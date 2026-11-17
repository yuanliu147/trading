#!/usr/bin/env python3
"""Build the three pre-registered scores and evaluate them at signal level."""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd
from scipy import stats
from feature_defs import THEORY_SIGN, add_derived

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/05_选股特征"
SRC = f"{BASE}/02_代码/cand_feat.parquet"
IC = f"{BASE}/03_结果/特征_IC.csv"
OUT = f"{BASE}/03_结果"
SIG = f"{BASE}/02_代码/信号_带评分.parquet"


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last: dict[str, int] = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def load_raw() -> pd.DataFrame:
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df = add_derived(df)
    df["limit_up_open"] = df["next_raw_open"] >= df["next_raw_preclose"] * (1 + df["limit_pct"] / 100.0) - 0.001
    df["gap"] = df["next_open"] / df["hfq_close"] - 1.0
    df["tradable"] = (df["next_status"] == 1) & (~df["limit_up_open"]) & (df["gap"] <= 0.05)
    v2 = df[df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok"] & df["tradable"]].copy()
    v2["year"] = v2["date"].dt.year
    return v2


def zscore(df: pd.DataFrame, dev: pd.DataFrame, feat: str, sign: int) -> pd.Series:
    d = dev[feat].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = d.quantile([0.01, 0.99]).values
    mu = d.clip(lo, hi).mean()
    sd = d.clip(lo, hi).std()
    x = df[feat].clip(lo, hi)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=df.index)
    return (sign * (x - mu) / sd).fillna(0.0)


def daily_ic(df: pd.DataFrame, col: str, target: str = "ex20", min_n: int = 5) -> pd.Series:
    t = df[["date", col, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(t) < 50:
        return pd.Series(dtype=float)
    t["rf"] = t.groupby("date")[col].rank(method="average")
    t["rt"] = t.groupby("date")[target].rank(method="average")
    g = t.groupby("date", sort=True)
    n = g.size().astype(float)
    sx = g["rf"].sum(); sy = g["rt"].sum()
    sxy = (t["rf"] * t["rt"]).groupby(t["date"]).sum()
    sx2 = (t["rf"] ** 2).groupby(t["date"]).sum()
    sy2 = (t["rt"] ** 2).groupby(t["date"]).sum()
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sx2 / n - (sx / n) ** 2
    vy = sy2 / n - (sy / n) ** 2
    return (cov / np.sqrt(vx * vy))[n >= min_n].replace([np.inf, -np.inf], np.nan).dropna()


def ic_stats(ic: pd.Series) -> dict:
    n = len(ic)
    if n < 5:
        return {"n_days": n, "ic": np.nan, "t": np.nan, "pos": np.nan}
    mean = ic.mean(); sd = ic.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else np.nan
    return {"n_days": n, "ic": mean, "t": t, "pos": (ic > 0).mean()}


def metrics(g: pd.DataFrame, score: str, period: str, group: str) -> dict:
    ex = g["ex20"].dropna()
    daily = g.groupby("date")["ex20"].mean().dropna()
    t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 2 and daily.std(ddof=1) > 0 else np.nan
    return {
        "score": score, "period": period, "group": group, "n": len(g), "n_days": g["date"].nunique(),
        "mean_ex20": ex.mean(), "median_ex20": ex.median(), "win_ex20": (ex > 0).mean(), "t_ex20": t,
        "mean_ex_ind20": g["ex_ind20"].mean(), "median_ex_ind20": g["ex_ind20"].median(), "win_ex_ind20": (g["ex_ind20"] > 0).mean(),
        "mean_r20": g["r20"].mean(), "median_r20": g["r20"].median(), "win_r20": (g["r20"] > 0).mean(),
        "mean_mae": g["mae"].mean(), "mean_mfe": g["mfe"].mean(),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    raw = load_raw()
    dev_all = raw[raw["date"].dt.year <= 2017].copy()
    dev = dedup_20(dev_all)
    print("raw V2 tradable", len(raw), "dev", len(dev), flush=True)

    ic = pd.read_csv(IC)
    # ---- A family-balanced: one per family, |IC|>=0.05, stable>=4, cross-family corr<0.70 ----
    cand = ic[(ic["pass_fdr"]) & (ic["ic_dev"].abs() >= 0.05) & (ic["stable_years"] >= 4)].copy()
    cand["abs_ic"] = cand["ic_dev"].abs()
    cfeat = [f for f in cand["feature"] if f in dev.columns]
    corr = dev[cfeat].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    reps = cand.sort_values("abs_ic", ascending=False).groupby("family", as_index=False).first()
    reps = reps.sort_values("abs_ic", ascending=False)
    a_family = []
    for _, row in reps.iterrows():
        f = row["feature"]
        if f not in corr.columns:
            continue
        if a_family and corr.loc[f, a_family].abs().max() >= 0.70:
            continue
        a_family.append(f)
        if len(a_family) >= 8:
            break
    print("\nA family-balanced:", a_family, flush=True)
    print(reps[reps.feature.isin(a_family)][["feature", "family", "ic_dev", "ic_oos"]].to_string(index=False), flush=True)

    # ---- A-greedy: max 2 per family, corr<0.70, max 8 ----
    greedy = []
    fam_count: dict[str, int] = {}
    for _, row in cand.sort_values("abs_ic", ascending=False).iterrows():
        f, fam = row["feature"], row["family"]
        if f not in corr.columns or fam_count.get(fam, 0) >= 2:
            continue
        if greedy and corr.loc[f, greedy].abs().max() >= 0.70:
            continue
        greedy.append(f); fam_count[fam] = fam_count.get(fam, 0) + 1
        if len(greedy) >= 8:
            break
    print("\nA-greedy:", greedy, flush=True)

    # ---- scores ----
    score_defs = {}
    score_defs["A"] = [(f, int(np.sign(ic.loc[ic.feature == f, "ic_dev"].iloc[0]))) for f in a_family]
    score_defs["A_greedy"] = [(f, int(np.sign(ic.loc[ic.feature == f, "ic_dev"].iloc[0]))) for f in greedy]
    score_defs["B"] = [(f, s) for f, s in THEORY_SIGN.items()]

    defs = []
    for name, items in score_defs.items():
        for f, s in items:
            defs.append({"score": name, "feature": f, "sign": s,
                         "ic_dev": ic.loc[ic.feature == f, "ic_dev"].iloc[0] if f in set(ic.feature) else np.nan,
                         "ic_oos": ic.loc[ic.feature == f, "ic_oos"].iloc[0] if f in set(ic.feature) else np.nan,
                         "family": ic.loc[ic.feature == f, "family"].iloc[0] if f in set(ic.feature) else ""})
    pd.DataFrame(defs).to_csv(f"{OUT}/评分_定义.csv", index=False, encoding="utf-8-sig")

    qrows, trows, yrows, irows, crows = [], [], [], [], []
    for name, items in score_defs.items():
        zs = [zscore(raw, dev, f, s) for f, s in items]
        raw[f"score_{name}"] = pd.concat(zs, axis=1).mean(axis=1)
        # dev percentile, fixed cut points
        dev_scored = dedup_20(raw[raw["date"].dt.year <= 2017])
        cut = dev_scored[f"score_{name}"].quantile([0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.9]).values
        crows.append({"score": name, "q50": cut[5], "q70": cut[4], "q90": cut[6],
                      "n_dev": len(dev_scored)})
        raw[f"score_{name}_pct"] = pd.cut(raw[f"score_{name}"],
                                          [-np.inf] + list(np.unique(cut[:4])) + [np.inf],
                                          labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        raw[f"top30_{name}"] = raw[f"score_{name}"] >= cut[4]
        raw[f"top50_{name}"] = raw[f"score_{name}"] >= cut[5]
        raw[f"top10_{name}"] = raw[f"score_{name}"] >= cut[6]

    # evaluation sample: dedup after score assignment
    ev = dedup_20(raw[raw["date"].dt.year <= 2026].copy())
    ev["period"] = np.where(ev["date"].dt.year <= 2017, "dev_2012_2017", "oos_2018_2026")

    for name in score_defs:
        for period, sub in [("dev_2012_2017", ev[ev["period"] == "dev_2012_2017"]),
                            ("oos_2018_2026", ev[ev["period"] == "oos_2018_2026"]),
                            ("full", ev)]:
            for g, gg in sub.groupby(f"score_{name}_pct", observed=True):
                qrows.append(metrics(gg, name, period, str(g)))
            q1 = sub[sub[f"score_{name}_pct"] == "Q1"]
            q5 = sub[sub[f"score_{name}_pct"] == "Q5"]
            if len(q1) and len(q5):
                a = q5.groupby("date")["ex20"].mean(); b = q1.groupby("date")["ex20"].mean()
                d = (a - b).dropna()
                t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
                qrows.append({"score": name, "period": period, "group": "Q5-Q1", "n": len(d), "n_days": len(d),
                              "mean_ex20": q5["ex20"].mean() - q1["ex20"].mean(),
                              "median_ex20": np.nan, "win_ex20": np.nan, "t_ex20": t,
                              "mean_ex_ind20": q5["ex_ind20"].mean() - q1["ex_ind20"].mean(),
                              "median_ex_ind20": np.nan, "win_ex_ind20": np.nan,
                              "mean_r20": np.nan, "median_r20": np.nan, "win_r20": np.nan,
                              "mean_mae": np.nan, "mean_mfe": np.nan})
            for label, col in [("top50", f"top50_{name}"), ("top30", f"top30_{name}"), ("top10", f"top10_{name}")]:
                g = sub[sub[col]]
                trows.append(metrics(g, name, period, label))

        for y, sub in ev.groupby("year"):
            g = sub[sub[f"top30_{name}"]]
            yrows.append({"score": name, "year": int(y), "n_all": len(sub), "ex20_all": sub["ex20"].mean(),
                          "n_top30": len(g), "ex20_top30": g["ex20"].mean(),
                          "ex_ind_top30": g["ex_ind20"].mean(), "win_top30": (g["ex20"] > 0).mean(),
                          "r20_top30": g["r20"].mean()})
        for period, sub in [("dev_2012_2017", ev[ev["period"] == "dev_2012_2017"]),
                            ("oos_2018_2026", ev[ev["period"] == "oos_2018_2026"]), ("full", ev)]:
            s = daily_ic(sub, f"score_{name}")
            si = daily_ic(sub, f"score_{name}", target="ex_ind20")
            st = ic_stats(s); sti = ic_stats(si)
            irows.append({"score": name, "period": period, "n_days": st["n_days"],
                          "ic": st["ic"], "t": st["t"], "pos": st["pos"],
                          "ic_ind": sti["ic"], "t_ind": sti["t"]})

    pd.DataFrame(qrows).to_csv(f"{OUT}/评分_分位.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trows).to_csv(f"{OUT}/评分_TOP.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yrows).to_csv(f"{OUT}/评分_年度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(irows).to_csv(f"{OUT}/评分_IC.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(crows).to_csv(f"{OUT}/评分_切点.csv", index=False, encoding="utf-8-sig")
    raw.to_parquet(SIG, index=False)

    print("\n=== top30 OOS ===")
    print(pd.DataFrame(trows).query("period=='oos_2018_2026'").to_string(index=False), flush=True)
    print("\n=== quintiles OOS ===")
    print(pd.DataFrame(qrows).query("period=='oos_2018_2026'").to_string(index=False), flush=True)
    print("\n=== IC ===")
    print(pd.DataFrame(irows).to_string(index=False), flush=True)
    print("saved", SIG, flush=True)


if __name__ == "__main__":
    main()
