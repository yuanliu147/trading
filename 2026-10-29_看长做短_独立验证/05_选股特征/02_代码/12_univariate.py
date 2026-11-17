#!/usr/bin/env python3
"""Univariate feature study for the selection layer.

Dev 2012-2017 is used for cut points and IC signs; OOS 2018-2026 is reported
only. All signals use the frozen V2 entry, execution filters and 20-day dedup.
"""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd
from scipy import stats
from feature_defs import ALL_FEATURES, FAMILY_OF, add_derived

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/05_选股特征"
SRC = f"{BASE}/02_代码/cand_feat.parquet"
OUT = f"{BASE}/03_结果"


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last: dict[str, int] = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def load_v2() -> pd.DataFrame:
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df = add_derived(df)
    df["limit_up_open"] = df["next_raw_open"] >= df["next_raw_preclose"] * (1 + df["limit_pct"] / 100.0) - 0.001
    df["gap"] = df["next_open"] / df["hfq_close"] - 1.0
    df["tradable"] = (df["next_status"] == 1) & (~df["limit_up_open"]) & (df["gap"] <= 0.05)
    v2 = df[df["base_ok"] & df["trend"] & df["pullback"] & df["reclaim"] & df["wok"] & df["mok"] & df["tradable"]].copy()
    v2 = dedup_20(v2)
    v2["year"] = v2["date"].dt.year
    return v2


def daily_ic(df: pd.DataFrame, feat: str, target: str = "ex20", min_n: int = 5) -> pd.Series:
    t = df[["date", feat, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(t) < 50:
        return pd.Series(dtype=float)
    t["rf"] = t.groupby("date")[feat].rank(method="average")
    t["rt"] = t.groupby("date")[target].rank(method="average")
    g = t.groupby("date", sort=True)
    n = g.size().astype(float)
    sx = g["rf"].sum()
    sy = g["rt"].sum()
    sxy = (t["rf"] * t["rt"]).groupby(t["date"]).sum()
    sx2 = (t["rf"] ** 2).groupby(t["date"]).sum()
    sy2 = (t["rt"] ** 2).groupby(t["date"]).sum()
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sx2 / n - (sx / n) ** 2
    vy = sy2 / n - (sy / n) ** 2
    ic = cov / np.sqrt(vx * vy)
    return ic[n >= min_n].replace([np.inf, -np.inf], np.nan).dropna()


def ic_stats(ic: pd.Series) -> dict:
    n = len(ic)
    if n < 5:
        return {"n_days": n, "ic": np.nan, "t": np.nan, "p": np.nan, "pos": np.nan}
    mean = ic.mean()
    sd = ic.std(ddof=1)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if np.isfinite(t) else np.nan
    return {"n_days": n, "ic": mean, "t": t, "p": p, "pos": (ic > 0).mean()}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        val = pv[idx] * m / (m - rank + 1)
        prev = min(prev, val)
        q[idx] = prev
    out[ok] = np.minimum(q, 1.0)
    return out


def assign_groups(df: pd.DataFrame, feat: str, dev: pd.DataFrame):
    d = dev[feat].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200 or d.nunique() < 2:
        return None
    lo, hi = d.quantile([0.01, 0.99]).values
    x = df[feat].clip(lo, hi)
    if d.nunique() <= 10:
        return x.round(8).astype(str)
    qs = np.unique(d.clip(lo, hi).quantile([0.2, 0.4, 0.6, 0.8]).values)
    if len(qs) < 2:
        return None
    labels = [f"Q{i+1}" for i in range(len(qs) + 1)]
    return pd.cut(x, [-np.inf] + list(qs) + [np.inf], labels=labels, include_lowest=True)


def group_metrics(g: pd.DataFrame, family: str, feat: str, period: str, grp: str) -> dict:
    ex = g["ex20"].dropna()
    exi = g["ex_ind20"].dropna()
    r = g["r20"].dropna()
    daily = g.groupby("date")["ex20"].mean().dropna()
    t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))) if len(daily) > 2 and daily.std(ddof=1) > 0 else np.nan
    return {
        "family": family, "feature": feat, "period": period, "group": grp,
        "n": len(g), "n_days": g["date"].nunique(),
        "mean_ex20": ex.mean(), "median_ex20": ex.median(), "win_ex20": (ex > 0).mean(),
        "t_ex20": t,
        "mean_ex_ind20": exi.mean(), "median_ex_ind20": exi.median(), "win_ex_ind20": (exi > 0).mean(),
        "mean_ex40": g["ex40"].mean(), "median_ex40": g["ex40"].median(),
        "mean_r20": r.mean(), "median_r20": r.median(), "win_r20": (r > 0).mean(),
        "mean_mae": g["mae"].mean(), "mean_mfe": g["mfe"].mean(),
        "mfe_mae": g["mfe"].mean() - g["mae"].mean(),
    }


def spread_metrics(g: pd.DataFrame, family: str, feat: str, period: str, top: str, bottom: str) -> dict:
    a = g[g["group"] == top].groupby("date")["ex20"].mean()
    b = g[g["group"] == bottom].groupby("date")["ex20"].mean()
    d = (a - b).dropna()
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
    t_ex = g[g["group"] == top]["ex20"].mean() - g[g["group"] == bottom]["ex20"].mean()
    t_ind = g[g["group"] == top]["ex_ind20"].mean() - g[g["group"] == bottom]["ex_ind20"].mean()
    return {
        "family": family, "feature": feat, "period": period,
        "group": f"{top}-{bottom}", "n": len(d), "n_days": len(d),
        "mean_ex20": t_ex, "median_ex20": np.nan, "win_ex20": np.nan,
        "t_ex20": t,
        "mean_ex_ind20": t_ind, "median_ex_ind20": np.nan, "win_ex_ind20": np.nan,
        "mean_ex40": np.nan, "median_ex40": np.nan,
        "mean_r20": np.nan, "median_r20": np.nan, "win_r20": np.nan,
        "mean_mae": np.nan, "mean_mfe": np.nan, "mfe_mae": np.nan,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_v2()
    print("V2 after dedup", len(df), "dev", (df.period == "dev_2012_2017").sum(),
          "oos", (df.period == "oos_2018_2026").sum(), flush=True)
    dev = df[df["period"] == "dev_2012_2017"].copy()
    oos = df[df["period"] == "oos_2018_2026"].copy()

    rows = []
    ic_rows = []
    for k, feat in enumerate(ALL_FEATURES, 1):
        family = FAMILY_OF[feat]
        if feat not in df.columns:
            print("missing", feat, flush=True)
            continue
        ic_dev = daily_ic(dev, feat)
        ic_oos = daily_ic(oos, feat)
        ic_full = daily_ic(df, feat)
        ic_ind_dev = daily_ic(dev, feat, target="ex_ind20")
        ic_ind_oos = daily_ic(oos, feat, target="ex_ind20")
        s_dev = ic_stats(ic_dev)
        s_oos = ic_stats(ic_oos)
        s_full = ic_stats(ic_full)
        s_ind_dev = ic_stats(ic_ind_dev)
        s_ind_oos = ic_stats(ic_ind_oos)

        yearly = {}
        for y in range(2012, 2018):
            yearly[y] = ic_stats(daily_ic(dev[dev["year"] == y], feat))["ic"]
        sign = np.sign(s_dev["ic"]) if np.isfinite(s_dev["ic"]) else 0
        stable = sum(1 for v in yearly.values() if np.isfinite(v) and np.sign(v) == sign and sign != 0)

        # groups
        grp = assign_groups(df, feat, dev)
        q5q1_dev = np.nan
        q5q1_oos = np.nan
        q5q1_ind_dev = np.nan
        q5q1_ind_oos = np.nan
        t_spread_dev = np.nan
        t_spread_oos = np.nan
        if grp is not None:
            df["_grp"] = grp
            for period, sub in (("dev_2012_2017", df[df["period"] == "dev_2012_2017"]),
                                ("oos_2018_2026", df[df["period"] == "oos_2018_2026"]),
                                ("full", df)):
                gg = sub[sub["_grp"].notna()]
                for name, g in gg.groupby("_grp", observed=True):
                    rows.append(group_metrics(g, family, feat, period, str(name)))
                labels = [str(x) for x in gg["_grp"].cat.categories] if hasattr(gg["_grp"], "cat") else sorted(gg["_grp"].unique())
                if len(labels) >= 2:
                    top, bottom = labels[-1], labels[0]
                    rows.append(spread_metrics(gg.rename(columns={"_grp": "group"}), family, feat, period, top, bottom))
                    if period == "dev_2012_2017":
                        q5q1_dev = rows[-1]["mean_ex20"]
                        t_spread_dev = rows[-1]["t_ex20"]
                        q5q1_ind_dev = rows[-1]["mean_ex_ind20"]
                    elif period == "oos_2018_2026":
                        q5q1_oos = rows[-1]["mean_ex20"]
                        t_spread_oos = rows[-1]["t_ex20"]
                        q5q1_ind_oos = rows[-1]["mean_ex_ind20"]
            df = df.drop(columns=["_grp"])

        ic_rows.append({
            "family": family, "feature": feat,
            "missing_rate": df[feat].isna().mean(),
            "n_dev": len(dev[feat].dropna()), "n_oos": len(oos[feat].dropna()),
            "n_days_dev": s_dev["n_days"], "n_days_oos": s_oos["n_days"],
            "ic_dev": s_dev["ic"], "t_dev": s_dev["t"], "p_dev": s_dev["p"], "pos_dev": s_dev["pos"],
            "ic_oos": s_oos["ic"], "t_oos": s_oos["t"], "p_oos": s_oos["p"], "pos_oos": s_oos["pos"],
            "ic_full": s_full["ic"], "t_full": s_full["t"],
            "ic_ind_dev": s_ind_dev["ic"], "t_ind_dev": s_ind_dev["t"],
            "ic_ind_oos": s_ind_oos["ic"], "t_ind_oos": s_ind_oos["t"],
            "yearly_ic": ";".join(f"{y}:{v:.4f}" for y, v in yearly.items()),
            "stable_years": stable,
            "q5q1_dev": q5q1_dev, "t_q5q1_dev": t_spread_dev,
            "q5q1_oos": q5q1_oos, "t_q5q1_oos": t_spread_oos,
            "q5q1_ind_dev": q5q1_ind_dev, "q5q1_ind_oos": q5q1_ind_oos,
        })
        if k % 10 == 0:
            print(f"{k}/{len(ALL_FEATURES)} {feat} ic_dev={s_dev['ic']:.4f} ic_oos={s_oos['ic']:.4f}", flush=True)

    uni = pd.DataFrame(rows)
    ic = pd.DataFrame(ic_rows)
    ic["q_dev"] = bh_fdr(ic["p_dev"].values)
    ic["pass_fdr"] = (ic["q_dev"] < 0.10) & (ic["ic_dev"].abs() >= 0.02) & (ic["stable_years"] >= 4)
    ic = ic.sort_values("ic_dev", key=lambda s: s.abs(), ascending=False)
    uni.to_csv(f"{OUT}/特征_单变量.csv", index=False, encoding="utf-8-sig")
    ic.to_csv(f"{OUT}/特征_IC.csv", index=False, encoding="utf-8-sig")

    print("\n=== top |IC| dev ===")
    print(ic[["feature", "family", "ic_dev", "t_dev", "q_dev", "stable_years", "ic_oos", "t_oos", "ic_ind_dev", "ic_ind_oos", "q5q1_dev", "q5q1_oos", "q5q1_ind_dev", "q5q1_ind_oos", "pass_fdr"]].head(30).to_string(index=False), flush=True)
    print("\npass_fdr", int(ic["pass_fdr"].sum()), "of", len(ic), flush=True)


if __name__ == "__main__":
    main()
