#!/usr/bin/env python3
"""Diagnostics and charts for the entry-variant study."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/06_入场变体"
SRC = f"{BASE}/02_代码/entries.parquet"
OUT = f"{BASE}/03_结果"
FIG = f"{BASE}/04_图表"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last: dict[str, int] = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def overlap(a: pd.DataFrame, b: pd.DataFrame, tol: int = 20) -> float:
    """Share of a-signals that have a b-signal on the same code within +-tol sessions."""
    idx: dict[str, np.ndarray] = {c: np.sort(g["cal_idx"].values) for c, g in b.groupby("code")}
    hit = 0
    for c, ci in zip(a["code"].values, a["cal_idx"].values):
        arr = idx.get(c)
        if arr is None:
            continue
        p = np.searchsorted(arr, ci)
        near = arr[max(0, p - 1):p + 1]
        if len(near) and np.min(np.abs(near - ci)) <= tol:
            hit += 1
    return hit / len(a)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    base = df["base_ok"] & df["trend"] & df["tradable"] & df["wok"] & df["mok"]
    sets = {
        "E0_回踩MA20": dedup_20(df[base & df["pullback"] & df["reclaim"]]),
        "E1_20日新高": dedup_20(df[base & df["e1"]]),
        "E1V_放量突破": dedup_20(df[base & df["e1v"]]),
        "E2_浅回踩MA10": dedup_20(df[base & df["e2"]]),
        "E3_平台突破": dedup_20(df[base & df["e3"]]),
    }

    ov = []
    for a, b in [("E2_浅回踩MA10", "E0_回踩MA20"), ("E1_20日新高", "E0_回踩MA20"),
                 ("E3_平台突破", "E1_20日新高"), ("E1_20日新高", "E2_浅回踩MA10")]:
        ov.append({"a": a, "b": b, "share_a_with_b_within20": overlap(sets[a], sets[b])})
    ovdf = pd.DataFrame(ov)
    ovdf.to_csv(f"{OUT}/买点重叠.csv", index=False, encoding="utf-8-sig")
    print(ovdf.to_string(index=False))

    # yearly portfolio returns
    yrs = []
    for name in ["P1_E0回踩MA20_fixed20", "P2_E2浅回踩MA10_fixed10", "P3_E2浅回踩MA10_fixed20",
                 "P4_E2浅回踩MA10_rule", "P5_E2浅回踩MA10_ma20only", "P6_E1_20日新高_fixed20",
                 "P7_E1_20日新高_rule", "P8_E1_20日新高_ma20only", "P9_E2浅回踩MA10_fixed5",
                 "P10_E2浅回踩MA10_ma5", "P11_E1V放量突破_fixed20", "P12_E3平台突破_fixed20"]:
        f = f"{OUT}/组合_{name}_权益.csv"
        if not os.path.exists(f):
            continue
        e = pd.read_csv(f, parse_dates=["date"]).set_index("date")["equity"]
        ann = e.resample("YE").last()
        ret = ann.pct_change()
        first = e.iloc[0]
        ret.iloc[0] = ann.iloc[0] / first - 1
        for d, v in ret.items():
            yrs.append({"config": name, "year": d.year, "ret": v})
    ydf = pd.DataFrame(yrs)
    ydf.to_csv(f"{OUT}/组合_入场_分年.csv", index=False, encoding="utf-8-sig")

    summ = pd.read_csv(f"{OUT}/组合_入场_汇总.csv")
    sig = pd.read_csv(f"{OUT}/入场_信号层.csv")

    # 1) signal excess bars
    oos = sig[(sig.period == "oos_2018_2026") & (sig.h == 20)].copy()
    order = ["E0_回踩MA20", "E1_20日新高", "E1V_放量突破", "E2_浅回踩MA10", "E3_平台突破",
             "E1d_仅日线突破", "E2d_仅日线浅回踩"]
    oos["entry"] = pd.Categorical(oos["entry"], order, ordered=True)
    oos = oos.sort_values("entry")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(oos))
    ax.bar(x, oos["mean_ex"] * 100, color=np.where(oos["mean_ex"] < 0, "#3b6ea5", "#c96a3a"), alpha=0.85)
    for i, (v, t) in enumerate(zip(oos["mean_ex"], oos["t_ex_cluster"])):
        ax.text(i, v * 100 + (0.02 if v > 0 else -0.06), f"t={t:.1f}", ha="center", fontsize=7, color="#444")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in oos["entry"]], fontsize=8)
    ax.set_ylabel("20 日平均超额（%）")
    ax.set_title("验证期 2018–2026：各买点相对等权全A的 20 日超额", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    plt.savefig(f"{FIG}/01_买点超额_OOS.png")
    plt.close()

    # 2) OOS equity curves
    fig, ax = plt.subplots(figsize=(10, 5))
    show = {"P1_E0回踩MA20_fixed20": "E0 固定20（对照）", "P3_E2浅回踩MA10_fixed20": "E2 固定20",
            "P6_E1_20日新高_fixed20": "E1 突破 固定20", "P11_E1V放量突破_fixed20": "E1V 放量突破",
            "P12_E3平台突破_fixed20": "E3 平台突破"}
    base_idx = None
    for name, lab in show.items():
        e = pd.read_csv(f"{OUT}/组合_{name}_权益.csv", parse_dates=["date"]).set_index("date")["equity"]
        e = e[e.index >= "2018-01-01"]
        if base_idx is None:
            base_idx = e.index
        ax.plot(e.index, e / e.iloc[0], lw=1.2, label=lab)
    ax.set_ylabel("净值（2018-01-03=1）")
    ax.set_title("验证期组合净值（含 0.20% 往返成本）", fontsize=11)
    ax.grid(alpha=0.2)
    # EW benchmark: same OOS window, compounded at the realised EW OOS CAGR
    ew_cagr = float(summ["EW_oos_CAGR"].iloc[0])
    idx = np.arange(len(base_idx))
    ax.plot(base_idx, (1 + ew_cagr) ** (idx / 252), "--", color="#777", lw=1,
            label=f"等权全A（年化 {ew_cagr:.1%}）")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(base_idx[0], base_idx[-1])
    plt.savefig(f"{FIG}/02_组合权益_OOS.png")
    plt.close()

    # 3) exit comparison on E2
    e2 = summ[summ["name"].str.contains("P2_|P3_|P4_|P5_|P9_|P10_")].copy()
    e2["lab"] = e2["exit"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(e2))
    ax.bar(x - 0.2, e2["oos_CAGR"] * 100, 0.4, label="验证期年化", color="#3b6ea5")
    ax.bar(x + 0.2, e2["oos_MDD"] * 100, 0.4, label="验证期最大回撤", color="#c96a3a", alpha=0.75)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(e2["lab"])
    ax.set_ylabel("%")
    ax.set_title("同一买点 E2 下不同卖出规则的验证期表现", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    plt.savefig(f"{FIG}/03_卖出规则对比.png")
    plt.close()

    # 4) yearly signal excess heatmap-style bars
    ys = pd.read_csv(f"{OUT}/入场_信号_分年.csv")
    ys = ys[ys.entry.isin(["E0_回踩MA20", "E1_20日新高", "E2_浅回踩MA10", "E3_平台突破"])]
    piv = ys.pivot_table(index="year", columns="entry", values="mean_ex20")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, c in enumerate(piv.columns):
        ax.plot(piv.index, piv[c] * 100, marker="o", ms=3, lw=1.2, label=c)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("20 日平均超额（%）")
    ax.set_title("分年 20 日超额：突破只在少数年份为正", fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    plt.savefig(f"{FIG}/04_分年超额.png")
    plt.close()

    # 5) exposure vs OOS metrics
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for _, r in summ.iterrows():
        ax.scatter(r["avg_pos"], r["oos_CAGR"] * 100, s=60, color="#3b6ea5", alpha=0.8)
        ax.annotate(r["name"].split("_")[0], (r["avg_pos"], r["oos_CAGR"] * 100),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(summ["EW_oos_CAGR"].iloc[0] * 100, color="#c96a3a", ls="--", lw=1,
               label=f"等权全A {summ['EW_oos_CAGR'].iloc[0]:.1%}")
    ax.set_xlabel("平均持仓只数（10 槽）")
    ax.set_ylabel("验证期年化（%）")
    ax.set_title("买点越稀疏越像“现金策略”，仍跑不赢等权", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    plt.savefig(f"{FIG}/05_暴露与收益.png")
    plt.close()

    print("charts done")
    print("\n=== yearly portfolio ===")
    print(ydf.pivot(index="year", columns="config", values="ret").round(3).to_string())


if __name__ == "__main__":
    main()
