#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""买入点复盘：全部时点 500 金叉信号，每笔固定 1 万元。

重点回答：
1. 哪些信号日特征与后续收益/亏损概率相关；
2. 哪些“走势”亏损概率更高；
3. 这些差异在 2010-2017、2018-2022、2023-2026 是否稳定；
4. 股性类型、市场状态是否只是换了暴露。

输出中文表头的 CSV 与图表。收益口径默认用“死叉卖出”（让盈利奔跑），
同时保留 b3 与 20 日固定持有的结果作对照。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA = HERE / "01_数据"
FIG = HERE / "02_图表"
FIG.mkdir(exist_ok=True)

# 中文字体
for name in ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

CN = {
    "ret1": "当日涨跌", "ret3": "前3日收益", "ret5": "前5日收益",
    "ret10": "前10日收益", "ret20": "前20日收益", "ret60": "前60日收益",
    "ret120": "前120日收益", "dist_ma5": "距MA5", "dist_ma10": "距MA10",
    "dist_ma20": "距MA20", "dist_ma60": "距MA60",
    "ma20_slope10": "MA20十日斜率", "ma60_slope20": "MA60二十日斜率",
    "ma_align": "均线多头排列数", "rsi14": "RSI14",
    "atr20": "ATR20占价", "vol20": "20日波动", "vol60": "60日波动",
    "vol_ratio": "波动比20/60", "max_dd60": "距60日高点",
    "dd250": "距250日高点", "pos250": "250日区间位置",
    "amt5_60": "5日均额/60日均额", "amt20_60": "20日均额/60日均额",
    "amt60_yi": "60日均额亿", "vol5_60": "5日均量/60日均量",
    "hist_norm": "MACD柱/价", "dif_norm": "DIF/价", "dea_norm": "DEA/价",
    "hist_slope5": "MACD柱5日变化", "dif_slope5": "DIF5日变化",
    "hist_accel": "MACD柱加速度", "n_golden_60": "60日金叉数",
    "n_death_60": "60日死叉数", "days_since_golden": "距上次金叉",
    "signal_day_ret": "信号日涨跌", "body": "实体占振幅",
    "upper_shadow": "上影占振幅", "lower_shadow": "下影占振幅",
    "close_pos": "收盘区间位置", "gap": "信号日跳空",
    "big_move_20": "20日大波动天数", "price": "价格", "log_price": "对数价格",
    "beta60": "60日Beta", "corr60": "60日相关性",
    "rs20": "20日相对强弱", "rs60": "60日相对强弱",
    "idx_ret20": "中证500前20日", "idx_ret60": "中证500前60日",
    "idx_vol20": "中证500 20日波动", "idx_dd250": "中证500距250日高点",
    "vol120": "120日波动", "range250": "250日振幅",
    "s_vol": "股性波动分", "s_heat": "股性活跃分",
    "breadth_ma20": "MA20上方占比", "entry_gap": "入场跳空",
}

FEATURES = [
    "ret1", "ret3", "ret5", "ret10", "ret20", "ret60", "ret120",
    "dist_ma5", "dist_ma10", "dist_ma20", "dist_ma60",
    "ma20_slope10", "ma60_slope20", "ma_align", "rsi14",
    "atr20", "vol20", "vol60", "vol_ratio", "max_dd60",
    "dd250", "pos250", "amt5_60", "amt20_60", "amt60_yi",
    "vol5_60", "hist_norm", "dif_norm", "dea_norm", "hist_slope5",
    "dif_slope5", "hist_accel", "n_golden_60", "n_death_60",
    "days_since_golden", "signal_day_ret", "body", "upper_shadow",
    "lower_shadow", "close_pos", "gap", "big_move_20", "price",
    "log_price", "beta60", "corr60", "rs20", "rs60",
    "idx_ret20", "idx_ret60", "idx_vol20", "idx_dd250",
    "breath_placeholder",
]
FEATURES = [f for f in FEATURES if f != "breath_placeholder"] + [
    "breadth_ma20", "vol120", "range250", "s_vol", "s_heat", "entry_gap"]

STAGES = [("设计 2010-2017", 2010, 2017),
          ("验证 2018-2022", 2018, 2022),
          ("样本外 2023-2026", 2023, 2026)]


def load() -> pd.DataFrame:
    tr = pd.read_csv(DATA / "全信号交易_分规则.csv", encoding="utf-8-sig",
                     parse_dates=["信号日", "买入日", "卖出日"])
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", encoding="utf-8-sig",
                      parse_dates=["signal_date"])
    g = tr[tr["规则"] == "死叉"].copy()
    g = g.merge(sig, left_on=["代码", "信号日"],
                right_on=["code", "signal_date"], how="left",
                suffixes=("", "_sig"))
    g["entry_gap"] = g["入场跳空"]
    g["年份"] = g["信号日"].dt.year
    g["阶段"] = np.select(
        [g["年份"] <= 2017, g["年份"] <= 2022],
        ["设计 2010-2017", "验证 2018-2022"], default="样本外 2023-2026")
    g["盈利"] = (g["净收益率"] > 0).astype(int)
    g["大亏"] = (g["净收益率"] <= -0.10).astype(int)
    return g


def bh_q(p: pd.Series) -> pd.Series:
    idx = p.index
    vals = p.astype(float).values
    n = len(vals)
    order = np.argsort(vals)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        val = min(prev, vals[i] * n / k)
        q[i] = val
        prev = val
    return pd.Series(q, index=idx)


def univariate(g: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    edges = {}
    quint_rows = []
    for f in FEATURES:
        x = pd.to_numeric(g[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = (x.notna() & g["净收益率"].notna()).to_numpy()
        if valid.sum() < 500:
            continue
        x2 = x.to_numpy()[valid]
        y2 = g["净收益率"].to_numpy()[valid]
        w2 = g["盈利"].to_numpy()[valid]
        big2 = g["大亏"].to_numpy()[valid]
        fwd2 = g["fwd20"].to_numpy()[valid]
        try:
            _, edges[f] = pd.qcut(x2, 5, retbins=True, duplicates="drop")
            b = pd.cut(x2, bins=edges[f], include_lowest=True, labels=False)
        except ValueError:
            continue
        if pd.Series(b).nunique() < 4:
            continue
        tab = []
        for k in sorted(pd.Series(b).dropna().unique()):
            m = (b == k)
            if m.sum() == 0:
                continue
            row = {
                "特征": f, "含义": CN.get(f, f), "分位": int(k) + 1,
                "区间下限": edges[f][int(k)],
                "区间上限": edges[f][int(k) + 1],
                "样本": int(m.sum()),
                "胜率": w2[m].mean() * 100,
                "平均收益": y2[m].mean() * 100,
                "收益中位数": np.median(y2[m]) * 100,
                "大亏率": big2[m].mean() * 100,
                "20日平均": np.nanmean(fwd2[m]) * 100,
            }
            tab.append(row)
            quint_rows.append(row)
        t = pd.DataFrame(tab)
        if len(t) < 4:
            continue
        q1, q5 = t.iloc[0], t.iloc[-1]
        rho, p = stats.spearmanr(x2, y2)
        try:
            auc = roc_auc_score(w2, x2)
        except ValueError:
            auc = np.nan
        rows.append({
            "特征": f, "含义": CN.get(f, f), "样本": int(valid.sum()),
            "Q1胜率": q1["胜率"], "Q5胜率": q5["胜率"],
            "胜率差Q5-Q1": q5["胜率"] - q1["胜率"],
            "Q1平均": q1["平均收益"], "Q5平均": q5["平均收益"],
            "收益差Q5-Q1": q5["平均收益"] - q1["平均收益"],
            "Q1大亏率": q1["大亏率"], "Q5大亏率": q5["大亏率"],
            "大亏差Q5-Q1": q5["大亏率"] - q1["大亏率"],
            "Q1的20日": q1["20日平均"], "Q5的20日": q5["20日平均"],
            "20日差Q5-Q1": q5["20日平均"] - q1["20日平均"],
            "Spearman": rho, "p值": p, "AUC": auc,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["q值"] = bh_q(out["p值"])
        out["方向一致性"] = np.sign(out["收益差Q5-Q1"]) == np.sign(out["20日差Q5-Q1"])
        out = out.sort_values("q值")
    return out, edges, pd.DataFrame(quint_rows)


def by_stage(g: pd.DataFrame, edges: dict) -> pd.DataFrame:
    rows = []
    for f, e in edges.items():
        for stage, lo, hi in STAGES:
            s = g[(g["年份"] >= lo) & (g["年份"] <= hi)]
            x = pd.to_numeric(s[f], errors="coerce")
            b = pd.cut(x, bins=e, include_lowest=True, labels=False)
            if b.nunique() < 3:
                continue
            for k in [b.min(), b.max()]:
                m = b == k
                if m.sum() == 0:
                    continue
                rows.append({
                    "特征": f, "含义": CN.get(f, f), "阶段": stage,
                    "端点": "Q1" if k == b.min() else "Q5",
                    "样本": int(m.sum()), "胜率": s.loc[m, "盈利"].mean() * 100,
                    "平均收益": s.loc[m, "净收益率"].mean() * 100,
                    "大亏率": s.loc[m, "大亏"].mean() * 100,
                    "20日平均": s.loc[m, "fwd20"].mean() * 100,
                })
    return pd.DataFrame(rows)


def yearly_sign(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        if f not in g.columns:
            continue
        for y, s in g.groupby("年份"):
            x = pd.to_numeric(s[f], errors="coerce")
            m = x.notna() & s["净收益率"].notna()
            if m.sum() < 100:
                continue
            rho, p = stats.spearmanr(x[m], s.loc[m, "净收益率"])
            rows.append({"特征": f, "含义": CN.get(f, f), "年份": int(y),
                         "Spearman": rho, "p值": p, "样本": int(m.sum())})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    agg = d.groupby(["特征", "含义"]).agg(
        正相关年数=("Spearman", lambda s: int((s > 0).sum())),
        负相关年数=("Spearman", lambda s: int((s < 0).sum())),
        中位相关=("Spearman", "median"),
        最小相关=("Spearman", "min"),
        最大相关=("Spearman", "max"),
        年数=("年份", "nunique"),
    ).reset_index()
    agg["符号稳定率"] = agg[["正相关年数", "负相关年数"]].max(axis=1) / agg["年数"]
    return agg.sort_values("中位相关")


def binary_rules(g: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "前20日涨幅>15%": g["ret20"] > 0.15,
        "前20日涨幅>25%": g["ret20"] > 0.25,
        "距MA20>8%": g["dist_ma20"] > 0.08,
        "距MA20>12%": g["dist_ma20"] > 0.12,
        "收盘低于MA60": g["dist_ma60"] < 0,
        "MA60向下": g["ma60_slope20"] < 0,
        "5日均额/60日均额<-20%": g["amt5_60"] < -0.20,
        "5日均额/60日均额>20%": g["amt5_60"] > 0.20,
        "信号日涨>5%": g["signal_day_ret"] > 0.05,
        "信号日涨>8%": g["signal_day_ret"] > 0.08,
        "ATR20>5%": g["atr20"] > 0.05,
        "20日波动>50%": g["vol20"] > 0.50,
        "RSI14>70": g["rsi14"] > 70,
        "250日位置>0.9": g["pos250"] > 0.9,
        "距250日高点<3%": g["dd250"] > -0.03,
        "60日金叉>=4次": g["n_golden_60"] >= 4,
        "中证500在MA20下": g["idx_above_ma20"] == 0,
        "中证500在MA60下": g["idx_above_ma60"] == 0,
        "MA20上方占比<40%": g["breadth_ma20"] < 0.40,
        "入场跳空>2%": g["entry_gap"] > 0.02,
        "价格<5元": g["price"] < 5,
        "高波活跃型": g["type"] == "高波活跃型",
    }
    rows = []
    for name, m in rules.items():
        for stage, lo, hi in STAGES + [("全样本", 2010, 2026)]:
            s = g[(g["年份"] >= lo) & (g["年份"] <= hi)]
            mm = m.loc[s.index]
            if mm.sum() < 50:
                continue
            rows.append({
                "规则": name, "阶段": stage, "命中数": int(mm.sum()),
                "命中占比": mm.mean() * 100,
                "命中胜率": s.loc[mm, "盈利"].mean() * 100,
                "命中平均": s.loc[mm, "净收益率"].mean() * 100,
                "命中大亏率": s.loc[mm, "大亏"].mean() * 100,
                "命中20日": s.loc[mm, "fwd20"].mean() * 100,
                "未命中胜率": s.loc[~mm, "盈利"].mean() * 100,
                "未命中平均": s.loc[~mm, "净收益率"].mean() * 100,
                "平均差": (s.loc[mm, "净收益率"].mean()
                         - s.loc[~mm, "净收益率"].mean()) * 100,
            })
    return pd.DataFrame(rows)


def behavior(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, lo, hi in STAGES + [("全样本", 2010, 2026)]:
        s = g[(g["年份"] >= lo) & (g["年份"] <= hi)]
        for t, x in s.groupby("type"):
            rows.append({
                "阶段": stage, "股性类型": t, "样本": len(x),
                "占比": len(x) / len(s) * 100,
                "胜率": x["盈利"].mean() * 100,
                "平均收益": x["净收益率"].mean() * 100,
                "收益中位数": x["净收益率"].median() * 100,
                "大亏率": x["大亏"].mean() * 100,
                "20日平均": x["fwd20"].mean() * 100,
                "平均最大不利": x["期间最低"].mean() * 100,
            })
        for sub, x in s[s["subtype"].notna() & (s["subtype"] != "")].groupby("subtype"):
            rows.append({
                "阶段": stage, "股性类型": sub, "样本": len(x),
                "占比": len(x) / len(s) * 100,
                "胜率": x["盈利"].mean() * 100,
                "平均收益": x["净收益率"].mean() * 100,
                "收益中位数": x["净收益率"].median() * 100,
                "大亏率": x["大亏"].mean() * 100,
                "20日平均": x["fwd20"].mean() * 100,
                "平均最大不利": x["期间最低"].mean() * 100,
            })
    return pd.DataFrame(rows)


def market(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, lo, hi in STAGES + [("全样本", 2010, 2026)]:
        s = g[(g["年份"] >= lo) & (g["年份"] <= hi)]
        for name, m in {
            "中证500在MA20上": s["idx_above_ma20"] == 1,
            "中证500在MA20下": s["idx_above_ma20"] == 0,
            "中证500在MA60上": s["idx_above_ma60"] == 1,
            "中证500在MA60下": s["idx_above_ma60"] == 0,
        }.items():
            x = s[m]
            rows.append({"阶段": stage, "状态": name, "样本": len(x),
                         "胜率": x["盈利"].mean() * 100,
                         "平均收益": x["净收益率"].mean() * 100,
                         "大亏率": x["大亏"].mean() * 100,
                         "20日平均": x["fwd20"].mean() * 100})
        for f, label in [("idx_ret20", "中证500前20日"),
                         ("breadth_ma20", "MA20上方占比"),
                         ("idx_vol20", "中证500 20日波动")]:
            try:
                b = pd.qcut(s[f], 5, labels=False, duplicates="drop")
            except ValueError:
                continue
            for k in sorted(b.dropna().unique()):
                x = s[b == k]
                rows.append({"阶段": stage, "状态": f"{label} Q{int(k)+1}",
                             "样本": len(x), "胜率": x["盈利"].mean() * 100,
                             "平均收益": x["净收益率"].mean() * 100,
                             "大亏率": x["大亏"].mean() * 100,
                             "20日平均": x["fwd20"].mean() * 100})
    return pd.DataFrame(rows)


def path(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, lo, hi in STAGES + [("全样本", 2010, 2026)]:
        s = g[(g["年份"] >= lo) & (g["年份"] <= hi)]
        for label, x in [("盈利单", s[s["盈利"] == 1]),
                         ("亏损单", s[s["盈利"] == 0])]:
            rows.append({
                "阶段": stage, "分组": label, "样本": len(x),
                "平均收益": x["净收益率"].mean() * 100,
                "平均峰值浮盈": x["峰值浮盈"].mean() * 100,
                "中位峰值浮盈": x["峰值浮盈"].median() * 100,
                "平均最大不利": x["期间最低"].mean() * 100,
                "中位最大不利": x["期间最低"].median() * 100,
                "平均持有": x["持有交易日"].mean(),
                "峰值>=5%占比": (x["峰值浮盈"] >= 0.05).mean() * 100,
                "峰值>=10%占比": (x["峰值浮盈"] >= 0.10).mean() * 100,
                "最低<=-5%占比": (x["期间最低"] <= -0.05).mean() * 100,
                "最低<=-8%占比": (x["期间最低"] <= -0.08).mean() * 100,
            })
    return pd.DataFrame(rows)


def exit_compare() -> pd.DataFrame:
    tr = pd.read_csv(DATA / "全信号交易_分规则.csv", encoding="utf-8-sig",
                     parse_dates=["信号日"])
    key = ["代码", "信号日"]
    base = tr[tr["规则"] == "死叉"].set_index(key)
    rows = []
    for rule in ["b3_10_8", "b3_8_6", "b3_12_8", "b3_15_10",
                 "死叉_硬止损8", "死叉_硬止损10", "b3_10_8_硬止损8",
                 "b3_10_8_时间40", "死叉_时间40"]:
        x = tr[tr["规则"] == rule].set_index(key)
        common = base.index.intersection(x.index)
        b, z = base.loc[common], x.loc[common]
        diff = z["净收益率"] - b["净收益率"]
        rows.append({
            "规则": rule, "配对样本": len(common),
            "规则平均": z["净收益率"].mean() * 100,
            "死叉平均": b["净收益率"].mean() * 100,
            "平均差": diff.mean() * 100,
            "差的中位数": diff.median() * 100,
            "规则更好占比": (diff > 0).mean() * 100,
            "规则胜率": (z["净收益率"] > 0).mean() * 100,
            "死叉胜率": (b["净收益率"] > 0).mean() * 100,
            "规则平均持有": z["持有交易日"].mean(),
            "死叉平均持有": b["持有交易日"].mean(),
            "规则平均最大不利": z["期间最低"].mean() * 100,
            "死叉平均最大不利": b["期间最低"].mean() * 100,
        })
    return pd.DataFrame(rows)


def plot_quintiles(g: pd.DataFrame, edges: dict, feats: list[str]) -> None:
    n = len(feats)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    for ax, f in zip(axes, feats):
        if f not in edges:
            ax.axis("off")
            continue
        b = pd.cut(pd.to_numeric(g[f], errors="coerce"), bins=edges[f],
                   include_lowest=True, labels=False)
        xs, wr, av = [], [], []
        for k in sorted(b.dropna().unique()):
            m = b == k
            xs.append(f"Q{int(k)+1}")
            wr.append(g.loc[m, "盈利"].mean() * 100)
            av.append(g.loc[m, "净收益率"].mean() * 100)
        ax.bar(xs, wr, color="#4C78A8", alpha=0.85, label="胜率")
        ax.set_ylim(0, max(50, max(wr) * 1.2))
        ax.set_title(CN.get(f, f), fontsize=11)
        ax.set_ylabel("胜率 %")
        ax2 = ax.twinx()
        ax2.plot(xs, av, color="#E45756", marker="o", lw=1.6, label="平均收益")
        ax2.set_ylabel("平均收益 %", color="#E45756")
        ax2.tick_params(axis="y", colors="#E45756")
        ax2.axhline(0, color="#E45756", lw=0.6, ls=":")
        for i, v in enumerate(av):
            ax2.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8, color="#E45756")
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.suptitle("信号日特征五分位：胜率（柱）与死叉卖出平均收益（线）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "01_分位胜率_核心特征.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(g: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(
            axes, ["盈利", "净收益率"],
            ["胜率 %", "平均收益 %"]):
        x = pd.qcut(g["ret20"], 3, labels=["前20日低", "前20日中", "前20日高"],
                    duplicates="drop")
        y = pd.qcut(g["dist_ma20"], 3, labels=["距MA20低", "距MA20中", "距MA20高"],
                    duplicates="drop")
        if metric == "盈利":
            p = g.pivot_table(index=y, columns=x, values=metric,
                              aggfunc="mean") * 100
        else:
            p = g.pivot_table(index=y, columns=x, values=metric,
                              aggfunc="mean") * 100
        im = ax.imshow(p.values, cmap="RdYlGn", aspect="auto",
                       vmin=p.values.min() - 0.5, vmax=p.values.max() + 0.5)
        ax.set_xticks(range(len(p.columns)), p.columns, fontsize=9)
        ax.set_yticks(range(len(p.index)), p.index, fontsize=9)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, f"{p.values[i, j]:.1f}", ha="center", va="center",
                        fontsize=11, color="black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("追高 × 偏离均线：死叉卖出的胜率与收益", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "02_追高与偏离热力图.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_behavior(b: pd.DataFrame) -> None:
    x = b[b["阶段"] == "全样本"].copy()
    x = x[x["股性类型"].isin(["低波低量型", "低波活跃型", "高波低量型", "高波活跃型"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(x))
    ax.bar(xs - 0.2, x["胜率"], width=0.4, color="#4C78A8", label="胜率 %")
    ax.set_xticks(xs, x["股性类型"], rotation=20)
    ax.set_ylabel("胜率 %")
    ax2 = ax.twinx()
    ax2.bar(xs + 0.2, x["平均收益"], width=0.4, color="#E45756", label="平均收益 %")
    ax2.set_ylabel("平均收益 %", color="#E45756")
    ax2.tick_params(axis="y", colors="#E45756")
    ax2.axhline(0, color="#E45756", lw=0.6, ls=":")
    fig.suptitle("股性类型：胜率与死叉卖出平均收益（全样本）")
    fig.tight_layout()
    fig.savefig(FIG / "03_股性类型.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_path(g: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in zip(
            axes, ["峰值浮盈", "期间最低"],
            ["持有期最大浮盈分布", "持有期最大不利分布"]):
        for label, color in [("盈利单", "#4C78A8"), ("亏损单", "#E45756")]:
            x = g[g["净收益率"] > 0] if label == "盈利单" else g[g["净收益率"] <= 0]
            v = x[col].clip(-0.3, 0.4) * 100
            ax.hist(v, bins=60, alpha=0.55, label=label, color=color, density=True)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("%")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_盈亏路径分布.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    g = load()
    print(f"信号 {len(g):,}，年份 {g['年份'].min()}-{g['年份'].max()}", flush=True)

    uni, edges, quint = univariate(g)
    uni.to_csv(DATA / "买入点_单变量分位.csv", index=False, encoding="utf-8-sig")
    quint.to_csv(DATA / "买入点_五分位明细.csv", index=False, encoding="utf-8-sig")
    print("单变量完成", flush=True)

    st = by_stage(g, edges)
    st.to_csv(DATA / "买入点_分阶段.csv", index=False, encoding="utf-8-sig")

    ys = yearly_sign(g)
    ys.to_csv(DATA / "买入点_年度符号.csv", index=False, encoding="utf-8-sig")

    br = binary_rules(g)
    br.to_csv(DATA / "买入点_二元规则.csv", index=False, encoding="utf-8-sig")

    bh = behavior(g)
    bh.to_csv(DATA / "买入点_股性.csv", index=False, encoding="utf-8-sig")

    mk = market(g)
    mk.to_csv(DATA / "买入点_市场状态.csv", index=False, encoding="utf-8-sig")

    pt = path(g)
    pt.to_csv(DATA / "买入点_路径.csv", index=False, encoding="utf-8-sig")

    ex = exit_compare()
    ex.to_csv(DATA / "卖出点_配对比较.csv", index=False, encoding="utf-8-sig")

    plot_quintiles(g, edges, ["ret20", "dist_ma20", "amt5_60",
                              "atr20", "idx_ret20", "breadth_ma20"])
    plot_heatmap(g)
    plot_behavior(bh)
    plot_path(g)

    print("\nTop 15 单变量（按 q 值）：")
    cols = ["特征", "含义", "Q1胜率", "Q5胜率", "胜率差Q5-Q1",
            "Q1平均", "Q5平均", "收益差Q5-Q1", "Spearman", "q值"]
    print(uni[cols].head(15).round(2).to_string(index=False))
    print("\n卖出规则配对：")
    print(ex.round(2).to_string(index=False))
    print("\n股性（全样本）：")
    print(bh[bh["阶段"] == "全样本"].round(2).to_string(index=False))
    print(f"\n完成：{DATA} / {FIG}")


if __name__ == "__main__":
    main()
