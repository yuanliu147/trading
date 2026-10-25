# -*- coding: utf-8 -*-
"""250 日涨幅（信号日前 250 个交易日）与单笔收益的关系。

只使用本地同花顺 DuckDB 派生数据：
- research_panel.parquet 里的 return_250d / volatility20 / amount20 / drawdown250
- 07 的前二 450 笔、母路径 99 笔、08 的全部信号 4,019 笔

输出到 10_250日涨幅与收益/03_结果。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
RES = BASE / "10_250日涨幅与收益" / "03_结果"
PANEL = BASE.parent / "2026-10-24_主板恐慌反转_年化30" / "01_数据" / "research_panel.parquet"
TOP2 = BASE / "07_市场阶段适配" / "03_结果" / "全部信号日逐只_带阶段.csv"
MOTHER = BASE.parent / "2026-10-24_主板恐慌反转_年化30" / "03_结果" / "最终_逐笔.csv"
ALL = BASE / "08_重复信号与买入时点" / "03_结果" / "全部信号重复序列_完整.csv"
CURRENT = BASE / "09_信号日回撤深度与收益" / "03_结果" / "当前信号_深度对照.csv"

BINS = [0, 0.10, 0.20, 0.50, 1.00, 2.00, np.inf]
BIN_LABELS = ["0–10%", "10–20%", "20–50%", "50–100%", "100–200%", ">200%"]
RNG = np.random.default_rng(20261025)
CN = {
    "n": "笔数", "mean": "单笔均值", "median": "单笔中位", "win": "胜率",
    "target": "止盈率", "p10": "p10", "p90": "p90", "worst": "最差",
    "best": "最好", "avg_dd": "平均回撤", "avg_vol": "平均个股波动",
    "ex_n": "剔除2026笔数", "ex_mean": "剔除2026均值", "ex_win": "剔除2026胜率",
    "ex_target": "剔除2026止盈率", "y26_n": "2026笔数", "y26_mean": "2026均值",
    "y26_win": "2026胜率", "y26_target": "2026止盈率",
    "r250均值": "250日涨幅均值", "r250中位": "250日涨幅中位",
    "平均波动": "平均个股波动",
    "Pearson": "Pearson相关", "Pearson_p": "Pearson p",
    "Spearman": "Spearman相关", "Spearman_p": "Spearman p",
}


def write(df: pd.DataFrame, name: str) -> None:
    df.rename(columns=CN).to_csv(RES / name, index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------- 数据
panel = pd.read_parquet(
    PANEL,
    columns=[
        "thscode", "date", "hfq_close", "return_250d", "volatility20",
        "amount20", "drawdown250", "rsi14", "close_location",
        "market_volatility20", "return_60d",
    ],
)
panel["date"] = pd.to_datetime(panel["date"])


def attach(z: pd.DataFrame, code_col: str, date_col: str) -> pd.DataFrame:
    z = z.merge(
        panel,
        left_on=[code_col, date_col],
        right_on=["thscode", "date"],
        how="left",
        suffixes=("", "_p"),
    )
    z = z.rename(columns={"return_250d": "r250"})
    z["target"] = (z["exit_reason"] == "target").astype(int)
    z["year"] = pd.to_datetime(z["signal_date"]).dt.year
    z["r250_bin"] = pd.cut(z["r250"], BINS, labels=BIN_LABELS, include_lowest=True)
    return z


top2 = pd.read_csv(TOP2, parse_dates=["signal_date", "entry_date", "exit_date"])
top2 = top2[top2["complete"] == True].copy()
top2 = attach(top2, "code", "signal_date")
top2["phase"] = top2["阶段"].str[0]
top2["log_amount"] = np.log(top2["amount20"].clip(lower=1))

mother = pd.read_csv(MOTHER, parse_dates=["signal_date", "entry_date", "exit_date"])
mother = mother[mother["ret"].notna()].copy()
mother = attach(mother, "thscode", "signal_date")
mother["phase"] = mother["signal_date"].map(
    top2.drop_duplicates("signal_date").set_index("signal_date")["phase"]
)

all_sig = pd.read_csv(ALL, parse_dates=["signal_date"])
all_sig = all_sig[all_sig["complete"] == True].copy()
all_sig = attach(all_sig, "code", "signal_date")
all_sig["log_amount"] = np.log(all_sig["amount20"].clip(lower=1))

# 校验定义：return_250d = hfq_close / 250 个交易日前 hfq_close - 1
chk = panel.sort_values(["thscode", "date"]).copy()
chk["calc"] = chk.groupby("thscode")["hfq_close"].pct_change(250)
chk = chk.dropna(subset=["return_250d", "calc"])
max_diff = float((chk["return_250d"] - chk["calc"]).abs().max())

assert len(top2) == 450, len(top2)
assert top2["r250"].notna().all()
assert len(mother) == 99, len(mother)
assert mother["r250"].notna().all()
assert len(all_sig) == 4019, len(all_sig)
assert all_sig["r250"].notna().all()
assert max_diff < 1e-12, max_diff


# ----------------------------------------------------------------------------- 统计
def summarize(z: pd.DataFrame) -> dict:
    z = z.dropna(subset=["ret", "r250"])
    ex = z[z["year"] < 2026]
    y26 = z[z["year"] == 2026]
    return {
        "n": int(len(z)),
        "mean": float(z["ret"].mean()),
        "median": float(z["ret"].median()),
        "win": float((z["ret"] > 0).mean()),
        "target": float((z["exit_reason"] == "target").mean()),
        "p10": float(z["ret"].quantile(0.10)),
        "p90": float(z["ret"].quantile(0.90)),
        "worst": float(z["ret"].min()),
        "best": float(z["ret"].max()),
        "avg_dd": float(z["drawdown250"].mean()),
        "avg_vol": float(z["volatility20"].mean()),
        "ex_n": int(len(ex)),
        "ex_mean": float(ex["ret"].mean()) if len(ex) else np.nan,
        "ex_win": float((ex["ret"] > 0).mean()) if len(ex) else np.nan,
        "ex_target": float((ex["exit_reason"] == "target").mean()) if len(ex) else np.nan,
        "y26_n": int(len(y26)),
        "y26_mean": float(y26["ret"].mean()) if len(y26) else np.nan,
        "y26_win": float((y26["ret"] > 0).mean()) if len(y26) else np.nan,
        "y26_target": float((y26["exit_reason"] == "target").mean()) if len(y26) else np.nan,
    }


def bin_table(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BIN_LABELS:
        g = z[z["r250_bin"] == b]
        if not len(g):
            continue
        s = summarize(g)
        rows.append({"涨幅档": b, **s})
    return pd.DataFrame(rows)


def quintile_table(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy()
    z["q"] = pd.qcut(z["r250"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rows = []
    for q, g in z.groupby("q", observed=True):
        s = summarize(g)
        rows.append(
            {
                "五分位": q,
                "最小": float(g["r250"].min()),
                "最大": float(g["r250"].max()),
                **s,
            }
        )
    return pd.DataFrame(rows)


def corr_row(z: pd.DataFrame, sample: str) -> dict:
    z = z.dropna(subset=["ret", "r250"])
    pear = stats.pearsonr(z["ret"], z["r250"])
    spear = stats.spearmanr(z["ret"], z["r250"])
    return {
        "样本": sample,
        "笔数": int(len(z)),
        "Pearson": float(pear.statistic),
        "Pearson_p": float(pear.pvalue),
        "Spearman": float(spear.statistic),
        "Spearman_p": float(spear.pvalue),
    }


def design(z: pd.DataFrame, controls: bool, fe: bool) -> tuple[np.ndarray, list[str]]:
    cols = [z[["r250"]].values]
    names = ["r250"]
    if controls:
        cols.append(
            z[["drawdown250", "volatility20", "log_amount", "rsi14",
               "close_location", "market_volatility20"]].values
        )
        names += ["drawdown250", "volatility20", "log_amount", "rsi14",
                  "close_location", "market_volatility20"]
    if fe:
        for y in sorted(z["year"].unique())[1:]:
            cols.append((z["year"] == y).astype(float).values.reshape(-1, 1))
            names.append(f"year_{y}")
    return np.hstack(cols), names


def boot_coef(z: pd.DataFrame, controls: bool = False, fe: bool = True,
              reps: int = 300) -> tuple[float, float, float, float]:
    z = z.dropna(subset=["ret", "r250"]).reset_index(drop=True)
    X, _ = design(z, controls, fe)
    y = z["ret"].values
    model = LinearRegression().fit(X, y)
    coef = float(model.coef_[0] * 100)
    r2 = float(model.score(X, y))
    co = []
    for _ in range(reps):
        idx = RNG.integers(0, len(z), len(z))
        co.append(float(LinearRegression().fit(X[idx], y[idx]).coef_[0] * 100))
    lo, hi = np.percentile(co, [2.5, 97.5])
    return coef, float(lo), float(hi), r2


def boot_or(z: pd.DataFrame, controls: bool = False, fe: bool = True,
            reps: int = 300) -> tuple[float, float, float, float]:
    z = z.dropna(subset=["target", "r250"]).reset_index(drop=True)
    X, _ = design(z, controls, fe)
    y = z["target"].values
    model = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    odds = float(np.exp(model.coef_[0][0]))
    auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    co = []
    for _ in range(reps):
        idx = RNG.integers(0, len(z), len(z))
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        try:
            co.append(float(np.exp(
                LogisticRegression(C=np.inf, max_iter=5000).fit(X[idx], yy).coef_[0][0]
            )))
        except Exception:
            continue
    lo, hi = np.percentile(co, [2.5, 97.5])
    return odds, float(lo), float(hi), auc


# ----------------------------------------------------------------------------- 样本与分档
sample_rows = [
    {"样本": "前二完整", "笔数": len(top2), "起始": str(top2.signal_date.min().date()),
     "结束": str(top2.signal_date.max().date()),
     "r250均值": top2.r250.mean(), "r250中位": top2.r250.median(),
     "单笔均值": top2.ret.mean(), "单笔中位": top2.ret.median(),
     "胜率": (top2.ret > 0).mean(), "止盈率": (top2.exit_reason == "target").mean()},
    {"样本": "母路径99笔", "笔数": len(mother), "起始": str(mother.signal_date.min().date()),
     "结束": str(mother.signal_date.max().date()),
     "r250均值": mother.r250.mean(), "r250中位": mother.r250.median(),
     "单笔均值": mother.ret.mean(), "单笔中位": mother.ret.median(),
     "胜率": (mother.ret > 0).mean(), "止盈率": (mother.exit_reason == "target").mean()},
    {"样本": "全部信号完整", "笔数": len(all_sig), "起始": str(all_sig.signal_date.min().date()),
     "结束": str(all_sig.signal_date.max().date()),
     "r250均值": all_sig.r250.mean(), "r250中位": all_sig.r250.median(),
     "单笔均值": all_sig.ret.mean(), "单笔中位": all_sig.ret.median(),
     "胜率": (all_sig.ret > 0).mean(), "止盈率": (all_sig.exit_reason == "target").mean()},
]
sample_df = pd.DataFrame(sample_rows)
write(sample_df, "样本说明.csv")

bin_top2 = bin_table(top2)
write(bin_top2, "六档成绩_前二.csv")
bin_all = bin_table(all_sig)
write(bin_all, "六档成绩_全部信号.csv")
bin_mother = bin_table(mother)
write(bin_mother, "六档成绩_母路径.csv")

q_top2 = quintile_table(top2)
write(q_top2, "五分位_前二.csv")

# ----------------------------------------------------------------------------- 相关性
corr_rows = [
    corr_row(top2, "前二全部"),
    corr_row(top2[top2.year < 2026], "前二_剔除2026"),
    corr_row(top2[top2.year == 2026], "前二_2026"),
    corr_row(mother, "母路径99笔"),
    corr_row(mother[mother.year < 2026], "母路径_剔除2026"),
    corr_row(all_sig, "全部信号"),
    corr_row(all_sig[all_sig.year < 2026], "全部信号_剔除2026"),
    corr_row(all_sig[all_sig.year == 2026], "全部信号_2026"),
]
corr = pd.DataFrame(corr_rows)
write(corr, "相关性.csv")

# ----------------------------------------------------------------------------- 回归
reg_rows = []
for label, z, controls, fe in [
    ("前二_全部_均值_年度FE", top2, False, True),
    ("前二_全部_均值_年度FE+控制", top2, True, True),
    ("前二_剔除2026_均值_年度FE", top2[top2.year < 2026], False, True),
    ("前二_2026_均值", top2[top2.year == 2026], False, False),
    ("母路径_均值_年度FE", mother, False, True),
    ("全部信号_均值_年度FE", all_sig, False, True),
]:
    coef, lo, hi, r2 = boot_coef(z, controls, fe, reps=300 if len(z) <= 500 else 200)
    reg_rows.append({"模型": label, "指标": "每+100pp的收益(pp)", "估计": coef,
                     "CI低": lo, "CI高": hi, "拟合": r2, "笔数": len(z)})

for label, z, controls, fe in [
    ("前二_全部_止盈_年度FE", top2, False, True),
    ("前二_全部_止盈_年度FE+控制", top2, True, True),
    ("前二_剔除2026_止盈_年度FE", top2[top2.year < 2026], False, True),
    ("前二_2026_止盈", top2[top2.year == 2026], False, False),
    ("母路径_止盈_年度FE", mother, False, True),
    ("全部信号_止盈_年度FE", all_sig, False, True),
]:
    odds, lo, hi, auc = boot_or(z, controls, fe, reps=300 if len(z) <= 500 else 200)
    reg_rows.append({"模型": label, "指标": "每+100pp的止盈OR", "估计": odds,
                     "CI低": lo, "CI高": hi, "拟合": auc, "笔数": len(z)})

reg = pd.DataFrame(reg_rows)
write(reg, "回归.csv")

# ----------------------------------------------------------------------------- 年度 / 阶段 / 深度 / 波动
year_rows = []
for y, g in top2.groupby("year"):
    r = stats.pearsonr(g.ret, g.r250)
    s = summarize(g)
    year_rows.append({
        "年份": int(y), "笔数": int(len(g)), "r250中位": g.r250.median(),
        "单笔均值": s["mean"], "单笔中位": s["median"], "胜率": s["win"],
        "止盈率": s["target"], "Pearson": float(r.statistic), "p值": float(r.pvalue),
    })
year_df = pd.DataFrame(year_rows)
write(year_df, "年度相关性_前二.csv")

med = top2.r250.median()
top2["r250_high"] = top2.r250 > med
phase_rows = []
for (ph, hi), g in top2.groupby(["phase", "r250_high"]):
    s = summarize(g)
    phase_rows.append({"阶段": ph, "250日涨幅": "高于中位" if hi else "低于中位",
                       **s})
phase_df = pd.DataFrame(phase_rows)
write(phase_df, "阶段拆分_前二.csv")

top2["deep"] = top2.drawdown250 < -0.65
depth_rows = []
for (deep, hi), g in top2.groupby(["deep", "r250_high"]):
    s = summarize(g)
    depth_rows.append({"回撤档": "≤-65%" if deep else ">-65%",
                       "250日涨幅": "高于中位" if hi else "低于中位", **s})
depth_df = pd.DataFrame(depth_rows)
write(depth_df, "深度拆分_前二.csv")

top2["vol_q"] = pd.qcut(top2.volatility20, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
vol_rows = []
for q, g in top2.groupby("vol_q", observed=True):
    s = summarize(g)
    vol_rows.append({"波动五分位": q, "平均波动": g.volatility20.mean(),
                     "r250均值": g.r250.mean(), **s})
vol_df = pd.DataFrame(vol_rows).drop(columns=["avg_vol"])
write(vol_df, "波动五分位_前二.csv")

# ----------------------------------------------------------------------------- 逐笔
top2_out = top2[[
    "signal_date", "code", "rank", "year", "phase", "dd250", "r250",
    "volatility20", "ret", "exit_reason", "target", "r250_bin",
    "entry_date", "exit_date", "持有期市场",
]].rename(columns={
    "signal_date": "信号日", "code": "代码", "rank": "排名", "year": "年份",
    "phase": "阶段", "dd250": "250日回撤", "r250": "250日涨幅",
    "volatility20": "个股波动率", "ret": "单笔收益", "exit_reason": "退出原因",
    "target": "是否止盈", "r250_bin": "涨幅档", "entry_date": "入场日",
    "exit_date": "退出日", "持有期市场": "持有期市场",
})
top2_out.to_csv(RES / "逐笔_前二_带250日涨幅.csv", index=False, encoding="utf-8-sig")

mother_out = mother[[
    "signal_date", "thscode", "rk", "year", "phase", "drawdown250", "r250",
    "volatility20", "ret", "exit_reason", "target", "r250_bin", "days",
]].rename(columns={
    "signal_date": "信号日", "thscode": "代码", "rk": "排名", "year": "年份",
    "phase": "阶段", "drawdown250": "250日回撤", "r250": "250日涨幅",
    "volatility20": "个股波动率", "ret": "单笔收益", "exit_reason": "退出原因",
    "target": "是否止盈", "r250_bin": "涨幅档", "days": "持有交易日",
})
mother_out.to_csv(RES / "逐笔_母路径_带250日涨幅.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------------- 当前信号
cur = pd.read_csv(CURRENT)
cur["r250"] = np.nan
for i, row in cur.iterrows():
    d = pd.to_datetime(row["信号日"])
    x = panel[(panel.thscode == row["代码"]) & (panel.date == d)]
    if len(x):
        cur.loc[i, "r250"] = x.return_250d.iloc[0]
cur["涨幅档"] = pd.cut(cur["r250"], BINS, labels=BIN_LABELS, include_lowest=True)
cur = cur.merge(
    bin_top2[["涨幅档", "n", "mean", "win", "target"]],
    on="涨幅档", how="left",
)
cur_out = cur[[
    "信号日", "排名", "代码", "信号日回撤_后复权", "r250", "涨幅档",
    "n", "mean", "win", "target", "备注",
]].rename(columns={
    "信号日回撤_后复权": "250日回撤(%)", "r250": "250日涨幅",
    "n": "同档笔数", "mean": "同档均值", "win": "同档胜率",
    "target": "同档止盈率",
})
cur_out["250日涨幅(%)"] = cur_out["250日涨幅"] * 100
cur_out["同档均值(%)"] = cur_out["同档均值"] * 100
cur_out["同档胜率(%)"] = cur_out["同档胜率"] * 100
cur_out["同档止盈率(%)"] = cur_out["同档止盈率"] * 100
cur_out = cur_out[[
    "信号日", "排名", "代码", "250日回撤(%)", "250日涨幅", "250日涨幅(%)",
    "涨幅档", "同档笔数", "同档均值", "同档均值(%)", "同档胜率",
    "同档胜率(%)", "同档止盈率", "同档止盈率(%)", "备注",
]]
cur_out.to_csv(RES / "当前信号_250日涨幅.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------------- 指标 JSON
metrics = {
    "样本": {
        "前二": len(top2),
        "母路径": len(mother),
        "全部信号": len(all_sig),
        "前二起始": str(top2.signal_date.min().date()),
        "前二结束": str(top2.signal_date.max().date()),
        "定义校验最大差": max_diff,
    },
    "相关性": {r["样本"]: {"n": int(r["笔数"]), "pearson": float(r["Pearson"]),
                         "p": float(r["Pearson_p"]), "spearman": float(r["Spearman"]),
                         "spearman_p": float(r["Spearman_p"])}
               for _, r in corr.iterrows()},
    "回归": {r["模型"]: {"估计": float(r["估计"]), "CI低": float(r["CI低"]),
                        "CI高": float(r["CI高"]), "拟合": float(r["拟合"]),
                        "n": int(r["笔数"])} for _, r in reg.iterrows()},
    "前二": {
        "r250均值": float(top2.r250.mean()),
        "r250中位": float(top2.r250.median()),
        "涨幅档": {r["涨幅档"]: {"n": int(r["n"]), "mean": float(r["mean"]),
                               "median": float(r["median"]), "win": float(r["win"]),
                               "target": float(r["target"]), "p10": float(r["p10"]),
                               "p90": float(r["p90"]), "worst": float(r["worst"])}
                   for _, r in bin_top2.iterrows()},
    },
    "当前": {r["代码"]: {"r250": float(r["r250"]), "档": str(r["涨幅档"])}
             for _, r in cur.iterrows()},
}
(RES / "指标.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

print("样本：", len(top2), len(mother), len(all_sig))
print("定义最大差：", max_diff)
print("\n相关性：\n", corr.round(3).to_string(index=False))
print("\n六档：\n", bin_top2[["涨幅档", "n", "mean", "median", "win", "target", "p10", "p90", "ex_mean", "ex_win", "y26_mean", "y26_win"]].round(3).to_string(index=False))
print("\n回归：\n", reg.round(3).to_string(index=False))
print("\n年度：\n", year_df.round(3).to_string(index=False))
print("\n当前：\n", cur_out.round(2).to_string(index=False))
