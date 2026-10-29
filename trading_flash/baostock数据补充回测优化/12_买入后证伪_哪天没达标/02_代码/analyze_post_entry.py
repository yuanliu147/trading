#!/usr/bin/env python3
"""统计买入后“预期未达标/特征转弱”与后续下跌的关系。

全部为描述性统计。所有信号在 d 日收盘可知；早退按 d+1 开盘价，避免前视。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_结果"
PATH = OUT / "买入后逐日路径.parquet"

DAYS = [1, 2, 3, 5, 10]
KEY_DAYS = [1, 2, 3, 5, 10]
THRESHOLDS = [-0.05, -0.02, 0.0, 0.02, 0.03, 0.05, 0.08]
FLAGS = ["below_entry", "no_new_high_since_entry", "break_entry_low_close",
         "below_ma5", "ma5_down", "two_lower", "weak_close", "gap_fade",
         "long_upper", "heavy_down", "vol_price_div"]
CAND_ORDER = ["1-2", "3-9", "10-19", "20-49", "50-99", "100+"]


def cand_bucket(n) -> str:
    if pd.isna(n):
        return "未知"
    if n <= 2:
        return "1-2"
    if n <= 9:
        return "3-9"
    if n <= 19:
        return "10-19"
    if n <= 49:
        return "20-49"
    if n <= 99:
        return "50-99"
    return "100+"


FLAG_CN = {
    "below_entry": "收盘低于买入价",
    "no_new_high_since_entry": "未创买入后新高",
    "break_entry_low_close": "收盘跌破买入日最低",
    "below_ma5": "收盘低于5日线",
    "ma5_down": "5日线拐头向下",
    "two_lower": "连续两日低点下移",
    "weak_close": "收盘在当日振幅下半部",
    "gap_fade": "高开低走",
    "long_upper": "长上影线",
    "heavy_down": "放量下跌",
    "vol_price_div": "涨缩量/跌放量",
}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PATH)
    df = df.sort_values(["signal_date", "code", "d"]).reset_index(drop=True)
    g = df.groupby(["signal_date", "code"], sort=False)
    df["next1_ret"] = g["hfq_close"].shift(-1) / df["hfq_close"] - 1
    df["next3_ret"] = g["hfq_close"].shift(-3) / df["hfq_close"] - 1
    counts = pd.read_csv(OUT / "信号日候选数.csv", encoding="utf-8-sig", parse_dates=["信号日"])
    counts = counts.rename(columns={"信号日": "signal_date", "候选数": "n_candidates"})
    df = df.merge(counts, on="signal_date", how="left")
    df["候选级别"] = pd.Categorical(df.n_candidates.map(cand_bucket), categories=CAND_ORDER, ordered=True)
    df["period"] = np.where(df.signal_date < pd.Timestamp("2017-01-01"), "2005-2016", "2017-2026")
    df["score"] = df[["below_entry", "no_new_high_since_entry", "break_entry_low_close",
                      "below_ma5", "two_lower"]].sum(axis=1)
    df["comp"] = (df.below_entry & df.no_new_high_since_entry & df.below_ma5)
    # 评价口径：非实际成交用模拟最终结果（+20% 止盈/15日）；实际成交用逐笔净收益
    use_logged = df.is_actual & df.logged_ret.notna() & (df.logged_exit_reason != "eod")
    df["outcome_ret"] = np.where(use_logged, df.logged_ret, df.final_ret_sim)
    df["outcome_net"] = use_logged
    df["outcome_loss"] = df.outcome_ret < 0
    df["forward_eval"] = (1 + df.outcome_ret) / (1 + df.ret) - 1
    df["early_exit_eval"] = df.exit_next_open - df.outcome_ret - np.where(use_logged, 0.002, 0.0)
    df["tail10"] = df.outcome_ret <= -0.10
    df["tail15"] = df.outcome_ret <= -0.15
    # 信号段：同一代码，信号日间隔 <=3 个交易日合并
    all_dates = pd.DatetimeIndex(sorted(pd.unique(df.date)))
    dmap = {d: i for i, d in enumerate(all_dates)}
    rows = []
    for code, g in df.groupby("code"):
        g = g[["signal_date"]].drop_duplicates().sort_values("signal_date")
        idx = g.signal_date.map(dmap)
        ep = (idx.diff().fillna(999) > 3).cumsum().to_numpy()
        rows.append(pd.DataFrame({"code": code, "signal_date": g.signal_date.to_numpy(), "episode_no": ep}))
    ep = pd.concat(rows, ignore_index=True)
    df = df.merge(ep, on=["code", "signal_date"], how="left")
    df["episode"] = df.code + "_" + df.episode_no.astype(str)
    return df


def risk_set(df: pd.DataFrame, d: int, scope: str = "all", period: str = "all",
             use_logged: bool = False) -> pd.DataFrame:
    x = df[df.d == d].copy()
    x = x[~x.tp_hit_so_far]
    if scope == "actual":
        x = x[x.is_actual]
    elif scope == "other":
        x = x[~x.is_actual]
    if period != "all" and period != "全期":
        x = x[x.period == period]
    if use_logged:
        x = x[x.outcome_net & x.outcome_ret.notna()]
        x["_outcome"] = x.logged_ret
        x["_net"] = True
    else:
        x = x[x.final_ret_sim.notna()]
        x["_outcome"] = x.final_ret_sim
        x["_net"] = False
    x = x[x._outcome.notna()].copy()
    x["_loss"] = x._outcome < 0
    x["_forward"] = (1 + x._outcome) / (1 + x.ret) - 1
    x["_early"] = x.exit_next_open - x._outcome - np.where(x._net, 0.002, 0.0)
    x["_tail10"] = x._outcome <= -0.10
    return x


def cluster_se(g: pd.DataFrame, cluster: str = "episode") -> float:
    if len(g) == 0:
        return np.nan
    p = g._loss.mean()
    s = g.groupby(cluster)["_loss"].agg(["sum", "count"])
    return float(np.sqrt(((s["sum"] - p * s["count"]) ** 2).sum()) / len(g))


def summarize(g: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    if mask is None:
        mask = pd.Series(True, index=g.index)
    e = g[mask]
    ne = g[~mask]
    out = {
        "N": len(e),
        "占比": len(e) / len(g) if len(g) else np.nan,
        "最终亏损率": e._loss.mean() if len(e) else np.nan,
        "未触发亏损率": ne._loss.mean() if len(ne) else np.nan,
        "差值pp": (e._loss.mean() - ne._loss.mean()) * 100 if len(e) and len(ne) else np.nan,
        "深度亏损率": e._tail10.mean() if len(e) else np.nan,
        "之后5日再跌5%": e.future_decline_5.mean() if len(e) else np.nan,
        "次日下跌率": (e.next1_ret < 0).mean() if len(e) else np.nan,
        "剩余收益均值": e._forward.mean() if len(e) else np.nan,
        "之后止盈率": e.tp_any.mean() if len(e) else np.nan,
        "早退相对持有": e._early.mean() if len(e) else np.nan,
        "持有均值": e._outcome.mean() if len(e) else np.nan,
        "早退均值": (e.exit_next_open - np.where(e._net, 0.002, 0.0)).mean() if len(e) else np.nan,
        "聚类数": e.episode.nunique() if len(e) else 0,
        "亏损率_se": cluster_se(e) if len(e) else np.nan,
    }
    out["亏损率_CI下"] = max(0, out["最终亏损率"] - 1.96 * out["亏损率_se"]) if len(e) else np.nan
    out["亏损率_CI上"] = min(1, out["最终亏损率"] + 1.96 * out["亏损率_se"]) if len(e) else np.nan
    return out


def trade_level(sub: pd.DataFrame, use_logged: bool, period: str = "全期") -> pd.DataFrame:
    if period != "全期":
        sub = sub[sub.period == period]
    t = sub.groupby(["signal_date", "code"]).agg(
        outcome=("logged_ret" if use_logged else "final_ret_sim", "last"),
        exit_reason=("logged_exit_reason", "last"),
        tp=("tp_any", "max"),
        hold15=("hold15_ret", "first"),
        period=("period", "first"),
        episode=("episode", "first"),
        cand=("候选级别", "first"),
        entry=("entry_open", "first"),
        runmin=("runmin_close", "last"),
    ).reset_index()
    t["maxdd"] = t["runmin"] / t["entry"] - 1
    if use_logged:
        t = t[(t.exit_reason != "eod") & t.outcome.notna()]
    else:
        t = t[t.outcome.notna()]
    return t


def baseline_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            s = sub if period == "全期" else sub[sub.period == period]
            t = trade_level(sub, use_logged, period)
            rows.append({
                "样本": scope, "期间": period, "笔数": len(t),
                "信号日": s.signal_date.nunique(), "信号段": s.episode.nunique(),
                "胜率": (t.outcome > 0).mean(), "最终亏损率": (t.outcome < 0).mean(),
                "均值": t.outcome.mean(), "中位": t.outcome.median(),
                "止盈率": t.tp.mean(), "15日亏损率": (t.hold15 < 0).mean(),
                "平均最大浮亏": t.maxdd.mean(), "浮亏<=-10%": (t.maxdd <= -0.10).mean(),
            })
    sample = pd.DataFrame(rows)

    path_rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            for d in [0] + DAYS:
                x = risk_set(sub, d, period=period, use_logged=use_logged)
                if not len(x):
                    continue
                xa = sub[sub.d == d]
                if period != "全期":
                    xa = xa[xa.period == period]
                path_rows.append({
                    "样本": scope, "期间": period, "天数": d,
                    "全样本笔数": len(xa), "全样本路径均值": xa.ret.mean(),
                    "全样本高于买入价概率": (xa.ret > 0).mean(),
                    "全样本最终止盈占比": xa.tp_any.mean(),
                    "持仓笔数": len(x), "持仓路径均值": x.ret.mean(), "持仓路径中位": x.ret.median(),
                    "持仓高于买入价概率": (x.ret > 0).mean(),
                    "持仓最终亏损率": x._loss.mean(),
                    "剩余收益均值": x._forward.mean(),
                    "之后5日再跌5%": x.future_decline_5.mean(),
                    "次日下跌率": (x.next1_ret < 0).mean(),
                    "3日下跌率": (x.next3_ret < 0).mean(),
                })
    return sample, pd.DataFrame(path_rows)


def threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            for d in DAYS:
                g = risk_set(sub, d, period=period, use_logged=use_logged)
                for th in THRESHOLDS:
                    m = g.ret < th
                    r = summarize(g, m)
                    r.update({"样本": scope, "期间": period, "天数": d, "阈值": th})
                    rows.append(r)
    return pd.DataFrame(rows)


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            for d in DAYS:
                g = risk_set(sub, d, period=period, use_logged=use_logged)
                for flag in FLAGS + ["comp"]:
                    if flag == "comp":
                        m = g.comp
                    else:
                        m = g[flag].fillna(False)
                    r = summarize(g, m)
                    r.update({"样本": scope, "期间": period, "天数": d,
                              "规则": FLAG_CN.get(flag, "跌破买入价+未创新高+破5日线")})
                    rows.append(r)
    return pd.DataFrame(rows)


def score_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            for d in DAYS:
                g = risk_set(sub, d, period=period, use_logged=use_logged)
                for score in range(6):
                    m = g.score == score
                    r = summarize(g, m)
                    r.update({"样本": scope, "期间": period, "天数": d, "预警分": score})
                    rows.append(r)
    return pd.DataFrame(rows)


def composite_table(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "收益<-2%": lambda g: g.ret < -0.02,
        "收益<0": lambda g: g.ret < 0,
        "未达+3%": lambda g: g.ret < 0.03,
        "跌破买入价": lambda g: g.below_entry,
        "未创买入后新高": lambda g: g.no_new_high_since_entry,
        "跌破买入日最低": lambda g: g.break_entry_low_close,
        "收盘低于5日线": lambda g: g.below_ma5,
        "预警分>=4": lambda g: g.score >= 4,
        "跌破买入价+未创新高+破5日线": lambda g: g.comp,
    }
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            for d in KEY_DAYS:
                g = risk_set(sub, d, period=period, use_logged=use_logged)
                for name, fn in rules.items():
                    r = summarize(g, fn(g))
                    r.update({"样本": scope, "期间": period, "天数": d, "规则": name})
                    rows.append(r)
    return pd.DataFrame(rows)


def auc(y: pd.Series, s: pd.Series) -> float:
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["2005-2016", "2017-2026", "全期"]:
            for d in DAYS:
                x = risk_set(sub, d, period=period, use_logged=use_logged)
                if len(x) < 20 or x._loss.nunique() < 2:
                    continue
                rows.append({
                    "样本": scope, "期间": period, "天数": d, "笔数": len(x),
                    "基准亏损率": x._loss.mean(),
                    "AUC_收益": auc(x._loss, -x.ret),
                    "AUC_预警分": auc(x._loss, x.score),
                    "AUC_复合": auc(x._loss, x.comp.astype(int)),
                })
    return pd.DataFrame(rows)


def early_exit_table(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "收益<-2%": lambda g: g.ret < -0.02,
        "收益<0": lambda g: g.ret < 0,
        "未达+3%": lambda g: g.ret < 0.03,
        "跌破买入价": lambda g: g.below_entry,
        "未创买入后新高": lambda g: g.no_new_high_since_entry,
        "跌破买入日最低": lambda g: g.break_entry_low_close,
        "收盘低于5日线": lambda g: g.below_ma5,
        "预警分>=4": lambda g: g.score >= 4,
        "跌破买入价+未创新高+破5日线": lambda g: g.comp,
    }
    rows = []
    for scope, sub, use_logged in [("全部信号前二", df, False), ("实际成交", df[df.is_actual], True)]:
        for period in ["全期", "2017-2026"]:
            for d in KEY_DAYS:
                g = risk_set(sub, d, period=period, use_logged=use_logged)
                for name, fn in rules.items():
                    e = g[fn(g)]
                    if len(e) < 10:
                        continue
                    exit_net = e.exit_next_open - np.where(e._net, 0.002, 0.0)
                    rows.append({
                        "样本": scope, "期间": period, "天数": d, "规则": name, "笔数": len(e),
                        "持有均值": e._outcome.mean(), "早退均值": exit_net.mean(),
                        "早退-持有pp": e._early.mean() * 100,
                        "持有<=-10%": (e._outcome <= -0.10).mean(),
                        "早退<=-10%": (exit_net <= -0.10).mean(),
                        "持有<=-15%": (e._outcome <= -0.15).mean(),
                        "早退<=-15%": (exit_net <= -0.15).mean(),
                        "持有最差": e._outcome.min(), "早退最差": exit_net.min(),
                    })
    return pd.DataFrame(rows)


def trade_summary(df: pd.DataFrame) -> pd.DataFrame:
    """每笔一行，只保留关键天数和结果，便于 Excel 阅读。"""
    keys = ["is_actual", "signal_date", "code"]
    base = df.groupby(keys).agg(
        entry_date=("date", "first"), period=("period", "first"),
        rank=("rank", "first"), drawdown=("drawdown250_signal", "first"),
        n_days=("n_days", "max"), partial=("partial", "max"),
        tp=("tp_any", "max"), outcome=("final_ret_sim", "last"),
        logged=("logged_ret", "last"), episode=("episode", "first"),
    ).reset_index()
    sub = df[df.d.isin([1, 3, 5, 10])].pivot_table(
        index=keys, columns="d",
        values=["ret", "score", "below_entry", "no_new_high_since_entry"],
        aggfunc="first",
    )
    sub.columns = [f"{a}_{b}" for a, b in sub.columns]
    out = base.merge(sub.reset_index(), on=keys, how="left")
    out["样本"] = np.where(out.is_actual, "实际成交", "其余信号前二")
    rename = {
        "signal_date": "信号日", "code": "代码", "entry_date": "买入日",
        "period": "期间", "rank": "排名", "drawdown": "信号回撤",
        "n_days": "评估天数", "partial": "是否部分", "tp": "是否止盈",
        "outcome": "模拟最终收益", "logged": "实际净收益", "episode": "信号段",
        "ret_1": "第1天收益", "ret_3": "第3天收益", "ret_5": "第5天收益", "ret_10": "第10天收益",
        "score_1": "第1天预警分", "score_3": "第3天预警分", "score_5": "第5天预警分", "score_10": "第10天预警分",
        "below_entry_1": "第1天低于买入价", "below_entry_3": "第3天低于买入价",
        "below_entry_5": "第5天低于买入价", "below_entry_10": "第10天低于买入价",
        "no_new_high_since_entry_1": "第1天未创新高", "no_new_high_since_entry_3": "第3天未创新高",
        "no_new_high_since_entry_5": "第5天未创新高", "no_new_high_since_entry_10": "第10天未创新高",
    }
    out = out.rename(columns=rename)
    first = ["样本", "信号日", "代码", "买入日", "期间", "排名", "信号回撤", "评估天数",
             "是否部分", "是否止盈", "模拟最终收益", "实际净收益", "信号段"]
    rest = [c for c in out.columns if c not in first and c not in {"is_actual"}]
    out = out[first + rest].sort_values(["信号日", "代码"])
    return out



def candidate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ["2005-2016", "2017-2026", "全期"]:
        t = trade_level(df, use_logged=False, period=period)
        for b in CAND_ORDER:
            g = t[t.cand == b]
            if not len(g):
                continue
            rows.append({
                "期间": period, "候选级别": b, "信号日数": g.signal_date.nunique(),
                "交易数": len(g), "胜率": (g.outcome > 0).mean(),
                "最终亏损率": (g.outcome < 0).mean(), "均值": g.outcome.mean(),
                "中位": g.outcome.median(), "止盈率": g.tp.mean(),
                "平均最大浮亏": g.maxdd.mean(),
            })
    return pd.DataFrame(rows)


def candidate_path(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in range(15):
        x = df[df.d == d]
        for b in CAND_ORDER:
            g = x[x["候选级别"] == b]
            if not len(g):
                continue
            rows.append({
                "候选级别": b, "天数": d, "笔数": g.groupby(["signal_date", "code"]).ngroups,
                "路径均值": g.ret.mean(), "路径中位": g.ret.median(),
                "高于买入价概率": (g.ret > 0).mean(),
            })
    return pd.DataFrame(rows)


def candidate_warning(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [("收益<-2%", lambda g: g.ret < -0.02),
             ("三项复合", lambda g: g.comp)]
    for period in ["2005-2016", "2017-2026", "全期"]:
        for d in [3, 5]:
            x = risk_set(df, d, period=period)
            for b in CAND_ORDER:
                g = x[x["候选级别"] == b]
                if not len(g):
                    continue
                for name, fn in rules:
                    e = g[fn(g)]
                    if not len(e):
                        continue
                    se = cluster_se(e)
                    loss = e._loss.mean()
                    rows.append({
                        "期间": period, "天数": d, "候选级别": b, "规则": name,
                        "笔数": len(g), "触发笔数": len(e), "触发率": len(e) / len(g),
                        "最终亏损率": loss, "同桶基准亏损率": g._loss.mean(),
                        "差值pp": (loss - g._loss.mean()) * 100,
                        "深度亏损率": e._tail10.mean(),
                        "之后止盈率": e.tp_any.mean(),
                        "早退相对持有": e._early.mean(),
                        "95%置信下": max(0, loss - 1.96 * se),
                        "95%置信上": min(1, loss + 1.96 * se),
                    })
    return pd.DataFrame(rows)

def main() -> None:
    df = load_data()
    sample, path = baseline_tables(df)
    thresholds = threshold_table(df)
    features = feature_table(df)
    scores = score_table(df)
    composites = composite_table(df)
    early = early_exit_table(df)
    aucs = auc_table(df)
    cand_base = candidate_baseline(df)
    cand_path = candidate_path(df)
    cand_warn = candidate_warning(df)
    trades = trade_summary(df)

    for name, table in [
        ("基准_样本.csv", sample), ("基准_逐日.csv", path), ("收益阈值.csv", thresholds),
        ("结构特征.csv", features), ("预警评分.csv", scores), ("组合规则.csv", composites),
        ("早退反事实.csv", early), ("预测力_AUC.csv", aucs), ("逐笔摘要.csv", trades),
        ("候选数量_基准.csv", cand_base), ("候选数量_逐日路径.csv", cand_path),
        ("候选数量_预警.csv", cand_warn),
    ]:
        table.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    # 控制台摘要
    pd.set_option("display.width", 220)
    print("=== 基准 ===")
    print(sample.round(3).to_string(index=False))
    print("\n=== D+5 关键规则（全部信号前二，全期）===")
    c = composites[(composites["样本"] == "全部信号前二") & (composites["期间"] == "全期") & (composites["天数"] == 5)]
    print(c[["规则", "N", "占比", "最终亏损率", "未触发亏损率", "差值pp", "深度亏损率",
             "次日下跌率", "剩余收益均值", "早退相对持有", "亏损率_CI下", "亏损率_CI上"]].round(3).to_string(index=False))
    print("\n=== D+3 关键规则（全部信号前二，全期）===")
    c = composites[(composites["样本"] == "全部信号前二") & (composites["期间"] == "全期") & (composites["天数"] == 3)]
    print(c[["规则", "N", "占比", "最终亏损率", "未触发亏损率", "差值pp", "深度亏损率",
             "次日下跌率", "剩余收益均值", "早退相对持有", "亏损率_CI下", "亏损率_CI上"]].round(3).to_string(index=False))
    print("\n=== D+5 收益阈值（全部信号前二，全期）===")
    t = thresholds[(thresholds["样本"] == "全部信号前二") & (thresholds["期间"] == "全期") & (thresholds["天数"] == 5)]
    print(t[["阈值", "N", "占比", "最终亏损率", "未触发亏损率", "差值pp", "深度亏损率",
             "次日下跌率", "剩余收益均值", "早退相对持有"]].round(3).to_string(index=False))
    print("\n=== 预警分（全部信号前二，全期）===")
    s = scores[(scores["样本"] == "全部信号前二") & (scores["期间"] == "全期") & (scores["天数"].isin([3, 5]))]
    print(s[["天数", "预警分", "N", "最终亏损率", "深度亏损率", "次日下跌率", "剩余收益均值"]].round(3).to_string(index=False))
    print("\n=== 早退反事实（实际成交，全期，D+5）===")
    e = early[(early["样本"] == "实际成交") & (early["期间"] == "全期") & (early["天数"] == 5)]
    print(e[["规则", "笔数", "持有均值", "早退均值", "早退-持有pp", "持有<=-10%", "早退<=-10%",
             "持有<=-15%", "早退<=-15%"]].round(3).to_string(index=False))
    print("\n=== 候选数量：基准（全期）===")
    print(cand_base[cand_base["期间"] == "全期"].round(3).to_string(index=False))
    print("\n=== 候选数量：D+5 收益<-2%（全期）===")
    cw = cand_warn[(cand_warn["期间"] == "全期") & (cand_warn["天数"] == 5) & (cand_warn["规则"] == "收益<-2%")]
    print(cw.round(3).to_string(index=False))
    print("\n=== AUC（全部信号前二）===")
    a = aucs[(aucs["样本"] == "全部信号前二")]
    print(a.pivot_table(index="天数", columns="期间", values=["AUC_收益", "AUC_预警分"]).round(3).to_string())

    meta = {
        "path_rows": len(df), "trades_all": df.groupby(["signal_date", "code"]).ngroups,
        "trades_actual": df[df.is_actual].groupby(["signal_date", "code"]).ngroups,
        "episodes": df.episode.nunique(), "signal_days": df.signal_date.nunique(),
        "start": str(df.date.min().date()), "end": str(df.date.max().date()),
    }
    (OUT / "口径.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n口径:", meta)


if __name__ == "__main__":
    main()
