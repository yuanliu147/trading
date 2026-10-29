#!/usr/bin/env python3
"""买入后 D+3 三态：上涨 / 中震荡 / 下跌，之后怎么走。

定义（D+3 收盘，且尚未止盈）：
  上涨    ret3 >= +3%
  中震荡  -2% < ret3 < +3%
  下跌    ret3 <= -2%

所有规则只用 D+3 / D+5 收盘及之前的数据；早退按 D+6 开盘。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "03_结果"
PATH = RES / "买入后逐日路径.parquet"
MARKET = BASE.parent / "01_数据" / "market_daily.parquet"

UP = 0.03
DOWN = -0.02
STATES = ["上涨", "中震荡", "下跌"]
PERIODS = ["2005-2016", "2017-2026", "全期"]
RULE_ORDER = {
    "上涨": ["D+5 回吐到成本下", "D+5 距 D+0–5 最高回撤≥10%", "D+5 比 D+3 收盘回吐≥5pp",
             "D+5 收盘跌破 D+3 最低", "D+5 未创新高（相对 D+3）", "D+5 < +3%"],
    "中震荡": ["D+5 < -2%", "D+5 < -5%", "D+5 预警分≥4", "D+5 三项复合",
              "D+5 跌破买入日低点", "D+5 < 0"],
    "下跌": ["D+5 < -5%", "D+5 预警分≥4", "D+5 三项复合", "D+5 比 D+3 再跌≥2pp",
             "D+5 低点低于 D+3 低点", "D+5 < -2%"],
}


def load_path() -> pd.DataFrame:
    df = pd.read_parquet(PATH)
    df = df.sort_values(["signal_date", "code", "d"]).reset_index(drop=True)
    df["period"] = np.where(df.signal_date < pd.Timestamp("2017-01-01"), "2005-2016", "2017-2026")
    df["score"] = df[["below_entry", "no_new_high_since_entry", "break_entry_low_close",
                      "below_ma5", "two_lower"]].sum(axis=1)
    df["comp"] = df.below_entry & df.no_new_high_since_entry & df.below_ma5
    use_logged = df.is_actual & df.logged_ret.notna() & (df.logged_exit_reason != "eod")
    df["outcome_net"] = use_logged
    return add_episode(df)


def add_episode(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(pd.unique(df.date)))
    dmap = {d: i for i, d in enumerate(dates)}
    rows = []
    for code, g in df.groupby("code"):
        g = g[["signal_date"]].drop_duplicates().sort_values("signal_date")
        idx = g.signal_date.map(dmap)
        ep = (idx.diff().fillna(999) > 3).cumsum().to_numpy()
        rows.append(pd.DataFrame({"code": code, "signal_date": g.signal_date.to_numpy(),
                                  "episode_no": ep}))
    ep = pd.concat(rows, ignore_index=True)
    df = df.merge(ep, on=["code", "signal_date"], how="left")
    df["episode"] = df.code + "_" + df.episode_no.astype(str)
    return df


def make_trades(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["signal_date", "code"]
    t = df.groupby(keys).agg(
        period=("period", "first"), is_actual=("is_actual", "max"),
        outcome_sim=("final_ret_sim", "last"), outcome_logged=("logged_ret", "last"),
        net=("outcome_net", "max"), episode=("episode", "first"),
        entry=("entry_open", "first"), runmin=("runmin_close", "last"),
        tp_any=("tp_any", "max"), exit_reason=("logged_exit_reason", "last"),
    ).reset_index()
    t["maxdd"] = t.runmin / t.entry - 1

    d3 = df[df.d == 3][["signal_date", "code", "ret", "tp_hit_so_far", "hfq_high", "hfq_low"]]
    d3 = d3.rename(columns={"ret": "ret3", "tp_hit_so_far": "tp3",
                            "hfq_high": "high3", "hfq_low": "low3"})
    d5_cols = ["signal_date", "code", "ret", "hfq_high", "hfq_low", "hfq_close",
               "score", "comp", "break_entry_low_close", "tp_hit_so_far", "exit_next_open"]
    d5 = df[df.d == 5][d5_cols].rename(columns={
        "ret": "ret5", "hfq_high": "high5", "hfq_low": "low5", "hfq_close": "close5",
        "score": "score5", "comp": "comp5", "break_entry_low_close": "break5",
        "tp_hit_so_far": "tp5", "exit_next_open": "exit5",
    })
    mh = df[df.d <= 5].groupby(keys).hfq_high.max().rename("maxhigh5").reset_index()
    t = t.merge(d3, on=keys, how="left").merge(d5, on=keys, how="left").merge(mh, on=keys, how="left")
    t["state"] = np.select([t.ret3 >= UP, t.ret3 <= DOWN], ["上涨", "下跌"], default="中震荡")
    t["held3"] = ~t.tp3.fillna(False).astype(bool)
    t["held5"] = ~t.tp5.fillna(False).astype(bool)
    return t


def cluster_se(g: pd.DataFrame, cluster: str = "episode") -> float:
    if len(g) == 0:
        return np.nan
    p = g._loss.mean()
    s = g.groupby(cluster)["_loss"].agg(["sum", "count"])
    return float(np.sqrt(((s["sum"] - p * s["count"]) ** 2).sum()) / len(g))


def eval_base(t: pd.DataFrame, sample: str, period: str) -> pd.DataFrame:
    """全部信号前二=模拟最终结果；实际成交=逐笔净收益。"""
    x = t.copy()
    if sample == "实际成交":
        x = x[x.is_actual & x.net & (x.exit_reason != "eod")]
        x["_outcome"] = x.outcome_logged
        x["_net"] = True
    else:
        x["_outcome"] = x.outcome_sim
        x["_net"] = False
    x = x[x.held3 & x._outcome.notna()]
    if period != "全期":
        x = x[x.period == period]
    x = x.copy()
    x["_loss"] = x._outcome < 0
    x["_tail10"] = x._outcome <= -0.10
    x["_fwd5"] = (1 + x._outcome) / (1 + x.ret5) - 1
    x["_early5"] = x.exit5 - x._outcome - np.where(x._net, 0.002, 0.0)
    return x


def baseline(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in ["全部信号前二", "实际成交"]:
        periods = PERIODS
        for period in periods:
            x = eval_base(t, sample, period)
            if not len(x):
                continue
            for st in STATES:
                g = x[x.state == st]
                if not len(g):
                    continue
                p = g._loss.mean()
                se = cluster_se(g)
                rows.append({
                    "样本": sample, "期间": period, "状态": st, "笔数": len(g),
                    "占比": len(g) / len(x),
                    "最终亏损率": p, "深度亏损率": g._tail10.mean(),
                    "均值": g._outcome.mean(), "中位": g._outcome.median(),
                    "止盈率": g.tp_any.mean(), "平均最大浮亏": g.maxdd.mean(),
                    "D+3均值": g.ret3.mean(), "D+5均值": g.ret5.mean(),
                    "信号段数": g.episode.nunique(),
                    "亏损率_se": se,
                    "95%置信下": max(0, p - 1.96 * se),
                    "95%置信上": min(1, p + 1.96 * se),
                })
    return pd.DataFrame(rows)


def path_table(t: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    base = t[t.held3 & t.outcome_sim.notna()][["signal_date", "code", "state", "ret3"]]
    x = df.merge(base, on=["signal_date", "code"], how="inner")
    rows = []
    for st in STATES:
        g = x[x.state == st]
        for d in range(15):
            y = g[g.d == d]
            if not len(y):
                continue
            rows.append({
                "状态": st, "天数": d, "笔数": y.groupby(["signal_date", "code"]).ngroups,
                "路径均值": y.ret.mean(), "路径中位": y.ret.median(),
                "高于买入价概率": (y.ret > 0).mean(),
                "高于D+3收盘概率": (y.ret > y.ret3).mean() if d >= 3 else np.nan,
            })
    return pd.DataFrame(rows)


def rule_defs() -> dict[str, list[tuple[str, str]]]:
    return {
        "上涨": [
            ("D+5 回吐到成本下", "ret5", lambda g: g.ret5 < 0),
            ("D+5 距 D+0–5 最高回撤≥10%", "ret5", lambda g: g.close5 / g.maxhigh5 - 1 <= -0.10),
            ("D+5 比 D+3 收盘回吐≥5pp", "ret5", lambda g: g.ret5 < g.ret3 - 0.05),
            ("D+5 收盘跌破 D+3 最低", "ret5", lambda g: g.close5 < g.low3),
            ("D+5 未创新高（相对 D+3）", "ret5", lambda g: g.high5 <= g.high3),
            ("D+5 < +3%", "ret5", lambda g: g.ret5 < 0.03),
        ],
        "中震荡": [
            ("D+5 < -2%", "ret5", lambda g: g.ret5 < -0.02),
            ("D+5 < -5%", "ret5", lambda g: g.ret5 < -0.05),
            ("D+5 预警分≥4", "score5", lambda g: g.score5 >= 4),
            ("D+5 三项复合", "comp5", lambda g: g.comp5),
            ("D+5 跌破买入日低点", "break5", lambda g: g.break5),
            ("D+5 < 0", "ret5", lambda g: g.ret5 < 0),
        ],
        "下跌": [
            ("D+5 < -5%", "ret5", lambda g: g.ret5 < -0.05),
            ("D+5 预警分≥4", "score5", lambda g: g.score5 >= 4),
            ("D+5 三项复合", "comp5", lambda g: g.comp5),
            ("D+5 比 D+3 再跌≥2pp", "ret5", lambda g: g.ret5 < g.ret3 - 0.02),
            ("D+5 低点低于 D+3 低点", "ret5", lambda g: g.low5 < g.low3),
            ("D+5 < -2%", "ret5", lambda g: g.ret5 < -0.02),
        ],
    }


def warning_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in ["全部信号前二", "实际成交"]:
        periods = PERIODS
        for period in periods:
            x = eval_base(t, sample, period)
            x = x[x.held5 & x.ret5.notna() & x._outcome.notna()].copy()
            for st in STATES:
                g = x[x.state == st].copy()
                if not len(g):
                    continue
                for name, _, fn in rule_defs()[st]:
                    mask = fn(g)
                    e = g[mask.fillna(False)]
                    if not len(e):
                        continue
                    base_loss = g._loss.mean()
                    loss = e._loss.mean()
                    se = cluster_se(e)
                    exit_net = e.exit5 - np.where(e._net, 0.002, 0.0)
                    rows.append({
                        "样本": sample, "期间": period, "状态": st, "规则": name,
                        "笔数": len(g), "触发笔数": len(e), "触发率": len(e) / len(g),
                        "最终亏损率": loss, "同态基准": base_loss,
                        "差值pp": (loss - base_loss) * 100,
                        "深度亏损率": e._tail10.mean(),
                        "之后止盈率": e.tp_any.mean(),
                        "剩余收益均值": e._fwd5.mean(),
                        "早退相对持有": e._early5.mean(),
                        "持有均值": e._outcome.mean(), "早退均值": exit_net.mean(),
                        "持有<=-15%": (e._outcome <= -0.15).mean(),
                        "早退<=-15%": (exit_net <= -0.15).mean(),
                        "信号段数": e.episode.nunique(),
                        "95%置信下": max(0, loss - 1.96 * se),
                        "95%置信上": min(1, loss + 1.96 * se),
                    })
    out = pd.DataFrame(rows)
    if len(out):
        order = {(st, i): i for st in STATES for i, r in enumerate(RULE_ORDER[st])}
        out["_o"] = [order.get((r.状态, r.规则), 99) for _, r in out.iterrows()]
        out["_s"] = out["状态"].map({s: i for i, s in enumerate(STATES)})
        out = out.sort_values(["样本", "期间", "_s", "_o"]).drop(columns=["_o", "_s"])
    return out


def threshold_table(t: pd.DataFrame) -> pd.DataFrame:
    x = t[t.held3 & t.outcome_sim.notna()].copy()
    x["_loss"] = x.outcome_sim < 0
    x["_tail10"] = x.outcome_sim <= -0.10
    rows = []
    for up in [0.02, 0.03, 0.05, 0.08]:
        st = np.select([x.ret3 >= up, x.ret3 <= DOWN], ["上涨", "下跌"], default="中震荡")
        for s in STATES:
            g = x[st == s]
            rows.append({
                "敏感性": "上涨阈值", "上涨阈值": up, "下跌阈值": DOWN, "状态": s,
                "笔数": len(g), "最终亏损率": g._loss.mean(), "深度亏损率": g._tail10.mean(),
                "均值": g.outcome_sim.mean(), "止盈率": g.tp_any.mean(),
            })
    for dn in [0.0, -0.01, -0.02, -0.03]:
        st = np.select([x.ret3 >= UP, x.ret3 <= dn], ["上涨", "下跌"], default="中震荡")
        for s in STATES:
            g = x[st == s]
            rows.append({
                "敏感性": "下跌阈值", "上涨阈值": UP, "下跌阈值": dn, "状态": s,
                "笔数": len(g), "最终亏损率": g._loss.mean(), "深度亏损率": g._tail10.mean(),
                "均值": g.outcome_sim.mean(), "止盈率": g.tp_any.mean(),
            })
    return pd.DataFrame(rows)


def market_regime() -> pd.DataFrame:
    m = pd.read_parquet(MARKET, columns=["date", "market_return"])
    m["date"] = pd.to_datetime(m.date)
    m = m.sort_values("date").reset_index(drop=True)
    m["idx"] = (1 + m.market_return.fillna(0)).cumprod()
    m["ma250"] = m.idx.rolling(250, min_periods=250).mean()
    m["ret120"] = m.idx / m.idx.shift(120) - 1
    m["slope250"] = m.ma250 / m.ma250.shift(20) - 1
    m["above250"] = m.idx / m.ma250 - 1

    def reg(r):
        if r.above250 > 0.02 and r.ret120 > 0.03 and r.slope250 > 0:
            return "上涨市"
        if r.above250 < -0.02 and r.ret120 < -0.03 and r.slope250 < 0:
            return "下跌市"
        return "中震荡市"

    m["市场环境"] = m.apply(reg, axis=1)
    return m[["date", "市场环境", "above250", "ret120", "slope250"]]


def market_table(t: pd.DataFrame) -> pd.DataFrame:
    m = market_regime()
    x = t[t.held3 & t.outcome_sim.notna()].copy()
    x["_loss"] = x.outcome_sim < 0
    x = x.merge(m, left_on="signal_date", right_on="date", how="left")
    rows = []
    for period in PERIODS:
        y = x if period == "全期" else x[x.period == period]
        for regime in ["上涨市", "中震荡市", "下跌市"]:
            g = y[y["市场环境"] == regime]
            if not len(g):
                continue
            rows.append({
                "期间": period, "市场环境": regime, "状态": "全部",
                "笔数": len(g), "信号日数": g.signal_date.nunique(),
                "最终亏损率": g._loss.mean(), "均值": g.outcome_sim.mean(),
            })
            for st in STATES:
                h = g[g.state == st]
                if not len(h):
                    continue
                rows.append({
                    "期间": period, "市场环境": regime, "状态": st,
                    "笔数": len(h), "信号日数": h.signal_date.nunique(),
                    "最终亏损率": h._loss.mean(), "均值": h.outcome_sim.mean(),
                })
    return pd.DataFrame(rows)


def peak_table(t: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    base = t[t.held3 & t.outcome_sim.notna()][["signal_date", "code", "state"]]
    x = df[df.d >= 3].merge(base, on=["signal_date", "code"], how="inner")
    keys = ["signal_date", "code"]
    hi = x.loc[x.groupby(keys).hfq_high.idxmax()][keys + ["d"]].rename(columns={"d": "峰值日_高"})
    cl = x.loc[x.groupby(keys).hfq_close.idxmax()][keys + ["d", "hfq_close"]].rename(columns={"d": "峰值日_收"})
    last = x.sort_values("d").groupby(keys).hfq_close.last().rename("期末收盘").reset_index()
    p = hi.merge(cl, on=keys).merge(last, on=keys).merge(base, on=keys)
    p["峰值回吐"] = p.期末收盘 / p.hfq_close - 1
    rows = []
    for st in STATES:
        g = p[p.state == st]
        rows.append({
            "状态": st, "笔数": len(g),
            "峰值日_收盘中位": g.峰值日_收.median(),
            "D+5前见顶占比": (g.峰值日_高 <= 5).mean(),
            "D+10前见顶占比": (g.峰值日_高 <= 10).mean(),
            "峰值到期末平均回吐": g.峰值回吐.mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_path()
    t = make_trades(df)
    tables = {
        "路径三态_基准.csv": baseline(t),
        "路径三态_逐日.csv": path_table(t, df),
        "路径三态_转弱.csv": warning_table(t),
        "路径三态_阈值.csv": threshold_table(t),
        "路径三态_市场环境.csv": market_table(t),
        "路径三态_峰值.csv": peak_table(t, df),
    }
    for name, table in tables.items():
        table.to_csv(RES / name, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    print("=== D+3 三态基准（全部信号前二，全期）===")
    b = tables["路径三态_基准.csv"]
    print(b[(b["样本"] == "全部信号前二") & (b["期间"] == "全期")].round(3).to_string(index=False))
    print("\n=== D+5 转弱（全部信号前二，全期）===")
    w = tables["路径三态_转弱.csv"]
    print(w[(w["样本"] == "全部信号前二") & (w["期间"] == "全期")][
        ["状态", "规则", "笔数", "触发笔数", "触发率", "最终亏损率", "同态基准",
         "差值pp", "深度亏损率", "剩余收益均值", "早退相对持有"]].round(3).to_string(index=False))
    print("\n=== 市场环境（全期）===")
    m = tables["路径三态_市场环境.csv"]
    print(m[m["期间"] == "全期"].pivot_table(index="市场环境", columns="状态",
                                           values="最终亏损率").round(3).to_string())
    print("\n=== 峰值 ===")
    print(tables["路径三态_峰值.csv"].round(3).to_string(index=False))
    print("\n三态样本：", t[t.held3 & t.outcome_sim.notna()].state.value_counts().to_dict())
    print("D+3 前已止盈：", int((t.tp3.fillna(False) & (t.exit_reason != "eod")).sum()),
          "未满 15 日：", int(t.outcome_sim.isna().sum()))


if __name__ == "__main__":
    main()
