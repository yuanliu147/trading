# -*- coding: utf-8 -*-
"""F09b 603778 换手率区间个案：只读单股核验。

只查 sh.603778：
- adjustflag=3（不复权，实际价）+ turn 字段（baostock turn 单位=百分比，如 6.4 表示 6.4%）
- adjustflag=1（后复权）做收益，T+1 开盘买、T+N 收盘卖
不做全表扫描。输出到本目录 CSV。
"""
import sqlite3
import numpy as np
import pandas as pd

BAO = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
OUT = "/Users/admin/WorkSpace/trading/2026-11-07_主板策略逐一审计/02_复算/F09b"
BINS = [0, 0.5, 1, 2, 3.5, 5, 8, 12, 20, 1e9]
LABELS = ["<0.5", "0.5-1", "1-2", "2-3.5", "3.5-5", "5-8", "8-12", "12-20", ">20"]


def load():
    con = sqlite3.connect(f"file:{BAO}?mode=ro", uri=True)
    raw = pd.read_sql(
        """select date,open,high,low,close,preclose,volume,amount,turn,pct_chg,is_st
           from stock_kline_d
           where code='sh.603778' and adjustflag=3 order by date""",
        con, parse_dates=["date"])
    hfq = pd.read_sql(
        """select date,open,high,low,close from stock_kline_d
           where code='sh.603778' and adjustflag=1 order by date""",
        con, parse_dates=["date"])
    con.close()
    df = raw.merge(hfq.rename(columns={c: f"{c}_h" for c in ["open", "high", "low", "close"]}),
                   on="date", how="left")
    return df


def build(df):
    df = df.dropna(subset=["turn"]).reset_index(drop=True)
    o1 = df.open_h.shift(-1)
    for n in (1, 5, 20, 60):
        df[f"f{n}"] = (df.close_h.shift(-n) / o1 - 1) * 100
    df["band"] = pd.cut(df.turn, BINS, labels=LABELS, right=False)
    df["year"] = df.date.dt.year
    return df


def main():
    raw = load()
    print("rows(adjustflag=3):", len(raw))
    print("date range:", raw.date.min().date(), "→", raw.date.max().date())
    print("turn NA:", raw.turn.isna().sum(), "| turn describe(%):")
    print(raw.turn.describe().round(2).to_string())
    # 单位核验：turn 是否等于 volume/流通股本*100
    sub = raw[(raw.date >= "2026-09-01")][["date", "close", "volume", "amount", "turn", "pct_chg"]]
    print("\n最近 10 日 turn（百分数口径）:")
    print(sub.tail(10).round(3).to_string(index=False))

    raw = raw[raw.date <= "2026-10-29"].copy()
    df = build(raw)
    print("\n有效交易日:", len(df), "| 停牌/缺 turn:", len(raw) - len(df))

    # 1) 区间表复算，与原交付 03_结果/换手区间统计.csv 对照
    g = df.groupby("band", observed=True)
    tab = pd.DataFrame({
        "天数": g.size(),
        "T+20中位": g.f20.median().round(2),
        "T+20均值": g.f20.mean().round(2),
        "T+20胜率": (g.f20.apply(lambda s: (s > 0).mean()) * 100).round(0),
        "T+60中位": g.f60.median().round(2),
        "T+60胜率": (g.f60.apply(lambda s: (s > 0).mean()) * 100).round(0),
    })
    tab.to_csv(f"{OUT}/F09b_603778_区间复算.csv", encoding="utf-8-sig")
    print("\n[复算] 区间表:")
    print(tab.to_string())

    # 2) 独立样本：每段连续同区间的天数合并成 1 个 episode
    eps = []
    run_id = (df.band.astype(str) != df.band.astype(str).shift()).cumsum()
    for b, s in df.groupby([run_id, "band"], observed=True):
        eps.append({"band": b[1], "start": s.date.iloc[0], "end": s.date.iloc[-1],
                    "days": len(s), "f20_first": s.f20.iloc[0]})
    ep = pd.DataFrame(eps)
    ep_sum = ep.groupby("band", observed=True).agg(段数=("days", "size"), 中位持续天数=("days", "median"))
    print("\n[独立样本] 连续区间段数:")
    print(ep_sum.to_string())
    ep.to_csv(f"{OUT}/F09b_603778_区间段.csv", index=False, encoding="utf-8-sig")

    # 3) 5-8% 带：剔除 2026 后
    b58 = df[df.band == "5-8"]
    b58_26 = b58[b58.year == 2026]
    b58_pre = b58[b58.year != 2026]
    print(f"\n[5-8%] 全部 {len(b58)} 天；2026 {len(b58_26)} 天；剔除2026 {len(b58_pre)} 天")
    print("  剔除2026: T+20 中位 %.2f 均值 %.2f 胜率 %.0f%% | T+60 中位 %.2f 胜率 %.0f%%" % (
        b58_pre.f20.median(), b58_pre.f20.mean(), (b58_pre.f20 > 0).mean() * 100,
        b58_pre.f60.median(), (b58_pre.f60 > 0).mean() * 100))
    ph = b58.copy()
    ph["阶段"] = np.where(ph.date < "2025-07-01", "A_2025-06前",
                        np.where(ph.date <= "2026-04-13", "B_2025-07至2026-04-13", "C_2026-04-14后"))
    print("  5-8% 按阶段:")
    print(ph.groupby("阶段").agg(天数=("f20", "size"), T20中位=("f20", "median"),
                                T20胜率=("f20", lambda s: (s > 0).mean() * 100)).round(1).to_string())

    # 4) 非重叠子样本：每 20 个交易日取 1 天（消除 20 日窗口重叠）
    idx = np.arange(0, len(df), 20)
    sub20 = df.iloc[idx]
    g2 = sub20.groupby("band", observed=True)
    sub_tab = pd.DataFrame({
        "天数": g2.size(),
        "T+20中位": g2.f20.median().round(2),
        "T+20均值": g2.f20.mean().round(2),
        "T+20胜率": (g2.f20.apply(lambda s: (s > 0).mean()) * 100).round(0),
    })
    sub_tab.to_csv(f"{OUT}/F09b_603778_非重叠子样本.csv", encoding="utf-8-sig")
    print(f"\n[非重叠] 每20日取1日，n={len(sub20)}:")
    print(sub_tab.to_string())

    # 5) 个股漂移与“负期望带”的相对比较
    print("\n[基准] 全样本 T+20 均值 %.2f%%，中位 %.2f%%，胜率 %.0f%%" % (
        df.f20.mean(), df.f20.median(), (df.f20 > 0).mean() * 100))
    prem = df[(df.date >= "2016-01-01") & (df.date <= "2025-06-30")]
    print("[基准] 2016-01-01→2025-06-30（剔除2025-07后主升与崩塌）T+20 均值 %.2f%%，中位 %.2f%%，胜率 %.0f%%" % (
        prem.f20.mean(), prem.f20.median(), (prem.f20 > 0).mean() * 100))
    print("  同窗口 5-8%% 带: n=%d, 均值 %.2f%%, 中位 %.2f%%, 胜率 %.0f%%" % (
        len(prem[prem.band == "5-8"]), prem[prem.band == "5-8"].f20.mean(),
        prem[prem.band == "5-8"].f20.median(), (prem[prem.band == "5-8"].f20 > 0).mean() * 100))

    # 6) 当前快照核验（2026-10-29 为库内最后一日）
    last = df.tail(3)[["date", "close", "turn", "f20"]]
    print("\n[最新]", last.round(3).to_string(index=False))
    pct_hist = (df.turn <= df.turn.iloc[-1]).mean() * 100
    d26 = df[df.year == 2026]
    pct_26 = (d26.turn <= d26.turn.iloc[-1]).mean() * 100
    print("  换手全历史分位 %.1f%% | 2026年内分位 %.1f%%" % (pct_hist, pct_26))

    # 7) 一句话结论的稳健性：把 2016 新股段剔除后 5-8% 还成立吗
    no16 = df[df.year > 2016]
    print("\n[稳健性] 剔除2016后 5-8%%: n=%d T+20 中位 %.2f%% 均值 %.2f%% 胜率 %.0f%%" % (
        len(no16[no16.band == "5-8"]), no16[no16.band == "5-8"].f20.median(),
        no16[no16.band == "5-8"].f20.mean(), (no16[no16.band == "5-8"].f20 > 0).mean() * 100))
    print("[稳健性] 剔除2016后 <1%%: n=%d T+20 中位 %.2f%% 均值 %.2f%% 胜率 %.0f%%" % (
        len(no16[no16.turn < 1]), no16[no16.turn < 1].f20.median(),
        no16[no16.turn < 1].f20.mean(), (no16[no16.turn < 1].f20 > 0).mean() * 100))


if __name__ == "__main__":
    main()
