#!/usr/bin/env python3
"""02 核心事件研究：MACD(12,26,9) 日线金叉，2010-2026 全 A

输出
----
03_结果/事件_金叉_12269.parquet
03_结果/统计_总体.csv
03_结果/统计_年度.csv
03_结果/统计_条件.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
from stats_utils import stat_block, bucket_table

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def path_features(P, horizons=(20, 60)):
    """未来 N 日内的 MFE/MAE，以 T+1 开盘为成本。"""
    out = {}
    for h in horizons:
        hi = pd.DataFrame(P["high"]).rolling(h, min_periods=h).max().to_numpy(dtype=np.float32)
        lo = pd.DataFrame(P["low"]).rolling(h, min_periods=h).min().to_numpy(dtype=np.float32)
        mfe = np.full(P["shape"], np.nan, dtype=np.float32)
        mae = np.full(P["shape"], np.nan, dtype=np.float32)
        n = P["shape"][0]
        if n > h:
            # t 日信号：窗口 high[t+1..t+h] = rolling(h)[t+h]；成本 T+1 开盘
            mfe[: n - h] = hi[h:] / P["open"][1 : n - h + 1] - 1
            mae[: n - h] = lo[h:] / P["open"][1 : n - h + 1] - 1
        out[f"mfe{h}"] = mfe
        out[f"mae{h}"] = mae
    return out


def add_buckets(e):
    e = e.copy()
    e["市场趋势"] = np.select(
        [e.index_above200 & (e.CSI300_ret20 > 0),
         e.index_above200 & (e.CSI300_ret20 <= 0)],
        ["强趋势(年线上+20日涨)", "弱趋势(年线上+20日跌)"], default="下跌(年线下)")
    e["宽度20"] = pd.cut(e.breadth20, [-0.01, 0.3, 0.5, 0.7, 1.01],
                        labels=["<30%", "30-50%", "50-70%", ">70%"])
    e["指数回撤"] = pd.cut(e.CSI300_dd250, [-1, -0.15, -0.05, 0.001],
                        labels=["<-15%", "-15~-5%", ">-5%"])
    e["指数波动"] = pd.qcut(e.CSI300_vol20, [0, 0.33, 0.67, 1],
                          labels=["低波动", "中波动", "高波动"], duplicates="drop")
    e["个股趋势"] = np.select(
        [e.above_ma200 & (e.ma200_slope20 > 0), e.above_ma200],
        ["年线上+年线升", "年线上+年线降"], default="年线下")
    e["距250日高"] = pd.cut(e.dist_high, [-1, -0.3, -0.15, -0.05, 0.001],
                          labels=["<-30%", "-30~-15%", "-15~-5%", ">-5%"])
    e["波动分组"] = pd.qcut(e.std20, [0, 0.33, 0.67, 1],
                          labels=["低波", "中波", "高波"], duplicates="drop")
    e["成交额分组"] = pd.qcut(e.amount20, [0, 0.33, 0.67, 1],
                           labels=["低额", "中额", "高额"], duplicates="drop")
    e["股价年龄"] = pd.cut(e.bars, [0, 250, 500, 1250, 10000],
                         labels=["<1年", "1-2年", "2-5年", ">5年"])
    e["60日金叉次数"] = pd.cut(e.cross60, [-1, 0, 1, 2, 100],
                            labels=["0", "1", "2", "3+"])
    e["绿柱长度"] = pd.cut(e.green_len, [-1, 5, 20, 60, 10000],
                         labels=["<=5", "6-20", "21-60", ">60"])
    e["量比"] = pd.cut(e.vol_ratio, [0, 0.7, 1.3, 2.0, 100],
                      labels=["<0.7", "0.7-1.3", "1.3-2", ">2"])
    e["DIF位置"] = pd.cut(e.dif_pct, [-100, 0, 0.5, 1.5, 100],
                        labels=["<0", "0-0.5", "0.5-1.5", ">1.5"])
    e["周线状态"] = np.select(
        [e.w_gold4 > 0, e.w_above],
        ["近4周周线金叉", "周线多头"], default="周线空头")
    return e


def main():
    P = M.load_panel()
    print("panel", P["shape"])
    dif, dea, hist = M.macd(P, 12, 26, 9)
    gold, death = M.crosses(dif, dea, P, min_bars=120)
    print("signals", int(gold.sum()), "death", int(death.sum()))

    feats = M.stock_features(P)
    mf = M.market_features(P)
    wf = M.weekly_state(P)
    fwd, ok, mkt, loo = M.forward_returns(P)
    pf = path_features(P)

    e = M.event_frame(P, gold, feats, mf, wf, fwd, ok, loo, dif, dea, hist, death, path=pf)
    print("events", len(e))
    e = add_buckets(e)
    e.to_parquet(os.path.join(RES, "事件_金叉_12269.parquet"), index=False)

    # 总体
    rows = []
    for h in M.HORIZONS:
        st = stat_block(e, f"ex{h}")
        st["周期"] = h
        rows.append(st)
    overall = pd.DataFrame(rows)
    overall.to_csv(os.path.join(RES, "统计_总体.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 总体 ===")
    print(overall.to_string(index=False))

    # 年度
    rows = []
    for h in M.HORIZONS:
        for y, g in e.dropna(subset=[f"ex{h}"]).groupby("year"):
            st = stat_block(g, f"ex{h}")
            st.update(年份=y, 周期=h)
            rows.append(st)
    yearly = pd.DataFrame(rows)
    yearly.to_csv(os.path.join(RES, "统计_年度.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 年度 ex20 ===")
    print(yearly[yearly.周期 == 20][["年份", "n", "mean_ex", "mean_daily", "hit", "t_nw"]].to_string(index=False))

    # 条件
    dims = ["市场趋势", "宽度20", "指数回撤", "指数波动", "个股趋势", "距250日高",
            "波动分组", "成交额分组", "股价年龄", "60日金叉次数", "绿柱长度",
            "量比", "DIF位置", "周线状态", "zero_state", "board", "is_st"]
    tabs = []
    for d in dims:
        t = bucket_table(e, d, horizons=M.HORIZONS, min_n=200)
        if len(t):
            tabs.append(t)
    cond = pd.concat(tabs, ignore_index=True)
    cond.to_csv(os.path.join(RES, "统计_条件.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 关键条件 ex20 (按均值排序) ===")
    key = cond[(cond.周期 == 20)].sort_values("mean_ex", ascending=False)
    print(key[["维度", "分组", "n", "n_days", "mean_ex", "mean_daily", "hit", "t_nw", "q_bh",
               "years_pos", "years_n"]].head(40).to_string(index=False))
    print("\n=== 关键条件 ex20 (最差) ===")
    print(key[["维度", "分组", "n", "n_days", "mean_ex", "mean_daily", "hit", "t_nw", "q_bh",
               "years_pos", "years_n"]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
