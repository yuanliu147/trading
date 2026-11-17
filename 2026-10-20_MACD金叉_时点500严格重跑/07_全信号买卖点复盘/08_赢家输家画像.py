#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把读图得到的假设落成数值：赢家/输家画像、b3 与死叉的时间先后、MACD 位置。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "01_数据"

CN = {
    "ret1": "当日涨跌", "ret5": "前5日收益", "ret10": "前10日收益",
    "ret20": "前20日收益", "ret60": "前60日收益", "ret120": "前120日收益",
    "dist_ma5": "距MA5", "dist_ma10": "距MA10", "dist_ma20": "距MA20",
    "dist_ma60": "距MA60", "ma20_slope10": "MA20十日斜率",
    "ma60_slope20": "MA60二十日斜率", "ma_align": "均线多头排列数",
    "rsi14": "RSI14", "atr20": "ATR20占价", "vol20": "20日波动",
    "vol60": "60日波动", "vol_ratio": "波动比20/60", "max_dd60": "距60日高点",
    "dd250": "距250日高点", "pos250": "250日区间位置",
    "amt5_60": "5日均额/60日均额", "amt20_60": "20日均额/60日均额",
    "amt60_yi": "60日均额亿", "vol5_60": "5日均量/60日均量",
    "hist_norm": "MACD柱/价", "dif_norm": "DIF/价", "dea_norm": "DEA/价",
    "hist_slope5": "MACD柱5日变化", "dif_slope5": "DIF5日变化",
    "hist_accel": "MACD柱加速度", "zero_axis": "零轴之上",
    "n_golden_60": "60日金叉数", "n_death_60": "60日死叉数",
    "days_since_golden": "距上次金叉", "signal_day_ret": "信号日涨跌",
    "body": "实体占振幅", "upper_shadow": "上影占振幅",
    "lower_shadow": "下影占振幅", "close_pos": "收盘区间位置",
    "gap": "信号日跳空", "big_move_20": "20日大波动天数",
    "price": "价格", "beta60": "60日Beta", "corr60": "60日相关性",
    "rs20": "20日相对强弱", "rs60": "60日相对强弱",
    "idx_ret20": "中证500前20日", "idx_ret60": "中证500前60日",
    "idx_vol20": "中证500 20日波动", "idx_dd250": "中证500距250日高点",
    "breadth_ma20": "MA20上方占比", "vol120": "120日波动",
    "range250": "250日振幅", "s_vol": "股性波动分", "s_heat": "股性活跃分",
    "entry_gap": "入场跳空", "zero_axis_f": "零轴之上",
}

FEATURES = [
    "ret5", "ret10", "ret20", "ret60", "dist_ma5", "dist_ma10",
    "dist_ma20", "dist_ma60", "ma20_slope10", "ma60_slope20", "ma_align",
    "rsi14", "atr20", "vol20", "vol60", "vol_ratio", "max_dd60", "dd250",
    "pos250", "amt5_60", "amt20_60", "amt60_yi", "vol5_60",
    "hist_norm", "dif_norm", "dea_norm", "hist_slope5", "dif_slope5",
    "hist_accel", "n_golden_60", "n_death_60", "days_since_golden",
    "signal_day_ret", "body", "upper_shadow", "lower_shadow", "close_pos",
    "gap", "big_move_20", "price", "beta60", "corr60", "rs20", "rs60",
    "idx_ret20", "idx_ret60", "idx_vol20", "idx_dd250", "breadth_ma20",
    "vol120", "range250", "s_vol", "s_heat", "entry_gap",
]


def load() -> pd.DataFrame:
    tr = pd.read_csv(DATA / "全信号交易_分规则.csv", encoding="utf-8-sig",
                     parse_dates=["信号日", "买入日", "卖出日"])
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", encoding="utf-8-sig",
                      parse_dates=["signal_date"])
    d = tr[tr["规则"] == "死叉"].merge(
        sig, left_on=["代码", "信号日"], right_on=["code", "signal_date"],
        how="left", suffixes=("", "_sig"))
    d["entry_gap"] = d["入场跳空"]
    b = tr[tr["规则"] == "b3_10_8"][
        ["代码", "信号日", "净收益率", "卖出日", "持有交易日"]].rename(
        columns={"净收益率": "b3收益", "卖出日": "b3卖出日",
                 "持有交易日": "b3持有"})
    d = d.merge(b, on=["代码", "信号日"], how="left")
    return d


def portrait(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        x = pd.to_numeric(d[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        w = x[d["净收益率"] > 0].dropna()
        l = x[d["净收益率"] <= 0].dropna()
        bw = x[d["净收益率"] > 0.15].dropna()
        bl = x[d["净收益率"] < -0.10].dropna()
        if len(w) < 100 or len(l) < 100:
            continue
        try:
            p = stats.mannwhitneyu(w, l, alternative="two-sided").pvalue
        except ValueError:
            p = np.nan
        rows.append({
            "特征": f, "含义": CN.get(f, f),
            "盈利单中位": w.median(), "亏损单中位": l.median(),
            "差值": w.median() - l.median(),
            "大赢中位": bw.median() if len(bw) else np.nan,
            "大亏中位": bl.median() if len(bl) else np.nan,
            "p值": p,
            "盈利单样本": len(w), "亏损单样本": len(l),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p值"] = out["p值"].clip(lower=1e-300)
        out = out.sort_values("p值")
    return out


def timing(d: pd.DataFrame) -> pd.DataFrame:
    x = d.dropna(subset=["b3卖出日", "卖出日"]).copy()
    x["b3先卖"] = x["b3卖出日"] < x["卖出日"]
    x["同日"] = x["b3卖出日"] == x["卖出日"]
    x["b3后卖"] = x["b3卖出日"] > x["卖出日"]
    x["日差"] = (x["b3卖出日"] - x["卖出日"]).dt.days
    x["收益差"] = x["b3收益"] - x["净收益率"]
    rows = [{
        "样本": len(x),
        "b3更早": x["b3先卖"].mean() * 100,
        "同日": x["同日"].mean() * 100,
        "b3更晚": x["b3后卖"].mean() * 100,
        "b3平均收益": x["b3收益"].mean() * 100,
        "死叉平均收益": x["净收益率"].mean() * 100,
        "平均差": x["收益差"].mean() * 100,
        "b3更早时平均差": x.loc[x["b3先卖"], "收益差"].mean() * 100,
        "b3更晚时平均差": x.loc[x["b3后卖"], "收益差"].mean() * 100,
        "b3更早时b3更好占比": (x.loc[x["b3先卖"], "收益差"] > 0).mean() * 100,
        "b3更晚时b3更好占比": (x.loc[x["b3后卖"], "收益差"] > 0).mean() * 100,
    }]
    return pd.DataFrame(rows)


def zero_and_macd(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, m in {
        "零轴之上": d["zero_axis"] == 1,
        "零轴之下": d["zero_axis"] == 0,
        "红柱5日走强": d["hist_slope5"] > 0,
        "红柱5日走弱": d["hist_slope5"] <= 0,
        "红柱加速": d["hist_accel"] > 0,
        "红柱减速": d["hist_accel"] <= 0,
        "收盘强实体": (d["body"] > 0.5) & (d["upper_shadow"] < 0.2),
        "长上影": d["upper_shadow"] > 0.3,
        "跳空高开>2%": d["entry_gap"] > 0.02,
        "跳空低开<-2%": d["entry_gap"] < -0.02,
    }.items():
        x = d[m]
        if len(x) < 100:
            continue
        rows.append({
            "分组": name, "样本": len(x), "占比": len(x) / len(d) * 100,
            "胜率": (x["净收益率"] > 0).mean() * 100,
            "平均收益": x["净收益率"].mean() * 100,
            "中位收益": x["净收益率"].median() * 100,
            "大亏率": (x["净收益率"] <= -0.10).mean() * 100,
            "平均最大不利": x["期间最低"].mean() * 100,
        })
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    p = portrait(d)
    p.to_csv(DATA / "赢家输家画像.csv", index=False, encoding="utf-8-sig")
    t = timing(d)
    t.to_csv(DATA / "b3与死叉时间差.csv", index=False, encoding="utf-8-sig")
    z = zero_and_macd(d)
    z.to_csv(DATA / "零轴与MACD形态.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 240)
    print("赢家/输家画像（按 p 值）：")
    print(p[["特征", "含义", "盈利单中位", "亏损单中位", "差值",
             "大赢中位", "大亏中位", "p值"]].head(25).round(4).to_string(index=False))
    print("\nb3 与死叉时间：")
    print(t.round(2).to_string(index=False))
    print("\n零轴/MACD 形态：")
    print(z.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
