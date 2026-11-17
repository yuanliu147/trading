#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全信号特征工程：每个金叉信号一行，只用信号日及以前的数据。

股票池：时点中证 500（快照次日生效，含退市股）。
区间：2010-07-01 至 2026-10-13（2010 年上半年做指标预热）。

输出：
- 信号特征_时点500.csv        中文表头，供复盘/筛选
- 信号特征_时点500_内部.csv   英文字段，供脚本读取
- 信号_股性标签.csv
- 日度市场状态.csv
- 特征字典.csv
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = Path("/Users/admin/WorkSpace/trading")
DATA_DIR = ROOT / "2026-10-18_成交量融合_2010-2025" / "01_数据"
OUT = Path(__file__).resolve().parent / "01_数据"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2010-07-01")
END = pd.Timestamp("2026-10-13")

spec = importlib.util.spec_from_file_location(
    "strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)


def _digits(code: str) -> str:
    return "".join(ch for ch in str(code) if ch.isdigit())[:6]


def load_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with open(DATA_DIR / "daily_cache_2016_2026.pkl", "rb") as f:
        b = pickle.load(f)
    for code, name in b.get("names", {}).items():
        if name:
            names[_digits(code)] = str(name).strip()
    pool = pd.read_csv(ROOT / "analysis" / "macd" / "MACD研究股票名单.csv")
    for _, r in pool.iterrows():
        names[_digits(r["thscode"])] = str(r["name"]).strip()
    try:
        extra = pd.read_csv(ROOT / "八系统回测" / "股票代码名称.csv",
                            dtype={"code": str})
        for _, r in extra.iterrows():
            k = _digits(r["code"])
            if k not in names:
                names[k] = str(r["name"]).strip()
    except Exception:
        pass
    return names


def bench_features(bench: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    b = bench.copy().reset_index(drop=True)
    c = b["close"].astype(float)
    b["idx_ret1"] = c.pct_change()
    b["idx_ret20"] = c.pct_change(20)
    b["idx_ret60"] = c.pct_change(60)
    b["idx_ma20"] = c.rolling(20, min_periods=20).mean()
    b["idx_ma60"] = c.rolling(60, min_periods=60).mean()
    b["idx_above_ma20"] = (c > b["idx_ma20"]).astype(int)
    b["idx_above_ma60"] = (c > b["idx_ma60"]).astype(int)
    b["idx_vol20"] = b["idx_ret1"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
    b["idx_dd250"] = c / c.rolling(250, min_periods=250).max() - 1.0
    return b.set_index("date")["idx_ret1"], b


def add_features(df: pd.DataFrame, bench_ret: pd.Series,
                 bench_feat: pd.DataFrame, code: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    o = out["open"].astype(float)
    a = out["amount"].astype(float)
    v = out["volume"].astype(float)
    prev_c = c.shift(1)

    out["bars"] = np.arange(1, len(out) + 1)
    out["ret1"] = c.pct_change()
    for n in (3, 5, 10, 20, 60, 120):
        out[f"ret{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 60, 120):
        out[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
        out[f"dist_ma{n}"] = c / out[f"ma{n}"] - 1.0
    out["ma20_slope10"] = out["ma20"] / out["ma20"].shift(10) - 1.0
    out["ma60_slope20"] = out["ma60"] / out["ma60"].shift(20) - 1.0
    out["ma_align"] = ((out["ma5"] > out["ma10"]).astype(int)
                       + (out["ma10"] > out["ma20"]).astype(int)
                       + (out["ma20"] > out["ma60"]).astype(int))

    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ru = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rd = down.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["rsi14"] = 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))

    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
                   axis=1).max(axis=1)
    out["atr20"] = tr.rolling(20, min_periods=20).mean() / c
    out["vol20"] = out["ret1"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
    out["vol60"] = out["ret1"].rolling(60, min_periods=60).std(ddof=0) * np.sqrt(252)
    out["vol_ratio"] = out["vol20"] / out["vol60"].replace(0.0, np.nan)

    roll_high60 = c.rolling(60, min_periods=60).max()
    roll_high250 = h.rolling(250, min_periods=250).max()
    roll_low250 = l.rolling(250, min_periods=250).min()
    out["max_dd60"] = c / roll_high60 - 1.0
    out["dd250"] = c / roll_high250 - 1.0
    out["pos250"] = (c - roll_low250) / (roll_high250 - roll_low250).replace(0.0, np.nan)

    amt5 = a.rolling(5, min_periods=5).mean()
    amt20 = a.rolling(20, min_periods=20).mean()
    amt60 = a.rolling(60, min_periods=60).mean()
    amt120 = a.rolling(120, min_periods=120).mean()
    out["amt5_60"] = amt5 / amt60.replace(0.0, np.nan) - 1.0
    out["amt20_60"] = amt20 / amt60.replace(0.0, np.nan) - 1.0
    out["amt60_yi"] = amt60 / 1e8
    vol5 = v.rolling(5, min_periods=5).mean()
    vol60 = v.rolling(60, min_periods=60).mean()
    out["vol5_60"] = vol5 / vol60.replace(0.0, np.nan) - 1.0

    out["hist_norm"] = out["hist"] / c
    out["dif_norm"] = out["dif"] / c
    out["dea_norm"] = out["dea"] / c
    out["hist_slope5"] = (out["hist"] - out["hist"].shift(5)) / c
    out["dif_slope5"] = (out["dif"] - out["dif"].shift(5)) / c
    out["hist_accel"] = (out["hist"] - 2 * out["hist"].shift(1)
                         + out["hist"].shift(2)) / c
    out["zero_axis"] = (out["dif"] > 0) & (out["dea"] > 0)
    out["dif_pos"] = out["dif"] > 0
    out["n_golden_60"] = out["golden_cross"].rolling(60, min_periods=60).sum()
    out["n_death_60"] = out["death_cross"].rolling(60, min_periods=60).sum()

    gc = out["golden_cross"].to_numpy()
    days = np.full(len(out), np.nan)
    last = -1
    for i in range(len(out)):
        if gc[i] == 1:
            last = i
        if last >= 0:
            days[i] = i - last
    out["days_since_golden"] = days

    rng = (h - l).replace(0.0, np.nan)
    out["signal_day_ret"] = out["ret1"]
    out["body"] = (c - o) / rng
    out["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng
    out["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng
    out["close_pos"] = (c - l) / rng
    out["gap"] = o / prev_c - 1.0
    out["big_move_20"] = (out["ret1"].abs() >= 0.095).rolling(
        20, min_periods=20).sum()
    out["price"] = c
    out["log_price"] = np.log(c)

    out["idx_ret1"] = out["date"].map(bench_ret)
    cov = out["ret1"].rolling(60, min_periods=60).cov(out["idx_ret1"])
    var = out["idx_ret1"].rolling(60, min_periods=60).var()
    out["beta60"] = cov / var.replace(0.0, np.nan)
    out["corr60"] = out["ret1"].rolling(60, min_periods=60).corr(out["idx_ret1"])
    bf = bench_feat.set_index("date")
    for col in ["idx_ret20", "idx_ret60", "idx_above_ma20", "idx_above_ma60",
                "idx_vol20", "idx_dd250"]:
        out[col] = out["date"].map(bf[col])
    out["rs20"] = out["ret20"] - out["idx_ret20"]
    out["rs60"] = out["ret60"] - out["idx_ret60"]

    out["vol120"] = out["ret1"].rolling(120, min_periods=120).std(ddof=0) * np.sqrt(252)
    out["range250"] = (h.rolling(250, min_periods=250).max()
                       / l.rolling(250, min_periods=250).min() - 1.0)
    out["act"] = amt20 / amt120.replace(0.0, np.nan) - 1.0
    digits = "".join(ch for ch in str(code) if ch.isdigit())[:6]
    thr = 0.195 if digits[:3] in ("300", "301", "688") else 0.095
    out["limfrac"] = (out["ret1"].abs() >= thr).rolling(250, min_periods=250).mean()
    out["gapfrac"] = (out["gap"].abs() > 0.02).rolling(250, min_periods=250).mean()
    return out


FEATURE_COLS = [
    "ret1", "ret3", "ret5", "ret10", "ret20", "ret60", "ret120",
    "dist_ma5", "dist_ma10", "dist_ma20", "dist_ma60",
    "ma20_slope10", "ma60_slope20", "ma_align", "rsi14",
    "atr20", "vol20", "vol60", "vol_ratio", "max_dd60",
    "dd250", "pos250", "amt5_60", "amt20_60", "amt60_yi",
    "vol5_60", "hist_norm", "dif_norm", "dea_norm", "hist_slope5",
    "dif_slope5", "hist_accel", "zero_axis", "dif_pos",
    "n_golden_60", "n_death_60", "days_since_golden",
    "signal_day_ret", "body", "upper_shadow", "lower_shadow",
    "close_pos", "gap", "big_move_20", "price", "log_price",
    "beta60", "corr60", "rs20", "rs60",
    "idx_ret20", "idx_ret60", "idx_above_ma20", "idx_above_ma60",
    "idx_vol20", "idx_dd250", "vol120", "range250",
]

CN = {
    "code": "代码", "name": "名称", "signal_date": "信号日", "close": "信号日收盘",
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
    "hist_accel": "MACD柱加速度", "zero_axis": "零轴之上",
    "dif_pos": "DIF为正", "n_golden_60": "60日金叉数",
    "n_death_60": "60日死叉数", "days_since_golden": "距上次金叉",
    "signal_day_ret": "信号日涨跌", "body": "实体占振幅",
    "upper_shadow": "上影占振幅", "lower_shadow": "下影占振幅",
    "close_pos": "收盘区间位置", "gap": "信号日跳空",
    "big_move_20": "20日大波动天数", "price": "价格", "log_price": "对数价格",
    "beta60": "60日Beta", "corr60": "60日相关性",
    "rs20": "20日相对强弱", "rs60": "60日相对强弱",
    "idx_ret20": "中证500前20日", "idx_ret60": "中证500前60日",
    "idx_above_ma20": "中证500在MA20上", "idx_above_ma60": "中证500在MA60上",
    "idx_vol20": "中证500 20日波动", "idx_dd250": "中证500距250日高点",
    "vol120": "120日波动", "range250": "250日振幅",
    "s_vol": "股性波动分", "s_heat": "股性活跃分",
    "type": "股性类型", "subtype": "股性子类",
    "breadth_ma20": "MA20上方占比",
}


def main() -> None:
    print("加载数据…", flush=True)
    data = eng.load_pit_data()
    eng.START, eng.END = START, END
    bench = eng.load_benchmark()
    bench_ret, bench_feat = bench_features(bench)
    names = load_names()

    sig_rows = []
    behavior_rows = []
    breadth_rows = []
    for k, (code, df) in enumerate(data.items(), 1):
        x = add_features(df, bench_ret, bench_feat, code)
        x["code"] = code
        m = (x["golden_cross"] == 1) & (x["date"] >= START) & (x["date"] <= END)
        if m.any():
            cols = ["code", "date", "close"] + FEATURE_COLS
            s = x.loc[m, cols].copy()
            s["name"] = names.get(_digits(code), "")
            sig_rows.append(s)
        b = x.loc[(x["bars"] >= 260) & (x["amt60_yi"] * 1e8 >= 2e7),
                  ["date", "code", "vol120", "range250", "act", "limfrac",
                   "gapfrac", "amt60_yi", "bars", "golden_cross"]].copy()
        if len(b):
            behavior_rows.append(b)
        breadth_rows.append(pd.DataFrame({
            "date": x["date"],
            "above": (x["close"] > x["ma20"]).astype(float),
            "valid": x["ma20"].notna().astype(float),
        }))
        if k % 400 == 0:
            print(f"  {k}/{len(data)}", flush=True)

    print("汇总信号…", flush=True)
    sig = pd.concat(sig_rows, ignore_index=True)
    sig = sig.rename(columns={"date": "signal_date"})
    print(f"  原始金叉信号 {len(sig):,} 行", flush=True)

    print("计算逐日股性…", flush=True)
    panel = pd.concat(behavior_rows, ignore_index=True)
    for c in ["vol120", "range250", "act", "limfrac", "gapfrac", "amt60_yi"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("float32")
    panel = panel[(panel["bars"] >= 260) & (panel["amt60_yi"] * 1e8 >= 2e7)].copy()
    g = panel.groupby("date", sort=False)
    panel["s_vol"] = (g["vol120"].rank(pct=True)
                      + g["range250"].rank(pct=True)) / 2.0
    panel["s_heat"] = (g["act"].rank(pct=True)
                       + g["limfrac"].rank(pct=True)
                       + g["gapfrac"].rank(pct=True)) / 3.0
    panel = panel.dropna(subset=["s_vol", "s_heat"]).copy()
    panel["type"] = np.select(
        [(panel.s_vol >= 0.5) & (panel.s_heat >= 0.5),
         (panel.s_vol >= 0.5) & (panel.s_heat < 0.5),
         (panel.s_vol < 0.5) & (panel.s_heat >= 0.5)],
        ["高波活跃型", "高波低量型", "低波活跃型"], default="低波低量型")
    panel["subtype"] = ""
    mask = panel["type"] == "高波活跃型"
    if mask.any():
        sub = panel[mask].copy()
        sub["r_lim"] = sub.groupby("date")["limfrac"].rank(pct=True)
        sub["r_gap"] = sub.groupby("date")["gapfrac"].rank(pct=True)
        sub["spec_score"] = (sub["r_lim"] + sub["r_gap"]) / 2.0
        panel.loc[mask, "subtype"] = np.where(
            sub["spec_score"] >= 0.8, "极端情绪型", "普通高波型")
    mask2 = panel["type"] == "低波低量型"
    if mask2.any():
        sub2 = panel[mask2].copy()
        sub2["liq_rank"] = sub2.groupby("date")["amt60_yi"].rank(pct=True)
        panel.loc[mask2, "subtype"] = np.select(
            [sub2["liq_rank"] >= 0.7, sub2["liq_rank"] <= 0.3],
            ["稳健低波型", "沉寂低波型"], default="普通低波型")

    labels = panel[panel["golden_cross"] == 1][
        ["date", "code", "s_vol", "s_heat", "type", "subtype"]].copy()
    labels = labels.rename(columns={"date": "signal_date"})
    sig = sig.merge(labels, on=["code", "signal_date"], how="left")

    br = pd.concat(breadth_rows, ignore_index=True)
    br = br[br["valid"] > 0]
    breadth = br.groupby("date")["above"].mean()
    sig["breadth_ma20"] = sig["signal_date"].map(breadth)
    sig["type"] = sig["type"].fillna("未分类")
    sig["subtype"] = sig["subtype"].fillna("")

    # 只保留信号日确实在时点 500 成分内的
    snaps = eng.load_snapshots()
    uni = eng.Universe(snaps)
    print("核对时点成分…", flush=True)
    member = [code in uni.snapshot_for(d) for code, d in
              zip(sig["code"], sig["signal_date"])]
    sig["in_universe"] = np.array(member, dtype=int)
    print(f"  成分内信号 {sig.in_universe.sum():,} / {len(sig):,}", flush=True)

    sig = sig[sig["in_universe"] == 1].copy()
    sig = sig.sort_values(["signal_date", "code"]).reset_index(drop=True)
    num_cols = sig.select_dtypes(include=[np.number]).columns
    sig[num_cols] = sig[num_cols].round(6)
    sig.to_csv(OUT / "信号特征_时点500_内部.csv", index=False, encoding="utf-8-sig")

    # 中文版
    cn = sig.rename(columns={k: v for k, v in CN.items() if k in sig.columns})
    cn.to_csv(OUT / "信号特征_时点500.csv", index=False, encoding="utf-8-sig")

    # 股性标签只保留成分内信号
    keep = sig[["code", "signal_date"]].drop_duplicates()
    labels = labels.merge(keep, on=["code", "signal_date"], how="inner")
    lab = labels.rename(columns={"signal_date": "信号日", "code": "代码",
                                 "s_vol": "波动分", "s_heat": "活跃分",
                                 "type": "股性类型", "subtype": "股性子类"})
    lab.to_csv(OUT / "信号_股性标签.csv", index=False, encoding="utf-8-sig")

    bench_out = bench_feat.copy()
    bench_out["breadth_ma20"] = bench_out["date"].map(breadth)
    bench_out = bench_out.rename(columns={"date": "日期"})
    bench_out.to_csv(OUT / "日度市场状态.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"字段": list(CN), "含义": list(CN.values())}).to_csv(
        OUT / "特征字典.csv", index=False, encoding="utf-8-sig")

    print("\n年度信号数：")
    print(sig.groupby(sig.signal_date.dt.year).size().to_string())
    print(f"\n完成：{OUT}")


if __name__ == "__main__":
    main()
