#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股性自适应买卖点验证：全信号等权，每笔 1 万，本金无限。

预注册口径（先写死，再跑，避免事后挑参数）
------------------------------------------------
买点：
  全信号          : 所有时点 500 金叉
  F2              : ret20<=15%, dist_ma20<=8%, rsi14<=70,
                    n_golden_60>=3, s_vol<=0.72, amt5_60<=26%,
                    dist_ma60>=-8%
  F2+指数         : F2 且信号日中证 500 在 MA20 上（主方案）
  类型动量+指数   : 高波活跃型用“趋势+20日动量+量能扩张”，
                    其它类型用 F2，且指数在 MA20 上
  类型突破+指数   : 高波活跃型用“近250日高位+量能扩张”，
                    其它类型用 F2，且指数在 MA20 上
  S1              : 趋势+回调+压缩+RSI+流动性（用户提出的“高把握”买点）
  S1+赔率         : S1 且 上方空间/止损 >= 2

卖点：
  死叉
  b3_10_8
  b3_15_10
  b3_10_8_硬止损8
  ATR自适应            : 硬止损 2.5*ATR，追踪 2.2*ATR，触发 max(10%,3*ATR)
  类型自适应ATR        : 按股性类型给不同止损/追踪；低波低量加 30 日时间止损
  类型自适应ATR_无时间 : 同上，但不加时间止损
  ATR追踪_无硬止损     : 只做 ATR 追踪，不加硬止损

执行：
  T 日收盘金叉，T+1 开盘买；涨停不买；买入费 0.025%；
  T 日收盘信号，T+1 开盘卖；跌停顺延；卖出费 0.125%；
  每笔独立 1 万元，不共享资金、不限制同时持仓数。

阶段：
  设计 2010-2017；样本外 2018-2026；细分 2018-2022 / 2023-2026。

输出：06_股性自适应验证/
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "06_股性自适应验证"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

START = pd.Timestamp("2010-07-01")
END = pd.Timestamp("2026-10-13")
ALLOC = 10_000.0
LIMIT = int(os.environ.get("LIMIT", "0"))

ENTRIES = {
    "全信号": "m_all",
    "F2": "m_f2",
    "F2+指数": "m_f2idx",
    "类型动量+指数": "m_typemom",
    "类型突破+指数": "m_typebrk",
    "S1": "m_s1",
    "S1+赔率": "m_s1odds",
}

EXITS = {
    "死叉": dict(kind="death"),
    "b3_10_8": dict(kind="b3", trigger=0.10, trail=0.08, hard=None),
    "b3_15_10": dict(kind="b3", trigger=0.15, trail=0.10, hard=None),
    "b3_10_8_硬止损8": dict(kind="b3", trigger=0.10, trail=0.08, hard=0.08),
    "ATR自适应": dict(kind="atr", stop_mult=2.5, trail_mult=2.2,
                   trigger_mult=3.0, hard=True, time_stop=None),
    "类型自适应ATR": dict(kind="type", trigger_mult=3.0, hard=True,
                      time_stop=30),
    "类型自适应ATR_无时间": dict(kind="type", trigger_mult=3.0, hard=True,
                          time_stop=None),
    "ATR追踪_无硬止损": dict(kind="atr", stop_mult=2.5, trail_mult=2.2,
                       trigger_mult=3.0, hard=False, time_stop=None),
}

TYPE_MULT = {
    "低波低量型": (2.2, 1.8),
    "低波活跃型": (2.5, 2.2),
    "高波低量型": (2.8, 2.4),
    "高波活跃型": (3.0, 2.6),
    "未分类": (2.5, 2.2),
}


def add_entry_masks(sig: pd.DataFrame) -> pd.DataFrame:
    x = sig.copy()
    trend = ((x["dist_ma60"] > 0) & (x["ma60_slope20"] > 0)
             & (x["ma_align"] >= 2))
    base = (x["rsi14"] <= 70) & (x["amt60_yi"] >= 0.2)
    pull = (x["dist_ma20"].between(-0.03, 0.05)
            & (x["ret20"] <= 0.12) & (x["max_dd60"] >= -0.15))
    comp = (x["vol_ratio"] <= 1.0) & (x["amt5_60"] <= 0.0)
    mom = (trend & base & (x["ret20"] >= 0.05) & (x["dist_ma20"] >= 0)
           & (x["pos250"] >= 0.7) & (x["dd250"] >= -0.15)
           & (x["vol_ratio"] >= 1.0) & (x["amt5_60"] >= 0))
    brk = (trend & base & (x["pos250"] >= 0.9) & (x["dd250"] >= -0.05)
           & (x["vol_ratio"] >= 1.2) & (x["amt5_60"] >= 0.2))
    f2 = ((x["ret20"] <= 0.15) & (x["dist_ma20"] <= 0.08)
          & (x["rsi14"] <= 70) & (x["n_golden_60"] >= 3)
          & (x["s_vol"] <= 0.72) & (x["amt5_60"] <= 0.26)
          & (x["dist_ma60"] >= -0.08))
    idx = x["idx_above_ma20"] == 1
    hv = x["type"] == "高波活跃型"
    s1 = trend & pull & comp & base

    # 赔率：上方空间用“距250日高点”或“250日振幅的一半”，封顶 50%
    stop_common = (2.5 * x["atr20"]).clip(0.06, 0.15)
    room = np.maximum(-x["dd250"], 0.5 * x["range250"]).clip(0, 0.5)
    x["赔率"] = room / stop_common.replace(0.0, np.nan)
    x["风险单位"] = stop_common
    x["赔率组"] = pd.qcut(x["赔率"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                        duplicates="drop")
    x["层数"] = (trend.astype(int) + pull.astype(int) + comp.astype(int)
               + (x["rsi14"] <= 70).astype(int)
               + (x["amt60_yi"] >= 0.2).astype(int))

    x["m_all"] = True
    x["m_f2"] = f2
    x["m_f2idx"] = f2 & idx
    x["m_typemom"] = ((hv & mom) | ((~hv) & f2)) & idx
    x["m_typebrk"] = ((hv & brk) | ((~hv) & f2)) & idx
    x["m_s1"] = s1
    x["m_s1odds"] = s1 & (x["赔率"] >= 2.0)
    return x


def resolve_exit(spec: dict, atr: float, typ: str) -> dict:
    atr = max(float(atr) if np.isfinite(atr) else 0.03, 0.005)
    if spec["kind"] == "death":
        return dict(kind="death", hard=None, trail=None, trigger=None, time=None)
    if spec["kind"] == "b3":
        return dict(kind="b3", hard=spec["hard"], trail=spec["trail"],
                    trigger=spec["trigger"], time=None)
    if spec["kind"] == "atr":
        hard = (float(np.clip(spec["stop_mult"] * atr, 0.06, 0.15))
                if spec["hard"] else None)
        trail = float(np.clip(spec["trail_mult"] * atr, 0.06, 0.18))
        trigger = max(0.10, spec["trigger_mult"] * atr)
        return dict(kind="atr", hard=hard, trail=trail, trigger=trigger,
                    time=None)
    if spec["kind"] == "type":
        sm, tm = TYPE_MULT.get(typ, TYPE_MULT["未分类"])
        hard = (float(np.clip(sm * atr, 0.06, 0.15))
                if spec["hard"] else None)
        trail = float(np.clip(tm * atr, 0.06, 0.18))
        trigger = max(0.10, spec["trigger_mult"] * atr)
        time_stop = spec["time_stop"] if typ == "低波低量型" else None
        return dict(kind="atr", hard=hard, trail=trail, trigger=trigger,
                    time=time_stop)
    raise ValueError(spec["kind"])


def simulate_exit(o, h, l, c, dc, dates, code: str, idx_e: int,
                  r: dict) -> dict:
    n = len(c)
    entry = float(o[idx_e])
    peak = entry
    pending = False
    pending_reason = ""
    exit_idx = None
    exit_px = None
    exit_at_open = False
    reason = ""

    for j in range(idx_e, n):
        if pending:
            if j > idx_e:
                prev_c = c[j - 1]
                lim = eng.limit_pct(code, dates.iat[j])
                if o[j] <= prev_c * (1.0 - lim) * 1.001:
                    # 跌停卖不出，继续等
                    continue
                exit_idx = j
                exit_px = float(o[j])
                exit_at_open = True
                reason = pending_reason
                break
            # 买入日不执行挂起卖出
            pending = False

        cl, lo, hi = c[j], l[j], h[j]
        # 1) 硬止损（收盘确认）
        if r["hard"] is not None and cl <= entry * (1.0 - r["hard"]):
            pending = True
            pending_reason = "硬止损"
        # 2) 时间止损
        elif r["time"] is not None and (j - idx_e + 1) >= r["time"]:
            pending = True
            pending_reason = "时间止损"
        # 3) 追踪（只用前一日收盘时已知的峰值）
        elif (r["trigger"] is not None
              and peak / entry - 1.0 >= r["trigger"]
              and lo <= peak * (1.0 - r["trail"])):
            pending = True
            pending_reason = ("b3追踪" if r["kind"] == "b3" else "ATR追踪")
        # 4) 死叉兜底
        elif int(dc[j]) == 1:
            pending = True
            pending_reason = "死叉"

        # 当天新高只用于下一天
        if hi > peak:
            peak = hi

    if exit_idx is None:
        exit_idx = n - 1
        exit_px = float(c[-1])
        exit_at_open = False
        reason = "期末/退市"

    last_hold = exit_idx - 1 if exit_at_open and exit_idx > idx_e else exit_idx
    last_hold = max(idx_e, min(last_hold, n - 1))
    seg = slice(idx_e, last_hold + 1)
    mfe = float(h[seg].max()) / entry - 1.0
    mae = float(l[seg].min()) / entry - 1.0
    return dict(exit_idx=exit_idx, exit_px=exit_px, reason=reason,
                exit_at_open=exit_at_open, hold_days=exit_idx - idx_e,
                peak_profit=peak / entry - 1.0, mfe=mfe, mae=mae)


def summarize(x: pd.DataFrame) -> pd.Series:
    n = len(x)
    if n == 0:
        return pd.Series(dtype=float)
    win = x["净收益率"] > 0
    loss = ~win
    pos = x.loc[win, "净盈亏"].sum()
    neg = x.loc[loss, "净盈亏"].sum()
    top_n = max(1, int(np.ceil(n * 0.05)))
    top = x["净盈亏"].nlargest(top_n).sum()
    return pd.Series({
        "笔数": n,
        "胜率%": win.mean() * 100,
        "平均收益率%": x["净收益率"].mean() * 100,
        "收益率中位数%": x["净收益率"].median() * 100,
        "平均R": x["R倍数"].mean(),
        "R中位数": x["R倍数"].median(),
        "平均盈利%": x.loc[win, "净收益率"].mean() * 100 if win.any() else np.nan,
        "平均亏损%": x.loc[loss, "净收益率"].mean() * 100 if loss.any() else np.nan,
        "盈亏比": (x.loc[win, "净收益率"].mean()
                 / abs(x.loc[loss, "净收益率"].mean())
                 if win.any() and loss.any() else np.nan),
        "PF": pos / abs(neg) if neg < 0 else np.nan,
        "p5%": x["净收益率"].quantile(0.05) * 100,
        "p95%": x["净收益率"].quantile(0.95) * 100,
        "最差单笔%": x["净收益率"].min() * 100,
        "平均持有": x["持有交易日"].mean(),
        "平均峰值浮盈%": x["峰值浮盈"].mean() * 100,
        "平均最大不利%": x["期间最低"].mean() * 100,
        "top5%盈利占比%": top / pos * 100 if pos > 0 else np.nan,
    })


def cluster_bootstrap_mean(x, cluster, n_boot: int = 2000,
                           seed: int = 42) -> tuple[float, float, float]:
    d = pd.DataFrame({"x": np.asarray(x, float),
                      "c": np.asarray(cluster)}).dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    g = d.groupby("c")["x"].agg(["sum", "count"])
    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return (float(d["x"].mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def build_daily_pnl(trades: pd.DataFrame, data: dict[str, pd.DataFrame],
                    idx_maps: dict[str, dict]) -> pd.DataFrame:
    daily = defaultdict(float)
    for _, t in trades.iterrows():
        code = t["代码"]
        df = data[code]
        c = df["close"].to_numpy(float)
        dates = df["date"].tolist()
        j0 = idx_maps[code][t["买入日"]]
        j1 = idx_maps[code][t["卖出日"]]
        shares = t["股数"]
        entry = t["买入价"]
        exit_px = t["卖出价"]
        daily[dates[j0]] += shares * (c[j0] - entry)
        daily[dates[j0]] -= t["买入费"]
        for j in range(j0 + 1, j1):
            daily[dates[j]] += shares * (c[j] - c[j - 1])
        if j1 > j0:
            daily[dates[j1]] += shares * (exit_px - c[j1 - 1])
        else:
            daily[dates[j1]] += shares * (exit_px - entry)
        daily[dates[j1]] -= t["卖出费"]
    s = pd.Series(daily).sort_index()
    return pd.DataFrame({"日期": s.index, "当日盈亏": s.values})


def main() -> None:
    print("加载数据与信号…", flush=True)
    data = eng.load_pit_data()
    eng.START, eng.END = START, END
    bench = eng.load_benchmark()
    dates = pd.to_datetime(bench["date"]).tolist()
    dates = [d for d in dates if START <= d <= END]
    snaps = eng.load_snapshots()
    uni = eng.Universe(snaps)

    sig = pd.read_csv(HERE / "01_数据" / "信号特征_时点500_内部.csv",
                      parse_dates=["signal_date"])
    sig = sig[sig["in_universe"] == 1].copy()
    sig = add_entry_masks(sig)
    if LIMIT:
        sig = (sig.sample(n=min(LIMIT, len(sig)), random_state=42)
               .sort_values("signal_date").reset_index(drop=True))
    print(f"  候选信号 {len(sig):,}", flush=True)
    print("  买点通过数：", {k: int(sig[v].sum()) for k, v in ENTRIES.items()},
          flush=True)

    idx_maps = {c: {d: i for i, d in enumerate(df["date"].tolist())}
                for c, df in data.items()}
    next_date = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    arr_cache = {}
    stats = Counter()
    rows = []

    for k, r in enumerate(sig.itertuples(index=False), 1):
        code = r.code
        sd = pd.Timestamp(r.signal_date)
        df = data.get(code)
        if df is None or sd not in idx_maps.get(code, {}):
            stats["信号日无行情"] += 1
            continue
        ed = next_date.get(sd)
        if ed is None:
            stats["无下一交易日"] += 1
            continue
        j0 = idx_maps[code].get(ed)
        if j0 is None:
            stats["买入日无行情"] += 1
            continue
        if code not in uni.snapshot_for(ed):
            stats["买入日已非成分"] += 1
            continue
        prev_c = float(df["close"].iat[j0 - 1]) if j0 > 0 else float(df["open"].iat[j0])
        entry = float(df["open"].iat[j0])
        lim = eng.limit_pct(code, ed)
        if entry >= prev_c * (1.0 + lim) * 0.999:
            stats["买入涨停跳过"] += 1
            continue
        if entry <= 0:
            stats["买入价异常"] += 1
            continue

        buy_fee = ALLOC * eng.COST_BUY
        shares = (ALLOC - buy_fee) / entry
        stats["买入成交"] += 1

        if code not in arr_cache:
            arr_cache[code] = (
                df["open"].to_numpy(float),
                df["high"].to_numpy(float),
                df["low"].to_numpy(float),
                df["close"].to_numpy(float),
                df["death_cross"].to_numpy(int),
            )
        o, h, l, c, dc = arr_cache[code]
        dts = df["date"]
        atr = float(r.atr20) if np.isfinite(r.atr20) else 0.03
        typ = getattr(r, "type", "未分类")
        risk_unit = float(np.clip(2.5 * max(atr, 0.005), 0.06, 0.15))

        for exit_name, spec in EXITS.items():
            rr = resolve_exit(spec, atr, typ)
            ex = simulate_exit(o, h, l, c, dc, dts, code, j0, rr)
            proceeds = shares * ex["exit_px"]
            sell_fee = proceeds * eng.COST_SELL
            net = proceeds - sell_fee
            pnl = net - ALLOC
            ret = pnl / ALLOC
            rows.append({
                "卖点": exit_name, "代码": code,
                "名称": getattr(r, "name", ""),
                "信号日": sd, "买入日": ed,
                "卖出日": df["date"].iat[ex["exit_idx"]],
                "买入价": entry, "卖出价": ex["exit_px"],
                "股数": shares, "买入费": buy_fee, "卖出费": sell_fee,
                "投入": ALLOC, "净回款": net, "净盈亏": pnl,
                "净收益率": ret, "R倍数": ret / risk_unit,
                "风险单位": risk_unit,
                "持有交易日": ex["hold_days"],
                "离场原因": ex["reason"],
                "峰值浮盈": ex["peak_profit"],
                "期间最高": ex["mfe"], "期间最低": ex["mae"],
                "信号日收盘": float(r.close),
                "入场跳空": entry / float(r.close) - 1.0,
                "股性类型": typ,
                "股性子类": getattr(r, "subtype", ""),
                "波动分": getattr(r, "s_vol", np.nan),
                "活跃分": getattr(r, "s_heat", np.nan),
                "赔率": getattr(r, "赔率", np.nan),
                "赔率组": getattr(r, "赔率组", None),
                "层数": getattr(r, "层数", np.nan),
                **{v: bool(getattr(r, v)) for v in ENTRIES.values()},
            })
        if k % 10000 == 0:
            print(f"  {k:,}/{len(sig):,}", flush=True)

    tr = pd.DataFrame(rows)
    tr["年份"] = pd.to_datetime(tr["信号日"]).dt.year
    tr["阶段"] = np.select(
        [tr["年份"] <= 2017, tr["年份"] <= 2022],
        ["设计2010-2017", "验证2018-2022"], default="样本外2023-2026")
    tr["阶段2"] = np.where(tr["年份"] <= 2017,
                       "设计2010-2017", "样本外2018-2026")
    tr["月份"] = pd.to_datetime(tr["信号日"]).dt.to_period("M").astype(str)

    # ── 汇总：买点 × 卖点 × 阶段 ──
    print("汇总…", flush=True)
    rows = []
    for entry_name, mask in ENTRIES.items():
        g0 = tr[tr[mask]]
        for exit_name, g1 in g0.groupby("卖点"):
            for stage, g2 in g1.groupby("阶段2"):
                s = summarize(g2)
                s["买点"] = entry_name
                s["卖点"] = exit_name
                s["阶段"] = stage
                rows.append(s)
    summary = pd.DataFrame(rows)
    summary = summary[["买点", "卖点", "阶段", "笔数", "胜率%", "平均收益率%",
                       "收益率中位数%", "平均R", "R中位数", "平均盈利%",
                       "平均亏损%", "盈亏比", "PF", "p5%", "p95%",
                       "最差单笔%", "平均持有", "平均峰值浮盈%",
                       "平均最大不利%", "top5%盈利占比%"]]
    summary.to_csv(OUT / "汇总_买点x卖点.csv", index=False, encoding="utf-8-sig")

    # ── 主买点：F2+指数 ──
    main = tr[tr["m_f2idx"]].copy()
    main_sum = []
    for exit_name, g in main.groupby("卖点"):
        for stage, x in g.groupby("阶段2"):
            s = summarize(x)
            s["卖点"] = exit_name
            s["阶段"] = stage
            main_sum.append(s)
    main_sum = pd.DataFrame(main_sum)
    main_sum = main_sum[["卖点", "阶段", "笔数", "胜率%", "平均收益率%",
                         "收益率中位数%", "平均R", "平均盈利%", "平均亏损%",
                         "盈亏比", "PF", "p5%", "p95%", "最差单笔%",
                         "平均持有", "平均峰值浮盈%", "平均最大不利%",
                         "top5%盈利占比%"]]
    main_sum.to_csv(OUT / "汇总_主买点.csv", index=False, encoding="utf-8-sig")

    # ── 主买点：三阶段 ──
    main3 = []
    for exit_name, g in main.groupby("卖点"):
        for stage, x in g.groupby("阶段"):
            s = summarize(x)
            s["卖点"] = exit_name
            s["阶段"] = stage
            main3.append(s)
    main3 = pd.DataFrame(main3)
    main3 = main3[["卖点", "阶段", "笔数", "胜率%", "平均收益率%",
                   "收益率中位数%", "平均R", "平均盈利%", "平均亏损%",
                   "盈亏比", "PF", "p5%", "p95%", "最差单笔%",
                   "平均持有", "平均峰值浮盈%", "平均最大不利%",
                   "top5%盈利占比%"]]
    main3.to_csv(OUT / "汇总_主买点_三阶段.csv", index=False,
                 encoding="utf-8-sig")

    # ── 分类型 ──
    type_rows = []
    for (typ, exit_name, stage), g in main.groupby(
            ["股性类型", "卖点", "阶段2"]):
        s = summarize(g)
        s["股性类型"] = typ
        s["卖点"] = exit_name
        s["阶段"] = stage
        type_rows.append(s)
    type_sum = pd.DataFrame(type_rows)
    type_sum = type_sum[["股性类型", "卖点", "阶段", "笔数", "胜率%",
                         "平均收益率%", "平均R", "平均盈利%", "平均亏损%",
                         "PF", "p5%", "p95%", "最差单笔%", "平均持有",
                         "平均峰值浮盈%", "平均最大不利%"]]
    type_sum.to_csv(OUT / "分类型_主买点.csv", index=False,
                    encoding="utf-8-sig")

    # ── 年度 ──
    year_rows = []
    for (exit_name, year), g in main.groupby(["卖点", "年份"]):
        s = summarize(g)
        s["卖点"] = exit_name
        s["年份"] = int(year)
        year_rows.append(s)
    year = pd.DataFrame(year_rows)
    year.to_csv(OUT / "分年_主买点.csv", index=False, encoding="utf-8-sig")

    # ── 离场原因 ──
    reason = (main.groupby(["卖点", "阶段2", "离场原因"])
              .size().rename("笔数").reset_index())
    reason.to_csv(OUT / "离场原因_主买点.csv", index=False,
                  encoding="utf-8-sig")

    # ── 条件层数 ──
    layer_rows = []
    for (exit_name, layer, stage), g in tr[tr["m_all"]].groupby(
            ["卖点", "层数", "阶段2"]):
        s = summarize(g)
        s["卖点"] = exit_name
        s["层数"] = int(layer)
        s["阶段"] = stage
        layer_rows.append(s)
    layer = pd.DataFrame(layer_rows)
    layer = layer[["卖点", "层数", "阶段", "笔数", "胜率%", "平均收益率%",
                   "平均R", "PF", "p5%", "最差单笔%"]]
    layer.to_csv(OUT / "分层_条件数.csv", index=False, encoding="utf-8-sig")

    # ── 赔率分层 ──
    odds_rows = []
    for entry_name, mask in [("全信号", "m_all"), ("F2+指数", "m_f2idx")]:
        for (group, exit_name, stage), g in tr[tr[mask] & tr["赔率组"].notna()].groupby(
                ["赔率组", "卖点", "阶段2"], observed=True):
            s = summarize(g)
            s["买点"] = entry_name
            s["赔率组"] = str(group)
            s["卖点"] = exit_name
            s["阶段"] = stage
            odds_rows.append(s)
    odds = pd.DataFrame(odds_rows)
    odds = odds[["买点", "赔率组", "卖点", "阶段", "笔数", "胜率%",
                 "平均收益率%", "平均R", "PF"]]
    odds.to_csv(OUT / "分层_赔率.csv", index=False, encoding="utf-8-sig")

    # ── bootstrap：主买点，OOS ──
    boot_rows = []
    oos = main[main["阶段2"] == "样本外2018-2026"]
    for exit_name, g in oos.groupby("卖点"):
        m, lo, hi = cluster_bootstrap_mean(g["净收益率"], g["月份"])
        mr, lor, hir = cluster_bootstrap_mean(g["R倍数"], g["月份"])
        boot_rows.append({"卖点": exit_name, "笔数": len(g),
                          "平均收益率%": m * 100, "收益CI下%": lo * 100,
                          "收益CI上%": hi * 100,
                          "平均R": mr, "R CI下": lor, "R CI上": hir})
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "bootstrap_主买点样本外.csv", index=False,
                encoding="utf-8-sig")

    # ── 配对差异：同一信号、不同卖点 ──
    pair_rows = []
    for stage in ["设计2010-2017", "样本外2018-2026"]:
        g = main[main["阶段2"] == stage]
        pivot = g.pivot_table(index=["代码", "信号日"], columns="卖点",
                              values=["净收益率", "R倍数"], aggfunc="first")
        for a, b in [("类型自适应ATR", "ATR自适应"),
                     ("类型自适应ATR", "b3_15_10"),
                     ("类型自适应ATR", "b3_10_8_硬止损8"),
                     ("ATR自适应", "b3_10_8_硬止损8"),
                     ("ATR追踪_无硬止损", "死叉"),
                     ("b3_15_10", "死叉"),
                     ("b3_10_8", "死叉")]:
            if (a not in g["卖点"].unique()) or (b not in g["卖点"].unique()):
                continue
            d = (pivot[("净收益率", a)] - pivot[("净收益率", b)]).dropna()
            dr = (pivot[("R倍数", a)] - pivot[("R倍数", b)]).dropna()
            months = pd.Series(d.index.get_level_values("信号日")).dt.to_period("M").astype(str).values
            m, lo, hi = cluster_bootstrap_mean(d.values, months)
            mr, lor, hir = cluster_bootstrap_mean(dr.values, months)
            pair_rows.append({"阶段": stage, "卖点A": a, "卖点B": b,
                              "配对笔数": len(d),
                              "平均收益差%": m * 100,
                              "收益差CI下%": lo * 100,
                              "收益差CI上%": hi * 100,
                              "平均R差": mr,
                              "R差CI下": lor, "R差CI上": hir})
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT / "配对差异_主买点.csv", index=False, encoding="utf-8-sig")

    # ── 交易记录 ──
    for exit_name in ["死叉", "b3_15_10", "b3_10_8_硬止损8",
                      "ATR自适应", "类型自适应ATR"]:
        g = main[main["卖点"] == exit_name].copy()
        g = g.sort_values(["信号日", "代码"])
        g.to_csv(OUT / f"交易_F2+指数_{exit_name}.csv", index=False,
                 encoding="utf-8-sig")
    # 各买点主卖点交易记录
    for entry_name, mask in ENTRIES.items():
        g = tr[tr[mask] & (tr["卖点"] == "类型自适应ATR")].copy()
        g = g.sort_values(["信号日", "代码"])
        g.to_csv(OUT / f"交易_{entry_name}_类型自适应ATR.csv", index=False,
                 encoding="utf-8-sig")

    # ── 权益（累计盈亏，不是组合净值） ──
    eq_rows = []
    for exit_name in ["死叉", "b3_15_10", "b3_10_8_硬止损8",
                      "ATR自适应", "类型自适应ATR"]:
        g = main[main["卖点"] == exit_name]
        pnl = build_daily_pnl(g, data, idx_maps)
        pnl["卖点"] = exit_name
        pnl["累计盈亏"] = pnl["当日盈亏"].cumsum()
        pnl["峰值"] = pnl["累计盈亏"].cummax()
        pnl["回撤"] = pnl["累计盈亏"] - pnl["峰值"]
        eq_rows.append(pnl)
    eq = pd.concat(eq_rows, ignore_index=True)
    eq.to_csv(OUT / "权益_主买点.csv", index=False, encoding="utf-8-sig")

    # ── 统计日志 ──
    with open(OUT / "运行日志.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\n候选信号: {len(sig)}\n")
        f.write(f"交易行数: {len(tr)}\n")
        f.write("\n买点通过数：\n")
        for k, v in ENTRIES.items():
            f.write(f"  {k}: {int(sig[v].sum())}\n")
        f.write("\n主买点 F2+指数 分阶段：\n")
        f.write(main_sum.round(4).to_string(index=False))
        f.write("\n\n主买点 F2+指数 三阶段：\n")
        f.write(main3.round(4).to_string(index=False))
        f.write("\n\nbootstrap：\n")
        f.write(boot.round(4).to_string(index=False))
        f.write("\n\n配对差异：\n")
        f.write(pair.round(4).to_string(index=False))
        f.write("\n")
    print(f"完成：{OUT}", flush=True)


if __name__ == "__main__":
    main()
