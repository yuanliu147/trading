#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全信号、每笔固定 1 万元、本金无限的卖出规则对照。

- 股票池：时点中证 500（含退市），信号日与买入日都核对成分；
- 入场：T 日收盘金叉，T+1 开盘买；涨停不买；买入费 0.025%；
- 卖出：T 日收盘信号，T+1 开盘卖；跌停顺延；卖出费 0.125%；
- 每笔独立 1 万元，不共享资金、不限制同时持仓数；
- 同一批信号分别跑死叉、b3、硬止损、时间止损等规则，便于配对比较。

输出：
- 全信号交易_分规则.csv
- 全信号汇总_分规则.csv
- 全信号年度_分规则.csv
- 全信号权益_分规则.csv
- 运行时统计.txt
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "01_数据"

spec = importlib.util.spec_from_file_location(
    "strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

START = pd.Timestamp("2010-07-01")
END = pd.Timestamp("2026-10-13")
ALLOC = 10_000.0

# 规则：strategy, trigger, trail, hard_stop, time_stop
RULES = {
    "死叉": dict(strategy="death", trigger=None, trail=None, hard_stop=None, time_stop=None),
    "b3_10_8": dict(strategy="b3", trigger=0.10, trail=0.08, hard_stop=None, time_stop=None),
    "b3_8_6": dict(strategy="b3", trigger=0.08, trail=0.06, hard_stop=None, time_stop=None),
    "b3_12_8": dict(strategy="b3", trigger=0.12, trail=0.08, hard_stop=None, time_stop=None),
    "b3_15_10": dict(strategy="b3", trigger=0.15, trail=0.10, hard_stop=None, time_stop=None),
    "死叉_硬止损8": dict(strategy="death", trigger=None, trail=None, hard_stop=0.08, time_stop=None),
    "死叉_硬止损10": dict(strategy="death", trigger=None, trail=None, hard_stop=0.10, time_stop=None),
    "b3_10_8_硬止损8": dict(strategy="b3", trigger=0.10, trail=0.08, hard_stop=0.08, time_stop=None),
    "b3_10_8_硬止损12": dict(strategy="b3", trigger=0.10, trail=0.08, hard_stop=0.12, time_stop=None),
    "b3_10_8_时间40": dict(strategy="b3", trigger=0.10, trail=0.08, hard_stop=None, time_stop=40),
    "死叉_时间40": dict(strategy="death", trigger=None, trail=None, hard_stop=None, time_stop=40),
}


def simulate(df: pd.DataFrame, code: str, idx_e: int, rule: dict):
    """从买入日开盘开始模拟，返回离场信息与路径。"""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    dc = df["death_cross"].to_numpy(int)
    n = len(df)
    entry = o[idx_e]
    peak = entry
    pending = None
    pending_reason = ""
    exit_idx = None
    exit_px = None
    exit_at_open = False
    reason = ""

    for j in range(idx_e, n):
        if pending is not None:
            if j > idx_e:
                prev_c = c[j - 1]
                lim = eng.limit_pct(code, df["date"].iat[j])
                if o[j] <= prev_c * (1.0 - lim) * 1.001:
                    # 跌停卖不出，继续等
                    continue
                exit_idx = j
                exit_px = o[j]
                exit_at_open = True
                reason = pending_reason
                break
            # 理论上不会在买入日执行挂起卖出
            pending = None

        cl, lo, hi = c[j], l[j], h[j]
        # 1) 硬止损（收盘确认）
        if rule["hard_stop"] is not None and cl <= entry * (1.0 - rule["hard_stop"]):
            pending = "hard"
            pending_reason = "硬止损"
        # 2) 时间止损
        elif rule["time_stop"] is not None and (j - idx_e + 1) >= rule["time_stop"]:
            pending = "time"
            pending_reason = "时间止损"
        # 3) b3 追踪（只用前一日收盘时已知的峰值）
        elif (rule["strategy"] == "b3"
              and peak / entry - 1.0 >= rule["trigger"]
              and lo <= peak * (1.0 - rule["trail"])):
            pending = "b3"
            pending_reason = "b3追踪"
        # 4) 死叉兜底
        elif dc[j] == 1:
            pending = "death"
            pending_reason = "死叉"

        # 当天新高只用于下一天
        if hi > peak:
            peak = hi

    if exit_idx is None:
        exit_idx = n - 1
        exit_px = c[-1]
        exit_at_open = False
        reason = "期末/退市"
    last_hold = exit_idx - 1 if exit_at_open and exit_idx > idx_e else exit_idx
    last_hold = max(idx_e, min(last_hold, n - 1))
    seg = slice(idx_e, last_hold + 1)
    mfe = float(h[seg].max()) / entry - 1.0
    mae = float(l[seg].min()) / entry - 1.0
    return dict(exit_idx=exit_idx, exit_px=exit_px, reason=reason,
                exit_at_open=exit_at_open,
                hold_days=exit_idx - idx_e, peak_profit=peak / entry - 1.0,
                mfe=mfe, mae=mae)


def build_daily_pnl(trades: pd.DataFrame, data: dict[str, pd.DataFrame],
                    idx_maps: dict[str, dict]) -> pd.DataFrame:
    """按日盯市累计盈亏（元）。只对已成交交易。"""
    daily = defaultdict(float)
    for _, t in trades.iterrows():
        code = t["代码"]
        df = data[code]
        o = df["open"].to_numpy(float)
        c = df["close"].to_numpy(float)
        dates = df["date"].tolist()
        j0 = idx_maps[code][t["买入日"]]
        j1 = idx_maps[code][t["卖出日"]]
        shares = t["股数"]
        entry = t["买入价"]
        exit_px = t["卖出价"]
        # 买入日收盘变动
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
    return s.rename("当日盈亏").reset_index().rename(columns={"index": "日期"})


def main() -> None:
    print("加载数据与信号…", flush=True)
    data = eng.load_pit_data()
    eng.START, eng.END = START, END
    bench = eng.load_benchmark()
    dates = pd.to_datetime(bench["date"]).tolist()
    dates = [d for d in dates if START <= d <= END]
    snaps = eng.load_snapshots()
    uni = eng.Universe(snaps)

    sig = pd.read_csv(OUT / "信号特征_时点500_内部.csv", parse_dates=["signal_date"])
    sig = sig[sig["in_universe"] == 1].copy()
    print(f"  候选信号 {len(sig):,}", flush=True)

    idx_maps = {c: {d: i for i, d in enumerate(df["date"].tolist())}
                for c, df in data.items()}
    next_date = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
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

        # 固定持有期（不计费，仅用于买入点分析）
        fixed = {}
        for hz in (5, 10, 20, 40):
            j = j0 + hz
            fixed[f"fwd{hz}"] = (float(df["close"].iat[j]) / entry - 1.0
                                 if j < len(df) else np.nan)

        for rule_name, rule in RULES.items():
            ex = simulate(df, code, j0, rule)
            proceeds = shares * ex["exit_px"]
            sell_fee = proceeds * eng.COST_SELL
            net = proceeds - sell_fee
            pnl = net - ALLOC
            rows.append({
                "规则": rule_name, "代码": code, "名称": r.name,
                "信号日": sd, "买入日": ed,
                "卖出日": df["date"].iat[ex["exit_idx"]],
                "买入价": entry, "卖出价": ex["exit_px"],
                "股数": shares, "买入费": buy_fee, "卖出费": sell_fee,
                "投入": ALLOC, "净回款": net, "净盈亏": pnl,
                "净收益率": pnl / ALLOC,
                "持有交易日": ex["hold_days"], "离场原因": ex["reason"],
                "峰值浮盈": ex["peak_profit"], "期间最高": ex["mfe"],
                "期间最低": ex["mae"],
                "信号日收盘": float(r.close),
                "入场跳空": entry / float(r.close) - 1.0,
                "fwd5": fixed["fwd5"], "fwd10": fixed["fwd10"],
                "fwd20": fixed["fwd20"], "fwd40": fixed["fwd40"],
                "股性类型": getattr(r, "type", "未分类"),
                "股性子类": getattr(r, "subtype", ""),
                "波动分": getattr(r, "s_vol", np.nan),
                "活跃分": getattr(r, "s_heat", np.nan),
            })
        if k % 10000 == 0:
            print(f"  {k:,}/{len(sig):,}", flush=True)

    tr = pd.DataFrame(rows)
    tr["年份"] = pd.to_datetime(tr["信号日"]).dt.year
    tr["阶段"] = np.select(
        [tr["年份"] <= 2017, tr["年份"] <= 2022],
        ["设计 2010-2017", "验证 2018-2022"], default="样本外 2023-2026")
    tr.to_csv(OUT / "全信号交易_分规则.csv", index=False, encoding="utf-8-sig")

    # 汇总
    def summ(x: pd.DataFrame) -> pd.Series:
        n = len(x)
        win = x[x["净收益率"] > 0]
        loss = x[x["净收益率"] <= 0]
        return pd.Series({
            "笔数": n,
            "胜率": len(win) / n * 100 if n else np.nan,
            "平均收益率": x["净收益率"].mean() * 100 if n else np.nan,
            "收益率中位数": x["净收益率"].median() * 100 if n else np.nan,
            "平均盈利": win["净收益率"].mean() * 100 if len(win) else np.nan,
            "平均亏损": loss["净收益率"].mean() * 100 if len(loss) else np.nan,
            "盈亏比": (win["净收益率"].mean() / abs(loss["净收益率"].mean())
                     if len(win) and len(loss) else np.nan),
            "PF": (win["净盈亏"].sum() / abs(loss["净盈亏"].sum())
                   if len(loss) else np.nan),
            "总净盈亏万": x["净盈亏"].sum() / 1e4,
            "平均持有": x["持有交易日"].mean(),
            "平均峰值浮盈": x["峰值浮盈"].mean() * 100,
            "平均最大不利": x["期间最低"].mean() * 100,
            "曾达10%占比": x["峰值浮盈"].ge(0.10).mean() * 100,
        })

    summ_rows = []
    for name, g in tr.groupby("规则"):
        s = summ(g)
        s["规则"] = name
        summ_rows.append(s)
    summary = pd.DataFrame(summ_rows).set_index("规则")
    summary = summary[["笔数", "胜率", "平均收益率", "收益率中位数", "平均盈利",
                       "平均亏损", "盈亏比", "PF", "总净盈亏万", "平均持有",
                       "平均峰值浮盈", "平均最大不利", "曾达10%占比"]]
    summary.to_csv(OUT / "全信号汇总_分规则.csv", encoding="utf-8-sig")

    # 分阶段
    stage_rows = []
    for (name, stage), g in tr.groupby(["规则", "阶段"]):
        s = summ(g)
        s["规则"] = name
        s["阶段"] = stage
        stage_rows.append(s)
    stage = pd.DataFrame(stage_rows)
    stage.to_csv(OUT / "全信号分阶段_分规则.csv", index=False, encoding="utf-8-sig")

    # 年度
    yearly_rows = []
    for (name, y), g in tr.groupby(["规则", "年份"]):
        s = summ(g)
        s["规则"] = name
        s["年份"] = int(y)
        yearly_rows.append(s)
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(OUT / "全信号年度_分规则.csv", index=False, encoding="utf-8-sig")

    # 权益：只给四条代表规则算逐日盯市
    eq_rows = []
    for name in ["死叉", "b3_10_8", "b3_10_8_硬止损8", "b3_8_6"]:
        g = tr[tr["规则"] == name]
        pnl = build_daily_pnl(g, data, idx_maps)
        pnl["规则"] = name
        pnl["累计盈亏"] = pnl["当日盈亏"].cumsum()
        pnl["峰值"] = pnl["累计盈亏"].cummax()
        pnl["回撤"] = pnl["累计盈亏"] - pnl["峰值"]
        eq_rows.append(pnl)
    eq = pd.concat(eq_rows, ignore_index=True)
    eq.to_csv(OUT / "全信号权益_分规则.csv", index=False, encoding="utf-8-sig")

    with open(HERE / "02_模拟统计.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\n候选信号: {len(sig)}\n")
        f.write(f"交易行数: {len(tr)}\n")
        for name, g in tr.groupby("规则"):
            pnl = eq[eq["规则"] == name]["累计盈亏"]
            mdd = (eq[eq["规则"] == name]["回撤"].min() / 1e4
                   if len(pnl) else np.nan)
            f.write(f"{name}: 笔数={len(g)}, 总盈亏={g['净盈亏'].sum()/1e4:.1f}万, "
                    f"最大回撤={mdd:.1f}万\n")

    print("\n汇总：")
    print(summary.round(3).to_string())
    print("\n统计：")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    print(f"\n完成：{OUT}")


if __name__ == "__main__":
    main()
