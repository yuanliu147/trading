#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2 槽 × 2 万本金的小资金回测。

约束：
- 最多同时持有 2 只；T 日收盘金叉，T+1 开盘买；
- 100 股整手（688 开头按 200 股一手）；不足一手不买；
- 买入佣金 0.025%（最低 5 元），卖出 0.125%（最低 5 元）；
- 单笔计划亏损 <= 本金的 3%（600 元），并同时不超过当前权益的 3%：
  risk_budget = min(600, 3% * 当前权益)
  目标市值 = min(risk_budget / risk_stop, 当前权益 * 50%, 可用现金)
  risk_stop 用 8%（止损刚好成交）或 10%（8% 止损 + 2% 跳空缓冲）；
- 离场：b3(10%/8%，前一日已知峰值) + 8% 硬止损（收盘确认）+ 死叉；
- 涨停不买、跌停顺延、停牌按最后收盘估值、退市/长期停牌按最后收盘强平；
- 期末未平仓按最后收盘强平，便于逐笔统计。

输出：
- 两槽小资金_汇总.csv
- 两槽小资金_年度.csv
- 两槽小资金_权益_*.csv
- 两槽小资金_交易_*.csv
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = HERE / "01_数据"
OUT = HERE / "05_两槽小资金"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

START = pd.Timestamp("2010-07-01")
END = pd.Timestamp("2026-10-13")
INITIAL = 20_000.0
COST_BUY = 0.00025
COST_SELL = 0.00125
MIN_COMM = 5.0
HARD_STOP = 0.08
TRAIL_TRIGGER = 0.10
TRAIL_STOP = 0.08

REASON_CN = {"hard_stop": "硬止损", "b3_trailing": "b3追踪",
             "death_cross": "死叉", "退市/长期停牌": "退市/长期停牌", "期末": "期末强平"}


def digits(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())[:6]


def lot_size(code: str) -> int:
    return 200 if digits(code).startswith("688") else 100


def load_inputs():
    data = eng.load_pit_data()
    eng.START, eng.END = START, END
    bench = eng.load_benchmark()
    dates = [d for d in pd.to_datetime(bench["date"]) if START <= d <= END]
    uni = eng.Universe(eng.load_snapshots())
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", parse_dates=["signal_date"])
    sig = sig[sig["in_universe"] == 1].copy()
    sig["hist_raw"] = sig["hist_norm"] * sig["close"]
    return data, bench, dates, uni, sig


def build_masks(sig: pd.DataFrame) -> dict[str, pd.Series]:
    A = (sig.ret20 <= 0.15) & (sig.dist_ma20 <= 0.08) & (sig.rsi14 <= 70)
    G = sig.n_golden_60 >= 3
    V = sig.s_vol <= 0.72
    Q = sig.amt5_60 <= 0.26
    T = sig.type != "高波活跃型"
    K = sig.dist_ma60 >= -0.08
    return {
        "F1_MA20": A & G & V & Q & T & (sig.idx_above_ma20 == 1),
        "F2_MA20": A & G & V & Q & K & (sig.idx_above_ma20 == 1),
        "F1_无MA": A & G & V & Q & T,
    }


def run_config(name: str, sig: pd.DataFrame, mask: pd.Series, score_col: str,
               data: dict, bench: pd.DataFrame, dates: list, uni,
               max_slots: int = 2, risk_stop: float = 0.10,
               exit_mode: str = "b3", initial: float = INITIAL,
               hard_stop: float = HARD_STOP) -> dict:
    """exit_mode: b3 / death。"""
    s = sig.loc[mask].copy()
    s = s[np.isfinite(s[score_col])]
    by_date = {
        d: list(g.sort_values(score_col, ascending=False)[["code", "name", score_col]].itertuples(index=False, name=None))
        for d, g in s.groupby("signal_date")
    }
    date_maps = {c: {d: i for i, d in enumerate(df["date"].tolist())} for c, df in data.items()}
    last_date = {c: df["date"].iloc[-1] for c, df in data.items()}
    cash = float(initial)
    positions: dict[str, dict] = {}
    pending_buys: list[tuple] = []
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    eq_records: list[dict] = []
    stats = Counter()
    entry_id = 0

    def equity_now_at_open() -> float:
        mv = sum(p["shares"] * last_close.get(c, p["entry_price"]) for c, p in positions.items())
        return cash + mv

    def close_position(code: str, date: pd.Timestamp, px: float, reason: str,
                       price_date: pd.Timestamp) -> None:
        nonlocal cash
        p = positions[code]
        proceeds = p["shares"] * px
        sell_fee = max(proceeds * COST_SELL, MIN_COMM)
        net = proceeds - sell_fee
        cash += net
        pnl = net - p["total_invested"]
        trades.append({
            "编号": p["entry_id"], "代码": code, "名称": p["name"],
            "信号日": p["signal_date"], "买入日": p["entry_date"],
            "买入价": p["entry_price"], "卖出日": date, "卖出价": px,
            "计价日": price_date, "股数": p["shares"],
            "投入资金": p["total_invested"], "净回款": net,
            "净盈亏": pnl, "净收益率": pnl / p["total_invested"] if p["total_invested"] > 0 else 0.0,
            "净收益率%": pnl / p["total_invested"] * 100 if p["total_invested"] > 0 else 0.0,
            "持有交易日": p["hold_days"], "离场原因": REASON_CN.get(reason, reason),
            "峰值浮盈": p["peak"] / p["entry_price"] - 1.0,
            "峰值浮盈%": (p["peak"] / p["entry_price"] - 1.0) * 100,
        })
        del positions[code]

    for di, date in enumerate(dates):
        # 1) 昨日挂起的卖出，今日开盘执行
        for code in list(positions):
            p = positions[code]
            if p["pending_exit"] is None:
                continue
            df = data.get(code)
            idx = date_maps[code].get(date) if df is not None else None
            if df is None or idx is None:
                if df is not None and last_date[code] < date:
                    close_position(code, date, last_close.get(code, p["entry_price"]),
                                   "退市/长期停牌", last_date[code])
                    stats["退市强平"] += 1
                continue
            prev_close = float(df["close"].iat[idx - 1]) if idx > 0 else float(df["open"].iat[idx])
            today_open = float(df["open"].iat[idx])
            lim = eng.limit_pct(code, date)
            if today_open <= prev_close * (1.0 - lim) * 1.001:
                stats["卖出跌停顺延"] += 1
                continue
            close_position(code, date, today_open, p["pending_reason"], date)
            stats["卖出成交"] += 1

        # 2) 昨日收盘挂起的买入，今日开盘按排序依次尝试
        for code, score, nm, sig_date in pending_buys:
            if len(positions) >= max_slots:
                break
            if code in positions:
                continue
            df = data.get(code)
            idx = date_maps[code].get(date) if df is not None else None
            if df is None or idx is None:
                stats["买入无行情"] += 1
                continue
            if code not in uni.snapshot_for(date):
                stats["买入已非成分"] += 1
                continue
            prev_close = float(df["close"].iat[idx - 1]) if idx > 0 else float(df["open"].iat[idx])
            today_open = float(df["open"].iat[idx])
            if today_open <= 0:
                stats["买入价异常"] += 1
                continue
            lim = eng.limit_pct(code, date)
            if today_open >= prev_close * (1.0 + lim) * 0.999:
                stats["买入涨停跳过"] += 1
                continue

            eq = equity_now_at_open()
            risk_budget = min(initial * 0.03, eq * 0.03)
            target = min(risk_budget / risk_stop, eq * 0.5, cash)
            lot = lot_size(code)
            if target < today_open * lot + MIN_COMM:
                stats["买入一手太贵"] += 1
                continue
            # 以 target 为含费用上限，按整手向下取
            fee0 = max(target * COST_BUY, MIN_COMM)
            shares = int((target - fee0) // (today_open * lot)) * lot
            while shares >= lot:
                gross = shares * today_open
                fee = max(gross * COST_BUY, MIN_COMM)
                cost = gross + fee
                if cost <= target and cost <= cash:
                    break
                shares -= lot
            if shares < lot:
                stats["买入资金不足"] += 1
                continue

            cash -= cost
            entry_id += 1
            positions[code] = {
                "entry_id": entry_id, "name": nm, "signal_date": sig_date,
                "entry_date": date, "entry_price": today_open, "shares": float(shares),
                "total_invested": float(cost), "buy_fee": float(fee),
                "peak": today_open, "hold_days": 0,
                "pending_exit": None, "pending_reason": "",
            }
            last_close[code] = today_open
            stats["买入成交"] += 1

        # 3) 收盘：更新持仓、检查离场
        for code, p in list(positions.items()):
            df = data.get(code)
            idx = date_maps[code].get(date) if df is not None else None
            if df is None or idx is None:
                if df is not None and last_date[code] < date:
                    close_position(code, date, last_close.get(code, p["entry_price"]),
                                   "退市/长期停牌", last_date[code])
                    stats["退市强平"] += 1
                continue
            p["hold_days"] += 1
            hi = float(df["high"].iat[idx]); lo = float(df["low"].iat[idx]); cl = float(df["close"].iat[idx])
            last_close[code] = cl
            if p["pending_exit"] is None:
                if cl <= p["entry_price"] * (1.0 - hard_stop):
                    p["pending_exit"] = "full"; p["pending_reason"] = "hard_stop"
                elif exit_mode == "b3" and p["peak"] / p["entry_price"] - 1.0 >= TRAIL_TRIGGER \
                        and lo <= p["peak"] * (1.0 - TRAIL_STOP):
                    p["pending_exit"] = "full"; p["pending_reason"] = "b3_trailing"
                elif int(df["death_cross"].iat[idx]) == 1:
                    p["pending_exit"] = "full"; p["pending_reason"] = "death_cross"
            p["peak"] = max(p["peak"], hi)

        # 4) 按收盘估值
        mv = sum(p["shares"] * last_close.get(c, p["entry_price"]) for c, p in positions.items())
        eq = cash + mv
        eq_records.append({"date": date, "equity": eq, "n_positions": len(positions),
                           "gross_market_value": mv,
                           "market_value_ratio": mv / eq if eq > 0 else 0.0})

        # 5) 生成明日买单：把“明日开盘会卖”的仓位也算作可用槽位
        pending_buys = []
        if di + 1 < len(dates):
            free = max_slots - len(positions) + sum(1 for p in positions.values() if p["pending_exit"])
            if free > 0:
                cands = by_date.get(date, [])
                picked = []
                for code, nm, score in cands:
                    if code in positions:
                        continue
                    picked.append((code, score, nm, date))
                    if len(picked) >= max(10, free + 5):
                        break
                pending_buys = picked

    # 期末强平，便于逐笔统计（权益已在最后一天按收盘计）
    for code, p in list(positions.items()):
        px = last_close.get(code, p["entry_price"])
        pdte = last_date[code] if last_date[code] <= dates[-1] else dates[-1]
        close_position(code, dates[-1], px, "期末", pdte)
        stats["期末强平"] += 1
    if eq_records:
        eq_records[-1]["equity"] = cash
        eq_records[-1]["n_positions"] = 0
        eq_records[-1]["gross_market_value"] = 0.0
        eq_records[-1]["market_value_ratio"] = 0.0

    tr = pd.DataFrame(trades)
    eq = pd.DataFrame(eq_records)
    return {"name": name, "equity": eq, "trades": tr, "stats": dict(stats),
            "final_equity": float(eq["equity"].iloc[-1]), "initial": initial}


def add_metrics(name: str, r: dict, bench: pd.DataFrame) -> list[dict]:
    rows = []
    eq = r["equity"].copy()
    tr = r["trades"].copy()
    for label, start in [("全期", START), ("2018起", pd.Timestamp("2018-01-01")),
                         ("2023起", pd.Timestamp("2023-01-01"))]:
        e = eq[eq["date"] >= start].copy()
        if len(e) < 2:
            continue
        m = eng.equity_metrics(e)
        t = tr[pd.to_datetime(tr["买入日"]) >= start] if len(tr) else tr
        if len(t):
            t_en = t.rename(columns={"净收益率": "total_ret", "净盈亏": "total_pnl",
                                     "持有交易日": "hold_days"})
            ts = eng.trade_metrics(t_en)
        else:
            ts = {"n_trades": 0, "win_rate": 0, "pf": 0, "avg_hold_days": 0, "avg_ret": 0}
        total_pnl = float(t["净盈亏"].sum()) if len(t) else 0.0
        top5 = float(t.sort_values("净盈亏", ascending=False)["净盈亏"].head(5).sum()) if len(t) else 0.0
        losses = -t["净盈亏"].min() if len(t) and t["净盈亏"].min() < 0 else 0.0
        n_over3 = int((t["净盈亏"] < -r["initial"] * 0.03).sum()) if len(t) else 0
        try:
            sm = eng.same_exposure_benchmark(e, bench)
        except Exception:
            sm = {"same_ann": np.nan, "same_mdd": np.nan}
        rows.append({
            "配置": name, "区间": label, "年化": m["ann_return"], "最大回撤": m["mdd"],
            "Calmar": m["calmar"], "Sharpe": m["sharpe"],
            "交易": ts["n_trades"], "胜率": ts["win_rate"], "平均每笔": ts["avg_ret"],
            "PF": ts["pf"], "平均仓位": m["avg_exposure"], "平均持仓数": m["avg_positions"],
            "最大单笔亏损": round(losses, 0), "最大单笔亏损占本金%": round(losses / r["initial"] * 100, 2),
            "亏损>3%笔数": n_over3,
            "前5笔占比%": round(top5 / total_pnl * 100, 2) if total_pnl > 0 else np.nan,
            "剔除前5笔盈亏": round(total_pnl - top5, 0),
            "同暴露基准年化": sm.get("same_ann"), "同暴露基准回撤": sm.get("same_mdd"),
        })
    return rows


def yearly(r: dict) -> pd.DataFrame:
    eq = r["equity"].copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    rows = []
    for y, g in eq.groupby("year"):
        g = g.sort_values("date")
        rows.append({"配置": r["name"], "年份": int(y),
                     "收益%": round((g["equity"].iloc[-1] / g["equity"].iloc[0] - 1) * 100, 2),
                     "最大回撤%": round(float((g["equity"] / g["equity"].cummax() - 1).min()) * 100, 2),
                     "平均仓位%": round(float(g["market_value_ratio"].mean()) * 100, 1)})
    return pd.DataFrame(rows)


def main():
    print("加载数据…", flush=True)
    data, bench, dates, uni, sig = load_inputs()
    masks = build_masks(sig)
    configs = [
        ("F1_MA20_b3_风险10%", "F1_MA20", "hist_norm", 0.10, "b3", 2),
        ("F1_MA20_b3_风险8%", "F1_MA20", "hist_norm", 0.08, "b3", 2),
        ("F2_MA20_b3_风险10%", "F2_MA20", "hist_norm", 0.10, "b3", 2),
        ("F2_MA20_b3_风险8%", "F2_MA20", "hist_norm", 0.08, "b3", 2),
        ("F1_MA20_死叉_风险10%", "F1_MA20", "hist_norm", 0.10, "death", 2),
        ("F1_MA20_b3_风险15%", "F1_MA20", "hist_norm", 0.15, "b3", 2),
        ("F1_MA20_b3_风险10%_1槽", "F1_MA20", "hist_norm", 0.10, "b3", 1),
        ("F1_MA20_b3_风险10%_原始排序", "F1_MA20", "hist_raw", 0.10, "b3", 2),
    ]
    all_rows = []
    yearly_rows = []
    for name, mk, score, risk, mode, slots in configs:
        print(f"\n运行 {name} …", flush=True)
        r = run_config(name, sig, masks[mk], score, data, bench, dates, uni,
                       max_slots=slots, risk_stop=risk, exit_mode=mode)
        all_rows += add_metrics(name, r, bench)
        yearly_rows.append(yearly(r))
        eq = r["equity"].copy(); eq.columns = ["日期", "权益", "持仓数", "持仓市值", "仓位比例"]
        eq.to_csv(OUT / f"权益_{name}.csv", index=False, encoding="utf-8-sig")
        tr = r["trades"].copy()
        tr.to_csv(OUT / f"交易_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  期末 {r['final_equity']:,.0f} 元；交易 {len(tr)}；统计 {r['stats']}", flush=True)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT / "两槽小资金_汇总.csv", index=False, encoding="utf-8-sig")
    yearly_df = pd.concat(yearly_rows, ignore_index=True)
    yearly_df.to_csv(OUT / "两槽小资金_年度.csv", index=False, encoding="utf-8-sig")
    cols = ["配置", "区间", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率",
            "平均每笔", "PF", "平均仓位", "平均持仓数", "最大单笔亏损占本金%",
            "亏损>3%笔数", "前5笔占比%", "同暴露基准年化", "同暴露基准回撤"]
    print("\n", summary[cols].round(2).to_string(index=False))
    print(f"\n保存到 {OUT}")


if __name__ == "__main__":
    main()
