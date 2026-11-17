#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实组合回测：有限资金、最多 N 槽、T+1、涨跌停、停牌、退市、成本。

对照：
- 全部信号 vs 三套入场过滤；
- 死叉 vs b3(10%,8%)；
- 10 槽×10%（主口径）；随后对优胜方案补 5 槽×20%、1 槽×100%。

所有过滤条件只用信号日及以前的数据；过滤集合来自 01_数据/信号特征_时点500_内部.csv。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "01_数据"
OUT = HERE / "03_真实回测"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

FULL_START = pd.Timestamp("2010-07-01")
FULL_END = pd.Timestamp("2026-10-13")
SUB_START = pd.Timestamp("2018-07-01")


def build_filters(sig: pd.DataFrame) -> dict[str, set]:
    s = sig[sig["in_universe"] == 1].copy()
    A = (s["ret20"] <= 0.15) & (s["dist_ma20"] <= 0.08) & (s["rsi14"] <= 70)
    G = s["n_golden_60"] >= 3
    V = s["s_vol"] <= 0.72
    Q = s["amt5_60"] <= 0.26
    T = s["type"] != "高波活跃型"
    K = s["dist_ma60"] >= -0.08
    masks = {
        "F1_收益优先": A & G & V & Q & T,
        "F2_低回撤": A & G & V & Q & K,
        "F3_中性": A & G & V,
    }
    out = {}
    for name, m in masks.items():
        x = s[m]
        out[name] = set(zip(x["code"], x["signal_date"]))
        print(f"  {name}: {len(x):,} 个信号（{len(x)/len(s)*100:.1f}%）", flush=True)
    return out


def period_metrics(eq: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    g = eq[(pd.to_datetime(eq["date"]) >= start)
           & (pd.to_datetime(eq["date"]) <= end)].copy()
    if len(g) < 2:
        return {}
    eqv = g["equity"].to_numpy(float)
    total = eqv[-1] / eqv[0] - 1.0
    years = max((pd.to_datetime(g["date"]).iloc[-1]
                 - pd.to_datetime(g["date"]).iloc[0]).days / 365.25, 0.01)
    ann = (1.0 + total) ** (1.0 / years) - 1.0
    dd = float((eqv / np.maximum.accumulate(eqv) - 1.0).min())
    dr = np.diff(eqv) / eqv[:-1]
    sharpe = ((np.mean(dr) * 250 - 0.02) / (np.std(dr) * np.sqrt(250))
              if len(dr) and np.std(dr) > 0 else 0.0)
    return {
        "累计收益": total * 100, "年化": ann * 100, "最大回撤": dd * 100,
        "Calmar": ann / abs(dd) if dd != 0 else 0.0, "Sharpe": sharpe,
        "平均仓位": g["market_value_ratio"].mean() * 100,
        "平均持仓数": g["n_positions"].mean(),
    }


def trade_stats(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"交易": 0, "胜率": np.nan, "平均每笔": np.nan, "PF": np.nan,
                "平均持有": np.nan}
    wins = tr[tr["total_ret"] > 0]
    losses = tr[tr["total_ret"] <= 0]
    gp = wins["total_pnl"].sum()
    gl = abs(losses["total_pnl"].sum())
    return {
        "交易": len(tr), "胜率": len(wins) / len(tr) * 100,
        "平均每笔": tr["total_ret"].mean() * 100,
        "PF": gp / gl if gl > 0 else np.nan,
        "平均持有": tr["hold_days"].mean(),
    }


def run_one(name: str, strategy: str, allowed: set | None, bench: pd.DataFrame,
            dates: list[pd.Timestamp], data: dict, uni, max_pos: int,
            weight: float, trail_trigger: float = 0.10,
            trail_stop: float = 0.08) -> dict:
    flt = None
    if allowed is not None:
        def flt(code, df, idx, date, _allow=allowed):
            return (code, date) in _allow
    print(f"\n运行 {name} …", flush=True)
    r = eng.run_backtest(
        data, dates, strategy, universe=uni,
        max_positions=max_pos, flat_weight=weight,
        entry_filter=flt, trail_trigger=trail_trigger,
        trail_stop=trail_stop, label=name)
    m = r["metrics"]
    m.update(trade_stats(r["trades"]))
    m["期末权益"] = r["final_equity"]
    m["未平仓"] = r["n_open"]
    m["买入成交"] = r["stats"].get("买入成交", 0)
    m["买入涨停跳过"] = r["stats"].get("买入涨停跳过", 0)
    return r


def main() -> None:
    print("加载数据…", flush=True)
    data = eng.load_pit_data()
    eng.START, eng.END = FULL_START, FULL_END
    bench = eng.load_benchmark()
    dates = pd.to_datetime(bench["date"]).tolist()
    dates = [d for d in dates if FULL_START <= d <= FULL_END]
    snaps = eng.load_snapshots()
    uni = eng.Universe(snaps)

    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", parse_dates=["signal_date"])
    filters = build_filters(sig)

    configs = []
    for fname, allowed in [("全部", None)] + list(filters.items()):
        for ename, strategy in [("死叉", "death_cross"),
                                ("b3_10_8", "b3_prior_peak")]:
            configs.append((f"{fname}_{ename}", strategy, allowed))

    results = {}
    rows = []
    for name, strategy, allowed in configs:
        r = run_one(name, strategy, allowed, bench, dates, data, uni, 10, 0.10)
        results[name] = r
        m = r["metrics"]
        base = {"配置": name, "槽位": 10, "单槽": "10%"}
        for label, start in [("全期", FULL_START), ("2018起", SUB_START)]:
            pm = period_metrics(r["equity_curve"], start, FULL_END)
            tr = r["trades"].copy()
            if len(tr):
                tr["entry_date"] = pd.to_datetime(tr["entry_date"])
                tr = tr[tr["entry_date"] >= start]
            ts = trade_stats(tr)
            rows.append({**base, "区间": label,
                         **{k: round(v, 3) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in {**pm, **ts}.items()},
                         "期末权益": round(m["final_equity"], 0) if label == "全期" else np.nan})
        print(f"  {name}: 全期年化 {m['ann_return']}% 回撤 {m['mdd']}% "
              f"交易 {m['n_trades']} 胜率 {m['win_rate']}%", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "组合汇总_10槽.csv", index=False, encoding="utf-8-sig")

    # 保存权益与交易，便于复核
    for name, r in results.items():
        eq = r["equity_curve"].copy()
        eq.columns = ["日期", "权益", "持仓数", "持仓市值", "仓位比例"]
        eq.to_csv(OUT / f"权益_{name}.csv", index=False, encoding="utf-8-sig")
        tr = r["trades"].copy()
        if not tr.empty:
            tr = tr.rename(columns={
                "entry_id": "编号", "code": "代码", "signal_date": "信号日",
                "entry_date": "买入日", "entry_price": "买入价",
                "exit_date": "卖出日", "exit_price": "卖出价",
                "exit_price_date": "计价日", "shares": "股数",
                "total_invested": "投入资金", "net_proceeds": "净回款",
                "total_pnl": "净盈亏", "total_ret": "净收益率",
                "hold_days": "持有交易日", "reason": "离场原因",
                "peak_profit_pct": "峰值浮盈",
            })
            tr.to_csv(OUT / f"交易_{name}.csv", index=False, encoding="utf-8-sig")

    print("\n汇总：")
    cols = ["配置", "区间", "年化", "最大回撤", "Calmar", "Sharpe",
            "交易", "胜率", "平均每笔", "PF", "平均仓位"]
    print(summary[cols].round(2).to_string(index=False))
    print(f"\n完成：{OUT}")


if __name__ == "__main__":
    main()
