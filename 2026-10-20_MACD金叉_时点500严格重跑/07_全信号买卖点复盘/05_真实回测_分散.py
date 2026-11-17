#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分散版真实回测：用更多槽位、每槽 1 万元左右，逼近“每个信号都买 1 万”。

10 槽 ×10% 是集中组合；本脚本测试 20×5%、50×2%、100×1%。
资金仍为 100 万，单槽资金分别约 5 万、2 万、1 万。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "01_数据"
OUT = HERE / "03_真实回测"

spec = importlib.util.spec_from_file_location("bt10", HERE / "04_真实回测.py")
bt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bt
spec.loader.exec_module(bt)


def main() -> None:
    print("加载数据…", flush=True)
    data = bt.eng.load_pit_data()
    bt.eng.START, bt.eng.END = bt.FULL_START, bt.FULL_END
    bench = bt.eng.load_benchmark()
    dates = pd.to_datetime(bench["date"]).tolist()
    dates = [d for d in dates if bt.FULL_START <= d <= bt.FULL_END]
    snaps = bt.eng.load_snapshots()
    uni = bt.eng.Universe(snaps)
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv",
                      parse_dates=["signal_date"])
    filters = bt.build_filters(sig)

    configs = [
        (20, 0.05, "全部_死叉", "death_cross", None),
        (20, 0.05, "F2_低回撤_死叉", "death_cross", filters["F2_低回撤"]),
        (50, 0.02, "全部_死叉", "death_cross", None),
        (50, 0.02, "F2_低回撤_死叉", "death_cross", filters["F2_低回撤"]),
        (50, 0.02, "F2_低回撤_b3", "b3_prior_peak", filters["F2_低回撤"]),
        (100, 0.01, "全部_死叉", "death_cross", None),
        (100, 0.01, "F2_低回撤_死叉", "death_cross", filters["F2_低回撤"]),
        (100, 0.01, "F2_低回撤_b3", "b3_prior_peak", filters["F2_低回撤"]),
        (100, 0.01, "F1_收益优先_b3", "b3_prior_peak", filters["F1_收益优先"]),
    ]

    rows = []
    for slots, w, name, strategy, allowed in configs:
        full_name = f"{slots}槽_{name}"
        r = bt.run_one(full_name, strategy, allowed, bench, dates, data, uni,
                       slots, w)
        for label, start in [("全期", bt.FULL_START), ("2018起", bt.SUB_START)]:
            pm = bt.period_metrics(r["equity_curve"], start, bt.FULL_END)
            tr = r["trades"].copy()
            if len(tr):
                tr["entry_date"] = pd.to_datetime(tr["entry_date"])
                tr = tr[tr["entry_date"] >= start]
            ts = bt.trade_stats(tr)
            rows.append({"配置": full_name, "槽位": slots,
                         "单槽": f"{w*100:.0f}%", "区间": label,
                         **{k: round(v, 3) if isinstance(v, (int, float)) else v
                            for k, v in {**pm, **ts}.items()}})
        m = r["metrics"]
        print(f"  {full_name}: 年化 {m['ann_return']}% 回撤 {m['mdd']}% "
              f"交易 {m['n_trades']}", flush=True)

        eq = r["equity_curve"].copy()
        eq.columns = ["日期", "权益", "持仓数", "持仓市值", "仓位比例"]
        eq.to_csv(OUT / f"权益_{full_name}.csv", index=False, encoding="utf-8-sig")
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
            tr.to_csv(OUT / f"交易_{full_name}.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "组合汇总_分散槽位.csv", index=False, encoding="utf-8-sig")
    print("\n汇总：")
    cols = ["配置", "区间", "年化", "最大回撤", "Calmar", "Sharpe",
            "交易", "胜率", "平均每笔", "PF", "平均仓位"]
    print(summary[cols].round(2).to_string(index=False))
    print(f"\n完成：{OUT}")


if __name__ == "__main__":
    main()
