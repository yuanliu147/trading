#!/usr/bin/env python3
"""只买第一名 vs 买前二：同为两个半仓，同一套成交规则。

变体只改一件事：每个信号日最多买 rank<=1；仓位仍为 2×50%，
所以只买第一名时经常只占用一个仓位，剩余现金空转。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from strategy_lib import (Config, annual, build_signals, load_matrices, load_panel,
                          load_stocks, metrics, run, save_result)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_只买第一名"
RES = OUT / "结果"
PERIODS = [
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2017-2026", "2017-10-19", "2026-10-23"),
    ("2005-2026", "2005-01-01", "2026-10-23"),
]
POOLS = [
    ("A_仅当前上市", "listed", "listed", ""),
    ("C_含退市_排除风险警示", "all", "all", "is_st == 0"),
]
RANKS = [(2, "买前二"), (1, "只买第一名")]


def main():
    panel = load_panel()
    stocks = load_stocks()
    M = load_matrices()

    rows = []
    rank_rows = []
    yearly = {}
    for period, start, end in PERIODS:
        for pool, universe, gate, extra in POOLS:
            by_rank = {}
            for rank, rank_label in RANKS:
                label = f"{pool}_{rank_label}"
                cfg = Config(start=start, end=end, universe=universe, gate=gate,
                             extra_filter=extra, max_rank=rank, label=label)
                sig = build_signals(panel, stocks, cfg)
                res = run(sig, M, stocks, cfg)
                m = metrics(res.equity, res.trades)
                save_result(res, cfg, RES / period, label)
                eq = res.equity
                avg_inv = float((1 - eq.cash / eq.equity).mean())
                buys = res.orders[res.orders.side == "buy"]
                rejects = buys[buys.reason.str.startswith("reject")]
                rows.append(dict(
                    期间=period, 股票池=pool, 选股=rank_label, 仓位数=2,
                    单票仓位=cfg.position_fraction, 年化收益=m["cagr"],
                    最大回撤=m["max_dd"], 卡玛=m["calmar"], 夏普=m["sharpe"],
                    交易笔数=m["n_trades"], 胜率=m["win"], 平均单笔=m["avg"],
                    中位单笔=m["median"], 平均仓位=avg_inv,
                    有持仓天数占比=m["exposure"], 期末净值=m["final"],
                    信号日数=int(sig.date.nunique()), 信号池行数=len(sig),
                    可买信号行数=int((sig["rank"] <= rank).sum()),
                    买入笔数=int((buys.reason == "buy").sum()),
                    拒买笔数=int(len(rejects)),
                    拒买_涨停=int((rejects.reason == "reject_limit_up").sum()),
                    拒买_停牌无K线=int((rejects.reason == "reject_no_bar").sum()),
                    拒买_资金不足=int((rejects.reason == "reject_budget").sum()),
                    首日=str(m["first_date"]), 末日=str(m["last_date"]),
                ))
                print(period, pool, rank_label,
                      f"cagr={m['cagr']:.4f} dd={m['max_dd']:.4f} "
                      f"trades={m['n_trades']} inv={avg_inv:.3f}", flush=True)

                by_rank[rank_label] = res.trades
                if pool.startswith("C"):
                    yearly[(period, rank_label)] = annual(res.equity)
                    tr = res.trades.merge(
                        sig[["date", "code", "rank"]],
                        left_on=["signal_date", "code"], right_on=["date", "code"],
                        how="left",
                    )
                    tr["净盈亏"] = tr.proceeds - tr.invested
                    for rk, g in tr.groupby("rank"):
                        rank_rows.append(dict(
                            期间=period, 运行=rank_label, 排名=int(rk), 笔数=len(g),
                            平均单笔=g.ret.mean(), 中位单笔=g.ret.median(),
                            胜率=(g.ret > 0).mean(), 净盈亏合计=g["净盈亏"].sum(),
                            平均持有日=g.holding_sessions.mean(),
                            止盈笔数=int((g.exit_reason.str.startswith("take_profit")).sum()),
                        ))
            # 只买第一 vs 买前二：交易集合差异（仅 C）
            if pool.startswith("C"):
                t1 = by_rank["只买第一名"]
                t2 = by_rank["买前二"]
                s1 = set(zip(t1.signal_date, t1.code))
                s2 = set(zip(t2.signal_date, t2.code))
                for tag, only, src in [
                    ("只买第一多出的交易", s1 - s2, t1),
                    ("仅前二买入的交易", s2 - s1, t2),
                ]:
                    if not only:
                        continue
                    mask = [k in only for k in zip(src.signal_date, src.code)]
                    g = src[mask]
                    rank_rows.append(dict(
                        期间=period, 运行="对比", 排名=tag, 笔数=len(g),
                        平均单笔=g.ret.mean(), 中位单笔=g.ret.median(),
                        胜率=(g.ret > 0).mean(),
                        净盈亏合计=(g.proceeds - g.invested).sum(),
                        平均持有日=g.holding_sessions.mean(),
                        止盈笔数=int(g.exit_reason.str.startswith("take_profit").sum()),
                    ))

    OUT.mkdir(parents=True, exist_ok=True)

    # 稳健性：叠加 25% 止损（仍为 2×50% 仓位），只看 C 池
    stop_rows = []
    for period, start, end in PERIODS:
        for rank, rank_label in RANKS:
            label = f"C_止损25_{rank_label}"
            cfg = Config(start=start, end=end, universe="all", gate="all",
                         extra_filter="is_st == 0", max_rank=rank, stop_loss=0.25,
                         label=label)
            sig = build_signals(panel, stocks, cfg)
            res = run(sig, M, stocks, cfg)
            m = metrics(res.equity, res.trades)
            save_result(res, cfg, RES / period, label)
            stop_rows.append(dict(期间=period, 选股=rank_label, 止损=0.25,
                                  单票仓位=0.5, **m))
            print(period, "止损25", rank_label,
                  f"cagr={m['cagr']:.4f} dd={m['max_dd']:.4f} trades={m['n_trades']}",
                  flush=True)
    pd.DataFrame(stop_rows).to_csv(OUT / "只买第一名_止损25.csv", index=False,
                                   encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "只买第一名_对比.csv", index=False, encoding="utf-8-sig")

    rk = pd.DataFrame(rank_rows)
    rk.to_csv(OUT / "只买第一名_排名分解.csv", index=False, encoding="utf-8-sig")

    yearly_rows = [
        dict(期间=period, 选股=rank_label, 年份=int(year), 收益=float(val))
        for (period, rank_label), s in yearly.items() for year, val in s.items()
    ]
    y = pd.DataFrame(yearly_rows).sort_values(["期间", "选股", "年份"])
    y.to_csv(OUT / "只买第一名_年度收益.csv", index=False, encoding="utf-8-sig")

    (OUT / "只买第一名_说明.json").write_text(json.dumps(dict(
        periods=PERIODS,
        pools=POOLS,
        execution="信号日 T 收盘排名，T+3 开盘买入；+20% 盘中止盈；15 个交易日收盘卖出；2 个半仓",
        max_rank="每个信号日最多买 rank<=N；只买第一名时第二名不递补",
        costs="滑点 0.1%/边，佣金 0.025%（最低 5 元），过户费，卖出印花税",
    ), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(df.to_string(index=False))
    print(rk.to_string(index=False))


if __name__ == "__main__":
    main()
