#!/usr/bin/env python3
"""36 事件组合：金叉后持有 20 日的非重叠组合，按股票子集比较

每个组合起点（每 20 个交易日）买入当日所有符合条件的金叉股，T+1 开盘等权，
持有 20 日，扣 0.15% 往返。输出绝对年化与相对全 A 的超额年化。
"""
import os

import numpy as np
import pandas as pd

RES = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度/03_结果"
H = 20
FEE = 0.0015


def perf(r):
    r = np.asarray(r, dtype=float)
    eq = np.cumprod(1 + r)
    n = len(r)
    return dict(期数=n, 年化=eq[-1] ** (252 / H / n) - 1 if n else np.nan,
                回撤=(eq / np.maximum.accumulate(eq) - 1).min() if n else np.nan,
                夏普=r.mean() / r.std() * np.sqrt(252 / H) if r.std() > 0 else np.nan,
                胜率=(r > 0).mean(), 平均每期=r.mean())


def main():
    ev = pd.read_parquet(os.path.join(RES, "个股_事件_增强.parquet"))
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[(ev["date"] >= pd.Timestamp("2010-01-01")) & ev["fwd20"].notna()].copy()
    ev["is_st"] = ev["is_st"].fillna(0).astype(bool)

    # 当日横截面分位
    for f in ["规模_log", "换手20", "特质波动250", "60日涨幅", "波动20", "成交额_log"]:
        ev["r_" + f] = ev.groupby("date")[f].rank(pct=True, method="average")

    z = np.load(os.path.join(RES, "个股_日频.npz"), allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    start = int(np.searchsorted(dates.values, np.datetime64("2010-01-01")))
    tgrid = set(dates.values[start::H])

    base = (~ev["is_st"]) & (ev["bars"] >= 120)
    conds = {
        "全A": base,
        "ST": ev["is_st"] & (ev["bars"] >= 120),
        "干净子集": base & (ev["r_规模_log"] > 0.5) & (ev["r_换手20"] < 0.5) &
                    (ev["r_特质波动250"] < 0.5) & (ev["r_60日涨幅"] < 0.5),
        "干净严格": base & (ev["r_规模_log"] > 0.6) & (ev["r_换手20"] < 0.4) &
                    (ev["r_特质波动250"] < 0.4) & (ev["r_60日涨幅"] < 0.4),
        "低换手Q1": base & (ev["r_换手20"] < 0.2),
        "低特质波Q1": base & (ev["r_特质波动250"] < 0.2),
        "低60日Q1": base & (ev["r_60日涨幅"] < 0.2),
        "大市值Q5": base & (ev["r_规模_log"] > 0.8),
        "低波Q1": base & (ev["r_波动20"] < 0.2),
        "低换手+低特质波": base & (ev["r_换手20"] < 0.2) & (ev["r_特质波动250"] < 0.2),
        "低换手+低特质波+低60日": base & (ev["r_换手20"] < 0.2) &
                                  (ev["r_特质波动250"] < 0.2) & (ev["r_60日涨幅"] < 0.2),
        "大市值+低换手+低特质波": base & (ev["r_规模_log"] > 0.8) &
                                  (ev["r_换手20"] < 0.2) & (ev["r_特质波动250"] < 0.2),
        "低换手+低60日": base & (ev["r_换手20"] < 0.2) & (ev["r_60日涨幅"] < 0.2),
    }

    rows = []
    for name, cond in conds.items():
        sub = ev[cond & ev["date"].isin(tgrid)].copy()
        if sub.empty:
            continue
        g = sub.groupby("date")
        rec = pd.DataFrame({
            "abs": g["fwd20"].mean() - FEE,
            "ex": g["ex20"].mean(),
            "n": g["fwd20"].size(),
        })
        rec = rec[rec["n"] >= 30]
        if len(rec) < 10:
            continue
        for seg, lo, hi in [("2010-2026", "2010-01-01", "2026-12-31"),
                            ("2010-2017", "2010-01-01", "2017-12-31"),
                            ("2018-2026", "2018-01-01", "2026-12-31")]:
            d = rec[(rec.index >= pd.Timestamp(lo)) & (rec.index <= pd.Timestamp(hi))]
            if len(d) < 8:
                continue
            a = perf(d["abs"])
            e = perf(d["ex"])
            rows.append(dict(子集=name, 区间=seg, **{f"绝对_{k}": v for k, v in a.items()},
                             **{f"超额_{k}": v for k, v in e.items()},
                             平均持仓=d["n"].mean()))
        print(name, len(rec), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "事件组合_子集.csv"), index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
