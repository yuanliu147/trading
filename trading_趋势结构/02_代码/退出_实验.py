"""退出与仓位实验：一次跑完 交易级 / 组合级 / 仓位敏感 / 状态标签检验。

用法：
    python3 02_代码/退出_实验.py            # 全部阶段
    python3 02_代码/退出_实验.py trades     # 只跑交易级
    python3 02_代码/退出_实验.py portfolio  # 只跑组合级
    python3 02_代码/退出_实验.py sizing     # 只跑仓位敏感
    python3 02_代码/退出_实验.py states     # 只跑状态标签检验

所有收益只落在 2008-01-01 ~ 2020-12-31；2021+ 由 assert_in_window 拦截。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from 持仓状态机 import ARMS  # noqa: E402
from 退出_数据层 import Panel  # noqa: E402
from 退出_回测引擎 import (  # noqa: E402
    BTConfig,
    build_signals,
    perf,
    run_portfolio,
    run_trades_multi,
    trade_stats,
)

OUT = os.path.join(ROOT, "03_结果")
CACHE = os.path.join(ROOT, "01_数据", "cache")
os.makedirs(OUT, exist_ok=True)

# 臂的阅读顺序（按机制family分组，不是按绩效排序）
ARM_ORDER = [
    "B0", "S_fix", "S_struct",
    "T2", "T3", "T45", "T6", "T_struct",
    "BE", "TS10", "TS20", "TS40",
    "SC", "TG2", "SE20", "MX60", "STK", "ALL",
]


def _pct(x, n=2):
    return f"{x:+.{n}f}" if isinstance(x, float) and np.isfinite(x) else str(x)


def stage_trades():
    t0 = time.time()
    p = Panel()
    for tb in ("TB2", "TB1"):
        sig = build_signals(p, tb, cache_dir=CACHE)
        cfg = BTConfig(testbed=tb, start="2008-01-01", end="2020-12-31")
        log: list = []
        res = run_trades_multi(p, sig, cfg, ARMS, state_log_arm="B0", state_log=log)
        rows = []
        for s in log:
            rows.append(dict(state=s[0], fwd5=s[1], window="dev" if s[2] else "val",
                             board="创业板" if s[3] else "主板"))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"退出_状态标签_{tb}.csv"), index=False)

        stat_rows = []
        year_rows = []
        for arm in ARM_ORDER:
            df = res[arm]
            df = df.assign(year=df.entry_date.str[:4].astype(int))
            for w in ("dev", "val"):
                sub = df[df.window == w]
                st = trade_stats(sub)
                if not st:
                    continue
                stat_rows.append(dict(testbed=tb, arm=arm, window=w, **st))
            for y, g in df.groupby("year"):
                year_rows.append(dict(testbed=tb, arm=arm, year=int(y), **trade_stats(g)))
        pd.DataFrame(stat_rows).to_csv(
            os.path.join(OUT, f"退出_臂对照_交易级_{tb}.csv"), index=False)
        pd.DataFrame(year_rows).to_csv(
            os.path.join(OUT, f"退出_分年_{tb}.csv"), index=False)
        # 退出原因分布
        rr = []
        for arm in ARM_ORDER:
            vc = res[arm].groupby(["window", "reason"]).size().reset_index(name="n")
            vc["arm"] = arm
            rr.append(vc)
        pd.concat(rr).to_csv(os.path.join(OUT, f"退出_退出原因_{tb}.csv"), index=False)
        # 逐笔明细留一份（B0 与 ALL，便于复盘）
        for arm in ("B0", "T3", "TG2", "ALL"):
            res[arm].to_csv(os.path.join(OUT, f"退出_明细_{tb}_{arm}.csv"), index=False)
        print(f"[trades] {tb} 完成，{time.time()-t0:.0f}s", flush=True)


def stage_portfolio():
    t0 = time.time()
    p = Panel()
    sig = build_signals(p, "TB1", cache_dir=CACHE)
    rows = []
    for arm in ARM_ORDER:
        for (s, e, w) in (("2008-01-01", "2015-12-31", "dev"),
                          ("2016-01-01", "2020-12-31", "val")):
            cfg = BTConfig(arm=arm, start=s, end=e, testbed="TB1")
            eq, tr, npos = run_portfolio(p, sig, cfg, ARMS[arm])
            pf = perf(eq)
            st = trade_stats(tr)
            rows.append(dict(
                arm=arm, window=w,
                cagr=pf.get("cagr"), mdd=pf.get("mdd"), calmar=pf.get("calmar"),
                sharpe=pf.get("sharpe"), vol=pf.get("vol"),
                nav_end=pf.get("nav_end"),
                n_trades=st.get("n"), win=st.get("win"), avg_R=st.get("avg_R"),
                med_R=st.get("med_R"), avg_hold=st.get("avg_hold"),
                avg_pos=float(np.mean(npos)) if len(npos) else np.nan,
            ))
            eq.to_csv(os.path.join(OUT, f"退出_净值_TB1_{arm}_{w}.csv"))
        print(f"[portfolio] {arm} 完成 {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "退出_臂对照_组合级.csv"), index=False)


def stage_sizing():
    t0 = time.time()
    p = Panel()
    sig = build_signals(p, "TB1", cache_dir=CACHE)
    rows = []
    base = dict(start="2008-01-01", end="2015-12-31", testbed="TB1")

    def add(tag, arm, cfg, window="dev", start="2008-01-01", end="2015-12-31"):
        c = BTConfig(arm=arm, start=start, end=end, testbed="TB1", **cfg)
        eq, tr, npos = run_portfolio(p, sig, c, ARMS[arm])
        pf = perf(eq)
        st = trade_stats(tr)
        rows.append(dict(study=tag, arm=arm, window=window,
                         sizing=c.sizing, risk_pct=c.risk_pct, vol_pct=c.vol_pct,
                         slots=c.slots, capital=c.capital, risk_cap=c.risk_cap,
                         cagr=pf.get("cagr"), mdd=pf.get("mdd"),
                         calmar=pf.get("calmar"), sharpe=pf.get("sharpe"),
                         n_trades=st.get("n"), avg_R=st.get("avg_R"),
                         avg_pos=float(np.mean(npos)) if len(npos) else np.nan))

    # 1) 仓位口径
    for arm in ("B0", "T3", "ALL"):
        add("仓位口径", arm, dict())
        for rp in (0.005, 0.01, 0.015, 0.02):
            add("仓位口径", arm, dict(sizing="RB", risk_pct=rp))
        for vp in (0.005, 0.01):
            add("仓位口径", arm, dict(sizing="VOL", vol_pct=vp))
        print(f"[sizing] {arm} 口径 {time.time()-t0:.0f}s", flush=True)

    # 2) 组合总风险上限（用 RB 1% 作底）
    for arm in ("T3", "ALL"):
        for cap in (None, 0.04, 0.06, 0.08):
            add("组合风险上限", arm, dict(sizing="RB", risk_pct=0.01, risk_cap=cap))
        print(f"[sizing] {arm} 风险上限 {time.time()-t0:.0f}s", flush=True)

    # 3) 验证窗口复核（只跑主结论需要的少数配置）
    for arm in ("T3", "ALL"):
        for cfgd, tag in (
            (dict(), "EQ"),
            (dict(sizing="RB", risk_pct=0.01, risk_cap=0.06), "RB1_cap6"),
        ):
            add("验证复核", arm, cfgd, window="val",
                start="2016-01-01", end="2020-12-31")

    # 4) 槽数与资金敏感性
    for arm in ("T3",):
        for slots in (5, 10, 20):
            for cap0 in (20_000, 100_000, 1_000_000):
                add("槽数与资金", arm, dict(slots=slots, capital=cap0))
        print(f"[sizing] 槽数 {time.time()-t0:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "退出_仓位敏感.csv"), index=False)


def stage_states():
    """状态标签是否真的有区分力：按状态统计未来 5 日收益（t+1 开盘 -> t+5 收盘）。"""
    rows = []
    for tb in ("TB2", "TB1"):
        for ext in (".csv", ".csv.gz"):
            f = os.path.join(OUT, f"退出_状态标签_{tb}{ext}")
            if os.path.exists(f):
                break
        else:
            continue
        d = pd.read_csv(f)
        for (w, s), g in d.groupby(["window", "state"]):
            rows.append(dict(testbed=tb, window=w, state=s, n=len(g),
                             fwd5_mean=g.fwd5.mean(), fwd5_median=g.fwd5.median(),
                             fwd5_pos=(g.fwd5 > 0).mean()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "退出_状态区分力.csv"), index=False)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["trades", "portfolio", "sizing", "states"]
    if "trades" in stages:
        stage_trades()
    if "portfolio" in stages:
        stage_portfolio()
    if "sizing" in stages:
        stage_sizing()
    if "states" in stages:
        stage_states()
    print("done")
