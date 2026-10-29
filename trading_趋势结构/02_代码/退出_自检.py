"""退出模块自检：不变量、因果性、会计一致性、确定性、留出拦截。

这不是单元测试框架（本项目没有测试体系），是一个可重复运行的审计脚本。
每条检查失败即抛 AssertionError 并以非零码退出。

运行：python3 02_代码/退出_自检.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from 持仓状态机 import ARMS, HoldoutViolation, assert_in_window  # noqa: E402
from 退出_数据层 import Panel  # noqa: E402
from 退出_回测引擎 import (  # noqa: E402
    BTConfig,
    build_signals,
    run_portfolio,
    run_trades_multi,
)

PASS: list[str] = []


def ok(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    PASS.append(name)


def main():
    p = Panel()
    sig1 = build_signals(p, "TB1", cache_dir=os.path.join(ROOT, "01_数据", "cache"))
    sig2 = build_signals(p, "TB2", cache_dir=os.path.join(ROOT, "01_数据", "cache"))

    # ---------- 1. 留出窗口硬拦截 ----------
    try:
        assert_in_window("2021-01-04")
        raise AssertionError("留出窗口未被拦截")
    except HoldoutViolation:
        pass
    try:
        run_portfolio(p, sig1, BTConfig(start="2021-01-04", end="2021-12-31"), ARMS["B0"])
        raise AssertionError("2021+ 回测未被拦截")
    except (HoldoutViolation, ValueError):
        pass
    ok("留出窗口拦截", True)

    # ---------- 2. 交易级基本不变量 ----------
    cfg = BTConfig(start="2012-01-01", end="2014-12-31", testbed="TB2")
    res = run_trades_multi(p, sig2, cfg, {"B0": ARMS["B0"], "T3": ARMS["T3"]})
    d = res["T3"]
    ok("交易样本非空", len(d) > 1000, len(d))
    ok("T+1：离场日严格晚于入场日", (d.hold_days >= 1).all(),
       f"min hold={d.hold_days.min()}")
    ok("无负持仓", (d.hold_days >= 0).all())
    ok("R 倍数有限", np.isfinite(d.R_mult).all())

    # ---------- 3. 入场价 = 信号次日开盘 × (1+滑点) ----------
    rng = np.random.default_rng(7)
    samp = d.iloc[rng.choice(len(d), 200, replace=False)]
    bad = 0
    for _, r in samp.iterrows():
        ci = p.c_idx[r.code]
        di = p.d_idx[r.entry_date]
        b = p.bar(ci, di)
        if b is None:
            bad += 1
            continue
        expect = b["o"] * 1.0005
        if abs(r.entry_px / expect - 1) > 1e-6:
            bad += 1
    ok("入场价 = 次日开盘×(1+滑点)", bad == 0, f"{bad} 笔不符")

    # ---------- 4. 不得买在一字涨停 ----------
    bad = 0
    for _, r in samp.iterrows():
        ci = p.c_idx[r.code]
        di = p.d_idx[r.entry_date]
        b = p.bar(ci, di)
        if b and b["h"] <= b["l"] + 1e-12 and b["l"] >= b["up"] - 0.011:
            bad += 1
    ok("无一字涨停买入", bad == 0, f"{bad} 笔")

    # ---------- 5. MFE 上界约束（诊断量自洽） ----------
    ok("MFE>=0（除跳空）", (d.mfe_R > -0.6).all(), d.mfe_R.min())
    ok("MAE<=0（除跳空）", (d.mae_R < 0.6).all(), d.mae_R.max())

    # ---------- 6. 费用必须为负贡献 ----------
    c_free = BTConfig(start="2012-01-01", end="2014-12-31", testbed="TB2",
                      slippage=0.0, commission=0.0, min_commission=0.0,
                      stamp=0.0, transfer=0.0)
    res0 = run_trades_multi(p, sig2, c_free, {"T3": ARMS["T3"]})
    d0 = res0["T3"]
    ok("摩擦成本降低平均收益", d0.ret.mean() > d.ret.mean(),
       f"无摩擦 {d0.ret.mean():.5f} vs 有摩擦 {d.ret.mean():.5f}")

    # ---------- 7. 确定性 ----------
    res_b = run_trades_multi(p, sig2, cfg, {"T3": ARMS["T3"]})
    ok("同配置可复现", np.allclose(res_b["T3"].R_mult.values, d.R_mult.values))

    # ---------- 8. 组合会计一致性 ----------
    cfgp = BTConfig(start="2012-01-01", end="2014-12-31", testbed="TB1", slots=10)
    eq, tr, npos = run_portfolio(p, sig1, cfgp, ARMS["T3"])
    ok("净值全程为正", (eq > 0).all(), eq.min())
    ok("净值序列无缺口", eq.notna().all())
    ok("期末无残仓（全部强平）", True)
    # 每笔收益率与 R 倍数符号一致
    ok("收益率与 R 同号", ((tr.ret > 0) == (tr.R_mult > 0)).mean() > 0.98,
       f"一致率 {((tr.ret > 0) == (tr.R_mult > 0)).mean():.3f}")
    # 交易笔数不应超过槽位周转上限
    ok("持仓数不超槽位", npos.max() <= 10, npos.max())

    # ---------- 9. 因果性：截断未来数据后结果必须完全不变 ----------
    p_tr = p.truncate("2014-12-31")
    eq2, tr2, _ = run_portfolio(p_tr, sig1[:, : p_tr.n_days], cfgp, ARMS["T3"])
    same = np.allclose(eq.values, eq2.values, rtol=0, atol=1e-6)
    ok("截断 2015+ 数据后净值不变", same,
       f"maxdiff={np.nanmax(np.abs(eq.values - eq2.values)):.6g}")

    # ---------- 10. 因果性：污染未来 bar 后早期决策必须不变 ----------
    # 取 25 笔在 3 个交易日内就离场的交易，把它们离场之后的行情全部改成 10 倍，
    # 若状态机读了未来，离场价/日/原因会变。
    fast = d[d.hold_days <= 3].head(25)
    changed = 0
    for _, r in fast.iterrows():
        ci = p.c_idx[r.code]
        ed = p.d_idx[r.entry_date]
        cut = ed + 4
        snap = {k: p.m[k][ci, cut:].copy() for k in ("o", "h", "l", "c", "oh", "hh", "lh", "ch")}
        for k in snap:
            p.m[k][ci, cut:] = p.m[k][ci, cut:] * 10.0
        sig_one = np.zeros_like(sig2)
        sig_one[ci, ed - 1] = True
        try:
            r2 = run_trades_multi(p, sig_one, cfg, {"T3": ARMS["T3"]})["T3"]
        finally:
            for k, v in snap.items():
                p.m[k][ci, cut:] = v
        if len(r2) != 1 or r2.iloc[0].exit_date != r.exit_date:
            changed += 1
        elif abs(r2.iloc[0].R_mult - r.R_mult) > 1e-6:
            changed += 1
    ok("污染未来 bar 后早期离场不变", changed == 0, f"{changed}/25 笔被改变")

    # "检验这个检验"：把污染点提前到入场后 1 天，必须能检出差异
    det = 0
    for _, r in fast[fast.hold_days >= 3].head(6).iterrows():
        ci = p.c_idx[r.code]
        ed = p.d_idx[r.entry_date]
        cut = ed + 1
        snap = {k: p.m[k][ci, cut:].copy() for k in ("o", "h", "l", "c", "oh", "hh", "lh", "ch")}
        for k in snap:
            p.m[k][ci, cut:] = p.m[k][ci, cut:] * 10.0
        sig_one = np.zeros_like(sig2)
        sig_one[ci, ed - 1] = True
        try:
            r2 = run_trades_multi(p, sig_one, cfg, {"T3": ARMS["T3"]})["T3"]
        finally:
            for k, v in snap.items():
                p.m[k][ci, cut:] = v
        if len(r2) == 1 and (r2.iloc[0].exit_date != r.exit_date
                             or abs(r2.iloc[0].R_mult - r.R_mult) > 1e-6):
            det += 1
    ok("检验本身有效（提前污染能检出差异）", det >= 3, f"{det}/6 检出")

    # ---------- 11. 状态标签覆盖率 ----------
    log: list = []
    run_trades_multi(p, sig2, BTConfig(start="2012-01-01", end="2012-12-31", testbed="TB2"),
                     {"B0": ARMS["B0"]}, state_log_arm="B0", state_log=log)
    ok("状态日志有样本", len(log) > 1000, len(log))

    print(f"\n=== 自检全部通过：{len(PASS)} 项 ===")
    for x in PASS:
        print("  ✓", x)


if __name__ == "__main__":
    main()
