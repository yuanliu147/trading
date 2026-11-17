#!/usr/bin/env python3
"""37 配对对照：低 60 日 / 低波子集的收益，是 MACD 的还是特征本身？

关键对照（同一批 20 日非重叠 cohort）：
- 全市场 vs 全市场金叉
- 全市场金叉 vs 金叉内低 60 日
- 全市场低 60 日 vs 低 60 日 + 金叉
- 全市场低 60 日 vs 低 60 日随机抽同样只数（控制分散度）
- 低波同理
并做 20 个 cohort 相位（每 20 日取一个起点）的稳健性。
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import macd_core as M
import stock_core as S

RES = S.RES
H = 20
FEE = 0.0015
RNG = np.random.default_rng(20261029)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def perf(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 8:
        return dict(期数=len(r), 年化=np.nan, 回撤=np.nan, 夏普=np.nan, 胜率=np.nan, 平均每期=np.nan)
    eq = np.cumprod(1 + r)
    return dict(期数=len(r), 年化=eq[-1] ** (252 / H / len(r)) - 1,
                回撤=(eq / np.maximum.accumulate(eq) - 1).min(),
                夏普=r.mean() / r.std() * np.sqrt(252 / H) if r.std() > 0 else np.nan,
                胜率=(r > 0).mean(), 平均每期=r.mean())


def rank_pct(x, base):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for t in range(x.shape[0]):
        idx = np.where(base[t] & np.isfinite(x[t]))[0]
        if len(idx) < 30:
            continue
        r = rankdata(x[t, idx], method="average")
        out[t, idx] = (r - 0.5) / len(idx)
    return out


def main():
    t0 = time.time()
    log("load panel")
    P = S.load_base()
    dates = P["dates"]
    n = len(dates)
    log("features")
    fwd_d, ok_d, _, _ = M.forward_returns(P, (H,))
    fwd, ok = fwd_d[H], ok_d[H]
    dif, dea, hist = M.macd(P)
    gold, _ = M.crosses(dif, dea, P)
    above = dif > dea
    c = P["close_ff"]
    ret60 = (c / M.shift(c, 60) - 1).astype(np.float32)
    std20 = S.roll(P["r_cc"], 20, "std") * np.sqrt(252)
    turn20 = S.roll(P["turn"], 20, "mean")
    mcap = P["amount"].astype(np.float64) / np.maximum(P["turn"].astype(np.float64) / 100.0, 1e-9)
    mcap[~np.isfinite(mcap) | (mcap <= 0)] = np.nan
    size = S.roll(np.log(mcap), 20, "median")
    base = P["valid"] & (P["bars"] >= 120) & ~P["is_st"].astype(bool) & ok & np.isfinite(fwd)
    log(f"base stock-days {base.sum()/1e6:.1f}M")

    log("rank matrices")
    r_ret_all = rank_pct(ret60, base)
    r_std_all = rank_pct(std20, base)
    r_turn_all = rank_pct(turn20, base)
    r_size_all = rank_pct(size, base)
    r_ret_gold = rank_pct(ret60, base & gold)
    r_std_gold = rank_pct(std20, base & gold)
    log("ranks done")

    def series(selector, minn=10):
        rows = []
        for t in range(n - H):
            if not base[t].any():
                continue
            sel = selector(t)
            if sel.sum() < minn:
                continue
            rows.append((t, float(np.mean(fwd[t, sel]) - FEE), int(sel.sum())))
        return pd.DataFrame(rows, columns=["t", "r", "n"])

    def low60_all(t):
        return base[t] & (r_ret_all[t] <= 0.2)

    def low60_gold(t):
        return base[t] & gold[t] & (r_ret_gold[t] <= 0.2)

    def low60_all_gold(t):
        return base[t] & gold[t] & (r_ret_all[t] <= 0.2)

    def lowvol_all(t):
        return base[t] & (r_std_all[t] <= 0.2)

    def lowvol_gold(t):
        return base[t] & gold[t] & (r_std_gold[t] <= 0.2)

    def lowvol_all_gold(t):
        return base[t] & gold[t] & (r_std_all[t] <= 0.2)

    def random_subset(t, base_sel, k, reps=20):
        idx = np.where(base_sel)[0]
        if len(idx) == 0 or k <= 0:
            return np.nan
        vals = []
        for _ in range(reps):
            pick = RNG.choice(idx, min(k, len(idx)), replace=False)
            vals.append(np.mean(fwd[t, pick]) - FEE)
        return float(np.mean(vals))

    log("build series (offset 0 and phase robustness)")
    start = int(np.searchsorted(dates.values, np.datetime64("2010-01-01")))
    offsets = list(range(20))
    phase_rows = []
    main_series = {}
    for off in offsets:
        tlist = range(start + off, n - H, 20)
        def run(selector, minn=10, use_tlist=tlist):
            vals = []
            for t in use_tlist:
                sel = selector(t)
                if sel.sum() >= minn:
                    vals.append((t, float(np.mean(fwd[t, sel]) - FEE), int(sel.sum())))
            return pd.DataFrame(vals, columns=["t", "r", "n"])

        alls = run(lambda t: base[t], 30)
        golds = run(lambda t: base[t] & gold[t], 10)
        l_all = run(low60_all, 10)
        l_gold = run(low60_gold, 5)
        l_ag = run(low60_all_gold, 5)
        v_all = run(lowvol_all, 10)
        v_gold = run(lowvol_gold, 5)
        v_ag = run(lowvol_all_gold, 5)
        if off == 0:
            main_series = dict(alls=alls, golds=golds, l_all=l_all, l_gold=l_gold,
                               l_ag=l_ag, v_all=v_all, v_gold=v_gold, v_ag=v_ag)
        for name, s in [("全A", alls), ("全A金叉", golds), ("低60全A", l_all),
                        ("低60金叉", l_gold), ("低60全A+金叉", l_ag),
                        ("低波全A", v_all), ("低波金叉", v_gold), ("低波全A+金叉", v_ag)]:
            if len(s) >= 8:
                p = perf(s.r)
                phase_rows.append(dict(相位=off, 对象=name, **p, 平均持仓=s.n.mean()))
        if off % 5 == 0:
            log(f"  offset {off} done")

    phase = pd.DataFrame(phase_rows)
    phase.to_csv(os.path.join(RES, "配对_相位.csv"), index=False, encoding="utf-8-sig")
    log("phase done")

    # 主表：同一日期配对
    def pair(a, b, name, seg_lo, seg_hi):
        m = a.merge(b, on="t", suffixes=("_a", "_b"))
        m = m[(m.t >= np.searchsorted(dates.values, np.datetime64(seg_lo))) &
              (m.t <= np.searchsorted(dates.values, np.datetime64(seg_hi)))]
        if len(m) < 8:
            return None
        pa, pb = perf(m.r_a), perf(m.r_b)
        d = m.r_a - m.r_b
        return dict(对照=name, 区间=f"{seg_lo[:4]}-{seg_hi[:4]}", 期数=len(m),
                    年化A=pa["年化"], 年化B=pb["年化"], 夏普A=pa["夏普"], 夏普B=pb["夏普"],
                    回撤A=pa["回撤"], 回撤B=pb["回撤"],
                    每期差=d.mean(), 差t=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan,
                    平均持仓A=m.n_a.mean(), 平均持仓B=m.n_b.mean())

    rows = []
    for lo, hi in [("2010-01-01", "2026-12-31"), ("2010-01-01", "2017-12-31"),
                   ("2018-01-01", "2026-12-31")]:
        rows.append(pair(main_series["golds"], main_series["alls"], "金叉 vs 全A", lo, hi))
        rows.append(pair(main_series["l_gold"], main_series["golds"], "金叉内低60 vs 金叉", lo, hi))
        rows.append(pair(main_series["l_ag"], main_series["l_all"], "低60+金叉 vs 低60", lo, hi))
        rows.append(pair(main_series["v_gold"], main_series["golds"], "金叉内低波 vs 金叉", lo, hi))
        rows.append(pair(main_series["v_ag"], main_series["v_all"], "低波+金叉 vs 低波", lo, hi))
    out = pd.DataFrame([r for r in rows if r])
    out.to_csv(os.path.join(RES, "配对_主表.csv"), index=False, encoding="utf-8-sig")
    log("main table done")

    # 随机对照：低60全A + 随机抽同样只数
    log("random control")
    rand_rows = []
    for lo, hi in [("2010-01-01", "2026-12-31"), ("2018-01-01", "2026-12-31")]:
        vals = []
        for t in range(start, n - H):
            if dates[t] < pd.Timestamp(lo) or dates[t] > pd.Timestamp(hi):
                continue
            bg = low60_gold(t)
            ba = low60_all(t)
            if bg.sum() < 5 or ba.sum() < 50:
                continue
            vals.append(random_subset(t, ba, int(bg.sum()), reps=20))
        vals = np.array([v for v in vals if np.isfinite(v)])
        rand_rows.append(dict(区间=lo[:4] + "-" + hi[:4], 期数=len(vals), 随机低60年化=perf(vals)["年化"],
                              随机低60均值=vals.mean()))
    pd.DataFrame(rand_rows).to_csv(os.path.join(RES, "配对_随机.csv"), index=False, encoding="utf-8-sig")

    print(out.round(4).to_string(index=False))
    log(f"done {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
