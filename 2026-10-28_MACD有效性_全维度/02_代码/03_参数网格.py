#!/usr/bin/env python3
"""03 参数网格：fast / slow / signal 全组合的事件研究

对每个组合：
- 2010-2026 全样本金叉
- 信号加权超额（5/10/20/60 日）+ 按信号日聚类的 t
- 日度加权超额
- 中位数、胜率
- 2010-2017 / 2018-2026 两段信号加权 ex20 与聚类 t

输出：03_结果/参数网格.csv
"""
import os
import sys
import gc
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")

FASTS = [5, 6, 8, 10, 12, 15, 20, 25]
SLOWS = [15, 20, 26, 30, 35, 40, 50, 60]
SIGS = [1, 3, 5, 9, 12, 15, 20]


def cluster_stats(e, dates):
    d = pd.DataFrame({"e": e, "d": dates}).dropna()
    if len(d) < 100:
        return np.nan, np.nan, len(d), 0
    g = d.groupby("d").e
    N = len(d)
    mean = d.e.mean()
    sums = g.sum().to_numpy()
    se = np.sqrt(np.sum(sums ** 2)) / N
    return mean, (mean / se if se > 0 else np.nan), N, len(sums)


def main():
    t0 = time.time()
    P = M.load_panel()
    fwd, ok, mkt, loo = M.forward_returns(P)
    dates = P["dates"]
    # 预取超额矩阵
    ex = {h: (fwd[h] - loo[h]).astype(np.float32) for h in M.HORIZONS}
    valid_ret = {h: np.isfinite(ex[h]) & ok[h] for h in M.HORIZONS}
    # 20 日横截面分位（主指标）
    f20 = fwd[20]; v20 = ok[20] & np.isfinite(f20)
    pct20 = np.full(P["shape"], np.nan, dtype=np.float32)
    for t in range(P["shape"][0]):
        idx = np.where(v20[t])[0]
        if len(idx) < 50:
            continue
        x = f20[t, idx]
        order = np.argsort(np.argsort(x))
        pct20[t, idx] = (order + 0.5) / len(idx)
    del fwd, mkt, loo
    gc.collect()

    rows = []
    combos = [(f, s, g) for f in FASTS for s in SLOWS for g in SIGS if f < s]
    print("combos", len(combos))
    for k, (f, s, g) in enumerate(combos):
        dif, dea, hist = M.macd(P, f, s, g)
        gold, death = M.crosses(dif, dea, P, min_bars=120)
        ii, jj = np.where(gold)
        keep = dates[ii] >= pd.Timestamp("2010-01-01")
        ii, jj = ii[keep], jj[keep]
        dd = dates[ii]
        rec = dict(fast=f, slow=s, signal=g, n=len(ii))
        for h in M.HORIZONS:
            v = valid_ret[h][ii, jj]
            mean, t, n, nd = cluster_stats(ex[h][ii, jj][v], dd[v])
            rec[f"ex{h}"] = mean
            rec[f"t{h}"] = t
            rec[f"n{h}"] = n
            if h == 20:
                vals = ex[h][ii, jj][v]
                rec["median20"] = np.nanmedian(vals) if len(vals) else np.nan
                rec["hit20"] = np.nanmean(vals > 0) if len(vals) else np.nan
                dm = pd.DataFrame({"e": vals, "d": dd[v]}).groupby("d").e.mean()
                rec["mean_daily20"] = dm.mean() if len(dm) else np.nan
                rec["t_daily20"] = (dm.mean() / (dm.std() / np.sqrt(len(dm)))
                                    if len(dm) > 10 else np.nan)
                pp = pct20[ii, jj][v]
                rec["pct20"] = np.nanmean(pp) if len(pp) else np.nan
                rec["share50"] = np.nanmean(pp > 0.5) if len(pp) else np.nan
        # 两段
        for lab, lo, hi in [("early", "2010-01-01", "2018-01-01"),
                            ("late", "2018-01-01", "2026-12-31")]:
            sel = (dd >= pd.Timestamp(lo)) & (dd < pd.Timestamp(hi))
            v = valid_ret[20][ii[sel], jj[sel]]
            mean, t, n, nd = cluster_stats(ex[20][ii[sel], jj[sel]][v], dd[sel][v])
            rec[f"ex20_{lab}"] = mean
            rec[f"t20_{lab}"] = t
            rec[f"n_{lab}"] = n
        rows.append(rec)
        if (k + 1) % 40 == 0:
            print(k + 1, "combos", round(time.time() - t0, 1), "s", flush=True)
        del dif, dea, hist, gold, death, ii, jj
        gc.collect()

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "参数网格.csv"), index=False, encoding="utf-8-sig")
    print(out.shape, "done", round(time.time() - t0, 1))
    print(out.sort_values("ex20", ascending=False).head(20).to_string(index=False))
    print("\n12/26/9:")
    print(out[(out.fast == 12) & (out.slow == 26) & (out.signal == 9)].to_string(index=False))


if __name__ == "__main__":
    main()
