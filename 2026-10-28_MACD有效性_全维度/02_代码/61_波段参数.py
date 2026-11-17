#!/usr/bin/env python3
"""61 波段参数稳健：不同 MACD/KDJ 参数下，拐点精确率与方向一致率。"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import macd_core as M
import kdj_core as K
import stock_core as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "03_结果")


def coverage(pv, n, nc, typ, win=5):
    cov = np.zeros((n, nc), dtype=bool)
    for r in pv[pv["type"] == typ].itertuples():
        cov[max(0, r.pivot_i - win):min(n, r.pivot_i + win + 1), r.ci] = True
    return cov


def main():
    t0 = time.time()
    P = S.load_base(); base = P["universe_inv"]
    pv = pd.read_parquet(os.path.join(RES, "59_波段_拐点.parquet"))
    pv = pv[pv["th"] == 0.10]
    n, nc = P["shape"]
    cov_low = coverage(pv, n, nc, -1)
    cov_high = coverage(pv, n, nc, 1)
    base_low = (cov_low & base).sum() / base.sum()
    base_high = (cov_high & base).sum() / base.sum()
    dirn = np.zeros((n, nc), dtype=np.int8)
    for ci, g in pv.sort_values(["ci", "confirm_i"]).groupby("ci"):
        for r in g.itertuples():
            dirn[r.confirm_i:, ci] = -r.type
    okdir = base & (dirn != 0)
    rows = []

    kdj_sets = [(5, 3, 3), (9, 3, 3), (14, 3, 3), (19, 3, 3), (9, 5, 3), (9, 3, 5)]
    for (nn, m1, m2) in kdj_sets:
        k, d, j, _ = K.kdj(P, nn, m1, m2)
        gold, death = K.kdj_crosses(k, d, P)
        bull = (k > d) & P["valid"] & (P["bars"] >= 120)
        lowg = gold & (k < 30)
        highd = death & (k > 70)
        for name, s in [("KDJ金叉", gold), ("KDJ低位金叉", lowg), ("KDJ高位死叉", highd)]:
            ss = s & base
            if ss.sum() == 0:
                continue
            rows.append(dict(类别="KDJ", 参数=f"{nn},{m1},{m2}", 信号=name,
                             信号数=int(ss.sum()), 每日占比=float(ss.sum() / base.sum()),
                             落低点5=float((ss & cov_low).sum() / ss.sum()),
                             低点基准=float(base_low),
                             低点提升=float((ss & cov_low).sum() / ss.sum() / max(base_low, 1e-9)),
                             落高点5=float((ss & cov_high).sum() / ss.sum()),
                             高点基准=float(base_high),
                             高点提升=float((ss & cov_high).sum() / ss.sum() / max(base_high, 1e-9)),
                             方向一致=float(np.mean(((bull & (dirn > 0)) | (~bull & (dirn < 0)))[okdir]))))
        print("kdj", nn, m1, m2, "%.1fs" % (time.time() - t0))

    macd_sets = [(8, 17, 9), (12, 26, 9), (16, 32, 9), (20, 40, 9)]
    for (f, s, sig) in macd_sets:
        dif, dea, hist = M.macd(P, f, s, sig)
        gold, death = M.crosses(dif, dea, P)
        bull = (dif > dea) & P["valid"] & (P["bars"] >= 120)
        for name, ss0 in [("MACD金叉", gold), ("MACD死叉", death)]:
            ss = ss0 & base
            if ss.sum() == 0:
                continue
            rows.append(dict(类别="MACD", 参数=f"{f},{s},{sig}", 信号=name,
                             信号数=int(ss.sum()), 每日占比=float(ss.sum() / base.sum()),
                             落低点5=float((ss & cov_low).sum() / ss.sum()),
                             低点基准=float(base_low),
                             低点提升=float((ss & cov_low).sum() / ss.sum() / max(base_low, 1e-9)),
                             落高点5=float((ss & cov_high).sum() / ss.sum()),
                             高点基准=float(base_high),
                             高点提升=float((ss & cov_high).sum() / ss.sum() / max(base_high, 1e-9)),
                             方向一致=float(np.mean(((bull & (dirn > 0)) | (~bull & (dirn < 0)))[okdir]))))
        print("macd", f, s, sig, "%.1fs" % (time.time() - t0))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "61_波段_参数.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
