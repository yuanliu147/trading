#!/usr/bin/env python3
"""65 零轴回踩+柱体放大：在已确认上行波段里，是“继续”还是“尾声”？"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import stock_core as S
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.5)


def main():
    t0 = time.time()
    P = S.load_base(); base = P["universe_inv"]
    mf = F.build_macd_features(P)
    dif, dea, hist = mf["dif"], mf["dea"], mf["hist"]
    c = P["close_ff"]
    h = (hist / c * 100).astype(np.float32)
    hp = M.shift(h, 1)
    peak20 = M.rolling_max(h, 20, minp=10)
    trough5 = M.rolling_min(h, 5, minp=3)
    red = h > 0; expand = red & (h > hp)
    zone = (dif > 0) & (dea > 0)
    mask = base & P["valid"] & (P["bars"] >= 120)
    pv = pd.read_parquet(os.path.join(RES, "59_波段_拐点.parquet"))
    pv = pv[(pv.th == 0.10) & (pv.type == -1) & (pv.next_type == 1) & (pv.next_pivot >= 0)]
    n, nc = P["shape"]
    up = np.zeros((n, nc), dtype=bool)
    rem = np.full((n, nc), np.nan, dtype=np.float32)
    dth = np.full((n, nc), np.nan, dtype=np.float32)
    for r in pv.itertuples():
        a, b = int(r.confirm_i), int(r.next_pivot)
        if b <= a:
            continue
        up[a:b, r.ci] = True
        rem[a:b, r.ci] = r.next_price / c[a:b, r.ci] - 1.0
        dth[a:b, r.ci] = b - np.arange(a, b)
    print("upswing cells", int(up.sum()), "%.1fs" % (time.time() - t0))

    rows = []
    for ratio in RATIOS:
        if ratio == 0.0:
            pull = trough5 <= 0; name = "回踩破0再翻红"
        else:
            pull = (trough5 > 0) & (trough5 <= ratio * peak20)
            name = f"红柱回踩≤{ratio:.0%}峰值"
        pat = mask & pull & expand
        for lab, p in [("全部", pat), ("零轴上", pat & zone)]:
            pp = p & up
            bb = up & ~p
            rr = rem[pp]; rb = rem[bb]
            dd = np.nanmean(np.where(pp, rem, np.nan), axis=1) - np.nanmean(np.where(bb, rem, np.nan), axis=1)
            dd = dd[np.isfinite(dd)]
            rows.append(dict(
                形态=name, 分组=lab,
                形态天数=int(pp.sum()), 基准天数=int(bb.sum()),
                形态中位剩余=float(np.nanmedian(rr)) if len(rr) else np.nan,
                基准中位剩余=float(np.nanmedian(rb)) if len(rb) else np.nan,
                形态均值剩余=float(np.nanmean(rr)) if len(rr) else np.nan,
                基准均值剩余=float(np.nanmean(rb)) if len(rb) else np.nan,
                形态剩余大于5=float(np.nanmean(rr > 0.05)) if len(rr) else np.nan,
                基准剩余大于5=float(np.nanmean(rb > 0.05)) if len(rb) else np.nan,
                形态中位天数=float(np.nanmedian(dth[pp])) if len(rr) else np.nan,
                基准中位天数=float(np.nanmedian(dth[bb])) if len(rb) else np.nan,
                日度差=float(np.nanmean(dd)) if len(dd) else np.nan,
                日度差t=float(nw_t(dd, 20)) if len(dd) else np.nan,
            ))
        print(name, "%.1fs" % (time.time() - t0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "65_零轴回踩_波段.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
