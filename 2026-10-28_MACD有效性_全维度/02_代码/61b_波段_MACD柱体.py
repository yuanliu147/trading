#!/usr/bin/env python3
"""61b MACD 柱体特征在波段拐点附近的精确率（补充 61）。"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import macd_core as M
import feature_core as F
import stock_core as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "03_结果")


def main():
    P = S.load_base(); base = P["universe_inv"]
    pv = pd.read_parquet(os.path.join(RES, "59_波段_拐点.parquet"))
    pv = pv[pv.th == 0.10]
    mf = F.build_macd_features(P)
    h, d = mf["hist"], mf["dif"]
    hp, dp = M.shift(h, 1), M.shift(d, 1)
    valid = P["valid"] & (P["bars"] >= 120)
    sig = {
        "柱体上行": (h > hp) & valid,
        "绿柱缩短": (h < 0) & (h > hp) & valid,
        "红柱缩短": (h > 0) & (h < hp) & valid,
        "DIF上行": (d > dp) & valid,
        "DIF下穿0": (d < 0) & (dp >= 0) & valid,
        "DIF上穿0": (d >= 0) & (dp < 0) & valid,
        "DIF>0": (d > 0) & valid,
    }
    n, nc = P["shape"]

    def cov(typ, w=5):
        c = np.zeros((n, nc), dtype=bool)
        for r in pv[pv.type == typ].itertuples():
            c[max(0, r.pivot_i - w):r.pivot_i + w + 1, r.ci] = True
        return c

    cl, ch = cov(-1), cov(1)
    bl = (cl & base).sum() / base.sum()
    bh = (ch & base).sum() / base.sum()
    rows = []
    low = pv[pv.type == -1]; high = pv[pv.type == 1]
    for name, s in sig.items():
        ss = s & base
        rows.append(dict(
            信号=name, 信号数=int(ss.sum()), 每日占比=float(ss.sum() / base.sum()),
            落低点5=float((ss & cl).sum() / ss.sum()),
            低点基准=float(bl),
            低点提升=float((ss & cl).sum() / ss.sum() / bl),
            落高点5=float((ss & ch).sum() / ss.sum()),
            高点基准=float(bh),
            高点提升=float((ss & ch).sum() / ss.sum() / bh),
            低点当天=float(s[low.pivot_i.to_numpy(), low.ci.to_numpy()].mean()),
            高点当天=float(s[high.pivot_i.to_numpy(), high.ci.to_numpy()].mean()),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "61_波段_MACD柱体.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
