#!/usr/bin/env python3
"""21 严格增量：加入 1 日反转、波动、换手、成交额控制

红柱放大可能只是“今天大涨”或高波动/高换手的代理。这里在 20 的模型上再加入：
ret1、std20、turn20、log(amount20)，全部按日 rank 到 [0,1]。
若状态系数仍显著，才承认是 MACD 的独立增量。

输出：03_结果/特征_严格增量.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def r01(x):
    return (rankdata(x) - 0.5) / len(x)


def fm(state, level, controls, ex, mask, dates, restrict=None, min_n=400):
    rows = []
    for t in range(len(dates)):
        m = mask[t] & np.isfinite(ex[t]) & np.isfinite(level[t])
        if restrict is not None:
            m = m & restrict[t]
        for c in controls:
            m = m & np.isfinite(c[t])
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        s = state[t, idx].astype(np.float64)
        if s.sum() < 40 or (1 - s).sum() < 40:
            continue
        lv = r01(level[t, idx]) - 0.5
        cs = [r01(c[t, idx]) for c in controls]
        y = ex[t, idx].astype(np.float64)
        X = np.column_stack([np.ones(len(idx)), lv, *cs, s, s * lv])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((dates[t], *beta))
    cols = ["date", "const", "level", *[f"c{i}" for i in range(len(controls))],
            "state", "state_x_level"]
    d = pd.DataFrame(rows, columns=cols).set_index("date")
    out = []
    for c in ["level", "state", "state_x_level"]:
        early = d.index < pd.Timestamp("2018-01-01")
        out.append(dict(项=c, 系数=d[c].mean(), t=nw_t(d[c], lag=20),
                        系数_2010_2017=d[c][early].mean(), 系数_2018_2026=d[c][~early].mean(),
                        正系数年份=int((d[c].groupby(d.index.year).mean() > 0).sum()),
                        年份数=int(d[c].groupby(d.index.year).mean().notna().sum())))
    return out, len(d)


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = fwd[20] - loo[20]
    mask = base & ok[20] & np.isfinite(ex)

    c = P["close_ff"]
    ret1 = np.full_like(c, np.nan)
    ret1[1:] = c[1:] / c[:-1] - 1
    ret5 = np.full_like(c, np.nan)
    ret5[5:] = c[5:] / c[:-5] - 1
    amount20 = M.rolling_mean(P["amount"], 20)
    logamt = np.log1p(np.maximum(amount20, 0)).astype(np.float32)
    controls = [ret1, ret5, st["ret20"], st["ret60"], st["std20"], st["turn20"], logamt]

    hist = macd["hist"]
    hist_pct = macd["feats"]["柱体水平"]
    dif_pct = macd["feats"]["DIF水平"]
    up = macd["states"]["柱体上行"]
    above = macd["states"]["DIF>DEA"]
    zero_up = macd["states"]["柱体上穿0"]
    red = hist > 0
    green = hist < 0

    cases = [
        ("柱体方向", up, hist_pct, None),
        ("绿柱缩短", up, hist_pct, green),
        ("红柱放大", up, hist_pct, red),
        ("DIF>DEA", above, dif_pct, None),
        ("柱体上穿0", zero_up, hist_pct, None),
    ]
    rows = []
    for name, state, level, restrict in cases:
        res, nd = fm(state, level, controls, ex, mask, P["dates"], restrict)
        for r in res:
            r["检验"] = name
            r["交易日数"] = nd
            rows.append(r)
        print(name, nd)
        for r in res:
            if r["项"] in ("state", "state_x_level"):
                print("  ", r["项"], f"{r['系数']*100:+.3f}%", "t=", round(r["t"], 2),
                      "早", f"{r['系数_2010_2017']*100:+.3f}%",
                      "晚", f"{r['系数_2018_2026']*100:+.3f}%")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "特征_严格增量.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
