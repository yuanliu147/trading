#!/usr/bin/env python3
"""20 Fama-MacBeth：MACD 状态在价格控制下还有多少增量

每个交易日做横截面回归：
  ex20 ~ 1 + MACD水平(中心化) + ret5 + ret20 + ret60 + 状态 + 状态×MACD水平
状态系数 = 在“同等 MACD 水平、同等 5/20/60 日涨幅”下，状态 1 相对状态 0 的
20 日超额收益差；交互项 = 该差随 MACD 水平的变化。
报告系数均值、Newey-West t、前后段。

输出：03_结果/特征_FamaMacBeth.csv
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


def fm_regression(state, level, ret5, ret20, ret60, ex, mask, dates, restrict=None, min_n=300):
    rows = []
    for t in range(len(dates)):
        m = mask[t] & np.isfinite(ex[t]) & np.isfinite(level[t])
        if restrict is not None:
            m = m & restrict[t]
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        s = state[t, idx].astype(np.float64)
        if s.sum() < 30 or (1 - s).sum() < 30:
            continue
        lv = r01(level[t, idx]) - 0.5
        c5 = r01(ret5[t, idx])
        c20 = r01(ret20[t, idx])
        c60 = r01(ret60[t, idx])
        y = ex[t, idx].astype(np.float64)
        X = np.column_stack([np.ones(len(idx)), lv, c5, c20, c60, s, s * lv])
        # 去掉常数列/共线列
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append((dates[t], *beta))
    cols = ["date", "const", "level", "ret5", "ret20", "ret60", "state", "state_x_level"]
    d = pd.DataFrame(rows, columns=cols).set_index("date")
    out = []
    for c in ["level", "ret5", "ret20", "ret60", "state", "state_x_level"]:
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
    ret5 = np.full_like(c, np.nan)
    ret5[5:] = c[5:] / c[:-5] - 1

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
        res, nd = fm_regression(state, level, ret5, st["ret20"], st["ret60"],
                                ex, mask, P["dates"], restrict)
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
    out.to_csv(os.path.join(RES, "特征_FamaMacBeth.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
