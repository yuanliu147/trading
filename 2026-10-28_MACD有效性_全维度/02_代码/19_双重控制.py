#!/usr/bin/env python3
"""19 双重控制：方向效应是否独立于“柱体高度”和“过去涨幅”

18 发现方向效应随 MACD 水平强烈变化；17 发现控制过去涨幅后方向差变负。
这里同时按 当日 MACD 水平五分位 × 过去 20 日涨幅五分位 分层，再比较状态：
- 柱体上行 vs 下行
- 绿柱缩短 vs 绿柱放大（只在绿柱内）
- 红柱放大 vs 红柱缩短（只在红柱内）
- DIF>DEA vs DIF<=DEA
- 柱体上穿 0

输出：03_结果/特征_双重控制.csv
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


def qcut_rows(x, mask, nq=5):
    out = np.full(x.shape, -1, dtype=np.int8)
    for t in range(x.shape[0]):
        idx = np.where(mask[t] & np.isfinite(x[t]))[0]
        if len(idx) < nq * 20:
            continue
        r = rankdata(x[t, idx])
        out[t, idx] = np.minimum(((r - 1) * nq / len(idx)).astype(np.int16), nq - 1)
    return out


def double_strat(state, level, ret, ex, mask, dates, nq=5, min_n=200, min_cell=10):
    daily = []
    for t in range(len(dates)):
        m = mask[t] & np.isfinite(ex[t]) & np.isfinite(level[t]) & np.isfinite(ret[t])
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        y = ex[t, idx]
        s = state[t, idx]
        lv = level[t, idx]
        rv = ret[t, idx]
        lq = np.minimum(((rankdata(lv) - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        rq = np.minimum(((rankdata(rv) - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        diffs, weights = [], []
        for a in range(nq):
            for b in range(nq):
                cell = (lq == a) & (rq == b)
                n1 = int((cell & s).sum())
                n0 = int((cell & ~s).sum())
                if n1 < min_cell or n0 < min_cell:
                    continue
                diffs.append(y[cell & s].mean() - y[cell & ~s].mean())
                weights.append(n1 + n0)
        if diffs:
            daily.append((dates[t], np.average(diffs, weights=weights)))
    d = pd.DataFrame(daily, columns=["date", "delta"]).set_index("date")
    if len(d) < 20:
        return None
    early = d.index < pd.Timestamp("2018-01-01")
    return dict(交易日数=len(d), 方向差=d.delta.mean(), 方向t=nw_t(d.delta, lag=20),
                方向差_2010_2017=d.delta[early].mean(), 方向差_2018_2026=d.delta[~early].mean(),
                差大于0年份=int((d.delta.groupby(d.index.year).mean() > 0).sum()),
                年份数=int(d.delta.groupby(d.index.year).mean().notna().sum()))


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = fwd[20] - loo[20]
    mask = base & ok[20] & np.isfinite(ex)

    hist = macd["hist"]
    dif = macd["dif"]
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
        m = mask if restrict is None else (mask & restrict)
        sm = state if restrict is None else (state & restrict)
        r = double_strat(sm, level, st["ret20"], ex, m, P["dates"])
        if r is None:
            continue
        r["检验"] = name
        rows.append(r)
        print(name, r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "特征_双重控制.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
