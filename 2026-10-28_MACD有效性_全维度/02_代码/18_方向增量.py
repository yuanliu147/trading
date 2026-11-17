#!/usr/bin/env python3
"""18 方向增量：MACD 的“方向”是否独立于“高度”

前面的状态差可能只是“柱体越高/ DIF 越高，未来越差”的水平效应。
这里在每个交易日内，先按 MACD 水平分五组，再在组内比较方向状态：
- 柱体上行 vs 下行（控制柱体水平）
- DIF>DEA vs DIF<=DEA（控制 DIF 水平）
- 红柱放大 vs 红柱缩短（只在红柱内，控制柱体水平）
- 绿柱缩短 vs 绿柱放大（只在绿柱内，控制柱体水平）
- 柱体上穿 0（控制柱体水平）

输出：03_结果/特征_方向增量.csv、特征_方向矩阵.csv
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


def strat_effect(state, level, ex, mask, dates, nq=5, min_n=200, min_cell=15):
    daily, per_level = [], {k: [] for k in range(nq)}
    for t in range(len(dates)):
        m = mask[t] & np.isfinite(ex[t]) & np.isfinite(level[t])
        idx = np.where(m)[0]
        if len(idx) < min_n:
            continue
        lv = level[t, idx]
        lr = rankdata(lv)
        lq = np.minimum(((lr - 1) * nq / len(idx)).astype(np.int16), nq - 1)
        y = ex[t, idx]
        s = state[t, idx]
        diffs, weights = [], []
        for k in range(nq):
            mk = lq == k
            s1 = mk & s
            s0 = mk & ~s
            n1, n0 = int(s1.sum()), int(s0.sum())
            if n1 < min_cell or n0 < min_cell:
                continue
            d = y[s1].mean() - y[s0].mean()
            diffs.append(d)
            weights.append(n1 + n0)
            per_level[k].append((dates[t], d, n1, n0))
        if diffs:
            daily.append((dates[t], np.average(diffs, weights=weights)))
    d = pd.DataFrame(daily, columns=["date", "delta"]).set_index("date")
    rows = []
    for k in range(nq):
        x = per_level[k]
        if len(x) < 20:
            rows.append(dict(水平组=k + 1, 交易日数=len(x), 方向差=np.nan,
                             方向t=np.nan, 状态1数=np.nan, 状态0数=np.nan))
            continue
        z = pd.DataFrame(x, columns=["date", "d", "n1", "n0"])
        rows.append(dict(水平组=k + 1, 交易日数=len(z), 方向差=z.d.mean(),
                         方向t=nw_t(z.d, lag=20), 状态1数=z.n1.mean(), 状态0数=z.n0.mean()))
    early = d.index < pd.Timestamp("2018-01-01")
    summary = dict(交易日数=len(d), 方向差=d.delta.mean(),
                   方向t=nw_t(d.delta, lag=20),
                   方向差_2010_2017=d.delta[early].mean(),
                   方向差_2018_2026=d.delta[~early].mean(),
                   差大于0年份=int((d.delta.groupby(d.index.year).mean() > 0).sum()),
                   年份数=int(d.delta.groupby(d.index.year).mean().notna().sum()))
    return summary, pd.DataFrame(rows)


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
    red = hist > 0
    green = hist < 0
    zero_up = macd["states"]["柱体上穿0"]

    cases = [
        ("柱体方向(全样本)", up, hist_pct, None),
        ("DIF>DEA(控制DIF)", above, dif_pct, None),
        ("红柱放大(红柱内)", up, hist_pct, red),
        ("绿柱缩短(绿柱内)", up, hist_pct, green),
        ("柱体上穿0(控制柱体)", zero_up, hist_pct, None),
    ]
    rows, mats = [], []
    for name, state, level, restrict in cases:
        m = mask if restrict is None else (mask & restrict)
        sm = state if restrict is None else (state & restrict)
        summ, lv = strat_effect(sm, level, ex, m, P["dates"])
        summ["检验"] = name
        rows.append(summ)
        lv["检验"] = name
        mats.append(lv)
        print(name, summ)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "特征_方向增量.csv"), index=False, encoding="utf-8-sig")
    pd.concat(mats).to_csv(os.path.join(RES, "特征_方向矩阵.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(pd.concat(mats).to_string(index=False))


if __name__ == "__main__":
    main()
