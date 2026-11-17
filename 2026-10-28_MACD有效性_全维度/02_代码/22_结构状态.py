#!/usr/bin/env python3
"""22 结构状态图：红柱放大 / DIF>DEA 在各趋势结构下的方向差

对 7 个结构维度 × 3 个桶，分别计算：
- 状态 1 与状态 0 的 20 日超额差（原始）
- 控制 MACD 水平三分位 × 过去 20 日涨幅三分位 后的差
- 状态 1 / 状态 0 的 20 日绝对收益、同期市场绝对收益
输出：03_结果/特征_结构状态.csv、特征_状态矩阵.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")

DIMS = {
    "长期趋势": ("trend", ["上行", "下行", "过渡"]),
    "中期排列": ("align", ["多头排列", "空头排列", "混合"]),
    "250日位置": ("pos", ["高位", "中位", "深位"]),
    "短期结构": ("struct20", ["高高低低", "低高低低", "收敛/其他"]),
    "波动": ("volq", ["低波", "中波", "高波"]),
    "换手": ("turnq", ["低换手", "中换手", "高换手"]),
    "市场状态": ("market", ["指数上行", "指数下行", "过渡"]),
}


def q3(x):
    r = rankdata(x)
    return np.minimum(((r - 1) * 3 / len(x)).astype(np.int16), 2)


def effect(state, level, ret20, absex, ex, mask, dates, min_n=150, min_cell=8):
    raw, adj, a1, a0, am, ns, ds = [], [], [], [], [], [], []
    for t in range(len(dates)):
        idx = np.where(mask[t] & np.isfinite(ex[t]) & np.isfinite(level[t]))[0]
        if len(idx) < min_n:
            continue
        s = state[t, idx]
        n1, n0 = int(s.sum()), int((~s).sum())
        if n1 < 20 or n0 < 20:
            continue
        y = ex[t, idx]
        raw.append(y[s].mean() - y[~s].mean())
        ds.append(dates[t])
        # 控制：水平三分位 × ret20 三分位
        lq = q3(level[t, idx])
        rq = q3(ret20[t, idx])
        diffs, w = [], []
        for a in range(3):
            for b in range(3):
                cell = (lq == a) & (rq == b)
                c1, c0 = int((cell & s).sum()), int((cell & ~s).sum())
                if c1 < min_cell or c0 < min_cell:
                    continue
                diffs.append(y[cell & s].mean() - y[cell & ~s].mean())
                w.append(c1 + c0)
        if diffs:
            adj.append(np.average(diffs, weights=w))
        a1.append(absex[t, idx][s].mean())
        a0.append(absex[t, idx][~s].mean())
        am.append(absex[t, idx].mean())
        ns.append(len(idx))
    if len(raw) < 20:
        return None
    r = pd.Series(raw, index=ds)
    a = pd.Series(adj, index=ds[:len(adj)])
    early = r.index < pd.Timestamp("2018-01-01")
    return dict(交易日数=len(r), 股票日=int(np.sum(ns)),
                原始差=r.mean(), 原始t=nw_t(r, lag=20),
                控制后差=a.mean() if len(a) else np.nan,
                控制t=nw_t(a, lag=20) if len(a) else np.nan,
                状态1绝对=np.mean(a1), 状态0绝对=np.mean(a0), 市场绝对=np.mean(am),
                原始差_2010_2017=r[early].mean(), 原始差_2018_2026=r[~early].mean())


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    st["volq"] = F.row_tercile(st["std20"], base)
    st["turnq"] = F.row_tercile(st["turn20"], base)
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    absex = fwd[20]
    ex = fwd[20] - loo[20]
    mask = base & ok[20] & np.isfinite(ex)

    hist = macd["hist"]
    states = {
        "红柱放大": (macd["states"]["红柱放大"], macd["feats"]["柱体水平"], hist > 0),
        "DIF>DEA": (macd["states"]["DIF>DEA"], macd["feats"]["DIF水平"], None),
    }
    rows = []
    for dim, (key, labels) in DIMS.items():
        code = st[key]
        for k, lab in enumerate(labels):
            m = mask & (code == k)
            if m.sum() < 1000:
                continue
            for name, (state, level, restrict) in states.items():
                mm = m if restrict is None else (m & restrict)
                sm = state if restrict is None else (state & restrict)
                r = effect(sm, level, st["ret20"], absex, ex, mm, P["dates"])
                if r is None:
                    continue
                r.update(维度=dim, 分组=lab, 状态=name)
                rows.append(r)
            print("完成", dim, lab)
    out = pd.DataFrame(rows)
    # BH：每个状态内
    out["q_bh"] = np.nan
    for s, idx in out.groupby("状态").groups.items():
        p = 2 * (1 - norm.cdf(np.abs(out.loc[idx, "控制t"].values)))
        order = np.argsort(p); m = len(p)
        q = np.empty(m); ranked = p[order] * m / (np.arange(m) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q[order] = np.minimum(ranked, 1)
        out.loc[idx, "q_bh"] = q
    out.to_csv(os.path.join(RES, "特征_结构状态.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
