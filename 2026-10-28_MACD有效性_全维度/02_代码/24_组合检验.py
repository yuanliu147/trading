#!/usr/bin/env python3
"""24 组合检验：把 MACD 特征做成 20 日非重叠组合

每个交易日历起点（每 20 个交易日）按 T 日信号选股，T+1 开盘等权买入，
持有 20 个交易日，扣 0.15% 往返成本。比较：
- 全 A 等权
- 剔除红柱放大 / 剔除柱体上行 / 剔除 DIF>DEA
- 只买 DIF 最低 20% / 只买柱体下行
- 空头排列 + 柱体下行 / 空头排列 + 绿柱缩短

注意：回撤是 20 日采样，不是日频路径；用于横向比较，不用于精确 MDD。
输出：03_结果/特征_组合.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
FEE = 0.0015
H = 20


def perf(rets):
    r = np.asarray(rets, dtype=float)
    eq = np.cumprod(1 + r)
    n = len(r)
    cagr = eq[-1] ** (252 / H / n) - 1 if n else np.nan
    mdd = (eq / np.maximum.accumulate(eq) - 1).min() if n else np.nan
    sharpe = r.mean() / r.std() * np.sqrt(252 / H) if r.std() > 0 else np.nan
    return dict(期数=n, 年化=cagr, 回撤=mdd, 夏普=sharpe,
                胜率=(r > 0).mean(), 平均每期=r.mean())


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    fwd, ok, mkt, loo = M.forward_returns(P, (H,))
    ex = fwd[H] - loo[H]
    mask0 = base & ok[H] & np.isfinite(ex)

    hist = macd["hist"]
    hist_pct = macd["feats"]["柱体水平"]
    dif_pct = macd["feats"]["DIF水平"]
    red_expand = macd["states"]["红柱放大"]
    hist_up = macd["states"]["柱体上行"]
    above = macd["states"]["DIF>DEA"]
    green_shrink = macd["states"]["绿柱缩短"]
    bear_align = st["align"] == 1

    # 每日横截面分位
    from scipy.stats import rankdata

    def rank_mat(x):
        out = np.full(x.shape, np.nan, dtype=np.float32)
        for t in range(len(P["dates"])):
            idx = np.where(mask0[t] & np.isfinite(x[t]))[0]
            if len(idx) < 100:
                continue
            out[t, idx] = (rankdata(x[t, idx]) - 0.5) / len(idx)
        return out

    dif_rank = rank_mat(dif_pct)
    ret20_rank = rank_mat(st["ret20"])
    ret60_rank = rank_mat(st["ret60"])
    dist20_rank = rank_mat(st["dist_ma20"])
    vol_rank = rank_mat(st["std20"])

    strat = {
        "全A等权": lambda t: mask0[t],
        "剔除红柱放大": lambda t: mask0[t] & ~red_expand[t],
        "剔除柱体上行": lambda t: mask0[t] & ~hist_up[t],
        "剔除DIF>DEA": lambda t: mask0[t] & ~above[t],
        "只买DIF最低20%": lambda t: mask0[t] & (dif_rank[t] <= 0.2),
        "只买20日涨幅最低20%": lambda t: mask0[t] & (ret20_rank[t] <= 0.2),
        "只买60日涨幅最低20%": lambda t: mask0[t] & (ret60_rank[t] <= 0.2),
        "只买距MA20最低20%": lambda t: mask0[t] & (dist20_rank[t] <= 0.2),
        "只买低波20%": lambda t: mask0[t] & (vol_rank[t] <= 0.2),
        "只买DIF最低20%+柱体下行": lambda t: mask0[t] & (dif_rank[t] <= 0.2) & ~hist_up[t],
        "只买DIF最低20%+空头排列": lambda t: mask0[t] & (dif_rank[t] <= 0.2) & bear_align[t],
        "只买柱体下行": lambda t: mask0[t] & ~hist_up[t],
        "空头排列+柱体下行": lambda t: mask0[t] & bear_align[t] & ~hist_up[t],
        "空头排列+绿柱缩短": lambda t: mask0[t] & bear_align[t] & green_shrink[t],
    }

    dates = P["dates"]
    start = np.searchsorted(dates, np.datetime64("2010-01-01"))
    tgrid = np.arange(start, len(dates) - H, H)
    rows = []
    for name, fn in strat.items():
        recs = []
        for t in tgrid:
            sel = fn(t)
            if sel.sum() < 30:
                continue
            r = fwd[H][t, sel].mean() - FEE
            if np.isfinite(r):
                recs.append((dates[t], r, int(sel.sum())))
        d = pd.DataFrame(recs, columns=["date", "r", "n"])
        if d.empty:
            continue
        for seg, dd in [("2010-2026", d), ("2018-2026", d[d.date >= "2018-01-01"])]:
            if len(dd) < 10:
                continue
            rows.append(dict(策略=name, 区间=seg, **perf(dd.r), 平均持仓=dd.n.mean()))
        print(name, len(d))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "特征_组合.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
