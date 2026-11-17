#!/usr/bin/env python3
"""05 信号变体：零轴、绿柱、周线、背离、多周期确认

统一口径：T 日收盘信号，T+1 开盘买入；20 日收益的当日横截面分位（pct20）为主指标，
辅以超额 ex20、按信号日聚类 t、2010-2017/2018-2026 分段。

输出：03_结果/信号变体.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def pct_rank(f, ok, min_n=50):
    n, d = f.shape
    pct = np.full((n, d), np.nan, dtype=np.float32)
    for t in range(n):
        idx = np.where(ok[t] & np.isfinite(f[t]))[0]
        if len(idx) < min_n:
            continue
        x = f[t, idx]
        order = np.argsort(np.argsort(x))
        pct[t, idx] = (order + 0.5) / len(idx)
    return pct


def stats_for(P, sig, pct, fwd, ok, loo, label, mask=None):
    ii, jj = np.where(sig)
    keep = P["dates"][ii] >= pd.Timestamp("2010-01-01")
    ii, jj = ii[keep], jj[keep]
    if mask is not None:
        m = mask[ii, jj]
        ii, jj = ii[m], jj[m]
    v = ok[20][ii, jj] & np.isfinite(fwd[20][ii, jj])
    p = pct[ii, jj][v]
    ex = (fwd[20][ii, jj] - loo[20][ii, jj])[v]
    dd = P["dates"][ii][v]
    if len(p) < 100:
        return None
    df = pd.DataFrame({"e": ex, "d": dd})
    sums = df.groupby("d").e.sum().to_numpy()
    tcl = ex.mean() / (np.sqrt(np.sum(sums ** 2)) / len(df))
    dm = df.groupby("d").e.mean()
    early = P["dates"][ii][v] < pd.Timestamp("2018-01-01")
    return dict(变体=label, n=len(p), mean_pct=p.mean(), median_pct=np.median(p),
                share50=(p > 0.5).mean(), mean_ex=ex.mean(),
                t_cluster=tcl, mean_daily=dm.mean(),
                pct_early=p[early].mean(), pct_late=p[~early].mean(),
                n_early=int(early.sum()), n_late=int((~early).sum()))


def bottom_divergence(P, gold, dif):
    """金叉处的 60/60 日双底背离：价格创新低、DIF 不创新低。"""
    ii, jj = np.where(gold)
    out = np.zeros(len(ii), dtype=bool)
    low, c = P["low"], P["close"]
    for k, (t, j) in enumerate(zip(ii, jj)):
        if t < 125:
            continue
        a = low[t - 59 : t + 1, j]
        b = low[t - 119 : t - 59, j]
        if not np.isfinite(a).any() or not np.isfinite(b).any():
            continue
        ia, ib = np.nanargmin(a), np.nanargmin(b)
        pa, pb = a[ia], b[ib]
        if not (pa < pb):
            continue
        da, db = dif[t - 59 + ia, j], dif[t - 119 + ib, j]
        if np.isfinite(da) and np.isfinite(db) and da > db:
            out[k] = True
    return out


def main():
    P = M.load_panel()
    dif, dea, hist = M.macd(P)
    gold, death = M.crosses(dif, dea, P)
    feats = M.stock_features(P)
    wf = M.weekly_state(P)
    fwd, ok, mkt, loo = M.forward_returns(P)
    pct = pct_rank(fwd[20], ok[20])

    # 基础信号
    prev_dif = M.shift(dif, 1)
    zero_up = P["valid"] & (P["bars"] >= 120) & (dif > 0) & (prev_dif <= 0)
    zero_dn = P["valid"] & (P["bars"] >= 120) & (dif < 0) & (prev_dif >= 0)
    # 绿柱缩短：hist 从负转升（连续两根），且尚未金叉
    h1, h2 = M.shift(hist, 1), M.shift(hist, 2)
    green_turn = P["valid"] & (P["bars"] >= 120) & (hist < 0) & (hist > h1) & (h1 <= h2)
    # 金叉的零轴/周线/密度掩码
    zpos = dif > 0
    wbear = ~wf["w_above"]
    wbull = wf["w_above"]
    wgold = wf["w_gold4"] > 0
    cross60 = np.cumsum(gold, axis=0) - M.shift(np.cumsum(gold, axis=0), 60, fill=0) - 1
    # 周线金叉：上周完成周线金叉后的第一个交易日
    wgold_sig = wf["w_gold"] & ~M.shift(wf["w_gold"], 1).astype(bool)
    wgold_sig = wgold_sig & P["valid"]
    div_all = bottom_divergence(P, gold, dif)
    gii, gjj = np.where(gold)
    gkeep = P["dates"][gii] >= pd.Timestamp("2010-01-01")
    div = div_all[gkeep]
    gii, gjj = gii[gkeep], gjj[gkeep]
    div_full = np.zeros(P["shape"], dtype=bool)
    div_full[gii[div], gjj[div]] = True
    div = div_full

    variants = [
        ("金叉(12,26,9)", gold, None),
        ("DIF上穿0", zero_up, None),
        ("DIF下穿0(反指)", zero_dn, None),
        ("绿柱缩短(未金叉)", green_turn, None),
        ("金叉+零轴下", gold, ~zpos),
        ("金叉+零轴上", gold, zpos),
        ("金叉+周线空头", gold, wbear),
        ("金叉+周线多头", gold, wbull),
        ("金叉+近4周周线金叉", gold, wgold),
        ("金叉+底背离", gold, div),
        ("金叉+非底背离", gold, ~div),
        ("金叉+60日首次", gold, cross60 == 0),
        ("金叉+60日第2次以上", gold, cross60 >= 1),
        ("周线金叉(完整周)", wgold_sig, None),
    ]
    rows = []
    for label, sig, mask in variants:
        st = stats_for(P, sig, pct, fwd, ok, loo, label, mask)
        if st:
            rows.append(st)
            print(st)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "信号变体.csv"), index=False, encoding="utf-8-sig")
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
