#!/usr/bin/env python3
"""55 解剖 K<20（MACD多内）：逐年、分布、先期收益/波动/成交额控制、匹配对照。"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import kdj_core as K
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def dm_series(sel, ex, min_n=30):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def contrast(a, b, ex):
    da, _ = dm_series(a, ex)
    db, _ = dm_series(b, ex)
    common = np.isfinite(da) & np.isfinite(db)
    if common.sum() < 20:
        return dict(A=np.nan, B=np.nan, 差值=np.nan, t=np.nan, 共同日=int(common.sum()))
    x = da - db
    return dict(A=float(np.nanmean(da[common])), B=float(np.nanmean(db[common])),
                差值=float(np.nanmean(x[common])), t=float(nw_t(x[common], 20)),
                共同日=int(common.sum()))


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = (fwd[20] - loo[20]).astype(np.float32)
    m20 = base & ok[20] & np.isfinite(ex) & (dates >= pd.Timestamp("2010-01-01"))[:, None]
    mb = macd["states"]["DIF>DEA"] & m20
    k = kd["k"]
    low = mb & (k < 20)
    rest = mb & (k >= 20)
    print("K<20 bull n", int(low.sum()), "days", int(np.isfinite(dm_series(low, ex)[0]).sum()))
    print("mean", np.nanmean(ex[low]), "median", np.nanmedian(ex[low]),
          "trim5%", np.nanmean(np.clip(ex[low], *np.nanpercentile(ex[low], [5, 95]))))
    dm, _ = dm_series(low, ex)
    for y in range(2010, 2027):
        sel = np.isfinite(dm) & (dates.year == y)
        if sel.sum() == 0:
            continue
        vals = dm[sel]
        print(y, "days", len(vals), "mean", np.nanmean(vals), "median", np.nanmedian(vals),
              "t", nw_t(vals, 5), "stockdays", int((low & (dates.year == y)[:, None]).sum()))

    # 口径对比
    summ = []
    summ.append(dict(口径="全部股票日", 均值=float(np.nanmean(ex[low])), 中位=float(np.nanmedian(ex[low])),
                     修剪均值=float(np.nanmean(np.clip(ex[low], *np.nanpercentile(ex[low], [5, 95])))),
                     股票日=int(low.sum()), 交易日=int(len(dates[np.where(low)[0]].unique())), t=np.nan))
    for mn in (1, 10, 30, 50):
        n = low.sum(axis=1); ss = np.nansum(np.where(low, ex, 0.0), axis=1)
        d = np.where(n >= mn, ss / np.maximum(n, 1), np.nan)
        ok = np.isfinite(d)
        summ.append(dict(口径=f"日度≥{mn}只", 均值=float(np.nanmean(d[ok])),
                         中位=float(np.nanmedian(d[ok])), 修剪均值=float(np.nanmean(np.clip(d[ok], *np.nanpercentile(d[ok], [5, 95])))),
                         股票日=int(low.sum()), 交易日=int(ok.sum()),
                         t=float(nw_t(d[ok], 20)) if ok.sum() >= 20 else np.nan))
    pd.DataFrame(summ).to_csv(os.path.join(RES, "KDJ_K20解剖.csv"), index=False, encoding="utf-8-sig")
    # 逐年
    yrows = []
    for y in range(2010, 2027):
        sel = np.isfinite(dm) & (dates.year == y)
        if sel.sum() == 0:
            continue
        vals = dm[sel]
        yrows.append(dict(年份=y, 交易日=int(sel.sum()), 日均超额=float(np.nanmean(vals)),
                          中位=float(np.nanmedian(vals)), t=float(nw_t(vals, 5)) if len(vals) >= 20 else np.nan,
                          股票日=int((low & (dates.year == y)[:, None]).sum())))
    pd.DataFrame(yrows).to_csv(os.path.join(RES, "KDJ_K20逐年.csv"), index=False, encoding="utf-8-sig")

    # 最 contributive days
    contrib = pd.DataFrame({"date": dates, "dm": dm}).dropna()
    contrib["year"] = contrib.date.dt.year
    top = contrib.reindex(contrib.dm.abs().sort_values(ascending=False).index).head(15)
    print("\ntop |daily mean| days")
    print(top.to_string(index=False))

    # 双重排序
    r5 = np.full(P["shape"], np.nan, dtype=np.float32)
    r5[5:] = P["close_ff"][5:] / P["close_ff"][:-5] - 1
    qr5 = F.row_quantile(r5, base, 5)
    qr20 = F.row_quantile(st["ret20"], base, 5)
    qv = F.row_quantile(st["std20"], base, 5)
    qa = F.row_quantile(M.rolling_mean(P["amount"], 20), base, 5)
    rows = []
    for nm, q in [("5日收益", qr5), ("20日收益", qr20), ("波动", qv), ("成交额", qa)]:
        for qi in range(5):
            cell = mb & (q == qi)
            a = cell & (k < 20)
            b = cell & (k >= 20)
            if a.sum() < 300 or b.sum() < 300:
                continue
            r = contrast(a, b, ex)
            rows.append(dict(维度=nm, 分位=f"Q{qi+1}", A股日=int(a.sum()), B股日=int(b.sum()), **r))
    ctrl = pd.DataFrame(rows)
    ctrl.to_csv(os.path.join(RES, "KDJ_K20控制.csv"), index=False, encoding="utf-8-sig")
    print("\nK<20 vs K>=20 控制")
    print(ctrl.to_string(index=False))

    # 匹配：每个 K<20 股票日，在同一天、同20日收益分位、同波动分位里随机抽一个 K>=20 对照
    rng = np.random.default_rng(20261031)
    match = np.zeros(P["shape"], dtype=bool)
    for t in range(P["shape"][0]):
        idx = np.where(low[t])[0]
        if len(idx) == 0:
            continue
        for j in idx:
            cand = np.where(mb[t] & (k[t] >= 20) & (qr20[t] == qr20[t, j]) & (qv[t] == qv[t, j]))[0]
            if len(cand) == 0:
                continue
            match[t, rng.choice(cand)] = True
    print("\nmatch n", int(match.sum()), "vs low", int(low.sum()))
    print("matched mean", np.nanmean(ex[match]), "low mean", np.nanmean(ex[low]))
    print(contrast(low, match, ex))
    # 匹配后的逐年差异
    da, _ = dm_series(low, ex); db, _ = dm_series(match, ex)
    common = np.isfinite(da) & np.isfinite(db)
    d = da - db
    rows = []
    for y in range(2010, 2027):
        sel = common & (dates.year == y)
        if sel.sum() < 20:
            continue
        rows.append(dict(年份=y, 差值=float(np.nanmean(d[sel])), t=float(nw_t(d[sel], 5)),
                         交易日=int(sel.sum())))
    mrow = pd.DataFrame(rows)
    mrow["低K均值"] = np.nanmean(ex[low])
    mrow["匹配均值"] = np.nanmean(ex[match])
    mrow.to_csv(os.path.join(RES, "KDJ_K20匹配.csv"), index=False, encoding="utf-8-sig")
    print(mrow.to_string(index=False))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
