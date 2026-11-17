#!/usr/bin/env python3
"""46 条件有效性：5/20/60 日三个期限的稳健性

只挑条件层的关键状态，检查“证据越强越差”是否只在 20 日成立。
口径同 45：T 日状态，T+1 开盘，fwd_h = close[T+h]/open[T+1]-1，
ex_h = fwd_h - 同日全市场等权（leave-one-out）。
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t

RES = os.path.join(os.path.dirname(HERE), "03_结果")
HORIZONS = (5, 20, 60)


def cell_stat(sel, ex, dates, lag):
    n = int(sel.sum())
    if n == 0:
        return dict(股票日=0, 交易日=0, 日均超额=np.nan, t值=np.nan,
                    前期=np.nan, 后期=np.nan)
    cnt = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    dm = np.where(cnt >= 50, s / np.maximum(cnt, 1), np.nan)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(股票日=n, 交易日=int(ok.sum()), 日均超额=np.nan, t值=np.nan,
                    前期=np.nan, 后期=np.nan)
    early = ok & (dates <= pd.Timestamp("2017-12-31"))
    late = ok & (dates >= pd.Timestamp("2018-01-01"))
    return dict(股票日=n, 交易日=int(ok.sum()),
                日均超额=float(np.nanmean(dm)), t值=float(nw_t(dm, lag=lag)),
                前期=float(np.nanmean(dm[early])) if early.any() else np.nan,
                后期=float(np.nanmean(dm[late])) if late.any() else np.nan)


def main():
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)

    fwd, ok, mkt, loo = M.forward_returns(P, HORIZONS)

    # 周/月
    wf = M.weekly_state(P)
    P11 = importlib.import_module("11_周期研究")
    perM = P11.build_period(P, "M")
    difM, deaM, _, _, _, _ = P11.period_signals(P, perM, 12, 26, 9, 36)
    stateM = difM > deaM
    mend = dates[perM["end"]]
    posM = np.searchsorted(mend.values, dates.values, side="left") - 1
    m_on_daily = np.full(P["shape"], np.nan, dtype=np.float32)
    for t in range(len(dates)):
        p = posM[t]
        if p >= 0:
            m_on_daily[t] = stateM[p]
    m_ok = np.isfinite(m_on_daily)
    m_above = (m_on_daily > 0.5) & m_ok
    w_ok = np.isfinite(wf["w_dif"])
    w_above = wf["w_above"].astype(bool) & w_ok

    # 证据分数
    dif, dea, hist = macd["dif"], macd["dea"], macd["hist"]
    score = np.zeros(P["shape"], dtype=np.int8)
    for x in [(dif > 0), (dif > dea), (hist > M.shift(hist, 1)),
              (dif > M.shift(dif, 1)), w_above, m_above]:
        score += x.astype(np.int8)
    score[~(base & w_ok & m_ok)] = -1

    # 条件
    zs = np.full(P["shape"], -1, dtype=np.int8)
    zs[(dif > 0) & (dea > 0)] = 0
    zs[(dif > 0) & (dea <= 0)] = 1
    zs[(dif <= 0) & (dea > 0)] = 2
    zs[(dif <= 0) & (dea <= 0)] = 3
    zs[~base] = -1
    hs = macd["hist_state"]
    c = P["close_ff"]
    dif_pct = macd["feats"]["DIF水平"]
    hist_pct = macd["feats"]["柱体水平"]
    slope = macd["feats"]["DIF5日斜率"]
    q_hist = F.row_quantile(hist_pct, base, 5)
    q_dif = F.row_quantile(dif_pct, base, 5)
    q_slope = F.row_quantile(slope, base, 5)
    high20 = M.rolling_max(c, 20)
    low20 = M.rolling_min(c, 20)
    dmax20 = M.rolling_max(dif_pct, 20)
    dmin20 = M.rolling_min(dif_pct, 20)
    top_div = base & (c >= high20 - 1e-6) & (dif_pct < dmax20 - 1e-9)
    bot_div = base & (c <= low20 + 1e-6) & (dif_pct > dmin20 + 1e-9)

    conds = [
        ("全部多头", base),
        ("零轴双上", zs == 0),
        ("零轴双下", zs == 3),
        ("红柱放大", base & (hs == 0)),
        ("绿柱缩短", base & (hs == 2)),
        ("柱体水平Q1", base & (q_hist == 0)),
        ("柱体水平Q3", base & (q_hist == 2)),
        ("柱体水平Q5", base & (q_hist == 4)),
        ("DIF水平Q2", base & (q_dif == 1)),
        ("DIF水平Q5", base & (q_dif == 4)),
        ("DIF斜率Q3", base & (q_slope == 2)),
        ("DIF斜率Q5", base & (q_slope == 4)),
        ("证据分数<=1", base & (score <= 1)),
        ("证据分数>=5", base & (score >= 5)),
        ("日周月共振", base & (dif > dea) & w_above & m_above),
        ("顶背离", top_div),
        ("底背离", bot_div),
        ("市场下行", base & (st["market"] == 1)),
    ]

    rows = []
    for h in HORIZONS:
        f = fwd[h]
        mask = base & ok[h] & np.isfinite(f)
        ex = (f - loo[h]).astype(np.float32)
        for lab, cm in conds:
            for side, sel in [("多", cm & (dif > dea)), ("空", cm & ~(dif > dea))]:
                s = sel & mask
                cst = cell_stat(s, ex, dates, lag=max(5, h))
                rows.append(dict(条件=lab, 方向=side, 周期=h, **cst))
        print(f"  {h}日 done")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "条件_期限.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
