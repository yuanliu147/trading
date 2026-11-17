#!/usr/bin/env python3
"""31 月线趋势定义稳健性：价格<MA200 / MA200斜率<0 / 两者同时

27-28 的“下行”= 价<MA200 且 MA200 20 日斜率<0。这里换定义，看月线柱体水平的
正 IC 是否只在某一种定义下成立。
输出：03_结果/月度下行_趋势定义.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t

spec = importlib.util.spec_from_file_location("m27", os.path.join(HERE, "27_月度下行.py"))
m27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m27)

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")


def main():
    P = M.load_panel()
    mf = M.market_features(P)
    st = F.build_structure(P, mf)
    per = m27.build_period(P)
    dif, dea, hist, valid = m27.sig(per)
    c = M._fill_lead(per["close"])
    hist_pct = hist / c * 100
    rets, ok, mkt, loo = m27.prets(P, per, [3, 12])

    close_d = P["close_ff"][per["end"], :]
    ma200 = st["ma200"][per["end"], :]
    slope = st["slope200"][per["end"], :]
    is_st = P["is_st"][per["end"], :] == 1
    bars = P["bars"][per["end"], :]
    price = P["raw_close"][per["end"], :]
    amt = M.rolling_mean(P["amount"], 20)[per["end"], :]

    base_liq = ~is_st & (bars >= 250) & (price > 2)
    liq = np.full(amt.shape, np.nan, dtype=np.float32)
    for p in range(per["nper"]):
        idx = np.where(base_liq[p] & np.isfinite(amt[p]))[0]
        if len(idx) >= 100:
            liq[p, idx] = (rankdata(amt[p, idx]) - 0.5) / len(idx)
    hi = base_liq & (liq >= 0.5)

    below = close_d < ma200
    slope_dn = slope < 0
    samples = {
        "价<MA200": below,
        "斜率<0": slope_dn,
        "价<MA200且斜率<0": below & slope_dn,
        "价>MA200": ~below,
        "斜率>0": slope > 0,
        "价>MA200且斜率>0": (~below) & (slope > 0),
    }
    rows = []
    for H in [3, 12]:
        ex = rets[H] - loo[H]
        base = valid & ok[H] & np.isfinite(ex)
        for name, sm in samples.items():
            m = base & hi & sm
            s = m27.ic_series(hist_pct, ex, m)
            if len(s) < 20:
                continue
            ss = pd.Series(s.ic.values, index=P["dates"][per["end"]][s.p.values])
            early = ss.index < pd.Timestamp("2018-01-01")
            # Q5-Q1 月均
            qs = []
            for p in range(per["nper"]):
                idx = np.where(m[p] & np.isfinite(rets[H][p]) & np.isfinite(hist_pct[p]))[0]
                if len(idx) < 100:
                    continue
                q = np.minimum(((rankdata(hist_pct[p, idx]) - 1) * 5 / len(idx)).astype(np.int16), 4)
                cnt = np.bincount(q, minlength=5)
                if (cnt == 0).any():
                    continue
                qa = np.bincount(q, weights=rets[H][p, idx], minlength=5) / cnt
                qs.append(qa[4] - qa[0])
            rows.append(dict(周期=H, 定义=name, 月数=len(ss), 股票月=int(m.sum()),
                             平均IC=ss.mean(), IC_t12=nw_t(ss, lag=12),
                             IC_2010_2017=ss[early].mean(), IC_2018_2026=ss[~early].mean(),
                             IC大于0占比=(ss > 0).mean(),
                             Q5_Q1月均=np.mean(qs) if qs else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "月度下行_趋势定义.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
