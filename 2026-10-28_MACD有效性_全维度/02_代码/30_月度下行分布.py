#!/usr/bin/env python3
"""30 月度下行分布：Q1/Q5 的均值、中位、胜率、尾部

回答“高柱体在下跌趋势里好”是不是典型交易也好，还是只有均值被右尾拉起。
样本：月线、长期趋势下行、非 ST、上市>=250 日、价格>2、成交额上半区。
输出：03_结果/月度下行_分布.csv
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
import importlib.util

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
    dif_pct = dif / c * 100
    rets, ok, mkt, loo = m27.prets(P, per, [3, 12])
    trend = st["trend"][per["end"], :]
    is_st = P["is_st"][per["end"], :] == 1
    bars = P["bars"][per["end"], :]
    price = P["raw_close"][per["end"], :]
    amt = M.rolling_mean(P["amount"], 20)[per["end"], :]

    liq = np.full(amt.shape, np.nan, dtype=np.float32)
    for p in range(per["nper"]):
        idx = np.where(valid[p] & ok[3][p] & np.isfinite(amt[p]) &
                       ~is_st[p] & (bars[p] >= 250) & (price[p] > 2))[0]
        if len(idx) >= 100:
            liq[p, idx] = (rankdata(amt[p, idx]) - 0.5) / len(idx)

    rows = []
    for H in [3, 12]:
        ex = rets[H] - loo[H]
        base = valid & ok[H] & np.isfinite(ex)
        m = base & (trend == 1) & ~is_st & (bars >= 250) & (price > 2) & (liq >= 0.5)
        for name, feat in [("柱体水平", hist_pct), ("DIF水平", dif_pct)]:
            rec = {0: [], 4: []}
            for p in range(per["nper"]):
                idx = np.where(m[p] & np.isfinite(rets[H][p]) & np.isfinite(feat[p]))[0]
                if len(idx) < 100:
                    continue
                q = np.minimum(((rankdata(feat[p, idx]) - 1) * 5 / len(idx)).astype(np.int16), 4)
                for k in [0, 4]:
                    sel = idx[q == k]
                    rec[k].append((rets[H][p, sel], ex[p, sel]))
            for k in [0, 4]:
                if not rec[k]:
                    continue
                f = np.concatenate([a for a, b in rec[k]])
                e = np.concatenate([b for a, b in rec[k]])
                rows.append(dict(周期=H, 特征=name, 分位=f"Q{k+1}", 样本数=len(f),
                                 均值绝对=f.mean(), 中位绝对=np.median(f),
                                 胜率绝对=(f > 0).mean(), P10=np.percentile(f, 10),
                                 P90=np.percentile(f, 90), 均值超额=e.mean(),
                                 中位超额=np.median(e)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "月度下行_分布.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
