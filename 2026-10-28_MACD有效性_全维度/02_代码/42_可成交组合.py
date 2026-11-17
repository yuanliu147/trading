#!/usr/bin/env python3
"""42 可成交组合：剔除一字涨停买不进 / 一字跌停卖不出后的组合结果

这是组合层的最终口径。逐股统计仍用未过滤版本（仅作异质性描述），
决策以本表为准。
"""
import os

import numpy as np
import pandas as pd

import stock_core as S

RES = S.RES
COST = S.COST_SIDE


def ann(x):
    eq = np.cumprod(1 + x)
    return eq[-1] ** (252 / len(x)) - 1 if len(x) and eq[-1] > 0 else np.nan


def mdd(x):
    eq = np.cumprod(1 + x)
    return (eq / np.maximum.accumulate(eq) - 1).min()


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan


def executable(pos, entry, exitb):
    pe = np.zeros_like(pos, dtype=np.float32)
    for t in range(pos.shape[0]):
        cur = pe[t - 1] if t > 0 else np.zeros(pos.shape[1], dtype=np.float32)
        buy = (pos[t] > 0.5) & (cur < 0.5) & ~entry[t]
        sell = (pos[t] < 0.5) & (cur > 0.5) & ~exitb[t]
        pe[t] = np.where(buy, 1.0, np.where(sell, 0.0, cur))
    return pe


def main():
    P = S.load_base()
    dates = P["dates"]
    codes = P["codes"].astype(str)
    z = np.load(os.path.join(RES, "个股_日频.npz"), allow_pickle=True)
    r = z["r_oo"].astype(np.float64)
    inv = z["universe_inv"]
    pos = {k: z[k].astype(np.float32) for k in ["pos_dif", "pos_ma60"]}
    entry = (P["raw_high"] == P["raw_low"]) & (P["raw_open"] > P["preclose"]) & (P["trade"] == 1)
    exitb = (P["raw_high"] == P["raw_low"]) & (P["raw_open"] < P["preclose"]) & (P["trade"] == 1)
    pe = {k: executable(v, entry, exitb) for k, v in pos.items()}

    star = pd.Series(codes).str.startswith(("688", "689")).to_numpy()
    cyb = pd.Series(codes).str.startswith(("300", "301", "302")).to_numpy()
    mainb = ~(star | cyb)

    def ew(mask, sl):
        m = mask[sl]
        n = m.sum(1)
        return np.where(n > 0, np.nansum(np.where(m, r[sl], 0), 1) / np.maximum(n, 1), 0)

    def port(mask, p, sl):
        pp = np.where(mask[sl], p[sl], 0)
        ret = pp * r[sl]
        d = np.abs(np.diff(pp, axis=0, prepend=np.zeros((1, pp.shape[1]))))
        ret = ret - COST * d
        n = mask[sl].sum(1)
        return np.nansum(np.where(mask[sl], ret, 0), 1) / np.maximum(n, 1), d

    def met(mask, p, lo, hi):
        sl = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        s, d = port(mask, p, sl)
        e = ew(mask & inv, sl)
        expo = np.nansum(np.where(mask[sl], p[sl], 0)) / max(np.nansum(mask[sl]), 1)
        b = expo * e
        return dict(年化净=ann(s), 年化同暴露=ann(b), 净超同暴露=ann(s) - ann(b),
                    年化持有=ann(e), 净超持有=ann(s) - ann(e),
                    夏普净=sharpe(s), 夏普同暴露=sharpe(b),
                    回撤净=mdd(s), 回撤同暴露=mdd(b), 暴露=expo,
                    年换手=np.nansum(d[mask[sl]]) / max(np.nansum(mask[sl]), 1) * 252)

    rows = []
    tests = [("全市场", inv, "pos_dif", "DIF>DEA"),
             ("全市场", inv, "pos_ma60", "价格>MA60"),
             ("科创板", inv & star, "pos_dif", "DIF>DEA"),
             ("科创板", inv & star, "pos_ma60", "价格>MA60"),
             ("创业板", inv & cyb, "pos_dif", "DIF>DEA"),
             ("主板", inv & mainb, "pos_dif", "DIF>DEA")]
    for name, mask, key, strat in tests:
        for lab, lo, hi in [("2010-2026", "2010-01-01", "2026-12-31"),
                            ("2010-2017", "2010-01-01", "2017-12-31"),
                            ("2018-2026", "2018-01-01", "2026-12-31"),
                            ("2019-2021", "2019-07-01", "2021-12-31"),
                            ("2022-2026", "2022-01-01", "2026-12-31")]:
            if name != "全市场" and lab in ("2010-2026", "2010-2017", "2018-2026") and name == "科创板":
                continue
            d = met(mask, pe[key], lo, hi)
            d.update(分组=name, 策略=strat, 区间=lab)
            rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "最终_个股适配_组合_可成交.csv"), index=False, encoding="utf-8-sig")

    # 平移零假（可成交口径）
    null_rows = []
    rng = np.random.default_rng(20261029)
    tests2 = [("全市场", inv, "pos_dif", "2010-2026", "2010-01-01", "2026-12-31"),
              ("全市场", inv, "pos_dif", "2010-2017", "2010-01-01", "2017-12-31"),
              ("全市场", inv, "pos_dif", "2018-2026", "2018-01-01", "2026-12-31"),
              ("科创板", inv & star, "pos_dif", "2022-2026", "2022-01-01", "2026-12-31")]
    for name, mask, key, lab, lo, hi in tests2:
        sl = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        e = ew(mask & inv, sl)
        expo = np.nansum(np.where(mask[sl], pe[key][sl], 0)) / max(np.nansum(mask[sl]), 1)
        b = expo * e
        actual = ann(port(mask, pe[key], sl)[0]) - ann(b)
        nd = int(sl.sum())
        vals = []
        for _ in range(100):
            k = int(rng.integers(50, nd - 50))
            ps = np.roll(pe[key], k, axis=0)
            vals.append(ann(port(mask, ps, sl)[0]) - ann(b))
        vals = np.array(vals)
        null_rows.append(dict(对象=name, 策略=key, 区间=lab, 实际=actual, 零假均值=vals.mean(),
                              零假P5=np.percentile(vals, 5), 零假P95=np.percentile(vals, 95),
                              p值=(1 + (vals >= actual).sum()) / (len(vals) + 1), 次数=len(vals)))
    pd.DataFrame(null_rows).to_csv(os.path.join(RES, "最终_个股适配_零假_可成交.csv"),
                                   index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(null_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
