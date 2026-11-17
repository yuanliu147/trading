#!/usr/bin/env python3
"""41 科创板例外 + 行业净超额表"""
import os

import numpy as np
import pandas as pd
from scipy.special import erfc

import stock_core as S
from stats_utils import _bh

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


def main():
    z = np.load(os.path.join(RES, "个股_日频.npz"), allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    codes = z["codes"].astype(str)
    r = z["r_oo"].astype(np.float64)
    pos = {k: z[k].astype(np.float32) for k in ["pos_dif", "pos_ma60"]}
    inv = z["universe_inv"]
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
        return np.nansum(np.where(mask[sl], ret, 0), 1) / np.maximum(n, 1)

    def met(mask, p, lo, hi):
        sl = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        s = port(mask, p, sl)
        e = ew(mask & inv, sl)
        expo = np.nansum(np.where(mask[sl], p[sl], 0)) / max(np.nansum(mask[sl]), 1)
        b = expo * e
        return dict(年化净=ann(s), 年化同暴露=ann(b), 净超同暴露=ann(s) - ann(b),
                    年化持有=ann(e), 净超持有=ann(s) - ann(e),
                    夏普净=sharpe(s), 夏普同暴露=sharpe(b),
                    回撤净=mdd(s), 回撤同暴露=mdd(b), 暴露=expo)

    rows = []
    for name, mask in [("科创板", star), ("创业板", cyb), ("主板", mainb)]:
        m = inv & mask
        for lab, lo, hi in [("2019-2021", "2019-07-01", "2021-12-31"),
                            ("2022-2026", "2022-01-01", "2026-12-31"),
                            ("2019-2026", "2019-07-01", "2026-12-31")]:
            d = met(m, pos["pos_dif"], lo, hi)
            d.update(板块=name, 区间=lab, 策略="DIF>DEA")
            rows.append(d)
        for lab, lo, hi in [("2019-2021", "2019-07-01", "2021-12-31"),
                            ("2022-2026", "2022-01-01", "2026-12-31"),
                            ("2019-2026", "2019-07-01", "2026-12-31")]:
            d = met(m, pos["pos_ma60"], lo, hi)
            d.update(板块=name, 区间=lab, 策略="价格>MA60")
            rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "最终_个股适配_科创板.csv"), index=False, encoding="utf-8-sig")

    # 科创板逐年
    rows = []
    m = inv & star
    for y in range(2019, 2027):
        sl = (dates >= pd.Timestamp(f"{y}-01-01")) & (dates <= pd.Timestamp(f"{y}-12-31"))
        if sl.sum() < 50:
            continue
        d = met(m, pos["pos_dif"], f"{y}-01-01", f"{y}-12-31")
        d.update(年份=y)
        rows.append(d)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "最终_个股适配_科创板逐年.csv"), index=False, encoding="utf-8-sig")

    # 科创板平移零假
    sl = (dates >= pd.Timestamp("2022-01-01")) & (dates <= pd.Timestamp("2026-12-31"))
    e = ew(m & inv, sl)
    expo = np.nansum(np.where(m[sl], pos["pos_dif"][sl], 0)) / max(np.nansum(m[sl]), 1)
    b = expo * e
    actual = ann(port(m, pos["pos_dif"], sl)) - ann(b)
    rng = np.random.default_rng(20261029)
    nd = int(sl.sum())
    vals = []
    for _ in range(200):
        k = int(rng.integers(50, nd - 50))
        ps = np.roll(pos["pos_dif"], k, axis=0)
        vals.append(ann(port(m, ps, sl)) - ann(b))
    vals = np.array(vals)
    pd.DataFrame([dict(对象="科创板 DIF>DEA 2022-2026", 实际=actual, 零假均值=vals.mean(),
                       零假P5=np.percentile(vals, 5), 零假P95=np.percentile(vals, 95),
                       p值=(1 + (vals >= actual).sum()) / (len(vals) + 1), 次数=len(vals))]).to_csv(
        os.path.join(RES, "最终_个股适配_科创板零假.csv"), index=False, encoding="utf-8-sig")

    # 行业净超额 + t + BH
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    tm = pd.read_csv(os.path.join(RES, "个股_择时表现_分期.csv"))
    d = st.merge(tm, on="code")
    rows = []
    for lab in ["前期", "后期"]:
        sub = d[d["期间"] == lab]
        for ind, s in sub.groupby("行业"):
            if len(s) < 10:
                continue
            x = s["净超同暴露"].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(dict(期间=lab, 行业=ind, 股票数=len(s), 净超同暴露=s["净超同暴露"].mean(),
                             毛超同暴露=s["毛超同暴露"].mean(), 夏普差=s["夏普差"].mean(),
                             t=x.mean() / x.std() * np.sqrt(len(x)) if x.std() > 0 else np.nan))
    ind = pd.DataFrame(rows)
    ind["q值"] = np.nan
    for lab, idx in ind.groupby("期间").groups.items():
        ind.loc[idx, "q值"] = _bh(ind.loc[idx, "t"].apply(
            lambda t: erfc(abs(t) / np.sqrt(2)) if np.isfinite(t) else np.nan).values)
    piv = ind.pivot(index="行业", columns="期间", values="净超同暴露")
    if "前期" in piv and "后期" in piv:
        piv["两期同为正"] = (piv["前期"] > 0) & (piv["后期"] > 0)
    piv.reset_index().to_csv(os.path.join(RES, "最终_个股适配_行业_择时.csv"), index=False, encoding="utf-8-sig")
    ind.to_csv(os.path.join(RES, "最终_个股适配_行业_择时_明细.csv"), index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    print(pd.read_csv(os.path.join(RES, "最终_个股适配_科创板零假.csv")).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
