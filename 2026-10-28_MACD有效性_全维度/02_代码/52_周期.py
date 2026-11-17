#!/usr/bin/env python3
"""52 周期配合：日线MACD×周/月KDJ；周线MACD×周线KDJ；月线MACD×月线KDJ。"""
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
MIN_N = 30


def dm_series(sel, ex, min_n=MIN_N):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def stat(sel, ex, label, lag=20):
    dm, n = dm_series(sel, ex)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(类别=label, n=int(sel.sum()), days=int(ok.sum()), mean=np.nan, t=np.nan)
    return dict(类别=label, n=int(sel.sum()), days=int(ok.sum()),
                mean=float(np.nanmean(dm)), t=float(nw_t(dm, lag)),
                early=float(np.nanmean(dm[ok & (np.arange(len(dm)) < len(dm) * 0.55)])),
                late=float(np.nanmean(dm[ok & (np.arange(len(dm)) >= len(dm) * 0.55)])))


def contrast(a, b, ex, lag=20):
    da, _ = dm_series(a, ex)
    db, _ = dm_series(b, ex)
    common = np.isfinite(da) & np.isfinite(db)
    if common.sum() < 20:
        return dict(A=np.nan, B=np.nan, 差值=np.nan, t=np.nan, 共同日=int(common.sum()))
    d = da - db
    return dict(A=float(np.nanmean(da[common])), B=float(np.nanmean(db[common])),
                差值=float(np.nanmean(d[common])), t=float(nw_t(d[common], lag)),
                共同日=int(common.sum()))


def did(mb, ms, cond, ex, lag=20):
    a, b = mb & cond, mb & ~cond
    c, d = ms & cond, ms & ~cond
    sa, _ = dm_series(a, ex); sb, _ = dm_series(b, ex)
    sc, _ = dm_series(c, ex); sd, _ = dm_series(d, ex)
    common = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(sc) & np.isfinite(sd)
    if common.sum() < 20:
        return dict(A=np.nan, B=np.nan, 差值=np.nan, t=np.nan, 共同日=int(common.sum()))
    x = (sa - sb) - (sc - sd)
    return dict(A=float(np.nanmean((sa - sb)[common])), B=float(np.nanmean((sc - sd)[common])),
                差值=float(np.nanmean(x[common])), t=float(nw_t(x[common], lag)),
                共同日=int(common.sum()))


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    wk = K.period_kdj_state(P, "W")
    mk = K.period_kdj_state(P, "M")
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex20 = (fwd[20] - loo[20]).astype(np.float32)
    m20 = base & ok[20] & np.isfinite(ex20)
    mb = macd["states"]["DIF>DEA"] & m20
    ms = ~macd["states"]["DIF>DEA"] & m20
    gold, _ = M.crosses(macd["dif"], macd["dea"], P)

    # ---------- A. 日线 MACD × 周/月 KDJ ----------
    rows = []
    for fam, w in [("周线KDJ", wk), ("月线KDJ", mk)]:
        for cond, cl in [(w["state"].astype(bool), "多"), (~w["state"].astype(bool) & w["ok"], "空")]:
            for side, sel in [("MACD多", mb), ("MACD空", ms)]:
                cell = sel & cond & m20
                r = stat(cell, ex20, f"{side}+{fam}{cl}")
                rows.append(dict(维度=fam, 方向=side, 状态=cl, **r))
        # DiD
        d = did(mb, ms, w["state"].astype(bool) & w["ok"], ex20)
        rows.append(dict(维度=fam, 方向="DiD", 状态="多vs空", **d))
    # 事件：MACD金叉 × 周/月 KDJ
    for fam, w in [("周线KDJ", wk), ("月线KDJ", mk)]:
        g = gold & m20
        for cond, cl in [(w["state"].astype(bool), "多"), (~w["state"].astype(bool) & w["ok"], "空")]:
            r = stat(g & cond, ex20, f"MACD金叉+{fam}{cl}")
            rows.append(dict(维度=fam, 方向="MACD金叉", 状态=cl, **r))
        r = contrast(g & w["state"].astype(bool), g & ~w["state"].astype(bool) & w["ok"], ex20)
        rows.append(dict(维度=fam, 方向="MACD金叉增量", 状态="多-空", **r))
        # 周/月 KDJ 极值
        for lo, hi, lab in [(80, 200, "K>80"), (0, 20, "K<20")]:
            c = (w["k"] > lo) & (w["k"] <= hi) & w["ok"]
            r = stat(g & c, ex20, f"MACD金叉+{fam}{lab}")
            rows.append(dict(维度=fam, 方向="MACD金叉", 状态=lab, **r))
            r2 = contrast(g & c, g & ~c & w["ok"], ex20)
            rows.append(dict(维度=fam, 方向="MACD金叉增量", 状态=lab, **r2))
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(RES, "KDJ_周期_日线x周月.csv"), index=False, encoding="utf-8-sig")
    print("A done")
    print(A.to_string(index=False))

    # ---------- B/C. 同周期 ----------
    all_rows = []
    for freq, lab, H, lag in [("W", "周线", 4, 4), ("M", "月线", 3, 3)]:
        per = K.period_ohlc(P, freq)
        h = M._fill_lead(per["high"]); l = M._fill_lead(per["low"]); c = M._fill_lead(per["close"])
        # 周期 MACD
        pdif = M.ema_filled(c, 12) - M.ema_filled(c, 26)
        pdea = M.ema_filled(pdif, 9)
        pvalid = ~np.isnan(per["close"])
        pbars = np.cumsum(pvalid, axis=0)
        pgold = pvalid & (pbars >= (60 if freq == "W" else 36)) & (pdif > pdea) & (M.shift(pdif, 1) <= M.shift(pdea, 1))
        pstate = (pdif > pdea) & pvalid
        # 周期 KDJ
        pk, pdk, pj, _ = K.kdj(P, 9, 3, 3, high=h, low=l, close=c)
        pkgold = pvalid & (pbars >= (26 if freq == "W" else 12)) & (pk > pdk) & (M.shift(pk, 1) <= M.shift(pdk, 1))
        pkstate = (pk > pdk) & pvalid
        # 周期收益
        import importlib
        P11 = importlib.import_module("11_周期研究")
        rets, oks, mkts, loos, pcts = P11.period_returns(P, per, [H])
        pex = (rets[H] - loos[H]).astype(np.float32)
        pm = oks[H] & np.isfinite(pex) & pvalid
        pmb = pstate & pm
        pms = ~pstate & pm
        rows = []
        for cond, cl in [(pkstate, "KDJ多"), (~pkstate & pvalid, "KDJ空")]:
            for side, sel in [("MACD多", pmb), ("MACD空", pms)]:
                r = stat(sel & cond, pex, f"{side}+{cl}", lag=lag)
                rows.append(dict(周期=lab, 维度="状态", 方向=side, 状态=cl, **r))
        d = did(pmb, pms, pkstate, pex, lag=lag)
        rows.append(dict(周期=lab, 维度="状态", 方向="DiD", 状态="多vs空", **d))
        # 事件：MACD 金叉 × KDJ；KDJ 金叉 × MACD
        g = pgold & pm
        for cond, cl in [(pkstate, "KDJ多"), (~pkstate & pvalid, "KDJ空"), (pkgold, "同日KDJ金叉")]:
            r = stat(g & cond, pex, f"MACD金叉+{cl}", lag=lag)
            rows.append(dict(周期=lab, 维度="MACD金叉", 方向="过滤", 状态=cl, **r))
        r = contrast(g & pkstate, g & ~pkstate & pvalid, pex, lag=lag)
        rows.append(dict(周期=lab, 维度="MACD金叉", 方向="增量", 状态="KDJ多-空", **r))
        kg = pkgold & pm
        for cond, cl in [(pstate, "MACD多"), (~pstate & pvalid, "MACD空")]:
            r = stat(kg & cond, pex, f"KDJ金叉+{cl}", lag=lag)
            rows.append(dict(周期=lab, 维度="KDJ金叉", 方向="过滤", 状态=cl, **r))
        r = contrast(kg & pstate, kg & ~pstate & pvalid, pex, lag=lag)
        rows.append(dict(周期=lab, 维度="KDJ金叉", 方向="增量", 状态="MACD多-空", **r))
        all_rows.append(pd.DataFrame(rows))
        print(lab, "done", time.time() - t0)
    BC = pd.concat(all_rows, ignore_index=True)
    BC.to_csv(os.path.join(RES, "KDJ_周期_同周期.csv"), index=False, encoding="utf-8-sig")
    print(BC.to_string(index=False))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
