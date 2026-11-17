#!/usr/bin/env python3
"""54 正式状态表：MACD×KDJ 四态、KDJ 条件的绝对/MA60/KDJ 三重增量与综合判定。"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import kdj_core as K
from stats_utils import nw_t, _bh

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
COST = 0.0015
EARLY_END = pd.Timestamp("2017-12-31")
LATE_START = pd.Timestamp("2018-01-01")
MIN_N = 3000


def dm_series(sel, ex, min_n=50):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def cell(sel, ex, dates):
    n = int(sel.sum())
    dm, _ = dm_series(sel, ex)
    ok = np.isfinite(dm)
    if n < MIN_N or ok.sum() < 150:
        return dict(股票日=n, 交易日=int(ok.sum()), 日均超额=np.nan, 中位=np.nan,
                    t值=np.nan, 正日占比=np.nan, 前期=np.nan, 后期=np.nan,
                    年数=0, 正年数=0, 有效年数=0, 有效正年数=0)
    early = ok & (dates <= EARLY_END)
    late = ok & (dates >= LATE_START)
    yr = pd.Series(dm[ok]).groupby(pd.Series(dates[ok]).dt.year).agg(["mean", "count"])
    eff = yr[yr["count"] >= 20]
    return dict(股票日=n, 交易日=int(ok.sum()), 日均超额=float(np.nanmean(dm)),
                中位=float(np.nanmedian(ex[sel])), t值=float(nw_t(dm, 20)),
                正日占比=float(np.nanmean(dm[ok] > 0)),
                前期=float(np.nanmean(dm[early])) if early.any() else np.nan,
                后期=float(np.nanmean(dm[late])) if late.any() else np.nan,
                年数=int(yr.shape[0]), 正年数=int((yr["mean"] > 0).sum()),
                有效年数=int(eff.shape[0]), 有效正年数=int((eff["mean"] > 0).sum()))


def spread(a, b, ex):
    da, _ = dm_series(a, ex)
    db, _ = dm_series(b, ex)
    common = np.isfinite(da) & np.isfinite(db)
    if common.sum() < 100:
        return dict(均值=np.nan, t=np.nan, 共同日=int(common.sum()))
    x = da - db
    return dict(均值=float(np.nanmean(x[common])), t=float(nw_t(x[common], 20)),
                共同日=int(common.sum()))


def stable(r, long=True):
    if r["有效年数"] < 6:
        return False
    pos_share = r["有效正年数"] / max(r["有效年数"], 1)
    if long:
        return pos_share >= 0.6 and r["中位"] > 0
    return pos_share <= 0.4 and r["中位"] < 0


def long_verdict(r):
    if r["股票日"] < MIN_N or r["交易日"] < 150:
        return "样本不足"
    if r["日均超额"] > COST and r["t值"] > 2 and r["前期"] > 0 and r["后期"] > 0:
        return "稳健有效" if stable(r, True) else "均值有效(不稳)"
    if r["日均超额"] < -COST and r["t值"] < -2 and r["前期"] < 0 and r["后期"] < 0:
        return "稳健反向" if stable(r, False) else "均值反向(不稳)"
    return "无效"


def short_verdict(r):
    if r["股票日"] < MIN_N or r["交易日"] < 150:
        return "样本不足"
    if r["日均超额"] < -COST and r["t值"] < -2 and r["前期"] < 0 and r["后期"] < 0:
        return "稳健回避" if stable(r, False) else "均值回避(不稳)"
    if r["日均超额"] > COST and r["t值"] > 2 and r["前期"] > 0 and r["后期"] > 0:
        return "稳健反向(该涨)" if stable(r, True) else "均值反向(不稳)"
    return "无效"


def inc_verdict(v, t):
    if not np.isfinite(v) or not np.isfinite(t):
        return "不可比"
    if v > 0 and t > 2:
        return "有增量"
    if v < 0 and t < -2:
        return "负增量"
    return "无显著增量"


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    wk = K.period_kdj_state(P, "W")
    mk = K.period_kdj_state(P, "M")
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = (fwd[20] - loo[20]).astype(np.float32)
    mask = base & ok[20] & np.isfinite(ex) & (dates >= pd.Timestamp("2010-01-01"))[:, None]
    bull = mask & macd["states"]["DIF>DEA"]
    bear = mask & ~macd["states"]["DIF>DEA"]
    ma60 = st["ma60"]
    ma60b = mask & (P["close_ff"] > ma60)
    ma60s = mask & (P["close_ff"] <= ma60)
    k, d, j = kd["k"], kd["d"], kd["j"]
    qk = F.row_quantile(k, base, 5)
    qgap = F.row_quantile(k - d, base, 5)
    qj = F.row_quantile(j, base, 5)
    age = np.abs(kd["age"])

    # ---------- 四态 ----------
    rows = []
    for kl, kcond in [("KDJ多", k > d), ("KDJ空", k <= d)]:
        for ml, mcond in [("MACD多", macd["states"]["DIF>DEA"]), ("MACD空", ~macd["states"]["DIF>DEA"])]:
            sel = mask & kcond & mcond
            r = cell(sel, ex, dates)
            rows.append(dict(KDJ=kl, MACD=ml, **r, 判定=long_verdict(r)))
    four = pd.DataFrame(rows)
    four.to_csv(os.path.join(RES, "KDJ_状态_四态.csv"), index=False, encoding="utf-8-sig")
    print("四态")
    print(four.to_string(index=False))

    # DiD
    A = mask & (k > d) & macd["states"]["DIF>DEA"]
    B = mask & (k <= d) & macd["states"]["DIF>DEA"]
    C = mask & (k > d) & ~macd["states"]["DIF>DEA"]
    D = mask & (k <= d) & ~macd["states"]["DIF>DEA"]
    did = spread(A, B, ex)
    did2 = spread(C, D, ex)
    da, _ = dm_series(A, ex); db, _ = dm_series(B, ex)
    dc, _ = dm_series(C, ex); dd, _ = dm_series(D, ex)
    common = np.isfinite(da) & np.isfinite(db) & np.isfinite(dc) & np.isfinite(dd)
    x = (da - db) - (dc - dd)
    print("DiD K>D:", np.nanmean(x[common]), nw_t(x[common], 20), int(common.sum()))
    pd.DataFrame([dict(对比="MACD多内 KDJ多-空", **did),
                  dict(对比="MACD空内 KDJ多-空", **did2),
                  dict(对比="DiD", 均值=float(np.nanmean(x[common])), t=float(nw_t(x[common], 20)),
                       共同日=int(common.sum()))]).to_csv(
        os.path.join(RES, "KDJ_状态_四态DiD.csv"), index=False, encoding="utf-8-sig")

    # ---------- 条件表 ----------
    def recent(ev, n):
        cs = np.cumsum(ev, axis=0)
        return (cs - M.shift(cs, n, fill=0)) > 0
    high20 = M.rolling_max(P["close_ff"], 20)
    low20 = M.rolling_min(P["close_ff"], 20)
    top_div = mask & (P["close_ff"] >= high20 - 1e-6) & (k < M.rolling_max(k, 20) - 1e-9)
    bot_div = mask & (P["close_ff"] <= low20 + 1e-6) & (k > M.rolling_min(k, 20) + 1e-9)

    conds = []
    def add(fam, lab, m):
        conds.append((fam, lab, m & mask))
    add("KDJ方向", "K>D", k > d)
    add("KDJ方向", "K<=D", k <= d)
    add("KDJ水平", "K>80", k > 80)
    add("KDJ水平", "K<20", k < 20)
    add("KDJ水平", "K>70", k > 70)
    add("KDJ水平", "K<30", k < 30)
    add("KDJ水平", "K 50-70", (k >= 50) & (k <= 70))
    add("KDJ水平", "K 30-50", (k >= 30) & (k < 50))
    add("J极值", "J>100", j > 100)
    add("J极值", "J<0", j < 0)
    for qi in range(5):
        add("K五分位", f"KQ{qi+1}", qk == qi)
        add("K-D差分位", f"GQ{qi+1}", qgap == qi)
        add("J五分位", f"JQ{qi+1}", qj == qi)
    for lo, hi in [(0, 5), (6, 15), (16, 40), (41, 120), (121, 9999)]:
        add("交叉年龄", f"{lo}-{hi}日", (age >= lo) & (age <= hi))
    for n in (5, 10, 20):
        add("近期金叉", f"{n}日内金叉", recent(kd["gold"], n))
        add("近期死叉", f"{n}日内死叉", recent(kd["death"], n))
    add("KDJ背离", "顶背离", top_div)
    add("KDJ背离", "底背离", bot_div)
    add("KDJ背离", "无背离", mask & ~top_div & ~bot_div)
    for lab, w in [("周线KDJ多", wk["state"]), ("周线KDJ空", ~wk["state"] & wk["ok"]),
                   ("月线KDJ多", mk["state"]), ("月线KDJ空", ~mk["state"] & mk["ok"])]:
        add("周月KDJ", lab, w)

    rows = []
    for fam, lab, cm in conds:
        mb, ms = cm & bull, cm & bear
        lb = cell(mb, ex, dates); ls = cell(ms, ex, dates)
        sp = spread(mb, ms, ex)
        # MA60 增量
        ma_sp = spread(cm & ma60b, cm & ma60s, ex)
        inc = sp["均值"] - ma_sp["均值"] if np.isfinite(sp["均值"]) and np.isfinite(ma_sp["均值"]) else np.nan
        da, _ = dm_series(mb, ex); db, _ = dm_series(ms, ex)
        dma, _ = dm_series(cm & ma60b, ex); dmb, _ = dm_series(cm & ma60s, ex)
        common = np.isfinite(da) & np.isfinite(db) & np.isfinite(dma) & np.isfinite(dmb)
        inc_t = nw_t((da - db - dma + dmb)[common], 20) if common.sum() >= 100 else np.nan
        # KDJ 增量：多/空内部 subset vs 补集
        comp_b = ~cm & mask & macd["states"]["DIF>DEA"]
        comp_s = ~cm & mask & ~macd["states"]["DIF>DEA"]
        kb = spread(mb, comp_b, ex)
        ks = spread(ms, comp_s, ex)
        # DiD
        a1, _ = dm_series(mb, ex); b1, _ = dm_series(comp_b, ex)
        c1, _ = dm_series(ms, ex); d1, _ = dm_series(comp_s, ex)
        cc = np.isfinite(a1) & np.isfinite(b1) & np.isfinite(c1) & np.isfinite(d1)
        didv = float(np.nanmean(((a1 - b1) - (c1 - d1))[cc])) if cc.sum() >= 100 else np.nan
        didt = nw_t(((a1 - b1) - (c1 - d1))[cc], 20) if cc.sum() >= 100 else np.nan
        rows.append(dict(
            条件族=fam, 条件=lab,
            多头日均=lb["日均超额"], 多头中位=lb["中位"], 多头t=lb["t值"], 多头前期=lb["前期"], 多头后期=lb["后期"],
            多头年数=lb["年数"], 多头正年数=lb["正年数"], 多头有效年数=lb["有效年数"], 多头有效正年数=lb["有效正年数"], 多头股票日=lb["股票日"],
            空头日均=ls["日均超额"], 空头中位=ls["中位"], 空头t=ls["t值"], 空头前期=ls["前期"], 空头后期=ls["后期"],
            空头股票日=ls["股票日"],
            多空差=sp["均值"], 多空差t=sp["t"], 共同日=sp["共同日"],
            MA60多空差=ma_sp["均值"], MACD增量=inc, MACD增量t=inc_t,
            KDJ增量_多=kb["均值"], KDJ增量_多t=kb["t"],
            KDJ增量_空=ks["均值"], KDJ增量_空t=ks["t"],
            DiD=didv, DiDt=didt,
            绝对判定=long_verdict(lb), 空头判定=short_verdict(ls),
            MACD增量判定=inc_verdict(inc, inc_t),
            KDJ增量_多判定=inc_verdict(kb["均值"], kb["t"]),
            KDJ增量_空判定=inc_verdict(ks["均值"], ks["t"]),
        ))
        print(f"{fam:8s} {lab:12s} 多{lb['日均超额']*100:+.3f} 空{ls['日均超额']*100:+.3f} "
              f"KDJ多增{kb['均值']*100:+.3f} t{kb['t']:+.1f} 空增{ks['均值']*100:+.3f} t{ks['t']:+.1f} "
              f"DiD{didv*100:+.3f} t{didt:+.1f}")
    out = pd.DataFrame(rows)
    # BH
    p1 = 2 * norm.sf(np.abs(out["多头t"].to_numpy(float)))
    out["多头q"] = _bh(p1)
    p2 = 2 * norm.sf(np.abs(out["MACD增量t"].to_numpy(float)))
    out["MACD增量q"] = _bh(p2)
    p3 = 2 * norm.sf(np.abs(out["KDJ增量_多t"].to_numpy(float)))
    out["KDJ增量_多q"] = _bh(p3)
    p4 = 2 * norm.sf(np.abs(out["DiDt"].to_numpy(float)))
    out["DiDq"] = _bh(p4)
    out.to_csv(os.path.join(RES, "KDJ_状态_条件.csv"), index=False, encoding="utf-8-sig")
    print("saved", out.shape)
    print("\n综合判定")
    print(out.groupby(["绝对判定", "MACD增量判定", "KDJ增量_多判定"]).size().to_string())
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
