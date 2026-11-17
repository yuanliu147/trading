#!/usr/bin/env python3
"""49c 关键格：逐年、市场状态、2x2 交互（DiD）、相对 MA60/单指标增量。"""
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


def daily_sel(sel, ex, min_n=50):
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def series(sel, ex):
    dm, _ = daily_sel(sel, ex)
    return dm


def nwt(x, lag=20):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return nw_t(x, lag) if len(x) >= 20 else np.nan


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    fwd, ok, mkt, loo = M.forward_returns(P, (5, 20, 60))
    exs = {h: (fwd[h] - loo[h]).astype(np.float32) for h in (5, 20, 60)}
    masks = {h: base & ok[h] & np.isfinite(exs[h]) for h in (5, 20, 60)}
    ex20 = exs[20]
    m20 = masks[20]
    mb = macd["states"]["DIF>DEA"] & m20
    ms = ~macd["states"]["DIF>DEA"] & m20
    k, d, j = kd["k"], kd["d"], kd["j"]
    ma60 = st["ma60"]
    ma60b = m20 & (P["close_ff"] > ma60)
    ma60s = m20 & (P["close_ff"] <= ma60)

    conds = {
        "K<20": k < 20, "K<30": k < 30, "D<20": d < 20, "D<30": d < 30,
        "J<0": j < 0, "J<-20": j < -20,
        "K>70": k > 70, "K>80": k > 80, "J>100": j > 100,
        "K>D": k > d, "K<D": k < d,
        "5日内KDJ金叉": (np.cumsum(kd["gold"], axis=0) - M.shift(np.cumsum(kd["gold"], axis=0), 5, fill=0)) > 0,
        "KDJ顶背离": (P["close_ff"] >= M.rolling_max(P["close_ff"], 20) - 1e-6) &
                    (k < M.rolling_max(k, 20) - 1e-9),
        "KDJ底背离": (P["close_ff"] <= M.rolling_min(P["close_ff"], 20) + 1e-6) &
                    (k > M.rolling_min(k, 20) + 1e-9),
    }

    # ---------- 1. 关键格逐年 ----------
    rows = []
    for lab, cond in conds.items():
        for side, sel in [("MACD多", mb), ("MACD空", ms)]:
            cell = sel & cond
            if cell.sum() < 3000:
                continue
            dm = series(cell, ex20)
            for y in range(2010, 2027):
                ym = (dates.year == y)
                if (cell & ym[:, None]).sum() < 200:
                    continue
                x = dm[ym & np.isfinite(dm)]
                if len(x) < 20:
                    continue
                rows.append(dict(条件=lab, 方向=side, 年份=y, 股票日=int((cell & ym[:, None]).sum()),
                                 交易日=int(len(x)), 日均超额=float(np.mean(x)),
                                 t值=float(nw_t(x, 5))))
    yr = pd.DataFrame(rows)
    yr.to_csv(os.path.join(RES, "探索_关键格逐年.csv"), index=False, encoding="utf-8-sig")
    print("year table", yr.shape)
    for lab in ["K<20", "D<20", "K>70", "K>80", "K>D", "K<D"]:
        for side in ["MACD多", "MACD空"]:
            x = yr[(yr.条件 == lab) & (yr.方向 == side)]
            if len(x) == 0:
                continue
            print(f"\n{lab} {side}: years={len(x)} mean={x.日均超额.mean()*100:+.3f}% "
                  f"pos={int((x.日均超额>0).sum())}/{len(x)} "
                  f"min={x.日均超额.min()*100:+.2f}% max={x.日均超额.max()*100:+.2f}%")
            print(x[["年份", "股票日", "日均超额", "t值"]].to_string(index=False))

    # ---------- 2. 市场状态 ----------
    mkt_state = st["market"][:, 0]
    print("\nmarket state counts", {i: int((mkt_state == i).sum()) for i in range(3)})
    mrows = []
    for lab, cond in [("K<20", k < 20), ("D<20", d < 20), ("K>70", k > 70), ("K>80", k > 80),
                      ("K>D", k > d), ("K<D", k < d)]:
        for side, sel in [("MACD多", mb), ("MACD空", ms)]:
            for ms_, ml in [(0, "指数上行"), (1, "指数下行"), (2, "指数震荡")]:
                cell = sel & cond & (mkt_state == ms_)[:, None] & m20
                if cell.sum() < 1000:
                    continue
                dm = series(cell, ex20)
                x = dm[np.isfinite(dm)]
                if len(x) < 20:
                    continue
                mrows.append(dict(条件=lab, 方向=side, 市场=ml, 股票日=int(cell.sum()),
                                  交易日=len(x), 日均超额=float(x.mean()), t值=float(nw_t(x, 20))))
    mk = pd.DataFrame(mrows)
    mk.to_csv(os.path.join(RES, "探索_关键格市场.csv"), index=False, encoding="utf-8-sig")
    print("\nmarket table", mk.shape)
    print(mk.to_string(index=False))

    # ---------- 3. 2x2 交互 DiD ----------
    # 对每个 KDJ 条件：A=M+ & cond, B=M+ & ~cond, C=M- & cond, D=M- & ~cond
    # DiD = (A-B) - (C-D)，按共同交易日配对；t 用 NW(20)
    did_rows = []
    for lab, cond in conds.items():
        A = mb & cond
        B = mb & ~cond
        C = ms & cond
        D = ms & ~cond
        if min(A.sum(), B.sum(), C.sum(), D.sum()) < 3000:
            continue
        sa, sb, sc, sd = series(A, ex20), series(B, ex20), series(C, ex20), series(D, ex20)
        common = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(sc) & np.isfinite(sd)
        if common.sum() < 100:
            continue
        did = (sa - sb) - (sc - sd)
        # 同时给 KDJ 条件在 MACD 多/空内部的增量
        inc_bull = sa - sb
        inc_bear = sc - sd
        did_rows.append(dict(
            条件=lab,
            M多_条件=float(np.nanmean(sa[common])), M多_无条件=float(np.nanmean(sb[common])),
            M空_条件=float(np.nanmean(sc[common])), M空_无条件=float(np.nanmean(sd[common])),
            MACD多内增量=float(np.nanmean(inc_bull[common])), 多内t=float(nw_t(inc_bull[common], 20)),
            MACD空内增量=float(np.nanmean(inc_bear[common])), 空内t=float(nw_t(inc_bear[common], 20)),
            DiD=float(np.nanmean(did[common])), DiD_t=float(nw_t(did[common], 20)),
            共同日=int(common.sum()),
            A股日=int(A.sum()), B股日=int(B.sum()), C股日=int(C.sum()), D股日=int(D.sum()),
        ))
    dd = pd.DataFrame(did_rows).sort_values("DiD", ascending=False)
    dd.to_csv(os.path.join(RES, "探索_KDJ交互DiD.csv"), index=False, encoding="utf-8-sig")
    print("\nDiD table")
    print(dd.to_string(index=False))

    # ---------- 4. 关键格 vs MA60 / 单指标 ----------
    # 对每个 cell：与同日 MA60 多头、MACD 多、KDJ 条件整体比较
    vrows = []
    for lab, cond in conds.items():
        for side, sel, ma_ref in [("MACD多", mb, ma60b), ("MACD空", ms, ma60s)]:
            cell = sel & cond
            if cell.sum() < 3000:
                continue
            sc = series(cell, ex20)
            # 对照 1：同方向 MA60
            sr = series(ma_ref, ex20)
            # 对照 2：同方向单指标（MACD 多/空 全体）
            ss = series(sel, ex20)
            # 对照 3：KDJ 条件整体（不分 MACD）
            sk = series(cond & m20, ex20)
            for nm, ref in [("vsMA60", sr), ("vsMACD单指标", ss), ("vsKDJ单指标", sk)]:
                common = np.isfinite(sc) & np.isfinite(ref)
                if common.sum() < 100:
                    continue
                diff = sc - ref
                vrows.append(dict(条件=lab, 方向=side, 对照=nm,
                                  格子=float(np.nanmean(sc[common])), 对照值=float(np.nanmean(ref[common])),
                                  差值=float(np.nanmean(diff[common])), t值=float(nw_t(diff[common], 20)),
                                  共同日=int(common.sum()), 股票日=int(cell.sum())))
    vv = pd.DataFrame(vrows)
    vv.to_csv(os.path.join(RES, "探索_关键格增量.csv"), index=False, encoding="utf-8-sig")
    print("\nincrement table", vv.shape)
    print(vv[(vv.条件.isin(["K<20", "D<20", "K>70", "K>80"]))].to_string(index=False))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
