#!/usr/bin/env python3
"""49d 网格：MACD 状态 × KDJ 水平五分组；零轴/柱体 × KDJ；逐年交互稳定性。"""
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


def stats(sel, ex, dates):
    n = int(sel.sum())
    dm, _ = daily_sel(sel, ex)
    ok = np.isfinite(dm)
    if n < 500 or ok.sum() < 20:
        return dict(n=n, days=int(ok.sum()), mean=np.nan, t=np.nan,
                    early=np.nan, late=np.nan, ypos=0, yn=0)
    early = ok & (dates <= pd.Timestamp("2017-12-31"))
    late = ok & (dates >= pd.Timestamp("2018-01-01"))
    yr = pd.Series(dm[ok]).groupby(pd.Series(dates[ok]).dt.year).mean()
    return dict(n=n, days=int(ok.sum()), mean=float(np.nanmean(dm)),
                t=float(nw_t(dm, 20)),
                early=float(np.nanmean(dm[early])) if early.any() else np.nan,
                late=float(np.nanmean(dm[late])) if late.any() else np.nan,
                ypos=int((yr > 0).sum()), yn=int(yr.notna().sum()))


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
    mb = macd["states"]["DIF>DEA"]
    k, d, j = kd["k"], kd["d"], kd["j"]
    ma60 = st["ma60"]
    ma60b = P["close_ff"] > ma60
    ma60s = P["close_ff"] <= ma60
    zs = np.full(P["shape"], -1, dtype=np.int8)
    zs[(macd["dif"] > 0) & (macd["dea"] > 0)] = 0
    zs[(macd["dif"] > 0) & (macd["dea"] <= 0)] = 1
    zs[(macd["dif"] <= 0) & (macd["dea"] > 0)] = 2
    zs[(macd["dif"] <= 0) & (macd["dea"] <= 0)] = 3

    qs = {}
    for nm, x in [("K", k), ("D", d), ("J", j), ("K-D差", k - d)]:
        qs[nm] = F.row_quantile(x, base, 5)

    # ---------- 2x5 ----------
    rows = []
    for nm, q in qs.items():
        for side, sel in [("MACD多", mb), ("MACD空", ~mb)]:
            for qi in range(5):
                cell = sel & (q == qi) & m20
                r = stats(cell, ex20, dates)
                rows.append(dict(维度=nm, 方向=side, 分组=f"Q{qi+1}", **r))
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(RES, "探索_2x5网格.csv"), index=False, encoding="utf-8-sig")
    print("2x5")
    print(g.pivot_table(index=["维度", "方向"], columns="分组", values="mean").to_string())

    # ---------- 零轴/柱体 x K五分位 ----------
    rows = []
    for z, zl in [(0, "双上"), (1, "DIF上DEA下"), (2, "DIF下DEA上"), (3, "双下")]:
        for qi in range(5):
            cell = (zs == z) & (qs["K"] == qi) & m20
            r = stats(cell, ex20, dates)
            rows.append(dict(维度="零轴", 状态=zl, 分组=f"Q{qi+1}", **r))
    hs = macd["hist_state"]
    for h, hl in [(0, "红柱放大"), (1, "红柱缩小"), (2, "绿柱缩短"), (3, "绿柱放大")]:
        for qi in range(5):
            cell = (hs == h) & (qs["K"] == qi) & m20
            r = stats(cell, ex20, dates)
            rows.append(dict(维度="柱体", 状态=hl, 分组=f"Q{qi+1}", **r))
    g2 = pd.DataFrame(rows)
    g2.to_csv(os.path.join(RES, "探索_结构x5网格.csv"), index=False, encoding="utf-8-sig")
    print("\n零轴/柱体 x K五分位")
    print(g2.pivot_table(index=["维度", "状态"], columns="分组", values="mean").to_string())

    # ---------- 关键对比：超买惩罚、超卖交互、中性区 ----------
    print("\n关键对比（20日）")
    contrasts = []
    def add_con(lab, a, b, note=""):
        sa, _ = daily_sel(a & m20, ex20)
        sb, _ = daily_sel(b & m20, ex20)
        common = np.isfinite(sa) & np.isfinite(sb)
        if common.sum() < 50:
            return
        diff = sa - sb
        contrasts.append(dict(对比=lab, A均值=float(np.nanmean(sa[common])),
                              B均值=float(np.nanmean(sb[common])),
                              差值=float(np.nanmean(diff[common])),
                              t值=float(nw_t(diff[common], 20)), 共同日=int(common.sum()),
                              A股日=int((a&m20).sum()), B股日=int((b&m20).sum()), 备注=note))
    add_con("MACD多: K>70 vs K<=70", mb & (k > 70), mb & (k <= 70))
    add_con("MACD多: K>80 vs K<=80", mb & (k > 80), mb & (k <= 80))
    add_con("MACD多: K>D vs K<D", mb & (k > d), mb & (k < d))
    add_con("MACD多: K<20 vs K>=20", mb & (k < 20), mb & (k >= 20))
    add_con("MACD多: D<20 vs D>=20", mb & (d < 20), mb & (d >= 20))
    add_con("MACD多: K Q3 vs 其他", mb & (qs["K"] == 2), mb & (qs["K"] != 2))
    add_con("MACD空: K>70 vs K<=70", ~mb & (k > 70), ~mb & (k <= 70))
    add_con("MACD空: K<20 vs K>=20", ~mb & (k < 20), ~mb & (k >= 20))
    add_con("MACD空: K<D vs K>D", ~mb & (k < d), ~mb & (k > d))
    add_con("MACD空: K Q3 vs 其他", ~mb & (qs["K"] == 2), ~mb & (qs["K"] != 2))
    add_con("MACD空: K Q3 vs Q1", ~mb & (qs["K"] == 2), ~mb & (qs["K"] == 0))
    # KDJ 顶/底背离
    top_div = (P["close_ff"] >= M.rolling_max(P["close_ff"], 20) - 1e-6) & (k < M.rolling_max(k, 20) - 1e-9)
    bot_div = (P["close_ff"] <= M.rolling_min(P["close_ff"], 20) + 1e-6) & (k > M.rolling_min(k, 20) + 1e-9)
    add_con("MACD多: 顶背离 vs 无", mb & top_div, mb & ~top_div)
    add_con("MACD多: 底背离 vs 无", mb & bot_div, mb & ~bot_div)
    add_con("MACD空: 顶背离 vs 无", ~mb & top_div, ~mb & ~top_div)
    add_con("MACD空: 底背离 vs 无", ~mb & bot_div, ~mb & ~bot_div)
    co = pd.DataFrame(contrasts).sort_values("t值")
    co.to_csv(os.path.join(RES, "探索_关键对比.csv"), index=False, encoding="utf-8-sig")
    print(co.to_string(index=False))

    # ---------- 逐年交互 ----------
    yr_rows = []
    year = dates.year
    for lab, a, b in [
        ("MACD多: K>70-K<=70", mb & (k > 70), mb & (k <= 70)),
        ("MACD多: K>D-K<D", mb & (k > d), mb & (k < d)),
        ("MACD多: K<20-K>=20", mb & (k < 20), mb & (k >= 20)),
        ("MACD空: K<20-K>=20", ~mb & (k < 20), ~mb & (k >= 20)),
        ("MACD空: K>D-K<D", ~mb & (k > d), ~mb & (k < d)),
        ("MACD空: K Q3-Q1", ~mb & (qs["K"] == 2), ~mb & (qs["K"] == 0)),
    ]:
        sa, _ = daily_sel(a & m20, ex20)
        sb, _ = daily_sel(b & m20, ex20)
        common = np.isfinite(sa) & np.isfinite(sb)
        diff = sa - sb
        for y in range(2010, 2027):
            sel = common & (year == y)
            if sel.sum() < 20:
                continue
            yr_rows.append(dict(对比=lab, 年份=y, 差值=float(np.nanmean(diff[sel])),
                                t值=float(nw_t(diff[sel], 5)), 交易日=int(sel.sum())))
    yr = pd.DataFrame(yr_rows)
    yr.to_csv(os.path.join(RES, "探索_逐年交互.csv"), index=False, encoding="utf-8-sig")
    print("\n逐年交互")
    for lab, x in yr.groupby("对比", sort=False):
        print(f"{lab}: 年数={len(x)} 负={int((x.差值<0).sum())} 正={int((x.差值>0).sum())} "
              f"均值={x.差值.mean()*100:+.3f}% 最小={x.差值.min()*100:+.2f}% 最大={x.差值.max()*100:+.2f}%")
        print(x[["年份", "差值", "t值", "交易日"]].to_string(index=False))

    # ---------- 中性区 vs MA60 ----------
    print("\n中性区 vs MA60")
    vrows = []
    for lab, cell, ref in [
        ("MACD多 KQ3", mb & (qs["K"] == 2), ma60b),
        ("MACD多 KQ3+K<=70", mb & (qs["K"] == 2), ma60b),
        ("MACD空 KQ3", ~mb & (qs["K"] == 2), ma60s),
        ("MACD空 KQ3 vs Q1", ~mb & (qs["K"] == 2), ~mb & (qs["K"] == 0)),
        ("MACD多 K>70 vs MA60多", mb & (k > 70), ma60b),
        ("MACD多 K<20 vs MA60多", mb & (k < 20), ma60b),
    ]:
        sc, _ = daily_sel(cell & m20, ex20)
        sr, _ = daily_sel(ref & m20, ex20)
        common = np.isfinite(sc) & np.isfinite(sr)
        if common.sum() < 50:
            continue
        diff = sc - sr
        vrows.append(dict(格子=lab, 格子均值=float(np.nanmean(sc[common])),
                          对照均值=float(np.nanmean(sr[common])),
                          差值=float(np.nanmean(diff[common])), t值=float(nw_t(diff[common], 20)),
                          共同日=int(common.sum())))
    print(pd.DataFrame(vrows).to_string(index=False))
    pd.DataFrame(vrows).to_csv(os.path.join(RES, "探索_中性区增量.csv"), index=False, encoding="utf-8-sig")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
