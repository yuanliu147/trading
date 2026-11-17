#!/usr/bin/env python3
"""49e 控制：中性区逐年、先期收益/波动双重排序、市场状态；判断 KDJ 中性区是否只是短期反转/低波。"""
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


def contrast(a, b, ex, dates, min_n=50):
    sa, _ = daily_sel(a, ex, min_n)
    sb, _ = daily_sel(b, ex, min_n)
    common = np.isfinite(sa) & np.isfinite(sb)
    if common.sum() < 20:
        return None
    diff = sa - sb
    early = common & (dates <= pd.Timestamp("2017-12-31"))
    late = common & (dates >= pd.Timestamp("2018-01-01"))
    return dict(A=float(np.nanmean(sa[common])), B=float(np.nanmean(sb[common])),
                差值=float(np.nanmean(diff[common])), t=float(nw_t(diff[common], 20)),
                前期=float(np.nanmean(diff[early])) if early.any() else np.nan,
                后期=float(np.nanmean(diff[late])) if late.any() else np.nan,
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
    m20 = base & ok[20] & np.isfinite(ex)
    mb = macd["states"]["DIF>DEA"] & m20
    k = kd["k"]
    qk = F.row_quantile(k, base, 5)
    neutral = (qk == 2)
    mid2 = (qk == 1) | (qk == 2)
    nonext = (k >= 20) & (k <= 80)

    # ---------- 1. 逐年 ----------
    print("=== 逐年：中性区 vs 其他（20日） ===")
    rows = []
    year = dates.year
    for lab, a, b in [
        ("MACD多 KQ3-其他", mb & neutral, mb & ~neutral),
        ("MACD多 KQ2+Q3-其他", mb & mid2, mb & ~mid2),
        ("MACD多 K20-80-极端", mb & nonext, mb & ~nonext),
        ("MACD空 KQ3-其他", ~mb & neutral, ~mb & ~neutral),
        ("MACD空 KQ2+Q3-其他", ~mb & mid2, ~mb & ~mid2),
        ("MACD空 K20-80-极端", ~mb & nonext, ~mb & ~nonext),
    ]:
        sa, _ = daily_sel(a, ex)
        sb, _ = daily_sel(b, ex)
        common = np.isfinite(sa) & np.isfinite(sb)
        diff = sa - sb
        for y in range(2010, 2027):
            sel = common & (year == y)
            if sel.sum() < 20:
                continue
            rows.append(dict(对比=lab, 年份=y, 差值=float(np.nanmean(diff[sel])),
                             t=float(nw_t(diff[sel], 5)), 交易日=int(sel.sum())))
    yr = pd.DataFrame(rows)
    yr.to_csv(os.path.join(RES, "探索_中性区逐年.csv"), index=False, encoding="utf-8-sig")
    for lab, x in yr.groupby("对比", sort=False):
        print(f"{lab}: 年数={len(x)} 正={int((x.差值>0).sum())} 均值={x.差值.mean()*100:+.3f}% "
              f"最小={x.差值.min()*100:+.2f}% 最大={x.差值.max()*100:+.2f}%")
        print(x[["年份", "差值", "t", "交易日"]].to_string(index=False))

    # ---------- 2. 先期 20 日收益 x 波动 双重排序 ----------
    r20 = st["ret20"]
    vol = st["std20"]
    qr = F.row_quantile(r20, base, 5)
    qv = F.row_quantile(vol, base, 5)
    print("\n=== 双重排序：MACD多 & KQ3 vs 其他 ===")
    rows = []
    for ri in range(5):
        for vi in range(5):
            cell = mb & (qr == ri) & (qv == vi)
            a = cell & neutral
            b = cell & ~neutral
            if a.sum() < 2000 or b.sum() < 2000:
                continue
            r = contrast(a, b, ex, dates)
            if r is None:
                continue
            rows.append(dict(先期收益分位=f"R{ri+1}", 波动分位=f"V{vi+1}",
                             A股日=int(a.sum()), B股日=int(b.sum()), **r))
    ds = pd.DataFrame(rows)
    ds.to_csv(os.path.join(RES, "探索_双重排序_收益波动.csv"), index=False, encoding="utf-8-sig")
    print(ds.to_string(index=False))
    print("正格数", int((ds.差值 > 0).sum()), "/", len(ds), " t>2 格数", int((ds.t > 2).sum()))

    # ---------- 3. 市场状态 ----------
    print("\n=== 市场状态 ===")
    mkt_state = st["market"][:, 0]
    rows = []
    for ms, ml in [(0, "指数上行"), (1, "指数下行"), (2, "指数震荡")]:
        for lab, a, b in [
            ("MACD多 KQ3-其他", mb & neutral, mb & ~neutral),
            ("MACD多 K20-80-极端", mb & nonext, mb & ~nonext),
            ("MACD空 KQ3-其他", ~mb & neutral, ~mb & ~neutral),
            ("MACD空 K20-80-极端", ~mb & nonext, ~mb & ~nonext),
        ]:
            mask = (mkt_state == ms)[:, None]
            r = contrast(a & mask, b & mask, ex, dates)
            if r is None:
                continue
            rows.append(dict(市场=ml, 对比=lab, **r))
    mk = pd.DataFrame(rows)
    mk.to_csv(os.path.join(RES, "探索_中性区市场.csv"), index=False, encoding="utf-8-sig")
    print(mk.to_string(index=False))

    # ---------- 4. 主板/创业板/科创板 ----------
    print("\n=== 板块 ===")
    code_pref = pd.Series(P["codes"]).str[:3].to_numpy()
    board = np.where(np.isin(code_pref, ["300", "301", "302"]), "创业板",
                     np.where(np.isin(code_pref, ["688", "689"]), "科创板", "主板"))
    rows = []
    for bd in ["主板", "创业板", "科创板"]:
        mask = (board == bd)[None, :]
        for lab, a, b in [
            ("MACD多 KQ3-其他", mb & neutral, mb & ~neutral),
            ("MACD多 K20-80-极端", mb & nonext, mb & ~nonext),
            ("MACD空 KQ3-其他", ~mb & neutral, ~mb & ~neutral),
        ]:
            r = contrast(a & mask, b & mask, ex, dates)
            if r is None:
                continue
            rows.append(dict(板块=bd, 对比=lab, **r))
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "探索_中性区板块.csv"), index=False, encoding="utf-8-sig")

    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
