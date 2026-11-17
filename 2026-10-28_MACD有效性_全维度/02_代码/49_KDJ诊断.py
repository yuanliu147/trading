#!/usr/bin/env python3
"""49 KDJ 基础诊断：状态重叠、四态收益、事件量级。

仅用于设计后续研究，不生成正式交付物。
"""
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


def stat(sel, ex, dates):
    dm, _ = daily_sel(sel, ex)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(n=int(sel.sum()), days=int(ok.sum()), mean=np.nan, t=np.nan,
                    early=np.nan, late=np.nan)
    return dict(n=int(sel.sum()), days=int(ok.sum()),
                mean=float(np.nanmean(dm)), t=float(nw_t(dm, 20)),
                early=float(np.nanmean(dm[ok & (dates <= pd.Timestamp("2017-12-31"))])),
                late=float(np.nanmean(dm[ok & (dates >= pd.Timestamp("2018-01-01"))])))


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    print("panel", P["shape"])
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    print("kdj done", time.time() - t0)

    fwd, ok, mkt, loo = M.forward_returns(P, (5, 20, 60))
    for h in (5, 20, 60):
        ex = (fwd[h] - loo[h]).astype(np.float32)
        mask = base & ok[h] & np.isfinite(ex)
        mb = macd["states"]["DIF>DEA"] & mask
        kb = kd["states"]["K>D"] & mask
        print(f"\n===== horizon {h} =====")
        print("overlap: MACD bull share %.3f, KDJ bull share %.3f, both %.3f, agree %.3f" % (
            mb.sum() / mask.sum(), kb.sum() / mask.sum(), (mb & kb).sum() / mask.sum(),
            ((mb & kb) | (~mb & ~kb)).sum() / mask.sum()))
        for ml, ms in [(True, "MACD多"), (False, "MACD空")]:
            for kl, ks in [(True, "KDJ多"), (False, "KDJ空")]:
                sel = mask & (mb if ml else ~mb) & (kb if kl else ~kb)
                r = stat(sel, ex, dates)
                print(f"{ms}+{ks}: n={r['n']:>9d} mean={r['mean']*100:+.3f}% t={r['t']:+.2f} "
                      f"early={r['early']*100:+.3f}% late={r['late']*100:+.3f}%")

    # KDJ 事件量级
    print("\nKDJ crosses total:", int(kd["gold"].sum()), int(kd["death"].sum()))
    print("MACD crosses total:", int(macd["states"]["DIF>DEA"].sum()))
    # MACD gold by KDJ state at signal close
    gold, death = M.crosses(macd["dif"], macd["dea"], P)
    kdj_gold = kd["gold"]
    both = gold & kdj_gold
    print("MACD gold & KDJ gold same day:", int(both.sum()))
    print("KDJ gold & MACD bull same day:", int((kdj_gold & macd["states"]["DIF>DEA"]).sum()))

    # 同日双金叉后 20 日
    ex20 = (fwd[20] - loo[20]).astype(np.float32)
    mask20 = base & ok[20] & np.isfinite(ex20)
    for lab, sel in [
        ("MACD金叉全部", gold & mask20),
        ("MACD金叉+KDJ多", gold & kd["states"]["K>D"] & mask20),
        ("MACD金叉+KDJ空", gold & ~kd["states"]["K>D"] & mask20),
        ("同日双金叉", both & mask20),
        ("KDJ金叉全部", kdj_gold & mask20),
        ("KDJ金叉+MACD多", kdj_gold & macd["states"]["DIF>DEA"] & mask20),
        ("KDJ金叉+MACD空", kdj_gold & ~macd["states"]["DIF>DEA"] & mask20),
        ("KDJ金叉+MACD双上", kdj_gold & (macd["dif"] > 0) & (macd["dea"] > 0) & mask20),
    ]:
        s = sel & mask20
        r = stat(s, ex20, dates)
        print(f"{lab:18s} n={r['n']:>7d} mean={r['mean']*100:+.3f}% t={r['t']:+.2f} "
              f"early={r['early']*100:+.3f}% late={r['late']*100:+.3f}%")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
