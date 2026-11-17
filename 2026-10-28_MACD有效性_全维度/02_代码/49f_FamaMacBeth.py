#!/usr/bin/env python3
"""49f Fama-MacBeth：控制先期收益、波动、成交额、MA60 后，KDJ 水平/状态是否仍有增量。

每日横截面回归 ex_h ~ MACD状态 + KDJ分组/状态 + 交互 + 控制；系数日序列 NW(20) t。
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


def zscore(x, mask):
    v = x[mask]
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def run_fm(P, ex, base, cols, names, dates, min_n=500, winsor=(0.005, 0.995)):
    """cols: list of 2D arrays; names: list of str. 返回系数日序列。"""
    coefs = []
    r2s = []
    ns = []
    used_dates = []
    for t in range(P["shape"][0]):
        m = base[t] & np.isfinite(ex[t])
        for c in cols:
            m &= np.isfinite(c[t])
        n = int(m.sum())
        if n < min_n:
            continue
        y = ex[t, m].astype(np.float64)
        lo, hi = np.quantile(y, winsor)
        y = np.clip(y, lo, hi)
        X = np.column_stack([np.ones(n)] + [c[t, m] for c in cols]).astype(np.float64)
        # 去掉零方差列
        sd = X.std(axis=0)
        keep = sd > 1e-12
        keep[0] = True
        X = X[:, keep]
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = X @ beta
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        full = np.full(len(names) + 1, np.nan)
        full[keep] = beta
        coefs.append(full)
        r2s.append(r2)
        ns.append(n)
        used_dates.append(dates[t])
    coefs = pd.DataFrame(coefs, columns=["截距"] + names, index=used_dates)
    return coefs, np.array(r2s), np.array(ns)


def summarize(coefs, r2, ns, label):
    rows = []
    for c in coefs.columns:
        x = coefs[c].dropna()
        if len(x) < 50:
            continue
        rows.append(dict(设定=label, 变量=c, 均值=float(x.mean()), t值=float(nw_t(x, 20)),
                         标准差=float(x.std()), 交易日=int(len(x))))
    out = pd.DataFrame(rows)
    out["平均R2"] = float(np.nanmean(r2))
    out["平均股票数"] = float(np.nanmean(ns))
    return out


def main():
    t0 = time.time()
    P = M.load_panel()
    dates = P["dates"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    fwd, ok, mkt, loo = M.forward_returns(P, (5, 20))
    exs = {h: (fwd[h] - loo[h]).astype(np.float32) for h in (5, 20)}
    masks = {h: base & ok[h] & np.isfinite(exs[h]) for h in (5, 20)}

    k = kd["k"]
    qk = F.row_quantile(k, base, 5)
    mb = macd["states"]["DIF>DEA"].astype(np.float32)
    kq = [((qk == i).astype(np.float32) if i > 0 else None) for i in range(5)]
    # 连续 K 的 z 分数（当日横截面）
    kz = np.full(k.shape, np.nan, dtype=np.float32)
    for t in range(k.shape[0]):
        m = base[t] & np.isfinite(k[t])
        kz[t] = zscore(k[t], m)
    ret5 = np.full(k.shape, np.nan, dtype=np.float32)
    ret5[5:] = P["close_ff"][5:] / P["close_ff"][:-5] - 1
    ret20 = st["ret20"]
    ret60 = st["ret60"]
    std20 = st["std20"]
    logamt = np.log1p(np.maximum(M.rolling_mean(P["amount"], 20), 0))
    ma60 = st["ma60"]
    above60 = (P["close_ff"] > ma60).astype(np.float32)
    disthigh = st["dist_high"]

    def zcol(x):
        out = np.full(x.shape, np.nan, dtype=np.float32)
        for t in range(x.shape[0]):
            m = base[t] & np.isfinite(x[t])
            out[t] = zscore(x[t], m)
        return out
    z_ret5, z_ret20, z_ret60 = zcol(ret5), zcol(ret20), zcol(ret60)
    z_std20, z_logamt, z_disthigh = zcol(std20), zcol(logamt), zcol(disthigh)

    controls = [z_ret5, z_ret20, z_std20, z_logamt]
    control_names = ["5日收益z", "20日收益z", "20日波动z", "成交额z"]
    controls2 = controls + [z_ret60, z_disthigh, above60]
    control_names2 = control_names + ["60日收益z", "距250日高z", "价格>MA60"]

    # 交互项
    inter = []
    for i in range(1, 5):
        inter.append((mb * kq[i]).astype(np.float32))
    # 设定 A：KQ2-Q5 + 交互 + 控制
    cols_a = [mb] + [kq[i] for i in range(1, 5)] + inter + controls
    names_a = ["MACD多", "KQ2", "KQ3", "KQ4", "KQ5",
               "MACD多×KQ2", "MACD多×KQ3", "MACD多×KQ4", "MACD多×KQ5"] + control_names

    # 设定 B：加 MA60/60日/位置
    cols_b = cols_a + [z_ret60, z_disthigh, above60]
    names_b = names_a + ["60日收益z", "距250日高z", "价格>MA60"]

    # 设定 C：连续 K + 二次项 + 交互
    kz2 = (kz ** 2).astype(np.float32)
    cols_c = [mb, kz, kz2, (mb * kz).astype(np.float32), (mb * kz2).astype(np.float32)] + controls
    names_c = ["MACD多", "K水平z", "K水平z^2", "MACD多×Kz", "MACD多×Kz^2"] + control_names

    # 设定 D：常见状态 K>D, K>70, K<20 + 交互
    kdum = kd["states"]["K>D"].astype(np.float32)
    kup = (k > 70).astype(np.float32)
    kdn = (k < 20).astype(np.float32)
    cols_d = [mb, kdum, kup, kdn,
              (mb * kdum).astype(np.float32), (mb * kup).astype(np.float32), (mb * kdn).astype(np.float32)] + controls
    names_d = ["MACD多", "K>D", "K>70", "K<20",
               "MACD多×K>D", "MACD多×K>70", "MACD多×K<20"] + control_names

    all_sum = []
    for h in (5, 20):
        ex = exs[h]
        mask = masks[h]
        print(f"\n===== horizon {h} =====")
        for label, cols, names in [
            ("A_K五分组", cols_a, names_a),
            ("B_加趋势控制", cols_b, names_b),
            ("C_连续K二次", cols_c, names_c),
            ("D_常用状态", cols_d, names_d),
        ]:
            print(label, "running", time.time() - t0)
            co, r2, ns = run_fm(P, ex, mask, cols, names, dates)
            s = summarize(co, r2, ns, f"{h}日_{label}")
            all_sum.append(s)
            print(s.to_string(index=False))
    out = pd.concat(all_sum, ignore_index=True)
    out.to_csv(os.path.join(RES, "探索_FamaMacBeth.csv"), index=False, encoding="utf-8-sig")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
