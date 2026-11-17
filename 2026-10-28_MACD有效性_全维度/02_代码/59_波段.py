#!/usr/bin/env python3
"""59 波段研究：MACD/KDJ 能否确定波段？

口径
----
- 波段用收盘价 ZigZag 标注：从摆动极值反转 >= 阈值 才确认拐点。
- 拐点在确认日才可知；策略最早在确认日 T+1 开盘执行。
- 指标只用 T 日及以前数据；MACD(12,26,9)、KDJ(9,3,3)。
- 股票池：上市满 250 日、非 ST、20 日成交额 >= 500 万；含退市股。
- 收益用后复权；成本单边 0.075%。

输出
----
- 59_波段_拐点概览.csv：各阈值波段数量、幅度、确认滞后、确认时已走比例。
- 59_波段_识别.csv：拐点前后信号命中/领先滞后/确认后滞后。
- 59_波段_精确率.csv：信号落在拐点附近的比例、随机基准与提升。
- 59_波段_翻转.csv：确认后 MACD/KDJ 状态翻转的额外滞后与额外让渡。
- 59_波段_方向.csv：指标方向与当前波段方向的一致率 vs 价格基准。
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
import stock_core as S

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
THRESHOLDS = (0.05, 0.08, 0.10, 0.15)
MAIN_TH = 0.10
BIG = 10 ** 9


# ---------------------------------------------------------------- ZigZag
def zigzag(close, valid, th, min_gap=3):
    """收盘价 ZigZag。返回 [(pivot_idx, type, confirm_idx)]，type=+1 高点/-1 低点。
    只用确认日及以前的数据；min_gap 为相邻拐点最小交易日间隔。"""
    idxs = np.where(valid & np.isfinite(close))[0]
    n = len(idxs)
    if n < 80:
        return []
    cl = close[idxs].astype(np.float64)
    piv = []
    direction = 0
    hi_i = lo_i = 0
    hi = lo = cl[0]
    for k in range(1, n):
        p = cl[k]
        if direction == 0:
            if p > hi:
                hi, hi_i = p, k
            if p < lo:
                lo, lo_i = p, k
            if p >= lo * (1.0 + th):
                cand = (idxs[lo_i], -1, idxs[k])
                if not piv or cand[0] - piv[-1][0] >= min_gap:
                    piv.append(cand)
                direction = 1
                hi, hi_i = p, k
            elif p <= hi * (1.0 - th):
                cand = (idxs[hi_i], 1, idxs[k])
                if not piv or cand[0] - piv[-1][0] >= min_gap:
                    piv.append(cand)
                direction = -1
                lo, lo_i = p, k
        elif direction == 1:
            if p > hi:
                hi, hi_i = p, k
            if p <= hi * (1.0 - th):
                cand = (idxs[hi_i], 1, idxs[k])
                if not piv or cand[0] - piv[-1][0] >= min_gap:
                    piv.append(cand)
                direction = -1
                lo, lo_i = p, k
        else:
            if p < lo:
                lo, lo_i = p, k
            if p >= lo * (1.0 + th):
                cand = (idxs[lo_i], -1, idxs[k])
                if not piv or cand[0] - piv[-1][0] >= min_gap:
                    piv.append(cand)
                direction = 1
                hi, hi_i = p, k
    return piv


def build_pivots(P, th, base):
    out = []
    ncol = P["shape"][1]
    for j in range(ncol):
        pv = zigzag(P["close_ff"][:, j], base[:, j], th)
        # 去掉第一个起点拐点（不是真实摆动极值）
        for q in pv[1:]:
            out.append((th, j, q[0], q[1], q[2]))
    df = pd.DataFrame(out, columns=["th", "ci", "pivot_i", "type", "confirm_i"])
    df["pivot_date"] = P["dates"].to_numpy()[df["pivot_i"]]
    df["confirm_date"] = P["dates"].to_numpy()[df["confirm_i"]]
    df["confirm_lag"] = df["confirm_i"] - df["pivot_i"]
    df["price"] = P["close_ff"][df["pivot_i"].to_numpy(), df["ci"].to_numpy()]
    return df


def add_swing_amp(P, pv):
    """每个拐点：下一个反向拐点的价格、幅度、波段方向。"""
    n = len(pv)
    next_pivot = np.full(n, -1, dtype=np.int32)
    next_price = np.full(n, np.nan)
    next_type = np.zeros(n, dtype=np.int8)
    for (th, ci), g in pv.groupby(["th", "ci"], sort=False):
        idx = g.index.to_numpy()
        pi = g["pivot_i"].to_numpy()
        ty = g["type"].to_numpy()
        pr = g["price"].to_numpy()
        for q in range(len(idx) - 1):
            if ty[q] == -1 and ty[q + 1] == 1 and pr[q + 1] > pr[q]:
                next_pivot[idx[q]] = pi[q + 1]
                next_price[idx[q]] = pr[q + 1]
                next_type[idx[q]] = 1
            elif ty[q] == 1 and ty[q + 1] == -1 and pr[q + 1] < pr[q]:
                next_pivot[idx[q]] = pi[q + 1]
                next_price[idx[q]] = pr[q + 1]
                next_type[idx[q]] = -1
    pv = pv.copy()
    pv["next_pivot"] = next_pivot
    pv["next_price"] = next_price
    pv["next_type"] = next_type
    pv["swing_amp"] = pv["next_price"] / pv["price"] - 1.0
    pv["swing_days"] = pv["next_pivot"] - pv["pivot_i"]
    pv["confirm_ret"] = P["close_ff"][pv["confirm_i"].to_numpy(), pv["ci"].to_numpy()] / pv["price"] - 1.0
    pv["confirm_share"] = pv["confirm_ret"] / pv["swing_amp"]
    return pv


# ---------------------------------------------------------------- math helpers
def dist_arrays(s):
    """s: bool 1D。返回 (带符号最近信号距离, 之后第一个信号下标)。
    距离只在 ±60 内有效，否则 BIG；之后信号无则 BIG。"""
    n = len(s)
    idx = np.arange(n, dtype=np.int32)
    last = np.maximum.accumulate(np.where(s, idx, -BIG))
    nxt = np.minimum.accumulate(np.where(s, idx, BIG)[::-1])[::-1]
    dprev = idx - last
    dnext = nxt - idx
    dist = np.where(dprev <= dnext, -dprev, dnext)
    dist[np.minimum(dprev, dnext) > 60] = BIG
    return dist.astype(np.int32), nxt.astype(np.int32)


def safe_ret(close, i0, i1, ci):
    """close[i1]/close[i0]-1，越界或缺失返回 nan。"""
    i0 = np.asarray(i0)
    i1 = np.asarray(i1)
    ci = np.asarray(ci)
    ok = (i1 >= 0) & (i1 < close.shape[0]) & (i0 >= 0) & (i0 < close.shape[0])
    out = np.full(len(i0), np.nan)
    a = np.where(ok, i0, 0)
    b = np.where(ok, i1, 0)
    out[ok] = close[b[ok], ci[ok]] / close[a[ok], ci[ok]] - 1.0
    return out


# ---------------------------------------------------------------- tables
def signal_pack(P, macd, kd):
    dif, dea = macd["dif"], macd["dea"]
    mg, md = M.crosses(dif, dea, P)
    kg, kdx = K.kdj_crosses(kd["k"], kd["d"], P)
    k, d, j = kd["k"], kd["d"], kd["j"]
    kp, jp = M.shift(k, 1), M.shift(j, 1)
    base = P["valid"] & (P["bars"] >= 120)
    sig = {
        "MACD金叉": mg, "MACD死叉": md,
        "MACD多头": (dif > dea) & base, "MACD空头": (dif <= dea) & base,
        "KDJ金叉": kg, "KDJ死叉": kdx,
        "KDJ多头": (k > d) & base, "KDJ空头": (k <= d) & base,
        "KDJ低位金叉": kg & (k < 30), "KDJ高位死叉": kdx & (k > 70),
        "K<20": (k < 20) & P["valid"], "K<30": (k < 30) & P["valid"],
        "K>70": (k > 70) & P["valid"], "K>80": (k > 80) & P["valid"],
        "J<0": (j < 0) & P["valid"], "J>100": (j > 100) & P["valid"],
        "K上穿20": (k >= 20) & (kp < 20) & P["valid"],
        "K下穿20": (k < 20) & (kp >= 20) & P["valid"],
        "K上穿80": (k >= 80) & (kp < 80) & P["valid"],
        "K下穿80": (k < 80) & (kp >= 80) & P["valid"],
        "J上穿0": (j >= 0) & (jp < 0) & P["valid"],
        "J下穿0": (j < 0) & (jp >= 0) & P["valid"],
        "J上穿100": (j >= 100) & (jp < 100) & P["valid"],
        "J下穿100": (j < 100) & (jp >= 100) & P["valid"],
    }
    return sig


EVENT_LOW = ["MACD金叉", "KDJ金叉", "KDJ低位金叉", "K下穿20", "K上穿20",
             "J下穿0", "J上穿0"]
EVENT_HIGH = ["MACD死叉", "KDJ死叉", "KDJ高位死叉", "K上穿80", "K下穿80",
              "J上穿100", "J下穿100"]
EVENTS = sorted(set(EVENT_LOW + EVENT_HIGH))


def event_table(P, pv, sig):
    close = P["close_ff"]
    groups = list(pv.groupby("ci", sort=False))
    rows = []
    for name in EVENTS:
        s_all = sig[name]
        n = len(pv)
        dist = np.full(n, BIG, dtype=np.int32)
        nxt = np.full(n, BIG, dtype=np.int32)
        for ci, g in groups:
            darr, narr = dist_arrays(s_all[:, ci])
            idx = g.index.to_numpy()
            dist[idx] = darr[g["pivot_i"].to_numpy()]
            nxt[idx] = narr[g["confirm_i"].to_numpy()]
        pv2 = pv.copy()
        pv2["dist"] = dist
        pv2["nxt"] = nxt
        pv2["lag_after"] = np.where(pv2["nxt"] < BIG,
                                    pv2["nxt"] - pv2["confirm_i"], np.nan)
        pv2["ret_after"] = safe_ret(close, pv2["confirm_i"].to_numpy(),
                                    pv2["nxt"].to_numpy(), pv2["ci"].to_numpy())
        pv2.loc[pv2["lag_after"] > 60, ["lag_after", "ret_after"]] = np.nan
        for (th, typ), g in pv2.groupby(["th", "type"]):
            d = g["dist"].to_numpy()
            ok = d < BIG
            rows.append(dict(
                阈值=th, 拐点类型="低点" if typ == -1 else "高点", 信号=name,
                拐点数=len(g),
                命中5=float(np.mean(np.abs(d) <= 5)),
                命中10=float(np.mean(np.abs(d) <= 10)),
                领先5=float(np.mean((d >= -5) & (d < 0))),
                滞后5=float(np.mean((d > 0) & (d <= 5))),
                同步=float(np.mean(d == 0)),
                六十日内无信号=float(np.mean(~ok)),
                中位距离=float(np.median(d[ok])) if ok.any() else np.nan,
                确认后中位滞后=float(np.nanmedian(g["lag_after"])),
                确认后中位变动=float(np.nanmedian(g["ret_after"])),
            ))
    return pd.DataFrame(rows)


def coverage_matrix(pv, n, nc, typ, win):
    cov = np.zeros((n, nc), dtype=bool)
    sub = pv[pv["type"] == typ]
    for r in sub.itertuples():
        cov[max(0, r.pivot_i - win):min(n, r.pivot_i + win + 1), r.ci] = True
    return cov


def precision_table(P, pv, sig, base):
    n, nc = P["shape"]
    rows = []
    for th in sorted(pv["th"].unique()):
        sub = pv[pv["th"] == th]
        cov_low = coverage_matrix(sub, n, nc, -1, 5)
        cov_high = coverage_matrix(sub, n, nc, 1, 5)
        b = base
        base_low = float((cov_low & b).sum() / b.sum())
        base_high = float((cov_high & b).sum() / b.sum())
        for name in EVENTS:
            s = sig[name] & b
            ns = int(s.sum())
            if ns == 0:
                continue
            pl = float((s & cov_low).sum() / ns)
            ph = float((s & cov_high).sum() / ns)
            rows.append(dict(
                阈值=th, 信号=name, 信号数=ns,
                落低点5=pl, 落高点5=ph, 低点基准=base_low, 高点基准=base_high,
                低点提升=pl / max(base_low, 1e-9), 高点提升=ph / max(base_high, 1e-9),
            ))
    return pd.DataFrame(rows)


def flip_table(P, pv, sig):
    """确认后状态翻转的额外滞后与额外让渡。"""
    close = P["close_ff"]
    groups = list(pv.groupby("ci", sort=False))
    rows = []
    cases = [
        ("MACD多头", -1), ("MACD空头", 1), ("KDJ多头", -1), ("KDJ空头", 1),
        ("K<20", -1), ("K<30", -1), ("K>70", 1), ("K>80", 1), ("J<0", -1), ("J>100", 1),
    ]
    for name, typ in cases:
        s_all = sig[name]
        n = len(pv)
        nxt = np.full(n, BIG, dtype=np.int32)
        for ci, g in groups:
            _, narr = dist_arrays(s_all[:, ci])
            nxt[g.index.to_numpy()] = narr[g["confirm_i"].to_numpy()]
        pv2 = pv[pv["type"] == typ].copy()
        # 对齐 nxt
        nxt_map = pd.Series(nxt, index=pv.index)
        pv2["nxt"] = nxt_map.loc[pv2.index].to_numpy()
        already = pv2["nxt"] == pv2["confirm_i"]
        # 只认“本波段内”的翻转：不能晚于下一个反向拐点
        in_swing = (pv2["nxt"] < BIG) & (pv2["nxt"] <= pv2["next_pivot"])
        pv2["flip_lag"] = np.where(in_swing, pv2["nxt"] - pv2["confirm_i"], np.nan)
        pv2["flip_ret"] = np.where(in_swing,
                                   safe_ret(close, pv2["confirm_i"].to_numpy(),
                                            pv2["nxt"].to_numpy(), pv2["ci"].to_numpy()),
                                   np.nan)
        pv2["total_ret"] = np.where(in_swing,
                                    safe_ret(close, pv2["pivot_i"].to_numpy(),
                                             pv2["nxt"].to_numpy(), pv2["ci"].to_numpy()),
                                    np.nan)
        pv2["total_share"] = pv2["total_ret"] / pv2["swing_amp"]
        # 拐点当天状态
        pv2["at_pivot"] = sig[name][pv2["pivot_i"].to_numpy(), pv2["ci"].to_numpy()]
        for th, g in pv2.groupby("th"):
            ga = g.loc[~already.loc[g.index] & in_swing.loc[g.index]]
            rows.append(dict(
                阈值=th, 拐点类型="低点" if typ == -1 else "高点", 状态=name,
                拐点数=len(g),
                拐点日已满足=float(g["at_pivot"].mean()),
                确认时已满足=float(already.loc[g.index].mean()),
                波段内未翻转=float((~in_swing.loc[g.index]).mean()),
                额外滞后中位=float(np.nanmedian(ga["flip_lag"])),
                额外让渡中位=float(np.nanmedian(ga["flip_ret"])),
                确认时波段已走=float(np.nanmedian(g["confirm_share"])),
                指标确认时已走=float(np.nanmedian(g["total_share"])),
            ))
    return pd.DataFrame(rows)


def direction_table(P, pv, sig, base):
    n, nc = P["shape"]
    dirn = np.zeros((n, nc), dtype=np.int8)
    order = pv[pv["th"] == MAIN_TH].sort_values(["ci", "confirm_i"])
    for ci, g in order.groupby("ci"):
        for r in g.itertuples():
            dirn[r.confirm_i:, ci] = -r.type
    c = P["close_ff"]
    ma20 = M.rolling_mean(c, 20)
    ret20 = np.full_like(c, np.nan)
    ret20[20:] = c[20:] / c[:-20] - 1.0
    preds = {
        "MACD多头": sig["MACD多头"],
        "KDJ多头": sig["KDJ多头"],
        "价格>MA20": c > ma20,
        "20日动量为正": ret20 > 0,
    }
    rows = []
    ok = base & (dirn != 0)
    for name, p in preds.items():
        known = ok & np.isfinite(p)
        agree = np.mean(((p & (dirn > 0)) | (~p & (dirn < 0)))[known])
        rows.append(dict(指标=name, 一致率=float(agree),
                         逆势看多=float(np.mean((p & (dirn < 0))[known])),
                         逆势看空=float(np.mean((~p & (dirn > 0))[known])),
                         样本日=int(known.sum())))
    a, b = sig["MACD多头"], sig["KDJ多头"]
    both = np.mean((((a & b) & (dirn > 0)) | ((~a & ~b) & (dirn < 0)))[ok])
    rows.append(dict(指标="双指标一致", 一致率=float(both), 逆势看多=np.nan,
                     逆势看空=np.nan, 样本日=int(ok.sum())))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    P = S.load_base()
    base = P["universe_inv"]
    macd = F.build_macd_features(P)
    kd = K.kdj_features(P)
    sig = signal_pack(P, macd, kd)
    print("loaded", P["shape"], "base", int(base.sum()), "%.1fs" % (time.time() - t0))

    pvs = []
    for th in THRESHOLDS:
        pv = build_pivots(P, th, base)
        pv = add_swing_amp(P, pv)
        pvs.append(pv)
        print("th", th, "pivots", len(pv), "%.1fs" % (time.time() - t0))
    pv = pd.concat(pvs, ignore_index=True)
    pv.to_parquet(os.path.join(RES, "59_波段_拐点.parquet"))

    ov = pv.groupby(["th", "type"]).agg(
        拐点数=("pivot_i", "size"),
        中位确认滞后=("confirm_lag", "median"),
        中位波段幅度=("swing_amp", lambda x: np.nanmedian(np.abs(x))),
        中位波段天数=("swing_days", "median"),
        确认时已走=("confirm_share", "median"),
    ).reset_index()
    ov["拐点类型"] = np.where(ov["type"] == -1, "低点", "高点")
    ov = ov.drop(columns="type")
    ov.to_csv(os.path.join(RES, "59_波段_拐点概览.csv"), index=False, encoding="utf-8-sig")
    print(ov.to_string(index=False))

    ev = event_table(P, pv, sig)
    ev.to_csv(os.path.join(RES, "59_波段_识别.csv"), index=False, encoding="utf-8-sig")
    print("event table done %.1fs" % (time.time() - t0))

    pr = precision_table(P, pv, sig, base)
    pr.to_csv(os.path.join(RES, "59_波段_精确率.csv"), index=False, encoding="utf-8-sig")
    print("precision done %.1fs" % (time.time() - t0))

    fl = flip_table(P, pv, sig)
    fl.to_csv(os.path.join(RES, "59_波段_翻转.csv"), index=False, encoding="utf-8-sig")
    print("flip done %.1fs" % (time.time() - t0))

    dr = direction_table(P, pv, sig, base)
    dr.to_csv(os.path.join(RES, "59_波段_方向.csv"), index=False, encoding="utf-8-sig")
    print(dr.to_string(index=False))
    print("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
