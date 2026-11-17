#!/usr/bin/env python3
"""35 组合检验：把逐股 MACD 择时做成可投资组合

- 等权、日频、T 日收盘信号 T+1 开盘执行；成本 0.075%/边。
- 对照：同暴露常数仓位、100% 持有、MA60 趋势、市场 MA200。
- 分组：VR20 / 规模 / 换手 / 特质波动 / 60 日涨幅 / 干净子集。
- 事前分组：前期特征 -> 后期组合。
- 循环平移零假：保留仓位序列与换手，只打乱与收益的对应。
"""
import os
import time

import numpy as np
import pandas as pd

import macd_core as M
import stock_core as S

RES = S.RES
COST = S.COST_SIDE
RNG = np.random.default_rng(20261029)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    z = np.load(os.path.join(RES, "个股_日频.npz"), allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    codes = z["codes"].astype(str)
    r = z["r_oo"].astype(np.float32)
    uni = z["universe"].astype(bool)
    inv = z["universe_inv"].astype(bool)
    pos = {k: z[k].astype(np.float32) for k in ["pos_dif", "pos_hist", "pos_ma60"]}
    return dates, codes, r, uni, inv, pos


def ew_return(r, mask):
    m = mask & np.isfinite(r)
    n = m.sum(axis=1)
    s = np.nansum(np.where(m, r, 0), axis=1)
    return np.where(n > 0, s / np.maximum(n, 1), 0.0).astype(np.float64)


def port_series(pos, mask, r, cost=True):
    p = np.where(mask, pos, 0.0).astype(np.float32)
    ret = p * r
    d = np.abs(np.diff(p, axis=0, prepend=np.zeros((1, p.shape[1]), dtype=np.float32)))
    if cost:
        ret = ret - COST * d
    n = mask.sum(axis=1)
    s = np.nansum(np.where(mask, ret, 0), axis=1) / np.maximum(n, 1)
    return s.astype(np.float64), p, d, n


def ann(x):
    if len(x) < 60:
        return np.nan
    path = np.cumprod(1.0 + x)
    if path[-1] <= 0:
        return -1.0
    return float(path[-1] ** (252.0 / len(x)) - 1.0)


def mdd(x):
    path = np.concatenate([[1.0], np.cumprod(1.0 + x)])
    peak = np.maximum.accumulate(path)
    return float(np.min(path / peak - 1.0))


def sharpe(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(252)) if sd > 0 else np.nan


def metrics(pos, mask, r, sl, cost=True):
    s, p, d, n = port_series(pos, mask, r, cost=cost)
    s, p, d, n = s[sl], p[sl], d[sl], n[sl]
    m = mask[sl]
    ew = ew_return(r[sl], m)
    nd = len(s)
    expo = float(np.nansum(p[m]) / max(np.nansum(m), 1))
    bench = expo * ew
    ann_s, ann_bh, ann_bm = ann(s), ann(ew), ann(bench)
    turn = float(np.nansum(d[m]) / max(np.nansum(m), 1) * 252)
    return dict(
        days=nd, avg_n=float(np.mean(n)), 暴露=expo,
        年化净=ann_s, 年化同暴露=ann_bm, 净超同暴露=ann_s - ann_bm,
        年化持有=ann_bh, 净超持有=ann_s - ann_bh,
        夏普净=sharpe(s), 夏普同暴露=sharpe(bench),
        回撤净=mdd(s), 回撤同暴露=mdd(bench),
        换手=turn,
    )


def gross_metrics(pos, mask, r, sl):
    return metrics(pos, mask, r, sl, cost=False)


def slice_for(dates, lo, hi):
    return (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))


def make_periods(dates):
    return {
        "全期": (dates >= pd.Timestamp("2010-01-01")),
        "前期": (dates >= pd.Timestamp("2010-01-01")) & (dates <= pd.Timestamp("2017-12-31")),
        "后期": (dates >= pd.Timestamp("2018-01-01")),
    }


def qstatic(x, n=5):
    x = pd.Series(x)
    r = x.rank(method="first", pct=True)
    q = np.floor(r * n).clip(0, n - 1)
    q[~np.isfinite(r)] = -1
    return q.to_numpy(dtype=float)


def main():
    t0 = time.time()
    log("load")
    dates, codes, r, uni, inv, pos = load()
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    st = st.set_index("code").reindex(codes).reset_index()
    periods = make_periods(dates)
    nc = len(codes)

    # 市场状态：等权指数 > MA200
    ew_all = ew_return(r, uni)
    idx = np.cumprod(1 + ew_all)
    ma200 = pd.Series(idx).rolling(200, min_periods=100).mean().to_numpy()
    mstate = (idx > ma200).astype(np.float32)
    pos_mkt = np.repeat(M.shift(mstate, 1)[:, None], nc, axis=1)
    log("market state done")

    rows = []
    # 全市场策略对比
    strategies = {"DIF>DEA": pos["pos_dif"], "柱体>0": pos["pos_hist"],
                  "价格>MA60": pos["pos_ma60"], "市场>MA200": pos_mkt,
                  "始终持有": np.ones_like(pos["pos_dif"])}
    for name, p in strategies.items():
        for lab, sl in periods.items():
            met = metrics(p, inv, r, sl, cost=True)
            g = gross_metrics(p, inv, r, sl)
            met.update(分组类型="全市场", 分组=name, 期间=lab,
                       毛超同暴露=g["净超同暴露"], 成本拖累=g["净超同暴露"] - met["净超同暴露"])
            rows.append(met)
        log(f"  策略 {name}")

    # 市场状态相关性
    frac_dif = np.nansum(np.where(inv, pos["pos_dif"], 0), axis=1) / np.maximum(inv.sum(axis=1), 1)
    corr_mkt = np.corrcoef(frac_dif[periods["全期"]], mstate[periods["全期"]])[0, 1]
    pd.DataFrame([dict(指标="DIF>DEA 持仓比例与市场MA200相关", 值=corr_mkt)]).to_csv(
        os.path.join(RES, "组合_市场相关.csv"), index=False, encoding="utf-8-sig")
    log(f"  与市场状态相关 {corr_mkt:.3f}")

    # 分组
    group_defs = []
    for f in ["VR20", "规模_log", "换手20", "特质波动250", "60日涨幅", "波动20"]:
        for lab in ["前期", "后期"]:
            x = st[f + "_" + lab].to_numpy()
            q = qstatic(x)
            for k in range(5):
                group_defs.append((f"{f}(同期)", lab, k + 1, q == k))
    # 事前：前期特征 -> 后期
    for f in ["VR20", "规模_log", "换手20", "特质波动250", "60日涨幅", "波动20"]:
        q = qstatic(st[f + "_前期"].to_numpy())
        for k in range(5):
            group_defs.append((f"{f}(事前)", "后期", k + 1, q == k))
    # 干净子集
    for lab in ["前期", "后期"]:
        cols = [f + "_" + lab for f in ["规模_log", "换手20", "特质波动250", "60日涨幅"]]
        X = st[cols].to_numpy(dtype=float)
        med = np.nanmedian(X, axis=0)
        ok = np.all(np.isfinite(X), axis=1)
        clean = ok & (X[:, 0] > med[0]) & (X[:, 1] < med[1]) & (X[:, 2] < med[2]) & (X[:, 3] < med[3])
        group_defs.append(("干净子集(同期)", lab, "干净", clean))
    # 事前干净
    cols = [f + "_前期" for f in ["规模_log", "换手20", "特质波动250", "60日涨幅"]]
    X = st[cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    ok = np.all(np.isfinite(X), axis=1)
    clean0 = ok & (X[:, 0] > med[0]) & (X[:, 1] < med[1]) & (X[:, 2] < med[2]) & (X[:, 3] < med[3])
    group_defs.append(("干净子集(事前)", "后期", "干净", clean0))

    for gtype, lab, gname, gmask in group_defs:
        mask = inv & gmask[None, :]
        if mask.sum() < 250 * 20:
            continue
        met = metrics(pos["pos_dif"], mask, r, periods[lab], cost=True)
        g = gross_metrics(pos["pos_dif"], mask, r, periods[lab])
        met.update(分组类型=gtype, 分组=str(gname), 期间=lab, 股票数=int(gmask.sum()),
                   毛超同暴露=g["净超同暴露"], 成本拖累=g["净超同暴露"] - met["净超同暴露"])
        rows.append(met)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "组合_择时.csv"), index=False, encoding="utf-8-sig")
    log(f"组合表 {out.shape}")

    # 循环平移零假：全市场、VR20 Q5 后期、干净子集后期
    null_rows = []
    tests = [("全市场", inv, pos["pos_dif"], "后期"),
             ("全市场", inv, pos["pos_dif"], "前期"),
             ("VR20_Q5", inv & (qstatic(st["VR20_后期"].to_numpy()) == 4)[None, :], pos["pos_dif"], "后期"),
             ("VR20_Q5_事前", inv & (qstatic(st["VR20_前期"].to_numpy()) == 4)[None, :], pos["pos_dif"], "后期"),
             ("干净子集", inv & clean0[None, :], pos["pos_dif"], "后期")]
    for name, mask, p, lab in tests:
        sl = periods[lab]
        actual = metrics(p, mask, r, sl, cost=True)["净超同暴露"]
        nd = int(sl.sum())
        vals = []
        for i in range(60):
            k = int(RNG.integers(50, max(51, nd - 50)))
            ps = np.roll(p, k, axis=0)
            vals.append(metrics(ps, mask, r, sl, cost=True)["净超同暴露"])
        vals = np.array(vals)
        pval = (1 + np.sum(vals >= actual)) / (len(vals) + 1)
        null_rows.append(dict(对象=name, 期间=lab, 实际=actual, 零假均值=vals.mean(),
                              零假P5=np.percentile(vals, 5), 零假P95=np.percentile(vals, 95),
                              p值=pval, 次数=len(vals)))
        log(f"  零假 {name} {lab}: 实际 {actual:.3f}, p={pval:.2f}")
    pd.DataFrame(null_rows).to_csv(os.path.join(RES, "组合_零假.csv"), index=False, encoding="utf-8-sig")

    # 分组表：只保留 DIF>DEA 的净/毛/夏普/回撤
    log(f"done {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
