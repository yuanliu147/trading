#!/usr/bin/env python3
"""60 波段捕获：同一批波段上，纯价格确认 vs 加 MACD/KDJ

口径
----
- 波段对 = ZigZag 低点 -> 下一个高点；低点确认后 T+1 开盘买，高点确认后 T+1 开盘卖。
- 过滤器在低点确认日判断（MACD 多头 / KDJ 多头 / 双多头 / 拐点日状态）。
- 择时 = 低点确认后等 MACD/KDJ 转多再买；高点确认后等转空再卖。
- 经典交叉 = MACD/KDJ 金叉 T+1 买、死叉 T+1 卖。
- 成本往返 0.15%；不可成交则顺延至最多 8 个交易日，仍不可成交则放弃。
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
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
COST = 0.0015
THRESHOLDS = (0.05, 0.08, 0.10, 0.15)


def fill_open(P, i, ci, side):
    """从 i 日起找第一个可成交开盘；买避开一字涨停，卖避一字跌停。"""
    n = P["shape"][0]
    for d in range(i, min(n, i + 9)):
        if P["trade"][d, ci] != 1 or not np.isfinite(P["open"][d, ci]):
            continue
        one = (P["raw_high"][d, ci] == P["raw_low"][d, ci])
        up = one and (P["raw_open"][d, ci] > P["preclose"][d, ci])
        dn = one and (P["raw_open"][d, ci] < P["preclose"][d, ci])
        if side == "buy" and up:
            continue
        if side == "sell" and dn:
            continue
        return d
    return -1


def build_pairs(P, pv, th):
    low = pv[(pv["th"] == th) & (pv["type"] == -1)].copy()
    high = pv[(pv["th"] == th) & (pv["type"] == 1)][
        ["ci", "pivot_i", "confirm_i", "price"]].rename(
        columns={"pivot_i": "hpivot_i", "confirm_i": "hconfirm_i", "price": "hprice"})
    d = low.merge(high, left_on=["ci", "next_pivot"], right_on=["ci", "hpivot_i"],
                  how="left", suffixes=("", "_h"))
    # 没有下一个高点：持有到该股最后一个有效日，按最后有效开盘强制退出
    last_valid = P["valid"].sum(axis=0) - 1
    d["forced"] = d["hconfirm_i"].isna()
    d["last_valid"] = last_valid[d["ci"].to_numpy()]
    d["ideal"] = np.where(d["forced"], np.nan, d["hprice"] / d["price"] - 1.0)
    d["entry_sig"] = d["confirm_i"] + 1
    d["exit_sig"] = np.where(d["forced"], d["last_valid"], d["hconfirm_i"] + 1)
    d = d[(d["ideal"].isna()) | (d["ideal"] > 0)].copy()
    d = d[d["exit_sig"] > d["entry_sig"]].copy()
    return d


def add_fills(P, d):
    ent = np.array([fill_open(P, int(i), int(ci), "buy")
                    for i, ci in zip(d["entry_sig"], d["ci"])])
    ext = np.array([fill_open(P, int(i), int(ci), "sell")
                    for i, ci in zip(d["exit_sig"], d["ci"])])
    d = d.copy()
    d["entry_i"] = ent
    d["exit_i"] = ext
    # 退出日不可成交：用最后有效开盘强制退出（退市/长期停牌）
    bad = (d["exit_i"] < 0)
    if bad.any():
        lv = d.loc[bad, "last_valid"].to_numpy()
        d.loc[bad, "exit_i"] = lv
        d.loc[bad, "forced"] = True
    d = d[(d["entry_i"] >= 0) & (d["exit_i"] > d["entry_i"])].copy()
    d["hold_days"] = d["exit_i"] - d["entry_i"]
    d["ret_gross"] = (P["open"][d["exit_i"].to_numpy(), d["ci"].to_numpy()] /
                      P["open"][d["entry_i"].to_numpy(), d["ci"].to_numpy()] - 1.0)
    d["ret_net"] = (1.0 + d["ret_gross"]) * (1.0 - COST) - 1.0
    d["capture"] = d["ret_gross"] / d["ideal"]
    return d


def add_states(P, d, sig, kd, ma20, ma200):
    ci = d["ci"].to_numpy()
    c = d["confirm_i"].to_numpy()
    p = d["pivot_i"].to_numpy()
    d = d.copy()
    d["MACD确认多"] = sig["MACD多头"][c, ci]
    d["KDJ确认多"] = sig["KDJ多头"][c, ci]
    d["MACD拐点多"] = sig["MACD多头"][p, ci]
    d["KDJ拐点多"] = sig["KDJ多头"][p, ci]
    d["价格>MA20"] = P["close_ff"][c, ci] > ma20[c, ci]
    d["价格>MA200"] = P["close_ff"][c, ci] > ma200[c, ci]
    d["K<30确认"] = kd["k"][c, ci] < 30
    d["K<20确认"] = kd["k"][c, ci] < 20
    return d


def summarize(d, label, th):
    if len(d) == 0:
        return dict(阈值=th, 方案=label, 交易数=0, 年交易数=0.0, 平均持有=np.nan,
                    平均理想波段=np.nan, 中位理想波段=np.nan, 平均收益=np.nan,
                    中位收益=np.nan, 胜率=np.nan, 盈亏比=np.nan, 平均捕获=np.nan,
                    中位捕获=np.nan, 捕获为正=np.nan)
    r = d["ret_net"].to_numpy()
    ideal = d["ideal"].to_numpy()
    cap = d["capture"].to_numpy()
    cap = cap[np.isfinite(cap)]
    ideal = ideal[np.isfinite(ideal)]
    up = r[r > 0]
    dn = r[r < 0]
    return dict(
        阈值=th, 方案=label, 交易数=len(d),
        年交易数=len(d) / 17.0,
        强制退出占比=float(d["forced"].mean()),
        平均持有=float(np.mean(d["hold_days"])),
        平均理想波段=float(np.mean(ideal)) if len(ideal) else np.nan,
        中位理想波段=float(np.median(ideal)) if len(ideal) else np.nan,
        平均收益=float(np.mean(r)),
        中位收益=float(np.median(r)),
        胜率=float(np.mean(r > 0)),
        盈亏比=float(up.sum() / max(-dn.sum(), 1e-9)),
        平均捕获=float(np.mean(cap)) if len(cap) else np.nan,
        中位捕获=float(np.median(cap)) if len(cap) else np.nan,
        捕获为正=float(np.mean(cap > 0)) if len(cap) else np.nan,
    )


def monthly_diff(d, flag):
    """按入场月份计算 选中-未选中 的平均收益差，NW t。"""
    x = d.copy()
    x["month"] = pd.to_datetime(P_dates[x["entry_i"].to_numpy()]).to_period("M")
    a = x[x[flag]].groupby("month")["ret_net"].mean()
    b = x[~x[flag]].groupby("month")["ret_net"].mean()
    diff = (a - b).dropna()
    return float(diff.mean()) if len(diff) else np.nan, float(nw_t(diff, 6)) if len(diff) else np.nan


def trade_stats(P, entry, exit_, ci, label):
    ret = P["open"][exit_, ci] / P["open"][entry, ci] - 1.0
    ret = (1.0 + ret) * (1.0 - COST) - 1.0
    return pd.DataFrame(dict(方案=label, ci=ci, entry_i=entry, exit_i=exit_,
                             hold_days=exit_ - entry, ret_net=ret))


def classic_trades(P, sig, name_gold, name_death, label):
    """金叉 T+1 买、死叉 T+1 卖；逐股配对。"""
    out = []
    n, nc = P["shape"]
    gold, death = sig[name_gold], sig[name_death]
    for j in range(nc):
        gd = np.where(gold[:, j])[0]
        dd = np.where(death[:, j])[0]
        if len(gd) == 0 or len(dd) == 0:
            continue
        lastv = int(np.where(P["valid"][:, j])[0][-1])
        for g in gd:
            # 先找 g 之后第一个死叉；没有死叉则持有到最后一个有效日
            k = np.searchsorted(dd, g, side="right")
            x_sig = dd[k] + 1 if k < len(dd) else lastv
            e = fill_open(P, g + 1, j, "buy")
            x = fill_open(P, x_sig, j, "sell")
            if x < 0:
                x = lastv
            if e < 0 or x <= e:
                continue
            out.append((j, e, x))
    if not out:
        return pd.DataFrame()
    a = np.array(out)
    return trade_stats(P, a[:, 1].astype(int), a[:, 2].astype(int),
                       a[:, 0].astype(int), label)


def timing_trades(P, pairs, sig, entry_state, exit_state, label):
    """低点确认后等状态转多买；高点确认后等状态转空卖（允许拖到下一段）。"""
    out = []
    ent_s, ext_s = entry_state, exit_state
    n = P["shape"][0]
    for r in pairs.itertuples():
        c = int(r.confirm_i)
        lastv = int(r.last_valid)
        # 入场：确认日或之后第一次状态为真；不晚于高点枢轴/最后有效日
        hp = lastv if bool(r.forced) else min(int(r.hpivot_i), lastv)
        a = np.where(ent_s[c:hp + 1, r.ci])[0]
        if len(a) == 0:
            continue
        e_sig = c + int(a[0])
        # 出场：高点确认日或之后第一次状态为假；强制退出则到最后有效日
        if bool(r.forced):
            x_sig = lastv
        else:
            b = np.where(ext_s[int(r.hconfirm_i):lastv + 1, r.ci])[0]
            x_sig = int(r.hconfirm_i) + int(b[0]) if len(b) else lastv
        e = fill_open(P, e_sig + 1, r.ci, "buy")
        x = fill_open(P, x_sig + 1, r.ci, "sell")
        if e < 0 or x <= e:
            continue
        out.append((r.ci, e, x))
    if not out:
        return pd.DataFrame()
    a = np.array(out)
    df = trade_stats(P, a[:, 1].astype(int), a[:, 2].astype(int),
                     a[:, 0].astype(int), label)
    return df


def portfolio(P, base, tr, label):
    """等权多标的组合；持仓从入场开盘到出场开盘。"""
    n, nc = P["shape"]
    pos = np.zeros((n, nc), dtype=np.int8)
    entries = np.zeros(n, dtype=np.int32)
    exits = np.zeros(n, dtype=np.int32)
    for r in tr.itertuples():
        pos[r.entry_i:r.exit_i, r.ci] = 1
        entries[r.entry_i] += 1
        exits[r.exit_i] += 1
    r_oo = P["r_oo"]
    active = pos.astype(bool)
    cnt = active.sum(axis=1).astype(float)
    s = np.nansum(np.where(active, r_oo, 0.0), axis=1)
    daily = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    # 成本：入场日与出场日按当日持仓数摊
    daily -= np.where(cnt > 0, 0.00075 * entries / np.maximum(cnt, 1), 0.0)
    # 出场日在持仓中已不计，用前一日持仓数近似
    prev = np.concatenate([[0], cnt[:-1]])
    daily -= np.where(prev > 0, 0.00075 * exits / np.maximum(prev, 1), 0.0)
    daily = pd.Series(daily, index=P["dates"])
    daily = daily.loc["2010-01-01":]
    nav = (1 + daily).cumprod()
    ann = nav.iloc[-1] ** (252.0 / len(daily)) - 1 if len(daily) and nav.iloc[-1] > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    return dict(方案=label, 年化=ann, 最大回撤=mdd, 夏普=sharpe,
                日均持仓=float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 0.0,
                在场天数=int((cnt > 0).sum()))


P_dates = None


def main():
    global P_dates
    t0 = time.time()
    P = S.load_base()
    P_dates = P["dates"]
    base = P["universe_inv"]
    macd = F.build_macd_features(P)
    kd = K.kdj_features(P)
    sig = {
        "MACD金叉": M.crosses(macd["dif"], macd["dea"], P)[0],
        "MACD死叉": M.crosses(macd["dif"], macd["dea"], P)[1],
        "MACD多头": (macd["dif"] > macd["dea"]) & P["valid"],
        "MACD空头": (macd["dif"] <= macd["dea"]) & P["valid"],
        "KDJ金叉": kd["gold"], "KDJ死叉": kd["death"],
        "KDJ多头": (kd["k"] > kd["d"]) & P["valid"],
        "KDJ空头": (kd["k"] <= kd["d"]) & P["valid"],
        "KDJ低位金叉": kd["gold"] & (kd["k"] < 30),
    }
    c = P["close_ff"]
    ma20 = M.rolling_mean(c, 20)
    ma200 = M.rolling_mean(c, 200)
    pv = pd.read_parquet(os.path.join(RES, "59_波段_拐点.parquet"))
    print("loaded", P["shape"], "pivots", len(pv), "%.1fs" % (time.time() - t0))

    rows = []
    pairs_all = {}
    for th in THRESHOLDS:
        d = build_pairs(P, pv, th)
        d = add_fills(P, d)
        d = add_states(P, d, sig, kd, ma20, ma200)
        pairs_all[th] = d
        print("th", th, "pairs", len(d), "%.1fs" % (time.time() - t0))

        rows.append(summarize(d, "价格确认", th))
        rows.append(summarize(d[d["MACD确认多"]], "MACD确认多", th))
        rows.append(summarize(d[d["KDJ确认多"]], "KDJ确认多", th))
        rows.append(summarize(d[d["MACD确认多"] & d["KDJ确认多"]], "双确认多", th))
        rows.append(summarize(d[d["MACD拐点多"]], "MACD拐点多", th))
        rows.append(summarize(d[d["KDJ拐点多"]], "KDJ拐点多", th))
        rows.append(summarize(d[d["价格>MA20"]], "价格>MA20", th))
        rows.append(summarize(d[d["价格>MA200"]], "价格>MA200", th))
        rows.append(summarize(d[d["K<30确认"]], "K<30确认", th))
        rows.append(summarize(d[d["K<20确认"]], "K<20确认", th))

        # 择时
        tt = timing_trades(P, d, sig, sig["MACD多头"], sig["MACD空头"], "MACD择时")
        if len(tt):
            r = tt["ret_net"].to_numpy()
            up, dn = r[r > 0], r[r < 0]
            rows.append(dict(阈值=th, 方案="MACD择时", 交易数=len(tt),
                             年交易数=len(tt) / 17.0,
                             平均持有=float(tt["hold_days"].mean()),
                             平均理想波段=np.nan, 中位理想波段=np.nan,
                             平均收益=float(r.mean()), 中位收益=float(np.median(r)),
                             胜率=float((r > 0).mean()),
                             盈亏比=float(up.sum() / max(-dn.sum(), 1e-9)),
                             平均捕获=np.nan, 中位捕获=np.nan, 捕获为正=np.nan))
        tt = timing_trades(P, d, sig, sig["KDJ多头"], sig["KDJ空头"], "KDJ择时")
        if len(tt):
            r = tt["ret_net"].to_numpy()
            up, dn = r[r > 0], r[r < 0]
            rows.append(dict(阈值=th, 方案="KDJ择时", 交易数=len(tt),
                             年交易数=len(tt) / 17.0,
                             平均持有=float(tt["hold_days"].mean()),
                             平均理想波段=np.nan, 中位理想波段=np.nan,
                             平均收益=float(r.mean()), 中位收益=float(np.median(r)),
                             胜率=float((r > 0).mean()),
                             盈亏比=float(up.sum() / max(-dn.sum(), 1e-9)),
                             平均捕获=np.nan, 中位捕获=np.nan, 捕获为正=np.nan))

        if th == 0.10:
            for flag, lab in [("MACD确认多", "MACD确认多"), ("KDJ确认多", "KDJ确认多"),
                              ("价格>MA20", "价格>MA20")]:
                m, t = monthly_diff(d, flag)
                print("  diff", lab, "%.4f t=%.2f" % (m, t))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "60_波段_捕获.csv"), index=False, encoding="utf-8-sig")
    print(out[out["阈值"] == 0.10].to_string(index=False))

    # 经典交叉（全样本）
    print("classic ... %.1fs" % (time.time() - t0))
    classic = []
    for name, g, x in [("MACD交叉", "MACD金叉", "MACD死叉"),
                       ("KDJ交叉", "KDJ金叉", "KDJ死叉")]:
        tr = classic_trades(P, sig, g, x, name)
        if len(tr):
            r = tr["ret_net"].to_numpy()
            up, dn = r[r > 0], r[r < 0]
            classic.append(dict(方案=name, 交易数=len(tr), 年交易数=len(tr) / 17.0,
                                平均持有=float(tr["hold_days"].mean()),
                                平均收益=float(r.mean()), 中位收益=float(np.median(r)),
                                胜率=float((r > 0).mean()),
                                盈亏比=float(up.sum() / max(-dn.sum(), 1e-9))))
            tr.to_parquet(os.path.join(RES, "60_%s_交易.parquet" % name))
    pd.DataFrame(classic).to_csv(os.path.join(RES, "60_经典交叉.csv"),
                                 index=False, encoding="utf-8-sig")
    print(pd.DataFrame(classic).to_string(index=False))

    # 组合：阈值 10% + 5%
    port = []
    bench_daily = np.nanmean(np.where(base, P["r_oo"], np.nan), axis=1)
    bench = pd.Series(bench_daily, index=P["dates"]).loc["2010-01-01":].fillna(0.0)
    nav = (1 + bench).cumprod()
    port.append(dict(方案="全A等权", 年化=nav.iloc[-1] ** (252.0 / len(bench)) - 1,
                     最大回撤=(nav / nav.cummax() - 1).min(),
                     夏普=bench.mean() / bench.std() * np.sqrt(252),
                     日均持仓=np.nan, 在场天数=len(bench)))
    for th in (0.10, 0.05):
        d = pairs_all[th]
        variants = [("价格确认", d),
                    ("MACD确认多", d[d["MACD确认多"]]),
                    ("KDJ确认多", d[d["KDJ确认多"]]),
                    ("双确认多", d[d["MACD确认多"] & d["KDJ确认多"]]),
                    ("价格>MA20", d[d["价格>MA20"]])]
        for lab, dd in variants:
            tr = pd.DataFrame(dict(ci=dd["ci"], entry_i=dd["entry_i"], exit_i=dd["exit_i"]))
            if len(tr):
                port.append(portfolio(P, base, tr, "%s_%.0f%%" % (lab, th * 100)))
    pd.DataFrame(port).to_csv(os.path.join(RES, "60_波段_组合.csv"),
                              index=False, encoding="utf-8-sig")
    print(pd.DataFrame(port).to_string(index=False))
    print("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
