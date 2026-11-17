#!/usr/bin/env python3
"""45 条件有效性：把 MACD 特征当作“判断条件”，而不是选股因子

问题
----
不是“用 MACD 选哪些股票”，而是：
给定今天的 MACD 特征状态（零轴、柱体四态、交叉年龄、柱体/DIF 位置、
斜率、背离、周/月共振、证据分数、市场状态），MACD 说多或说空，
未来 20 日到底对不对？

口径
----
- 全 A 日线，2010-2026；剔除 ST、上市不足 120 日、不可成交。
- 判断：T 日收盘 DIF>DEA 为“多”，否则为“空”。
- 结果：fwd20 = close[T+20]/open[T+1]-1；
  ex20 = fwd20 - 同日全市场等权同口径收益（leave-one-out）。
- 有效：多头 ex20 > 0.15%（覆盖往返成本）且 t>2 且前后两段同为正；
  空头有效：空头 ex20 < -0.15% 且 t<-2 且前后两段同为负（回避有效）。
- t：日度横截面均值序列，Newey-West lag=20（20 日重叠）。
- 增量：同一条件内，MACD 多空差 - 价格>MA60 多空差；正数才算 MACD 有独立判断力。

输出
----
03_结果/条件_多头.csv、条件_空头.csv、条件_多空差.csv、
条件_证据分数.csv、条件_事件.csv、条件_经典过滤.csv、
条件_逐年.csv、条件_市场状态.csv
"""
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
COST = 0.0015
EARLY_END = pd.Timestamp("2017-12-31")
LATE_START = pd.Timestamp("2018-01-01")
MIN_N = 50
MIN_CELL_N = 3000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def daily_sel(sel, ex, min_n=MIN_N):
    """条件子集的日度横截面均值序列。"""
    n = sel.sum(axis=1)
    s = np.nansum(np.where(sel, ex, 0.0), axis=1)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan), n


def cell_stat(sel, ex, dates):
    n = int(sel.sum())
    if n == 0:
        return dict(股票日=0, 交易日=0, 日均超额=np.nan, 中位超额=np.nan,
                    t值=np.nan, 正日占比=np.nan, 前期=np.nan, 后期=np.nan)
    dm, _ = daily_sel(sel, ex)
    ok = np.isfinite(dm)
    if ok.sum() < 20:
        return dict(股票日=n, 交易日=int(ok.sum()), 日均超额=np.nan, 中位超额=np.nan,
                    t值=np.nan, 正日占比=np.nan, 前期=np.nan, 后期=np.nan)
    early = ok & (dates <= EARLY_END)
    late = ok & (dates >= LATE_START)
    return dict(
        股票日=n, 交易日=int(ok.sum()),
        日均超额=float(np.nanmean(dm)),
        中位超额=float(np.nanmedian(ex[sel])),
        t值=float(nw_t(dm, lag=20)),
        正日占比=float(np.nanmean(dm[ok] > 0)),
        前期=float(np.nanmean(dm[early])) if early.any() else np.nan,
        后期=float(np.nanmean(dm[late])) if late.any() else np.nan,
    )


def verdict_long(m):
    if m["股票日"] < MIN_CELL_N or m["交易日"] < 150:
        return "样本不足"
    if (m["日均超额"] > COST and m["t值"] > 2 and m["前期"] > 0 and m["后期"] > 0):
        return "有效"
    if (m["日均超额"] < -COST and m["t值"] < -2 and m["前期"] < 0 and m["后期"] < 0):
        return "反向"
    return "无效"


def verdict_short(m):
    if m["股票日"] < MIN_CELL_N or m["交易日"] < 150:
        return "样本不足"
    if (m["日均超额"] < -COST and m["t值"] < -2 and m["前期"] < 0 and m["后期"] < 0):
        return "有效(回避)"
    if (m["日均超额"] > COST and m["t值"] > 2 and m["前期"] > 0 and m["后期"] > 0):
        return "反向(该涨)"
    return "无效"


def q_marks(x, base, nq=5):
    q = F.row_quantile(x, base, nq)
    return q


def q_name(k, nq=5, low="最低", high="最高"):
    return f"Q{k + 1}({low if k == 0 else high if k == nq - 1 else k + 1})"


def add_cond(conds, fam, label, mask):
    conds.append((fam, label, mask))


def main():
    t0 = time.time()
    log("load panel")
    P = M.load_panel()
    dates, codes = P["dates"], P["codes"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)

    log("forward returns")
    fwd, ok, mkt, loo = M.forward_returns(P, (20,))
    ex = (fwd[20] - loo[20]).astype(np.float32)
    mask = base & ok[20] & np.isfinite(ex)
    bull = mask & macd["states"]["DIF>DEA"]
    bear = mask & ~macd["states"]["DIF>DEA"]

    c = P["close_ff"]
    dif = macd["dif"]
    dea = macd["dea"]
    hist = macd["hist"]
    dif_pct = macd["feats"]["DIF水平"]
    hist_pct = macd["feats"]["柱体水平"]
    dif_slope5 = macd["feats"]["DIF5日斜率"]
    hist_z60 = macd["feats"]["柱体60日z"]
    red_share20 = macd["feats"]["20日红柱占比"]
    cross_age = macd["feats"]["交叉后天数"]

    # 周/月状态（已收盘、滞后一期）
    log("weekly/monthly states")
    wf = M.weekly_state(P)
    P11 = importlib.import_module("11_周期研究")
    perM = P11.build_period(P, "M")
    difM, deaM, _, _, _, _ = P11.period_signals(P, perM, 12, 26, 9, 36)
    stateM = (difM > deaM)
    mend = dates[perM["end"]]
    posM = np.searchsorted(mend.values, dates.values, side="left") - 1
    m_on_daily = np.full(P["shape"], np.nan, dtype=np.float32)
    for t in range(len(dates)):
        p = posM[t]
        if p >= 0:
            m_on_daily[t, :] = stateM[p, :]
    w_ok = np.isfinite(wf["w_dif"])
    m_ok = np.isfinite(m_on_daily)
    w_above = wf["w_above"].astype(bool) & w_ok
    m_above = (m_on_daily > 0.5) & m_ok

    # 证据分数：6 项多头证据
    log("confirmation score")
    comps = [
        (dif > 0),
        (dif > dea),
        (hist > M.shift(hist, 1)),
        (dif > M.shift(dif, 1)),
        w_above,
        m_above,
    ]
    score = np.zeros(P["shape"], dtype=np.int8)
    for x in comps:
        score += x.astype(np.int8)
    score_ok = base & w_ok & m_ok
    score[~score_ok] = -1

    # 背离：价格创 20 日新高/新低，但 DIF 未同步
    high20 = M.rolling_max(c, 20)
    low20 = M.rolling_min(c, 20)
    dmax20 = M.rolling_max(dif_pct, 20)
    dmin20 = M.rolling_min(dif_pct, 20)
    top_div = mask & (c >= high20 - 1e-6) & (dif_pct < dmax20 - 1e-9)
    bot_div = mask & (c <= low20 + 1e-6) & (dif_pct > dmin20 + 1e-9)

    ma60 = st["ma60"]
    ma60b = mask & (c > ma60)
    ma60s = mask & (c <= ma60)

    # 条件表
    conds = []
    # 1 零轴四态
    zs = np.full(P["shape"], -1, dtype=np.int8)
    zs[(dif > 0) & (dea > 0)] = 0
    zs[(dif > 0) & (dea <= 0)] = 1
    zs[(dif <= 0) & (dea > 0)] = 2
    zs[(dif <= 0) & (dea <= 0)] = 3
    zs[~mask] = -1
    for k, lab in enumerate(["双上(DIF>0,DEA>0)", "DIF上DEA下", "DIF下DEA上", "双下(DIF<=0,DEA<=0)"]):
        add_cond(conds, "零轴四态", lab, zs == k)
    # 2 柱体四态
    hs = macd["hist_state"]
    for k, lab in enumerate(["红柱放大", "红柱缩小", "绿柱缩短", "绿柱放大"]):
        add_cond(conds, "柱体四态", lab, mask & (hs == k))
    # 3 交叉年龄
    age = np.abs(cross_age)
    for lo, hi, lab in [(0, 5, "距交叉0-5日"), (6, 15, "6-15日"), (16, 40, "16-40日"),
                        (41, 120, "41-120日"), (121, 9999, "120日以上")]:
        add_cond(conds, "交叉年龄", lab, mask & (age >= lo) & (age <= hi))
    # 4-9 连续特征五分位
    # 柱体水平 = 2×(DIF-DEA)/价格，与“DIF-DEA距离”完全同序，只保留一个，避免重复检验。
    conts = [
        ("柱体水平", hist_pct, "柱体水平"),
        ("DIF水平", dif_pct, "DIF水平"),
        ("DIF5日斜率", dif_slope5, "DIF5日斜率"),
        ("柱体60日z", hist_z60, "柱体60日z"),
        ("20日红柱占比", red_share20, "20日红柱占比"),
    ]
    qcache = {}
    for fam, x, _ in conts:
        q = q_marks(x, base)
        qcache[fam] = q
        for k in range(5):
            add_cond(conds, fam, q_name(k), mask & (q == k))
    # 10 背离
    add_cond(conds, "背离", "顶背离(价新高DIF未新高)", top_div)
    add_cond(conds, "背离", "底背离(价新低DIF未新低)", bot_div)
    add_cond(conds, "背离", "无背离", mask & ~top_div & ~bot_div)
    # 11 周/月状态
    for wk, wl in [(True, "周线多头"), (False, "周线空头")]:
        for mw, ml in [(True, "月线多头"), (False, "月线空头")]:
            sel = mask & w_ok & m_ok & (w_above if wk else ~w_above) & (m_above if mw else ~m_above)
            add_cond(conds, "周月状态", f"{wl}+{ml}", sel)
    # 12 证据分数
    for s in range(7):
        add_cond(conds, "证据分数", f"{s}项", mask & (score == s))
    # 13 市场状态（注意不要用 mk 这个名字，事件层会用它循环月线）
    mkt_state = st["market"]
    for k, lab in [(0, "指数在MA200上且上行"), (1, "指数在MA200下且下行"), (2, "指数震荡")]:
        add_cond(conds, "市场状态", lab, mask & (mkt_state == k))

    log(f"conditions {len(conds)}")
    long_rows, short_rows, diff_rows = [], [], []
    for fam, lab, cm in conds:
        mb = cm & bull
        ms = cm & bear
        lb = cell_stat(mb, ex, dates)
        ls = cell_stat(ms, ex, dates)
        dmb, _ = daily_sel(mb, ex)
        dms, _ = daily_sel(ms, ex)
        sp = dmb - dms
        common = np.isfinite(sp)
        sp_mean = float(np.nanmean(sp[common])) if common.any() else np.nan
        sp_t = float(nw_t(sp[common], lag=20)) if common.sum() >= 20 else np.nan
        bull_common = float(np.nanmean(dmb[common])) if common.any() else np.nan
        bear_common = float(np.nanmean(dms[common])) if common.any() else np.nan
        # 相对 MA60 的增量：只用四个序列都有的同一批交易日
        dmmb, _ = daily_sel(cm & ma60b, ex)
        dmms, _ = daily_sel(cm & ma60s, ex)
        ma_sp = dmmb - dmms
        inc = sp - ma_sp
        ci = np.isfinite(inc)
        inc_mean = float(np.nanmean(inc[ci])) if ci.any() else np.nan
        inc_t = float(nw_t(inc[ci], lag=20)) if ci.sum() >= 20 else np.nan
        ma_common = float(np.nanmean(ma_sp[ci])) if ci.any() else np.nan
        long_rows.append(dict(条件族=fam, 条件=lab, **lb, 判定=verdict_long(lb),
                              相对MA60增量=inc_mean, 增量t=inc_t))
        short_rows.append(dict(条件族=fam, 条件=lab, **ls, 判定=verdict_short(ls)))
        diff_rows.append(dict(条件族=fam, 条件=lab, 多头日均=bull_common,
                              空头日均=bear_common, 多空差=sp_mean, 多空差t=sp_t,
                              共同交易日=int(common.sum()),
                              MA60多空差=ma_common,
                              MACD增量=inc_mean, 增量t=inc_t,
                              增量共同交易日=int(ci.sum()),
                              多头股票日=lb["股票日"], 空头股票日=ls["股票日"]))
        log(f"  {fam:8s} {lab:22s} 多 {bull_common*100:+.3f}% t={lb['t值']:+.1f} "
            f"空 {bear_common*100:+.3f}% 差 {sp_mean*100:+.3f}% 增量 {inc_mean*100:+.3f}%")

    ld = pd.DataFrame(long_rows)
    sd = pd.DataFrame(short_rows)
    dd = pd.DataFrame(diff_rows)

    # 两道门槛：绝对有效（扣成本、t>2、前后同号）与增量有效（减 MA60 后仍 t>2）
    def inc_verdict(r):
        v, t = r["相对MA60增量"], r["增量t"]
        if not np.isfinite(v) or not np.isfinite(t):
            return "不可比"
        if v > 0 and t > 2:
            return "有增量"
        if v < 0 and t < -2:
            return "负增量"
        return "无显著增量"

    def combo_verdict(r):
        if r["判定"] == "有效" and r["增量判定"] == "有增量":
            return "有效"
        if r["判定"] == "有效":
            return "仅价格趋势"
        if r["增量判定"] == "有增量":
            return "仅相对有效"
        return r["判定"]

    ld["增量判定"] = ld.apply(inc_verdict, axis=1)
    ld["综合判定"] = ld.apply(combo_verdict, axis=1)
    # 多重检验：对多头表的 t 值做 BH-FDR（同表全部条件）
    from scipy.stats import norm as _norm
    from stats_utils import _bh
    pv = 2 * _norm.sf(np.abs(ld["t值"].to_numpy(dtype=float)))
    ld["q值"] = _bh(pv)
    ld.to_csv(os.path.join(RES, "条件_多头.csv"), index=False, encoding="utf-8-sig")
    sd.to_csv(os.path.join(RES, "条件_空头.csv"), index=False, encoding="utf-8-sig")
    dd.to_csv(os.path.join(RES, "条件_多空差.csv"), index=False, encoding="utf-8-sig")
    log(f"state tables {ld.shape}")
    log("综合判定:\n" + ld["综合判定"].value_counts().to_string())

    # 每个条件族的最优/最差/最大增量
    fam_rows = []
    for fam, g in ld.groupby("条件族", sort=False):
        b = g.loc[g["日均超额"].idxmax()]
        w = g.loc[g["日均超额"].idxmin()]
        inc = g.dropna(subset=["相对MA60增量"])
        bi = inc.loc[inc["相对MA60增量"].idxmax()] if len(inc) else None
        fam_rows.append(dict(
            条件族=fam,
            最优多头条件=b["条件"], 最优多头超额=b["日均超额"], 最优多头t=b["t值"], 最优多头q=b["q值"],
            最差多头条件=w["条件"], 最差多头超额=w["日均超额"], 最差多头t=w["t值"],
            最大增量条件=bi["条件"] if bi is not None else "",
            最大增量=bi["相对MA60增量"] if bi is not None else np.nan,
            增量t=bi["增量t"] if bi is not None else np.nan,
            绝对有效格数=int((g["判定"] == "有效").sum()),
            有增量格数=int((g["增量判定"] == "有增量").sum()),
            综合有效格数=int((g["综合判定"] == "有效").sum()),
        ))
    pd.DataFrame(fam_rows).to_csv(os.path.join(RES, "条件_族汇总.csv"), index=False, encoding="utf-8-sig")

    # 证据分数专门表
    sc = ld[ld.条件族 == "证据分数"].copy()
    sc["分数"] = sc["条件"].str.replace("项", "", regex=False).astype(int)
    sc = sc.sort_values("分数")
    sc.to_csv(os.path.join(RES, "条件_证据分数.csv"), index=False, encoding="utf-8-sig")

    # 市场状态表
    mk_tab = ld[ld.条件族 == "市场状态"].copy()
    mk_tab.to_csv(os.path.join(RES, "条件_市场状态.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 事件层 ----------------
    log("event layer")
    e = pd.read_parquet(os.path.join(RES, "事件_金叉_12269.parquet"))
    e = e[np.isfinite(e.ex20)].copy()
    di = pd.Index(dates).get_indexer(e.date)
    ci = pd.Index(codes).get_indexer(e.code)
    good = (di >= 0) & (ci >= 0)
    e = e[good].copy()
    di, ci = di[good], ci[good]
    e["di"], e["ci"] = di, ci
    e["age"] = age[di, ci]
    e["score"] = score[di, ci]
    e["zero"] = zs[di, ci]
    e["gapq"] = qcache["柱体水平"][di, ci]
    e["difq"] = qcache["DIF水平"][di, ci]
    e["histq"] = qcache["柱体水平"][di, ci]
    e["top_div"] = top_div[di, ci]
    e["bot_div"] = bot_div[di, ci]
    e["w"] = w_above[di, ci]
    e["m"] = m_above[di, ci]
    e["mkt"] = st["market"][di, ci]
    e["ma60b"] = ma60b[di, ci]
    e["净超额"] = e["ex20"] - COST

    def ev_stat(sub):
        sub = sub.dropna(subset=["ex20"])
        if len(sub) < 100:
            return dict(信号数=len(sub), 交易日=0, 毛超额=np.nan, 净超额=np.nan,
                        中位=np.nan, t值=np.nan, 正日占比=np.nan, 前期=np.nan, 后期=np.nan)
        dm = sub.groupby("date")["ex20"].mean()
        early = dm[dm.index <= EARLY_END]
        late = dm[dm.index >= LATE_START]
        return dict(
            信号数=len(sub), 交易日=len(dm),
            毛超额=float(dm.mean()), 净超额=float(dm.mean() - COST),
            中位=float(sub["ex20"].median()),
            t值=float(nw_t(dm, lag=20)), 正日占比=float((dm > 0).mean()),
            前期=float(early.mean()) if len(early) else np.nan,
            后期=float(late.mean()) if len(late) else np.nan,
        )

    ev_rows = []
    # 金叉当日交叉年龄恒为 0，年龄维度只在状态层有意义。
    fam_defs = [
        ("零轴四态", "zero", {0: "双上", 1: "DIF上DEA下", 2: "DIF下DEA上", 3: "双下"}),
        ("证据分数", "score", {k: f"{k}项" for k in range(7)}),
        ("DIF水平", "difq", {k: q_name(k) for k in range(5)}),
        ("柱体水平", "histq", {k: q_name(k) for k in range(5)}),
        ("周月状态", None, {}),
        ("背离", None, {}),
        ("市场状态", "mkt", {0: "指数在MA200上且上行", 1: "指数在MA200下且下行", 2: "指数震荡"}),
    ]
    for fam, col, mapping in fam_defs:
        if fam == "周月状态":
            for wk, wl in [(True, "周线多头"), (False, "周线空头")]:
                for mw, ml in [(True, "月线多头"), (False, "月线空头")]:
                    sub = e[(e.w == wk) & (e.m == mw)]
                    if len(sub) == 0:
                        continue
                    stt = ev_stat(sub)
                    ev_rows.append(dict(条件族=fam, 条件=f"{wl}+{ml}", **stt))
            continue
        if fam == "背离":
            for flag, lab in [("top_div", "顶背离"), ("bot_div", "底背离")]:
                sub = e[e[flag].astype(bool)]
                if len(sub) == 0:
                    continue
                ev_rows.append(dict(条件族=fam, 条件=lab, **ev_stat(sub)))
            sub = e[~e.top_div.astype(bool) & ~e.bot_div.astype(bool)]
            ev_rows.append(dict(条件族=fam, 条件="无背离", **ev_stat(sub)))
            continue
        for k, lab in mapping.items():
            sub = e[e[col] == k]
            if len(sub) == 0:
                continue
            ev_rows.append(dict(条件族=fam, 条件=lab, **ev_stat(sub)))
    ev = pd.DataFrame(ev_rows)
    # 判定：净超额 >0 且 t>2 且前后同号
    def ev_verdict(r):
        if r["信号数"] < 2000 or r["交易日"] < 150:
            return "样本不足"
        if r["净超额"] > 0 and r["t值"] > 2 and r["前期"] > 0 and r["后期"] > 0:
            return "有效"
        if r["净超额"] < 0 and r["t值"] < -2 and r["前期"] < 0 and r["后期"] < 0:
            return "反向"
        return "无效"
    ev["判定"] = ev.apply(ev_verdict, axis=1)
    ev.to_csv(os.path.join(RES, "条件_事件.csv"), index=False, encoding="utf-8-sig")
    log(f"event table {ev.shape}")

    # 经典过滤：事件层
    classic = []
    def add_classic(lab, sel):
        classic.append(dict(过滤=lab, **ev_stat(e[sel])))
    add_classic("全部金叉(基准)", np.ones(len(e), bool))
    add_classic("零轴上金叉(双上)", e.zero == 0)
    add_classic("零轴下金叉(双下)", e.zero == 3)
    add_classic("周线多头", e.w.astype(bool))
    add_classic("月线多头", e.m.astype(bool))
    add_classic("周线多头+月线多头", e.w.astype(bool) & e.m.astype(bool))
    add_classic("证据分数<=1", e.score <= 1)
    add_classic("证据分数>=4", e.score >= 4)
    add_classic("顶背离", e.top_div.astype(bool))
    add_classic("底背离", e.bot_div.astype(bool))
    add_classic("DIF-DEA距离Q1(最窄)", e.gapq == 0)
    add_classic("DIF-DEA距离Q5(最宽)", e.gapq == 4)
    cl = pd.DataFrame(classic)
    cl["净超额"] = cl["毛超额"] - COST
    cl.to_csv(os.path.join(RES, "条件_经典过滤.csv"), index=False, encoding="utf-8-sig")
    log(f"classic {cl.shape}")

    # ---------------- 逐年稳定性 ----------------
    log("yearly")
    year = dates.year
    key_masks = [
        ("全部多头", bull),
        ("零轴双上", bull & (zs == 0)),
        ("零轴双下", bull & (zs == 3)),
        ("红柱放大", bull & (hs == 0)),
        ("绿柱缩短", bear & (hs == 2)),
        ("证据分数<=1", bull & (score <= 1)),
        ("证据分数>=5", bull & (score >= 5)),
        ("日周月共振", bull & w_above & m_above),
        ("顶背离", top_div),
        ("底背离", bot_div),
        ("市场下行", bull & (mkt_state == 1)),
        ("市场上行", bull & (mkt_state == 0)),
    ]
    yr_rows = []
    for lab, sel in key_masks:
        for y in range(2010, 2027):
            m = sel & (year == y)[:, None]
            if m.sum() < 500:
                continue
            dm, _ = daily_sel(m, ex)
            if np.isfinite(dm).sum() < 20:
                continue
            yr_rows.append(dict(条件=lab, 年份=y, 日均超额=float(np.nanmean(dm)),
                                股票日=int(m.sum())))
    pd.DataFrame(yr_rows).to_csv(os.path.join(RES, "条件_逐年.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 组合：零轴 x 证据分数 ----------------
    log("combos")
    comb_rows = []
    for z, zl in [(0, "双上"), (1, "DIF上DEA下"), (2, "DIF下DEA上"), (3, "双下")]:
        for s in range(7):
            sel = mask & (zs == z) & (score == s)
            m = cell_stat(sel & bull, ex, dates)
            if m["股票日"] < 1000:
                continue
            comb_rows.append(dict(零轴=zl, 证据分数=s, **m, 判定=verdict_long(m)))
    pd.DataFrame(comb_rows).to_csv(os.path.join(RES, "条件_零轴x分数.csv"), index=False, encoding="utf-8-sig")

    # 市场状态 x 零轴
    mz_rows = []
    for m, ml in [(0, "指数上行"), (1, "指数下行"), (2, "指数震荡")]:
        for z, zl in [(0, "双上"), (1, "DIF上DEA下"), (2, "DIF下DEA上"), (3, "双下")]:
            sel = mask & (mkt_state == m) & (zs == z)
            if sel.sum() < 1000:
                continue
            c = cell_stat(sel & bull, ex, dates)
            mz_rows.append(dict(市场=ml, 零轴=zl, **c, 判定=verdict_long(c)))
    pd.DataFrame(mz_rows).to_csv(os.path.join(RES, "条件_市场x零轴.csv"), index=False, encoding="utf-8-sig")

    log(f"done {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
