# -*- coding: utf-8 -*-
"""
F05 八系统（书里的系统）只读复算。
只读 八系统回测/ 交付物；所有输出写本目录（02_复算/F05/）。
不重跑回测引擎，只从交付的权益/交易/信号 CSV 独立重算指标与统计。
"""
import itertools
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

SRC = "/Users/admin/WorkSpace/trading/八系统回测"
HIST = os.path.join(SRC, "结果_历史成分")
OLD = os.path.join(SRC, "结果")
OUT = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["MACD入门", "MACD波段", "波段为王1", "波段为王2", "看盘", "短线30技", "裸K", "KDJ"]
INIT = 1_000_000.0
PERIODS = 252

log_lines = []


def log(s=""):
    print(s, flush=True)
    log_lines.append(str(s))


def read_eq(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return df


def eq_metrics(eq):
    e = eq.set_index("date")["equity"].astype(float)
    total = e.iloc[-1] / e.iloc[0] - 1
    days = (e.index[-1] - e.index[0]).days
    ann365 = (1 + total) ** (365 / days) - 1
    r = e.pct_change().dropna()
    ann252 = (1 + total) ** (PERIODS / len(r)) - 1
    dd = (e / e.cummax() - 1).min()
    yearly = e.groupby(e.index.year).last() / e.groupby(e.index.year).last().shift(1) - 1
    yearly.iloc[0] = e[e.index.year == e.index[0].year].iloc[-1] / e.iloc[0] - 1
    return dict(total=total, ann365=ann365, ann252=ann252, dd=dd,
                **{f"y{y}": v for y, v in yearly.items()})


def tr_metrics(tr):
    wins = tr[tr.pnl > 0]
    losses = tr[tr.pnl <= 0]
    pf = wins.pnl.sum() / abs(losses.pnl.sum()) if losses.pnl.sum() != 0 else np.nan
    t = tr.ret.mean() / (tr.ret.std(ddof=1) / math.sqrt(len(tr))) if len(tr) > 1 else np.nan
    top = tr.loc[tr.pnl.idxmax()]
    ex_top = tr.pnl.sum() - top.pnl
    w2 = tr[tr.pnl > 0].drop(index=tr.pnl.idxmax(), errors="ignore")
    pf_ex = w2.pnl.sum() / abs(losses.pnl.sum()) if losses.pnl.sum() != 0 else np.nan
    return dict(trades=len(tr), pnl_sum=tr.pnl.sum(), win=(tr.pnl > 0).mean(), pf=pf, t=t,
                top_code=top.code, top_pnl=top.pnl, top_share=top.pnl / tr.pnl.sum() if tr.pnl.sum() != 0 else np.nan,
                pnl_ex_top=ex_top, pf_ex_top=pf_ex)


# ---------------------------------------------------------------------------
# 1. 八套系统单独指标：时点成分口径 + 旧口径
# ---------------------------------------------------------------------------
def part1():
    log("=" * 70)
    log("Part 1 八系统单独指标复算（权益曲线→收益/MDD；逐笔→胜率/PF/t/集中度）")
    claimed = pd.read_csv(os.path.join(HIST, "汇总_含年度_历史成分.csv"))
    rows = []
    for s in SYSTEMS:
        eq = read_eq(os.path.join(HIST, f"权益_历史成分_{s}.csv"))
        tr = pd.read_csv(os.path.join(HIST, f"交易_历史成分_{s}.csv"))
        m = dict(system=s, **eq_metrics(eq), **tr_metrics(tr))
        c = claimed[claimed.system == s].iloc[0]
        m["claim_total"] = c.total
        m["d_total"] = m["total"] - c.total
        m["d_dd"] = m["dd"] - c.dd
        m["d_ann"] = m["ann365"] - c.ann
        m["claim_trades"] = c.trades
        m["d_win"] = m["win"] - c.win
        rows.append(m)
        log(f"{s:8s} total={m['total']:+.4%} (台账 {c.total:+.4%}, Δ={m['d_total']:+.2e}) "
            f"ann365={m['ann365']:+.4%} dd={m['dd']:.4%} (Δ={m['d_dd']:+.2e}) "
            f"trades={m['trades']}({int(c.trades)}) win={m['win']:.3f} pf={m['pf']:.3f} t={m['t']:.3f} "
            f"top={m['top_code']} {m['top_pnl']/1e4:.2f}万 占{m['top_share']:.1%} 剔后{m['pnl_ex_top']/1e4:+.2f}万 PF_ex={m['pf_ex_top']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "P1_八系统_历史成分_复算.csv"), index=False, encoding="utf-8-sig")
    log(f"最大绝对偏差: total={df.d_total.abs().max():.2e} dd={df.d_dd.abs().max():.2e} ann={df.d_ann.abs().max():.2e} win={df.d_win.abs().max():.2e}")

    # 旧口径
    claimed_old = pd.read_csv(os.path.join(OLD, "汇总_含年度_旧池.csv"))
    rows = []
    for s in SYSTEMS:
        eq = read_eq(os.path.join(OLD, f"权益_{s}.csv"))
        tr = pd.read_csv(os.path.join(OLD, f"交易_{s}.csv"))
        m = dict(system=s, **eq_metrics(eq), **tr_metrics(tr))
        c = claimed_old[claimed_old.system == s].iloc[0]
        m["claim_total"] = c.total
        m["d_total"] = m["total"] - c.total
        rows.append(m)
        log(f"[旧池] {s:8s} total={m['total']:+.4%} (交付汇总 {c.total:+.4%}, Δ={m['d_total']:+.2e}) dd={m['dd']:.4%} trades={m['trades']}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "P1_八系统_旧池_复算.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 2. 八套等权组合与同暴露基准
# ---------------------------------------------------------------------------
def part2():
    log("=" * 70)
    log("Part 2 八套等权组合复算")
    curves = {}
    for s in SYSTEMS:
        eq = read_eq(os.path.join(HIST, f"权益_历史成分_{s}.csv")).set_index("date")["equity"]
        curves[s] = eq / eq.iloc[0]
    C = pd.DataFrame(curves).dropna()
    ew = C.mean(axis=1)
    total = ew.iloc[-1] - 1
    r = ew.pct_change().dropna()
    ann = (1 + total) ** (PERIODS / len(r)) - 1
    dd = (ew / ew.cummax() - 1).min()
    log(f"八套等权(买入持有): total={total:+.4%} ann252={ann:+.4%} MDD={dd:.4%}  (报告 -5.57% / -1.42% / -24.81%)")

    delivered = read_eq(os.path.join(HIST, "组合", "等权组合_权益.csv")).set_index("date")
    dew = delivered["equity"] / delivered["equity"].iloc[0]
    diff = (dew - ew.reindex(dew.index)).abs().max()
    log(f"交付 等权组合_权益.csv 与独立均值曲线最大偏差: {diff:.2e}")
    same = delivered["same_exposure"]
    st = same.iloc[-1] / same.iloc[0] - 1
    sdd = (same / same.cummax() - 1).min()
    log(f"同暴露基准: total={st:+.4%} MDD={sdd:.4%}  (报告 +4.70% / -14.99%)")
    alpha = ew.reindex(same.index).pct_change().dropna() - same.pct_change().dropna()
    t_alpha = alpha.mean() / (alpha.std(ddof=1) / math.sqrt(len(alpha)))
    ann_alpha_arith = alpha.mean() * 252  # 交付口径：日均alpha×252（算术年化）
    ann_alpha_geo = (1 + total) ** (PERIODS / len(r)) - (1 + st) ** (PERIODS / len(r))
    days = (ew.index[-1] - ew.index[0]).days
    ann365 = (1 + total) ** (365 / days) - 1
    log(f"等权 ann365={ann365:+.4%} (交付 -1.42%); 年化 alpha 算术={ann_alpha_arith:+.4%} (交付 -2.33%), "
        f"几何差={ann_alpha_geo:+.4%}, 日alpha t={t_alpha:.3f} (交付 -0.61)")
    # 年度
    y = ew.groupby(ew.index.year).last()
    yr = y / y.shift(1) - 1
    yr.iloc[0] = y.iloc[0] - 1
    log("等权年度: " + " ".join(f"{k}:{v:+.2%}" for k, v in yr.items()))
    pd.DataFrame({"date": ew.index, "ew_equity_norm": ew.values}).to_csv(
        os.path.join(OUT, "P2_等权组合_复算权益.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 3. 主板占比与主板子集
# ---------------------------------------------------------------------------
def is_main(code):
    return code.startswith("sh.60") or code.startswith("sz.00")


def part3():
    log("=" * 70)
    log("Part 3 主板占比与主板子集（逐笔盈亏归因，非账户重跑）")
    snaps = {}
    with open(os.path.join(SRC, "hist_constituents.txt")) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) > 1:
                snaps[p[0]] = set(p[1:])
    rows = []
    for d, cs in snaps.items():
        mb = {c for c in cs if is_main(c)}
        rows.append(dict(snapshot=d, n=len(cs), main=len(mb), share=len(mb) / len(cs)))
        log(f"快照 {d}: 成分 {len(cs)}, 主板 {len(mb)} ({len(mb)/len(cs):.1%})")
    union = set().union(*snaps.values())
    umb = {c for c in union if is_main(c)}
    log(f"并集 {len(union)} 只, 主板 {len(umb)} ({len(umb)/len(union):.1%})")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "P3_快照主板占比.csv"), index=False, encoding="utf-8-sig")

    rows = []
    for s in SYSTEMS:
        tr = pd.read_csv(os.path.join(HIST, f"交易_历史成分_{s}.csv"))
        tr["main"] = tr.code.map(is_main)
        for flag, sub in [("全部", tr), ("主板", tr[tr.main]), ("非主板", tr[~tr.main])]:
            if len(sub) == 0:
                continue
            m = tr_metrics(sub)
            rows.append(dict(system=s, subset=flag, **m))
        mb = tr[tr.main]
        log(f"{s:8s} 主板笔数 {len(mb)}/{len(tr)} ({len(mb)/len(tr):.1%}) "
            f"主板盈亏 {mb.pnl.sum()/1e4:+.2f}万 / 全部 {tr.pnl.sum()/1e4:+.2f}万 "
            f"主板胜率 {(mb.pnl>0).mean():.1%} 最大单笔 {mb.loc[mb.pnl.idxmax(),'code']} {mb.pnl.max()/1e4:.2f}万 "
            f"剔后 {(mb.pnl.sum()-mb.pnl.max())/1e4:+.2f}万")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "P3_主板子集_逐笔归因.csv"), index=False, encoding="utf-8-sig")
    allt = pd.read_csv(os.path.join(HIST, "全部交易_历史成分.csv"))
    allt["main"] = allt.code.map(is_main)
    mb = allt[allt.main]
    log(f"全部八系统合计: 主板 {len(mb)}/{len(allt)} 笔 ({len(mb)/len(allt):.1%}), "
        f"主板盈亏 {mb.pnl.sum()/1e4:+.2f}万 vs 全部 {allt.pnl.sum()/1e4:+.2f}万")
    # 等权口径下的主板贡献近似（1/8 资金 × 主板 pnl / 100万）
    contrib = df[df.subset == "主板"].set_index("system").pnl_sum / 8 / INIT
    log(f"等权组合主板贡献近似合计: {contrib.sum():+.4%} (组合实际 {(-0.0557):+.2%}; 归因近似,忽略仓位交互)")


# ---------------------------------------------------------------------------
# 4. 组合与共振抽查
# ---------------------------------------------------------------------------
def part4():
    log("=" * 70)
    log("Part 4 组合与共振分析抽查")
    # 4.1 共振组合集中度
    tr = pd.read_csv(os.path.join(HIST, "组合", "共振组合_交易.csv"))
    rows = []
    for v, sub in tr.groupby("variant"):
        m = tr_metrics(sub)
        top3 = sub.nlargest(3, "pnl").pnl.sum()
        rows.append(dict(variant=v, **m, top3=top3, rest=sub.pnl.sum() - top3))
        log(f"{v}: 净利 {m['pnl_sum']/1e4:+.2f}万 最大单笔 {m['top_code']} {m['top_pnl']/1e4:.2f}万 "
            f"剔后 {m['pnl_ex_top']/1e4:+.2f}万 PF_ex={m['pf_ex_top']:.2f} 前三 {top3/1e4:.2f}万 其余 {(sub.pnl.sum()-top3)/1e4:+.2f}万")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "P4_共振组合_集中度复算.csv"), index=False, encoding="utf-8-sig")

    # 4.2 共振分组前瞻超额（信号层，从明细重算均值/中位数/胜率）
    sig = pd.read_csv(os.path.join(HIST, "组合", "信号_前瞻收益.csv"))
    log(f"信号总数 {len(sig)} (报告 35,070)")
    rows = []
    for w in [0, 5, 10]:
        col = f"n_sys_{w}"
        s = sig.dropna(subset=["ex20"])
        for gname, mask in [("1套", s[col] == 1), ("2套", s[col] == 2), ("3套+", s[col] >= 3)]:
            sub = s[mask]
            rows.append(dict(window=w, group=gname, n=len(sub), mean_ex20=sub.ex20.mean(),
                             median=sub.ex20.median(), win=(sub.ex20 > 0).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "P4_共振分组_复算.csv"), index=False, encoding="utf-8-sig")
    claim = pd.read_csv(os.path.join(HIST, "组合", "共振分组_前瞻超额.csv"))
    c20 = claim[claim.horizon == 20]
    for r in df.itertuples():
        wname = f"n_sys_{r.window}"
        crow = c20[(c20.window == wname) & (c20.group == r.group)]
        if len(crow):
            d = r.mean_ex20 - crow.iloc[0]["mean"]
            log(f"窗口{r.window}日 {r.group}: n={r.n} 均值={r.mean_ex20:+.4%} (交付 {crow.iloc[0]['mean']:+.4%}, Δ={d:+.2e}) 中位={r.median:+.4%}")
    s5 = sig.dropna(subset=["ex20"])
    log(f"5日窗口 3套+ 占比: {(s5.n_sys_5>=3).mean():.1%} (报告 46.3%)")

    # 4.3 配对增量 Holm
    pairs = pd.read_csv(os.path.join(HIST, "组合", "信号对_增量_显著性.csv"))
    pos_sig = pairs[(pairs.ci_lo > 0)]
    neg_sig = pairs[(pairs.ci_hi < 0)]
    holm_pos = pairs[(pairs.ci_lo > 0) & (pairs.p_holm < 0.05)]
    log(f"56方向对: 实际 {len(pairs)}; 未校正 CI>0: {len(pos_sig)} ({list(pos_sig.a+'→'+pos_sig.b)}); "
        f"CI<0: {len(neg_sig)}; Holm后正向显著: {len(holm_pos)}")
    kdj = pairs[(pairs.a == "KDJ") & (pairs.b == "裸K")].iloc[0]
    log(f"KDJ→裸K: n={kdj.n} ex20={kdj.overlap_ex20:+.4%} CI[{kdj.ci_lo:+.4%},{kdj.ci_hi:+.4%}] p_boot={kdj.p_boot} p_holm={kdj.p_holm}")
    # 独立重做 Holm（对 p_boot 双侧）
    p = pairs.p_boot.values.copy()
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    runmax = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        runmax = max(runmax, val)
        adj[idx] = min(1.0, runmax)
    dmax = np.abs(adj - pairs.p_holm.values).max()
    log(f"独立 Holm 重算与交付 p_holm 最大偏差: {dmax:.3f} (交付为近似校正)")
    pairs_out = pairs.copy()
    pairs_out["p_holm_recalc"] = adj
    pairs_out.to_csv(os.path.join(OUT, "P4_配对_Holm复算.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 5. 股性适配抽查
# ---------------------------------------------------------------------------
def part5():
    log("=" * 70)
    log("Part 5 股性适配分析抽查")
    tr = pd.read_csv(os.path.join(HIST, "股性适配", "交易_股性明细.csv"))
    rows = []
    for (strat, typ), sub in tr.groupby(["策略", "股性类型"]):
        pnl = sub["盈亏（元）"]
        ret = sub["净收益（含费用）"]
        top = pnl.max()
        rows.append(dict(strategy=strat, type=typ, trades=len(sub), pnl=pnl.sum(),
                         pnl_ex_top=pnl.sum() - top, mean_ret=ret.mean(), median_ret=ret.median(),
                         win=(pnl > 0).mean(), top_code=sub.loc[pnl.idxmax(), "股票代码"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "P5_股性_交易层复算.csv"), index=False, encoding="utf-8-sig")
    for r in df.itertuples():
        log(f"{r.strategy} × {r.type}: {r.trades}笔 pnl={r.pnl:+,.0f} 剔最大后={r.pnl_ex_top:+,.0f} "
            f"平均={r.mean_ret:+.2%} 中位={r.median_ret:+.2%} 胜率={r.win:.1%}")
    claim = pd.read_csv(os.path.join(HIST, "股性适配", "交易层_股性统计.csv"))
    log("交付交易层列: " + ",".join(claim.columns))

    from scipy.stats import trim_mean
    sig = pd.read_csv(os.path.join(HIST, "股性适配", "信号_股性明细.csv"))
    rows = []
    for (strat, typ), sub in sig.groupby(["策略", "股性类型"]):
        x = sub["20日超额"].dropna()
        trimmed = trim_mean(x.to_numpy(), 0.05)  # 交付口径：scipy trim_mean 双侧各截5%
        rows.append(dict(strategy=strat, type=typ, n=len(x), mean=x.mean(), median=x.median(),
                         trim5=trimmed, win=(x > 0).mean()))
    ds = pd.DataFrame(rows)
    ds.to_csv(os.path.join(OUT, "P5_股性_信号层复算.csv"), index=False, encoding="utf-8-sig")
    for r in ds.itertuples():
        log(f"[信号] {r.strategy} × {r.type}: n={r.n} 均值={r.mean:+.2%} 中位={r.median:+.2%} 缩尾5%={r.trim5:+.2%}")


# ---------------------------------------------------------------------------
# 6. 过拟合检验复现（PBO / DSR / MinBTL / 去重）
# ---------------------------------------------------------------------------
def sr_daily(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1)


def cscv(X, S):
    T, N = X.shape
    idx = np.array_split(np.arange(T), S)
    omegas, oos_best, oos_med = [], [], []
    for is_idx in itertools.combinations(range(S), S // 2):
        is_set = set(is_idx)
        ir = np.concatenate([idx[i] for i in is_idx])
        orow = np.concatenate([idx[i] for i in range(S) if i not in is_set])
        IS, OOS = X[ir], X[orow]
        is_sr = np.nanmean(IS, axis=0) / np.nanstd(IS, axis=0, ddof=1)
        oos_sr = np.nanmean(OOS, axis=0) / np.nanstd(OOS, axis=0, ddof=1)
        best = int(np.nanargmax(is_sr))
        omega = rankdata(oos_sr, method="average")[best] / (N + 1)
        omegas.append(omega)
        oos_best.append(oos_sr[best])
        oos_med.append(np.nanmedian(oos_sr))
    omegas = np.array(omegas)
    return dict(S=S, combos=len(omegas), PBO=float((omegas < 0.5).mean()),
                is_oos_sr_ann=float(np.mean(oos_best)), oos_med_sr_ann=float(np.mean(oos_med)))


def dsr(r, N, var_sr):
    r = np.asarray(r.dropna(), float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    s = pd.Series(r)
    skew = float(s.skew())
    kurt = float(s.kurtosis() + 3.0)
    g = 0.5772156649015329
    sr0 = math.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr0) * math.sqrt(T - 1) / den
    return dict(sr_ann=sr * math.sqrt(PERIODS), sr0_ann=sr0 * math.sqrt(PERIODS),
                dsr=float(norm.cdf(z)), T=T, skew=skew, kurt=kurt)


def part6():
    log("=" * 70)
    log("Part 6 过拟合检验复现")
    OF = os.path.join(HIST, "过拟合检验")
    reg = pd.read_csv(os.path.join(OF, "试验登记表.csv"))
    log(f"登记配置 {len(reg)} 个; 纳入矩阵 {(reg['是否纳入矩阵']=='是').sum()} 个 (报告 29→25)")
    START, END = pd.Timestamp("2022-01-04"), pd.Timestamp("2025-12-31")
    rets = {}
    for r in reg[reg["是否纳入矩阵"] == "是"].itertuples():
        path = os.path.join(SRC, r.来源)
        eq = read_eq(path).set_index("date")["equity"].astype(float)
        eq = eq[(eq.index >= START) & (eq.index <= END)]
        rets[r.试验名称] = eq.pct_change().dropna()
    R = pd.DataFrame(rets).sort_index().dropna(how="any")
    log(f"从源权益重建矩阵: {R.shape[0]} 天 × {R.shape[1]} 条")
    saved = pd.read_csv(os.path.join(OF, "收益矩阵_日.csv"), index_col=0, parse_dates=True)
    common_cols = [c for c in R.columns if c in saved.columns]
    diff = (R[common_cols] - saved.loc[R.index, common_cols]).abs().to_numpy().max()
    log(f"重建矩阵 vs 交付 收益矩阵_日.csv 最大绝对偏差: {diff:.2e}; 列一致: {set(R.columns)==set(saved.columns)}")
    # 去重核验：29 个登记里重复的 4 个
    dups = reg[reg["是否纳入矩阵"] == "否"]
    for r in dups.itertuples():
        log(f"去重剔除: {r.试验名称} 重复于 {r.重复于}")

    N = R.shape[1]
    var_sr = float(R.apply(sr_daily).var(ddof=1))
    log(f"N={N}, var(daily SR)={var_sr:.8f}")

    # PBO
    rows = []
    M = (1 + R).resample("ME").prod() - 1
    log(f"月度矩阵 {M.shape[0]} 期")
    for freq, mat in [("日", R), ("月", M)]:
        X = mat.values.astype(float)
        for S in [6, 8, 10, 12]:
            res = cscv(X, S)
            rows.append(dict(freq=freq, **res))
    pbo = pd.DataFrame(rows)
    pbo.to_csv(os.path.join(OUT, "P6_PBO_复算.csv"), index=False, encoding="utf-8-sig")
    claim = pd.read_csv(os.path.join(OF, "PBO_CSCV.csv"))
    for r in pbo.itertuples():
        c = claim[(claim["频率"] == r.freq) & (claim["S"] == r.S)].iloc[0]
        log(f"PBO {r.freq} S={r.S}: 复算={r.PBO:.3f} 交付={c['PBO']:.3f} 组合数={r.combos}")

    # DSR 共同区间
    srs = R.apply(sr_daily) * math.sqrt(PERIODS)
    best = srs.idxmax()
    d = dsr(R[best], N, var_sr)
    log(f"共同区间最优: {best} SR_ann={d['sr_ann']:.3f} DSR(N={N})={d['dsr']:.3f} (报告 cooldown10 0.607 / 0.438)")
    # DSR 扩展 2020 低波活跃
    p = os.path.join(HIST, "扩展2020", "股性过滤", "权益_原版三法取二_只做低波活跃型.csv")
    eq = read_eq(p).set_index("date")["equity"].astype(float)
    r_long = eq.pct_change().dropna()
    dl = dsr(r_long, N, var_sr)
    log(f"2020-2025 低波活跃(原版): SR_ann={dl['sr_ann']:.3f} DSR(N={N})={dl['dsr']:.3f} T={dl['T']} (报告 0.556 / 0.378)")
    # MinBTL
    for label, sr, actual in [("共同最优", d["sr_ann"], 4.0), ("低波活跃2020-2025", dl["sr_ann"], 6.0)]:
        for NN in [25, 50, 100]:
            yrs = 2 * math.log(NN) / sr ** 2
            log(f"MinBTL {label} SR={sr:.3f} N={NN}: {yrs:.1f} 年 (实际 {actual} 年)")
    pd.DataFrame([dict(item="共同最优", **d), dict(item="低波活跃2020_2025", **dl)]).to_csv(
        os.path.join(OUT, "P6_DSR_复算.csv"), index=False, encoding="utf-8-sig")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    with open(os.path.join(OUT, "f05_recalc_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()