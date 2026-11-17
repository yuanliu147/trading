#!/usr/bin/env python3
"""33 适配检验：MACD 是否只适合部分股票？

四件事：
A. 分期逐股择时（前期/后期），供持续性检验。
B. 事件/择时的劈半信度与前期→后期持续性（“过去好的股票未来是否继续好”）。
C. 事件层：按个股特征分位，检验金叉超额；同期分层 + 前期特征→后期事件（事前）。
D. 择时层：同一套特征分位；同期 + 前期特征→后期择时；OOS 评分。
E. 行业、聚类（股票类型）。
F. 事件/择时 Fama-MacBeth 多元回归。
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

import stock_core as S
from stats_utils import nw_t, _bh

warnings.simplefilter("ignore")
RES = S.RES
RNG = np.random.default_rng(20261029)

EV_FEATS = [
    "规模_log", "换手20", "波动20", "beta250", "特质波动250", "ER60", "VR20",
    "AC1_60", "MA60择时250", "动量12_1", "距250日高", "年龄_天", "偏度250",
    "最大日涨250", "PE_TTM", "PB_MRQ",
]
# 分位表多放几个描述性特征
Q_FEATS = EV_FEATS + ["成交额_log", "Amihud20", "价格", "MA20择时250", "20日涨幅", "60日涨幅", "ST占比250"]
PERIODS = ["全期", "前期", "后期"]


def log(msg):
    print(msg, flush=True)


def bh_frame(df, by, pcol="日t"):
    df = df.copy()
    df["q_bh"] = np.nan
    if pcol not in df.columns:
        return df
    for _, idx in df.groupby(by).groups.items():
        t = df.loc[idx, pcol].to_numpy(dtype=float)
        p = np.where(np.isfinite(t), 2 * (1 - _norm_cdf(np.abs(t))), np.nan)
        df.loc[idx, "q_bh"] = _bh(p)
    return df


def _norm_cdf(x):
    from scipy.special import erf
    return 0.5 * (1 + erf(np.asarray(x, dtype=float) / np.sqrt(2)))


def welch_t(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 20 or len(b) < 20:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if va + vb > 0 else np.nan


# ---------------- A. 分期择时 ----------------
def build_period_timing():
    z = np.load(os.path.join(RES, "个股_日频.npz"), allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    codes = z["codes"]
    r, pos, uni = z["r_oo"], z["pos_dif"], z["universe"]
    valid = np.isfinite(z["raw_close"]) & (z["raw_close"] > 0)
    mkt = np.nanmean(np.where(uni, r, np.nan), axis=1)
    mkt = np.nan_to_num(mkt)
    parts = []
    for lab, lo, hi in [("全期", "2010-01-01", "2026-12-31"),
                        ("前期", "2010-01-01", "2017-12-31"),
                        ("后期", "2018-01-01", "2026-12-31")]:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        Psub = dict(valid=valid[m], r_oo=r[m], shape=(int(m.sum()), len(codes)))
        ts = S.timing_stats(Psub, pos[m], mkt[m])
        df = pd.DataFrame({"code": codes, "期间": lab})
        for k, v in ts.items():
            df[k] = v
        parts.append(df)
        log(f"  分期择时 {lab} done")
    out = pd.concat(parts, ignore_index=True)
    out["毛超同暴露"] = out["年化毛"] - out["年化同暴露"]
    out["夏普差"] = out["夏普净"] - out["夏普持有"]
    out.to_csv(os.path.join(RES, "个股_择时表现_分期.csv"), index=False, encoding="utf-8-sig")
    return out


# ---------------- B. 持续性与信度 ----------------
def persistence(ev, tm):
    rows = []
    # 事件：前期→后期
    st = pd.read_csv(os.path.join(RES, "个股_事件表现.csv"))
    pe = st.pivot(index="code", columns="口径", values="event_ex_style")
    ne = st.pivot(index="code", columns="口径", values="信号数")
    pe = pe.dropna()
    ne = ne.loc[pe.index]
    rows.append(dict(对象="事件超额", 指标="Pearson", 值=pe["前期"].corr(pe["后期"])))
    rows.append(dict(对象="事件超额", 指标="Spearman", 值=spearmanr(pe["前期"], pe["后期"]).correlation))
    q = pd.qcut(pe["前期"], 10, labels=False, duplicates="drop")
    dec = pe.groupby(q).agg(前期=("前期", "mean"), 后期=("后期", "mean"), 股票数=("后期", "size"))
    dec["后期加权"] = [np.average(pe.loc[q[q == k].index, "后期"],
                                 weights=ne.loc[q[q == k].index, "后期"]) for k in dec.index]
    dec.to_csv(os.path.join(RES, "适配_事件持续性十分位.csv"), encoding="utf-8-sig")

    # 择时：前期→后期
    a = tm[tm["期间"] == "前期"].set_index("code")
    b = tm[tm["期间"] == "后期"].set_index("code")
    j = a.join(b, lsuffix="_前", rsuffix="_后")
    for col in ["净超同暴露", "毛超同暴露", "夏普差", "择时差", "回撤超同暴露"]:
        x, y = j[col + "_前"], j[col + "_后"]
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append(dict(对象=f"择时/{col}", 指标="Pearson", 值=x[ok].corr(y[ok])))
        rows.append(dict(对象=f"择时/{col}", 指标="Spearman",
                         值=spearmanr(x[ok], y[ok]).correlation))
    q = pd.qcut(j["毛超同暴露_前"], 10, labels=False, duplicates="drop")
    dec = j.groupby(q).agg(前期=("毛超同暴露_前", "mean"), 后期=("毛超同暴露_后", "mean"),
                           股票数=("毛超同暴露_后", "size"))
    dec.to_csv(os.path.join(RES, "适配_择时持续性十分位.csv"), encoding="utf-8-sig")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "适配_持续性.csv"), index=False, encoding="utf-8-sig")

    # 事件劈半信度
    e = ev[ev["ex20_style"].notna()][["code", "ex20_style", "date"]].copy()
    cors = []
    for rep in range(20):
        e["_h"] = RNG.integers(0, 2, len(e))
        h = e.groupby(["code", "_h"])["ex20_style"].mean().unstack()
        h = h.dropna()
        if len(h) > 100:
            cors.append(h[0].corr(h[1]))
    rh = np.nanmean(cors)
    rel = 2 * rh / (1 + rh) if rh > -1 else np.nan
    pd.DataFrame([dict(对象="事件超额", 劈半相关=rh, SpearmanBrown=rel,
                       说明="20 次随机劈半平均；Spearman-Brown 校正到全样本")]).to_csv(
        os.path.join(RES, "适配_事件信度.csv"), index=False, encoding="utf-8-sig")
    log(f"  事件劈半相关 {rh:.4f}, 校正 {rel:.4f}")
    return out


# ---------------- C. 事件分位 ----------------
def event_quintiles(ev, feats):
    rows, spreads = [], []
    per = {
        "全期": ev["date"] >= pd.Timestamp("2010-01-01"),
        "前期": (ev["date"] >= pd.Timestamp("2010-01-01")) & (ev["date"] <= pd.Timestamp("2017-12-31")),
        "后期": ev["date"] >= pd.Timestamp("2018-01-01"),
    }
    for f in feats:
        pct = ev.groupby("date")[f].rank(pct=True, method="average")
        q = np.floor(pct * 5).clip(0, 4)
        for lab, m in per.items():
            sub = ev[m & q.notna() & ev["ex20_style"].notna()].copy()
            if len(sub) < 1000:
                continue
            qq = q.loc[sub.index].astype(int)
            sub["_q"] = qq
            dm = sub.groupby(["date", "_q"])["ex20_style"].mean()
            means = []
            for k in range(5):
                s = sub[sub["_q"] == k]
                d = dm.xs(k, level=1) if k in dm.index.get_level_values(1) else pd.Series(dtype=float)
                means.append(s["ex20_style"].mean())
                rows.append(dict(特征=f, 期间=lab, 分位=k + 1, 事件数=len(s),
                                 均值=s["ex20_style"].mean(), 中位=s["ex20_style"].median(),
                                 胜率=(s["ex20_style"] > 0).mean(), 日t=nw_t(d)))
            piv = dm.unstack("_q")
            sp = (piv[4] - piv[0]).dropna() if (4 in piv.columns and 0 in piv.columns) else pd.Series(dtype=float)
            spreads.append(dict(特征=f, 期间=lab, Q5均值=means[4], Q1均值=means[0],
                                Q5Q1=means[4] - means[0], 日t=nw_t(sp),
                                单调性=spearmanr(range(5), means).correlation))
    out = bh_frame(pd.DataFrame(rows), ["特征", "期间"])
    sp = bh_frame(pd.DataFrame(spreads), ["期间"])
    out.to_csv(os.path.join(RES, "适配_事件分位.csv"), index=False, encoding="utf-8-sig")
    sp.to_csv(os.path.join(RES, "适配_事件价差.csv"), index=False, encoding="utf-8-sig")
    return out, sp


# ---------------- D. 择时分位 ----------------
def timing_quintiles(tm, st, feats):
    d = st.merge(tm, on="code", how="inner")
    rows, spreads = [], []
    for f in feats:
        for lab in ["前期", "后期"]:
            col = f"{f}_{lab}"
            if col not in d.columns:
                continue
            sub = d[(d["期间"] == lab) & np.isfinite(d[col])].copy()
            if len(sub) < 500:
                continue
            q = pd.qcut(sub[col], 5, labels=False, duplicates="drop")
            sub["_q"] = q
            means, means_net = {}, {}
            for k in sorted(sub["_q"].dropna().unique()):
                s = sub[sub["_q"] == k]
                means[int(k)] = s["毛超同暴露"].mean()
                means_net[int(k)] = s["净超同暴露"].mean()
                rows.append(dict(特征=f, 期间=lab, 分位=int(k) + 1, 股票数=len(s),
                                 净超同暴露=s["净超同暴露"].mean(),
                                 毛超同暴露=s["毛超同暴露"].mean(),
                                 夏普差=s["夏普差"].mean(),
                                 回撤超同暴露=s["回撤超同暴露"].mean(),
                                 在场比例=s["在场比例"].mean(),
                                 年换手=s["年换手次数"].mean(),
                                 特质波动=s["波动净"].mean()))
            if len(means) >= 2:
                lo, hi = min(means), max(means)
                a = sub[sub["_q"] == hi]["毛超同暴露"]
                b = sub[sub["_q"] == lo]["毛超同暴露"]
                spreads.append(dict(特征=f, 期间=lab, Q5均值=means[hi], Q1均值=means[lo],
                                    Q5Q1=means[hi] - means[lo], 股票t=welch_t(a, b),
                                    Q5Q1净=means_net[hi] - means_net[lo]))
    out = pd.DataFrame(rows)
    sp = bh_frame(pd.DataFrame(spreads), ["期间"], pcol="股票t")
    out.to_csv(os.path.join(RES, "适配_择时分位.csv"), index=False, encoding="utf-8-sig")
    sp.to_csv(os.path.join(RES, "适配_择时价差.csv"), index=False, encoding="utf-8-sig")
    return out, sp


# ---------------- F. FM 回归 ----------------
def fm_regression(ev, tm, feats):
    rows = []
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    for lab, m in [("前期", (ev["date"] <= pd.Timestamp("2017-12-31"))),
                   ("后期", (ev["date"] >= pd.Timestamp("2018-01-01")))]:
        sub = ev[m & ev["ex20_ind"].notna()].copy()
        R = pd.DataFrame(index=sub.index)
        for f in feats:
            R[f] = sub.groupby("date")[f].rank(pct=True).sub(0.5)
        R = R.fillna(0.0)
        X = np.column_stack([np.ones(len(sub)), R.to_numpy(dtype=float)])
        y = sub["ex20_ind"].to_numpy(dtype=float)
        coefs, r2s = [], []
        for _, idx in sub.groupby("date").indices.items():
            if len(idx) < 50:
                continue
            Xi, yi = X[idx], y[idx]
            try:
                b = np.linalg.lstsq(Xi, yi, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            coefs.append(b)
            pred = Xi @ b
            ss = np.sum((yi - yi.mean()) ** 2)
            r2s.append(1 - np.sum((yi - pred) ** 2) / ss if ss > 0 else np.nan)
        coefs = np.array(coefs)
        for k, f in enumerate(feats):
            c = coefs[:, k + 1]
            rows.append(dict(期间=lab, 特征=f, 系数=c.mean(), t=nw_t(c), 日期数=len(c)))
        rows.append(dict(期间=lab, 特征="R2", 系数=np.nanmean(r2s), t=np.nan, 日期数=len(r2s)))
        log(f"  FM 事件 {lab}: {len(coefs)} 日")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "适配_事件FM.csv"), index=False, encoding="utf-8-sig")
    return out


def fm_timing(tm, st, feats):
    d = st.merge(tm, on="code", how="inner")
    rows = []
    for lab in ["前期", "后期"]:
        sub = d[d["期间"] == lab].copy()
        y = sub["毛超同暴露"].to_numpy(dtype=float)
        cols = [f"{f}_{lab}" for f in feats if f"{f}_{lab}" in sub.columns]
        X = sub[cols].to_numpy(dtype=float)
        # 标准化，缺失填 0
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, med)
        sd = X.std(axis=0)
        sd[sd == 0] = 1
        X = (X - med) / sd
        ok = np.isfinite(y)
        X, y, days = X[ok], y[ok], sub["天数"].to_numpy(dtype=float)[ok]
        w = np.sqrt(np.maximum(days, 1))
        Xw = np.column_stack([np.ones(len(X)), X]) * w[:, None]
        yw = y * w
        b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        e = yw - Xw @ b
        # 简单 OLS 标准误（加权后）
        try:
            se = np.sqrt(np.diag(np.linalg.inv(Xw.T @ Xw) * np.sum(e ** 2) / (len(yw) - Xw.shape[1])))
        except np.linalg.LinAlgError:
            se = np.full(Xw.shape[1], np.nan)
        for k, f in enumerate(cols):
            rows.append(dict(期间=lab, 特征=f.replace(f"_{lab}", ""), 系数=b[k + 1],
                             t=b[k + 1] / se[k + 1] if se[k + 1] > 0 else np.nan,
                             股票数=len(y)))
        pred = Xw @ b
        ss = np.sum((yw - yw.mean()) ** 2)
        rows.append(dict(期间=lab, 特征="R2", 系数=1 - np.sum((yw - pred) ** 2) / ss if ss > 0 else np.nan,
                         t=np.nan, 股票数=len(y)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "适配_择时FM.csv"), index=False, encoding="utf-8-sig")
    return out


# ---------------- E. 行业 ----------------
def industry_tables(ev, tm, st):
    # 事件
    rows = []
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    for lab, m in [("全期", ev["date"] >= pd.Timestamp("2010-01-01")),
                   ("前期", ev["date"] <= pd.Timestamp("2017-12-31")),
                   ("后期", ev["date"] >= pd.Timestamp("2018-01-01"))]:
        sub = ev[m & ev["ex20_style"].notna()]
        dm = sub.groupby(["date", "行业"])["ex20_style"].mean()
        for ind, s in sub.groupby("行业"):
            d = dm.xs(ind, level=1) if ind in dm.index.get_level_values(1) else pd.Series(dtype=float)
            rows.append(dict(期间=lab, 行业=ind, 事件数=len(s), 均值=s["ex20_style"].mean(),
                             中位=s["ex20_style"].median(), 日t=nw_t(d)))
    out = bh_frame(pd.DataFrame(rows), ["期间"])
    out.to_csv(os.path.join(RES, "适配_行业_事件.csv"), index=False, encoding="utf-8-sig")

    # 择时
    d = st.merge(tm, on="code", how="inner")
    rows = []
    for lab in ["前期", "后期"]:
        sub = d[d["期间"] == lab]
        for ind, s in sub.groupby("行业"):
            if len(s) < 10:
                continue
            rows.append(dict(期间=lab, 行业=ind, 股票数=len(s),
                             净超同暴露=s["净超同暴露"].mean(),
                             毛超同暴露=s["毛超同暴露"].mean(),
                             夏普差=s["夏普差"].mean(),
                             回撤超同暴露=s["回撤超同暴露"].mean()))
    out2 = pd.DataFrame(rows)
    out2.to_csv(os.path.join(RES, "适配_行业_择时.csv"), index=False, encoding="utf-8-sig")
    return out, out2


# ---------------- F. 聚类 ----------------
def clusters(ev, tm, st):
    cols = [f"{f}_前期" for f in ["规模_log", "换手20", "波动20", "beta250", "ER60",
                                  "VR20", "AC1_60", "MA60择时250", "动量12_1", "偏度250", "年龄_天"]]
    cols = [c for c in cols if c in st.columns]
    X = st[cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    Z = (X - med) / sd
    km = KMeans(n_clusters=6, n_init=20, random_state=42).fit(Z)
    st["类型"] = km.labels_
    prof = pd.DataFrame(km.cluster_centers_, columns=cols)
    prof.index.name = "类型"
    prof.to_csv(os.path.join(RES, "适配_股票类型_画像.csv"), encoding="utf-8-sig")

    # 事件
    ev2 = ev.merge(st[["code", "类型"]], on="code", how="left")
    rows = []
    for lab, m in [("前期", ev2["date"] <= pd.Timestamp("2017-12-31")),
                   ("后期", ev2["date"] >= pd.Timestamp("2018-01-01"))]:
        sub = ev2[m & ev2["ex20_style"].notna() & ev2["类型"].notna()]
        dm = sub.groupby(["date", "类型"])["ex20_style"].mean()
        for k, s in sub.groupby("类型"):
            d = dm.xs(k, level=1) if k in dm.index.get_level_values(1) else pd.Series(dtype=float)
            rows.append(dict(期间=lab, 类型=int(k), 事件数=len(s), 均值=s["ex20_style"].mean(),
                             中位=s["ex20_style"].median(), 日t=nw_t(d)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "适配_股票类型_事件.csv"), index=False, encoding="utf-8-sig")

    # 择时
    d = st.merge(tm, on="code", how="inner")
    rows = []
    for lab in ["前期", "后期"]:
        sub = d[d["期间"] == lab]
        for k, s in sub.groupby("类型"):
            rows.append(dict(期间=lab, 类型=int(k), 股票数=len(s),
                             净超同暴露=s["净超同暴露"].mean(),
                             毛超同暴露=s["毛超同暴露"].mean(),
                             夏普差=s["夏普差"].mean(),
                             回撤超同暴露=s["回撤超同暴露"].mean()))
    out2 = pd.DataFrame(rows)
    out2.to_csv(os.path.join(RES, "适配_股票类型_择时.csv"), index=False, encoding="utf-8-sig")
    return st[["code", "类型"]], prof, out, out2


def main():
    log("load")
    ev = pd.read_parquet(os.path.join(RES, "个股_事件_增强.parquet"))
    ev["date"] = pd.to_datetime(ev["date"])
    if "PE_TTM" not in ev.columns:
        ev["PE_TTM"] = ev["pe_ttm"]
        ev["PB_MRQ"] = ev["pb_mrq"]
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    log("A. 分期择时")
    tm = build_period_timing()

    log("C. 事件分位")
    eq, esp = event_quintiles(ev, Q_FEATS)
    log("D. 择时分位")
    tq, tsp = timing_quintiles(tm, st, Q_FEATS)
    log("F. FM 事件")
    fm_regression(ev, tm, EV_FEATS)
    log("F. FM 择时")
    fm_timing(tm, st, EV_FEATS)
    log("E. 行业")
    industry_tables(ev, tm, st)
    log("F. 聚类")
    clusters(ev, tm, st)
    log("B. 持续性")
    persistence(ev, tm)
    log("done")


if __name__ == "__main__":
    main()
