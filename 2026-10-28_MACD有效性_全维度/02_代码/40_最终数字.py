#!/usr/bin/env python3
"""40 最终数字：把个股适配性结果整理成中文表，供报告/HTML/Excel 使用"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import stock_core as S
from stats_utils import nw_t, _bh

RES = S.RES
KEY_EVENT = ["换手20", "特质波动250", "规模_log", "60日涨幅", "波动20", "ER60", "VR20",
             "AC1_60", "距250日高", "PE_TTM", "PB_MRQ"]
KEY_TIMING = ["VR20", "ER60", "AC1_60", "换手20", "规模_log", "波动20", "特质波动250",
              "MA60择时250", "动量12_1", "60日涨幅", "距250日高", "偏度250"]
ALL_FEATS = ["规模_log", "换手20", "波动20", "beta250", "特质波动250", "ER60", "VR20",
             "AC1_60", "MA60择时250", "动量12_1", "距250日高", "年龄_天", "偏度250",
             "最大日涨250", "PE_TTM", "PB_MRQ"]


def rankq(x, n=5):
    r = pd.Series(x).rank(method="first", pct=True)
    q = np.floor(r * n).clip(0, n - 1)
    q[~np.isfinite(r)] = -1
    return q.to_numpy()


def welch_t(a, b):
    a = pd.Series(a).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    b = pd.Series(b).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(a) < 20 or len(b) < 20:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if va + vb > 0 else np.nan


def main():
    st = pd.read_csv(os.path.join(RES, "个股_特征.csv"))
    tm = pd.read_csv(os.path.join(RES, "个股_择时表现_分期.csv"))
    ev = pd.read_parquet(os.path.join(RES, "个股_事件_增强.parquet"))
    ev["date"] = pd.to_datetime(ev["date"])
    if "PE_TTM" not in ev.columns:
        ev["PE_TTM"], ev["PB_MRQ"] = ev["pe_ttm"], ev["pb_mrq"]
    st["板块"] = st["板块"].fillna("主板")

    # ---------- 1. 总览 ----------
    rows = []
    est = pd.read_csv(os.path.join(RES, "个股_事件表现.csv"))
    for lab in ["全期", "前期", "后期"]:
        s = est[est["口径"] == lab]
        rows.append(dict(项目="金叉逐股超额", 期间=lab, 指标="均值", 值=s["event_ex_style"].mean()))
        rows.append(dict(项目="金叉逐股超额", 期间=lab, 指标="中位", 值=s["event_ex_style"].median()))
        rows.append(dict(项目="金叉逐股超额", 期间=lab, 指标="为正比例", 值=(s["event_ex_style"] > 0).mean()))
        rows.append(dict(项目="金叉逐股超额", 期间=lab, 指标="t<-2比例", 值=(s["event_t"] < -2).mean()))
    for lab in ["全期", "前期", "后期"]:
        s = tm[tm["期间"] == lab]
        rows.append(dict(项目="状态择时逐股", 期间=lab, 指标="净超同暴露中位", 值=s["净超同暴露"].median()))
        rows.append(dict(项目="状态择时逐股", 期间=lab, 指标="毛超同暴露中位", 值=s["毛超同暴露"].median()))
        x = s["净超同暴露"].dropna()
        rows.append(dict(项目="状态择时逐股", 期间=lab, 指标="净超同暴露为正比例", 值=(x > 0).mean()))
        rows.append(dict(项目="状态择时逐股", 期间=lab, 指标="可评估股票数", 值=len(x)))
        rows.append(dict(项目="状态择时逐股", 期间=lab, 指标="成本拖累中位", 值=s["成本拖累"].median()))
        rows.append(dict(项目="状态择时逐股", 期间=lab, 指标="夏普差中位", 值=s["夏普差"].median()))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "最终_个股适配_总览.csv"), index=False, encoding="utf-8-sig")

    # ---------- 2. 事件分位（关键特征） ----------
    eq = pd.read_csv(os.path.join(RES, "适配_事件分位.csv"))
    eq = eq[eq["特征"].isin(KEY_EVENT)]
    eq = eq.rename(columns={"特征": "特征", "期间": "期间", "分位": "分位", "事件数": "事件数",
                            "均值": "平均超额", "中位": "中位超额", "胜率": "胜率", "日t": "日t", "q_bh": "q值"})
    eq.to_csv(os.path.join(RES, "最终_个股适配_事件分位.csv"), index=False, encoding="utf-8-sig")

    esp = pd.read_csv(os.path.join(RES, "适配_事件价差.csv"))
    esp = esp[esp["特征"].isin(KEY_EVENT)].rename(columns={
        "Q5均值": "Q5平均", "Q1均值": "Q1平均", "Q5Q1": "Q5-Q1", "日t": "日t",
        "单调性": "单调性", "q_bh": "q值"})
    esp.to_csv(os.path.join(RES, "最终_个股适配_事件价差.csv"), index=False, encoding="utf-8-sig")

    # ---------- 3. 择时分位 ----------
    tq = pd.read_csv(os.path.join(RES, "适配_择时分位.csv"))
    tq = tq[tq["特征"].isin(KEY_TIMING)].rename(columns={
        "股票数": "股票数", "净超同暴露": "净超同暴露", "毛超同暴露": "毛超同暴露",
        "夏普差": "夏普差", "回撤超同暴露": "回撤超同暴露", "在场比例": "在场比例",
        "年换手": "年换手", "特质波动": "组合波动"})
    tq.to_csv(os.path.join(RES, "最终_个股适配_择时分位.csv"), index=False, encoding="utf-8-sig")

    tsp = pd.read_csv(os.path.join(RES, "适配_择时价差.csv"))
    tsp = tsp[tsp["特征"].isin(KEY_TIMING)].rename(columns={
        "Q5均值": "Q5毛超", "Q1均值": "Q1毛超", "Q5Q1": "Q5-Q1毛超",
        "股票t": "股票t", "Q5Q1净": "Q5-Q1净超", "q_bh": "q值"})
    tsp.to_csv(os.path.join(RES, "最终_个股适配_择时价差.csv"), index=False, encoding="utf-8-sig")

    # ---------- 4. 事前：前期特征 -> 后期事件/择时 ----------
    late = ev[ev["date"] >= pd.Timestamp("2018-01-01")]
    erows, srows = [], []
    for f in KEY_EVENT + [x for x in KEY_TIMING if x not in KEY_EVENT]:
        col = f + "_前期"
        if col not in st.columns:
            continue
        q = rankq(st[col].to_numpy())
        # 事件
        sub = late.merge(st[["code", col]], on="code", how="left")
        sub = sub[np.isfinite(sub[col]) & np.isfinite(sub["ex20_style"])].copy()
        sub["q"] = rankq(sub[col])
        dm = sub.groupby(["date", "q"])["ex20_style"].mean()
        means = []
        for k in range(5):
            s = sub[sub["q"] == k]
            d = dm.xs(k, level=1) if k in dm.index.get_level_values(1) else pd.Series(dtype=float)
            means.append(s["ex20_style"].mean())
            erows.append(dict(特征=f, 分位=k + 1, 事件数=len(s), 平均超额=s["ex20_style"].mean(),
                              中位超额=s["ex20_style"].median(), 胜率=(s["ex20_style"] > 0).mean(),
                              日t=nw_t(d)))
        piv = dm.unstack("q")
        sp = (piv[4] - piv[0]).dropna()
        srows.append(dict(对象="事件", 特征=f, Q5Q1=means[4] - means[0], t=nw_t(sp),
                          单调性=spearmanr(range(5), means).correlation))
        # 择时
        sub2 = tm[(tm["期间"] == "后期")].merge(st[["code", col]], on="code", how="left")
        sub2 = sub2[np.isfinite(sub2[col])].copy()
        sub2["q"] = rankq(sub2[col])
        m = sub2.groupby("q")["毛超同暴露"].mean()
        mn = sub2.groupby("q")["净超同暴露"].mean()
        sp2 = m.get(4, np.nan) - m.get(0, np.nan)
        srows.append(dict(对象="择时", 特征=f, Q5Q1=sp2, t=welch_t(sub2[sub2.q == 4]["毛超同暴露"],
                                                                 sub2[sub2.q == 0]["毛超同暴露"]),
                          单调性=spearmanr(m.index, m.values).correlation if len(m) == 5 else np.nan))
    event_ex = pd.DataFrame(erows)
    spread = pd.DataFrame(srows)
    event_ex.to_csv(os.path.join(RES, "最终_个股适配_事前事件.csv"), index=False, encoding="utf-8-sig")
    spread.to_csv(os.path.join(RES, "最终_个股适配_事前价差.csv"), index=False, encoding="utf-8-sig")

    # ---------- 5. 持续性/信度 ----------
    st2 = pd.read_csv(os.path.join(RES, "个股_事件表现.csv"))
    pe = st2.pivot(index="code", columns="口径", values="event_ex_style").dropna()
    pers = [dict(对象="事件超额", 指标="前期-后期Pearson", 值=pe["前期"].corr(pe["后期"])),
            dict(对象="事件超额", 指标="前期-后期Spearman", 值=spearmanr(pe["前期"], pe["后期"]).correlation)]
    a = tm[tm["期间"] == "前期"].set_index("code")
    b = tm[tm["期间"] == "后期"].set_index("code")
    j = a.join(b, lsuffix="_前", rsuffix="_后")
    for col in ["净超同暴露", "毛超同暴露", "夏普差"]:
        x, y = j[col + "_前"], j[col + "_后"]
        ok = np.isfinite(x) & np.isfinite(y)
        pers.append(dict(对象="状态择时/" + col, 指标="前期-后期Pearson", 值=x[ok].corr(y[ok])))
        pers.append(dict(对象="状态择时/" + col, 指标="前期-后期Spearman",
                         值=spearmanr(x[ok], y[ok]).correlation))
    # 特征稳定性（唯一股票）
    for f in ["VR20", "ER60", "MA60择时250", "换手20", "规模_log", "波动20", "特质波动250", "AC1_60", "年龄_天"]:
        x, y = st[f + "_前期"], st[f + "_后期"]
        ok = np.isfinite(x) & np.isfinite(y) & (st["有效天数_前期"] >= 500) & (st["有效天数_后期"] >= 500)
        pers.append(dict(对象="特征稳定性/" + f, 指标="前期-后期Spearman",
                         值=spearmanr(x[ok], y[ok]).correlation))
    pers.append(dict(对象="事件超额", 指标="劈半信度(SpearmanBrown)",
                     值=pd.read_csv(os.path.join(RES, "适配_事件信度.csv"))["SpearmanBrown"].iloc[0]))
    pd.DataFrame(pers).to_csv(os.path.join(RES, "最终_个股适配_持续性.csv"), index=False, encoding="utf-8-sig")

    # 事件方差分解
    e = st2[st2["口径"] == "全期"].copy()
    obs = e["event_ex_style"].std()
    se = (e["event_ex_style"] / e["event_t"]).replace([np.inf, -np.inf], np.nan).dropna()
    avg_se = se.mean()
    true_sd = np.sqrt(max(obs ** 2 - avg_se ** 2, 0))
    pd.DataFrame([dict(观测标准差=obs, 平均标准误=avg_se, 真实标准差=true_sd,
                       真实方差占比=(true_sd ** 2) / (obs ** 2))]).to_csv(
        os.path.join(RES, "最终_个股适配_事件方差分解.csv"), index=False, encoding="utf-8-sig")

    # 择时劈半信度：前期/后期各切两半
    z = np.load(os.path.join(RES, "个股_日频.npz"), allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    codes = z["codes"]
    r, pos = z["r_oo"], z["pos_dif"]
    valid = np.isfinite(z["raw_close"]) & (z["raw_close"] > 0)
    mkt = np.nan_to_num(np.nanmean(np.where(z["universe"], r, np.nan), axis=1))
    halves = [("前期1", "2010-01-01", "2013-12-31"), ("前期2", "2014-01-01", "2017-12-31"),
              ("后期1", "2018-01-01", "2021-12-31"), ("后期2", "2022-01-01", "2026-12-31")]
    dfs = {}
    for lab, lo, hi in halves:
        m = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
        Psub = dict(valid=valid[m], r_oo=r[m], shape=(int(m.sum()), len(codes)))
        ts = S.timing_stats(Psub, pos[m], mkt[m])
        d = pd.DataFrame({"code": codes, **ts})
        d["毛超同暴露"] = d["年化毛"] - d["年化同暴露"]
        dfs[lab] = d.set_index("code")
    rr = []
    for a1, b1, name in [("前期1", "前期2", "前期"), ("后期1", "后期2", "后期")]:
        x = dfs[a1]["毛超同暴露"]
        y = dfs[b1]["毛超同暴露"]
        ok = np.isfinite(x) & np.isfinite(y)
        rho = x[ok].corr(y[ok])
        rr.append(dict(期间=name, 劈半相关=rho, SpearmanBrown=2 * rho / (1 + rho) if rho > -1 else np.nan,
                       股票数=int(ok.sum())))
    pd.DataFrame(rr).to_csv(os.path.join(RES, "最终_个股适配_择时信度.csv"), index=False, encoding="utf-8-sig")

    # ---------- 6. 配对对照 ----------
    pair = pd.read_csv(os.path.join(RES, "配对_主表.csv"))
    phase = pd.read_csv(os.path.join(RES, "配对_相位.csv"))
    ph = phase.groupby("对象").agg(相位均值=("年化", "mean"), 相位最小=("年化", "min"),
                                   相位最大=("年化", "max"), 相位期数=("期数", "mean")).reset_index()
    ph.to_csv(os.path.join(RES, "最终_个股适配_配对相位.csv"), index=False, encoding="utf-8-sig")
    pair.to_csv(os.path.join(RES, "最终_个股适配_配对.csv"), index=False, encoding="utf-8-sig")

    # ---------- 7. 组合 ----------
    port = pd.read_csv(os.path.join(RES, "组合_择时.csv"))
    keep = port[(port["分组类型"].isin(["全市场"])) |
                (port["分组类型"].str.contains("VR20|干净")) |
                (port["分组类型"].str.contains("规模|换手", regex=True))]
    keep.to_csv(os.path.join(RES, "最终_个股适配_组合.csv"), index=False, encoding="utf-8-sig")
    pd.read_csv(os.path.join(RES, "组合_零假.csv")).to_csv(
        os.path.join(RES, "最终_个股适配_零假.csv"), index=False, encoding="utf-8-sig")

    # ---------- 8. 行业 ----------
    ind_e = pd.read_csv(os.path.join(RES, "适配_行业_事件.csv"))
    ind_t = pd.read_csv(os.path.join(RES, "适配_行业_择时.csv"))
    ind_t2 = ind_t.pivot(index="行业", columns="期间", values="毛超同暴露")
    ind_t2["两期同为正"] = (ind_t2.get("前期", 0) > 0) & (ind_t2.get("后期", 0) > 0)
    ind_t2.reset_index().to_csv(os.path.join(RES, "最终_个股适配_行业.csv"), index=False, encoding="utf-8-sig")

    # ---------- 9. 类型 ----------
    prof = pd.read_csv(os.path.join(RES, "适配_类型画像_v2.csv")).rename(columns={"Unnamed: 0": "类型"})
    prof.to_csv(os.path.join(RES, "最终_个股适配_类型画像.csv"), index=False, encoding="utf-8-sig")
    pd.read_csv(os.path.join(RES, "适配_类型事件_v2.csv")).to_csv(
        os.path.join(RES, "最终_个股适配_类型事件.csv"), index=False, encoding="utf-8-sig")
    pd.read_csv(os.path.join(RES, "适配_类型择时_v2.csv")).to_csv(
        os.path.join(RES, "最终_个股适配_类型择时.csv"), index=False, encoding="utf-8-sig")

    # ---------- 10. FM ----------
    pd.read_csv(os.path.join(RES, "适配_事件FM.csv")).to_csv(
        os.path.join(RES, "最终_个股适配_FM事件.csv"), index=False, encoding="utf-8-sig")
    pd.read_csv(os.path.join(RES, "适配_择时FM.csv")).to_csv(
        os.path.join(RES, "最终_个股适配_FM择时.csv"), index=False, encoding="utf-8-sig")

    # ---------- 11. OOS 评分 ----------
    # 事件：前期 FM 系数 * 后期事件特征分位
    fm = pd.read_csv(os.path.join(RES, "适配_事件FM.csv"))
    ce = fm[(fm["期间"] == "前期") & (fm["特征"] != "R2") & np.isfinite(fm["系数"])]
    ce = ce.set_index("特征")["系数"]
    late2 = late.copy()
    score = np.zeros(len(late2))
    for f, c in ce.items():
        if f not in late2.columns:
            continue
        rk = late2.groupby("date")[f].rank(pct=True) - 0.5
        score += c * rk.fillna(0).to_numpy()
    late2["评分"] = score
    late2["评分分位"] = rankq(score)
    rows = []
    for k in range(5):
        s = late2[late2["评分分位"] == k]
        d = s.groupby("date")["ex20_style"].mean()
        rows.append(dict(对象="事件", 分位=k + 1, 事件数=len(s), 平均超额=s["ex20_style"].mean(),
                         中位=s["ex20_style"].median(), 日t=nw_t(d)))
    oos_e = pd.DataFrame(rows)
    # 择时：前期 FM 系数 * 前期特征标准化 -> 后期
    fmt = pd.read_csv(os.path.join(RES, "适配_择时FM.csv"))
    ct = fmt[(fmt["期间"] == "前期") & (fmt["特征"] != "R2") & np.isfinite(fmt["系数"])]
    ct = ct.set_index("特征")["系数"]
    d = st.merge(tm[tm["期间"] == "后期"], on="code")
    X = []
    for f, c in ct.items():
        col = f + "_前期"
        if col not in d.columns:
            continue
        x = d[col].to_numpy(float)
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med)
        sd = np.nanstd(x)
        x = (x - med) / (sd if sd > 0 else 1)
        X.append(c * x)
    if X:
        d["评分"] = np.sum(X, axis=0)
        d["评分分位"] = rankq(d["评分"])
        rows = []
        for k in range(5):
            s = d[d["评分分位"] == k]
            rows.append(dict(对象="择时", 分位=k + 1, 股票数=len(s), 毛超同暴露=s["毛超同暴露"].mean(),
                             净超同暴露=s["净超同暴露"].mean(), 夏普差=s["夏普差"].mean()))
        oos_t = pd.DataFrame(rows)
    else:
        oos_t = pd.DataFrame()
    pd.concat([oos_e, oos_t], ignore_index=True).to_csv(
        os.path.join(RES, "最终_个股适配_OOS.csv"), index=False, encoding="utf-8-sig")

    print("done")


if __name__ == "__main__":
    main()
