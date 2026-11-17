#!/usr/bin/env python3
"""32 个股层：逐股 MACD 事件超额 + 状态择时 + 个股特征

输出
----
03_结果/个股_事件_增强.parquet   532k 金叉事件，含市场/行业/风格/自身中性超额
03_结果/个股_事件表现.csv         逐股事件统计（全期/前期/后期）
03_结果/个股_择时表现.csv         逐股 DIF>DEA 状态择时统计
03_结果/个股_特征.csv             逐股规模/流动性/波动/趋势性/行业/估值
03_结果/个股_日频.npz             日频收益、仓位、行业、风格格，供组合检验复用
"""
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd

import macd_core as M
import stock_core as S
from feature_core import row_quantile

warnings.simplefilter("ignore", RuntimeWarning)
RES = S.RES
DB = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("load panel")
    P = S.load_base()
    n, nc = P["shape"]
    codes = P["codes"]
    log(f"panel {n} x {nc}")

    st, aux = S.build_states(P)
    pos = {k: S.position(P, v) for k, v in st.items()}
    log("states/positions done")

    # ---------- 行业 ----------
    ind, ind_names = S.industry_matrix(P)
    log(f"industry coverage {(ind >= 0).mean():.3f}, {len(ind_names)} industries")

    # ---------- 规模/波动/风格格 ----------
    mcap = P["amount"].astype(np.float64) / np.maximum(P["turn"].astype(np.float64) / 100.0, 1e-9)
    mcap[~np.isfinite(mcap) | (mcap <= 0)] = np.nan
    size = S.roll(np.log(mcap), 20, "median")
    vol = S.roll(P["r_cc"], 20, "std") * np.sqrt(252)
    size_q = row_quantile(size, P["universe"], 5)
    vol_q = row_quantile(vol, P["universe"], 5)
    cell = np.where((size_q >= 0) & (vol_q >= 0), size_q * 5 + vol_q, -1).astype(np.int8)
    log("size/vol quintiles done")

    # ---------- 事件表 ----------
    log("load events")
    ev = pd.read_parquet(os.path.join(RES, "事件_金叉_12269.parquet"))
    ev = ev[ev["date"] >= pd.Timestamp("2010-01-01")].copy()
    ev = ev[np.isfinite(ev["fwd20"]) & ev["ok20"].astype(bool)].copy()
    log(f"events {len(ev)}")

    c2j = {c: j for j, c in enumerate(codes)}
    ii = np.searchsorted(P["dates"].values, ev["date"].values.astype("datetime64[ns]"))
    jj = ev["code"].map(c2j).to_numpy()
    keep = np.isfinite(jj.astype(float))
    ev, ii, jj = ev.loc[keep].copy(), ii[keep].astype(int), jj[keep].astype(int)
    log(f"events mapped {len(ev)}")

    # ---------- 四种中性超额 ----------
    fwd_d, ok_d, _, _ = M.forward_returns(P, (20,))
    fwd20, ok20 = fwd_d[20], ok_d[20]
    sums_s, cnts_s = S.group_sums(fwd20, ok20, cell, 25)
    sums_i, cnts_i = S.group_sums(fwd20, ok20, ind, len(ind_names))
    ev["ex20_style"] = S.loo_excess(sums_s, cnts_s, cell, fwd20, ii, jj)
    ev["ex20_ind"] = S.loo_excess(sums_i, cnts_i, ind, fwd20, ii, jj)
    own = {}
    for lab, lo, hi in [("前期", "2010-01-01", "2017-12-31"), ("后期", "2018-01-01", "2026-12-31")]:
        m = (P["dates"] >= pd.Timestamp(lo)) & (P["dates"] <= pd.Timestamp(hi))
        own[lab] = np.nanmean(np.where(ok20[m], fwd20[m], np.nan), axis=0)
    late_mask = (ev["date"] >= pd.Timestamp("2018-01-01")).to_numpy()
    own_mean = np.where(late_mask, own["后期"][jj], own["前期"][jj])
    ev["ex20_own"] = ev["fwd20"].to_numpy() - own_mean
    ev["行业"] = [ind_names[x] if x >= 0 else "未分类" for x in ind[ii, jj]]
    ev["风格格"] = cell[ii, jj]
    ev["规模5"] = size_q[ii, jj]
    ev["波动5"] = vol_q[ii, jj]
    ev["ii"] = ii
    ev["jj"] = jj
    log("neutral excess done")

    # ---------- 估值（月末快照） ----------
    log("valuation snapshots")
    dts = pd.Series(P["dates"])
    mends = dts.groupby(dts.dt.to_period("M")).max().dt.strftime("%Y-%m-%d").tolist()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = ("select code,date,pe_ttm,pb_mrq from stock_kline_d "
         f"where adjustflag=1 and date in ({','.join(['?'] * len(mends))})")
    val = pd.read_sql_query(q, con, params=mends)
    con.close()
    val["code"] = val["code"].map(lambda x: x.split(".")[1].zfill(6) + (".SH" if x.startswith("sh.") else ".SZ"))
    val["date"] = pd.to_datetime(val["date"]).astype("datetime64[ns]")
    ev["date"] = pd.to_datetime(ev["date"]).astype("datetime64[ns]")
    val = val.sort_values(["date", "code"])
    ev = pd.merge_asof(ev.sort_values(["date", "code"]), val, by="code", on="date",
                       direction="backward")
    ev = ev.sort_values(["date", "code"]).reset_index(drop=True)
    log(f"valuation merged {ev['pe_ttm'].notna().mean():.3f}")

    ii2 = ev["ii"].to_numpy()
    jj2 = ev["jj"].to_numpy()
    masks = S.period_masks(P)
    stock = pd.DataFrame({"code": codes})

    def attach(name, F):
        F = F.astype(np.float32)
        ev[name] = F[ii2, jj2]
        Fv = np.where(P["valid"], F, np.float32(np.nan))
        for lab, m in masks.items():
            # 只在实际交易日上取中位数，避免上市前 bfill 的 0 值污染
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stock[f"{name}_{lab}"] = np.nanmedian(Fv[m], axis=0)
        log(f"  {name} attached")

    # ---------- 个股特征（逐个计算、逐个附加，避免同时占用内存） ----------
    log("build feature matrices")
    logp = np.log(P["close_ff"].astype(np.float64)).astype(np.float32)
    r_cc = P["r_cc"]
    c = P["close_ff"]
    mkt = P["ew"]
    lr = np.log1p(np.clip(r_cc, -0.999, None)).astype(np.float32)

    attach("规模_log", size)
    attach("成交额_log", np.log(P["amount20"].astype(np.float64) + 1).astype(np.float32))
    attach("换手20", S.roll(P["turn"], 20, "mean"))
    attach("波动20", vol)
    amihud = np.divide(np.abs(r_cc), P["amount"].astype(np.float64),
                       out=np.full(P["shape"], np.nan, np.float64),
                       where=P["amount"].astype(np.float64) > 0)
    attach("Amihud20", (S.roll(amihud, 20, "mean") * 1e9).astype(np.float32))
    del amihud
    attach("价格", P["raw_close"])

    age_info = pd.read_parquet(os.path.join(RES, "面板_日线.parquet"),
                               columns=["thscode", "ipo_date"]).drop_duplicates("thscode")
    age_info = age_info.set_index("thscode")["ipo_date"].reindex(codes)
    ipo = pd.to_datetime(age_info).values
    age = (P["dates"].values[:, None] - ipo[None, :]) / np.timedelta64(1, "D")
    age = np.clip(age.astype(np.float64), 0, None).astype(np.float32)
    attach("年龄_天", age)
    del age

    # beta / 特质波动
    mr = S.roll(r_cc, 250, "mean")
    er2 = S.roll(r_cc * r_cc, 250, "mean")
    erm = S.roll(r_cc * mkt[:, None], 250, "mean")
    em2 = pd.Series(mkt.astype(np.float64) ** 2).rolling(250, min_periods=125).mean().to_numpy()
    mm = pd.Series(mkt.astype(np.float64)).rolling(250, min_periods=125).mean().to_numpy()
    cov = erm - mr * mm[:, None]
    vm = (em2 - mm ** 2)[:, None]
    beta = np.divide(cov, vm, out=np.full(P["shape"], np.nan, np.float32), where=vm > 1e-12)
    var_r = er2 - mr ** 2
    idio = np.sqrt(np.maximum(var_r - beta ** 2 * vm, 0)) * np.sqrt(252)
    attach("beta250", beta)
    attach("特质波动250", idio)
    del mr, er2, erm, em2, mm, cov, vm, beta, var_r, idio

    attach("偏度250", S.roll(r_cc, 250, "skew"))
    attach("最大日涨250", S.roll(r_cc, 250, "max"))

    abs_r = np.abs(r_cc)
    den = S.roll(abs_r, 60, "mean") * 60
    num = np.abs(logp - M.shift(logp, 60))
    attach("ER60", np.divide(num, den, out=np.full(P["shape"], np.nan, np.float32), where=den > 1e-8))
    del abs_r, den, num

    r20 = logp - M.shift(logp, 20)
    v20 = S.roll(r20, 60, "var")
    v1 = S.roll(r_cc, 60, "var")
    attach("VR20", np.divide(v20, 20 * v1, out=np.full(P["shape"], np.nan, np.float32),
                             where=(v1 > 1e-12)))
    del r20, v20, v1

    x = r_cc
    y = M.shift(x, 1)
    mx = S.roll(x, 60, "mean")
    my = S.roll(y, 60, "mean")
    exy = S.roll(x * y, 60, "mean")
    ex2 = S.roll(x * x, 60, "mean")
    ey2 = S.roll(y * y, 60, "mean")
    acov = exy - mx * my
    vx = ex2 - mx ** 2
    vy = ey2 - my ** 2
    attach("AC1_60", np.divide(acov, np.sqrt(vx * vy),
                               out=np.full(P["shape"], np.nan, np.float32),
                               where=(vx > 1e-12) & (vy > 1e-12)))
    del x, y, mx, my, exy, ex2, ey2, acov, vx, vy

    # 趋势性：MA 择时过去 250 日毛边际
    edge20 = (S.roll(pos["价格>MA20"] * lr, 250, "mean") - S.roll(lr, 250, "mean")) * 250
    attach("MA20择时250", edge20)
    del edge20
    edge60 = (S.roll(pos["价格>MA60"] * lr, 250, "mean") - S.roll(lr, 250, "mean")) * 250
    attach("MA60择时250", edge60)
    del edge60

    attach("动量12_1", M.shift(logp, 21) - M.shift(logp, 252))
    attach("20日涨幅", c / M.shift(c, 20) - 1)
    attach("60日涨幅", c / M.shift(c, 60) - 1)
    attach("距250日高", c / M.rolling_max(c, 250) - 1)
    attach("ST占比250", S.roll(P["is_st"], 250, "mean"))

    # 行业众数 + 板块
    ind_mode = np.full(nc, -1, dtype=np.int16)
    for j in range(nc):
        g = ind[masks["全期"], j]
        g = g[g >= 0]
        if len(g):
            ind_mode[j] = np.bincount(g).argmax()
    stock["行业"] = [ind_names[x] if x >= 0 else "未分类" for x in ind_mode]
    code_s = pd.Series(codes).astype(str)
    stock["板块"] = np.select(
        [code_s.str.startswith(("300", "301", "302")),
         code_s.str.startswith(("688", "689"))],
        ["创业板", "科创板"], default="主板")
    stock["上市日"] = age_info.values
    stock["状态"] = pd.read_parquet(os.path.join(RES, "面板_日线.parquet"),
                                    columns=["thscode", "status"]).drop_duplicates("thscode").set_index("thscode")["status"].reindex(codes).values
    for lab, m in masks.items():
        stock[f"有效天数_{lab}"] = P["valid"][m].sum(axis=0).astype(int)

    for lab, lo, hi in [("前期", "2010-01-01", "2017-12-31"), ("后期", "2018-01-01", "2026-12-31")]:
        m = (ev["date"] >= pd.Timestamp(lo)) & (ev["date"] <= pd.Timestamp(hi))
        g = ev.loc[m].groupby("code")[["pe_ttm", "pb_mrq"]].median()
        stock[f"PE_TTM_{lab}"] = stock["code"].map(g["pe_ttm"])
        stock[f"PB_MRQ_{lab}"] = stock["code"].map(g["pb_mrq"])
    stock["PE_TTM_全期"] = stock["code"].map(ev.groupby("code")["pe_ttm"].median())
    stock["PB_MRQ_全期"] = stock["code"].map(ev.groupby("code")["pb_mrq"].median())
    log("stock features done")

    # ---------- 逐股事件统计 ----------
    log("per-stock event stats")
    parts = []
    for lab, m in masks.items():
        sub = ev[(ev["date"] >= P["dates"][m][0]) & (ev["date"] <= P["dates"][m][-1])]
        parts.append(S.per_stock_event(sub, lab))
    evstat = pd.concat(parts, ignore_index=True)
    evstat.to_csv(os.path.join(RES, "个股_事件表现.csv"), index=False, encoding="utf-8-sig")
    log(f"event stats {evstat.shape}")

    # ---------- 逐股择时 ----------
    log("per-stock timing")
    ts = S.timing_stats(P, pos["DIF>DEA"], mkt)
    timing = pd.DataFrame({"code": codes})
    for k, v in ts.items():
        timing[k] = v
    timing.to_csv(os.path.join(RES, "个股_择时表现.csv"), index=False, encoding="utf-8-sig")
    log(f"timing {timing.shape}")

    # ---------- 落盘 ----------
    ev.drop(columns=["ii", "jj"], errors="ignore").to_parquet(
        os.path.join(RES, "个股_事件_增强.parquet"), index=False)
    stock.to_csv(os.path.join(RES, "个股_特征.csv"), index=False, encoding="utf-8-sig")
    np.savez_compressed(
        os.path.join(RES, "个股_日频.npz"),
        dates=P["dates"].values.astype("datetime64[ns]"),
        codes=codes.astype(str),
        r_oo=P["r_oo"], pos_dif=pos["DIF>DEA"].astype(np.int8),
        pos_hist=pos["柱体>0"].astype(np.int8),
        pos_ma60=pos["价格>MA60"].astype(np.int8),
        universe=P["universe"], universe_inv=P["universe_inv"],
        cell=cell, industry=ind, amount20=P["amount20"].astype(np.float32),
        raw_close=P["raw_close"].astype(np.float32),
    )
    log(f"saved. elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
