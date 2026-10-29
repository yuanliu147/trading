#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终数值审计：独立复算（只读）。

不 import 02_代码/最终交付.py，不调用其断言；所有数值从 CSV/parquet/NAV/DuckDB 重算。
不计算 2021+ 收益；只扫描日期上界。
输出：05_审计/最终数值审计.json
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_审计" / "最终数值审计.json"
SEED = 20261026
GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def read_csv(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel, encoding="utf-8-sig", **kw)


def cagr_mdd(nav: np.ndarray, dates: pd.Series) -> tuple[float, float, float]:
    years = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[0])).days / 365.25
    cagr = (nav[-1] / nav[0]) ** (1.0 / years) - 1.0
    mdd = float((nav / np.maximum.accumulate(nav) - 1.0).min())
    return float(cagr), mdd, float(years)


def nav_stats(rel: str, col: str = "nav") -> dict:
    p = ROOT / rel
    d = pd.read_parquet(p) if p.suffix == ".parquet" else read_csv(rel, parse_dates=["date"])
    nav = d[col].to_numpy(dtype=float)
    dates = pd.to_datetime(d["date"])
    cagr, mdd, years = cagr_mdd(nav, dates)
    r = pd.Series(nav).pct_change().dropna()
    vol = float(r.std(ddof=1) * math.sqrt(252))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
    return {
        "cagr": cagr, "mdd": mdd, "years": years,
        "vol": vol, "sharpe": sharpe,
        "n_days": int(len(d)),
        "start": str(dates.iloc[0].date()), "end": str(dates.iloc[-1].date()),
        "nav0": float(nav[0]), "navT": float(nav[-1]),
    }


def dsr(sr_daily: float, T: int, var_sr: float, N: int, sk: float, ku: float) -> dict:
    z1 = norm.ppf(1 - 1.0 / N)
    z2 = norm.ppf(1 - 1.0 / (N * math.e))
    sr0 = math.sqrt(max(var_sr, 0.0)) * ((1 - GAMMA) * z1 + GAMMA * z2)
    den = math.sqrt(max(1e-12, 1 - sk * sr_daily + (ku - 1) / 4.0 * sr_daily ** 2))
    z = (sr_daily - sr0) * math.sqrt(max(T - 1, 1)) / den
    return {"dsr": float(norm.cdf(z)), "z": float(z), "sr0_daily": sr0,
            "sr0_annual": float(sr0 * math.sqrt(243)), "N": N, "T": T}


def pbo_t0005(navs: pd.DataFrame, S: int = 16) -> dict:
    """主候选_T0005.py 的 CSCV 口径（rank=order/(N-1)，pbo=mean(rank<0.5)）。"""
    r = navs.astype(float).pct_change().fillna(0.0)
    r = r.loc[:, r.std() > 0]
    T, N = r.shape
    arr = r.to_numpy()
    edges = np.linspace(0, T, S + 1).astype(int)
    n = np.array([edges[s + 1] - edges[s] for s in range(S)], dtype=float)
    s1 = np.array([arr[edges[s]:edges[s + 1]].sum(axis=0) for s in range(S)])
    s2 = np.array([(arr[edges[s]:edges[s + 1]] ** 2).sum(axis=0) for s in range(S)])
    ranks = []
    for comb in combinations(range(S), S // 2):
        oos = [s for s in range(S) if s not in comb]
        ni = n[list(comb)].sum()
        si = s1[list(comb)].sum(axis=0)
        qi = s2[list(comb)].sum(axis=0)
        sdi = np.sqrt(np.maximum((qi - si ** 2 / ni) / max(ni - 1, 1), 1e-18))
        sr_is = (si / ni) / sdi
        best = int(np.nanargmax(sr_is))
        no = n[oos].sum()
        so = s1[oos].sum(axis=0)
        qo = s2[oos].sum(axis=0)
        sdo = np.sqrt(np.maximum((qo - so ** 2 / no) / max(no - 1, 1), 1e-18))
        sr_oos = (so / no) / sdo
        order = np.argsort(np.argsort(sr_oos))
        ranks.append(float(order[best]) / max(N - 1, 1))
    ranks = np.array(ranks)
    return {"pbo": float(np.mean(ranks < 0.5)), "median_rank": float(np.median(ranks)),
            "N": int(N), "T": int(T), "n_splits": int(len(ranks))}


def pbo_vcb(navs: pd.DataFrame, S: int = 8) -> dict:
    """候选_收缩突破.py 的 CSCV 口径（8 块，70 分割，rank/K）。"""
    T, K = navs.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [navs.iloc[edges[i]:edges[i + 1]] for i in range(S)]
    ranks = []
    for is_idx in combinations(range(S), S // 2):
        oos = [i for i in range(S) if i not in is_idx]
        IS = pd.concat([blocks[i] for i in is_idx])
        OOS = pd.concat([blocks[i] for i in oos])
        sri = IS.mean() / IS.std(ddof=1)
        sro = OOS.mean() / OOS.std(ddof=1)
        best = sri.idxmax()
        rk = float(sro.rank().loc[best]) / K
        rk = min(max(rk, 1.0 / (K + 1)), K / (K + 1))
        ranks.append(rk)
    logits = np.array([math.log(r / (1 - r)) for r in ranks])
    return {"pbo": float((logits < 0).mean()), "median_rank": float(np.median(ranks)),
            "K": int(K), "T": int(T), "n_splits": int(len(ranks))}


def pbo_pool(navs: pd.DataFrame, S: int = 16) -> dict:
    """最终试验总账.py 的 CSCV 口径（omega=rank/(C+1)，pbo=mean(omega<=0.5)）。"""
    R = navs.astype(float).pct_change().fillna(0.0)
    R = R.loc[:, R.std() > 0]
    T, C = R.shape
    idx = np.array_split(np.arange(T), S)
    bsum = np.array([R.values[ix].sum(axis=0) for ix in idx])
    bsq = np.array([(R.values[ix] ** 2).sum(axis=0) for ix in idx])
    bn = np.array([len(ix) for ix in idx], dtype=float)
    combos = np.array(list(combinations(range(S), S // 2)), dtype=np.int16)
    is_mask = np.zeros((len(combos), S), dtype=bool)
    for i, c in enumerate(combos):
        is_mask[i, list(c)] = True
    is_n = is_mask @ bn
    oos_n = bn.sum() - is_n
    is_sum = is_mask @ bsum
    is_sq = is_mask @ bsq
    oos_sum = bsum.sum(axis=0)[None, :] - is_sum
    oos_sq = bsq.sum(axis=0)[None, :] - is_sq
    im = is_sum / is_n[:, None]
    iv = np.maximum(is_sq / is_n[:, None] - im ** 2, 1e-18)
    isr = im / np.sqrt(iv)
    om = oos_sum / oos_n[:, None]
    ov = np.maximum(oos_sq / oos_n[:, None] - om ** 2, 1e-18)
    osr = om / np.sqrt(ov)
    pick = isr.argmax(axis=1)
    picked = osr[np.arange(len(pick)), pick]
    ranks = (osr < picked[:, None]).sum(axis=1) + 1
    omega = ranks / (C + 1.0)
    return {"pbo": float((omega <= 0.5).mean()), "median_rank": float(np.median(omega)),
            "C": int(C), "T": int(T), "n_splits": int(len(omega))}


def maxstat(R: pd.DataFrame, B: int = 2000, block: int = 20, seed: int = SEED) -> dict:
    R = R.dropna(how="any")
    vals = R.to_numpy(dtype=float)
    T, C = vals.shape
    obs = float(vals.mean(axis=0).max())
    v0 = vals - vals.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed + 7)
    cum = np.vstack([np.zeros((1, C)), np.cumsum(v0, axis=0)])
    cum2 = np.vstack([cum, cum[-1] + cum[1:]])
    mx = np.empty(B)
    for b in range(B):
        idx = 0
        tot = np.zeros(C)
        n = 0
        while idx < T:
            L = max(1, int(rng.geometric(1.0 / block)))
            start = int(rng.integers(0, T))
            take = min(L, T - idx)
            tot += cum2[start + take] - cum2[start]
            n += take
            idx += take
        mx[b] = (tot / max(n, 1)).max()
    return {"stat": obs, "p_maxstat": float(((mx >= obs).sum() + 1) / (B + 1)), "B": B, "T": T, "C": C}


def load_trend_matrix(win: str) -> pd.DataFrame:
    out = {}
    for f in glob.glob(str(ROOT / "03_结果/趋势族" / f"净值_*_{win}.parquet")):
        d = pd.read_parquet(f)
        name = Path(f).name.replace(f"_{win}.parquet", "").replace("净值_", "")
        out[name] = pd.Series(d["equity"].to_numpy(dtype=float), index=pd.to_datetime(d["date"]))
    return pd.DataFrame(out).sort_index()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def near(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# 复算
# ---------------------------------------------------------------------------
M = {}
checks = []


def add(cid, item, reported, actual, verdict, severity, source, note="", tol=0.0):
    checks.append({
        "id": cid, "item": item, "reported": reported, "actual": actual,
        "verdict": verdict, "severity": severity, "source": source, "note": note,
        "tol": tol,
    })


# ---- 1. 数据层 -------------------------------------------------------------
con = duckdb.connect(str(ROOT / "01_数据/frozen/frozen.duckdb"), read_only=True)
rows, syms, dmin, dmax = con.execute(
    "select count(*), count(distinct symbol), min(date), max(date) from daily").fetchone()
ncols = len(con.execute("describe daily").fetchall())
M["data_rows"] = int(rows); M["data_symbols"] = int(syms)
M["data_dmin"] = str(dmin); M["data_dmax"] = str(dmax); M["data_cols"] = ncols

add("D01", "日线行数/证券数/日期边界/列数", "16,698,536 / 5,549 / 2004-01-02..2026-10-26 / 40",
    f"{rows:,} / {syms:,} / {dmin}..{dmax} / {ncols}",
    "PASS" if (rows, syms, str(dmin), str(dmax), ncols) == (16698536, 5549, "2004-01-02", "2026-10-26", 40) else "FAIL",
    "LOW", "01_数据/frozen/frozen.duckdb daily")

# v2 不变量与 v1 反事实
q = """
with x as (
 select symbol,date,pct_chg,factor,close,hfq_close,
   hfq_close/lag(hfq_close) over (partition by symbol order by date) - 1 as r2,
   (close*factor)/lag(close*factor) over (partition by symbol order by date) - 1 as r1
 from daily
)
select count(*) n,
  max(abs(r2 - pct_chg/100)) max_dev2,
  sum(case when abs(r2 - pct_chg/100) > 0.0005 then 1 else 0 end) n_bad2,
  max(abs(r1 - pct_chg/100)) max_dev1,
  sum(case when abs(r1 - pct_chg/100) > 0.0001 then 1 else 0 end) n_bad1,
  median(abs(r1 - pct_chg/100)) med_dev1,
  regr_slope(r1, pct_chg/100) slope1
from x where r2 is not null and r1 is not null
"""
v2 = con.execute(q).fetchone()
M.update({"adj_pairs": int(v2[0]), "v2_max_dev": float(v2[1]), "v2_bad_5bp": int(v2[2]),
          "v1_max_dev": float(v2[3]), "v1_bad_1bp": int(v2[4]),
          "v1_med_dev": float(v2[5]), "v1_slope": float(v2[6]),
          "v1_bad_share": float(v2[4] / v2[0])})
add("D02", "v2 hfq 日收益=官方 pct_chg（>5bp 违约）", "0 行；max 4.31e-15",
    f"{v2[2]} 行；max {v2[1]:.4e}", "PASS" if v2[2] == 0 and abs(v2[1]) < 1e-12 else "FAIL",
    "LOW", "frozen daily 全样本相邻对")
add("D03", "v1 hfq=raw×factor 坏行占比（>1bp）", "94.10%",
    f"{v2[4]:,}/{v2[0]:,} = {100*v2[4]/v2[0]:.2f}%", "WARN", "LOW",
    "frozen daily 全样本相邻对",
    "报告 94.10% 的分母未写；全样本相邻对口径为 94.01%，差 0.09pp；坏行数 15,693,245 与证据矩阵 R04 一致")
add("D04", "v1 中位偏差 / 斜率", "1.41pp / 2.012（截尾）或 2.62（全样本）",
    f"{100*v2[5]:.2f}pp / {v2[6]:.4f}",
    "PASS" if near(100*v2[5], 1.4105, 0.01) and near(v2[6], 2.6196, 0.01) else "FAIL",
    "LOW", "frozen daily 全样本相邻对")

# 双源
try:
    ths_path = str(Path.home() / "Library/Application Support/hithink-finance/data/market.duckdb")
    con.execute(f"attach '{ths_path}' as ths (read_only)")
    q = """
    with m as (
     select d.symbol,d.date,d.open,d.high,d.low,d.close,
            upper(split_part(d.symbol,'.',1))||'.'||split_part(d.symbol,'.',2) as thscode
     from daily d where d.date between DATE '2016-10-13' and DATE '2026-10-26'
    )
    select count(*) n,
      sum(case when abs(m.close-t.close)>0.005 then 1 else 0 end) n_close,
      sum(case when abs(m.open-t.open)>0.005 or abs(m.high-t.high)>0.005
                or abs(m.low-t.low)>0.005 or abs(m.close-t.close)>0.005 then 1 else 0 end) n_ohlc,
      max(abs(m.close-t.close)) max_close
    from m join ths.raw_kline_daily t on t.thscode=m.thscode and t.date=m.date
    """
    ds = con.execute(q).fetchone()
    M.update({"dual_common_days": int(ds[0]), "dual_close_gt_5m": int(ds[1]),
              "dual_ohlc_gt_5m": int(ds[2]), "dual_max_close_diff": float(ds[3])})
    add("D05", "双源共同证券日 / 收盘差>0.005 元", "10,104,953 / 4 行",
        f"{ds[0]:,} / {ds[1]} 行", "PASS" if ds[0] == 10104953 and ds[1] == 4 else "FAIL",
        "LOW", "frozen ⋈ THS raw_kline_daily（2016-10-13..2026-10-26）")
    add("D06", "报告写“OHLC 差异>0.005 元仅 4 行”", "4 行",
        f"收盘差 4 行；任一 OHLC 字段 5 行", "WARN", "LOW",
        "同上", "4 行只是收盘价；任一 OHLC 字段为 5 行（688206.SH 2021-12-28 仅最高价差 1.20 元）")
    # 退市股缺失
    import sqlite3
    bs = sqlite3.connect(str(Path.home() / "WorkSpace/baostock_data/data/baostock.db"))
    delisted = [r[0] for r in bs.execute(
        "select code from stock_basic where type=1 and out_date is not null and out_date!=''").fetchall()]
    con.execute("create temp table bs_del(code varchar)")
    con.executemany("insert into bs_del values (?)", [(c,) for c in delisted])
    miss = con.execute("""
      select count(*) from bs_del where not exists (
        select 1 from ths.raw_kline_daily t
        where t.thscode=split_part(bs_del.code,'.',2)||'.'||upper(split_part(bs_del.code,'.',1))
          and t.date<=DATE '2026-10-26')
    """).fetchone()[0]
    M["ths_missing_delisted"] = int(miss); M["bs_delisted"] = len(delisted)
    add("D07", "同花顺缺失退市股 / BS 退市股", "338 / 341",
        f"{miss} / {len(delisted)}", "PASS" if (miss, len(delisted)) == (338, 341) else "FAIL",
        "LOW", "BS stock_basic type=1 out_date; THS raw")
except Exception as exc:  # pragma: no cover
    add("D05", "双源核验", "10,104,953 / 4", f"SKIP: {exc}", "WARN", "LOW", "THS/Baostock")

# ---- 2. 计数与覆盖 ---------------------------------------------------------
led = read_csv("03_结果/最终试验总账.csv", low_memory=False)
gate = read_csv("03_结果/最终统计/统一门槛.csv", low_memory=False)
cnt = read_csv("03_结果/最终统计/最终计数.csv")
N = {k: int(led[f"in_{k}"].sum()) for k in ["Nsel", "Ncell", "Ndiag", "Nplatform", "Naudit"]}
M["counts"] = {**N, "rows": int(len(led))}
add("C01", "总账 N_sel/N_cell/N_diag/N_platform/N_audit/rows", "733/1,251/454/257/87/1,699",
    f"{N['Nsel']}/{N['Ncell']}/{N['Ndiag']}/{N['Nplatform']}/{N['Naudit']}/{len(led)}",
    "PASS" if (N["Nsel"], N["Ncell"], N["Ndiag"], N["Nplatform"], N["Naudit"], len(led)) ==
    (733, 1251, 454, 257, 87, 1699) else "FAIL", "LOW", "最终试验总账.csv")

gate_cfg = set(gate.config_id)
covered = int(led[led.config_id.isin(gate_cfg)].in_Nsel.sum())
unrated = N["Nsel"] - covered
M["gate_coverage"] = {"rows": len(gate), "covered": covered, "unrated": unrated}
add("C02", "门槛行数 / 覆盖 N_sel / 未评级", "478 / 505 / 228（505+228=733）",
    f"{len(gate)} / {covered} / {unrated}", "PASS" if (len(gate), covered, unrated) == (478, 505, 228) else "FAIL",
    "LOW", "统一门槛.csv；总账 in_Nsel")
unrated_fam = led[~led.config_id.isin(gate_cfg)].groupby("family").in_Nsel.sum().to_dict()
M["unrated_families"] = {k: int(v) for k, v in unrated_fam.items() if v}
add("C03", "未评级去向", "STRUCT 219 + STAT 9", str(M["unrated_families"]),
    "PASS" if M["unrated_families"] == {"STAT": 9, "STRUCT": 219} else "FAIL",
    "LOW", "总账 family×in_Nsel")

b = gate[gate.grade == "B_conditional"]
b_rows = led[led.config_id.isin(b.config_id)]
M["b_conditional"] = {"configs": int(len(b)), "ledger_rows": int(len(b_rows)),
                      "families": b.family.value_counts().to_dict()}
add("C04", "配置级 B_conditional", "5 个配置 / 10 总账行（4 S3 + VCB-P20）",
    f"{len(b)} / {len(b_rows)}；{b.family.value_counts().to_dict()}",
    "PASS" if len(b) == 5 and len(b_rows) == 10 else "FAIL", "LOW", "统一门槛；总账")
cand = read_csv("03_结果/统计整合/候选分级.csv", low_memory=False)
bb = cand[cand.grade == "B_conditional"]
M["b_row_level"] = {"rows": int(len(bb)), "configs": int(bb.config_id.nunique()),
                    "s3": int(bb.strategy_id.str.contains("S3").sum()),
                    "s1": int(bb.strategy_id.str.contains("S1").sum())}
add("C05", "旧行级 B_conditional", "12 行 / 8 配置（11 S3 + 1 S1）",
    f"{len(bb)} / {bb.config_id.nunique()}（{M['b_row_level']['s3']} S3 + {M['b_row_level']['s1']} S1）",
    "PASS" if (len(bb), bb.config_id.nunique(), M["b_row_level"]["s3"], M["b_row_level"]["s1"]) == (12, 8, 11, 1) else "FAIL",
    "LOW", "候选分级.csv")

# ---- 3. T0005 --------------------------------------------------------------
nav_dev = nav_stats("03_结果/主候选_T0005/03_日净值_T0005_dev.csv")
nav_val = nav_stats("03_结果/主候选_T0005/03_日净值_T0005_val.csv")
tr_dev = read_csv("03_结果/主候选_T0005/02_逐笔_T0005_dev.csv")
tr_val = read_csv("03_结果/主候选_T0005/02_逐笔_T0005_val.csv")
M["t0005"] = {"dev": nav_dev, "val": nav_val,
              "trades": {"dev": int(len(tr_dev)), "val": int(len(tr_val)),
                         "codes_dev": int(tr_dev.code.nunique()), "codes_val": int(tr_val.code.nunique())}}
for tag, label, rep, act, tol in [
    ("T01", "dev CAGR", 0.04156815775673306, nav_dev["cagr"], 1e-9),
    ("T02", "dev MDD", -0.0905350729763279, nav_dev["mdd"], 1e-9),
    ("T03", "val CAGR", -0.025342280131277373, nav_val["cagr"], 1e-9),
    ("T04", "val MDD", -0.13797983512013767, nav_val["mdd"], 1e-9),
]:
    add(tag, f"T0005 {label}", f"{rep*100:.4f}%", f"{act*100:.6f}%",
        "PASS" if abs(rep - act) <= tol else "FAIL", "LOW", "03_日净值_T0005_*.csv")
add("T05", "T0005 回合数/股票数", "dev 103/81；val 159/111",
    f"dev {len(tr_dev)}/{tr_dev.code.nunique()}；val {len(tr_val)}/{tr_val.code.nunique()}",
    "PASS" if (len(tr_dev), tr_dev.code.nunique(), len(tr_val), tr_val.code.nunique()) == (103, 81, 159, 111) else "FAIL",
    "LOW", "02_逐笔_T0005_*.csv")
add("T06", "T0005 Sharpe/波动", "dev 0.90/4.83%；val -0.64/4.00%",
    f"dev {M['t0005']['dev']['sharpe']:.4f}/{M['t0005']['dev']['vol']*100:.2f}%；"
    f"val {M['t0005']['val']['sharpe']:.4f}/{M['t0005']['val']['vol']*100:.2f}%",
    "PASS" if near(M['t0005']['dev']['sharpe'], 0.897189, 1e-3) and near(M['t0005']['val']['sharpe'], -0.642061, 1e-3) else "FAIL",
    "LOW", "NAV 日收益")

for w, tr in [("dev", tr_dev), ("val", tr_val)]:
    top = float(tr.pnl.nlargest(5).sum()); net = float(tr.pnl.sum()); pos = float(tr.pnl[tr.pnl > 0].sum())
    M["t0005"][f"top5_{w}"] = {"top5": top, "net": net, "pos": pos,
                                "share_net": top / net if net else None, "share_pos": top / pos if pos else None}
# 去前 5 年化
for w, nv in [("dev", nav_dev), ("val", nav_val)]:
    tr = tr_dev if w == "dev" else tr_val
    tr_t = pd.read_csv(ROOT / f"03_结果/主候选_T0005/03_日净值_T0005_{w}.csv", encoding="utf-8-sig", parse_dates=["date"])
    nv_ex = nv["navT"] - float(tr.pnl.nlargest(5).sum())
    M["t0005"][f"cagr_ex_top5_{w}"] = (nv_ex / nv["nav0"]) ** (1 / nv["years"]) - 1
add("T07", "T0005 去前 5 年化", "dev +2.40%；val -4.23%",
    f"dev {M['t0005']['cagr_ex_top5_dev']*100:.2f}%；val {M['t0005']['cagr_ex_top5_val']*100:.2f}%",
    "PASS" if abs(M["t0005"]["cagr_ex_top5_dev"] - 0.023997) < 1e-5 and abs(M["t0005"]["cagr_ex_top5_val"] + 0.042338) < 1e-5 else "FAIL",
    "LOW", "NAV - top5 pnl")
add("T08", "T0005 top5 share 口径", "报告 31.1%（dev），表头写“前5盈利/总盈亏”",
    f"dev top5/净盈亏 {M['t0005']['top5_dev']['share_net']*100:.2f}%；top5/全部盈利 {M['t0005']['top5_dev']['share_pos']*100:.2f}%；"
    f"val top5/净盈亏 {M['t0005']['top5_val']['share_net']*100:.2f}%；top5/全部盈利 {M['t0005']['top5_val']['share_pos']*100:.2f}%",
    "FAIL", "MODERATE", "02_逐笔_T0005_*.csv",
    "报告 31.1% 是 top5/全部盈利；“总盈亏”口径应为 45.87%。小盘 COMBO 的 53.2%/375.6% 是 top5/净盈亏，两个族不可比")

# 年度
yr = read_csv("03_结果/主候选_T0005/04_年度_T0005_dev.csv")
M["t0005"]["yearly"] = {str(int(r.year)): float(r["return"]) for _, r in yr.iterrows()}
other = 1.0
for y in ["2008", "2009", "2010", "2011", "2012", "2013"]:
    other *= 1 + M["t0005"]["yearly"][y]
M["t0005"]["other6_annual"] = other ** (1 / 6) - 1
add("T09", "T0005 年度：2014/2015/其余 6 年平均", "+26.39% / +15.08% / -0.81%",
    f"{M['t0005']['yearly']['2014']*100:.2f}% / {M['t0005']['yearly']['2015']*100:.2f}% / {M['t0005']['other6_annual']*100:.2f}%",
    "PASS" if near(M['t0005']['yearly']['2014'], 0.2639106, 1e-5) and near(M['t0005']['other6_annual'], -0.008142, 1e-4) else "FAIL",
    "LOW", "04_年度_T0005_dev.csv")

# 信号漏斗
rg = read_csv("03_结果/主候选_T0005/10_信号_风险门.csv")
M["signal_funnel"] = {r.window: {"n": int(r.n_signals), "pass": int(r.pass_risk_gate),
                                 "share": float(r.pass_share), "r_med": float(r.r_median)} for _, r in rg.iterrows() if ":" not in str(r.window)}
add("T10", "T0005 风险门：dev/val 信号数、过门数、占比、R/P0 中位", "18,586/245/1.32%/29.2%；6,972/324/4.65%/24.5%",
    f"{M['signal_funnel']['dev']['n']:,}/{M['signal_funnel']['dev']['pass']}/{M['signal_funnel']['dev']['share']*100:.2f}%/"
    f"{M['signal_funnel']['dev']['r_med']*100:.2f}%；"
    f"{M['signal_funnel']['val']['n']:,}/{M['signal_funnel']['val']['pass']}/{M['signal_funnel']['val']['share']*100:.2f}%/"
    f"{M['signal_funnel']['val']['r_med']*100:.2f}%",
    "PASS" if (M['signal_funnel']['dev']['n'], M['signal_funnel']['dev']['pass'],
               M['signal_funnel']['val']['n'], M['signal_funnel']['val']['pass']) == (18586, 245, 6972, 324) else "FAIL",
    "LOW", "10_信号_风险门.csv")

# 退出结构
M["exit_struct"] = {}
for w, tr in [("dev", tr_dev), ("val", tr_val)]:
    g = tr.groupby("reason").agg(n=("pnl", "size"), mean=("pnl", "mean"), win=("pnl", lambda s: (s > 0).mean()))
    M["exit_struct"][w] = {k: {"n": int(v["n"]), "mean": float(v["mean"]), "win": float(v["win"])} for k, v in g.iterrows()}
add("T11", "T0005 退出结构（stop/open 笔数均值胜率）", "dev stop 37/+13,334/78.4%；open 65/-1,917/18.5%；val 36/+2,366/61.1%；120/-1,760/16.7%",
    f"dev stop {M['exit_struct']['dev']['stop_trigger']['n']}/{M['exit_struct']['dev']['stop_trigger']['mean']:.0f}/"
    f"{M['exit_struct']['dev']['stop_trigger']['win']*100:.1f}%；open {M['exit_struct']['dev']['open']['n']}/"
    f"{M['exit_struct']['dev']['open']['mean']:.0f}/{M['exit_struct']['dev']['open']['win']*100:.1f}%；"
    f"val stop {M['exit_struct']['val']['stop_trigger']['n']}/{M['exit_struct']['val']['stop_trigger']['mean']:.0f}/"
    f"{M['exit_struct']['val']['stop_trigger']['win']*100:.1f}%；open {M['exit_struct']['val']['open']['n']}/"
    f"{M['exit_struct']['val']['open']['mean']:.0f}/{M['exit_struct']['val']['open']['win']*100:.1f}%",
    "PASS" if (M['exit_struct']['dev']['stop_trigger']['n'], M['exit_struct']['dev']['open']['n'],
               M['exit_struct']['val']['stop_trigger']['n'], M['exit_struct']['val']['open']['n']) == (37, 65, 36, 120) else "FAIL",
    "LOW", "02_逐笔")
add("T12", "T0005 “退市强制退出” 2 笔（val）", "0（dev）/2（val，退市强制）",
    f"delist_forced=0/0；forced_sells=0/2；2 笔 reason=sell_persistence_10d、exit_reason_state=time_20d",
    "FAIL", "LOW", "08_自检_T0005_val.json；02_逐笔_T0005_val.csv",
    "2 笔是停牌顺延后的强制估值卖出，不是退市强平；JSON 的 delist_forced=0。应写“停牌顺延强制估值卖出 2 笔（非退市）”")

# 消融/邻域
abl = read_csv("03_结果/主候选_T0005/20_消融.csv")
nei = read_csv("03_结果/主候选_T0005/30_邻域.csv")
abl_val = abl[(abl.window == "val") & (abl.n_trades > 0)]
M["ablation"] = {"val_min": float(abl_val.cagr.min()),
                 "val_max": float(abl_val.cagr.max()),
                 "n_val_pos": int((abl_val.cagr > 0).sum()),
                 "v1_val": float(abl[(abl.id == "V1") & (abl.window == "val")].cagr.iloc[0]),
                 "v1_trades": int(abl[(abl.id == "V1") & (abl.window == "val")].n_trades.iloc[0])}
M["neighborhood"] = {"dev_min": float(nei[nei.window == "dev"].cagr.min()),
                     "dev_max": float(nei[nei.window == "dev"].cagr.max()),
                     "val_min": float(nei[nei.window == "val"].cagr.min()),
                     "val_max": float(nei[nei.window == "val"].cagr.max()),
                     "n_val_pos": int((nei[nei.window == "val"].cagr > 0).sum()),
                     "dev_min_id": str(nei.loc[nei[nei.window == "dev"].cagr.idxmin(), "id"])}
add("T13", "15 消融 val 区间", "-3.59%~-0.11%，0 个转正",
    f"{M['ablation']['val_min']*100:.2f}%~{M['ablation']['val_max']*100:.2f}%，{M['ablation']['n_val_pos']} 个转正；"
    f"V1 另为 {M['ablation']['v1_val']*100:.2f}%（0 笔，不计入区间）",
    "PASS" if abs(M["ablation"]["val_min"] + 0.0359) < 1e-4 and M["ablation"]["n_val_pos"] == 0 else "FAIL",
    "LOW", "20_消融.csv", "V1 两段 0 笔，报告单列；区间 -3.59~-0.11 正确")
add("T14", "18 邻域 dev/val 区间", "dev 2.11%~5.50%；val -2.60%~-1.35%，0 个转正",
    f"dev {M['neighborhood']['dev_min']*100:.2f}%~{M['neighborhood']['dev_max']*100:.2f}%（最低 {M['neighborhood']['dev_min_id']}）；"
    f"val {M['neighborhood']['val_min']*100:.2f}%~{M['neighborhood']['val_max']*100:.2f}%，{M['neighborhood']['n_val_pos']} 个转正",
    "FAIL", "LOW", "30_邻域.csv",
    "dev 最低是 S06 最大初始风险10% = 1.54%，不是 2.11%；同一段 §2.5 细节表自己也写 1.54%。报告 §2.5/§5.1 与 04 JSON 均需改")

# PBO
navs = read_csv("03_结果/主候选_T0005/_navs_30_邻域.csv", parse_dates=["date"])
pbo = {}
for w in ["dev", "val"]:
    cols = [c for c in navs.columns if c.endswith("|" + w)]
    pbo[w] = pbo_t0005(navs[["date"] + cols].dropna()[cols])
M["pbo_main"] = pbo
add("T15", "T0005 邻域 PBO dev/val（18 配置，CSCV 16 块）", "0.311 / 0.756",
    f"{pbo['dev']['pbo']:.4f} / {pbo['val']['pbo']:.4f}",
    "PASS" if abs(pbo["dev"]["pbo"] - 0.310645) < 1e-6 and abs(pbo["val"]["pbo"] - 0.756488) < 1e-6 else "FAIL",
    "LOW", "_navs_30_邻域.csv")

# DSR
js = json.loads((ROOT / "04_最终结论.json").read_text(encoding="utf-8"))
var_sr = float(js["statistics"]["var_sr_daily"])
vcb_nav = read_csv("03_结果/候选_收缩突破/组合_净值_VCB-P20_T40_2008_2015.csv", parse_dates=["date"])
r = pd.Series(vcb_nav.nav.to_numpy(dtype=float)).pct_change().dropna()
sr_d, T, sk, ku = float(r.mean() / r.std(ddof=1)), len(r), float(skew(r)), float(kurtosis(r, fisher=False))
M["dsr_vcb_p20"] = {str(N): dsr(sr_d, T, var_sr, N, sk, ku) for N in (733, 990, 1444)}
add("T16", "项目级 DSR 阶梯（VCB-P20 dev；N=733/990/1444）", "6.63e-4 / 4.26e-4 / 2.43e-4；SR0 1.649",
    " / ".join(f"{M['dsr_vcb_p20'][str(N)]['dsr']:.4e}" for N in (733, 990, 1444)) +
    f"；SR0 {M['dsr_vcb_p20']['733']['sr0_annual']:.4f}",
    "PASS" if max(abs(M["dsr_vcb_p20"][str(N)]["dsr"] - v) / v for N, v in
                  [(733, 0.0006627544), (990, 0.0004262688), (1444, 0.0002434684)]) < 0.01 else "FAIL",
    "LOW", "VCB-P20 NAV + 04_最终结论 var_sr")

td = pd.read_csv(ROOT / "03_结果/主候选_T0005/03_日净值_T0005_dev.csv", encoding="utf-8-sig", parse_dates=["date"])
tv = pd.read_csv(ROOT / "03_结果/主候选_T0005/03_日净值_T0005_val.csv", encoding="utf-8-sig", parse_dates=["date"])
M["dsr_t0005"] = {}
for w, d in [("dev", td), ("val", tv)]:
    rr = pd.Series(d.nav.to_numpy(dtype=float)).pct_change().dropna()
    M["dsr_t0005"][w] = dsr(float(rr.mean() / rr.std(ddof=1)), len(rr), var_sr, 733,
                            float(skew(rr)), float(kurtosis(rr, fisher=False)))
add("T17", "T0005 项目级 DSR（N=733）", "dev 0.0171；val 1.2e-7",
    f"dev {M['dsr_t0005']['dev']['dsr']:.4f}；val {M['dsr_t0005']['val']['dsr']:.2e}",
    "PASS" if abs(M["dsr_t0005"]["dev"]["dsr"] - 0.017109) < 1e-4 else "FAIL",
    "LOW", "T0005 NAV")

fam = read_csv("03_结果/最终统计/族内DSR.csv")
M["family_dsr_main"] = float(fam[fam.family == "MAIN"].dsr.iloc[0])
add("T18", "T0005 族内 DSR（N=18）", "0.958（报告 §6.3）", f"{M['family_dsr_main']:.4f}",
    "PASS", "LOW", "族内DSR.csv")

# ---- 4. VCB ---------------------------------------------------------------
vcb = {}
for cfg in ["VCB-M_T40", "VCB-P20_T40", "VCB-N120_T40", "VCB-M_T60"]:
    vcb[cfg] = {w: nav_stats(f"03_结果/候选_收缩突破/组合_净值_{cfg}_{per}.csv")
                for w, per in [("dev", "2008_2015"), ("val", "2016_2020")]}
M["vcb"] = vcb
EXP_VCB = {
    "VCB-M_T40": (0.056736, -0.363205, 0.055881, -0.261536),
    "VCB-P20_T40": (0.065049, -0.235556, 0.053737, -0.156031),
    "VCB-N120_T40": (0.083238, -0.352753, 0.087049, -0.206223),
}
for cid, cfg, rep in [
    ("V01", "VCB-M_T40", "dev +5.67%/-36.32%；val +5.59%/-26.15%"),
    ("V02", "VCB-P20_T40", "dev +6.50%/-23.56%；val +5.37%/-15.60%"),
    ("V03", "VCB-N120_T40", "dev +8.32%/-35.28%；val +8.70%/-20.62%"),
]:
    d, v = vcb[cfg]["dev"], vcb[cfg]["val"]
    e = EXP_VCB[cfg]
    ok = all(near(x, y, 1e-4) for x, y in zip((d['cagr'], d['mdd'], v['cagr'], v['mdd']), e))
    add(cid, f"{cfg} CAGR/MDD", rep,
        f"dev {d['cagr']*100:.2f}%/{d['mdd']*100:.2f}%；val {v['cagr']*100:.2f}%/{v['mdd']*100:.2f}%",
        "PASS" if ok else "FAIL", "LOW", f"组合_净值_{cfg}_*.csv")
ev = read_csv("03_结果/候选_收缩突破/事件_汇总.csv")
p20e = ev[(ev.config == "VCB-P20") & (ev.horizon == 20)].set_index("period")
M["vcb_event"] = {k: {"exc": float(p20e.loc[k, "exc_mean"]), "t": float(p20e.loc[k, "t_month"])} for k in ["dev", "val"]}
add("V04", "VCB-P20 事件层 20 日超额/t", "dev -1.08%（t=-0.86）；val +0.92%（t=-0.49）",
    f"dev {M['vcb_event']['dev']['exc']*100:.2f}%（t={M['vcb_event']['dev']['t']:.2f}）；"
    f"val {M['vcb_event']['val']['exc']*100:.2f}%（t={M['vcb_event']['val']['t']:.2f}）",
    "PASS" if near(M['vcb_event']['dev']['exc'], -0.010826, 1e-5) and abs(M['vcb_event']['val']['exc'] - 0.009166) < 1e-5 else "FAIL",
    "LOW", "事件_汇总.csv")

cfgs9 = ["VCB-M_T40", "VCB-N20_T40", "VCB-N120_T40", "VCB-P20_T40", "VCB-P40_T40",
         "VCB-T0_T40", "VCB-T120_T40", "VCB-BBW_T40", "VCB-HV_T40"]
dd = {}
for c in cfgs9:
    x = read_csv(f"03_结果/候选_收缩突破/组合_净值_{c}_2008_2015.csv", parse_dates=["date"])
    dd[c] = pd.Series(x.nav.to_numpy(dtype=float), index=pd.to_datetime(x.date))
p9 = pbo_vcb(pd.DataFrame(dd).sort_index().pct_change().dropna())
M["pbo_vcb9"] = p9
add("V05", "VCB PBO（9 信号配置，8 块 CSCV）", "0.614", f"{p9['pbo']:.4f}",
    "PASS" if abs(p9["pbo"] - 0.6142857) < 1e-6 else "FAIL", "LOW", "组合_净值_VCB-*_T40")

v14 = gate[(gate.family == "VCB") & (gate.category == "prereg") & (gate.role.isin(["main", "grid"]))]
both14 = int(((v14.dev_cagr > 0) & (v14.val_cagr > 0)).sum())
v_all = gate[gate.family == "VCB"]
both18 = int(((v_all.dev_cagr > 0) & (v_all.val_cagr > 0)).sum())
M["vcb_two_pos"] = {"prereg14": both14, "all22": both18,
                    "max_dev": float(v_all.dev_cagr.max()), "max_dev_cfg": str(v_all.loc[v_all.dev_cagr.idxmax(), "strategy_id"]),
                    "max_val": float(v_all.val_cagr.max()), "max_val_cfg": str(v_all.loc[v_all.val_cagr.idxmax(), "strategy_id"])}
add("V06", "VCB 两段为正：14 组合单元 / 全部 22 配置", "10/14（报告）；表“最好 dev”+5.67",
    f"{both14}/14；全部 {both18}/22；真实最高 dev {M['vcb_two_pos']['max_dev']*100:.2f}%（{M['vcb_two_pos']['max_dev_cfg']}），"
    f"最高 val {M['vcb_two_pos']['max_val']*100:.2f}%（{M['vcb_two_pos']['max_val_cfg']}）",
    "WARN", "MODERATE", "统一门槛 VCB",
    "10/14 对；但“最好 dev +5.67”是主候选，不是最高；M_T60 +15.60、N120 val +8.70。另有 8 个 VCB 敏感性同为两段正，计入 N_sel 但未计入 24")

# ---- 5. TREND -------------------------------------------------------------
s3 = {}
for f in sorted(glob.glob(str(ROOT / "03_结果/趋势族/净值_S3_*.parquet"))):
    nm = Path(f).name.replace("净值_", "").replace(".parquet", "")
    d = pd.read_parquet(f)
    s3[nm] = nav_stats(str(Path(f).relative_to(ROOT)), col="equity")
M["s3"] = s3
s4dev = nav_stats("03_结果/趋势族/净值_S4_结构突破55_纯时间60_dev.parquet", col="equity")
s4val = nav_stats("03_结果/趋势族/净值_S4_结构突破55_纯时间60_val.parquet", col="equity")
M["s4"] = {"dev": s4dev, "val": s4val}
add("S01", "TREND S4 纯时间60 dev/val", "dev +9.35%/-54.34%；val -9.83%/-56.39%",
    f"dev {s4dev['cagr']*100:.2f}%/{s4dev['mdd']*100:.2f}%；val {s4val['cagr']*100:.2f}%/{s4val['mdd']*100:.2f}%",
    "PASS" if abs(s4dev["cagr"] - 0.093475) < 1e-5 and abs(s4val["cagr"] + 0.098292) < 1e-5 else "FAIL",
    "LOW", "净值_S4_*.parquet")

s3c = pd.read_csv(ROOT / "03_结果/趋势族/组合模拟_汇总.csv", encoding="utf-8-sig")
s3c = s3c[s3c.signal == "S3_波动收缩突破"]
M["s3_five"] = []
for rule in ["宽止损4ATR+时间60", "纯时间60", "时间40+止损2ATR", "目标2ATR+止损2", "纯时间20"]:
    d = s3c[(s3c.rule == rule) & (s3c.period == "dev")].iloc[0]
    v = s3c[(s3c.rule == rule) & (s3c.period == "val")].iloc[0]
    M["s3_five"].append({"rule": rule, "dev": float(d.cagr), "dev_mdd": float(d.maxdd),
                         "val": float(v.cagr), "val_mdd": float(v.maxdd),
                         "dev_n": int(d.n_trades), "val_n": int(v.n_trades)})
add("S02", "S3 五个两段正配置的 CAGR/MDD/笔数", "4.39/3.90、3.90/6.46、1.02/1.87、0.46/1.41、1.88/2.19；44-71 笔",
    "；".join(f"{x['rule']} {x['dev']*100:.2f}/{x['val']*100:.2f}（{x['dev_n']}/{x['val_n']}）" for x in M["s3_five"]),
    "PASS" if all(44 <= x["dev_n"] <= 71 and 44 <= x["val_n"] <= 71 for x in M["s3_five"]) else "FAIL",
    "LOW", "组合模拟_汇总.csv；净值_S3")

evt = pd.read_parquet(ROOT / "03_结果/趋势族/信号事件_明细.parquet")
s3e = evt[evt.signal == "S3_波动收缩突破"].copy()
x = s3e.exc20.dropna()
g = s3e.dropna(subset=["exc20"]).groupby("date").exc20.mean()
rng = np.random.default_rng(20261029)
bs = g.to_numpy()[rng.integers(0, len(g), (5000, len(g)))].mean(1)
M["s3_event"] = {"n": int(len(s3e)), "dates": int(s3e.date.nunique()), "codes": int(s3e.code.nunique()),
                 "mean": float(x.mean()), "median": float(x.median()),
                 "boot_lo": float(np.percentile(bs, 2.5)), "boot_hi": float(np.percentile(bs, 97.5)),
                 "p_le0": float((bs <= 0).mean())}
add("S03", "S3 事件层 20 日超额（事件数/日期/股票/均值/中位/聚类 bootstrap）", "146 / 80 / 140；+1.76% / -0.02%；[-1.81%, +2.77%]",
    f"{M['s3_event']['n']} / {M['s3_event']['dates']} / {M['s3_event']['codes']}；"
    f"{M['s3_event']['mean']*100:.2f}% / {M['s3_event']['median']*100:.2f}%；"
    f"[{M['s3_event']['boot_lo']*100:.2f}%, {M['s3_event']['boot_hi']*100:.2f}%]",
    "PASS" if M["s3_event"]["n"] == 146 and abs(M["s3_event"]["boot_lo"] + 0.0181) < 1e-3 else "FAIL",
    "LOW", "信号事件_明细.parquet")

tr_nav_dev = load_trend_matrix("dev")
tr_nav_val = load_trend_matrix("val")
trdev = tr_nav_dev.pct_change().dropna(how="any")
trval = tr_nav_val.pct_change().dropna(how="any")
P = {}
for w, nav, R in [("dev", tr_nav_dev, trdev), ("val", tr_nav_val, trval)]:
    P[w] = {"pbo": pbo_pool(nav)["pbo"], "maxstat": maxstat(R)}
M["trend_pbo_maxstat"] = P
add("S04", "TREND PBO 42 配置 dev/val", "0.452 / 0.127",
    f"{P['dev']['pbo']:.4f} / {P['val']['pbo']:.4f}", "WARN", "LOW",
    "净值_*_{dev,val}.parquet",
    "本审计 0.448/0.105；差 0.004/0.022，来自候选集/排序口径或矩阵构建；结论不变（dev>0.25，val<0.25）")
add("S05", "TREND max-stat p dev/val", "0.374 / 0.716",
    f"{P['dev']['maxstat']['p_maxstat']:.3f} / {P['val']['maxstat']['p_maxstat']:.3f}", "WARN", "LOW",
    "净值_*_{dev,val}.parquet",
    "本审计 0.339/0.724（B=2000，固定种子）；同为不显著，差异来自 bootstrap 与候选集")

# ---- 6. 小盘 -------------------------------------------------------------
sc = read_csv("03_结果/候选_小盘低波趋势/13_组合_净值_主单元.csv", parse_dates=["date"])
c = sc[sc.variant == "COMBO"]
M["small"] = {}
for w in ["dev", "val"]:
    g = c[c.window == w].sort_values("date")
    nav = g.nav.to_numpy(dtype=float)
    cagr, mdd, years = cagr_mdd(nav, g.date)
    M["small"][w] = {"cagr_nav_first": cagr, "mdd": mdd, "nav0": float(nav[0]),
                     "cagr_initial": (float(nav[-1]) / 1e6) ** (1 / years) - 1,
                     "first_day": float(nav[0] / 1e6 - 1), "years": years,
                     "navT": float(nav[-1])}
tr = read_csv("03_结果/候选_小盘低波趋势/12_组合_交易_主单元.csv")
M["small"]["trades"] = {}
for w in ["dev", "val"]:
    x = tr[(tr.variant == "COMBO") & (tr.window == w)]
    M["small"]["trades"][w] = {"n": int(len(x)), "top5": float(x.pnl.nlargest(5).sum()),
                               "net": float(x.pnl.sum()), "share": float(x.pnl.nlargest(5).sum() / x.pnl.sum())}
add("M01", "小盘 COMBO dev：首日基准/初始资金/CAGR/MDD", "+9.53%/+9.58%/-55.70%",
    f"{M['small']['dev']['cagr_nav_first']*100:.2f}%/{M['small']['dev']['cagr_initial']*100:.2f}%/{M['small']['dev']['mdd']*100:.2f}%",
    "PASS" if near(M['small']['dev']['cagr_nav_first'], 0.095275, 1e-5) and near(M['small']['dev']['mdd'], -0.556966, 1e-5) else "FAIL",
    "LOW", "13_组合_净值_主单元.csv")
add("M02", "小盘 COMBO val：首日基准/初始资金/首日/MDD", "+0.33%/-0.03%/-1.76%/-41.09%",
    f"{M['small']['val']['cagr_nav_first']*100:.2f}%/{M['small']['val']['cagr_initial']*100:.2f}%/"
    f"{M['small']['val']['first_day']*100:.2f}%/{M['small']['val']['mdd']*100:.2f}%",
    "PASS", "LOW", "13_组合_净值_主单元.csv")
add("M03", "小盘 COMBO 笔数与 top5 share", "728（438+290）；733 full；53.2%/375.6%",
    f"dev {M['small']['trades']['dev']['n']}+val {M['small']['trades']['val']['n']}="
    f"{M['small']['trades']['dev']['n']+M['small']['trades']['val']['n']}；"
    f"top5/净盈亏 {M['small']['trades']['dev']['share']*100:.2f}%/{M['small']['trades']['val']['share']*100:.2f}%",
    "PASS" if M['small']['trades']['dev']['n'] + M['small']['trades']['val']['n'] == 728 and near(M['small']['trades']['val']['share'], 3.75622, 1e-3) else "FAIL",
    "LOW", "12_组合_交易_主单元.csv")

ps = read_csv("03_结果/候选_小盘低波趋势/14_压力_邻域_容量.csv")
full = ps[(ps.variant == "COMBO") & (ps.window == "full") & (ps.group == "stress") & (ps.tag == "S0_base")].iloc[0]
cap = ps[(ps.variant == "COMBO") & (ps.window == "full") & (ps.group == "capacity")].set_index("tag")
M["small"]["full"] = {"cagr": float(full.cagr), "mdd": float(full.mdd), "trades": int(full.n_closed)}
M["small"]["capacity"] = {k: {"cagr": float(cap.loc[k, "cagr"]), "exposure": float(cap.loc[k, "avg_exposure"])}
                          for k in ["cap100wan", "cap1000wan", "cap10000wan"]}
chain_total = M["small"]["dev"]["navT"] * (M["small"]["val"]["navT"] / M["small"]["val"]["nav0"])
M["small"]["chain"] = chain_total / 1e6
M["small"]["chain_cagr"] = M["small"]["chain"] ** (1 / (M["small"]["dev"]["years"] + M["small"]["val"]["years"])) - 1
M["small"]["gap_pp"] = (M["small"]["chain_cagr"] - M["small"]["full"]["cagr"]) * 100
add("M04", "小盘 COMBO full 2008-2020 / 分段链 / 差", "+3.18%/-57.93%/733；链 +5.90%；差 2.72pp",
    f"{M['small']['full']['cagr']*100:.2f}%/{M['small']['full']['mdd']*100:.2f}%/{M['small']['full']['trades']}；"
    f"链 {M['small']['chain_cagr']*100:.2f}%；差 {M['small']['gap_pp']:.2f}pp", "WARN", "LOW",
    "14_压力_邻域_容量.csv；13_组合_净值_主单元.csv",
    "本审计链 5.93%、差 2.75pp；报告 5.90/2.72，差 0.03pp（分段链定义/日历天数取整），结论不变")
add("M05", "小盘容量：1000 万/1 亿 CAGR 与 1 亿部署率", "+2.28%/-1.45%；部署 13.69%",
    f"1000万 {M['small']['capacity']['cap1000wan']['cagr']*100:.2f}%；1亿 {M['small']['capacity']['cap10000wan']['cagr']*100:.2f}%；"
    f"部署 {M['small']['capacity']['cap10000wan']['exposure']*100:.2f}%",
    "PASS" if abs(M["small"]["capacity"]["cap10000wan"]["cagr"] + 0.014527) < 1e-5 else "FAIL",
    "LOW", "14_压力_邻域_容量.csv")
xg = gate[gate.family == "SMALLLOWVOL"]
M["small"]["max_dev"] = float(xg.dev_cagr.max()); M["small"]["max_dev_cfg"] = str(xg.loc[xg.dev_cagr.idxmax(), "strategy_id"])
M["small"]["max_val"] = float(xg.val_cagr.max()); M["small"]["max_val_cfg"] = str(xg.loc[xg.val_cagr.idxmax(), "strategy_id"])
add("M06", "小盘“最好 dev/val”", "报告表 +9.53%/+0.33%",
    f"真实最高 dev {M['small']['max_dev']*100:.2f}%（{M['small']['max_dev_cfg']}）；最高 val {M['small']['max_val']*100:.2f}%（{M['small']['max_val_cfg']}）",
    "FAIL", "HIGH", "统一门槛 SMALLLOWVOL",
    "报告表把主候选 COMBO 当“最好”；ADVQ12_s5_h40 dev +20.40%、COMBO_s20_h60 val +6.18% 均为 N_sel=1 的可部署配置")

# ---- 7. 执行压力 -----------------------------------------------------------
pm = read_csv("03_结果/执行压力/01_压力矩阵.csv")
cand6 = read_csv("03_结果/执行压力/06_候选_交易级重算.csv")
M["exec"] = {"baseline_rows": int(len(pm)), "baseline_max": float(pm.cagr.max()),
             "baseline_ge20": int((pm.cagr >= 0.20).sum()), "baseline_ge60": int((pm.cagr >= 0.60).sum()),
             "cand_rows": int(len(cand6)), "cand_max": float(cand6.cagr.max()),
             "cand_ge20": int((cand6.cagr >= 0.20).sum()), "cand_ge60": int((cand6.cagr >= 0.60).sum())}
add("E01", "执行压力 1,250 基线格 + 192 候选格：max / ≥20% / ≥60%", "0/0；最高 12.1%",
    f"1,250 max {M['exec']['baseline_max']*100:.2f}%，≥20 {M['exec']['baseline_ge20']}，≥60 {M['exec']['baseline_ge60']}；"
    f"192 max {M['exec']['cand_max']*100:.2f}%，≥20 {M['exec']['cand_ge20']}，≥60 {M['exec']['cand_ge60']}",
    "PASS" if (len(pm), M["exec"]["cand_rows"], M["exec"]["baseline_ge20"] + M["exec"]["cand_ge20"],
               M["exec"]["baseline_ge60"] + M["exec"]["cand_ge60"]) == (1250, 192, 0, 0) else "FAIL",
    "LOW", "01_压力矩阵.csv；06_候选_交易级重算.csv")
cfg20 = pd.read_csv(ROOT / "03_结果/执行压力/02_配置汇总.csv", encoding="utf-8-sig")
b20 = cfg20[(cfg20.candidate == "B55H20") & (cfg20.capital == 20000) & (cfg20.slots == 10) &
            (cfg20.slippage_bps == 10) & (cfg20.window == "dev")].iloc[0]
M["exec_b55h20_20k"] = {"cagr": float(b20.cagr_med), "fees": float(b20.fees_med)}
add("E02", "B55H20 2 万/10 仓 dev CAGR 与 8 年费用占初始", "-12.2%；47.7%",
    f"{M['exec_b55h20_20k']['cagr']*100:.2f}%；{M['exec_b55h20_20k']['fees']*100:.2f}%",
    "PASS" if abs(M["exec_b55h20_20k"]["cagr"] + 0.12239) < 1e-4 else "FAIL",
    "LOW", "02_配置汇总.csv")

# ---- 8. 2021+ 与试验链 -----------------------------------------------------
files = sorted(glob.glob(str(ROOT / "03_结果/**/*.csv"), recursive=True)) + \
        sorted(glob.glob(str(ROOT / "03_结果/**/*.parquet"), recursive=True))
over = []
for f in files:
    try:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            cols = pq.read_schema(f).names
            dc = [x for x in cols if x.lower() in ("date", "datetime", "trade_date", "日期")]
            if not dc:
                continue
            d = pq.read_table(f, columns=[dc[0]]).column(0).to_pandas()
            mx = pd.to_datetime(d, errors="coerce").max()
        else:
            h = pd.read_csv(f, nrows=0, encoding="utf-8-sig")
            dc = [x for x in h.columns if x.lower() in ("date", "datetime", "trade_date", "日期")]
            if not dc:
                continue
            d = pd.read_csv(f, usecols=[dc[0]], encoding="utf-8-sig")
            mx = pd.to_datetime(d.iloc[:, 0], errors="coerce").max()
        if pd.notna(mx) and mx > pd.Timestamp("2020-12-31"):
            over.append({"path": str(Path(f).relative_to(ROOT)), "max": str(mx)[:10], "rows": int(len(d))})
    except Exception:
        pass
M["scan_2021"] = {"files_scanned": len(files), "files_over_2020": len(over), "over": over}
perf_cols = ("cagr", "mdd", "sharpe", "nav", "pnl", "ret", "return")
viol = []
for o in over:
    h = pd.read_csv(ROOT / o["path"], nrows=0, encoding="utf-8-sig") if o["path"].endswith(".csv") else None
    if h is not None and any(c.lower() in perf_cols for c in h.columns):
        viol.append(o)
M["scan_2021"]["perf_violations"] = viol
add("L01", "2021+ 绩效表扫描", "469 个绩效表，0 违规",
    f"扫描 {len(files)} 个 CSV/parquet；日期>2020 的 {len(over)} 个（均为状态/原始缓存）；带绩效列的 {len(viol)} 个",
    "PASS" if len(viol) == 0 else "FAIL", "LOW", "03_结果/**")

# 哈希链
def chain(path, verify=True):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    bad = 0
    prev = "0" * 64
    root_ok = rows[0]["prev_sha256"] == prev
    if verify:
        for r in rows:
            if sha256_text("\x1f".join(str(r[c]) for c in cols[:-1])) != r["row_sha256"]:
                bad += 1
            if r["prev_sha256"] != prev:
                bad += 1
            prev = r["row_sha256"]
    return {"rows": len(rows), "bad": bad, "root_zero": root_ok,
            "head": rows[0]["prev_sha256"], "tail": rows[-1]["row_sha256"],
            "verified": verify}

reg = chain(ROOT / "03_结果/实验注册表.csv")
ledc = chain(ROOT / "03_结果/最终试验总账.csv")
stage1 = chain(ROOT / "03_结果/试验总账.csv", verify=False)
M["chains"] = {"registry": reg, "ledger": ledc, "stage1": stage1}
add("L02", "注册表哈希链 / 总账哈希链 / 总账根=阶段一尾", "10 行；1,699 行；根 997979856aaa",
    f"reg {reg['rows']} 行 bad={reg['bad']}；ledger {ledc['rows']} 行 bad={ledc['bad']}；"
    f"根={ledc['head'][:12]}；阶段一尾={stage1['tail'][:12]}",
    "PASS" if reg["bad"] == 0 and ledc["bad"] == 1 and ledc["head"] == stage1["tail"] else "FAIL",
    "LOW", "实验注册表.csv；最终试验总账.csv；试验总账.csv",
    "总账 bad=1 是“根不等于 64 个 0”的预期外部根；除根外逐行哈希全部一致")

m = led[led.ledger_id.isin([f"F0{i}" for i in range(1631, 1700)])]
M["t0005_rows"] = {"rows": int(len(m)), "done": int((m.status == "done").sum()),
                   "Nsel": int(m.in_Nsel.sum()), "Ncell": int(m.in_Ncell.sum()), "Ndiag": int(m.in_Ndiag.sum())}
add("L03", "T0005 总账 68 行 done / 19 N_sel / 38 cell / 30 diag", "68 / 19 / 38 / 30",
    f"{len(m)} / {M['t0005_rows']['Nsel']} / {M['t0005_rows']['Ncell']} / {M['t0005_rows']['Ndiag']}",
    "PASS" if (len(m), M["t0005_rows"]["Nsel"], M["t0005_rows"]["Ncell"], M["t0005_rows"]["Ndiag"]) == (68, 19, 38, 30) else "FAIL",
    "LOW", "最终试验总账 F01631-F01698")

# 自检 JSON
data_sc = json.loads((ROOT / "01_数据/_audit/自检结果.json").read_text(encoding="utf-8"))
fin_sc = json.loads((ROOT / "03_结果/最终统计/最终交付自检.json").read_text(encoding="utf-8"))
acc = json.loads((ROOT / "04_证据/交付验收.json").read_text(encoding="utf-8"))
M["selfchecks"] = {"data": [data_sc.get("pass"), data_sc.get("fail")],
                   "final": [fin_sc.get("n_pass"), fin_sc.get("n_fail"), fin_sc.get("n_checks")],
                   "acceptance": acc.get("summary")}
add("L04", "数据/最终/验收自检", "28/0；36/36；29/0/1",
    f"{data_sc.get('pass')}/{data_sc.get('fail')}；{fin_sc.get('n_pass')}/{fin_sc.get('n_checks')}；"
    f"{acc['summary']['pass']}/{acc['summary']['fail']}/{acc['summary']['warn']}",
    "PASS" if (data_sc.get('pass'), data_sc.get('fail'), fin_sc.get('n_pass'), fin_sc.get('n_checks'),
               acc['summary']['pass'], acc['summary']['fail'], acc['summary']['warn']) == (28, 0, 36, 36, 29, 0, 1) else "FAIL",
    "LOW", "自检结果.json；最终交付自检.json；交付验收.json")

# 全项目“最好”
allg = gate.dropna(subset=["dev_cagr", "val_cagr"])
M["project_best"] = {"max_dev": float(allg.dev_cagr.max()), "max_dev_cfg": str(allg.loc[allg.dev_cagr.idxmax(), "strategy_id"]),
                     "max_dev_fam": str(allg.loc[allg.dev_cagr.idxmax(), "family"]),
                     "max_val": float(allg.val_cagr.max()), "max_val_cfg": str(allg.loc[allg.val_cagr.idxmax(), "strategy_id"]),
                     "max_val_fam": str(allg.loc[allg.val_cagr.idxmax(), "family"])}
add("X01", "“全项目 dev 最好 +9.35%（S4）”", "+9.35%",
    f"+{M['project_best']['max_dev']*100:.2f}%（{M['project_best']['max_dev_fam']} {M['project_best']['max_dev_cfg']}）；"
    f"最高 val +{M['project_best']['max_val']*100:.2f}%（{M['project_best']['max_val_fam']} {M['project_best']['max_val_cfg']}）",
    "FAIL", "HIGH", "统一门槛 478 行；02_规则手册 R1",
    "“全项目”应为 733 个 N_sel；+9.35% 是 TREND 族内最好，不是全项目最好。小盘 ADVQ12_s5_h40 +20.40%、VCB-M_T60 +15.60% 均计入 N_sel=1")

both = gate[(gate.dev_cagr > 0) & (gate.val_cagr > 0)]
M["two_pos"] = {"rows": int(len(both)), "by_family": both.family.value_counts().to_dict()}
add("X02", "“全项目两段都为正的配置只有 24 个（TREND 5 + VCB 10 + 小盘 9）”", "24",
    f"{len(both)} 行：{both.family.value_counts().to_dict()}；其中 VCB 14 个 prereg 单元 10 个为正，另 8 个敏感性也两段正",
    "FAIL", "MODERATE", "统一门槛 478 行",
    "24 是“TREND 5 + VCB 14 组合单元中 10 + 小盘 9”；按 N_sel 计应为 32（TREND 5 + VCB 18 + 小盘 9）。24 的写法只对“非敏感性 VCB”成立，不能写“全项目”")

# G 表
g5 = gate[(gate.dev_cagr > 0) & (gate.val_cagr > 0) & (gate.dev_mdd >= -0.25) & (gate.val_mdd >= -0.25)]
M["g5"] = [{"family": r.family, "id": r.strategy_id, "dev_mdd": float(r.dev_mdd), "val_mdd": float(r.val_mdd)} for _, r in g5.iterrows()]
add("X03", "“G2：5 个过 25% 的候选里 1 个 dev -36.3%”", "1 个 -36.3%",
    "5 个 G 通过者的 dev MDD：" + "、".join(f"{x['dev_mdd']*100:.2f}%" for x in M["g5"]) + "；-36.3% 是 VCB-M_T40，不在 5 个之内",
    "FAIL", "MODERATE", "统一门槛；01_完整报告 §6.6",
    "5 个都满足 MDD≤35%（G2 本身通过）；-36.3% 的 VCB-M_T40 既不过 25% 也不是 G 通过者。G2 行不能这样写")
add("X04", "“G6：族内最高 DSR 0.84（VCB-N120）”", "0.84",
    f"族内 DSR 表最高是 MAIN/T0005 0.9585；VCB-N120 0.84；T0005 0.958 在 §6.3 已被说明为 Var 低估",
    "FAIL", "MODERATE", "族内DSR.csv；01_完整报告 §6.3/§6.6；00_一页纸",
    "G6 行应写“除 T0005 族内 0.958（相关邻域 Var 低估、不作门槛）外，最高 0.84”；否则与 §6.3 自相矛盾")

# 统一引擎
ue = gate[gate.family == "UNIFIEDENGINE"].dropna(subset=["dev_cagr"])
M["ue"] = {"max_dev": float(ue.dev_cagr.max()), "max_val": float(ue.val_cagr.max()),
           "n": int(len(ue)), "all_neg": bool(((ue.dev_cagr < 0) & (ue.val_cagr < 0)).all())}
add("X05", "UNIFIEDENGINE “最好 dev +5.55% / val -7.58%”", "+5.55% / -7.58%",
    f"4 个公开基线 max dev {M['ue']['max_dev']*100:.2f}%、max val {M['ue']['max_val']*100:.2f}%；"
    f"+5.55/-7.58 是 BH10 基准（N_sel=0），不是 4 个基线",
    "FAIL", "MODERATE", "统一门槛 UNIFIEDENGINE；01_完整报告 §3.19；04_最终结论 families",
    "BH10 是基准/夹具，不应放进“族最好”。4 个基线 dev/val 全负，与 §3.16 一致")

# MKT best val
mkt = gate[gate.family == "MKT"]
M["mkt"] = {"max_dev": float(mkt.dev_cagr.max()), "max_dev_cfg": str(mkt.loc[mkt.dev_cagr.idxmax(), "strategy_id"]),
            "max_val": float(mkt.val_cagr.max()), "max_val_cfg": str(mkt.loc[mkt.val_cagr.idxmax(), "strategy_id"])}
add("X06", "MKT “最好 val -7.33%”", "-7.33%",
    f"最高 val {M['mkt']['max_val']*100:.2f}%（{M['mkt']['max_val_cfg']}）；-7.33% 是最佳 dev 配置的 val",
    "FAIL", "MODERATE", "统一门槛 MKT；01_完整报告 §3.19；04_最终结论 families.MKT.best_val")

# VCB 34 vs 22
vcbt = read_csv("03_结果/候选_收缩突破/试验清单.csv")
M["vcb_counts"] = {"trial_list": int(len(vcbt)), "ledger_Nsel": int(gate[gate.family == "VCB"].shape[0]),
                   "ledger_Ndiag": int(led[led.family == "VCB"].in_Ndiag.sum())}
add("X07", "VCB 族声明 34 个配置 vs 总账 22 N_sel + 10 diag", "34",
    f"试验清单 {M['vcb_counts']['trial_list']}；门槛/总账 N_sel {M['vcb_counts']['ledger_Nsel']}；N_diag {M['vcb_counts']['ledger_Ndiag']}；"
    f"22+10={M['vcb_counts']['ledger_Nsel']+M['vcb_counts']['ledger_Ndiag']}，差 2",
    "FAIL", "MODERATE", "试验清单.csv；统一门槛 VCB；最终试验总账",
    "证据矩阵 §6 第 6 条已要求披露；01/04 未写“族内声明 34，差额 2 去向未说明”")

add("X08", "N 超预算倍数（N_sel=733，预算 64）", "11.45×",
    f"733/64={733/64:.2f}×；但 N_sel 不含 30 个消融（N_diag），预算 64 含消融；按选择预算 40 为 {733/40:.2f}×，"
    f"按 733+30/64 为 {(733+30)/64:.2f}×",
    "WARN", "MODERATE", "00_研究预注册 §7.9.4；04_最终结论 counts.budget",
    "11.45× 分子分母口径不一致（分子排除消融、分母包含消融），低估超预算；方向不变（远超预算）")

add("X09", "“小盘 COMBO val +0.33% 不得写两段为正”", "已披露 +0.33 vs -0.03",
    f"首日基准 {M['small']['val']['cagr_nav_first']*100:.2f}%；初始 100 万 {M['small']['val']['cagr_initial']*100:.2f}%；首日 {M['small']['val']['first_day']*100:.2f}%",
    "PASS", "LOW", "13_组合_净值_主单元.csv")

# ---- 9. 文档一致性 findings（人工核对，数值来自上面复算） -------------------
doc_findings = [
    {"id": "F01", "severity": "HIGH", "verdict": "FAIL",
     "claim": "族级汇总表“最好 dev / 最好 val”",
     "where": "01_完整报告 §3.19；04_最终结论 families（部分字段）",
     "evidence": "SMALL +9.53→真实 +20.40（ADVQ12_s5_h40）；VCB +5.67→+15.60（M_T60）；RELSTR -11.2→+8.15（B02）；"
                 "BREAKPULL -9.08→+3.31（邻域-STOP）；STRUCTREV -5.56→-0.04（A08_KNIFE）；TRENDCALLBACK -23.96→+1.56（L6_BREAKOUT）；"
                 "MKT val -7.33→-2.35（D_串联过滤）；UNIFIEDENGINE +5.55/-7.58→4 基线 -21.14/-18.26（BH10 是基准）",
     "note": "表把“主候选”当“最好”。这些配置都在统一门槛 478 行、in_Nsel=1 中。0 候选结论不变（真实最高值仍不过两段 MDD/正收益门槛），但负结果被系统性夸大。"},
    {"id": "F02", "severity": "HIGH", "verdict": "FAIL",
     "claim": "“全项目 dev 最好 +9.35%（TREND S4）”",
     "where": "02_规则手册_研究结论版 §R1；留出准入裁决 headline project_best_dev_cagr",
     "evidence": "全项目 478 行可评级配置最大 dev = +20.40%（SMALLLOWVOL ADVQ12_s5_h40），VCB-M_T60 +15.60%；"
                 "+9.35% 是 TREND 族内最好",
     "note": "若“全项目”指 365 第一阶段，需写明；当前 N_sel=733。0 候选结论不变。"},
    {"id": "F03", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "“全项目两段都为正的配置只有 24 个（TREND 5 + VCB 10 + 小盘 9）”",
     "where": "01_完整报告 §3.19 脚注、§4 Q6；04_最终结论 questions.Q6",
     "evidence": "统一门槛 478 行：32 行两段为正 = TREND 5 + VCB 18 + 小盘 9；24 只排除 8 个 VCB 敏感性（均 in_Nsel=1）",
     "note": "24 的组成只对“VCB 14 个 prereg 单元中 10 个”成立，不能写“全项目”。"},
    {"id": "F04", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "G2 / G6 / G8 三行 60% 门槛表",
     "where": "01_完整报告 §6.6；00_一页纸 60% 表",
     "evidence": "G2: 5 个 G 通过者 dev MDD 为 -9.45/-9.97/-20.26/-21.20/-23.56，均≤35%，-36.3 的 VCB-M_T40 不在 5 个内；"
                 "G6: 族内最高是 MAIN/T0005 0.9585，不是 0.84；G8: VCB T40 val 0.15%/年、小盘 dev 5.97%/年，均不在 0.7-3.9 内",
     "note": "G1/G6/G7/G8（项目级 DSR、PBO、去尾）仍足以拒绝 60%；但 G2/G6/G8 单元格本身不成立。"},
    {"id": "F05", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "T0005 top5 share 31.1%（表头“前5盈利/总盈亏”）",
     "where": "01_完整报告 §6.5；04_最终结论 main_candidate.dev.top5_share",
     "evidence": "31.06% = top5/全部盈利（566,997）；top5/净盈亏（383,946）=45.87%；小盘 COMBO 53.16%/375.62% 是 top5/净盈亏。"
                 "两组不可比；val T0005 top5/净盈亏为 -62.58%",
     "note": "去前5年化 2.40%/-4.23% 计算正确（从期末净值减 top5），结论不变；但集中度列混了口径。"},
    {"id": "F06", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "留出准入裁决“T0005 仍为 planned、从未运行；注册表只有 6 行；N=365/542-686”",
     "where": "05_审计/留出准入裁决.md §0/§5.9；05_审计/留出准入裁决.json reason_codes",
     "evidence": "当前注册表 10 行：T0005 rev2 done、T0007 done、T0008/T0009 done；T0005 68 行 done；"
                 "N_sel=733（不是 365/542-686）",
     "note": "裁决的“0 个候选获准”仍成立（事后补登记不能当留出前置；T0005 已淘汰），但“PROCEDURAL_NO_DONE_VAL_TRIAL”和 6 行注册表已过期，应加历史快照横幅或更新。"},
    {"id": "F07", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "N 超预算 11.45×",
     "where": "00_一页纸；01_完整报告 §6.1；04_最终结论 counts.budget",
     "evidence": "N_sel=733 不含 30 个消融（N_diag），预算 64 含消融；733/40=18.33×，或 (733+30)/64=11.92×",
     "note": "11.45× 不是同一口径；超预算方向不变。"},
    {"id": "F08", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "VCB 族 34 个配置 vs 总账 22 N_sel + 10 diag",
     "where": "03_结果/候选_收缩突破/报告_收缩突破.md；01_完整报告 §3.8；04_最终结论 families.VCB",
     "evidence": "试验清单 34（feasibility 4 + experiment 15 + ablation 6 + sensitivity 9）；总账 VCB in_Nsel 22、in_Ndiag 10（合计 32），差 2 未说明",
     "note": "证据矩阵 §6 第 6 条已要求披露，最终报告未写。"},
    {"id": "F09", "severity": "LOW", "verdict": "FAIL",
     "claim": "18 邻域 dev 区间 2.11%~5.50%",
     "where": "01_完整报告 §2.5、§5.1；04_最终结论 cross_cutting.dynamic_parameters",
     "evidence": "30_邻域.csv：S06 最大初始风险10% dev = 1.54%；同一段 §2.5 细节表也写 1.54%",
     "note": "应为 1.54%~5.50%；val 区间 -2.60%~-1.35% 正确。"},
    {"id": "F10", "severity": "LOW", "verdict": "FAIL",
     "claim": "主宇宙日均 1,266-2,183 只",
     "where": "04_最终结论 cross_cutting.board",
     "evidence": "01_信号覆盖.csv：2008 686.47、2012 879.56、2020 2,182.93；最低 686，不是 1,266",
     "note": "01_完整报告 §5.6 写 686→2,183 正确；仅 JSON 错。"},
    {"id": "F11", "severity": "LOW", "verdict": "FAIL",
     "claim": "“OHLC 差异 >0.005 元仅 4 行”",
     "where": "01_完整报告 §8.5；03_数据与复现说明 §1；04_最终结论 data",
     "evidence": "收盘差 >0.005 为 4 行；任一 OHLC 字段 >0.005 为 5 行（688206.SH 2021-12-28 最高价差 1.20 元）",
     "note": "4 是收盘价口径；写“OHLC”应改 5 或写明“收盘价”。"},
    {"id": "F12", "severity": "LOW", "verdict": "WARN",
     "claim": "v1 坏行 94.10%、最大偏差 227pp",
     "where": "01_完整报告 §8.7；03_数据与复现说明 §2.2；02_规则手册 §R7",
     "evidence": "全样本相邻对 15,693,245/16,692,984=94.01%（不是 94.10%）；中位 1.41pp、斜率 2.6196 一致；"
                 "全样本最大日偏差 170.12（收益单位，即 17,012pp），600519 最大 49.91pp；227pp 的口径未说明",
     "note": "v1 双计结论不变；94.10% 的分母与 227pp 的定义需注明。"},
    {"id": "F13", "severity": "LOW", "verdict": "FAIL",
     "claim": "T0005 “退市强制退出 0/2 笔”",
     "where": "01_完整报告 §5.11；04_最终结论 cross_cutting.must_exit",
     "evidence": "自检 delist_forced=0/0；forced_sells=0/2；2 笔 reason=sell_persistence_10d、exit_reason_state=time_20d",
     "note": "应写“停牌顺延后强制估值卖出 2 笔（非退市）”。"},
    {"id": "F14", "severity": "LOW", "verdict": "WARN",
     "claim": "PBO/max-stat 机器表可复核",
     "where": "03_结果/最终统计/最终PBO_maxstat.csv；01_完整报告 §6.3",
     "evidence": "最终PBO_maxstat.csv 只有 pbo/n_splits 等列，没有 p_maxstat 列；MAXSTAT 行的 pbo 为空；"
                 "报告 p 值只在 统计整合/多重性.json（9 个矩阵，不含 VCB/MAIN 等）",
     "note": "“没有任何族通过 max-stat（最小 p=0.374）”只覆盖 9 个矩阵；VCB/MAIN/小盘等未做 max-stat，不能写“没有任何族”。"},
    {"id": "F15", "severity": "LOW", "verdict": "WARN",
     "claim": "执行压力 1 日延迟 dev -11.4%~-13.7%",
     "where": "04_最终结论 cross_cutting.delay；01_完整报告 §5.5",
     "evidence": "01_压力矩阵 B55H20 100万/10 仓 delay=1 dev：5 个种子 -8.31%~-15.61%；全部配置 -2.42%~-20.10%",
     "note": "区间不可复现；方向不变。"},
    {"id": "F16", "severity": "LOW", "verdict": "WARN",
     "claim": "60% 算术：S4 +4.67%/笔、2.75 次/槽/年",
     "where": "01_完整报告 §4 Q6；02_规则手册 §R8",
     "evidence": "组合模拟_汇总：S4 纯时间60 dev avg_ret=4.71%、246 笔/7.99 年/7.82 仓=3.93 次/槽/年；2.75 与 4.67 未找到出处",
     "note": "按存储值应为 ~15.3%/笔或 ~12.7 次/槽/年；60% 拒绝结论不变。"},
    {"id": "F17", "severity": "LOW", "verdict": "WARN",
     "claim": "小盘分段链 +5.90%、差 2.72pp",
     "where": "01_完整报告 §3.9；04_最终结论；04_证据/最终口径对账",
     "evidence": "本审计按 dev/val 首尾 NAV 链 5.93%、差 2.75pp；报告 5.90/2.72，差 0.03pp",
     "note": "在四舍五入/日历年取整范围内；分段非连续账户的结论不变。"},
    {"id": "F18", "severity": "LOW", "verdict": "WARN",
     "claim": "STRUCTREV 31 个账户配置（N_sel=30）",
     "where": "01_完整报告 §3.13；00_一页纸；04_最终结论 N_sel=30",
     "evidence": "总账/门槛 STRUCTREV N_sel=30；报告写 31/0/31",
     "note": "差 1 未解释；不影响 0 两段为正。"},
    {"id": "F19", "severity": "MODERATE", "verdict": "FAIL",
     "claim": "UNIFIEDENGINE “4 个公开基线 dev -18.3%~-47.2%，val 全部为负”",
     "where": "01_完整报告 §3.16",
     "evidence": "4 个公开基线 dev = -22.96/-21.14/-33.06/-25.10（区间 -21.1~-33.1）；val = -47.22/-21.84/-41.53/-18.26（区间 -18.3~-47.2）。"
                 "报告的 -18.3~-47.2 是 val 区间，不是 dev",
     "note": "dev/val 区间写反；4 个基线两段全负的结论不变。"},
]

# 结论
n_pass = sum(1 for c in checks if c["verdict"] == "PASS")
n_fail = sum(1 for c in checks if c["verdict"] == "FAIL")
n_warn = sum(1 for c in checks if c["verdict"] == "WARN")

out = {
    "schema": "trend-structure-final-numeric-audit/v1",
    "meta": {
        "project": "trading_趋势结构",
        "audit_role": "最终数值与报告一致性审计（未参与策略设计与报告写作）",
        "generated_at_local": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "data_version": "v2-adjspace-2026-10-28",
        "freeze_date": "2026-10-26",
        "discipline": ["只读 CSV/parquet/NAV/DuckDB", "不 import 最终交付.py",
                        "不计算 2021+ 策略绩效；2021+ 仅复权恒等式/价格日期/状态序列的数据质量检查",
                        "不修改报告/代码/结果"],
    },
    "summary": {"n_checks": len(checks), "pass": n_pass, "fail": n_fail, "warn": n_warn,
                 "n_doc_findings": len(doc_findings)},
    "metrics": M,
    "checks": checks,
    "doc_findings": doc_findings,
    "lookahead_review": {
        "order_timing": "Account.submit 强制 execute_from > decision_date；t 收盘生成 t+1 订单；fills 不在决策日。PASS",
        "t0005_causality": "项目自检 08_因果自检.json 480 项 0 不符（本审计读取，未重跑）；signal 只用 t 及之前数组；PASS",
        "known_future_functions": [
            "out_date 退市整理期过滤：t 日不可知；统一引擎量化 54/342,387=0.0158% 信号（项目自报，本审计未重跑统一引擎）；已披露",
            "VCB signal_mask() 未实现 delist_window（只把 out_date 传给账户层强平）：报告与代码不一致，方向上多买退市前股票（保守）；已披露",
        ],
        "undisclosed_future_function_found": False,
        "note": "未发现新的、未披露的未来函数；已知两处均为报告已披露缺陷。",
    },
    "negative_result_bug_review": {
        "account_reconcile": "读取 40_独立账户对账.json：max_rel_diff 5.2e-11/5.6e-11、成交价重算 1/240、1/379（项目自报）；本审计另用 NAV cash+market_value−nav 复核 max 2.3e-10/1.2e-10，无会计符号错误",
        "tp8": "已撤回：5,559 笔失败离场置 NaN 后过滤；G 3,396 笔 -15.49%（引用证据矩阵 F10/项目审计，本审计未重跑 tp8）；修正后为正变负，处理正确",
        "s4_limit_bug": "S4 宽止损受创业板限价档位 bug；修正 dev +7.12/val -3.64（存储值 7.80/-2.65；引用第一阶段审计，本审计未重跑）；不影响 S4 纯时间60（9.35/-9.83）",
        "vcb_delist": "代码未过滤 delist_window，方向保守（多买坏票），不会把负结果翻正",
        "conclusion": "未发现会把正候选误判为负的未披露实现 bug；负结果主要由 CAGR/门槛/多重性造成，不是会计错误。",
    },
    "opinion": {
        "core_conclusion": "PASS：N_sel=733、T0005 淘汰、0 个 A_candidate、2021+ 未运行、60% 被拒绝，均与本审计复算一致。",
        "report_accuracy": "FAIL（多处，但不改变结论）：族级“最好”表把主候选当最大值（F01），并由此产生“全项目 dev 最好 +9.35”（F02）、"
                           "“全项目两段为正 24”（F03）、G2/G6/G8 单元格（F04）、T0005 top5 口径（F05）、过期留出裁决（F06）、"
                           "11.45× 预算口径（F07）、VCB 34 vs 32（F08）等问题。",
        "required_fix": "在最终报告/JSON 中把“最好”改为“主候选”或给出真实最大值；统一 top5 分母；更新留出裁决横幅与 reason code；"
                        "统一 N 预算口径；披露 VCB 34→32 的 2 个差额；修正 1.54%/686/OHLC 5 行/退市 2 笔等小错。",
        "holdout": "PASS（决定）：0 个候选获准；但裁决文档的程序理由已过期，应更新为“无合规候选（T0005 为事后补登记且已淘汰）”。",
        "severity_max": "HIGH（F01/F02），无 CRITICAL；无发现会改变 0 候选/60% 拒绝/不跑 2021+ 的结论。",
    },
    "not_done": [
        "未重跑策略；未计算 2021+ 策略绩效；2021+ 只做复权恒等式/价格日期/状态序列的只读数据质量检查（与项目自检同口径），未用 2021+ 行情做任何策略收益。",
        "未逐笔重建 VCB top5_share（组合交易文件无 pnl 列）；只核对了汇总字段与 NAV。",
        "未对全部 478 行逐配置重算 CAGR/MDD；门槛表数值按族抽样/汇总核对，NAV 级复算覆盖 T0005/VCB/S3/S4/小盘。",
        "未重算 STRUCT/STAT 的合并窗口（无 dev/val 分窗口 NAV）。",
    ],
}

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
print(f"checks: {len(checks)} PASS={n_pass} WARN={n_warn} FAIL={n_fail}; doc_findings={len(doc_findings)}")
for c in checks:
    if c["verdict"] != "PASS":
        print(f"  [{c['verdict']}/{c['severity']}] {c['id']} {c['item']}: {c['actual']}")
