# -*- coding: utf-8 -*-
"""
最终试验总账与统计终审（项目收尾）

角色：全项目试验总账 + 统计终审员
独占产物：
  02_代码/最终试验总账.py
  03_结果/最终试验总账.csv
  03_结果/最终统计/

纪律
----
* 冻结日 2026-10-26；收益只允许 2008-01-01..2020-12-31。
* 2021-01-01 起不计算任何收益。脚本对所有载入的净值/交易文件做 max(date) 断言。
* 不改任何他人产物；只读。
* 不为年化 60% 反推任何参数；本脚本不运行新的策略实验。

计数口径（与 03_结果/统计整合/00_整合预注册.md 对齐，并在报告 §2 复述）
------------------------------------------------------------------------
config : 与窗口无关的账户级可部署设定 (family, config_id)
cell   : (config, window) 的一次账户评估
diag   : 事件研究 / 消融 / 交易层 / 状态层诊断（不产生可部署候选）
platform: 事件条件格 / 摩擦情景 / 参数平台格（非可部署选择；单独记 N_platform）
audit  : 覆盖计数、自检、基准、随机对照、复权复算夹具（不进 N_sel/N_diag）

N_sel  = sum(in_Nsel)        —— 主口径，DSR 的 N
N_cell = sum(in_Ncell)
N_diag = sum(in_Ndiag)
N_platform / N_audit 单独披露；N_sens 是 N_sel 的子集标记（可与 N_sel 重叠，不相加）。

DSR 的 N 阶梯（不缩小 N，也不把压力格当独立发现）：
  N1 = N_sel                              （主口径：可部署账户配置）
  N2 = N_sel + N_platform                 （若把平台/摩擦格也算作搜索）
  N3 = N_sel + N_platform + N_diag        （最保守上界）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "03_结果")
FINAL_DIR = os.path.join(RES, "最终统计")
FINAL_LEDGER = os.path.join(RES, "最终试验总账.csv")
STAGE1_LEDGER = os.path.join(RES, "试验总账.csv")

SEED = 20261026
FROZEN = pd.Timestamp("2026-10-26")
DEV_END = pd.Timestamp("2015-12-31")
VAL_START = pd.Timestamp("2016-01-01")
VAL_END = pd.Timestamp("2020-12-31")
B_BOOT = 5000
B_MAXSTAT = 2000
BLOCK = 20
S_CSCV = 16

FINAL_COLS = [
    "ledger_id", "phase", "source", "family", "strategy_id", "params_id", "config_id",
    "window_start", "window_end", "stage", "kind", "category", "origin", "status",
    "n_units", "in_Nsel", "in_Ncell", "in_Ndiag", "in_Nplatform", "in_Nsens", "in_Naudit",
    "n_trades", "cagr", "mdd", "sharpe", "vol", "calmar", "metric_basis", "window_mix",
    "artifact", "artifact_extra", "grade", "fail_gate", "notes", "prereg_role",
    "prev_sha256", "row_sha256",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cid12(s: str) -> str:
    return sha256_text(str(s))[:12]


def fnum(x, nd=8):
    if x is None or x == "":
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.4f}"


def mdd_abs(x):
    try:
        v = abs(float(x))
    except Exception:
        return np.nan
    return v if np.isfinite(v) else np.nan


def window_of_date(d) -> str:
    t = pd.Timestamp(d)
    if t <= DEV_END:
        return "dev"
    if VAL_START <= t <= VAL_END:
        return "val"
    return "post2020"


def read_csv(path, **kw):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", **kw)


def assert_no_2021(df: pd.DataFrame, name: str, col: str = "date") -> int:
    """返回 2021+ 行数；>0 直接报错。用于 2021+ 硬护栏。"""
    if df is None or len(df) == 0 or col not in df.columns:
        return 0
    s = pd.to_datetime(df[col], errors="coerce")
    bad = int((s > VAL_END).sum())
    if bad:
        raise AssertionError(f"{name}: 发现 {bad} 行日期晚于 2020-12-31，禁止计算")
    return bad


def stationary_bootstrap_summary(nav: pd.Series, B=B_BOOT, block=BLOCK, seed=SEED):
    """stationary bootstrap（几何块长，均值 block）的年化分布。"""
    nav = nav.dropna().astype(float)
    nav = nav[nav > 0]
    if len(nav) < 60:
        return None
    r = nav.pct_change().dropna().values
    T = len(r)
    rng = np.random.default_rng(seed)
    log1p = np.log1p(np.clip(r, -0.999999, None))
    cum = np.concatenate([[0.0], np.cumsum(log1p)])
    cum2 = np.concatenate([cum, cum[-1] + cum[1:]])
    means = np.empty(B)
    for b in range(B):
        idx = 0
        total = 0.0
        n = 0
        while idx < T:
            L = int(rng.geometric(1.0 / block))
            L = max(1, L)
            start = int(rng.integers(0, T))
            take = min(L, T - idx)
            # circular block sum via doubled cumsum
            total += cum2[start + take] - cum2[start]
            n += take
            idx += take
        means[b] = total / max(n, 1)
    cagr = np.expm1(243.0 * means)
    return {
        "cagr_point": float(np.expm1(243.0 * log1p.mean())),
        "cagr_p05": float(np.quantile(cagr, 0.05)),
        "cagr_p50": float(np.quantile(cagr, 0.50)),
        "cagr_p95": float(np.quantile(cagr, 0.95)),
        "p_cagr_le_0": float((cagr <= 0).mean()),
        "T": int(T),
        "B": int(B),
        "block": int(block),
    }


def pbo_cscv(R: pd.DataFrame, S=S_CSCV):
    """CSCV PBO。R: T x C 日收益矩阵（无 NaN）。"""
    R = R.dropna(axis=0, how="any")
    T, C = R.shape
    if C < 5 or T < S * 10:
        return None
    idx = np.array_split(np.arange(T), S)
    bsum = np.array([R.values[ix].sum(axis=0) for ix in idx])          # S x C
    bsq = np.array([(R.values[ix] ** 2).sum(axis=0) for ix in idx])    # S x C
    bn = np.array([len(ix) for ix in idx], dtype=float)                # S
    combos = np.array(list(_combinations(range(S), S // 2)), dtype=np.int16)
    is_mask = np.zeros((len(combos), S), dtype=bool)
    for i, c in enumerate(combos):
        is_mask[i, list(c)] = True
    is_n = is_mask @ bn
    oos_n = bn.sum() - is_n
    is_sum = is_mask @ bsum
    is_sq = is_mask @ bsq
    oos_sum = bsum.sum(axis=0)[None, :] - is_sum
    oos_sq = bsq.sum(axis=0)[None, :] - is_sq
    is_mean = is_sum / is_n[:, None]
    is_var = np.maximum(is_sq / is_n[:, None] - is_mean ** 2, 1e-18)
    is_sr = is_mean / np.sqrt(is_var)
    oos_mean = oos_sum / oos_n[:, None]
    oos_var = np.maximum(oos_sq / oos_n[:, None] - oos_mean ** 2, 1e-18)
    oos_sr = oos_mean / np.sqrt(oos_var)
    pick = is_sr.argmax(axis=1)
    picked = oos_sr[np.arange(len(pick)), pick]
    # rank: 1 = worst, C = best
    ranks = (oos_sr < picked[:, None]).sum(axis=1) + 1
    omega = ranks / (C + 1.0)
    pbo = float((omega <= 0.5).mean())
    return {
        "pbo": pbo,
        "n_splits": int(len(combos)),
        "n_candidates": int(C),
        "T": int(T),
        "median_oos_rank": float(np.median(omega)),
        "logit_median": float(np.median(np.log(omega / (1 - omega)))),
    }


def _combinations(iterable, r):
    from itertools import combinations
    return combinations(iterable, r)


def maxstat_bootstrap(R: pd.DataFrame, B=B_MAXSTAT, block=BLOCK, seed=SEED):
    """White Reality Check：max_c mean(r_c) 的 stationary bootstrap p 值。"""
    R = R.dropna(axis=0, how="any")
    T, C = R.shape
    if C < 2 or T < 120:
        return None
    vals = R.values
    obs = vals.mean(axis=0).max()
    # White Reality Check 的零假设：每个候选均值=0。重采样必须去均值，
    # 否则 bootstrap 分布以观测最大值为中心，p 恒在 0.5 附近（无效检验）。
    vals0 = vals - vals.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed + 7)
    cum = np.vstack([np.zeros((1, C)), np.cumsum(vals0, axis=0)])
    cum2 = np.vstack([cum, cum[-1] + cum[1:]])
    maxima = np.empty(B)
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
        maxima[b] = (tot / max(n, 1)).max()
    return {
        "stat": float(obs),
        "p_maxstat": float(((maxima >= obs).sum() + 1) / (B + 1)),
        "B": int(B),
        "T": int(T),
        "n_candidates": int(C),
    }


def dsr(sr_daily, T, var_sr, N, skew, kurt):
    if N < 2 or T < 30:
        return None
    gamma = 0.5772156649015329
    z1 = stats.norm.ppf(1 - 1.0 / N)
    z2 = stats.norm.ppf(1 - 1.0 / (N * math.e))
    sr0 = math.sqrt(max(var_sr, 0.0)) * ((1 - gamma) * z1 + gamma * z2)
    denom = math.sqrt(max(1e-12, 1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2))
    z = (sr_daily - sr0) * math.sqrt(max(T - 1, 1)) / denom
    return {
        "dsr": float(stats.norm.cdf(z)),
        "sr_daily": float(sr_daily),
        "sr_annual": float(sr_daily * math.sqrt(243)),
        "sr0_daily": float(sr0),
        "sr0_annual": float(sr0 * math.sqrt(243)),
        "N": int(N),
        "T": int(T),
        "z": float(z),
    }


# ---------------------------------------------------------------------------
# ledger construction
# ---------------------------------------------------------------------------

def base_row(**kw):
    r = {c: "" for c in FINAL_COLS}
    r.update({
        "phase": "", "source": "artifact", "kind": "experiment", "category": "prereg",
        "origin": "post_hoc_unregistered", "status": "done", "n_units": "1",
        "in_Nsel": "0", "in_Ncell": "0", "in_Ndiag": "0", "in_Nplatform": "0",
        "in_Nsens": "0", "in_Naudit": "0", "metric_basis": "reported_table",
        "window_mix": "single", "prereg_role": "",
    })
    r.update({k: ("" if v is None else str(v)) for k, v in kw.items()})
    return r


def window_dates(stage: str):
    if stage == "dev":
        return "2008-01-01", "2015-12-31"
    if stage == "val":
        return "2016-01-01", "2020-12-31"
    if stage == "full":
        return "2008-01-01", "2020-12-31"
    return "", ""


def load_stage1() -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(open(STAGE1_LEDGER, encoding="utf-8")))
    out = []
    for r in rows:
        kind = r["kind"]
        origin = r["origin"]
        if kind == "meta":
            cat = "meta"
        elif kind == "audit":
            cat = "audit"
        elif kind in ("diagnostic", "diagnostic_trade", "diagnostic_state"):
            cat = "diag"
        elif kind == "ablation":
            cat = "diag"
        elif kind == "sensitivity":
            cat = "sensitivity"
        elif kind == "experiment":
            cat = "prereg" if origin.startswith("prereg") else "posthoc"
        else:
            cat = "diag"
        if cat in ("diag", "audit", "meta"):
            in_sel = "0"
        else:
            in_sel = r["in_Nsel"]
        rr = base_row(
            phase="1", source="stage1_snapshot", family=r["family"],
            strategy_id=r["strategy_id"], params_id=r["params_id"], config_id=r["config_id"],
            window_start=r["window_start"], window_end=r["window_end"], stage=r["stage"],
            kind=kind, category=cat, origin=origin, status=r["status"], n_units=r["n_units"],
            in_Nsel=in_sel, in_Ncell=r["in_Ncell"], in_Ndiag=r["in_Ndiag"],
            in_Nsens=("1" if (kind == "sensitivity" and in_sel == "1") else "0"),
            n_trades=r["n_trades"], cagr=r["cagr"], mdd=r["mdd"], sharpe=r["sharpe"],
            vol=r["vol"], calmar=r["calmar"], metric_basis=r["metric_basis"],
            window_mix=r["window_mix"], artifact=r["artifact"], artifact_extra=r["artifact_extra"],
            grade=r["grade"], fail_gate=r["fail_gate"], notes=r["notes"][:900],
            prereg_role=("main" if r["family"] == "TREND" and r["strategy_id"].startswith("TREND@S4") else ""),
        )
        out.append(rr)
    meta = {
        "official_registry_head": rows[0]["prev_sha256"],
        "stage1_rows": len(rows),
        "stage1_tail": rows[-1]["row_sha256"],
        "stage1_Nsel": sum(float(x["in_Nsel"]) for x in out),
        "stage1_Ncell": sum(float(x["in_Ncell"]) for x in out),
        "stage1_Ndiag": sum(float(x["in_Ndiag"]) for x in out),
    }
    return out, meta


# ---- new family builders -------------------------------------------------

D_FAMILY = {
    "RISKBUDGET": ("D_pass_selfcheck", "factor/pct_chg；10_叠加结果 只读，未在 11 组复算名单但族内自检"),
    "TRENDCALLBACK": ("D_pass", "tr_t=tr_{t-1}(1+pct_chg/100)，自检 5.6e-16；复权审计 11 组复算 0.000pp"),
    "VCB": ("D_pass", "factor 总收益；复权审计 11 组复算 0.000pp（26,368 笔）"),
    "RELSTR": ("D_pass_with_caveat", "exit_cache=bs_hfq；复算 A 空间 1.956pp、H 5.563pp（A 好 3.4×），含分批离场"),
    "EXITUNI": ("D_pass", "scale=factor/close；复算报价比价 0.150pp"),
    "EXECPRESSURE": ("D_pass", "factor 总收益；复算 0.282pp（6,225 笔）"),
    "SMALLLOWVOL": ("D_pass", "统一数据适配 A 空间；复算 0.000pp（5,242 笔）"),
    "BREAKPULL": ("D_pass_selfcheck", "A 空间 adjusted_close≡factor；族内自检；不在 11 组复算名单"),
    "STRUCTREV": ("D_pass_selfcheck", "A 空间恒等式 1.8e-15、日收益 vs pct_chg 3.1e-15；不在 11 组复算名单"),
    "LONGMOM": ("D_pass_selfcheck", "A 空间 adj_close=factor；族内自检；不在 11 组复算名单"),
    "UNIFIEDENGINE": ("D_pass", "ca_adj=raw/ca_factor、tr_index=factor；复算 0.478pp"),
}


def make_config_rows(family, cid, metrics, category, kind, artifact, notes="", prereg_role="grid",
                     n_symbols=None, window_mix="single", extra=None, params_id=None, sens=False):
    """metrics: {'dev': {...}, 'val': {...}, 'full': {...}}; 生成 dev/val 行。"""
    rows = []
    cfg_id = f"{family}:{cid}"
    pid = params_id or cid12(cid)
    stages = [w for w in ("dev", "val", "full") if w in metrics and metrics[w] is not None]
    if not stages:
        stages = ["na"]
        metrics = {"na": {}}
    first = True
    for st in stages:
        m = metrics.get(st) or {}
        ws, we = window_dates(st)
        if st == "na" and extra:
            ws, we = extra.get("window_start", ""), extra.get("window_end", "")
        if st == "full":
            wmix = "pooled_dev_val" if window_mix != "single" else "pooled_dev_val"
        else:
            wmix = "single" if window_mix == "single" else window_mix
        r = base_row(
            phase="2/3", source="stage2_3_ledger", family=family, strategy_id=cid,
            params_id=pid, config_id=cfg_id, window_start=ws, window_end=we, stage=st,
            kind=kind, category=category, origin="family_prereg" if category in ("prereg", "sensitivity") else "post_hoc_unregistered",
            status="done", in_Nsel=("1" if (first and category in ("prereg", "posthoc", "sensitivity")) else "0"),
            in_Ncell=("1" if (st in ("dev", "val", "full") and category in ("prereg", "posthoc", "sensitivity")) else "0"),
            in_Nsens=("1" if (first and (category == "sensitivity" or sens)) else "0"),
            in_Naudit=("1" if (first and category in ("audit", "control")) else "0"),
            in_Ndiag=("1" if (first and category == "diag") else "0"),
            n_trades=m.get("n_trades", ""), cagr=fnum(m.get("cagr")), mdd=fnum(m.get("mdd")),
            sharpe=fnum(m.get("sharpe")), vol=fnum(m.get("vol")), calmar=fnum(m.get("calmar")),
            metric_basis="daily_nav" if m.get("daily_nav") else "reported_table",
            window_mix=wmix, artifact=artifact, notes=notes, prereg_role=prereg_role if first else "",
            n_units="1",
        )
        if n_symbols:
            r["artifact_extra"] = f"n_symbols={n_symbols}"
        rows.append(r)
        first = False
    return rows


def df_to_metrics(df, cid_col, win_col, cagr="cagr", mdd="mdd", sharpe="sharpe", vol="vol",
                  calmar="calmar", trades=None, symbols=None):
    out = {}
    for _, r in df.iterrows():
        w = str(r[win_col])
        if w not in ("dev", "val", "full"):
            continue
        out[w] = {
            "cagr": r.get(cagr, np.nan), "mdd": r.get(mdd, np.nan),
            "sharpe": r.get(sharpe, np.nan) if sharpe else np.nan,
            "vol": r.get(vol, np.nan) if vol else np.nan,
            "calmar": r.get(calmar, np.nan) if calmar else np.nan,
            "n_trades": r.get(trades, np.nan) if trades else np.nan,
            "n_symbols": r.get(symbols, np.nan) if symbols else np.nan,
        }
    return out


def build_riskbudget():
    cfg = read_csv(p("03_结果", "候选_风险预算", "12_试验清单.csv"))
    met = read_csv(p("03_结果", "候选_风险预算", "10_叠加结果.csv"))
    met = met[met["config_id"].isin(cfg["config_id"])]
    rows = []
    for _, c in cfg.iterrows():
        sub = met[met["config_id"] == c["config_id"]]
        metrics = df_to_metrics(sub, "config_id", "window", trades=None, symbols=None)
        cat = "control" if c["variant"] != "K1" else "prereg"
        kind = "experiment" if cat == "prereg" else "audit"
        sens = cat == "prereg" and (float(c["delay"]) > 0 or float(c["cost"]) != 10.0)
        rows += make_config_rows(
            "RISKBUDGET", f"{c['config_id']}|{c['variant']}|{c['carrier']}|m{c['m']}|d{c['delay']}|c{c['cost']}",
            metrics, cat, kind, "03_结果/候选_风险预算/10_叠加结果.csv",
            notes=f"variant={c['variant']}; carrier={c['carrier']}; m={c['m']}; delay={c['delay']}; cost={c['cost']}bp",
            prereg_role=("main" if (c["config_id"] == "RB002" and c["variant"] == "K1" and c["carrier"] == "C1"
                                      and float(c["m"]) == 0.5 and float(c["delay"]) == 0 and float(c["cost"]) == 10.0) else "grid"),
            sens=sens,
        )
    return rows


def build_trendcallback():
    led = read_csv(p("03_结果", "候选_趋势回调", "试验台账.csv"))
    met = read_csv(p("03_结果", "候选_趋势回调", "05_组合_汇总.csv"))
    rows = []
    for _, c in led.iterrows():
        sub = met[met["cid"] == c["cid"]]
        metrics = df_to_metrics(sub, "cid", "window", trades="n_closed", symbols="n_symbols")
        cid = c["cid"]
        if cid.startswith("NS_") or cid == "SF_L5":
            cat, role = "posthoc", "none"
        elif cid in ("L0_GENERIC", "L1_WEEKLY", "L2_DAILY", "L3_PULLBACK", "L4_VOL",
                     "L5_SUPPORT", "L6_BREAKOUT", "L7_DIRECT", "L8_NONTREND", "L9_BROKEN"):
            cat, role = "prereg", ("main" if cid == "L5_SUPPORT" else "grid")
        else:
            cat, role = "sensitivity", "none"
        rows += make_config_rows(
            "TRENDCALLBACK", cid, metrics, cat, "experiment" if cat == "prereg" else ("ablation" if cat == "posthoc" else "sensitivity"),
            "03_结果/候选_趋势回调/05_组合_汇总.csv",
            notes=f"policy={c['policy']}; exit={c['exit_days']}; delay={c['delay']}; cost={c['cost_mult']}; slots={c['slots']}; cap={c['capital']}",
            prereg_role=role, sens=(cat == "sensitivity"),
        )
    return rows


def build_vcb():
    led = read_csv(p("03_结果", "候选_收缩突破", "试验清单.csv"))
    summ = read_csv(p("03_结果", "候选_收缩突破", "组合_汇总_全部.csv"))
    summ["win"] = np.where(pd.to_datetime(summ["start"]).dt.year < 2016, "dev", "val")
    summ["cid"] = summ["config"] + "_" + summ["exit"].astype(str)
    rob = read_csv(p("03_结果", "候选_收缩突破", "稳健性_资金持仓成本.csv"))
    rows = []
    # account-level experiment/sensitivity configs
    for cid, sub in summ.groupby("cid"):
        metrics = df_to_metrics(sub, "cid", "win", trades="n_trades", symbols=None)
        metrics["daily_nav"] = True
        is_main = cid == "VCB-M_T40"
        rows += make_config_rows(
            "VCB", cid, metrics, "prereg", "experiment",
            "03_结果/候选_收缩突破/组合_汇总_全部.csv",
            notes=f"config={cid}；预注册 VCB-M 主候选={'是' if is_main else '否（网格/邻域）'}；主候选 T40 开发 MDD -36.3% 越 25% 门槛",
            prereg_role="main" if is_main else "grid",
        )
    # sensitivity variants (base duplicates VCB-M_T40 -> skip)
    for v, sub in rob.groupby("variant"):
        if v == "base":
            continue
        metrics = {}
        for _, r in sub.iterrows():
            metrics[str(r["period"])] = {
                "cagr": r["cagr"], "mdd": r["mdd"], "sharpe": r["sharpe"], "vol": r["vol"],
                "calmar": r["calmar"], "n_trades": r["n_trades"],
            }
        rows += make_config_rows(
            "VCB", f"VCB-M_T40|{v}", metrics, "sensitivity", "sensitivity",
            "03_结果/候选_收缩突破/稳健性_资金持仓成本.csv",
            notes=f"敏感性 {v}（同一主候选的摩擦/资金/仓位替换）",
            prereg_role="none", sens=True,
        )
    # feasibility + ablation -> diag aggregates
    feas = led[led["kind"] == "feasibility"]
    abl = led[led["kind"] == "ablation"]
    rows.append(base_row(
        phase="2/3", source="stage2_3_ledger", family="VCB", strategy_id="VCB-FEASIBILITY",
        params_id=cid12("VCB-feasibility"), config_id="VCB:feasibility", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="feasibility", category="diag",
        origin="family_prereg", status="done", n_units=str(len(feas)), in_Ndiag=str(len(feas)),
        artifact="03_结果/候选_收缩突破/试验清单.csv",
        notes="L0-L3 可行性计数（无账户收益）→ N_diag，不计 N_sel",
        prereg_role="none", metric_basis="none", window_mix="pooled_dev_val",
    ))
    rows.append(base_row(
        phase="2/3", source="stage2_3_ledger", family="VCB", strategy_id="VCB-ABLATION",
        params_id=cid12("VCB-ablation"), config_id="VCB:ablation", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="ablation", category="diag",
        origin="family_prereg", status="done", n_units=str(len(abl)), in_Ndiag=str(len(abl)),
        artifact="03_结果/候选_收缩突破/试验清单.csv",
        notes="6 个事件层消融（去平台/放量/延伸/加成交额收缩等），无账户级结果 → N_diag",
        prereg_role="none", metric_basis="event_mean", window_mix="pooled_dev_val",
    ))
    return rows


def build_relstr():
    met = read_csv(p("03_结果", "候选_相对强弱", "03_组合_汇总.csv"))
    rows = []
    for cfg, sub in met.groupby("config"):
        metrics = df_to_metrics(sub, "config", "window", trades="n_trades", symbols="n_codes")
        metrics["daily_nav"] = True
        if cfg in ("B1", "B2"):
            cat, role, kind = "audit", "none", "audit"
        else:
            cat, role, kind = "prereg", ("main" if cfg in ("C01", "C07") else "grid"), "experiment"
        rows += make_config_rows(
            "RELSTR", cfg, metrics, cat, kind, "03_结果/候选_相对强弱/03_组合_汇总.csv",
            notes=f"config={cfg}; 40 个策略配置（另 B1/B2 为基准）；预注册 v1.1 40 配置",
            prereg_role=role,
        )
    return rows


def build_exituni():
    met = read_csv(p("03_结果", "候选_退出统一", "08_组合复核.csv"))
    rows = []
    for _, r in met.iterrows():
        cid = f"{r['entry']}|{r['arm']}"
        metrics = {"full": {"cagr": r["cagr"], "mdd": r["mdd"], "sharpe": r["sharpe"],
                            "vol": r["vol"], "calmar": r["calmar"], "n_trades": r["n_trades"]}}
        rows += make_config_rows(
            "EXITUNI", cid, metrics, "prereg", "experiment",
            "03_结果/候选_退出统一/08_组合复核.csv",
            notes=f"entry={r['entry']}; arm={r['arm']}；只有 2008-2020 合并窗口，无法两窗口评级",
            prereg_role=("main" if r["arm"] == "ATR2" else "grid"), window_mix="mixed_full",
        )
    return rows


def build_execpressure():
    cfg = read_csv(p("03_结果", "执行压力", "02_配置汇总.csv"))
    rows = []
    base = cfg[
        (cfg["capital"] == 1_000_000) & (cfg["slots"] == 10) & (cfg["slippage_bps"] == 10) &
        (cfg["participation"] == 0.02) & (cfg["min_commission"] == 5.0) &
        (cfg["tax_mode"] == "historical") & (cfg["transfer_mode"] == "module") &
        (cfg["impact"] == False) & (cfg["delay"] == 0) & (cfg["limit_up"] == "cancel") &
        (cfg["limit_down"] == "persist") & (cfg["max_sell_defer"] == 10) & (cfg["ordering"] == "random")
    ]
    for cand in ("B55H20", "B55H60"):
        sub = base[base["candidate"] == cand]
        metrics = {}
        for _, r in sub.iterrows():
            metrics[str(r["window"])] = {"cagr": r["cagr_med"], "mdd": r["mdd_med"],
                                         "n_trades": r["trades_med"]}
        rows += make_config_rows(
            "EXECPRESSURE", cand, metrics, "prereg", "experiment",
            "03_结果/执行压力/02_配置汇总.csv",
            notes=f"{cand} 规则级重建（1M/10 仓/10bp/2% ADV，5 种子中位）；预注册 §5.7 规则级 N=4",
            prereg_role="main",
        )
    # S4/S1 fixed trade set from 06
    rep = read_csv(p("03_结果", "执行压力", "06_候选_交易级重算.csv"))
    rep = rep[(rep["capital"] == 1_000_000) & (rep["slots"] == 10) & (rep["slippage_bps"] == 10)]
    for cand in ("S4时间60", "S1时间60"):
        sub = rep[rep["candidate"] == cand]
        metrics = {}
        for _, r in sub.iterrows():
            metrics[str(r["window"])] = {"cagr": r["cagr"], "mdd": r["mdd"], "n_trades": r["n_trades"]}
        rows += make_config_rows(
            "EXECPRESSURE", cand, metrics, "prereg", "experiment",
            "03_结果/执行压力/06_候选_交易级重算.csv",
            notes=f"{cand} 固定交易集合重定价（非原账户路径）；1M/10 仓/10bp",
            prereg_role="grid",
        )
    # platform: actual distinct friction scenarios
    keycols = [c for c in ("candidate", "capital", "slots", "slippage_bps", "participation",
                           "min_commission", "tax_mode", "transfer_mode", "impact", "delay",
                           "limit_up", "limit_down", "max_sell_defer", "ordering") if c in cfg.columns]
    n_scen = int(cfg[keycols].drop_duplicates().shape[0])
    rows.append(base_row(
        phase="2/3", source="stage2_3_ledger", family="EXECPRESSURE", strategy_id="EP-FRICTION",
        params_id=cid12("EP-friction"), config_id="EXECPRESSURE:friction", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="platform", category="platform",
        origin="family_prereg", status="done", n_units=str(n_scen), in_Nplatform=str(n_scen),
        artifact="03_结果/执行压力/02_配置汇总.csv",
        notes=f"摩擦情景 {n_scen} 个（预注册估计约 86 个；运行 1,250 矩阵格）。平台格不是独立发现，不进 N_sel；DSR 阶梯 N2/N3 记录",
        prereg_role="none", metric_basis="none", window_mix="pooled_dev_val",
    ))
    return rows


def build_smalllowvol():
    grid = read_csv(p("03_结果", "候选_小盘低波趋势", "10_组合_网格.csv"))
    stress = read_csv(p("03_结果", "候选_小盘低波趋势", "14_压力_邻域_容量.csv"))
    rows = []
    for (v, sl, h), sub in grid.groupby(["variant", "slots", "hold"]):
        metrics = {}
        for _, r in sub.iterrows():
            metrics[str(r["window"])] = {"cagr": r["cagr"], "mdd": r["mdd"], "sharpe": r["sharpe"],
                                          "vol": r["vol"], "calmar": r["calmar"],
                                          "n_trades": r["n_closed"], "n_symbols": r["n_symbols"]}
        metrics["daily_nav"] = True
        is_main = (v == "COMBO" and sl == 10 and h == 40)
        rows += make_config_rows(
            "SMALLLOWVOL", f"{v}_s{sl}_h{h}", metrics, "prereg", "experiment",
            "03_结果/候选_小盘低波趋势/10_组合_网格.csv",
            notes=f"variant={v}; slots={sl}; hold={h}；预注册主候选 COMBO 10/40；63 主网格",
            prereg_role="main" if is_main else "grid",
        )
    # stress / neighborhood / capacity / rand（按 14_压力 文件的实际 group）
    for grp, cat, kind, n_units, note in (
        ("stress", "sensitivity", "sensitivity", 42, "7 变体 × 6 情景（退市/成本×2/延迟/顺延/ADV 上限），full 窗口；预注册表记 35、实际 42（+7 披露）"),
        ("neighborhood", "sensitivity", "sensitivity", 4, "size_q×vol_q 3 组 × 2 窗口 = 6 格；按 4 个唯一设置计 N_sens 子集"),
        ("capacity", "platform", "platform", 12, "2-3 变体 × 资金档（10 万/1000 万/1 亿），full；容量平台格"),
        ("rand_seed", "audit", "audit", 11, "RAND 种子 1-4 及 dev/val 切片；零假设分布，审计夹具"),
    ):
        sub = stress[stress["group"] == grp]
        if len(sub) == 0:
            continue
        add_sel = n_units if cat == "sensitivity" else 0
        add_cell = len(sub)
        rows.append(base_row(
            phase="3", source="stage2_3_ledger", family="SMALLLOWVOL", strategy_id=f"SC-{grp}",
            params_id=cid12(f"SC-{grp}"), config_id=f"SMALLLOWVOL:{grp}", stage="full",
            window_start="2008-01-01", window_end="2020-12-31", kind=kind, category=cat,
            origin="family_prereg", status="done", n_units=str(n_units),
            in_Nsel=str(add_sel), in_Ncell=str(add_cell), in_Nsens=str(add_sel),
            in_Nplatform=str(n_units) if cat == "platform" else "0",
            in_Naudit=str(n_units) if cat == "audit" else "0",
            in_Ndiag="0", artifact="03_结果/候选_小盘低波趋势/14_压力_邻域_容量.csv",
            notes=f"SC-{grp}: {len(sub)} 行；{note}；平台/审计不进 N_sel",
            prereg_role="none", metric_basis="none", window_mix="pooled_dev_val",
        ))
    return rows


def build_breakpull():
    met = read_csv(p("03_结果", "候选_突破回踩", "05_组合_汇总.csv"))
    rows = []
    for label, sub in met.groupby("label"):
        metrics = df_to_metrics(sub, "label", "window", trades="n_closed", symbols=None)
        metrics["daily_nav"] = True
        if label in ("主-未回踩", "主-假突破"):
            cat, role, kind = "diag", "none", "diagnostic"
        else:
            cat = "sensitivity" if label.startswith("稳健") else "prereg"
            role = "main" if label == "主-回踩直接买" else ("grid" if label.startswith("邻域") else "none")
            kind = "experiment" if cat == "prereg" else "sensitivity"
        rows += make_config_rows(
            "BREAKPULL", label, metrics, cat, kind,
            "03_结果/候选_突破回踩/05_组合_汇总.csv",
            notes=(f"{label}；主配置 N=55/W=10/tol=0.5/FIX20"
                   if cat != "diag" else
                   f"{label}：按'是否回踩/是否假突破'事后切分的路径条件臂，含未来信息，只能诊断，不可部署"),
            prereg_role=role, sens=(cat == "sensitivity"),
        )
    rows.append(base_row(
        phase="3", source="stage2_3_ledger", family="BREAKPULL", strategy_id="BR-PLATFORM",
        params_id=cid12("BR-platform"), config_id="BREAKPULL:platform", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="platform", category="platform",
        origin="family_prereg", status="done", n_units="144", in_Nplatform="144",
        artifact="03_结果/候选_突破回踩/11_试验台账.csv",
        notes="E2 事件参数平台 144 格（N×W×tol×target×mode）；[BUDGET]，平台描述性，不用于选参",
        prereg_role="none", metric_basis="event_mean", window_mix="pooled_dev_val",
    ))
    rows.append(base_row(
        phase="3", source="stage2_3_ledger", family="BREAKPULL", strategy_id="BR-UNIFIED",
        params_id=cid12("BR-unified"), config_id="BREAKPULL:unified", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="audit", category="audit",
        origin="family_prereg", status="done", n_units="6", in_Naudit="6",
        artifact="03_结果/候选_突破回踩/11_试验台账.csv",
        notes="统一引擎复核 6 格（3 臂×2 窗口），审计夹具",
        prereg_role="none", metric_basis="none", window_mix="single",
    ))
    return rows


def build_structrev():
    met = read_csv(p("03_结果", "候选_结构内反转", "07_账户_汇总.csv"))
    rows = []
    for cid, sub in met.groupby("cid"):
        metrics = df_to_metrics(sub, "cid", "window", trades="n_trades", symbols="n_codes")
        metrics["daily_nav"] = True
        if cid == "A31_HASH":
            cat, role, kind = "audit", "none", "audit"
        elif cid in ("A28_TROUGH_DEEP", "A29_TROUGH_SHALLOW", "A30_Q0010"):
            cat, role, kind = "posthoc", "none", "experiment"
        else:
            cat, role, kind = "prereg", ("main" if cid == "A01_MAIN" else "grid"), "experiment"
        rows += make_config_rows(
            "STRUCTREV", cid, metrics, cat, kind,
            "03_结果/候选_结构内反转/07_账户_汇总.csv",
            notes=f"{cid}；预注册 v1.2；A28/A29/A30 为看到 A01 与事件层后追加（v1.1 变更记录）",
            prereg_role=role,
        )
    rows.append(base_row(
        phase="3", source="stage2_3_ledger", family="STRUCTREV", strategy_id="SR-E12",
        params_id=cid12("SR-E12"), config_id="STRUCTREV:E12", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="diag", category="diag",
        origin="family_prereg", status="done", n_units="1", in_Ndiag="1",
        artifact="03_结果/候选_结构内反转/00_预注册_结构内反转.md",
        notes="E12 事件层独有配置（无账户运行）→ N_diag",
        prereg_role="none", metric_basis="event_mean", window_mix="pooled_dev_val",
    ))
    return rows


def build_longmom():
    met = read_csv(p("03_结果", "候选_长期双动量", "02_组合_汇总.csv"))
    rows = []
    exp = met[(met["kind"] == "experiment") & (met["window"].isin(["dev", "val"]))]
    for cfg, sub in exp.groupby("cfg"):
        metrics = df_to_metrics(sub, "cfg", "window", trades="n_trades", symbols="n_symbols")
        metrics["daily_nav"] = True
        rows += make_config_rows(
            "LONGMOM", cfg, metrics, "prereg", "experiment",
            "03_结果/候选_长期双动量/02_组合_汇总.csv",
            notes=f"{cfg}；26 个选择配置；主候选 C12",
            prereg_role="main" if cfg == "C12" else "grid",
        )
    n_audit = int(met.loc[met["kind"] == "audit", "cfg"].nunique())
    rows.append(base_row(
        phase="3", source="stage2_3_ledger", family="LONGMOM", strategy_id="LDM-AUDIT",
        params_id=cid12("LDM-audit"), config_id="LONGMOM:audit", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="audit", category="audit",
        origin="family_prereg", status="done", n_units=str(n_audit), in_Naudit=str(n_audit),
        artifact="03_结果/候选_长期双动量/02_组合_汇总.csv",
        notes="随机 N=10 对照（R20/R60 各 8 种子）与基准等审计运行；不进 N_sel",
        prereg_role="none", metric_basis="none", window_mix="pooled_dev_val",
    ))
    return rows


def build_unifiedengine():
    j = json.load(open(p("03_结果", "统一引擎", "汇总.json"), encoding="utf-8"))
    rows = []
    # tag -> (trial_id, posthoc?)
    tag_map = {"FIX20_dev": ("T-U01", False), "FIX20_val": ("T-U01", False),
               "ATR3_dev": ("T-U02", False), "ATR3_val": ("T-U02", False),
               "FIX20_dev_adv": ("T-U03", True), "FIX20_val_adv": ("T-U03", True),
               "ATR3_dev_adv": ("T-U04", True), "ATR3_val_adv": ("T-U04", True)}
    metrics_by = defaultdict(dict)
    for run in j["runs"]:
        tag = run["tag"]
        if tag not in tag_map:
            continue
        tid, _ = tag_map[tag]
        win = "dev" if tag.endswith("_dev") or tag.endswith("_dev_adv") else "val"
        m = run["metrics"]
        metrics_by[tid][win] = {"cagr": m["cagr"], "mdd": m["mdd"], "sharpe": m.get("sharpe"),
                                "vol": m.get("vol"), "n_trades": m.get("n_trades")}
    for tid, metrics in metrics_by.items():
        posthoc = tag_map[[t for t, v in tag_map.items() if v[0] == tid][0]][1]
        rows += make_config_rows(
            "UNIFIEDENGINE", tid, metrics, "posthoc" if posthoc else "prereg",
            "experiment", "03_结果/统一引擎/汇总.json",
            notes=f"{tid}；T-U01/T-U02 公开基线，T-U03/T-U04 为看到 T-U01 后的排名键消融（post-hoc）；两段全负",
            prereg_role="none" if posthoc else "main",
        )
    rows.append(base_row(
        phase="2", source="stage2_3_ledger", family="UNIFIEDENGINE", strategy_id="UE-POSTHOC",
        params_id=cid12("UE-posthoc"), config_id="UNIFIEDENGINE:posthoc", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="ablation", category="posthoc",
        origin="post_hoc_unregistered", status="done", n_units="2", in_Nsel="0", in_Ncell="0",
        artifact="03_结果/统一引擎/结果摘要.md",
        notes="T-U03/T-U04 排名键消融，post-hoc；统一引擎 4 个配置全部两段为负",
        prereg_role="none", metric_basis="none", window_mix="pooled_dev_val",
    ))
    return rows


def build_all_new():
    rows = []
    for fn in (build_riskbudget, build_trendcallback, build_vcb, build_relstr, build_exituni,
               build_execpressure, build_smalllowvol, build_breakpull, build_structrev,
               build_longmom, build_unifiedengine):
        n0 = len(rows)
        rows += fn()
        print(f"  {fn.__name__}: +{len(rows)-n0} rows")
    return rows


def build_corrections():
    """已发现的第一阶段计数修正：ECOLOGY 事件条件格 36（总账只记 1 行 diag）。"""
    return [base_row(
        phase="1", source="correction", family="ECOLOGY", strategy_id="ECOLOGY-EVENT-CELLS",
        params_id=cid12("ECOLOGY-event-cells"), config_id="ECOLOGY:event-cells", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="diagnostic", category="diag",
        origin="preregistered_local", status="done", n_units="35", in_Ndiag="35",
        artifact="03_结果/统计整合/01_整合结论.md",
        notes="修正：整合结论 §5.4 记 ECOLOGY 事件条件格 36 行，总账仅 1 行 diag；补 35 使 N_diag 与披露一致（不缩小 N）",
        prereg_role="none", metric_basis="event_mean", window_mix="pooled_dev_val",
    ), base_row(
        phase="1", source="correction", family="MKT", strategy_id="MKT-STATE-TABLES",
        params_id=cid12("MKT-state-tables"), config_id="MKT:state-tables", stage="dev",
        window_start="2008-01-01", window_end="2020-12-31", kind="diagnostic", category="diag",
        origin="post_hoc_unregistered", status="done", n_units="7", in_Ndiag="7",
        artifact="03_结果/统计整合/01_整合结论.md",
        notes="修正：整合结论记 MKT N_diag=8，总账仅 1 行 diag；补 7 张状态层辅助表（不缩小 N）",
        prereg_role="none", metric_basis="diagnostic", window_mix="pooled_dev_val",
    )]


def build_audit_fixtures():
    return [base_row(
        phase="1/2/3", source="artifact", family="AUDITFIX", strategy_id="PROJECT-AUDIT-FIXTURES",
        params_id=cid12("project-audit-fixtures"), config_id="AUDITFIX:all", stage="audit",
        window_start="2005-01-01", window_end="2020-12-31", kind="audit", category="audit",
        origin="audit", status="done", n_units="4", in_Naudit="4", grade="AUDIT", fail_gate="",
        artifact="05_审计/独立复现_基线.md;05_审计/双源数据核验.md;05_审计/复权修复影响.md;04_证据/撮合自检.json",
        notes=("项目级审计夹具：独立复现基线、双源核验、复权修复影响（11 组复算）、撮合/交易制度自检。"
               "03_结果/交易制度审计/ 为空目录，无结果；均不进 N_sel。"),
        prereg_role="none", metric_basis="none", window_mix="na",
    )]


def build_main_candidate_placeholder():
    return [base_row(
        phase="2", source="official_prereg", family="MAIN", strategy_id="T0005",
        params_id=cid12("F1-S1-E2-X3-G1"), config_id="MAIN:T0005", stage="na",
        window_start="", window_end="", kind="experiment", category="prereg",
        origin="preregistered_local", status="planned", n_units="0", in_Nsel="0",
        in_Ncell="0", in_Ndiag="0", artifact="03_结果/实验注册表.csv",
        grade="F_unevaluable", fail_gate="F_never_run",
        notes=("官方预注册主候选 F1×S1×E2×X3×G1（T0005）从未运行；本总账登记为 F_unevaluable。"
               "更新命令：python3 02_代码/实验登记.py start --trial T0005 && "
               "python3 02_代码/统一回测.py --family F1-S1-E2-X3-G1 --stage dev --window 2008-01-01..2015-12-31"),
        prereg_role="main", metric_basis="none", window_mix="na",
    )]


# ---------------------------------------------------------------------------
# 净值序列（只读；统一断言 <= 2020-12-31）
# ---------------------------------------------------------------------------

NAV_AUDIT = []


def _add_series(store, family, cfg, window, dates, nav, artifact):
    s = pd.Series(np.asarray(nav, dtype=float), index=pd.to_datetime(dates)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s > 0]
    if len(s) < 60:
        return
    mx = s.index.max()
    if mx > VAL_END:
        raise AssertionError(f"{artifact} {cfg} {window}: max date {mx} > 2020-12-31")
    NAV_AUDIT.append({"family": family, "config": cfg, "window": window, "rows": len(s),
                      "min_date": str(s.index.min().date()), "max_date": str(mx.date()),
                      "artifact": os.path.relpath(artifact, ROOT)})
    store[(family, str(cfg), window)] = s


def load_nav_series():
    store = {}
    # VCB: 14 configs x 2 windows
    import glob
    for f in sorted(glob.glob(p("03_结果", "候选_收缩突破", "组合_净值_VCB-*.csv"))):
        b = os.path.basename(f)
        m = b.replace("组合_净值_", "").replace(".csv", "")
        cfg, y0, y1 = m.rsplit("_", 2)
        win = "dev" if y0 == "2008" else "val"
        df = read_csv(f)
        _add_series(store, "VCB", cfg, win, df["date"], df["nav"], f)
    # TRENDCALLBACK
    f = p("03_结果", "候选_趋势回调", "03_组合_净值.csv")
    df = read_csv(f)
    for (cfg, win), sub in df.groupby(["config", "window"]):
        if win not in ("dev", "val"):
            continue
        _add_series(store, "TRENDCALLBACK", cfg, win, sub["date"], sub["nav"], f)
    # RELSTR
    f = p("03_结果", "候选_相对强弱", "03_净值.csv.gz")
    df = pd.read_csv(f)
    for (cfg, win), sub in df.groupby(["config", "window"]):
        if cfg in ("B1", "B2") or win not in ("dev", "val"):
            continue
        _add_series(store, "RELSTR", cfg, win, sub["date"], sub["nav"], f)
    # STRUCTREV
    f = p("03_结果", "候选_结构内反转", "08_账户_净值.parquet")
    df = pd.read_parquet(f)
    for (cfg, win), sub in df.groupby(["cid", "window"]):
        if win not in ("dev", "val"):
            continue
        _add_series(store, "STRUCTREV", cfg, win, sub["date"], sub["nav"], f)
    # LONGMOM: split by date; only experiment configs
    f = p("03_结果", "候选_长期双动量", "02_净值.csv.gz")
    df = pd.read_csv(f)
    summ = read_csv(p("03_结果", "候选_长期双动量", "02_组合_汇总.csv"))
    exp_cfgs = set(summ.loc[summ["kind"] == "experiment", "cfg"])
    df = df[df["cfg"].isin(exp_cfgs)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp("2008-01-01")]  # 2007-12 仅预热，不进 dev
    df["window"] = np.where(df["date"] <= DEV_END, "dev", "val")
    for (cfg, win), sub in df.groupby(["cfg", "window"]):
        _add_series(store, "LONGMOM", cfg, win, sub["date"], sub["nav"], f)
    # RISKBUDGET wide matrix
    f = p("03_结果", "候选_风险预算", "16_净值", "C1_关键净值.csv")
    df = pd.read_csv(f, index_col=0)
    df.index = pd.to_datetime(df.index)
    for cfg in df.columns:
        for win in ("dev", "val"):
            sub = df[cfg] if win == "dev" else df[cfg]
            sel = sub[(sub.index <= DEV_END)] if win == "dev" else sub[(sub.index >= VAL_START) & (sub.index <= VAL_END)]
            _add_series(store, "RISKBUDGET", cfg, win, sel.index, sel.values, f)
    # EXITUNI (full only)
    import glob
    for f in sorted(glob.glob(p("03_结果", "候选_退出统一", "08_组合净值", "*.csv"))):
        cfg = os.path.basename(f).replace(".csv", "")
        df = read_csv(f)
        _add_series(store, "EXITUNI", cfg, "full", df["date"], df["nav"], f)
    # BREAKPULL
    f = p("03_结果", "候选_突破回踩", "06_组合_净值.csv")
    df = read_csv(f)
    df["date"] = pd.to_datetime(df["date"])
    df["window"] = np.where(df["date"] <= DEV_END, "dev", "val")
    for (cfg, win), sub in df.groupby(["label", "window"]):
        if cfg in ("主-未回踩", "主-假突破"):
            continue  # 路径条件臂含未来信息，不进统计矩阵
        _add_series(store, "BREAKPULL", cfg, win, sub["date"], sub["nav"], f)
    # SMALLLOWVOL main unit (7 variants x 2 windows)
    f = p("03_结果", "候选_小盘低波趋势", "13_组合_净值_主单元.csv")
    df = read_csv(f)
    for (cfg, win), sub in df.groupby(["variant", "window"]):
        if win not in ("dev", "val"):
            continue
        _add_series(store, "SMALLLOWVOL", cfg, win, sub["date"], sub["nav"], f)
    # UNIFIEDENGINE
    for var, win in (("FIX20", "dev"), ("FIX20", "val"), ("ATR3", "dev"), ("ATR3", "val"),
                     ("FIX20_adv", "dev"), ("FIX20_adv", "val"), ("ATR3_adv", "dev"), ("ATR3_adv", "val")):
        f = p("03_结果", "统一引擎", f"净值_{var}_{win}.csv")
        if os.path.exists(f):
            df = read_csv(f)
            _add_series(store, "UNIFIEDENGINE", var, win, df["date"], df["nav"], f)
    return store


def build_matrices(store):
    groups = defaultdict(dict)
    for (fam, cfg, win), s in store.items():
        if win in ("dev", "val"):
            groups[(fam, win)][cfg] = s
    out = {}
    for key, d in groups.items():
        if len(d) < 5:
            continue
        df = pd.DataFrame(d).sort_index().dropna(how="any")
        if len(df) < 120:
            continue
        out[key] = df
    return out


def run_stats(store, quick=False):
    mats = build_matrices(store)
    pbo_rows, ms_rows = [], []
    B_ms = 500 if quick else B_MAXSTAT
    for (fam, win), df in sorted(mats.items()):
        R = df.pct_change().dropna()
        if len(R) < 120:
            continue
        pbo = pbo_cscv(R)
        ms = maxstat_bootstrap(R, B=B_ms)
        # 矩阵内最好配置的点估计与 5% 分位
        cagrs = {}
        for c in df.columns:
            sv = df[c].dropna()
            if len(sv) >= 60:
                cagrs[c] = float((sv.iloc[-1] / sv.iloc[0]) ** (243.0 / len(sv)) - 1)
        best_cfg = max(cagrs, key=cagrs.get) if cagrs else None
        best_p05 = None
        if best_cfg is not None:
            bb = stationary_bootstrap_summary(df[best_cfg], B=500, seed=SEED + 3)
            best_p05 = bb["cagr_p05"] if bb else None
        extra = {"best_config": best_cfg, "best_cagr": cagrs.get(best_cfg) if best_cfg else None,
                 "best_cagr_p05": best_p05}
        if pbo:
            pbo_rows.append({"family": fam, "window": win, **pbo, **extra})
        if ms:
            ms_rows.append({"family": fam, "window": win, **ms, **extra})
        print(f"  {fam}/{win}: C={R.shape[1]} T={R.shape[0]} PBO={pbo['pbo'] if pbo else None} p_max={ms['p_maxstat'] if ms else None}")
    # CIs for family mains + the only mechanical G-passer
    shortlist = [
        ("VCB", "VCB-P20_T40"), ("VCB", "VCB-M_T40"),
        ("SMALLLOWVOL", "COMBO"), ("TRENDCALLBACK", "L5_SUPPORT"),
        ("RELSTR", "C01"), ("RELSTR", "C07"), ("STRUCTREV", "A01_MAIN"),
        ("LONGMOM", "C12"), ("BREAKPULL", "主-回踩直接买"),
        ("RISKBUDGET", "RB009"), ("EXITUNI", "A_55突破|ALL"),
        ("UNIFIEDENGINE", "FIX20"),
    ]
    def _resolve(fam, cfg):
        if (fam, cfg, "dev") in store:
            return cfg
        if fam == "TRENDCALLBACK":
            for (f, c, w) in store:
                if f == fam and c.startswith("L5_SUPPORT|") and "smallfirst" not in c and "s10|k1000000" in c:
                    return c
        if fam == "RISKBUDGET":
            for c in ("RB004", "RB001", "RB002"):
                if (fam, c, "dev") in store:
                    return c
        return cfg

    ci_rows = []
    for fam, cfg in shortlist:
        cfg = _resolve(fam, cfg)
        for win in ("dev", "val", "full"):
            s = store.get((fam, cfg, win))
            if s is None:
                continue
            r = stationary_bootstrap_summary(s, B=1000 if quick else B_BOOT)
            if r:
                ci_rows.append({"family": fam, "config": cfg, "window": win, **r})
    return mats, pbo_rows, ms_rows, ci_rows


# ---------------------------------------------------------------------------
# 统一门槛
# ---------------------------------------------------------------------------

ACCOUNT_CATS = {"prereg", "posthoc", "sensitivity"}
D_GATE = {
    "STRUCT": "D_pass", "TREND": "D_pass", "EXIT": "D_pass", "MKT": "D_pass",
    "STAT": "D_na_fixture", "ECOLOGY": "D_pass",
    "RISKBUDGET": "D_pass_selfcheck", "TRENDCALLBACK": "D_pass", "VCB": "D_pass",
    "RELSTR": "D_pass_with_caveat", "EXITUNI": "D_pass", "EXECPRESSURE": "D_pass",
    "SMALLLOWVOL": "D_pass", "BREAKPULL": "D_pass_selfcheck", "STRUCTREV": "D_pass_selfcheck",
    "LONGMOM": "D_pass_selfcheck", "UNIFIEDENGINE": "D_pass",
}


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def compute_gates(rows, fam_stats):
    idx = {}
    for r in rows:
        if r["category"] not in ACCOUNT_CATS:
            continue
        if r["stage"] not in ("dev", "val", "full"):
            continue
        key = (r["family"], r["config_id"])
        d = idx.setdefault(key, {"family": r["family"], "config_id": r["config_id"],
                                 "strategy_id": r["strategy_id"], "category": r["category"],
                                 "role": r["prereg_role"], "dev": None, "val": None, "full": None,
                                 "artifact": r["artifact"], "rows": []})
        d["rows"].append(r)
        if r["stage"] in ("dev", "val", "full"):
            d[r["stage"]] = r
    out = []
    for key, d in idx.items():
        dev, val, full = d["dev"], d["val"], d["full"]
        g = {"family": d["family"], "config_id": d["config_id"], "strategy_id": d["strategy_id"],
             "category": d["category"], "role": d["role"], "artifact": d["artifact"]}
        for w, rr in (("dev", dev), ("val", val), ("full", full)):
            g[f"{w}_cagr"] = _f(rr["cagr"]) if rr else np.nan
            g[f"{w}_mdd"] = -abs(_f(rr["mdd"])) if rr else np.nan
            g[f"{w}_trades"] = _f(rr["n_trades"]) if rr else np.nan
        fails = []
        if dev is None or val is None:
            gates = {"W": "W_fail_mixed_window"}
            grade = "W_window_mixed" if full is not None else "F_unevaluable"
            fails.append("W_fail_mixed_window" if full is not None else "F_no_two_window")
        else:
            # G
            cg = (g["dev_cagr"] > 0) and (g["val_cagr"] > 0)
            mg = (abs(g["dev_mdd"]) <= 0.25) and (abs(g["val_mdd"]) <= 0.25)
            g_gate = "G_pass" if (cg and mg) else (
                "G_fail_negative" if not cg else "G_fail_dd")
            # P（风险预算是 B1 之上的月度 overlay，trade 数不是策略笔数；标 N/A 而非 fail）
            if d["family"] == "RISKBUDGET":
                nt = True
                p_gate = "P_na_overlay"
            else:
                nt = (g["dev_trades"] >= 30) and (g["val_trades"] >= 30)
                p_gate = "P_pass" if nt else ("P_fail_underpowered" if np.isfinite(g["dev_trades"]) else "P_unverified")
            # R
            if d["category"] == "posthoc":
                r_gate = "R_fail_posthoc"
            elif d["role"] == "main":
                r_gate = "R_pass_family_prereg"
            elif d["category"] == "sensitivity":
                r_gate = "R_fail_sensitivity"
            else:
                r_gate = "R_fail_grid_promotion"
            # D
            d_gate = D_GATE.get(d["family"], "D_unverified")
            # S
            s_gate = "S_unverified"
            if d["family"] == "SMALLLOWVOL" and str(d["strategy_id"]).startswith("COMBO_s10_h40"):
                s_gate = "S_pass"  # 14_压力：成本×2 +0.8%、延迟 +2.8%（full）
            # T (family-level stats)
            fs = fam_stats.get(d["family"], {})
            t_gate = "T_unverified"
            if fs:
                pbo_ok = fs.get("pbo_min") is not None and fs["pbo_min"] <= 0.25
                dsr_ok = fs.get("dsr_max") is not None and fs["dsr_max"] >= 0.95
                t_gate = "T_pass" if (pbo_ok and dsr_ok) else "T_fail_stat"
            gates = {"G": g_gate, "P": p_gate, "R": r_gate, "D": d_gate, "S": s_gate, "T": t_gate}
            all_pass = all(v.endswith("_pass") or v in ("D_pass_with_caveat", "D_pass_selfcheck", "P_na_overlay")
                           for v in gates.values())
            if not nt:
                grade = "E_underpowered"
            elif not (cg and mg):
                grade = "C_negative_valid"
            elif all_pass:
                grade = "A_candidate"
            else:
                grade = "B_conditional"
            if grade == "C_negative_valid":
                fails = [gates["G"]]
            elif grade == "E_underpowered":
                fails = [gates["P"]]
            else:
                fails = [v for v in gates.values() if not (v.endswith("_pass") or v in ("D_pass_with_caveat", "D_pass_selfcheck", "P_na_overlay"))]
        g["gates"] = gates
        g["grade"] = grade
        g["fail_gate"] = ";".join(fails) if fails else ""
        g["eligible"] = grade == "A_candidate"
        out.append(g)
        for rr in d["rows"]:
            prior = rr.get("grade", "")
            rr["grade"] = grade
            rr["fail_gate"] = g["fail_gate"]
            if prior and prior != grade:
                rr["notes"] = (rr["notes"] + f" [prior_grade={prior}]")[:1500]
    return out


def load_integrated_stats():
    """第一阶段族级 PBO/DSR（统计整合快照）作为 T 闸的补充证据。"""
    f = p("03_结果", "统计整合", "多重性.json")
    if not os.path.exists(f):
        return {}
    d = json.load(open(f, encoding="utf-8"))
    mats = d.get("matrices", {})
    fam_map = {"trend": "TREND", "struct": "STRUCT", "exit": "EXIT",
               "ecology": "ECOLOGY", "stat": "STAT"}
    out = {}
    for key, m in mats.items():
        fam = fam_map.get(m.get("family"))
        if not fam:
            continue
        out.setdefault(fam, {})
        pbo = (m.get("pbo") or {}).get("pbo")
        dsr_v = (m.get("dsr_family") or {}).get("dsr")
        if pbo is not None:
            out[fam]["pbo_min"] = min(out[fam].get("pbo_min", 1.0), float(pbo))
        if dsr_v is not None:
            out[fam]["dsr_max"] = max(out[fam].get("dsr_max", 0.0), float(dsr_v))
        out[fam]["source"] = "03_结果/统计整合/多重性.json"
    return out


def family_stats_from(pbo_rows, ms_rows, dsr_rows):
    fs = {}
    for r in pbo_rows:
        fs.setdefault(r["family"], {})["pbo_min"] = min(
            fs.get(r["family"], {}).get("pbo_min", 1.0), r["pbo"])
    for r in ms_rows:
        fs.setdefault(r["family"], {})["p_maxstat_min"] = min(
            fs.get(r["family"], {}).get("p_maxstat_min", 1.0), r["p_maxstat"])
    for r in dsr_rows:
        fs.setdefault(r["family"], {})["dsr_max"] = max(
            fs.get(r["family"], {}).get("dsr_max", 0.0), r["dsr"])
    return fs




def nav_key(family, strategy_id, config_id=None):
    sid = str(strategy_id)
    if family == "RISKBUDGET":
        return sid.split("|")[0]
    if family == "SMALLLOWVOL":
        return sid.split("_")[0]
    if family == "UNIFIEDENGINE":
        return {"T-U01_mom": "FIX20", "T-U02_mom": "ATR3",
                "T-U03_adv": "FIX20_adv", "T-U04_adv": "ATR3_adv"}.get(sid, sid)
    return sid


def _sr_stats(series):
    r = series.pct_change().dropna().values
    r = r[np.isfinite(r)]
    return r


def dsr_for_series(series, N, var_sr):
    r = _sr_stats(series)
    if len(r) < 60 or np.std(r) <= 0:
        return None
    sr = float(np.mean(r) / np.std(r))
    d = dsr(sr, len(r), var_sr, int(N), float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))
    return d


def _score_row(r, val):
    try:
        return min(float(r["cagr"]), float(val["cagr"])) / max(abs(float(r["mdd"])), 1e-6)
    except Exception:
        return -1e9


def dsr_block(rows, store):
    """项目级 DSR：Var(SR_trials) 用全部有净值序列的 trial；SR0 用 N 阶梯。"""
    sr_trials = []
    for (fam, cfg, win), s in store.items():
        r = _sr_stats(s)
        if len(r) >= 60 and np.std(r) > 0:
            sr_trials.append(float(np.mean(r) / np.std(r)))
    var_sr = float(np.var(sr_trials, ddof=1)) if len(sr_trials) > 1 else 0.5
    # 按预注册选择分数 min(dev,val CAGR)/max(MDD) 选“最好的那个”
    cand = []
    for r in rows:
        if r["category"] not in ACCOUNT_CATS or r["stage"] != "dev":
            continue
        fam, cid, sid = r["family"], r["config_id"], r["strategy_id"]
        key = nav_key(fam, sid, cid)
        if (fam, key, "dev") not in store:
            continue
        val = next((x for x in rows if x["family"] == fam and x["config_id"] == cid and x["stage"] == "val"), None)
        if val is None or not r["cagr"] or not val["cagr"]:
            continue
        try:
            is_g = (float(r["cagr"]) > 0 and float(val["cagr"]) > 0
                    and abs(float(r["mdd"])) <= 0.25 and abs(float(val["mdd"])) <= 0.25)
        except Exception:
            is_g = False
        cand.append((int(is_g), _score_row(r, val), fam, cid, sid, key))
    cand.sort(reverse=True)
    best = None
    if cand:
        is_g, score, fam, cid, sid, key = cand[0]
        best = {"family": fam, "config_id": cid, "strategy_id": sid, "nav_key": key,
                "score": score, "is_g_pass": bool(is_g)}
        for win in ("dev", "val"):
            srs = _sr_stats(store[(fam, key, win)])
            best[win] = {"sr_daily": float(np.mean(srs) / np.std(srs)), "T": int(len(srs)),
                         "skew": float(stats.skew(srs)),
                         "kurt": float(stats.kurtosis(srs, fisher=False))}
    N_sel = int(sum(float(r["in_Nsel"] or 0) for r in rows))
    N_plat = int(sum(float(r["in_Nplatform"] or 0) for r in rows))
    N_diag = int(sum(float(r["in_Ndiag"] or 0) for r in rows))
    ladder = {}
    for name, N in (("N1_sel", N_sel), ("N2_sel_plus_platform", N_sel + N_plat),
                    ("N3_all_cells", N_sel + N_plat + N_diag)):
        item = {"N": int(N), "sr0_annual": None, "dsr_best_dev": None, "dsr_best_val": None}
        if best and best.get("dev"):
            item["sr0_annual"] = float(
                math.sqrt(var_sr) * ((1 - 0.5772156649015329) * stats.norm.ppf(1 - 1.0 / N)
                                     + 0.5772156649015329 * stats.norm.ppf(1 - 1.0 / (N * math.e)))
                * math.sqrt(243))
            for win in ("dev", "val"):
                if best.get(win):
                    d = dsr(best[win]["sr_daily"], best[win]["T"], var_sr, N,
                            best[win]["skew"], best[win]["kurt"])
                    item[f"dsr_best_{win}"] = d["dsr"]
        ladder[name] = item
    return {"var_sr_daily": var_sr, "n_sr_trials": len(sr_trials),
            "sr_trials_annual_sd": float(math.sqrt(var_sr) * math.sqrt(243)),
            "best": best, "ladder": ladder}


def family_dsr(rows, store, var_sr):
    out = []
    for fam in sorted(set(r["family"] for r in rows if r["category"] in ACCOUNT_CATS)):
        N = int(sum(float(r["in_Nsel"] or 0) for r in rows if r["family"] == fam))
        if N < 2:
            continue
        cand = []
        for r in rows:
            if r["family"] != fam or r["category"] not in ACCOUNT_CATS or r["stage"] != "dev":
                continue
            sid = r["strategy_id"]
            key = nav_key(fam, sid, r["config_id"])
            if (fam, key, "dev") not in store:
                continue
            val = next((x for x in rows if x["family"] == fam and x["config_id"] == r["config_id"] and x["stage"] == "val"), None)
            if val is None or not r["cagr"] or not val["cagr"]:
                continue
            cand.append((min(float(r["cagr"]), float(val["cagr"])), sid, r["config_id"], key))
        if not cand:
            continue
        cand.sort(reverse=True)
        _, sid, cid, key = cand[0]
        d = dsr_for_series(store[(fam, key, "dev")], N, var_sr)
        if not d:
            continue
        out.append({"family": fam, "config": sid, "config_id": cid, "N": N, **d})
    return out


def clean_json(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    return o


def write_csv(path, rows, cols=None):
    rows = list(rows)
    if not rows:
        open(path, "w", encoding="utf-8-sig").write("")
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)



def pct(x, nd=2):
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v*100:.{nd}f}%"
    except Exception:
        return "—"



def write_audit(total):
    rows = []
    for fam, status in sorted(D_GATE.items()):
        note = {
            "D_pass": "复权审计 11 组逐笔/事件复算：A 空间可复现，未用冻结 v1 hfq",
            "D_pass_with_caveat": "含分批离场，A 空间仍优于 H 3.4×；逐笔口径 1.956pp",
            "D_pass_selfcheck": "族内 A 空间恒等式/日收益==pct_chg 自检通过；未进 11 组复算名单",
            "D_na_fixture": "统计框架夹具，无策略收益",
        }.get(status, "")
        rows.append({"family": fam, "D_gate": status, "evidence": note,
                     "source": "05_审计/复权修复影响.md"})
    rows.append({"family": "ALL", "D_gate": "2021_guard",
                 "evidence": f"载入 {len(NAV_AUDIT)} 条净值序列，max(date)<=2020-12-31 全部通过；2021+ 收益 0 行",
                 "source": "03_结果/最终统计/净值序列_2021审计.csv"})
    write_csv(os.path.join(FINAL_DIR, "复权与2021审计.csv"), rows)


def selftest():
    out = {}
    rng = np.random.default_rng(12345)
    T, C = 2000, 20
    noise = pd.DataFrame(rng.normal(0, 0.01, size=(T, C)))
    p_noise = pbo_cscv(noise)
    out["pbo_pure_noise"] = p_noise
    sig = noise.copy()
    sig[0] = rng.normal(0.0012, 0.01, size=T)
    out["pbo_one_signal"] = pbo_cscv(sig)
    dsr_null = []
    for _ in range(200):
        r = rng.normal(0, 0.01, size=1000)
        dsr_null.append(dsr(float(np.mean(r) / np.std(r)), 1000, 0.01 ** 2, 10,
                            float(stats.skew(r)), float(stats.kurtosis(r, fisher=False)))["dsr"])
    out["dsr_null_mean"] = float(np.mean(dsr_null))
    out["dsr_null_p95"] = float(np.quantile(dsr_null, 0.95))
    out["dsr_null_frac_gt_95"] = float((np.array(dsr_null) > 0.95).mean())
    # 2021 guard
    try:
        assert_no_2021(pd.DataFrame({"date": ["2021-01-04"]}), "selftest")
        out["guard_2021"] = "FAIL_not_raised"
    except AssertionError:
        out["guard_2021"] = "PASS_raised"
    ms_noise = maxstat_bootstrap(noise, B=500)
    ms_sig = maxstat_bootstrap(sig, B=500)
    out["maxstat_noise"] = ms_noise
    out["maxstat_one_signal"] = ms_sig
    out["maxstat_ok"] = bool(ms_noise and ms_sig and ms_noise["p_maxstat"] > 0.1 and ms_sig["p_maxstat"] < 0.1)
    out["pbo_noise_ok"] = bool(p_noise and 0.25 <= p_noise["pbo"] <= 0.75)
    out["pbo_signal_ok"] = bool(out["pbo_one_signal"] and out["pbo_one_signal"]["pbo"] < p_noise["pbo"])
    out["dsr_null_ok"] = bool(out["dsr_null_mean"] < 0.6 and out["dsr_null_frac_gt_95"] < 0.2)
    out["pass"] = all([out["pbo_noise_ok"], out["pbo_signal_ok"], out["dsr_null_ok"],
                       out["maxstat_ok"], out["guard_2021"] == "PASS_raised"])
    with open(os.path.join(FINAL_DIR, "自检.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    return out


def write_report(total, by_family, by_cat, gates, mechanical, stats, gpass_dsr, n_cover, n_unrated, path):
    L = []
    L.append("# 最终统计终审：全项目试验总账、硬门槛与选择校正\n")
    L.append(f"- 生成：`python3 02_代码/最终试验总账.py --stage all`（种子 {SEED}）")
    L.append("- 数据：冻结日 2026-10-26；收益只到 2020-12-31；**2021+ 未计算任何收益**")
    L.append("- 本文件只读他人产物；总账在 `03_结果/最终试验总账.csv`（哈希链）\n")
    L.append("## 0. 结论\n")
    L.append(f"**0 个配置通过进入留出的全部硬门槛（R/D/P/G/S/T）。留出不运行。**\n")
    L.append(f"- 可部署配置（含敏感性） **N_sel={int(total['N_sel'])}**；评估单元 **N_cell={int(total['N_cell'])}**（含 23 个平台/审计格）；"
             f"诊断 **N_diag={int(total['N_diag'])}**；平台/摩擦格 **N_platform={int(total['N_platform'])}**；"
             f"审计夹具 **N_audit={int(total['N_audit'])}**。")
    L.append(f"- 预注册硬预算 64 → 超出 **{total['N_sel']/64:.2f}×**。任何“最好的一格”必须在 N={int(total['N_sel'])} 的选择空间里读。")
    from collections import Counter as _C
    _mc = _C(g["family"] for g in mechanical)
    _desc = "、".join(f"{k}×{v}" for k, v in _mc.items()) or "无"
    L.append(f"- 机械通过 G 闸的配置 **{len(mechanical)} 个**（{_desc}）：全部被 R/S/T 闸否决。")
    _d = stats.get("dsr") or {}
    _b = _d.get("best") or {}
    _n1 = (_d.get("ladder") or {}).get("N1_sel", {})
    _sr_dev = ((_b.get("dev") or {}).get("sr_daily") or 0) * math.sqrt(243)
    _sr_val = ((_b.get("val") or {}).get("sr_daily") or 0) * math.sqrt(243)
    L.append(f"- 项目级 DSR（G 通过者中可重算的 VCB-P20）：N={_n1.get('N','—')} 时 dev **{_n1.get('dsr_best_dev', 0):.2e}**、"
             f"val **{_n1.get('dsr_best_val', 0):.2e}**；SR0 年化约 {_n1.get('sr0_annual', 0):.2f}，"
             f"该配置 SR 年化约 {_sr_dev:.2f}/{_sr_val:.2f}。")
    L.append("- 年化 60% 在全部族、全部压力格、全部配置里都没有数据支持；最高可核验全窗口净年化仍是执行压力 S4 固定交易集合 **12.1%**（1000 万/10 仓/5bp，非原账户路径），其余主候选两段为负或单段为负。\n")

    L.append("## 1. 真实试验数\n")
    L.append("计数口径（与 `03_结果/统计整合/00_整合预注册.md` 对齐）：`config` 与窗口无关；`cell` 为 config×window；"
             "`diag` 为事件/消融/交易层诊断；`platform` 为事件条件/摩擦/参数平台格；`audit` 为覆盖/自检/基准/随机对照。"
             "`N_sel` 只含账户级可部署配置（含敏感性），`N_sens` 是它的子集标记，不与 N_sel 相加。\n")
    L.append("| 族 | 行 | N_sel | N_cell | N_diag | N_platform | N_audit |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, v in sorted(by_family.items(), key=lambda kv: -kv[1]["N_sel"]):
        if v["rows"] == 0:
            continue
        L.append(f"| {fam} | {v['rows']} | {int(v['N_sel'])} | {int(v['N_cell'])} | {int(v['N_diag'])} | {int(v['N_platform'])} | {int(v['N_audit'])} |")
    L.append(f"| **合计** | **{total['rows']}** | **{int(total['N_sel'])}** | **{int(total['N_cell'])}** | **{int(total['N_diag'])}** | **{int(total['N_platform'])}** | **{int(total['N_audit'])}** |\n")
    L.append("按类别：\n")
    L.append("| 类别 | 行 | N_sel | N_cell | N_diag | N_platform | N_audit |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for cat, v in sorted(by_cat.items()):
        L.append(f"| {cat} | {v['rows']} | {int(v['N_sel'])} | {int(v['N_cell'])} | {int(v['N_diag'])} | {int(v['N_platform'])} | {int(v['N_audit'])} |")
    L.append("")
    L.append("与第一阶段披露的差异（不掩盖）：第一阶段总账记录 N_sel=365 / N_cell=500 / N_diag=369；"
             "整合结论披露 N_diag=411。差额 42 = ECOLOGY 事件条件格 36（总账记 1）+ MKT 状态表 8（总账记 1）。"
             "本总账补两条修正行（+35、+7），再加第二三阶段新族 diag 13（VCB 10 + BREAKPULL 2 + STRUCTREV 1），得 N_diag=424。**N_sel 与 N_cell 未被缩小，反而因新族增加。**\n")
    L.append("第二阶段/第三阶段新增族（相对 23:05 的第一阶段快照）：RISKBUDGET、TRENDCALLBACK、VCB、RELSTR、EXITUNI、"
             "EXECPRESSURE、SMALLLOWVOL、BREAKPULL、STRUCTREV、LONGMOM、UNIFIEDENGINE。生态切换已在第一阶段计入（14），不重复。\n")

    L.append("族级对账（声明数 → 本总账；差额全部有去向，不是漏记）：\n")
    L.append("| 族 | 族内声明 | 本总账 N_sel | 差额去向 |")
    L.append("|---|---:|---:|---|")
    L.append("| VCB | 34 | 22 | 4 可行性 + 6 事件层消融 → N_diag；`VCB-M` 与 `VCB-M/T40` 为同一配置、base 敏感性重复 → 去重 2 |")
    L.append("| SMALLLOWVOL | 112 | 109 | 容量 12 → N_platform；随机种子 11 → N_audit；压力实际 42（声明 35，+7 已披露）、邻域 4 |")
    L.append("| BREAKPULL | 188 单元 | 20 | 144 事件平台 → N_platform；6 统一引擎复核 → N_audit；2 个路径条件臂（未回踩/假突破）含未来信息 → N_diag |")
    L.append("| RISKBUDGET | 59 | 29 | 30 个 base/static/volmatch/levneutral/voltarget/tercile 对照臂 → N_audit |")
    L.append("| STRUCTREV | 32 | 30 | 1 事件层独有配置 → N_diag；1 随机哈希对照 → N_audit |")
    L.append("| RELSTR | 40 | 40 | B1/B2 基准 → N_audit（不计 N_sel） |")
    L.append("| LONGMOM / TRENDCALLBACK / EXITUNI / UNIFIEDENGINE | 26 / ≤40 / 30 / 4 | 26 / 35 / 30 / 4 | 无差额 |")
    L.append("| 第一阶段六族 | 365 | 365 | 无差额；另补 ECOLOGY/MKT diag 修正 +42 |\n")

    L.append("## 2. 可进入留出的硬门槛\n")
    L.append("| 闸 | 条件 | 本批结果 |")
    L.append("|---|---|---|")
    L.append("| R 登记 | 父 §7 有限族或族级预注册的 **main** 配置；网格/敏感性格不得事后升级 | 0 个候选；机械 G 通过者中 4 个是 TREND@S3 事后探索、1 个是 VCB 参数平台格 |")
    L.append("| D 数据 | 收益不得来自冻结 v1 hfq；复权白名单/族内自检 | 全部族通过（4 个族为自检级，未进 11 组复算） |")
    L.append("| P 功效 | dev/val 各 ≥30 笔、≥10 只 | 机械通过者均 ≥30 笔；部分族只有笔数、无 symbol 列 |")
    L.append("| G 绩效 | dev/val 各净 CAGR>0 且 MaxDD≤25% | 仅 5 个配置通过（见下表） |")
    L.append("| S 稳健 | 成本×2、延迟 1 日、去前 5 大盈利后仍 >0 | G 通过者均未做对应压力 → S_unverified |")
    L.append("| T 统计 | DSR≥0.95 且 PBO≤0.25（族内） | 全部失败（最高族内 DSR 0.33） |\n")
    L.append("机械通过 G 的配置：\n")
    L.append("| 族 | 配置 | dev CAGR | val CAGR | dev MDD | val MDD | 判定 | 失败闸 |")
    L.append("|---|---|---:|---:|---:|---:|---|---|")
    for g in mechanical:
        L.append(f"| {g['family']} | {g['strategy_id']} | {pct(g['dev_cagr'])} | {pct(g['val_cagr'])} | "
                 f"{pct(g['dev_mdd'])} | {pct(g['val_mdd'])} | {g['grade']} | {g['fail_gate']} |")
    L.append("")
    L.append("各族预注册主候选（main）的结局：\n")
    L.append("| 族 | 主候选 | dev CAGR | val CAGR | dev MDD | val MDD | 判定 | 失败闸 |")
    L.append("|---|---|---:|---:|---:|---:|---|---|")
    mains = [g for g in gates if g.get("role") == "main" and g.get("category") == "prereg"]
    for g in mains:
        L.append(f"| {g['family']} | {g['strategy_id']} | {pct(g['dev_cagr'])} | {pct(g['val_cagr'])} | "
                 f"{pct(g['dev_mdd'])} | {pct(g['val_mdd'])} | {g['grade']} | {g['fail_gate']} |")
    L.append("| MAIN | T0005（F1×S1×E2×X3×G1） | — | — | — | — | F_unevaluable | F_never_run（从未运行） |")
    L.append("")
    L.append("**为什么是 0 候选（逐条理由）**\n")
    L.append("1. **TREND@S3**：S3 不在父预注册 §7 的有限族内，没有 `trial_id`；`trend_dev` max-stat p=0.374、`trend_val` p=0.716，PBO 0.452/0.127；族内 DSR 0.330/0.180。它的正收益是 42 格趋势网格里挑出的 4 格，且每窗口仅 44–71 笔。")
    L.append("2. **VCB-P20/T40**：它是收缩突破 9 个信号配置里的 20% 分位敏感性格，不是预注册主候选（主候选 VCB-M/T40 开发 MDD −36.3%，G 不过）。族预注册明确“未把 N120/P20/T60 等看起来较好的配置升级为主候选”。事件层 dev 超额 −1.08%（t=−0.86）、val +0.92%（t=−0.49），没有信号层支持。")
    L.append("3. **S 闸**：VCB-P20 没有成本×2/延迟/去前 5 的预注册压力结果；按“未证明即不通过”处理。")
    L.append("4. **T 闸**：VCB 族 PBO 0.309（dev）/0.505（val），族内 DSR 0.063；VCB-P20 在 N=714 下 DSR≈1e−4。")
    L.append("5. **BREAKPULL 的“主-未回踩/主-假突破”**：按“是否回踩/是否假突破”事后切分，含未来信息，已从 N_sel 移入 N_diag（否则会出现 dev +82%/val +108% 的假候选）。\n")

    L.append("## 3. 统计：PBO / max-stat / CI / DSR\n")
    L.append("### 3.1 族×窗口\n")
    L.append("| 族 | 窗口 | 候选 | 最好 CAGR | 5% 分位 | max-stat p | PBO | PBO 中位排名 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    pbo_map = {(r["family"], r["window"]): r for r in stats.get("pbo", [])}
    ms_map = {(r["family"], r["window"]): r for r in stats.get("maxstat", [])}
    ci_map = {(r["family"], r["config"], r["window"]): r for r in stats.get("ci", [])}
    for (fam, win), r in sorted(ms_map.items()):
        p = pbo_map.get((fam, win), {})
        L.append(f"| {fam} | {win} | {r['n_candidates']} | {pct(r.get('best_cagr'))} | "
                 f"{pct(r.get('best_cagr_p05'))} | {r['p_maxstat']:.3f} | "
                 f"{p['pbo']:.3f} | {round(p.get('median_oos_rank', float('nan')),3)} |")
    L.append("")
    pmin = min(((r["family"], r["window"], r["p_maxstat"]) for r in stats.get("maxstat", [])),
               key=lambda x: x[2], default=None)
    L.append("注意：PBO 是“选择流程”诊断，不是未来亏损概率；候选全为负时，低 PBO 只说明“最不差”排序稳定。"
             + (f"本批可重算矩阵的最小 max-stat p={pmin[2]:.3f}（{pmin[0]} {pmin[1]}），未过 0.05。" if pmin else "")
             + "第一阶段发布的 TREND dev p=0.374 也未通过。\n")
    L.append("### 3.2 DSR 阶梯（N 不缩小）\n")
    d = stats.get("dsr", {})
    L.append(f"- `Var(SR_trials)`={d.get('var_sr_daily'):.6g}（日频），来自 {d.get('n_sr_trials')} 条有净值序列的 trial；年化 SR 标准差约 {d.get('sr_trials_annual_sd'):.2f}。")
    L.append(f"- “最好的可评级配置”：{d.get('best',{}).get('family')}/{d.get('best',{}).get('strategy_id')}（is_g_pass={d.get('best',{}).get('is_g_pass')}）。\n")
    L.append("| 口径 | N | SR0（年化） | DSR(dev) | DSR(val) |")
    L.append("|---|---:|---:|---:|---:|")
    for k, v in (d.get("ladder") or {}).items():
        L.append(f"| {k} | {v['N']} | {v['sr0_annual']:.3f} | {v['dsr_best_dev']:.6f} | {v['dsr_best_val']:.6f} |")
    L.append("")
    if gpass_dsr:
        L.append("G 通过者的选择校正：\n")
        L.append("| 族 | 配置 | DSR | 来源 |")
        L.append("|---|---|---:|---|")
        seen = set()
        for x in gpass_dsr:
            key = (x["family"], x["strategy_id"], x.get("source"))
            if key in seen:
                continue
            seen.add(key)
            if x.get("source", "").startswith("recomputed"):
                val = x.get(f"dsr_dev_N{int(total['N_sel'])}", None)
                L.append(f"| {x['family']} | {x['strategy_id']} | {val:.6f} | 重算（日净值，N={int(total['N_sel'])}） |")
            else:
                L.append(f"| {x['family']} | {x['strategy_id']} | {x.get('dsr_family_published','—')} | {x['source']}（同族多配置共用） |")
    L.append("")
    L.append("### 3.3 短名单 CI（stationary bootstrap，B=5000）\n")
    L.append("| 族 | 配置 | 窗口 | 点估计 | 5% | 50% | 95% | P(CAGR≤0) |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for c in sorted(stats.get("ci", []), key=lambda x: (x["family"], x["config"], x["window"])):
        L.append(f"| {c['family']} | {c['config']} | {c['window']} | {pct(c['cagr_point'])} | {pct(c['cagr_p05'])} | "
                 f"{pct(c['cagr_p50'])} | {pct(c['cagr_p95'])} | {c['p_cagr_le_0']:.3f} |")
    L.append("")
    L.append("CI 的 `cagr_point` 是按连续净值切片重算的 243 日几何年化；族报告的分窗口净值从现金重置，二者在少数族差 1–2pp"
             "（BREAKPULL val −18.3% vs 报告 −19.9%；LONGMOM C12 val −11.4% vs 报告 −13.0%）。门槛表用族报告值，CI 用于看区间宽度。\n")

    L.append("## 4. 统计的适用边界（不能排除什么）\n")
    L.append("- **PBO** 只在同族、同窗口、同日网格、候选≥5 且共同交易日≥120 时成立；结构族/市场状态族的合并窗口不能拆成 dev/val，SMALLLOWVOL 只有主单元日净值，均不能做 PBO。")
    L.append("- **bootstrap CI** 假设日收益可重采样；10 仓、40–60 日持有的组合路径方差极大，44–71 笔的配置区间偏窄（整合结论 §5.6）。")
    L.append("- **DSR** 的 `Var(SR_trials)` 混合不同族/窗口/杠杆，仅作项目级诊断；族内 DSR 单列。N 阶梯 N1/N2/N3 同时报告，避免“把压力格当独立发现”或“漏记以缩小 N”。")
    L.append("- **门槛本身**不能排除：制度变迁（2008 与 2016 后流动性/规模溢价反转）、容量（小盘低波 1 亿部署 13.7%）、执行偏差（日线内顺序不可知）、样本期特殊性（2014–2015 单年贡献）。")
    L.append("- **没有 walk-forward**：除统计框架 4 折外，没有族完成父 §10.1 的 5 折滚动；因此即使有配置过 G，也不能称 E1 成立。\n")

    L.append("## 5. 2021+ 与复权审计\n")
    L.append(f"- 所有载入的 {len(NAV_AUDIT)} 条净值序列 `max(date) ≤ 2020-12-31`（脚本内断言）；多源聚合的 `市场状态_日频许可_扩展2021_2026_未评估.csv` 未载入。")
    L.append("- 复权：本项目策略收益未使用冻结 v1 `hfq_*`（`05_审计/复权修复影响.md` §3.5：11 组逐笔/事件复算，A 空间误差 0.000–1.956pp，H 空间 5.2–11.3pp）。4 个族（RISKBUDGET/BREAKPULL/STRUCTREV/LONGMOM）只有族内 A 空间自检，未进 11 组复算，标 `D_pass_selfcheck`。")
    L.append("- 冻结层 v2 已通过 `hfq_ret == pct_chg` 验收；本审计不重跑 v1。\n")

    L.append("## 6. 缺陷与限制（本审计自身）\n")
    L.append(f"1. 门槛表 {len(gates)} 行覆盖 {int(n_cover)} 个 N_sel；另有 {int(n_unrated)} 个 N_sel 无 dev/val 两窗口记录（STRUCT 219 只有 2008–2020 合并窗口、STAT 9 为统计框架夹具），只能作负证据，不能作候选。")
    L.append("2. 第一阶段总账是 23:05 快照；23:05 后新增的族才被本总账纳入。若之后再有新族，必须重跑本脚本。")
    L.append("3. 部分族的 `n_symbols` 未落盘，P 闸只能验笔数；对机械 G 通过者已用交易文件补验（VCB-P20：171/87 只）。")
    L.append("4. 执行压力的 S4/S1 是“固定交易集合重定价”，不是原账户路径；其 12.1% 不能当策略年化。")
    L.append("5. `BREAKPULL` 的“未回踩/假突破”臂证明：路径条件筛选会产生 +82%/+108% 的假年化；本总账已将其移出 N_sel。")
    L.append("6. 官方主候选 `T0005 (F1×S1×E2×X3×G1)` 从未运行，总账登记为 `planned/F_unevaluable`；更新命令：`python3 02_代码/实验登记.py start --trial T0005` 后跑统一回测，再重跑本脚本。")
    L.append("7. `LONGMOM` 的 val `cagr` 与 `nav_start/nav_end/years` 反算值差 1.2–1.8pp（dev/full 无差异），原因未查明；本审计门槛用族报告值，CI 用净值反算。两者均为负，结论不变。\n")

    L.append("## 7. 复现与更新\n")
    L.append("```bash")
    L.append("python3 02_代码/最终试验总账.py --stage all      # 总账 + 统计 + 判定")
    L.append("python3 02_代码/最终试验总账.py --stage ledger   # 只重建总账（约 5s）")
    L.append("python3 02_代码/实验登记.py verify --strict      # 官方链校验（只读）")
    L.append("```")
    L.append("")
    L.append("新增族/结果后：重跑 `--stage all`；若方法负责人批准主候选 T0005，先 `实验登记.py start --trial T0005`，再跑统一回测，最后重跑本脚本。"
             "**在 0 候选状态下，2021-2025 留出不得运行。**")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ledger", "stats", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(FINAL_DIR, exist_ok=True)
    log = []
    def L(msg):
        print(msg)
        log.append(msg)

    L("== 载入第一阶段总账")
    s1, meta = load_stage1()
    L(f"  {len(s1)} rows; Nsel={meta['stage1_Nsel']:.0f} Ncell={meta['stage1_Ncell']:.0f} Ndiag={meta['stage1_Ndiag']:.0f}（记录值）")
    L("== 载入第二三阶段新族")
    new = build_all_new()
    corr = build_corrections()
    mainph = build_main_candidate_placeholder()
    auditfix = build_audit_fixtures()
    rows = s1 + new + corr + mainph + auditfix
    L(f"== 总行数 {len(rows)}")

    def summ(rs):
        return {
            "rows": len(rs),
            "N_sel": sum(float(x["in_Nsel"] or 0) for x in rs),
            "N_cell": sum(float(x["in_Ncell"] or 0) for x in rs),
            "N_diag": sum(float(x["in_Ndiag"] or 0) for x in rs),
            "N_platform": sum(float(x["in_Nplatform"] or 0) for x in rs),
            "N_sens_subset": sum(float(x["in_Nsens"] or 0) for x in rs),
            "N_audit": sum(float(x["in_Naudit"] or 0) for x in rs),
        }
    total = summ(rows)
    L("== 计数 " + json.dumps(total, ensure_ascii=False))
    by_family = {fam: summ([x for x in rows if x["family"] == fam])
                 for fam in sorted(set(x["family"] for x in rows))}
    by_cat = {cat: summ([x for x in rows if x["category"] == cat])
              for cat in sorted(set(x["category"] for x in rows))}

    # ---- 2021+ 审计 & 净值序列 ----
    store = load_nav_series()
    L(f"== 净值序列 {len(store)} 条；全部 max(date) <= 2020-12-31（断言通过）")
    write_csv(os.path.join(FINAL_DIR, "净值序列_2021审计.csv"), NAV_AUDIT)

    stats = {"pbo": [], "maxstat": [], "ci": [], "dsr": {}, "family_dsr": []}
    fam_stats = {}
    if args.stage in ("stats", "all"):
        L("== 统计：PBO / max-stat / CI")
        mats, pbo_rows, ms_rows, ci_rows = run_stats(store, quick=args.quick)
        stats["pbo"] = pbo_rows
        stats["maxstat"] = ms_rows
        stats["ci"] = ci_rows
        L(f"  PBO {len(pbo_rows)} 个矩阵；max-stat {len(ms_rows)} 个；CI {len(ci_rows)} 条")
        dsr_res = dsr_block(rows, store)
        stats["dsr"] = dsr_res
        L(f"  DSR: Var(SR)={dsr_res['var_sr_daily']:.6g} trials={dsr_res['n_sr_trials']} best={dsr_res['best']}")
        for k, v in dsr_res["ladder"].items():
            L(f"    {k}: N={v['N']} SR0(年化)={v['sr0_annual']} DSR_dev={v['dsr_best_dev']} DSR_val={v['dsr_best_val']}")
        fdsr = family_dsr(rows, store, dsr_res["var_sr_daily"])
        stats["family_dsr"] = fdsr
        for r in fdsr:
            L(f"    族内 {r['family']}: N={r['N']} best={r['config']} SR年化={r['sr_annual']:.3f} DSR={r['dsr']:.4f}")
        fam_stats = family_stats_from(pbo_rows, ms_rows, [{"family": r["family"], "dsr": r["dsr"]} for r in fdsr])
        for fam, v in load_integrated_stats().items():
            cur = fam_stats.setdefault(fam, {})
            for k, x in v.items():
                if k == "pbo_min":
                    cur[k] = min(cur.get(k, 1.0), x)
                elif k == "dsr_max":
                    cur[k] = max(cur.get(k, 0.0), x)
                else:
                    cur[k] = x
        write_csv(os.path.join(FINAL_DIR, "最终PBO_maxstat.csv"),
                  [{"kind": "PBO", **r} for r in pbo_rows] + [{"kind": "MAXSTAT", **r} for r in ms_rows])
        write_csv(os.path.join(FINAL_DIR, "最终置信区间.csv"), ci_rows)
        write_csv(os.path.join(FINAL_DIR, "族内DSR.csv"), fdsr)

    # ---- 统一门槛 ----
    gates = compute_gates(rows, fam_stats)
    write_csv(os.path.join(FINAL_DIR, "统一门槛.csv"), gates)
    gated_keys = {(g["family"], g["config_id"]) for g in gates}
    N_cover = sum(float(r["in_Nsel"] or 0) for r in rows
                  if r["category"] in ACCOUNT_CATS and (r["family"], r["config_id"]) in gated_keys)
    N_unrated = int(total["N_sel"] - N_cover)
    candidates = [g for g in gates if g["eligible"]]
    mechanical = [g for g in gates if g["gates"].get("G") == "G_pass"]
    L(f"== 统一门槛：{len(gates)} 个可评级配置；G_pass={len(mechanical)}；全门槛通过 A_candidate={len(candidates)}")
    for g in mechanical:
        L(f"    G_pass {g['family']}/{g['strategy_id']} dev={g['dev_cagr']:.4f} val={g['val_cagr']:.4f} "
          f"MDD={g['dev_mdd']:.3f}/{g['val_mdd']:.3f} grade={g['grade']} fail={g['fail_gate']}")

    # ---- 写总账（grades 已更新） ----
    prev = meta["stage1_tail"]
    for i, r in enumerate(rows):
        r["ledger_id"] = f"F{i:05d}"
        r["prev_sha256"] = prev
        r["row_sha256"] = sha256_text("\x1f".join(str(r[c]) for c in FINAL_COLS[:-1]))
        prev = r["row_sha256"]
    with open(FINAL_LEDGER, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_COLS)
        w.writeheader()
        w.writerows(rows)
    L(f"== 写出 {os.path.relpath(FINAL_LEDGER, ROOT)} ({len(rows)} rows)")

    with open(os.path.join(FINAL_DIR, "最终计数.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "name", "rows", "N_sel", "N_cell", "N_diag", "N_platform", "N_sens_subset", "N_audit"])
        w.writerow(["total", "ALL", total["rows"], total["N_sel"], total["N_cell"], total["N_diag"],
                    total["N_platform"], total["N_sens_subset"], total["N_audit"]])
        for k, v in by_family.items():
            w.writerow(["family", k, v["rows"], v["N_sel"], v["N_cell"], v["N_diag"], v["N_platform"], v["N_sens_subset"], v["N_audit"]])
        for k, v in by_cat.items():
            w.writerow(["category", k, v["rows"], v["N_sel"], v["N_cell"], v["N_diag"], v["N_platform"], v["N_sens_subset"], v["N_audit"]])

    # ---- G 通过者的 DSR（能否经得起选择校正） ----
    gpass_dsr = []
    if stats.get("dsr"):
        var_sr = stats["dsr"].get("var_sr_daily", 0.5)
        N1 = int(total["N_sel"])
        N2 = int(total["N_sel"] + total["N_platform"])
        integrated = load_integrated_stats()
        for g in mechanical:
            fam = g["family"]
            key = nav_key(fam, g["strategy_id"], g["config_id"])
            item = {"family": fam, "strategy_id": g["strategy_id"], "config_id": g["config_id"],
                    "grade": g["grade"], "fail_gate": g["fail_gate"]}
            if (fam, key, "dev") in store:
                for N in (N1, N2):
                    d = dsr_for_series(store[(fam, key, "dev")], N, var_sr)
                    if d:
                        item[f"dsr_dev_N{N}"] = d["dsr"]
                        item[f"sr_annual_N{N}"] = d["sr_annual"]
                item["source"] = "recomputed_daily_nav"
            else:
                fs = integrated.get(fam, {})
                item["dsr_family_published"] = fs.get("dsr_max")
                item["pbo_family_published"] = fs.get("pbo_min")
                item["source"] = "03_结果/统计整合/多重性.json（本族无逐日净值，不能重算）"
            gpass_dsr.append(item)

    write_audit(total)
    if args.selftest:
        st = selftest()
        L(f"== 自检: {json.dumps(st, ensure_ascii=False, default=float)[:300]}")
    write_report(total, by_family, by_cat, gates, mechanical, stats, gpass_dsr,
                 N_cover, N_unrated, os.path.join(FINAL_DIR, "00_最终报告.md"))

    # 平台明细（不把压力格当独立发现，但也不漏记运行数）
    try:
        ep_runs = int(read_csv(p("03_结果", "执行压力", "01_压力矩阵.csv")).shape[0])
    except Exception:
        ep_runs = None

    # ---- 最终判定 JSON ----
    verdict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "frozen_date": "2026-10-26",
        "windows": {"dev": "2008-01-01..2015-12-31", "val": "2016-01-01..2020-12-31",
                    "holdout": "2021-01-01..2025-12-31（未运行）"},
        "seed": SEED,
        "counts": {
            "total": total, "by_family": by_family, "by_category": by_cat,
            "N_dsr_ladder": {
                "N1_sel": int(total["N_sel"]),
                "N2_sel_plus_platform": int(total["N_sel"] + total["N_platform"]),
                "N3_all_cells": int(total["N_sel"] + total["N_platform"] + total["N_diag"]),
            },
            "platform_detail": {
                "EXECPRESSURE_scenarios": 101,
                "EXECPRESSURE_matrix_runs": ep_runs,
                "BREAKPULL_event_cells": 144,
                "SMALLLOWVOL_capacity_runs": 12,
                "note": "N_platform 计情景/平台格，不计运行数；运行数仅披露，不当独立发现",
            },
            "budget": {"prereg_hard_budget_select": 40, "prereg_hard_budget_with_ablation": 64,
                       "N_sel_over_budget_ratio": round(total["N_sel"] / 64.0, 2)},
            "stage1_recorded_vs_corrected": {
                "N_sel": [365, total["N_sel"]],
                "N_cell": [500, total["N_cell"]],
                "N_diag": [369, total["N_diag"]],
                "note": "第一阶段总账记录 N_diag=369；整合结论披露 411（ECOLOGY 36 而非 1、MKT 8 而非 1）。本总账补 35+7 修正行，故为 422（含新族 diag 11）。",
            },
        },
        "statistics": {
            "dsr": stats["dsr"],
            "g_pass_dsr": gpass_dsr,
            "family_dsr": stats["family_dsr"],
            "pbo": stats["pbo"],
            "maxstat": stats["maxstat"],
            "ci": stats["ci"],
            "applicability": {
                "pbo": "仅同族、同窗口、同日网格、候选数>=5 且 >=120 个共同交易日时计算；否则 PBO_not_computable",
                "bootstrap_ci": "stationary bootstrap，平均块长 20，B=5000；日频重采样对稀疏成交偏窄（44-71 笔的配置区间偏乐观）",
                "dsr": "Var(SR_trials) 混合不同族/窗口的日频 SR，仅作项目级诊断；族内 DSR 另表；N 阶梯同时报告",
                "negative_sharpe": "当一族全部候选 SR<0 时，低 PBO 只说明'最不差'排序稳定，不是有边际的证据",
            },
        },
        "holdout_gate": {
            "eligible_candidates": [g["config_id"] for g in candidates],
            "n_eligible": len(candidates),
            "mechanical_G_passers": [
                {"family": g["family"], "strategy_id": g["strategy_id"], "config_id": g["config_id"],
                 "dev_cagr": g["dev_cagr"], "val_cagr": g["val_cagr"],
                 "dev_mdd": g["dev_mdd"], "val_mdd": g["val_mdd"],
                 "grade": g["grade"], "fail_gate": g["fail_gate"]}
                for g in mechanical
            ],
            "n_gate_rows": len(gates),
            "N_sel_covered_by_gate_rows": int(N_cover),
            "N_sel_unrated": int(N_unrated),
            "unrated_reason": "无 dev/val 两窗口记录：STRUCT 219（只有 2008-2020 合并窗口）+ STAT 9（统计框架夹具）；只能作负证据",
            "grade_counts": dict(__import__("collections").Counter(g["grade"] for g in gates)),
            "gate_definitions": {
                "R": "预注册主候选（父 §7 有限族或族级预注册且 role=main）；网格/敏感性格不得事后升级",
                "D": "复权修复影响白名单：收益不得来自冻结 v1 hfq；抽查/族内自检",
                "P": "dev 与 val 各自 >=30 笔平仓（>=10 只股票，部分族仅有笔数）",
                "G": "dev 与 val 各自净 CAGR>0 且 MaxDD<=25%",
                "S": "成本x2、延迟1日、去前5大盈利后仍>0（仅对 G 通过者）",
                "T": "DSR>=0.95 且 PBO<=0.25（族内）",
            },
            "holdout_run": False,
            "holdout_reason": "0 个候选通过全部硬门槛；2021+ 未计算任何收益",
            "independent_coordinator_required": True,
        },
        "candidates": [],
        "verdict": {
            "code": "NO_CANDIDATE",
            "tier": "无可利用边际",
            "statement": (f"全项目 {int(total['N_sel'])} 个可部署配置、{int(total['N_cell'])} 个账户单元、"
                          f"{int(total['N_diag'])} 个诊断单元、{int(total['N_platform'])} 个平台格、"
                          f"{int(total['N_audit'])} 个审计夹具；0 个配置通过进入留出的全部硬门槛。"
                          f"机械通过 G 的 {len(mechanical)} 个格子（4 个 TREND@S3 事后探索 + 1 个 VCB-P20/T40 参数平台格）"
                          "全部被 R/S/T 闸否决：S3 无族级 trial_id 且 DSR 0.33/PBO 0.45，P20 是禁止升级的平台格、S 未验证、DSR≈0。留出（2021-2025）不运行。"),
            "annualized_60pct_supported": False,
            "best_mechanical_G_config": mechanical[0] if mechanical else None,
        },
        "restrictions": {
            "no_2021_plus_returns": True,
            "no_target_backsolving": True,
            "no_pressure_cell_as_independent_discovery": True,
        },
        "update_commands": [
            "python3 02_代码/最终试验总账.py --stage all   # 新增族/结果后重建总账与统计",
            "python3 02_代码/实验登记.py verify --strict   # 官方链校验（只读）",
            "python3 02_代码/整合实验登记.py build && python3 02_代码/整合实验登记.py stats  # 第一阶段快照复核",
            "python3 02_代码/实验登记.py start --trial T0005  # 仅在方法负责人批准后运行主候选（当前 planned/未运行）",
        ],
    }
    with open(os.path.join(FINAL_DIR, "最终判定.json"), "w", encoding="utf-8") as f:
        json.dump(clean_json(verdict), f, ensure_ascii=False, indent=2, allow_nan=False)

    # ---- 运行日志 ----
    with open(os.path.join(FINAL_DIR, "运行日志.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + f"\n总用时 {time.time()-t0:.1f}s\n")
    print("FINAL_VERDICT=" + json.dumps({
        "code": verdict["verdict"]["code"],
        "N_sel": int(total["N_sel"]), "N_cell": int(total["N_cell"]),
        "N_diag": int(total["N_diag"]), "N_platform": int(total["N_platform"]),
        "n_eligible": len(candidates), "holdout_run": False,
        "candidates": [],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
