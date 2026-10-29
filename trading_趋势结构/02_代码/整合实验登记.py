#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整合实验登记.py — 第一阶段全量试验的补登记、计数、统一推断与定级。

定位
----
本文件不产生任何新策略，也不改任何他人文件。它做四件事：

1. **盘点**：扫描第一阶段已落盘的全部结果表，把每一次实际评估登记成一行
   `03_结果/试验总账.csv`（含哈希链、计数权重、定级）。
2. **补登记**：官方 `实验注册表.csv` 只有 6 行、N=0，而实际评估数百次。
   本文件以官方链尾 `row_sha256` 为锚，另起**独立哈希链**补登记，
   官方文件零改动；`emit-registry` 可生成官方 append 命令供方法负责人并入。
3. **审查**：独立复算官方链（不调用其 verify），报告代码漂移、窗口、计数漏洞。
4. **统一推断**：stationary bootstrap 区间、DSR、PBO/CSCV、max-statistic，
   全部按 `统计整合/00_整合预注册.md` 的定义，种子固定，可复现。

边界
----
* 收益只算 2005-01-01..2020-12-31；2021-01-01 起只做覆盖扫描，不算收益。
* 只读源数据与他人产物；只写本文件、试验总账.csv、03_结果/统计整合/。

用法
----
    python3 02_代码/整合实验登记.py build          # 盘点 + 补登记 + 定级
    python3 02_代码/整合实验登记.py verify         # 校验总账哈希链
    python3 02_代码/整合实验登记.py audit-registry # 审查官方注册表
    python3 02_代码/整合实验登记.py defects        # 数据缺陷暴露与 2021+ 扫描
    python3 02_代码/整合实验登记.py nullcheck      # 随机组合零点复算
    python3 02_代码/整合实验登记.py stats          # 置信区间 / DSR / PBO / max-stat
    python3 02_代码/整合实验登记.py summary        # 终端摘要
    python3 02_代码/整合实验登记.py emit-registry  # 生成官方 append 命令
    python3 02_代码/整合实验登记.py selftest       # 机制自检（临时目录）
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
RES = ROOT / "03_结果"
INT = RES / "统计整合"
TREND = RES / "趋势族"
LEDGER = RES / "试验总账.csv"
REGISTRY = RES / "实验注册表.csv"
PREREG = ROOT / "00_研究预注册.md"
IPREREG = INT / "00_整合预注册.md"

FREEZE = dt.date(2026, 10, 26)
SEED = 20261026
BOOT = 2000
BOOT_SELECTED = 5000
BLOCK = 20
ANN = 243.0
PBO_S = 16
PBO_B = PBO_S // 2

LEDGER_COLUMNS = [
    "ledger_id", "source", "family", "strategy_id", "params_id", "config_id",
    "window_start", "window_end", "stage", "kind", "origin", "status",
    "n_units", "in_Nsel", "in_Ncell", "in_Ndiag",
    "n_trades", "cagr", "mdd", "sharpe", "vol", "calmar",
    "metric_basis", "window_mix", "artifact", "artifact_extra",
    "grade", "fail_gate", "notes",
    "prev_sha256", "row_sha256",
]
N_HASH = len(LEDGER_COLUMNS) - 2
SEP = "\x1f"
ZERO = "0" * 64

PANELS = {
    "trend_dev": (TREND, "净值_", "_dev.parquet"),
    "trend_val": (TREND, "净值_", "_val.parquet"),
}

# ---------------------------------------------------------------- 基础


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iv(v, default=None):
    """安全 int：NaN/缺失返回 default。"""
    try:
        if v is None:
            return default
        f = float(v)
        if not math.isfinite(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        if not math.isfinite(v):
            return ""
        return repr(round(v, 10))
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, np.bool_):
        return str(bool(v))
    return str(v)


def row_canonical(row: dict) -> str:
    return SEP.join(fmt(row[c]) for c in LEDGER_COLUMNS[:N_HASH])


def row_hash(row: dict) -> str:
    return sha256_text(row_canonical(row))


def pid(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False))[:12]


def cid(family: str, strategy: str, params: dict) -> str:
    return pid({"f": family, "s": strategy, "p": params})


def registry_anchor() -> str:
    if not REGISTRY.exists():
        return ZERO
    with REGISTRY.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1]["row_sha256"] if rows else ZERO


# ---------------------------------------------------------------- 盘点构造器

def mk(family, strategy, params, start, end, stage, kind, origin, status,
       n_units=1, n_trades=None, cagr=None, mdd=None, sharpe=None, vol=None,
       calmar=None, metric_basis="none", window_mix="single", artifact="",
       artifact_extra="", notes=""):
    p = params if isinstance(params, dict) else {"v": params}
    return {
        "source": "artifact",
        "family": family,
        "strategy_id": strategy,
        "params_id": pid(p),
        "config_id": cid(family, strategy, p),
        "window_start": start,
        "window_end": end,
        "stage": stage,
        "kind": kind,
        "origin": origin,
        "status": status,
        "n_units": n_units,
        "in_Nsel": 0, "in_Ncell": 0, "in_Ndiag": 0,
        "n_trades": n_trades,
        "cagr": cagr, "mdd": mdd, "sharpe": sharpe, "vol": vol, "calmar": calmar,
        "metric_basis": metric_basis,
        "window_mix": window_mix,
        "artifact": artifact,
        "artifact_extra": artifact_extra,
        "grade": "", "fail_gate": "",
        "notes": notes,
    }


PERIOD_WIN = {
    "dev": ("2008-01-01", "2015-12-31", "dev"),
    "dev_sub": ("2008-01-02", "2012-12-31", "dev_subwindow"),
    "val": ("2016-01-01", "2020-12-31", "val"),
    "preheat": ("2005-01-01", "2007-12-31", "preheat"),
    "pooled": ("2008-01-01", "2020-12-31", "pooled"),
    "pooled_0520": ("2005-01-01", "2020-12-31", "pooled"),
}
TREND_PERIOD = {"预热0507": "preheat", "开发0815": "dev", "验证1620": "val",
                "全样本0520": "pooled_0520", "dev": "dev", "val": "val"}
MKT_PERIOD = {"早期_2006-2007": ("2006-01-04", "2007-12-31", "preheat"),
              "开发_2008-2015": ("2008-01-01", "2015-12-31", "dev"),
              "验证_2016-2020": ("2016-01-01", "2020-12-31", "val"),
              "全段_2006-2020": ("2006-01-04", "2020-12-31", "pooled")}
ORIGIN = "post_hoc_unregistered"


def inv_official(rows):
    """官方注册表里已存在的行（只读镜像，用于口径对齐与未运行标记）。"""
    if not REGISTRY.exists():
        return
    with REGISTRY.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "meta":
                continue
            if r["kind"] == "audit":
                rows.append(mk("OFFICIAL", "official_audit", {"trial": r["trial_id"]},
                               r["window_start"], r["window_end"], "audit", "audit",
                               "preregistered", "done", n_units=1,
                               metric_basis="coverage", artifact="03_结果/实验注册表.csv",
                               notes=f"官方行 {r['exp_id']}；metrics={r['metrics_json'][:120]}"))
            elif r["status"] in ("planned", "blocked"):
                rows.append(mk("OFFICIAL", f"PREREG:{r['family']}", json.loads(r["params_json"]),
                               r["window_start"], r["window_end"], r["stage"], "experiment",
                               "preregistered", "not_run", n_units=0,
                               metric_basis="none", artifact="03_结果/实验注册表.csv",
                               notes=f"{r['trial_id']} 状态={r['status']}；"
                                     "预注册主候选从未运行，故不进入任何计数"))


def inv_trend(rows):
    p = TREND / "组合模拟_汇总.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[TREND_PERIOD[r.period]]
            rows.append(mk("TREND", f"TREND@{r.signal}", {"rule": r.rule}, st, en, stage,
                           "experiment", ORIGIN, "done", n_trades=int(r.n_trades),
                           cagr=r.cagr, mdd=r.maxdd, sharpe=r.sharpe, vol=r.vol,
                           calmar=r.calmar, metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes="6 信号 × 7 退出规则；非预注册 §7 的 S/E/X 定义"))

    p = TREND / "敏感性_资金与持仓数.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[TREND_PERIOD[r.period]]
            rows.append(mk("TREND", f"TREND@{r.signal}",
                           {"capital": int(r.capital), "n_pos": int(r.n_pos)}, st, en, stage,
                           "sensitivity", ORIGIN, "done", n_trades=int(r.n_trades),
                           cagr=r.cagr, mdd=r.maxdd, sharpe=r.sharpe, vol=r.vol,
                           calmar=r.calmar, metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes="资金 × 持仓数敏感性；报告未用于择优"))

    p = TREND / "参数邻域.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[TREND_PERIOD[r.period]]
            rows.append(mk("TREND", "TREND@S1_月周日共振突破",
                           {"param": r.param, "value": float(r.value)}, st, en, stage,
                           "sensitivity", ORIGIN, "done", n_units=1,
                           metric_basis="event_mean", artifact=str(p.relative_to(ROOT)),
                           notes=f"事件层 exc20={r.exc20:.4f}，单参数邻域，未做组合层"))

    p = TREND / "信号事件_汇总.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[TREND_PERIOD[r.period]]
            rows.append(mk("TREND", f"TREND@{r.signal}", {"view": "event20"},
                           st, en, stage, "diagnostic", ORIGIN, "done", n_units=1,
                           metric_basis="event_mean", artifact=str(p.relative_to(ROOT)),
                           notes=f"n={int(r.n)} exc20={r.exc20_mean:.4f}"))

    p = TREND / "消融_汇总.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[TREND_PERIOD[r.period]]
            rows.append(mk("TREND", f"TREND@ABL:{r.signal}", {"view": "event20"},
                           st, en, stage, "ablation", ORIGIN, "done", n_units=1,
                           metric_basis="event_mean", artifact=str(p.relative_to(ROOT)),
                           notes=f"n={int(r.n)} exc20={r.exc20_mean:.4f}"))


STRUCT_WIN = [
    ("全部2008-2020", "pooled", ("2008-01-01", "2020-12-31")),
    ("开发2008-2015", "dev", ("2008-01-01", "2015-12-31")),
    ("验证2016-2020", "val", ("2016-01-01", "2020-12-31")),
]


def inv_struct(rows):
    for k in (2, 3, 5):
        p = RES / f"结构族_事件研究_信号汇总_K{k}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            seg = dict((a, (c[0], c[1], b)) for a, b, c in STRUCT_WIN).get(r.seg)
            if seg is None:
                continue
            rows.append(mk("STRUCT", f"STRUCT@{r.sig}", {"K": k, "view": "event20"},
                           seg[0], seg[1], seg[2], "diagnostic", ORIGIN, "done",
                           metric_basis="event_mean", artifact=str(p.relative_to(ROOT)),
                           notes=f"n={int(r.n)} exc20={r.exc20:.4f}"))

    for k in (2, 3, 5):
        p = RES / f"结构族_离场对比_K{k}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", f"STRUCT@{r.sig}", {"K": k, "rule": r.rule},
                           "2008-01-01", "2020-12-31", "pooled", "diagnostic_trade",
                           ORIGIN, "done", n_trades=int(r.n), metric_basis="trade_mean",
                           artifact=str(p.relative_to(ROOT)),
                           notes=f"交易层 mean={r['mean']:.4f} 中位={r['median']:.4f}"))

    p = RES / "结构族_信号×离场_矩阵.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", f"STRUCT@{r.sig}", {"rule": r.rule},
                           "2008-01-01", "2020-12-31", "pooled", "experiment", ORIGIN,
                           "done", n_trades=int(r.n), cagr=r.CAGR, sharpe=None,
                           metric_basis="reported_table", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)),
                           notes=f"6 种子均值 CAGR={r.CAGR:.4f}, 种子 sd={r.sd:.4f}；"
                                 "无逐日净值，Sharpe 未汇报"))

    p = RES / "结构族_回测_分层_K3_hold40.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", f"STRUCT@{r.sig}", {"K": 3, "layer": r.layer},
                           "2008-01-01", "2020-12-31", "pooled", "experiment", ORIGIN,
                           "done", n_trades=iv(r.trades, 0), cagr=r.CAGR, mdd=r.MaxDD,
                           sharpe=r.Sharpe, vol=r["年化波动"], calmar=r.Calmar,
                           metric_basis="reported_table", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)),
                           notes=f"8 种子均值；CAGR sd={r.CAGR_sd:.4f}"))

    p = RES / "结构族_敏感性_K3_hold40.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", f"STRUCT@{r.sig}",
                           {"K": 3, "rule": "hold40", "capital": int(r.capital),
                            "slots": int(r.slots)},
                           "2008-01-01", "2020-12-31", "pooled", "sensitivity", ORIGIN,
                           "done", n_trades=int(r.trades), cagr=r.CAGR, mdd=r.MaxDD,
                           sharpe=r.Sharpe, vol=r["年化波动"], calmar=r.Calmar,
                           metric_basis="reported_table", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)), notes="资金 × 槽数敏感性"))

    p = RES / "结构族_成本敏感_K3_hold40.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", "STRUCT@ALL_SIGNALS",
                           {"K": 3, "rule": "hold40", "slippage": float(r.slippage),
                            "stamp": float(r.stamp)},
                           "2008-01-01", "2020-12-31", "pooled", "sensitivity", ORIGIN,
                           "done", n_trades=int(r.trades), cagr=r.CAGR, mdd=r.MaxDD,
                           sharpe=r.Sharpe, vol=r["年化波动"], calmar=r.Calmar,
                           metric_basis="reported_table", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)), notes="成本压力"))

    p = RES / "结构族_枢轴K对比.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", f"STRUCT@{r.sig}", {"K": int(r.k), "rule": r.rule},
                           "2008-01-01", "2020-12-31", "pooled", "experiment", ORIGIN,
                           "done", n_trades=int(r.n), cagr=r.CAGR,
                           metric_basis="reported_table", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)),
                           notes=f"K 对比；种子 sd={r.sd:.4f}"))

    p = RES / "结构族_离场规则_组合_K3.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STRUCT", "STRUCT@ALL_SIGNALS", {"K": 3, "rule": r.rule},
                           "2008-01-01", "2020-12-31", "pooled", "experiment", ORIGIN,
                           "done", n_trades=int(r.trades), cagr=r.CAGR, mdd=r.MaxDD,
                           sharpe=r.Sharpe, vol=r["年化波动"], calmar=r.Calmar,
                           metric_basis="reported_table", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)), notes="全信号合并，单路径"))

    # 逐信号 × 层级 × struct 离场的 32 条净值（早期探索产物，与 hold40 表并列）
    p = RES / "结构族_回测_净值_K3_struct.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        r_all = d.pct_change().dropna()
        n = len(r_all)
        for c in d.columns:
            sig, layer = c.split("|", 1)
            nav = d[c].to_numpy(float)
            x = r_all[c].to_numpy(float)
            cagr = float((nav[-1] / nav[0]) ** (ANN / n) - 1)
            mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
            sharpe = float(x.mean() / x.std(ddof=1) * math.sqrt(ANN))
            rows.append(mk("STRUCT", f"STRUCT@{sig}",
                           {"K": 3, "layer": layer, "rule": "struct"},
                           "2008-01-01", "2020-12-31", "pooled", "experiment", ORIGIN,
                           "done", n_trades=None, cagr=cagr, mdd=mdd, sharpe=sharpe,
                           vol=float(x.std(ddof=1) * math.sqrt(ANN)),
                           calmar=float(cagr / abs(mdd)) if mdd else None,
                           metric_basis="daily_nav", window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)),
                           notes="逐信号 struct 离场；无成交笔数，按净值反算"))


def inv_market(rows):
    p = RES / "市场状态" / "市场状态_分窗口绩效.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = MKT_PERIOD[r["窗口"]]
            bench = str(r["方案"]).startswith("基线")
            rows.append(mk("MKT", f"MKT@{r['方案']}", {"scheme": r["方案"]},
                           st, en, stage, "benchmark" if bench else "experiment",
                           ORIGIN, "done", n_units=1, cagr=r.cagr, mdd=r.mdd,
                           sharpe=r.sharpe, vol=r.vol, calmar=r.calmar,
                           metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes="策略表；基准行不计入候选"))

    p = RES / "市场状态" / "市场状态_参数敏感性.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = MKT_PERIOD[r["窗口"]]
            rows.append(mk("MKT", f"MKT@{r['方案']}", {"scheme": r["方案"]},
                           st, en, stage, "sensitivity", ORIGIN, "done", n_units=1,
                           cagr=r.cagr, mdd=r.mdd, sharpe=r.sharpe, calmar=r.calmar,
                           metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes="市场状态参数敏感性"))

    aux = ["市场状态_崩盘依赖.csv", "市场状态_平移检验.csv", "市场状态_风险可预测性.csv",
           "市场状态_新鲜度条件收益.csv", "市场状态_空仓门诊断.csv",
           "市场状态_静态等仓对照.csv", "市场状态_状态条件收益.csv",
           "市场状态_驻留区间.csv"]
    have = [f for f in aux if (RES / "市场状态" / f).exists()]
    if have:
        rows.append(mk("MKT", "MKT@diagnostics", {"tables": len(have)},
                       "2006-01-04", "2020-12-31", "pooled", "diagnostic", ORIGIN,
                       "done", n_units=len(have), metric_basis="diagnostic",
                       artifact="03_结果/市场状态/",
                       notes="状态层辅助分析：" + "|".join(have)))


EXIT_TB = {"TB1": "TREND突破入场", "TB2": "日历控制入场"}


def inv_exit(rows):
    p = RES / "退出_臂对照_组合级.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[r.window]
            rows.append(mk("EXIT", f"EXIT@TB1/{r.arm}", {"testbed": "TB1", "arm": r.arm},
                           st, en, stage, "experiment", ORIGIN, "done",
                           n_trades=int(r.n_trades), cagr=r.cagr, mdd=r.mdd,
                           sharpe=r.sharpe, vol=r.vol, calmar=r.calmar,
                           metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes="18 臂 × 2 窗口；组合级仅 TB1"))

    for tb in ("TB1", "TB2"):
        p = RES / f"退出_臂对照_交易级_{tb}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[r.window]
            rows.append(mk("EXIT", f"EXIT@{tb}/{r.arm}",
                           {"testbed": tb, "arm": r.arm, "view": "trade"},
                           st, en, stage, "diagnostic_trade", ORIGIN, "done",
                           n_trades=int(r.get("n_trades", r.get("n", 0)) or 0),
                           metric_basis="trade_mean",
                           artifact=str(p.relative_to(ROOT)),
                           notes="交易层平均 R；无组合净值"))

    p = RES / "退出_仓位敏感.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            st, en, stage = PERIOD_WIN[r.window]
            params = {"study": r.study, "arm": r.arm, "sizing": r.sizing,
                      "risk_pct": None if pd.isna(r.risk_pct) else float(r.risk_pct),
                      "slots": int(r.slots),
                      "capital": None if pd.isna(r.capital) else float(r.capital)}
            rows.append(mk("EXIT", f"EXIT@TB1/{r.arm}", params, st, en, stage,
                           "sensitivity", ORIGIN, "done", n_trades=int(r.n_trades),
                           cagr=r.cagr, mdd=r.mdd, sharpe=r.sharpe, calmar=r.calmar,
                           metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes=f"{r.study}；仓位口径敏感性"))


def inv_ecology(rows):
    """后续并行族（ECOLOGY-REV-BRK）：自带预注册与试验清单，未写中央注册表。"""
    d0 = RES / "候选_生态切换"
    p = d0 / "11_分窗口绩效.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    for _, r in d.iterrows():
        st, en, stage = PERIOD_WIN[r.window]
        rows.append(mk("ECOLOGY", f"ECOLOGY@{r.config}", {"config": r.config},
                       st, en, stage, "experiment", "preregistered_local", "done",
                       n_trades=None,
                       cagr=r.cagr, mdd=-abs(r.mdd), sharpe=r.sharpe, vol=r.vol,
                       calmar=r.calmar, metric_basis="reported_table",
                       artifact=str(p.relative_to(ROOT)),
                       notes="生态切换族（并行）；本地预注册，无中央 trial_id"))

    p = d0 / "05_组合绩效.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("ECOLOGY", f"ECOLOGY@{r.config}", {"config": r.config},
                           "2008-01-01", "2020-12-31", "pooled", "experiment",
                           "preregistered_local", "done", n_trades=int(r.n_trades),
                           cagr=r.cagr, mdd=-abs(r.mdd), sharpe=r.sharpe, vol=r.vol,
                           calmar=r.calmar, metric_basis="reported_table",
                           window_mix="pooled_dev_val",
                           artifact=str(p.relative_to(ROOT)),
                           notes="全段 2008-2020；分窗口见 11_分窗口绩效.csv"))

    p = d0 / "03_事件条件边际.csv"
    if p.exists():
        d = pd.read_csv(p)
        rows.append(mk("ECOLOGY", "ECOLOGY@event_conditional", {"view": "event"},
                       "2008-01-01", "2020-12-31", "pooled", "diagnostic",
                       "preregistered_local", "done", n_units=max(len(d), 1),
                       metric_basis="event_mean", artifact=str(p.relative_to(ROOT)),
                       notes=f"事件条件格 {len(d)} 个，族自报 N 上界 32"))


def inv_stat(rows):
    p = RES / "统计验证" / "候选日收益矩阵.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        r = d.dropna()
        n = len(r)
        for c in d.columns:
            x = r[c].to_numpy()
            nav = (1 + x).cumprod()
            cagr = nav[-1] ** (ANN / n) - 1
            sharpe = x.mean() / x.std(ddof=1) * math.sqrt(ANN)
            mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
            rows.append(mk("STAT", f"STAT@{c}", {"candidate": c},
                           "2008-01-02", "2012-12-31", "dev_subwindow", "experiment",
                           "post_hoc_unregistered", "done", n_units=1,
                           cagr=float(cagr), mdd=mdd, sharpe=float(sharpe),
                           vol=float(x.std(ddof=1) * math.sqrt(ANN)),
                           calmar=float(cagr / abs(mdd)) if mdd else None,
                           metric_basis="daily_nav",
                           artifact=str(p.relative_to(ROOT)),
                           notes="统计验证框架的 9 个 regime 网格候选（参考夹具）"))

    p = RES / "统计验证" / "walkforward_逐折.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append(mk("STAT", "STAT@walkforward", {"fold": int(r.fold)},
                           str(r.test_start), str(r.test_end), "dev", "diagnostic",
                           ORIGIN, "done", n_units=1, cagr=r.test_cagr,
                           metric_basis="reported_table",
                           artifact=str(p.relative_to(ROOT)),
                           notes=f"train {r.train_start}~{r.train_end}"))


# ---------------------------------------------------------------- 缺陷与定级

def defect_table():
    """冻结层里 Baostock hfq 与官方 pct_chg 背离 >0.5pp 的证券日。"""
    import duckdb
    con = duckdb.connect(str(ROOT / "01_数据/frozen/frozen.duckdb"), read_only=True)
    q = """
    with d as (
      select symbol, date, pct_chg, bs_hfq_close,
             lag(bs_hfq_close) over (partition by symbol order by date) as prev_hfq
      from daily where date between date '2005-01-01' and date '2020-12-31'
    )
    select symbol as sym, date, pct_chg,
           (bs_hfq_close/prev_hfq-1)*100 - pct_chg as gap
    from d
    where prev_hfq is not null
      and abs((bs_hfq_close/prev_hfq-1)*100 - pct_chg) > 0.5
    """
    df = con.execute(q).fetchdf()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _sym(code: str) -> str:
    code = str(code)
    return code[3:] + "." + code[:2].upper()


def contamination(defects):
    """按族统计受影响成交/事件比例与 NAV 扭曲上界。"""
    out = []
    dts = defects[["sym", "date", "gap"]].rename(columns={"date": "def_date"})

    p = TREND / "交易明细_全部.parquet"
    if p.exists():
        t = pd.read_parquet(p)
        t["sym"] = t.code.map(_sym)
        t["entry_date"] = pd.to_datetime(t.entry_date)
        t["exit_date"] = pd.to_datetime(t.exit_date)
        m = t.reset_index().merge(dts, on="sym", how="inner")
        m = m[(m.def_date >= m.entry_date) & (m.def_date <= m.exit_date)]
        hit = m["index"].nunique()
        gap_sum = float(m.groupby("index").gap.apply(lambda s: s.abs().max()).sum()) \
            if hit else 0.0
        out.append(dict(family="TREND", unit="trade", total=len(t), affected=int(hit),
                        ratio=100 * hit / max(len(t), 1),
                        nav_pp=gap_sum * 0.10,
                        artifact=str(p.relative_to(ROOT))))

    p = TREND / "信号事件_明细.parquet"
    if p.exists():
        e = pd.read_parquet(p)
        e["sym"] = e.code.map(_sym)
        e["d0"] = pd.to_datetime(e["date"])
        m = e.reset_index().merge(dts, on="sym", how="inner")
        m = m[(m.def_date > m.d0) & (m.def_date <= m.d0 + pd.Timedelta(days=84))]
        hit = m["index"].nunique()
        out.append(dict(family="TREND-event", unit="event(60d)", total=len(e),
                        affected=int(hit), ratio=100 * hit / max(len(e), 1), nav_pp=0.0,
                        artifact=str(p.relative_to(ROOT))))

    for arm in ("B0", "T3", "TG2", "ALL"):
        p = RES / f"退出_明细_TB1_{arm}.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        t["sym"] = t.code.map(_sym)
        t["entry_date"] = pd.to_datetime(t.entry_date)
        t["exit_date"] = pd.to_datetime(t.exit_date)
        m = t.reset_index().merge(dts, on="sym", how="inner")
        m = m[(m.def_date >= m.entry_date) & (m.def_date <= m.exit_date)]
        hit = m["index"].nunique()
        out.append(dict(family=f"EXIT-TB1/{arm}", unit="trade", total=len(t),
                        affected=int(hit), ratio=100 * hit / max(len(t), 1),
                        nav_pp=0.0, artifact=str(p.relative_to(ROOT))))

    for k in (2, 3, 5):
        p = RES / f"结构族_事件_K{k}.parquet"
        if not p.exists():
            continue
        e = pd.read_parquet(p)
        col = "code" if "code" in e.columns else e.columns[0]
        dcol = "date" if "date" in e.columns else None
        if dcol is None:
            continue
        e["sym"] = e[col].map(_sym)
        e["d0"] = pd.to_datetime(e[dcol])
        m = e.reset_index().merge(dts, on="sym", how="inner")
        m = m[(m.def_date > m.d0) & (m.def_date <= m.d0 + pd.Timedelta(days=84))]
        hit = m["index"].nunique()
        out.append(dict(family=f"STRUCT-K{k}-event", unit="event(60d)", total=len(e),
                        affected=int(hit), ratio=100 * hit / max(len(e), 1),
                        nav_pp=0.0, artifact=str(p.relative_to(ROOT))))
    return pd.DataFrame(out)


def grade_rows(rows, contam_df):
    contam = {}
    if not contam_df.empty:
        for fam, g in contam_df.groupby("family"):
            contam[fam.split("-")[0]] = float(g.ratio.max())

    for r in rows:
        gates = []
        fam = r["family"]
        c = contam.get(fam, 0.0)
        if r["status"] != "done" or (r["cagr"] is None
                                     and r["metric_basis"] in ("none", "coverage")):
            r["grade"] = "F_unevaluable"
            r["fail_gate"] = "F_no_result"
            continue
        if r["kind"] in ("diagnostic", "diagnostic_trade", "ablation", "audit"):
            r["grade"] = "DIAG"
            r["fail_gate"] = ""
            continue
        if r["kind"] == "benchmark":
            r["grade"] = "BENCH"
            r["fail_gate"] = ""
            continue

        if r["window_mix"] != "single":
            gates.append("W_mixed_window")
        if r["stage"] == "preheat":
            gates.append("X_preheat_not_evidence")
        if r["metric_basis"] == "reported_table" and c > 1.0:
            gates.append("D_fail_contaminated")
        if r["n_trades"] is not None and r["n_trades"] < 30:
            gates.append("P_fail_underpowered")
        elif r["n_trades"] is None:
            gates.append("P_unknown")
        if r["cagr"] is not None and r["metric_basis"] in ("reported_table", "daily_nav"):
            if not (r["cagr"] > 0):
                gates.append("G_fail_negative")
            if r["mdd"] is not None and r["mdd"] < -0.25:
                gates.append("G_fail_dd")
        if r["origin"] == "preregistered_local":
            gates.append("R_fail_no_central_trial_id")
        elif r["origin"] != "preregistered":
            gates.append("R_fail_unregistered")

        if "P_fail_underpowered" in gates:
            grade = "E_underpowered"
        elif "D_fail_contaminated" in gates:
            grade = "D_contaminated"
        elif (not any(x.startswith("G_fail") for x in gates)
              and "W_mixed_window" not in gates and "P_unknown" not in gates
              and "X_preheat_not_evidence" not in gates):
            grade = "B_conditional"
        else:
            grade = "C_negative_valid"
        r["grade"] = grade
        r["fail_gate"] = ";".join(gates)


def assign_counts(rows):
    """每个 config 只在一个行上计入 N_sel（优先 dev），每个组合级 cell 计入 N_cell。

    只有产出组合层数值（reported_table / daily_nav）的配置才算“可被选择的试验”；
    事件层敏感性（event_mean）计入 N_diag——它们不能作为可部署策略被选中。
    """
    order = {"dev": 0, "dev_subwindow": 1, "pooled": 2, "val": 3, "preheat": 4,
             "audit": 5, "na": 6}
    idx = sorted(range(len(rows)),
                 key=lambda i: (rows[i]["config_id"], order.get(rows[i]["stage"], 9), i))
    seen = set()
    for i in idx:
        r = rows[i]
        portfolio = r["metric_basis"] in ("reported_table", "daily_nav")
        if r["kind"] in ("experiment", "sensitivity") and r["status"] == "done" and portfolio:
            r["in_Ncell"] = 1
            if r["config_id"] not in seen:
                seen.add(r["config_id"])
                r["in_Nsel"] = 1
        elif r["kind"] in ("diagnostic", "diagnostic_trade", "ablation", "audit") \
                or (r["kind"] == "sensitivity" and r["metric_basis"] == "event_mean"):
            r["in_Ndiag"] = 1
    for i, r in enumerate(rows, start=1):
        r["ledger_id"] = f"L{i:05d}"


# ---------------------------------------------------------------- 总账读写

def write_ledger(rows, rebuild=True):
    anchor = registry_anchor()
    meta = {
        "ledger_id": "L00000", "source": "meta", "family": "META",
        "strategy_id": "INTEGRATION_PREREG", "params_id": pid({"p": str(IPREREG.name)}),
        "config_id": "meta", "window_start": "2006-01-01", "window_end": str(FREEZE),
        "stage": "na", "kind": "meta", "origin": "preregistered", "status": "done",
        "n_units": 0, "in_Nsel": 0, "in_Ncell": 0, "in_Ndiag": 0,
        "n_trades": None, "cagr": None, "mdd": None, "sharpe": None, "vol": None,
        "calmar": None, "metric_basis": "none", "window_mix": "na",
        "artifact": str(IPREREG.relative_to(ROOT)), "artifact_extra": "",
        "grade": "META", "fail_gate": "",
        "notes": json.dumps({
            "prereg_sha256": sha256_file(IPREREG) if IPREREG.exists() else "",
            "official_registry_head": anchor,
            "official_registry_rows": _registry_nrows(),
            "seed": SEED,
        }, ensure_ascii=False, sort_keys=True),
        "prev_sha256": anchor, "row_sha256": "",
    }
    meta["row_sha256"] = row_hash(meta)
    all_rows = [meta] + rows
    delta = 0
    prev = meta["row_sha256"]
    for r in all_rows[1:]:
        r["ledger_id"] = f"L{len([1])+delta:05d}"
        delta += 1
        r["prev_sha256"] = prev
        r["row_sha256"] = row_hash(r)
        prev = r["row_sha256"]

    path = LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and path.exists():
        bak = path.with_suffix(".csv.bak")
        shutil.copy2(path, bak)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(LEDGER_COLUMNS)
        for r in all_rows:
            w.writerow([fmt(r[c]) for c in LEDGER_COLUMNS])
    tmp.replace(path)
    return all_rows


def _registry_nrows():
    if not REGISTRY.exists():
        return 0
    with REGISTRY.open("r", encoding="utf-8", newline="") as fh:
        return max(0, len(list(csv.DictReader(fh))))


def read_ledger(path=LEDGER):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def verify_ledger(path=LEDGER, verbose=True):
    rows = read_ledger(path)
    errs, warns = [], []
    prev = rows[0]["prev_sha256"] if rows else ZERO
    anchor = registry_anchor()
    if rows and rows[0]["prev_sha256"] != anchor:
        errs.append(f"首行锚点 {rows[0]['prev_sha256'][:12]} != 官方链尾 {anchor[:12]}")
    for i, r in enumerate(rows, 1):
        if r["prev_sha256"] != prev:
            errs.append(f"第{i}行 prev 断链")
        if row_hash(r) != r["row_sha256"]:
            errs.append(f"第{i}行({r['ledger_id']}) 内容被改")
        prev = r["row_sha256"]
    nsel = sum(int(r["n_units"]) for r in rows if r["in_Nsel"] == "1")
    ncell = sum(int(r["n_units"]) for r in rows if r["in_Ncell"] == "1")
    ndiag = sum(int(r["n_units"]) for r in rows if r["in_Ndiag"] == "1")
    if verbose:
        print(f"总账行数：{len(rows)}  N_sel={nsel}  N_cell={ncell}  N_diag={ndiag}")
        for w in warns:
            print("WARN", w)
        for e in errs:
            print("FAIL", e)
        print("结论：" + ("OK" if not errs else f"FAIL({len(errs)})"))
    return errs, (nsel, ncell, ndiag)


# ---------------------------------------------------------------- 官方链审查

def audit_registry():
    errs, notes = [], []
    if not REGISTRY.exists():
        return ["官方注册表不存在"], []
    with REGISTRY.open("r", encoding="utf-8", newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        raw = [r for r in rdr if r]
    from importlib import util
    spec = util.spec_from_file_location("regtool", ROOT / "02_代码/实验登记.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if header != mod.COLUMNS:
        errs.append("表头与 实验登记.py 的 COLUMNS 不一致")
    rows = [dict(zip(mod.COLUMNS, r)) for r in raw]
    prev = mod.ZERO_HASH
    for i, r in enumerate(rows, 1):
        if r["prev_sha256"] != prev:
            errs.append(f"第{i}行 prev 断链")
        if mod.row_hash(r) != r["row_sha256"]:
            errs.append(f"第{i}行 {r['exp_id']} row_sha256 不匹配")
        prev = r["row_sha256"]
    # 代码漂移
    drift = []
    for r in rows:
        if r["code_path"]:
            p = ROOT / r["code_path"]
            if p.is_file() and r["code_sha256"] and sha256_file(p) != r["code_sha256"]:
                drift.append((r["exp_id"], r["trial_id"], r["kind"], r["code_path"]))
    # 窗口
    win_bad = []
    for r in rows:
        if r["stage"] in ("dev", "val", "preheat") and r["window_end"] >= "2021-01-01":
            win_bad.append(r["exp_id"])
    n = max(mod.prefix_cumulative(rows).values()) if rows else 0
    planned = [r["trial_id"] for r in rows if r["status"] in ("planned", "blocked")]
    notes.append(f"官方链独立复算：{len(rows)} 行，N={n}，断链 {len(errs)} 处。")
    notes.append(f"代码漂移 {len(drift)} 处：{drift}")
    notes.append(f"窗口越界 {len(win_bad)} 处：{win_bad}")
    notes.append(f"未运行试验：{planned}")
    return errs, notes


def write_registry_audit():
    INT.mkdir(parents=True, exist_ok=True)
    errs, notes = audit_registry()
    led = read_ledger() if LEDGER.exists() else []
    nsel = sum(int(r["n_units"]) for r in led if r["in_Nsel"] == "1")
    ncell = sum(int(r["n_units"]) for r in led if r["in_Ncell"] == "1")
    ndiag = sum(int(r["n_units"]) for r in led if r["in_Ndiag"] == "1")
    lines = ["# 登记审计：官方哈希链与补登记缺口", "",
             f"- 生成：`整合实验登记.py audit-registry`（种子 {SEED}）",
             f"- 官方注册表：`03_结果/实验注册表.csv`",
             f"- 补登记总账：`03_结果/试验总账.csv`", "", "## 1. 官方链复核", ""]
    for e in errs:
        lines.append(f"- FAIL {e}")
    for n in notes:
        lines.append(f"- {n}")
    lines += ["", "## 2. 登记缺口", "",
              "| 项 | 官方注册表 | 本总账补登记 |",
              "|---|---:|---:|",
              f"| 行数 | {_registry_nrows()} | {len(led)} |",
              f"| N（选择口径） | 0 | {nsel} |",
              f"| N_cell（含窗口复跑） | 0 | {ncell} |",
              f"| N_diag（事件/消融/诊断） | 0 | {ndiag} |",
              "",
              "官方链的 `trials_cumulative` 只统计已登记行，因此 N=0；"
              "实际评估数由本总账按 artifact 反推。两者相差两个数量级，"
              "这不是算术错误，而是**先跑后登记**的直接后果。", "",
              "## 3. 哈希链机制审查", "",
              "机制本身按设计工作，但有五处结构性弱点（不改代码，只记录）：", "",
              "1. **无外部锚**：链只保证内部一致，任何人可重写全表并重算链。"
              "`anchor` 只锚定预注册文件，不锚定注册表链尾。"
              "本总账把官方链尾写进首行 `prev_sha256`，也只是把两个内部链拼起来，"
              "仍不能证明未被整体重写——除非把链尾发布到项目外。",
              "2. **`n_configs` 自报**：所谓“防少报”只是拒绝手填 `trials_cumulative`，"
              "`n_configs` 仍由登记者填。本次缺口就是证据：填 0 就真的记 0。",
              "3. **`audit` 行豁免代码漂移**：`verify_rows` 对 `kind=audit` 的代码漂移只告警。"
              "E00002 记录了 20 项 PASS，但它的 `code_sha256` 与当前 `实验登记.py` 不一致，"
              "`verify --strict` 仍返回 OK。",
              "4. **`meta` 行不绑定代码**：锚点行 `code_path` 为空，预注册文本与工具版本无关联。",
              "5. **无“实际运行”对账**：工具不知道磁盘上有多少结果文件，"
              "无法发现“跑了没登记”。", "",
              "## 4. 结论与建议", "",
              f"- 官方链**未被篡改**（断链 0），但它**不完整**：N=0 对 N={nsel}。",
              "- 按预注册 §12.4，“未登记就运行并事后补登记”本身构成研究失败的一部分，"
              "必须在报告首页披露。",
              "- 建议（不代做）：方法负责人用 `整合实验登记.py emit-registry` 生成的命令，"
              "把本总账按 family×window 聚合成官方 `append` 行；官方链将自动累加 `n_configs`。",
              ]
    (INT / "登记审计.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return errs


def emit_registry():
    led = read_ledger()
    out = []
    for fam, g in pd.DataFrame(led).groupby("family"):
        g = g[g["kind"].isin(["experiment", "sensitivity", "ablation"])]
        for stage, gg in g.groupby("stage"):
            if stage not in ("dev", "val", "preheat", "pooled"):
                continue
            n = len(gg)
            if n == 0:
                continue
            w0, w1 = gg["window_start"].mode().iloc[0], gg["window_end"].mode().iloc[0]
            if stage == "pooled":
                w0, w1, stage = "2008-01-01", "2020-12-31", "dev"
                note = "[RETRO-REGISTERED][POST-HOC] 2008-2020 合并窗口无法按 dev/val 复核"
            else:
                note = "[RETRO-REGISTERED] 第一阶段事后补登记；运行发生在登记之前"
            metrics = {"n_trades": int(gg["n_trades"].fillna(0).astype(float).sum()),
                       "cagr": float(pd.to_numeric(gg["cagr"], errors="coerce").min()),
                       "mdd": float(pd.to_numeric(gg["mdd"], errors="coerce").min())}
            out.append(
                "python3 02_代码/实验登记.py append --kind experiment "
                f"--stage {stage} --status done "
                f"--hypothesis \"{fam} 族 {n} 个配置的事后补登记（最差 CAGR 作为汇总指标）\" "
                f"--family {fam} --window-start {w0} --window-end {w1} "
                "--universe U-MAINBOARD-V1 --cost-model C-REAL-V1 "
                f"--params-json '{json.dumps({'family': fam, 'stage': stage, 'n': n}, ensure_ascii=False)}' "
                f"--n-configs {n} --code 02_代码/整合实验登记.py "
                f"--metrics-json '{json.dumps(metrics)}' --decision discard "
                f"--ack-budget --budget-reason \"实际评估数超过 64 硬预算\" "
                f"--notes \"{note}\"")
    print("\n".join(out))
    return out

# ---------------------------------------------------------------- 统计推断

def stationary_indices(n: int, L: int = BLOCK, B: int = BOOT, seed: int = SEED):
    """stationary bootstrap 索引（Politis & Romano）。"""
    rng = np.random.default_rng(seed)
    idx = np.empty((B, n), dtype=np.int32)
    for b in range(B):
        pos = 0
        while pos < n:
            start = int(rng.integers(0, n))
            ln = int(rng.geometric(1.0 / L))
            ln = min(ln, n - pos)
            idx[b, pos:pos + ln] = (start + np.arange(ln)) % n
            pos += ln
    return idx


def boot_all(R: np.ndarray, idx: np.ndarray, ann: float = ANN):
    B, n = idx.shape
    C = R.shape[1]
    cagr = np.empty((B, C))
    sh = np.empty((B, C))
    mdd = np.empty((B, C))
    for b in range(B):
        X = R[idx[b]]
        cagr[b] = np.expm1(np.log1p(X).mean(0) * ann)
        mu = X.mean(0)
        sd = X.std(0, ddof=1)
        sh[b] = np.where(sd > 0, mu / sd * math.sqrt(ann), np.nan)
        nav = np.cumprod(1.0 + X, axis=0)
        mdd[b] = (nav / np.maximum.accumulate(nav, axis=0) - 1.0).min(0)
    return cagr, sh, mdd


def point_stats(R: np.ndarray, ann: float = ANN):
    n = len(R)
    cagr = np.expm1(np.log1p(R).mean(0) * ann)
    mu = R.mean(0)
    sd = R.std(0, ddof=1)
    sh = np.where(sd > 0, mu / sd * math.sqrt(ann), np.nan)
    nav = np.cumprod(1 + R, axis=0)
    mdd = (nav / np.maximum.accumulate(nav, axis=0) - 1).min(0)
    return cagr, sh, mdd


def maxstat(R: np.ndarray, idx: np.ndarray):
    """White Reality Check：max_c mean(r_c) 的 bootstrap p 值 + 逐候选 p。

    零假设 mean_c = 0 必须由**中心化后的重采样分布**施加，否则 p 值无意义。
    """
    obs = R.mean(0)
    stat = float(obs.max())
    Rc = R - obs
    boot = np.empty((len(idx), R.shape[1]))
    for b in range(len(idx)):
        boot[b] = Rc[idx[b]].mean(0)
    pmax = float((boot.max(1) >= stat).mean())
    p_each = (boot >= obs).mean(0)
    return stat, pmax, p_each


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def pbo_cscv(R: np.ndarray, S: int = PBO_S):
    """CSCV：S 个连续块，枚举 C(S,S/2)，返回 PBO 与 logit 分布。"""
    n, C = R.shape
    if C < 5 or n < S * 5:
        return None
    bnd = np.linspace(0, n, S + 1).astype(int)
    bsum = np.array([R[bnd[i]:bnd[i + 1]].sum(0) for i in range(S)])
    bsq = np.array([(R[bnd[i]:bnd[i + 1]] ** 2).sum(0) for i in range(S)])
    bcnt = np.array([bnd[i + 1] - bnd[i] for i in range(S)], float)
    ranks = []
    for comb in itertools.combinations(range(S), PBO_B):
        oos = [i for i in range(S) if i not in comb]
        s1 = bsum[list(comb)].sum(0); q1 = bsq[list(comb)].sum(0); c1 = bcnt[list(comb)].sum()
        s2 = bsum[oos].sum(0); q2 = bsq[oos].sum(0); c2 = bcnt[oos].sum()
        v1 = q1 / c1 - (s1 / c1) ** 2
        v2 = q2 / c2 - (s2 / c2) ** 2
        ok = (v1 > 0) & (v2 > 0)
        if ok.sum() < 2:
            continue
        sr1 = np.where(ok, (s1 / c1) / np.sqrt(np.where(ok, v1, 1)), -np.inf)
        sr2 = np.where(ok, (s2 / c2) / np.sqrt(np.where(ok, v2, 1)), -np.inf)
        best = int(np.argmax(sr1))
        if not ok[best]:
            continue
        rank = (sr2[ok] < sr2[best]).sum() / max(ok.sum() - 1, 1)
        ranks.append(rank)
    if not ranks:
        return None
    ranks = np.array(ranks)
    # rank 为"比选中的更差的候选占比"：1=OOS 最好，0=OOS 最差
    # PBO 定义：选中的策略在 OOS 落到中位数以下的比例
    pbo = float((ranks < 0.5).mean())
    eps = 1e-6
    logit = np.log(np.clip(ranks, eps, 1 - eps) / (1 - np.clip(ranks, eps, 1 - eps)))
    return dict(pbo=pbo, n_comb=len(ranks), median_rank=float(np.median(ranks)),
                logit_mean=float(logit.mean()), logit_std=float(logit.std()))


def dsr(sr_daily, T, N, var_sr, skew, kurt):
    from scipy import stats as st
    g = 0.5772156649
    if N < 2 or not np.isfinite(var_sr) or var_sr <= 0:
        var_sr = 0.25
    sr0 = math.sqrt(var_sr) * ((1 - g) * st.norm.ppf(1 - 1.0 / N)
                               + g * st.norm.ppf(1 - 1.0 / (N * math.e)))
    denom = 1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2
    denom = max(denom, 1e-9)
    z = (sr_daily - sr0) * math.sqrt(max(T - 1, 1)) / math.sqrt(denom)
    return dict(dsr=float(st.norm.cdf(z)), sr_daily=float(sr_daily),
                sr0_daily=float(sr0), sr0_annual=float(sr0 * math.sqrt(ANN)),
                n_trials=int(N), var_sr=float(var_sr), z=float(z))


def load_daily_returns():
    """返回 {matrix_name: (DataFrame date×config_id of returns, meta)}。"""
    mats = {}

    # 趋势族：84 个 parquet 净值
    for tag, (dirp, pre, suf) in PANELS.items():
        cols = {}
        dates = None
        for f in sorted(dirp.glob(pre + "*" + suf)):
            d = pd.read_parquet(f)
            d["date"] = pd.to_datetime(d["date"])
            key = cid("TREND", f"TREND@{d['signal'].iloc[0]}", {"rule": d["rule"].iloc[0]})
            cols[key] = d.set_index("date")["equity"].astype(float)
        if cols:
            df = pd.DataFrame(cols).sort_index()
            r = df.pct_change().dropna(how="all").fillna(0.0)
            mats[tag] = (r, "trend")

    # 退出族：18 臂 × 2 窗口
    for win in ("dev", "val"):
        cols = {}
        for f in sorted(RES.glob(f"退出_净值_TB1_*_{win}.csv")):
            arm = f.name.replace("退出_净值_TB1_", "").replace(f"_{win}.csv", "")
            d = pd.read_csv(f)
            d.columns = ["date", "nav"]
            d["date"] = pd.to_datetime(d["date"])
            key = cid("EXIT", f"EXIT@TB1/{arm}", {"testbed": "TB1", "arm": arm})
            cols[key] = d.set_index("date")["nav"].astype(float)
        if cols:
            df = pd.DataFrame(cols).sort_index()
            r = df.pct_change().dropna(how="all").fillna(0.0)
            mats[f"exit_{win}"] = (r, "exit")

    # 结构族：32 配置的净值矩阵（2008-2020 全段，hold40）
    p = RES / "结构族_回测_净值_K3_hold40.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        cols = {}
        for c in d.columns:
            sig, layer = c.split("|", 1)
            cols[cid("STRUCT", f"STRUCT@{sig}", {"K": 3, "layer": layer})] = d[c].astype(float)
        df = pd.DataFrame(cols).sort_index()
        mats["struct_pooled"] = (df.pct_change().dropna(how="all").fillna(0.0), "struct")

    # 结构族：逐信号 struct 离场 32 条净值
    p = RES / "结构族_回测_净值_K3_struct.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        cols = {}
        for c in d.columns:
            sig, layer = c.split("|", 1)
            cols[cid("STRUCT", f"STRUCT@{sig}",
                     {"K": 3, "layer": layer, "rule": "struct"})] = d[c].astype(float)
        df = pd.DataFrame(cols).sort_index()
        mats["struct_struct"] = (df.pct_change().dropna(how="all").fillna(0.0), "struct")

    # 结构族：11 条全信号聚合离场规则净值
    p = RES / "结构族_离场规则_净值_K3.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        cols = {cid("STRUCT", "STRUCT@ALL_SIGNALS", {"K": 3, "rule": c}): d[c].astype(float)
                for c in d.columns}
        df = pd.DataFrame(cols).sort_index()
        mats["struct_rules"] = (df.pct_change().dropna(how="all").fillna(0.0), "struct")

    # 统计验证族：9 候选日收益矩阵
    p = RES / "统计验证" / "候选日收益矩阵.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        cols = {cid("STAT", f"STAT@{c}", {"candidate": c}): d[c].astype(float)
                for c in d.columns}
        mats["stat_dev"] = (pd.DataFrame(cols).dropna(), "stat")

    p = RES / "候选_生态切换" / "06_组合净值.csv"
    if p.exists():
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        cols = {cid("ECOLOGY", f"ECOLOGY@{c}", {"config": c}): d[c].astype(float)
                for c in d.columns}
        mats["ecol_full"] = (pd.DataFrame(cols).pct_change().dropna(), "ecology")
    return mats


def cmd_stats(args):
    INT.mkdir(parents=True, exist_ok=True)
    led = pd.DataFrame(read_ledger())
    mats = load_daily_returns()
    ci_rows, multi = [], {"seed": SEED, "block": BLOCK, "boot": args.boot,
                          "definition": "exp(243*mean(log1p(r)))-1; stationary bootstrap",
                          "matrices": {}}

    for name, (r, fam) in mats.items():
        r = r.dropna(axis=1, how="all")
        keep = [c for c in r.columns if c in set(led["config_id"])]
        if len(keep) < 2:
            continue
        R = r[keep].to_numpy(float)
        n, C = R.shape
        idx = stationary_indices(n, BLOCK, args.boot, SEED)
        cagr_b, sh_b, mdd_b = boot_all(R, idx)
        cagr_p, sh_p, mdd_p = point_stats(R)
        for j, c in enumerate(keep):
            ci_rows.append(dict(
                matrix=name, config_id=c, n_days=n,
                cagr_point=cagr_p[j],
                cagr_p5=np.nanpercentile(cagr_b[:, j], 5),
                cagr_p50=np.nanpercentile(cagr_b[:, j], 50),
                cagr_p95=np.nanpercentile(cagr_b[:, j], 95),
                sharpe_point=sh_p[j],
                sharpe_p5=np.nanpercentile(sh_b[:, j], 5),
                sharpe_p50=np.nanpercentile(sh_b[:, j], 50),
                sharpe_p95=np.nanpercentile(sh_b[:, j], 95),
                mdd_point=mdd_p[j],
                mdd_p5=np.nanpercentile(mdd_b[:, j], 5),
                mdd_p95=np.nanpercentile(mdd_b[:, j], 95),
                ci_method="stationary_bootstrap",
            ))

        stat, pmax, p_each = maxstat(R, idx)
        best = int(np.argmax(cagr_p))
        x = R[:, best]
        pbo = pbo_cscv(R)
        var_sr = float(np.nanvar(sh_p / math.sqrt(ANN), ddof=1))
        d = dsr(sh_p[best] / math.sqrt(ANN), n, C, var_sr,
                float(pd.Series(x).skew()), float(pd.Series(x).kurt() + 3))
        # 5% 分位年化的 max-stat 版本
        multi["matrices"][name] = dict(
            family=fam, n_candidates=C, n_days=n,
            best_config=keep[best], best_cagr=float(cagr_p[best]),
            best_cagr_p5=float(np.nanpercentile(cagr_b[:, best], 5)),
            p_maxstat=pmax, stat_max_mean_daily=stat,
            p_each_holm=dict(zip(keep, map(float, holm(p_each)))),
            pbo=pbo, dsr_family=d,
            var_sr_daily=var_sr,
        )

    ci = pd.DataFrame(ci_rows)
    ci.to_csv(INT / "统一置信区间.csv", index=False, encoding="utf-8")

    # 项目级 DSR：N = N_sel；Var(SR) 用全部有 Sharpe 的组合级配置
    port = led[(led["kind"].isin(["experiment", "sensitivity"]))
               & (led["sharpe"].notna()) & (led["sharpe"] != "")]
    port = port.drop_duplicates("config_id")
    sr_ann = pd.to_numeric(port["sharpe"], errors="coerce").dropna().to_numpy()
    var_project = float(np.var(sr_ann / math.sqrt(ANN), ddof=1))
    nsel = int(sum(int(x) for x in led[led["in_Nsel"] == "1"]["n_units"]))
    ncell = int(sum(int(x) for x in led[led["in_Ncell"] == "1"]["n_units"]))

    # 项目级"最优开发配置"：只在有日收益序列的配置里选（DSR 需要 SR/T/skew/kurt）
    all_cols = set()
    for name, (r, fam) in mats.items():
        all_cols |= set(r.columns)
    dev = led[(led["stage"].isin(["dev", "dev_subwindow"]))
              & (led["cagr"].notna()) & (led["cagr"] != "")
              & (led["kind"].isin(["experiment", "sensitivity"]))
              & (led["config_id"].isin(all_cols))].copy()
    dev["cagr_f"] = pd.to_numeric(dev["cagr"])
    dev = dev.sort_values("cagr_f", ascending=False)
    best_row = dev.iloc[0]
    best_series = None
    best_matrix = ""
    for name, (r, fam) in mats.items():
        if best_row["config_id"] in r.columns and name.endswith(("_dev", "dev")):
            best_series = r[best_row["config_id"]].to_numpy(float)
            best_matrix = name
            break
    # 另记：全项目 dev 最高（可能没有日序列）
    dev_all = led[(led["stage"].isin(["dev", "dev_subwindow"]))
                  & (led["cagr"].notna()) & (led["cagr"] != "")
                  & (led["kind"].isin(["experiment", "sensitivity"]))].copy()
    dev_all["cagr_f"] = pd.to_numeric(dev_all["cagr"])
    top_any = dev_all.sort_values("cagr_f", ascending=False).iloc[0]
    proj = {}
    if best_series is not None:
        T = len(best_series)
        proj = dsr(best_series.mean() / best_series.std(ddof=1), T, nsel, var_project,
                   float(pd.Series(best_series).skew()),
                   float(pd.Series(best_series).kurt() + 3))
        proj_cell = dsr(best_series.mean() / best_series.std(ddof=1), T, ncell, var_project,
                        float(pd.Series(best_series).skew()),
                        float(pd.Series(best_series).kurt() + 3))
    else:
        proj_cell = {}
    multi["project"] = dict(
        n_sel=nsel, n_cell=ncell,
        n_configs_with_sharpe=int(len(sr_ann)),
        var_sr_daily=var_project,
        best_dev_config=str(best_row["config_id"]),
        best_dev_strategy=str(best_row["strategy_id"]),
        best_dev_cagr=float(best_row["cagr_f"]),
        best_dev_matrix=best_matrix,
        dsr_N_sel=proj, dsr_N_cell=proj_cell,
        top_dev_any_strategy=str(top_any["strategy_id"]),
        top_dev_any_cagr=float(top_any["cagr_f"]),
        top_dev_any_has_daily=bool(top_any["config_id"] in all_cols),
    )
    multi["budget"] = dict(prereg_budget=64, n_sel=nsel,
                           overrun_x=round(nsel / 64.0, 2))
    (INT / "多重性.json").write_text(
        json.dumps(multi, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    # 候选分级：总账 + 统计标记
    g = led.copy()
    fam_p = {}
    for name, m in multi["matrices"].items():
        fam_p.setdefault(m["family"], []).append(m["p_maxstat"])
    g["family_p_maxstat"] = g["family"].map(
        lambda f: min(fam_p.get(f, [np.nan])) if fam_p.get(f) else np.nan)
    ci_map = {}
    for _, row in ci.iterrows():
        ci_map.setdefault(row["config_id"], {})[row["matrix"]] = (
            row["cagr_p5"], row["cagr_p50"], row["cagr_p95"])

    def pick_ci(c, stage):
        d = ci_map.get(c, {})
        if not d:
            return "", np.nan, np.nan, np.nan
        order = []
        if stage == "dev":
            order = [k for k in d if k.endswith("_dev")]
        elif stage == "val":
            order = [k for k in d if k.endswith("_val")]
        order += [k for k in d if not k.endswith(("_dev", "_val"))]
        if not order:
            order = list(d)
        k = order[0]
        return k, d[k][0], d[k][1], d[k][2]

    picked = [pick_ci(c, s) for c, s in zip(g["config_id"], g["stage"])]
    g["ci_matrix"] = [p[0] for p in picked]
    g["cagr_p5"] = [p[1] for p in picked]
    g["cagr_p50_ci"] = [p[2] for p in picked]
    g["cagr_p95"] = [p[3] for p in picked]
    g["n_trades_f"] = pd.to_numeric(g["n_trades"], errors="coerce")
    g["power_note"] = np.where(g["n_trades_f"] < 100, "low_power(<100)", "")
    g["ci_note"] = np.where(
        (pd.to_numeric(g["cagr"], errors="coerce") > 0)
        & (g["cagr_p5"] < 0) & (g["cagr_p95"] > 0), "CI含0", "")
    # 配置级配对：只有 dev 与 val 两行都过 G 才算“两窗口都对”
    pair = {}
    for cid_, gg in g.groupby("config_id"):
        st = set(gg["stage"])
        gr = set(gg["grade"])
        if {"dev", "val"} <= st and gr == {"B_conditional"}:
            pair[cid_] = "B_both_windows"
        elif "B_conditional" in gr:
            pair[cid_] = "B_one_window_only"
        elif "E_underpowered" in gr:
            pair[cid_] = "E_underpowered"
        elif "D_contaminated" in gr:
            pair[cid_] = "D_contaminated"
        else:
            pair[cid_] = "negative_or_other"
    g["pair_grade"] = g["config_id"].map(pair)
    cols = ["ledger_id", "config_id", "family", "strategy_id", "stage", "kind", "origin",
            "n_units", "n_trades", "cagr", "mdd", "sharpe", "window_mix",
            "grade", "pair_grade", "fail_gate", "family_p_maxstat", "ci_matrix", "cagr_p5",
            "cagr_p50_ci", "cagr_p95", "power_note", "ci_note",
            "artifact"]
    g[cols].to_csv(INT / "候选分级.csv", index=False, encoding="utf-8")
    print(f"置信区间 {len(ci)} 行 -> {INT/'统一置信区间.csv'}")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if kk in
                           ("n_sel", "n_cell", "best_dev_strategy", "best_dev_cagr")})
                      for k, v in multi["project"].items()}, ensure_ascii=False, indent=2))
    return multi


# ---------------------------------------------------------------- 缺陷 / 零点

def cmd_defects(args):
    INT.mkdir(parents=True, exist_ok=True)
    d = defect_table()
    c = contamination(d)
    c.to_csv(INT / "数据缺陷暴露.csv", index=False, encoding="utf-8")

    # 2021+ 扫描
    scan = []
    for p in sorted(RES.rglob("*")):
        if p.suffix not in (".csv", ".parquet") or p.is_dir():
            continue
        try:
            if p.suffix == ".parquet":
                x = pd.read_parquet(p)
            else:
                x = pd.read_csv(p, nrows=100000)
        except Exception:
            continue
        mx = None
        for col in x.columns:
            s = x[col]
            if str(s.dtype).startswith("datetime"):
                mx = s.max() if mx is None else max(mx, s.max())
            elif pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
                ss = s.astype(str)
                v = ss[ss.str.match(r"^\d{4}-\d{2}-\d{2}")]
                if len(v):
                    vm = v.max()
                    mx = vm if mx is None else max(mx, vm)
        if mx is not None and str(mx)[:10] > "2020-12-31":
            names = " ".join(map(str, x.columns)).lower()
            perf = any(k in names for k in
                       ("nav", "equity", "cagr", "return", "pnl", "净值", "收益",
                        "mdd", "sharpe", "daily_ret"))
            if "统计整合" in str(p) or p.name in ("实验注册表.csv", "试验总账.csv"):
                kind = "metadata_self"
            elif perf:
                kind = "PERFORMANCE_2021+_SUSPECT"
            elif "未评估" in p.name or "缓存" in str(p):
                kind = "derived_state_only"
            else:
                kind = "non_performance_metadata"
            scan.append(dict(file=str(p.relative_to(ROOT)), max_date=str(mx)[:10],
                             kind=kind, rows=int(x.shape[0])))
    pd.DataFrame(scan).to_csv(INT / "2021后日期扫描.csv", index=False, encoding="utf-8")
    print(c.to_string(index=False))
    print(f"\n2021+ 日期文件 {len(scan)} 个（明细 2021后日期扫描.csv）")
    return c, scan


def cmd_nullcheck(args):
    import duckdb
    INT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "01_数据/frozen/frozen.duckdb"), read_only=True)
    d = con.execute("""
        select symbol, date, pct_chg, close, is_st, td_since_list, amount
        from daily where date between date '2008-01-01' and date '2020-12-31'
    """).fetchdf()
    con.close()
    d["date"] = pd.to_datetime(d.date)
    px = d.pivot(index="date", columns="symbol", values="pct_chg") / 100.0
    close = d.pivot(index="date", columns="symbol", values="close")
    st = d.pivot(index="date", columns="symbol", values="is_st")
    tdl = d.pivot(index="date", columns="symbol", values="td_since_list")
    amt = d.pivot(index="date", columns="symbol", values="amount")
    elig = ((tdl >= 250) & (st == 0) & (close >= 2)
            & (amt.rolling(20, min_periods=10).mean() >= 5e7) & px.notna())
    R = px.fillna(0).to_numpy()
    E = elig.to_numpy()
    dates = px.index
    rows = []
    rng_seed = SEED
    for lab, a, b in [("dev", "2008-01-01", "2015-12-31"),
                      ("val", "2016-01-01", "2020-12-31"),
                      ("full", "2008-01-01", "2020-12-31")]:
        m = (dates >= a) & (dates <= b)
        r2, E2 = R[m], E[m]
        n = len(r2)
        ew = float(np.prod(1 + r2.mean(1)) ** (ANN / n) - 1)
        for cost in (0.0, 0.0025):
            rng = np.random.default_rng(rng_seed)
            res = []
            for s in range(args.seeds):
                rets = np.zeros(n)
                for i in range(0, n, 21):
                    if i + 21 > n:
                        break
                    el = np.where(E2[i])[0]
                    if len(el) < 10:
                        continue
                    pick = rng.choice(el, 10, replace=False)
                    blk = r2[i:i + 21][:, pick].mean(1).copy()
                    blk[0] -= cost
                    rets[i:i + 21] = blk
                res.append(np.prod(1 + rets) ** (ANN / n) - 1)
            res = np.array(res)
            rows.append(dict(window=lab, cost_per_rotation=cost, seeds=args.seeds,
                             random10_mean=float(res.mean()),
                             random10_p5=float(np.percentile(res, 5)),
                             random10_p95=float(np.percentile(res, 95)),
                             ew_daily_rebal=ew, n_days=n))
    out = pd.DataFrame(rows)
    out.to_csv(INT / "零假设校准.csv", index=False, encoding="utf-8")
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------- 命令

def cmd_build(args):
    rows = []
    inv_official(rows)
    inv_trend(rows)
    inv_struct(rows)
    inv_market(rows)
    inv_exit(rows)
    inv_stat(rows)
    inv_ecology(rows)
    d = defect_table()
    c = contamination(d)
    grade_rows(rows, c)
    assign_counts(rows)
    allrows = write_ledger(rows)
    INT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["n_units"] = pd.to_numeric(df["n_units"], errors="coerce").fillna(0).astype(int)
    df["w_sel"] = df.n_units * df.in_Nsel
    df["w_cell"] = df.n_units * df.in_Ncell
    df["w_diag"] = df.n_units * df.in_Ndiag
    cnt = (df.groupby(["family", "kind"]).agg(
        rows=("ledger_id", "count"), n_units=("n_units", "sum"),
        n_sel=("w_sel", "sum"), n_cell=("w_cell", "sum"),
        n_diag=("w_diag", "sum")).reset_index())
    total = dict(rows=len(df), n_sel=int(df.w_sel.sum()), n_cell=int(df.w_cell.sum()),
                 n_diag=int(df.w_diag.sum()))
    tot = pd.DataFrame([{"family": "TOTAL", "kind": "all", "rows": total["rows"],
                         "n_units": total["n_sel"] + total["n_cell"],
                         "n_sel": total["n_sel"], "n_cell": total["n_cell"],
                         "n_diag": total["n_diag"]}])
    pd.concat([cnt, tot]).to_csv(INT / "试验计数.csv", index=False, encoding="utf-8")
    print(f"总账 {len(allrows)} 行（含 meta）-> {LEDGER}")
    print(f"N_sel={total['n_sel']}  N_cell={total['n_cell']}  N_diag={total['n_diag']}")
    print(f"预算 64，超出 {total['n_sel']/64:.2f}×")
    gr = df.grade.value_counts().to_dict()
    print("定级：", gr)
    return total


def cmd_summary(args):
    led = pd.DataFrame(read_ledger())
    print(f"总账：{len(led)} 行  锚点 {led.iloc[0]['prev_sha256'][:16]}…")
    print(led.groupby(["family", "grade"]).size().to_string())
    print()
    nsel = led[led.in_Nsel == "1"]
    print("N_sel 按族：", nsel.groupby("family").size().to_dict())
    pos = nsel[(pd.to_numeric(nsel.cagr, errors="coerce") > 0)]
    print(f"dev/val 任一窗口净 CAGR>0 的配置：{len(pos)} / {len(nsel)}")
    print(pos[["family", "strategy_id", "stage", "cagr", "mdd"]].to_string(index=False))


def cmd_verify(args):
    errs, n = verify_ledger()
    return 0 if not errs else 1


def cmd_audit_registry(args):
    e = write_registry_audit()
    print(f"登记审计 -> {INT/'登记审计.md'}（FAIL {len(e)}）")
    for x in e:
        print("FAIL", x)
    return e


# ---------------------------------------------------------------- 自检

def cmd_selftest(args):
    tmp = Path(tempfile.mkdtemp(prefix="intledger_"))
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra and not cond else ""))
        ok &= bool(cond)

    # 1 链
    rows = [mk("T", "s", {"a": 1}, "2008-01-01", "2015-12-31", "dev", "experiment",
               "post_hoc_unregistered", "done", cagr=-0.1, mdd=-0.5,
               metric_basis="reported_table")]
    rows += [mk("T", "s", {"a": 1}, "2016-01-01", "2020-12-31", "val", "experiment",
                "post_hoc_unregistered", "done", cagr=-0.2, mdd=-0.4,
                metric_basis="reported_table")]
    assign_counts(rows)
    p = tmp / "led.csv"
    global LEDGER
    old = LEDGER
    LEDGER = p
    try:
        allr = write_ledger(rows)
        errs, n = verify_ledger(p, verbose=False)
        check("链建立且可校验", not errs and n[0] == 1, str(errs))
        text = p.read_text(encoding="utf-8").splitlines()
        text[2] = text[2].replace("-0.1", "-0.9")
        p.write_text("\n".join(text) + "\n", encoding="utf-8")
        errs, _ = verify_ledger(p, verbose=False)
        check("篡改可检测", bool(errs))
    finally:
        LEDGER = old

    # 2 DSR 单调
    d1 = dsr(0.02, 2000, 1, 1e-4, 0, 3)
    d100 = dsr(0.02, 2000, 100, 1e-4, 0, 3)
    check("DSR 随 N 递减", d1["dsr"] > d100["dsr"])

    # 3 PBO：噪声≈0.5（单次方差大，取 8 个实现均值）；有真实优势时应接近 0
    rng = np.random.default_rng(1)
    pbos = []
    for _ in range(8):
        Rn = rng.normal(0, 0.01, size=(1500, 20))
        pb = pbo_cscv(Rn)
        if pb:
            pbos.append(pb["pbo"])
    check("噪声 PBO 均值在 0.3~0.7", 0.3 < float(np.mean(pbos)) < 0.7,
          str([round(p, 2) for p in pbos]))
    Rk = rng.normal(0, 0.01, size=(1500, 12))
    Rk[:, 3] += 0.002
    check("真实优势 PBO≈0", pbo_cscv(Rk)["pbo"] < 0.10, str(pbo_cscv(Rk)["pbo"]))

    # 4 max-stat 对强信号可检出
    Rn = rng.normal(0, 0.01, size=(800, 8))
    sig = rng.normal(0.002, 0.01, size=(800, 8))
    idx = stationary_indices(800, 20, 500, 7)
    _, p_good, _ = maxstat(sig, idx)
    _, p_bad, _ = maxstat(Rn, idx)
    check("max-stat 强信号 p 更小", p_good < p_bad and p_good < 0.10,
          f"{p_good:.3f} vs {p_bad:.3f}")

    # 5 bootstrap 点估计落在区间内
    X = rng.normal(0.0002, 0.01, size=(500, 1))
    idx = stationary_indices(500, 20, 500, 11)
    cb, _, _ = boot_all(X, idx)
    lo, hi = np.percentile(cb[:, 0], [5, 95])
    pt = float(np.expm1(np.log1p(X[:, 0]).mean() * ANN))
    check("bootstrap 覆盖点估计", lo <= pt <= hi, f"[{lo:.4f},{hi:.4f}] pt={pt:.4f}")

    print(f"\n结论：{'全部通过' if ok else '存在失败项'}（{tmp}）")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="整合实验登记：盘点、补登记、统一推断")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("build", cmd_build), ("verify", cmd_verify),
                     ("audit-registry", cmd_audit_registry), ("defects", cmd_defects),
                     ("nullcheck", cmd_nullcheck), ("summary", cmd_summary),
                     ("emit-registry", lambda a: (emit_registry(), 0)[1]),
                     ("selftest", cmd_selftest)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
    p = sub.add_parser("stats")
    p.add_argument("--boot", type=int, default=BOOT)
    p.set_defaults(func=cmd_stats)
    p = sub.add_parser("nullcheck2")
    p.add_argument("--seeds", type=int, default=200)
    p.set_defaults(func=cmd_nullcheck)
    args = ap.parse_args(argv)
    if not hasattr(args, "seeds"):
        args.seeds = 200
    r = args.func(args)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
