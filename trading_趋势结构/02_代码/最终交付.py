# -*- coding: utf-8 -*-
"""
最终交付入口（趋势结构项目收尾）

角色：最终整合负责人。独占产物：
  00_一页纸.md / 01_完整报告.md / 02_规则手册_研究结论版.md / 03_数据与复现说明.md
  04_最终结论.json
  03_结果/实验注册表.csv（只追加）
  03_结果/最终试验总账.csv（只追加）与 03_结果/最终统计/
  02_代码/最终交付.py（本文件）

纪律
----
* 冻结日 2026-10-26；收益只允许 2008-01-01..2020-12-31。2021+ 不计算任何收益。
* 注册表与总账都是**只追加**：历史 planned 行保留原样，T0005 的真实 done 结果另起新行。
* 不重跑任何策略；只读 `03_结果/主候选_T0005/` 的现成结果并做数字对账。
* 不为 60% 反推任何参数；不提供任何当前买卖建议。

子命令
------
  register    把 T0005 的 done 状态与指标合规登记进官方注册表（保留历史 planned 行）
  ledger      把 T0005 真实结果追加进最终总账，并重算最终统计聚合
  conclusion  生成 04_最终结论.json（机器可读总交付）
  remediation 生成 04_证据/审计整改验收.json（四份审计逐项处置 + 验收断言）
  verify      注册表链 / 总账链 / T0005 数字对账 / 计数一致性 / 2021 护栏 / 整改断言
  selfcheck   交付物自检（文档口径 + 禁用表述 + 整改覆盖）
  all         register -> ledger -> conclusion -> remediation -> verify -> selfcheck
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "02_代码")
RES = os.path.join(ROOT, "03_结果")
FINAL_DIR = os.path.join(RES, "最终统计")
REGISTRY = os.path.join(RES, "实验注册表.csv")
LEDGER = os.path.join(RES, "最终试验总账.csv")
STAGE1_LEDGER = os.path.join(RES, "试验总账.csv")
T0005_DIR = os.path.join(RES, "主候选_T0005")
CONCLUSION = os.path.join(ROOT, "04_最终结论.json")
SELFCHECK = os.path.join(FINAL_DIR, "最终交付自检.json")

SEED = 20261026
TAG = "[FINAL-DELIVERY-2026-10-29]"
FINAL_STATUS = "四原审计员终验均PASS，方法N7/N8已随后修正"
FINAL_VERDICT = "FINAL_ACCEPTANCE_PASS_N7_N8_FIXED"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
FULL = ("2008-01-01", "2020-12-31")
STAGE_END_CAP = pd.Timestamp("2020-12-31")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def load_module(rel: str, name: str):
    path = p(rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, rows: list[dict], cols: list[str], crlf: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\r\n" if crlf else "\n",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_rows(path: str, rows: list[dict], cols: list[str], crlf: bool) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\r\n" if crlf else "\n",
                           extrasaction="ignore")
        w.writerows(rows)


def fnum(x, nd: int = 8) -> str:
    if x is None or x == "":
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def jdump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def jload(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# T0005 源结果（只读）
# ---------------------------------------------------------------------------

def source_numbers() -> dict:
    stats = jload(os.path.join(T0005_DIR, "07_统计_主候选.json"))
    ledger = pd.read_csv(os.path.join(T0005_DIR, "60_试验台账.csv"))
    mult = jload(os.path.join(T0005_DIR, "41_多重性.json"))
    rec = jload(os.path.join(T0005_DIR, "40_独立账户对账.json"))
    sc = {w: jload(os.path.join(T0005_DIR, f"08_自检_T0005_{w}.json")) for w in ("dev", "val")}
    funnel = jload(os.path.join(T0005_DIR, "01_信号覆盖_漏斗.json"))
    gate = pd.read_csv(os.path.join(T0005_DIR, "10_信号_风险门.csv"))
    exits = pd.read_csv(os.path.join(T0005_DIR, "11_退出原因.csv"))
    yearly = {w: pd.read_csv(os.path.join(T0005_DIR, f"04_年度_T0005_{w}.csv")) for w in ("dev", "val")}
    return {"stats": stats, "ledger": ledger, "mult": mult, "recon": rec,
            "selfcheck": sc, "funnel": funnel, "gate": gate, "exits": exits, "yearly": yearly}


def nav_series(src: dict, window: str) -> pd.Series:
    df = pd.read_csv(os.path.join(T0005_DIR, f"03_日净值_T0005_{window}.csv"))
    s = pd.Series(df["nav"].astype(float).values, index=pd.to_datetime(df["date"]))
    return s.sort_index()


# ---------------------------------------------------------------------------
# register：官方注册表只追加
# ---------------------------------------------------------------------------

def _registry_args(**kw):
    base = dict(kind="experiment", stage="dev", status="done", decision="discard",
                window_start=DEV[0], window_end=DEV[1], n_configs="1",
                params_json="{}", metrics_json="", family="F1-S1-E2-X3-G1",
                universe_id="U-MAINBOARD-V1", cost_model_id="C-REAL-V1",
                code="02_代码/主候选_T0005.py", parent_trial="", researcher="最终整合负责人",
                model="deepseek-flash", notes="", post_holdout_reason="",
                ack_budget=False, budget_reason="", independent_runner="",
                hypothesis="")
    base.update(kw)
    return argparse.Namespace(**base)


def stage_register(force: bool = False) -> dict:
    R = load_module("02_代码/实验登记.py", "regtool")
    path = R.ROOT / "03_结果" / "实验注册表.csv"
    rows = R.read_rows(path)
    if any(TAG in r.get("notes", "") for r in rows) and not force:
        print("register: 已登记（存在 %s 标记），跳过" % TAG)
        return {"skipped": True, "n_rows": len(rows)}

    src = source_numbers()
    dev_m = src["stats"]["dev"]["metrics"]
    val_m = src["stats"]["val"]["metrics"]
    base_planned = R.latest_rows(rows)["T0005"]
    params_json = base_planned["params_json"]

    metrics_dev = {
        "n_trades": int(dev_m["n_trades"]), "cagr": round(float(dev_m["cagr"]), 10),
        "mdd": round(float(dev_m["mdd"]), 10), "sharpe": round(float(dev_m["sharpe"]), 8),
        "vol": round(float(dev_m["vol"]), 8), "calmar": round(float(dev_m["calmar"]), 8),
        "n_codes": int(dev_m["n_codes"]), "avg_positions": round(float(dev_m["avg_positions"]), 6),
        "time_in_market": round(float(dev_m["time_in_market"]), 6),
        "cagr_ex_top5": round(float(dev_m["cagr_ex_top5"]), 10),
        "window": "dev", "source": "03_结果/主候选_T0005/07_统计_主候选.json",
    }
    metrics_val = {
        "n_trades": int(val_m["n_trades"]), "cagr": round(float(val_m["cagr"]), 10),
        "mdd": round(float(val_m["mdd"]), 10), "sharpe": round(float(val_m["sharpe"]), 8),
        "vol": round(float(val_m["vol"]), 8), "calmar": round(float(val_m["calmar"]), 8),
        "n_codes": int(val_m["n_codes"]), "avg_positions": round(float(val_m["avg_positions"]), 6),
        "time_in_market": round(float(val_m["time_in_market"]), 6),
        "cagr_ex_top5": round(float(val_m["cagr_ex_top5"]), 10),
        "window": "val", "source": "03_结果/主候选_T0005/07_统计_主候选.json",
    }

    # 1) T0005 修订 2：dev done（保留 revision=1 planned 行）
    changes = {
        "status": "done",
        "metrics_json": R.canonical_json(json.dumps(metrics_dev), "metrics_json"),
        "decision": "discard",
        "code_path": "02_代码/主候选_T0005.py",
        "code_sha256": sha256_file(p("02_代码/主候选_T0005.py")),
        "notes": (base_planned["notes"] +
                  (" | " if base_planned["notes"] else "") +
                  f"{TAG} 2026-10-29 00:29-00:53 由主候选执行员实际运行 dev；"
                  "本修订行由最终整合负责人补登记，不改写 revision=1 planned 行。"
                  "dev 净年化 4.16%、MDD -9.05%、103 回合；val 见 T0007。结论：淘汰（val 为负）。"),
    }
    r2 = R.build_revision_row(rows, "T0005", changes)

    # 2) T0007：val done（新试验行，parent=T0005）
    rows2 = rows + [r2]
    r3 = R.build_new_row(rows2, _registry_args(
        stage="val", window_start=VAL[0], window_end=VAL[1], parent_trial="T0005",
        params_json=params_json, metrics_json=json.dumps(metrics_val), decision="discard",
        hypothesis="主候选 F1-S1-E2-X3-G1 在 2016-2020 的净年化为正（§7.9.5 两段>0 的后半段）",
        notes=(f"{TAG} 验证段补登记：运行时间与 dev 同批（2026-10-29 00:29-00:53），"
               "参数与 T0005 planned 行逐字段相同；val 净年化 -2.53%（MDD -13.80%，159 回合）→ "
               "§7.9.5 不通过，主候选淘汰。本行是事后补登记，不伪造 planned 前置。"),
        n_configs="1"))

    # 3) T0008：消融 V1-V8 + A1-A7 共 15 个配置（dev/val 各一次）
    abl = src["ledger"][src["ledger"]["kind"] == "ablation"]
    abl_dev = abl[abl["stage"] == "dev"]
    abl_val = abl[abl["stage"] == "val"]
    metrics_abl = {
        "n_trades": int(abl["n_trades"].sum()),
        "cagr": round(float(abl_dev["cagr"].max()), 10),
        "mdd": round(float(abl_dev.loc[abl_dev["cagr"].idxmax(), "mdd"]), 10),
        "n_variants": int(abl["trial"].nunique()),
        "best_dev_cagr": round(float(abl_dev["cagr"].max()), 10),
        "best_val_cagr": round(float(abl_val["cagr"].max()), 10),
        "worst_val_cagr": round(float(abl_val["cagr"].min()), 10),
        "n_val_positive": int((abl_val["cagr"] > 0).sum()),
        "metric_semantics": "cagr/mdd 取 dev 最好变体（V7 固定2R目标）；val 最好仅 -0.11%",
    }
    params_abl = json.dumps({"base": json.loads(params_json), "ablation_set": "V1-V8+A1-A7",
                             "n_variants": int(abl["trial"].nunique())}, sort_keys=True,
                            ensure_ascii=False)
    rows3 = rows2 + [r3]
    r4 = R.build_new_row(rows3, _registry_args(
        kind="ablation", stage="na", window_start=FULL[0], window_end=FULL[1],
        parent_trial="T0005", params_json=params_abl, n_configs=str(abl["trial"].nunique()),
        metrics_json=json.dumps(metrics_abl), decision="discard",
        hypothesis="主候选的单因素替换（F/S/E/X/G 与 A1-A7 阶梯）中至少有一个在 dev/val 两段都不差于主候选",
        notes=(f"{TAG} 15 个消融变体 × dev/val = 30 次账户运行（03_结果/主候选_T0005/20_消融.csv）。"
               "V1（F2 季/月）两段 0 笔成交，是“无法交易”不是“无损失”。"
               "全部 15 个变体 val 年化为负（最好 A1 -0.11%）。")))

    # 4) T0009：参数邻域 S01-S18
    sens = src["ledger"][src["ledger"]["kind"] == "sensitivity"]
    sens_dev = sens[sens["stage"] == "dev"]
    sens_val = sens[sens["stage"] == "val"]
    metrics_sens = {
        "n_trades": int(sens["n_trades"].sum()),
        "cagr": round(float(dev_m["cagr"]), 10),
        "mdd": round(float(dev_m["mdd"]), 10),
        "n_variants": int(sens["trial"].nunique()),
        "best_val_cagr": round(float(sens_val["cagr"].max()), 10),
        "worst_val_cagr": round(float(sens_val["cagr"].min()), 10),
        "n_val_positive": int((sens_val["cagr"] > 0).sum()),
        "metric_semantics": "cagr/mdd 为主候选 dev 参照值；18 个邻域 val 全部为负",
    }
    rows4 = rows3 + [r4]
    params_sens = json.dumps({"base": json.loads(params_json), "sensitivity_set": "S01-S18",
                              "n_variants": int(sens["trial"].nunique())}, sort_keys=True,
                             ensure_ascii=False)
    r5 = R.build_new_row(rows4, _registry_args(
        kind="sensitivity", stage="na", window_start=FULL[0], window_end=FULL[1],
        parent_trial="T0005", params_json=params_sens, n_configs=str(sens["trial"].nunique()),
        metrics_json=json.dumps(metrics_sens), decision="discard",
        hypothesis="主候选在 §7.8 的 18 个参数邻域里符号稳定（两段同号）",
        notes=(f"{TAG} 18 个参数邻域 × dev/val = 36 次账户运行（03_结果/主候选_T0005/30_邻域.csv）。"
               "dev 2.11%~5.50%、val -2.60%~-1.35%，**没有一个邻域在 val 转正**。"
               "敏感性只报告稳健性，不参与选择（§7.8）。")))

    new_rows = [r2, r3, r4, r5]
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        for r in new_rows:
            w.writerow([r[c] for c in R.COLUMNS])
        f.flush()
        os.fsync(f.fileno())

    all_rows = R.read_rows(path)
    errors, warnings = R.verify_rows(all_rows, strict=False, allow_code_drift=False)
    print(f"register: 追加 {len(new_rows)} 行；注册表 {len(all_rows)} 行；"
          f"verify errors={len(errors)} warnings={len(warnings)}")
    for e in errors:
        print("  FAIL " + e)
    cum = R.prefix_cumulative(all_rows)
    print(f"register: trials_cumulative 末值 = {max(cum.values()) if cum else 0}")
    return {"skipped": False, "n_rows": len(all_rows), "appended": len(new_rows),
            "errors": errors, "warnings": warnings, "cumulative": max(cum.values())}


# ---------------------------------------------------------------------------
# ledger：最终总账只追加 + 最终统计重算
# ---------------------------------------------------------------------------

LEDGER_COLS = ["ledger_id", "phase", "source", "family", "strategy_id", "params_id", "config_id",
               "window_start", "window_end", "stage", "kind", "category", "origin", "status",
               "n_units", "in_Nsel", "in_Ncell", "in_Ndiag", "in_Nplatform", "in_Nsens", "in_Naudit",
               "n_trades", "cagr", "mdd", "sharpe", "vol", "calmar", "metric_basis", "window_mix",
               "artifact", "artifact_extra", "grade", "fail_gate", "notes", "prereg_role",
               "prev_sha256", "row_sha256"]


def ledger_row_hash(r: dict) -> str:
    txt = "\x1f".join(str(r[c]) for c in LEDGER_COLS[:-1])
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def verify_ledger_chain(rows: list[dict]) -> list[str]:
    errs = []
    s1 = read_rows(STAGE1_LEDGER)
    prev = s1[-1]["row_sha256"]
    if rows and rows[0]["prev_sha256"] != prev:
        errs.append(f"F00000 prev_sha256 != 第一阶段总账尾哈希")
    for r in rows:
        if r["prev_sha256"] != prev:
            errs.append(f"{r['ledger_id']}: prev_sha256 断链")
        if ledger_row_hash(r) != r["row_sha256"]:
            errs.append(f"{r['ledger_id']}: row_sha256 不匹配")
        prev = r["row_sha256"]
    return errs


def _base_ledger_row(**kw) -> dict:
    r = {c: "" for c in LEDGER_COLS}
    r.update({
        "phase": "2", "source": "official_prereg", "kind": "experiment", "category": "prereg",
        "origin": "preregistered_local", "status": "done", "n_units": "1",
        "in_Nsel": "0", "in_Ncell": "0", "in_Ndiag": "0", "in_Nplatform": "0",
        "in_Nsens": "0", "in_Naudit": "0", "metric_basis": "daily_nav", "window_mix": "single",
        "prereg_role": "", "artifact_extra": "",
    })
    r.update({k: ("" if v is None else str(v)) for k, v in kw.items()})
    return r


def build_t0005_ledger_rows(prev_hash: str, start_idx: int) -> list[dict]:
    src = source_numbers()
    M = load_module("02_代码/最终试验总账.py", "finalledger")
    dev_m = src["stats"]["dev"]["metrics"]
    val_m = src["stats"]["val"]["metrics"]
    params_id = M.cid12("F1-S1-E2-X3-G1")
    out = []

    def add(**kw):
        r = _base_ledger_row(**kw)
        r["prev_sha256"] = ""
        r["row_sha256"] = ""
        out.append(r)

    add(family="MAIN", strategy_id="T0005", params_id=params_id, config_id="MAIN:T0005",
        window_start=DEV[0], window_end=DEV[1], stage="dev", category="prereg",
        in_Nsel="1", in_Ncell="1", n_trades=str(int(dev_m["n_trades"])),
        cagr=fnum(dev_m["cagr"]), mdd=fnum(dev_m["mdd"]), sharpe=fnum(dev_m["sharpe"]),
        vol=fnum(dev_m["vol"]), calmar=fnum(dev_m["calmar"]),
        artifact="03_结果/主候选_T0005/07_统计_主候选.json", prereg_role="main",
        notes=(f"{TAG} 实测行（dev）：净年化 4.16%、MDD -9.05%、103 回合/81 只；"
               "平均持仓 0.83 只、在场 24.5%、去前 5 大盈利后 2.40%。"
               "同 config 的历史 planned 行 F01629 保留 F_unevaluable，不改写。"))
    add(family="MAIN", strategy_id="T0005", params_id=params_id, config_id="MAIN:T0005",
        window_start=VAL[0], window_end=VAL[1], stage="val", category="prereg",
        in_Ncell="1", n_trades=str(int(val_m["n_trades"])),
        cagr=fnum(val_m["cagr"]), mdd=fnum(val_m["mdd"]), sharpe=fnum(val_m["sharpe"]),
        vol=fnum(val_m["vol"]), calmar=fnum(val_m["calmar"]),
        artifact="03_结果/主候选_T0005/07_统计_主候选.json",
        notes=(f"{TAG} 实测行（val）：净年化 -2.53%、MDD -13.80%、159 回合/111 只；"
               "平均持仓 1.62 只、在场 54.8%、去前 5 大盈利后 -4.23%。§7.9.5 两段>0 不通过 → 淘汰。"))

    led = src["ledger"]
    for kind, cat in (("ablation", "diag"), ("sensitivity", "sensitivity")):
        sub = led[led["kind"] == kind]
        for trial, g in sub.groupby("trial"):
            gd = g[g["stage"] == "dev"]
            gv = g[g["stage"] == "val"]
            if len(gd):
                gd = gd.iloc[0]
                if kind == "sensitivity":
                    add(family="MAIN", strategy_id=trial, params_id=M.cid12(trial),
                        config_id=f"MAIN:{trial}", window_start=DEV[0], window_end=DEV[1],
                        stage="dev", kind="sensitivity", category="sensitivity",
                        in_Nsel="1", in_Ncell="1", in_Nsens="1",
                        n_trades=str(int(gd["n_trades"])), cagr=fnum(gd["cagr"]),
                        mdd=fnum(gd["mdd"]),
                        artifact="03_结果/主候选_T0005/30_邻域.csv", prereg_role="none",
                        notes=f"{TAG} 参数邻域 {trial}：{gd['desc']}（敏感性，不参与选择）")
                else:
                    add(family="MAIN", strategy_id=trial, params_id=M.cid12(trial),
                        config_id=f"MAIN:{trial}", window_start=DEV[0], window_end=DEV[1],
                        stage="dev", kind="ablation", category="diag",
                        in_Ndiag="1", n_trades=str(int(gd["n_trades"])), cagr=fnum(gd["cagr"]),
                        mdd=fnum(gd["mdd"]), grade="DIAG",
                        artifact="03_结果/主候选_T0005/20_消融.csv", prereg_role="none",
                        notes=f"{TAG} 消融 {trial}：{gd['desc']}（诊断，不计 N_sel）")
            if len(gv):
                gv = gv.iloc[0]
                if kind == "sensitivity":
                    add(family="MAIN", strategy_id=trial, params_id=M.cid12(trial),
                        config_id=f"MAIN:{trial}", window_start=VAL[0], window_end=VAL[1],
                        stage="val", kind="sensitivity", category="sensitivity",
                        in_Ncell="1", n_trades=str(int(gv["n_trades"])), cagr=fnum(gv["cagr"]),
                        mdd=fnum(gv["mdd"]), artifact="03_结果/主候选_T0005/30_邻域.csv",
                        prereg_role="none",
                        notes=f"{TAG} 参数邻域 {trial}：{gv['desc']}（敏感性，不参与选择）")
                else:
                    add(family="MAIN", strategy_id=trial, params_id=M.cid12(trial),
                        config_id=f"MAIN:{trial}", window_start=VAL[0], window_end=VAL[1],
                        stage="val", kind="ablation", category="diag",
                        in_Ndiag="1", n_trades=str(int(gv["n_trades"])), cagr=fnum(gv["cagr"]),
                        mdd=fnum(gv["mdd"]), grade="DIAG",
                        artifact="03_结果/主候选_T0005/20_消融.csv", prereg_role="none",
                        notes=f"{TAG} 消融 {trial}：{gv['desc']}（诊断，不计 N_sel）")

    # 编号与哈希链
    prev = prev_hash
    for i, r in enumerate(out):
        r["ledger_id"] = f"F{start_idx + i:05d}"
        r["prev_sha256"] = prev
        r["row_sha256"] = ledger_row_hash(r)
        prev = r["row_sha256"]
    return out


def build_fam_stats(src: dict | None = None) -> dict:
    """复现最终统计的族级 PBO/DSR 输入，并补 MAIN 族。"""
    fs: dict = {}
    pbo = read_rows(p("03_结果/最终统计/最终PBO_maxstat.csv"))
    for r in pbo:
        if r.get("kind") != "PBO":
            continue
        fam = r["family"]
        fs.setdefault(fam, {})["pbo_min"] = min(fs.get(fam, {}).get("pbo_min", 1.0), float(r["pbo"]))
    for r in read_rows(p("03_结果/最终统计/族内DSR.csv")):
        fam = r["family"]
        fs.setdefault(fam, {})["dsr_max"] = max(fs.get(fam, {}).get("dsr_max", 0.0), float(r["dsr"]))
    M = load_module("02_代码/最终试验总账.py", "finalledger")
    for fam, v in M.load_integrated_stats().items():
        cur = fs.setdefault(fam, {})
        for k, x in v.items():
            if k == "pbo_min":
                cur[k] = min(cur.get(k, 1.0), x)
            elif k == "dsr_max":
                cur[k] = max(cur.get(k, 0.0), x)
            else:
                cur[k] = x
    mult = (src or source_numbers())["mult"]
    fs["MAIN"] = {
        "pbo_min": float(min(mult["pbo_neighbor"]["dev"]["pbo"], mult["pbo_neighbor"]["val"]["pbo"])),
        "dsr_max": float(mult["dev_family_dsr"]["dsr"]),
        "source": "03_结果/主候选_T0005/41_多重性.json（18 配置邻域；族内 DSR 因变体高度相关而偏高，仅诊断）",
    }
    return fs


def summarize(rows: list[dict]) -> dict:
    out = {}
    for k in ("in_Nsel", "in_Ncell", "in_Ndiag", "in_Nplatform", "in_Nsens", "in_Naudit"):
        out[k] = float(sum(float(r[k] or 0) for r in rows))
    out["rows"] = len(rows)
    return out


def stage_ledger(force: bool = False) -> dict:
    M = load_module("02_代码/最终试验总账.py", "finalledger")
    M.D_GATE.setdefault("MAIN", "D_pass")  # 运行时补 D 闸：A 空间 + 独立账户对账
    rows = read_rows(LEDGER)
    errs = verify_ledger_chain(rows)
    if errs:
        raise SystemExit("总账链校验失败：" + "; ".join(errs[:5]))

    added = 0
    if any(TAG in r.get("notes", "") for r in rows) and not force:
        print("ledger: 已包含 T0005 实测行（存在 %s 标记），只重算聚合" % TAG)
    else:
        new = build_t0005_ledger_rows(rows[-1]["row_sha256"], len(rows))
        rows = rows + new
        added = len(new)

    fam_stats = build_fam_stats()
    errors, warnings = [], []
    if added:
        snapshot = {r["ledger_id"]: (r["grade"], r["fail_gate"], r["notes"]) for r in rows if r["ledger_id"] < new[0]["ledger_id"]}
        gates = M.compute_gates(rows, fam_stats)
        # 还原历史行的 grade/fail_gate/notes（只追加，不改写历史）
        for r in rows:
            if r["ledger_id"] in snapshot:
                r["grade"], r["fail_gate"], r["notes"] = snapshot[r["ledger_id"]]
        gmap = {(g["family"], g["config_id"]): g for g in gates}
        for r in rows:
            if r["ledger_id"] < new[0]["ledger_id"]:
                continue
            g = gmap.get((r["family"], r["config_id"]))
            if g is None:
                r["grade"] = r["grade"] or "DIAG"
                continue
            r["grade"] = g["grade"]
            r["fail_gate"] = g["fail_gate"]
        new = rows[-added:]
    else:
        gates = M.compute_gates(rows, fam_stats)

    # 重算所有行的哈希链（历史行逐字节不变：字段未改，哈希不变）
    prev = read_rows(STAGE1_LEDGER)[-1]["row_sha256"]
    for r in rows:
        if r["ledger_id"] in {x["ledger_id"] for x in (new if added else [])}:
            r["prev_sha256"] = prev
            r["row_sha256"] = ledger_row_hash(r)
        prev = r["row_sha256"]

    write_rows(LEDGER, rows, LEDGER_COLS, crlf=True)
    errs = verify_ledger_chain(rows)
    if errs:
        errors.extend(errs)

    total = summarize(rows)
    by_family = {fam: summarize([x for x in rows if x["family"] == fam])
                 for fam in sorted({x["family"] for x in rows})}
    by_cat = {cat: summarize([x for x in rows if x["category"] == cat])
              for cat in sorted({x["category"] for x in rows})}

    def _cnt(scope, name, v):
        return {"scope": scope, "name": name, "rows": v["rows"], "N_sel": v["in_Nsel"],
                "N_cell": v["in_Ncell"], "N_diag": v["in_Ndiag"],
                "N_platform": v["in_Nplatform"], "N_sens_subset": v["in_Nsens"],
                "N_audit": v["in_Naudit"]}

    write_rows(os.path.join(FINAL_DIR, "最终计数.csv"),
               [_cnt("total", "ALL", total)] +
               [_cnt("family", k, v) for k, v in by_family.items()] +
               [_cnt("category", k, v) for k, v in by_cat.items()],
               ["scope", "name", "rows", "N_sel", "N_cell", "N_diag", "N_platform", "N_sens_subset", "N_audit"])

    gate_cols = ["family", "config_id", "strategy_id", "category", "role", "artifact",
                 "dev_cagr", "dev_mdd", "dev_trades", "val_cagr", "val_mdd", "val_trades",
                 "full_cagr", "full_mdd", "full_trades", "gates", "grade", "fail_gate", "eligible"]
    gate_rows = []
    for g in gates:
        rr = dict(g)
        rr["gates"] = json.dumps(g.get("gates", {}), ensure_ascii=False, sort_keys=True)
        rr["eligible"] = bool(g.get("eligible"))
        gate_rows.append(rr)
    write_rows(os.path.join(FINAL_DIR, "统一门槛.csv"), gate_rows, gate_cols)

    # 族内 DSR：追加 MAIN
    dsr_rows = read_rows(os.path.join(FINAL_DIR, "族内DSR.csv"))
    if not any(r["family"] == "MAIN" for r in dsr_rows):
        src = source_numbers()
        M2 = load_module("02_代码/最终试验总账.py", "finalledger2")
        dev_s = nav_series(src, "dev")
        r = dev_s.pct_change().dropna().values
        sr = float(np.mean(r) / np.std(r))
        d = M2.dsr(sr, len(r), float(src["mult"]["var_sr_family_daily"]),
                   int(src["mult"]["dev_family_dsr"]["n_trials"]),
                   float(src["stats"]["dev"]["metrics"]["skew"]),
                   float(src["stats"]["dev"]["metrics"]["kurtosis"]))
        dsr_rows.append({"family": "MAIN", "config": "T0005", "config_id": "MAIN:T0005",
                         "N": 34, "dsr": d["dsr"], "sr_daily": d["sr_daily"],
                         "sr_annual": d["sr_annual"], "sr0_daily": d["sr0_daily"],
                         "sr0_annual": d["sr0_annual"], "T": d["T"], "z": d["z"]})
        write_rows(os.path.join(FINAL_DIR, "族内DSR.csv"), dsr_rows,
                   ["family", "config", "config_id", "N", "dsr", "sr_daily", "sr_annual",
                    "sr0_daily", "sr0_annual", "T", "z"])

    print(f"ledger: 追加 {added} 行；总行数 {len(rows)}；"
          f"N_sel={total['in_Nsel']:.0f} N_cell={total['in_Ncell']:.0f} "
          f"N_diag={total['in_Ndiag']:.0f} N_platform={total['in_Nplatform']:.0f}")
    return {"added": added, "rows": len(rows), "total": total,
            "by_family": by_family, "by_category": by_cat,
            "gates": gates, "errors": errors}


# ---------------------------------------------------------------------------
# 最终统计：T0005 纳入后的聚合与判定
# ---------------------------------------------------------------------------

CURATED_FAMILIES = {
    "MAIN": {
        "label": "官方主候选 T0005（F1×S1×E2×X3×G1）",
        "n_configs_registered": 34,
        "n_sel": 19,
        "n_sel_note": "T0005 主候选 + 18 个参数邻域（in_Nsel=1）；18 邻域不是 N_diag",
        "dev": {"cagr": 0.04156815775673306, "mdd": -0.0905350729763279, "trades": 103},
        "val": {"cagr": -0.025342280131277373, "mdd": -0.13797983512013767, "trades": 159},
        "verdict": "淘汰",
        "reason": "§7.9.5 两段净收益>0 不通过（val -2.53%）；15 个消融与 18 个参数邻域无一在 val 转正；PBO 0.311/0.756、项目级 DSR≈0。",
    },
    "TREND": {
        "label": "趋势族（多周期趋势/结构/触发）",
        "best_dev": {"config": "S4 纯时间60", "cagr": 0.0935, "mdd": -0.543},
        "best_val": {"config": "S3 纯时间60", "cagr": 0.0646, "mdd": -0.096},
        "caliber": "主口径 S4；族内真实最高 dev/val 即本行；S3 的 5 个两段正配置带创业板限价执行口径污染",
        "verdict": "无可冻结",
        "reason": "dev/val 符号翻转；5 个两段为正的配置全部是 S3 事后探索（无 trial_id、每窗口 44-71 笔、事件层 20 日超额中位为负、去前 5 即转负）；S3 交易 16.46% 非主板、5 个规则 16.52% 非主板，且创业板 2020-08-24 前限价档位 bug 污染执行；max-stat p 0.374/0.716（9 矩阵口径）。",
    },
    "STRUCT": {
        "label": "结构族（非重绘 HH/HL）",
        "best_pooled": 0.0116,
        "verdict": "无入场 alpha",
        "reason": "219 个配置只有 2008-2020 合并窗口，无法两窗口评级；事件层 20 日超额 -1.21%~+0.11%；tp8 正收益因丢弃左尾撤回；结构止损是最差离场。",
    },
    "EXIT": {
        "label": "退出族（18 臂 × 2 测试床）",
        "dev_range": [-0.376, -0.2014],
        "val_range": [-0.458, -0.198],
        "family_best": {"dev": -0.201996, "val": -0.1984,
                        "caliber": "18 臂，不含 B0 基准与 RB 仓位敏感性"},
        "b0_baseline": {"config": "TB1/B0", "dev": -0.2268, "val": -0.2243,
                        "type": "baseline", "excluded_from_18arm_best": True,
                        "source": "03_结果/退出_臂对照_组合级.csv"},
        "excluded_rb_sensitivity": {"config": "TB1/B0 + RB 仓位（risk 0.5%）", "dev": -0.0883, "val": None,
                                    "source": "03_结果/退出_仓位敏感.csv", "scope": "dev only",
                                    "note": "不是 B0 full-window 基准；原稿误配给 B0"},
        "verdict": "退出规则不改变符号",
        "reason": "同一入场换 18 种退出，验证窗口全负；18 臂 dev best -20.20%（S_fix）/val best -19.84%（S_struct）；B0 baseline -22.68%/-22.43%；-8.83% 是 退出_仓位敏感.csv 的 RB 仓位敏感性（风险 0.5%，仅 dev、无 val，不是 B0）；T2 跟踪把 dev 平均 R 从 +0.98 打到 -0.01；STK 六态状态机为 dev 最差臂。",
    },
    "MKT": {
        "label": "市场状态族",
        "main": {"config": "A_状态切换_偏广度", "dev": 0.1389, "val": -0.0733},
        "main_dev_config_val": -0.0733,
        "family_best_dev": {"config": "A_状态切换_偏广度", "cagr": 0.1389},
        "family_best_val": {"config": "D_串联过滤", "cagr": -0.0235},
        "caliber": "主口径是 A 偏广度（dev 13.89%/val -7.33%，main_dev_config_val）；族内真实最高 val 是 D_串联过滤 -2.35%（family_best_val）。两者分开，不得把 -7.33% 当族内最好。",
        "verdict": "不是择时 alpha",
        "reason": "dev 好、val 负；收益改善来自降 beta；全段正收益集中在 2006-2007 与 2014-2015 的暴露。",
    },
    "ECOLOGY": {
        "label": "生态切换（反转×趋势 + 连续风险预算）",
        "dev_best": -0.0013,
        "val_best": -0.022,
        "verdict": "拒绝",
        "reason": "54,070 个事件、扣除成本后 10 日超额 -0.75%/-1.38%（REV）与 -1.02%/-2.28%（BRK），按信号日聚类 t -10.9~-18.0；36 个状态条件格平均超额全负。",
    },
    "STAT": {
        "label": "统计框架夹具",
        "best": -0.0629,
        "verdict": "夹具",
        "reason": "2008-2012 参考网格，仅用于校验统计框架；不作为候选。",
    },
    "VCB": {
        "label": "收缩突破（VCB）",
        "main": {"dev": 0.0567, "val": 0.0559, "dev_mdd": -0.363, "val_mdd": -0.262},
        "best_gate": {"config": "VCB-P20/T40", "dev": 0.0651, "val": 0.0538, "dev_mdd": -0.236, "val_mdd": -0.156},
        "closest": {"config": "VCB-N120/T40", "dev": 0.083238, "val": 0.087049, "dev_mdd": -0.352753},
        "family_best_dev": {"config": "VCB-M_T60", "cagr": 0.155999, "mdd": -0.254860, "val": 0.014596},
        "family_best_val": {"config": "VCB-N120_T40", "cagr": 0.087049, "dev": 0.083238, "dev_mdd": -0.352753},
        "cost_model": "10bp 固定滑点 + 10% 参与率上限，无 0.02*sqrt(part) 冲击项；补算后 dev 约高估 0.44pp/年",
        "delisting_missing_days": 250,
        "count_reconciliation": {"declared_rows": 34, "unique_configs": 32, "n_sel": 22, "n_diag": 10, "aliases": 2,
                                 "note": "VCB-M≡VCB-M/T40；敏感性 base≡VCB-M/T40"},
        "verdict": "拒绝",
        "reason": "低个位数两段正但事件层为零（-1.08%/+0.92%，t<1）；PBO 0.614、DSR 0.622（N=24）/0.1139（族内 N=22）；随机排序（+7.20/+6.16）高于突破强度排序（+5.67/+5.59）；去前 5 后 val 仅 +0.7%/年；成本无冲击项、250 日停牌强平（002075.SZ）两项均使 VCB 偏乐观。",
    },
    "SMALLLOWVOL": {
        "label": "小盘低波趋势",
        "main": {"dev": 0.0953, "val": 0.0033, "dev_mdd": -0.557, "val_mdd": -0.411,
                 "val_cagr_nav_first": 0.0033, "val_cagr_initial_capital": -0.0003,
                 "val_first_day_return": -0.0176,
                 "val_nav_first": 982381.89, "val_nav_start_capital": 1000000.0},
        "family_best_dev": {"config": "ADVQ12_s5_h40", "cagr": 0.203991, "mdd": -0.691508, "val": 0.032787},
        "family_best_val": {"config": "COMBO_s20_h60", "cagr": 0.061825, "dev": 0.130291, "dev_mdd": -0.508762},
        "verdict": "不成立",
        "reason": ("val +0.33% 以首日收盘净值 982,382 为基准（2016-01-04 熔断日 -1.76% 未计入）；"
                   "按初始 100 万计 val -0.03%，不是“两段为正”。族内真实最高 dev +20.40%（ADVQ12_s5_h40，dev MDD -69.15%）、"
                   "最高 val +6.18%（COMBO_s20_h60，dev MDD -50.88%）；按真实最高值仍 0 个过两段 MDD≤25%。"
                   "val 的 P(CAGR≤0)=49.95%；val 去前 5 赢家 -9.42%；1 亿部署率 13.7%、CAGR -1.5%。"),
    },
    "TRENDCALLBACK": {
        "label": "趋势回调 vs 结构破坏",
        "event_l5": 0.00077,
        "main": {"dev": -0.2396, "val": -0.1978},
        "verdict": "拒绝",
        "reason": "健康回调 20 日超额 +0.077%（t=0.78，中位 -1.74%）；主候选扣费 -23.96%/-19.78%；族内真实最高 dev +1.56%（L6_BREAKOUT）、最高 val -0.77%（SF_L5）；35 个配置 0 个两段为正。",
    },
    "RELSTR": {
        "label": "横截面相对强弱（RS-T）",
        "main": {"dev": -0.112, "val": -0.363},
        "family_best_dev": {"config": "B02", "cagr": 0.0814571, "val": -0.255681},
        "family_best_val": {"config": "N02", "cagr": -0.043718, "dev": 0.037473},
        "family_best": {"dev": {"config": "B02", "cagr": 0.0814571, "mdd": -0.4254},
                        "val": {"config": "N02", "cagr": -0.043718, "mdd": -0.4976}},
        "caliber": "family_best_dev 的 val 是 B02 自己的 -25.5681%；family_best_val 的 dev 是 N02 自己的 +3.7473%。-4.37% 是 N02 的 val，不是 B02 的 val。",
        "verdict": "拒绝（反向信号）",
        "reason": "24 个 IC 全部为负；40 个配置 0 个两段同为正；族内真实最高 dev +8.15%（B02，val -25.57%）、最高 val -4.37%（N02，dev +3.75%），按真实最高值 val 仍负；负边际来自选股而非成本（毛已 -13.4%/年）。",
    },
    "BREAKPULL": {
        "label": "突破回踩 vs 立即追涨",
        "main": {"dev": -0.1902, "val": -0.1985},
        "lookahead_arm": {"dev": 0.82, "val": 1.0755},
        "family_best_dev": {"config": "邻域-STOP", "cagr": 0.033025, "mdd": -0.2553, "val": -0.050716},
        "family_best_val": {"config": "邻域-STOP", "cagr": -0.050716},
        "verdict": "拒绝",
        "reason": "回踩买价低 +0.20%/+0.61% 每笔，但政策层 dev 144 格全负、val 144 格全正（符号翻转）；账户层两段约 -19%；族内真实最高 dev +3.31%（邻域-STOP）、最高 val -5.09%；+82%/+107.5% 是前视子集，不得当候选。",
    },
    "STRUCTREV": {
        "label": "结构内短期温和反转",
        "main": {"dev": -0.2361, "val": -0.218},
        "family_best_dev": {"config": "A08_KNIFE", "cagr": -0.000421, "val": 0.007266},
        "family_best_val": {"config": "A08_KNIFE", "cagr": 0.007266, "dev": -0.000421},
        "count_reconciliation": {"n_sel": 30, "n_audit": 1, "account_rows": 31,
                                 "note": "A31_HASH 是 audit 随机对照，不是第 31 个可部署配置"},
        "verdict": "拒绝",
        "reason": "30 个 N_sel + 1 个 audit 随机对照（A31_HASH）= 31 个账户行，0 个两段为正；最好 A08_KNIFE dev -0.04%/val +0.73%（不是 A17 -5.56%/-4.96%）；按日等权 net dev -1.09%/笔、val -0.85%/笔；排序键把“回撤最浅优先”操作化成“已反弹最多优先”。",
    },
    "LONGMOM": {
        "label": "长期双动量",
        "main": {"dev": -0.0837, "val": -0.1302},
        "verdict": "无正边际",
        "reason": "26 个配置全负；月度 IC dev -0.0308（t=-1.99）；双动量退化（绝对动量过滤从不改变 Top-N）；DSR 0.038、PBO 0.020（低 PBO 只说明“最不差”稳定）。",
    },
    "EXITUNI": {
        "label": "退出统一比较（2 入口 × 15 臂）",
        "entry_a": -0.0151,
        "entry_b": -0.0087,
        "verdict": "拒绝",
        "reason": "两个公开入口扣费后 -1.51%/-0.87% 每笔；28 个非基准比较 0 个通过 6 条判定；唯一显著的相对改善（结构止损 +0.18pp）本身仍为负。",
    },
    "RISKBUDGET": {
        "label": "市场状态 → 风险预算",
        "family_best_dev": {"config": "RB001", "cagr": 0.1163, "val": -0.0314},
        "family_best_val": {"config": "RB021", "cagr": -0.0314},
        "verdict": "拒绝",
        "reason": "dev 有效、val 有害；控制当期 20 日已实现波动后偏相关 +0.036/-0.002/+0.003；状态只是“波动已经很高”的另一种说法。族内最高 dev 11.63%（RB001）是诊断叠加，不是可部署候选。",
    },
    "UNIFIEDENGINE": {
        "label": "统一引擎公开基线",
        "dev_range": [-0.330569, -0.211377],
        "val_range": [-0.472211, -0.182581],
        "benchmark_bh10": {"dev": 0.055472, "val": -0.075814, "type": "benchmark"},
        "verdict": "基线应淘汰",
        "reason": "4 个策略（T-U01..T-U04）dev -33.06%~-21.14%、val -47.22%~-18.26%，全部为负；BH10（买入持有 10 只）dev +5.55%/val -7.58% 是基准，不是策略，已单列；60 日新高突破的前向 20 日超额 11/13 年为负。",
    },
    "EXECPRESSURE": {
        "label": "执行压力与容量",
        "max_full_window": 0.121,
        "original_account_dev_range": [-0.137, -0.053],
        "original_account_val_range": [-0.275, -0.188],
        "reprice_best": {"config": "S4时间60", "dev": 0.1116, "val": -0.0209},
        "verdict": "60% 不成立",
        "reason": "1,250 个基线压力格 + 192 个候选重定格，0 格 ≥20%、0 格 ≥60%；原账户 dev -13.7%~-5.3%、val -27.5%~-18.8%；最高 12.1% 是固定交易集合重定价（另一口径），不是原账户路径。",
    },
}

# 跨族口径矩阵：宇宙 / 复权 / 引擎 / 成本 / 限制（对应报告 §3.20）。
# 标“族报告口径”的格子表示本批只核对了族报告/代码表面，没有逐笔穿透。
FAMILY_SCOPE = {
    "MAIN": {"universe": "主板 U-MAINBOARD-V1", "adjust": "A 空间", "engine": "统一引擎 UnifiedAccount",
             "cost": "C-REAL-V1", "limits": "out_date 前视 0.0158%；2 笔停牌顺延强平"},
    "TREND": {"universe": "sh.60/68 + sz.00/30（含创业板/科创板）", "adjust": "Baostock flag1",
              "engine": "族自建简化引擎", "cost": "5bp 固定滑点+印花税，无冲击/参与率上限",
              "limits": "创业板限价档位 bug；成交 15.84% 非主板；S3 16.46%；5 个两段正 S3 16.52%"},
    "STRUCT": {"universe": "主板+中小板+创业板（未排除 300/688）", "adjust": "Baostock flag1",
               "engine": "族自建简化引擎", "cost": "10bp 固定滑点+印花税，无冲击/参与率上限",
               "limits": "事件 13.93% 非主板；多数表只有 2008-2020 合并窗口；tp8 已撤回"},
    "EXIT": {"universe": "族报告口径（未统一核验）", "adjust": "Baostock flag1", "engine": "族自建简化引擎",
             "cost": "5bp 固定滑点+印花税", "limits": "两窗口组合级全负；非统一引擎"},
    "MKT": {"universe": "指数/ETF 层", "adjust": "指数价格", "engine": "族自建状态引擎",
            "cost": "族报告口径", "limits": "规则含创业板/科创板涨跌停；未在个股账户复核"},
    "ECOLOGY": {"universe": "族报告口径", "adjust": "族报告口径（未统一声明）", "engine": "族自建",
                "cost": "族报告口径", "limits": "事件层为主；组合层 14 配置全负"},
    "STAT": {"universe": "参考网格", "adjust": "族报告口径", "engine": "夹具", "cost": "族报告口径",
             "limits": "仅校验统计框架"},
    "VCB": {"universe": "主板 board='主板'", "adjust": "A 空间", "engine": "族自建组合引擎（基类 Account）",
            "cost": "10bp 固定+10% 参与率上限，无冲击项", "limits": "250 日停牌强平 1 笔；signal_mask 未实现 delist_window；delist_haircut=0 偏乐观约 0.6pp"},
    "SMALLLOWVOL": {"universe": "主板 board='主板'", "adjust": "factor 当价格（总收益指数）",
                    "engine": "族自建组合引擎（UnifiedAccount）", "cost": "C-REAL-V1",
                    "limits": "size_pit=a_close×fs_pit 规模排序被分红抬高；主单元 104 笔停牌强平"},
    "TRENDCALLBACK": {"universe": "族报告口径", "adjust": "Baostock flag1", "engine": "族自建",
                      "cost": "族报告口径", "limits": "事件层 +0.077%（t=0.78）；组合层 -23.96%/-19.78%"},
    "RELSTR": {"universe": "族报告口径", "adjust": "Baostock flag1", "engine": "族自建",
               "cost": "族报告口径", "limits": "IC 全负；40 配置 0 两段正"},
    "BREAKPULL": {"universe": "族报告口径", "adjust": "族报告口径（未统一声明）", "engine": "族自建 + 统一引擎复核",
                  "cost": "族报告口径", "limits": "nopull 前视；政策层符号翻转"},
    "STRUCTREV": {"universe": "族报告口径", "adjust": "族报告口径（未统一声明）", "engine": "族自建",
                  "cost": "族报告口径", "limits": "排序键操作化错误；30 N_sel + 1 audit"},
    "LONGMOM": {"universe": "族报告口径", "adjust": "Baostock flag1", "engine": "族自建", "cost": "族报告口径",
                "limits": "26 配置全负；val CAGR 与 NAV 反算差 1.2-1.8pp"},
    "EXITUNI": {"universe": "族报告口径", "adjust": "族报告口径（未统一声明）", "engine": "族自建",
                "cost": "族报告口径", "limits": "两入口扣费为负"},
    "RISKBUDGET": {"universe": "指数层 + 状态", "adjust": "指数价格", "engine": "族自建", "cost": "族报告口径",
                   "limits": "状态≈波动代理；偏相关≈0"},
    "UNIFIEDENGINE": {"universe": "主板", "adjust": "A 空间", "engine": "统一引擎", "cost": "C-REAL-V1",
                      "limits": "仅 4 个公开基线；BH10 是基准不是策略"},
    "EXECPRESSURE": {"universe": "主板", "adjust": "A 空间 + 固定交易集合", "engine": "平台重定价",
                     "cost": "5bp 档", "limits": "重定价 ≠ 原账户路径"},
}


def _gate_rows() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FINAL_DIR, "统一门槛.csv"))


def audit_facts() -> dict:
    """从只看结果表重算审计整改所需的关键事实（不运行策略、不碰 2021+）。"""
    g = _gate_rows()
    g = g.dropna(subset=["dev_cagr", "val_cagr"]).copy()
    g["min2"] = g[["dev_cagr", "val_cagr"]].min(axis=1)
    bd = g.loc[g.dev_cagr.idxmax()]
    bv = g.loc[g.val_cagr.idxmax()]
    bm = g.loc[g.min2.idxmax()]
    pos = g[(g.dev_cagr > 0) & (g.val_cagr > 0)]
    pos_by_fam = {k: int(v) for k, v in pos.groupby("family").size().items()}
    gpass = g[(g.dev_cagr > 0) & (g.val_cagr > 0) & (g.dev_mdd >= -0.25) & (g.val_mdd >= -0.25)]
    # 5 个 SEL-G 通过者的 Calmar / 最差滚动 12 月
    gpass_info = []
    for _, r in gpass.iterrows():
        gpass_info.append({"family": r.family, "config_id": r.config_id, "strategy_id": r.strategy_id,
                           "dev_cagr": float(r.dev_cagr), "dev_mdd": float(r.dev_mdd),
                           "val_cagr": float(r.val_cagr), "val_mdd": float(r.val_mdd),
                           "dev_calmar": float(r.dev_cagr / abs(r.dev_mdd)) if r.dev_mdd else None,
                           "val_calmar": float(r.val_cagr / abs(r.val_mdd)) if r.val_mdd else None})
    # max-stat p（最终判定.json）
    fv = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    ms = fv.get("statistics", {}).get("maxstat", []) or []
    pvals = [{"family": x.get("family"), "window": x.get("window"), "p_maxstat": x.get("p_maxstat"),
              "best_config": x.get("best_config"), "best_cagr": x.get("best_cagr")} for x in ms]
    valid = [x for x in pvals if x["p_maxstat"] is not None]
    pmin = min(valid, key=lambda x: x["p_maxstat"]) if valid else None
    # T0005 top5 分母
    tdev = pd.read_csv(os.path.join(T0005_DIR, "02_逐笔_T0005_dev.csv"))
    tval = pd.read_csv(os.path.join(T0005_DIR, "02_逐笔_T0005_val.csv"))

    def top5(df):
        pnl = df["pnl"].astype(float)
        return {"top5": float(pnl.nlargest(5).sum()), "net": float(pnl.sum()),
                "gross_profit": float(pnl[pnl > 0].sum()),
                "top5_over_net": float(pnl.nlargest(5).sum() / pnl.sum()) if pnl.sum() else None,
                "top5_over_gross": float(pnl.nlargest(5).sum() / pnl[pnl > 0].sum()) if (pnl > 0).any() else None}
    # 邻域 dev/val 区间
    nb = pd.read_csv(os.path.join(T0005_DIR, "30_邻域.csv"))
    nb_dev = nb[nb.window == "dev"]["cagr"].astype(float)
    nb_val = nb[nb.window == "val"]["cagr"].astype(float)
    # 45 空仓率 / 平均持仓（S3）
    s3 = []
    import glob
    for f in sorted(glob.glob(os.path.join(RES, "趋势族", "净值_S3_*.parquet"))):
        d = pd.read_parquet(f)
        s3.append({"file": os.path.basename(f), "flat_share": float((d.n_pos == 0).mean()),
                   "avg_positions": float(d.n_pos.mean())})
    # TREND 非主板成交占比
    tr = pd.read_parquet(os.path.join(RES, "趋势族", "交易明细_全部.parquet"))
    def _board(code):
        c = str(code).split(".")[-1]
        return "GEM" if c.startswith("30") else ("STAR" if c.startswith("68") else "MAIN")
    tb = tr["code"].map(_board)
    s3tr = tr[tr["signal"].astype(str).str.contains("S3")]
    s3b = s3tr["code"].map(_board)
    rules5 = ["纯时间60", "时间40+止损2ATR", "目标2ATR+止损2", "宽止损4ATR+时间60", "纯时间20"]
    s3r = s3tr[s3tr["rule"].isin(rules5)]
    s3rb = s3r["code"].map(_board)
    # VCB 34 vs 32
    led = pd.read_csv(LEDGER)
    v = led[led.family == "VCB"]
    vcb_nsel = int(v.in_Nsel.sum()); vcb_diag = int(v.in_Ndiag.sum())
    # 2021+ 缓存
    caches = []
    for rel in ["03_结果/市场状态_缓存/index_feat.parquet", "03_结果/市场状态_缓存/market_feature.parquet"]:
        f = os.path.join(ROOT, rel)
        if os.path.exists(f):
            cols = [c for c in pd.read_parquet(f).columns if "ret" in str(c)]
            caches.append({"path": rel, "return_like_columns": cols[:6]})
    return {
        "best_dev": {"config_id": bd.config_id, "strategy_id": bd.strategy_id, "family": bd.family,
                     "cagr": float(bd.dev_cagr), "mdd": float(bd.dev_mdd), "val_cagr": float(bd.val_cagr)},
        "best_val": {"config_id": bv.config_id, "strategy_id": bv.strategy_id, "family": bv.family,
                     "cagr": float(bv.val_cagr), "dev_cagr": float(bv.dev_cagr), "dev_mdd": float(bv.dev_mdd)},
        "best_min": {"config_id": bm.config_id, "strategy_id": bm.strategy_id, "family": bm.family,
                     "min_cagr": float(bm.min2), "dev_cagr": float(bm.dev_cagr), "val_cagr": float(bm.val_cagr),
                     "dev_mdd": float(bm.dev_mdd), "val_mdd": float(bm.val_mdd)},
        "n_two_segment_positive": int(len(pos)),
        "two_segment_positive_by_family": pos_by_fam,
        "g_pass": gpass_info,
        "maxstat": {"n_p": len(valid), "min": pmin, "values": pvals},
        "top5": {"dev": top5(tdev), "val": top5(tval)},
        "neighborhood": {"dev_min": float(nb_dev.min()), "dev_max": float(nb_dev.max()),
                         "val_min": float(nb_val.min()), "val_max": float(nb_val.max())},
        "s3_exposure": {"flat_min": min(x["flat_share"] for x in s3), "flat_max": max(x["flat_share"] for x in s3),
                        "pos_min": min(x["avg_positions"] for x in s3), "pos_max": max(x["avg_positions"] for x in s3)},
        "trend_non_main": {"all_trades": int(len(tr)), "non_main": int((tb != "MAIN").sum()),
                           "share": float((tb != "MAIN").mean()),
                           "s3_trades": int(len(s3tr)), "s3_non_main": int((s3b != "MAIN").sum()),
                           "s3_share": float((s3b != "MAIN").mean()),
                           "s3_5rules_trades": int(len(s3r)), "s3_5rules_non_main": int((s3rb != "MAIN").sum()),
                           "s3_5rules_share": float((s3rb != "MAIN").mean())},
        "vcb_count": {"declared_rows": 34, "unique_configs": vcb_nsel + vcb_diag,
                      "n_sel": vcb_nsel, "n_diag": vcb_diag, "aliases": 2},
        "structrev_count": {"n_sel": int(led[led.family == "STRUCTREV"].in_Nsel.sum()),
                            "n_audit": int(led[(led.family == "STRUCTREV") & (led.kind == "audit")].strategy_id.nunique()),
                            "account_rows": int(led[led.family == "STRUCTREV"].in_Nsel.sum())
                                            + int(led[(led.family == "STRUCTREV") & (led.kind == "audit")].strategy_id.nunique())},
        "caches_2021": caches,
        "relstr": _relstr_fields(g),
        "main_count": {"n_sel": int(led[led.family == "MAIN"].in_Nsel.sum())},
        "exit_best": _exit_best(g),
        "g4": _g4_range(),
        "seal": _seal_facts(),
        "transfer_fee": _transfer_fee_facts(),
        "disposition": _disposition_facts(),
        "evidence_matrix_banner": _evidence_matrix_banner(),
    }


def _relstr_fields(g):
    r = g[g.family == "RELSTR"]
    b02 = r[r.config_id.astype(str).str.contains("B02")]
    n02 = r[r.config_id.astype(str).str.contains("N02")]
    return {"B02": {"dev": float(b02.dev_cagr.iloc[0]), "val": float(b02.val_cagr.iloc[0])} if len(b02) else None,
            "N02": {"dev": float(n02.dev_cagr.iloc[0]), "val": float(n02.val_cagr.iloc[0])} if len(n02) else None}


def _exit_best(g):
    e = g[g.family == "EXIT"].dropna(subset=["dev_cagr", "val_cagr"])
    return {"dev": float(e.dev_cagr.max()), "val": float(e.val_cagr.max()), "n": int(len(e))}


def _g4_range():
    cfgs = [("纯时间20", os.path.join(RES, "趋势族", "净值_S3_波动收缩突破_纯时间20_dev.parquet")),
            ("宽止损4ATR+时间60", os.path.join(RES, "趋势族", "净值_S3_波动收缩突破_宽止损4ATR+时间60_dev.parquet")),
            ("时间40+止损2ATR", os.path.join(RES, "趋势族", "净值_S3_波动收缩突破_时间40+止损2ATR_dev.parquet")),
            ("目标2ATR+止损2", os.path.join(RES, "趋势族", "净值_S3_波动收缩突破_目标2ATR+止损2_dev.parquet"))]
    vals = []
    for k, f in cfgs:
        d = pd.read_parquet(f)
        s = d.set_index("date")["equity"].astype(float)
        vals.append((k, float((s / s.shift(252) - 1).min())))
    f = os.path.join(RES, "候选_收缩突破", "组合_净值_VCB-P20_T40_2008_2015.csv")
    d = pd.read_csv(f, parse_dates=["date"])
    s = d.set_index("date")["nav"].astype(float)
    vals.append(("VCB-P20_T40", float((s / s.shift(252) - 1).min())))
    return {"values": vals, "min": min(v for _, v in vals), "max": max(v for _, v in vals)}


def _seal_facts():
    path = os.path.join(ROOT, "05_审计", "2021+未评估文件封存清单.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, encoding="utf-8-sig")
    grade = d["封存等级"].astype(str)
    perf = d["含策略绩效列"].astype(str).str.strip().isin(["否", "False", "false"])
    return {"rows": int(len(d)), "A": int(grade.str.startswith("A").sum()),
            "B": int(grade.str.startswith("B").sum()), "C": int(grade.str.startswith("C").sum()),
            "sha_nonempty": int(d["SHA256"].astype(str).str.len().ge(64).sum()),
            "strategy_performance_zero": bool(perf.all())}


def _transfer_fee_facts():
    code = ""
    p = os.path.join(CODE, "统一回测.py")
    if os.path.exists(p):
        code = open(p, encoding="utf-8").read()
    return {"code_default_0_0010": "transfer_par_rate: float = 0.0010" in code,
            "code_comment_0_0006_sensitivity": "0.0006 只做敏感性" in code,
            "main_is_0_06": False}


def _disposition_facts():
    path = os.path.join(ROOT, "05_审计", "审计问题处置表.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, encoding="utf-8-sig")
    text = d.astype(str).agg(" ".join, axis=1)
    return {"rows": int(len(d)),
            "needs_sync_rows": int(text.str.contains("需再同步|待同步|未同步").sum())}


def _evidence_matrix_banner():
    path = os.path.join(ROOT, "05_审计", "最终结论证据矩阵.md")
    if not os.path.exists(path):
        return False
    head = "\n".join(open(path, encoding="utf-8").read().splitlines()[:15])
    return ("505/228" in head) and ("486/228" in head)


# ---------------------------------------------------------------------------
# 审计整改清单（四份最终审计的逐项处置）
# disposition: FIXED / FIXED_ROOT_DOCS_SUPERSEDE / LIMITATION_DECLARED
# ---------------------------------------------------------------------------
REMEDIATION = [
    # 最终数值审计 F01-F19
    {"source": "最终数值审计", "id": "F01", "severity": "HIGH", "disposition": "FIXED",
     "claim": "族级“最好 dev / 最好 val”表把主候选当家族最好",
     "where": ["01_完整报告.md §3.19", "04_最终结论.json families[].family_best_*"],
     "evidence": "§3.19 拆成“主口径 / 族内真实最高”；真实最高 dev +20.40%（ADVQ12_s5_h40）、val +8.70%（N120_T40）；8 个族的偏差逐行列明。",
     "check": "family_best_matches_gate_table"},
    {"source": "最终数值审计", "id": "F02", "severity": "HIGH", "disposition": "FIXED",
     "claim": "“全项目 dev 最好 +9.35%（TREND S4）”",
     "where": ["02_规则手册_研究结论版.md R1", "01_完整报告.md §0/§3.19"],
     "evidence": "改为 +20.40%（小盘 ADVQ12_s5_h40）；+9.35% 明确标注为 TREND 族内最好。",
     "check": "report_says_project_best_dev"},
    {"source": "最终数值审计", "id": "F03", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "“全项目两段为正只有 24 个”",
     "where": ["01_完整报告.md §3.19/§4 Q6/§10", "00_一页纸.md", "04_最终结论.json questions[Q6]"],
     "evidence": "统一门槛 478 行重算：32 行 = TREND 5 + VCB 18 + 小盘 9；24 只在 VCB 非敏感性 14 单元子集成立。",
     "check": "two_segment_positive_32"},
    {"source": "最终数值审计", "id": "F04", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "G2/G6/G8 单元格自相矛盾",
     "where": ["00_一页纸.md 60% 裁决", "01_完整报告.md §6.6"],
     "evidence": "G2：5 个 SEL-G 通过者全部过 35%（−9.45/−21.20/−9.97/−20.26/−23.56），−36.3% 的 M_T40 不在 5 个内；G6：T0005 0.9585 / 可用最高 VCB-N120 0.1139，0.84 是 dsr_null_p95；G8：0.15%-5.97%、多数为负。",
     "check": "gate_table_recomputed"},
    {"source": "最终数值审计", "id": "F05", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "T0005 top5 share 混用毛/净分母",
     "where": ["01_完整报告.md §6.5", "04_最终结论.json statistics.top5_denominators"],
     "evidence": "T0005 dev 45.9%（净）/31.1%（毛）；val −62.6%（净）/40.9%（毛）；全表主数统一为净 PnL。",
     "check": "top5_two_denominators"},
    {"source": "最终数值审计", "id": "F06", "severity": "MODERATE", "disposition": "FIXED_ROOT_DOCS_SUPERSEDE",
     "claim": "留出准入裁决的理由已过期（6 行/从未运行/N=365）",
     "where": ["00_一页纸.md 留出", "01_完整报告.md §7", "04_最终结论.json holdout.supersedes"],
     "evidence": "根文档与 JSON 改为当前事实（10 行、T0005 rev2 done discard、T0007 val done discard、无通过 val 的父试验、N=733）；裁决文件为 v2 修订版（02:12），00:37 初版在附录 A/initial_snapshot。",
     "check": "rereview_holdout_v2"},
    {"source": "最终数值审计", "id": "F07", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "N 超预算 11.45× 的分子分母不同口径",
     "where": ["00_一页纸.md 关键数字", "01_完整报告.md §6.1", "README.md §1.1", "04_最终结论.json statistics.budget"],
     "evidence": "写明预算 40/64、超出 18.33×/11.45×、N_sel 不含 30 个消融；(733+30)/64=11.92× 也给出。",
     "check": "budget_two_calibers"},
    {"source": "最终数值审计", "id": "F08", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "VCB 34 vs 总账 22+10=32",
     "where": ["01_完整报告.md §3.8/§3.19", "README.md §1.2", "04_最终结论.json families[VCB].count_reconciliation"],
     "evidence": "34 = 4 可行性 + 15 实验 + 6 消融 + 9 敏感性；去重后 32 = 22 N_sel + 10 诊断；差额 2 是别名（VCB-M≡M/T40；base≡M/T40）。",
     "check": "vcb_count_reconciled"},
    {"source": "最终数值审计", "id": "F09", "severity": "LOW", "disposition": "FIXED",
     "claim": "18 邻域 dev 区间 2.11%~5.50%",
     "where": ["01_完整报告.md §2.5/§5.1", "04_最终结论.json statistics.neighborhood_range"],
     "evidence": "30_邻域.csv min=1.5396%（S06）；改为 1.54%~5.50%。",
     "check": "neighborhood_range_154"},
    {"source": "最终数值审计", "id": "F10", "severity": "LOW", "disposition": "FIXED",
     "claim": "主宇宙日均 1,266-2,183 只",
     "where": ["01_完整报告.md §5.6", "README.md §4"],
     "evidence": "01_信号覆盖.csv：2008 年 686.47、2020 年 2,182.93；改为 686-2,183。",
     "check": "universe_min_686"},
    {"source": "最终数值审计", "id": "F11", "severity": "LOW", "disposition": "FIXED",
     "claim": "“OHLC 差异 >0.005 元仅 4 行”",
     "where": ["01_完整报告.md §8", "03_数据与复现说明.md §1", "04_最终结论.json data"],
     "evidence": "收盘 4 行；任一 OHLC 5 行（688206.SH 的 2021 年 12 月 28 日 high 56.20 vs 55.00）。",
     "check": "ohlc_close_vs_any"},
    {"source": "最终数值审计", "id": "F12", "severity": "LOW", "disposition": "FIXED",
     "claim": "v1 坏行 94.10%、最大偏差 227pp",
     "where": ["01_完整报告.md §8", "03_数据与复现说明.md §2.2", "02_规则手册 R7", "04_最终结论.json data.v1_defect"],
     "evidence": "94.10%（v1 构建 15,693,248/16,676,415）与 94.01%（审计全样本 15,693,245/16,692,984）并列；最大 170.1（收益小数，000631）；227pp 无可复现定义、撤回。",
     "check": "v1_two_denominators"},
    {"source": "最终数值审计", "id": "F13", "severity": "LOW", "disposition": "FIXED",
     "claim": "T0005 “退市强制退出 0/2 笔”",
     "where": ["01_完整报告.md §5.11", "04_最终结论.json cross_cutting.must_exit"],
     "evidence": "实为停牌顺延后强制估值卖出（reason=sell_persistence_10d），delist_forced=0/0、forced_sells=0/2。",
     "check": "delist_label_fixed"},
    {"source": "最终数值审计", "id": "F14", "severity": "LOW", "disposition": "FIXED",
     "claim": "max-stat 机器表没有 p 值、只覆盖 9 矩阵",
     "where": ["01_完整报告.md §6.3", "04_最终结论.json statistics.maxstat"],
     "evidence": "16 个 p 全部回填；全局最小 0.071964（VCB dev）；9 矩阵口径 0.374 单列；最终PBO_maxstat.csv 无 p 列的事实已写明。",
     "check": "maxstat_global_min"},
    {"source": "最终数值审计", "id": "F15", "severity": "LOW", "disposition": "FIXED",
     "claim": "执行压力 1 日延迟 dev −11.4%~−13.7%",
     "where": ["01_完整报告.md §5.5", "04_最终结论.json cross_cutting.delay"],
     "evidence": "B55H20 100 万/10 仓 delay=1 的 5 个种子 −8.31%~−15.61%；全部配置 −2.42%~−20.10%。",
     "check": "delay_range_fixed"},
    {"source": "最终数值审计", "id": "F16", "severity": "LOW", "disposition": "FIXED",
     "claim": "60% 算术 S4 +4.67%/笔、2.75 次/槽/年",
     "where": ["01_完整报告.md §4 Q6", "02_规则手册 R8"],
     "evidence": "S4 纯时间60：246 笔/10 槽/7.99 年=3.0771 次/槽/年、+4.7128%/笔、暴露 76.55%；需 15.3%（对数）/19.5%（简单）每笔或约 10 次/槽/年。",
     "check": "sixty_arithmetic_fixed"},
    {"source": "最终数值审计", "id": "F17", "severity": "LOW", "disposition": "FIXED",
     "claim": "小盘分段链 +5.90%、差 2.72pp",
     "where": ["01_完整报告.md §3.9"],
     "evidence": "族报告 5.90/2.72，按首尾 NAV 链复算 5.93/2.75，差 0.03pp，已注明两种口径。",
     "check": "small_chain_footnote"},
    {"source": "最终数值审计", "id": "F18", "severity": "LOW", "disposition": "FIXED",
     "claim": "STRUCTREV 31 个账户配置（N_sel=30）",
     "where": ["01_完整报告.md §3.13/§3.19", "04_最终结论.json families[STRUCTREV].count_reconciliation"],
     "evidence": "30 个 N_sel + 1 个 audit 随机对照（A31_HASH）= 31 个账户行；差额 1 的去向已写明。",
     "check": "structrev_count_reconciled"},
    {"source": "最终数值审计", "id": "F19", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "UNIFIEDENGINE 把 dev/val 区间写反",
     "where": ["01_完整报告.md §3.16/§3.19", "04_最终结论.json families[UNIFIEDENGINE]"],
     "evidence": "4 策略 dev −33.06%~−21.14%、val −47.22%~−18.26%；BH10 dev +5.55%/val −7.58% 单列基准。",
     "check": "bh10_separated"},

    # 最终交付审计 M01-M23
    {"source": "最终交付审计", "id": "M01", "severity": "HIGH", "disposition": "FIXED",
     "claim": "“所有结论只覆盖主板”与 TREND/STRUCT 实际成交不符",
     "where": ["01_完整报告.md §1.4/§3.20/§5.6", "02_规则手册 §1", "README.md §4", "00_一页纸.md"],
     "evidence": "改为分族声明 + 新增 §3.20 矩阵：TREND 成交 15.84% 非主板、STRUCT 事件 13.93% 非主板；T0005/VCB/小盘为主板。",
     "check": "scope_not_all_mainboard"},
    {"source": "最终交付审计", "id": "M02", "severity": "HIGH", "disposition": "FIXED",
     "claim": "TREND 创业板限价档位 bug 污染 5 个 S3 正结果",
     "where": ["01_完整报告.md §3.1/§8.9/§9.2", "03_数据与复现说明.md §3.3", "04_最终结论.json limits"],
     "evidence": "披露 2020-08-24 前创业板应为 10%、代码 20%；17-18 个涨停买入/45 个跌停卖出；S3 826 笔中 136 笔、5 个规则 581 笔中 96 笔非主板；S3 标为“执行口径污染”，未重跑。",
     "check": "trend_limit_disclosed"},
    {"source": "最终交付审计", "id": "M03", "severity": "HIGH", "disposition": "FIXED",
     "claim": "G6 “族内最高 0.84”与最终统计矛盾",
     "where": ["00_一页纸.md G6", "01_完整报告.md §6.3/§6.6"],
     "evidence": "T0005 0.9585（不可用）、可用最高 VCB-N120 0.1139；0.84=dsr_null_p95=0.8407 不是 DSR；VCB 自身 DSR 0.622（N=24）。",
     "check": "g6_dsr_fixed"},
    {"source": "最终交付审计", "id": "M04", "severity": "HIGH", "disposition": "FIXED",
     "claim": "max-stat 最小值写成 0.374，实际 0.071964",
     "where": ["00_一页纸.md 关键数字", "01_完整报告.md §6.3", "04_最终结论.json statistics.maxstat"],
     "evidence": "全局最小 0.071964（VCB dev）；9 矩阵最小 0.374（trend_dev）；两者都写明，均未过 0.05。",
     "check": "maxstat_global_min"},
    {"source": "最终交付审计", "id": "M05", "severity": "HIGH", "disposition": "FIXED_ROOT_DOCS_SUPERSEDE",
     "claim": "留出理由仍是 no done val，与 T0007 冲突",
     "where": ["00_一页纸.md", "01_完整报告.md §7", "04_最终结论.json holdout"],
     "evidence": "改为“无通过 val 的父试验 + 二者均 discard + 事后补登记”；supersedes 指向 initial_snapshot；裁决文件为 v2 修订版（02:12）+ 附录 A 初版。",
     "check": "rereview_holdout_v2"},
    {"source": "最终交付审计", "id": "M06", "severity": "HIGH", "disposition": "FIXED",
     "claim": "跨族引擎/价格口径不一致未进根文档",
     "where": ["01_完整报告.md §1.4/§3.20", "README.md §1.2/§8", "04_最终结论.json cross_family_matrix"],
     "evidence": "新增 18 行 × 宇宙/复权/引擎/成本/限制矩阵；§1.2“换引擎后剩 0”限定为 4 个公开基线 + T0005；跨族表不得横向排名。",
     "check": "cross_matrix_complete"},
    {"source": "最终交付审计", "id": "M07", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "VCB 34 vs 22；README 断链到 §1",
     "where": ["README.md §1.2", "01_完整报告.md §3.8"],
     "evidence": "指针改到 §3.8，并给出 34→32 的别名映射。",
     "check": "vcb_count_reconciled"},
    {"source": "最终交付审计", "id": "M08", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "过户费文档 0.1% vs 代码 0.06%",
     "where": ["03_数据与复现说明.md §3.2"],
     "evidence": "按代码统一为面额 0.06%（每股 0.0006 元），并量化影响 0.001%-0.02%/年。",
     "check": "transfer_fee_aligned"},
    {"source": "最终交付审计", "id": "M09", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "60% 算术 21.8%/12.8 次重复计入部署率",
     "where": ["01_完整报告.md §4 Q6", "02_规则手册 R8"],
     "evidence": "3.08 次/槽/年、+4.71%/笔；需 15.3%/19.5% 每笔或约 10 次/槽/年；原口径撤回。",
     "check": "sixty_arithmetic_fixed"},
    {"source": "最终交付审计", "id": "M10", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "邻域 dev 区间 2.11%~5.50%",
     "where": ["01_完整报告.md §2.5/§5.1"],
     "evidence": "1.54%~5.50%（S06 最低）。",
     "check": "neighborhood_range_154"},
    {"source": "最终交付审计", "id": "M11", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "2021+ 间接风险只披露 1 个 state/pos 文件",
     "where": ["01_完整报告.md §7", "03_数据与复现说明.md §4", "README.md §5", "04_最终结论.json holdout.exposure_paths"],
     "evidence": "列出 3 个路径：state/pos CSV、index_feat.parquet（ret1）、market_feature.parquet（ew_ret1/csd_ret1）；封存仍未做，作为 LIMITATION 单列。",
     "check": "caches_listed"},
    {"source": "最终交付审计", "id": "M12", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "文件清单缺审计 JSON/CSV",
     "where": ["README.md §8/§9", "03_数据与复现说明.md §6"],
     "evidence": "采纳审计建议的第二选项：README §9 明示收录规则——05_审计/ 只收 *.md 报告类审计，*.json/*.csv 不收录，机器证据在对应目录保留；04_证据/*.json 全收。清单由 交付验收.py 生成（本次 350→随文件变化），不是逐字节全量。",
     "check": "inventory_disclosed"},
    {"source": "最终交付审计", "id": "M13", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "预算只写 64，不写 40",
     "where": ["00_一页纸.md", "01_完整报告.md §6.1", "README.md §1.1", "04_最终结论.json statistics.budget"],
     "evidence": "40/64 两口径 + 18.33×/11.45×。",
     "check": "budget_two_calibers"},
    {"source": "最终交付审计", "id": "M14", "severity": "LOW", "disposition": "FIXED",
     "claim": "G2 行“5 个过 25% 里 1 个 dev −36.3%”逻辑不通",
     "where": ["01_完整报告.md §6.6"],
     "evidence": "5 个全部过 35%（−9.45~−23.56）；−36.3% 的 M_T40 不在 5 个内。",
     "check": "gate_table_recomputed"},
    {"source": "最终交付审计", "id": "M15", "severity": "LOW", "disposition": "FIXED",
     "claim": "G8 一页纸 0.7%~3.9% vs 报告 0.15%-3.9%",
     "where": ["00_一页纸.md G8", "01_完整报告.md §6.6"],
     "evidence": "统一为 0.15%-5.97%，并注明多数为负。",
     "check": "g8_range_fixed"},
    {"source": "最终交付审计", "id": "M16", "severity": "LOW", "disposition": "FIXED",
     "claim": "README 85 行/30 护栏 vs 审计时实际 87/32",
     "where": ["README.md §5"],
     "evidence": "改为“92 行 / 36 护栏 / 1 条未分类说明字符串”（整改后重跑 L4 的当前值；审计时 87/32、原稿 85/30 均已废）。",
     "check": "readme_92_36"},
    {"source": "最终交付审计", "id": "M17", "severity": "LOW", "disposition": "FIXED",
     "claim": "“敏感性格”错字",
     "where": ["00_一页纸.md", "01_完整报告.md", "04_最终结论.json"],
     "evidence": "统一为“敏感性网格”。",
     "check": "typo_fixed"},
    {"source": "最终交付审计", "id": "M18", "severity": "LOW", "disposition": "FIXED",
     "claim": "“数据层可以信”过于绝对",
     "where": ["03_数据与复现说明.md §9"],
     "evidence": "改为“复权/价格层可以信；ST、退市、成交量、停牌语义、Baostock 重定基仍有限制”。",
     "check": "data_trust_nuanced"},
    {"source": "最终交付审计", "id": "M19", "severity": "LOW", "disposition": "FIXED",
     "claim": "“差一个数量级”实为 6.9 倍",
     "where": ["00_一页纸.md", "01_完整报告.md §0", "02_规则手册 §5"],
     "evidence": "改为“约 7 倍”（60/8.70）。",
     "check": "no_order_of_magnitude"},
    {"source": "最终交付审计", "id": "M20", "severity": "LOW", "disposition": "FIXED",
     "claim": "v1 斜率 2.012 未注明截尾",
     "where": ["03_数据与复现说明.md §2.2"],
     "evidence": "全样本 2.6196 / 截尾 2.0119 并列。",
     "check": "v1_slope_fixed"},
    {"source": "最终交付审计", "id": "M21", "severity": "LOW", "disposition": "FIXED",
     "claim": "Baostock 重定基 63 证券日 vs 自检 79 行",
     "where": ["01_完整报告.md §8", "03_数据与复现说明.md §7", "README.md §8"],
     "evidence": "63 个证券日 >0.5pp（39 只）与广义 79 行 / 16,682,010 行并列。",
     "check": "baostock_63_79"},
    {"source": "最终交付审计", "id": "M22", "severity": "LOW", "disposition": "FIXED",
     "claim": "“上市 ≥250 日”漏掉 td_since_list ≥120",
     "where": ["01_完整报告.md §1.4", "README.md §4", "02_规则手册 §1"],
     "evidence": "补全 `td_since_list ≥ 120` 且 `days_since_ipo ≥ 250`。",
     "check": "listing_rules_complete"},
    {"source": "最终交付审计", "id": "M23", "severity": "LOW", "disposition": "FIXED_ROOT_DOCS_SUPERSEDE",
     "claim": "证据矩阵 P03 仍写“最终判定报 486/228”",
     "where": ["01_完整报告.md §6.1", "04_最终结论.json holdout.gate_coverage"],
     "evidence": "根文档统一为 505/228（478 行、01:28 口径）；证据矩阵属他人审计文件，不改写，以根文档为准。",
     "check": "gate_coverage_505"},

    # 最终反方审计 F01-F16
    {"source": "最终反方审计", "id": "F01", "severity": "HIGH", "disposition": "FIXED",
     "claim": "分族“最好”表把主候选当家族最好", "where": ["01_完整报告.md §3.19"], "evidence": "同数值审计 F01。", "check": "family_best_matches_gate_table"},
    {"source": "最终反方审计", "id": "F02", "severity": "HIGH", "disposition": "FIXED",
     "claim": "“没有一条给出可复现的正净边际”与 24 个两段为正自相矛盾", "where": ["01_完整报告.md §10"], "evidence": "改为“0 个通过全部硬门槛；32 个两段为正未过事件层/去尾/多重性/风险”。", "check": "report_no_absolute_claim"},
    {"source": "最终反方审计", "id": "F03", "severity": "HIGH", "disposition": "FIXED",
     "claim": "G2/G4 门槛行写反；G3 未定义候选", "where": ["00_一页纸.md 60% 裁决", "01_完整报告.md §6.6"], "evidence": "G2：5 个全过 35%；G4：5 个全过（−6.44%~−18.22%）；G3：限定“5 个 SEL-G 通过者”dev Calmar 0.05-0.28 / val 0.18-0.36，并列出 VCB-M_T60 0.61/T0005-V7 1.35 不属 5 个。", "check": "rereview_g4_range"},
    {"source": "最终反方审计", "id": "F04", "severity": "HIGH", "disposition": "FIXED_ROOT_DOCS_SUPERSEDE",
     "claim": "留出裁决书事实过期", "where": ["04_最终结论.json holdout.supersedes", "01_完整报告.md §7"], "evidence": "同数值 F06；裁决文件为 v2 修订版（02:12），00:37 初版在附录 A/initial_snapshot。", "check": "rereview_holdout_v2"},
    {"source": "最终反方审计", "id": "F05", "severity": "HIGH", "disposition": "FIXED",
     "claim": "UNIFIEDENGINE 把 BH10 基准混进策略区间", "where": ["01_完整报告.md §3.16/§3.19", "04_最终结论.json"], "evidence": "策略区间 −33.06~−21.14 / −47.22~−18.26；BH10 单列 +5.55/−7.58。", "check": "bh10_separated"},
    {"source": "最终反方审计", "id": "F06", "severity": "HIGH", "disposition": "LIMITATION_DECLARED",
     "claim": "2021+ 状态序列未封存", "where": ["01_完整报告.md §7/§8", "03_数据与复现说明.md §4", "README.md §5"], "evidence": "逻辑封存已完成（34 文件、A3/B7/C24、逐文件 SHA-256、0 策略绩效、禁止用途）；物理隔离/独立保管未做，作为未关闭残余风险声明。", "check": "rereview_seal_logical"},
    {"source": "最终反方审计", "id": "F07", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "T0005 top5 分母与同表其他族不一致", "where": ["01_完整报告.md §6.5"], "evidence": "45.9%（净）/31.1%（毛）；val −62.6%/40.9%。", "check": "top5_two_denominators"},
    {"source": "最终反方审计", "id": "F08", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "max-stat 只覆盖 9 矩阵；最终表没有 p 值", "where": ["01_完整报告.md §6.3", "04_最终结论.json statistics.maxstat"], "evidence": "16 个 p 回填 + 9/16 两口径 + 全局最小 0.071964。", "check": "maxstat_global_min"},
    {"source": "最终反方审计", "id": "F09", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "DSR 跨 N 口径不单调，报告未解释", "where": ["01_完整报告.md §6.2", "04_最终结论.json statistics.dsr_cross_caliber"], "evidence": "写明 365 的 DSR 3.00e−4 / var_sr 1.34e−3 / SR 0.466，733 的 6.63e−4 / 1.116e−3 / 0.498；只有同候选集/同 var_sr 内才随 N 单调。", "check": "dsr_cross_caliber"},
    {"source": "最终反方审计", "id": "F10", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "S3 “46-50% 空仓 / 1.3-2.1 只”不准", "where": ["01_完整报告.md §3.1"], "evidence": "按 14 个 S3 净值重算：全空仓 46.2%-83.8%、平均持仓 0.39-2.13；纯时间60 暴露最高。", "check": "s3_exposure_fixed"},
    {"source": "最终反方审计", "id": "F11", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "STRUCTREV “最好 A17”不是最好", "where": ["01_完整报告.md §3.13/§3.19"], "evidence": "最好 A08_KNIFE −0.04/+0.73；A17 −5.56/−4.96 是另一配置。", "check": "structrev_best_a08"},
    {"source": "最终反方审计", "id": "F12", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "VCB “缺 delist_window 方向保守”不成立", "where": ["01_完整报告.md §3.8/§8.9", "02_规则手册 §3.1"], "evidence": "改为“方向未证明”；基准 delist_haircut=0 偏乐观，delist30 val 4.96% vs 5.59%（−0.63pp）。", "check": "vcb_conservative_removed"},
    {"source": "最终反方审计", "id": "F13", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "一页纸 G3/G6/G7/G8 范围过宽", "where": ["00_一页纸.md 60% 裁决"], "evidence": "逐条限定“5 个 SEL-G 通过者”并给全项目补充：G7 小盘 dev 0.241 过/val 0.338 不过；G8 0.15-5.97、多数为负。", "check": "gate_table_recomputed"},
    {"source": "最终反方审计", "id": "F14", "severity": "LOW", "disposition": "FIXED",
     "claim": "cost_paid 不含滑点却写成“累计成本”", "where": ["01_完整报告.md §5.4", "04_最终结论.json cross_cutting.cost"], "evidence": "改“显性费用”；补“滑点已含在成交价、全成本约 0.4-0.5%/年”。", "check": "cost_paid_wording"},
    {"source": "最终反方审计", "id": "F15", "severity": "LOW", "disposition": "FIXED",
     "claim": "G/G2 命名三重含义、MKT 宇宙、VCB 34/22 指针、L3 说明、24/32", "where": ["00_一页纸.md 术语", "01_完整报告.md §5.6/§6.3/§7", "README.md"], "evidence": "SEL-G/RISK-G2/MKT-G2 定义；686-2,183；VCB 指针 §3.8；L3 假阳性说明；32 个。", "check": "naming_and_pointers"},
    {"source": "最终反方审计", "id": "F16", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "“60% 被拒绝”措辞过强", "where": ["00_一页纸.md", "01_完整报告.md §0/§6.6/§10", "02_规则手册 §5", "README.md 头"], "evidence": "统一为“在本项目测试空间内不被支持；G1 未被直接检验；不等于任何策略都不可能 60%”。", "check": "sixty_wording"},

    # 最终方法代码审计 F1-F12
    {"source": "最终方法代码审计", "id": "F1", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "VCB 成本模型偏离 C-REAL-V1（无冲击项、10% 参与率）", "where": ["01_完整报告.md §1.4/§3.8/§5.4/§3.20", "03_数据与复现说明.md §3.2", "02_规则手册 §1"], "evidence": "披露 + 量化 dev 高估约 0.44pp/年；不把 VCB 与 T0005 无注明并表。", "check": "vcb_cost_disclosed"},
    {"source": "最终方法代码审计", "id": "F2", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "VCB delisting_missing_days=250 在停牌期制造卖出", "where": ["01_完整报告.md §3.8/§8.9", "03_数据与复现说明.md §3.3"], "evidence": "002075.SZ 526 日停牌、2017-09-25 按 16.12 卖出；复牌 14.48、2019 约 7.2；val 偏乐观 0.14-0.77pp/年。", "check": "vcb_250d_disclosed"},
    {"source": "最终方法代码审计", "id": "F3", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "停牌期 sell_persistence 强平跨族通用但只披露 T0005", "where": ["01_完整报告.md §5.11/§8", "04_最终结论.json cross_cutting.must_exit"], "evidence": "T0005 val 2、小盘主单元 104、COMBO full 16/733、VCB val 1。", "check": "forced_liquidation_counts"},
    {"source": "最终方法代码审计", "id": "F4", "severity": "MEDIUM", "disposition": "LIMITATION_DECLARED",
     "claim": "2021+ 状态序列是 MKT 留出的实质泄漏路径", "where": ["01_完整报告.md §7", "03_数据与复现说明.md §4", "README.md §5"], "evidence": "逻辑封存已完成（34 文件、逐文件 SHA-256、0 策略绩效）；物理隔离/移出/独立保管未做，作为未关闭残余风险。", "check": "rereview_seal_logical"},
    {"source": "最终方法代码审计", "id": "F5", "severity": "LOW", "disposition": "FIXED",
     "claim": "out_date 前视影响 0.0158%", "where": ["01_完整报告.md §8", "03_数据与复现说明.md §7"], "evidence": "保留披露与量化；列入下一版必修。", "check": "outdate_disclosed"},
    {"source": "最终方法代码审计", "id": "F6", "severity": "LOW", "disposition": "FIXED",
     "claim": "§12.2 分档用 min(dev,val) 代替 min(E1,E2)", "where": ["01_完整报告.md §0/§2.3/§4 Q6", "00_一页纸.md", "README.md"], "evidence": "全部改为“按 §12.2 阈值作描述性套用，不是预注册分档”。", "check": "descriptive_tier"},
    {"source": "最终方法代码审计", "id": "F7", "severity": "LOW", "disposition": "FIXED_ROOT_DOCS_SUPERSEDE",
     "claim": "证据矩阵仍说 covered=486", "where": ["01_完整报告.md §6.1", "04_最终结论.json holdout.gate_coverage"], "evidence": "根文档统一 505/228；矩阵不改写。", "check": "gate_coverage_505"},
    {"source": "最终方法代码审计", "id": "F8", "severity": "LOW", "disposition": "FIXED",
     "claim": "两个 2021+ 扫描器口径不一致（L3 假阳性）", "where": ["01_完整报告.md §7", "03_数据与复现说明.md §4", "README.md §5"], "evidence": "写明 L3 的 1 条 suspect 是 试验总账.csv 两行 meta/audit 的假阳性；扫描器本批不修。", "check": "scan_l3_disclosed"},
    {"source": "最终方法代码审计", "id": "F9", "severity": "LOW", "disposition": "FIXED_ROOT_DOCS_SUPERSEDE",
     "claim": "留出裁决仍写 T0005 从未运行", "where": ["01_完整报告.md §7", "04_最终结论.json holdout"], "evidence": "同 M05；裁决文件为 v2 修订版（02:12）+ 附录 A 初版。", "check": "rereview_holdout_v2"},
    {"source": "最终方法代码审计", "id": "F10", "severity": "LOW", "disposition": "FIXED",
     "claim": "size_pit 用总收益指数当价格，报告未披露", "where": ["01_完整报告.md §3.20", "04_最终结论.json families[SMALLLOWVOL].scope"], "evidence": "§3.20 写明 size_pit=a_close×fs_pit 会把高分红股规模抬高；影响横截面排序，不改变正负号。", "check": "size_pit_disclosed"},
    {"source": "最终方法代码审计", "id": "F11", "severity": "LOW", "disposition": "FIXED",
     "claim": "12% 风险门在 t+1 跳空后实现风险可达 13.8%/1.43%", "where": ["01_完整报告.md §2.7"], "evidence": "保留披露（dev 1 笔、val 2 笔跳空，最大 1.43%/1.33%）。", "check": "risk_gap_disclosed"},
    {"source": "最终方法代码审计", "id": "F12", "severity": "LOW", "disposition": "FIXED",
     "claim": "“60% 差一个数量级”实为 6.9 倍", "where": ["00_一页纸.md", "01_完整报告.md §0", "02_规则手册 §5"], "evidence": "改为约 7 倍。", "check": "no_order_of_magnitude"},
]

# 四份复核（2026-10-29 02:40-02:43）的 FAIL/WARN 逐项关闭清单。
# 这些不是首轮 70 项的重复；每项都是复核员在 rev2 上重新发现/残留的具体字段。
REREVIEW_ITEMS = [
    # 数值复核
    {"source": "数值复核", "id": "F01", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "01 §5.8/04 cross_cutting MKT 主口径 val -7.33 未与族内 best -2.35 分开；04 RELSTR 冗余字段错；01 §3.19 MAIN/EXIT 行仍错",
     "where": ["01_完整报告.md §5.8/§3.19", "04_最终结论.json cross_cutting.market_state / families.MKT / families.RELSTR / families.EXIT"],
     "evidence": "01 §5.8 改“主口径 A 偏广度 +13.89%/−7.33%；族内真实最高 val D_串联过滤 −2.35%”；04 MKT 用 main_dev_config_val=-0.0733 与 family_best_val=-0.0235；RELSTR B02 val=-0.255681、N02 dev=+0.037473；MAIN N_sel=19、EXIT 18 臂 best=−20.20%/−19.84%、表内合计 733。",
     "check": "rereview_f01_mkt_relstr_main_exit"},
    {"source": "数值复核", "id": "F06", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "根文档/04 仍把 v2 修订版留出裁决写成“00:37 历史快照”",
     "where": ["00_一页纸.md", "01_完整报告.md §7/§12", "03_数据与复现说明.md", "04_最终结论.json holdout"],
     "evidence": "统一为“v2 修订版（2026-10-29 02:12）；00:37 初版只作附录 A/initial_snapshot”；04 supersedes.path 指向 `#initial_snapshot`、as_of=02:12。",
     "check": "rereview_holdout_v2"},
    {"source": "数值复核", "id": "F09", "severity": "LOW", "disposition": "FIXED",
     "claim": "04 cross_cutting.dynamic_parameters 仍写 2.11%~5.50%",
     "where": ["04_最终结论.json cross_cutting.dynamic_parameters", "01_完整报告.md §2.5/§5.1"],
     "evidence": "04 改 1.54%~5.50%（S06 最低）；与 statistics.neighborhood_range 一致。",
     "check": "rereview_dynamic_154"},
    {"source": "数值复核", "id": "N01", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "03 把过户费敏感性 0.06% 写成主口径；transfer_fee_aligned 把错误固化为 PASS",
     "where": ["03_数据与复现说明.md §3.2", "02_代码/最终交付.py transfer_fee_aligned"],
     "evidence": "03 改主口径 0.1%（0.0010），0.06% 仅 `--transfer-par 0.0006` 敏感性；影响 T0005 dev 57 笔沪市、约 543 元/8 年≈0.007%/年；检查改为读代码默认 0.0010 + 文档主/敏感性双写。",
     "check": "rereview_transfer_fee"},
    {"source": "数值复核", "id": "N02", "severity": "MODERATE", "disposition": "FIXED",
     "claim": "根文档/04 说 2021+ “未封存/未哈希封存”，与 34 文件封存清单矛盾",
     "where": ["01_完整报告.md §7/§9.2", "03_数据与复现说明.md §4/§7", "README.md §5/§8", "04_最终结论.json holdout"],
     "evidence": "统一为“逻辑封存已完成（34 文件、A3/B7/C24、逐文件 SHA-256、0 策略绩效、禁止用途）；未做物理隔离（未移出/改权限/独立保管）”；并链接封存清单/说明。",
     "check": "rereview_seal_logical"},
    {"source": "数值复核", "id": "N03", "severity": "HIGH", "disposition": "FIXED",
     "claim": "验收 70 项/60 checks 的闭合声明过度；检查过窄",
     "where": ["04_证据/审计整改验收.json", "02_代码/最终交付.py _acceptance_checks"],
     "evidence": "验收改为“内部检查通过，等待独立复核”；新增 30+ 断言覆盖 MKT/RELSTR/MAIN/EXIT/版本标签/过户费/封存/G3-G7/预算 11.92/top5 毛净/证据矩阵横幅/交付清单；新增 `04_证据/最终审计剩余项关闭.json` 逐项映射。",
     "check": "rereview_acceptance_status"},
    {"source": "数值复核", "id": "N04", "severity": "LOW", "disposition": "FIXED",
     "claim": "G4 区间 −7.2~−18.2 漏掉 −6.44",
     "where": ["00_一页纸.md G4", "01_完整报告.md §6.6"],
     "evidence": "改 −6.44%~−18.22%（逐配置 −6.44/−7.17/−18.22/−15.33/−14.74，252 日滚动复算）。",
     "check": "rereview_g4_range"},
    # 交付复核
    {"source": "交付复核", "id": "R01", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "修订版留出裁决被根文档描述成 00:37 历史快照", "where": ["00/01/03/04"],
     "evidence": "同数值复核 F06；另 `05_审计/留出准入裁决.md` 头部改为“附录 A 可由内容复算；初版文件已被覆盖，文件清单只记录修订版”。",
     "check": "rereview_holdout_v2"},
    {"source": "交付复核", "id": "R02", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "04 cross_cutting 仍 2.11%；top5_share 旧字段未标毛/净", "where": ["04_最终结论.json"],
     "evidence": "dynamic 改 1.54；main_candidate.dev/val 增加 top5_share_gross/top5_share_net/top5_share_denominator（dev net 0.4587/gross 0.3106；val net −0.6258/gross 0.4092）。",
     "check": "rereview_top5_fields"},
    {"source": "交付复核", "id": "R03", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "“未封存/未哈希封存”与逻辑封存清单冲突", "where": ["01/03/README/04"],
     "evidence": "同数值复核 N02；链接 34 文件清单与说明。", "check": "rereview_seal_logical"},
    {"source": "交付复核", "id": "R04", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "验收闭合声明与处置表状态标注冲突", "where": ["04_证据/审计整改验收.json", "05_审计/审计问题处置表.csv"],
     "evidence": "验收状态改“内部检查通过，等待独立复核”；处置表快照状态同步为“已同步（本轮）”，最终状态见剩余项关闭 JSON；不再作外部闭合声明。",
     "check": "rereview_acceptance_status"},
    {"source": "交付复核", "id": "R05", "severity": "LOW", "disposition": "FIXED",
     "claim": "证据矩阵 P03 仍 486/228，无横幅", "where": ["05_审计/最终结论证据矩阵.md"],
     "evidence": "文件顶部加横幅：01:12 历史；P03/§6#11 的 486/228 已被根文档 505/228/733 取代；不改历史复算。",
     "check": "rereview_evidence_banner"},
    {"source": "交付复核", "id": "M10", "severity": "LOW-MEDIUM", "disposition": "FIXED",
     "claim": "04 cross_cutting.dynamic_parameters 仍 2.11%~5.50%（M10 WARN）",
     "where": ["04_最终结论.json cross_cutting.dynamic_parameters", "01_完整报告.md §2.5/§5.1"],
     "evidence": "04 改 1.54%~5.50%；与 statistics.neighborhood_range 一致。",
     "check": "rereview_dynamic_154"},
    {"source": "交付复核", "id": "M23", "severity": "LOW", "disposition": "FIXED",
     "claim": "证据矩阵 P03 486/228 无横幅（M23 WARN）",
     "where": ["05_审计/最终结论证据矩阵.md"],
     "evidence": "顶部横幅已加；历史复算不改。",
     "check": "rereview_evidence_banner"},
    # 反方复核
    {"source": "反方复核", "id": "F01", "severity": "HIGH", "disposition": "FIXED",
     "claim": "01 §3.19 MAIN N_sel=1/EXIT −8.83；表内 715≠733；00 MAIN“diagnostic”/EXIT“同左”",
     "where": ["01_完整报告.md §3.19", "00_一页纸.md"],
     "evidence": "MAIN 19（T0005+18 邻域）、EXIT 18 臂 best −20.20/−19.84；表内 N_sel 合计 733；00 MAIN 标“N_sel 18 邻域”、EXIT 标“18 臂 best”。",
     "check": "rereview_f01_mkt_relstr_main_exit"},
    {"source": "反方复核", "id": "F04", "severity": "HIGH", "disposition": "FIXED",
     "claim": "00/01/03/04 仍把修订版裁决写成 00:37 快照；01 §7 行 669/670 自相矛盾",
     "where": ["00/01/03/04"], "evidence": "同数值复核 F06；01 §7 统一为 v2 修订版 + 附录 A 初版。",
     "check": "rereview_holdout_v2"},
    {"source": "反方复核", "id": "F06", "severity": "HIGH", "disposition": "FIXED",
     "claim": "根文档仍写“未做哈希封存/未封存”", "where": ["00/01/03/04/README"],
     "evidence": "同数值复核 N02；统一逻辑封存 + 未物理隔离。", "check": "rereview_seal_logical"},
    {"source": "反方复核", "id": "F13", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "G3 只写 dev；G5 把 T0005 混入 5 个；G6 漏 VCB T40 0.622；G7 把 TREND 42 矩阵当 S3",
     "where": ["00_一页纸.md G3/G5/G6/G7", "01_完整报告.md §6.6"],
     "evidence": "G3 dev 0.05–0.28 / val 0.18–0.36；G5 只列 5 个并单列 T0005；G6 8 族表 0.1139 + VCB T40 0.622（N=24）；G7 标 TREND 42 矩阵、S3 是子集。",
     "check": "rereview_g3_g5_g6_g7"},
    {"source": "反方复核", "id": "F15", "severity": "LOW-MEDIUM", "disposition": "FIXED",
     "claim": "00/02 预算未给 11.92；00 “最好 PBO”漏全族；交付清单缺封存/处置表/修订裁决/整改验收",
     "where": ["00_一页纸.md", "02_规则手册 R5", "01_完整报告.md §12", "README.md"],
     "evidence": "预算补 11.92× 可比总口径；PBO 行改“选中族，不是全族最低”并补 ECOLOGY/LONGMOM/STRUCTREV/小盘；交付清单补 6 项。",
     "check": "rereview_budget_pbo_delivery"},
    {"source": "反方复核", "id": "R01", "severity": "HIGH", "disposition": "FIXED",
     "claim": "MAIN/EXIT 两行与 N_sel 合计", "where": ["01 §3.19", "00"], "evidence": "同反方 F01。", "check": "rereview_f01_mkt_relstr_main_exit"},
    {"source": "反方复核", "id": "R02", "severity": "HIGH", "disposition": "FIXED",
     "claim": "修订版裁决被标 00:37", "where": ["00/01/03/04"], "evidence": "同数值 F06。", "check": "rereview_holdout_v2"},
    {"source": "反方复核", "id": "R03", "severity": "HIGH", "disposition": "FIXED",
     "claim": "未封存表述与 34 清单矛盾", "where": ["00/01/03/04/README"], "evidence": "同数值 N02。", "check": "rereview_seal_logical"},
    {"source": "反方复核", "id": "R04", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "G7 0.127/0.452 是 TREND 42 矩阵", "where": ["00 G7", "01 §6.6"], "evidence": "已标 TREND 42 矩阵、S3 子集。", "check": "rereview_g3_g5_g6_g7"},
    {"source": "反方复核", "id": "R05", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "G6 漏 VCB T40 0.622", "where": ["00 G6", "01 §6.6/§6.3", "02 R5"], "evidence": "已写 8 族表 0.1139 + T40 0.622（N=24）。", "check": "rereview_g3_g5_g6_g7"},
    {"source": "反方复核", "id": "R06", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "G3 只 dev；G5 混入 T0005", "where": ["00 G3/G5", "01 §6.6"], "evidence": "G3 dev/val 分列；G5 单列 T0005。", "check": "rereview_g3_g5_g6_g7"},
    {"source": "反方复核", "id": "R07", "severity": "LOW-MEDIUM", "disposition": "FIXED",
     "claim": "预算缺 11.92", "where": ["00", "02 R5"], "evidence": "已补 11.92 可比总口径。", "check": "rereview_budget_pbo_delivery"},
    {"source": "反方复核", "id": "R08", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "“最好 PBO”漏全族", "where": ["00"], "evidence": "改“选中族，不是全族最低”并补 4 个低 PBO。", "check": "rereview_budget_pbo_delivery"},
    {"source": "反方复核", "id": "R09", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "验收闭合声明 / verify 不查文档", "where": ["04_证据/审计整改验收.json", "最终交付.py"],
     "evidence": "新增 30+ 文档/JSON 字段断言；验收状态改内部检查通过；剩余项关闭 JSON。", "check": "rereview_acceptance_status"},
    {"source": "反方复核", "id": "R10", "severity": "LOW", "disposition": "FIXED",
     "claim": "00/01 交付清单缺封存/处置表/修订裁决/整改验收", "where": ["00/01"], "evidence": "已补 6 项。", "check": "rereview_budget_pbo_delivery"},
    # 方法复核
    {"source": "方法复核", "id": "N1", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "根文档称 2021+ 未封存", "where": ["01/03/README/04"], "evidence": "同数值 N02。", "check": "rereview_seal_logical"},
    {"source": "方法复核", "id": "N2", "severity": "MEDIUM", "disposition": "FIXED",
     "claim": "修订版裁决被标未改写/保留原样", "where": ["00/01/03/04"], "evidence": "同数值 F06；01 §9.4 已改“v2 修订版”。", "check": "rereview_holdout_v2"},
    {"source": "方法复核", "id": "N3", "severity": "LOW", "disposition": "FIXED",
     "claim": "附录 A 哈希被谎称与当前文件清单一致", "where": ["05_审计/留出准入裁决.md"],
     "evidence": "头注改为“附录 A 当前内容 42,886 bytes，SHA-256 可按附录内容复算；初版 42,550 bytes/SHA ddd8a220 已被覆盖，项目内不保留原件；文件清单只记录修订版文件（大小与 SHA-256 见清单）”。",
     "check": "rereview_appendix_hash"},
    {"source": "方法复核", "id": "N4", "severity": "LOW", "disposition": "FIXED",
     "claim": "“三个文件”未指向 34 文件清单，未提 C 级数据源", "where": ["01 §7", "03 §4", "README §5"], "evidence": "已写 34 个（A3/B7/C24），C 级 24 个可重建 2021+ 行情，链接清单/说明。", "check": "rereview_seal_logical"},
    {"source": "方法复核", "id": "N5", "severity": "LOW", "disposition": "FIXED",
     "claim": "03_结果 内 2021+ 文件 6 vs 9 口径未说明", "where": ["01 §7", "03 §4", "README §5"], "evidence": "已注明 6=date 列口径、9=date/calendar_date/out_date 口径（A3+B6）。", "check": "rereview_six_nine"},
    {"source": "方法复核", "id": "N6", "severity": "LOW", "disposition": "FIXED",
     "claim": "E1/E2 与入场规则重名", "where": ["00/01/02/04"], "evidence": "证据等级统一 E1-EVID/E2-EVID；入场规则 E1-ENTRY/E2-ENTRY。", "check": "rereview_e1e2_naming"},
]


def _acceptance_checks(facts: dict | None = None) -> list[dict]:
    """整改验收的数据断言（只读结果表；不运行策略）。"""
    f = facts or audit_facts()
    g = _gate_rows()
    out = []

    def rec(name, ok, detail):
        out.append(_rec(name, bool(ok), detail, "03_结果/最终统计/统一门槛.csv"))

    bd, bv, bm = f["best_dev"], f["best_val"], f["best_min"]
    rec("family_best_matches_gate_table",
        abs(bd["cagr"] - 0.203991) < 1e-5 and abs(bv["cagr"] - 0.087049) < 1e-5
        and abs(bm["min_cagr"] - 0.083238) < 1e-5,
        f"best dev {bd['cagr']:.6f} ({bd['config_id']}); best val {bv['cagr']:.6f} ({bv['config_id']}); "
        f"best min {bm['min_cagr']:.6f}")
    rec("two_segment_positive_32",
        f["n_two_segment_positive"] == 32
        and f["two_segment_positive_by_family"].get("TREND") == 5
        and f["two_segment_positive_by_family"].get("VCB") == 18
        and f["two_segment_positive_by_family"].get("SMALLLOWVOL") == 9,
        f"n={f['n_two_segment_positive']} by_family={f['two_segment_positive_by_family']}")
    rec("budget_two_calibers",
        733 / 40.0 > 18.32 and 733 / 64.0 > 11.44,
        f"733/40={733/40.0:.2f}; 733/64={733/64.0:.2f}")
    ms = f["maxstat"]
    rec("maxstat_global_min",
        ms["n_p"] >= 16 and ms["min"] is not None and abs(ms["min"]["p_maxstat"] - 0.071964) < 1e-6,
        f"n_p={ms['n_p']}; min={ms['min']}")
    ue = g[(g.family == "UNIFIEDENGINE")].dropna(subset=["dev_cagr", "val_cagr"])
    rec("bh10_separated",
        len(ue) == 4 and ue.dev_cagr.max() < -0.20 and ue.val_cagr.max() < -0.18,
        f"UNIFIEDENGINE strategies={len(ue)} dev max={ue.dev_cagr.max():.4f} val max={ue.val_cagr.max():.4f} (BH10 0.0555/-0.0758 单列)")
    t5d, t5v = f["top5"]["dev"], f["top5"]["val"]
    rec("top5_two_denominators",
        abs(t5d["top5_over_net"] - 0.4587) < 5e-4 and abs(t5d["top5_over_gross"] - 0.31061) < 5e-4
        and t5v["top5_over_net"] < 0,
        f"dev net={t5d['top5_over_net']:.4f} gross={t5d['top5_over_gross']:.4f}; val net={t5v['top5_over_net']:.4f}")
    rec("vcb_count_reconciled",
        f["vcb_count"]["declared_rows"] == 34 and f["vcb_count"]["unique_configs"] == 32
        and f["vcb_count"]["n_sel"] == 22 and f["vcb_count"]["n_diag"] == 10,
        f"{f['vcb_count']}")
    sr = g[g.family == "STRUCTREV"]
    src2 = f["structrev_count"]
    rec("structrev_count_reconciled",
        len(sr) == 30 and src2["n_sel"] == 30 and src2["n_audit"] == 1 and src2["account_rows"] == 31,
        f"STRUCTREV gate rows={len(sr)}; ledger N_sel={src2['n_sel']} + audit={src2['n_audit']} -> {src2['account_rows']} account rows")
    nb = f["neighborhood"]
    rec("neighborhood_range_154",
        abs(nb["dev_min"] - 0.015396) < 1e-5 and abs(nb["dev_max"] - 0.054984) < 1e-5,
        f"dev {nb['dev_min']:.6f}~{nb['dev_max']:.6f}; val {nb['val_min']:.6f}~{nb['val_max']:.6f}")
    s3 = f["s3_exposure"]
    rec("s3_exposure_fixed",
        s3["flat_min"] < 0.47 and s3["flat_max"] > 0.83 and s3["pos_min"] < 0.40,
        f"flat {s3['flat_min']:.3f}-{s3['flat_max']:.3f}; positions {s3['pos_min']:.2f}-{s3['pos_max']:.2f}")
    tn = f["trend_non_main"]
    rec("trend_limit_disclosed",
        tn["non_main"] > 5000 and abs(tn["share"] - 0.1584) < 5e-4 and tn["s3_non_main"] > 130,
        f"TREND non-main {tn['non_main']}/{tn['all_trades']} ({tn['share']:.4f}); S3 {tn['s3_non_main']}/{tn['s3_trades']}; "
        f"5 rules {tn['s3_5rules_non_main']}/{tn['s3_5rules_trades']}")
    rec("caches_listed",
        len(f["caches_2021"]) == 2 and all(x["return_like_columns"] for x in f["caches_2021"]),
        f"{f['caches_2021']}")
    gp = f["g_pass"]
    rec("gate_table_recomputed",
        len(gp) == 5 and all(x["dev_mdd"] >= -0.25 and x["val_mdd"] >= -0.25 for x in gp)
        and max(x["dev_mdd"] for x in gp) > -0.24,
        f"5 SEL-G pass; dev MDD={[round(x['dev_mdd'],4) for x in gp]}; Calmar={[round(x['dev_calmar'],3) for x in gp]}")
    # 复核新增的数据断言
    rs = f["relstr"]
    rec("rereview_relstr_fields",
        rs["B02"] and rs["N02"] and abs(rs["B02"]["val"] + 0.255681) < 1e-5
        and abs(rs["N02"]["dev"] - 0.037473) < 1e-5,
        f"B02={rs['B02']}; N02={rs['N02']}")
    rec("rereview_main_nsel_19", f["main_count"]["n_sel"] == 19,
        f"MAIN N_sel={f['main_count']['n_sel']}")
    eb = f["exit_best"]
    rec("rereview_exit_18arm_best",
        abs(eb["dev"] + 0.201996) < 1e-5 and abs(eb["val"] + 0.1984) < 1e-5 and eb["n"] >= 18,
        f"EXIT gate rows={eb['n']}; best dev={eb['dev']:.6f} val={eb['val']:.6f}")
    g4 = f["g4"]
    rec("rereview_g4_range_data",
        abs(g4["min"] + 0.182214) < 1e-4 and abs(g4["max"] + 0.064356) < 1e-4,
        f"g4 values={[(k, round(v,4)) for k,v in g4['values']]}")
    sl = f["seal"]
    rec("rereview_seal_34",
        sl and sl["rows"] == 34 and sl["A"] == 3 and sl["B"] == 7 and sl["C"] == 24
        and sl["sha_nonempty"] == 34 and sl["strategy_performance_zero"],
        f"{sl}")
    tf = f["transfer_fee"] if f.get("transfer_fee") else {}
    rec("rereview_transfer_fee_code",
        tf.get("code_default_0_0010") and tf.get("code_comment_0_0006_sensitivity"),
        f"{tf}")
    dp = f["disposition"] if f.get("disposition") else {}
    rec("rereview_disposition_no_sync",
        dp.get("needs_sync_rows") == 0,
        f"{dp}")
    rec("rereview_evidence_banner_data", bool(f["evidence_matrix_banner"]),
        f"evidence_matrix_banner={f['evidence_matrix_banner']}")
    return out


def stage_remediation() -> dict:
    """生成 04_证据/审计整改验收.json（不运行策略、不写交易结果）。"""
    facts = audit_facts()
    checks = _acceptance_checks(facts)
    # 文档断言（只读文本）
    docs = {}
    for relp in ["00_一页纸.md", "01_完整报告.md", "02_规则手册_研究结论版.md",
                 "03_数据与复现说明.md", "README.md"]:
        try:
            with open(p(relp), encoding="utf-8") as fh:
                docs[relp] = fh.read()
        except Exception:
            docs[relp] = ""
    all_text = "\n".join(docs.values())
    fc = jload(CONCLUSION)

    def doc_check(name, ok, detail):
        checks.append(_rec(name, bool(ok), detail, "根文档"))

    doc_check("report_says_project_best_dev", "20.40" in docs["01_完整报告.md"] and "20.40" in docs["02_规则手册_研究结论版.md"],
              "报告与规则手册均写 +20.40%（全项目 dev 最高）")
    doc_check("report_no_absolute_claim",
              "0 个通过全部硬门槛；32 个两段为正" in docs["01_完整报告.md"],
              "§10 已改为 0 个过门槛 + 32 个未过事件层/去尾/多重性/风险（旧句只保留在撤回索引）")
    doc_check("scope_not_all_mainboard",
              "T0005/VCB/小盘为主板宇宙" in docs["01_完整报告.md"]
              and "15.84%" in docs["01_完整报告.md"] and "13.93%" in docs["01_完整报告.md"],
              "含 TREND 15.84% / STRUCT 13.93% 非主板，并改为分族声明（旧句只出现在“不能说”条目）")
    doc_check("sixty_wording",
              "G1 未被直接检验" in all_text and "任何策略都不可能 60%" in all_text
              and "差一个数量级地拒绝" not in docs["00_一页纸.md"],
              "60% 措辞统一 + G1 未检验 + 明确不等于任何策略都不可能")
    doc_check("descriptive_tier", "描述性" in docs["01_完整报告.md"] and "不是完整预注册分档" in all_text,
              "§12.2 标注为描述性套用，不是预注册分档")
    doc_check("g6_dsr_fixed", "0.9585" in docs["01_完整报告.md"] and "0.1139" in docs["01_完整报告.md"]
              and "0.84（VCB-N120）" not in docs["00_一页纸.md"],
              "G6 写 T0005 0.9585 / 可用最高 VCB-N120 0.1139；删除 0.84（VCB-N120）")
    doc_check("g8_range_fixed", ("0.15%–5.97%" in docs["00_一页纸.md"] or "0.15%-5.97%" in docs["00_一页纸.md"])
              and "0.15%-5.97%" in docs["01_完整报告.md"],
              "G8 统一为 0.15%-5.97%")
    doc_check("delay_range_fixed", "−8.31%~−15.61%" in docs["01_完整报告.md"], "执行压力延迟改为 5 种子区间 + 全配置区间")
    doc_check("sixty_arithmetic_fixed", "3.08 次/槽/年" in docs["01_完整报告.md"] and "15.3%/笔" in docs["02_规则手册_研究结论版.md"],
              "S4 3.08 次/槽/年、+4.71%/笔；15.3%/19.5% 或约 10 次/槽/年")
    doc_check("vcb_conservative_removed", "方向未证明" in docs["01_完整报告.md"]
              and "方向上对收益保守" not in docs["01_完整报告.md"], "VCB 删“方向保守”，改“方向未证明 + 基准偏乐观约 0.6pp”")
    doc_check("vcb_cost_disclosed", "无冲击项" in docs["01_完整报告.md"] and "0.44pp/年" in docs["01_完整报告.md"],
              "VCB 成本口径与 0.44pp/年 已披露")
    doc_check("vcb_250d_disclosed", "002075" in docs["01_完整报告.md"] and "250" in docs["01_完整报告.md"],
              "VCB 250 日停牌强平已披露")
    doc_check("forced_liquidation_counts", "104 笔" in docs["01_完整报告.md"], "跨族停牌强平笔数已补")
    doc_check("ohlc_close_vs_any", "任一 OHLC" in docs["01_完整报告.md"] and "4 行" in docs["01_完整报告.md"]
              and "5 行" in docs["01_完整报告.md"], "OHLC 写成 close 4 / any 5")
    doc_check("v1_two_denominators", "94.01%" in docs["03_数据与复现说明.md"] and "170.1" in docs["03_数据与复现说明.md"]
              and "227pp" in docs["03_数据与复现说明.md"], "94.10/94.01 并列 + 170.1 + 227 撤回")
    doc_check("v1_slope_fixed", "2.6196" in docs["03_数据与复现说明.md"] and "2.0119" in docs["03_数据与复现说明.md"],
              "斜率全样本/截尾并列")
    doc_check("transfer_fee_aligned",
              "主口径按面额 0.1%" in docs["03_数据与复现说明.md"]
              and "0.06%" in docs["03_数据与复现说明.md"]
              and "敏感性" in docs["03_数据与复现说明.md"]
              and "0.0010" in docs["03_数据与复现说明.md"]
              and "按代码统一为面额 0.06%" not in docs["03_数据与复现说明.md"],
              "过户费主口径 0.1%（0.0010）；0.06% 只作敏感性")
    doc_check("data_trust_nuanced", "复权/价格层可以信" in docs["03_数据与复现说明.md"], "数据层一句话已限定")
    doc_check("readme_92_36", "92 行" in docs["README.md"] and "36 行为护栏" in docs["README.md"]
              and "未分类 1 行" in docs["README.md"], "README 92/36 + 1 条未分类说明")
    doc_check("baostock_63_79", "63 个证券日" in docs["README.md"] and "79 行" in docs["README.md"], "63 日 / 79 行并列")
    doc_check("listing_rules_complete", "td_since_list ≥ 120" in docs["01_完整报告.md"], "上市规则补 td_since_list ≥ 120")
    doc_check("typo_fixed", "敏感性格" not in all_text, "“敏感性格”已清")
    doc_check("no_order_of_magnitude", "差一个数量级地拒绝" not in docs["00_一页纸.md"] and "约 **7 倍**" in docs["01_完整报告.md"],
              "一页纸不再写“差一个数量级地拒绝”；报告写约 7 倍")
    doc_check("cost_paid_wording", "显性费用" in docs["01_完整报告.md"], "cost_paid 改“显性费用”")
    doc_check("size_pit_disclosed", "size_pit" in docs["01_完整报告.md"], "size_pit 口径扭曲已披露")
    doc_check("risk_gap_disclosed", "1.43%" in docs["01_完整报告.md"], "跳空风险超限已披露")
    doc_check("outdate_disclosed", "0.0158%" in docs["01_完整报告.md"], "out_date 前视已披露")
    doc_check("scan_l3_disclosed", "假阳性" in docs["01_完整报告.md"], "L3 假阳性已说明")
    doc_check("naming_and_pointers", "SEL-G" in docs["00_一页纸.md"] and "§3.8" in docs["README.md"], "术语与 VCB 指针已改")
    doc_check("inventory_disclosed", "不是逐字节全量" in docs["README.md"], "清单非全量已声明")
    doc_check("gate_coverage_505", "505" in docs["01_完整报告.md"] and "505" in docs["README.md"], "505/228 已统一")
    doc_check("holdout_reasons_current", "没有任何通过 val 的父试验" in docs["01_完整报告.md"]
              or "没有任何“通过 val”的父试验" in docs["01_完整报告.md"], "留出理由改为当前事实")
    doc_check("delist_label_fixed", "停牌顺延后强制估值卖出" in docs["01_完整报告.md"], "T0005 2 笔改为停牌顺延强平")
    doc_check("universe_min_686", "686 只" in docs["01_完整报告.md"] and "2,183 只" in docs["01_完整报告.md"],
              "主宇宙 686-2,183（旧值只保留在“原稿漏掉”的说明里）")
    doc_check("small_chain_footnote", "5.93%" in docs["01_完整报告.md"], "分段链两口径")
    doc_check("structrev_best_a08", "A08_KNIFE" in docs["01_完整报告.md"] and "A17" in docs["01_完整报告.md"], "STRUCTREV best A08")
    doc_check("report_says_project_best_dev2", "20.40" in docs["00_一页纸.md"], "一页纸含 +20.40%")
    doc_check("two_segment_32_docs", "32 个" in docs["00_一页纸.md"] and "32 个" in docs["01_完整报告.md"], "24→32")
    doc_check("budget_docs", "18.33" in docs["00_一页纸.md"] and "18.33" in docs["README.md"], "预算两口径")
    doc_check("maxstat_docs", "0.071964" in docs["00_一页纸.md"] and "0.071964" in docs["01_完整报告.md"], "max-stat 全局最小 p")
    doc_check("top5_docs", "45.9%" in docs["01_完整报告.md"], "top5 净分母")
    doc_check("cross_matrix_json", len(fc.get("cross_family_matrix", [])) == 18
              and all(all(k in x for k in ("universe", "adjust", "engine", "cost", "limits")) for x in fc.get("cross_family_matrix", [])),
              "04_最终结论.json cross_family_matrix 18 行 × 5 列")
    doc_check("cross_matrix_complete", len(fc.get("cross_family_matrix", [])) == 18
              and all(all(str(x.get(k) or "").strip() for k in ("universe", "adjust", "engine", "cost", "limits"))
                      for x in fc.get("cross_family_matrix", [])),
              "18 行 × 5 列全部非空")
    doc_check("dsr_cross_caliber", "dsr_cross_caliber" in fc.get("statistics", {})
              and "跨口径" in docs["01_完整报告.md"],
              "DSR 跨 365/733 不单调的解释已写入 JSON 与报告 §6.2")
    doc_check("families_count_18_json", len(fc.get("families", [])) == 18, "families=18")
    fam_map = {x.get("family"): x for x in fc.get("families", [])}
    doc_check("bh10_json", (fam_map.get("UNIFIEDENGINE", {}).get("benchmark_bh10", {}) or {}).get("type") == "benchmark",
              "BH10 单列 benchmark")
    doc_check("holdout_supersedes_json", "supersedes" in fc.get("holdout", {}), "holdout.supersedes 存在")

    # --- 四份复核 33 项的文档/JSON 断言 ---
    fam = {x.get("family"): x for x in fc.get("families", [])}
    mkt = fam.get("MKT", {})
    rel = fam.get("RELSTR", {})
    ex = fam.get("EXIT", {})
    mn = fam.get("MAIN", {})
    exb = ex.get("family_best", {}) or {}
    ex_dev = (exb.get("dev", {}) or {}).get("cagr") if isinstance(exb.get("dev"), dict) else exb.get("dev")
    doc_check("rereview_f01_mkt_relstr_main_exit",
              "MKT 族**主口径**" in docs["01_完整报告.md"] and "−2.35%" in docs["01_完整报告.md"]
              and "N_sel 合计核对" in docs["01_完整报告.md"] and "−20.20%" in docs["01_完整报告.md"]
              and abs(float(mn.get("n_sel", 0)) - 19) < 0.1
              and abs(float(mkt.get("main_dev_config_val", 0)) + 0.0733) < 1e-4
              and abs(float(mkt.get("family_best_val", {}).get("cagr", 0)) + 0.0235) < 1e-4
              and abs(float(rel.get("family_best_dev", {}).get("val", 0)) + 0.255681) < 1e-5
              and abs(float(rel.get("family_best_val", {}).get("dev", 0)) - 0.037473) < 1e-5
              and abs(float(ex_dev or 0) + 0.201996) < 1e-5,
              "01 §5.8/§3.19 与 04 MKT/RELSTR/MAIN/EXIT 字段全部对齐")
    b0 = ex.get("b0_baseline", {}) or {}
    rb = ex.get("excluded_rb_sensitivity", {}) or {}
    doc_check("rereview_exit_b0_vs_rb",
              abs(float(b0.get("dev", 0)) + 0.2268) < 5e-4
              and abs(float(b0.get("val", 0)) + 0.2243) < 5e-4
              and abs(float(rb.get("dev", 0)) + 0.0883) < 5e-4
              and rb.get("val") is None
              and rb.get("scope") == "dev only"
              and "退出_仓位敏感.csv" in str(rb.get("source", ""))
              and "benchmark_full_window" not in ex
              and "RB 仓位敏感性" in docs["00_一页纸.md"]
              and "RB 仓位敏感性" in docs["01_完整报告.md"]
              and "RB 仓位敏感性" in docs["README.md"],
              "EXIT B0 baseline -22.68%/-22.43%；-8.83% 归 RB 敏感性（dev only）；18 臂 best -20.20/-19.84 不变")
    doc_check("rereview_holdout_v2",
              "v2 修订版" in all_text and "00:37 历史快照" not in docs["00_一页纸.md"]
              and "00:37 的历史快照" not in docs["01_完整报告.md"]
              and "00:37 历史快照" not in docs["03_数据与复现说明.md"]
              and "v2 修订版" in str(fc.get("holdout", {}).get("adjudication_note", ""))
              and "initial_snapshot" in str(fc.get("holdout", {}).get("supersedes", {}).get("path", "")),
              "留出裁决统一为 v2 修订版 + 附录 A/initial_snapshot 初版")
    doc_check("rereview_dynamic_154",
              "1.54%~5.50%" in str(fc.get("cross_cutting", {}).get("dynamic_parameters", {}).get("evidence", ""))
              and "2.11%~5.50%" not in str(fc.get("cross_cutting", {}).get("dynamic_parameters", {}).get("evidence", "")),
              "04 cross_cutting.dynamic_parameters = 1.54%~5.50%")
    doc_check("rereview_transfer_fee",
              "主口径按面额 0.1%" in docs["03_数据与复现说明.md"]
              and "0.06%" in docs["03_数据与复现说明.md"]
              and "敏感性" in docs["03_数据与复现说明.md"]
              and "0.0010" in docs["03_数据与复现说明.md"]
              and "按代码统一为面额 0.06%" not in docs["03_数据与复现说明.md"],
              "过户费主口径 0.1%（0.0010）；0.06% 只作敏感性")
    doc_check("rereview_seal_logical",
              "逻辑封存" in all_text and "物理隔离" in all_text
              and "2021+未评估文件封存清单.csv" in all_text
              and "逻辑封存" in str(fc.get("holdout", {}).get("exposure_note", ""))
              and bool(fc.get("holdout", {}).get("seal", {}).get("logical_seal")),
              "逻辑封存 34 文件 + 未物理隔离，已链接清单/说明")
    doc_check("rereview_acceptance_status",
              fc.get("meta", {}).get("review_status") == FINAL_STATUS,
              f"review_status={fc.get('meta', {}).get('review_status')}")
    doc_check("rereview_g4_range",
              "−6.44%~−18.22%" in docs["00_一页纸.md"] and "−6.44%~−18.22%" in docs["01_完整报告.md"],
              "G4 区间改为 −6.44%~−18.22%")
    t5 = fc.get("main_candidate", {})
    doc_check("rereview_top5_fields",
              abs(float(t5.get("dev", {}).get("top5_share_net", 0)) - 0.4587) < 5e-4
              and abs(float(t5.get("dev", {}).get("top5_share_gross", 0)) - 0.3106) < 5e-4
              and abs(float(t5.get("val", {}).get("top5_share_net", 0)) + 0.6258) < 5e-4,
              "top5_share_gross/net 双字段已回填")
    doc_check("rereview_evidence_banner", bool(facts["evidence_matrix_banner"]),
              "证据矩阵顶部含 486/228→505/228 横幅")
    doc_check("rereview_g3_g5_g6_g7",
              "val Calmar 0.18–0.36" in docs["00_一页纸.md"] and "val Calmar 0.18–0.36" in docs["01_完整报告.md"]
              and "T0005 dev −0.03% 不在 5 个" in docs["00_一页纸.md"]
              and "VCB T40 0.622" in docs["00_一页纸.md"] and "VCB T40 0.622" in docs["01_完整报告.md"]
              and "TREND 42 矩阵" in docs["00_一页纸.md"] and "TREND 42 矩阵" in docs["01_完整报告.md"],
              "G3 dev/val、G5 单列 T0005、G6 T40 0.622、G7 TREND 42 矩阵")
    doc_check("rereview_budget_pbo_delivery",
              "11.92" in docs["00_一页纸.md"] and "11.92" in docs["02_规则手册_研究结论版.md"]
              and "选中族，不是全族最低" in docs["00_一页纸.md"]
              and "05_审计/2021+未评估文件封存清单.csv" in docs["00_一页纸.md"]
              and "审计问题处置表.csv" in docs["00_一页纸.md"]
              and "04_证据/最终审计剩余项关闭.json" in docs["00_一页纸.md"],
              "预算 11.92、PBO 选中族、交付清单 6 项")
    doc_check("rereview_six_nine",
              ("6 vs 9" in docs["01_完整报告.md"] or "6 个**含 >2020 日期" in docs["01_完整报告.md"])
              and ("6 vs 9" in docs["03_数据与复现说明.md"] or "9 个（A3+B6）" in docs["03_数据与复现说明.md"]),
              "6=date 列口径、9=A3+B6 已在根文档说明")
    doc_check("rereview_e1e2_naming",
              "E1-EVID" in docs["01_完整报告.md"] and "E2-EVID" in docs["01_完整报告.md"]
              and "E1-ENTRY" in docs["01_完整报告.md"] and "E1-EVID" in docs["00_一页纸.md"],
              "证据等级 E1-EVID/E2-EVID；入场 E1-ENTRY/E2-ENTRY")
    ruling = ""
    rp = os.path.join(ROOT, "05_审计", "留出准入裁决.md")
    if os.path.exists(rp):
        ruling = open(rp, encoding="utf-8").read()
    doc_check("rereview_appendix_hash",
              "可按附录 A 内容复算" in ruling and "初版文件已被覆盖" in ruling
              and "只记录修订版文件" in ruling
              and "与 `文件清单.csv` 记录的初版哈希前缀" not in ruling,
              "附录 A 哈希改为“可按附录 A 内容复算”；不谎称清单保留初版")

    items = []
    for it in REMEDIATION:
        items.append({**it, "status": "DONE" if it["disposition"] != "LIMITATION_DECLARED" else "DONE_WITH_DECLARED_LIMITATION"})
    for it in REREVIEW_ITEMS:
        items.append({**it, "status": "FIXED_IN_THIS_BATCH"})
    n_fixed = sum(1 for x in items if x["disposition"] == "FIXED")
    n_super = sum(1 for x in items if x["disposition"] == "FIXED_ROOT_DOCS_SUPERSEDE")
    n_limit = sum(1 for x in items if x["disposition"] == "LIMITATION_DECLARED")
    n_open = sum(1 for x in items if x["disposition"] not in ("FIXED", "FIXED_ROOT_DOCS_SUPERSEDE", "LIMITATION_DECLARED"))
    missing_check = [x["id"] for x in items if x.get("check") and x["check"] not in {c["check"] for c in checks}]
    failed = [c["check"] for c in checks if not c["pass"]]
    out = {
        "schema": "trend-structure-audit-remediation/v2",
        "generated_at_local": dt.datetime.now().isoformat(timespec="seconds"),
        "tool_sha256": sha256_file(__file__),
        "project": "trading_趋势结构",
        "role": "审计整改负责人（只读结果表；未重跑策略、未计算 2021+ 收益、未修改交易结果/历史证据）",
        "audits": [
            {"path": "05_审计/最终数值审计.json", "findings": "F01-F19", "n": 19},
            {"path": "05_审计/最终交付审计.json", "findings": "M01-M23", "n": 23},
            {"path": "05_审计/最终反方审计.json", "findings": "F01-F16", "n": 16},
            {"path": "05_审计/最终方法代码审计.json", "findings": "F1-F12", "n": 12},
            {"path": "05_审计/最终数值审计_复核.json", "findings": "F01/F06/F09/N01-N04", "n": 7},
            {"path": "05_审计/最终交付审计_复核.json", "findings": "R01-R05", "n": 5},
            {"path": "05_审计/最终反方审计_复核.json", "findings": "F01/F04/F06/F13/F15/R01-R10", "n": 15},
            {"path": "05_审计/最终方法代码审计_复核.json", "findings": "N1-N6", "n": 6},
        ],
        "summary": {
            "n_items": len(items),
            "n_first_round": len(REMEDIATION),
            "n_rereview": len(REREVIEW_ITEMS),
            "n_fixed": n_fixed, "n_supersede": n_super,
            "n_declared_limitation": n_limit,
            "n_open_internal": n_open,
            "n_needs_resync": len(missing_check),
            "n_checks": len(checks), "n_check_pass": sum(1 for c in checks if c["pass"]),
            "n_check_fail": len(failed),
            "status": FINAL_STATUS,
            "verdict": FINAL_VERDICT,
            "claim_scope": "本 JSON 只证明“本清单 + 本检查”通过；四份终验均 PASS（`05_审计/*_终验.md/.json`），但仍不代表无残余。",
            "external_claim": "四份终验均 PASS；不宣称 0 open——三项 declared residuals 与签字范围限制仍有效。",
        },
        "items": items,
        "checks": checks,
        "missing_check_mapping": missing_check,
        "declared_limitations": [
            "2021+ 文件已逻辑封存（34 个、A3/B7/C24、逐文件 SHA-256、0 策略绩效、禁止用途；清单见 `05_审计/2021+未评估文件封存清单.csv`，说明见 `05_审计/2021+未评估文件封存说明.md`）；**物理隔离未做**（未移出/改权限/独立保管），A 级文件仍在项目树内。",
            "TREND 创业板限价档位 bug 未修、5 个 S3 未重跑：只标“执行口径污染”，不得当干净两段正证据。",
            "VCB 成本模型/250 日强平/delist_window 未改代码：只披露与量化，VCB 仍按原结果判负。",
            "跨族引擎/价格口径未统一：新增矩阵只作披露，跨族表不得横向排名。",
            "`05_审计/留出准入裁决.md/.json` 已修订为 v2（当前事实 + 附录 A 初版）；`05_审计/最终结论证据矩阵.md` 只加“486/228→505/228”横幅，历史复算不变。",
            "`文件清单.csv` 明示只收 `05_审计/*.md`，审计 JSON/CSV 不收录；清单不是逐字节全量。",
            "交付验收 L4 仍有 1 条未分类：`封存2021文件.py:66` 的说明字符串（无收益列），不是策略代码；本批不改他人脚本。",
            "v1 偏差有 94.10%/94.01% 两个分母；“227pp”撤回，改用 170.1（收益小数）。",
        ],
        "not_run": ["任何策略回测", "任何 2021+ 收益", "任何 2021+ 行情对照", "任何交易结果/历史证据修改"],
        "commands": {
            "verify": "python3 02_代码/最终交付.py verify",
            "selfcheck": "python3 02_代码/最终交付.py selfcheck",
            "remediation": "python3 02_代码/最终交付.py remediation",
            "closure": "python3 02_代码/最终交付.py remediation（同时写 04_证据/最终审计剩余项关闭.json）",
            "acceptance": "python3 02_代码/交付验收.py",
        },
    }
    jdump(os.path.join(ROOT, "04_证据", "审计整改验收.json"), out)
    stage_closure(out)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + " | " + c["detail"])
    print(f"remediation: {len(checks)-len(failed)}/{len(checks)} checks; items={len(items)} "
          f"open_internal={n_open} needs_resync={len(missing_check)}; status={out['summary']['status']}")
    return out


def stage_closure(remediation: dict | None = None) -> dict:
    """逐项映射四份复核的全部 FAIL/WARN，写 04_证据/最终审计剩余项关闭.json。"""
    checks = {c["check"]: c for c in (remediation or {}).get("checks", [])}
    items = []
    for it in REREVIEW_ITEMS:
        c = checks.get(it.get("check"))
        items.append({
            "source": it["source"], "id": it["id"], "severity": it["severity"],
            "claim": it["claim"], "where": it.get("where", []),
            "disposition": it["disposition"], "status": "FIXED_IN_THIS_BATCH",
            "verified_by": it.get("check"), "check_pass": bool(c and c.get("pass")),
            "check_detail": (c or {}).get("detail", ""),
            "evidence": it.get("evidence", ""),
            "needs_resync": False,
        })
    aliases = {"交付复核:M10": "交付复核:R02", "交付复核:M23": "交付复核:R05"}
    coverage = {
        "数值复核": ["F01", "F06", "F09", "N01", "N02", "N03", "N04"],
        "交付复核": ["R01", "R02", "R03", "R04", "R05", "M10", "M23"],
        "反方复核": ["F01", "F04", "F06", "F13", "F15", "R01", "R02", "R03", "R04",
                     "R05", "R06", "R07", "R08", "R09", "R10"],
        "方法复核": ["N1", "N2", "N3", "N4", "N5", "N6"],
    }
    mapped = {(x["source"], x["id"]) for x in items}
    missing = []
    for src, ids in coverage.items():
        for i in ids:
            if (src, i) not in mapped and f"{src}:{i}" not in aliases:
                missing.append(f"{src}:{i}")
    n_check_fail = sum(1 for x in items if not x["check_pass"])
    out = {
        "schema": "trend-structure-final-audit-residual-closure/v1",
        "generated_at_local": dt.datetime.now().isoformat(timespec="seconds"),
        "project": "trading_趋势结构",
        "status": FINAL_VERDICT,
        "status_zh": FINAL_STATUS,
        "source_audits": [
            {"path": "05_审计/最终数值审计_复核.json", "fail_warn": ["F01", "F06", "F09", "N01", "N02", "N03", "N04"]},
            {"path": "05_审计/最终交付审计_复核.json", "fail_warn": ["R01", "R02", "R03", "R04", "R05", "M10", "M23"]},
            {"path": "05_审计/最终反方审计_复核.json",
             "fail_warn": ["F01", "F04", "F06", "F13", "F15", "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10"]},
            {"path": "05_审计/最终方法代码审计_复核.json", "fail_warn": ["N1", "N2", "N3", "N4", "N5", "N6"]},
        ],
        "aliases": aliases,
        "summary": {
            "n_source_items": sum(len(v) for v in coverage.values()),
            "n_dedup_items": len(items),
            "n_fixed": len(items),
            "n_check_fail": n_check_fail,
            "n_unmapped": len(missing),
            "n_needs_resync": 0,
            "external_open_claims": 0,
            "claim_scope": "仅本文件映射的复核项；四份终验均 PASS，N7/N8 已随后修正；仍不宣称无残余。",
        },
        "items": items,
        "unmapped": missing,
        "verification": {
            "verify": "python3 02_代码/最终交付.py verify",
            "selfcheck": "python3 02_代码/最终交付.py selfcheck",
            "remediation": "python3 02_代码/最终交付.py remediation",
            "delivery_acceptance": "python3 02_代码/交付验收.py",
            "seal_scan": "python3 02_代码/封存2021文件.py",
            "inventory": "python3 02_代码/交付验收.py --inventory-only",
        },
        "declared_residuals": [
            "2021+ 物理隔离/独立保管未做（逻辑封存已完成；清单 `05_审计/2021+未评估文件封存清单.csv`、说明 `05_审计/2021+未评估文件封存说明.md`）。",
            "TREND 创业板限价 bug、VCB 成本/250 日强平未改代码、未重跑。",
            "跨族引擎/价格口径未统一，矩阵只作披露。",
        ],
        "not_run": ["任何策略回测", "任何 2021+ 收益", "任何交易结果/注册表/总账修改"],
    }
    jdump(os.path.join(ROOT, "04_证据", "最终审计剩余项关闭.json"), out)
    return out


def _t0005_block() -> dict:
    src = source_numbers()
    M = load_module("02_代码/最终试验总账.py", "fl3")
    dev_m = src["stats"]["dev"]["metrics"]
    val_m = src["stats"]["val"]["metrics"]
    old = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    var_sr = float(old["statistics"]["dsr"]["var_sr_daily"])
    n_old = int(old["statistics"]["dsr"]["n_sr_trials"])

    def _top5_net(window):
        t = pd.read_csv(os.path.join(T0005_DIR, f"02_逐笔_T0005_{window}.csv"))
        pnl = t["pnl"].astype(float)
        return float(pnl.nlargest(5).sum() / pnl.sum()) if pnl.sum() else None

    top5_net_dev = _top5_net("dev")
    top5_net_val = _top5_net("val")

    def dsr_of(series, N):
        r = series.pct_change().dropna().values
        r = r[np.isfinite(r)]
        sr = float(np.mean(r) / np.std(r))
        from scipy import stats as st
        return M.dsr(sr, len(r), var_sr, int(N), float(st.skew(r)), float(st.kurtosis(r, fisher=False)))

    dev_s, val_s = nav_series(src, "dev"), nav_series(src, "val")
    out = {
        "trial_id": "T0005",
        "family": "F1-S1-E2-X3-G1",
        "strategy": "F1(月/周/日) × S1(已确认分型) × E2(突破回踩) × X3(奔跑+谨慎态) × G1(无市场门)",
        "params_default_7_8": {
            "k": 2, "atr": 14, "breakout_buffer_atr": 0.05, "stop_atr": 0.3,
            "max_risk_pct": 0.12, "be_mfe_r": 1.0, "trail_mfe_r": 2.0, "trail_atr": 3.0,
            "fail_days": 3, "fail_mfe_r": 0.5, "half_days": 10, "time_stop_days": 20,
            "port_dd_half": 0.10, "port_dd_stop": 0.15, "risk_per_trade": 0.012,
            "max_pos": 10, "rank": "breakout_strength",
        },
        "dev": {"cagr": dev_m["cagr"], "mdd": dev_m["mdd"], "sharpe": dev_m["sharpe"],
                "vol": dev_m["vol"], "calmar": dev_m["calmar"], "n_trades": dev_m["n_trades"],
                "n_codes": dev_m["n_codes"], "win_rate": dev_m["win_rate"],
                "avg_hold_days": dev_m["avg_hold_days"], "avg_positions": dev_m["avg_positions"],
                "time_in_market": dev_m["time_in_market"],
                "cagr_ex_top5": dev_m["cagr_ex_top5"],
                "top5_share": dev_m["top5_share"],
                "top5_share_gross": dev_m["top5_share"],
                "top5_share_net": top5_net_dev,
                "top5_share_denominator": "top5_share is legacy gross (top5 / sum of positive pnl); top5_share_net = top5 / net pnl",
                "total_return": dev_m["total_return"]},
        "val": {"cagr": val_m["cagr"], "mdd": val_m["mdd"], "sharpe": val_m["sharpe"],
                "vol": val_m["vol"], "calmar": val_m["calmar"], "n_trades": val_m["n_trades"],
                "n_codes": val_m["n_codes"], "win_rate": val_m["win_rate"],
                "avg_hold_days": val_m["avg_hold_days"], "avg_positions": val_m["avg_positions"],
                "time_in_market": val_m["time_in_market"],
                "cagr_ex_top5": val_m["cagr_ex_top5"],
                "top5_share": val_m["top5_share"],
                "top5_share_gross": val_m["top5_share"],
                "top5_share_net": top5_net_val,
                "top5_share_denominator": "top5_share is legacy gross (top5 / sum of positive pnl); top5_share_net = top5 / net pnl",
                "total_return": val_m["total_return"]},
        "grading": {
            "dev_cagr_gt0": bool(dev_m["cagr"] > 0),
            "val_cagr_gt0": bool(val_m["cagr"] > 0),
            "dev_mdd_le25": bool(abs(dev_m["mdd"]) <= 0.25),
            "val_mdd_le25": bool(abs(val_m["mdd"]) <= 0.25),
            "dev_trades_ge30": bool(dev_m["n_trades"] >= 30),
            "val_trades_ge30": bool(val_m["n_trades"] >= 30),
            "dev_codes_ge10": bool(dev_m["n_codes"] >= 10),
            "val_codes_ge10": bool(val_m["n_codes"] >= 10),
            "selection_pass_7_9_5": False,
            "tier_12_2": "无可利用边际（min(dev,val) = -2.53% < 20%）",
        },
        "multiplicity": {
            "n_new_configs_registered": int(src["mult"]["n_new_configs"]),
            "n_configs_unique_paths": 32,
            "n_sel_before": int(src["mult"]["n_project_sel"]),
            "n_for_dsr_in_t0005_report": int(src["mult"]["n_total_for_dsr"]),
            "pbo_neighbor_dev": src["mult"]["pbo_neighbor"]["dev"]["pbo"],
            "pbo_neighbor_val": src["mult"]["pbo_neighbor"]["val"]["pbo"],
            "dsr_project_dev_N397": src["mult"]["dev"]["dsr"],
            "dsr_project_val_N397": src["mult"]["val"]["dsr"],
            "family_dsr_dev_N18": src["mult"]["dev_family_dsr"]["dsr"],
            "family_dsr_note": "18 个邻域变体高度相关，Var(SR_trials) 被系统低估，族内 DSR 偏高；门槛口径是项目级 N。",
        },
        "ablation": {
            "n_variants": 15,
            "dev_best": "V7 X3→X1 固定2R目标 +8.12%（MDD -6.01%）",
            "val_best": "A1 E2+固定20日 -0.11%（MDD -12.68%）",
            "val_range": [-0.0359, -0.0011],
            "n_val_positive": 0,
            "v1_zero_trades": "F1→F2（季/月）在 12% 初始风险门下信号集为空，两段 0 笔——是无法交易，不是无损失。",
        },
        "neighborhood": {
            "n_variants": 18,
            "dev_range": [0.0154, 0.0550],
            "val_range": [-0.0260, -0.0135],
            "n_val_positive": 0,
        },
        "exit_structure": {
            "stop_trigger": {"dev": {"n": 37, "mean_pnl": 13334.0, "win_rate": 0.784},
                             "val": {"n": 36, "mean_pnl": 2366.0, "win_rate": 0.611}},
            "open_exits": {"dev": {"n": 65, "mean_pnl": -1917.0, "win_rate": 0.185},
                           "val": {"n": 120, "mean_pnl": -1760.0, "win_rate": 0.167}},
            "note": "stop_trigger 含跟踪/保本止损，是“锁住利润”的那一类；时间/结构退出（open）是亏损来源。",
        },
        "signal_funnel": {
            "n_signals_2007_2020": src["funnel"]["n_signals"],
            "risk_gate_pass_share": {"dev": 0.0132, "val": 0.0465},
            "risk_gate_median_r_over_p0": {"dev": 0.2924, "val": 0.2452},
            "note": "12% 初始风险门是主要约束：周线结构低点离现价太远，232 个有效成交信号来自 25558 个信号。",
        },
        "selfcheck": src["selfcheck"],
        "reconciliation": {
            "dev": src["recon"]["dev"]["independent_ledger"],
            "val": src["recon"]["val"]["independent_ledger"],
            "fill_price_mismatch_gt_1e-6": {
                "dev": src["recon"]["dev"]["fill_price_check"]["n_price_mismatch_gt_1e-6"],
                "val": src["recon"]["val"]["fill_price_check"]["n_price_mismatch_gt_1e-6"],
            },
            "causality": src["recon"]["causality"],
        },
        "yearly": {
            w: {int(r["year"]): {"return": float(r["return"]), "mdd": float(r["mdd"])}
                for _, r in src["yearly"][w].iterrows()} for w in ("dev", "val")
        },
        "dsr_at_project_N": {},
    }
    return out


def _append_t0005_pbo_ci() -> None:
    """把 MAIN（T0005 18 配置邻域）的 PBO 与 T0005 的 bootstrap CI 补进最终统计（幂等）。"""
    src = source_numbers()
    mult = src["mult"]
    nb = pd.read_csv(os.path.join(T0005_DIR, "30_邻域.csv"))

    pbo_path = os.path.join(FINAL_DIR, "最终PBO_maxstat.csv")
    pbo_rows = read_rows(pbo_path)
    if not any(r["family"] == "MAIN" for r in pbo_rows):
        cols = list(pbo_rows[0].keys())
        for w in ("dev", "val"):
            sub = nb[nb["window"] == w]
            best = sub.loc[sub["cagr"].idxmax()]
            m = mult["pbo_neighbor"][w]
            pbo_rows.append({
                "kind": "PBO", "family": "MAIN", "window": w,
                "pbo": m["pbo"], "n_splits": m["n_comb"], "n_candidates": m["n_configs"],
                "T": m["n_days"], "median_oos_rank": m["median_rank"],
                "logit_median": m.get("logit_mean", ""),
                "best_config": f"{best['id']}_{best['desc']}",
                "best_cagr": float(best["cagr"]),
                "best_cagr_p05": float(best["bootstrap_cagr_lo"]),
            })
        write_rows(pbo_path, pbo_rows, cols)

    ci_path = os.path.join(FINAL_DIR, "最终置信区间.csv")
    ci_rows = read_rows(ci_path)
    if not any(r["family"] == "MAIN" for r in ci_rows):
        cols = list(ci_rows[0].keys())
        for w in ("dev", "val"):
            b = src["stats"][w]["metrics"]["bootstrap"]
            ci_rows.append({
                "family": "MAIN", "config": "T0005", "window": w,
                "cagr_point": b["cagr_point"], "cagr_p05": b["cagr_lo"],
                "cagr_p50": b["cagr_median"], "cagr_p95": b["cagr_hi"],
                "p_cagr_le_0": b["p_cagr_le_0"], "T": b["n_obs"], "B": b["n_boot"],
                "block": int(b["block_len"]),
            })
        write_rows(ci_path, ci_rows, cols)

    # 口径说明
    sm_path = os.path.join(FINAL_DIR, "T0005_纳入后_汇总.json")
    sm = jload(sm_path)
    sm["pbo_ci_note"] = (
        "最终PBO_maxstat.csv 与 最终置信区间.csv 已补 MAIN 行：PBO 来自 18 配置邻域（12870 组合）；"
        "CI 来自 T0005 日净值 stationary bootstrap（块长 20，B=2000），其 cagr_point 用 243 交易日年化，"
        "与主指标 365.25 日历天年化差约 0.15pp（dev 4.31% vs 4.16%，val -2.62% vs -2.53%）。")
    jdump(sm_path, sm)


def stage_final_stats() -> dict:
    rows = read_rows(LEDGER)
    total = summarize(rows)
    by_family = {fam: summarize([x for x in rows if x["family"] == fam])
                 for fam in sorted({x["family"] for x in rows})}
    by_cat = {cat: summarize([x for x in rows if x["category"] == cat])
              for cat in sorted({x["category"] for x in rows})}
    N1 = int(total["in_Nsel"])
    N2 = int(total["in_Nsel"] + total["in_Nplatform"])
    N3 = int(total["in_Nsel"] + total["in_Nplatform"] + total["in_Ndiag"])
    gates = read_rows(os.path.join(FINAL_DIR, "统一门槛.csv"))
    grade_counts = dict(Counter(g["grade"] for g in gates))
    t5 = _t0005_block()

    # 门槛覆盖率必须用“总账 in_Nsel 求和”，不能用门槛行数代替：
    # 478 行门槛表覆盖 505 个 N_sel（505+228=733）；T0005 的 19 行是 714→733 的增量。
    gated_keys = {(g["family"], g["config_id"]) for g in gates}
    account_cats = {"prereg", "posthoc", "sensitivity"}
    N_cover = int(sum(float(r["in_Nsel"] or 0) for r in rows
                      if r["category"] in account_cats and (r["family"], r["config_id"]) in gated_keys))
    N_unrated = int(N1 - N_cover)
    b_rows = [g for g in gates if g["grade"] == "B_conditional"]
    b_keys = {(g["family"], g["config_id"]) for g in b_rows}
    b_ledger_rows = sum(1 for r in rows if (r["family"], r["config_id"]) in b_keys)
    cg = pd.read_csv(os.path.join(RES, "统计整合", "候选分级.csv"))
    cg_b = cg[cg["grade"] == "B_conditional"]
    candidate_grading = {
        "config_level": {
            "caliber": "最终试验总账/统一门槛：全项目、配置级；每配置 1 行门槛、总账 5×2=10 行",
            "n_configs": len(b_rows),
            "n_rows": b_ledger_rows,
            "trend_s3_configs": sum(1 for g in b_rows if g["strategy_id"].startswith("TREND@S3")),
            "vcb_configs": sum(1 for g in b_rows if g["family"] == "VCB"),
            "config_ids": sorted(g["config_id"] for g in b_rows),
            "source": "03_结果/最终统计/统一门槛.csv；03_结果/最终试验总账.csv",
        },
        "row_level_6_families": {
            "caliber": "候选分级.csv：仅第一阶段 6 族（TREND/STRUCT/EXIT/MKT/STAT/ECOLOGY）、行级；不含 VCB 等后加族",
            "n_rows": int(len(cg_b)),
            "n_configs": int(cg_b["config_id"].nunique()),
            "trend_s3_rows": int((cg_b["strategy_id"] == "TREND@S3_波动收缩突破").sum()),
            "trend_s1_rows": int((cg_b["strategy_id"] == "TREND@S1_月周日共振突破").sum()),
            "source": "03_结果/统计整合/候选分级.csv",
        },
        "note": "两个口径族集合与粒度不同，不能混用；'全部 S3' 只在配置级 5 个成立，行级 12 行含 1 行 S1。",
    }
    gate_coverage = {
        "n_gate_rows": len(gates),
        "N_sel_covered_by_gate_rows": N_cover,
        "N_sel_unrated": N_unrated,
        "N_sel_total": N1,
        "identity": f"{N_cover}+{N_unrated}={N_cover + N_unrated}",
        "legacy_714_caliber": {"covered": N_cover - 19, "unrated": N_unrated,
                               "note": "旧值 486/228 是 714 口径（486+228=714）；不含 T0005 的 19 个 N_sel。当前 478 行门槛表含 T0005 19 行，故 505/228。"},
        "unrated_reason": "无 dev/val 两窗口记录：STRUCT 219（只有 2008-2020 合并窗口）+ STAT 9（统计框架夹具）；只能作负证据",
    }

    old = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    var_sr = float(old["statistics"]["dsr"]["var_sr_daily"])
    best = old["statistics"]["dsr"]["best"]
    M = load_module("02_代码/最终试验总账.py", "fl4")

    def ladder(N):
        sr0 = math.sqrt(var_sr) * ((1 - 0.5772156649015329) * __import__("scipy").stats.norm.ppf(1 - 1.0 / N)
                                   + 0.5772156649015329 * __import__("scipy").stats.norm.ppf(1 - 1.0 / (N * math.e)))
        out = {"N": int(N), "sr0_annual": float(sr0 * math.sqrt(243))}
        for w in ("dev", "val"):
            d = best[w]
            r = M.dsr(d["sr_daily"], d["T"], var_sr, int(N), d["skew"], d["kurt"])
            out[f"dsr_best_{w}"] = float(r["dsr"])
        return out

    ladder_all = {"N1_sel": ladder(N1), "N2_sel_plus_platform": ladder(N2), "N3_all_cells": ladder(N3)}

    # T0005 在项目 N 下的 DSR
    src = source_numbers()
    from scipy import stats as st
    dev_s, val_s = nav_series(src, "dev"), nav_series(src, "val")
    for w, s in (("dev", dev_s), ("val", val_s)):
        r = s.pct_change().dropna().values
        sr = float(np.mean(r) / np.std(r))
        for N in (N1, N2, N3):
            d = M.dsr(sr, len(r), var_sr, N, float(st.skew(r)), float(st.kurtosis(r, fisher=False)))
            t5["dsr_at_project_N"].setdefault(f"N{N}", {})[w] = d["dsr"]

    summary = {
        "generated_at_local": dt.datetime.now().isoformat(timespec="seconds"),
        "tag": TAG,
        "counts": {"total": total, "by_family": by_family, "by_category": by_cat,
                   "N_dsr_ladder": {"N1_sel": N1, "N2_sel_plus_platform": N2, "N3_all_cells": N3},
                   "budget": {"prereg_hard_budget_select": 40, "prereg_hard_budget_with_ablation": 64,
                              "N_sel_over_budget_ratio": round(N1 / 64.0, 2)}},
        "grade_counts": grade_counts,
        "candidate_grading": candidate_grading,
        "gate_coverage": gate_coverage,
        "t0005": t5,
        "dsr_ladder": ladder_all,
        "var_sr_daily": var_sr,
        "var_sr_note": f"Var(SR_trials) 沿用 {old['statistics']['dsr']['n_sr_trials']} 条日净值序列的估计；T0005 的 2 条序列未重估（影响 <1%，且 N 上升使 SR0 上升、DSR 只降不升）。",
        "families": CURATED_FAMILIES,
        "a_candidate": [g for g in gates if g["grade"] == "A_candidate"],
        "mechanical_G_pass": [g for g in gates if "G_pass" in g["gates"]],
        "n_gate_rows": len(gates),
    }
    jdump(os.path.join(FINAL_DIR, "T0005_纳入后_汇总.json"), summary)

    # 补丁最终判定.json
    v = copy.deepcopy(old)
    v["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    v["generated_at_local"] = dt.datetime.now().isoformat(timespec="seconds")
    v["supersedes_note"] = (f"{TAG} 本文件由 `02_代码/最终交付.py ledger` 补丁：T0005 已实际完成 "
                            "dev/val（+4.16% / -2.53%）并淘汰；计数按总账追加行重算。")
    v["counts"]["total"] = total
    v["counts"]["by_family"] = by_family
    v["counts"]["by_category"] = by_cat
    v["counts"]["N_dsr_ladder"] = {"N1_sel": N1, "N2_sel_plus_platform": N2, "N3_all_cells": N3}
    v["counts"]["budget"] = {"prereg_hard_budget_select": 40, "prereg_hard_budget_with_ablation": 64,
                             "N_sel_over_budget_ratio": round(N1 / 64.0, 2)}
    v["counts"]["stage1_recorded_vs_corrected"] = {
        "N_sel": [365, N1],
        "N_cell": [500, int(total["in_Ncell"])],
        "N_diag": [369, int(total["in_Ndiag"])],
        "note": ("第一阶段总账记录 N_diag=369；整合结论披露 411（ECOLOGY 36 而非 1、MKT 8 而非 1）。"
                 f"当前总账（含 T0005）为 N_sel={N1} / N_cell={int(total['in_Ncell'])} / "
                 f"N_diag={int(total['in_Ndiag'])}；00:51 快照的 714/1,213/424 是历史口径，已被当前值取代。"),
    }
    v["statistics"]["dsr"]["ladder"] = ladder_all
    v["statistics"]["dsr"]["var_sr_note"] = summary["var_sr_note"]
    v["holdout_gate"]["main_candidate_T0005"] = {
        "trial_id": "T0005",
        "dev": {"cagr": t5["dev"]["cagr"], "mdd": t5["dev"]["mdd"], "n_trades": t5["dev"]["n_trades"]},
        "val": {"cagr": t5["val"]["cagr"], "mdd": t5["val"]["mdd"], "n_trades": t5["val"]["n_trades"]},
        "grade": "C_negative_valid", "fail_gate": "G_fail_negative",
        "decision": "discard",
        "reason": "§7.9.5 两段净收益>0 不通过；PBO 0.311/0.756；15 个消融与 18 个邻域无一 val 转正。",
    }
    v["holdout_gate"]["grade_counts"] = grade_counts
    v["holdout_gate"]["n_gate_rows"] = len(gates)
    v["holdout_gate"]["N_sel_covered_by_gate_rows"] = N_cover
    v["holdout_gate"]["N_sel_unrated"] = N_unrated
    v["holdout_gate"]["N_sel_total"] = N1
    v["holdout_gate"]["coverage_note"] = (
        f"当前门槛表 {len(gates)} 行覆盖 {N_cover} 个 N_sel，未评级 {N_unrated} 个，"
        f"{N_cover}+{N_unrated}={N_cover + N_unrated}。旧值 486/228 是 714 口径"
        "（486+228=714，不含 T0005 的 19 个 N_sel）；478 行含 T0005 19 行后应为 505/228。"
        "覆盖数按总账 in_Nsel 求和，不能与门槛行数互换。")
    v["holdout_gate"]["coverage_derivation"] = {
        "non_MAIN_gate_rows": len([g for g in gates if g["family"] != "MAIN"]),
        "non_MAIN_covered": N_cover - 19,
        "MAIN_gate_rows": len([g for g in gates if g["family"] == "MAIN"]),
        "MAIN_covered": 19,
        "current_covered": N_cover,
        "current_unrated": N_unrated,
        "current_total": N1,
        "stale_714": {"covered": N_cover - 19, "unrated": N_unrated, "sum": N_cover - 19 + N_unrated},
    }
    v["grade_calibers"] = candidate_grading
    v["verdict"]["statement"] = (
        f"全项目 {N1} 个可部署配置、{int(total['in_Ncell'])} 个账户单元、{int(total['in_Ndiag'])} 个诊断单元、"
        f"{int(total['in_Nplatform'])} 个平台格、{int(total['in_Naudit'])} 个审计夹具；"
        "0 个配置通过进入留出的全部硬门槛（R/D/P/G/S/T）。"
        "官方主候选 T0005 已完成 dev/val：dev +4.16%/MDD -9.05%、val -2.53%/MDD -13.80%，"
        "验证期不通过 §7.9.5，主候选淘汰；15 个消融与 18 个参数邻域无一在 val 转正。"
        "机械通过 G 的 5 个格子（4 个 TREND@S3 事后探索 + 1 个 VCB-P20/T40 参数平台格）全部被 R/S/T 闸否决。"
        "留出（2021-2025）不运行。")
    v["update_commands"] = [
        "python3 02_代码/最终交付.py verify      # 注册表链 + 总账链 + T0005 对账 + 计数一致性",
        "python3 02_代码/最终交付.py selfcheck   # 交付物自检（36 项）",
        "python3 02_代码/最终试验总账.py --stage all  # 注意：该脚本不含 T0005 行，重跑会覆盖本目录聚合",
        "python3 02_代码/实验登记.py verify --strict    # 官方链校验（只读）",
    ]
    jdump(os.path.join(FINAL_DIR, "最终判定.json"), v)

    _append_t0005_pbo_ci()

    # 报告补丁横幅（幂等）：当前口径覆盖在历史正文之前，不改历史数字
    rep = os.path.join(FINAL_DIR, "00_最终报告.md")
    txt = open(rep, encoding="utf-8").read()
    banner_mark = "历史快照：生成于 2026-10-29 00:51"
    if banner_mark not in txt:
        banner = (
            f"> **[{banner_mark}；当前口径以本横幅为准 / {TAG}]**\n"
            ">\n"
            "> **当前口径（2026-10-29 收尾；本文件正文的旧数字不得作为当前事实引用）**\n"
            f"> - **计数**：`N_sel={N1}`、`N_cell={int(total['in_Ncell'])}`、`N_diag={int(total['in_Ndiag'])}`、"
            f"`N_platform={int(total['in_Nplatform'])}`、`N_audit={int(total['in_Naudit'])}`；总账 {int(total['rows'])} 行。"
            "正文的 714/1,213/424 与 1,631 行是 **00:51 历史快照**。\n"
            "> - **T0005**：已于 00:29-00:53 实际完成 dev/val；dev 净年化 **+4.16%**（MDD -9.05%，103 回合）、"
            "val 净年化 **-2.53%**（MDD -13.80%，159 回合），按 §7.9.5 **淘汰**。正文 §2 的 T0005 旧状态（`F_never_run`）"
            "与 §6 第 6 条是**历史状态，不是当前事实**；总账同时保留 `F00002 not_run`、`F01629 planned` 两行历史快照，"
            "当前事实以 **68 行 done**（19 N_sel / 38 cell / 30 diag）为准。\n"
            "> - **门槛覆盖**：当前门槛表 **478 行**覆盖 **505** 个 `N_sel`、未评级 **228** 个（505+228=733）。"
            "正文 §6 第 1 条的“459 行覆盖 486 / 228”是 **714 口径**（486+228=714，不含 T0005 的 19 个 N_sel）。\n"
            "> - **B_conditional**：配置级 **5 个**（4 个 `TREND@S3` + `VCB-P20/T40`；总账 5×2=10 行）；"
            "旧行级 **12 行/8 配置**（11 行 S3 + 1 行 S1，仅第一阶段 6 族）。两个口径不能混用。\n"
            "> - **小盘 COMBO**：val **+0.33%** 以首日收盘净值 982,382 为基准（漏掉 2016-01-04 熔断日 -1.76%）；"
            "按初始 100 万计 val **-0.03%**，不得写成“两段为正”。\n"
            "> - **2021+ 扫描**：最近一次交付验收扫描 **469** 个绩效表、0 违规；历史 README 的 470 为上一轮文件集，"
            "差 1 个文件，结论不变。\n"
            ">\n"
            "> 最新机器口径见 `03_结果/最终统计/T0005_纳入后_汇总.json`、`03_结果/最终统计/最终判定.json`、"
            "根目录 `04_最终结论.json` 与 `04_证据/最终口径对账.json`。\n\n"
            "---\n\n"
            "> **以下为 2026-10-29 00:51 历史正文（正文数字均为当时口径；当前口径见上）。**\n\n")
        open(rep, "w", encoding="utf-8").write(banner + txt)

    print(f"final_stats: N_sel={N1} N_cell={int(total['in_Ncell'])} N_diag={int(total['in_Ndiag'])}; "
          f"A_candidate={len(summary['a_candidate'])}")
    return summary


# ---------------------------------------------------------------------------
# conclusion：04_最终结论.json
# ---------------------------------------------------------------------------

def stage_conclusion() -> dict:
    summary = jload(os.path.join(FINAL_DIR, "T0005_纳入后_汇总.json"))
    v = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    pbo = read_rows(os.path.join(FINAL_DIR, "最终PBO_maxstat.csv"))
    ci = read_rows(os.path.join(FINAL_DIR, "最终置信区间.csv"))
    facts = audit_facts()
    total = summary["counts"]["total"]
    N1 = summary["counts"]["N_dsr_ladder"]["N1_sel"]
    t5 = summary["t0005"]
    t5.setdefault("multiplicity", {})["ledger_note"] = (
        "32/397 是 T0005 族报告自己的 DSR 口径（365+32，15 个消融×2 + 18 邻域 + 主候选去重）；"
        "项目总账只把 19 个配置计 N_sel（主候选 dev + S01-S18），15 个消融×2 计 N_diag。"
        "当前项目 N=733，引用 DSR 时不能用 397 代替项目 N。")
    # top5 旧字段是毛分母：显式补 gross/net 与 denominator，避免机器读者误读
    for _w in ("dev", "val"):
        _src = facts["top5"][_w]
        t5.setdefault(_w, {})
        t5[_w]["top5_share"] = _src["top5_over_gross"]
        t5[_w]["top5_share_gross"] = _src["top5_over_gross"]
        t5[_w]["top5_share_net"] = _src["top5_over_net"]
        t5[_w]["top5_share_denominator"] = "legacy top5_share is gross; top5_share_net = top5 / net pnl"

    _g = _gate_rows().dropna(subset=["dev_cagr", "val_cagr"])

    def _fam_best(key):
        x = _g[_g.family == key]
        if x.empty:
            return None
        bd = x.loc[x.dev_cagr.idxmax()]
        bv = x.loc[x.val_cagr.idxmax()]
        return {"dev": {"config_id": bd.config_id, "strategy_id": bd.strategy_id,
                        "cagr": float(bd.dev_cagr), "mdd": float(bd.dev_mdd)},
                "val": {"config_id": bv.config_id, "strategy_id": bv.strategy_id,
                        "cagr": float(bv.val_cagr), "mdd": float(bv.val_mdd)}}

    def fam_entry(key):
        c = CURATED_FAMILIES[key]
        return {"family": key, **c, "scope": FAMILY_SCOPE.get(key, {}),
                "family_best": _fam_best(key)}

    _pmap = {(x["family"], x["window"]): x["p_maxstat"]
             for x in facts["maxstat"]["values"] if x.get("p_maxstat") is not None}
    pbo_enriched = []
    for r in pbo:
        r2 = dict(r)
        if r2.get("kind") == "MAXSTAT":
            r2["p_maxstat"] = _pmap.get((r2.get("family"), r2.get("window")))
        pbo_enriched.append(r2)

    questions = [
        {"id": "Q1", "question": "已确认分型 + 高低点抬高的结构，在扣费后是否有正期望",
         "experiment": "主候选 vs A1（E2+固定20日）；结构族事件研究；统一引擎基线",
         "answer": "否",
         "evidence": ("A1（只进场+固定20日）dev +5.79% / val -0.11%；全状态机 A7 dev +4.16% / val -2.53%。"
                      "结构族 219 个配置只有 2008-2020 合并窗口，事件层 20 日超额 -1.21%~+0.11%。"
                      "统一引擎 60 日新高突破的前向 20 日超额 11/13 年为负。"),
         "implication": "结构本身无边际；后续周期叠加与退出微调都没有可证伪的正期望。"},
        {"id": "Q2", "question": "月/周/日三层叠加是否优于单层日线",
         "experiment": "V1（F1→F2 季/月）、V2（F1→F3 周/日）对照主候选",
         "answer": "否（“更少交易”不是“更多信息”）",
         "evidence": ("V1 两段 0 笔成交——12% 初始风险门下信号集为空，是“无法交易”。"
                      "V2 dev -1.72% / val -3.44%（vs 主候选 +4.16% / -2.53%），去掉高周期反而更差；"
                      "但主候选 dev 的正收益几乎全来自 2014-2015（+26.4%/+15.1%），其余 6 年平均 -0.81%。"),
         "implication": "多周期层的作用是把信号压到很少（dev 仅 103 回合、平均持仓 0.83 只），不是提供独立 alpha。"},
        {"id": "Q3", "question": "“买入后不符合预期”的早期识别，是否提高终值",
         "experiment": "A6（去掉 3 日失败退出）vs A5（含 3 日失败）；退出族 FAIL3 臂",
         "answer": "否",
         "evidence": ("A6 dev +5.70% / val -1.46%，比含 3 日失败的 A5（+4.16% / -2.53%）两段都更好；"
                      "退出族 FAIL3 对“第 10 日前已退出组”改善 +1.46/+1.93pp，但对幸存组伤害 -0.06~-1.92pp，净 +0.04/+0.02pp（CI 跨 0）。"),
         "implication": "3 日失败退出只是把噪声提前兑现成成本；没有证据支持它提高终值。"},
        {"id": "Q4", "question": "谨慎态（个股减半 + 组合降险）是否降低回撤而不显著降低收益",
         "experiment": "V6（X3→X2 去谨慎态）、V8（G1→G2 市场门）对照；退出族六态状态机",
         "answer": "不稳定，不能作为降回撤的证据",
         "evidence": ("去谨慎态 V6：dev +4.38%（MDD -9.13%）/ val -3.01%（MDD -15.60%）；含谨慎态 A7：dev +4.16%（-9.05%）/ val -2.53%（-13.80%）。"
                      "dev 略差、val 略好，符号不一致；退出族里完整六态状态机是 dev 最差臂（年份等权 -0.128R）。"),
         "implication": "谨慎态改变的是回撤路径的形态，不是收益符号；把它当“降回撤工具”没有跨窗口证据。"},
        {"id": "Q5", "question": "“让利润奔跑”（无目标 + 跟踪）是否优于固定 2R 目标",
         "experiment": "V7（X3→X1 固定 2R 目标）vs V6（X2 奔跑）/ A3；退出族 B0（无目标）vs TG2（2R）",
         "answer": "否——本样本右尾不足，固定目标更好",
         "evidence": ("V7 dev +8.12%（MDD -6.01%）/ val -2.13% 优于 V6 dev +4.38% / val -3.01%，两段都更好；"
                      "退出族不设目标的 B0 只在 2014-2015 大胜、2016-2020 小负，2R 目标 TG2 反过来——是 regime 下注，不是普适优化。"),
         "implication": "“让利润奔跑”在本样本没有可复现的正边际；但它也没有被证明有害到必须禁止，属未验证政策。"},
        {"id": "Q6", "question": "上述净边际的量级与不确定性，是否支持年化 60%",
         "experiment": "全项目门槛 R/D/P/G/S/T 与 §12.1 G1-G10",
         "answer": "否；在本项目测试空间内 60% 不被支持，G1 未被直接检验",
         "evidence": ("全项目两段都为正的配置是 32 个（TREND 5 + VCB 18 + 小盘 9），"
                      "同时过两段 MDD≤25% 的只有 5 个（4 个 TREND@S3 未登记 + VCB-P20/T40 敏感性网格）。"
                      "最高全窗 dev 是 +20.40%（小盘 ADVQ12_s5_h40，MDD -69.15%）；最高 val 是 +8.70%（VCB-N120_T40）；"
                      "最高 min(dev,val) 是 8.32%（VCB-N120_T40，dev MDD -35.3%、DSR 0.622/0.1139、PBO 0.614）。"
                      "执行压力 1,250+192 格中 0 格 ≥20%、0 格 ≥60%，最高 12.1%（固定交易集合重定价）。"),
         "implication": "按 §12.2 阈值作描述性套用：min(dev,val) < 20% → “无可利用边际”。60% 与 8.70% 约 7 倍差距；不等于任何策略都不可能 60%。"},
    ]

    cross = {
        "dynamic_parameters": {
            "evidence": ("T0005 18 个参数邻域：dev **1.54%~5.50%**（S06 最大初始风险 10% 最低）、val -2.60%~-1.35%，0 个 val 转正。"
                         "VCB 9 个信号配置：dev 0/9 的 20 日超额为正、val 7/9 为正，PBO 0.614。"
                         "趋势族 30 个参数邻域只报告符号稳定性，未取最优。"),
            "conclusion": "参数邻域内符号不稳定或全负；不存在“调对了参数就能转正”的证据。敏感性只作稳健性披露，不参与选择。",
        },
        "slots_5_10_20": {
            "evidence": ("T0005：5 仓 S16 dev +2.09%/val -2.48%；10 仓主候选 +4.16%/-2.53%；20 仓 S17 dev +2.59%/val -1.39%。"
                         "执行压力 B55H20：5 仓 dev -12.9%、10 仓 -13.2%；小盘低波 5/10/20 仓网格差 3-8pp。"),
            "conclusion": "仓位数改变噪声量级（3-8pp/年），不改变符号；20 仓把小盘/低波暴露摊薄，但仍在负区间。",
        },
        "capital": {
            "evidence": ("执行压力：B55H20 20k dev -12.2%/val -36.3%；100 万 -13.2%/-18.8%；1000 万 -12.9%/-22.1%。"
                         "20k/10 的 8 年累计费用占初始资金 47.7%，最低佣金是主要杀手。"
                         "小盘低波容量：100 万部署 91.4%/CAGR +3.2%；1000 万 67.6%/+2.3%；1 亿 13.7%/-1.5%。"),
            "conclusion": "小账户不是 60% 的来源；大资金受 2% ADV20 硬约束。实际容量约 1,000 万元（小盘族）。",
        },
        "cost": {
            "evidence": ("C-REAL-V1（主口径）：佣金 2.5bp（最低 5 元）+ 历史印花税/过户费 + 滑点 10bp + 0.02√参与率 + 2% ADV20。"
                         "T0005 显性费用 18,198（dev）/20,130（val）；滑点已含在成交价、未计入 cost_paid，全成本约 0.4-0.5%/年；"
                         "退出族每笔往返约 0.36%；小盘 COMBO 成本×2 从 +3.2% 降到 +0.8%（full）。"
                         "VCB 不适用本表：10bp 固定 + 10% 参与率、无冲击项，dev 高估约 0.44pp/年。"),
            "conclusion": "成本压力只移动几个百分点，没有任何配置因成本从正翻负为 60% 级别；但跨族成本口径不同，VCB 与 T0005 不能无注明并表。",
        },
        "delay": {
            "evidence": ("小盘 COMBO 延迟 1 日 full +2.8%（基准 +3.2%）；执行压力 B55H20 100 万/10 仓 delay=1 的 5 个种子 dev -8.31%~-15.61%，"
                         "全部配置 dev -2.42%~-20.10%；T0005 未做延迟压力（缺口，已披露）。"),
            "conclusion": "1 日延迟通常损失 0.4-4pp/年，不改变符号；T0005 缺该项压力，是其 S 闸未验证的原因之一。",
        },
        "board": {
            "evidence": ("T0005/VCB/小盘的账户层宇宙是 U-MAINBOARD-V1：sh.60*/sz.00*（含原中小板），不含创业板/科创板/北交所；ST、td_since_list<120 与 days_since_ipo<250 排除；"
                         "主宇宙日均约 686-2,183 只（2008 年 686、2020 年 2,183）。"
                         "但 TREND 宇宙为 sh.60/68+sz.00/30（成交 15.84% 非主板）、STRUCT 未排除 300/688（事件 13.93% 非主板），MKT 规则含创业板/科创板。"),
            "conclusion": "主板结论只适用于 T0005/VCB/小盘；TREND/STRUCT 的结论带更宽宇宙与执行口径污染；不能写“所有结论只覆盖主板”，也不能把主板结论外推到创业板/科创板/北交所。",
        },
        "liquidity": {
            "evidence": ("主宇宙要求 ADV20 ≥ 3,000 万元；小盘族实测 ADV 五分位毛价差 18.0pp/年，规模三分位仅 5.6pp；"
                         "规模溢价 dev +1.45%/20 日（t=3.36）、val +0.06%（t=0.15）；"
                         "趋势回调 L5 超额随成交额单调下降（Q1 +1.26% → Q5 -1.09%）。"),
            "conclusion": "“规模效应”主要是流动性代理，且 2016 年后消失；低流动性层的正超额无法在 1,000 万以上资金复制。",
        },
        "market_state": {
            "evidence": ("MKT 族**主口径**（A 偏广度）dev +13.89%、val -7.33%（main_dev_config_val）；**族内真实最高 val 是 D_串联过滤 -2.35%（family_best_val）**，不是 -7.33%。"
                         "风险预算 dev 有效、val 有害，控制当期波动后偏相关≈0；"
                         "T0005 的 G2 市场门（V8）dev +4.23%/val -2.16%，与无门的 +4.16%/-2.53% 差异在噪声内。"),
            "conclusion": "市场状态层最多改变暴露与回撤速度，不制造 alpha；把它当择时信号没有跨窗口证据。",
        },
        "expectation_met": {
            "definition": "“符合预期” = 入场后 10 个交易日内 MFE ≥ 1R（预注册 §8.3 原文，不得事后改成“最终盈利”）",
            "evidence": ("退出族：第 10 日 MFE≥2R 的交易里，最终仍有 60.6%(A)/56.8%(B) 亏损；"
                         "第 10 日 MFE<1R 的减半在 T0005 的 S12（去掉 10 日减半）对照下 dev +4.92% vs +4.16%，小幅更好。"),
            "conclusion": "“符合预期”有信息（后续分布不同），但没有无条件可执行增益；不能把“曾经浮盈”当“会赚钱”。",
        },
        "expectation_missed": {
            "evidence": ("A6 去掉 3 日失败退出后两段都更好；退出族 FAIL3 净改善 +0.04/+0.02pp（CI 跨 0）；"
                         "T0005 的 open 类退出（时间/结构）dev 65 笔均值 -1,917 元、胜率 18.5%，val 120 笔均值 -1,760 元、胜率 16.7%。"),
            "conclusion": "“不符合预期就早退”在本样本是负边际；真正亏钱的是时间/结构退出那一组，而不是止损。",
        },
        "caution_state": {
            "evidence": ("C1 个股（连续 2 日收盘<MA20 减半）与 C3 组合（回撤 10% 减半 / 15% 停开）在 V6 vs A7 的对照里："
                         "dev +4.38% vs +4.16%（去谨慎态更好）、val -3.01% vs -2.53%（含谨慎态更好）。"),
            "conclusion": "谨慎态是“不稳定”的降险工具：dev 负边际、val 正边际，不能宣称它降低回撤而不伤收益。",
        },
        "must_exit": {
            "evidence": ("T0005 退出结构：stop_trigger（含保本/跟踪止损）dev 37 笔均值 +13,334 元、胜率 78.4%；"
                         "val 36 笔均值 +2,366 元、胜率 61.1%。open（时间/结构）退出 dev 65 笔 -1,917 元、val 120 笔 -1,760 元。"
                         "T0005 的 2 笔是**停牌顺延后强制估值卖出**（reason=sell_persistence_10d，非退市）；"
                         "跨族同机制：小盘主单元 104 笔、COMBO full 16/733、VCB val 1 笔（250 日口径）。"),
            "conclusion": "“必须退出”的证据只支持“止损类退出锁住了利润”，不支持“时间止损/结构破坏退出”有增益；"
                          "这两类在 T0005 里是亏损的主要来源。停牌强平是跨族通用机制，方向通常偏乐观。",
        },
        "let_profits_run": {
            "evidence": ("V7 固定 2R 目标 dev +8.12%/val -2.13% 优于 V6 无目标 dev +4.38%/val -3.01%；"
                         "T0005 A1（固定 20 日）val -0.11% 优于 A3（保本+跟踪）val -3.01%；"
                         "退出族 B0（无目标）与 TG2（2R）互为 regime 胜负。"),
            "conclusion": "本样本右尾不足，“让利润奔跑”没有可复现正边际；但它也没有被证伪到可以反向使用——属未验证退出规则。",
        },
    }

    stats_block = {
        "N_sel": N1,
        "N_cell": int(total["in_Ncell"]),
        "N_diag": int(total["in_Ndiag"]),
        "N_platform": int(total["in_Nplatform"]),
        "N_audit": int(total["in_Naudit"]),
        "N_sens_subset": int(total["in_Nsens"]),
        "budget": {"select": 40, "with_ablation": 64,
                   "overrun_vs_select": 18.33,
                   "overrun_vs_with_ablation": 11.45,
                   "overrun_comparable_total": 11.92,
                   "overrun_ratio": 11.45,
                   "note": "N_sel=733 不含 30 个消融（在 N_diag）；预算 64 含消融，因此 733/64=11.45× 是混合口径。按选择预算 40 为 18.33×；可比总口径 (733+30)/64=11.92×。引用时必须说明是哪一个。"},
        "g4_range": {"configs": ["TREND@S3_纯时间20", "TREND@S3_宽止损4ATR+时间60",
                                 "TREND@S3_时间40+止损2ATR", "TREND@S3_目标2ATR+止损2", "VCB-P20_T40"],
                     "worst_rolling_12m_dev": [-0.064356, -0.153304, -0.182214, -0.071746, -0.147357],
                     "range": [-0.182214, -0.064356],
                     "note": "252 交易日滚动；5 个 SEL-G 全部 > -20%；原稿 -7.2~-18.2 漏掉 -6.44。"},
        "pbo_all_family_min": {"ECOLOGY": 0.002, "LONGMOM": 0.020, "STRUCTREV": 0.043, "SMALLLOWVOL": 0.241,
                               "TREND": 0.452, "note": "全族最低 PBO 不是 TREND 0.452；低 PBO ≠ 能赚钱。"},
        "family_best": {
            "best_dev": facts["best_dev"],
            "best_val": facts["best_val"],
            "best_min_dev_val": facts["best_min"],
            "note": "全部取自 统一门槛.csv（in_Nsel=1）；主候选不是族内最高，报告 §3.19 已分列。",
        },
        "two_segment_positive": {
            "n": facts["n_two_segment_positive"],
            "by_family": facts["two_segment_positive_by_family"],
            "note": "32 = TREND 5 + VCB 18 + 小盘 9；原稿 24 只排除 8 个 VCB 敏感性。",
        },
        "g_pass": {"n": len(facts["g_pass"]), "configs": facts["g_pass"],
                   "note": "5 个 SEL-G 通过者（4 个 TREND@S3 + VCB-P20/T40）；旧行级 B_conditional 12 行/8 配置是另一口径。"},
        "dsr_ladder": summary["dsr_ladder"],
        "var_sr_daily": summary["var_sr_daily"],
        "dsr_cross_caliber": {
            "stage1_N365": {"dsr": 0.000300, "var_sr_daily": 0.001345, "selected_sr_annual": 0.466,
                            "source": "03_结果/统计整合/01_整合结论.md"},
            "current_N733": {"dsr": 0.000662754421907755, "var_sr_daily": 0.0011164356612968494,
                             "selected_sr_annual": 0.4978, "selected": "VCB-P20_T40"},
            "note": "DSR 只在同一候选集/同一 var_sr 内随 N 单调；365→733 同时换了 var_sr 与被选配置，不能读成“N 增加 DSR 必降”。",
        },
        "maxstat": {"n_p": facts["maxstat"]["n_p"], "min": facts["maxstat"]["min"],
                    "nine_matrix_min": {"p": 0.374, "matrix": "trend_dev",
                                        "source": "03_结果/统计整合/多重性.json"},
                    "all_values": facts["maxstat"]["values"],
                    "note": "全项目 8 族 16 个 p 的全局最小是 0.071964（VCB dev）；9 个矩阵口径的最小才是 0.374。均未过 0.05。"},
        "pbo_maxstat": pbo_enriched,
        "top5_denominators": {"dev": facts["top5"]["dev"], "val": facts["top5"]["val"],
                              "note": "主数为 top5/净盈亏；括号/毛分母另列。T0005 val 净分母为负（-62.6%），该比例无经济意义。"},
        "neighborhood_range": facts["neighborhood"],
        "s3_exposure": facts["s3_exposure"],
        "trend_non_main": facts["trend_non_main"],
        "vcb_count": facts["vcb_count"],
        "bootstrap_ci_count": len(ci),
        "ci_includes_zero": int(sum(1 for r in ci if float(r["cagr_p05"]) <= 0 <= float(r["cagr_p95"]))),
        "null_calibration": {
            "dev_random_10slots_21d": 0.0457,
            "val_random_10slots_21d": -0.0581,
            "full_random_10slots_21d": 0.0086,
            "swing_between_windows_pp": 10.4,
            "expected_max_of_42_trials": {"dev": 0.134, "val": 0.040},
            "expected_max_of_365_trials": {"dev": 0.170, "val": 0.079},
            "note": "零点不是 0；账户层 2-6pp 的“边际”小于零点的窗口间漂移。42 次独立性假设会高估期望最大（相关变体降低增益），对“否定 S4”是保守方向。",
        },
        "applicability": {
            "pbo": "仅同族/同窗口/同日网格、候选≥5、共同交易日≥120；候选全负时低 PBO 只说明“最不差”排序稳定。",
            "bootstrap": "stationary bootstrap（块长 20）；日频重采样对 44-71 笔的稀疏成交偏窄。",
            "dsr": "Var(SR_trials) 混合不同族/窗口，仅作项目级诊断；族内 DSR 单列；N 阶梯 N1/N2/N3 同时报告；跨口径不可直接比。",
            "walk_forward": "E1-EVID 未建立：现有 walk-forward 只有 4 折、训练段不选参；没有任何配置完成 §10.1 的 5 折滚动。",
        },
    }

    data_block = {
        "version": "v2-adjspace-2026-10-28",
        "freeze_date": "2026-10-26",
        "daily_rows": 16698536,
        "symbols": 5549,
        "date_range": "2004-01-02..2026-10-26",
        "columns": 40,
        "dual_source_common_security_days": 10104953,
        "close_diff_gt_0_005_rows": 4,
        "ohlc_any_field_diff_gt_0_005_rows": 5,
        "ohlc_5th_row": "688206.SH 的 2021 年 12 月 28 日 high 56.20 vs 55.00（close 均 42.71，不在主宇宙）",
        "ths_missing_delisted_symbols": 338,
        "baostock_includes_delisted": True,
        "adjust_space": "A 空间：adjusted_close ≡ factor ≡ Π(1+pct_chg/100)；hfq_px = raw_px × A_t；日收益严格等于 pct_chg（仅 T0005/VCB/小盘/统一引擎）",
        "v1_defect": ("v1 hfq=raw×factor 双计收益：94.10% 相邻对偏离 pct_chg >1bp（v1 构建 15,693,248/16,676,415；审计全样本重算 15,693,245/16,692,984=94.01%），"
                      "中位 1.41pp，最大单日偏差 170.1（收益小数，即 17,010pp；000631.SZ 2009-06-05），600519 最大 49.9pp；"
                      "原稿“227pp”无可复现定义、已撤回；v2 全样本 0 行违约（最大 4.31e-15）"),
        "selfcheck": "01_数据/_audit/自检结果.json：28 PASS / 0 FAIL（另 7 项 INFO）",
    }

    holdout = {
        "run": False,
        "window": "2021-01-01..2025-12-31",  # 留出窗口：未运行（护栏，禁止计算）
        "adjudication": "05_审计/留出准入裁决.json（**v2 修订版，2026-10-29 02:12**；0 个候选获准；00:37 初版在 `initial_snapshot`/附录 A）",
        "adjudication_note": "该裁决书是 **v2 修订版（2026-10-29 02:12）**，当前事实为注册表 10 行、T0005 rev2 done discard、T0007 val done discard、holdout 0、N=733、DSR 6.63e−4；**00:37 初版只作为附录 A / `initial_snapshot` 保留**，不是当前文件。",
        "supersedes": {"path": "05_审计/留出准入裁决.json#initial_snapshot",
                       "current_revision": "v2",
                       "current_as_of": "2026-10-29T02:12",
                       "initial_as_of": "2026-10-29T00:37",
                       "as_of": "2026-10-29T02:12",
                       "replaced_reason_codes": ["PROCEDURAL_NO_DONE_VAL_TRIAL"],
                       "current_reason_codes": ["NO_ELIGIBLE_CANDIDATE", "POSTHOC_REGISTRATION", "G1_NOT_TESTED"],
                       "current_facts": {"registry_rows": 10, "holdout_rows": 0,
                                         "t0005": "rev2 dev done discard", "t0007": "val done discard",
                                         "N_sel": 733, "DSR_N733": 0.000662754421907755}},
        "reasons": [
            "程序（当前事实，裁决文件为 v2 修订版）：注册表 10 行——T0005 rev1 planned（历史）→ rev2 dev done/discard、T0007 val done/discard、T0008/T0009 done；holdout 行 0。没有任何“通过 val”的父试验，T0005/T0007 又都是事后补登记，不能倒过来当留出前置；绕过注册表直接跑即 §12.4 的“未登记运行”。",
            "门槛：§12.1 G1 要求 E1-EVID 与 E2-EVID 都 ≥60%；E1-EVID 未建立（无 5 折 walk-forward），E2-EVID 单独成立也无意义；G1 未被直接检验。",
            "统计：5 年留出、σ 20-30% 时年化标准误 9-13pp；真实 6% 的候选 5 年观测落在 0 以下的概率约 25%，留出无法“确认”低个位数候选。",
            "资源：§5.2 全局只允许一个非 rejected/invalid 的 holdout；为 3-6% 的候选消耗唯一 E2-EVID 是负期望。",
            "证据等级：E2-EVID 通过也只能否决、不能证明泛化；把“2021-2025 没亏”写成“样本外验证通过”被明文禁止。",
        ],
        "no_2021_returns": True,
        "scan": ("交付验收 L1：469 个绩效表，0 个含 2021+ 收益行（历史一轮 470，差 1 个文件集）；"
                 "数值审计另扫 682 个结果文件，6 个含 >2020 日期（状态/原始缓存/审计缓存；按 `date` 列口径）、带绩效列 0 个；"
                 "封存清单按 `date/calendar_date/out_date` 口径在 03_结果/ 内为 9 个（A3+B6）。"
                 "L3 的 1 条 PERFORMANCE_2021+_SUSPECT 是旧扫描对 试验总账.csv 的假阳性。"),
        "exposure_paths": [
            "03_结果/市场状态/市场状态_日频许可_扩展2021_2026_未评估.csv（A 级；1,406 行状态/仓位，无收益列）",
            "03_结果/市场状态_缓存/index_feat.parquet（A 级；含 ret1 等收益列，max 2026-10-26）",
            "03_结果/市场状态_缓存/market_feature.parquet（A 级；含 ew_ret1/csd_ret1 等）",
        ],
        "seal": {
            "logical_seal": True,
            "physical_isolation": False,
            "n_files": 34,
            "grades": {"A_forbidden": 3, "B_readonly": 7, "C_source": 24},
            "n_strategy_performance_files": 0,
            "list": "05_审计/2021+未评估文件封存清单.csv",
            "description": "05_审计/2021+未评估文件封存说明.md",
            "note": "逻辑封存=逐文件 SHA-256 + 机器清单 + 禁止用途契约；未删除/移动/改权限/独立保管，A 级文件仍在项目树内。",
        },
        "exposure_note": "**已完成逻辑封存**：34 个 2021+ 文件（A=3/B=7/C=24）逐文件 SHA-256，含策略绩效列 0 个；**未做物理隔离**（未移出/改权限/独立保管），A 级文件仍在项目树内。C 级 24 个是可重建 2021+ 行情的数据源。",
        "gate_coverage": summary["gate_coverage"],
        "gate_coverage_note": ("478 行门槛表覆盖 505 个 N_sel、未评级 228 个（505+228=733）；"
                               "旧 486/228 是 714 口径（486+228=714），不含 T0005 的 19 个 N_sel。"),
    }

    limits = [
        "无分钟数据：日线内“先到高点还是低点”不可知；同日止损/目标同时触及按止损优先（保守）。",
        "ST 历史标记可能不完整：2005-2007 的 ST 与股改口径不可靠，该段只作压力/预热。",
        "退市处置只有“最后价”与“最后价×0.7”两档，真实三板流动性更差；VCB 另有 250 日停牌强平（002075.SZ），方向乐观。",
        "生存者偏差：DuckDB 日线自 2016-10-13 起且只含现存代码；2016 前的退市股必须走 Baostock 适配器。",
        "2021+ 文件已逻辑封存（34 个、A3/B7/C24、逐文件 SHA-256、0 策略绩效、禁止用途）；物理隔离/独立保管未做，A 级文件仍在项目树内（`05_审计/2021+未评估文件封存清单.csv` + `封存说明.md`）。",
        "Baostock 后复权年末重定基：63 个证券日 >0.5pp（涉及 39 只；广义 79 行 / 16,682,010 行）；趋势/结构/退出/相对强弱/趋势回调/长期双动量使用该口径，逐笔复算与 A 空间一致但引用必须注明“Baostock flag1”。",
        "out_date 前视：父规则要求“退市整理期前 30 个交易日不新开仓”，out_date 在 t 日不可知；统一引擎量化影响 54/342,387 信号（0.0158%），不足以解释 2-6pp 的候选差异，但必须在下一版修掉。",
        "VCB 有三处口径差：① delist_window 未实现（方向未证明；基准 delist_haircut=0 偏乐观约 0.6pp）；② delisting_missing_days=250 把 002075.SZ 的 526 日停牌当退市（val 偏乐观 0.14-0.77pp/年）；③ 成本 10bp 固定、10% 参与率、无冲击项（dev 偏乐观约 0.44pp/年）。",
        "TREND 创业板限价档位 bug：2020-08-24 前创业板实际 10%，代码全期 20%；17-18 个唯一（代码,日）涨停开盘仍买入、45 个跌停开盘仍卖出；5 个 S3 正结果（581 笔中 96 笔非主板）因此带执行口径污染，不得当干净的两段正证据。",
        "T0005 未做成本×2/延迟 1 日/去尾的完整 S 闸压力；其 S 闸按“未证明即不通过”处理。",
        "T0005 的 val 自检有 2 笔停牌顺延后强制估值成交（非退市）与 2 处风险超限（跳空），已在 50_结论.md §7 披露；小盘主单元 104 笔、VCB val 1 笔为同机制。",
        "结构族与市场状态族的部分表只有 2008-2020 合并窗口，无法按预注册两窗口准则评级，只能作负证据。",
        "E1-EVID/E2-EVID 缺失：没有项目级 walk-forward 主证据；2021-2025 留出未运行；结论的证据等级是“描述性开发/验证”，不是样本外验证，也不是完整预注册分档。入场规则的 E1/E2 记为 E1-ENTRY/E2-ENTRY。",
        "预注册 §15 的四列（window_start/window_end/adjust/cost_model_id）未在所有表落实；验收记为 WARN（T2）；跨族口径改由 cross_family_matrix 披露。",
        "文件清单不是逐字节全量：>16 MB 的文件不记 SHA-256，只记字节数；收录规则是“05_审计/ 只收 *.md，审计 JSON/CSV 不收录；04_证据/*.json 全收”（README §9）。",
        "费用模型对停牌/复牌/涨跌停的顺延上限做了简化（最长 10 日），极端行情下可能高估可成交性。",
        "N 有三个口径：第一阶段 365（6 族，DSR 0.0003，5.70×）；00:51 快照 714（1,631 行，DSR 6.89e-4，11.16×，不含 T0005）；当前 733（1,699 行，DSR 6.63e-4，11.45×/18.33×/11.92×，含 T0005 的 19 个 N_sel/38 cell/30 diag）。引用必须绑定 N 与来源；746 已作废。",
        "B_conditional 有两个口径：配置级 5 个（4 个 TREND@S3 + VCB-P20/T40；统一门槛 5 行、总账 10 行）；旧行级 12 行/8 配置（11 行 S3 + 1 行 S1，仅第一阶段 6 族，候选分级.csv）。族集合与粒度不同，不能混用，也不能写成“全部 S3”。",
        "小盘低波 COMBO 的 val +0.33% 以首日收盘净值 982,382 为基准，漏掉 2016-01-04 熔断日 -1.76%；按初始 100 万计 val 为 -0.03%，不得写成“两段为正”或“val 正收益说明有边际”。",
        "T0005 的旧 planned/not_run 行（F01629、F00002）是历史快照；当前事实是 68 行 done、19 个 N_sel、38 cell、30 diag，主候选已淘汰。检索总账必须按 status 过滤。",
    ]

    deliverables = {
        "一页纸": "00_一页纸.md",
        "完整报告": "01_完整报告.md",
        "规则手册（研究结论版）": "02_规则手册_研究结论版.md",
        "数据与复现说明": "03_数据与复现说明.md",
        "机器可读结论": "04_最终结论.json",
        "官方注册表": "03_结果/实验注册表.csv",
        "最终试验总账": "03_结果/最终试验总账.csv",
        "最终统计目录": "03_结果/最终统计/",
        "最终交付脚本": "02_代码/最终交付.py",
        "主候选原始结果": "03_结果/主候选_T0005/",
        "留出准入裁决": "05_审计/留出准入裁决.json（v2 修订版；00:37 初版在 initial_snapshot）",
        "交付验收": "04_证据/交付验收.json",
        "最终口径对账": "04_证据/最终口径对账.json",
        "审计整改验收": "04_证据/审计整改验收.json",
        "复核剩余项关闭": "04_证据/最终审计剩余项关闭.json",
        "2021封存清单": "05_审计/2021+未评估文件封存清单.csv",
        "2021封存说明": "05_审计/2021+未评估文件封存说明.md",
        "审计问题处置表": "05_审计/审计问题处置表.csv",
        "四份复核": ["05_审计/最终数值审计_复核.json", "05_审计/最终交付审计_复核.json",
                     "05_审计/最终反方审计_复核.json", "05_审计/最终方法代码审计_复核.json"],
    }

    out = {
        "schema": "trend-structure-final-conclusion/v2",
        "meta": {
            "project": "trading_趋势结构",
            "generated_at_local": dt.datetime.now().isoformat(timespec="seconds"),
            "generated_by": "最终整合负责人",
            "model": "deepseek-flash",
            "seed": SEED,
            "tool_sha256": sha256_file(__file__),
            "data_version": data_block["version"],
            "freeze_date": data_block["freeze_date"],
            "role": "最终整合（只读他人产物；只追加注册表/总账；不重跑策略）",
            "revision": "audit-remediation-2026-10-29-rev3",
            "review_status": FINAL_STATUS,
        },
        "verdict": {
            "code": "NO_CANDIDATE",
            "tier": "无可利用边际（§12.2 阈值的描述性套用，不是预注册分档）",
            "annualized_60pct_supported": False,
            "annualized_60pct_statement": "在本项目测试空间内，年化 60%（全窗口 CAGR）不被支持；G1 未被直接检验；不等于任何策略都不可能 60%。",
            "g1_directly_tested": False,
            "n_candidates": 0,
            "holdout_run": False,
            "scope": "A 股日线、仅做多、无杠杆、10 仓框架、2008-2020、18 族/733 配置；T0005/VCB/小盘为主板宇宙，TREND/STRUCT 含创业板/科创板暴露。",
            "statement": v["verdict"]["statement"],
            "main_candidate": {"trial_id": "T0005", "decision": "discard",
                               "dev_cagr": t5["dev"]["cagr"], "val_cagr": t5["val"]["cagr"],
                               "dev_mdd": t5["dev"]["mdd"], "val_mdd": t5["val"]["mdd"],
                               "reason": "§7.9.5 两段净收益>0 不通过；15 个消融 + 18 个邻域无一 val 转正；PBO 0.311/0.756。"},
        },
        "windows": {
            "dev": "2008-01-01..2015-12-31",
            "val": "2016-01-01..2020-12-31",
            "holdout": "2021-01-01..2025-12-31（未运行）",
            "preheat": "2005-01-01..2007-12-31（仅预热/压力，不作证据）",
            "evidence_level": "描述性开发/验证（E1-EVID 未建立、E2-EVID 未运行、E3 不存在）；不是样本外验证，不是完整预注册分档。入场规则的 E1/E2 记为 E1-ENTRY/E2-ENTRY。",
        },
        "data": data_block,
        "counts": summary["counts"],
        "grade_counts": summary["grade_counts"],
        "candidate_grading": summary["candidate_grading"],
        "main_candidate": t5,
        "families": [fam_entry(k) for k in CURATED_FAMILIES],
        "cross_family_matrix": [{"family": k, **FAMILY_SCOPE[k]} for k in CURATED_FAMILIES],
        "questions": questions,
        "cross_cutting": cross,
        "statistics": stats_block,
        "holdout": holdout,
        "limits": limits,
        "deliverables": deliverables,
        "reproduce": [
            "python3 02_代码/最终交付.py verify",
            "python3 02_代码/最终交付.py selfcheck",
            "python3 02_代码/实验登记.py verify --strict",
            "python3 02_代码/最终试验总账.py --stage all   # 注意：不含 T0005，重跑会覆盖最终统计聚合",
            "python3 02_代码/主候选_T0005.py main && python3 02_代码/主候选_T0005.py report",
            "python3 02_代码/交付验收.py",
        ],
        "restrictions": {
            "no_2021_plus_returns": True,
            "no_target_backsolving": True,
            "no_current_buy_sell_advice": True,
            "no_live_trading_claim": True,
            "negative_results_retained": True,
            "append_only_registry_and_ledger": True,
        },
        "audit_remediation": {
            "acceptance_json": "04_证据/审计整改验收.json",
            "residual_closure_json": "04_证据/最终审计剩余项关闭.json",
            "audits": ["05_审计/最终数值审计.json", "05_审计/最终交付审计.json",
                       "05_审计/最终反方审计.json", "05_审计/最终方法代码审计.json"],
            "rereviews": ["05_审计/最终数值审计_复核.json", "05_审计/最终交付审计_复核.json",
                          "05_审计/最终反方审计_复核.json", "05_审计/最终方法代码审计_复核.json"],
            "n_first_round": len(REMEDIATION),
            "n_rereview": len(REREVIEW_ITEMS),
            "status": FINAL_STATUS,
            "note": f"首轮 {len(REMEDIATION)} 项 + 四份复核 {len(REREVIEW_ITEMS)} 项；逐项处置见 acceptance_json，复核剩余项映射见 residual_closure_json；四份终验均 PASS，但不宣称 0 open（declared residuals 仍在）。",
        },
    }
    jdump(CONCLUSION, out)

    # 同步 最终判定.json 的族级字段（该文件由 stage_final_stats 生成；本批只改 families，不动统计数字）
    if v.get("families") != CURATED_FAMILIES:
        v["families"] = CURATED_FAMILIES
        v["families_note"] = "族级字段已按 2026-10-29 审计整改更新：主候选与族内真实最高分开；BH10 单列基准；跨族口径见 04_最终结论.json cross_family_matrix。"
        jdump(os.path.join(FINAL_DIR, "最终判定.json"), v)

    print(f"conclusion: {rel(CONCLUSION)} 已生成")
    return out


# ---------------------------------------------------------------------------
# verify：注册表 + 总账 + 对账 + 计数 + 2021 护栏
# ---------------------------------------------------------------------------

def _rec(name: str, ok: bool, detail: str, evidence: str = "") -> dict:
    return {"check": name, "pass": bool(ok), "detail": detail, "evidence": evidence}


def stage_verify(verbose: bool = True) -> dict:
    checks: list[dict] = []
    R = load_module("02_代码/实验登记.py", "regtool_v")
    src = source_numbers()
    dev_m = src["stats"]["dev"]["metrics"]
    val_m = src["stats"]["val"]["metrics"]

    # 1 注册表链
    reg_rows = R.read_rows(R.ROOT / "03_结果" / "实验注册表.csv")
    errs, warns = R.verify_rows(reg_rows, strict=False, allow_code_drift=False)
    checks.append(_rec("registry_chain", len(errs) == 0,
                       f"rows={len(reg_rows)} errors={len(errs)} warnings={len(warns)}",
                       "03_结果/实验注册表.csv"))
    # 2 T0005 planned 保留
    first = [r for r in reg_rows if r["trial_id"] == "T0005" and r["revision"] == "1"]
    checks.append(_rec("registry_planned_preserved",
                       bool(first) and first[0]["status"] == "planned",
                       f"revision=1 status={first[0]['status'] if first else 'MISSING'}",
                       "03_结果/实验注册表.csv"))
    # 3 T0005 done（dev）
    latest = R.latest_rows(reg_rows)
    t5 = latest.get("T0005")
    ok = bool(t5) and t5["status"] == "done" and t5["stage"] == "dev"
    dev_cagr = dev_mdd = dev_tr = None
    if ok:
        mt = json.loads(t5["metrics_json"])
        dev_cagr, dev_mdd, dev_tr = mt["cagr"], mt["mdd"], mt["n_trades"]
        ok = (abs(float(dev_cagr) - float(dev_m["cagr"])) < 1e-9
              and abs(float(dev_mdd) - float(dev_m["mdd"])) < 1e-9
              and int(dev_tr) == int(dev_m["n_trades"]))
    checks.append(_rec("registry_t0005_dev_done", ok,
                       f"cagr={dev_cagr} mdd={dev_mdd} n_trades={dev_tr} (source {dev_m['cagr']:.6f}/{dev_m['mdd']:.6f}/{dev_m['n_trades']})",
                       "03_结果/实验注册表.csv"))
    # 4 val done
    val_rows = [r for r in reg_rows if r["stage"] == "val" and r["status"] == "done"
                and r["parent_trial"] == "T0005"]
    ok = False
    detail = "no val done row"
    if val_rows:
        mt = json.loads(val_rows[-1]["metrics_json"])
        ok = (abs(float(mt["cagr"]) - float(val_m["cagr"])) < 1e-9
              and abs(float(mt["mdd"]) - float(val_m["mdd"])) < 1e-9
              and int(mt["n_trades"]) == int(val_m["n_trades"]))
        detail = f"cagr={mt['cagr']} mdd={mt['mdd']} n_trades={mt['n_trades']} ({val_rows[-1]['trial_id']})"
    checks.append(_rec("registry_t0005_val_done", ok, detail, "03_结果/实验注册表.csv"))
    # 5 无留出
    holds = [r for r in latest.values() if r["stage"] == "holdout"]
    checks.append(_rec("registry_no_holdout", len(holds) == 0, f"holdout rows={len(holds)}",
                       "03_结果/实验注册表.csv"))

    # 6 总账链
    led = read_rows(LEDGER)
    lerr = verify_ledger_chain(led)
    checks.append(_rec("ledger_chain", len(lerr) == 0, f"rows={len(led)} errors={len(lerr)}",
                       "03_结果/最终试验总账.csv"))
    # 7 总账 T0005 实测行
    t5rows = [r for r in led if r["family"] == "MAIN" and r["strategy_id"] == "T0005"
              and r["status"] == "done"]
    d = next((r for r in t5rows if r["stage"] == "dev"), None)
    v_ = next((r for r in t5rows if r["stage"] == "val"), None)
    ok = d is not None and v_ is not None
    if ok:
        ok = (abs(float(d["cagr"]) - float(dev_m["cagr"])) < 1e-8
              and abs(float(d["mdd"]) - float(dev_m["mdd"])) < 1e-8
              and int(float(d["n_trades"])) == int(dev_m["n_trades"])
              and abs(float(v_["cagr"]) - float(val_m["cagr"])) < 1e-8
              and abs(float(v_["mdd"]) - float(val_m["mdd"])) < 1e-8
              and int(float(v_["n_trades"])) == int(val_m["n_trades"]))
    checks.append(_rec("ledger_t0005_matches_source", ok,
                       "dev/val cagr/mdd/n_trades 与 07_统计_主候选.json 一致" if ok else "不一致或缺失",
                       "03_结果/最终试验总账.csv"))
    # 8 历史 planned 行未改
    planned = [r for r in led if r["strategy_id"] == "T0005" and r["status"] == "planned"]
    checks.append(_rec("ledger_planned_row_intact",
                       len(planned) == 1 and planned[0]["grade"] == "F_unevaluable"
                       and planned[0]["fail_gate"] == "F_never_run",
                       f"planned rows={len(planned)} grade={planned[0]['grade'] if planned else 'NA'}",
                       "03_结果/最终试验总账.csv"))

    # 9 计数一致性
    total = summarize(led)
    cnt = read_rows(os.path.join(FINAL_DIR, "最终计数.csv"))
    tot_row = next(r for r in cnt if r["scope"] == "total")
    ok = all(abs(float(tot_row[k]) - total[v]) < 1e-9 for k, v in (
        ("rows", "rows"), ("N_sel", "in_Nsel"), ("N_cell", "in_Ncell"),
        ("N_diag", "in_Ndiag"), ("N_platform", "in_Nplatform"), ("N_audit", "in_Naudit")))
    checks.append(_rec("counts_file_matches_ledger", ok,
                       f"N_sel={total['in_Nsel']:.0f} N_cell={total['in_Ncell']:.0f} N_diag={total['in_Ndiag']:.0f}",
                       "03_结果/最终统计/最终计数.csv"))
    # 10 最终判定一致
    fv = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    ok = (int(fv["counts"]["total"]["in_Nsel"]) == int(total["in_Nsel"])
          and int(fv["counts"]["N_dsr_ladder"]["N1_sel"]) == int(total["in_Nsel"]))
    checks.append(_rec("verdict_json_matches_ledger", ok,
                       f"json N_sel={fv['counts']['N_dsr_ladder']['N1_sel']}", "03_结果/最终统计/最终判定.json"))
    # 11 04_最终结论 一致
    fc = jload(CONCLUSION) if os.path.exists(CONCLUSION) else {}
    ok = bool(fc) and int(fc["counts"]["total"]["in_Nsel"]) == int(total["in_Nsel"])
    checks.append(_rec("conclusion_json_matches_ledger", ok,
                       f"04_最终结论 N_sel={fc.get('counts', {}).get('total', {}).get('in_Nsel')}",
                       "04_最终结论.json"))
    # 12 A_candidate
    gates = read_rows(os.path.join(FINAL_DIR, "统一门槛.csv"))
    n_a = sum(1 for g in gates if g["grade"] == "A_candidate")
    checks.append(_rec("no_a_candidate", n_a == 0, f"A_candidate={n_a}", "03_结果/最终统计/统一门槛.csv"))
    # 13 T0005 门槛判定
    g5 = [g for g in gates if g["family"] == "MAIN" and g["config_id"] == "MAIN:T0005"]
    ok = bool(g5) and g5[0]["grade"] == "C_negative_valid" and "G_fail_negative" in g5[0]["fail_gate"]
    checks.append(_rec("t0005_graded_negative", ok,
                       f"grade={g5[0]['grade'] if g5 else 'NA'} fail={g5[0]['fail_gate'] if g5 else 'NA'}",
                       "03_结果/最终统计/统一门槛.csv"))
    # 14 2021 收益护栏
    bad = [r["ledger_id"] for r in led
           if r.get("cagr") and r.get("window_end") and pd.Timestamp(r["window_end"]) > STAGE_END_CAP]
    checks.append(_rec("no_post_2020_performance", len(bad) == 0,
                       f"带绩效且 window_end>2020-12-31 的行 = {len(bad)}", "03_结果/最终试验总账.csv"))
    # 15 T0005 净值上界
    mx = {}
    for w in ("dev", "val"):
        s = nav_series(src, w)
        mx[w] = str(s.index.max().date())
    ok = all(pd.Timestamp(v) <= STAGE_END_CAP for v in mx.values())
    checks.append(_rec("t0005_nav_max_date", ok, f"max date dev={mx['dev']} val={mx['val']}",
                       "03_结果/主候选_T0005/03_日净值_T0005_*.csv"))
    # 16 T0005 自检 rows_after_2020
    ok = all(int(src["selfcheck"][w]["rows_after_2020"]) == 0 for w in ("dev", "val"))
    checks.append(_rec("t0005_selfcheck_no_2020plus", ok,
                       "rows_after_2020 dev=0 val=0", "03_结果/主候选_T0005/08_自检_T0005_*.json"))
    # 17 独立账户对账
    rel_diff = max(src["recon"][w]["independent_ledger"]["max_rel_diff"] for w in ("dev", "val"))
    checks.append(_rec("t0005_independent_ledger", rel_diff < 1e-6,
                       f"max_rel_diff={rel_diff:.2e}", "03_结果/主候选_T0005/40_独立账户对账.json"))
    # 18 因果自检
    caus = src["recon"]["causality"]
    checks.append(_rec("t0005_causality", bool(caus["pass"]) and int(caus["n_mismatch"]) == 0,
                       f"n_checked={caus['n_checked']} mismatch={caus['n_mismatch']}",
                       "03_结果/主候选_T0005/40_独立账户对账.json"))
    # 19 门槛覆盖与当前 N 自洽（478 行 → 505+228=733；旧 486 为 714 口径）
    fv = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    gates = read_rows(os.path.join(FINAL_DIR, "统一门槛.csv"))
    gk = {(g["family"], g["config_id"]) for g in gates}
    cover = int(sum(float(r["in_Nsel"] or 0) for r in led
                    if r["category"] in {"prereg", "posthoc", "sensitivity"}
                    and (r["family"], r["config_id"]) in gk))
    total_n = int(summarize(led)["in_Nsel"])
    unrated = total_n - cover
    ok = (len(gates) == 478 and cover == 505 and unrated == 228 and cover + unrated == total_n == 733
          and int(fv["holdout_gate"]["n_gate_rows"]) == 478
          and int(fv["holdout_gate"]["N_sel_covered_by_gate_rows"]) == cover
          and int(fv["holdout_gate"]["N_sel_unrated"]) == unrated)
    checks.append(_rec("holdout_gate_coverage", ok,
                       f"rows={len(gates)} covered={cover} unrated={unrated} total={total_n} "
                       f"(json {fv['holdout_gate']['N_sel_covered_by_gate_rows']}/"
                       f"{fv['holdout_gate']['N_sel_unrated']})",
                       "03_结果/最终统计/最终判定.json；统一门槛.csv"))

    # 20-30 审计整改断言（2026-10-29 rev2）：从结果表重算，不依赖文档
    facts = audit_facts()
    for c in _acceptance_checks(facts):
        if c["check"] in {x["check"] for x in checks}:
            continue
        checks.append(c)
    fc2 = jload(CONCLUSION) if os.path.exists(CONCLUSION) else {}
    st2 = fc2.get("statistics", {})
    checks.append(_rec("conclusion_family_best",
                       bool(st2.get("family_best")) and abs(st2["family_best"]["best_dev"]["cagr"] - 0.203991) < 1e-5,
                       f"04 最终结论 family_best.best_dev={st2.get('family_best', {}).get('best_dev', {}).get('cagr')}",
                       "04_最终结论.json"))
    checks.append(_rec("conclusion_budget_two_calibers",
                       st2.get("budget", {}).get("select") == 40 and st2.get("budget", {}).get("with_ablation") == 64
                       and abs(float(st2.get("budget", {}).get("overrun_vs_select", 0)) - 18.33) < 0.01,
                       f"budget={st2.get('budget')}", "04_最终结论.json"))
    _pm_rows = [r for r in st2.get("pbo_maxstat", []) if r.get("kind") == "MAXSTAT"]
    _pm_ok = bool(_pm_rows) and all(r.get("p_maxstat") is not None for r in _pm_rows)
    checks.append(_rec("conclusion_maxstat_p",
                       bool(st2.get("maxstat", {}).get("min")) and abs(st2["maxstat"]["min"]["p_maxstat"] - 0.071964) < 1e-6
                       and st2.get("maxstat", {}).get("n_p") >= 16 and _pm_ok,
                       f"maxstat={st2.get('maxstat', {}).get('min')} n_p={st2.get('maxstat', {}).get('n_p')} "
                       f"MAXSTAT rows with p={sum(1 for r in _pm_rows if r.get('p_maxstat') is not None)}/{len(_pm_rows)}",
                       "04_最终结论.json"))
    checks.append(_rec("conclusion_cross_matrix",
                       len(fc2.get("cross_family_matrix", [])) == 18
                       and all(all(str(x.get(k) or "").strip() for k in ("universe", "adjust", "engine", "cost", "limits"))
                               for x in fc2.get("cross_family_matrix", [])),
                       f"cross_family_matrix rows={len(fc2.get('cross_family_matrix', []))}", "04_最终结论.json"))
    fam_map = {x.get("family"): x for x in fc2.get("families", [])}
    checks.append(_rec("conclusion_bh10_benchmark",
                       (fam_map.get("UNIFIEDENGINE", {}).get("benchmark_bh10", {}) or {}).get("type") == "benchmark"
                       and fam_map.get("UNIFIEDENGINE", {}).get("dev_range", [0])[0] < -0.3,
                       f"UNIFIEDENGINE dev_range={fam_map.get('UNIFIEDENGINE', {}).get('dev_range')}",
                       "04_最终结论.json"))
    checks.append(_rec("conclusion_holdout_supersedes",
                       "supersedes" in fc2.get("holdout", {})
                       and fc2["holdout"]["supersedes"].get("current_reason_codes"),
                       f"supersedes={bool(fc2.get('holdout', {}).get('supersedes'))}", "04_最终结论.json"))
    checks.append(_rec("conclusion_top5_denominators",
                       "top5_denominators" in st2 and abs(st2["top5_denominators"]["dev"]["top5_over_net"] - 0.4587) < 5e-4,
                       f"top5={st2.get('top5_denominators', {}).get('dev')}", "04_最终结论.json"))
    checks.append(_rec("conclusion_vcb_count",
                       st2.get("vcb_count", {}).get("declared_rows") == 34 and st2.get("vcb_count", {}).get("unique_configs") == 32,
                       f"vcb_count={st2.get('vcb_count')}", "04_最终结论.json"))
    rem_path = os.path.join(ROOT, "04_证据", "审计整改验收.json")
    rem = jload(rem_path) if os.path.exists(rem_path) else {}
    checks.append(_rec("remediation_acceptance",
                       bool(rem) and rem.get("summary", {}).get("status") == FINAL_STATUS
                       and rem.get("summary", {}).get("verdict") == FINAL_VERDICT
                       and rem.get("summary", {}).get("n_open_internal") == 0
                       and rem.get("summary", {}).get("n_needs_resync") == 0
                       and rem.get("summary", {}).get("n_check_fail") == 0,
                       f"items={rem.get('summary', {}).get('n_items')} status={rem.get('summary', {}).get('status')} "
                       f"open_internal={rem.get('summary', {}).get('n_open_internal')} "
                       f"needs_resync={rem.get('summary', {}).get('n_needs_resync')} "
                       f"checks={rem.get('summary', {}).get('n_check_pass')}/{rem.get('summary', {}).get('n_checks')}",
                       "04_证据/审计整改验收.json"))
    checks.append(_rec("remediation_covers_all_ids",
                       bool(rem) and len({x["source"] + ":" + x["id"] for x in rem.get("items", [])}) >= 100,
                       f"items={len(rem.get('items', []))}", "04_证据/审计整改验收.json"))
    clo_path = os.path.join(ROOT, "04_证据", "最终审计剩余项关闭.json")
    clo = jload(clo_path) if os.path.exists(clo_path) else {}
    checks.append(_rec("audit_closure_complete",
                       bool(clo) and clo.get("summary", {}).get("n_unmapped") == 0
                       and clo.get("summary", {}).get("n_needs_resync") == 0
                       and clo.get("summary", {}).get("n_check_fail") == 0
                       and clo.get("status") == FINAL_VERDICT,
                       f"items={clo.get('summary', {}).get('n_dedup_items')} unmapped={clo.get('summary', {}).get('n_unmapped')} "
                       f"needs_resync={clo.get('summary', {}).get('n_needs_resync')} fail={clo.get('summary', {}).get('n_check_fail')}",
                       "04_证据/最终审计剩余项关闭.json"))

    npass = sum(1 for c in checks if c["pass"])
    out = {"generated_at_local": dt.datetime.now().isoformat(timespec="seconds"),
           "tag": TAG, "n_checks": len(checks), "n_pass": npass,
           "n_fail": len(checks) - npass, "checks": checks}
    if verbose:
        for c in checks:
            print(("PASS " if c["pass"] else "FAIL ") + c["check"] + " | " + c["detail"])
        print(f"verify: {npass}/{len(checks)} PASS")
    return out


# ---------------------------------------------------------------------------
# selfcheck：交付物自检
# ---------------------------------------------------------------------------

FORBIDDEN = [
    "年化60%可达", "年化 60% 可达", "已实现年化60", "已实现年化 60%",
    "样本外验证通过", "成功实现60%", "稳定盈利系统", "可直接实盘",
    "建议买入", "建议卖出", "推荐买入", "推荐卖出", "当前买入", "当前卖出",
    "包赚", "稳赚", "保证收益",
]
NEGATIONS = ("不得", "不能", "不应", "禁止", "勿", "不要", "不允许", "不是", "并非", "别", "拒绝")


def _claim_hits(text: str, window: int = 18) -> list[str]:
    """只把“未被否定的表述”算作违规；引用禁令（不得写成/不能说/禁止）不算。"""
    hits = []
    for w in FORBIDDEN:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            ctx = text[max(0, i - window):i] + text[i + len(w):i + len(w) + window]
            if not any(n in ctx for n in NEGATIONS):
                hits.append(w + " @ " + text[max(0, i - 20):i + len(w) + 12].replace("\n", " "))
            start = i + len(w)
    return hits


def _read_text(rel_path: str) -> str:
    with open(p(rel_path), encoding="utf-8") as f:
        return f.read()


def stage_selfcheck() -> dict:
    from pathlib import Path

    checks: list[dict] = []

    need = ["00_一页纸.md", "01_完整报告.md", "02_规则手册_研究结论版.md",
            "03_数据与复现说明.md", "README.md", "04_最终结论.json",
            "04_证据/审计整改验收.json", "03_结果/实验注册表.csv",
            "03_结果/最终试验总账.csv", "03_结果/最终统计/T0005_纳入后_汇总.json",
            "02_代码/最终交付.py"]
    missing = [x for x in need if not os.path.exists(p(x))]
    checks.append(_rec("deliverables_exist", not missing, f"missing={missing}", "根目录"))

    texts = {}
    for x in need:
        if x.endswith(".md"):
            try:
                texts[x] = _read_text(x)
            except Exception as exc:
                texts[x] = ""
                checks.append(_rec(f"read_{x}", False, str(exc), x))
    one = texts.get("00_一页纸.md", "")
    rep = texts.get("01_完整报告.md", "")
    rb = texts.get("02_规则手册_研究结论版.md", "")
    dat = texts.get("03_数据与复现说明.md", "")

    try:
        fc = jload(CONCLUSION)
        jok = isinstance(fc, dict)
    except Exception:
        fc, jok = {}, False
    checks.append(_rec("conclusion_json_parses", jok, "04_最终结论.json", "04_最终结论.json"))
    req = ["meta", "verdict", "windows", "data", "counts", "main_candidate", "families",
           "questions", "cross_cutting", "statistics", "holdout", "limits", "deliverables",
           "reproduce", "restrictions"]
    miss = [k for k in req if k not in fc]
    checks.append(_rec("conclusion_schema", not miss, f"missing={miss}", "04_最终结论.json"))
    checks.append(_rec("conclusion_no_candidate",
                       fc.get("verdict", {}).get("n_candidates") == 0,
                       f"n_candidates={fc.get('verdict', {}).get('n_candidates')}", "04_最终结论.json"))
    checks.append(_rec("conclusion_60_rejected",
                       fc.get("verdict", {}).get("annualized_60pct_supported") is False,
                       "annualized_60pct_supported=False", "04_最终结论.json"))
    checks.append(_rec("conclusion_holdout_not_run",
                       fc.get("holdout", {}).get("run") is False,
                       "holdout.run=False", "04_最终结论.json"))
    checks.append(_rec("conclusion_main_candidate_discard",
                       fc.get("verdict", {}).get("main_candidate", {}).get("decision") == "discard",
                       "T0005 decision=discard", "04_最终结论.json"))

    # 一页纸
    ok = all(k in one for k in ("60%", "DSR", "PBO", "N_sel", "T0005", "留出", "证据等级"))
    checks.append(_rec("onepager_required_terms", ok,
                       "含 60%/DSR/PBO/N_sel/T0005/留出/证据等级", "00_一页纸.md"))
    # 完整报告问题
    qs = [f"Q{i}" for i in range(1, 7)]
    missq = [q for q in qs if q not in rep]
    checks.append(_rec("report_covers_q1_q6", not missq, f"missing={missq}", "01_完整报告.md"))
    # 全部族
    fams = ["TREND", "STRUCT", "EXIT", "MKT", "ECOLOGY", "STAT", "VCB", "SMALLLOWVOL",
            "TRENDCALLBACK", "RELSTR", "BREAKPULL", "STRUCTREV", "LONGMOM", "EXITUNI",
            "RISKBUDGET", "UNIFIEDENGINE", "EXECPRESSURE", "MAIN"]
    missf = [f for f in fams if f not in rep]
    checks.append(_rec("report_covers_all_families", not missf, f"missing={missf}", "01_完整报告.md"))
    # 横切
    cross_terms = ["动态参数", "5/10/20", "资金", "成本", "延迟", "板块", "流动性", "市场状态",
                   "符合预期", "不符合预期", "谨慎", "必须退出", "让利润奔跑"]
    missc = [t for t in cross_terms if t not in rep]
    checks.append(_rec("report_covers_cross_cutting", not missc, f"missing={missc}", "01_完整报告.md"))
    # 负结果不被隐藏
    ok = ("V1" in rep and "0 笔" in rep) and ("tp8" in rep) and ("撤回" in rep)
    checks.append(_rec("report_keeps_negative_and_retractions", ok,
                       "含 V1 0 笔、tp8 撤回", "01_完整报告.md"))
    # 规则手册标注
    ok = ("研究结论版" in rb) and ("未通过验证" in rb)
    checks.append(_rec("rulebook_label", ok, "含“研究结论版/未通过验证”", "02_规则手册_研究结论版.md"))
    ok = ("不构成实盘" in rb or "不是可实盘" in rb) and ("可证伪" in rb)
    checks.append(_rec("rulebook_not_live_system", ok,
                       "含“不构成实盘/不是可实盘”与“可证伪”", "02_规则手册_研究结论版.md"))
    ok = ("未验证" in rb) and ("风险原则" in rb)
    checks.append(_rec("rulebook_separates_verified_unverified", ok,
                       "可证伪风险原则与未验证退出规则分节", "02_规则手册_研究结论版.md"))
    # 数据事实
    facts = ["16,698,536", "5,549", "2004-01-02", "2026-10-26", "10,104,953", "338", "v2-adjspace-2026-10-28"]
    missd = [f for f in facts if f not in dat]
    checks.append(_rec("data_doc_facts", not missd, f"missing={missd}", "03_数据与复现说明.md"))
    # 禁用表述（引用禁令不算违规）
    all_text = one + rep + rb + dat
    hits = _claim_hits(all_text)
    checks.append(_rec("no_forbidden_claims", not hits, f"hits={hits}", "四份 Markdown"))
    # 2021 声明
    ok = ("2021" in rep) and ("未运行" in rep)
    checks.append(_rec("report_states_holdout_not_run", ok, "报告声明 2021+ 未运行", "01_完整报告.md"))
    # 留出理由数量
    ok = rep.count("不跑留出") + rep.count("留出不运行") + rep.count("没有留出") >= 1
    checks.append(_rec("report_holdout_reasons", ok, "含无留出理由", "01_完整报告.md"))

    # 注册表 / 总账链
    v = stage_verify(verbose=False)
    checks.append(_rec("verify_all_pass", v["n_fail"] == 0,
                       f"{v['n_pass']}/{v['n_checks']} PASS", "verify()"))
    led = read_rows(LEDGER)
    total = summarize(led)
    checks.append(_rec("counts_match_current_ledger",
                       total["in_Nsel"] == 733 and total["in_Ncell"] == 1251
                       and total["in_Ndiag"] == 454 and total["rows"] == 1699,
                       f"N_sel={total['in_Nsel']:.0f}, N_cell={total['in_Ncell']:.0f}, "
                       f"N_diag={total['in_Ndiag']:.0f}, rows={total['rows']} (expect 733/1251/454/1699)",
                       "03_结果/最终试验总账.csv"))
    checks.append(_rec("t0005_in_ledger",
                       any(r["family"] == "MAIN" and r["strategy_id"] == "T0005" and r["status"] == "done" for r in led),
                       "T0005 done 行存在", "03_结果/最终试验总账.csv"))
    checks.append(_rec("budget_ratio_documented",
                       "11.45" in rep or "11.45×" in rep or "11.45" in one,
                       "N_sel/64 的倍数在报告披露", "01_完整报告.md"))
    checks.append(_rec("t0005_numbers_in_docs",
                       ("4.16" in rep) and ("2.53" in rep),
                       "报告含 dev 4.16% 与 val 2.53%", "01_完整报告.md"))
    cg = fc.get("candidate_grading", {})
    fv2 = jload(os.path.join(FINAL_DIR, "最终判定.json"))
    ok_cal = (
        cg.get("config_level", {}).get("n_configs") == 5
        and cg.get("config_level", {}).get("n_rows") == 10
        and cg.get("row_level_6_families", {}).get("n_rows") == 12
        and cg.get("row_level_6_families", {}).get("n_configs") == 8
        and cg.get("row_level_6_families", {}).get("trend_s1_rows") == 1
        and int(fv2["holdout_gate"]["N_sel_covered_by_gate_rows"]) == 505
        and int(fv2["holdout_gate"]["N_sel_unrated"]) == 228
        and ("-0.03" in one or "−0.03" in one) and ("12 行" in one)
        and ("505" in rep) and ("-0.03" in rep or "−0.03" in rep)
    )
    checks.append(_rec("final_calibers_consistent", ok_cal,
                       "配置级 5/10 行 + 行级 12/8（1 S1）+ 覆盖 505/228 + 小盘 val -0.03% 已写入",
                       "04_最终结论.json；最终判定.json；00_一页纸.md；01_完整报告.md"))

    # 结构完整性（机器可读结论）
    fam_list = fc.get("families", [])
    checks.append(_rec("families_count_18", len(fam_list) == 18,
                       f"families={len(fam_list)}", "04_最终结论.json"))
    checks.append(_rec("questions_count_6", len(fc.get("questions", [])) == 6,
                       f"questions={len(fc.get('questions', []))}", "04_最终结论.json"))
    cross_keys = ["dynamic_parameters", "slots_5_10_20", "capital", "cost", "delay", "board",
                  "liquidity", "market_state", "expectation_met", "expectation_missed",
                  "caution_state", "must_exit", "let_profits_run"]
    missk = [k for k in cross_keys if k not in fc.get("cross_cutting", {})]
    checks.append(_rec("cross_cutting_keys_13", not missk, f"missing={missk}", "04_最终结论.json"))
    checks.append(_rec("limits_count_ge_10", len(fc.get("limits", [])) >= 10,
                       f"limits={len(fc.get('limits', []))}", "04_最终结论.json"))
    restr = fc.get("restrictions", {})
    checks.append(_rec("restrictions_all_true",
                       bool(restr) and all(v is True for v in restr.values()),
                       f"restrictions={restr}", "04_最终结论.json"))
    checks.append(_rec("statement_mentions_t0005",
                       "T0005" in fc.get("verdict", {}).get("statement", ""),
                       "verdict.statement 提到 T0005", "04_最终结论.json"))

    # 总账/注册表结构
    reg = read_rows(REGISTRY)
    checks.append(_rec("registry_rows_10", len(reg) == 10,
                       f"registry rows={len(reg)}", "03_结果/实验注册表.csv"))
    checks.append(_rec("ledger_ablation_rows_30",
                       sum(1 for r in led if r["family"] == "MAIN" and r["kind"] == "ablation") == 30,
                       "30 行消融（15 配置×2 窗口）", "03_结果/最终试验总账.csv"))
    checks.append(_rec("ledger_sensitivity_rows_36",
                       sum(1 for r in led if r["family"] == "MAIN" and r["kind"] == "sensitivity") == 36,
                       "36 行邻域（18 配置×2 窗口）", "03_结果/最终试验总账.csv"))

    # 文档引用数据层事实与 2021 扫描
    checks.append(_rec("report_cites_2021_scan",
                       ("绩效表" in rep) and ("0 个" in rep),
                       "报告含绩效表扫描与 0 个 2021+ 收益", "01_完整报告.md"))
    checks.append(_rec("rulebook_has_param_table",
                       ("最大初始风险" in rb) and ("跟踪宽度" in rb),
                       "规则手册含研究用参数表", "02_规则手册_研究结论版.md"))

    # 整改版口径（文档层；数据层在 verify 的 20-30 项）
    checks.append(_rec("docs_no_stale_family_best",
                       ("族内真实最高" in rep) and ("32 个两段为正的配置" in rep)
                       and ("+20.40%" in rep) and ("ADVQ12_s5_h40" in rep),
                       "报告已把主候选与族内真实最高分列，并写 32 个两段为正（旧值只在撤回索引）", "01_完整报告.md"))
    checks.append(_rec("docs_no_stale_gate_cells",
                       ("族内最高 0.84" not in one) and ("5 个过 25% 的候选里 1 个 dev −36.3%" not in rep)
                       and ("全部 0.7%~3.9%/年" not in one),
                       "G2/G6/G8 旧单元格已清", "00_一页纸.md；01_完整报告.md"))
    checks.append(_rec("docs_scope_and_60pct",
                       ("T0005/VCB/小盘为主板宇宙" in rep) and ("G1 未被直接检验" in rep)
                       and ("任何策略都不可能 60%" in rep),
                       "主板声明已限定为分族；60% 措辞已改", "01_完整报告.md"))
    checks.append(_rec("docs_vcb_disclosures",
                       ("002075" in rep) and ("无冲击项" in rep) and ("0.44pp/年" in rep),
                       "VCB 250 日强平与成本偏离已披露", "01_完整报告.md"))
    checks.append(_rec("docs_data_limits",
                       ("任一 OHLC" in rep) and ("94.01%" in dat) and ("170.1" in dat),
                       "双源 OHLC 与 v1 两分母已改", "01_完整报告.md；03_数据与复现说明.md"))
    checks.append(_rec("acceptance_status_final",
                       fc.get("meta", {}).get("review_status") == FINAL_STATUS,
                       f"review_status={fc.get('meta', {}).get('review_status')}", "04_最终结论.json"))
    clo_path2 = os.path.join(ROOT, "04_证据", "最终审计剩余项关闭.json")
    clo2 = jload(clo_path2) if os.path.exists(clo_path2) else {}
    checks.append(_rec("closure_file_present",
                       bool(clo2) and clo2.get("status") == FINAL_VERDICT,
                       f"closure items={clo2.get('summary', {}).get('n_dedup_items')} "
                       f"unmapped={clo2.get('summary', {}).get('n_unmapped')}",
                       "04_证据/最终审计剩余项关闭.json"))
    checks.append(_rec("no_external_zero_open_claim",
                       "不宣称 0 open" in str(fc.get("audit_remediation", {}).get("note", ""))
                       or "不宣称外部 0 open" in str(fc.get("audit_remediation", {}).get("note", "")),
                       "根 JSON 明确不宣称外部 0 open", "04_最终结论.json"))

    npass = sum(1 for c in checks if c["pass"])
    out = {"generated_at_local": dt.datetime.now().isoformat(timespec="seconds"),
           "tag": TAG, "n_checks": len(checks), "n_pass": npass,
           "n_fail": len(checks) - npass, "checks": checks}
    jdump(SELFCHECK, out)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + " | " + c["detail"])
    print(f"selfcheck: {npass}/{len(checks)} PASS")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="趋势结构项目最终交付")
    ap.add_argument("cmd", choices=["register", "ledger", "stats", "conclusion", "remediation", "closure",
                                    "verify", "selfcheck", "all"])
    ap.add_argument("--force", action="store_true", help="忽略幂等标记，强制重建（会追加重复行）")
    args = ap.parse_args()

    if args.cmd in ("register", "all"):
        stage_register(force=args.force)
    if args.cmd in ("ledger", "all", "stats"):
        stage_ledger(force=args.force)
        stage_final_stats()
    if args.cmd in ("conclusion", "all"):
        stage_conclusion()
    if args.cmd in ("remediation", "all"):
        stage_remediation()
    if args.cmd == "closure":
        rem = jload(os.path.join(ROOT, "04_证据", "审计整改验收.json"))
        stage_closure(rem)
        print("closure: 04_证据/最终审计剩余项关闭.json 已生成")
    if args.cmd in ("verify", "all"):
        v = stage_verify()
        if v["n_fail"]:
            sys.exit(1)
    if args.cmd in ("selfcheck", "all"):
        s = stage_selfcheck()
        if s["n_fail"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
