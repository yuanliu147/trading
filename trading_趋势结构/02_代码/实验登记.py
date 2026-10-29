#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验登记与审计工具（趋势结构项目）。

设计目标
--------
1. 只追加：注册表 CSV 的行一经写入不可修改；修订写新行（revision+1）。
2. 可校验：每行含 prev_sha256 / row_sha256，构成哈希链；verify 可发现任何改动。
3. 防越界：窗口不得晚于 2026-10-26；dev/val 不得触及 2021+；留出只允许一次。
4. 防少报：trials_cumulative 由脚本计算，不接受手填，DSR 的 N 由此而来。
5. 可自检：selftest 在临时目录跑完整生命周期与篡改检测。

用法见 `python3 02_代码/实验登记.py -h` 及各子命令 `-h`。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- 常量

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
DEFAULT_REGISTRY = ROOT / "03_结果" / "实验注册表.csv"
DEFAULT_PREREG = ROOT / "00_研究预注册.md"

DATA_FREEZE = dt.date(2026, 10, 26)
MIN_DATE = dt.date(2005, 1, 1)

STAGE_END_RANGE = {
    "preheat": (dt.date(2005, 1, 1), dt.date(2007, 12, 31)),
    "dev": (dt.date(2008, 1, 1), dt.date(2015, 12, 31)),
    "val": (dt.date(2016, 1, 1), dt.date(2020, 12, 31)),
    "holdout": (dt.date(2021, 1, 1), dt.date(2025, 12, 31)),
    "extension": (dt.date(2026, 1, 1), DATA_FREEZE),
    "audit": (MIN_DATE, DATA_FREEZE),
    "na": (MIN_DATE, DATA_FREEZE),
}

KINDS = {"meta", "audit", "experiment", "ablation", "sensitivity"}
STAGES = set(STAGE_END_RANGE)
STATUSES = {"planned", "running", "done", "failed", "invalid", "rejected", "blocked"}
DECISIONS = {"keep", "discard", "inconclusive", "na"}

COUNTED_KINDS = {"experiment", "ablation", "sensitivity"}
UNCOUNTED_STATUS = {"planned", "blocked"}
TERMINAL_STATUS = {"done", "failed", "invalid", "rejected"}

AUDIT_METRIC_ALLOW = {
    "n_rows", "n_symbols", "n_dates", "n_susp", "n_cov", "n_missing",
    "n_delisted", "n_st", "n_events", "n_future_events", "n_after_freeze",
    "coverage", "n_trades", "n_signals", "n_fail_orders", "n_resume",
    "sample_rows", "n_holdout_runs", "n_trials",
}
FORBIDDEN_METRIC = {
    "cagr", "annual_return", "return", "returns", "ret", "sharpe", "pnl",
    "nav", "equity", "drawdown", "mdd", "win_rate", "profit_factor",
    "calmar", "alpha", "ir", "excess_return",
}

BUDGET_SELECT = 40
BUDGET_TOTAL = 64

COLUMNS = [
    "exp_id", "trial_id", "revision", "kind", "stage", "status", "parent_trial",
    "registered_at_utc", "researcher", "model", "hypothesis", "family",
    "window_start", "window_end", "universe_id", "cost_model_id",
    "params_json", "params_hash", "n_configs", "trials_cumulative",
    "code_path", "code_sha256", "metrics_json", "decision", "notes",
    "prev_sha256", "row_sha256",
]
N_HASHED = len(COLUMNS) - 1  # 前 26 列参与哈希
SEP = "\x1f"
ZERO_HASH = "0" * 64


# ---------------------------------------------------------------- 基础工具

def eprint(*a):
    print(*a, file=sys.stderr)


def canonical_json(text: str, field: str) -> str:
    if text is None or text.strip() == "":
        return ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} 不是合法 JSON: {exc}") from exc
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_date(text: str, field: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} 必须是 YYYY-MM-DD：{text}") from exc


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_canonical(row: dict) -> str:
    return SEP.join(str(row[c]) for c in COLUMNS[:N_HASHED])


def row_hash(row: dict) -> str:
    return sha256_text(row_canonical(row))


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return []
        if header != COLUMNS:
            raise ValueError(
                "注册表表头与分析预期不一致；拒绝操作。\n"
                f"期望: {COLUMNS}\n实际: {header}"
            )
        rows = []
        for lineno, rec in enumerate(reader, start=2):
            if not rec:
                continue
            if len(rec) != len(COLUMNS):
                raise ValueError(f"第 {lineno} 行有 {len(rec)} 列，应为 {len(COLUMNS)} 列")
            rows.append(dict(zip(COLUMNS, rec)))
        return rows


def write_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerow(COLUMNS)
        fh.flush()
        os.fsync(fh.fileno())


@contextlib.contextmanager
def locked(path: Path, exclusive: bool = True):
    """对注册表加 flock；读写都在锁内完成。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_header(path)
    mode = "r+" if exclusive else "r"
    with path.open(mode, encoding="utf-8", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield fh
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_rows_from_handle(fh) -> list:
    fh.seek(0)
    reader = csv.reader(fh)
    header = next(reader)
    if header != COLUMNS:
        raise ValueError("注册表表头不一致，拒绝操作")
    rows = []
    for rec in reader:
        if not rec:
            continue
        if len(rec) != len(COLUMNS):
            raise ValueError(f"注册表存在损坏行（{len(rec)} 列）")
        rows.append(dict(zip(COLUMNS, rec)))
    return rows


def append_row(path: Path, row: dict):
    with locked(path, exclusive=True) as fh:
        fh.seek(0, os.SEEK_END)
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow([row[c] for c in COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------- 统计口径

def latest_rows(rows: list) -> dict:
    """trial_id -> 最新 revision 行。"""
    latest = {}
    for r in rows:
        tid = r["trial_id"]
        if tid not in latest or int(r["revision"]) > int(latest[tid]["revision"]):
            latest[tid] = r
    return latest


def trial_first_order(rows: list) -> list:
    seen = []
    for r in rows:
        if r["trial_id"] not in seen:
            seen.append(r["trial_id"])
    return seen


def cumulative_map(rows: list) -> dict:
    """每个 trial 的最新 revision 对应的累计配置数。"""
    latest = latest_rows(rows)
    out = {}
    total = 0
    for tid in trial_first_order(rows):
        r = latest[tid]
        if r["kind"] in COUNTED_KINDS and r["status"] not in UNCOUNTED_STATUS:
            total += int(r["n_configs"])
        out[tid] = total
    return out


def prefix_cumulative(rows: list) -> dict:
    """exp_id -> 该行写入时点的累计配置数（用于校验历史行）。"""
    out = {}
    latest = {}
    total = 0
    for r in rows:
        tid = r["trial_id"]
        prev = latest.get(tid)
        if prev is not None and r["kind"] in COUNTED_KINDS and prev["status"] not in UNCOUNTED_STATUS:
            total -= int(prev["n_configs"])
        latest[tid] = r
        if r["kind"] in COUNTED_KINDS and r["status"] not in UNCOUNTED_STATUS:
            total += int(r["n_configs"])
        out[r["exp_id"]] = total
    return out


# ---------------------------------------------------------------- 校验

def inside_root(p: Path) -> bool:
    try:
        p.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def resolve_code(code_path: str) -> tuple:
    p = Path(code_path)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if not inside_root(p):
        raise ValueError(f"code 路径必须位于工作区 {ROOT} 内：{p}")
    if not p.is_file():
        raise ValueError(f"code 文件不存在：{p}")
    return p, sha256_file(p)


def validate_window(stage: str, start: dt.date, end: dt.date):
    if end > DATA_FREEZE:
        raise ValueError(
            f"window_end={end} 晚于数据冻结日 {DATA_FREEZE}；禁止使用冻结后行情"
        )
    if start < MIN_DATE:
        raise ValueError(f"window_start={start} 早于数据起点 {MIN_DATE}")
    if start > end:
        raise ValueError("window_start 晚于 window_end")
    lo, hi = STAGE_END_RANGE[stage]
    if not (lo <= end <= hi):
        raise ValueError(
            f"stage={stage} 的 window_end 必须落在 {lo}..{hi}，收到 {end}"
        )
    if stage in {"preheat", "dev", "val"} and end >= dt.date(2021, 1, 1):
        raise ValueError("研究阶段不得触及 2021 年及以后；2021+ 只允许 audit/holdout")


def validate_metrics(kind: str, metrics_json: str, status: str):
    if not metrics_json:
        if kind in COUNTED_KINDS and status == "done":
            raise ValueError("done 的 experiment/ablation/sensitivity 必须提供 metrics_json")
        return
    obj = json.loads(metrics_json)
    if not isinstance(obj, dict):
        raise ValueError("metrics_json 必须是 JSON 对象")
    if kind == "audit":
        bad = {k.lower() for k in obj} & FORBIDDEN_METRIC
        if bad:
            raise ValueError(
                f"audit 行不得含绩效指标键 {sorted(bad)}；2021+ 只允许覆盖类审计"
            )
        unknown = {k for k in obj if k not in AUDIT_METRIC_ALLOW}
        if unknown:
            raise ValueError(f"audit 行含未白名单键 {sorted(unknown)}")
    if kind in COUNTED_KINDS and status == "done":
        for key in ("n_trades", "cagr", "mdd"):
            if key not in obj:
                raise ValueError(f"done 指标缺少必需键 {key}")


def ensure_unique_config(rows, params_hash, stage, kind):
    if kind not in COUNTED_KINDS:
        return
    latest = latest_rows(rows)
    for r in latest.values():
        if r["kind"] not in COUNTED_KINDS:
            continue
        if r["params_hash"] == params_hash and r["stage"] == stage:
            if r["status"] not in {"invalid", "rejected"}:
                raise ValueError(
                    f"重复登记：同一参数+窗口+宇宙+成本已在 {r['trial_id']} "
                    f"(status={r['status']}) 登记于 stage={stage}"
                )


def ensure_holdout_slot(rows, stage, kind, parent_trial):
    if stage != "holdout" or kind not in COUNTED_KINDS:
        return
    latest = latest_rows(rows)
    for r in latest.values():
        if r["stage"] == "holdout" and r["status"] not in {"rejected", "invalid"}:
            raise ValueError(
                f"留出已存在：{r['trial_id']} (status={r['status']})。"
                "留出只允许运行一次；如需作废，先登记 rejected/invalid 修订行"
            )
    if not parent_trial:
        raise ValueError("holdout 必须用 --parent-trial 指向一个已完成的 val 试验")
    p = latest.get(parent_trial)
    if p is None:
        raise ValueError(f"parent_trial={parent_trial} 不存在")
    if p["stage"] != "val" or p["status"] != "done":
        raise ValueError("holdout 的父试验必须是已完成的 val 试验")


def ensure_extension_slot(rows, stage, kind, parent_trial):
    if stage != "extension" or kind not in COUNTED_KINDS:
        return
    latest = latest_rows(rows)
    for r in latest.values():
        if r["stage"] == "extension" and r["status"] not in {"rejected", "invalid"}:
            raise ValueError(f"延伸阶段已存在：{r['trial_id']}")
    if not parent_trial:
        raise ValueError("extension 必须用 --parent-trial 指向已完成的 holdout 试验")
    p = latest.get(parent_trial)
    if p is None or p["stage"] != "holdout" or p["status"] != "done":
        raise ValueError("extension 的父试验必须是已完成的 holdout 试验")


def ensure_post_holdout_guard(rows, stage, kind, reason):
    if kind not in COUNTED_KINDS or stage not in {"dev", "val"}:
        return ""
    latest = latest_rows(rows)
    done_holdout = [
        r for r in latest.values()
        if r["stage"] == "holdout" and r["status"] == "done"
    ]
    if not done_holdout:
        return ""
    if not reason:
        raise ValueError(
            "留出已运行；新的 dev/val 试验必须带 --post-holdout-reason，"
            "并在 notes 标注 [POST-HOLDOUT]"
        )
    return f"[POST-HOLDOUT] {reason}"


def ensure_budget(rows, ack_budget, reason):
    cmap = cumulative_map(rows)
    total = max(cmap.values()) if cmap else 0
    if total > BUDGET_TOTAL and not ack_budget:
        raise ValueError(
            f"试验数 {total} 超过硬预算 {BUDGET_TOTAL}；如确需继续，"
            "使用 --ack-budget 并在 notes 说明"
        )
    if ack_budget and reason:
        return f"[BUDGET] {reason}"
    return ""


def next_ids(rows):
    exp_max = 0
    trial_max = 0
    for r in rows:
        m = re.fullmatch(r"E(\d{5})", r["exp_id"])
        if m:
            exp_max = max(exp_max, int(m.group(1)))
        m = re.fullmatch(r"T(\d{4})", r["trial_id"])
        if m:
            trial_max = max(trial_max, int(m.group(1)))
    return f"E{exp_max + 1:05d}", f"T{trial_max + 1:04d}"


# ---------------------------------------------------------------- 构建行

def build_new_row(rows, args) -> dict:
    kind = args.kind
    stage = args.stage
    status = args.status
    if kind not in KINDS:
        raise ValueError(f"未知 kind：{kind}")
    if stage not in STAGES:
        raise ValueError(f"未知 stage：{stage}")
    if status not in STATUSES:
        raise ValueError(f"未知 status：{status}")
    decision = args.decision or ("na" if kind in {"meta", "audit"} else "inconclusive")
    if decision not in DECISIONS:
        raise ValueError(f"未知 decision：{decision}")

    start = parse_date(args.window_start, "window_start")
    end = parse_date(args.window_end, "window_end")
    validate_window(stage, start, end)

    n_configs = int(args.n_configs)
    if kind in {"meta", "audit"}:
        if n_configs != 0:
            raise ValueError(f"{kind} 行的 n_configs 必须为 0")
    else:
        if n_configs < 1:
            raise ValueError("experiment/ablation/sensitivity 的 n_configs 必须 ≥ 1")

    params_json = canonical_json(args.params_json or "{}", "params_json")
    metrics_json = canonical_json(args.metrics_json or "", "metrics_json")
    validate_metrics(kind, metrics_json, status)

    params_hash = sha256_text(
        SEP.join([
            str(args.family), str(start), str(end),
            str(args.universe_id), str(args.cost_model_id), params_json,
        ])
    )

    if kind in COUNTED_KINDS:
        ensure_unique_config(rows, params_hash, stage, kind)
        ensure_holdout_slot(rows, stage, kind, args.parent_trial)
        ensure_extension_slot(rows, stage, kind, args.parent_trial)

    notes_parts = []
    extra = ensure_post_holdout_guard(rows, stage, kind, args.post_holdout_reason)
    if extra:
        notes_parts.append(extra)
    extra = ensure_budget(rows, args.ack_budget, args.budget_reason)
    if extra:
        notes_parts.append(extra)
    if args.independent_runner:
        notes_parts.append(f"[INDEPENDENT] {args.independent_runner}")
    if args.notes:
        notes_parts.append(args.notes)
    notes = " | ".join(notes_parts)

    code_path = ""
    code_hash = ""
    if args.code:
        p, code_hash = resolve_code(args.code)
        code_path = str(p.relative_to(ROOT))
    elif status in {"done", "failed"} and kind in COUNTED_KINDS:
        raise ValueError("done/failed 的试验必须用 --code 指定运行脚本")

    exp_id, trial_id = next_ids(rows)
    row = {
        "exp_id": exp_id,
        "trial_id": trial_id,
        "revision": "1",
        "kind": kind,
        "stage": stage,
        "status": status,
        "parent_trial": args.parent_trial or "",
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "researcher": args.researcher or os.environ.get("USER", "unspecified"),
        "model": args.model or os.environ.get("KIMI_MODEL") or "unspecified",
        "hypothesis": args.hypothesis,
        "family": args.family,
        "window_start": str(start),
        "window_end": str(end),
        "universe_id": args.universe_id,
        "cost_model_id": args.cost_model_id,
        "params_json": params_json,
        "params_hash": params_hash,
        "n_configs": str(n_configs),
        "trials_cumulative": "",
        "code_path": code_path,
        "code_sha256": code_hash,
        "metrics_json": metrics_json,
        "decision": decision,
        "notes": notes,
        "prev_sha256": rows[-1]["row_sha256"] if rows else ZERO_HASH,
        "row_sha256": "",
    }
    provisional = rows + [row]
    cmap = cumulative_map(provisional)
    row["trials_cumulative"] = str(cmap[trial_id])
    row["row_sha256"] = row_hash(row)
    return row


def build_revision_row(rows, trial_id, changes) -> dict:
    latest = latest_rows(rows)
    if trial_id not in latest:
        raise ValueError(f"trial_id={trial_id} 不存在")
    base = latest[trial_id]
    row = dict(base)
    row["exp_id"], _ = next_ids(rows)
    row["revision"] = str(int(base["revision"]) + 1)
    row["registered_at_utc"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    row.update(changes)
    if row["status"] not in STATUSES:
        raise ValueError(f"未知 status：{row['status']}")
    if row["decision"] not in DECISIONS:
        raise ValueError(f"未知 decision：{row['decision']}")
    validate_metrics(row["kind"], row["metrics_json"], row["status"])
    if row["status"] in {"done", "failed"} and row["kind"] in COUNTED_KINDS:
        if not row["code_path"]:
            raise ValueError("done/failed 的试验必须有 code_path")
    provisional = rows + [row]
    cmap = cumulative_map(provisional)
    row["trials_cumulative"] = str(cmap[trial_id])
    row["prev_sha256"] = rows[-1]["row_sha256"]
    row["row_sha256"] = row_hash(row)
    return row


# ---------------------------------------------------------------- 校验器

def verify_rows(rows, strict=False, allow_code_drift=False):
    errors = []
    warnings = []
    prev = ZERO_HASH
    for i, r in enumerate(rows, start=1):
        tag = f"第{i}行({r.get('exp_id','?')})"
        if r["prev_sha256"] != prev:
            errors.append(f"{tag}: prev_sha256 断链")
        if row_hash(r) != r["row_sha256"]:
            errors.append(f"{tag}: row_sha256 不匹配（内容被修改）")
        prev = r["row_sha256"]
        if r["kind"] not in KINDS:
            errors.append(f"{tag}: 非法 kind={r['kind']}")
        if r["stage"] not in STAGES:
            errors.append(f"{tag}: 非法 stage={r['stage']}")
        if r["status"] not in STATUSES:
            errors.append(f"{tag}: 非法 status={r['status']}")
        if r["decision"] not in DECISIONS:
            errors.append(f"{tag}: 非法 decision={r['decision']}")
        try:
            start = parse_date(r["window_start"], "window_start")
            end = parse_date(r["window_end"], "window_end")
            validate_window(r["stage"], start, end)
        except ValueError as exc:
            errors.append(f"{tag}: 窗口违规 {exc}")
        try:
            validate_metrics(r["kind"], r["metrics_json"], r["status"])
        except ValueError as exc:
            errors.append(f"{tag}: 指标违规 {exc}")
        if r["kind"] in {"meta", "audit"} and r["n_configs"] != "0":
            errors.append(f"{tag}: {r['kind']} 行 n_configs 必须为 0")
        if r["kind"] in COUNTED_KINDS and int(r["n_configs"]) < 1:
            errors.append(f"{tag}: n_configs 必须 ≥ 1")
        if r["code_path"]:
            p = ROOT / r["code_path"]
            if not p.is_file():
                warnings.append(f"{tag}: code_path 不存在 {r['code_path']}")
            elif r["code_sha256"] and sha256_file(p) != r["code_sha256"]:
                msg = f"{tag}: 代码已漂移 {r['code_path']}"
                if allow_code_drift or r["kind"] not in COUNTED_KINDS:
                    warnings.append(msg)
                else:
                    errors.append(msg)

    # 修订链连续
    by_trial = {}
    for r in rows:
        by_trial.setdefault(r["trial_id"], []).append(r)
    for tid, rs in by_trial.items():
        revs = [int(x["revision"]) for x in rs]
        if revs != list(range(1, len(revs) + 1)):
            errors.append(f"{tid}: revision 不连续 {revs}")
        if len(rs) > 1:
            for a, b in zip(rs, rs[1:]):
                if a["status"] in {"done"} and b["status"] == "done":
                    warnings.append(f"{tid}: 已 done 后再次 done（确认是修正而非重跑）")

    # 累计数校验
    pc = prefix_cumulative(rows)
    for r in rows:
        if int(r["trials_cumulative"]) != pc[r["exp_id"]]:
            errors.append(
                f"{r['exp_id']}: trials_cumulative={r['trials_cumulative']} "
                f"应为 {pc[r['exp_id']]}"
            )

    # 留出唯一
    latest = latest_rows(rows)
    holdouts = [
        r for r in latest.values()
        if r["stage"] == "holdout" and r["kind"] in COUNTED_KINDS
        and r["status"] not in {"rejected", "invalid"}
    ]
    if len(holdouts) > 1:
        errors.append(f"留出试验多于 1 个：{[r['trial_id'] for r in holdouts]}")

    # 留出后新增 dev/val 必须显式标注
    done_holdout_rows = [
        r for r in rows
        if r["stage"] == "holdout" and r["status"] == "done"
        and r["kind"] in COUNTED_KINDS
    ]
    if done_holdout_rows:
        cutoff = min(r["exp_id"] for r in done_holdout_rows)
        for r in rows:
            if (r["exp_id"] > cutoff and r["kind"] in COUNTED_KINDS
                    and r["stage"] in {"dev", "val"}
                    and "[POST-HOLDOUT]" not in r["notes"]):
                errors.append(
                    f"{r['exp_id']}: 留出已完成，新增 {r['stage']} 试验未标注 [POST-HOLDOUT]"
                )

    # 重复配置
    seen = {}
    for r in latest.values():
        if r["kind"] not in COUNTED_KINDS or r["status"] in {"invalid", "rejected"}:
            continue
        key = (r["params_hash"], r["stage"])
        if key in seen:
            errors.append(f"重复配置：{seen[key]} 与 {r['trial_id']} (stage={r['stage']})")
        else:
            seen[key] = r["trial_id"]

    if strict:
        open_trials = [
            r["trial_id"] for r in latest.values()
            if r["kind"] in COUNTED_KINDS and r["status"] in {"planned", "running"}
        ]
        if open_trials:
            warnings.append(f"未闭合试验：{open_trials}")
        n = max(pc.values()) if pc else 0
        if n > BUDGET_TOTAL:
            warnings.append(f"试验数 {n} 超预算 {BUDGET_TOTAL}")

    return errors, warnings


# ---------------------------------------------------------------- 子命令

def cmd_init(args):
    path = Path(args.registry)
    if path.exists():
        print(f"已存在，未改动：{path}")
        return 0
    write_header(path)
    print(f"已创建：{path}")
    print(f"列数：{len(COLUMNS)}")
    return 0


def cmd_append(args):
    path = Path(args.registry)
    with locked(path, exclusive=True) as fh:
        rows = read_rows_from_handle(fh)
        row = build_new_row(rows, args)
        if args.dry_run:
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 0
        fh.seek(0, os.SEEK_END)
        csv.writer(fh, lineterminator="\n").writerow([row[c] for c in COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{row['exp_id']} trial={row['trial_id']} rev={row['revision']} "
          f"kind={row['kind']} stage={row['stage']} status={row['status']} "
          f"n_configs={row['n_configs']} cum={row['trials_cumulative']}")
    print(f"row_sha256={row['row_sha256']}")
    if row["model"] != "deepseek-flash":
        eprint("提示：model 未记为 deepseek-flash；项目要求使用 deepseek-flash。")
    return 0


def cmd_start(args):
    path = Path(args.registry)
    with locked(path, exclusive=True) as fh:
        rows = read_rows_from_handle(fh)
        row = build_revision_row(rows, args.trial, {"status": "running"})
        if args.dry_run:
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 0
        fh.seek(0, os.SEEK_END)
        csv.writer(fh, lineterminator="\n").writerow([row[c] for c in COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{row['exp_id']} trial={row['trial_id']} rev={row['revision']} status=running")
    return 0


def cmd_finish(args):
    path = Path(args.registry)
    if args.status not in TERMINAL_STATUS:
        raise ValueError(f"finish 只接受终态 {sorted(TERMINAL_STATUS)}")
    with locked(path, exclusive=True) as fh:
        rows = read_rows_from_handle(fh)
        latest = latest_rows(rows)
        base = latest.get(args.trial)
        if base is None:
            raise ValueError(f"trial_id={args.trial} 不存在")
        metrics = canonical_json(args.metrics_json or "", "metrics_json")
        notes = base["notes"]
        if args.notes:
            notes = (notes + " | " if notes else "") + args.notes
        row = build_revision_row(rows, args.trial, {
            "status": args.status,
            "metrics_json": metrics,
            "decision": args.decision or base["decision"],
            "notes": notes,
        })
        if args.dry_run:
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 0
        fh.seek(0, os.SEEK_END)
        csv.writer(fh, lineterminator="\n").writerow([row[c] for c in COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())
    print(f"{row['exp_id']} trial={row['trial_id']} rev={row['revision']} "
          f"status={row['status']} decision={row['decision']}")
    return 0


def cmd_verify(args):
    path = Path(args.registry)
    if not path.exists():
        eprint(f"注册表不存在：{path}")
        return 2
    with locked(path, exclusive=False) as fh:
        rows = read_rows_from_handle(fh)
    errors, warnings = verify_rows(rows, strict=args.strict,
                                   allow_code_drift=args.allow_code_drift)
    print(f"行数：{len(rows)}  试验数：{max(prefix_cumulative(rows).values()) if rows else 0}")
    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"FAIL {e}")
    if errors:
        print(f"结论：FAIL（{len(errors)} 项）")
        return 1
    print("结论：OK（哈希链、窗口、指标、计数、留出一致性均通过）")
    return 0


def cmd_list(args):
    path = Path(args.registry)
    with locked(path, exclusive=False) as fh:
        rows = read_rows_from_handle(fh)
    latest = latest_rows(rows)
    items = sorted(latest.values(), key=lambda r: r["exp_id"])
    print(f"{'trial':7} {'stage':9} {'kind':11} {'status':8} {'dec':12} "
          f"{'cum':>4} {'window':23} family")
    for r in items:
        if args.kind and r["kind"] not in args.kind.split(","):
            continue
        if args.stage and r["stage"] not in args.stage.split(","):
            continue
        if args.status and r["status"] not in args.status.split(","):
            continue
        print(f"{r['trial_id']:7} {r['stage']:9} {r['kind']:11} {r['status']:8} "
              f"{r['decision']:12} {r['trials_cumulative']:>4} "
              f"{r['window_start']+'~'+r['window_end']:23} {r['family']}")
    return 0


def cmd_summary(args):
    path = Path(args.registry)
    with locked(path, exclusive=False) as fh:
        rows = read_rows_from_handle(fh)
    latest = latest_rows(rows)
    pc = prefix_cumulative(rows)
    n = max(pc.values()) if pc else 0
    print(f"文件：{path}")
    print(f"总行数：{len(rows)}  逻辑试验：{len(latest)}  累计配置数(N)：{n}")
    print()
    by = {}
    for r in latest.values():
        by[(r["kind"], r["stage"], r["status"])] = by.get((r["kind"], r["stage"], r["status"]), 0) + 1
    print(f"{'kind':11} {'stage':9} {'status':8} {'count':>5}")
    for (k, s, st), c in sorted(by.items()):
        print(f"{k:11} {s:9} {st:8} {c:>5}")
    print()
    counted = [r for r in latest.values()
               if r["kind"] in COUNTED_KINDS and r["status"] not in UNCOUNTED_STATUS]
    neg = [r for r in counted if r["status"] in {"failed", "invalid", "rejected"}]
    print(f"计入 N 的试验：{len(counted)}；其中失败/作废/否决：{len(neg)} "
          f"（{len(neg)/len(counted):.0%}）" if counted else "计入 N 的试验：0（仅 planned/blocked）")
    planned = [r for r in latest.values()
               if r["kind"] in COUNTED_KINDS and r["status"] in UNCOUNTED_STATUS]
    print(f"未运行（planned/blocked）：{len(planned)}")
    print(f"预算：选择 {sum(1 for r in counted if r['stage'] in {'dev','val'})}/{BUDGET_SELECT}，"
          f"总配置 {n}/{BUDGET_TOTAL}")
    holds = [r for r in latest.values() if r["stage"] == "holdout"]
    print(f"留出行：{len(holds)} " + (f"-> {holds[0]['trial_id']} {holds[0]['status']}" if holds else ""))
    if n:
        print(f"DSR 的 N 取：{n}")
    return 0


def cmd_anchor(args):
    prereg = Path(args.prereg)
    if not prereg.exists():
        eprint(f"预注册不存在：{prereg}")
        return 2
    digest = sha256_file(prereg)
    size = prereg.stat().st_size
    path = Path(args.registry)
    if args.check:
        with locked(path, exclusive=False) as fh:
            rows = read_rows_from_handle(fh)
        metas = [r for r in latest_rows(rows).values()
                 if r["kind"] == "meta" and r["family"] == "PREREG"]
        if not metas:
            print("未找到 PREREG 锚点")
            return 1
        last = max(metas, key=lambda r: r["exp_id"])
        recorded = json.loads(last["params_json"]).get("sha256")
        ok = recorded == digest
        print(f"文件 sha256：{digest}")
        print(f"注册表锚点：{recorded} ({last['trial_id']})")
        print("结论：" + ("一致" if ok else "不一致（预注册已被修改，需追加新锚点）"))
        return 0 if ok else 1
    args2 = argparse.Namespace(
        registry=args.registry, kind="meta", stage="na", status="done",
        parent_trial="", hypothesis="预注册文件内容锚定（SHA256）",
        family="PREREG", window_start="2026-10-26", window_end="2026-10-26",
        universe_id="NA", cost_model_id="NA",
        params_json=json.dumps({"sha256": digest, "bytes": size, "path": "00_研究预注册.md"}),
        n_configs="0", metrics_json="", decision="na",
        code="", notes="锚点随预注册修订追加，不覆盖；meta 行不绑定工具代码哈希",
        researcher=args.researcher, model=args.model,
        post_holdout_reason="", ack_budget=False, budget_reason="",
        independent_runner="", dry_run=False,
    )
    with locked(path, exclusive=True) as fh:
        rows = read_rows_from_handle(fh)
        row = build_new_row(rows, args2)
        fh.seek(0, os.SEEK_END)
        csv.writer(fh, lineterminator="\n").writerow([row[c] for c in COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())
    print(f"已锚定 {row['exp_id']} trial={row['trial_id']} sha256={digest}")
    return 0


# ---------------------------------------------------------------- 数据审计

def _audit_duckdb(deep=False):
    import duckdb
    db = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
    con = duckdb.connect(db, read_only=True)
    out = []
    q = lambda s, *p: con.execute(s, list(p)).fetchone() if p else con.execute(s).fetchone()
    total, dmin, dmax, codes = q(
        "select count(*), min(date), max(date), count(distinct thscode) from raw_kline_daily")
    out.append(("D1", "DuckDB 行数/代码数", f"{total:,} / {codes:,}", "记录", "INFO"))
    out.append(("D2", "DuckDB 最大日期", str(dmax), "晚于冻结日（证明必须过滤）",
                "PASS" if dmax > DATA_FREEZE else "FAIL"))
    after, after_codes = q(
        "select count(*), count(distinct thscode) from raw_kline_daily where date > ?", DATA_FREEZE)
    out.append(("D3", "冻结后行数（应被过滤）", f"{after:,} / {after_codes:,}", ">0 证明必须过滤", "INFO"))
    null_prev = q("select count(*) filter (where prev_close is null)*1.0/count(*) from raw_kline_daily")[0]
    out.append(("D4", "prev_close 非空率", f"{(1-null_prev)*100:.2f}%", "0（不可用）", "PASS" if null_prev > 0.999 else "FAIL"))
    zero = q("select count(*) from raw_kline_daily where volume = 0")[0]
    out.append(("D5", "volume=0 行数", f"{zero:,}", "0（无停牌行）", "PASS" if zero == 0 else "INFO"))
    ev, ev_future, ev_max = q(
        "select count(*), count(*) filter (where ex_date > ?), max(ex_date) from raw_adjustment_events", DATA_FREEZE)
    out.append(("D6", "公司行动/冻结后事件", f"{ev:,} / {ev_future}", f"最晚 {ev_max}", "INFO"))
    hle = q("select count(*) from raw_kline_daily where high = low")[0]
    out.append(("D7", "high=low 行数（一字板）", f"{hle:,}", "记录", "INFO"))
    con.close()
    return out


def _audit_baostock(deep=False):
    import sqlite3
    db = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    q = lambda s, *p: con.execute(s, p if p else ()).fetchone()
    for flag in (1, 2, 3):
        dmax = q("select max(date) from stock_kline_d where adjustflag=?", flag)[0]
        out.append((f"B{flag}", f"adj={flag} 最大日期", str(dmax), ">= 2026-10-26", "PASS" if dmax and dmax >= str(DATA_FREEZE) else "FAIL"))
    n_codes = q("select count(*) from stock_basic where type=1")[0]
    n_del = q("select count(*) from stock_basic where type=1 and status=0")[0]
    out.append(("B4", "股票代码/其中退市", f"{n_codes:,} / {n_del:,}", "含退市", "PASS" if n_del > 200 else "WARN"))
    n_idx, idx_min, idx_max = q("select count(distinct code), min(date), max(date) from index_kline_d")
    out.append(("B5", "指数数/区间", f"{n_idx} / {idx_min}..{idx_max}", "<= 2026-10-26", "PASS" if idx_max <= str(DATA_FREEZE) else "FAIL"))
    td = q("select count(*) from trade_dates where is_trading_day=1 and calendar_date between '2016-10-13' and '2026-10-26'")[0]
    out.append(("B6", "窗口内交易日", f"{td:,}", "2437", "PASS" if td == 2437 else "FAIL"))
    st, st_n = q("""select sum(case when is_st=1 then 1 else 0 end), count(*)
        from stock_kline_d where adjustflag=3 and date between '2024-01-01' and '2024-12-31'""")
    out.append(("B7", "2024 年 ST 行数/总行数", f"{st:,} / {st_n:,}", "ST 行存在",
                "PASS" if (st or 0) > 0 else "FAIL"))
    # 抽样月：停牌与复牌
    susp, vol0 = q("""select sum(case when trade_status=0 then 1 else 0 end),
        sum(case when volume=0 or volume is null then 1 else 0 end)
        from stock_kline_d where adjustflag=3 and date between '2024-06-01' and '2024-06-30'""")
    out.append(("B8", "样本月停牌/零量行", f"{susp} / {vol0}", "停牌行存在", "PASS" if (susp or 0) > 0 else "FAIL"))
    # 双源一致
    import duckdb
    dcon = duckdb.connect("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb", read_only=True)
    sample = dcon.execute("""select thscode, date, close, volume from raw_kline_daily
        where date between '2017-01-03' and '2017-03-31'
        and thscode in ('600519.SH','000001.SZ','600000.SH') order by thscode, date""").fetchdf()
    diff_close = diff_vol = 0
    matched = 0
    for _, r in sample.iterrows():
        code = ("sh." if r.thscode.endswith(".SH") else "sz.") + r.thscode[:6]
        got = con.execute("select close, volume from stock_kline_d where code=? and date=? and adjustflag=3",
                          (code, str(r.date)[:10])).fetchone()
        if not got:
            continue
        matched += 1
        diff_close = max(diff_close, abs(float(got[0]) - float(r.close)))
        diff_vol = max(diff_vol, abs(float(got[1]) - float(r.volume)))
    dcon.close()
    out.append(("B9", "双源抽样行/最大差", f"{matched} 行; close {diff_close:.4f}, vol {diff_vol:.0f}",
                "差异为 0", "PASS" if matched > 50 and diff_close < 1e-6 and diff_vol < 1 else "FAIL"))
    # 涨停幅度制度
    chinext20 = q("""select count(*) from stock_kline_d where adjustflag=3
        and date between '2020-09-01' and '2020-09-30' and code like 'sz.30%'
        and preclose>0 and abs(close/preclose-1-0.20) < 0.0021""")[0]
    main20 = q("""select count(*) from stock_kline_d where adjustflag=3
        and date between '2020-09-01' and '2020-09-30'
        and (code like 'sh.60%' or code like 'sz.00%')
        and preclose>0 and abs(close/preclose-1-0.20) < 0.0021""")[0]
    out.append(("B10", "2020-09 创业板+20%/主板+20%", f"{chinext20} / {main20}",
                "创业板>0 且 主板=0", "PASS" if chinext20 > 0 and main20 == 0 else "FAIL"))
    # 除权参考价
    row = con.execute("""select preclose, close from stock_kline_d
        where code='sh.688050' and date='2024-06-05' and adjustflag=3""").fetchone()
    if row:
        chg = row[1] / row[0] - 1
        out.append(("B11", "除权日 688050 涨跌", f"{chg:+.2%}", "接近 -1.4%，不是 -45%",
                    "PASS" if -0.05 < chg < 0.05 else "FAIL"))
    else:
        out.append(("B11", "除权日 688050 涨跌", "无行", "存在", "WARN"))
    # 复牌滞后
    rr = con.execute("""select count(*) from stock_kline_d a
        where a.adjustflag=3 and a.date='2024-03-26' and a.code='sz.002089'
        and abs(a.close/a.preclose-1) > 0.5""").fetchone()[0]
    out.append(("B12", "复牌首日 preclose 滞后", "是" if rr else "否", "确认需 resume 规则",
                "PASS" if rr else "INFO"))
    # 北交所
    bj = q("select count(*) from stock_basic where code like 'bj%'")[0]
    out.append(("B13", "北交所代码", f"{bj}", "0（不覆盖）", "INFO"))
    con.close()
    return out


def cmd_audit_data(args):
    checks = []
    if args.only in ("all", "duckdb"):
        checks += _audit_duckdb()
    if args.only in ("all", "baostock"):
        checks += _audit_baostock()
    width = max(len(c[1]) for c in checks)
    fails = 0
    for cid, name, obs, expect, verdict in checks:
        if verdict == "FAIL":
            fails += 1
        print(f"{cid:4} {name:{width}}  实测={obs}  期望={expect}  [{verdict}]")
    print(f"\n合计 {len(checks)} 项，FAIL {fails} 项")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([dict(id=c[0], name=c[1], observed=c[2], expected=c[3], verdict=c[4])
                        for c in checks], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写 {args.json_out}")
    return 1 if fails else 0


# ---------------------------------------------------------------- 可行性算术

def cmd_feasibility(args):
    target = args.target
    log_t = math.log(1 + target)
    print(f"目标年化 {target:.0%} -> 对数收益 {log_t:.4f}")
    print()
    print(f"{'持有日':>6} {'轮/槽/年':>9} {'满仓每笔净':>10} {'d=80%':>8} {'d=60%':>8} {'d=40%':>8}")
    for h in (5, 10, 20, 40, 60, 120):
        turns = 243.0 / h
        full = log_t / turns
        vals = [log_t / (d * turns) for d in (0.8, 0.6, 0.4)]
        print(f"{h:>6} {turns:>9.1f} {full*100:>9.2f}% "
              f"{vals[0]*100:>7.2f}% {vals[1]*100:>7.2f}% {vals[2]*100:>7.2f}%")
    print()
    sig20 = args.vol * math.sqrt(args.hold_days / 243.0)
    mu = log_t / (args.deployment * 243.0 / args.hold_days) + args.cost
    print(f"假设：持有 {args.hold_days} 日、部署率 {args.deployment:.0%}、往返成本 {args.cost:.2%}、"
          f"个股年化波动 {args.vol:.0%}（{args.hold_days} 日 σ={sig20:.1%}）")
    print(f"所需每笔毛边际 {mu:.2%} = {mu/sig20:.2f}σ")
    print()
    print(f"{'盈亏比':>6} {'所需胜率':>9}")
    for payoff in (1.0, 1.5, 2.0, 2.5, 3.0):
        z = mu / sig20
        p = (z + 1) / (payoff + 1)
        print(f"{payoff:>6.1f} {p*100:>8.1f}%")
    print()
    print("年化估计抽样误差（σ/√T）：")
    print(f"{'T年':>4} " + " ".join(f"{v:.0%}vol" for v in (0.25, 0.30, 0.40)))
    for t in (3, 5, 10, 15, 20):
        print(f"{t:>4} " + " ".join(f"{v/math.sqrt(t)*100:>6.1f}%" for v in (0.25, 0.30, 0.40)))
    return 0


# ---------------------------------------------------------------- 自检

def _run(args):
    return subprocess.run([sys.executable, str(HERE)] + args,
                          capture_output=True, text=True)


def cmd_selftest(args):
    tmp = Path(tempfile.mkdtemp(prefix="regaudit_"))
    reg = tmp / "reg.csv"
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra and not cond else ""))
        if not cond:
            ok = False

    base = ["--registry", str(reg)]
    r = _run(base + ["init"])
    check("init 创建注册表", r.returncode == 0 and reg.exists())

    common = ["append", "--kind", "experiment", "--stage", "dev",
              "--hypothesis", "selftest", "--family", "F-S-E-X-G",
              "--window-start", "2008-01-01", "--window-end", "2015-12-31",
              "--universe", "U-T", "--cost-model", "C-T",
              "--params-json", '{"k":2}', "--n-configs", "2",
              "--code", "02_代码/实验登记.py"]
    r = _run(base + common)
    check("登记 dev 试验", r.returncode == 0, r.stderr)
    tid = r.stdout.split("trial=")[1].split()[0] if "trial=" in r.stdout else "T0001"

    r = _run(base + common)
    check("拒绝重复配置", r.returncode != 0 and "重复登记" in (r.stderr + r.stdout))

    r = _run(base + ["append", "--kind", "experiment", "--stage", "dev",
                     "--hypothesis", "freeze", "--family", "F2",
                     "--window-start", "2015-01-01", "--window-end", "2026-10-27",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":3}', "--n-configs", "1"])
    check("拒绝冻结后窗口", r.returncode != 0 and "冻结" in (r.stderr + r.stdout))

    r = _run(base + ["append", "--kind", "experiment", "--stage", "dev",
                     "--hypothesis", "2021", "--family", "F3",
                     "--window-start", "2016-01-01", "--window-end", "2021-06-30",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":4}', "--n-configs", "1"])
    check("拒绝 dev 触及 2021+", r.returncode != 0 and "2021" in (r.stderr + r.stdout))

    r = _run(base + ["append", "--kind", "audit", "--stage", "audit",
                     "--hypothesis", "audit", "--family", "AUD",
                     "--window-start", "2021-01-01", "--window-end", "2025-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", "{}", "--n-configs", "0",
                     "--metrics-json", '{"cagr":0.6}', "--status", "done"])
    check("拒绝 audit 含绩效指标", r.returncode != 0 and "绩效" in (r.stderr + r.stdout))

    r = _run(base + ["append", "--kind", "audit", "--stage", "audit",
                     "--hypothesis", "audit-ok", "--family", "AUD",
                     "--window-start", "2021-01-01", "--window-end", "2025-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", "{}", "--n-configs", "0",
                     "--metrics-json", '{"n_rows":123,"n_symbols":45}', "--status", "done"])
    check("接受覆盖类 audit", r.returncode == 0, r.stderr)

    r = _run(base + ["start", "--trial", tid])
    check("planned->running", r.returncode == 0, r.stderr)
    r = _run(base + ["finish", "--trial", tid, "--status", "done",
                     "--metrics-json", '{"n_trades":40,"cagr":0.12,"mdd":0.2}',
                     "--decision", "keep"])
    check("running->done", r.returncode == 0, r.stderr)

    # 留出的父试验必须是已完成的 val 试验
    r = _run(base + ["append", "--kind", "experiment", "--stage", "val",
                     "--hypothesis", "selftest-val", "--family", "F-S-E-X-G",
                     "--window-start", "2016-01-01", "--window-end", "2020-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":2}', "--n-configs", "1",
                     "--code", "02_代码/实验登记.py"])
    vtid = r.stdout.split("trial=")[1].split()[0] if "trial=" in r.stdout else "T0002"
    check("登记 val 试验", r.returncode == 0, r.stderr)
    _run(base + ["start", "--trial", vtid])
    r = _run(base + ["finish", "--trial", vtid, "--status", "done",
                     "--metrics-json", '{"n_trades":35,"cagr":0.09,"mdd":0.18}',
                     "--decision", "keep"])
    check("val -> done", r.returncode == 0, r.stderr)

    r = _run(base + ["append", "--kind", "experiment", "--stage", "holdout",
                     "--hypothesis", "holdout", "--family", "F-S-E-X-G",
                     "--window-start", "2021-01-01", "--window-end", "2025-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":2}', "--n-configs", "1",
                     "--parent-trial", vtid, "--code", "02_代码/实验登记.py"])
    htid = r.stdout.split("trial=")[1].split()[0] if "trial=" in r.stdout else "T0003"
    check("登记唯一留出", r.returncode == 0, r.stderr)
    r = _run(base + ["append", "--kind", "experiment", "--stage", "holdout",
                     "--hypothesis", "holdout2", "--family", "F-S-E-X-G2",
                     "--window-start", "2021-01-01", "--window-end", "2025-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":5}', "--n-configs", "1",
                     "--parent-trial", vtid])
    check("拒绝第二个留出", r.returncode != 0 and "留出已存在" in (r.stderr + r.stdout))

    _run(base + ["start", "--trial", htid])
    r = _run(base + ["finish", "--trial", htid, "--status", "done",
                     "--metrics-json", '{"n_trades":30,"cagr":0.2,"mdd":0.25}',
                     "--decision", "keep"])
    check("留出 -> done", r.returncode == 0, r.stderr)
    r = _run(base + ["append", "--kind", "experiment", "--stage", "dev",
                     "--hypothesis", "post-holdout-tuning", "--family", "F-S-E-X-G",
                     "--window-start", "2008-01-01", "--window-end", "2015-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":9}', "--n-configs", "1"])
    check("拒绝留出后无标注调参", r.returncode != 0 and "留出已运行" in (r.stderr + r.stdout))
    r = _run(base + ["append", "--kind", "experiment", "--stage", "dev",
                     "--hypothesis", "post-holdout-tuning", "--family", "F-S-E-X-G",
                     "--window-start", "2008-01-01", "--window-end", "2015-12-31",
                     "--universe", "U-T", "--cost-model", "C-T",
                     "--params-json", '{"k":9}', "--n-configs", "1",
                     "--post-holdout-reason", "selftest"])
    check("接受带标注的留出后调参", r.returncode == 0, r.stderr)

    r = _run(base + ["verify", "--strict"])
    check("verify 通过", r.returncode == 0, r.stdout + r.stderr)

    # 篡改检测
    text = reg.read_text(encoding="utf-8").splitlines()
    if len(text) > 2:
        text[1] = text[1].replace("selftest", "tampered", 1)
        reg.write_text("\n".join(text) + "\n", encoding="utf-8")
    r = _run(base + ["verify"])
    check("检测到篡改", r.returncode != 0 and "row_sha256" in (r.stdout + r.stderr))

    # 并发追加
    reg2 = tmp / "conc.csv"
    _run(["--registry", str(reg2), "init"])
    procs = []
    for i in range(8):
        p = subprocess.Popen([sys.executable, str(HERE), "--registry", str(reg2), "append",
                              "--kind", "experiment", "--stage", "dev",
                              "--hypothesis", f"conc{i}", "--family", f"C{i}",
                              "--window-start", "2008-01-01", "--window-end", "2015-12-31",
                              "--universe", "U-T", "--cost-model", "C-T",
                              "--params-json", json.dumps({"i": i}), "--n-configs", "1"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        procs.append(p)
    errs = [p.communicate()[1] for p in procs]
    r = _run(["--registry", str(reg2), "verify"])
    check("并发 8 写无损坏", r.returncode == 0 and all(p.returncode == 0 for p in procs),
          str(errs[:2]))

    print(f"\n结论：{'全部通过' if ok else '存在失败项'}（临时目录 {tmp}）")
    return 0 if ok else 1


# ---------------------------------------------------------------- CLI

def add_common(p):
    p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p.add_argument("--prereg", default=str(DEFAULT_PREREG))


def main(argv=None):
    ap = argparse.ArgumentParser(description="实验登记与数据审计（只追加 + 哈希链）")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="注册表路径")
    ap.add_argument("--prereg", default=str(DEFAULT_PREREG), help="预注册路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="创建注册表")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("append", help="追加一个新试验（revision=1）")
    p.add_argument("--kind", required=True, choices=sorted(KINDS))
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--status", default="planned", choices=sorted(STATUSES))
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--window-start", required=True)
    p.add_argument("--window-end", required=True)
    p.add_argument("--universe", dest="universe_id", required=True)
    p.add_argument("--cost-model", dest="cost_model_id", required=True)
    p.add_argument("--params-json", default="{}")
    p.add_argument("--n-configs", default="1")
    p.add_argument("--metrics-json", default="")
    p.add_argument("--decision", default=None)
    p.add_argument("--code", default="")
    p.add_argument("--parent-trial", default="")
    p.add_argument("--researcher", default="")
    p.add_argument("--model", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--post-holdout-reason", default="")
    p.add_argument("--ack-budget", action="store_true")
    p.add_argument("--budget-reason", default="")
    p.add_argument("--independent-runner", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_append)

    p = sub.add_parser("start", help="planned -> running（追加修订行）")
    p.add_argument("--trial", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("finish", help="追加终态修订行")
    p.add_argument("--trial", required=True)
    p.add_argument("--status", required=True, choices=sorted(TERMINAL_STATUS))
    p.add_argument("--metrics-json", default="")
    p.add_argument("--decision", default=None, choices=sorted(DECISIONS))
    p.add_argument("--notes", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_finish)

    p = sub.add_parser("verify", help="校验哈希链与纪律")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--allow-code-drift", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("list", help="列出最新修订")
    p.add_argument("--kind", default="")
    p.add_argument("--stage", default="")
    p.add_argument("--status", default="")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("summary", help="汇总与 N")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("anchor", help="锚定预注册 SHA256")
    p.add_argument("--check", action="store_true")
    p.add_argument("--researcher", default="")
    p.add_argument("--model", default="")
    p.set_defaults(func=cmd_anchor)

    p = sub.add_parser("audit-data", help="对照数据源核验预注册中的数据事实")
    p.add_argument("--only", default="all", choices=["all", "duckdb", "baostock"])
    p.add_argument("--json-out", default="")
    p.set_defaults(func=cmd_audit_data)

    p = sub.add_parser("feasibility", help="60%% 目标的算术要求（不读行情）")
    p.add_argument("--target", type=float, default=0.60)
    p.add_argument("--hold-days", type=int, default=20)
    p.add_argument("--deployment", type=float, default=0.80)
    p.add_argument("--cost", type=float, default=0.0030)
    p.add_argument("--vol", type=float, default=0.55)
    p.set_defaults(func=cmd_feasibility)

    p = sub.add_parser("selftest", help="在临时目录自检全部机制")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        eprint(f"错误：{exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
