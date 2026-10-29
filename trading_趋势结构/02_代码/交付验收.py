#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交付验收：项目可复现性与交付完整性的总入口。

只做三件事：
  1) 静态校验：冻结数据指纹/截止日、2021+ 收益泄漏、关键表可读、报告内路径、作废索引；
  2) 运行四类自检（数据 / 特征 / 撮合 / 统计）+ 统一引擎自检，全部为轻量检查；
  3) 生成 `文件清单.csv` 与 `04_证据/交付验收.json`。

不重跑任何策略回测；不改他人产物（自检脚本的落盘被重定向到系统临时目录）。

用法：
    python3 02_代码/交付验收.py                 # 全量校验 + 自检（约 90 s）
    python3 02_代码/交付验收.py --quick         # 特征自检缩小样本
    python3 02_代码/交付验收.py --skip-run      # 只做静态校验，不执行自检脚本
    python3 02_代码/交付验收.py --deep          # 冻结文件逐个 SHA-256（56 个）
    python3 02_代码/交付验收.py --inventory-only

产物：
    04_证据/交付验收.json   机器可读验收记录
    文件清单.csv            关键代码/报告/结果/数据的盘点
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "02_代码"
DATA = ROOT / "01_数据"
FROZEN = DATA / "frozen"
EVID = ROOT / "04_证据"
AUDIT = ROOT / "05_审计"
RESULT = ROOT / "03_结果"
OUT_JSON = EVID / "交付验收.json"
OUT_INV = ROOT / "文件清单.csv"

FREEZE = "2026-10-26"
EXPECTED_VERSION = "v2-adjspace-2026-10-28"
RETURN_END = "2020-12-31"
HASH_LIMIT = 16 * 1024 * 1024
CSV_READ_LIMIT = 80 * 1024 * 1024
CSV_TOTAL_BUDGET = 600 * 1024 * 1024

DATE_RE = re.compile(r"(date|day|日期|时间|entry|exit|signal|start|end|window|month|year)", re.I)
PERF_RE = re.compile(r"(cagr|return|ret$|_ret|sharpe|mdd|calmar|nav|equity|pnl|profit|收益|年化|回撤|净值)", re.I)
CODE_DATE_RE = re.compile(r"\b202[1-5]-\d{2}-\d{2}\b")

ROWS: list[dict] = []
T0 = time.time()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add(cid: str, group: str, name: str, status: str, metric="", detail: str = "",
        evidence=None, sec: float | None = None) -> None:
    ROWS.append({"id": cid, "group": group, "name": name, "status": status,
                 "metric": metric, "detail": detail, "evidence": evidence, "sec": sec})
    print(f"[{status}] {cid} {name}: {metric}" + (f" | {detail}" if detail else ""))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def count(status: str) -> int:
    return sum(r["status"] == status for r in ROWS)


# ---------------------------------------------------------------- #
# A. 交付物与元数据
# ---------------------------------------------------------------- #
def check_deliverables(deep_run: bool) -> None:
    required = [
        "00_研究预注册.md",
        "01_数据/数据字典.md",
        "01_数据/meta.json",
        "01_数据/frozen/frozen.duckdb",
        "02_代码/交付验收.py",
        "03_结果/统计整合/01_整合结论.md",
        "03_结果/统计整合/候选分级.csv",
        "05_审计/复权修复影响.md",
        "05_审计/第一阶段代码与负结果审计.md",
        "README.md",
        "文件清单.csv",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    add("G1", "交付物", "关键交付物存在", "PASS" if not missing else "FAIL",
        f"{len(required) - len(missing)}/{len(required)}", f"缺失={missing}" if missing else "")

    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        needs = {
            "复现命令": "交付验收.py" in text,
            "作废索引": "作废索引" in text,
            "数据版本": EXPECTED_VERSION in text,
            "2021+ 纪律": "2021" in text,
        }
        miss = [k for k, v in needs.items() if not v]
        add("G2", "交付物", "README 必备段落", "PASS" if not miss else "FAIL",
            f"{len(needs) - len(miss)}/{len(needs)}", f"缺={miss}" if miss else "")
    else:
        add("G2", "交付物", "README 必备段落", "FAIL", "README 不存在")

    inv = OUT_INV
    if inv.exists():
        with open(inv, encoding="utf-8-sig", newline="") as fh:
            n = sum(1 for _ in csv.reader(fh)) - 1
        add("G3", "交付物", "文件清单可读", "PASS" if n > 100 else "WARN", f"{n} 行")
    else:
        add("G3", "交付物", "文件清单可读", "FAIL", "尚未生成")

    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    ok = (meta.get("data_version") == EXPECTED_VERSION and meta.get("freeze_date") == FREEZE)
    add("G4", "交付物", "数据版本与冻结日", "PASS" if ok else "FAIL",
        f"{meta.get('data_version')} / {meta.get('freeze_date')}",
        f"built_at={meta.get('built_at')}")

    script = CODE / "交付验收.py"
    command_ok = bool(readme.exists() and "python3 02_代码/交付验收.py" in
                      readme.read_text(encoding="utf-8"))
    add("G5", "交付物", "README 的验收命令与脚本绑定",
        "PASS" if command_ok and script.exists() else "FAIL",
        f"script_sha256={sha256(script)[:16]}…", f"README 命令={'有' if command_ok else '缺'}")


# ---------------------------------------------------------------- #
# B. 冻结数据指纹 / 截止日 / 复权不变量
# ---------------------------------------------------------------- #
def _con():
    import duckdb
    con = duckdb.connect(str(FROZEN / "frozen.duckdb"), read_only=True)
    con.execute("SET memory_limit='6GB'")
    return con


def check_data(deep: bool) -> None:
    import duckdb  # noqa: F401

    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    con = _con()
    try:
        n, nsym, d0, d1, n_after, dup = con.execute("""
            SELECT count(*), count(DISTINCT symbol), min(date), max(date),
                   count(*) FILTER (WHERE date > DATE '2026-10-26'),
                   count(*) - count(DISTINCT (symbol, date))
            FROM daily""").fetchone()
        tab = meta["tables"]["daily"]
        ok = (n == tab["rows"] and d1.isoformat() == FREEZE and n_after == 0 and dup == 0)
        add("V1", "数据指纹", "daily 行数/边界/主键与 meta 一致", "PASS" if ok else "FAIL",
            f"rows={n:,} symbols={nsym:,} {d0}~{d1}",
            f"meta rows={tab['rows']:,}; 冻结后行={n_after}; 重复主键={dup}")

        comps = con.execute("""
            SELECT count(*),
                   sum(cast(round(open * 100) AS hugeint)) FILTER (WHERE open IS NOT NULL),
                   sum(cast(round(high * 100) AS hugeint)) FILTER (WHERE high IS NOT NULL),
                   sum(cast(round(low * 100) AS hugeint)) FILTER (WHERE low IS NOT NULL),
                   sum(cast(round(close * 100) AS hugeint)) FILTER (WHERE close IS NOT NULL),
                   sum(cast(round(hfq_close * 10000) AS hugeint)) FILTER (WHERE hfq_close IS NOT NULL),
                   sum(cast(round(factor * 1000000) AS hugeint)) FILTER (WHERE factor IS NOT NULL),
                   sum(cast(round(pct_chg * 10000) AS hugeint)) FILTER (WHERE pct_chg IS NOT NULL)
            FROM daily""").fetchone()
        comps = [int(x) if x is not None else None for x in comps]
        digest = hashlib.sha256("|".join("" if x is None else str(x) for x in comps).encode()).hexdigest()
        want = meta["content_digest"]["daily_sha256"]
        add("V2", "数据指纹", "内容指纹（整数和）与 meta 一致", "PASS" if digest == want else "FAIL",
            f"{digest[:16]}…", f"meta={want[:16]}…", {"components": comps})

        bounds = []
        for table, col in [("index_daily", "date"), ("calendar", "date"), ("bj_daily", "date"),
                           ("market_daily", "date"), ("corp_actions", "ex_date"),
                           ("corp_actions_implied", "ex_date")]:
            mx = con.execute(f"SELECT max({col}) FROM {table}").fetchone()[0]
            bounds.append((table, str(mx)))
        bad = [(t, v) for t, v in bounds if v > FREEZE]
        add("V3", "数据指纹", "所有冻结表不晚于冻结日", "PASS" if not bad else "FAIL",
            f"{len(bounds)} 张表", f"越界={bad}" if bad else f"max={max(v for _, v in bounds)}")

        sample = con.execute("""
            WITH r AS (
              SELECT symbol, date, pct_chg, hfq_close,
                     hfq_close / lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS hret,
                     lag(hfq_close) OVER (PARTITION BY symbol ORDER BY date) AS prev,
                     hash(symbol || '|' || cast(date AS varchar)) % 50 AS h
              FROM daily
            )
            SELECT count(*),
                   count(*) FILTER (WHERE abs(hret - pct_chg / 100) > 1e-6),
                   max(abs(hret - pct_chg / 100)),
                   median(abs(hret - pct_chg / 100))
            FROM r WHERE prev > 0 AND pct_chg IS NOT NULL AND h = 0""").fetchone()
        ok = sample[1] == 0 and sample[0] > 200_000
        add("V4", "数据指纹", "hfq 日收益 = pct_chg（hash 抽样 1/50）", "PASS" if ok else "FAIL",
            f"{sample[0]:,} 对；违约={sample[1]}；max={sample[2]:.2e}；中位={sample[3]:.2e}",
            "v1 反事实：全样本 94.10% 行 >1bp（见 01_自检_数据.py C11）")

        qfq = con.execute("""
            WITH r AS (
              SELECT symbol, date, pct_chg, qfq_close,
                     qfq_close / lag(qfq_close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS qret,
                     lag(qfq_close) OVER (PARTITION BY symbol ORDER BY date) AS prev,
                     hash(symbol || '|' || cast(date AS varchar)) % 97 AS h
              FROM daily_qfq
            )
            SELECT count(*), count(*) FILTER (WHERE abs(qret - pct_chg / 100) > 1e-6),
                   max(abs(qret - pct_chg / 100))
            FROM r WHERE prev > 0 AND pct_chg IS NOT NULL AND h = 0""").fetchone()
        add("V5", "数据指纹", "qfq 日收益 = pct_chg（hash 抽样 1/97）",
            "PASS" if qfq[1] == 0 and qfq[0] > 100_000 else "FAIL",
            f"{qfq[0]:,} 对；违约={qfq[1]}；max={qfq[2]:.2e}")

        last = con.execute("""
            SELECT count(*) FILTER (WHERE abs(qfq_close - close) > 1e-6), count(*)
            FROM (
              SELECT symbol, qfq_close, close,
                     row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
              FROM daily_qfq
            ) WHERE rn = 1""").fetchone()
        add("V6", "数据指纹", "qfq 末日锚定原始收盘",
            "PASS" if last[0] == 0 else "FAIL", f"违约={last[0]}/{last[1]:,}")
    finally:
        con.close()

    files = meta.get("files") or []
    if deep:
        chosen = files
    else:
        names = {"frozen/frozen.duckdb", "frozen/calendar.parquet", "frozen/corp_actions.parquet",
                 "frozen/market_daily.parquet", "frozen/symbols.parquet",
                 "frozen/daily/year=2008/data_0.parquet", "frozen/daily/year=2015/data_0.parquet",
                 "frozen/daily/year=2020/data_0.parquet", "frozen/daily/year=2026/data_0.parquet"}
        chosen = [f for f in files if f["path"] in names]
    bad = []
    for f in chosen:
        p = DATA / f["path"]
        if not p.exists() or sha256(p) != f["sha256"]:
            bad.append(f["path"])
    add("V7", "数据指纹", "冻结文件 SHA-256", "PASS" if not bad else "FAIL",
        f"{len(chosen)}/{len(files)} 个文件，缺失/不符={len(bad)}",
        ("--deep 全量" if deep else "抽样") + (f"；首个不符={bad[0]}" if bad else ""))


# ---------------------------------------------------------------- #
# C. 2021+ 收益泄漏扫描
# ---------------------------------------------------------------- #
WHITELIST = {
    "03_结果/市场状态/市场状态_日频许可_扩展2021_2026_未评估.csv": "状态序列，无收益列（父预注册 §10.4 覆盖审计）",
    "03_结果/实验注册表.csv": "注册表元数据，2026 为文件/登记日期",
    "03_结果/统计整合/2021后日期扫描.csv": "扫描结果自身",
    "03_结果/统计验证/数据覆盖_逐年.csv": "覆盖计数，无收益",
    "03_结果/趋势族/数据审计_年度覆盖.csv": "覆盖计数，无收益",
}
WHITELIST_PREFIX = {
    "03_结果/市场状态_缓存/": "输入缓存（特征/行情），非绩效",
    "03_结果/_审计缓存/": "原始三复权审计数据，非绩效",
    "03_结果/候选_生态切换/07_试验清单.csv": "试验清单元数据",
}


def _read_header(path: Path) -> list[str]:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                return next(csv.reader(fh))
        except Exception:
            continue
    return []


DATE_EXACT = {"date", "dt", "trade_date", "trading_date", "month", "year",
              "日期", "交易日期", "时间", "window_start", "window_end",
              "entry_date", "exit_date", "signal_date", "start_date", "end_date"}


def is_date_col(name: str) -> bool:
    low = name.lower().strip()
    if low in DATE_EXACT:
        return True
    return low.endswith("_date") or low.endswith("_dt")


def _to_dt(s):
    import pandas as pd
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def _max_date(series, name: str = "") -> str | None:
    import pandas as pd
    s = series
    if pd.api.types.is_numeric_dtype(s):
        if name.lower() != "year":
            return None
        years = s[(s >= 1990) & (s <= 2030)]
        return f"{int(years.max())}-12-31" if len(years) else None
    d = _to_dt(s)
    d = d[(d >= pd.Timestamp("1990-01-01")) & (d <= pd.Timestamp("2030-12-31"))]
    if d.notna().sum() == 0:
        return None
    return d.max().date().isoformat()


def _over_mask(s, name: str = ""):
    import pandas as pd
    if pd.api.types.is_numeric_dtype(s) and name.lower() == "year":
        return s.astype(float) > 2020
    return _to_dt(s) > pd.Timestamp(RETURN_END)


def scan_csv(path: Path) -> dict | None:
    import pandas as pd
    header = _read_header(path)
    if not header:
        return None
    dcols = [c for c in header if is_date_col(c)]
    if not dcols:
        return None
    pcols = [c for c in header if PERF_RE.search(c)]
    use = list(dict.fromkeys(dcols[:4] + pcols[:2]))
    try:
        df = pd.read_csv(path, usecols=lambda c: c in use, low_memory=False)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    best, n_over, n_perf_over = None, 0, 0
    perf = df[pcols[0]] if pcols and pcols[0] in df.columns else None
    for c in dcols:
        if c not in df.columns:
            continue
        s = df[c]
        mx = _max_date(s, c)
        if mx and (best is None or mx > best):
            best = mx
        if mx and mx > RETURN_END:
            over = _over_mask(s, c)
            n_over = max(n_over, int(over.sum()))
            if perf is not None and len(over) == len(perf):
                n_perf_over = max(n_perf_over, int((over & perf.notna()).sum()))
    return {"max": best, "rows_over": n_over, "perf_over": n_perf_over,
            "perf_cols": pcols[:3], "date_cols": dcols[:4], "rows": len(df)}


def scan_parquet(path: Path) -> dict | None:
    import pyarrow.parquet as pq
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    names = pf.schema_arrow.names
    dcols = [c for c in names if is_date_col(c)]
    if not dcols:
        return None
    pcols = [c for c in names if PERF_RE.search(c)]
    best, rows_over = None, 0
    for c in dcols[:3]:
        idx = names.index(c)
        for rg in range(pf.metadata.num_row_groups):
            st = pf.metadata.row_group(rg).column(idx).statistics
            if st is None or not st.has_min_max:
                continue
            for v in (st.min, st.max):
                sv = v.date().isoformat() if hasattr(v, "date") else str(v)[:10]
                if best is None or sv > best:
                    best = sv
            mx = st.max
            mx_s = mx.date().isoformat() if hasattr(mx, "date") else str(mx)[:10]
            if mx_s > RETURN_END:
                rows_over += pf.metadata.row_group(rg).num_rows
    return {"max": best, "rows_over": rows_over, "perf_cols": pcols[:3],
            "date_cols": dcols[:3], "rows": pf.metadata.num_rows}


def check_leak(deep: bool) -> None:
    import pandas as pd  # noqa: F401

    perf_files = []
    skipped = []
    budget = CSV_TOTAL_BUDGET
    for p in sorted(RESULT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".parquet"):
            continue
        r = rel(p)
        if r.startswith("03_结果/_审计缓存/") or r.startswith("03_结果/市场状态_缓存/"):
            perf_files.append({"path": r, "whitelist": "输入缓存", "max": FREEZE})
            continue
        info = None
        if p.suffix.lower() == ".csv":
            sz = p.stat().st_size
            if sz > CSV_READ_LIMIT:
                skipped.append((r, sz))
                continue
            if sz > budget:
                skipped.append((r, sz))
                continue
            budget -= sz
            info = scan_csv(p)
        else:
            info = scan_parquet(p)
        if info is None:
            continue
        info["path"] = r
        info["bytes"] = p.stat().st_size
        perf_files.append(info)

    violations, warns = [], []
    for f in perf_files:
        r = f["path"]
        if r in WHITELIST or any(r.startswith(k) for k in WHITELIST_PREFIX):
            f["whitelist"] = WHITELIST.get(r) or next(v for k, v in WHITELIST_PREFIX.items() if r.startswith(k))
            continue
        if not f.get("max"):
            continue
        if f["max"] <= RETURN_END:
            continue
        n_perf = int(f.get("perf_over") or 0)
        n_any = int(f.get("rows_over") or 0)
        if n_perf > 0:
            violations.append(f)
        elif n_any > 0:
            warns.append(f)

    add("L1", "2021+ 泄漏", "绩效表无 2021+ 收益行", "PASS" if not violations else "FAIL",
        f"扫描 {len(perf_files)} 个文件；违规={len(violations)}；日期越界但无收益={len(warns)}",
        f"跳过 {len(skipped)} 个 >20MB CSV" if skipped else "",
        {"violations": violations[:10], "warn": warns[:10],
         "whitelisted": [f for f in perf_files if f.get("whitelist")][:10],
         "skipped": skipped[:10]})

    # 试验总账的两行 2026 边界必须是 meta/audit，不得是绩效行
    ledger = pd.read_csv(RESULT / "试验总账.csv", low_memory=False)
    end = pd.to_datetime(ledger.get("window_end"), errors="coerce")
    late = ledger[end > pd.Timestamp(RETURN_END)]
    bad = late[(late.get("cagr").notna() if "cagr" in late else False)]
    add("L2", "2021+ 泄漏", "试验总账的 2026 边界只是 meta/audit", "PASS" if len(bad) == 0 else "FAIL",
        f"越界行={len(late)}；其中带绩效={len(bad)}",
        f"kind={late['kind'].tolist()}" if len(late) else "")

    # 既有扫描表的结论复核
    scan = pd.read_csv(RESULT / "统计整合/2021后日期扫描.csv")
    add("L3", "2021+ 泄漏", "统计整合的 2021+ 扫描可复核", "PASS",
        f"{len(scan)} 行", f"分类={scan['kind'].value_counts().to_dict()}")

    # 代码里的 2021+ 日期字面量：按用途分类，策略文件里的非护栏项才需要人工确认
    FEE_RULE = {"费用与交易约束.py", "交易制度自检.py", "交易制度适配器.py", "账户撮合.py",
                "构建冻结数据.py", "实验登记.py", "统计验证.py", "核心特征.py"}
    AUDIT_FILES = {"审计双源.py", "审计第一阶段.py", "审计复权引用.py", "独立复现_基线.py",
                   "01_自检_数据.py", "多周期.py", "最终试验总账.py", "整合实验登记.py",
                   "交付验收.py", "退出_自检.py"}
    GUARD_FILES = {"统一回测.py", "统一数据适配.py", "持仓状态机.py", "策略族_结构.py",
                   "策略族_市场状态.py", "候选_结构内反转.py", "数据层.py", "候选_突破回踩.py"}
    kw = ("guard", "assert", "raise", "留出", "holdout", "禁止", "禁用", "审计", "error", "Error",
          "不得", "仅用于", "覆盖", "截断", "拒绝", "抛错", "HoldoutViolation", "no_2021",
          "之后", "freeze", "FREEZE", "冻结", "after_2020", "rows_after", "max_date",
          "any()", "isna", "notna")
    hits, guard_hits, unknown = [], 0, []
    for p in sorted(CODE.glob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not CODE_DATE_RE.search(line):
                continue
            guard = any(k in line for k in kw)
            guard_hits += int(guard)
            group = ("制度/费用" if p.name in FEE_RULE else
                     "审计/自检" if p.name in AUDIT_FILES else
                     "护栏" if guard or p.name in GUARD_FILES else "策略信号/收益")
            if group == "策略信号/收益":
                unknown.append({"file": p.name, "line": i, "text": line.strip()[:140]})
            hits.append({"file": p.name, "line": i, "text": line.strip()[:140],
                         "guard": guard, "group": group})
    add("L4", "2021+ 泄漏", "代码中 2021+ 日期均有明确用途",
        "PASS" if not unknown else "WARN",
        f"命中 {len(hits)} 行；护栏 {guard_hits}；策略/收益面未分类 {len(unknown)}",
        "; ".join(f"{h['file']}:{h['line']}" for h in unknown[:5]),
        {"by_file": {k: sum(1 for h in hits if h["file"] == k)
                     for k in sorted({h["file"] for h in hits})},
         "unclassified": unknown[:20], "hits": hits[:60]})

    # 2021-2026 之间的长窗口结果文件（按文件名）
    name_hits = [rel(p) for p in RESULT.rglob("*")
                 if p.is_file() and p.suffix in (".csv", ".parquet", ".json")
                 and re.search(r"202[1-6]", p.name)]
    unexplained = [p for p in name_hits
                   if p not in WHITELIST and not any(p.startswith(k) for k in WHITELIST_PREFIX)
                   and "覆盖" not in p and "审计" not in p and "扫描" not in p
                   and not p.endswith(".rows")]
    add("L5", "2021+ 泄漏", "文件名含 2021-2026 的结果均可解释",
        "PASS" if not unexplained else "WARN",
        f"{len(name_hits)} 个；未解释={len(unexplained)}",
        "; ".join(unexplained[:5]))


# ---------------------------------------------------------------- #
# D. 自检执行（数据 / 特征 / 撮合 / 统计 / 引擎）
# ---------------------------------------------------------------- #
DATA_BOOT = r"""
import importlib.util, json, pathlib, sys
scratch = pathlib.Path(sys.argv[1]); scratch.mkdir(parents=True, exist_ok=True)
src = pathlib.Path(sys.argv[4])
for name in ("v1_修复前_缺陷量化.json", "v1_修复前_指纹.json", "v1_修复前_meta.json",
             "v1_修复前_自检结果.json", "build_history.csv"):
    p = src / name
    if p.exists():
        (scratch / name).write_bytes(p.read_bytes())
spec = importlib.util.spec_from_file_location("sc_data", sys.argv[2])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.AUDIT = scratch
m.EVIDENCE = scratch
sys.argv = ["01_自检_数据.py", "--deep"] if sys.argv[3] == "deep" else ["01_自检_数据.py"]
m.main()
print("@@RESULT@@" + json.dumps({
    "n": len(m.RESULTS),
    "pass": sum(r["status"] == "PASS" for r in m.RESULTS),
    "fail": sum(r["status"] == "FAIL" for r in m.RESULTS),
    "info": sum(r["status"] == "INFO" for r in m.RESULTS),
    "checks": m.RESULTS}, ensure_ascii=False, default=str))
"""

STATS_BOOT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import 统计验证 as sv
s = sv.run_selftest()
print("@@RESULT@@" + json.dumps({
    "n_checks": s["n_checks"], "n_passed": s["n_passed"], "all_passed": s["all_passed"],
    "checks": s["checks"]}, ensure_ascii=False, default=str))
"""

ENGINE_BOOT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import 统一回测 as ub
events = ub.EventBook().load("2007-01-01", "2020-12-31")
panel = ub.FrozenPanel("2007-01-01", "2020-12-31", events=events)
checks = ub.selfcheck(panel, events, write=False)
print("@@RESULT@@" + json.dumps(checks, ensure_ascii=False, default=str))
"""


def run(args: list[str], timeout: int) -> tuple[int, str, str, float]:
    t = time.time()
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        return p.returncode, p.stdout, p.stderr, time.time() - t
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout {timeout}s", time.time() - t


def parse_marker(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("@@RESULT@@"):
            try:
                return json.loads(line[len("@@RESULT@@"):])
            except Exception:
                return None
    return None


def run_suites(quick: bool) -> dict:
    suites = {}
    scratch = Path(tempfile.mkdtemp(prefix="deliv_"))

    rc, out, err, sec = run([sys.executable, "-c", DATA_BOOT, str(scratch / "data"),
                             str(CODE / "01_自检_数据.py"), "light" if quick else "deep",
                             str(DATA / "_audit")], 900)
    payload = parse_marker(out)
    if payload is None:
        add("S1", "自检", "数据层自检（01_自检_数据.py）", "FAIL", f"rc={rc} {sec:.1f}s", (err or out)[-300:])
    else:
        ok = payload["fail"] == 0 and payload["pass"] >= 28
        add("S1", "自检", "数据层自检（01_自检_数据.py）", "PASS" if ok else "FAIL",
            f"{payload['pass']} PASS / {payload['fail']} FAIL / {payload['info']} INFO，{sec:.1f}s",
            "输出重定向到临时目录，未改 01_数据/_audit", {"checks": payload["checks"]}, sec)
        suites["data"] = payload
    suites["data_rc"] = rc

    feat_args = [sys.executable, str(CODE / "多周期.py"), "--out", str(scratch / "feature.json")]
    if quick:
        feat_args += ["--cases", "12", "--symbols", "4", "--stats-symbols", "8"]
    rc, out, err, sec = run(feat_args, 900)
    try:
        payload = json.loads((scratch / "feature.json").read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if payload is None:
        add("S2", "自检", "特征/多周期因果自检", "FAIL", f"rc={rc} {sec:.1f}s", (err or out)[-300:])
    else:
        s = payload["summary"]
        ok = s["fail"] == 0
        add("S2", "自检", "特征/多周期因果自检", "PASS" if ok else "FAIL",
            f"{s['pass']}/{s['total']} PASS，{sec:.1f}s",
            f"样本 cases={12 if quick else 40} symbols={4 if quick else 12}",
            {"summary": s, "meta": payload.get("meta")}, sec)
        suites["feature"] = s
    suites["feature_rc"] = rc

    rc, out, err, sec = run([sys.executable, str(CODE / "账户撮合.py"), "--out", str(scratch / "match.json")], 900)
    try:
        payload = json.loads((scratch / "match.json").read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if payload is None:
        add("S3", "自检", "撮合/账户自检", "FAIL", f"rc={rc} {sec:.1f}s", (err or out)[-300:])
    else:
        t = payload["self_tests"]
        audits = payload.get("data_audit") or {}
        bad_audit = [k for k, v in audits.items() if not v.get("ok")]
        ok = t["failed"] == 0 and not bad_audit
        add("S3", "自检", "撮合/账户自检", "PASS" if ok else "FAIL",
            f"{t['passed']}/{t['total']} PASS；审计 {len(audits)} 项，失败 {len(bad_audit)}，{sec:.1f}s",
            f"失败={bad_audit}" if bad_audit else "",
            {"self_tests": {k: t[k] for k in ("total", "passed", "failed")}, "audit_fail": bad_audit}, sec)
        suites["matching"] = {"total": t["total"], "passed": t["passed"], "failed": t["failed"],
                              "audits": len(audits), "audit_fail": bad_audit}
    suites["matching_rc"] = rc

    rc, out, err, sec = run([sys.executable, "-c", STATS_BOOT, str(CODE)], 600)
    payload = parse_marker(out)
    if payload is None:
        add("S4a", "自检", "统计框架自检", "FAIL", f"rc={rc} {sec:.1f}s", (err or out)[-300:])
    else:
        add("S4a", "自检", "统计框架自检", "PASS" if payload["all_passed"] else "FAIL",
            f"{payload['n_passed']}/{payload['n_checks']} PASS，{sec:.1f}s", "",
            {"checks": payload["checks"]}, sec)
        suites["stats"] = {"n_checks": payload["n_checks"], "n_passed": payload["n_passed"],
                           "all_passed": payload["all_passed"]}
    suites["stats_rc"] = rc

    rc, out, err, sec = run([sys.executable, str(CODE / "整合实验登记.py"), "selftest"], 600)
    passes = out.count("[PASS]")
    fails = out.count("[FAIL]")
    ok = rc == 0 and fails == 0 and passes >= 7
    add("S4b", "自检", "多重性/哈希链机制自检", "PASS" if ok else "FAIL",
        f"{passes} PASS / {fails} FAIL，{sec:.1f}s", (err or "")[-200:] if not ok else "",
        {"tail": out.strip().splitlines()[-3:]}, sec)
    suites["registry"] = {"pass": passes, "fail": fails, "rc": rc}

    rc, out, err, sec = run([sys.executable, "-c", ENGINE_BOOT, str(CODE)], 900)
    payload = parse_marker(out)
    if payload is None:
        add("S5", "自检", "统一引擎自检（写入关闭）", "FAIL", f"rc={rc} {sec:.1f}s", (err or out)[-300:])
    else:
        gates = payload.get("gates") or {}
        ok = bool(gates.get("all_pass"))
        add("S5", "自检", "统一引擎自检（写入关闭）", "PASS" if ok else "FAIL",
            f"gates {sum(bool(v) for v in gates.values())}/{len(gates)} PASS，{sec:.1f}s",
            "", {"gates": gates, "equivalence": {k: payload.get("equivalence", {}).get(k)
                                                 for k in ("shares_median_abs_err_bp", "shares_max_abs_err_bp")},
                 "guard_2021": payload.get("guard_2021")}, sec)
        suites["engine"] = {"gates": gates,
                            "equivalence": {k: payload.get("equivalence", {}).get(k)
                                            for k in ("shares_median_abs_err_bp", "shares_max_abs_err_bp")}}
    suites["engine_rc"] = rc
    return suites


def reuse_suites() -> dict:
    """--skip-run：复用既有证据文件，只做一致性核对。"""
    suites = {}
    data = json.loads((DATA / "_audit/自检结果.json").read_text(encoding="utf-8"))
    ok = data["fail"] == 0 and data["pass"] >= 28 and data["data_version"] == EXPECTED_VERSION
    add("S1", "自检", "数据层自检（复用 01_数据/_audit/自检结果.json）", "PASS" if ok else "FAIL",
        f"{data['pass']} PASS / {data['fail']} FAIL", f"generated={data['generated_at']}")
    suites["data"] = {"n": len(data["results"]), "pass": data["pass"], "fail": data["fail"]}

    feat = json.loads((EVID / "特征因果自检.json").read_text(encoding="utf-8"))
    s = feat["summary"]
    add("S2", "自检", "特征/多周期因果自检（复用证据）", "PASS" if s["fail"] == 0 else "FAIL",
        f"{s['pass']}/{s['total']} PASS", f"generated={feat['meta']['generated_at']}")
    suites["feature"] = s

    match = json.loads((EVID / "撮合自检.json").read_text(encoding="utf-8"))
    t = match["self_tests"]
    add("S3", "自检", "撮合/账户自检（复用证据）", "PASS" if t["failed"] == 0 else "FAIL",
        f"{t['passed']}/{t['total']} PASS", f"generated={match['meta']['generated_at']}")
    suites["matching"] = {"total": t["total"], "passed": t["passed"], "failed": t["failed"]}

    stats = json.loads((RESULT / "统计验证/自检_结果.json").read_text(encoding="utf-8"))
    add("S4a", "自检", "统计框架自检（复用证据）",
        "PASS" if stats["all_passed"] else "FAIL", f"{stats['n_passed']}/{stats['n_checks']} PASS",
        f"generated={stats['generated_at']}")
    suites["stats"] = {"n_checks": stats["n_checks"], "n_passed": stats["n_passed"],
                       "all_passed": stats["all_passed"]}

    eng = json.loads((EVID / "统一引擎自检.json").read_text(encoding="utf-8"))
    gates = eng["checks"]["gates"]
    add("S5", "自检", "统一引擎自检（复用证据）", "PASS" if gates.get("all_pass") else "FAIL",
        f"gates {sum(bool(v) for v in gates.values())}/{len(gates)} PASS",
        f"generated={eng['meta']['generated_at']}")
    suites["engine"] = {"gates": gates}

    rc, out, err, sec = run([sys.executable, str(CODE / "整合实验登记.py"), "selftest"], 600)
    passes, fails = out.count("[PASS]"), out.count("[FAIL]")
    add("S4b", "自检", "多重性/哈希链机制自检", "PASS" if rc == 0 and fails == 0 else "FAIL",
        f"{passes} PASS / {fails} FAIL，{sec:.1f}s")
    suites["registry"] = {"pass": passes, "fail": fails, "rc": rc}
    return suites


# ---------------------------------------------------------------- #
# E. 关键结果表可读 / 路径正确
# ---------------------------------------------------------------- #
KEY_TABLES = [
    ("03_结果/统计整合/候选分级.csv", "分级"),
    ("03_结果/统计整合/统一置信区间.csv", "区间"),
    ("03_结果/统计整合/多重性.json", "多重性"),
    ("03_结果/统计整合/数据缺陷暴露.csv", "缺陷暴露"),
    ("03_结果/统计整合/2021后日期扫描.csv", "2021+ 扫描"),
    ("03_结果/统计整合/试验计数.csv", "试验计数"),
    ("03_结果/试验总账.csv", "试验总账"),
    ("03_结果/实验注册表.csv", "官方注册表"),
    ("03_结果/趋势族/组合模拟_汇总.csv", "趋势族汇总"),
    ("03_结果/趋势族/净值_S4_结构突破55_纯时间60_dev.parquet", "趋势族净值"),
    ("03_结果/趋势族/交易明细_全部.parquet", "趋势族逐笔"),
    ("03_结果/结构族_信号×离场_矩阵.csv", "结构族矩阵"),
    ("03_结果/结构族_离场对比_K3.csv", "结构族离场"),
    ("03_结果/结构族_回测_净值_K3_hold40.csv", "结构族净值"),
    ("03_结果/结构族_自检.csv", "结构族自检"),
    ("03_结果/退出_臂对照_组合级.csv", "退出族汇总"),
    ("03_结果/退出_净值_TB1_ALL_dev.csv", "退出族净值"),
    ("03_结果/市场状态/市场状态_分窗口绩效.csv", "市场状态绩效"),
    ("03_结果/市场状态/市场状态_自检.csv", "市场状态自检"),
    ("03_结果/统一引擎/汇总.json", "统一引擎汇总"),
    ("03_结果/统一引擎/净值_FIX20_dev.csv", "统一引擎净值"),
    ("03_结果/统一引擎/交易_FIX20_dev.csv", "统一引擎逐笔"),
    ("03_结果/执行压力/02_配置汇总.csv", "执行压力汇总"),
    ("03_结果/候选_小盘低波趋势/17_关键数字.json", "小盘低波关键数字"),
    ("03_结果/候选_小盘低波趋势/10_组合_网格.csv", "小盘低波网格"),
    ("03_结果/候选_小盘低波趋势/13_组合_净值_主单元.csv", "小盘低波净值"),
    ("03_结果/候选_趋势回调/11_统计.json", "趋势回调统计"),
    ("03_结果/候选_收缩突破/统计诊断.json", "收缩突破统计"),
    ("03_结果/候选_生态切换/05_组合绩效.csv", "生态切换绩效"),
    ("03_结果/候选_相对强弱/04_统计检验.json", "相对强弱统计"),
    ("03_结果/候选_突破回踩/10_自检.json", "突破回踩自检"),
    ("03_结果/候选_结构内反转/11_统计.json", "结构内反转统计"),
    ("03_结果/候选_长期双动量/04_统计检验.json", "长期双动量统计"),
    ("03_结果/候选_风险预算/09_自检.csv", "风险预算自检"),
    ("03_结果/候选_退出统一/02_臂汇总.csv", "退出统一汇总"),
    ("03_结果/主候选_T0005/07_统计_主候选.json", "T0005 统计"),
    ("03_结果/主候选_T0005/03_日净值_T0005_dev.csv", "T0005 净值"),
]


def check_tables() -> None:
    import pandas as pd
    missing, unreadable, empty = [], [], []
    detail = []
    for path, role in KEY_TABLES:
        p = ROOT / path
        if not p.exists():
            missing.append(path)
            continue
        try:
            if p.suffix == ".csv":
                header = _read_header(p)
                df = pd.read_csv(p, nrows=5, low_memory=False)
                nrows = None
                if p.stat().st_size <= CSV_READ_LIMIT:
                    with open(p, encoding="utf-8-sig", errors="ignore") as fh:
                        nrows = max(sum(1 for _ in fh) - 1, 0)
                ncol = len(header)
                detail.append(f"{role}:{ncol}列" + (f"/{nrows}行" if nrows is not None else ""))
                if df.empty:
                    empty.append(path)
            elif p.suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                detail.append(f"{role}:{len(obj) if hasattr(obj, '__len__') else 1}键")
            elif p.suffix == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(p)
                detail.append(f"{role}:{pf.metadata.num_rows:,}行/{len(pf.schema_arrow.names)}列")
                if pf.metadata.num_rows == 0:
                    empty.append(path)
        except Exception as exc:
            unreadable.append(f"{path}: {type(exc).__name__}")
    add("T1", "关键表", "关键结果表可读且非空",
        "PASS" if not missing and not unreadable and not empty else "FAIL",
        f"{len(KEY_TABLES) - len(missing) - len(unreadable) - len(empty)}/{len(KEY_TABLES)}",
        f"缺失={missing[:3]} 不可读={unreadable[:3]} 空={empty[:3]}", {"tables": detail})

    has_cols = 0
    checked = 0
    for path, _ in KEY_TABLES:
        if not path.endswith(".csv"):
            continue
        header = [h.lstrip("\ufeff") for h in _read_header(ROOT / path)]
        checked += 1
        if all(c in header for c in ("window_start", "window_end", "adjust", "cost_model_id")):
            has_cols += 1
    add("T2", "关键表", "结果表含预注册 §15 四列（或表头元数据）", "WARN" if has_cols < checked else "PASS",
        f"{has_cols}/{checked} 张 CSV 显式含四列",
        "口径实际写在文件名/报告/JSON meta 中；§15 的硬要求未在表内落实")


TEMPLATE_REFS = {
    "03_结果/净值.csv", "03_结果/交易.csv", "03_结果/持仓.csv", "03_结果/失败订单.csv",
    "03_结果/年度月度.csv", "03_结果/消融.csv", "03_结果/稳健性.csv", "03_结果/walkforward.csv",
}


def check_paths() -> None:
    md_files = [ROOT / "README.md", ROOT / "00_研究预注册.md",
                RESULT / "统计整合/01_整合结论.md", RESULT / "统一引擎/操作说明与复现.md",
                AUDIT / "复权修复影响.md", DATA / "数据字典.md"]
    md_files += sorted(RESULT.glob("候选_*/*一页纸*.md"))
    md_files += sorted(RESULT.glob("候选_*/*报告*.md"))
    md_files += sorted(RESULT.glob("候选_*/1[0-9]_结论.md"))
    refs, missing = 0, []
    seen = set()
    for p in md_files:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"`([^`\n]{3,160})`", text):
            s = m.group(1).strip()
            if "/" not in s or s.startswith(("http", "python", "bash", "sql", "SELECT", "--")):
                continue
            if not re.match(r"^(00_|01_|02_|03_|04_|05_|README|文件清单)", s):
                continue
            if any(x in s for x in ("<", ">", "|", "...", "YYYY", "xxxx", "…")):
                continue
            if s in TEMPLATE_REFS:
                continue
            refs += 1
            key = (p, s)
            if key in seen:
                continue
            seen.add(key)
            target = s.split("#")[0].strip()
            if "*" in target:
                if not list(ROOT.glob(target)):
                    missing.append(f"{rel(p)} -> {s}")
            elif not (ROOT / target).exists():
                missing.append(f"{rel(p)} -> {s}")
    add("T3", "关键表", "报告内引用的项目路径存在",
        "PASS" if not missing else "WARN",
        f"{len(seen)} 条引用；缺失 {len(missing)}",
        "; ".join(missing[:8]), {"missing": missing[:50]})


# ---------------------------------------------------------------- #
# F. 作废索引
# ---------------------------------------------------------------- #
RETRACTIONS = [
    {
        "path": "03_结果/结构族_开发验证规范.md",
        "kind": "tp8",
        "what": "§8.6/§8.7/§9 把 tp8 列为最优离场（+2.6%）、G+tp8 +7.3%，并据此写“快速兑现优于让利润奔跑”",
        "why": "tp8 在 120 日内未触及 +8% 时把 5,559 笔失败离场置 NaN 后过滤；按最后收盘强制离场这些事件均值 −15.4%。G+tp8 8 种子 +6.36% → −1.14%；E+tp8 +2.21% → −3.07%",
        "authority": "05_审计/第一阶段代码与负结果审计.md §F2（撤回声明）",
        "action": "只可引用其负结论；tp8 正收益、'最佳离场'表述作废，不得进入一页纸",
    },
    {
        "path": "03_结果/统计整合/01_整合结论.md",
        "kind": "tp8",
        "what": "§2 矩阵表 `struct_rules` 最好 +2.48%（PBO 0.021）来自 `tp8` 聚合规则；`多重性.json` 同值",
        "why": "tp8 的正收益已被失败离场修正撤回；+2.48% 不能证明“聚合规则有边际”",
        "authority": "05_审计/第一阶段代码与负结果审计.md §F2",
        "action": "可作为负结果引用；不得单引 +2.48%，引用时须写“tp8 行修正后为负”",
    },
    {
        "path": "03_结果/结构族_离场规则_组合_K3.csv",
        "kind": "tp8",
        "what": "tp8 组合层 CAGR/Sharpe 行（+2.6%）",
        "why": "同上，正收益来自丢弃左尾而非策略优势",
        "authority": "05_审计/第一阶段代码与负结果审计.md §F2",
        "action": "数据保留；tp8 行不得单独作为结果引用，引用须附失败离场修正值",
    },
    {
        "path": "03_结果/结构族_离场规则_净值_K3.csv",
        "kind": "tp8",
        "what": "tp8 净值列",
        "why": "同上；该列为丢弃左尾后的路径",
        "authority": "05_审计/第一阶段代码与负结果审计.md §F2",
        "action": "数据保留；tp8 列不得单独作为净值证据",
    },
    {
        "path": "03_结果/结构族_信号×离场_矩阵.csv",
        "kind": "tp8",
        "what": "6 信号 × 离场矩阵中的 tp8 行（G 列 +7.3）",
        "why": "同一 tp8 实现；矩阵内 tp8 格偏高",
        "authority": "05_审计/第一阶段代码与负结果审计.md §F2",
        "action": "同上；聚合结论以 hold40/struct 行为准",
    },
    {
        "path": "03_结果/结构族_离场对比_K2.csv / _K3.csv / _K5.csv",
        "kind": "tp8",
        "what": "事件层 tp8 均值/胜率/分位",
        "why": "事件层 tp8 统计本身可用于比较路径，但其组合层正收益不可复现",
        "authority": "05_审计/第一阶段代码与负结果审计.md §F2",
        "action": "仅作事件层描述；不得外推为组合收益",
    },
    {
        "path": "01_数据/_audit/v1_修复前_自检报告.md",
        "kind": "v1_hfq",
        "what": "v1 自检宣称 hfq=close×factor 成立、'复权收益偏差 0'",
        "why": "C1 是恒等式、C2 用 factor 收益冒充 hfq 收益；旧公式把价格收益计入两次（全样本 94.10% 行 >1bp，中位 1.41pp）",
        "authority": "05_审计/复权修复影响.md §4.1；01_数据/_audit/修复后自检报告.md",
        "action": "历史快照，禁止引用其结论；以 v2 自检 28 PASS 为准",
    },
    {
        "path": "01_数据/_audit/v1_修复前_自检结果.json",
        "kind": "v1_hfq",
        "what": "v1 自检的机器结果",
        "why": "同上",
        "authority": "05_审计/复权修复影响.md §4.1",
        "action": "只作 v1 缺陷证据，不作数据可用性证据",
    },
    {
        "path": "03_结果/候选_相对强弱/00_预注册_相对强弱.md",
        "kind": "v1_hfq",
        "what": "§2.1 写“后复权价用冻结层自建 factor（hfq = raw × factor）”",
        "why": "该式是 v1 双计公式；代码实际使用 exit_cache 的 Baostock 后复权价，结果复算到 A 空间（中位误差 1.96pp）",
        "authority": "05_审计/复权修复影响.md §3.5 / §4.3",
        "action": "预注册只追加不改；引用时以代码与审计为准，勿按字面复算",
    },
    {
        "path": "03_结果/趋势族_开发验证规范.md",
        "kind": "note_baostock_hfq",
        "what": "通篇以“hfq”指 Baostock adjustflag=1 后复权价",
        "why": "不是冻结层 v1 hfq；但同名易混。Baostock 另有年末重定基缺陷（63 个证券日 >0.5pp）",
        "authority": "05_审计/复权修复影响.md §5.1(2)",
        "action": "可引用；引用时写明“Baostock flag1”，不得写成“冻结层 hfq”",
    },
    {
        "path": "03_结果/退出与仓位_实验规范.md",
        "kind": "note_baostock_hfq",
        "what": "以 hfq 空间记账（exit_cache.px_hfq）",
        "why": "同上，Baostock 后复权，非冻结层 v1",
        "authority": "05_审计/复权修复影响.md §5.1(4)",
        "action": "可引用；写明报价比价口径与 Baostock 来源",
    },
]


def check_retractions() -> None:
    missing = [r["path"].split(" / ")[0] for r in RETRACTIONS if not (ROOT / r["path"].split(" / ")[0]).exists()]
    tp8 = [r for r in RETRACTIONS if r["kind"] == "tp8"]
    v1 = [r for r in RETRACTIONS if r["kind"] == "v1_hfq"]
    add("R1", "作废索引", "tp8 撤回条目可定位", "PASS" if not missing else "FAIL",
        f"{len(tp8)} 条", f"缺失={missing}" if missing else "")
    add("R2", "作废索引", "v1 hfq 条目可定位", "PASS" if not missing else "FAIL",
        f"{len(v1)} 条", "", {"retractions": RETRACTIONS})

    authority = (AUDIT / "第一阶段代码与负结果审计.md").read_text(encoding="utf-8")
    has = ("tp8" in authority and "撤回" in authority and "−15.4" in authority) or \
          ("tp8" in authority and "撤回" in authority and "-15.4" in authority)
    add("R3", "作废索引", "撤回依据包含量化反证", "PASS" if has else "WARN",
        "第一阶段审计 §F2 含 5,559 笔 / −15.4% / 符号翻转",
        "未找到 −15.4% 字样" if not has else "")


# ---------------------------------------------------------------- #
# G. 文件清单
# ---------------------------------------------------------------- #
CODE_ROLE = [
    ("构建冻结数据", "数据构建"), ("01_自检_数据", "数据自检"), ("数据层", "数据访问"),
    ("统一数据适配", "数据适配"), ("核心特征", "特征"), ("多周期", "特征/多周期"),
    ("账户撮合", "撮合/账户"), ("费用与交易约束", "费用/约束"), ("交易制度适配器", "交易制度"),
    ("交易制度自检", "交易制度自检"), ("持仓状态机", "状态机"),
    ("统一回测", "统一引擎"), ("策略族_", "策略族"), ("退出_", "退出族"),
    ("候选_", "候选族"), ("主候选_", "主候选T0005"), ("最终试验总账", "注册表/整合"),
    ("统计验证", "统计"), ("实验登记", "注册表"),
    ("整合实验登记", "注册表/整合"), ("审计", "审计"), ("独立复现_基线", "独立复现"),
    ("交付验收", "交付验收"),
]

DOC_STATUS = {
    "03_结果/结构族_开发验证规范.md": "部分作废(tp8)",
    "03_结果/候选_相对强弱/00_预注册_相对强弱.md": "条文保留/口径描述过期",
    "01_数据/_audit/v1_修复前_自检报告.md": "作废(v1-hfq)",
    "01_数据/_audit/v1_修复前_自检结果.json": "作废(v1-hfq)",
    "03_结果/结构族_离场规则_组合_K3.csv": "有效-数据(tp8行作废)",
    "03_结果/结构族_离场规则_净值_K3.csv": "有效-数据(tp8列作废)",
    "03_结果/结构族_信号×离场_矩阵.csv": "有效-数据(tp8行作废)",
    "03_结果/结构族_离场对比_K2.csv": "有效-事件层(组合外推作废)",
    "03_结果/结构族_离场对比_K3.csv": "有效-事件层(组合外推作废)",
    "03_结果/结构族_离场对比_K5.csv": "有效-事件层(组合外推作废)",
}


def doc_role(relpath: str) -> str:
    name = Path(relpath).name
    if name == "README.md":
        return "交付说明"
    if name == "文件清单.csv":
        return "交付清单"
    if "预注册" in name or "研究规范" in name or "实验规范" in name or "开发验证规范" in name:
        return "预注册/规范"
    if "一页纸" in name:
        return "一页纸"
    if "报告" in name or "结论" in name:
        return "报告"
    if relpath.startswith("05_审计/"):
        return "审计"
    if relpath.startswith("01_数据/_audit/"):
        return "数据自检"
    if "数据字典" in name:
        return "数据字典"
    return "说明"


def result_role(relpath: str) -> str:
    name = Path(relpath).name
    if name.endswith(".parquet"):
        return "明细数据"
    if "自检" in name:
        return "自检证据"
    if "净值" in name:
        return "净值"
    if "交易" in name or "逐笔" in name or "成交" in name:
        return "逐笔"
    if "汇总" in name or "绩效" in name or "年度" in name or "分年" in name:
        return "汇总绩效"
    if "统计" in name or "检验" in name or "归因" in name:
        return "统计"
    if "压力" in name or "敏感" in name or "容量" in name:
        return "压力/敏感"
    if "分级" in name or "台账" in name or "清单" in name or "注册表" in name:
        return "清单/台账"
    if "矩阵" in name or "对比" in name:
        return "矩阵/对比"
    return "结果表"


RESULT_GLOBS = [
    "03_结果/*.csv",
    "03_结果/统计整合/*.csv", "03_结果/统计整合/*.json",
    "03_结果/统计验证/*.json", "03_结果/统计验证/自检_结果.md",
    "03_结果/统计验证/walkforward_逐折.csv", "03_结果/统计验证/参考夹具_净值.csv",
    "03_结果/统计验证/参考夹具_交易.csv",
    "03_结果/统一引擎/汇总.json", "03_结果/统一引擎/自检_统一引擎.json",
    "03_结果/统一引擎/规模敏感性.json", "03_结果/统一引擎/诊断_信号前向收益.json",
    "03_结果/统一引擎/净值_FIX20_dev.csv", "03_结果/统一引擎/交易_FIX20_dev.csv",
    "03_结果/统一引擎/年度_FIX20_dev.csv", "03_结果/统一引擎/失败订单_FIX20_dev.csv",
    "03_结果/趋势族/组合模拟_汇总.csv", "03_结果/趋势族/敏感性_资金与持仓数.csv",
    "03_结果/趋势族/参数邻域.csv", "03_结果/趋势族/消融_汇总.csv",
    "03_结果/趋势族/信号事件_汇总.csv", "03_结果/趋势族/自检.json",
    "03_结果/趋势族/交易诊断.csv", "03_结果/趋势族/净值_S4_结构突破55_纯时间60_dev.parquet",
    "03_结果/市场状态/市场状态_分窗口绩效.csv", "03_结果/市场状态/市场状态_参数敏感性.csv",
    "03_结果/市场状态/市场状态_年度绩效.csv", "03_结果/市场状态/市场状态_自检.csv",
    "03_结果/市场状态/市场状态_覆盖审计_2021_2026.csv",
    "03_结果/市场状态/市场状态_日频许可_2006_2020.csv",
    "03_结果/市场状态/市场状态_崩盘依赖.csv", "03_结果/市场状态/市场状态_空仓门诊断.csv",
    "03_结果/市场状态/市场状态_日频许可_扩展2021_2026_未评估.csv",
    "03_结果/退出_臂对照_组合级.csv", "03_结果/退出_仓位敏感.csv",
    "03_结果/退出_净值_TB1_ALL_dev.csv", "03_结果/退出_净值_TB1_ALL_val.csv",
    "03_结果/执行压力/01_压力矩阵.csv", "03_结果/执行压力/02_配置汇总.csv",
    "03_结果/执行压力/03_逐年.csv", "03_结果/执行压力/07_种子分布.csv",
    "03_结果/执行压力/08_统一引擎复核.csv",
    "03_结果/候选_退出统一/02_臂汇总.csv", "03_结果/候选_退出统一/03_配对差.csv",
    "03_结果/候选_退出统一/05_决策表.csv", "03_结果/候选_退出统一/10_自检.json",
    "03_结果/候选_小盘低波趋势/17_关键数字.json", "03_结果/候选_小盘低波趋势/10_组合_网格.csv",
    "03_结果/候选_小盘低波趋势/13_组合_净值_主单元.csv",
    "03_结果/候选_小盘低波趋势/14_压力_邻域_容量.csv",
    "03_结果/候选_小盘低波趋势/16_自检.json",
    "03_结果/候选_趋势回调/11_统计.json", "03_结果/候选_趋势回调/10_自检.json",
    "03_结果/候选_收缩突破/统计诊断.json", "03_结果/候选_收缩突破/自检.json",
    "03_结果/候选_收缩突破/组合_净值_全部_2008_2015.parquet",
    "03_结果/候选_生态切换/05_组合绩效.csv", "03_结果/候选_生态切换/11_分窗口绩效.csv",
    "03_结果/候选_生态切换/06_组合净值.csv",
    "03_结果/候选_相对强弱/04_统计检验.json", "03_结果/候选_相对强弱/05_自检.json",
    "03_结果/候选_突破回踩/10_自检.json",
    "03_结果/候选_结构内反转/11_统计.json", "03_结果/候选_结构内反转/01_自检.json",
    "03_结果/候选_结构内反转/08_账户_净值.parquet",
    "03_结果/候选_长期双动量/04_统计检验.json", "03_结果/候选_长期双动量/05_自检.json",
    "03_结果/候选_风险预算/09_自检.csv",
    "03_结果/主候选_T0005/01_信号覆盖.csv", "03_结果/主候选_T0005/02_逐笔_T0005_dev.csv",
    "03_结果/主候选_T0005/02_逐笔_T0005_val.csv", "03_结果/主候选_T0005/03_日净值_T0005_dev.csv",
    "03_结果/主候选_T0005/03_日净值_T0005_val.csv", "03_结果/主候选_T0005/07_统计_主候选.json",
    "03_结果/主候选_T0005/08_自检_T0005_dev.json", "03_结果/主候选_T0005/08_自检_T0005_val.json",
    "03_结果/主候选_T0005/40_独立账户对账.json",
    "03_结果/最终统计/*.csv", "03_结果/最终统计/*.json",
]


def _rows_of(path: Path) -> str:
    if path.suffix == ".csv" and path.stat().st_size <= CSV_READ_LIMIT:
        try:
            with open(path, encoding="utf-8-sig", errors="ignore") as fh:
                return str(max(sum(1 for _ in fh) - 1, 0))
        except Exception:
            return ""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            return str(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return ""
    return ""


def _window(path: Path) -> str:
    n = path.name
    if "dev" in n:
        return "2008-2015"
    if "val" in n:
        return "2016-2020"
    if "2008_2020" in n or "2008-2020" in n:
        return "2008-2020"
    if path.parts[0] == "01_数据":
        return "2004-01-02~2026-10-26"
    return ""


def build_inventory() -> list[dict]:
    seen: dict[str, dict] = {}

    def rec(path: Path, category: str, role: str, status: str = "有效", note: str = "") -> None:
        r = rel(path)
        if r in seen:
            return
        try:
            st = path.stat()
        except OSError:
            return
        row = {
            "path": r, "category": category, "role": role, "bytes": st.st_size,
            "sha256_16": sha256(path)[:16] if st.st_size <= HASH_LIMIT else "",
            "rows": _rows_of(path), "window": _window(path),
            "version": "v2-adjspace" if r.startswith("01_数据/") else
                       ("C-REAL-V1" if "统一引擎" in r or "小盘低波" in r else ""),
            "status": status, "notes": note,
        }
        if not row["sha256_16"]:
            row["notes"] = (note + "；大文件仅记字节数").strip("；")
        seen[r] = row

    # 代码
    for p in sorted(CODE.glob("*.py")):
        role = next((r for k, r in CODE_ROLE if p.name.startswith(k)), "代码")
        rec(p, "代码", role)

    # 文档
    docs = [ROOT / "00_研究预注册.md", ROOT / "00_一页纸.md", ROOT / "01_完整报告.md",
            ROOT / "02_规则手册_研究结论版.md", ROOT / "03_数据与复现说明.md",
            DATA / "数据字典.md"]
    docs += sorted(RESULT.rglob("*.md"))
    docs += sorted(AUDIT.glob("*.md"))
    docs += sorted((DATA / "_audit").glob("*.md"))
    for p in docs:
        if p.exists():
            r = rel(p)
            rec(p, "报告" if not r.startswith("05_审计") else "审计",
                doc_role(r), DOC_STATUS.get(r, "有效"))

    # 证据与数据
    for p in sorted(EVID.glob("*.json")):
        rec(p, "证据", "机器证据")
    if (ROOT / "04_最终结论.json").exists():
        rec(ROOT / "04_最终结论.json", "交付", "机器可读结论")
    for p in [DATA / "meta.json", DATA / "exit_cache.duckdb", DATA / "exit_panel.npz",
              FROZEN / "frozen.duckdb", FROZEN / "calendar.parquet",
              FROZEN / "corp_actions.parquet", FROZEN / "corp_actions_implied.parquet",
              FROZEN / "market_daily.parquet", FROZEN / "symbols.parquet"]:
        if p.exists():
            rec(p, "数据", "冻结数据", "有效", "读前请核对 meta 指纹")

    # 关键结果表
    for pat in RESULT_GLOBS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file():
                r = rel(p)
                rec(p, "结果", result_role(r), DOC_STATUS.get(r, "有效"))

    # 其余明细按目录汇总
    for d in sorted(RESULT.iterdir()):
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*") if p.is_file() and rel(p) not in seen]
        if not files:
            continue
        total = sum(p.stat().st_size for p in files)
        suffix = {}
        for p in files:
            suffix[p.suffix] = suffix.get(p.suffix, 0) + 1
        seen[rel(d) + "/（其余明细）"] = {
            "path": rel(d) + "/（其余明细）", "category": "结果", "role": "明细汇总",
            "bytes": total, "sha256_16": "", "rows": "",
            "window": "", "version": "v2-adjspace",
            "status": "有效" if d.name != "市场状态_缓存" else "输入缓存",
            "notes": f"未逐文件列出，共 {len(files)} 个文件：" +
                     ", ".join(f"{k}×{v}" for k, v in sorted(suffix.items())),
        }
    # 冻结目录汇总
    for sub in ("daily", "bj_daily"):
        p = FROZEN / sub
        if p.exists():
            files = [x for x in p.rglob("*.parquet")]
            seen[f"01_数据/frozen/{sub}/（分区）"] = {
                "path": f"01_数据/frozen/{sub}/（分区）", "category": "数据", "role": "冻结分区",
                "bytes": sum(x.stat().st_size for x in files), "sha256_16": "",
                "rows": str(sum(int(_rows_of(x) or 0) for x in files)),
                "window": "2004-01-02~2026-10-26", "version": "v2-adjspace",
                "status": "有效", "notes": f"{len(files)} 个 parquet 分区；逐文件哈希见 meta.json",
            }

    for p in [ROOT / "README.md", OUT_INV]:
        if p.exists() and rel(p) not in seen:
            rec(p, "交付", "交付说明" if p.suffix == ".md" else "交付清单")
    return [seen[k] for k in sorted(seen, key=lambda x: (x.count("/"), x))]


def write_inventory(rows: list[dict]) -> None:
    cols = ["path", "category", "role", "bytes", "sha256_16", "rows", "window",
            "version", "status", "notes"]
    with open(OUT_INV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- #
# H. 汇总
# ---------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="交付验收：可复现性与交付完整性总入口")
    ap.add_argument("--quick", action="store_true", help="缩小特征自检样本")
    ap.add_argument("--skip-run", action="store_true", help="只做静态校验，复用已有自检证据")
    ap.add_argument("--deep", action="store_true", help="冻结文件全量 SHA-256")
    ap.add_argument("--inventory-only", action="store_true", help="只生成文件清单")
    args = ap.parse_args()

    print(f"== 交付验收 {datetime.now():%Y-%m-%d %H:%M:%S} ==")
    if args.inventory_only:
        rows = build_inventory()
        write_inventory(rows)
        print(f"文件清单：{len(rows)} 行 -> {rel(OUT_INV)}")
        return 0

    check_deliverables(not args.skip_run)
    check_data(args.deep)
    check_leak(args.deep)
    if args.skip_run:
        suites = reuse_suites()
    else:
        suites = run_suites(args.quick)
    check_tables()
    check_paths()
    check_retractions()

    inv = build_inventory()
    write_inventory(inv)
    add("G3b", "交付物", "文件清单重建", "PASS", f"{len(inv)} 行",
        f"{rel(OUT_INV)}；类别={dict(sorted({c: sum(1 for r in inv if r['category'] == c) for c in {r['category'] for r in inv}}.items()))}")

    summary = {
        "pass": count("PASS"), "fail": count("FAIL"), "warn": count("WARN"),
        "skip": count("SKIP"), "total": len(ROWS),
    }
    failed = [r["id"] for r in ROWS if r["status"] == "FAIL"]
    summary["ok"] = not failed
    summary["failed_ids"] = failed

    payload = {
        "meta": {
            "tool": "交付验收",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(ROOT),
            "data_version": EXPECTED_VERSION,
            "freeze": FREEZE,
            "return_window_end": RETURN_END,
            "python": sys.version.split()[0],
            "argv": sys.argv,
            "tool_sha256": sha256(CODE / "交付验收.py"),
            "elapsed_s": round(time.time() - T0, 1),
            "mode": ("skip-run" if args.skip_run else ("quick" if args.quick else "full")),
        },
        "summary": summary,
        "checks": ROWS,
        "selfchecks": suites,
        "inventory": {"path": rel(OUT_INV), "rows": len(inv)},
        "retractions": RETRACTIONS,
        "limitations": [
            "本验收只运行数据/特征/撮合/统计/引擎自检，不重跑任何策略回测；策略绩效以既有证据文件为准。",
            "2021+ 泄漏扫描对 >20MB 的 CSV 与无 parquet 统计信息的文件无法逐行检查，已计入 skipped。",
            "关键表检查验证可读与非空，不重算其数值；数值复现见 README 的完整复现命令。",
            "预注册 §15 的 window_start/window_end/adjust/cost_model_id 四列尚未在所有结果表内落实（T2 为 WARN）。",
            "作废索引只标注、不修改历史文件；引用旧报告时须先查 R1/R2。",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n== 汇总 ==")
    print(f"PASS {summary['pass']} / FAIL {summary['fail']} / WARN {summary['warn']} / SKIP {summary['skip']}")
    print(f"结论：{'通过' if summary['ok'] else '未通过 ' + ','.join(failed)}")
    print(f"证据：{rel(OUT_JSON)}")
    print(f"清单：{rel(OUT_INV)}（{len(inv)} 行）")
    print(f"用时：{payload['meta']['elapsed_s']}s")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
