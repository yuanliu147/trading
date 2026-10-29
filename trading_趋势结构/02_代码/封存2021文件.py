#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2021+ 未评估文件封存扫描（只读）。

用途
----
扫描项目内含 2021-01-01 之后日期的派生状态/特征/行情缓存，登记
路径、SHA-256、行数、列、日期上界、生成代码、读取代码、是否含收益列、
是否含策略绩效列、是否被任何策略结果读取，输出机器清单：

    05_审计/2021+未评估文件封存清单.csv

纪律
----
- 只读：不删除、不移动、不改写任何被扫描文件；只写上述 CSV。
- 不读取 2021+ 数值做策略分析：只读列名、行数、日期上界与文件哈希。
- 不重跑策略；不计算任何 2021+ 收益。
- 不修改冻结层、结果、报告。

退出码
------
0  扫描完成，且没有“未登记文件”和“带策略绩效列的 2021+ 文件”
2  发现未登记文件（需人工补 manifest）
3  发现含策略绩效列的 2021+ 文件（泄漏，必须处理）
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import duckdb

PROJECT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT / "05_审计" / "2021+未评估文件封存清单.csv"
SCAN_ROOTS = ["01_数据", "02_代码", "03_结果", "04_证据", "05_审计"]
SKIP_DIRS = {"__pycache__", ".git", ".obsidian", "_tmp_spill"}
DATE_NAMES = {"date", "calendar_date", "trade_date", "dt", "日期", "年"}
RET_PAT = re.compile(
    r"^(ret1?|ew_ret1|ew_ret1_lq|med_ret1|csd_ret1|pct_chg|"
    r"r\d+|(hs300|csi500|csi1000|cyb|sse)_ret1|(hs300|csi500|csi1000|cyb|sse)_r\d+|"
    r".+_ret1|ret_.+)$"
)
PERF_PAT = re.compile(
    r"^(nav|equity|cagr|mdd|sharpe|calmar|pnl|profit|alpha|beta|ir|dsr|pbo|"
    r".*_cagr|.*_mdd|.*_sharpe|.*_nav)$"
)
PERF_NAME_HINTS = ("净值", "逐笔", "成交", "绩效", "门槛", "总账", "判定", "压力矩阵")

# 路径 -> 元数据。生成/读取位置以 2026-10-29 的代码为准。
MANIFEST = {
    "03_结果/市场状态/市场状态_日频许可_扩展2021_2026_未评估.csv": dict(
        tier="A_禁止读取", category="派生状态序列（state/pos）",
        generator="02_代码/策略族_市场状态.py:943-952（stage_audit；规则冻结后前向应用）",
        readers="02_代码/交付验收.py:267,1066（白名单/泄漏扫描，不进入绩效）；"
                "02_代码/最终试验总账.py:1590（明确声明未载入）",
        read_scope="仅审计扫描读取，未进入任何策略/绩效表",
        used_by_results="否",
        action="只读保留，不删除；列入禁止读取清单；留出运行前不得查看",
        note="1,406 行 2021-01-04..2026-10-26 的 state/B1-B5/P/gate/pos_A-D；无收益列。"
             "MKT 规则是 state→仓位，与 2021+ 行情 join 即可间接评估 MKT 族留出表现。"
             "同名 .rows 边车（内容 1406）随主文件一并封存。",
    ),
    "03_结果/市场状态_缓存/index_feat.parquet": dict(
        tier="A_禁止读取", category="特征缓存（含指数收益列）",
        generator="02_代码/策略族_市场状态.py:287",
        readers="无（全项目仅 1 处代码引用，为写出）",
        read_scope="未被任何脚本读取", used_by_results="否",
        action="只读保留；禁止用于 2021+ 对照/选参",
        note="44,162 行；收益列 ret1/r20/r60/r120；无策略绩效列。",
    ),
    "03_结果/市场状态_缓存/market_feature.parquet": dict(
        tier="A_禁止读取", category="特征缓存（含市场收益列）",
        generator="02_代码/策略族_市场状态.py:290-315",
        readers="02_代码/策略族_市场状态.py:496,796,851,914,943（MKT 族）",
        read_scope="MKT 族读取；绩效评估在 EVAL_END=2020-12-31 截断"
                   "（策略族_市场状态.py:56,497,797,915）；2021+ 行仅用于 stage_audit 状态外推",
        used_by_results="是（仅 ≤2020 切片进入 MKT 结果；2021+ 行未进入任何绩效表）",
        action="只读保留；2021+ 行禁止用于选参/收益计算",
        note="5,540 行；ew_ret1/med_ret1/csd_ret1/ew_ret1_lq 及 5 个指数 ret1；无策略绩效列。",
    ),
    "03_结果/市场状态_缓存/panel_hfq.parquet": dict(
        tier="B_只读封存", category="行情面板缓存（后复权）",
        generator="02_代码/策略族_市场状态.py:99-110",
        readers="02_代码/策略族_市场状态.py:249,905,956",
        read_scope="MKT 族评估 ≤2020；stage_audit 仅用 2021+ 做覆盖计数（不计算收益）",
        used_by_results="是（仅 ≤2020 切片）；2021+ 行仅覆盖计数",
        action="只读保留；2021+ 行禁止用于收益/选参",
        note="16,764,215 行；含 OHLC/成交/停牌/ST；无收益列。",
    ),
    "03_结果/市场状态_缓存/index_daily.parquet": dict(
        tier="B_只读封存", category="指数行情缓存",
        generator="02_代码/策略族_市场状态.py:112-121",
        readers="02_代码/策略族_市场状态.py（MKT 族）",
        read_scope="MKT 族评估 ≤2020；2021+ 仅覆盖审计",
        used_by_results="是（仅 ≤2020 切片）",
        action="只读保留；2021+ 行禁止用于收益/选参",
        note="44,162 行；含 pct_chg；无策略绩效列。",
    ),
    "03_结果/市场状态_缓存/calendar.parquet": dict(
        tier="B_只读封存", category="交易日历缓存",
        generator="02_代码/策略族_市场状态.py:132-140",
        readers="02_代码/策略族_市场状态.py（MKT 族）",
        read_scope="日历；不产生收益", used_by_results="否",
        action="只读保留", note="8,427 行；仅日期与是否交易日。",
    ),
    "03_结果/市场状态_缓存/stock_basic.parquet": dict(
        tier="B_只读封存", category="证券主数据（含退市日）",
        generator="02_代码/策略族_市场状态.py:124-131",
        readers="02_代码/策略族_市场状态.py（MKT 族）",
        read_scope="元数据（ipo/out_date）；不产生收益", used_by_results="否",
        action="只读保留", note="5,564 行；含 2021+ 退市日，属元数据。",
    ),
    "03_结果/_审计缓存/bs_window_all_flags.parquet": dict(
        tier="B_只读封存", category="审计标记缓存（原始三复权 + 停牌/ST）",
        generator="02_代码/审计双源.py:53（可重建）",
        readers="02_代码/审计双源.py；05_审计/最终结论证据矩阵_复算.py:140",
        read_scope="仅审计；未进入策略结果", used_by_results="否",
        action="只读保留；禁止用于选参",
        note="31,769,553 行；含 OHLC/成交/停牌/ST；无收益列。",
    ),
    "05_审计/双源数据核验明细.csv": dict(
        tier="B_只读封存", category="审计明细（双源价格对照）",
        generator="02_代码/审计双源.py:50,908",
        readers="无（审计输出，供报告引用）", read_scope="仅审计",
        used_by_results="否", action="只读保留；禁止用于选参",
        note="7,857 行；含 2021+ 价格对照 diff；无策略绩效列。",
    ),
    "03_结果/市场状态/市场状态_覆盖审计_2021_2026.csv": dict(
        tier="B_只读封存", category="覆盖审计计数（按年）",
        generator="02_代码/策略族_市场状态.py:958-964",
        readers="无（审计输出）", read_scope="仅覆盖计数，不含收益",
        used_by_results="否", action="只读保留",
        note="6 行（2021-2026）；行数/股票数/停牌/ST 计数。",
    ),
    "01_数据/exit_cache.duckdb": dict(
        tier="C_数据源", category="原始行情缓存（退出族/相对强弱）",
        generator="02_代码/退出_数据层.py:21（建库）；表 px_hfq/px_raw/idx/cal/basic",
        readers="02_代码/退出_数据层.py；02_代码/候选_相对强弱.py:1278",
        read_scope="数据源；两类族的分析窗口 ≤2020；2021+ 行未进入结果",
        used_by_results="是（仅 ≤2020 切片）",
        action="只读保留；2021+ 行禁止用于收益/选参",
        note="5 张表：px_hfq 16,367,202 行、px_raw 16,377,690 行、idx 44,162 行、"
             "cal/basic 元数据；max date 2026-10-26；无收益/绩效列。",
    ),
    "01_数据/frozen/": dict(
        tier="C_数据源", category="冻结行情数据源（原始）",
        generator="02_代码/构建冻结数据.py（freeze=2026-10-26）",
        readers="全部策略脚本（统一在 2021-01-01 前截断；见 02_代码/统一数据适配.py:147,292 等）",
        read_scope="数据源；策略结果 ≤2020；2021+ 仅数据覆盖审计",
        used_by_results="是（仅 ≤2020 切片）",
        action="不封存（数据源）；逐文件 SHA-256 见 01_数据/meta.json",
        note="56 个 parquet + frozen.duckdb；行数取 daily 表 16,698,536；"
             "max date 2026-10-26。逐文件哈希已登记在 01_数据/meta.json。",
    ),
}

# 2021+ 日期字段的元数据/自扫描文件：不是状态/特征缓存，不封存，但登记排除原因。
EXCLUDED_METADATA = {
    "03_结果/实验注册表.csv": "注册表元数据（registered_at_utc 为登记时间），无收益/状态",
    "03_结果/统计整合/2021后日期扫描.csv": "旧扫描输出自身（14 行，已过期；本清单取代其口径）",
    "03_结果/候选_生态切换/07_试验清单.csv": "试验清单元数据（generated_at），无绩效",
    "03_结果/统计验证/数据覆盖_逐年.csv": "逐年数据覆盖计数（first/last_date），无收益",
    "03_结果/趋势族/数据审计_年度覆盖.csv": "逐年数据覆盖计数（first/last_day），无收益",
}

# 日期在 out_date/ipo_date 而非 date 列的元数据：强制登记。
FORCE_2021 = {
    "03_结果/市场状态_缓存/stock_basic.parquet": ("out_date", "1990-12-19", "2026-10-28"),
}


RAW_GENERATORS = {
    "01_数据/_tmp/raw/": ("02_代码/构建冻结数据.py:160（Baostock 不复权分区落盘）",
                          "02_代码/构建冻结数据.py:217,243（生成冻结层）",
                          "否（构建冻结层的数据源；策略结果 ≤2020）",
                          "构建冻结层"),
    "01_数据/_tmp/bs_hfq/": ("02_代码/构建冻结数据.py:180（Baostock adjustflag=1 分区落盘）",
                             "02_代码/构建冻结数据.py:243（生成冻结层）",
                             "否（构建冻结层的数据源；策略结果 ≤2020）",
                             "构建冻结层"),
    "02_代码/cache/bs_": ("02_代码/统计验证.py:334（按年缓存 Baostock 截面）",
                          "02_代码/统计验证.py（按年读取；绩效入口 guard_performance 对 2021+ 抛错）",
                          "否（统计验证按年读取；2021+ 年份缓存未进入绩效）",
                          "统计验证；绩效入口对 2021+ 抛错"),
    "02_代码/_cache/kline_raw.parquet": ("02_代码/策略族_结构.py:122-150（COPY，date ≤ FREEZE）",
                                         "02_代码/策略族_结构.py:230",
                                         "是（仅 ≤2020 切片）",
                                         "结构族读取；面板在 2021-01-01 截断（策略族_结构.py:1449）"),
    "02_代码/_cache/kline_hfq.parquet": ("02_代码/策略族_结构.py:122-150（COPY，date ≤ FREEZE）",
                                         "02_代码/策略族_结构.py:229",
                                         "是（仅 ≤2020 切片）",
                                         "结构族读取；面板在 2021-01-01 截断（策略族_结构.py:1449）"),
    "02_代码/_cache/index_kline.parquet": ("02_代码/策略族_结构.py:126,152", "02_代码/策略族_结构.py:201",
                                            "是（仅 ≤2020 切片）", "结构族读取；指数仅作基准"),
    "02_代码/_cache/trade_dates.parquet": ("02_代码/策略族_结构.py:127,155", "02_代码/策略族_结构.py:202；02_代码/审计第一阶段.py:60",
                                            "是（仅 ≤2020 切片）", "日历；结构族读取"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _maxmin_from_stats(pf: pq.ParquetFile, names: list[str]):
    dcol = next((c for c in names if c.lower() in DATE_NAMES), None)
    if dcol is None:
        return None, None, None
    ci = names.index(dcol)
    lo = hi = None
    for rg in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(rg).column(ci).statistics
        if not st or st.min is None or st.max is None:
            continue
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    return dcol, lo, hi


def _norm_date(x) -> str | None:
    if x is None:
        return None
    s = str(x)[:10]
    return s if len(s) == 10 else None


def inspect(path: Path):
    rec = dict(size_mb=round(path.stat().st_size / 1e6, 3))
    if path.suffix.lower() == ".parquet":
        pf = pq.ParquetFile(path)
        names = list(pf.schema_arrow.names)
        rec.update(rows=pf.metadata.num_rows, cols=len(names), columns=names)
        dcol, lo, hi = _maxmin_from_stats(pf, names)
        rec.update(date_col=dcol, min_date=_norm_date(lo), max_date=_norm_date(hi))
    elif path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh))
        names = [h.strip() for h in header]
        idx = [i for i, c in enumerate(names) if c.lower() in DATE_NAMES]
        dcol = names[idx[0]] if idx else None
        if idx:
            s = pd.read_csv(path, usecols=[names[i] for i in idx], engine="python",
                            on_bad_lines="skip")
            if dcol == "年":
                yrs = pd.to_numeric(s[dcol], errors="coerce").dropna()
                lo = f"{int(yrs.min())}-01-01" if len(yrs) else None
                if len(yrs) and int(yrs.max()) >= 2026:
                    hi = "2026-10-26"
                else:
                    hi = f"{int(yrs.max())}-12-31" if len(yrs) else None
            else:
                vals = [_norm_date(v) for col in s.columns for v in s[col].tolist()]
                vals = [v for v in vals if v]
                lo, hi = (min(vals), max(vals)) if vals else (None, None)
        else:
            s, lo, hi = None, None, None
        rec.update(rows=(len(s) if s is not None else None), cols=len(names),
                   columns=names, date_col=dcol, min_date=lo, max_date=hi)
    elif path.suffix.lower() == ".duckdb":
        con = duckdb.connect(str(path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("show tables").fetchall()]
            cols, rows, lo, hi = [], 0, None, None
            for t in tables:
                desc = con.execute(f'describe "{t}"').fetchall()
                tcols = [r[0] for r in desc]
                cols += [f"{t}.{c}" for c in tcols]
                rows += int(con.execute(f'select count(*) from "{t}"').fetchone()[0])
                dcol = next((c for c in tcols if c.lower() in DATE_NAMES), None)
                if dcol:
                    a, b = con.execute(
                        f'select min("{dcol}"), max("{dcol}") from "{t}"').fetchone()
                    lo = a if lo is None else min(lo, a)
                    hi = b if hi is None else max(hi, b)
        finally:
            con.close()
        rec.update(rows=rows, cols=len(cols), columns=cols, date_col="date",
                   min_date=_norm_date(lo), max_date=_norm_date(hi))
    else:
        rec.update(rows=None, cols=None, columns=[], date_col=None,
                   min_date=None, max_date=None)
    return rec


def refine(rec: dict, rel: str, path: Path):
    names = rec["columns"] or []
    rets = sorted({c for c in names if RET_PAT.match(str(c).lower())})
    perfs = sorted({c for c in names if PERF_PAT.match(str(c).lower())})
    rec["ret_cols"] = rets
    rec["perf_cols"] = perfs
    rec["has_return"] = "是" if rets else "否"
    # 文件名像绩效表 + 有收益/绩效列，才判为“含策略绩效”
    name_hint = any(k in path.name for k in PERF_NAME_HINTS)
    rec["has_perf"] = "是" if (perfs or (name_hint and rets)) else "否"

    meta = MANIFEST.get(rel)
    if meta:
        rec.update(meta)
        return rec

    for prefix, (gen, readers, used, scope) in RAW_GENERATORS.items():
        if rel.startswith(prefix) or rel == prefix:
            rec.update(
                tier="C_数据源", category="原始行情缓存",
                generator=gen, readers=readers,
                read_scope=scope, used_by_results=used,
                action="只读保留；2021+ 行禁止用于收益/选参",
                note="原始行情缓存，可由源库重建；不属于派生状态/特征。",
            )
            return rec

    rec.update(tier="X_未登记", category="未登记（需人工复核）",
               generator="未知", readers="未知", read_scope="未知",
               used_by_results="未知", action="暂停使用，补登记后再处理",
               note="扫描发现但 manifest 未登记；请补 02_代码/封存2021文件.py。")
    return rec


def main() -> int:
    rows, unknown, leaks, excluded = [], [], [], []
    for root in SCAN_ROOTS:
        base = PROJECT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet", ".duckdb"}:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            rel = path.relative_to(PROJECT).as_posix()
            if rel.startswith("01_数据/frozen/"):
                continue  # 冻结层按目录登记一行，逐文件哈希见 01_数据/meta.json
            if rel in EXCLUDED_METADATA:
                excluded.append((rel, EXCLUDED_METADATA[rel]))
                continue
            rel_key = rel
            try:
                rec = inspect(path)
            except Exception as exc:  # 损坏/无权限文件不应静默跳过
                print(f"[WARN] 无法读取 {rel}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if rel in FORCE_2021:
                dcol, lo, hi = FORCE_2021[rel]
                rec.update(date_col=dcol, min_date=lo, max_date=hi)
            mx = rec.get("max_date")
            if not mx or mx <= "2020-12-31":
                continue
            rec = refine(rec, rel_key, path)
            rec["path"] = rel_key
            rec["sha256"] = sha256(path)
            rows.append(rec)

    # 冻结层单行
    fz = PROJECT / "01_数据" / "frozen" / "frozen.duckdb"
    if fz.exists():
        meta = MANIFEST["01_数据/frozen/"]
        rows.append(dict(path="01_数据/frozen/", size_mb=round(fz.stat().st_size / 1e6, 3),
                         rows=16698536, cols=None, columns=["daily(16,698,536 行) + 55 个 parquet"],
                         date_col="date", min_date="2004-01-02", max_date="2026-10-26",
                         ret_cols=["daily.pct_chg"], perf_cols=[], has_return="是", has_perf="否",
                         sha256=sha256(fz), **meta))

    tier_order = {"A_禁止读取": 0, "B_只读封存": 1, "C_数据源": 2, "X_未登记": 9}
    rows.sort(key=lambda r: (tier_order.get(r["tier"], 8), r["path"]))

    cols = ["序号", "封存等级", "类别", "路径", "文件大小MB", "行数", "列数", "列名",
            "日期列", "最早日期", "最晚日期", "SHA256", "含收益列", "收益列清单",
            "含策略绩效列", "生成代码", "读取代码", "读取范围", "是否被策略结果读取",
            "封存动作", "禁止用途", "备注"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(rows, 1):
            w.writerow([
                i, r["tier"], r["category"], r["path"], r["size_mb"],
                r["rows"] if r["rows"] is not None else "",
                r["cols"] if r["cols"] is not None else "",
                "|".join(map(str, r["columns"]))[:4000],
                r["date_col"] or "", r["min_date"] or "", r["max_date"] or "",
                r["sha256"], r["has_return"], "|".join(r["ret_cols"]),
                r["has_perf"], r["generator"], r["readers"], r["read_scope"],
                r["used_by_results"], r["action"],
                "禁止用于选参/收益计算/策略分析；2021+ 行只允许元数据核对" if r["tier"].startswith(("A", "B", "C")) else "暂停使用",
                r["note"],
            ])
            if r["tier"] == "X_未登记":
                unknown.append(r["path"])
            if r["has_perf"] == "是":
                leaks.append(r["path"])

    print(f"扫描根：{', '.join(SCAN_ROOTS)}")
    print(f"2021+ 文件：{len(rows)} 个 "
          f"(A={sum(r['tier'].startswith('A') for r in rows)}, "
          f"B={sum(r['tier'].startswith('B') for r in rows)}, "
          f"C={sum(r['tier'].startswith('C') for r in rows)}, "
          f"未登记={len(unknown)})")
    print(f"含收益列：{sum(r['has_return'] == '是' for r in rows)} 个；"
          f"含策略绩效列：{len(leaks)} 个")
    print(f"排除元数据（非缓存，无收益/状态）：{len(excluded)} 个")
    for p, why in excluded:
        print(f"  - {p}：{why}")
    print(f"清单：{OUT_CSV.relative_to(PROJECT)}")
    if unknown:
        print("[FAIL] 未登记文件：" + ", ".join(unknown), file=sys.stderr)
        return 2
    if leaks:
        print("[FAIL] 含策略绩效列的 2021+ 文件：" + ", ".join(leaks), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
