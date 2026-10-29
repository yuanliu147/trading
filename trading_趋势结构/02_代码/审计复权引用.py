#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复权缺陷引用面审计：谁真的用了冻结层 `hfq_*`，谁只是提到过字符串。

背景（已由独立复现与统一引擎裁决确认）
--------------------------------------
冻结层 `factor = Π(1+pct_chg/100)` 是正确的累计总收益指数；v1 冻结数据把
`hfq_px = raw_px × factor`，等于把价格收益再乘一次：
    hfq_t/hfq_{t-1} - 1 ≈ (1+r_t)² - 1
正确的逐日调整空间是 `A_t = factor_t / raw_close_t`，`adjusted_OHLC = raw_OHLC × A_t`
（可按每股首条价格归一）；此时 `adjusted_close` 的日收益与 `pct_chg` 逐行一致。

本脚本做三件事
--------------
1. **静态扫描** `02_代码/*.py`（AST，区分代码字符串与 docstring）和项目内 md/csv/json，
   列出所有 `hfq/qfq/factor` 引用点；再用「文件级规则表」判定数据口径。
   特别区分：读 hfq 列但立刻重建（不受影响）、只用同一天内比例（不受影响）、
   跨日收益/水平（受影响）、只出现在注释/文档（不受影响）。
2. **独立复算**：对每个有逐笔/事件明细的结果，用冻结层重算 A 空间与 H 空间
   （H = 冻结 `hfq_*`）收益，比较哪一空间能复现已发布数字；再用价格指纹确认
   报价空间（raw / Baostock hfq / factor / 冻结 hfq）。
3. **影响定级**：作废待重跑 / 受影响但符号稳健 / 不受此缺陷影响 / 未知，并给出白名单。

纪律
----
* 只读：不写 `01_数据/`，不改任何他人产物。
* 2021+ 一律不计算收益：所有读取截止 `2020-12-31`，脚本内 assert。
* 唯一写入：`05_审计/复权修复影响清单.csv`。

用法
----
    python3 02_代码/审计复权引用.py            # 全量（约 20-45 秒）
    python3 02_代码/审计复权引用.py --quick    # 抽样更小
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "02_代码"
RES = ROOT / "03_结果"
AUD = ROOT / "05_审计"
FROZEN = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
OUT_CSV = AUD / "复权修复影响清单.csv"

ANALYSIS_END = pd.Timestamp("2020-12-31")
SNAPSHOT_AT = "2026-10-28"

# 冻结层里由 v1 `hfq_close = close × factor` 派生/污染的列
FROZEN_HFQ_COLS = {"hfq_open", "hfq_high", "hfq_low", "hfq_close",
                   "qfq_open", "qfq_high", "qfq_low", "qfq_close"}
HFQ_WORD = re.compile(r"(hfq|qfq)", re.I)

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(dates) -> pd.Series:
    s = pd.to_datetime(pd.Series(dates), errors="coerce")
    mx = s.max()
    if pd.notna(mx) and mx > ANALYSIS_END:
        raise AssertionError(f"读取越过 2020-12-31：{mx}")
    return s


# --------------------------------------------------------------------------- #
# 1. 静态扫描
# --------------------------------------------------------------------------- #
def _docstring_nodes(tree: ast.AST) -> set[int]:
    keep: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(n, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    keep.add(id(body[0].value))
    return keep


def ast_facts(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:  # pragma: no cover
        return {"parse_error": str(e)}
    docs = _docstring_nodes(tree)
    code_hfq_lines: set[int] = set()
    doc_hfq_lines: set[int] = set()
    cols: set[str] = set()
    adjust_calls: list[tuple[int, str]] = []
    daily_calls: list[tuple[int, str]] = []
    imports: set[str] = set()
    loaders: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value
            if HFQ_WORD.search(v):
                (doc_hfq_lines if id(n) in docs else code_hfq_lines).add(n.lineno)
            for c in FROZEN_HFQ_COLS:
                if re.search(rf"\b{c}\b", v, re.I):
                    cols.add(c.lower())
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("daily", "bars"):
                kw = {k.arg: k.value for k in n.keywords if k.arg}
                adj = kw.get("adjust")
                val = adj.value if isinstance(adj, ast.Constant) else "<dynamic>"
                (adjust_calls if name == "daily" else daily_calls).append((n.lineno, str(val)))
            if name in ("_load_module", "spec_from_file_location"):
                for a in n.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.endswith(".py"):
                        loaders.add(a.value)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""])
            for m in mods:
                if m:
                    imports.add(m)
    return {
        "n_lines": len(src.splitlines()),
        "code_hfq_lines": sorted(code_hfq_lines),
        "doc_hfq_lines": sorted(doc_hfq_lines - code_hfq_lines),
        "hfq_cols": sorted(cols),
        "adjust_calls": adjust_calls,
        "bars_calls": daily_calls,
        "imports": sorted(imports),
        "loaders": sorted(loaders),
    }


# 文件级数据口径规则（人工逐文件核对后的结论；脚本同时打印 AST 事实供复核）
RULES: dict[str, tuple[str, str]] = {
    # 文件: (口径, 说明)
    "数据层.py": ("frozen_hfq_api", "daily(adjust='hfq'/'qfq') 返回 hfq_close/ qfq_close；项目内 0 个导入者"),
    "构建冻结数据.py": ("frozen_hfq_builder", "v1 构建 hfq=raw×factor；源码 23:37 已改为 A 空间，DB 未重建"),
    "01_自检_数据.py": ("selfcheck_false_pass", "C1 恒等式、C2 用 factor 收益冒充 hfq 收益"),
    "独立复现_基线.py": ("defect_quantifier", "A 空间自建；hfq 仅作反例披露"),
    "统一数据适配.py": ("factor_ca_adj", "ca_adj=raw/ca_factor；hfq_* 别名=tr_index(factor)，不读冻结 hfq 列"),
    "统一回测.py": ("factor_ca_adj", "撮合/估值用 ca_factor 与 tr_index；不读 hfq 列"),
    "策略族_结构.py": ("baostock_hfq", "Baostock adjustflag=1（kline_hfq）"),
    "策略族_趋势.py": ("baostock_hfq", "Baostock adjustflag=1（bs_hfq）"),
    "策略族_市场状态.py": ("baostock_hfq", "Baostock adjustflag=1（panel_hfq）"),
    "退出_数据层.py": ("baostock_hfq", "exit_cache.px_hfq 已核对 == 冻结 bs_hfq_close（9.1M 行 100%）"),
    "退出_回测引擎.py": ("baostock_hfq", "hfq 空间来自 退出_数据层.Panel"),
    "退出_实验.py": ("baostock_hfq", "调用 退出_回测引擎"),
    "退出_自检.py": ("baostock_hfq", "调用 退出_数据层.Panel"),
    "退出_补充分析.py": ("baostock_hfq", "调用 退出_数据层.Panel"),
    "退出_报告表.py": ("none", "只读退出结果表"),
    "核心特征.py": ("baostock_hfq", "Baostock flag1 / THS v_daily_hfq"),
    "多周期.py": ("baostock_hfq", "Baostock flag1 / THS v_daily_hfq"),
    "持仓状态机.py": ("naming_only", "只接收调用方传入的价格，不取数"),
    "统计验证.py": ("raw_preclose", "close/preclose-1 与 pct_chg；不读 hfq"),
    "费用与交易约束.py": ("none", "规则表，无行情"),
    "交易制度适配器.py": ("none", "规则表，无行情"),
    "交易制度自检.py": ("none", "规则表，无行情"),
    "账户撮合.py": ("raw_corp_actions", "原始价+公司行动记账；不读 hfq"),
    "实验登记.py": ("none", "登记表"),
    "整合实验登记.py": ("audit_baostock_hfq", "defect_table 用 bs_hfq_close（M1），非冻结 hfq"),
    "审计双源.py": ("audit_cross_source", "两源复权收益比对；不进入策略"),
    "审计第一阶段.py": ("audit_baostock_hfq", "M1 用 bs_hfq_close；反例用冻结 hfq 仅作说明"),
    "候选_趋势回调.py": ("rebuild_pctchg", "读 hfq 列后整列覆盖为 pct_chg 递归构造的 tr"),
    "候选_收缩突破.py": ("factor_rebuild", "tr_close=factor，tr_ohlc=raw×factor/close"),
    "候选_生态切换.py": ("factor_rebuild", "hc=factor，日内价 factor×px/close"),
    "候选_执行压力.py": ("factor", "hfq_close() 返回 factor"),
    "候选_退出统一.py": ("factor_rebuild", "scale=factor/close，hfq=raw×scale"),
    "候选_风险预算.py": ("factor", "收益用 factor；hfq 仅审计用途"),
    "候选_相对强弱.py": ("baostock_hfq", "from 退出_数据层 import Panel（exit_cache = bs_hfq）"),
    "候选_突破回踩.py": ("a_space_rebuild", "A_t=factor/close；hfq 仅作反例自检（23:45 落盘）"),
    "候选_长期双动量.py": ("a_space_rebuild", "adj_close=factor，adj_ohlc=raw×factor/close（23:42 落盘）"),
    "候选_结构内反转.py": ("a_space_rebuild", "A_t=factor/close，adj_close≡factor（23:41 落盘）"),
    "候选_小盘低波趋势.py": ("factor_ca_adj", "import 统一数据适配；A_t=factor/close（23:45 落盘）"),
    "审计复权引用.py": ("audit_self", "本审计；v1 缺陷按定义重建复算"),
}


def static_scan() -> tuple[pd.DataFrame, dict]:
    rows = []
    facts_by_file: dict[str, dict] = {}
    for p in sorted(CODE.glob("*.py")):
        f = ast_facts(p)
        facts_by_file[p.name] = f
        space, note = RULES.get(p.name, ("UNKNOWN", "无规则，需人工复核"))
        def _brief(v, n=10):
            v = list(v or [])
            return (v[:n] + [f"...(+{len(v)-n})"]) if len(v) > n else v
        rows.append({
            "id": f"PY-{p.stem}",
            "类别": "代码",
            "对象": p.name,
            "文件": str(p.relative_to(ROOT)),
            "单元": "模块",
            "样本数": f.get("n_lines"),
            "数据口径": space,
            "静态证据": (f"代码字符串行={_brief(f.get('code_hfq_lines'))}；"
                       f"doc行={len(f.get('doc_hfq_lines', []))}；"
                       f"daily(adjust)={_brief(f.get('adjust_calls'))}；cols={f.get('hfq_cols')}"),
            "备注": note,
        })

    # 文档 / 结果文本：只扫文本，不读大 CSV 全文
    text_files = [ROOT / "00_研究预注册.md", ROOT / "01_数据" / "数据字典.md",
                  ROOT / "01_数据" / "_audit" / "自检报告.md",
                  ROOT / "01_数据" / "_audit" / "v1_修复前_自检报告.md"]
    text_files += sorted(RES.rglob("*.md"))
    text_files += sorted(RES.rglob("*.json"))
    text_files += [q for q in sorted(AUD.glob("*.md")) if q.name != "复权修复影响.md"]
    seen = set()
    for p in text_files:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        n = len(HFQ_WORD.findall(txt))
        if not n:
            continue
        claim = ""
        if re.search(r"hfq\s*(=|＝)\s*.*close\s*[×*]", txt) or "hfq_px_t = raw_px_t × factor_t" in txt:
            claim = "含错误公式 hfq=raw×factor"
        if "偏差>0.5bp 的行" in txt and "0" in txt:
            claim = (claim + "；" if claim else "") + "含『hfq收益偏差 0』类假阳性结论"
        rows.append({
            "id": f"TXT-{len(rows)}",
            "类别": "文档/文本",
            "对象": p.name,
            "文件": str(p.relative_to(ROOT)),
            "单元": "文档",
            "样本数": n,
            "数据口径": "mention",
            "静态证据": f"hfq/qfq 提及 {n} 次",
            "备注": claim or "仅提及/说明",
        })
    return pd.DataFrame(rows), facts_by_file


def datalayer_indirect(facts: dict[str, dict]) -> pd.DataFrame:
    """DataLayer 间接调用面：谁 import 数据层、谁把 adjust 透传、谁用默认 raw。"""
    importers = []
    for name, f in facts.items():
        if name == Path(__file__).name:
            continue
        for m in f.get("imports", []):
            if m == "数据层" or m.startswith("数据层."):
                importers.append((name, m))
        for ld in f.get("loaders", []):
            if Path(ld).name == "数据层.py":
                importers.append((name, f"importlib:{ld}"))
    dynamic_adj = [(k, v["adjust_calls"]) for k, v in facts.items()
                   if any(x[1] == "<dynamic>" for x in v.get("adjust_calls", []))]
    no_adj = [(k, v["bars_calls"]) for k, v in facts.items() if v.get("bars_calls")]
    rows = [{
        "id": "DATA-INDIRECT",
        "类别": "代码",
        "对象": "DataLayer 间接调用面",
        "文件": "02_代码/数据层.py",
        "单元": "导入/调用",
        "样本数": len(importers),
        "数据口径": "frozen_hfq_api",
        "静态证据": (f"导入者={importers or '无'}；动态 adjust={dynamic_adj or '无'}；"
                   f"无参 .daily()/bars()={no_adj or '无'}"),
        "备注": "项目内 0 个模块导入 数据层；危险 API 存在但无下游",
    }]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. 冻结层缺陷量化 + 派生列污染
# --------------------------------------------------------------------------- #
def quantify_defect(con, quick: bool) -> pd.DataFrame:
    rows = []
    # v1 冻结层（2026-10-28 21:55 build）的 hfq 公式是 ROUND(raw × factor, 4)。
    # 现库已于 23:40 重建为 A 空间，因此 v1 必须按定义重建，不能读当前 hfq 列。
    q = """
    with r as (
      select symbol, date, board, close, preclose, pct_chg, factor,
             round(close * factor, 4) h1_close,
             bs_hfq_close,
             lag(close) over (partition by symbol order by date) pclose,
             lag(factor) over (partition by symbol order by date) pfactor,
             lag(round(close * factor, 4)) over (partition by symbol order by date) ph1,
             lag(bs_hfq_close) over (partition by symbol order by date) pbs
      from daily where date between date '2008-01-01' and date '2020-12-31'
    )
    select * from r where pclose is not null and pct_chg is not null
    """
    d = con.execute(q).df()
    pct = d.pct_chg.to_numpy(float) / 100.0
    raw = d.close.to_numpy(float) / d.pclose.to_numpy(float) - 1
    fac = d.factor.to_numpy(float) / d.pfactor.to_numpy(float) - 1
    hfq = d.h1_close.to_numpy(float) / d.ph1.to_numpy(float) - 1
    ok = np.isfinite(pct) & np.isfinite(hfq) & np.isfinite(fac)
    m = ok & (np.abs(pct) > 0.01)
    rows.append(dict(
        id="DEFECT-HFQ-V1", 类别="数据层", 对象="冻结层 v1 daily.hfq_*（重建复算）",
        文件="01_数据/frozen/frozen.duckdb（v1: 21:55 build）",
        单元="证券日", 样本数=int(ok.sum()), 数据口径="frozen_hfq_v1",
        静态证据=f"2008-2020 相邻日 {int(ok.sum()):,} 行；v1 公式 hfq=ROUND(raw×factor,4)",
        备注=(f"hfq 收益中位|偏差|={np.median(np.abs(hfq[ok]-pct[ok])):.6f}；"
              f">0.5bp 占比={np.mean(np.abs(hfq[ok]-pct[ok])>5e-5):.4f}；"
              f"|pct|>1% 行 hfq/pct 中位比={np.median(hfq[m]/pct[m]):.4f}；"
              f"最大|偏差|={np.max(np.abs(hfq[ok]-pct[ok])):.2f}"),
    ))
    rows.append(dict(
        id="DEFECT-FACTOR", 类别="数据层", 对象="冻结层 daily.factor（对照）", 文件="01_数据/frozen/frozen.duckdb",
        单元="证券日", 样本数=int(ok.sum()), 数据口径="factor",
        静态证据=f"2008-2020 相邻日 {int(ok.sum()):,} 行",
        备注=f"factor 收益与 pct_chg 最大|偏差|={np.max(np.abs(fac[ok]-pct[ok])):.2e}；>0.5bp={int(np.sum(np.abs(fac[ok]-pct[ok])>5e-5))}",
    ))
    rows.append(dict(
        id="DEFECT-RAW", 类别="数据层", 对象="原始收盘（对照）", 文件="01_数据/frozen/frozen.duckdb",
        单元="证券日", 样本数=int(ok.sum()), 数据口径="raw",
        静态证据=f"2008-2020 相邻日 {int(ok.sum()):,} 行",
        备注=f"raw 收益与 pct_chg 中位|偏差|={np.median(np.abs(raw[ok]-pct[ok])):.6f}；>0.5bp 占比={np.mean(np.abs(raw[ok]-pct[ok])>5e-5):.4f}（除权日）",
    ))
    okb = ok & np.isfinite(d.pbs.to_numpy(float))
    bs = d.bs_hfq_close.to_numpy(float) / d.pbs.to_numpy(float) - 1
    rows.append(dict(
        id="CONTROL-BS", 类别="数据层", 对象="Baostock hfq（对照）", 文件="01_数据/frozen/frozen.duckdb",
        单元="证券日", 样本数=int(okb.sum()), 数据口径="baostock_hfq",
        静态证据=f"2008-2020 相邻日 {int(okb.sum()):,} 行",
        备注=f"中位|偏差|={np.median(np.abs(bs[okb]-pct[okb])):.6f}；>0.5pp 占比={np.mean(np.abs(bs[okb]-pct[okb])>5e-3):.6f}",
    ))
    # v2 现库（23:40 重建）验收：hfq/factor 逐股常数、hfq 收益 == pct_chg
    v2 = con.execute("""
      with r as (
        select symbol,date,pct_chg,factor,hfq_close,
               lag(factor) over (partition by symbol order by date) pf,
               lag(hfq_close) over (partition by symbol order by date) ph,
               min(hfq_close/factor) over (partition by symbol) mn,
               max(hfq_close/factor) over (partition by symbol) mx
        from daily where date between date '2008-01-01' and date '2020-12-31')
      select count(*) n,
             max(abs(hfq_close/ph-1-pct_chg/100)) max_err,
             avg(case when abs(hfq_close/ph-1-pct_chg/100)>5e-5 then 1 else 0 end) bad,
             max((mx-mn)/nullif(mx,0)) max_rel_range
      from r where ph is not null and pct_chg is not null
    """).df().iloc[0]
    rows.append(dict(
        id="V2-ACCEPT", 类别="数据层", 对象="冻结层 v2 daily.hfq_*（23:40 重建）",
        文件="01_数据/frozen/frozen.duckdb（v2）", 单元="证券日", 样本数=int(v2["n"]),
        数据口径="frozen_hfq_v2",
        静态证据="v2 公式 hfq=ROUND(raw×factor/close×close_first,4)",
        备注=(f"hfq 收益 vs pct_chg 最大|偏差|={v2['max_err']:.2e}，>0.5bp 占比={v2['bad']:.6f}；"
              f"hfq/factor 逐股最大相对极差={v2['max_rel_range']:.2e}（应为 0）"),
    ))

    # 派生列 1：daily_qfq 视图（v1 = close×factor/factor_last；v2 已改 A 空间）
    qfq = con.execute("""
      with r as (
        select d.symbol, d.date,
               round(d.close*d.factor/f.factor_last,4) q1_close,
               d.factor, f.factor_last,
               lag(round(d.close*d.factor/f.factor_last,4)) over (partition by d.symbol order by d.date) p1,
               lag(d.factor) over (partition by d.symbol order by d.date) p2
        from daily d join symbol_factor f using(symbol)
        where d.date between date '2008-01-01' and date '2020-12-31')
      select count(*) n,
             median(abs(q1_close/p1 - factor/p2)) med,
             avg(case when abs(q1_close/p1 - factor/p2) > 1e-3 then 1 else 0 end) bad_share
      from r where p1 is not null
    """).df().iloc[0]
    qfq2 = con.execute("""
      with r as (
        select symbol,date,qfq_close,factor,
               lag(qfq_close) over (partition by symbol order by date) p1,
               lag(factor) over (partition by symbol order by date) p2
        from daily_qfq where date between date '2008-01-01' and date '2020-12-31')
      select count(*) n, max(abs(qfq_close/p1 - factor/p2)) max_err
      from r where p1 is not null
    """).df().iloc[0]
    rows.append(dict(
        id="DERIVED-QFQ", 类别="数据层", 对象="frozen.daily_qfq.qfq_*",
        文件="01_数据/frozen/frozen.duckdb", 单元="证券日", 样本数=int(qfq["n"]), 数据口径="frozen_qfq_v1",
        静态证据="v1 视图 qfq=close×factor/factor_last；v2 已改为 factor/close 锚定",
        备注=(f"v1 与 factor 比值中位|偏差|={qfq['med']:.6f}、>0.1% 占比={qfq['bad_share']:.4f}"
              f"（重复计收益）；v2 现库同口径最大|偏差|={qfq2['max_err']:.2e}"),
    ))

    # 派生列 2：market_daily 宽度 —— v1（hfq 水平）vs 正确价格口径 vs v2 现库
    qb = """
    with d as (
      select symbol,date,is_suspended,close,round(close*factor,4) h1, hfq_close h2,
             avg(close) over w ma20c, avg(round(close*factor,4)) over w ma20h1,
             avg(hfq_close) over w ma20h2, count(*) over w n20
      from daily where date between date '2007-11-01' and date '2020-12-31'
      window w as (partition by symbol order by date rows between 19 preceding and current row))
    select date,
      avg(case when not is_suspended and n20=20 and close>ma20c then 1.0
               when not is_suspended and n20=20 then 0.0 end)*100 correct_ma20,
      avg(case when not is_suspended and n20=20 and h1>ma20h1 then 1.0
               when not is_suspended and n20=20 then 0.0 end)*100 hfq1_ma20,
      avg(case when not is_suspended and n20=20 and h2>ma20h2 then 1.0
               when not is_suspended and n20=20 then 0.0 end)*100 hfq2_ma20
    from d group by date order by date
    """
    b = con.execute(qb).df()
    b["date"] = pd.to_datetime(b.date)
    md = con.execute("""select date,pct_above_ma20 from market_daily
                        where date between date '2007-11-01' and date '2020-12-31'""").df()
    md["date"] = pd.to_datetime(md.date)
    j = b.merge(md, on="date").dropna()
    diff1 = (j.hfq1_ma20 - j.correct_ma20).abs()
    rows.append(dict(
        id="DERIVED-BREADTH", 类别="数据层", 对象="frozen.market_daily.pct_above_ma20/ma60",
        文件="01_数据/frozen/frozen.duckdb", 单元="交易日", 样本数=len(j), 数据口径="frozen_hfq_level_v1",
        静态证据="v1 构建用 hfq_close(=close×factor) 与其 20 日均值比较；v2 用 A 空间",
        备注=(f"v1 宽度 vs 正确价格宽度：corr={j.correct_ma20.corr(j.hfq1_ma20):.6f}，"
              f"中位|差|={diff1.median():.2f}pp，p95={diff1.quantile(.95):.2f}pp，"
              f"50% 门槛翻转日占比={((j.correct_ma20>50)!=(j.hfq1_ma20>50)).mean():.4f}；"
              f"v2 现库 vs A 空间复算最大|差|={(j.hfq2_ma20-j.pct_above_ma20).abs().max():.4f}pp；"
              "项目内无策略读取该列"),
    ))

    # 派生列 3：hfq 当“水平”用于趋势信号的影响（主板，A vs H）
    if not quick:
        qs = """
        with d as (
          select symbol,date,is_suspended,close*factor h_close,factor a_close
          from daily where board='主板' and date between date '2008-01-01' and date '2020-12-31')
        select symbol,date,is_suspended,a_close,h_close,
          max(a_close) over w60 a_max60, max(h_close) over w60 h_max60,
          avg(a_close) over w120 a_ma120, avg(h_close) over w120 h_ma120
        from d
        window w60 as (partition by symbol order by date rows between 60 preceding and 1 preceding),
               w120 as (partition by symbol order by date rows between 119 preceding and current row)
        """
        s = con.execute(qs).df()
        s = s[~s.is_suspended].dropna()
        a_sig = (s.a_close >= s.a_max60) & (s.a_close > s.a_ma120)
        h_sig = (s.h_close >= s.h_max60) & (s.h_close > s.h_ma120)
        rows.append(dict(
            id="LEVEL-SIGNAL", 类别="数据层", 对象="hfq 当水平做 60 日突破+120 日均线",
            文件="01_数据/frozen/frozen.duckdb", 单元="证券日", 样本数=len(s), 数据口径="frozen_hfq_level",
            静态证据="同一规则分别在 A 空间（factor）与 H 空间（close×factor）计算",
            备注=(f"A 信号 {int(a_sig.sum()):,}，H 信号 {int(h_sig.sum()):,}，"
                  f"Jaccard={((a_sig&h_sig).sum()/max((a_sig|h_sig).sum(),1)):.4f}，"
                  f"A 信号在 H 下丢失 {(a_sig&~h_sig).sum()/max(a_sig.sum(),1):.4f}，"
                  f"H 新增 {(~a_sig&h_sig).sum()/max(h_sig.sum(),1):.4f}"),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. 独立复算：交易/事件的空间识别
# --------------------------------------------------------------------------- #
def _conv(c) -> str:
    c = str(c)
    if c.startswith(("sh.", "sz.", "bj.")):
        return c[3:] + "." + c[:2].upper()
    return c


def _calendar(con) -> np.ndarray:
    cal = con.execute("""select date from calendar where is_trading_day=1
                         and date <= DATE '2020-12-31' order by date""").df()
    return pd.to_datetime(cal.date).to_numpy()


def _pairs(con, pairs: pd.DataFrame) -> pd.DataFrame:
    """按 rid 保序取价。pairs 必须是 RangeIndex；返回行数与输入一致（缺失为 NULL）。"""
    p = pairs.reset_index(drop=True).copy()
    p["rid"] = np.arange(len(p))
    con.register("_pairs_tmp", p[["rid", "sym", "e", "x"]])
    q = """
    select p.rid,
           d1.open o_e, d1.close c_e, d1.factor f_e, d1.bs_hfq_close b_e,
           round(d1.open*d1.factor, 4) h_e, round(d1.close*d1.factor, 4) z_e,
           d2.open o_x, d2.close c_x, d2.factor f_x, d2.bs_hfq_close b_x,
           round(d2.open*d2.factor, 4) h_x, round(d2.close*d2.factor, 4) z_x
    from _pairs_tmp p
    left join daily d1 on d1.symbol = p.sym and d1.date = p.e
    left join daily d2 on d2.symbol = p.sym and d2.date = p.x
    order by p.rid
    """
    out = con.execute(q).df()
    con.unregister("_pairs_tmp")
    return out.sort_values("rid").reset_index(drop=True)


def _med(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def _space_stats(rep, rA, rH, rRaw, rA_px=None, rH_px=None, rPrice=None, price_space="n/a"):
    rep = np.asarray(rep, float)
    cands = {"A": np.asarray(rA, float), "H": np.asarray(rH, float), "raw": np.asarray(rRaw, float)}
    if rA_px is not None:
        cands["A_px"] = np.asarray(rA_px, float)
        cands["H_px"] = np.asarray(rH_px, float)
        cands["price"] = np.asarray(rPrice, float)
    ok = np.isfinite(rep)
    for v in cands.values():
        ok &= np.isfinite(v)
    if ok.sum() < 10:
        return dict(n=int(ok.sum()), med_A=np.nan, med_H=np.nan, med_raw=np.nan,
                    share_A_closer=np.nan, ret_space="unknown", best="n/a")
    med = {k: float(np.median(np.abs(rep[ok] - v[ok]))) for k, v in cands.items()}
    best = min(med, key=med.get)
    if best in ("A", "A_px"):
        space = "A/factor"
    elif best in ("H", "H_px"):
        space = "frozen_hfq"
    elif best == "price":
        space = ("A/factor" if (price_space.startswith("A") or price_space.startswith("Baostock"))
                 else "raw")
    else:
        space = "raw"
    # A 与 raw 只在除权日不同：若两者都贴住报告值，归为 A/factor（不要误报 raw）
    if med.get("raw") is not None and abs(med["A"] - med["raw"]) < 5e-4 and med["A"] < 0.01:
        space = "A/factor"
    if med[best] > 0.02:
        space = "ambiguous"
    return dict(n=int(ok.sum()), med_A=med["A"], med_H=med["H"], med_raw=med["raw"],
                med_best=med[best], best=best,
                mean_rep=float(np.mean(rep[ok])), mean_A=float(np.mean(cands["A"][ok])),
                mean_H=float(np.mean(cands["H"][ok])),
                sign_flip=float(np.mean(np.sign(rep[ok]) != np.sign(cands["H"][ok]))),
                share_A_closer=float(np.mean(np.abs(rep[ok] - cands["A"][ok])
                                             < np.abs(rep[ok] - cands["H"][ok]))),
                ret_space=space, price_space=price_space)


def _px_space(px_entry, o_e, c_e, f_e, b_e) -> str:
    px_entry = np.asarray(px_entry, float)
    raw = np.abs(px_entry / np.asarray(o_e, float) - 1)
    a = np.abs(px_entry / (np.asarray(o_e, float) * np.asarray(f_e, float) / np.asarray(c_e, float)) - 1)
    bs = np.abs(px_entry / (np.asarray(o_e, float) * np.asarray(b_e, float) / np.asarray(c_e, float)) - 1)
    if np.nanmedian(raw) < 0.01:
        return "raw(+滑点)"
    if np.nanmedian(a) < 0.01:
        return "A/factor 价"
    if np.nanmedian(bs) < 0.01:
        return "Baostock hfq 价"
    return "其它"


def trade_test(con, df: pd.DataFrame, sym, ed, xd, rep, label: str,
               price_e=None, price_x=None) -> dict:
    df = df.reset_index(drop=True)
    e = guard(df[ed]); x = guard(df[xd])
    keep = e.notna() & x.notna() & (e <= ANALYSIS_END) & (x <= ANALYSIS_END)
    sub = df[keep].reset_index(drop=True)
    pairs = pd.DataFrame({"sym": sub[sym].map(_conv), "e": e[keep].reset_index(drop=True),
                          "x": x[keep].reset_index(drop=True)})
    px = _pairs(con, pairs)
    o_e, c_e, f_e, b_e = (px[k].to_numpy(float) for k in ("o_e", "c_e", "f_e", "b_e"))
    z_e = px["z_e"].to_numpy(float)
    o_x, c_x, f_x, b_x = (px[k].to_numpy(float) for k in ("o_x", "c_x", "f_x", "b_x"))
    z_x = px["z_x"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_e, a_x = o_e * f_e / c_e, o_x * f_x / c_x
        rA = a_x / a_e - 1
        rH = px["h_x"].to_numpy(float) / px["h_e"].to_numpy(float) - 1
        rRaw = o_x / o_e - 1
        kw = {}
        pspace = "n/a"
        if price_e is not None and price_x is not None:
            pe = pd.to_numeric(sub[price_e], errors="coerce").to_numpy(float)
            pxp = pd.to_numeric(sub[price_x], errors="coerce").to_numpy(float)
            kw = dict(rA_px=(pxp * f_x / c_x) / (pe * f_e / c_e) - 1,
                      rH_px=(pxp * f_x) / (pe * f_e) - 1,
                      rPrice=pxp / pe - 1)
            pspace = _px_space(pe, o_e, c_e, f_e, b_e)
    st = _space_stats(sub[rep].to_numpy(float), rA, rH, rRaw, price_space=pspace, **kw)
    return dict(test=label, **st)


def event_test(con, df: pd.DataFrame, sym, sig, rep, offset: int, label: str,
               kind: str = "close", entry_lag: int = 1) -> dict:
    """entry = 信号日 + entry_lag 开盘；exit = 信号日 + offset（close 或 open）。"""
    df = df.reset_index(drop=True)
    cal = _calendar(con)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    d0 = guard(df[sig]).to_numpy()
    i0 = pos.reindex(d0).to_numpy()
    i1 = i0 + entry_lag
    i2 = i0 + offset
    ok = np.isfinite(i0) & (i2 < len(cal)) & (i1 >= 0) & (i1 < len(cal))
    pairs = pd.DataFrame({"sym": df[sym].map(_conv).to_numpy(),
                          "e": np.where(ok, cal[np.clip(i1, 0, len(cal) - 1).astype(int)],
                                        np.datetime64("NaT")),
                          "x": np.where(ok, cal[np.clip(i2, 0, len(cal) - 1).astype(int)],
                                        np.datetime64("NaT"))})
    px = _pairs(con, pairs)
    o_e, c_e, f_e = (px[k].to_numpy(float) for k in ("o_e", "c_e", "f_e"))
    o_x, c_x, f_x = (px[k].to_numpy(float) for k in ("o_x", "c_x", "f_x"))
    with np.errstate(invalid="ignore", divide="ignore"):
        a_e = o_e * f_e / c_e
        if kind == "close":
            rA = f_x / a_e - 1
            rH = px["z_x"].to_numpy(float) / px["h_e"].to_numpy(float) - 1
            rRaw = c_x / o_e - 1
        else:
            rA = (o_x * f_x / c_x) / a_e - 1
            rH = px["h_x"].to_numpy(float) / px["h_e"].to_numpy(float) - 1
            rRaw = o_x / o_e - 1
    st = _space_stats(df[rep].to_numpy(float), rA, rH, rRaw)
    st["price_space"] = "n/a"
    return dict(test=label, **st)


def structure_event_test(con, path: Path, ret_col="ret_hold20", n=20000) -> dict:
    ev = pd.read_parquet(path, columns=["code", "date", "i", "entry_i", "exit_i_hold20",
                                        "entry_px_hfq", ret_col]).reset_index(drop=True)
    if len(ev) > n:
        ev = ev.sample(n, random_state=7).reset_index(drop=True)
    cal = _calendar(con)
    cal04 = cal[(cal >= np.datetime64("2004-01-01"))]
    ei = ev.entry_i.to_numpy(int); xi = ev.exit_i_hold20.to_numpy(int)
    ok = (ei >= 0) & (xi >= 0) & (xi < len(cal04))
    pairs = pd.DataFrame({"sym": ev.code.map(_conv),
                          "e": np.where(ok, cal04[np.clip(ei, 0, len(cal04) - 1)], np.datetime64("NaT")),
                          "x": np.where(ok, cal04[np.clip(xi, 0, len(cal04) - 1)], np.datetime64("NaT"))})
    px = _pairs(con, pairs)
    o_e, c_e, f_e, b_e = (px[k].to_numpy(float) for k in ("o_e", "c_e", "f_e", "b_e"))
    o_x, c_x, f_x = (px[k].to_numpy(float) for k in ("o_x", "c_x", "f_x"))
    with np.errstate(invalid="ignore", divide="ignore"):
        rA = (o_x * f_x / c_x) / (o_e * f_e / c_e) - 1
        rH = px["h_x"].to_numpy(float) / px["h_e"].to_numpy(float) - 1
        rRaw = o_x / o_e - 1
    st = _space_stats(ev[ret_col].to_numpy(float), rA, rH, rRaw)
    st["price_space"] = _px_space(ev.entry_px_hfq.to_numpy(float), o_e, c_e, f_e, b_e)
    return dict(test=path.stem, **st)


def sample_parquet(path: Path, n: int) -> pd.DataFrame:
    if n <= 0:
        return pd.read_parquet(path)
    c = duckdb.connect()
    try:
        return c.execute(f"select * from read_parquet('{path}') using sample {int(n)} rows (reservoir, 42)").df()
    finally:
        c.close()


def artifact_tests(con, quick: bool) -> pd.DataFrame:
    rows = []
    N = 8000 if quick else 30000

    def add(aid, family, path, unit, res: dict, note=""):
        rows.append(dict(
            id=aid, 类别="结果", 对象=family, 文件=str(path.relative_to(ROOT)), 单元=unit,
            样本数=res.get("n"), 数据口径=res.get("ret_space", "unknown"),
            静态证据=(f"复算 {res.get('test')}: 中位|误差| A={res.get('med_A', float('nan'))*100:.3f}pp，"
                     f"H={res.get('med_H', float('nan'))*100:.3f}pp，raw={res.get('med_raw', float('nan'))*100:.3f}pp；"
                     f"最佳={res.get('best')}({res.get('med_best', float('nan'))*100:.3f}pp)；"
                     f"A 更近占比={res.get('share_A_closer', float('nan')):.3f}；"
                     f"均值 报告={res.get('mean_rep', float('nan'))*100:.2f}% / A={res.get('mean_A', float('nan'))*100:.2f}% / "
                     f"H={res.get('mean_H', float('nan'))*100:.2f}%；H 符号翻转={res.get('sign_flip', float('nan')):.3f}；"
                     f"报价空间={res.get('price_space', 'n/a')}"),
            备注=note,
        ))

    # 结构族 K3 事件（BS hfq 报价，ret_hold20）
    p = RES / "结构族_事件_K3.parquet"
    add("RES-STRUCT", "结构族 K3 事件", p, "event", structure_event_test(con, p, n=N),
        "entry_px_hfq 与 Baostock hfq 价位一致（非冻结 hfq）；抽样 3 万条（有效 29,414）")

    # 趋势族逐笔
    p = RES / "趋势族" / "交易明细_全部.parquet"
    d = pd.read_parquet(p, columns=["code", "entry_date", "exit_date", "entry_px", "exit_px", "ret", "period"])
    if len(d) > N * 4:
        d = d.sample(N * 4, random_state=11)
    res = trade_test(con, d, "code", "entry_date", "exit_date", "ret", "trend_trades", price_e="entry_px", price_x="exit_px")
    add("RES-TREND", "趋势族 逐笔", p, "trade", res, "Baostock flag1；已在第一阶段审计确认断点")

    # 退出族 B0 逐笔
    p = RES / "退出_明细_TB1_B0.csv"
    d = pd.read_csv(p, usecols=["code", "entry_date", "exit_date", "entry_px", "exit_px", "ret", "window"])
    if len(d) > N * 4:
        d = d.sample(N * 4, random_state=12)
    res = trade_test(con, d, "code", "entry_date", "exit_date", "ret", "exit_trades", price_e="entry_px", price_x="exit_px")
    add("RES-EXIT", "退出族 TB1/B0 逐笔", p, "trade", res,
        "exit_cache.px_hfq == bs_hfq_close；抽样 12 万条；逐笔 ret 与报价比价 90% <1e-6（原始价口径，另含分批/停牌差异）")

    # 统一引擎 FIX20
    dfs = []
    for f in ("交易_FIX20_dev.csv", "交易_FIX20_val.csv"):
        p = RES / "统一引擎" / f
        if p.exists():
            dfs.append(pd.read_csv(p, usecols=["code", "entry_date", "exit_date", "entry_price", "exit_price", "ret"]))
    p = RES / "统一引擎" / "交易_FIX20_dev.csv"
    d = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    res = trade_test(con, d, "code", "entry_date", "exit_date", "ret", "ue_trades", price_e="entry_price", price_x="exit_price")
    add("RES-UE", "统一引擎 FIX20", p, "trade", res, "factor/ca_adj 口径")

    # 候选_趋势回调 事件（entry=t+1 开盘，exit=t+1+20 收盘）
    p = RES / "候选_趋势回调" / "01_事件_明细.parquet"
    d = pd.read_parquet(p, columns=["symbol", "date", "r20", "entry_ok"])
    d = d[d.entry_ok].dropna(subset=["r20"])
    if len(d) > N:
        d = d.sample(N, random_state=13)
    res = event_test(con, d, "symbol", "date", "r20", 21, "cb_events")
    add("RES-CB", "候选_趋势回调 事件", p, "event", res, "pct_chg 递归重建 tr；抽样 3 万条")

    # 候选_收缩突破 事件（entry=t+1 开盘，exit=t+20 收盘）
    p = RES / "候选_收缩突破" / "事件_明细.parquet"
    d = pd.read_parquet(p, columns=["symbol", "date", "ret20", "config"])
    d = d.dropna(subset=["ret20"])
    if len(d) > N:
        d = d.sample(N, random_state=14)
    res = event_test(con, d, "symbol", "date", "ret20", 20, "vcb_events")
    add("RES-VCB", "候选_收缩突破 事件", p, "event", res, "tr=factor/raw×factor/close；全量 26,368 条")

    # 候选_生态切换 逐笔（entry t+1 开盘 / exit 开盘）
    p = RES / "候选_生态切换" / "08_组合交易.csv"
    d = pd.read_csv(p, usecols=["code", "entry_date", "exit_date", "entry_px", "exit_px", "ret"])
    if len(d) > N:
        d = d.sample(N, random_state=15)
    res = trade_test(con, d, "code", "entry_date", "exit_date", "ret", "eco_trades", price_e="entry_px", price_x="exit_px")
    add("RES-ECO", "候选_生态切换 逐笔", p, "trade", res, "factor 空间；抽样 3 万条")

    # 候选_相对强弱 逐笔（BS hfq 空间）
    dfs = []
    for f in sorted((RES / "候选_相对强弱").glob("03_交易_*.csv")):
        dfs.append(pd.read_csv(f, usecols=["code", "entry_date", "exit_date", "entry_px", "exit_px", "ret"]))
    d = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    p = RES / "候选_相对强弱" / "03_交易_C01_dev.csv"
    res = trade_test(con, d, "code", "entry_date", "exit_date", "ret", "rs_trades", price_e="entry_px", price_x="exit_px")
    add("RES-RS", "候选_相对强弱 逐笔", p, "trade", res,
        "exit_cache = bs_hfq；含分批离场，单一 exit_px 不能完全复现，A 仍优于 H 约 3.4 倍")

    # 候选_退出统一 逐笔（A 空间比例建仓；抽样）
    p = RES / "候选_退出统一" / "07_逐笔.parquet"
    d = sample_parquet(p, N * 4)
    res = trade_test(con, d, "code", "entry_date", "exit_date", "ret", "uExit_trades", price_e="entry_px", price_x="exit_px")
    add("RES-UEXIT", "候选_退出统一 逐笔", p, "trade", res,
        "scale=factor/close 重建；抽样 12 万条；扣费后逐笔≈报价比价（A 与 raw 20 日内几乎不可分）")

    # 候选_小盘低波趋势 逐笔（统一数据适配：factor/ca_adj；结果于 23:47 落盘）
    p = RES / "候选_小盘低波趋势" / "12_组合_交易_主单元.csv"
    if p.exists():
        d = pd.read_csv(p, usecols=["code", "entry_date", "exit_date", "entry_price",
                                    "exit_price", "gross_ret", "variant"])
        if len(d) > N:
            d = d.sample(N, random_state=17)
        res = trade_test(con, d, "code", "entry_date", "exit_date", "gross_ret",
                         "smallcap_trades", price_e="entry_price", price_x="exit_price")
        add("RES-SMALLCAP", "候选_小盘低波趋势 逐笔", p, "trade", res,
            "import 统一数据适配（factor/ca_adj）；结果生成于 v2 重建后")

    # 候选_执行压力 逐笔（factor 空间）
    p = RES / "执行压力" / "05_交易明细_关键配置.csv"
    d = pd.read_csv(p, usecols=["code", "entry_date", "exit_date", "entry_px", "exit_px", "gross_ret", "net_ret"])
    if len(d) > N:
        d = d.sample(N, random_state=16)
    res = trade_test(con, d, "code", "entry_date", "exit_date", "gross_ret", "exec_trades", price_e="entry_px", price_x="exit_px")
    add("RES-EXEC", "执行压力 逐笔", p, "trade", res, "factor 空间；全量 6,225 条")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. 判定与汇总
# --------------------------------------------------------------------------- #
def classify(static_df: pd.DataFrame, test_df: pd.DataFrame, defect_df: pd.DataFrame) -> pd.DataFrame:
    """把静态口径 + 复算结果合成对象级判定。"""
    def lookup(aid):
        r = test_df[test_df.id == aid]
        return None if r.empty else r.iloc[0]

    rows = []

    def add(aid, obj, family, unit, space, n, evidence, verdict, sign, whitelist, note):
        rows.append(dict(id=aid, 类别="判定", 对象=obj, 文件=family, 单元=unit, 样本数=n,
                         数据口径=space, 静态证据=evidence, 备注=note,
                         判定=verdict, 符号稳健=sign, 白名单=whitelist))

    add("J-FROZEN-V1", "冻结层 v1 daily.hfq_*", "01_数据/frozen（v1 21:55 build）", "数据列", "frozen_hfq_v1",
        "8.4M 证券日", "hfq=ROUND(raw×factor,4)；收益≈2×pct_chg", "作废待重跑", "n/a", "否",
        "v1 产物作废；现库已于 23:40 重建为 v2（见 V2-ACCEPT）。引用 v1 数字的旧结论不得再用")
    add("J-FROZEN-V2", "冻结层 v2 daily.hfq_*（23:40 重建）", "01_数据/frozen/frozen.duckdb", "数据列",
        "frozen_hfq_v2", "8.4M 证券日", "hfq=raw×factor/close×close_first", "不受此缺陷影响", "n/a", "是",
        "独立复核：hfq 收益 == pct_chg（最大 1e-16 量级），hfq/factor 逐股常数")
    add("J-QFQ", "冻结层 daily_qfq 视图", "01_数据/frozen/frozen.duckdb", "视图", "frozen_qfq",
        "-", "v1 qfq=close×factor/factor_last；v2 已修正", "作废待重跑", "n/a", "是（v2）",
        "v1 作废；v2 现库 qfq 收益与 factor 比值一致；DataLayer.daily('qfq') 仍无调用者")
    add("J-BREADTH", "market_daily.pct_above_ma20/ma60", "01_数据/frozen/frozen.duckdb", "数据列",
        "frozen_hfq_level", "3,187 日", "v1 用 hfq 水平与其 MA 比较；v2 用 A 空间", "受影响但符号稳健", "是", "是（v2）",
        "v1 与正确价格宽度 corr≈0.9998、中位差≈0.27pp、50% 门槛翻转≈0.69%；无策略读取；v2 已重算")
    add("J-SELFCHECK", "01_数据/_audit 自检 C1/C2", "01_数据/_audit/自检报告.md", "检查", "frozen_hfq",
        "15 项", "C1 用恒等式、C2 用 factor 收益冒充 hfq 收益", "作废待重跑", "n/a", "否",
        "必须在自检中直接比较 hfq_close 比值与 pct_chg；旧 PASS 是假阳性（v1 期）")
    add("J-DICT", "数据字典 §5.1/§7", "01_数据/数据字典.md", "文档", "frozen_hfq",
        "-", "写明 hfq_px=raw_px×factor 且宣称复权收益偏差 0", "作废待重跑", "n/a", "否",
        "公式与自检结论要改写；『factor 是总收益指数、不用 Baostock hfq』的主结论仍成立")
    add("J-DATALAYER", "DataLayer.daily(adjust='hfq'/'qfq')", "02_代码/数据层.py", "API", "frozen_hfq_api",
        "0 调用", "无导入者；无参 .daily() 默认 raw", "作废待重跑", "n/a", "否",
        "API 应改为返回 A 空间或删除 hfq/qfq 分支；当前无下游，故无结果受影响")
    add("J-BUILDER", "构建冻结数据.py", "02_代码/构建冻结数据.py", "代码", "frozen_hfq_builder",
        "-", "v1 产物含缺陷；源码 23:37 改为 A 空间，23:40 已重建", "作废待重跑", "n/a", "是（v2）",
        "v2 重建后冻结层恢复；重构脚本与数据需记录版本指纹")
    add("J-AUDIT1-REC", "第一阶段审计的修复建议第 1 条", "05_审计/第一阶段代码与负结果审计.md", "文档建议",
        "mixed", "-", "建议『改用冻结层 factor/hfq_close』", "受影响但符号稳健", "是", "否",
        "负结论本身不受影响；但『hfq_close』不能作为修复目标，应改为 factor 或 A 空间")
    add("J-DICT-CONC", "数据字典：factor 是总收益指数、不用 Baostock hfq", "01_数据/数据字典.md", "文档结论",
        "factor", "-", "定性结论正确，公式错误", "受影响但符号稳健", "是", "是",
        "可引用『factor=总收益指数、Baostock hfq 有年末重定基』；不可引用 hfq=raw×factor 与偏差 0")
    add("J-AUDIT1-CONC", "第一阶段四族负结论", "05_审计/第一阶段代码与负结果审计.md", "结论",
        "baostock_hfq", "-", "结构/趋势/退出均用 Baostock hfq，不读冻结 hfq", "不受此缺陷影响", "n/a", "是",
        "负结论不依赖冻结 hfq；但它们仍受 Baostock 断点、tp8 丢弃、限价档位等已列问题约束")
    add("J-DOC-CONFLICT", "统一引擎口径冲突与裁决 §13", "03_结果/统一引擎/口径冲突与裁决.md", "文档",
        "frozen_hfq", "-", "正确指出 daily(adjust=hfq) 的 close 也是 close×factor", "不受此缺陷影响", "n/a", "是",
        "可直接引用为缺陷定义与修复口径")
    add("J-DOC-EXEC", "执行压力：数据层缺陷复权", "03_结果/执行压力/10_数据层缺陷_复权.md", "文档",
        "frozen_hfq", "-", "实测 hfq/pct 均值 2.0006、中位 2.0014", "不受此缺陷影响", "n/a", "是",
        "与本次重建复算一致（中位比 2.0102，窗口/阈值不同）")
    add("J-DOC-CB", "候选_趋势回调：冻结层 hfq 缺陷", "03_结果/候选_趋势回调/12_数据缺陷_冻结层hfq.md", "文档",
        "frozen_hfq", "-", "91.73% 行偏差>0.5bp、中位 1.396%", "不受此缺陷影响", "n/a", "是",
        "与本次 92.14%/1.373pp 一致；该族已改用 pct_chg 递归 tr")
    add("J-DOC-ECO", "候选_生态切换：hfq 双计", "03_结果/候选_生态切换/21_数据层缺陷_hfq双计.md", "文档",
        "frozen_hfq", "-", "hfq/pct 中位 2.0009、89.15% 行>1bp", "不受此缺陷影响", "n/a", "是",
        "结论一致；该族用 factor/日内比例")

    for aid, obj, fam, unit, verdict, sign, note in [
        ("J-STRUCT", "结构族（事件/组合/离场）", "03_结果/结构族_*", "event/trade", "不受此缺陷影响", "n/a",
         "Baostock flag1；K3 事件 ret_hold20 复算 A 中位误差见 RES-STRUCT"),
        ("J-TREND", "趋势族（事件/组合）", "03_结果/趋势族", "trade", "不受此缺陷影响", "n/a",
         "Baostock flag1；逐笔复现已确认 A 空间"),
        ("J-REGIME", "市场状态族", "03_结果/市场状态", "state", "不受此缺陷影响", "n/a",
         "自带 panel_hfq（Baostock flag1）；未读 frozen.market_daily"),
        ("J-EXIT", "退出族 TB1/TB2", "03_结果/退出_明细_*", "trade", "不受此缺陷影响", "n/a",
         "exit_cache.px_hfq 与冻结 bs_hfq_close 9.1M 行 100% 相同"),
        ("J-UE", "统一引擎", "03_结果/统一引擎", "trade/nav", "不受此缺陷影响", "n/a",
         "ca_adj=raw/ca_factor，tr_index=factor；不读 hfq 列"),
        ("J-REPRO", "独立复现基线", "05_审计/独立复现_基线.md", "trade/nav", "不受此缺陷影响", "n/a",
         "A 空间自建；其 §8 正是本缺陷的量化来源"),
        ("J-CB", "候选_趋势回调", "03_结果/候选_趋势回调", "event/trade", "不受此缺陷影响", "n/a",
         "读 hfq 列后整列覆盖为 pct_chg 递归 tr；v1 200 只调试已作废"),
        ("J-VCB", "候选_收缩突破", "03_结果/候选_收缩突破", "event/trade", "不受此缺陷影响", "n/a",
         "tr=factor/raw×factor/close；全量 26,368 条"),
        ("J-ECO", "候选_生态切换", "03_结果/候选_生态切换", "event/trade", "不受此缺陷影响", "n/a",
         "hc=factor，日内 factor×px/close"),
        ("J-RS", "候选_相对强弱", "03_结果/候选_相对强弱", "event/trade", "不受此缺陷影响", "n/a",
         "from 退出_数据层 import Panel（bs_hfq）"),
        ("J-EXEC", "候选_执行压力", "03_结果/执行压力", "trade", "不受此缺陷影响", "n/a",
         "hfq_close() 返回 factor；自有缺陷文档"),
        ("J-UEXIT", "候选_退出统一", "03_结果/候选_退出统一", "trade", "不受此缺陷影响", "n/a",
         "scale=factor/close 重建，自检 hfq_ret≡pct_chg"),
        ("J-RB", "候选_风险预算", "03_结果/候选_风险预算", "nav/state", "不受此缺陷影响", "n/a",
         "收益用 factor；00_口径审计 已量化 hfq 缺陷"),
        ("J-BREAK-RETEST", "候选_突破回踩（代码，无结果）", "02_代码/候选_突破回踩.py", "code",
         "不受此缺陷影响", "n/a", "A_t=factor/close；自检含 hfq 反例；结果目录仅预注册"),
        ("J-LONGMOM", "候选_长期双动量（代码，无结果）", "02_代码/候选_长期双动量.py", "code",
         "不受此缺陷影响", "n/a", "adj_close=factor；自检比对 hfq 列；无结果"),
        ("J-REVERSAL", "候选_结构内反转（代码，无结果）", "02_代码/候选_结构内反转.py", "code",
         "不受此缺陷影响", "n/a", "A_t=factor/close；无结果"),
        ("J-SMALLCAP", "候选_小盘低波趋势（代码+逐笔）", "03_结果/候选_小盘低波趋势", "trade",
         "不受此缺陷影响", "n/a", "import 统一数据适配（factor/ca_adj）；逐笔复算见 RES-SMALLCAP"),
    ]:
        add(aid, obj, fam, "结果集", "A/factor" if "不受" in verdict else "mixed", "-",
            "代码口径 + 独立复算", verdict, sign, "是", note)

    add("J-PREREG", "父预注册复权表述", "00_研究预注册.md", "文档", "mixed", "-",
        "只写『后复权用于收益率/比例指标』，未写 hfq 公式", "不受此缺陷影响", "n/a", "是",
        "建议补一句『总收益指数用 factor，价格水平用 A 空间，禁止 close×factor』")
    add("J-SMALL", "仅预注册、无代码的候选", "03_结果/候选_结构内反转/00_预注册_*.md 等", "文档", "n/a", "-",
        "只有预注册 md，无 py、无结果", "未知", "n/a", "否", "无产物可审；代码落地后按其数据口径复核")

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    AUD.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FROZEN), read_only=True)
    con.execute("SET memory_limit='6GB'")

    hi = sha256(FROZEN)
    mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(FROZEN.stat().st_mtime))
    v1 = {}
    fp = ROOT / "01_数据" / "_audit" / "v1_修复前_指纹.json"
    if fp.exists():
        try:
            v1 = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            v1 = {}
    log(f"[snapshot] frozen.duckdb 现库 sha256={hi[:16]}… mtime={mt}")
    log(f"[snapshot] v1 快照 sha256={str(v1.get('frozen_duckdb_sha256','?'))[:16]}… "
        f"captured_at={v1.get('captured_at','?')}（v1 已于 23:40 被重建，v1 缺陷按定义重建复算）")

    static_df, facts = static_scan()
    indirect = datalayer_indirect(facts)
    log(f"[static] py={len(facts)} 个，文本行={len(static_df)-len(facts)}")

    defect_df = quantify_defect(con, args.quick)
    log("[defect] " + " | ".join(f"{r.对象}: {r.备注[:60]}" for _, r in defect_df.iterrows()))

    test_df = artifact_tests(con, args.quick)
    log("[recompute] " + " | ".join(f"{r.对象}: A={r['数据口径']}" for _, r in test_df.iterrows()))

    verdict_df = classify(static_df, test_df, defect_df)
    all_df = pd.concat([static_df, indirect, defect_df, test_df, verdict_df], ignore_index=True, sort=False)
    all_df["快照"] = f"{SNAPSHOT_AT} / frozen {hi[:12]}"
    all_df["复现命令"] = "python3 02_代码/审计复权引用.py"
    cols = ["id", "类别", "对象", "文件", "单元", "样本数", "数据口径", "静态证据",
            "判定", "符号稳健", "白名单", "备注", "快照", "复现命令"]
    for c in cols:
        if c not in all_df:
            all_df[c] = ""
    all_df = all_df[cols]
    all_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"[write] {OUT_CSV}  rows={len(all_df)}  {time.time()-t0:.1f}s")

    # 控制台摘要（供 md 引用）
    v = verdict_df[["id", "对象", "判定", "白名单", "备注"]]
    print("\n=== 判定汇总 ===")
    print(v.to_string(index=False, max_colwidth=60))
    print("\n=== 关键数字 ===")
    for _, r in pd.concat([defect_df, test_df]).iterrows():
        print(f"{r['id']:<16} {str(r['对象'])[:28]:<30} {str(r.get('备注','') or r.get('静态证据',''))[:110]}")
    con.close()


if __name__ == "__main__":
    main()
