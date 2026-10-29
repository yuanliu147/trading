#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终口径对账（只读）

把最终交付里最容易互相矛盾的六类口径做成机器可复核的清单：
  1) N 计数与 DSR 阶梯（733 / 1,251 / 454 / 257 / 87；1,699 行）
  2) T0005 的当前状态（68 行 done、19 N_sel/38 cell/30 diag；旧 planned/not_run 仅历史）
  3) 门槛覆盖（478 行 → 505 covered + 228 unrated = 733；旧 486/228 为 714 口径）
  4) B_conditional 两个口径（配置级 5 个/10 行；旧行级 12 行/8 配置，含 1 行 S1）
  5) 小盘 COMBO val 基准（首日 +0.33% vs 初始 100 万 -0.03%；first day -1.76%）
  6) 2021+ 扫描时点（当前 469；历史 470；0 违规）

只读：不重跑策略、不写 03_结果/、不改注册表/总账。唯一写入 04_证据/最终口径对账.json。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "04_证据", "最终口径对账.json")


def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def jload(rel: str) -> dict:
    with open(p(rel), encoding="utf-8") as f:
        return json.load(f)


def read_text(rel: str) -> str:
    with open(p(rel), encoding="utf-8") as f:
        return f.read()


CHECKS: list[dict] = []


def add(cid: str, item: str, expected, actual, passed: bool, source: str, note: str = "") -> None:
    CHECKS.append({
        "id": cid, "item": item, "expected": expected, "actual": actual,
        "pass": bool(passed), "source": source, "note": note,
    })


def eq(cid: str, item: str, expected, actual, source: str, note: str = "") -> None:
    add(cid, item, expected, actual, expected == actual, source, note)


def main() -> dict:
    led = pd.read_csv(p("03_结果", "最终试验总账.csv"))
    gates = pd.read_csv(p("03_结果", "最终统计", "统一门槛.csv"))
    cg = pd.read_csv(p("03_结果", "统计整合", "候选分级.csv"))
    fj = jload("03_结果/最终统计/最终判定.json")
    fc = jload("04_最终结论.json")
    sm = jload("03_结果/最终统计/T0005_纳入后_汇总.json")
    acc = jload("04_证据/交付验收.json")
    mx = jload("05_审计/最终结论证据矩阵_复算.json")

    # ---- 1. N 计数 ----
    counts = {
        "rows": int(len(led)),
        "N_sel": int(led["in_Nsel"].sum()),
        "N_cell": int(led["in_Ncell"].sum()),
        "N_diag": int(led["in_Ndiag"].sum()),
        "N_platform": int(led["in_Nplatform"].sum()),
        "N_audit": int(led["in_Naudit"].sum()),
    }
    eq("N01", "总账计数（733/1251/454/257/87；1699 行）",
       {"rows": 1699, "N_sel": 733, "N_cell": 1251, "N_diag": 454, "N_platform": 257, "N_audit": 87},
       counts, "03_结果/最终试验总账.csv")
    eq("N02", "最终计数.csv 合计与总账一致",
       counts,
       {k: int(pd.read_csv(p("03_结果", "最终统计", "最终计数.csv")).query("scope=='total'")[v].iloc[0])
        for k, v in [("rows", "rows"), ("N_sel", "N_sel"), ("N_cell", "N_cell"),
                     ("N_diag", "N_diag"), ("N_platform", "N_platform"), ("N_audit", "N_audit")]},
       "03_结果/最终统计/最终计数.csv")
    fc_total = fc["counts"]["total"]
    eq("N03", "04_最终结论.json 计数与总账一致",
       counts,
       {"rows": int(fc_total["rows"]), "N_sel": int(fc_total["in_Nsel"]),
        "N_cell": int(fc_total["in_Ncell"]), "N_diag": int(fc_total["in_Ndiag"]),
        "N_platform": int(fc_total["in_Nplatform"]), "N_audit": int(fc_total["in_Naudit"])},
       "04_最终结论.json")
    eq("N04", "最终判定.json 计数与总账一致",
       {"N_sel": 733, "N_cell": 1251, "N_diag": 454},
       {k: int(fj["counts"]["total"][v]) for k, v in [("N_sel", "in_Nsel"), ("N_cell", "in_Ncell"), ("N_diag", "in_Ndiag")]},
       "03_结果/最终统计/最终判定.json")
    eq("N05", "DSR 阶梯（733/990/1444；6.6275e-4；SR0 1.6489）",
       {"N1": 733, "N2": 990, "N3": 1444, "dsr_dev": 0.0006627544, "sr0": 1.648895},
       {"N1": fc["statistics"]["dsr_ladder"]["N1_sel"]["N"],
        "N2": fc["statistics"]["dsr_ladder"]["N2_sel_plus_platform"]["N"],
        "N3": fc["statistics"]["dsr_ladder"]["N3_all_cells"]["N"],
        "dsr_dev": round(fc["statistics"]["dsr_ladder"]["N1_sel"]["dsr_best_dev"], 10),
        "sr0": round(fc["statistics"]["dsr_ladder"]["N1_sel"]["sr0_annual"], 6)},
       "04_最终结论.json")
    eq("N06", "超预算倍数（5.7031/11.1562/11.4531）",
       {"stage1": 5.7031, "snapshot_714": 11.1562, "current_733": 11.4531},
       {"stage1": mx["overrun"]["stage1_x"], "snapshot_714": mx["overrun"]["snapshot_x"],
        "current_733": mx["overrun"]["ledger_x"]},
       "05_审计/最终结论证据矩阵_复算.json")

    # ---- 2. 门槛覆盖 ----
    gated = {(r.family, r.config_id) for r in gates.itertuples()}
    account_cats = {"prereg", "posthoc", "sensitivity"}
    cover = int(sum(float(r.in_Nsel or 0) for r in led.itertuples()
                    if r.category in account_cats and (r.family, r.config_id) in gated))
    unrated = int(counts["N_sel"] - cover)
    eq("C01", "门槛行数 478（含 MAIN 19 行）",
       {"rows": 478, "main_rows": 19, "non_main_rows": 459},
       {"rows": int(len(gates)), "main_rows": int((gates.family == "MAIN").sum()),
        "non_main_rows": int((gates.family != "MAIN").sum())},
       "03_结果/最终统计/统一门槛.csv")
    eq("C02", "当前覆盖 505 + 未评级 228 = 733",
       {"covered": 505, "unrated": 228, "sum": 733},
       {"covered": cover, "unrated": unrated, "sum": cover + unrated},
       "03_结果/最终试验总账.csv；统一门槛.csv")
    eq("C03", "最终判定.json 覆盖字段已改为 505/228/733",
       {"covered": 505, "unrated": 228, "total": 733, "rows": 478},
       {"covered": int(fj["holdout_gate"]["N_sel_covered_by_gate_rows"]),
        "unrated": int(fj["holdout_gate"]["N_sel_unrated"]),
        "total": int(fj["holdout_gate"]["N_sel_total"]),
        "rows": int(fj["holdout_gate"]["n_gate_rows"])},
       "03_结果/最终统计/最终判定.json")
    eq("C04", "旧 486/228 是 714 口径（486+228=714）",
       {"covered_714": 486, "unrated": 228, "sum": 714},
       {"covered_714": int(mx["ledger_snapshot_0051"]["N_sel"] - unrated),
        "unrated": unrated, "sum": int(mx["ledger_snapshot_0051"]["N_sel"])},
       "05_审计/最终结论证据矩阵_复算.json")
    eq("C05", "未评级去向（STRUCT 219 + STAT 9）",
       {"STRUCT": 219, "STAT": 9, "sum": 228},
       {"STRUCT": int(led[(led.family == "STRUCT") & (led.in_Nsel > 0)].in_Nsel.sum()),
        "STAT": int(led[(led.family == "STAT") & (led.in_Nsel > 0)].in_Nsel.sum()),
        "sum": int(led[(led.family == "STRUCT") & (led.in_Nsel > 0)].in_Nsel.sum()
                   + led[(led.family == "STAT") & (led.in_Nsel > 0)].in_Nsel.sum())},
       "03_结果/最终试验总账.csv",
       "两种族在总账中各只有合并窗口/夹具，无 dev/val 两窗口记录")

    # ---- 3. T0005 状态 ----
    main = led[led.family == "MAIN"]
    done = main[main.status == "done"]
    eq("T01", "T0005 登记：68 行 done（34 dev + 34 val）",
       {"rows": 68, "dev": 34, "val": 34},
       {"rows": int(len(done)), "dev": int((done.stage == "dev").sum()), "val": int((done.stage == "val").sum())},
       "03_结果/最终试验总账.csv")
    eq("T02", "T0005 计数增量：19 N_sel / 38 cell / 30 diag",
       {"N_sel": 19, "N_cell": 38, "N_diag": 30},
       {"N_sel": int(done.in_Nsel.sum()), "N_cell": int(done.in_Ncell.sum()), "N_diag": int(done.in_Ndiag.sum())},
       "03_结果/最终试验总账.csv")
    eq("T03", "总账仍保留 2 行历史状态（F00002 not_run、F01629 planned）",
       {"stale_ids": ["F00002", "F01629"], "statuses": ["not_run", "planned"]},
       {"stale_ids": sorted(led[led.status != "done"].ledger_id.tolist()),
        "statuses": sorted(led[led.status != "done"].status.tolist())},
       "03_结果/最终试验总账.csv",
       "历史快照；当前事实以 done 行为准，不删旧行")
    eq("T04", "T0005 dev/val 数字",
       {"dev_cagr": 0.04156816, "dev_mdd": -0.09053507, "dev_trades": 103,
        "val_cagr": -0.02534228, "val_mdd": -0.13797984, "val_trades": 159},
       {"dev_cagr": round(float(mx["t0005"]["dev"]["cagr_nav_first"]), 8),
        "dev_mdd": round(float(mx["t0005"]["dev"]["mdd"]), 8),
        "dev_trades": int(mx["t0005"]["dev"]["n_trades"]),
        "val_cagr": round(float(mx["t0005"]["val"]["cagr_nav_first"]), 8),
        "val_mdd": round(float(mx["t0005"]["val"]["mdd"]), 8),
        "val_trades": int(mx["t0005"]["val"]["n_trades"])},
       "05_审计/最终结论证据矩阵_复算.json")
    eq("T05", "04_最终结论 T0005 判定为 discard",
       {"decision": "discard", "n_candidates": 0},
       {"decision": fc["verdict"]["main_candidate"]["decision"],
        "n_candidates": fc["verdict"]["n_candidates"]},
       "04_最终结论.json")

    # ---- 4. B_conditional ----
    b_cfg = gates[gates.grade == "B_conditional"]
    b_keys = set(zip(b_cfg.family, b_cfg.config_id))
    b_ledger_rows = int(sum(1 for r in led.itertuples() if (r.family, r.config_id) in b_keys))
    b_row = cg[cg.grade == "B_conditional"]
    eq("B01", "配置级 B_conditional：5 个配置 / 10 总账行（4 S3 + 1 VCB）",
       {"configs": 5, "ledger_rows": 10, "trend_s3": 4, "vcb": 1},
       {"configs": int(b_cfg.config_id.nunique()), "ledger_rows": b_ledger_rows,
        "trend_s3": int(b_cfg.strategy_id.str.startswith("TREND@S3").sum()),
        "vcb": int((b_cfg.family == "VCB").sum())},
       "03_结果/最终统计/统一门槛.csv；最终试验总账.csv")
    eq("B02", "旧行级 B_conditional：12 行 / 8 配置（11 S3 + 1 S1；仅 TREND）",
       {"rows": 12, "configs": 8, "s3_rows": 11, "s1_rows": 1, "families": ["TREND"]},
       {"rows": int(len(b_row)), "configs": int(b_row.config_id.nunique()),
        "s3_rows": int((b_row.strategy_id == "TREND@S3_波动收缩突破").sum()),
        "s1_rows": int((b_row.strategy_id == "TREND@S1_月周日共振突破").sum()),
        "families": sorted(b_row.family.unique().tolist())},
       "03_结果/统计整合/候选分级.csv")
    eq("B03", "04_最终结论 同时记录两个口径",
       {"config_n": 5, "config_rows": 10, "row_n": 12, "row_configs": 8, "s1_rows": 1},
       {"config_n": fc["candidate_grading"]["config_level"]["n_configs"],
        "config_rows": fc["candidate_grading"]["config_level"]["n_rows"],
        "row_n": fc["candidate_grading"]["row_level_6_families"]["n_rows"],
        "row_configs": fc["candidate_grading"]["row_level_6_families"]["n_configs"],
        "s1_rows": fc["candidate_grading"]["row_level_6_families"]["trend_s1_rows"]},
       "04_最终结论.json")

    # ---- 5. 小盘 COMBO 基准 ----
    sc = pd.read_csv(p("03_结果", "候选_小盘低波趋势", "13_组合_净值_主单元.csv"))
    combo = {}
    for w in ("dev", "val"):
        s = sc[(sc.variant == "COMBO") & (sc.window == w)].sort_values("date")
        nav = s.nav.astype(float).to_numpy()
        dates = pd.to_datetime(s.date)
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        combo[w] = {
            "start": str(dates.iloc[0].date()),
            "nav_first": round(float(nav[0]), 2),
            "nav_last": round(float(nav[-1]), 2),
            "cagr_nav_first": round(float((nav[-1] / nav[0]) ** (1 / years) - 1), 8),
            "cagr_initial_capital": round(float((nav[-1] / 1_000_000) ** (1 / years) - 1), 8),
            "first_day_return": round(float(nav[0] / 1_000_000 - 1), 8),
            "mdd": round(float((nav / pd.Series(nav).cummax().to_numpy() - 1).min()), 8),
        }
    eq("S01", "小盘 COMBO dev：首日基准 9.5275% / 初始资金 9.5844%",
       {"nav_first": 0.09527546, "initial": 0.09584395},
       {"nav_first": combo["dev"]["cagr_nav_first"], "initial": combo["dev"]["cagr_initial_capital"]},
       "03_结果/候选_小盘低波趋势/13_组合_净值_主单元.csv")
    eq("S02", "小盘 COMBO val：首日基准 +0.3270% / 初始资金 -0.0297%（首日 -1.7618%）",
       {"nav_first": 0.00326993, "initial": -0.00029673, "first_day": -0.01761811, "nav_first_value": 982381.89},
       {"nav_first": combo["val"]["cagr_nav_first"], "initial": combo["val"]["cagr_initial_capital"],
        "first_day": combo["val"]["first_day_return"], "nav_first_value": combo["val"]["nav_first"]},
       "03_结果/候选_小盘低波趋势/13_组合_净值_主单元.csv",
       "报告 +0.33% 是 nav-first；按初始 100 万为 -0.03%，不得写两段为正")
    eq("S03", "小盘 COMBO full 与分段链差 2.72pp/年",
       {"full": 0.031776, "chain": 0.0589544, "gap_pp": 2.72, "trades_full": 733},
       {"full": round(float(mx["small_combo"]["full"]["cagr"]), 6),
        "chain": round(float(mx["small_combo"]["chained_dev_val_cagr"]), 7),
        "gap_pp": mx["small_combo"]["full_vs_chain_gap_pp"],
        "trades_full": int(mx["small_combo"]["full"]["n_closed"])},
       "05_审计/最终结论证据矩阵_复算.json")

    # ---- 6. 2021+ 扫描 ----
    l1 = next((c for c in acc["checks"] if c.get("id") == "L1"), {})
    scan_n = None
    if l1:
        m = __import__("re").search(r"扫描\s*(\d+)\s*个", l1.get("metric", ""))
        scan_n = int(m.group(1)) if m else None
    eq("A01", "2021+ 绩效表扫描：469 个 / 0 违规",
       {"files": 469, "violations": 0},
       {"files": scan_n, "violations": len(l1.get("evidence", {}).get("violations", [])) if l1 else None},
       "04_证据/交付验收.json L1",
       "历史一轮 470 个，差 1 个文件集；04_最终结论 已同时注明")
    eq("A02", "净值审计：418 条，max(date)=2020-12-31，0 越界",
       {"rows": 418, "max_date": "2020-12-31", "after": 0},
       {"rows": mx["nav_2021_audit"]["rows"], "max_date": mx["nav_2021_audit"]["max_date"],
        "after": mx["nav_2021_audit"]["n_after_2020"]},
       "05_审计/最终结论证据矩阵_复算.json")
    eq("A03", "总账无 2020 后带绩效行",
       {"bad": 0}, {"bad": int(((led.window_end.astype(str) > "2020-12-31") & led.cagr.notna()).sum())},
       "03_结果/最终试验总账.csv")

    # ---- 7. 文档一致性 ----
    readme = read_text("README.md")
    rep = read_text("01_完整报告.md")
    one = read_text("00_一页纸.md")
    rb = read_text("02_规则手册_研究结论版.md")
    dat = read_text("03_数据与复现说明.md")
    hist = read_text("03_结果/最终统计/00_最终报告.md")

    def has_neg(text: str, number: str = "0.03") -> bool:
        return (f"-{number}" in text) or (f"−{number}" in text)

    add("D01", "README 当前口径：733 / 1251 / 454，且 B_conditional 两个口径与 -0.03% 已写明",
        ["733", "1,251", "454", "12 行 / 8 配置", "1 行 S1", "-0.03% 或 −0.03%", "505"],
        [x for x in ["733", "1,251", "454", "12 行 / 8 配置", "1 行 S1", "505"] if x in readme]
        + (["-0.03% 或 −0.03%"] if has_neg(readme, "0.03%") else []),
        all(x in readme for x in ["733", "1,251", "454", "12 行 / 8 配置", "1 行 S1", "505"])
        and has_neg(readme, "0.03%"),
        "README.md")
    add("D02", "README 不再把 365/470/未计入 N 当作当前事实",
        {"N_sel_365_as_current": False, "470_as_current": False, "not_counted": False},
        {"N_sel_365_as_current": "N_sel=365" in readme,
         "470_as_current": "470 个绩效表 0 违规" in readme,
         "not_counted": "未计入 N" in readme},
        not any(x in readme for x in ["N_sel=365", "470 个绩效表 0 违规", "未计入 N"]),
        "README.md")
    add("D03", "01_完整报告：N 三口径 + 覆盖 505/228 + 小盘 -0.03%",
        ["365", "714", "733", "505", "228", "12 行/8 配置", "-0.03 或 −0.03"],
        [x for x in ["365", "714", "733", "505", "228", "12 行/8 配置"] if x in rep]
        + (["-0.03 或 −0.03"] if has_neg(rep) else []),
        all(x in rep for x in ["365", "714", "733", "505", "228", "12 行/8 配置"]) and has_neg(rep),
        "01_完整报告.md")
    add("D04", "00_一页纸：配置级 5 与行级 12/8 + 小盘 -0.03%",
        ["5 个", "12 行/8 配置", "-0.03 或 −0.03"],
        [x for x in ["5 个", "12 行/8 配置"] if x in one]
        + (["-0.03 或 −0.03"] if has_neg(one) else []),
        all(x in one for x in ["5 个", "12 行/8 配置"]) and has_neg(one),
        "00_一页纸.md")
    add("D05", "02_规则手册：两个口径已写明",
        ["12 行/8 配置", "-0.03 或 −0.03"],
        [x for x in ["12 行/8 配置"] if x in rb] + (["-0.03 或 −0.03"] if has_neg(rb) else []),
        ("12 行/8 配置" in rb) and has_neg(rb),
        "02_规则手册_研究结论版.md")
    add("D06", "03_数据说明：469 扫描与 19 项 verify 已更新",
        ["469", "19 项"],
        [x for x in ["469", "19 项"] if x in dat],
        all(x in dat for x in ["469", "19 项"]),
        "03_数据与复现说明.md")
    add("D07", "历史 00_最终报告已加当前口径横幅与内联历史标记",
        ["历史快照", "F_never_run（历史快照", "当前已 done", "505"],
        [x for x in ["历史快照", "F_never_run（历史快照", "当前已 done", "505"] if x in hist],
        all(x in hist for x in ["历史快照", "F_never_run（历史快照", "当前已 done", "505"]),
        "03_结果/最终统计/00_最终报告.md",
        "不删旧行/不改历史数字，只加当前口径与历史标记")

    # ---- 8. 证据矩阵交叉核对 ----
    matrix = {
        "R10_final": mx["ledger_final"],
        "R10_snapshot": mx["ledger_snapshot_0051"],
        "R09_stage1": mx["ledger_stage1"],
        "R11_overrun": mx["overrun"],
        "R12_row_level": mx["candidate_grades"],
        "R14_gates": mx["gates"],
        "R24_small_combo": mx["small_combo"],
        "R38_nav_2021": mx["nav_2021_audit"],
    }
    m_ok = (
        mx["ledger_final"]["N_sel"] == 733 and mx["ledger_final"]["N_cell"] == 1251
        and mx["ledger_final"]["N_diag"] == 454 and mx["ledger_final"]["rows"] == 1699
        and mx["ledger_snapshot_0051"]["N_sel"] == 714 and mx["ledger_snapshot_0051"]["rows"] == 1631
        and mx["ledger_stage1"]["N_sel"] == 365 and mx["ledger_stage1"]["rows"] == 875
        and mx["overrun"]["ledger_x"] == 11.4531
        and mx["candidate_grades"]["B_rows"] == 12 and mx["candidate_grades"]["B_unique_configs"] == 8
        and mx["gates"]["rows"] == 478 and mx["gates"]["pass_both_windows_mdd25"] == 5
        and mx["small_combo"]["val"]["cagr_nav_first"] == 0.00326993
        and mx["small_combo"]["val"]["cagr_initial_capital"] == -0.00029673
        and mx["nav_2021_audit"]["rows"] == 418 and mx["nav_2021_audit"]["max_date"] == "2020-12-31"
    )
    add("M01", "证据矩阵复算值与当前机器表一致", True, m_ok, m_ok,
        "05_审计/最终结论证据矩阵_复算.json",
        "矩阵 P03/P05 指出的 486 覆盖已在最终判定改为 505；其余 478/5/12/8/733 均一致")

    # ---- 9. 清单与交付物 ----
    inv = pd.read_csv(p("文件清单.csv"))
    inv_paths = set(inv.path.astype(str))
    need = ["00_一页纸.md", "01_完整报告.md", "02_规则手册_研究结论版.md",
            "03_数据与复现说明.md", "04_最终结论.json", "README.md",
            "03_结果/最终统计/最终判定.json", "03_结果/最终统计/00_最终报告.md",
            "04_证据/最终口径对账.json", "02_代码/最终口径对账.py"]
    missing = [x for x in need if x not in inv_paths]
    add("L01", "文件清单包含本次修改的关键文件", [], missing, not missing,
        "文件清单.csv")
    dup = int(inv.path.duplicated().sum())
    add("L02", "文件清单无重复路径，且行数 > 300",
        {"duplicates": 0, "rows_min": 300},
        {"duplicates": dup, "rows": int(len(inv))},
        dup == 0 and len(inv) > 300,
        "文件清单.csv")

    # ---- 汇总 ----
    npass = sum(1 for c in CHECKS if c["pass"])
    out = {
        "schema": "trend-structure-final-reconciliation/v1",
        "meta": {
            "project": "trading_趋势结构",
            "generated_at_local": datetime.now().isoformat(timespec="seconds"),
            "role": "最终交付口径对账（只读；不重跑策略；不计算 2021+ 收益）",
            "data_version": "v2-adjspace-2026-10-28",
            "freeze_date": "2026-10-26",
            "ledger_rows": counts["rows"],
        },
        "conclusion": {
            "n_candidates": 0,
            "annualized_60pct_supported": False,
            "holdout_run": False,
            "main_candidate": "T0005 discard（dev +4.16% / val -2.53%）",
            "n_sel": 733,
            "n_cell": 1251,
            "n_diag": 454,
        },
        "checks": CHECKS,
        "summary": {"n_checks": len(CHECKS), "n_pass": npass, "n_fail": len(CHECKS) - npass},
        "fixes": [
            {"file": "00_一页纸.md", "change": "新增 B_conditional 配置级 5/行级 12-8；小盘 COMBO val 首日 +0.33% 与初始 100 万 -0.03% 双基准；交付清单加对账文件"},
            {"file": "01_完整报告.md", "change": "新增 N 三口径、B_conditional 两口径、门槛 478→505/228、小盘 full/分段链 2.72pp 与笔数 709/728/733；verify/selfcheck 改为 19/36 项"},
            {"file": "02_规则手册_研究结论版.md", "change": "新增“两个容易误引的口径”框（B_conditional 5 vs 12/8；小盘 val -0.03%）"},
            {"file": "03_数据与复现说明.md", "change": "2021+ 扫描统一为 469（470 为历史文件集）；verify 19 项/selfcheck 36 项；文件清单 343 行；加对账命令"},
            {"file": "04_最终结论.json", "change": "新增 candidate_grading（5/10 vs 12/8）与 holdout.gate_coverage（478/505/228/733）；小盘 val 双基准；limits 补 N/B_conditional/小盘/T0005 历史行"},
            {"file": "README.md", "change": "§1.1 改为 N=733/1251/454、DSR 6.63e-4、B_conditional 两口径、T0005 已完成；§1.2/§5/§8/§10 的小盘基准与 469 扫描；文件清单 343 行、脚本 44 个"},
            {"file": "03_结果/最终统计/00_最终报告.md", "change": "顶部加“当前口径”横幅与“以下为 00:51 历史正文”分隔；正文 T0005 旧状态/486 覆盖/小盘 +0.33 加历史标记，不删旧行、不改历史数字"},
            {"file": "03_结果/最终统计/最终判定.json", "change": "N_sel_covered_by_gate_rows 486→505；加 N_sel_unrated 228/N_sel_total 733/coverage_note/coverage_derivation；stage1 对比改 365→733/500→1251/369→454；加 grade_calibers"},
            {"file": "03_结果/最终统计/T0005_纳入后_汇总.json", "change": "加 gate_coverage（478/505/228/733）与 candidate_grading 两口径"},
            {"file": "02_代码/最终交付.py", "change": "stage_final_stats 重算覆盖与 B_conditional 口径；报告横幅改为当前口径；verify 加 holdout_gate_coverage（19 项）；selfcheck 改精确计数并加 final_calibers_consistent（36 项）"},
            {"file": "02_代码/最终口径对账.py", "change": "新增只读对账脚本（35 项），输出 04_证据/最终口径对账.json"},
            {"file": "02_代码/交付验收.py", "change": "inventory 纳入 00_一页纸/01_完整报告/02_规则手册/03_数据说明/04_最终结论 与 03_结果/最终统计 的 JSON/CSV（文件清单 343 行）"},
            {"file": "文件清单.csv", "change": "由 交付验收.py 重建，343 行；包含最终交付文档、最终统计与对账文件"},
        ],
        "verification": {
            "experiment_registry_verify": "OK（2 WARN 代码漂移 02_代码/实验登记.py；哈希链/窗口/指标/计数/留出一致性通过）",
            "final_delivery_verify": "19/19 PASS",
            "final_delivery_selfcheck": "36/36 PASS",
            "delivery_acceptance_full": "29 PASS / 0 FAIL / 1 WARN（30 项；WARN=T2 预注册四列）",
            "evidence_matrix_recompute": "05_审计/最终结论证据矩阵_复算.py 运行 OK；R10/R11/R12/R14/R24/R38 与当前表一致",
            "reconciliation": f"{npass}/{len(CHECKS)} PASS",
        },
        "computed": {
            "counts": counts,
            "gate_coverage": {"n_gate_rows": int(len(gates)), "covered": cover,
                              "unrated": unrated, "total": counts["N_sel"],
                              "identity": f"{cover}+{unrated}={cover + unrated}"},
            "b_conditional": {
                "config_level": {"n_configs": int(b_cfg.config_id.nunique()), "n_ledger_rows": b_ledger_rows,
                                 "trend_s3": int(b_cfg.strategy_id.str.startswith("TREND@S3").sum()),
                                 "vcb": int((b_cfg.family == "VCB").sum())},
                "row_level_6_families": {"n_rows": int(len(b_row)), "n_configs": int(b_row.config_id.nunique()),
                                         "s3_rows": int((b_row.strategy_id == "TREND@S3_波动收缩突破").sum()),
                                         "s1_rows": int((b_row.strategy_id == "TREND@S1_月周日共振突破").sum())},
            },
            "t0005": {"done_rows": int(len(done)), "N_sel": int(done.in_Nsel.sum()),
                      "N_cell": int(done.in_Ncell.sum()), "N_diag": int(done.in_Ndiag.sum()),
                      "stale_rows": sorted(led[led.status != "done"].ledger_id.tolist())},
            "small_combo": combo,
            "scan_2021": {"files": scan_n, "violations": len(l1.get("evidence", {}).get("violations", [])) if l1 else None},
        },
        "evidence_matrix_crosscheck": matrix,
        "not_recomputed": [
            "未运行任何策略回测；未重跑 2021+ 收益。",
            "未修改源数据、注册表、总账历史行或各族交易结果。",
            "未重跑 02_代码/最终试验总账.py --stage all（该脚本不含 T0005，会覆盖最终统计聚合）。",
            "门槛覆盖、B_conditional、小盘基准均从已落盘 CSV/JSON 重算；未改族报告。",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"最终口径对账：{npass}/{len(CHECKS)} PASS -> {OUT}")
    for c in CHECKS:
        if not c["pass"]:
            print("FAIL", c["id"], c["item"], "expected=", c["expected"], "actual=", c["actual"])
    return out


if __name__ == "__main__":
    main()
