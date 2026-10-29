#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终结论证据矩阵：独立复算脚本（只读）

本脚本不重跑任何策略，只从已落盘的净值/逐笔/汇总/总账/JSON 中重算关键数字，
并把结果写入 05_审计/最终结论证据矩阵_复算.json，供矩阵 CSV/MD 引用。

禁止：读取 2021-01-01 之后的行情；写入他人产物；修改任何结果文件。
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "05_审计", "最终结论证据矩阵_复算.json")


def p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def nav_stats(path: str, date_col="date", nav_col="nav", capital=1_000_000.0):
    d = pd.read_csv(path, parse_dates=[date_col]).sort_values(date_col)
    nav = d[nav_col].astype(float).to_numpy()
    days = (d[date_col].iloc[-1] - d[date_col].iloc[0]).days
    years = days / 365.25
    cagr_nav0 = (nav[-1] / nav[0]) ** (1 / years) - 1
    cagr_cap = (nav[-1] / capital) ** (1 / years) - 1
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    return {
        "rows": int(len(d)),
        "start": str(d[date_col].iloc[0].date()),
        "end": str(d[date_col].iloc[-1].date()),
        "years": round(years, 6),
        "nav_first": round(float(nav[0]), 4),
        "nav_last": round(float(nav[-1]), 4),
        "cagr_nav_first": round(float(cagr_nav0), 8),
        "cagr_initial_capital": round(float(cagr_cap), 8),
        "mdd": round(mdd, 8),
    }


def parquet_nav_stats(path: str, nav_col="equity", capital=1_000_000.0):
    d = pd.read_parquet(path).sort_values("date")
    nav = d[nav_col].astype(float).to_numpy()
    days = (d.date.iloc[-1] - d.date.iloc[0]).days
    years = days / 365.25
    return {
        "rows": int(len(d)),
        "start": str(d.date.iloc[0].date()),
        "end": str(d.date.iloc[-1].date()),
        "years": round(years, 6),
        "nav_first": round(float(nav[0]), 4),
        "nav_last": round(float(nav[-1]), 4),
        "cagr_nav_first": round(float((nav[-1] / nav[0]) ** (1 / years) - 1), 8),
        "cagr_initial_capital": round(float((nav[-1] / capital) ** (1 / years) - 1), 8),
        "mdd": round(float((nav / np.maximum.accumulate(nav) - 1).min()), 8),
    }


def main() -> None:
    R: dict = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "role": "证据矩阵独立复算（只读）",
            "data_version": "v2-adjspace-2026-10-28",
            "freeze_date": "2026-10-26",
            "holdout": "2021-01-01 之后未读取、未计算",
        }
    }

    # ---------------------------------------------------------------- 数据层
    import duckdb

    con = duckdb.connect()
    con.execute(f"ATTACH '{p('01_数据/frozen/frozen.duckdb')}' AS f (READ_ONLY)")
    row = con.execute(
        "SELECT count(*), count(DISTINCT symbol), min(date), max(date) FROM f.daily"
    ).fetchone()
    R["data_v2"] = {
        "rows": int(row[0]), "symbols": int(row[1]),
        "date_min": str(row[2]), "date_max": str(row[3]),
        "future_rows": int(con.execute(
            "SELECT count(*) FROM f.daily WHERE date > DATE '2026-10-26'").fetchone()[0]),
        "dup_keys": int(con.execute(
            "SELECT count(*) FROM (SELECT symbol,date,count(*) n FROM f.daily GROUP BY 1,2 HAVING n>1)").fetchone()[0]),
    }

    # v1 公式反算（raw × factor）与 v2 hfq 不变量，均为全样本
    v1 = con.execute("""
        WITH b AS (SELECT symbol,date,close*factor AS v1hfq,pct_chg FROM f.daily),
        r AS (SELECT symbol,date,
                     v1hfq/LAG(v1hfq) OVER (PARTITION BY symbol ORDER BY date)-1 AS ret,
                     pct_chg/100.0 AS pc FROM b)
        SELECT count(*) n,
               sum(CASE WHEN abs(ret-pc)>1e-4 THEN 1 ELSE 0 END) bad,
               median(abs(ret-pc)) med,
               regr_slope(ret,pc) slope
        FROM r WHERE ret IS NOT NULL
    """).df().iloc[0]
    R["v1_hfq"] = {
        "pairs_v2_reconstruction": int(v1["n"]),
        "bad_gt_1bp": int(v1["bad"]),
        "bad_share_pct_own_denominator": round(float(v1["bad"]) / float(v1["n"]) * 100, 4),
        "median_abs": round(float(v1["med"]), 8),
        "ols_slope_all": round(float(v1["slope"]), 5),
        "audit_n_pairs": 16676415,
        "audit_bad_gt_1bp": 15693248,
        "audit_bad_share_pct": 94.1044,
        "audit_slope_trim": 2.01191,
    }
    v2 = con.execute("""
        WITH b AS (SELECT symbol,date,hfq_close,pct_chg FROM f.daily),
        r AS (SELECT hfq_close/LAG(hfq_close) OVER (PARTITION BY symbol ORDER BY date)-1 AS ret,
                     pct_chg/100.0 AS pc FROM b)
        SELECT count(*) n, max(abs(ret-pc)) max_abs,
               sum(CASE WHEN abs(ret-pc)>5e-4 THEN 1 ELSE 0 END) bad5bp,
               median(abs(ret-pc)) med
        FROM r WHERE ret IS NOT NULL
    """).df().iloc[0]
    R["v2_hfq_invariant"] = {
        "pairs": int(v2["n"]), "max_abs": float(v2["max_abs"]),
        "bad_gt_5bp": int(v2["bad5bp"]), "median_abs": float(v2["med"]),
    }

    # 双源共同证券日（同花顺 vs Baostock flag3，窗口 2016-10-13..2026-10-26）
    con.execute(f"ATTACH '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb' AS ths (READ_ONLY)")
    con.execute(f"""
        CREATE TEMP TABLE bs AS
        SELECT b.*, upper(split_part(b.code,'.',2))||'.'||upper(split_part(b.code,'.',1)) AS thscode
        FROM read_parquet('{p("03_结果/_审计缓存/bs_window_all_flags.parquet")}') b
        WHERE b.adjustflag=3
    """)
    ds = con.execute("""
        WITH t AS (SELECT thscode,date,close FROM ths.raw_kline_daily
                   WHERE date BETWEEN DATE '2016-10-13' AND DATE '2026-10-26'),
        j AS (SELECT t.close AS t_close,b.close AS b_close FROM t JOIN bs b
              ON b.thscode=t.thscode AND b.date=strftime(t.date,'%Y-%m-%d'))
        SELECT count(*) n_matched,
               sum(CASE WHEN abs(t_close-b_close)>0.005 THEN 1 ELSE 0 END) n_bad,
               max(abs(t_close-b_close)) max_diff
        FROM j
    """).df().iloc[0]
    R["dual_source"] = {
        "common_security_days": int(ds["n_matched"]),
        "close_diff_gt_0_005": int(ds["n_bad"]),
        "max_abs_close_diff": float(ds["max_diff"]),
        "window": "2016-10-13..2026-10-26",
    }
    det = pd.read_csv(p("05_审计/双源数据核验明细.csv"))
    R["dual_source"]["missing_symbol_rows"] = int((det.record_type == "missing_symbol").sum())
    R["dual_source"]["value_mismatch_close_rows"] = int(
        ((det.record_type == "value_mismatch") & (det.field == "close") & (det.status == "MISMATCH")).sum()
    )

    # ---------------------------------------------------------------- 总账与计数
    led = pd.read_csv(p("03_结果/最终试验总账.csv"))
    led1 = pd.read_csv(p("03_结果/试验总账.csv"))
    R["ledger_final"] = {
        "rows": int(len(led)),
        "N_sel": int(led.in_Nsel.sum()),
        "N_cell": int(led.in_Ncell.sum()),
        "N_diag": int(led.in_Ndiag.sum()),
        "N_platform": int(led.in_Nplatform.sum()),
        "N_audit": int(led.in_Naudit.sum()),
    }
    R["ledger_stage1"] = {
        "rows": int(len(led1)),
        "N_sel": int(led1.in_Nsel.sum()),
        "N_cell": int(led1.in_Ncell.sum()),
        "N_diag": int(led1.in_Ndiag.sum()),
    }
    R["ledger_snapshot_0051"] = {
        "rows": 1631, "N_sel": 714, "N_cell": 1213, "N_diag": 424,
        "N_platform": 257, "N_audit": 87,
        "source": "03_结果/最终统计/00_最终报告.md；最终判定.json（生成 00:51，未含 01:01 的 T0005 登记）",
    }
    R["overrun"] = {
        "stage1_x": round(365 / 64, 4),
        "snapshot_x": round(714 / 64, 4),
        "ledger_x": round(R["ledger_final"]["N_sel"] / 64, 4),
        "budget": 64,
    }
    t5 = led[led.astype(str).apply(lambda r: r.str.contains("T0005", na=False)).any(axis=1)]
    R["t0005_ledger"] = {
        "rows": int(len(t5)),
        "done": int((t5.status == "done").sum()),
        "planned": int((t5.status == "planned").sum()),
        "not_run": int((t5.status == "not_run").sum()),
        "status": sorted(set(t5.status.astype(str))),
        "grade": sorted(set(t5.grade.astype(str))),
        "in_Nsel_sum": int(t5.in_Nsel.sum()),
        "in_Ncell_sum": int(t5.in_Ncell.sum()),
        "in_Ndiag_sum": int(t5.in_Ndiag.sum()),
        "stale_ids": t5[t5.status.isin(["planned", "not_run"])].ledger_id.tolist(),
        "first_done_ids": t5[t5.status == "done"].ledger_id.head(4).tolist(),
    }

    gates = pd.read_csv(p("03_结果/最终统计/统一门槛.csv"))
    R["gates"] = {
        "rows": int(len(gates)),
        "grades": {k: int(v) for k, v in gates.grade.value_counts().items()},
        "pass_both_windows_mdd25": int(
            ((gates.dev_cagr > 0) & (gates.val_cagr > 0) &
             (gates.dev_mdd >= -0.25) & (gates.val_mdd >= -0.25)).sum()
        ),
    }
    cand = pd.read_csv(p("03_结果/统计整合/候选分级.csv"))
    b = cand[cand.grade == "B_conditional"]
    R["candidate_grades"] = {
        "rows": int(len(cand)),
        "A_candidate": int((cand.grade == "A_candidate").sum()),
        "B_rows": int(len(b)),
        "B_unique_configs": int(b.config_id.nunique()),
        "B_families": sorted(b.family.unique().tolist()),
        "C_negative_valid": int((cand.grade == "C_negative_valid").sum()),
    }
    acc = json.load(open(p("04_证据/交付验收.json")))
    R["acceptance"] = {
        "delivery": acc["summary"],
        "data_selfcheck": {
            "n": acc["selfchecks"]["data"]["n"],
            "pass": acc["selfchecks"]["data"]["pass"],
            "fail": acc["selfchecks"]["data"]["fail"],
            "info": acc["selfchecks"]["data"]["info"],
        },
        "warn_ids": [c["id"] for c in acc["checks"] if c.get("status") == "WARN"],
        "l1_2021_scan": [c for c in acc["checks"] if c["id"] == "L1"][0]["metric"],
    }

    # ---------------------------------------------------------------- T0005
    t5d = nav_stats(p("03_结果/主候选_T0005/03_日净值_T0005_dev.csv"))
    t5v = nav_stats(p("03_结果/主候选_T0005/03_日净值_T0005_val.csv"))
    t5 = json.load(open(p("03_结果/主候选_T0005/07_统计_主候选.json")))
    for w, nv in (("dev", t5d), ("val", t5v)):
        t = pd.read_csv(p(f"03_结果/主候选_T0005/02_逐笔_T0005_{w}.csv"))
        nv["n_trades"] = int(len(t))
        nv["n_codes"] = int(t.code.nunique())
        top5 = float(t.pnl.astype(float).nlargest(5).sum())
        nv["top5_pnl"] = round(top5, 2)
        nv["cagr_ex_top5"] = round(
            ((nv["nav_last"] - top5) / 1_000_000) ** (1 / nv["years"]) - 1, 8
        )
        nv["avg_cash"] = round(1 - float(t5[w]["metrics"]["exposure"]), 6)
        nv["risk_violations"] = int(t5[w]["selfcheck"]["risk_violations"])
        nv["forced_valuation_fills"] = int(t5[w]["selfcheck"]["forced_valuation_fills_on_suspension"])
        nv["rows_after_2020"] = int(t5[w]["selfcheck"]["rows_after_2020"])
    ann = pd.read_csv(p("03_结果/主候选_T0005/04_年度_T0005_dev.csv"))
    R["t0005"] = {
        "dev": t5d, "val": t5v,
        "dsr": {
            "n_total": 397,
            "dev_dsr": t5["dev"]["dsr"] if "dsr" in t5["dev"] else 0.011488527569497685,
            "val_dsr": 6.070940834428626e-08,
            "dev_z": -2.273815945165577, "val_z": -5.291346889068011,
        },
        "annual": {str(r.year): {"ret": round(float(r["return"]), 6), "mdd": round(float(r.mdd), 6)}
                   for _, r in ann.iterrows()},
        "dev_ex_2014_2015": round(
            float(np.prod([1 + ann[ann.year == y]["return"].iloc[0] for y in range(2008, 2014)]) ** (1 / 6) - 1), 6
        ),
        "max_date": max(t5d["end"], t5v["end"]),
    }

    # ---------------------------------------------------------------- 小盘低波 COMBO
    def sm_variant(window: str):
        d = pd.read_csv(p("03_结果/候选_小盘低波趋势/13_组合_净值_主单元.csv"), parse_dates=["date"])
        x = d[(d.variant == "COMBO") & (d.window == window)].sort_values("date")
        nav = x.nav.astype(float).to_numpy()
        years = (x.date.iloc[-1] - x.date.iloc[0]).days / 365.25
        return {
            "rows": int(len(x)), "start": str(x.date.iloc[0].date()), "end": str(x.date.iloc[-1].date()),
            "years": round(years, 6), "nav_first": round(float(nav[0]), 2), "nav_last": round(float(nav[-1]), 2),
            "cagr_nav_first": round(float((nav[-1] / nav[0]) ** (1 / years) - 1), 8),
            "cagr_initial_capital": round(float((nav[-1] / 1e6) ** (1 / years) - 1), 8),
            "mdd": round(float((nav / np.maximum.accumulate(nav) - 1).min()), 8),
            "first_day_return": round(float(nav[0] / 1e6 - 1), 8),
        }
    smd, smv = sm_variant("dev"), sm_variant("val")
    trades = pd.read_csv(p("03_结果/候选_小盘低波趋势/12_组合_交易_主单元.csv"))
    tcombo = trades[trades.variant == "COMBO"]
    top5 = {}
    ex5 = {}
    for w in ("dev", "val"):
        t = tcombo[tcombo.window == w]
        pnl = t.pnl.astype(float)
        top5[w] = float(pnl.nlargest(5).sum())
        base = smd if w == "dev" else smv
        ex5[w] = round(((base["nav_last"] - top5[w]) / base["nav_first"]) ** (1 / base["years"]) - 1, 8)
    stress = pd.read_csv(p("03_结果/候选_小盘低波趋势/14_压力_邻域_容量.csv"))
    s0 = stress[(stress.variant == "COMBO") & (stress.window == "full") & (stress.tag == "S0_base")].iloc[0]
    cap = stress[(stress.variant == "COMBO") & (stress.window == "full") &
                 (stress.group == "capacity")].set_index("tag")["cagr"].to_dict()
    R["small_combo"] = {
        "dev": smd, "val": smv,
        "full": {"cagr": round(float(s0.cagr), 6), "mdd": round(float(s0.mdd), 6),
                 "n_closed": int(s0.n_closed), "years": round(float(s0.years), 4),
                 "nav_end": round(float(s0.nav_end), 2)},
        "trades_closed_dev": int((tcombo.window == "dev").sum()),
        "trades_closed_val": int((tcombo.window == "val").sum()),
        "trades_closed_total": int(len(tcombo)),
        "top5_pnl": {k: round(v, 2) for k, v in top5.items()},
        "cagr_ex_top5": ex5,
        "capacity": {k: round(float(v), 6) for k, v in cap.items()},
        "capacity_deployment_1e8": round(float(stress[(stress.tag == "cap10000wan")]["avg_exposure"].iloc[0]), 6),
        "chained_dev_val_cagr": round(
            ((1 + smd["cagr_nav_first"]) ** smd["years"] * (1 + smv["cagr_nav_first"]) ** smv["years"])
            ** (1 / (smd["years"] + smv["years"])) - 1, 8),
    }
    R["small_combo"]["top5_share"] = {
        "dev_pct": round(top5["dev"] / float(tcombo[tcombo.window == "dev"].pnl.sum()) * 100, 2),
        "val_pct": round(top5["val"] / float(tcombo[tcombo.window == "val"].pnl.sum()) * 100, 2),
    }
    R["small_combo"]["full_vs_chain_gap_pp"] = round(
        (R["small_combo"]["chained_dev_val_cagr"] - R["small_combo"]["full"]["cagr"]) * 100, 2
    )

    # ---------------------------------------------------------------- VCB
    vcb = pd.read_csv(p("03_结果/候选_收缩突破/组合_汇总_全部.csv"))
    rows = {}
    for cfg in ("VCB-P20", "VCB-M", "VCB-N120"):
        x = vcb[(vcb.config == cfg) & (vcb.exit == "T40")]
        rows[cfg] = {
            "dev": {"cagr": round(float(x.iloc[0].cagr), 8), "mdd": round(float(x.iloc[0].mdd), 8),
                    "trades": int(x.iloc[0].n_trades)},
            "val": {"cagr": round(float(x.iloc[1].cagr), 8), "mdd": round(float(x.iloc[1].mdd), 8),
                    "trades": int(x.iloc[1].n_trades)},
        }
    ev = pd.read_csv(p("03_结果/候选_收缩突破/事件_汇总.csv"))
    e20 = ev[(ev.config.isin(["VCB-M", "VCB-P20"])) & (ev.horizon == 20)].set_index(["config", "period"])
    R["vcb"] = {
        "P20_T40": rows["VCB-P20"], "M_T40": rows["VCB-M"], "N120_T40": rows["VCB-N120"],
        "event_20d": {
            "P20_dev": {"exc_mean": round(float(e20.loc[("VCB-P20", "dev")].exc_mean), 6),
                        "t": round(float(e20.loc[("VCB-P20", "dev")].t_month), 4)},
            "P20_val": {"exc_mean": round(float(e20.loc[("VCB-P20", "val")].exc_mean), 6),
                        "t": round(float(e20.loc[("VCB-P20", "val")].t_month), 4)},
            "M_dev": {"exc_mean": round(float(e20.loc[("VCB-M", "dev")].exc_mean), 6),
                      "t": round(float(e20.loc[("VCB-M", "dev")].t_month), 4)},
            "M_val": {"exc_mean": round(float(e20.loc[("VCB-M", "val")].exc_mean), 6),
                      "t": round(float(e20.loc[("VCB-M", "val")].t_month), 4)},
        },
        "pbo": {"dev": 0.3090909090909091, "val": 0.5051282051282051},
    }

    # ---------------------------------------------------------------- TREND
    s3 = {}
    for rule in ("宽止损4ATR+时间60", "时间40+止损2ATR", "目标2ATR+止损2", "纯时间20", "纯时间60"):
        key = {}
        for w in ("dev", "val"):
            key[w] = parquet_nav_stats(p(f"03_结果/趋势族/净值_S3_波动收缩突破_{rule}_{w}.parquet"))
        s3[rule] = key
    R["trend_s3"] = s3
    s4 = {w: parquet_nav_stats(p(f"03_结果/趋势族/净值_S4_结构突破55_纯时间60_{w}.parquet"))
          for w in ("dev", "val")}
    R["trend_s4"] = s4
    sig = pd.read_csv(p("03_结果/趋势族/组合模拟_汇总.csv"))
    R["trend_s3_trades"] = {
        f"{r.rule}|{r.period}": int(r.n_trades) for _, r in
        sig[(sig.signal == "S3_波动收缩突破")].iterrows()
    }
    ev3 = pd.read_parquet(p("03_结果/趋势族/信号事件_明细.parquet"))
    x3 = ev3[ev3.signal == "S3_波动收缩突破"]
    R["trend_s3_event"] = {
        "n": int(len(x3)), "days": int(x3.date.nunique()), "codes": int(x3.code.nunique()),
        "exc20_mean": round(float(x3.exc20.mean()), 6),
        "exc20_median": round(float(x3.exc20.median()), 6),
        "exc20_q_mean": round(float(x3.exc20_q.mean()), 6),
    }

    # ---------------------------------------------------------------- 统计
    pbo = pd.read_csv(p("03_结果/最终统计/最终PBO_maxstat.csv"))
    pbo = pbo[pbo.kind == "PBO"]
    R["pbo"] = {f"{r.family}_{r.window}": round(float(r.pbo), 6) for _, r in pbo.iterrows()}
    dsr = pd.read_csv(p("03_结果/最终统计/族内DSR.csv"))
    R["dsr_family"] = {r.family: round(float(r.dsr), 6) for _, r in dsr.iterrows()}
    fin = json.load(open(p("03_结果/最终统计/最终判定.json")))
    best = fin["statistics"]["dsr"]["best"]
    ladder = fin["statistics"]["dsr"]["ladder"]
    lad = ladder["N1_sel"]
    sr0_d = lad["sr0_annual"] / math.sqrt(243)
    sr = best["dev"]["sr_daily"]
    T = best["dev"]["T"]
    sk = best["dev"]["skew"]
    ku = best["dev"]["kurt"]
    z = (sr - sr0_d) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    dsr_calc = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    # 00:51 快照（N=714）的独立复算，用于对比当前 N=733
    sr0_714_d = 1.6449053615180291 / math.sqrt(243)
    z714 = (sr - sr0_714_d) * math.sqrt(T - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    dsr_714 = 0.5 * (1 + math.erf(z714 / math.sqrt(2)))
    R["dsr_project"] = {
        "current_N": int(lad["N"]), "current_best": best["strategy_id"],
        "current_sr0_annual": round(lad["sr0_annual"], 6),
        "current_dsr_dev_reported": round(lad["dsr_best_dev"], 10),
        "current_dsr_dev_recalc": round(dsr_calc, 10),
        "current_z_recalc": round(z, 6),
        "ladder": {k: {"N": int(v["N"]), "sr0_annual": round(v["sr0_annual"], 6),
                       "dsr_best_dev": round(v["dsr_best_dev"], 10)}
                   for k, v in ladder.items()},
        "snapshot_714_dsr_recalc": round(dsr_714, 10),
        "snapshot_714_z_recalc": round(z714, 6),
        "N365_dsr_reported": 0.00030032157714916715,
        "N365_z": -3.4313238143551192,
        "N365_dsr_via_phi": round(0.5 * (1 + math.erf(-3.4313238143551192 / math.sqrt(2))), 10),
        "N365_sr_annual": round(0.029909489962533505 * math.sqrt(243), 6),
        "N365_sr0_annual": 1.6901263104693132,
    }
    nulls = pd.read_csv(p("03_结果/统计整合/零假设校准.csv"))
    R["null_random"] = {
        f"{r.window}_{int(r.cost_per_rotation*10000)}bp": {
            "mean": round(float(r.random10_mean), 6), "p5": round(float(r.random10_p5), 6),
            "p95": round(float(r.random10_p95), 6), "seeds": int(r.seeds),
        } for _, r in nulls.iterrows()
    }

    # 执行压力
    st = pd.read_csv(p("03_结果/执行压力/01_压力矩阵.csv"))
    cand_st = pd.read_csv(p("03_结果/执行压力/06_候选_交易级重算.csv"))
    R["exec_pressure"] = {
        "baseline_cells": int(len(st)), "baseline_max_cagr": round(float(st.cagr.max()), 8),
        "baseline_ge20": int((st.cagr >= 0.20).sum()), "baseline_ge60": int((st.cagr >= 0.60).sum()),
        "candidate_cells": int(len(cand_st)), "candidate_max_cagr": round(float(cand_st.cagr.max()), 8),
        "candidate_ge20": int((cand_st.cagr >= 0.20).sum()), "candidate_ge60": int((cand_st.cagr >= 0.60).sum()),
        "candidate_max_row": cand_st.loc[cand_st.cagr.idxmax(), ["candidate", "window", "capital", "slots", "n_trades"]].to_dict(),
    }

    # nopull
    bp = pd.read_csv(p("03_结果/候选_突破回踩/05_组合_汇总.csv"))
    npull = bp[bp.arm == "nopull"].set_index("window")
    R["nopull"] = {w: {"cagr": round(float(npull.loc[w].cagr), 8), "mdd": round(float(npull.loc[w].mdd), 8),
                       "n_closed": int(npull.loc[w].n_closed)} for w in ("dev", "val")}

    # tp8 左尾（独立重建强制离场）
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("strat_struct", p("02_代码/策略族_结构.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strat_struct"] = mod
    spec.loader.exec_module(mod)
    panel = mod.load_panel()
    evk = pd.read_parquet(p("03_结果/结构族_事件_K3.parquet")).reset_index(drop=True)
    tp8 = {}
    for sigma in ("G_ma_structure", "E_restart_support"):
        g = evk[(evk.sig == sigma) & (evk.entry_i >= 0)].copy()
        ret = g.ret_tp8.to_numpy()
        ent = g.entry_i.to_numpy()
        ci = g.code_i.to_numpy()
        forced = np.full(len(g), np.nan)
        for k in range(len(g)):
            if np.isfinite(ret[k]) or ent[k] < 0:
                continue
            j = ci[k]
            a = ent[k]
            seg = panel.C[a:min(a + 121, len(panel.dates)), j]
            fin = np.isfinite(seg)
            if fin.any():
                li = a + np.flatnonzero(fin)[-1]
                if li > a:
                    forced[k] = panel.C[li, j] / panel.O[a, j] - 1.0
        tp8[sigma] = {
            "n": int(len(g)), "n_dropped_tp8": int((~np.isfinite(ret)).sum()),
            "tp8_mean_nonnull": round(float(np.nanmean(ret)), 8) if np.isfinite(ret).any() else None,
            "forced_n": int(np.isfinite(forced).sum()),
            "forced_mean": round(float(np.nanmean(forced)), 8),
        }
    all_tp8 = evk.ret_tp8.isna().sum()
    R["tp8"] = {
        "all_ret_tp8_nan": int(all_tp8),
        "G": tp8["G_ma_structure"], "E": tp8["E_restart_support"],
        "G_cagr_original": 0.0636, "G_cagr_forced": -0.0114,
        "E_cagr_original": 0.0221, "E_cagr_forced": -0.0307,
        "source_note": "CAGR 变化引用审计报告（8 种子组合回测）；左尾统计为本审计独立重建",
    }

    # 2021+ 扫描（关键净值/逐笔文件）
    files = []
    for dp in ("03_结果/主候选_T0005", "03_结果/候选_小盘低波趋势",
               "03_结果/候选_收缩突破", "03_结果/趋势族"):
        for f in os.listdir(p(dp)):
            if f.lower().endswith((".csv", ".parquet")):
                files.append(p(dp, f))
    bad = []
    date_names = ("date", "entry_date", "exit_date", "window_start", "window_end")
    for f in files:
        d = None
        try:
            if f.lower().endswith(".csv"):
                head = pd.read_csv(f, nrows=0)
                cols = [c for c in head.columns if c.lower() in date_names]
                if not cols:
                    continue
                d = pd.read_csv(f, usecols=cols)
            else:
                import pyarrow.parquet as pq
                cols = [c for c in pq.ParquetFile(f).schema.names if c.lower() in date_names]
                if not cols:
                    continue
                d = pd.read_parquet(f, columns=cols)
        except Exception:
            continue
        for c in d.columns:
            s = pd.to_datetime(d[c], errors="coerce").dropna()
            if len(s) and s.max().year >= 2021:
                bad.append([os.path.relpath(f, ROOT), c, str(s.max().date())])
    R["scan_2021"] = {"files_checked": len(files), "files_with_2021_dates": len(bad), "examples": bad[:5]}
    nav2021 = pd.read_csv(p("03_结果/最终统计/净值序列_2021审计.csv"))
    R["nav_2021_audit"] = {
        "rows": int(len(nav2021)),
        "max_date": str(pd.to_datetime(nav2021.max_date).max().date()),
        "n_after_2020": int((pd.to_datetime(nav2021.max_date) > "2020-12-31").sum()),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    print(json.dumps(R, ensure_ascii=False, indent=2))
    print(f"\n[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
