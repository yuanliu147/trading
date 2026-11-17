#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F09a 早期系统审计复算（只读旧文件，只写本目录 CSV）

覆盖：
1) trading_hithink 初始研究 final_summary + 各阶段 daily_equity 复算 CAGR/MDD
2) trading_hithink flash 审计修复 O03 全窗与 2024-2026 子段
3) 阶梯双K 书版逐笔毛/净 PF、平均 R、尾部集中度
4) 阶梯双K 原 1 万账户净值复算
5) 阶梯双K 修正版组合/随机路径复算
6) trading_趋势结构 统一门槛 max dev/val、A_candidate/B_conditional、DSR/maxstat、T0005
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/admin/WorkSpace/trading")
OUT = Path(__file__).resolve().parent


def cagr_mdd(equity: pd.Series) -> tuple[float, float, float]:
    equity = equity.astype(float).dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return cagr, mdd, years


def hithink() -> None:
    base = ROOT / "trading_hithink/初始研究"
    summary = json.loads((base / "results/final_summary.json").read_text())
    rows = []
    rows.append({"item": "stage", "value": summary["strategy"]["stage"]})
    rows.append({"item": "account_variants_json", "value": summary["research_attempts"]["account_variants"]})
    rows.append({"item": "passing_development_json", "value": summary["research_attempts"]["account_variants_passing_development_gates"]})
    rows.append({"item": "validation_finalists_json", "value": summary["research_attempts"]["validation_finalists"]})
    rows.append({"item": "holdout_strategies_json", "value": summary["research_attempts"]["holdout_strategies_run_for_selection"]})
    for k, v in summary["targets"].items():
        rows.append({"item": f"targets.{k}", "value": v})
    for stage in ("development", "validation", "holdout"):
        for cap, item in summary[stage].items():
            rows.append({"item": f"{stage}.{cap}.cagr", "value": item["cagr"]})
            rows.append({"item": f"{stage}.{cap}.max_drawdown", "value": item["max_drawdown"]})
            rows.append({"item": f"{stage}.{cap}.trade_count", "value": item["trade_count"]})
            rows.append({"item": f"{stage}.{cap}.win_rate", "value": item["win_rate"]})
            rows.append({"item": f"{stage}.{cap}.top5_profit_share", "value": item["top_5_profit_share"]})
    pd.DataFrame(rows).to_csv(OUT / "F09a_hithink_final_summary.csv", index=False)

    # 独立复算各账户 daily_equity
    rows = []
    for stage in ("development", "validation", "holdout"):
        for cap in (20000, 50000):
            pattern = f"results/{stage}/accounts/deep_lowamount_h19_h19_p2_c{cap}_s10_l10/daily_equity.csv"
            matches = sorted(base.glob(pattern))
            if not matches:
                continue
            frame = pd.read_csv(matches[0], parse_dates=["date"]).set_index("date")
            cagr, mdd, years = cagr_mdd(frame["equity"])
            rows.append({
                "stage": stage, "capital": cap, "path": str(matches[0].relative_to(ROOT)),
                "start": frame.index[0].date().isoformat(), "end": frame.index[-1].date().isoformat(),
                "equity_start": frame["equity"].iloc[0], "equity_end": frame["equity"].iloc[-1],
                "cagr": cagr, "mdd": mdd, "years": years,
            })
    pd.DataFrame(rows).to_csv(OUT / "F09a_hithink_accounts_recomputed.csv", index=False)

    # 当前 all_account_trials 与交付 JSON 的计数差异
    trials = pd.read_csv(base / "results/development/all_account_trials.csv")
    pd.DataFrame([{
        "current_rows": len(trials),
        "current_passed_gates": int(trials["passed_gates"].sum()),
        "json_account_variants": summary["research_attempts"]["account_variants"],
        "json_passing": summary["research_attempts"]["account_variants_passing_development_gates"],
        "current_unique_trial_ids": trials["trial_id"].nunique(),
    }]).to_csv(OUT / "F09a_hithink_trial_count_gap.csv", index=False)


def flash() -> None:
    base = ROOT / "trading_hithink/flash 审计修复"
    summary = pd.read_csv(base / "results/summary.csv")
    rows = []
    for _, r in summary[summary["scenario"].astype(str).str.startswith("O03")].iterrows():
        rows.append({
            "scenario": r["scenario"], "kind": r["kind"], "capital": r["initial_capital"],
            "start": r["start"], "end": r["end"], "cagr": r["cagr"], "mdd": r["max_drawdown"],
            "closed_trades": r["trade_count"], "open": r["ending_open_positions"],
            "status": r["evaluation_status"],
        })
        # 2024-2026 子段
        eq_path = base / "results/accounts" / r["scenario"] / "daily_equity.csv"
        if eq_path.exists():
            eq = pd.read_csv(eq_path, parse_dates=["date"]).set_index("date")["equity"]
            sub = eq.loc["2024-01-02":]
            cagr, mdd, years = cagr_mdd(sub)
            rows[-1].update({"cagr_2024_2026": cagr, "mdd_2024_2026": mdd, "years_2024_2026": years})
    pd.DataFrame(rows).to_csv(OUT / "F09a_flash_o03.csv", index=False)


def ladder() -> None:
    pooled_path = ROOT / "analysis/ladder_doublek_trades_pooled.csv"
    d = pd.read_csv(pooled_path)
    d["notional"] = d["entry"] * d["shares"]
    d["exit_dt"] = pd.to_datetime(d["exit_date"])
    gross_pf = d.loc[d.pnl > 0, "pnl"].sum() / -d.loc[d.pnl < 0, "pnl"].sum()
    # 报告净费口径：印花税 0.1%/0.05% 分段 + 佣金万1双边；滑点已含在成交价
    stamp = np.where(d.exit_dt >= pd.Timestamp("2023-08-28"), 0.0005, 0.001)
    net = d.pnl - d.notional * (stamp + 0.0002)
    net_pf = net[net > 0].sum() / -net[net < 0].sum()
    top5 = d.nlargest(int(len(d) * 0.05), "pnl").pnl.sum()
    rows = [{
        "trades": len(d), "win_rate": (d.pnl > 0).mean(),
        "gross_pf": gross_pf, "gross_mean_R": d.R.mean(), "gross_median_R": d.R.median(),
        "net_pf": net_pf, "net_mean_R": (net / d.risk_amt).mean(),
        "net_median_R": (net / d.risk_amt).median(),
        "top5pct_profit_share_of_net": top5 / d.pnl.sum(),
        "min_R": d.R.min(), "max_R": d.R.max(),
        "mean_R_excl_worst1": d.nlargest(len(d) - 1, "R").R.mean(),
        "mean_R_excl_worst10": d.nlargest(len(d) - 10, "R").R.mean(),
        "mean_R_winsor_1_99": d.R.clip(d.R.quantile(.01), d.R.quantile(.99)).mean(),
    }]
    pd.DataFrame(rows).to_csv(OUT / "F09a_ladder_book_stats.csv", index=False)

    # 原 1 万账户净值
    eq = pd.read_csv(ROOT / "analysis/portfolio_equity_1w.csv", index_col=0, parse_dates=True).iloc[:, 0]
    cagr, mdd, years = cagr_mdd(eq)
    pd.DataFrame([{
        "path": "analysis/portfolio_equity_1w.csv",
        "start": eq.index[0].date().isoformat(), "end": eq.index[-1].date().isoformat(),
        "equity_start": eq.iloc[0], "equity_end": eq.iloc[-1],
        "total_return": eq.iloc[-1] / 10000 - 1, "cagr": cagr, "mdd": mdd, "years": years,
    }]).to_csv(OUT / "F09a_ladder_original_account.csv", index=False)

    # 修正版组合与随机路径
    comp = pd.read_csv(ROOT / "analysis/交易系统_审计修正/修正版_组合方案对比.csv")
    comp.to_csv(OUT / "F09a_ladder_corrected_portfolios.csv", index=False)
    paths = pd.read_csv(ROOT / "analysis/交易系统_审计修正/修正版_组合路径分布_ALL_2016_2026.csv")
    rnd = paths[paths.policy == "random"]
    pd.DataFrame([{
        "n_random": len(rnd),
        "cagr_p05": rnd.cagr.quantile(.05), "cagr_median": rnd.cagr.median(), "cagr_p95": rnd.cagr.quantile(.95),
        "mdd_p05": rnd.mdd.quantile(.05), "mdd_median": rnd.mdd.median(), "mdd_p95": rnd.mdd.quantile(.95),
        "final_median": rnd.final.median(), "taken_median": rnd.taken.median(),
    }]).to_csv(OUT / "F09a_ladder_random_paths.csv", index=False)

    # 全样本逐笔 CSV 现存行数（核对 --quick 覆盖）
    rows = []
    for f in sorted((ROOT / "analysis/交易系统_审计修正").glob("修正版_交易明细_*ALL_2016_2026.csv")):
        x = pd.read_csv(f)
        rows.append({"file": f.name, "rows": len(x), "symbols": x["symbol"].nunique() if "symbol" in x else None,
                     "sys": ",".join(sorted(x["sys"].unique())) if "sys" in x else None})
    pd.DataFrame(rows).to_csv(OUT / "F09a_ladder_trade_csv_audit.csv", index=False)


def trend() -> None:
    base = ROOT / "trading_趋势结构"
    gate = pd.read_csv(base / "03_结果/最终统计/统一门槛.csv")
    x = gate.dropna(subset=["dev_cagr", "val_cagr"])
    rows = []
    imax = x.dev_cagr.idxmax()
    vmax = x.val_cagr.idxmax()
    mmax = x.assign(min_cagr=x[["dev_cagr", "val_cagr"]].min(axis=1)).sort_values("min_cagr", ascending=False).iloc[0]
    rows.append({"item": "evaluable_rows", "value": len(x)})
    rows.append({"item": "max_dev_config", "value": x.loc[imax, "config_id"]})
    rows.append({"item": "max_dev_cagr", "value": x.loc[imax, "dev_cagr"]})
    rows.append({"item": "max_dev_mdd", "value": x.loc[imax, "dev_mdd"]})
    rows.append({"item": "max_val_config", "value": x.loc[vmax, "config_id"]})
    rows.append({"item": "max_val_cagr", "value": x.loc[vmax, "val_cagr"]})
    rows.append({"item": "max_val_dev_cagr", "value": x.loc[vmax, "dev_cagr"]})
    rows.append({"item": "max_val_mdd", "value": x.loc[vmax, "val_mdd"]})
    rows.append({"item": "max_min_cagr_config", "value": mmax["config_id"]})
    rows.append({"item": "max_min_cagr", "value": mmax["min_cagr"]})
    rows.append({"item": "A_candidate", "value": int((gate.grade == "A_candidate").sum())})
    rows.append({"item": "B_conditional", "value": int((gate.grade == "B_conditional").sum())})
    rows.append({"item": "grade_counts", "value": gate.grade.value_counts().to_dict()})

    judge = json.loads((base / "03_结果/最终统计/最终判定.json").read_text())
    ladder = judge["statistics"]["dsr"]["ladder"]["N1_sel"]
    rows.append({"item": "N_sel", "value": ladder["N"]})
    rows.append({"item": "DSR_N733", "value": ladder["dsr_best_dev"]})
    pvals = [v["p_maxstat"] for v in judge["statistics"]["maxstat"]]
    rows.append({"item": "maxstat_min_p", "value": min(pvals)})
    rows.append({"item": "maxstat_n", "value": len(pvals)})
    rows.append({"item": "holdout_run", "value": judge["holdout_gate"].get("run") if isinstance(judge.get("holdout_gate"), dict) else None})

    t5 = json.loads((base / "03_结果/主候选_T0005/07_统计_主候选.json").read_text())
    for stage in ("dev", "val"):
        m = t5[stage]["metrics"]
        rows.append({"item": f"T0005_{stage}_cagr", "value": m["cagr"]})
        rows.append({"item": f"T0005_{stage}_mdd", "value": m["mdd"]})
        rows.append({"item": f"T0005_{stage}_trades", "value": m["n_trades"]})
        rows.append({"item": f"T0005_{stage}_exposure", "value": m["exposure"]})
    pd.DataFrame(rows).to_csv(OUT / "F09a_trend_key_numbers.csv", index=False)

    final = json.loads((base / "04_最终结论.json").read_text())
    pd.DataFrame([{
        "verdict": final["verdict"]["code"], "n_candidates": final["verdict"]["n_candidates"],
        "holdout_run": final["holdout"]["run"], "holdout_window": final["holdout"]["window"],
        "dev_window": final["windows"]["dev"], "val_window": final["windows"]["val"],
        "evidence_level": final["windows"]["evidence_level"],
    }]).to_csv(OUT / "F09a_trend_verdict.csv", index=False)


if __name__ == "__main__":
    hithink()
    flash()
    ladder()
    trend()
    print("F09a done ->", OUT)
