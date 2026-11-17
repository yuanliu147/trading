"""Frozen robustness and mechanical gates for recovered candidate B."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import random

import pandas as pd

from .broker import run_backtest
from .data import load_panel
from .legacy_b_policy import LegacyBPolicy
from .legacy_b_runner import CANDIDATE_B
from .legacy_config import load_legacy_config
from .legacy_robustness import (
    GATE_IDS,
    RANDOM_PATHS,
    RANDOM_POSITIVE_MIN,
    _annual_equity_pnl,
    _checks,
    _remaining_years_positive,
    _save_backtest,
    _scenario_analysis,
    _sha256,
    _write_json,
    doubled_cost_context,
    evaluate_legacy_gates,
)
from .legacy_runner import PROJECT_ROOT, prepare_legacy_frames


@dataclass(frozen=True)
class LegacyBRobustnessResult:
    status: str
    summary: dict
    output_dir: Path


class RandomOrderLegacyBPolicy(LegacyBPolicy):
    def __init__(self, config, seed, path_id):
        super().__init__(config)
        self.seed = int(seed)
        self.path_id = int(path_id)
        self.audit = []

    def order_candidates(self, candidates):
        if not candidates:
            return []
        frame = pd.DataFrame(candidates).sort_values(
            ["date", "legacy_rank_score", "code"], ascending=[True, False, True], kind="stable"
        )
        codes = frame["code"].tolist()
        dates = frame["date"].astype(str).tolist()
        if len(set(dates)) != 1:
            raise ValueError("Random ordering expects one signal date")
        rng = random.Random(f"{self.seed}|{self.path_id}|{dates[0]}")
        rng.shuffle(codes)
        for rank, code in enumerate(codes):
            self.audit.append(
                {"path_id": self.path_id, "date": dates[0], "code": code, "random_rank": rank}
            )
        for row in frame.to_dict("records"):
            self.pending[str(row["code"])] = row
        return codes


def run_legacy_b_robustness(config, config_path=None, root: Path | None = None) -> LegacyBRobustnessResult:
    root = PROJECT_ROOT if root is None else Path(root)
    main_dir = root / "results" / "legacy_candidates" / CANDIDATE_B
    freeze = json.loads((root / ".sdd-legacy" / "b_main_freeze.json").read_text(encoding="utf-8"))
    for name, registered in freeze["files"].items():
        if _sha256(main_dir / name) != registered["sha256"]:
            raise ValueError(f"Frozen B main result changed before robustness: {name}")

    main_perf = json.loads((main_dir / "performance_summary.json").read_text(encoding="utf-8"))
    main_manifest = json.loads((main_dir / "run_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "results" / "legacy_b_summary.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "EVALUABLE":
        raise ValueError("Candidate B is not evaluable")
    audit_path = root / "results" / "legacy_b_recovery_audit.json"
    if not audit_path.exists():
        raise ValueError("Candidate B recovery audit is missing")
    recovery = json.loads(audit_path.read_text(encoding="utf-8"))
    if recovery.get("restored") is not True or recovery.get("status") != "EVALUABLE":
        raise ValueError("Candidate B recovery audit is not restored/EVALUABLE")
    if config_path is None or _sha256(Path(config_path)) != recovery["config"]["sha256"]:
        raise ValueError("Candidate B robustness --config does not match the recovery audit")
    if main_manifest["config"]["sha256"] != recovery["config"]["sha256"]:
        raise ValueError("Candidate B main manifest config differs from recovery audit")
    if main_manifest["recovery_audit"]["sha256"] != _sha256(audit_path):
        raise ValueError("Candidate B recovery audit changed after the main run")
    for name, registered in main_manifest["code"].items():
        if _sha256(root / name) != registered["sha256"]:
            raise ValueError(f"Candidate B code changed after the main run: {name}")

    panel_all = load_panel(root / "data" / "main_panel.parquet")
    signals_all = pd.read_parquet(
        frozen["outputs"]["signals"]["path"],
        columns=["date", "code", "entry_signal", "atr_pct", "amt20",
                 "legacy_rank_score", "rank_return_120", "stop_pct"],
    )
    actions = pd.read_parquet(root / "data" / "corporate_actions.parquet")
    panel, signals = prepare_legacy_frames(panel_all, signals_all, config)

    output_dir = root / "results" / "legacy_b_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "input_manifest.json",
        {
            "signals": {"path": frozen["outputs"]["signals"]["path"],
                        "sha256": frozen["outputs"]["signals"]["sha256"]},
            "config": {"path": str(root / "config" / "legacy_b_recovery.json"),
                       "sha256": _sha256(root / "config" / "legacy_b_recovery.json")},
            "recovery_audit": {"path": str(root / "results" / "legacy_b_recovery_audit.json"),
                               "sha256": _sha256(root / "results" / "legacy_b_recovery_audit.json")},
            "code": {name: {"sha256": _sha256(root / name)} for name in main_manifest["code"]},
            "b_main_freeze": freeze,
        },
    )

    double_config = replace(
        config,
        commission_rate=config.commission_rate * 2,
        min_commission=config.min_commission * 2,
        slippage_bps=20,
    )
    double_summary_path = output_dir / "double_cost" / "performance_summary.json"
    double_checks_path = output_dir / "double_cost" / "checks.json"
    if double_summary_path.exists() and double_checks_path.exists():
        double_perf = json.loads(double_summary_path.read_text(encoding="utf-8"))
        double_checks = json.loads(double_checks_path.read_text(encoding="utf-8"))
    else:
        with doubled_cost_context():
            double_bt = run_backtest(
                panel, signals, actions, double_config, policy=LegacyBPolicy(double_config)
            )
            double_checks = _checks(panel, signals, double_bt, double_config)
        double_perf = __import__("trend_system.metrics", fromlist=["performance_summary"]).performance_summary(
            double_bt.daily_account, double_bt.round_trips
        )
        _save_backtest(output_dir / "double_cost", double_bt)
        _write_json(output_dir / "double_cost" / "performance_summary.json", double_perf)
        _write_json(output_dir / "double_cost" / "checks.json", double_checks)

    random_records = []
    random_positive = 0
    random_early_stop = None
    for path_id in range(RANDOM_PATHS):
        path_dir = output_dir / "random_order" / f"path_{path_id:02d}"
        summary_path = path_dir / "performance_summary.json"
        if summary_path.exists():
            perf = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            completed = len(random_records)
            if completed - random_positive > RANDOM_PATHS - RANDOM_POSITIVE_MIN:
                random_early_stop = (
                    f"{completed} completed paths include "
                    f"{completed - random_positive} negatives; maximum possible positives "
                    f"{RANDOM_PATHS - (completed - random_positive)} < {RANDOM_POSITIVE_MIN}"
                )
                break
            policy = RandomOrderLegacyBPolicy(config, config.random_seed, path_id)
            backtest = run_backtest(panel, signals, actions, config, policy=policy)
            perf = __import__("trend_system.metrics", fromlist=["performance_summary"]).performance_summary(
                backtest.daily_account, backtest.round_trips
            )
            path_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(policy.audit).to_csv(path_dir / "random_order_log.csv", index=False)
            _write_json(summary_path, perf)
        positive = perf["total_return"] > 0
        random_positive += int(positive)
        random_records.append(
            {"path_id": path_id, "total_return": perf["total_return"],
             "max_drawdown": perf["max_drawdown"], "positive": bool(positive)}
        )

    annual_pnl = _annual_equity_pnl(
        pd.read_parquet(main_dir / "daily_account.parquet"), config.initial_cash
    )
    remaining_ok, remaining_detail = _remaining_years_positive(annual_pnl)
    integrity = main_manifest["checks"]["execution_integrity"]
    data_integrity = bool(
        integrity.get("all_passed") is True
        and main_manifest["checks"]["cash_nonnegative"] is True
        and main_manifest["checks"]["whole_lot_buys"] is True
        and main_manifest["checks"]["equity_identity"] is True
        and main_manifest["checks"]["trades_after_signal_date"] is True
        and frozen.get("required_entry_coverage") == 1.0
        and frozen.get("industry_entry_coverage") == 1.0
    )
    gates = {
        "data_execution_integrity": {"passed": data_integrity},
        "main_window_positive": {"passed": bool(main_perf["total_return"] > 0)},
        "rolling_3y_majority_positive": {
            "passed": bool(main_perf["rolling_3y_positive_share"] > 0.5)
        },
        "double_cost_positive_and_executable": {
            "passed": bool(
                double_perf["total_return"] > 0
                and double_checks["cash_nonnegative"]
                and double_checks["whole_lot_buys"]
                and double_checks["equity_identity"]
                and double_checks["execution_integrity"]["all_passed"]
            )
        },
        "random_order_at_least_16_positive": {
            "passed": bool(random_positive >= RANDOM_POSITIVE_MIN)
        },
        "remaining_years_equity_pnl_positive": {"passed": bool(remaining_ok)},
        "fully_after_close_reproducible": {
            "passed": bool(
                integrity.get("all_passed") is True
                and main_manifest["checks"]["trades_after_signal_date"] is True
            )
        },
        "drawdown_within_limit": {"passed": bool(main_perf["max_drawdown"] <= 0.20)},
    }
    summary = {
        "candidate_status": "EVALUABLE",
        "main_max_drawdown": float(main_perf["max_drawdown"]),
        "gates": gates,
        "main": {
            "total_return": main_perf["total_return"],
            "cagr": main_perf["cagr"],
            "final_equity": main_perf["final_equity"],
            "rolling_3y_positive_share": main_perf["rolling_3y_positive_share"],
            "closed_round_trips": main_perf["round_trip_count"],
            "win_rate": main_perf["win_rate"],
        },
        "double_cost": {
            "total_return": double_perf["total_return"],
            "max_drawdown": double_perf["max_drawdown"],
        },
        "random_order": {
            "positive_paths": random_positive,
            "paths_completed": len(random_records),
            "early_stop_reason": random_early_stop,
            "paths": random_records,
        },
        "remaining_years": remaining_detail,
        "annual_equity_pnl": annual_pnl,
        "stress_2010_2015": {
            "status": "NOT_APPLICABLE",
            "reason": "原规则资格日期为 2017-10-13，2010-2015 结构性无入场",
        },
        "scenario": _scenario_analysis(root, main_dir, config),
    }
    summary["verdict"] = evaluate_legacy_gates(summary)
    _write_json(output_dir / "summary.json", summary)
    _write_reports(root, summary)
    return LegacyBRobustnessResult(
        status=summary["verdict"], summary=summary, output_dir=output_dir
    )


def _write_reports(root: Path, summary: dict) -> None:
    gates = summary["gates"]
    lines = [
        "# 候选 B 旧研究系统统一复核结论",
        "",
        f"- 状态：**{summary['verdict']}**",
        f"- 主窗口总收益：{summary['main']['total_return']:.6f}",
        f"- 主窗口 CAGR：{summary['main']['cagr']:.6f}",
        f"- 主窗口最大回撤：{summary['main_max_drawdown']:.6f}",
        f"- 主窗口闭合往返：{summary['main']['closed_round_trips']}；胜率：{summary['main']['win_rate']:.4f}",
        f"- 双倍费用+20bp：总收益 {summary['double_cost']['total_return']:.6f}，MDD {summary['double_cost']['max_drawdown']:.6f}",
        f"- 随机排序正收益路径：{summary['random_order']['positive_paths']}/{summary['random_order']['paths_completed']} 条已完成；提前停止：{summary['random_order']['early_stop_reason']}",
        "- 2010—2015 压力：原规则资格日期为 2017-10-13，结构性无入场，不伪造空账户结果。",
        "",
        "## 机械门槛",
        "",
        "| 门槛 | 结果 |",
        "| --- | --- |",
    ]
    for gate_id in GATE_IDS:
        lines.append(f"| `{gate_id}` | {'通过' if gates[gate_id]['passed'] else '失败'} |")
    lines += [
        "",
        "## 年度权益盈亏",
        "",
        "| 年份 | 盈亏（元） |",
        "| --- | ---: |",
    ]
    for year, pnl in summary["annual_equity_pnl"].items():
        lines.append(f"| {year} | {pnl:.2f} |")
    lines += ["", "本结论不构成实盘授权或收益保证。"]
    (root / "95_候选B复核结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scenario = summary["scenario"]
    lines = [
        "# 候选 B 失败与场景归因",
        "",
        "只分析已冻结主路径，不新增消融或改参版本。",
        "",
        f"- 全部成交费用：{scenario['total_trade_costs']:.2f} 元",
        f"- 闭合交易净盈亏：{scenario['closed_net_pnl']:.2f} 元",
        f"- 闭合交易毛盈亏（含滑点成交价）：{scenario['closed_gross_pnl']:.2f} 元",
        "",
        "## 退出原因",
        "",
        "| 原因 | 笔数 | 净盈亏 | 毛盈亏 | 费用 | 平均收益率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario["exit_reasons"]:
        lines.append(
            f"| {row['exit_reason']} | {row['count']} | {row['net_pnl']:.2f} | "
            f"{row['gross_pnl']:.2f} | {row['costs']:.2f} | {row['avg_return']:.4f} |"
        )
    lines += [
        "",
        "## 指数状态（事后描述，不构成交易闸门）",
        "",
        "| 上证状态 | 天数 | 平均日收益 | 平均暴露 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in scenario["market_state"]:
        lines.append(
            f"| {row['state']} | {row['days']} | {row['avg_daily_return']:.6f} | "
            f"{row['avg_exposure']:.4f} |"
        )
    lines += ["", "分组仅作描述，不参与门槛，不用于反向调参。"]
    (root / "96_候选B失败与场景.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
