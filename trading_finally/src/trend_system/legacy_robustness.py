"""Frozen robustness, mechanical gates and scenario attribution for candidate A."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd
from unittest.mock import patch

from . import broker, execution
from .broker import execution_integrity, run_backtest
from .data import load_panel
from .legacy_data import A_ENTRY_REQUIRED_FIELDS, SIGNAL_FIELDS, candidate_entry_valid
from .legacy_policy import LegacyPolicy
from .legacy_runner import CANDIDATE_A, PROJECT_ROOT, SIGNAL_COLUMNS, prepare_legacy_frames
from .metrics import annual_and_rolling, performance_summary


GATE_IDS = (
    "data_execution_integrity",
    "main_window_positive",
    "rolling_3y_majority_positive",
    "double_cost_positive_and_executable",
    "random_order_at_least_16_positive",
    "remaining_years_equity_pnl_positive",
    "fully_after_close_reproducible",
    "drawdown_within_limit",
)
STRESS_START = "2010-01-04"
STRESS_END = "2015-12-31"
RANDOM_PATHS = 20
RANDOM_POSITIVE_MIN = 16


@dataclass(frozen=True)
class LegacyRobustnessResult:
    status: str
    summary: dict
    output_dir: Path


def evaluate_legacy_gates(summary: dict) -> str:
    status = summary.get("candidate_status")
    if status == "NOT_EVALUABLE":
        return "NOT_EVALUABLE"
    if status != "EVALUABLE":
        return "REJECTED"
    gates = summary.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(GATE_IDS):
        return "REJECTED"
    for gate_id in GATE_IDS:
        gate = gates[gate_id]
        if not isinstance(gate, dict):
            return "REJECTED"
        passed = gate.get("passed")
        if type(passed) is not bool:
            return "REJECTED"
        if not passed:
            return "REJECTED"
    mdd = summary.get("main_max_drawdown")
    if not isinstance(mdd, (int, float)) or not math.isfinite(mdd) or mdd < 0:
        return "REJECTED"
    if mdd <= 0.10:
        return "UNREJECTED"
    if mdd <= 0.20:
        return "BOUNDARY"
    return "REJECTED"


class RandomOrderLegacyPolicy(LegacyPolicy):
    """Same frozen rules; only same-day candidate order is randomly permuted."""

    def __init__(self, config, seed, path_id):
        super().__init__(config)
        self.seed = int(seed)
        self.path_id = int(path_id)
        self.audit: list[dict] = []

    def order_candidates(self, candidates):
        if not candidates:
            return []
        frame = pd.DataFrame(candidates)
        ordered = frame.sort_values(["date", "legacy_rank_score", "code"], ascending=[True, False, True], kind="stable")
        codes = ordered["code"].tolist()
        dates = ordered["date"].astype(str).tolist()
        if len(set(dates)) != 1:
            raise ValueError("Random ordering expects a single signal date per call")
        rng = random.Random(f"{self.seed}|{self.path_id}|{dates[0]}")
        rng.shuffle(codes)
        for rank, code in enumerate(codes):
            self.audit.append(
                {"path_id": self.path_id, "date": dates[0], "code": code, "random_rank": rank}
            )
        return codes


@contextmanager
def doubled_cost_context():
    """Double every fee component in sizing and fills, not only config rates."""
    original = execution.transaction_costs

    def doubled(date, amount, side, multiplier=1.0):
        return original(date, amount, side, 2.0 * multiplier)

    with patch.object(execution, "transaction_costs", doubled), patch.object(
        broker, "transaction_costs", doubled
    ):
        yield


def derive_entries_for_window(signals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Derive entries for another window from the same frozen signal rows."""
    frame = signals.copy()
    frame["date"] = frame["date"].astype(str)
    valid = candidate_entry_valid(frame)
    in_window = frame["date"].between(start, end)
    proposed = (
        frame["accepted_signal"].astype(bool)
        & in_window
        & frame["is_main_board"].fillna(False).astype(bool)
    )
    frame["entry_signal"] = (
        proposed & valid & frame["is_st"].fillna(1).eq(0)
    ).astype(bool)
    return frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_backtest(directory: Path, backtest) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "daily_account",
        "daily_positions",
        "trades",
        "orders",
        "round_trips",
        "action_log",
    ):
        getattr(backtest, name).to_parquet(directory / f"{name}.parquet", index=False)
    annual_and_rolling(backtest.daily_account).to_csv(
        directory / "annual_and_rolling.csv", index=False
    )


def _checks(panel, signals, backtest, config) -> dict:
    account = backtest.daily_account
    trades = backtest.trades
    buys = trades.loc[trades["side"] == "buy"]
    integrity = execution_integrity(panel, signals, backtest, config)
    return {
        "cash_nonnegative": bool((account["cash"] >= -1e-8).all()),
        "whole_lot_buys": bool((buys["quantity"] % 100 == 0).all()),
        "equity_identity": bool(
            (
                (account["equity"] - account["cash"] - account["receivables"]
                 - account["market_value"]).abs()
                <= 1e-6
            ).all()
        ),
        "trades_after_signal_date": bool((trades["date"] > trades["signal_date"]).all()),
        "execution_integrity": integrity,
    }


def _annual_equity_pnl(daily_account: pd.DataFrame, initial_equity: float) -> dict:
    account = daily_account.copy()
    account["date"] = account["date"].astype(str)
    account["year"] = account["date"].str.slice(0, 4)
    year_end = account.groupby("year", sort=True)["equity"].last()
    previous = year_end.shift(1).fillna(initial_equity)
    pnl = (year_end - previous).to_dict()
    return {year: float(value) for year, value in pnl.items()}


def _remaining_years_positive(annual_pnl: dict) -> tuple[bool, dict]:
    if not annual_pnl:
        return False, {}
    best_year = max(annual_pnl, key=annual_pnl.get)
    remaining = sum(value for year, value in annual_pnl.items() if year != best_year)
    return remaining > 0, {"best_year": best_year, "remaining_pnl": float(remaining)}


def _scenario_analysis(root: Path, main_dir: Path, config) -> dict:
    account = pd.read_parquet(main_dir / "daily_account.parquet")
    account["date"] = account["date"].astype(str)
    trips = pd.read_parquet(main_dir / "round_trips.parquet")
    trades = pd.read_parquet(main_dir / "trades.parquet")
    index = pd.read_parquet(
        root / "data" / "index_panel.parquet",
        columns=["date", "code", "close"],
    ).loc[lambda frame: frame["code"].eq("000001.SH")].copy()
    index["date"] = index["date"].astype(str)
    index = index.sort_values("date")
    index["ma20"] = index["close"].rolling(20).mean()
    index["state"] = (index["close"] > index["ma20"]).map({True: "above_ma20", False: "below_ma20"})
    merged = account.merge(index[["date", "state"]], on="date", how="left")
    merged["daily_return"] = merged["equity"].pct_change()

    year_end = merged.groupby(merged["date"].str.slice(0, 4))["equity"].last()
    year_pnl = year_end.diff()
    if len(year_pnl):
        year_pnl.iloc[0] = year_end.iloc[0] - config.initial_cash

    state_stats = (
        merged.groupby("state", dropna=False)
        .agg(days=("date", "size"), avg_daily_return=("daily_return", "mean"),
             avg_exposure=("exposure", "mean"))
        .reset_index()
    )
    exits = (
        trips.groupby("exit_reason")
        .agg(count=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
             gross_pnl=("gross_pnl", "sum"), costs=("total_costs", "sum"),
             avg_return=("return_pct", "mean"))
        .reset_index()
    )
    return {
        "annual_equity_pnl": {str(year): float(value) for year, value in year_pnl.items()},
        "market_state": state_stats.to_dict(orient="records"),
        "exit_reasons": exits.to_dict(orient="records"),
        "total_trade_costs": float(trades["costs"].sum()),
        "closed_net_pnl": float(trips["net_pnl"].sum()),
        "closed_gross_pnl": float(trips["gross_pnl"].sum()),
    }


def run_legacy_robustness(config, root: Path | None = None) -> LegacyRobustnessResult:
    root = PROJECT_ROOT if root is None else Path(root)
    main_dir = root / "results" / "legacy_candidates" / CANDIDATE_A
    freeze = json.loads((root / ".sdd-legacy" / "main_freeze.json").read_text(encoding="utf-8"))
    for name, registered in freeze["files"].items():
        if _sha256(main_dir / name) != registered["sha256"]:
            raise ValueError(f"Frozen main result changed before robustness: {name}")

    main_perf = json.loads((main_dir / "performance_summary.json").read_text(encoding="utf-8"))
    main_manifest = json.loads((main_dir / "run_manifest.json").read_text(encoding="utf-8"))
    frozen_summary = json.loads(
        (root / "results" / "legacy_a_summary.json").read_text(encoding="utf-8")
    )
    signal_columns = list(dict.fromkeys([
        *SIGNAL_COLUMNS,
        "accepted_signal",
        "is_main_board",
        "is_st",
        *A_ENTRY_REQUIRED_FIELDS,
    ]))
    signals_all = pd.read_parquet(
        frozen_summary["outputs"]["signals"]["path"], columns=signal_columns
    )
    panel_all = load_panel(root / "data" / "main_panel.parquet")
    actions = pd.read_parquet(root / "data" / "corporate_actions.parquet")
    main_panel, main_signals = prepare_legacy_frames(panel_all, signals_all, config)

    output_dir = root / "results" / "legacy_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "input_manifest.json",
        {
            "signal": {"path": frozen_summary["outputs"]["signals"]["path"],
                       "sha256": frozen_summary["outputs"]["signals"]["sha256"]},
            "main_panel": {"sha256": _sha256(root / "data" / "main_panel.parquet")},
            "corporate_actions": {"sha256": _sha256(root / "data" / "corporate_actions.parquet")},
            "main_freeze": freeze,
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
                main_panel, main_signals, actions, double_config, policy=LegacyPolicy(double_config)
            )
            double_checks = _checks(main_panel, main_signals, double_bt, double_config)
        double_perf = performance_summary(double_bt.daily_account, double_bt.round_trips)
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
            policy = RandomOrderLegacyPolicy(config, config.random_seed, path_id)
            backtest = run_backtest(main_panel, main_signals, actions, config, policy=policy)
            perf = performance_summary(backtest.daily_account, backtest.round_trips)
            path_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(policy.audit).to_csv(path_dir / "random_order_log.csv", index=False)
            _write_json(summary_path, perf)
        positive = perf["total_return"] > 0
        random_positive += int(positive)
        random_records.append(
            {"path_id": path_id, "total_return": perf["total_return"],
             "max_drawdown": perf["max_drawdown"], "positive": bool(positive)}
        )

    main_derived = derive_entries_for_window(signals_all, config.start_date, config.end_date)
    mismatch = int((main_derived["entry_signal"].astype(bool) != signals_all["entry_signal"].astype(bool)).sum())
    if mismatch:
        raise ValueError(f"Stress-entry derivation differs on main window: {mismatch} rows")
    stress_signals_all = derive_entries_for_window(signals_all, STRESS_START, STRESS_END)
    stress_config = replace(config, start_date=STRESS_START, end_date=STRESS_END)
    stress_panel, stress_signals = prepare_legacy_frames(
        panel_all, stress_signals_all, stress_config
    )
    stress_bt = run_backtest(
        stress_panel, stress_signals, actions, stress_config,
        policy=LegacyPolicy(stress_config),
    )
    stress_perf = performance_summary(stress_bt.daily_account, stress_bt.round_trips)
    stress_checks = _checks(stress_panel, stress_signals, stress_bt, stress_config)
    stress_dir = root / "results" / "legacy_stress_2010_2015"
    _save_backtest(stress_dir, stress_bt)
    _write_json(stress_dir / "performance_summary.json", stress_perf)
    _write_json(stress_dir / "checks.json", stress_checks)

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
        and frozen_summary["required_entry_coverage"]["passes"] is True
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
                and main_manifest["checks"]["no_signal_day_trade"] is True
            )
        },
        "drawdown_within_limit": {
            "passed": bool(main_perf["max_drawdown"] <= 0.20)
        },
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
            "total_return": stress_perf["total_return"],
            "max_drawdown": stress_perf["max_drawdown"],
            "final_equity": stress_perf["final_equity"],
            "closed_round_trips": stress_perf["round_trip_count"],
        },
        "scenario": _scenario_analysis(root, main_dir, config),
    }
    summary["verdict"] = evaluate_legacy_gates(summary)
    _write_json(output_dir / "summary.json", summary)
    _write_reports(root, summary)
    return LegacyRobustnessResult(status=summary["verdict"], summary=summary, output_dir=output_dir)


def _write_reports(root: Path, summary: dict) -> None:
    gates = summary["gates"]
    lines = [
        "# 旧候选 A 统一复核结论",
        "",
        f"- 状态：**{summary['verdict']}**",
        f"- 主窗口总收益：{summary['main']['total_return']:.6f}",
        f"- 主窗口 CAGR：{summary['main']['cagr']:.6f}",
        f"- 主窗口最大回撤：{summary['main_max_drawdown']:.6f}",
        f"- 主窗口闭合往返：{summary['main']['closed_round_trips']}；胜率：{summary['main']['win_rate']:.4f}",
        f"- 双倍费用+20bp：总收益 {summary['double_cost']['total_return']:.6f}，MDD {summary['double_cost']['max_drawdown']:.6f}",
        f"- 随机排序正收益路径：{summary['random_order']['positive_paths']}/{summary['random_order']['paths_completed']} 条已完成；提前停止：{summary['random_order']['early_stop_reason']}",
        f"- 2010—2015 场景压力：总收益 {summary['stress_2010_2015']['total_return']:.6f}，MDD {summary['stress_2010_2015']['max_drawdown']:.6f}",
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
    lines += [
        "",
        "候选 B 已按同源恢复预注册 `14_预注册_候选B恢复.md` 通过数据闸门并运行唯一主版本；其最终结果为 REJECTED，详见 `95_候选B复核结论.md` 与 `96_候选B失败与场景.md`。A、B 均不进入纯模拟。",
        "",
        "本结论不构成实盘授权或收益保证。",
    ]
    (root / "93_旧候选复核结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scenario = summary["scenario"]
    lines = [
        "# 旧候选 A 失败与场景归因",
        "",
        "只分析已冻结主路径，不新增消融或改参版本。",
        "",
        "## 交易与费用",
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
    lines += [
        "",
        "分组仅作描述，不参与门槛，不用于反向调参。",
    ]
    (root / "94_旧候选失败与场景.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
