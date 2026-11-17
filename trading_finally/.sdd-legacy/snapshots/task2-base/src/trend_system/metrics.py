"""Account metrics, research benchmarks, and the frozen baseline run."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow

from .broker import BacktestResult, run_backtest
from .config import ResearchConfig
from .data import load_panel
from .signals import compute_signals, select_backtest_window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
DESIGN_DOCUMENTS = (
    "00_设计_主板趋势波段交易系统研究.md",
    "01_实施计划_主板趋势波段交易系统.md",
    "02_预注册_极简趋势消融.md",
)
_ANNUAL_ROLLING_COLUMNS = [
    "period_type",
    "period",
    "start_date",
    "end_date",
    "return_pct",
    "is_partial",
]


@dataclass(frozen=True)
class BaselineResult:
    backtest: BacktestResult
    summary: dict
    annual_rolling: pd.DataFrame
    benchmarks: pd.DataFrame
    benchmark_metadata: dict
    manifest: dict
    output_dir: Path


def _ordered_account(daily_account: pd.DataFrame) -> pd.DataFrame:
    if daily_account.empty:
        raise ValueError("daily_account must not be empty")
    account = daily_account.copy()
    account["_date"] = pd.to_datetime(account["date"])
    account = account.sort_values("_date", kind="stable").reset_index(drop=True)
    account["equity"] = account["equity"].astype(float)
    return account


def annual_and_rolling(daily_account: pd.DataFrame) -> pd.DataFrame:
    """Return calendar-year and month-end rolling three-year returns.

    The first and last years are marked partial when the account does not cover
    January and December respectively. Annual returns after the first year use
    the previous observed year-end equity as their base. Rolling windows use an
    actual date three years earlier, or the latest account date before it within
    31 calendar days. Windows requiring an older asof observation are skipped.
    """
    account = _ordered_account(daily_account)
    rows = []
    previous_year_end_equity = None
    previous_year_end_date = None
    groups = list(account.groupby(account["_date"].dt.year, sort=True))
    for group_index, (year, group) in enumerate(groups):
        first_date = group["_date"].iloc[0]
        end_date = group["_date"].iloc[-1]
        if previous_year_end_equity is None:
            base_equity = group["equity"].iloc[0]
            start_date = first_date
        else:
            base_equity = previous_year_end_equity
            start_date = previous_year_end_date
        is_partial = (
            (group_index == 0 and first_date.month != 1)
            or (group_index == len(groups) - 1 and end_date.month != 12)
        )
        rows.append(
            {
                "period_type": "annual_partial" if is_partial else "annual",
                "period": str(year),
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "return_pct": float(group["equity"].iloc[-1] / base_equity - 1),
                "is_partial": bool(is_partial),
            }
        )
        previous_year_end_equity = group["equity"].iloc[-1]
        previous_year_end_date = end_date

    account_dates = account["_date"]
    month_ends = account.groupby(account["_date"].dt.to_period("M"), sort=True).tail(1)
    for _, row in month_ends.iterrows():
        end_date = row["_date"]
        target_date = end_date - pd.DateOffset(years=3)
        if target_date < account_dates.iloc[0]:
            continue
        start_index = int(account_dates.searchsorted(target_date, side="right") - 1)
        if start_index < 0:
            continue
        start_date = account_dates.iloc[start_index]
        if (target_date - start_date).days > 31:
            continue
        start_equity = account["equity"].iloc[start_index]
        rows.append(
            {
                "period_type": "rolling_3y",
                "period": str(end_date.to_period("M")),
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "return_pct": float(row["equity"] / start_equity - 1),
                "is_partial": False,
            }
        )
    return pd.DataFrame(rows, columns=_ANNUAL_ROLLING_COLUMNS)


def performance_summary(daily_account: pd.DataFrame, round_trips: pd.DataFrame) -> dict:
    """Summarize account performance and closed, cost-adjusted round trips.

    Turnover is the full-window two-sided closed-trade notional (entries plus
    exits) divided by mean daily equity. Expectancy and average win/loss are
    currency amounts per closed round trip after costs.
    """
    account = _ordered_account(daily_account)
    annual_rolling = annual_and_rolling(account)
    annual = annual_rolling.loc[
        annual_rolling["period_type"].isin(["annual", "annual_partial"])
    ]
    rolling = annual_rolling.loc[annual_rolling["period_type"] == "rolling_3y"]
    initial_equity = float(account["equity"].iloc[0])
    final_equity = float(account["equity"].iloc[-1])
    elapsed_days = int((account["_date"].iloc[-1] - account["_date"].iloc[0]).days)
    total_return = final_equity / initial_equity - 1
    cagr = (
        (final_equity / initial_equity) ** (365.25 / elapsed_days) - 1
        if elapsed_days > 0
        else 0.0
    )
    drawdown = account["equity"] / account["equity"].cummax() - 1

    trips = round_trips.copy()
    if not trips.empty:
        trips["_exit_date"] = pd.to_datetime(trips["exit_date"])
        trips = trips.sort_values(["_exit_date", "code"], kind="stable")
        pnl = trips["net_pnl"].astype(float)
        wins = pnl.loc[pnl > 0]
        losses = pnl.loc[pnl < 0]
        longest_losing_streak = 0
        current_losing_streak = 0
        for value in pnl:
            current_losing_streak = current_losing_streak + 1 if value < 0 else 0
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        realized_annual_contribution = {
            str(year): float(value)
            for year, value in pnl.groupby(trips["_exit_date"].dt.year).sum().items()
        }
        realized_stock_contribution = {
            str(code): float(value)
            for code, value in pnl.groupby(trips["code"], sort=True).sum().items()
        }
        top_five = wins.nlargest(5)
        gross_profit = float(wins.sum())
        mean_equity = float(account["equity"].mean())
        turnover = (
            float(trips["entry_amount"].sum() + trips["exit_amount"].sum()) / mean_equity
            if mean_equity
            else 0.0
        )
        trade_metrics = {
            "round_trip_count": int(len(trips)),
            "closed_round_trip_turnover": turnover,
            "net_expectancy": float(pnl.mean()),
            "win_rate": float((pnl > 0).mean()),
            "average_win": float(wins.mean()) if len(wins) else 0.0,
            "average_loss": float(losses.mean()) if len(losses) else 0.0,
            "longest_losing_streak": int(longest_losing_streak),
            "realized_annual_contribution": realized_annual_contribution,
            "realized_stock_contribution": realized_stock_contribution,
            "top5_closed_profitable_trades_net_pnl": float(top_five.sum()),
            "top5_closed_profitable_trades_share_of_gross_profit": (
                float(top_five.sum() / gross_profit) if gross_profit > 0 else 0.0
            ),
        }
    else:
        trade_metrics = {
            "round_trip_count": 0,
            "closed_round_trip_turnover": 0.0,
            "net_expectancy": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "longest_losing_streak": 0,
            "realized_annual_contribution": {},
            "realized_stock_contribution": {},
            "top5_closed_profitable_trades_net_pnl": 0.0,
            "top5_closed_profitable_trades_share_of_gross_profit": 0.0,
        }

    return {
        "start_date": account["_date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": account["_date"].iloc[-1].strftime("%Y-%m-%d"),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(-drawdown.min()),
        "annual_returns": {
            str(row.period): float(row.return_pct)
            for row in annual.itertuples(index=False)
        },
        "partial_years": annual.loc[annual["is_partial"], "period"].tolist(),
        "rolling_3y_returns": {
            str(row.period): float(row.return_pct)
            for row in rolling.itertuples(index=False)
        },
        "rolling_3y_positive_share": (
            float((rolling["return_pct"] > 0).mean()) if len(rolling) else 0.0
        ),
        "average_exposure": float(account["exposure"].mean()),
        "max_exposure": float(account["exposure"].max()),
        **trade_metrics,
    }


def _research_benchmarks(
    main_panel: pd.DataFrame,
    index_panel: pd.DataFrame,
    daily_account: pd.DataFrame,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, dict]:
    dates = pd.DatetimeIndex(pd.to_datetime(daily_account["date"]))
    frames = []

    def add_benchmark(name, code, nav):
        normalized = pd.Series(nav, index=dates, dtype=float)
        normalized = normalized / normalized.iloc[0]
        frames.append(
            pd.DataFrame(
                {
                    "benchmark": name,
                    "code": code,
                    "date": dates.strftime("%Y-%m-%d"),
                    "nav": normalized.to_numpy(),
                    "return_pct": normalized.to_numpy() - 1,
                }
            )
        )

    add_benchmark("现金", "", np.ones(len(dates)))
    for name, code in (("上证综指", "000001.SH"), ("沪深300", "000300.SH")):
        selected = index_panel.loc[index_panel["code"] == code, ["date", "close"]].copy()
        selected["date"] = pd.to_datetime(selected["date"])
        selected = selected.sort_values("date").drop_duplicates("date", keep="last")
        closes = selected.set_index("date")["close"].reindex(dates).ffill()
        if closes.isna().any():
            raise ValueError(f"Benchmark {code} does not cover the full account window")
        add_benchmark(name, code, closes.to_numpy())

    performance = main_panel.loc[
        main_panel["date"].between(config.start_date, config.end_date),
        ["date", "code", "signal_close"],
    ].copy()
    performance = performance.sort_values(["code", "date"], kind="stable")
    previous = performance.groupby("code", sort=False)["signal_close"].shift(1)
    performance["stock_return"] = performance["signal_close"] / previous - 1
    daily_returns = performance.groupby("date")["stock_return"].mean()
    aligned_returns = daily_returns.reindex(dates.strftime("%Y-%m-%d")).fillna(0.0)
    aligned_returns.iloc[0] = 0.0
    add_benchmark("主板等权", "", (1.0 + aligned_returns).cumprod().to_numpy())

    metadata = {
        "现金": {"code": "", "dividends_included": False, "note": "现金研究基准，收益率设为0。"},
        "上证综指": {
            "code": "000001.SH",
            "dividends_included": False,
            "note": "价格指数基准，不含股息。",
        },
        "沪深300": {
            "code": "000300.SH",
            "dividends_included": False,
            "note": "价格指数基准，不含股息。",
        },
        "主板等权": {
            "code": "",
            "dividends_included": True,
            "note": "主板同宇宙后复权收益日度等权再平衡研究基准，不可由2万元账户直接复制。",
        },
    }
    return pd.concat(frames, ignore_index=True), metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_info(frame: pd.DataFrame, path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "codes": int(frame["code"].nunique()) if "code" in frame else 0,
        "first_date": str(frame["date"].min()) if len(frame) else None,
        "last_date": str(frame["date"].max()) if len(frame) else None,
    }


def _manifest(
    config: ResearchConfig,
    inputs: dict,
    config_path: str | Path | None = None,
) -> dict:
    root = PROJECT_ROOT
    if config_path is None:
        canonical = json.dumps(
            asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        config_info = {
            "source": "canonical_config_values",
            "path": None,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "values": asdict(config),
        }
    else:
        actual_config_path = Path(config_path).resolve()
        config_info = {
            "source": "file",
            "path": str(actual_config_path),
            "sha256": _sha256(actual_config_path),
            "values": asdict(config),
        }
    code_paths = sorted((SOURCE_ROOT / "src" / "trend_system").glob("*.py"))
    code_paths.append(SOURCE_ROOT / "scripts" / "run_baseline.py")
    return {
        "config": config_info,
        "design_documents": [
            {"path": name, "sha256": _sha256(root / name)} for name in DESIGN_DOCUMENTS
        ],
        "code_files": [
            {
                "path": str(path.relative_to(SOURCE_ROOT)),
                "sha256": _sha256(path),
            }
            for path in code_paths
        ],
        "inputs": inputs,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "duckdb": duckdb.__version__,
        },
    }


def _benchmark_summary(benchmarks: pd.DataFrame) -> dict:
    result = {}
    for name, group in benchmarks.groupby("benchmark", sort=False):
        nav = group["nav"].astype(float)
        result[name] = {
            "total_return": float(nav.iloc[-1] - 1),
            "max_drawdown": float(-(nav / nav.cummax() - 1).min()),
        }
    return result


def _markdown_summary(
    config: ResearchConfig,
    summary: dict,
    annual_rolling: pd.DataFrame,
    benchmarks: pd.DataFrame,
    benchmark_metadata: dict,
) -> str:
    lines = [
        "# 冻结研究基线摘要",
        "",
        "本文件仅记录预注册研究基线的可复现结果，不构成收益宣传或投资建议。",
        "",
        f"- 主窗口：{config.start_date} 至 {config.end_date}",
        f"- 初始资金：{config.initial_cash}",
        f"- 完整往返交易：{summary['round_trip_count']}",
        "",
        "## 账户与交易指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    labels = (
        ("total_return", "总收益"),
        ("cagr", "CAGR"),
        ("max_drawdown", "最大回撤"),
        ("average_exposure", "平均持仓暴露"),
        ("max_exposure", "最大持仓暴露"),
        ("closed_round_trip_turnover", "已闭合往返双边换手/平均权益（全窗口）"),
        ("net_expectancy", "扣费后每笔期望（元）"),
        ("win_rate", "胜率"),
        ("average_win", "平均盈利（元）"),
        ("average_loss", "平均亏损（元）"),
        ("longest_losing_streak", "最长连续亏损笔数"),
        ("top5_closed_profitable_trades_net_pnl", "前五已闭合盈利交易净贡献（元）"),
        ("top5_closed_profitable_trades_share_of_gross_profit", "前五已闭合盈利交易/总盈利"),
    )
    for key, label in labels:
        lines.append(f"| {label} | {summary[key]:.6f} |")
    lines.extend(
        [
            "",
            "交易统计仅覆盖已闭合往返交易，不含未闭合交易、未实现盈亏和分红；已实现年度贡献不与账户年度权益盈亏直接相等。",
            "",
            "## 年度收益",
            "",
            "部分年度单独标记，不与完整年度直接比较。",
            "",
            "| 年度 | 类型 | 起始估值日 | 结束日 | 收益 |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    annual = annual_rolling.loc[annual_rolling["period_type"] != "rolling_3y"]
    for row in annual.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.period_type} | {row.start_date} | {row.end_date} | {row.return_pct:.6f} |"
        )
    rolling = annual_rolling.loc[annual_rolling["period_type"] == "rolling_3y"]
    lines.extend(
        [
            "",
            "## 月末滚动三年",
            "",
            f"窗口数：{len(rolling)}；正收益窗口占比：{summary['rolling_3y_positive_share']:.6f}。完整逐月结果见 `annual_and_rolling.csv`。",
            "",
            "## 研究基准",
            "",
            "| 基准 | 区间收益 | 最大回撤 | 口径 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    benchmark_stats = _benchmark_summary(benchmarks)
    for name, stats in benchmark_stats.items():
        lines.append(
            f"| {name} | {stats['total_return']:.6f} | {stats['max_drawdown']:.6f} | {benchmark_metadata[name]['note']} |"
        )
    lines.extend(
        [
            "",
            "完整年度与单票贡献、逐笔指标见 `performance_summary.json`；复现哈希、输入行数、数据窗口和依赖版本见 `run_manifest.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_baseline(config: ResearchConfig, config_path: str | Path | None = None) -> BaselineResult:
    """Run the frozen baseline using only local parquet panels."""
    root = PROJECT_ROOT
    data_directory = root / "data"
    main_path = data_directory / "main_panel.parquet"
    index_path = data_directory / "index_panel.parquet"
    actions_path = data_directory / "corporate_actions.parquet"
    main_panel = load_panel(main_path)
    index_panel = pd.read_parquet(index_path)
    corporate_actions = pd.read_parquet(actions_path)

    signals = compute_signals(main_panel, config)
    backtest_panel, backtest_signals = select_backtest_window(main_panel, signals, config)
    backtest = run_backtest(backtest_panel, backtest_signals, corporate_actions, config)
    summary = performance_summary(backtest.daily_account, backtest.round_trips)
    annual_rolling = annual_and_rolling(backtest.daily_account)
    benchmarks, benchmark_metadata = _research_benchmarks(
        main_panel, index_panel, backtest.daily_account, config
    )
    manifest = _manifest(
        config,
        {
            "main_panel": _input_info(main_panel, main_path),
            "backtest_panel": _input_info(backtest_panel, main_path),
            "index_panel": _input_info(index_panel, index_path),
            "corporate_actions": _input_info(corporate_actions, actions_path),
        },
        config_path=config_path,
    )
    manifest["benchmarks"] = benchmark_metadata
    manifest["alignment"] = {
        "method": "signals自身日期筛选后与面板按(date,code)一对一连接；run_backtest前硬校验",
        "panel_rows": int(len(backtest_panel)),
        "signal_rows": int(len(backtest_signals)),
        "key_difference_count": 0,
        "window_start": str(backtest_panel["date"].min()),
        "window_end": str(backtest_panel["date"].max()),
    }

    output_directory = root / "results" / "baseline"
    output_directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "daily_account.parquet": backtest.daily_account,
        "daily_positions.parquet": backtest.daily_positions,
        "trades.parquet": backtest.trades,
        "orders.parquet": backtest.orders,
        "round_trips.parquet": backtest.round_trips,
        "action_log.parquet": backtest.action_log,
    }
    for name, frame in tables.items():
        frame.to_parquet(output_directory / name, index=False)
    annual_rolling.to_csv(output_directory / "annual_and_rolling.csv", index=False)
    benchmarks.to_csv(output_directory / "benchmarks.csv", index=False)
    (output_directory / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / "summary.md").write_text(
        _markdown_summary(config, summary, annual_rolling, benchmarks, benchmark_metadata),
        encoding="utf-8",
    )
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BaselineResult(
        backtest=backtest,
        summary=summary,
        annual_rolling=annual_rolling,
        benchmarks=benchmarks,
        benchmark_metadata=benchmark_metadata,
        manifest=manifest,
        output_dir=output_directory,
    )
