"""仅合成数据；python3 -B 本文件。不联网、不读真实行情、不生成真实冻结/留出输出。"""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import 账户回测 as account
from 统计检验 import probability_backtest_overfit
from 策略注册 import registry
from 现金账本 import run_broker
from 测试_现金账本 import LOOSE, change, fixture, order


class FreezeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="账户冻结单测_", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.strategy = registry()[0]
        self.enterContext(patch.object(account, "ROOT", self.root))
        (self.root / "02_代码").mkdir()
        (self.root / "01_数据").mkdir()
        for name in account.FREEZE_SOURCES:
            (self.root / "02_代码" / name).write_text("# 合成源码 " + name, encoding="utf-8")
        inputs = {}
        for period in ("开发", "验证"):
            relative = f"01_数据/{period}合成输入.csv"
            (self.root / relative).write_text("合成输入\n1\n", encoding="utf-8")
            inputs[period] = {relative: account._file_sha256(self.root / relative)}
        self.frozen = {"schema_version": 1, "allowed_strategies": {self.strategy.name: asdict(self.strategy)},
                       "source_sha256": {name: account._file_sha256(self.root / "02_代码" / name)
                                         for name in account.FREEZE_SOURCES},
                       "input_sha256": inputs, "allowed_scenarios": deepcopy(account.FREEZE_SCENARIOS)}

    def freeze_text(self, text):
        # 连临时目录的04_证据也不创建；只模拟唯一指定路径的冻结读取。
        def read(path, *args, **kwargs):
            self.assertEqual(path, self.root / "04_证据" / "留出前冻结.json")
            if text is None:
                raise FileNotFoundError(path)
            return text
        return patch.object(Path, "read_text", autospec=True, side_effect=read)

    def reject_before_orders(self, frozen=None, *, text=None, strategy=None, label="历史留出",
                             slippage=10, unknown_limit="normal", action_lag=0, match="冻结|策略"):
        if frozen is not None:
            text = json.dumps(frozen, ensure_ascii=False)
        data = SimpleNamespace(unknown_limit=unknown_limit)  # 故意没有dates；拒绝必须发生在访问日期前。
        with self.freeze_text(text), patch.object(account.pd, "read_parquet") as reader, \
                patch.object(account, "run_broker") as broker, patch.object(account, "actions_for") as actions:
            with self.assertRaisesRegex(RuntimeError, match):
                account.simulate(strategy or self.strategy, label, data, self.root / "不应生成输出",
                                 slippage, action_lag)
            reader.assert_not_called()
            broker.assert_not_called()
            actions.assert_not_called()
        self.assertFalse((self.root / "04_证据").exists())
        self.assertFalse((self.root / "不应生成输出").exists())

    def test_development_validation_do_not_need_freeze(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("不应读取冻结")):
            for label in ("开发", "验证"):
                self.assertIsNone(account.validate_freeze(self.strategy, label))

    def test_direct_simulate_rejects_missing_freeze_for_all_protected_periods(self):
        for label in ("历史留出", "二零二六", "全期"):
            with self.subTest(label=label):
                self.reject_before_orders(label=label)

    def test_empty_invalid_and_incomplete_freeze_fail_closed(self):
        for text in ("", "{}", "[]", "null", "{", '{"schema_version": 1}'):
            with self.subTest(text=text):
                self.reject_before_orders(text=text)
        for key in self.frozen:
            frozen = deepcopy(self.frozen)
            del frozen[key]
            with self.subTest(missing=key):
                self.reject_before_orders(frozen)

    def test_nested_missing_fields_and_invalid_hashes_rejected(self):
        variants = []
        for key in ("allowed_strategies", "source_sha256", "input_sha256", "allowed_scenarios"):
            frozen = deepcopy(self.frozen)
            frozen[key] = {}
            variants.append(frozen)
        frozen = deepcopy(self.frozen)
        del frozen["allowed_strategies"][self.strategy.name]["risk_per_trade"]
        variants.append(frozen)
        frozen = deepcopy(self.frozen)
        frozen["input_sha256"]["验证"] = {}
        variants.append(frozen)
        frozen = deepcopy(self.frozen)
        frozen["source_sha256"]["现金账本.py"] = "不是哈希"
        variants.append(frozen)
        frozen = deepcopy(self.frozen)
        frozen["input_sha256"]["开发"] = {"../越界.csv": "0" * 64}
        variants.append(frozen)
        for i, frozen in enumerate(variants):
            with self.subTest(case=i):
                self.reject_before_orders(frozen)

    def test_strategy_name_and_full_snapshot_must_match(self):
        self.reject_before_orders(self.frozen, strategy=registry()[1])
        self.reject_before_orders(self.frozen, strategy=replace(self.strategy, risk_per_trade=.02))

    def test_valid_schema_then_each_source_hash_change_is_rejected(self):
        with self.freeze_text(json.dumps(self.frozen)):
            self.assertEqual(account.validate_freeze(self.strategy, "历史留出"), self.frozen)
        for name in account.FREEZE_SOURCES:
            frozen = deepcopy(self.frozen)
            frozen["source_sha256"][name] = "0" * 64
            with self.subTest(source=name):
                self.reject_before_orders(frozen, match="SHA-256不符")
        path = self.root / "02_代码" / "现金账本.py"
        path.write_text("# 改动后的合成源码", encoding="utf-8")
        self.reject_before_orders(self.frozen, match="SHA-256不符")

    def test_development_and_validation_input_content_hashes_are_checked(self):
        for period in ("开发", "验证"):
            frozen = deepcopy(self.frozen)
            relative = next(iter(frozen["input_sha256"][period]))
            frozen["input_sha256"][period][relative] = "0" * 64
            with self.subTest(period=period):
                self.reject_before_orders(frozen, match="SHA-256不符")
        relative = next(iter(self.frozen["input_sha256"]["验证"]))
        (self.root / relative).write_text("改动后的合成输入", encoding="utf-8")
        self.reject_before_orders(self.frozen, match="SHA-256不符")

    def test_pressure_scenarios_must_be_explicitly_registered(self):
        with self.freeze_text(json.dumps(self.frozen)):
            for slip in (10, 20, 30):
                for limit in ("normal", "conservative"):
                    for lag in (0, 20):
                        account.validate_freeze(self.strategy, "历史留出", slip, limit, lag)
        frozen = deepcopy(self.frozen)
        frozen["allowed_scenarios"] = {"slippage_bps": [10], "unknown_limit": ["normal"], "action_lag": [0]}
        for kwargs in ({"slippage": 20}, {"unknown_limit": "conservative"}, {"action_lag": 20}):
            with self.subTest(kwargs=kwargs):
                self.reject_before_orders(frozen, **kwargs)
        frozen["allowed_scenarios"]["slippage_bps"] = [99]
        self.reject_before_orders(frozen, slippage=99)

    def test_cli_validates_before_constructing_data(self):
        with self.freeze_text(None), patch("sys.argv", ["账户回测.py", "--period", "历史留出",
                                                       "--names", self.strategy.name]), \
                patch.object(account, "TradingData") as data:
            with self.assertRaisesRegex(RuntimeError, "冻结"):
                account.main()
            data.assert_not_called()


def synthetic_orders(days, extra=False):
    rows = [order(days[0])]
    if extra:
        rows.append(order(days[1]))
    return pd.DataFrame(rows).assign(stop_distance=.1, episode_key="合成结构", st_status_known=True)


def audited_one_buy(result, orders, bars):
    """为旧成交fixture补充已知的真实合成开仓值，不从信号收盘猜风险。"""
    fills = result["成交"]
    buy_index = fills.index[fills["方向"].eq("买入")]
    if len(buy_index) != 1:
        raise AssertionError("此合成辅助只用于单笔买入")
    i = buy_index[0]
    buy = fills.loc[i]
    factor = bars.loc[bars.date.eq(buy["日期"]) & bars.code.eq(buy["代码"]), "adj_factor"].iloc[0]
    stop = orders.iloc[0].stop_adj / factor
    fills.loc[i, "信号日期"] = orders.iloc[0].signal_date
    fills.loc[i, "初始止损原价"] = stop
    fills.loc[i, "初始风险金额"] = (buy["成交原价"] - stop) * buy["股数"]
    fills.loc[i, "开仓权益"] = 20000.0
    fills.loc[i, "开仓权重"] = buy["成交金额"] / 20000.0


class AccountReconciliationTests(unittest.TestCase):
    def open_case(self):
        days, bars = fixture(start="2018-01-02")
        change(bars, days[-1], close=12, high=12.1)
        orders = synthetic_orders(days)
        result = run_broker(bars, orders, days, LOOSE)
        return days, bars, orders, result

    def test_open_position_total_pnl_includes_terminal_mark_to_market(self):
        days, _, orders, result = self.open_case()
        net = account.roundtrips(result, orders, days)
        row = net.iloc[0]
        self.assertFalse(row["已闭合"])
        self.assertEqual(row["净盈亏"], row["已实现净盈亏"])
        self.assertEqual(row["已实现净盈亏"], 0)
        self.assertGreater(row["未实现净盈亏"], 0)
        self.assertAlmostEqual(row["总净盈亏"], result["每日账户"].iloc[-1]["账户权益"] - 20000)
        self.assertTrue(pd.isna(row["净收益率"]))
        self.assertAlmostEqual(row["含浮盈净收益率"], row["总净盈亏"] / row["含费投入"])
        account.assert_accounting(result, net)

    def test_partial_sale_preserves_realized_and_remaining_unrealized_pnl(self):
        days, bars, orders, _ = self.open_case()
        result = run_broker(bars, orders, days, LOOSE,
                            close_instructions=[dict(date=days[1], code="000001", next_open_sell_fraction=.5)])
        net = account.roundtrips(result, orders, days)
        self.assertFalse(net.iloc[0]["已闭合"])
        self.assertTrue(pd.isna(net.iloc[0]["卖出日期"]))
        self.assertAlmostEqual(net["已实现净盈亏"].sum(), result["成交"]["已实现盈亏"].sum())
        account.assert_accounting(result, net)

    def test_same_day_exit_reentry_and_dividend_stays_with_original_position(self):
        days, bars = fixture(start="2018-01-02")
        orders = synthetic_orders(days, extra=True)
        old_pnl = []
        for pay_date in (days[-1], days[-1] + pd.Timedelta(days=30)):
            with self.subTest(pay_date=pay_date):
                actions = [dict(code="000001", ex_date=days[2], pay_date=pay_date,
                                share_multiplier=1, cash_dividend_per_old_share=.1)]
                result = run_broker(bars, orders, days, LOOSE, corp_actions=actions,
                                    close_instructions=[dict(date=days[1], code="000001", next_open_sell_fraction=1)])
                net = account.roundtrips(result, orders, days)
                first, second = net.iloc[0], net.iloc[1]
                self.assertTrue(first["已闭合"])
                self.assertFalse(second["已闭合"])
                self.assertEqual(first["卖出日期"], second["买入日期"])
                dividend = result["公司行动"]["新增应收"].sum()
                self.assertGreater(dividend, 0)
                self.assertAlmostEqual(first["已实现净盈亏"], result["成交"]["已实现盈亏"].sum() + dividend)
                self.assertEqual(second["已实现净盈亏"], 0)
                self.assertEqual(first["未实现净盈亏"], 0)
                self.assertAlmostEqual(first["净收益率"], first["总净盈亏"] / first["含费投入"])
                self.assertAlmostEqual(result["每日账户"].iloc[-1]["分红应收"], dividend if pay_date > days[-1] else 0)
                account.assert_accounting(result, net)
                old_pnl.append(first["总净盈亏"])
        self.assertAlmostEqual(*old_pnl)

    def test_share_identity_includes_bonus_shares_without_fictitious_trade(self):
        days, bars, orders, _ = self.open_case()
        result = run_broker(bars, orders, days, LOOSE, corp_actions=[
            dict(code="000001", ex_date=days[2], pay_date=days[-1], share_multiplier=1.5,
                 stock_pay_date=days[-1] + pd.Timedelta(days=30), cash_dividend_per_old_share=.1)])
        net = account.roundtrips(result, orders, days)
        self.assertGreater(result["期末持仓"].iloc[0]["锁定股数"], 0)
        account.assert_accounting(result, net)

    def test_empty_broker_result_still_reconciles(self):
        days, bars = fixture()
        orders = synthetic_orders(days).iloc[:0]
        result = run_broker(bars, orders, days, LOOSE)
        net = account.roundtrips(result, orders, days, require_entry_audit=True)
        self.assertTrue(net.empty)
        account.assert_accounting(result, net)

    def test_actual_stop_distance_uses_fill_and_initial_stop_not_signal_close(self):
        days, bars = fixture(start="2018-01-02")
        change(bars, days[1], open=10.5, high=10.6, low=10.4, close=10.5)
        orders = synthetic_orders(days)
        result = run_broker(bars, orders, days, LOOSE)
        audited_one_buy(result, orders, bars)
        row = account.roundtrips(result, orders, days, require_entry_audit=True).iloc[0]
        self.assertTrue(row["开仓审计完整"])
        self.assertAlmostEqual(row["计划止损距离"], .1)
        self.assertAlmostEqual(row["实际初始止损距离"], 1.5 / 10.5)
        self.assertAlmostEqual(row["初始风险金额"], 1.5 * row["初始股数"])
        self.assertEqual(row["信号日期"], days[0])

    def test_inconsistent_entry_risk_or_signal_date_fails_strict_audit(self):
        days, bars, orders, result = self.open_case()
        audited_one_buy(result, orders, bars)
        for field, value in (("初始风险金额", 1), ("开仓权益", 10000),
                             ("开仓权重", .9), ("信号日期", days[1])):
            with self.subTest(field=field):
                broken = {key: frame.copy(deep=True) for key, frame in result.items()}
                broken["成交"].loc[0, field] = value
                with self.assertRaisesRegex(AssertionError, "真实开仓审计"):
                    account.roundtrips(broken, orders, days, require_entry_audit=True)

    def test_old_fixture_compatibility_does_not_fabricate_entry_audit(self):
        days, _, orders, result = self.open_case()
        result["成交"] = result["成交"].drop(columns=["信号日期", "初始止损原价", "初始风险金额", "开仓权益", "开仓权重"], errors="ignore")
        row = account.roundtrips(result, orders, days).iloc[0]
        self.assertFalse(row["开仓审计完整"])
        self.assertTrue(pd.isna(row["实际初始止损距离"]))
        self.assertTrue(pd.isna(row["初始风险金额"]))
        with self.assertRaisesRegex(AssertionError, "真实开仓审计"):
            account.roundtrips(result, orders, days, require_entry_audit=True)

    def test_simulate_with_real_broker_and_only_synthetic_data(self):
        days, bars, orders, _ = self.open_case()
        lookup = {date: group.set_index("code").to_dict("index") for date, group in bars.groupby("date")}
        data = SimpleNamespace(dates=days, unknown_limit="normal", day_index={d: i for i, d in enumerate(days)},
                               get_day=lambda date, codes: {c: lookup[date][c] for c in codes},
                               features=lambda code: pd.DataFrame(index=days))
        with patch.object(account.pd, "read_parquet", return_value=orders), \
                patch.object(account, "actions_for", return_value=pd.DataFrame()), \
                patch.object(account.duckdb, "connect", side_effect=AssertionError("不得打开真实数据库")), \
                patch.object(pd.DataFrame, "to_csv") as save, \
                patch.object(Path, "mkdir"), patch.object(Path, "write_text"):
            metrics = account.simulate(LOOSE, "开发", data, Path("/tmp/不写出的合成账户"))
            self.assertEqual(metrics["买入笔数"], 1)
            self.assertGreater(metrics["期末权益"], 20000)
            self.assertTrue(save.called)

    def test_simulate_enforces_accounting_before_writing_results(self):
        days, bars, orders, base = self.open_case()
        audited_one_buy(base, orders, bars)
        data = SimpleNamespace(dates=days, unknown_limit="normal")
        for fault in (None, "现金", "股份", "总净盈亏", "开仓审计"):
            with self.subTest(fault=fault):
                result = {key: frame.copy(deep=True) for key, frame in base.items()}
                if fault == "现金":
                    result["每日账户"].loc[result["每日账户"].index[-1], "可用现金"] += 1
                elif fault == "股份":
                    result["每日持仓"].loc[result["每日持仓"].index[-1], "股数"] += 1
                elif fault == "开仓审计":
                    result["成交"]["初始风险金额"] = np.nan
                with patch.object(account.pd, "read_parquet", return_value=orders), \
                        patch.object(account, "actions_for", return_value=pd.DataFrame()), \
                        patch.object(account, "run_broker", return_value=result), \
                        patch.object(pd.DataFrame, "to_csv") as save, \
                        patch.object(Path, "mkdir"), patch.object(Path, "write_text") as write:
                    if fault == "总净盈亏":
                        net = account.roundtrips(result, orders, days, require_entry_audit=True)
                        net.loc[0, "总净盈亏"] += 1
                        with patch.object(account, "roundtrips", return_value=net):
                            with self.assertRaisesRegex(AssertionError, "逐笔总净盈亏"):
                                account.simulate(LOOSE, "开发", data, Path("/tmp/不写出的合成账户"))
                    elif fault:
                        with self.assertRaisesRegex(AssertionError, fault):
                            account.simulate(LOOSE, "开发", data, Path("/tmp/不写出的合成账户"))
                    else:
                        metrics = account.simulate(LOOSE, "开发", data, Path("/tmp/不写出的合成账户"))
                        self.assertGreater(metrics["期末权益"], 20000)
                        self.assertTrue(save.called)
                    if fault:
                        save.assert_not_called()
                        write.assert_not_called()


class SharpeSelectorDiagnosticTests(unittest.TestCase):
    def test_identical_candidates_are_indeterminate_not_one_hundred_percent(self):
        returns = np.random.default_rng(1).normal(.0001, .01, 160)
        for values in (returns, np.zeros(160)):
            summary, detail = probability_backtest_overfit(np.column_stack([values] * 4))
            self.assertTrue(np.isnan(summary["过拟合概率诊断"]))
            self.assertEqual(summary["非并列分母"], 0)
            self.assertEqual(summary["并列数"], summary["组合数量"])
            self.assertTrue(detail["并列"].all())
            self.assertTrue(detail["对数胜算"].eq(0).all())
            self.assertIn("不可判定", summary["判定状态"])
            self.assertIn("非预注册选择流程PBO", summary["诊断标签"])

    def test_partial_median_ties_are_excluded_and_explained(self):
        noise = np.tile([-.01, .01], 10)
        candidates = np.concatenate([np.column_stack([noise + shift, noise + .002, noise])
                                     for shift in (.003, .003, .001, .001)])
        summary, detail = probability_backtest_overfit(candidates, blocks=4)
        self.assertEqual(summary["组合数量"], 6)
        self.assertEqual(summary["并列数"], 2)
        self.assertEqual(summary["非并列分母"], 4)
        non_tied = detail.loc[~detail["并列"], "对数胜算"]
        self.assertEqual(summary["过拟合概率诊断"], (non_tied < 0).mean())
        self.assertTrue(detail.loc[detail["并列"], "并列原因"].str.contains("logit=0").all())
        self.assertIn("排除分母", summary["并列处理"])
        self.assertIn("首列", summary["并列处理"])
        self.assertIn("不是预注册", summary["说明"])

    def test_non_tied_denominator_and_failure_fraction(self):
        candidates = np.random.default_rng(7).normal(0, .01, size=(160, 4))
        summary, detail = probability_backtest_overfit(candidates)
        self.assertEqual(summary["并列数"], 0)
        self.assertEqual(summary["非并列分母"], 70)
        self.assertEqual(summary["过拟合概率诊断"], (detail["对数胜算"] < 0).mean())


if __name__ == "__main__":
    unittest.main(verbosity=2)
