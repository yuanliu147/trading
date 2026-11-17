import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NEGATIVE_RESULT = ROOT / "92_负结果与后续边界.md"


class RejectedDeliveryTest(unittest.TestCase):
    def read_delivery(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"缺少负结果交付文档: {path.name}")
        return path.read_text(encoding="utf-8")

    def test_negative_result_explains_real_failure_evidence(self) -> None:
        text = self.read_delivery(NEGATIVE_RESULT)
        summary = json.loads(
            (ROOT / "results" / "robustness" / "summary.json").read_text(encoding="utf-8")
        )
        baseline = summary["baseline"]
        common = summary["common_window"]
        stress = summary["stress"]
        variants = summary["variants"]
        neighborhood = [row for row in variants if row["family"] == "neighborhood"]
        random_order = [row for row in variants if row["family"] == "random_order"]
        double = next(row for row in variants if row["family"] == "double_cost")
        required_evidence = (
            summary["verdict"],
            f"{baseline['total_return']:.6f}",
            f"{baseline['max_drawdown']:.6f}",
            "毛亏损",
            "费用",
            f"{baseline['win_rate']:.6f}",
            f"{baseline['average_win']:.6f}",
            f"{baseline['average_loss']:.6f}",
            "不是亏损上限",
            summary["profit_concentration"]["top_year"],
            f"{summary['profit_concentration']['remaining_equity_pnl']:.2f}",
            str(len(baseline["rolling_3y_returns"])),
            f"{sum(row['total_return'] > 0 for row in neighborhood)}/{len(neighborhood)}",
            f"{sum(row['total_return'] > 0 for row in random_order)}/{len(random_order)}",
            f"{double['total_return']:.6f}",
            f"{common['hithink_total_return']:.6f}",
            f"{stress['total_return']:.6f}",
            f"{stress['max_drawdown']:.6f}",
            "数据已支持",
            "尚属假设",
            "非门槛",
            "停牌陈旧估值",
        )
        for evidence in required_evidence:
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, text)

    def test_negative_result_has_stop_boundaries_without_rescue_guidance(self) -> None:
        text = self.read_delivery(NEGATIVE_RESULT)
        for boundary in ("不得进入实盘观察", "另立预注册", "全部失败", "停止新策略探索", "不无限调参"):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, text)
        for forbidden in (
            "换个参数",
            "继续救系统",
            "尝试其他参数",
            "加入市场闸门后",
            "可进入实盘观察",
            "建议实盘",
            "实盘候选",
            "买入建议",
            "卖出建议",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_cost_attribution_does_not_exclude_slippage(self) -> None:
        text = self.read_delivery(NEGATIVE_RESULT)
        for required in (
            "显式费用不是亏损唯一来源",
            "毛盈亏仍含滑点",
            "没有零滑点",
            "独立滑点消融",
            "未证实假设",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        for overclaim in ("不能把失败完全归因于佣金或滑点", "原本正收益"):
            with self.subTest(overclaim=overclaim):
                self.assertNotIn(overclaim, text)

    def test_rejected_delivery_does_not_create_live_manual_or_daily_files(self) -> None:
        forbidden_paths = (
            "91_系统手册.md",
            "03_盘后模拟协议.md",
            "scripts/daily_plan.py",
            "src/trend_system/daily.py",
            "state/positions.csv",
            "state/forward_ledger.csv",
        )
        for relative_path in forbidden_paths:
            with self.subTest(path=relative_path):
                self.assertFalse((ROOT / relative_path).exists(), f"REJECTED 不应交付: {relative_path}")

    def test_readme_contains_complete_reproduction_and_io_boundaries(self) -> None:
        text = self.read_delivery(README)
        commands = (
            "PYTHONPATH=src python3 -B scripts/build_panel.py --config config/baseline.json",
            "PYTHONPATH=src python3 -B -m unittest discover -s tests -v",
            "PYTHONPATH=src python3 -B scripts/run_baseline.py --config config/baseline.json",
            "PYTHONPATH=src python3 -B scripts/run_robustness.py --config config/baseline.json",
        )
        positions = []
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(command, text)
                positions.append(text.index(command))
        self.assertEqual(positions, sorted(positions), "复现命令顺序不完整")
        for boundary in (
            "read_only=True",
            "只读",
            "大结果",
            "Parquet",
            "CSV",
            "summary.json",
            "run_manifest.json",
            "标准输出",
            "90_研究结论.md",
            "92_负结果与后续边界.md",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, text)


if __name__ == "__main__":
    unittest.main()
