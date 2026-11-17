#!/usr/bin/env python3
"""纯合成统计测试；不读取行情、不执行真实策略。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from 研究主流程 import bootstrap, control_id, specs, summarize


class StatisticalTests(unittest.TestCase):
    def setUp(self):
        self.core = [SimpleNamespace(id=f"K{i}", exit_kind="none", months=3) for i in range(12)]
        self.control = SimpleNamespace(id="C_M3", exit_kind="none", months=3)
        self.dates = pd.bdate_range("2010-12-31", periods=505).strftime("%Y-%m-%d").tolist()

    def test_constant_positive_and_identical_pair(self):
        logs = np.full((13, 8, 504), 0.0002)
        table = bootstrap(self.core, self.core + [self.control], logs, self.dates)
        own = table[table["比较"] == "账户对数收益对零"]
        paired = table[table["比较"] == "同种子同退出控制差"]
        self.assertEqual(len(table), 24)
        self.assertTrue((own["95点区间下限"] > 0).all())
        self.assertTrue((own["Holm校正p"] < 0.05).all())
        self.assertTrue((paired["年化对数均值"] == 0).all())
        self.assertTrue((paired["单侧p"] == 1).all())

    def test_constant_negative(self):
        logs = np.full((13, 8, 504), -0.0002)
        table = bootstrap(self.core, self.core + [self.control], logs, self.dates)
        own = table[table["比较"] == "账户对数收益对零"]
        self.assertTrue((own["单侧p"] == 1).all())
        self.assertTrue((own["95点区间上限"] < 0).all())

    def test_frontier_excludes_dominated_and_diagnostic(self):
        rows = []
        for ident, cagr, drawdown, group in [
            ("A", .1, -.2, "core"), ("B", .05, -.15, "core"),
            ("C", .02, -.3, "core"), ("D", .8, -.01, "diagnostic"),
        ]:
            for seed in range(8):
                row = {"配置": ident, "类别": group, "种子": seed, "CAGR": cagr,
                       "最大回撤": drawdown, "2011-2015_CAGR": cagr, "2016-2020_CAGR": cagr}
                for key in ["期末净值", "平均仓位", "完整回合数", "胜率", "盈亏比", "费用合计", "最大单票权重"]:
                    row[key] = 1.0
                rows.append(row)
        summary = summarize(pd.DataFrame(rows), ["A", "B", "C"])
        self.assertEqual(set(summary.loc[summary["描述性前沿"], "配置"]), {"A", "B"})

    def test_registered_count_and_exact_controls(self):
        core, controls, diagnostics = specs()
        self.assertEqual((len(core), len(controls), len(diagnostics)), (12, 8, 10))
        self.assertEqual(len({s.id for s in core + controls + diagnostics}), 30)
        mapping = {s.id: s for s in controls}
        for s in core:
            matched = mapping[control_id(s)]
            self.assertEqual((s.months, s.exit_kind, s.window, s.multiple, s.slots),
                             (matched.months, matched.exit_kind, matched.window, matched.multiple, matched.slots))
            self.assertEqual(matched.score, "control")


if __name__ == "__main__":
    unittest.main(verbosity=2)
