### Task 7: 执行稳健性、历史压力和机械否决验收

**Files:**
- Create: `trading_finally/src/trend_system/robustness.py`
- Create: `trading_finally/src/trend_system/report.py`
- Create: `trading_finally/scripts/run_robustness.py`
- Create: `trading_finally/90_研究结论.md`
- Test: `trading_finally/tests/test_robustness.py`
- Test: `trading_finally/tests/test_report.py`

**Interfaces:**
- Consumes: 冻结的 `run_baseline`、信号、面板和公司行动。
- Produces: `run_robustness(config) -> RobustnessResult`；`evaluate_gates(summary) -> str`，返回值严格为 `UNREJECTED | BOUNDARY | REJECTED`；`render_final_report(...) -> Path`。

- [ ] **Step 1: 写变体完整性和验收命名失败测试**

断言邻域只能单因子改变：入场 90/150、趋势 180/220、斜率 10/30、退出 15/25，共 8 个；不得从结果中选一个替换基线。构造权益摘要用例：最大回撤大于 10% 为 `REJECTED`，8%—10% 为 `BOUNDARY`，其余条件通过且不超过 8% 也只能为 `UNREJECTED`，任何输出不得包含 `OOS_PASS`。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_robustness tests.test_report -v`
Expected: FAIL，提示稳健性或报告模块不存在。

- [ ] **Step 3: 实现预先冻结的稳健性矩阵**

固定执行：
- 8 个单因子邻域；
- 同花顺后复权信号面板的独立对照：从该源具备完整指标热身历史的共同日期开始，复用同一原价执行与组合规则，报告信号差异、收益和回撤，不把较短共同窗口冒充完整主窗口；
- 双倍全部费用和单边 20bp 滑点；
- 固定种子 20261101 的 20 个随机排序路径；
- 每个信号日保留信号数量、用当日其他合格股票替换入场的 20 个随机入场对照；
- 20 个交易日块、1,000 次日期块自助，仅作探索性区间；
- 2 槽位×20% 和 4 槽位×15% 两个组合敏感性；
- 年度、月末滚动三年、指数趋势/震荡/下跌和大小盘风格分组；
- 主结果及配置哈希冻结后，最后运行 2010—2015 单源压力测试。

机械门槛固定为：主窗口净收益为正；超过半数月末滚动三年收益为正；双倍成本净收益为正；8 个邻域至少 6 个净收益为正；20 个随机排序路径至少 16 个净收益为正；去掉贡献最高年份后剩余年份合计收益为正；同花顺复权信号对照的共同窗口净收益也为正且最大回撤不超过 10%，两源方向相反时不得未否决。回撤不超过 8% 才可能 `UNREJECTED`，8%—10% 为 `BOUNDARY`，超过 10% 或其他门槛失败为 `REJECTED`。

- [ ] **Step 4: 运行测试和完整稳健性流程**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_robustness tests.test_report -v`
Expected: PASS。

Run: `PYTHONPATH=src python3 -B scripts/run_robustness.py --config config/baseline.json`
Expected: `results/robustness/` 和 `results/stress_2010_2015/` 完整落盘；报告同时展示负结果、利润集中度、随机路径和单源局限，不只展示最佳变体。

- [ ] **Step 5: 生成最终研究裁决**

Run: `PYTHONPATH=src python3 -B -m unittest discover -s tests -v`
Expected: 全部 PASS；`90_研究结论.md` 明确列出每项机械门槛及 `UNREJECTED/BOUNDARY/REJECTED`，历史正收益不被称为确认性 OOS。

---

