### Task 6: 生成账户指标、基准和固定基线结果

**Files:**
- Create: `trading_finally/src/trend_system/metrics.py`
- Create: `trading_finally/scripts/run_baseline.py`
- Test: `trading_finally/tests/test_metrics.py`
- Test: `trading_finally/tests/test_baseline_integration.py`

**Interfaces:**
- Produces: `performance_summary(daily_account, round_trips) -> dict`；`annual_and_rolling(daily_account) -> pd.DataFrame`；`run_baseline(config) -> BaselineResult`。
- 指标包括总收益、CAGR、最大回撤、年度收益、月末滚动三年收益、持仓暴露、换手、扣费后期望、胜率、平均盈亏、连续亏损、年度与单票贡献、前五盈利交易贡献。

- [ ] **Step 1: 写已知权益路径和盈亏测试**

用固定权益序列断言最大回撤和年度收益；构造两笔一盈一亏交易，断言扣费后期望、平均盈利/亏损和最长连续亏损。月末滚动三年窗口必须按实际日期计数，2016 部分年度单独标记。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_metrics tests.test_baseline_integration -v`
Expected: FAIL，提示指标或基线入口不存在。

- [ ] **Step 3: 实现指标与一次基线运行**

`run_baseline` 只读取冻结配置，不接受策略参数命令行覆盖。加入现金、上证综指、沪深300、主板等权研究基准；价格指数基准注明不含股息，主板等权注明不可由 2 万账户直接复制。

每次运行保存配置与设计文档 SHA256、代码文件 SHA256、输入面板行数、数据窗口和依赖版本到 `results/baseline/run_manifest.json`。

- [ ] **Step 4: 先跑合成集成测试，再跑真实基线**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_metrics tests.test_baseline_integration -v`
Expected: PASS；合成结果重复运行完全一致，且没有 T 日收盘成交。

Run: `PYTHONPATH=src python3 -B scripts/run_baseline.py --config config/baseline.json`
Expected: 完整写出 `results/baseline/` 的账户、交易、指标和 Markdown 摘要；无论策略盈亏都正常完成，不为通过验收修改参数。

---

