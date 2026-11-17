### Task 5: 实现完整账户账本、顺延退出和公司行动

**Files:**
- Create: `trading_finally/src/trend_system/broker.py`
- Test: `trading_finally/tests/test_broker.py`

**Interfaces:**
- Consumes: `compute_signals` 输出、公司行动表和 Task 4 执行原语。
- Produces: `run_backtest(panel, signals, corporate_actions, config) -> BacktestResult`。
- `BacktestResult` 包含 `trades/orders/round_trips/daily_account/daily_positions/action_log` DataFrame；账户表至少有现金、应收、持仓市值、权益、暴露、回撤和陈旧估值标记。

- [ ] **Step 1: 写账户时序失败测试**

用小型合成面板逐项覆盖：
- T 日入场只能在 T+1 开盘成交，买入当日不能卖出；
- 买入未成交仅当日取消，后续必须有新信号；
- 连续两个开盘跌停后，卖单在第三个可执行开盘成交并逐日计滑点；
- 停牌使用最后实际收盘估值并标记，不能以缺价当现金；
- T 日已有持仓拟退出且有新信号时，新买只能使用 T 日已有现金和空槽；
- 每日现金不为负、持股数不为负、权益等于现金加应收加持仓市值。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_broker -v`
Expected: FAIL，提示 `run_backtest` 不存在。

- [ ] **Step 3: 实现逐日账户引擎**

T 日收盘先冻结次日订单和买入预算；T+1 开盘先处理卖单，再处理已冻结买单，但买单不得使用当日卖出释放的槽位或预期回款。买单只尝试一次，卖单持续保留。已有持仓不重复加仓，排序只决定同日有限预算下的先后顺序。

持仓遇到入场后 `data_valid=False` 或 `execution_conflict=True` 时中止运行并输出诊断，不能悄悄用未来值或删除交易；`adjustment_source_conflict=True` 只记录并交给双源敏感性报告，不触发执行中止。期末不强平；退市后仍无法退出且缺少清算对价时按保守零残值处理并单列日志，绝不提前使用 `out_date` 卖出。

- [ ] **Step 4: 加入公司行动及风险测试并跑通**

现金分红在除权日计入应收，并在当日收盘转为可用现金；这是本地缺少派息到账日的明确近似。由于 T+1 买入预算已在 T 日冻结，除权日新增现金不得用于当日买单。送转增加真实股数，新增股数下一交易日可卖；配股不自动缴款，记录保守不参与的假设。加入退市零残值、ST 次日退出和陈旧估值压力测试。

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_broker tests.test_execution -v`
Expected: PASS，全部账户恒等式在每个合成交易日成立。

---

