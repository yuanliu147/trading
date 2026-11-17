# Task 5 报告：完整账户账本、顺延退出和公司行动

## 1. 状态与范围

已在 `trading_finally` 内完成合成数据执行引擎及测试。未运行真实收益回测、未调参、未访问源数据库，未修改配置、信号、执行原语、预注册或真实数据，未执行任何 Git 暂存、提交或推送。

## 2. 文件

- 新增：`/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/broker.py`
- 新增：`/Users/admin/WorkSpace/trading/trading_finally/tests/test_broker.py`
- 本报告：`/Users/admin/WorkSpace/trading/trading_finally/.sdd/task-5-report.md`

## 3. 接口与输出

### `run_backtest(panel, signals, corporate_actions, config) -> BacktestResult`

直接消费 Task 4 的 `size_buy` 和 `open_fill`。买入数量在执行日用实际开盘 bar 和前一收盘冻结的资源快照计算；成交价格、费用、现金变动全部取自 `Fill`，拒单及顺延处置取自 `Rejection`。账户层没有重算滑点、费用或涨跌停。

`BacktestResult` 包含六个 DataFrame，即使为空也保留稳定列：

- `trades`：`order_id/signal_date/date/code/side/quantity/price/amount/commission/stamp_duty/transfer_fee/costs/cash_delta`。
- `orders`：`order_id/signal_date/date/code/side/quantity/status/reason`；`status` 为 `filled/cancelled/pending`，卖单的多次尝试共用 `order_id`。
- `round_trips`：`code/entry_date/exit_date/quantity/entry_quantity/exit_quantity/entry_amount/entry_costs/exit_amount/exit_costs/total_costs/gross_pnl/net_pnl/return_pct/exit_reason`；只含已闭合往返，`quantity` 与 `entry_quantity` 均为入场股数，`exit_quantity` 累计实际卖出及退市核销股数（包含送转新增股），`net_pnl` 使用实际现金流并扣除双边费用。
- `daily_account`：`date/cash/receivables/market_value/equity/exposure/drawdown/stale_valuation/position_count`。
- `daily_positions`：`date/code/entry_date/quantity/available_quantity/close/market_value/stale_valuation/pending_exit`。
- `action_log`：`date/code/action/quantity/amount/detail`，覆盖分红应收与结转、送转、配股不参与、复权源冲突和退市零残值。

账户恒等式为 `equity = cash + receivables + market_value`；`exposure = market_value / equity`（权益非正时记零）；`drawdown = equity / historical_peak - 1`，即回撤为非正数。分红在当日收盘结转，因此日终 `receivables` 通常为零，应收及结转过程保留在 `action_log`。

## 4. 精确 TDD 证据

### 初始 RED

在只有 `tests/test_broker.py`、没有 `broker.py` 时运行：

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest tests.test_broker -v
```

结果：退出码 1，`Ran 1 test ... FAILED (errors=1)`，核心错误为：

```text
ModuleNotFoundError: No module named 'trend_system.broker'
```

修正测试数据日期后再次运行，得到相同的缺失模块 RED。

### 自审新增边界的 RED

首轮 GREEN 后，自审发现“往返股数在完全卖出后变成零”和“公司行动空送转字段触发 NaN 转整数错误”两个边界。先增加断言，再运行：

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest \
  tests.test_broker.BrokerTest.test_entry_executes_next_open_and_bought_shares_are_not_sellable_same_day \
  tests.test_broker.BrokerTest.test_null_corporate_action_fields_are_treated_as_zero -v
```

结果：退出码 1，一项 `AssertionError: np.int64(0) != np.int64(400)`，一项 `ValueError: cannot convert float NaN to integer`。随后分别保存 `entry_quantity`、增加空数值到零的转换，同一命令变为 `Ran 2 tests ... OK`。

### 首轮指定 GREEN（历史）

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest tests.test_broker tests.test_execution -v
```

最终结果：退出码 0，`Ran 37 tests in 0.055s`，`OK`。其中 broker 18 项、execution 19 项。

### 首轮全项目 GREEN（历史）

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest discover -v
```

最终结果：退出码 0，`Ran 64 tests in 0.220s`，`OK`，无失败、错误或警告。

## 5. 关键时序选择

1. **T 日收盘**：用当日实际收盘或陈旧估值完成日终估值；按 `rank_return_120` 降序、代码升序确定候选；已有持仓不重复加仓。冻结全部合格候选以及 T 日 `slots/cash/equity/invested` 快照，不按空槽预截断。拟卖但未卖的持仓仍占槽位和 invested。
2. **T+1 开盘前**：先纯验证全部已有持仓的 `data_valid` 和 `execution_conflict`，全部通过后才更新 `out_date` 或写复权冲突日志。任一持仓存在阻断条件立即抛出带 `date/code/flag` 的 `RuntimeError`，不删除交易、不以未来数据替代，也不对部分持仓先做状态更新。`adjustment_source_conflict` 只写日志。
3. **可卖数量**：每个新交易日先把既有持仓全部转为可卖；当日买入数量记为不可卖。送转在除权日增加真实股数，但新增股数不加入当日可卖数量，下一交易日才释放。
4. **公司行动**：分红按交易前持仓计算并先计应收；送转在实际卖出前应用；配股不自动缴款，只记录保守不参与假设。
5. **先卖后买但资源不串联**：卖单先调用 `open_fill`；随后买单仍按 T 日排名和代码顺序处理，只有实际成交扣减冻结槽位，拒单、缺 bar、无数量等取消不扣槽，后续候选可有序递补；槽位耗尽后剩余订单以 `no_slot` 取消。买单用 T+1 实际 bar 调用 `size_buy/open_fill`，现金参数取 `min(冻结剩余现金, 当前实际现金)`，权益和已投资额度使用冻结快照及本批买单自身消耗。正向卖出回款和除权日分红不增加冻结预算，负向卖出净支出会降低可用预算，T+1 信息不参与重排。
6. **订单生命周期**：买单只尝试 T+1 一次，拒单、缺 bar 或数量不足均取消；后续必须出现新入场信号。卖单未成交持续保留，停牌、缺 bar、连续跌停均顺延；每次可执行尝试都使用当日 bar，由执行原语施加当日滑点。若卖出 `cash_delta` 会使现金小于零，则不改现金或持仓，记录 `insufficient_cash_for_sell` 并继续顺延。公司行动导致部分股数暂不可卖时，卖单对剩余数量继续顺延。
7. **收盘**：分红应收转为现金并记录事件；正常交易持仓用实际收盘估值，停牌或缺 bar 持仓沿用最后实际收盘并标记 `stale_valuation`。随后生成下一交易日退出和入场批次。面板期末不强平。
8. **退出与退市**：普通退出和 ST 退出都在信号日收盘后冻结，下一交易日才尝试。只有当前账户日期严格晚于 `out_date`、持仓仍无法退出且无清算对价字段时，才按零残值核销并单独记录；绝不把 `out_date` 用作提前卖出信号。

## 6. 合成测试覆盖

18 个 broker 测试覆盖：

- 次日开盘成交、买入当日不可卖、ST 次日退出。
- 买单取消后必须由新信号重新生成。
- 连续两个开盘跌停后的第三个可执行开盘成交，且最终价格使用成交日开盘滑点。
- 停牌和缺 bar 的陈旧估值，持仓市值不被当作现金。
- T 日拟卖持仓继续占槽，预期卖出回款不能资助 T+1 买单。
- 每日现金非负、股数非负、权益恒等式。
- 分红当日应收与收盘结转、当日新增分红不能资助冻结买单。
- 送转新增股数下一交易日可卖，已有卖单对新增股数顺延。
- 配股不自动缴款，公司行动空数值字段按零处理。
- 退市零残值且不提前卖出、期末不强平。
- 持仓无效数据和执行冲突明确中止并诊断，复权源冲突只记录。
- 排名顺序与已有持仓不重复加仓。
- 完整往返的入场股数、实际净现金流和退出原因。

## 7. 自审与顾虑

- **资源前视检查**：买入数量需要 T+1 开盘 bar，因此冻结的是 T 日候选和资源，而不是在 T 日提前用未来开盘价计算股数；实际股数只在 T+1 执行时计算。
- **现金检查**：卖出和分红不会增加 `remaining_cash`；多笔买单只消耗冻结预算。执行现金只应用 `Fill.cash_delta`，测试逐日验证非负和恒等式。
- **公司行动近似**：现金分红沿用简报指定的“除权日应收、当日收盘可用”近似；送转产生非整数股时向下取整数股；配股不参与。这些不是收益调参。
- **退市边界**：零残值在严格晚于 `out_date` 的首个账户日期确认。如果面板在 `out_date` 当日或之前结束，则依照“期末不强平”仍保留持仓及陈旧估值；真实退市清算对价不在当前字段契约中。
- **验证边界**：所有验证均为合成单元测试；按任务要求没有运行真实基线，因此不对真实收益或全市场运行性能作结论。

## 8. 独立审查 fix round 1

### 修复内容

- **有序递补**：取消 `candidates[:empty_slots]` 预截断。T 日冻结全部合格候选、原排名和空槽快照；T+1 仅实际成交扣槽，首候选涨停、缺 bar 或 `no_size` 后均由次候选递补，不用 T+1 信息重排。多买继续按排名扣减冻结现金和 invested。
- **卖出净支出与现金保护**：卖出现金入账前先检查 `cash + fill.cash_delta`；为负则保留卖单并记录 `insufficient_cash_for_sell`。买单 sizing 现金改为 `min(remaining_cash, cash)`，使已接受的负向卖出净支出降低预算，正向卖出回款仍不能增加冻结预算。
- **往返数量**：新增 `entry_quantity/exit_quantity`，保留原 `quantity` 作为入场股数兼容字段。送转后完整往返验证入场 400 股、累计退出 600 股；退市核销剩余股数也计入 `exit_quantity`，核销现金流为零。
- **中止原子性**：持仓预检拆成两个循环，先纯验证全部持仓，全部通过后才更新 `out_date` 和写复权冲突日志。
- **聚焦回归**：新增完整账户日历、缺 bar 卖单顺延、分红与卖出同日按售前股数计算，以及同日多买预算扣减测试。

### 本轮 RED 证据

先增加审查对应测试，再运行：

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest tests.test_broker -v
```

结果：退出码 1，`Ran 25 tests`，`FAILED (failures=6, errors=1)`。具体命中：

- `test_failed_first_candidate_does_not_block_ordered_backfill` 的 `limit_up/missing/no_size` 三个子场景均未让第二候选成交。
- `test_negative_small_lot_sell_is_kept_when_it_would_overdraw_cash` 产生现金 `-0.02`。
- `test_negative_sell_reduces_cash_available_to_frozen_buy` 产生现金 `-1.51`。
- `test_multi_position_abort_precedes_all_position_updates` 检出了中止前的非空 `out_date` 更新。
- 送转往返断言因缺少 `entry_quantity` 字段报错。
- 同日多买预算、完整日历、缺 bar 顺延和分红同日 guard 在 RED 运行中已通过，用于防止修复引入旁路回归。

### 本轮 GREEN 证据

修复后单独运行 `tests.test_broker`：`Ran 25 tests in 0.082s`，`OK`。

指定回归：

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest tests.test_broker tests.test_execution -v
```

结果：退出码 0，`Ran 44 tests in 0.080s`，`OK`，其中 broker 25 项、execution 19 项。

全项目：

```text
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest discover -v
```

结果：退出码 0，`Ran 71 tests in 0.253s`，`OK`，无失败、错误或警告。
