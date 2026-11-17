# Task 4 报告：实现原价涨跌停、费用、滑点和整手

## 状态

DONE

## 文件变更

- 新建 `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/execution.py`
  - 实现 `price_limits(prev_close, is_st)`。
  - 实现 `transaction_costs(date, amount, side, multiplier=1.0)`。
  - 实现 `Fill`、`Rejection` 与 `open_fill(order, bar, config)`。
  - 实现 `size_buy(order, equity, cash, invested_value, config)`。
- 新建 `/Users/admin/WorkSpace/trading/trading_finally/tests/test_execution.py`
  - 初版增加 18 项标准库 `unittest` 测试，fix round 1 后共 19 项；覆盖精确涨跌停、历史费用、最低佣金、费用倍率、原价目标市值、现金不足减手、总暴露、双边开盘成交、滑点边界、固定拒绝原因、订单处理方向及 sizing/fill 最终成交价一致性。
- 新建 `/Users/admin/WorkSpace/trading/trading_finally/.sdd/task-4-report.md`。

未修改既有数据、信号、配置、预注册或账户代码；未访问源数据库；未执行账户循环或收益回测。

## 固定接口契约

简报未展开 `order/bar` 的字段结构，本任务将其固定为支持 `[]` 和 `.get()` 的记录（字典或 pandas 行）：

- `open_fill` 的 `order` 使用 `side`、`quantity`；`side` 仅接受 `buy/sell`。买入数量必须为正整数且是 100 股整数倍；卖出数量必须为正整数，允许公司行动造成的零股。
- `open_fill` 的 `bar` 使用 `date/open/high/low/prev_close/is_st`，并读取 `trade_status/data_valid/execution_conflict`。后三项缺省分别按 `1/True/False`，显式异常值会触发拒绝。
- `size_buy` 的 `order` 使用执行日 `date` 和未复权、未滑点的 `open`；可携带 `high/low/prev_close/is_st`，以便与 `open_fill` 使用完全相同的最终成交价边界。未携带这些可选边界时保留原有仅按开盘价施加滑点的行为。
- `Fill` 固定字段为 `price/quantity/amount/costs/cash_delta`。`amount` 是精确到分的成交额；`costs` 包含 `commission/stamp_duty/transfer_fee/total`；`cash_delta` 已包含全部费用，买入为负、卖出为正，Task 5 不需重算成交费用。
- `Rejection` 固定字段为 `reason/disposition`。买入拒绝为 `disposition="cancel"`，卖出拒绝为 `disposition="keep"`；本原语只返回处理方向，不实现 Task 5 的逐日顺延。
- 固定机器原因包括 `invalid_order/invalid_bar/suspended/execution_conflict/price_out_of_bounds/buy_limit_up/sell_limit_down`。

## 关键实现选择

- 涨跌停以 `Decimal(str(prev_close))` 计算，主板非 ST 为 10%、ST 为 5%，按 `ROUND_HALF_UP` 保留到 0.01 元；9.99 元得到 10.99/8.99。
- 佣金固定万 2.5、最低 5 元；卖出印花税在 2023-08-28 起从 0.1% 降为 0.05%；双边过户费在 2022-04-29 起从 0.00002 降为 0.00001。
- 各费用组件以 Decimal 计算并四舍五入到分，`total` 为已舍入组件之和。`multiplier` 同时放大最低佣金及其他费用，供冻结成本压力使用。
- 买入滑点为开盘价上浮 `config.slippage_bps`，卖出为下浮；成交价被限制在当日高低价和涨跌停合法范围内，不以盘中后续打开涨跌停推定开盘可成交。
- 开盘涨停拒绝买入，开盘跌停拒绝卖出；无效数据、停牌、执行冲突及开盘价越界均在成交前明确拒绝。
- `size_buy` 复用 `ResearchConfig.max_weight/max_exposure/slippage_bps`，不新增策略参数。数量上限先按未滑点开盘价及 `min(20% 单票额度, 60% 总暴露剩余额度)` 向下取整手，再按与 `open_fill` 共用的最终成交价、成交额和全部费用检查现金；不足时每次减少 100 股，直到可负担或为零。
- 2 万权益、10 元开盘的基准例子为 400 股：未滑点目标市值 4,000 元；10.01 元成交价对应成交额 4,004 元、最低佣金 5 元、过户费 0.04 元，总现金扣款 4,009.04 元。现金仅 4,005 元时减至 300 股。

## TDD 证据

### RED

先创建完整失败测试，未创建执行模块，随后从 `trading_finally/` 执行规定命令：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_execution -v
```

结果按预期失败：

```text
ModuleNotFoundError: No module named 'trend_system.execution'
Ran 1 test in 0.000s
FAILED (errors=1)
```

失败原因确为执行模块不存在，而非测试数据或断言错误。

### GREEN：执行原语测试

实现后执行同一命令：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_execution -v
```

结果：`Ran 18 tests in 0.001s`，`OK`。18 项全部通过，无失败、错误或警告；测试中的所有成交数量均为 100 股整数倍。

### GREEN：完整项目回归

执行：

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

结果：`Ran 45 tests in 0.177s`，`OK`。配置 5 项、数据 15 项、执行 18 项、信号 7 项全部通过，无错误或警告。

## Fix round 1：sizing 与 fill 共用最终成交价

### 审查问题与修复

独立审查指出：`open_fill` 使用高低价和涨跌停边界夹取后的价格，而 `size_buy` 使用未夹取滑点价，导致现金充足时错误减手。

本次仅修复该 Important：

- 提取 `_final_fill_price(record, side, config)`，`open_fill` 与 `size_buy` 均调用此函数计算最终成交价。
- `size_buy` 可通过 `order` 的 `high/low/prev_close/is_st` 获得与成交相同的边界；函数签名不变。
- 未提供边界的订单仍按原滑点价估算，原现金不足减手行为不变。
- 审查例子 `open=high=10/low=9.5/prev_close=10` 下，400 股最终成交价为 10 元，总扣款为 4,005.04 元；现金 4,006 元时买入 400 股，现金 4,005.03 元时减为 300 股，与 `Fill.cash_delta` 精确一致。

### RED：审查回归测试

先新增 `test_size_buy_uses_same_clamped_fill_price_and_cash_delta_as_open_fill`，未修改实现，执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_execution -v
```

结果按预期失败，其余 18 项通过：

```text
AssertionError: 300 != 400
Ran 19 tests in 0.001s
FAILED (failures=1)
```

### GREEN：修复后执行测试

实现共用最终成交价函数后执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_execution -v
```

结果：`Ran 19 tests in 0.001s`，`OK`。新增审查例子、原无边界例子及原现金不足减手测试全部通过。

### GREEN：修复后全项目回归

执行：

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

结果：`Ran 46 tests in 0.172s`，`OK`。配置 5 项、数据 15 项、执行 19 项、信号 7 项全部通过，无错误或警告。

## 自审

- **原价边界**：所有成交、费用、整手和涨跌停只使用未复权 OHLC/prev_close；接口不读取信号复权价。
- **现金一致性**：`size_buy` 与 `open_fill` 共用最终成交价和费用函数，成交额及现金变动精确到分；取整后现金不足会减手，不会透支，也不会因未复用价格夹取而误减手。
- **仓位口径**：单票和总暴露目标按未滑点开盘价计算，避免把滑点支出误计入目标市值；既有持仓市值通过 `invested_value` 扣除总暴露额度。
- **执行边界**：买入开盘涨停、卖出开盘跌停、停牌、无效数据、执行冲突和价格越界都有测试；滑点不会穿出高低价或合法涨跌停范围。
- **任务边界**：未实现 T+1 可卖、卖单实际延期、账户账本、公司行动或收益回测，均留给后续任务。
- **配置边界**：只读取冻结配置已有字段，未新增配置项或参数覆盖入口。
- **改动范围**：未修改相邻模块、历史文档或真实数据，未访问数据库，未执行 `git add/commit/push`，未派发子代理。

## 顾虑

无阻塞顾虑。Task 5 应直接复用上述记录字段、`Fill.cash_delta` 和 `Rejection.disposition`，避免在账户层另建一套执行口径。
