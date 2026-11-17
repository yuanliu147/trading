### Task 4: 实现原价涨跌停、费用、滑点和整手

**Files:**
- Create: `trading_finally/src/trend_system/execution.py`
- Test: `trading_finally/tests/test_execution.py`

**Interfaces:**
- Produces: `price_limits(prev_close: float, is_st: bool) -> tuple[float, float]`；`transaction_costs(date, amount, side, multiplier=1.0) -> dict`；`open_fill(order, bar, config) -> Fill | Rejection`；`size_buy(order, equity, cash, invested_value, config) -> int`。
- `Fill` 使用原价成交，`Rejection` 给出固定机器原因；后续组合引擎不自行重算这些规则。

- [ ] **Step 1: 写精确金额和边界失败测试**

至少断言：非 ST 前收 10 元的涨跌停为 11.00/9.00，ST 为 10.50/9.50；9.99 元按 `Decimal ROUND_HALF_UP` 得 10.99/8.99。买入 4,000 元佣金最低 5 元，卖出在 2023-08-28 前后印花税分别为 0.1%/0.05%。10 元开盘、20% 单票上限和 2 万权益时应买入 400 股，含 10bp 买入滑点、最低佣金和过户费后仍不破现金或单票市值上限。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_execution -v`
Expected: FAIL，提示执行模块不存在。

- [ ] **Step 3: 实现执行原语**

佣金率为 `0.00025`、最低 5 元；印花税卖出按历史日期；过户费在 2022-04-29 前为成交额 `0.00002`、此后为 `0.00001`。基线滑点为开盘价的不利 10bp，成交价不得越过当日高低价和合法价格边界。

开盘涨停或不可交易时买入返回当日取消；开盘跌停或不可交易时卖出返回保留。`size_buy` 同时受 20% 单票、60% 总暴露、已有现金和 100 股整手约束，不用后复权价格计算；目标市值按未滑点开盘价估值，实际现金扣款按滑点后的成交价和费用计算。

- [ ] **Step 4: 运行测试并确认通过**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_execution -v`
Expected: PASS；所有测试成交数量均为 100 的整数倍，现金费用精确到分。

---

