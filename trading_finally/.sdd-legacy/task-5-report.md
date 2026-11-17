# Task 5：候选 A 可选 LegacyPolicy 账户钩子报告

日期：2026-11-04

## 1. 状态与范围

- 状态：**完成**。候选 A 的 `LegacyPolicy` 已作为可选对象接入既有 `run_backtest`；候选 B 仍为 `NOT_EVALUABLE`，未创建、未实现、未运行。
- 修改：`/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/broker.py`
- 新增：`/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/legacy_policy.py`
- 测试：`/Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_policy.py`
- 所有新内容均在 `trading_finally/`。未执行 git、未派发子代理、未连接活源库、未修改 `.sdd-legacy/snapshots/task5-base`。
- 只运行合成候选 A 反例和既有固定负收益 Donchian 基线；**未运行真实候选 A/B 账户收益**。

## 2. TDD：RED / GREEN 证据

### 2.1 首轮 RED

先创建合成测试，再运行：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_policy.py
```

实际结果：

```text
ModuleNotFoundError: No module named 'trend_system.legacy_policy'
Ran 1 test in 0.000s
FAILED (errors=1)
```

失败原因是被要求的新策略接口尚不存在，随后才创建 `legacy_policy.py` 并修改 `broker.py`。

### 2.2 最小实现后的反例校正

首次实现后实际运行 9 项测试，出现 2 项断言失败：

1. 同分候选的实际顺序为 `000001.SZ`、`600000.SH`，测试最初把代码 tie-break 的先后写反；按冻结的代码升序规则修正测试预期。
2. 既有 `orders.reason` 是执行状态诊断，filled 行保持空字符串；完整退出原因由卖单内部 `reason` 传入并最终写入 `round_trips.exit_reason`。测试改为验证既有真实接口，没有为候选 A 改变默认 orders 账本。

随后候选 A 与既有账户联合回归：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_policy.py \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_broker.py
```

实际结果：`Ran 38 tests ... OK`。其中新增候选 A 测试 9/9 通过。

### 2.3 最终全项目 GREEN

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest discover -s /Users/admin/WorkSpace/trading/trading_finally/tests
```

实际结果：

```text
Ran 168 tests in 3.586s
OK
```

## 3. 接口与默认分支隔离

### 3.1 可选接口

```python
run_backtest(panel, signals, corporate_actions, config, policy=None)
policy = LegacyPolicy(legacy_research_config)
```

`LegacyPolicy` 从 `LegacyResearchConfig.candidates` 中只选择 `three_of_two_low_vol`，使用其中冻结的 `max_slots=4`、`max_weight=0.20`、`max_new_per_day=2`、`entry_stop_pct=0.08`、`max_hold=40` 和入场否决开关。

- 排序：调用 Task 3 的 `order_legacy_a_candidates`，输入权威字段 `legacy_rank_score`，原分降序、代码升序 tie-break。没有读取 `rank_return_120` 作为候选 A 排序来源，也没有重算 Donchian 或 120 日收益。
- sizer：复用既有 `size_buy`，只覆盖候选 A 的单票 20% 权重；组合只受 4 槽、实际现金、100 股整手和实际成交费用约束。适配器的总暴露上限为 1.0，用来避免默认 Donchian 的 60% 上限泄漏到候选 A，不是新增择时或仓位规则。
- 入场状态：保存固定的实际尺度止损锚、入场点时因子和持有计数；没有每日重算 low10。
- 退出：按持仓状态、当前实际 bar 和 Task 3 收盘指标生成原因，仍由 broker 创建并顺延唯一卖单。

### 3.2 `policy=None` 不变

默认分支仍只读取原字段 `entry_signal/exit_signal/st_exit/rank_return_120`，仍使用 `config.slots`、原 `size_buy`、原退出原因和原订单顺序。新增的 `policy_state` 只是 `_Position` 的内部可选状态，不进入任何输出列。

默认测试显式比较“省略 policy”与“`policy=None`”的 `trades/orders/round_trips/daily_account/daily_positions/action_log`，逐表完全相同；输入信号不含任何 LegacyPolicy 专用字段。真实固定基线的七类文件哈希也逐项不变，见第 6 节。

## 4. 合成账户时序与反例

| 反例 | 独立预期与实际证据 |
| --- | --- |
| 权威排序、代码 tie-break、每日 2 开 | 两只原分 5 分候选按代码排在两只 4 分候选前；Donchian 兼容别名故意设为相反数量级但不影响选择。实际只在后续首个开盘买入 2 只，其余 `no_slot`；T 日成交数为 0。 |
| 4 槽、20%、整手、不加仓 | 初始权益 100,000、开盘价 10 元时首日各买 2,000 股；扣除首日费用后的次日权益使 20% 目标不足 20,000 元，次日各买 1,900 股而不是 2,000 股。已持有股票再次出现信号没有加仓，第五只股票无成交，最终 4 只。 |
| 费用与现金边界 | 所有成交费用由原 `open_fill/transaction_costs` 产生且大于 0；逐日现金非负，期末现金等于初始现金加全部 `cash_delta`，逐日权益等于现金、应收和市值之和。 |
| 拒单和锚否决递补 | 第一名开盘涨停拒单，第二名固定锚恰好等于实际买入价而触发 `stop_at_or_above_entry`；两者都不耗每日槽，第三、四名按冻结顺序实际成交，完成每日 2 开。买入委托不顺延。 |
| 固定 low10 与 `>=` 边界 | 信号日 `stop_ref=9.5`，后续信号日故意改成 1.0；实际收盘 9.5 仍按原锚触发止损，证明未每日重算。另一反例保留锚等于实际买入价的 `>=` 否决边界。 |
| 公司行动和实际/复权尺度 | 入场时 `F_entry=2`、`stop_ref=19`，结构锚转为实际尺度 `19/2=9.5`；公司行动后 `F_current=1`，锚变为 `9.5×1/2=4.75`。实际收盘 6 元时不止损，收盘 4.75 元时确认，下一开盘卖出；既有送股账本使 400 股变为 800 股。 |
| 40 日持有计数 | 入场日计第 1 日；一个停牌日、一个零成交日和一个缺 bar 日均不计数。第 40 个有效持有日在全局日期索引 43 收盘确认，索引 44 开盘卖出，未提前或延后。 |
| 多退出与 ST | 同一收盘同时满足止损、40 日到期、MA5<MA20、魏氏柱<0、破 MA20 和 ST；只生成一笔卖单和一笔卖出成交。旧原因顺序为 `stop_loss|max_hold|ma5_below_ma20|whist_negative|close_below_ma20`，新增 ST 以独立后缀 `;st_exit` 披露，不插入旧优先级；仅 ST 时仍为 `st_exit`。 |
| 跌停顺延 | 止损确认后的两个开盘均 `sell_limit_down`，订单保持 pending；第三个开盘以当时实际开盘价及统一滑点成交，测试价格为 `7×0.999=6.993`。 |

所有测试中的买入和卖出均满足成交日晚于信号日；买入严格为后续首个面板交易日，卖出延期具有逐日 pending 证据。T+1、分红、送股、退市、停牌估值和缺 bar 账本没有另写实现。

## 5. 固定止损与退出口径

- 入场实际结构锚：`stop_ref / adjustment_factor`；其中既有面板 `adjustment_factor=signal_close/close`。
- 入场固定锚：`max(实际结构锚, fill.price × (1-0.08))`，使用含不利滑点的实际成交价。
- 若固定锚 `>= fill.price`，取消本次买入并有序递补，不扣现金、不建仓、不耗槽。
- 持仓后的当前实际止损锚：`entry_stop_anchor × F_current/F_entry`，再与当前实际未复权收盘比较，收盘 `<=` 触发。结构分与买入价百分比分使用相同因子比例，因此等价于分别调整两个锚后取 max。
- 有个股正常行情且 `volume>0` 才增加 held 并评估旧收盘退出；停牌、零成交和缺 bar 不增加持有计数。ST 作为统一新增风险退出单独检查，停牌或跌停时由既有卖单 pending 机制顺延。

## 6. 默认固定基线 SHA256 回归

在修改前实际运行一次、在全部实现和测试后再次运行同一命令：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B \
  /Users/admin/WorkSpace/trading/trading_finally/scripts/run_baseline.py \
  --config /Users/admin/WorkSpace/trading/trading_finally/config/baseline.json
```

两次均为既有负收益基线：2,440 个账户日、337 条成交、169 个闭合往返。修改前已存文件与修改前实算文件哈希相同；下表“修改前”和“修改后”也是逐项完全相同。

| 输出 | 修改前 SHA256 | 修改后 SHA256 | 结论 |
| --- | --- | --- | --- |
| `daily_account.parquet` | `3f29fcfca7487aa6da2d53a7f575bf9b44fd8e4e50133e5a59c53d29a1ea90ad` | `3f29fcfca7487aa6da2d53a7f575bf9b44fd8e4e50133e5a59c53d29a1ea90ad` | 一致 |
| `daily_positions.parquet` | `ac5292fb9f20b48576d0b0a5bf44f1d72b27943c876af516869f86e942393002` | `ac5292fb9f20b48576d0b0a5bf44f1d72b27943c876af516869f86e942393002` | 一致 |
| `trades.parquet` | `0c7026571bd30ac24371efaf39041d93b2d46ba60e1df9841b709c8a6a389191` | `0c7026571bd30ac24371efaf39041d93b2d46ba60e1df9841b709c8a6a389191` | 一致 |
| `orders.parquet` | `47bcf05275e214ce9ba4fb12f69f0b01d4187a57e6c2808f4526791117ab4fc1` | `47bcf05275e214ce9ba4fb12f69f0b01d4187a57e6c2808f4526791117ab4fc1` | 一致 |
| `round_trips.parquet` | `3a9d3538a06edad251efc2963203c4dd12b600698d8dd9675a47b2f39f12456b` | `3a9d3538a06edad251efc2963203c4dd12b600698d8dd9675a47b2f39f12456b` | 一致 |
| `action_log.parquet` | `196945b236132f049f97b429340a14a14478a92614d048e85dc7a1fa0f6d9517` | `196945b236132f049f97b429340a14a14478a92614d048e85dc7a1fa0f6d9517` | 一致 |
| `performance_summary.json` | `fd9ba0124212c327bf6b171bbd5e48222e8bf5c7415d9a828e92b714fae1994b` | `fd9ba0124212c327bf6b171bbd5e48222e8bf5c7415d9a828e92b714fae1994b` | 一致 |

结论：**默认 Donchian 基线七类指定输出 SHA256 全部不变，允许交付本可选钩子；未进入真实候选运行。**

## 7. 限制与后续边界

1. 本任务只证明合成规则边界和默认回归，不证明真实候选 A 的收益、稳健性或可实盘性；真实候选 A/B 账户收益均未运行。
2. 真实运行仍依赖 Task 2/3 冻结面板的点时因子、ST、行情、量额和公司行动质量；本任务没有修补或重建源数据。
3. `orders` 继续承担执行状态/pending 证据，filled 行的执行诊断字段保持原语义；完整多退出原因最终保存在 `round_trips.exit_reason`，没有为新增原因改变默认输出 schema。
4. 持有计数、入场因子和固定止损锚是内部持仓状态，未额外暴露到 `daily_positions`，避免改变既有账本字段。
5. 候选 B 的 12 项不可还原问题没有变化；本任务不进入 Task 6 的真实候选评价。

## 8. 控制方接管审查与复权方向修复（2026-11-04）

- 原独立审查代理在开始后因 API 周额度 429 中断，未产出报告。控制方接管 scoped 审查。
- 控制方用真实 A 股样本核验因子方向：面板 `adjustment_factor = signal_close / close`。`002695.SZ` 于 2016-09-22 发生大比例送转，实际收盘 74.66→19.25，因子 1.0294→4.1255，信号价 76.86→79.42 连续。因此因子在送转时上升，固定止损实际价应按 `入场锚 × 入场因子 / 当前因子` 缩放。
- 原实现与旧测试使用了相反方向 `入场锚 × 当前因子 / 入场因子`，并构造了因子由 2 降至 1 的非真实路径。真实送转会令止损价被错误上调，除权后立即假止损。
- 修复：先以真实方向的送转反例取得 RED（原实现于除权日错误触发止损），再改 `legacy_policy.py` 为 `stop_anchor × entry_factor / current_factor`。修复后账户策略 9/9、全项目 168/168 PASS。
- 默认 Donchian 不受影响：重新运行固定基线，七类核心输出 SHA256 与本报告第 6 节完全一致。
- 13 预注册已在不改变经济规则的前提下补充因子定义：设计中的 F 等于面板 `adjustment_factor` 的倒数，并保留真实样本证据。
- 仍未运行任何真实候选 A/B 账户收益。该修复必须在 Task 6 真实回测前完成，否则送转持仓会产生系统性假止损。
