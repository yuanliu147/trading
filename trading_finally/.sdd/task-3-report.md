# Task 3 报告：实现无未来函数的唯一基线信号

## 状态

DONE

## 文件变更

- 新建 `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/signals.py`
  - 实现 `compute_signals(panel: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame`。
  - 按 `code/date` 稳定排序并保留输入字段与原 `(date, code)` 键。
  - 新增 `ma200/channel_high_120/channel_low_120/rank_return_120/entry_signal/exit_signal/st_exit`。
- 新建 `/Users/admin/WorkSpace/trading/trading_finally/tests/test_signals.py`
  - 增加 7 项信号测试，覆盖通道移位、完整窗口、点时截断、当日修改不回写历史、ST 不回填、全部入场门槛、复权冲突保留规则、主业绩窗口、退出标记及跨股票隔离。
- 新建本报告 `/Users/admin/WorkSpace/trading/trading_finally/.sdd/task-3-report.md`。

未修改 Task 1-2 源码、真实面板或源数据库。

## 固定实现契约

- `ma200 = signal_close.rolling(200).mean()`，按股票独立计算。
- 均线向上：`ma200 > ma200.shift(20)`，按股票独立移位。
- `channel_high_120 = signal_high.shift(1).rolling(120).max()`。
- `channel_low_120 = signal_low.shift(1).rolling(20).min()`。
- `rank_return_120 = signal_close / signal_close.shift(120) - 1`。
- 入场同时使用以下条件：
  - 收盘价高于 MA200；
  - MA200 斜率向上；
  - 收盘价高于由此前数据计算的 120 日通道高点；
  - 120 日排名收益非空；
  - 非 ST、`trade_status == 1`、`data_valid == True`；
  - 无 `execution_conflict`；
  - 日期位于 `config.start_date` 与 `config.end_date` 之间（含边界）。
- `adjustment_source_conflict` 与审计字段原样保留，不作为独立入场阻断。
- `exit_signal = signal_close < channel_low_120`；`st_exit` 由当日 `is_st` 单独产生。
- 所有 rolling 均使用默认完整窗口要求；窗口不足、关键字段值缺失或布尔比较无法成立时不生成入场。

## TDD 证据

### RED

先写 `tests/test_signals.py`，仅在信号模块保留模块说明，未实现 `compute_signals`，随后执行规定命令：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_signals -v
```

结果按预期失败：

```text
ImportError: cannot import name 'compute_signals' from 'trend_system.signals'
Ran 1 test in 0.000s
FAILED (errors=1)
```

失败原因确为接口尚未实现，而非测试数据或断言错误。

### GREEN：信号测试

实现固定公式与门槛后执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_signals -v
```

结果：`Ran 7 tests ... OK`。

通过的测试：

1. `test_channels_use_only_prior_rows_and_windows_are_complete`
2. `test_current_record_does_not_rewrite_yesterday`
3. `test_entry_gates_and_adjustment_conflict_rule`
4. `test_entry_is_limited_to_main_window_and_exit_is_separate`
5. `test_future_ex_right_rows_do_not_change_prior_signals`
6. `test_st_is_not_backfilled_to_history`
7. `test_stocks_are_sorted_and_calculated_independently`

### GREEN：规定的数据回归

执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_signals tests.test_data -v
```

结果：`Ran 22 tests ... OK`，信号 7 项与数据 15 项全部通过。

### 完整项目回归

额外执行：

```bash
PYTHONPATH=src python3 -B -m unittest discover -v
```

结果：`Ran 27 tests ... OK`，配置、数据、信号全部通过，无错误或警告。

## 真实主面板检查

只读加载 `/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet` 的必要列，使用 `/Users/admin/WorkSpace/trading/trading_finally/config/baseline.json`，未运行账户回测或收益参数比较。

断言及结果：

| 检查 | 结果 |
|---|---:|
| 面板行数 | 10,776,532 |
| 入场信号行数 | 74,492 |
| 最早入场日期 | 2016-10-13 |
| 最晚入场日期 | 2026-10-29 |
| 早于主窗口的入场 | 0 |
| 晚于主窗口的入场 | 0 |
| 重复 `(date, code)` 键 | 0 |
| 入场行 `rank_return_120` 缺失 | 0 |
| 入场行 `execution_conflict` | 0 |
| 入场行非可交易 / ST / 无效数据 | 均为 0（断言通过） |
| 保留 `adjustment_source_conflict=True` 的入场行 | 19,704 |

所有规定断言通过；复权模型差异仅保留供后续敏感性任务使用，未被错误地作为独立阻断。

## 自审

- **点时性**：所有分组计算只使用当前行及历史行；通道先 `shift(1)` 再 rolling，当前高点不会进入当日通道。
- **截断一致性**：除权日前截断测试对截断前后该日前全部 7 个输出字段使用 `check_exact=True`，结果完全一致。
- **无历史回填**：当日价格突变不改写昨日输出；ST 标记只从实际出现日开始生效。
- **窗口完整性**：MA200、20 日斜率滞后、120/20 日通道与 120 日收益均有完整历史后才可能产生入场。
- **冲突绑定**：仅 `execution_conflict` 独立阻断；`adjustment_source_conflict` 不参与入场否定条件。
- **接口范围**：函数只接收收盘后面板和不可变研究配置，不接收账户状态，不读取未来行情或公司行动。
- **改动范围**：未改相邻模块、未做额外策略抽象、未添加参数或收益筛选，未运行账户回测。
- **Git**：未 commit、未 push、未修改 Git 状态。

## 顾虑

无阻塞顾虑。
