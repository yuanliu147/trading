# Task 6 完成报告：账户指标、研究基准与冻结真实基线

## 状态与范围

- 状态：完成，并已完成审查 fix round 1。指标、四类研究基准、复现清单和一次真实基线均已生成。
- 新增实现：`src/trend_system/metrics.py`、`scripts/run_baseline.py`。
- 新增测试：`tests/test_metrics.py`、`tests/test_baseline_integration.py`。
- 未修改 `config/baseline.json`、设计/预注册文档或任何上游模块；未访问源数据库，未执行 Git 操作。
- 未运行 2010—2015 压力测试、邻域变体、随机排序变体或任何参数搜索。真实基线只运行一次。

## 指标口径

- 总收益以主窗口首个账户权益为基准；CAGR 使用实际起止天数和 365.25 天年化。
- 最大回撤报告为权益从历史峰值下跌的正数幅度。
- 年度收益在首个年度之后以前一实际年末账户权益为基准；2016 和 2026 均单独标为 `annual_partial`。
- 月末滚动三年收益使用月末实际日期减三年的目标日，并取目标日当日或之前最近一个账户日期，不按 36 行或固定交易日数计数。asof 向前取数距目标日最多 31 个自然日，超过则跳过该窗口；这是数据质量保护阈值，不是策略参数。
- 持仓暴露披露日均值和最大值。
- `closed_round_trip_turnover` 为全窗口已闭合交易的双边成交额（买入加卖出）除以日均权益，未年化；未闭合交易不进入该交易统计。
- 扣费后期望、平均盈利和平均亏损均为每笔完整往返交易的人民币净盈亏；胜率按 `net_pnl > 0` 计算，连续亏损按退出日期排序。
- `realized_annual_contribution` 按退出年度汇总已闭合交易净盈亏，`realized_stock_contribution` 按证券代码汇总已闭合交易净盈亏；`top5_closed_profitable_trades_net_pnl` 和 `top5_closed_profitable_trades_share_of_gross_profit` 分别披露前五已闭合盈利交易的净盈亏总额及其占全部正净盈亏的比例。
- 上述闭合交易统计不含未闭合交易、未实现盈亏和分红，不伪造全账户归因；已实现年度贡献不与账户年度权益盈亏直接相等。Task 7 年度门槛应使用账户年度权益收益。

## TDD 证据

### RED

首次仅写测试后执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_metrics tests.test_baseline_integration -v
```

结果为 `FAILED (errors=2)`。`tests.test_metrics` 和 `tests.test_baseline_integration` 均因以下接口缺失而无法导入：

```text
ModuleNotFoundError: No module named 'trend_system.metrics'
```

### GREEN

实现后重复同一命令：

```text
Ran 7 tests in 0.097s
OK
```

其中合成基线在同一临时输入上连续运行两次，11 个输出文件的 SHA256 全部一致；账户、交易、年度/滚动指标、基准、摘要和复现清单也逐项一致。合成交易严格发生在信号日之后，买入价为开盘价加 10bp 滑点、卖出价为开盘价减 10bp 滑点，不是 T 日收盘成交。

随后运行完整套件：

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

结果：`Ran 78 tests ... OK`。

CLI 帮助核验仅提供 `--config`，没有任何策略参数覆盖项。

### 审查 fix round 1 的 RED/GREEN

先更新/新增针对性测试，覆盖五个字段重命名、44 天 asof 缺口跳过、恰好 31 天缺口保留、实际配置路径传入，以及无配置路径时不得读取默认 `baseline.json`。实现前运行同一 targeted 命令：

```text
Ran 10 tests in 0.071s
FAILED (failures=1, errors=5)
```

失败证据包括新字段 `KeyError`、44 天缺口对应的 `2020-02` 窗口仍被错误保留、无默认文件时 `FileNotFoundError`，以及 `run_baseline()` 尚不接受 `config_path`。实现后：

```text
Ran 10 tests in 0.112s
OK
```

完整项目测试随后为 `Ran 81 tests ... OK`。任意日历部分年度判断按裁决未修改。

## 唯一一次真实基线运行

执行命令：

```bash
PYTHONPATH=src python3 -B scripts/run_baseline.py --config config/baseline.json
```

- 执行次数：1。
- 退出码：0。
- 异常或执行契约中止：无。
- 主窗口：2016-10-13 至 2026-10-29。
- 未因结果为负而修改参数、删除持仓或绕过执行契约。

### 输入行数和窗口

| 输入 | 行数 | 代码数 | 数据窗口 |
| --- | ---: | ---: | --- |
| `main_panel.parquet` 全量 | 10,776,532 | 3,425 | 2010-01-04 至 2026-10-29 |
| 主窗口回测面板 | 7,335,154 | 3,400 | 2016-10-13 至 2026-10-29 |
| `index_panel.parquet` | 44,189 | 9 | 2006-01-04 至 2026-10-29 |
| `corporate_actions.parquet` | 43,577 | 3,178 | 1991-02-26 至 2026-10-29 |

依赖版本记录为 Python 3.13.13、pandas 3.0.3、NumPy 2.5.0、pyarrow 26.0.0、DuckDB 1.5.6。DuckDB 仅用于版本披露，没有连接源数据库。

### 输出表行数

| 输出 | 行数 |
| --- | ---: |
| `daily_account.parquet` | 2,440 |
| `daily_positions.parquet` | 6,975 |
| `orders.parquet` | 54,562 |
| `trades.parquet` | 345 |
| `round_trips.parquet` | 171 |
| `action_log.parquet` | 1,095 |
| `annual_and_rolling.csv` | 96（11 个年度行、85 个滚动三年行） |
| `benchmarks.csv` | 9,760 |

完整输出位于 `results/baseline/`，另含 `performance_summary.json`、`run_manifest.json` 和 `summary.md`。大表未在终端回显。

## 冻结基线主要结果

以下仅为本次固定规则、固定数据窗口下的研究记录，不构成收益宣传或投资建议。

| 指标 | 结果 |
| --- | ---: |
| 初始权益 | 20,000.00 元 |
| 最终权益 | 2,301.81 元 |
| 总收益 | -88.4910% |
| CAGR | -19.3692% |
| 最大回撤 | 88.7073% |
| 完整往返交易 | 171 笔 |
| 扣费后每笔期望 | -104.444795 元 |
| 胜率 | 29.2398% |
| 平均盈利 | 232.149000 元 |
| 平均亏损 | -243.533140 元 |
| 最长连续亏损 | 13 笔 |
| 平均持仓暴露 | 48.7901% |
| 最大持仓暴露 | 64.3657% |
| 全窗口已闭合往返双边换手/平均权益 | 53.887864 |
| 前五已闭合盈利交易净贡献 | 5,626.60 元 |
| 前五已闭合盈利交易占全部正净盈亏 | 48.4740% |
| 月末滚动三年窗口 | 85 个 |
| 滚动三年正收益窗口占比 | 0% |

2016 部分年度收益为 -3.2242%，2026 部分年度收益为 -15.9908%，均未与完整年度混同标记。逐年度账户权益收益、已实现年度贡献、已实现单票贡献及 85 个滚动窗口的完整数值保存在结果文件中；已实现贡献不与账户年度权益盈亏直接相等。

最大暴露高于 60% 买入配置上限，原样披露；上游规则只在买入 sizing 时限制暴露，买入后市值变化没有强制再平衡规则，本次未因此改变规则。

## 研究基准

| 基准 | 区间收益 | 最大回撤 | 口径 |
| --- | ---: | ---: | --- |
| 现金 | 0% | 0% | 收益设为 0 |
| 上证综指 | 19.2755% | 30.7659% | 价格指数，不含股息 |
| 沪深300 | 22.3818% | 45.6026% | 价格指数，不含股息 |
| 主板等权 | 46.6144% | 43.4000% | 同宇宙后复权收益日度等权再平衡研究基准，不可由 2 万元账户直接复制 |

基准只用于研究对照；指数股息口径和主板等权的可投资性限制已同时写入 `summary.md` 与 `run_manifest.json`。

## fix round 1 对既有真实结果的处理

**没有重跑真实基线或账户回测。** 字段重命名和配置来源修正不要求重新计算交易；31 天 asof 数据质量阈值经既有账户序列验证不改变任何真实滚动窗口。

实际只执行了确定性汇总代码：读取已保存的 `daily_account.parquet`、`round_trips.parquet`、`annual_and_rolling.csv` 和 `benchmarks.csv`，调用 `performance_summary`、`annual_and_rolling`、`_markdown_summary` 与 `_manifest`，并使用实际配置文件 `/Users/admin/WorkSpace/trading/trading_finally/config/baseline.json` 作为 `config_path`。没有调用 `run_baseline` 或 `run_backtest`。

仅重写以下三个文件：

- `performance_summary.json`：替换为明确的已闭合/已实现字段名；映射前后数值逐项断言相等。
- `summary.md`：同步新名称和统计范围说明。
- `run_manifest.json`：记录实际配置文件的 `source=file` 和路径，并更新代码 SHA256。

重生成前后，年度/滚动表逐行相等，85 个滚动窗口保持不变，真实最大 asof 缺口为 8 天；账户、持仓、订单、成交、往返交易、行动日志、年度/滚动 CSV 和基准 CSV 的 SHA256 前后完全一致。

## 复现与运行后校验

`results/baseline/run_manifest.json` 包含：

- CLI 实际 `--config` 文件路径、`source=file`、配置值和 SHA256；本次配置 SHA256 为 `eb0cca4439a4da3a3a6f835eb18f09d055a7b56a16cb8b1e8d875dd239c3e8ca`。直接调用 `run_baseline(config)` 且未提供路径时，来源明确标为 `source=canonical_config_values`、路径为 null，并对实际 `config.values` 的规范 JSON（排序键、紧凑分隔符、UTF-8）计算 SHA256，不再无条件读取或声称使用默认文件。
- 三份设计/实施/预注册文档 SHA256。
- 全部 `src/trend_system/*.py` 与 `scripts/run_baseline.py` 的 SHA256。
- 输入面板行数、代码数、首尾日期和依赖版本。
- 四类基准的口径说明。

真实运行完成后另行只读校验通过：

- 345 笔成交均满足 `trade.date > trade.signal_date`。
- 2,440 个账户日均满足 `equity = cash + receivables + market_value`（容差 1e-8）。
- 配置、三份设计文档和全部清单内代码文件的当前 SHA256 与清单一致。
