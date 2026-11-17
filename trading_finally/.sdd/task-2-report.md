# Task 2 实施报告：双源审计和点时行情面板

## 状态

DONE_WITH_CONCERNS

实现、单元测试、真实构建和独立落盘验收均已完成。顾虑来自真实数据质量：重叠面板有 1,080,277 行源冲突，不能在后续策略研究中忽略。

## 文件变更

本任务新建源码、SQL、脚本和测试：

- `/Users/admin/WorkSpace/trading/trading_finally/sql/baostock_panel.sql`
- `/Users/admin/WorkSpace/trading/trading_finally/sql/hithink_overlap.sql`
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/data.py`
- `/Users/admin/WorkSpace/trading/trading_finally/scripts/build_panel.py`
- `/Users/admin/WorkSpace/trading/trading_finally/tests/test_data.py`

真实数据输出：

- `/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet`
- `/Users/admin/WorkSpace/trading/trading_finally/data/stress_panel.parquet`
- `/Users/admin/WorkSpace/trading/trading_finally/data/corporate_actions.parquet`
- `/Users/admin/WorkSpace/trading/trading_finally/data/index_panel.parquet`（“指数另存”输出）
- `/Users/admin/WorkSpace/trading/trading_finally/results/data_audit.json`
- `/Users/admin/WorkSpace/trading/trading_finally/results/data_conflicts.parquet`
- `/Users/admin/WorkSpace/trading/trading_finally/results/data_exclusions.csv`

过程报告：

- `/Users/admin/WorkSpace/trading/trading_finally/.sdd/task-2-report.md`

未修改 Task 1 的配置、设计文档或预注册。

## 实现口径

- `build_panel(config) -> PanelBuildResult` 使用唯一 `ResearchConfig`；`load_panel(path) -> pd.DataFrame` 校验必需字段和 `(date, code)` 唯一键。
- baostock SQLite 通过 URI `mode=ro` 打开；同花顺 DuckDB 通过 `read_only=True` 打开。未执行同步、迁移、修复、建表或任何写 SQL。
- baostock `adjustflag=3` 提供原价，`adjustflag=1` 按同日同代码左连接提供信号后复权价；不做未来行回填。
- 主面板从 2010-01-01 起保留热身历史，主业绩窗口仍为 2016-10-13 至 2026-10-29；压力面板固定为 2010—2015 单源切片。
- 主板范围为沪深普通 A 股的 `600/601/603/605/000/001/002/003` 前缀，其他证券按代码记录排除，不因数据缺失或冲突删除整只股票。
- 原价 OHLC 绝对差阈值严格为 `> 0.011` 元；成交量、成交额相对差阈值分别为 `> 1e-5`、`> 1e-6`。相对差分母为两源绝对值的较大者，双方均为零时差异为零。
- 复权因子使用同日 `hfq_close/raw_close`。每只股票取重叠期首个共同有效因子作归一化基准，复权 OHLC 缩放到原价可比口径后使用 0.011 元阈值；没有直接拼接或比较两套复权绝对价格。
- 面板保留 baostock 原价和复权值，冲突行设置 `source_conflict=True`；冲突文件同时保存两源数值、差异、因子和字段，未静默择值。
- 无效原价或任一后复权 OHLC 缺失时 `data_valid=False`，原行保留。baostock 独有行保留并计数；同花顺独有行会进入冲突和排除记录，本次实际为 0 行。
- 公司行动仅保留 `ex_date <= 2026-10-29` 的记录，包含现金分红、送转、配股、配股价、批次和来源日期。源表只提供除权日，`source_date` 如实保留该日期，没有虚构公告日或派息到账日。
- SQL 文本、SQL SHA256、数据库 SHA256、行数、窗口、冲突、排除和依赖版本均写入 `data_audit.json`。

## TDD 记录

### Step 1：先写合成测试

`tests/test_data.py` 共 10 项，覆盖：

1. `sh.600000 -> 600000.SH`、`sz.000001 -> 000001.SZ` 及已规范代码；
2. 北交所、长度错误、缺少交易所等无法转换代码抛错；
3. 原价存在而后复权缺失时保留原行、置无效且不以未来复权行回填；
4. 同键重复抛错；
5. 原价或后复权负价格抛错；
6. 原价 0.011 元阈值严格“超阈值才算冲突”；
7. 成交量和成交额相对差阈值；
8. 复权序列按股票归一化后比较，绝对尺度不同本身不构成冲突；
9. 公司行动除权当日生效、未来事件不得进入；
10. `load_panel` Parquet 往返。

### Step 2：RED

从 `trading_finally/` 执行简报命令：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_data -v
```

退出码 1，关键输出：

```text
ImportError: Failed to import test module: test_data
ModuleNotFoundError: No module named 'trend_system.data'
FAILED (errors=1)
```

失败原因确为数据模块不存在，与简报预期一致。

### Step 3/4：实现与 GREEN

首次实现后的阈值用例暴露出测试夹具同时改变原价和复权因子的问题；将夹具固定为相同复权因子后，原价阈值与复权归一化测试各自独立验证。最终指定命令结果：

```text
Ran 10 tests in 0.109s
OK
```

最终全量测试命令：

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

结果：

```text
Ran 15 tests in 0.104s
OK
```

无测试错误或警告。

## 真实构建记录

### 首次构建发现性能问题

首次真实构建长期停留在 SQLite 只读统计。原生栈采样和 `EXPLAIN QUERY PLAN` 确认：额外编制的“非主板证券历史行数”统计对每只证券重复扫描全市场日期索引，形成不必要的嵌套扫描。该额外行数不是简报要求，因此终止这次尚未落盘的构建，移除连接，只保留代码级排除记录。

修正后排除查询仅扫描 `stock_basic`，全部 15 项测试保持通过。没有通过修改源库、加索引或建表解决性能问题。

### 最终构建命令与输出

```bash
PYTHONPATH=src python3 -B scripts/build_panel.py --config config/baseline.json
```

退出码 0。 stdout 仅包含小摘要，大结果全部落盘：

```text
main_rows: 10776532
stress_rows: 3002066
conflict_rows: 1080277
exclusion_rows: 1090327
```

## 独立真实验收

另用 DuckDB/Parquet 读取落盘结果，并重新计算两个数据库的 SHA256。验收断言全部通过：

- 主面板最大日期严格为 `2026-10-29`；必需字段完整；`(date, code)` 重复键为 0。
- 主面板、压力面板、公司行动、指数、冲突和排除文件行数与审计 JSON 一致。
- 重叠行数恒等式成立：`matched + baostock_only = baostock_rows`，`matched + hithink_only = hithink_rows`。
- 面板内冲突行数、冲突文件行数、审计冲突行数一致。
- 公司行动无未来日期、无重复键；压力面板无双源冲突标记。
- 复权不完整股票计数为预期的 58 只。
- 两个源库重新计算的 SHA256 均与构建前审计值一致，证明源文件内容未变化。

### 面板行数与窗口

| 输出 | 行数 | 代码数 | 窗口 |
|---|---:|---:|---|
| 主面板 | 10,776,532 | 3,425 | 2010-01-04 至 2026-10-29 |
| 其中主业绩窗口 | 7,335,154 | — | 2016-10-13 至 2026-10-29 |
| 2010—2015 压力面板 | 3,002,066 | 2,342 | 2010-01-04 至 2015-12-31 |
| 公司行动 | 43,577 | 3,178 | 1991-02-26 至 2026-10-29 |
| 指数面板 | 44,189 | 9 | 2006-01-04 至 2026-10-29 |

### 源数据与重叠计数

- baostock：后复权/前复权各 18,604,908 行，原价 18,620,738 行；均为 1990-12-19 至 2026-10-29、5,564 个代码。
- 同花顺 `v_daily` 与 `v_daily_hfq`：各 10,403,697 行，2016-10-13 至 2026-10-30、5,580 个代码。
- 主板重叠窗口 baostock 7,335,154 行，同花顺 6,939,274 行。
- 同键匹配 6,939,274 行；baostock 独有 395,880 行；同花顺独有 0 行。

### 复权缺口与宇宙计数

- 全源复权行数不完整：58 只；其中主板前缀 8 只。
- 主面板实际后复权字段缺失：4,482 行、10 只；全部保留并设置 `data_valid=False`，没有以后复权未来行回填。
- 主面板无效行合计 4,482 行、10 只。
- 主板股票宇宙 3,489 只，其中退市 295 只、当前上市 3,194 只；主面板实际有行情代码 3,425 只。
- 非主板股票排除 2,075 个代码；非股票证券排除 3,461 个代码。

### 全部冲突计数

- 冲突文件及面板标记：1,080,277 行，涉及 1,628 只。
- 原价价格冲突：1 行（`close` 字段计数 1）。
- 成交额超阈值：527 行；成交量超阈值：0 行。
- 复权序列冲突：1,079,751 行。
- 分字段计数：`signal_open=1,059,876`、`signal_high=1,079,750`、`signal_low=1,041,948`、`signal_close=1,060,602`。字段与类别可在同一行重叠，不能相加作为总行数。
- 抽样核查显示典型复权冲突来自两源实际复权精度/序列差异，归一化后仍略超 0.011 元；并非直接比较两套复权绝对尺度造成。

### 排除记录计数

`data_exclusions.csv` 共 1,090,327 条记录：

| 原因 | 记录数 | 处理 |
|---|---:|---|
| `execution_conflict` | 1 | 面板保留；基线入场和持仓不合格 |
| `adjustment_source_conflict` | 1,079,750 | 面板保留；只进入独立复权敏感性 |
| `source_conflict` | 526 | 面板保留；仅作量额审计，不独立阻断 |
| `data_valid_false` | 4,482 | 面板保留，标记无效 |
| `non_stock_security` | 3,461 | 代码级宇宙排除 |
| `non_main_board` | 2,075 | 代码级宇宙排除 |
| `future_corporate_action` | 32 | 不进入点时公司行动输出 |

本次没有同花顺独有键，因此相应排除记录为 0。

### 数据库 SHA256 复核

- baostock：`77a12ced7de4f21d1a9c26b60fbaca518437e2fc808ef994ca634b558b2c8a6e`，构建后复算一致。
- 同花顺：`563cdd4b59f00ac8928ed61c8f2476aca0164b3139ae2faa92792f9ca81d7b3f`，构建后复算一致。

## 自审

- 简报要求的接口、字段、六个指定输出及指数另存均已实现；主键唯一和最大日期经独立读取验证。
- 复权缺口、冲突和无效行均保留，没有通过删除整只股票美化覆盖率。
- 公司行动只按已生效除权日输出；没有把未来公司行动接入历史面板。
- 所有真实大结果均写入 Parquet/CSV/JSON；stdout 仅有计数摘要和少量验收摘要。
- 仅使用现有 Python 3.13、pandas、DuckDB、pyarrow，未安装依赖，未派发子代理或审查代理。
- 两个源库只读访问且哈希前后一致；未执行任何源库写操作。
- 未执行 `git add`、`git commit` 或其他 Git 状态修改操作；未创建提交。

## Concerns

1. 复权序列冲突规模具有实质性：1,079,751 行进入 `adjustment_source_conflict` 独立敏感性，不能忽略本审计后继续形成有利结论；基线入场和持仓只由 `execution_conflict` 阻断，不再以 `source_conflict` 总标志单独阻断。
2. 58 只全源复权不完整股票以及面板实际 10 只缺复权股票已诚实保留和计数，但会减少可用信号宇宙；后续报告不得把排除后的宇宙描述为完整无缺口数据。
3. 公司行动源表没有独立派息到账日或公告日，本任务未推断这些字段；真实现金到账近似必须继续受预注册和 Task 5 测试约束。

---

## 修正追加：执行冲突与复权模型冲突拆分

### 修正范围与预注册

根据更新后的 Task 2 简报，再次修改：

- `src/trend_system/data.py`：新增 `execution_conflict`、`adjustment_source_conflict`，保留 `source_conflict` 作为审计并集；增加完整同花顺复权面板和归一化相对差诊断。
- `scripts/build_panel.py`：输出新增面板路径。
- `tests/test_data.py`：新增冲突语义、完整同花顺面板、尺度/精度/实质差异及冲突子集分布测试。
- `data/hithink_signal_panel.parquet`：新增全部共同重叠行的同花顺后复权 OHLC，不只保存冲突行。
- `results/data_audit.json`、`results/data_conflicts.parquet`、主/压力面板：写入拆分字段和新增诊断。
- `02_预注册_极简趋势消融.md`：追加“任何策略运行前的数据审计修订”，规定 Task 7 在共同可计算窗口做独立同花顺复权信号敏感性；不得按收益选择来源、替换基线或改写验收方向。

字段语义现在是：

- `execution_conflict`：仅原价 OHLC 超阈值差异。
- `adjustment_source_conflict`：归一化复权序列差异，是模型风险，不表示原价无法成交。
- `source_conflict`：原价 OHLC、量额和复权审计差异的并集。量额差异本身不设置 `execution_conflict`。

策略参数、成本、试验预算和收益否决方向保持冻结；同花顺双源机械门槛以文末 fix round 1 修订为准。本次没有运行任何策略回测或收益比较。

### 修正 TDD RED/GREEN

第一轮新增测试先运行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_data -v
```

RED 证据：

```text
ImportError: cannot import name '_hithink_signal_panel' from 'trend_system.data'
FAILED (errors=1)
```

实现过程中，测试还先发现新增标志列初始化遗漏（7 个错误），修正后执行冲突夹具因同时引入原价和复权收盘差异而失败；将夹具改为只改变原价 `open`，使执行与复权语义各自独立。为核对冲突子集中位数，再增加子集分布断言，RED 为：

```text
KeyError: 'adjustment_conflict_subset'
FAILED (errors=1)
```

最终实现后的指定测试：

```text
Ran 14 tests in 0.144s
OK
```

全量测试：

```text
Ran 19 tests in 0.139s
OK
```

### 修正后的真实构建与验收

重新执行同一命令，最终退出码 0：

```bash
PYTHONPATH=src python3 -B scripts/build_panel.py --config config/baseline.json
```

主面板行数和窗口保持不变，新增验收结果：

| 检查 | 结果 |
|---|---:|
| `source_conflict` | 1,080,277 行 |
| `execution_conflict` | 1 行 |
| `adjustment_source_conflict` | 1,079,751 行 |
| 量额冲突行 | 527 行 |
| 同花顺复权面板 | 6,939,274 行、3,197 只 |
| 同花顺面板窗口 | 2016-10-13 至 2026-10-29 |
| 同花顺面板重复键 / 无法匹配主面板 | 0 / 0 |
| 执行或复权标志未包含于总源冲突的违例 | 0 |

独立读取还确认：同花顺面板保存的是全部 6,939,274 个共同重叠键，而非 1,080,277 个冲突键；主面板最大日期仍为 2026-10-29，主键重复为 0，压力面板三个冲突标志均为 0。两个源库 SHA256 再次复算一致。

### 归一化复权相对差分布

相对差定义为：两源归一化复权收盘绝对差除以两者绝对值的较大者。独立 DuckDB 重算与构建审计一致。

全部具有有效归一化收盘的 6,939,262 个重叠行：

- 中位数：`0.00006406668896080168`；p90=`0.0012423228456005923`，p95=`0.002191018058220545`，p99=`0.006982051607460676`。
- 完全相等：657,216 行。
- `(0, 0.001]` 精度带：5,410,647 行。
- `(0.001, 0.01]`：831,357 行。
- `> 0.01` 实质差异：40,042 行。

`adjustment_source_conflict` 中具有有效收盘相对差的 1,079,739 行：

- 中位数：`0.0012168279905122163`，独立核对了参考诊断的约 0.001217。
- `(0, 0.001]`：471,294 行；`(0.001, 0.01]`：568,403 行；`> 0.01`：40,042 行。
- 另有 12 个复权冲突行缺少有效归一化收盘相对差，仍保留冲突标志，未从审计中删除。

绝对尺度与序列差异也已拆分：同日 `hfq_close/raw_close` 因子相对差超过 `1e-6` 的有 6,290,685 行，其中 5,419,286 行归一化收盘相对差不超过 0.001。这说明大量绝对尺度差异不等于实质序列差异；同时 40,042 行超过 1% 也不能被称为纯精度噪声。

### 修正后的剩余顾虑

原先“百万级复权差异被全部等同执行无效”的顾虑已解除：真实原价 OHLC 执行冲突只有 1 行，复权模型冲突单独标记，并保留完整同花顺复权面板供独立敏感性。剩余顾虑是 40,042 行超过 1% 的实质复权序列差异及既有复权缺口；后续必须按新增预注册披露敏感性，不得按收益选择来源。

本次修正仍未执行 `git add`、`git commit` 或其他 Git 状态修改操作。

---

## Fix round 1：统一执行阻断和双源机械门槛

### 两项 Important 的修正

1. 预注册第 3.1、5、7 节现已统一：基线入场和已有持仓仅由 `execution_conflict` 阻断；`source_conflict` 只作审计总标志；`adjustment_source_conflict` 不单独阻断，进入独立复权敏感性。`data.py` 的排除记录不再把全部 `source_conflict` 写成不合格。
2. 第 5 节正式新增同花顺后复权共同可计算窗口机械门槛：扣除成本后净收益为正且最大回撤不超过 10%；方向相反或不满足时不得 `UNREJECTED`，不得以 baostock 基线结果抵消。第 7 节明确这是任何策略运行前完成的数据审计修订。除该新增双源门槛外，所有策略参数、成本、试验预算、其他机械门槛和收益否决方向保持不变。

报告中此前“全部源冲突不合格”和“机械门槛未改”的过时表述已删除或改写，最终口径以本节及预注册为准。

### TDD RED/GREEN

新增 `_conflict_exclusions` 合成测试，覆盖：执行冲突不合格、复权冲突只进入敏感性、量额总源冲突仅审计，以及执行与复权重叠时只生成执行阻断记录。

RED：

```text
ImportError: cannot import name '_conflict_exclusions' from 'trend_system.data'
FAILED (errors=1)
```

实现后测试又发现审计文案含 `not independently ineligible`，容易与“不单独判定不合格”的机器断言混淆；改为无歧义的 `audit only; no independent blocking decision` 后通过。最终命令与输出：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_data -v
# Ran 15 tests in 0.147s
# OK

PYTHONPATH=src python3 -B -m unittest discover -s tests -v
# Ran 20 tests in 0.143s
# OK
```

### 必要真实重建与联合验收

重新执行：

```bash
PYTHONPATH=src python3 -B scripts/build_panel.py --config config/baseline.json
```

退出码 0。最终排除分类为：

| 原因 | 记录数 | 最终语义 |
|---|---:|---|
| `execution_conflict` | 1 | 基线入场和持仓不合格 |
| `adjustment_source_conflict` | 1,079,750 | 独立敏感性，不单独阻断 |
| `source_conflict` | 526 | 仅量额审计，不独立阻断 |
| 其他既有排除 | 10,050 | 数据无效、宇宙和未来公司行动规则不变 |

排除总记录仍为 1,090,327 条。真实面板中执行冲突 1 行、复权模型冲突 1,079,751 行、审计并集 1,080,277 行；其中 1 行同时具有执行和复权标志，排除时按执行阻断优先级只记录一次。

联合验收通过：主面板仍为 10,776,532 行、最大日期 2026-10-29、重复主键 0；同花顺复权面板仍为全部 6,939,274 个重叠键；`source_conflict` 和 `adjustment_source_conflict` 排除文案中含 `ineligible` 的记录为 0；执行阻断记录数与面板执行冲突数一致；审计 JSON 与 CSV 分类计数一致；两个源库 SHA256 再次复算一致。

本轮未运行任何策略回测或收益比较，未按收益选择来源，未执行 `git add`、`git commit` 或其他 Git 状态修改操作。
