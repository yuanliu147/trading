# Task 2：全 A 点时上下文与历史行业可行性面板报告（含 fix round 1）

- 初次执行日期：2026-11-03
- 修正执行日期：2026-11-04
- 任务范围：只读数据构建、点时/覆盖率测试和数据可行性审计
- 主业绩窗口：2016-10-13 至 2026-10-29
- 全 A 面板窗口：2010-01-04 至 2026-10-29（保留主窗口前热身历史）
- 未实现任何候选信号，未生成候选交易，未运行任何真实候选收益或机械收益裁决

## 1. 当前结论（fix round 1 后）

| 候选 | Task 1 状态 | 初次构建结果 | 修正后 Task 2 状态 | 后续约束 |
| --- | --- | --- | --- | --- |
| A `three_of_two_low_vol` | `PENDING_DATA_GATE` | `NOT_EVALUABLE`：18 日覆盖失败、旧全活跃入场分母不足 100% | `PENDING_DATA_GATE`：修正后全 A 逐日闸门通过 | Task 3 必须对实际触发且拟入场候选完成 100% 严格有效行检查，之后才可运行账户收益 |
| B `early_trend_breadth` | `NOT_EVALUABLE`，12 项公式/溯源问题 | `NOT_EVALUABLE` | `NOT_EVALUABLE` | 无历史行业有效日期来源，12 项规则冻结未解除 |

A 没有被判定为 `EVALUABLE`。B 的规则、12 项 `not_evaluable_fields` 和冻结状态均未修改，也未因本任务的数据构建或补齐而恢复。初次构建失败证据和修正后结果并列保留，详见第 8 节。

## 2. TDD：RED / GREEN 证据

### 2.1 先写失败测试

新建 `tests/test_legacy_data.py`，先行覆盖：

1. 未来行业记录在 `valid_from` 之前不可见。
2. 没有 `valid_from` 的当前行业快照不能回填历史。
3. 逐日分母固定为点时活跃宇宙；非活跃证券即使字段完整也不能进入分子。
4. 98.99% 必须失败，恰好 99% 可以通过。
5. 必需入场字段整列缺失时覆盖率为 0，部分缺失也不能视为完整。
6. A 的实际入场字段缺失时必须 `NOT_EVALUABLE`。
7. A 只有自身全部数据闸门通过时才为 `EVALUABLE`。
8. B 缺少有效日期行业源时必须 `NOT_EVALUABLE`；即使所有数据闸门假设通过，12 项规则冻结仍禁止恢复。

首次运行命令：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_data.py
```

实际 RED：

```text
ModuleNotFoundError: No module named 'trend_system.legacy_data'
FAILED (errors=1)
```

失败原因是要求的生产接口不存在，不是测试数据或断言错误。

### 2.2 最小实现与 GREEN

实现后首次 GREEN 检查又发现一个真实分子错误：非活跃的 `300001.SZ` 被计入 2024-01-03 分子，测试得到 `[1, 3]` 而非手工预期的 `[1, 2]`。随后将完整记录与固定活跃键内连接后再计算分子，没有修改测试期望。

最终结果：

```text
Ran 10 tests in 0.009s
OK
```

数据/冻结配置联合回归：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_data.py \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_config.py \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_data.py
```

实际结果：

```text
Ran 43 tests in 0.154s
OK
```

未运行包含账户或收益执行的项目完整套件，因为本任务明确禁止运行任何账户收益；这不是对未运行测试的通过声明。

## 3. 输入、SQL 与严格只读约束

### 3.1 权威输入

- 配置：`/Users/admin/WorkSpace/trading/trading_finally/config/legacy_candidates.json`
- 预注册：`/Users/admin/WorkSpace/trading/trading_finally/13_预注册_旧候选统一复核.md`
- 主板冻结面板：`/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet`
  - 构建时核验 SHA256：`6a51ed0e7958aa1549dcf156e4387badcfced0236fd812d2bc1fd894aa3acc1a`
- 指数冻结面板：`/Users/admin/WorkSpace/trading/trading_finally/data/index_panel.parquet`
  - 构建时核验 SHA256：`8b8763e0a552894eee84407f1b0c5e6efa4ff7eb8640ddb149b637539bef387a`
- 新建只读查询：`/Users/admin/WorkSpace/trading/trading_finally/sql/legacy_context.sql`
  - SHA256：`212e4163e2601dae09dc8dadc8d1201cc13b1b12c1bf5b037c8270dfa55566f8`

SQL 按年份参数化读取。原始 OHLC、量额和状态来自 Baostock `stock_kline_d` 的 `adjustflag=3`；点时信号 OHLC 来自同一源表 `adjustflag=1` 的后复权序列。没有使用未来行前复权回填，也没有用当前 ST 或当前上市状态过滤历史。

### 3.2 数据库指纹：构建前、构建后与预注册完全一致

| 数据库 | 字节数 | 构建前 SHA256 | 构建后 SHA256 | 结论 |
| --- | ---: | --- | --- | --- |
| `/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb` | 1,863,593,984 | `67561be6368efa5653107c884a6515dc249a7992a99d9f273f6a5cfd77e42f61` | `67561be6368efa5653107c884a6515dc249a7992a99d9f273f6a5cfd77e42f61` | 一致 |
| `/Users/admin/WorkSpace/baostock_data/data/baostock.db` | 10,330,697,728 | `77a12ced7de4f21d1a9c26b60fbaca518437e2fc808ef994ca634b558b2c8a6e` | `77a12ced7de4f21d1a9c26b60fbaca518437e2fc808ef994ca634b558b2c8a6e` | 一致 |

- SQLite 使用 `file:...?mode=ro` URI。
- DuckDB 使用 `read_only=True`。
- 只执行 SELECT、schema 只读检查和文件字节读取；未同步、迁移、修复、下载、建表或写源库。
- 构建后又在进程外用 `stat` 和 `shasum -a 256` 独立复核，结果仍一致。

## 4. 构建口径

### 4.1 全 A 点时活跃宇宙

- 范围为 `stock_basic.type=1` 且代码属于 SH/SZ，排除 BJ；不按当前 `status` 删除历史退市股票。
- 逐日固定分母为 `ipo_date <= date < out_date` 的证券；空 `out_date` 保持活跃。
- 日历使用 `trade_dates.is_trading_day=1`。
- 输出保留原始 OHLC、点时复权 OHLC、量额、交易状态、ST、上市/退市日期、主板标识、字段完整标识和既有冲突标识。
- 全 A 仅用于横截面股性、市场宽度和行业上下文；`is_main_board` 仅允许 `600/601/603/605/000/001/002/003` 进入执行宇宙。
- 源 `stock_basic` 有 5,564 只 SH/SZ A 股，其中 64 只在 2010-01-04 或之前已退市，因此面板实际出现 5,500 只；没有缺失 IPO 日期或因未来 IPO 而排除的代码。

### 4.2 执行冲突与复权差异

主板冲突列直接复用冻结 `main_panel.parquet`，没有另建冲突定义：

- `execution_conflict` 只来自实际未复权 OHLC 缺失或绝对差超过 0.011 元（含既有容差）。
- `adjustment_source_conflict` 和归一化复权差异仅披露，不冒充执行冲突。
- `source_conflict` 是审计并集；量额差异不单独构成执行冲突。
- 非主板没有冻结双源执行冲突审计，相应冲突字段保留为 NULL，不伪造为无冲突。

全 A 输出中继承的标记行数：执行冲突 1 行、复权来源冲突 1,079,751 行、审计并集冲突 1,080,277 行。这些计数不被当作覆盖率通过的替代品。

### 4.3 历史行业

只读检查同花顺 91 个字段和 Baostock 61 个字段，没有任何 `industry` 或 `sector` 字段。实际源库不仅没有带有效起止日的历史行业表，也没有可登记的当前行业快照字段。

因此行业面板为空表，保留 `code/industry/valid_from/valid_to/source/source_sha256/current_snapshot_only` 架构；审计 JSON 保留检查范围与无匹配字段证据。没有生成伪历史、没有当前行业回填，也没有从后续策略版本借行业来源。

## 5. 真实构建产物与独立结构验证

真实构建命令：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B \
  /Users/admin/WorkSpace/trading/trading_finally/scripts/build_legacy_context.py \
  --config /Users/admin/WorkSpace/trading/trading_finally/config/legacy_candidates.json
```

| 产物 | 行数 | 窗口 | 主键 | 字节数 | SHA256 |
| --- | ---: | --- | --- | ---: | --- |
| `/Users/admin/WorkSpace/trading/trading_finally/data/legacy_full_a_panel.parquet` | 14,595,370 | 2010-01-04 至 2026-10-29 | `(date, code)` | 841,170,761 | `8183acce5607b04946d80c05477fc26fd60791748a590200e5096d2d0e525a85` |
| `/Users/admin/WorkSpace/trading/trading_finally/data/legacy_industry_panel.parquet` | 0 | 无可用历史窗口 | `(code, valid_from)` | 3,629 | `6f49b7207340b863380c7c6c389224024ab639def8c64e3ce3b3ee9530a2f53d` |
| `/Users/admin/WorkSpace/trading/trading_finally/results/legacy_context_audit.json` | 不适用 | 覆盖主窗口 2,440 日 | 不适用 | 718,079 | `c30375094d8584131e529a50833d69ab5fabe479042e68bc05d2bc5f1a4ce899` |

独立使用 DuckDB 读取产物 Parquet 验证：

- 行数 14,595,370，`COUNT(DISTINCT date || '|' || code)` 也是 14,595,370，主键无重复。
- 证券数 5,500，BJ 行数为 0。
- 首末日期与构建窗口一致。
- `ipo_date > date` 或 `out_date <= date` 的活跃规则违规行数为 0。
- 独立逐日聚合得到的 2,440 日分母、分子与 audit JSON 逐项一致。
- 大表未在终端回显。

## 6. 初次构建数据闸门结果（pre-amendment 历史证据）

### 6.1 全 A 点时逐日覆盖：失败

- 分母：每日点时活跃全 A 证券数，不用全期 pooled 比率代替。
- 分子：原始/点时复权 OHLC、前收、量额、交易状态、ST、IPO 日期均完整且数值有效的证券数。
- 主窗口交易日：2,440 日。
- 阈值：每日均须 `>=99%`。
- 最低值：2022-05-05 为 98.133616%，分母 4,715，分子 4,627。
- 低于 99% 的天数：18 日。
- 闸门：`False`。

全部低覆盖日期如下；完整 2,440 日分母、分子和覆盖率在 audit JSON 的 `coverage.full_a.daily` 中。

| 日期 | 覆盖率 | 分母 | 分子 |
| --- | ---: | ---: | ---: |
| 2022-05-05 | 98.133616% | 4,715 | 4,627 |
| 2022-05-06 | 98.833757% | 4,716 | 4,661 |
| 2022-05-09 | 98.812553% | 4,716 | 4,660 |
| 2022-05-10 | 98.876166% | 4,716 | 4,663 |
| 2022-05-11 | 98.876166% | 4,716 | 4,663 |
| 2022-05-12 | 98.855447% | 4,718 | 4,664 |
| 2022-05-13 | 98.897838% | 4,718 | 4,666 |
| 2022-05-16 | 98.876643% | 4,718 | 4,665 |
| 2022-05-17 | 98.898072% | 4,719 | 4,667 |
| 2022-05-18 | 98.855932% | 4,720 | 4,666 |
| 2022-05-19 | 98.813559% | 4,720 | 4,664 |
| 2022-05-20 | 98.856659% | 4,723 | 4,669 |
| 2022-05-23 | 98.856417% | 4,722 | 4,668 |
| 2022-05-24 | 98.898538% | 4,721 | 4,669 |
| 2022-05-25 | 98.962304% | 4,722 | 4,673 |
| 2023-05-04 | 98.632891% | 4,974 | 4,906 |
| 2024-04-30 | 98.787642% | 5,114 | 5,052 |
| 2025-04-29 | 98.951456% | 5,150 | 5,096 |

### 6.2 候选 A 旧全活跃 stock-day 入场诊断：失败（该口径已由授权修正废止）

本任务不生成信号，因此使用主窗口内所有点时活跃主板 stock-day 作为“可进行实际候选入场决策”的保守分母；不是已触发入场或实际成交样本。

- 分母：7,334,981 行。
- 所有必需字段完整的分子：7,322,998 行。
- 至少缺一个必需字段：11,983 行。
- 联合字段覆盖率：99.83663216032871%。
- 数值有效覆盖率：99.83663216032871%。
- 阈值：100%。
- 上证综指 `000001.SH` 的市场闸门收盘字段：2,440/2,440 日，覆盖率 100%，均为正值。
- 总闸门：`False`，指数完整不能抵消个股必需字段缺失。

独立逐字段缺失计数：

| 字段 | 缺失行数 |
| --- | ---: |
| 原始 `open/high/low/close/prev_close` | 各 0 |
| 点时复权 `signal_open/signal_high/signal_low/signal_close` | 各 1,196 |
| `volume` | 10,964 |
| `amount` | 10,964 |
| `trade_status/is_st/ipo_date/execution_conflict` | 各 0 |

字段之间存在重叠，不能把逐字段缺失数直接相加；联合缺失为 11,983 行。原始收盘价缺失为 0，说明本次失败主要不是原始价缺失，而是量额和部分点时复权序列不完整。

### 6.3 历史行业与候选 B：失败

- 有效日期行业源：不存在。
- 行业面板记录：0。
- 主窗口逐日行业分子：0；分母沿用同日点时活跃全 A 宇宙；每日覆盖率均为 0%。
- 实际候选入场行业字段覆盖率：0%。
- 99% 逐日闸门和 100% 入场闸门均失败。
- 即使假设数据完整，Task 1 的 12 项公式/溯源问题仍使 B 保持 `NOT_EVALUABLE`。

## 7. 限制与后续边界

1. 本任务只证明当前冻结输入下的数据可行性，不证明策略有效性，也不构成实盘授权。
2. 点时活跃宇宙依赖 Baostock `stock_basic` 的 IPO/退市日期和交易日历；未把当前上市状态当成历史状态，但无法超越源库本身的完整性。
3. 第 6.2 节的全活跃主板 stock-day 口径仅保留为原失败证据；2026-11-04 起，100% 门槛只适用于 Task 3 实际触发且拟入场候选，不得再回到全活跃分母，也不得只检查实际成交样本。
4. 行业来源完全缺失，不能用当前快照、后续优化版或代码前缀推断替代。
5. 复权差异仍只作披露；没有因大量辅助复权差异而把它们错误计入执行冲突。
6. 未修改上一阶段模块、候选配置或 `.sdd-legacy/snapshots/task2-base`，未执行任何 git 操作。

## 8. Fix round 1（2026-11-04，用户批准的数据语义修正）

### 8.1 修正边界与原证据保留

本轮严格依据协调者转达的用户批准：

1. 100% 字段门槛只检查 Task 3 实际触发且拟入场候选，不再检查全部活跃主板 stock-day；检查仍在任何账户收益之前。
2. 全 A 每日 99% 门槛保留。正常停牌不进入可排名横截面，但 NULL 量额不再自动视为数据损坏；停牌仍在活跃分母且不得交易。
3. 当前活源库正被外部进程写入。本轮没有连接、查询或计算两个源库的当前哈希，没有更新 13 预注册中的历史数据库 manifest。
4. 只使用 SHA256 已核验匹配的原冻结 Parquet。未实现候选信号，未运行真实候选收益。

原审计已复制并保留：

- 路径：`/Users/admin/WorkSpace/trading/trading_finally/results/legacy_context_audit_pre_amendment.json`
- SHA256：`c30375094d8584131e529a50833d69ab5fabe479042e68bc05d2bc5f1a4ce899`
- 原全 A 文件仍为 `/Users/admin/WorkSpace/trading/trading_finally/data/legacy_full_a_panel.parquet`
- 原全 A SHA256 仍为 `8183acce5607b04946d80c05477fc26fd60791748a590200e5096d2d0e525a85`

当前可复现性明确拆分：

- `source_level_rebuild_reproducible=false`：活源库已外部变化，旧源库快照不可用，不能从当前源库复现原构建。
- `frozen_input_reproducible=true`：本轮所有输入 Parquet 和原审计均在构建前后核验哈希且未变化，修正后的下游构建可由这些冻结输入复现。

### 8.2 第二轮 RED / GREEN

在修改实现前扩充 `tests/test_legacy_data.py`，新增以下反例：

- 正常停牌且 NULL 量额：`context_complete=true`、`rank_eligible=false`。
- 正常交易且 NULL 成交额：`context_complete=false`。
- 严格候选入场有效行：正常完整行通过；停牌、零成交量、NULL 成交额、execution conflict 分别失败。
- hfq 精确同键补齐：只补 NULL，已有非空值不覆盖，其他日期不填充。
- 预期交易日整日缺失：相对独立日期集合必须抛出错误。
- A 全 A 通过后只能为 `PENDING_DATA_GATE`；B 即使假设数据完整也不能解除 12 项冻结。

实际 RED：

```text
ImportError: cannot import name 'assert_expected_trading_dates'
FAILED (errors=1)
```

实现后：

```text
Ran 20 tests in 0.034s
OK
```

按本轮要求运行全项目 unittest：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest discover -s /Users/admin/WorkSpace/trading/trading_finally/tests
```

实际结果：

```text
Ran 147 tests in 2.589s
OK
```

完整套件中的既有账户测试使用合成夹具并输出零收益；没有实现或运行 A/B 真实候选信号与收益。

### 8.3 正常停牌与可排名横截面

`refresh_context_flags` 现在分别处理：

- `trade_status=0`：量额可为 NULL 或非负值；原始/后复权价格、前收、ST、IPO 等其他上下文字段仍必须完整有效；`rank_eligible=false`。
- `trade_status=1`：所有上下文字段必须完整，量额必须非空且非负；只有上下文完整的正常交易行才可 `rank_eligible=true`。
- 其他或缺失交易状态：不作为完整正常交易/正常停牌处理。
- 活跃宇宙分母完全不变，停牌行没有被删除。

真实输出中，全窗口有 16,146 个正常停牌且量额至少一项为 NULL 的行，其中 15,672 行在其他字段有效后转为上下文完整；其余 474 行仍因其他必需字段无效而不完整。所有正常停牌行的 `rank_eligible=true` 数量为 0。正常交易且 NULL 量额但被标为上下文完整的行数为 0。

### 8.4 全 A 每日 99%：修正前后并列

| 指标 | 修正前 | 修正后 |
| --- | ---: | ---: |
| 主窗口预期/实际交易日 | 2,440 / 2,440 | 2,440 / 2,440 |
| 活跃分母 | 原 IPO/退市网格 | 逐日完全相同 |
| 低于 99% 天数 | 18 | 0 |
| 最低每日覆盖率 | 98.133616%（2022-05-05） | 99.870130%（2022-01-18，4,614/4,620） |
| 每日 99% 闸门 | 失败 | 通过 |

修正前后的 2,440 条逐日分母、分子和覆盖率均保存在新 audit JSON 的 `coverage.full_a.before/after`。独立从 completed Parquet 重算后，修正后每日分母和分子与 JSON 逐项一致，未用 pooled 比率替代逐日门槛。主窗口仍有 3,280 行上下文不完整，但每日不足均未超过活跃宇宙的 1%，因此没有伪装成全字段完整。

### 8.5 12 个精确键的 hfq 同字段补齐

冻结输入：

- `/Users/admin/WorkSpace/trading/trading_finally/data/hithink_signal_panel.parquet`
- SHA256：`6c7c8c070eca05e273fa67463787d085bbc8c8fbe38ed8ffc00d1b7245ccc53d`

实际处理结果：

- 原正常交易主板后复权信号不完整行：1,019。
- 精确 `(date,code)` 且同花顺四个 hfq 价均为正的候选键：12。
- 实际补齐键：12；每个键补齐 `signal_open/signal_high/signal_low/signal_close` 四个 NULL 字段，共 48 个字段值。
- 补齐后正常交易主板后复权信号不完整行：1,007。
- 非空原值被覆盖：0 行。
- 独立比较 12 键的 completed 与同花顺四价：不一致 0 行。
- 未做前值填充、未来填充、跨日期插值，未推断或修改 `trade_status/is_st`，未清除任何执行冲突。
- 10,964 个主板正常停牌 NULL 量额行保持原值，没有补成 0 或伪造成可入场。
- 12 个键及每个键的补齐字段已逐键写入新 audit JSON 的 `hfq_completion.keys`；补齐前后剩余原因在 `hfq_completion.before_after_remaining`。

补齐只使用授权内同一字段，不代表实际候选入场 100% 门槛通过。

### 8.6 严格候选入场有效行：保留给 Task 3

公共谓词为 `trend_system.legacy_data.candidate_entry_valid`。必须同时满足：

1. A 所需原始价、前收、hfq 四价、量额、交易状态、ST、IPO 和执行冲突字段完整。
2. 原始价、前收和 hfq 四价均为正。
3. `trade_status=1`。
4. `volume>0` 且 `amount>0`。
5. `execution_conflict=false`；NULL 冲突标志不接受。

停牌、零成交、NULL 量额和冲突均有独立合成反例。Task 3 应先得到实际触发且拟入场候选，再对该候选集合应用此谓词并计算 100%；当前 Task 2 不生成该候选集合，因此新审计的入场 `denominator/coverage/passes` 均为 NULL，`deferred_to_task3=true`。

为便于复核谓词，audit 另含一个明确标记为“不是闸门”的全活跃主板诊断：7,334,981 行中严格有效 7,225,283 行，独立 SQL 重算一致。该诊断不决定 A 的状态，也不得替代 Task 3 的实际候选分母。

### 8.7 独立预期交易日与整日缺失检测

预期交易日不再从实际 full-A 行自证，而是取冻结 `main_panel.parquet` 主窗口的 distinct 日期：

- 冻结主板 SHA256：`6a51ed0e7958aa1549dcf156e4387badcfced0236fd812d2bc1fd894aa3acc1a`
- 预期 2,440 日，completed 实际 2,440 日。
- 整日缺失 0，意外日期 0。
- 合成测试删除 2024-01-03 后实际触发 `Missing expected trading dates`，证明整日缺失不能从覆盖表中静默消失。

### 8.8 修正后产物与独立结构检查

| 产物 | 行数 | 字节数 | SHA256 |
| --- | ---: | ---: | --- |
| `/Users/admin/WorkSpace/trading/trading_finally/data/legacy_full_a_panel_completed.parquet` | 14,595,370 | 834,672,567 | `12e92da7600f59c72851da194ac13fca237d593d1cb8c2a0c329e4c28c23d9fb` |
| `/Users/admin/WorkSpace/trading/trading_finally/results/legacy_context_audit.json` | 不适用 | 1,518,262 | `d2e375ae169f078522d8b979dbea13f457b678d5325d3c77f2a8877d37d56c98` |

completed 面板保留原 14,595,370 行、5,500 codes、2010-01-04 至 2026-10-29 窗口和唯一 `(date,code)` 主键，并新增/更新 `context_complete`、`rank_eligible`。原面板未覆盖。行业面板仍为 0 行，B 的历史行业覆盖仍为 0。

### 8.9 当前状态与复审请求

- A：修正后的全 A 逐日 99% 闸门通过，状态仅为 `PENDING_DATA_GATE`；必须等待 Task 3 实际触发且拟入场候选的严格 100% 检查。
- B：`NOT_EVALUABLE`，12 项公式/溯源冻结和历史行业缺失均未改变。
- 13 预注册已记录两项批准语义、冻结输入补齐边界和可复现性限制。
- 本轮未执行 git、未派发子代理、未连接活源库、未运行真实候选收益。
- 已备齐原审查 Critical 1、Important 1/2、Minor 1 的修正证据，请交原审查者按 2026-11-04 新批准口径复审。
