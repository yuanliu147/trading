# Task 1 报告：旧候选唯一规则、预注册与严格配置

- 执行日期：2026-11-03
- 工作目录：`/Users/admin/WorkSpace/trading/trading_finally`
- 结论：**DONE_WITH_CONCERNS**。Task 1 的文件、严格加载和测试已完成；候选 B 因权威母版缺少多个必需公式及历史行业溯源而冻结为 `NOT_EVALUABLE`，候选 A 为 `PENDING_DATA_GATE`。
- 未运行任何真实候选收益，未联网、未连接券商、未执行 git 操作、未派发子代理，未修改账户引擎或 `.sdd-legacy/snapshots/task1-base`。数据库仅以只读文件字节流计算 SHA256 并识别文件头，未通过 DuckDB/SQLite 驱动连接或查询。

## 1. 交付文件

1. `/Users/admin/WorkSpace/trading/trading_finally/13_预注册_旧候选统一复核.md`
2. `/Users/admin/WorkSpace/trading/trading_finally/config/legacy_candidates.json`
3. `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/legacy_config.py`
4. `/Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_config.py`

冻结文件哈希：

| 文件 | SHA256 |
| --- | --- |
| `config/legacy_candidates.json` | `09096fc0cd796e0cfa0fd603a898f7b307b2a216ca771c4ae0500c55515e5e74` |
| `src/trend_system/legacy_config.py` | `56b05420c1afa09540222ffc1b770080d49e287b37cd266460977d72953dcaca` |
| `tests/test_legacy_config.py` | `09aff5e8f69cbb2bb314a75317cd93ef224b9571f6fe70426af644af62cd70cf` |
| `13_预注册_旧候选统一复核.md` | `131b7e3c58d936090d2415dbfafbeca37a938e796b6d6f703cfb402170d62eb7` |

## 2. TDD 的 RED/GREEN 证据

### RED（先写测试，模块尚不存在）

命令：

```text
PYTHONPATH=src python3 -B -m unittest tests.test_legacy_config
```

实际结果：退出码 1，`Ran 1 test ... FAILED (errors=1)`；导入错误为：

```text
ModuleNotFoundError: No module named 'trend_system.legacy_config'
```

这是预期的模块缺失 RED，不是断言误写或其他测试失败。RED 后才创建 `legacy_config.py`。

### GREEN（实现后新增测试）

命令：

```text
PYTHONPATH=src python3 -B -m unittest tests.test_legacy_config
```

实际结果：退出码 0，`Ran 17 tests in 0.007s`，`OK`。

### 既有套件回归

命令：

```text
PYTHONPATH=src python3 -B -m unittest discover -s tests
```

实际结果：退出码 0，`Ran 126 tests in 2.927s`，`OK`。126 = 既有 109 + 新增 17，既有账户引擎及基线测试无回归。数据库 manifest 补齐后已复跑相关测试与全套测试，结果即为本节所列；套件中的 robustness 输出均来自既有合成测试，不是两个旧候选的真实收益运行。

## 3. 权威旧源码路径与 SHA256

哈希命令为 `shasum -a 256 <三个精确路径>`，只读计算结果如下：

| 候选 | 权威源文件 | SHA256 |
| --- | --- | --- |
| A | `/Users/admin/WorkSpace/trading/2026-10-18_最优策略_原版三法取二_低波活跃/04_复现与代码/backtest.py` | `73b5eb05cba0f7090386f2d07de471bbff75b37c68c6151bc7f201635253c8a3` |
| A | `/Users/admin/WorkSpace/trading/2026-10-18_最优策略_原版三法取二_低波活跃/04_复现与代码/股性分类.py` | `52400a17160771a7846cf8f58a9e4899cdc2be5c3f541769dcbf32ace7ad2406` |
| B | `/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/02_代码/30b_组合回测_含退市.py` | `8406120aedd2ad21cbbca35e87fb784262434730b3399697b7afbe822b4686e8` |

预注册还登记了既有统一面板、`config.py`、`data.py`、`execution.py`、`broker.py` 的精确路径和 SHA256。

### 数据库输入 manifest（仅只读文件字节）

两个配置已固定的数据库文件已补齐准确路径、字节数、只读文件版本信息及 SHA256。哈希使用 `shasum -a 256` 流式读取文件字节，大小使用 `stat`，版本仅来自文件头及 `file`；没有使用 DuckDB/SQLite 驱动连接、没有 SQL 查询、没有写模式、同步、修复或迁移，也没有以 mtime 作为内容版本。

| 数据库精确路径 | 文件字节数 | 只读文件版本信息 | SHA256 |
| --- | ---: | --- | --- |
| `/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb` | 1,863,593,984 | DuckDB 文件头版本字符串 `v1.5.4`；头部标识 `08e34c447b` | `67561be6368efa5653107c884a6515dc249a7992a99d9f273f6a5cfd77e42f61` |
| `/Users/admin/WorkSpace/baostock_data/data/baostock.db` | 10,330,697,728 | SQLite 3.x；文件头 last-writer version number `3050004`（3.50.4），`version-valid-for=6` | `77a12ced7de4f21d1a9c26b60fbaca518437e2fc808ef994ca634b558b2c8a6e` |

配置结构无需新增字段：两个路径仍由 `hithink_db_path`、`baostock_db_path` 固定，完整字节 manifest 已登记在预注册和本报告。此次补齐不改变任何策略规则、参数、成本、数据阈值或候选状态。

## 4. 候选 A 逐项规则对照

| 规则项 | 权威旧源码还原结果 | 冻结状态 |
| --- | --- | --- |
| 市场闸门 | 上证综指收盘 > MA20 且 MA20 高于前一日；20 日简单均值含 T | 已冻结 |
| 个股趋势 | 收盘 > MA20 且 MA20 高于前一日 | 已冻结 |
| 成交额 | 含 T 的 20 日均成交额 >=3,000 万元 | 已冻结 |
| M1 | MA5 当日上穿 MA20：当日 >、前一日 <= | 已冻结 |
| M2 | EMA34-EMA200，WDEA 为差值 EMA13，柱为两倍差值；柱 >0 且 WDEA 上升；EMA `adjust=False` | 已冻结 |
| M3 | 收盘突破不含 T 的前 20 个完整交易日最高价，且量 >= 含 T 的 20 日均量 1.3 倍 | 已冻结 |
| 三法取二 | 命中数 >=2 | 已冻结 |
| 冷却 | 同一股票已接受信号原始行索引差 >=5，不是离场后冷却 | 已冻结 |
| `s_vol` | `vol120` 与 `range250` 横截面百分位均值；过滤 `<0.5` | 已冻结 |
| `s_heat` | `act`、`limfrac`、`gapfrac` 横截面百分位均值；过滤 `>=0.5` | 已冻结 |
| 股性历史闸门 | 最后 300 行、至少 260 行、快照当日有行情、60 日均额 >=2,000 万元 | 已冻结 |
| 股性指标细节 | `vol120` 用 `np.nanstd` 默认口径乘 `sqrt(252)`；range250 为高/低-1；act 为 20/120 均额-1；limfrac 阈值为 0.195（300/301/688）或 0.095；gapfrac 为 >2% | 已冻结 |
| 横截面 | 沪深 A、排除 BJ，快照质量与流动性筛选后 `rank(pct=True)`；不是当前名单 | 已冻结 |
| 排序 | 命中数×2 + 量/20 日均量×0.2 降序；代码最终 tie-break 来自已批准设计 | 已冻结 |
| 仓位 | 4 仓、单票 20%、单日最多新开 2、不加仓 | 已冻结 |
| **`low10` 是否含当日** | **含信号当日**：`rolling(10).min()` 后未 shift；用于入场时固定止损锚，不是每日重算 | 已冻结 |
| 买入价止损 | 入场锚为 `max(信号日low10, 实际买入价×0.92)`；公司行动按点时因子保持尺度 | 已冻结 |
| 时间退出 | `held>=40` 收盘确认，后续首个可执行开盘卖出 | 已冻结 |
| **持有/停牌计数** | **入场日计第 1 日；有个股行情且成交量 >0 才递增；停牌/零成交不计数** | 已冻结 |
| 结构退出 | MA5<MA20、魏氏柱<0、收盘<MA20 | 已冻结 |
| **原退出优先级** | **止损 > 到期 > MA5<MA20 > 魏氏柱<0 > 破MA20** | 已冻结 |
| ST | 两候选统一新增，不冒充旧规则；同日记录全部原因、只发一笔卖单，单独披露影响 | 已冻结（设计新增） |

候选 A 尚未执行全 A 点时覆盖和实际入场字段覆盖审计，故状态不是“已通过”，而是 `PENDING_DATA_GATE`。

## 5. 候选 B 逐项规则对照

| 规则项 | 权威母版还原结果 | 冻结状态 |
| --- | --- | --- |
| 原硬日期闸门 | `valid` 内含 `date>=2017-10-13`；主业绩窗口仍为 2016-10-13 起，不删除旧过滤器 | 已冻结 |
| 市场宽度 | 主方案 breadth 闸门，`MB>0.50` | 阈值可还原；公式 NOT_EVALUABLE |
| 趋势结构 | `U` 为真；设计含义为收盘>MA60、MA20>MA60、MA60 上行 | 条件保留；公式 NOT_EVALUABLE |
| **趋势年龄** | `5<=AGE<=30` | 阈值可还原；年龄算法 NOT_EVALUABLE |
| **C>=2** | 母版确实存在；统一规则为实际未复权收盘 >=2 元 | 已冻结 |
| **ATR** | `ATR<=0.044`，同时用于 sizing 的 `clip(2*ATR,0.04,0.12)` | 阈值可还原；ATR 公式 NOT_EVALUABLE |
| **DD60** | `DD60>-0.08` | 阈值可还原；公式 NOT_EVALUABLE |
| 20 日无涨停 | `LU==0` | 条件保留；计数公式 NOT_EVALUABLE |
| 行业 | `SBR>=0.60` 且 `SR20>0.05` | 阈值可还原；公式及历史行业 NOT_EVALUABLE |
| 成交额 | `AMT>=3,000 万`，主方案按 AMT 升序 | 阈值/方向保留；amt20 公式及并列次序 NOT_EVALUABLE |
| 原跳空过滤 | `gap_filter=True`，`NG[t-1]>0.05` 跳过；未静默省略 | 条件保留；公式/时点 NOT_EVALUABLE |
| 仓位 | 6 仓、单票 20%、不加仓、无单日新开上限 | 已冻结 |
| 单笔风险 | `risk_amt=equity×0.005` | 已冻结 |
| **风险股数** | `min(risk_amt/(entry×sd_size), equity×0.20/entry, cash/(entry×(1+cost)))`；统一账户再向下取 100 股；目标不足权益 1% 跳过 | 已冻结 |
| **heat** | `sum(shares×entry×sd_size)/equity`；`heat+0.005>0.0301` 跳过，保留旧源码实际 0.0301 拒绝线 | 已冻结 |
| 灾难止损 | 母版主方案 cat20；盘后修正为实际收盘相对调整后买入锚 <=-20%，下一可执行开盘卖 | 已冻结（含设计修正） |
| **持有/停牌计数** | `exit_i=entry_i+20-1`，入场日第 1 日；全局交易日索引前进，故个股停牌日也计入 20 日 | 已冻结 |
| 到期执行 | 第 20 日收盘确认，后续首个可执行开盘卖；停牌/跌停顺延 | 已冻结（含设计修正） |
| **退出优先级** | 原灾难止损 > 到期；ST 为新增并列披露，不虚构旧优先级 | 已冻结 |

### 候选 B 的 12 个 `NOT_EVALUABLE` 配置项

1. `up_struct_formula`
2. `trend_age_formula`
3. `atr_pct_formula`
4. `dd60_formula`
5. `lu_count20_formula`
6. `market_breadth20_formula`
7. `sector_breadth20_formula`
8. `sector_ret20_formula`
9. `amt20_formula`
10. `historical_industry_membership`
11. `next_gap_formula_and_timestamp`
12. `amount_ranking_tie_break`

权威母版仅加载这些矩阵，没有定义其构造。未读取后续版本来补缺，未用当前行业替代，未删除任何过滤器继续回测。

## 6. 配置与预注册一致性

### 6.1 共同值

| 配置字段 | JSON / 加载值 | 预注册位置 |
| --- | --- | --- |
| `start_date` / `end_date` | `2016-10-13` / `2026-10-29` | 第 2 节 |
| `initial_cash` | `20000` | 第 2 节 |
| `commission_rate` / `min_commission` | `0.00025` / `5` | 第 2 节 |
| `slippage_bps` | `10` | 第 2 节 |
| `random_seed` | `20261101` | 第 5、7 节 |
| `unrejected_max_drawdown` / `max_main_drawdown` | `0.1` / `0.2` | 第 5、7 节 |
| `random_paths` / `random_positive_min` | `20` / `16` | 第 5、7 节 |
| `stress_cost_multiplier` / `stress_slippage_bps` | `2` / `20` | 第 5、7 节 |
| `rolling_three_year_positive_threshold` | `0.5`，裁决为严格超过 | 第 5、7 节 |
| `point_in_time_coverage_min` | `0.99` | 第 6 节 |
| `industry_coverage_min` | `0.99` | 第 6 节 |
| `required_entry_coverage` | `1.0` | 第 6 节 |
| 数据库路径 | 与既有 `baseline.json` 相同 | 第 1.2 节 |

### 6.2 候选与源文件

- JSON 候选顺序及名称为 `three_of_two_low_vol`、`early_trend_breadth`，无第三候选。
- A 状态 `PENDING_DATA_GATE`，两份 source 路径/哈希与本报告第 3 节一致；B 状态 `NOT_EVALUABLE`，一份 source 路径/哈希一致。
- A 的冷却、三法窗口、low10、40 日、停牌不计数、4 仓/20%/2 新开、股性全部窗口与阈值、排序权重及退出顺序逐项对应预注册第 3 节。
- B 的旧硬日期、6 仓/20%/0.5%/0.0301、20 日且停牌计数、全部入场阈值、gap 过滤、20% 灾难退出、ATR sizing 上下限、1% 最小仓位、退出顺序和 12 个不可评价键逐项对应预注册第 4 节。
- 新增测试直接断言上述关键值、源哈希、状态、持有计数、退出优先级和不可评价键，防止加载时漂移。

### 6.3 严格性

实现沿用现有 `ResearchConfig` 的 `json.load` + frozen dataclass + 字段集合拒绝风格，并增加嵌套候选/来源的严格解析：

- 顶层、候选、来源的未知字段和缺失字段均拒绝；额外策略参数拒绝。
- 候选缺失、重复、额外或未知名称均拒绝；输出固定为两个冻结候选。
- 非对象根、错误标量类型、重复 JSON 键、`NaN`、`Infinity`、`-Infinity`、 malformed JSON 和缺失文件均拒绝。
- 配置、候选、来源均为 frozen dataclass，数组转为元组。
- 不读取环境变量或 CLI 参数；测试证明两种外部输入均不能覆盖 20,000 元等冻结值。

## 7. 数据闸门与执行冲突冻结

- 全 A 上下文为沪深 A、排除 BJ；逐日点时活跃宇宙覆盖 >=99%，不是全期 pooled 覆盖。
- 实际候选入场所需字段及历史行业入场字段均须 100%；历史行业逐日覆盖 >=99%，须有有效日期，不足即对应候选不可评价。
- `execution_conflict` 沿用 `data.py`：实际未复权 OHLC 单侧缺失或绝对差 >0.011 元（含既有容差）；量额差异不单独构成执行冲突。
- `adjustment_source_conflict` 只披露和进入既定敏感性，不冒充成交冲突；`source_conflict` 为审计并集，不单独阻断。
- Task 1 未执行数据构建、覆盖审计或真实候选收益。

## 8. 关键顾虑与后续边界

1. **候选 B 当前不可评价**：核心公式、历史行业和 `next_gap` 时点无法由唯一母版还原。不能以“参数已知”为由绕过公式缺失，也不能从后续版本择优补全。
2. **候选 B 的原跳空过滤存在盘后时点疑点**：已保留并标 `NOT_EVALUABLE`，未擅自改成可执行规则或删除。
3. **候选 A 仍待数据闸门**：源码规则唯一不等于数据已点时合格；未通过覆盖审计前不得查看真实收益。
4. **ST 与盘后执行属于统一修正**：已与旧原规则分开披露；这些变更不能被描述成旧策略原版，也不得因其结果不佳而删除。

## 9. Fix round 1：仅处理审查两项

### Important：补齐候选 A 实际入场否决

- 权威依据：`/Users/admin/WorkSpace/trading/2026-10-18_最优策略_原版三法取二_低波活跃/04_复现与代码/backtest.py` 第 616—618 行。
- 13 预注册第 3.3 节新增明确规则：实际入场锚 `max(信号日low10, 实际买入价*0.92) >= 实际买入价` 时放弃该次买入，保留 `>=` 边界；信号仍按原冷却规则收集，不建立仓位，不改成止损成交或下调锚。
- JSON 与 frozen dataclass 新增 `skip_if_stop_at_or_above_entry=true`；既有严格加载器将该布尔字段纳入必填、未知/缺失字段和标量类型校验。后续策略实现可消费该字段，本任务未实现信号或账户逻辑。

### Minor：补齐精确回归断言

- 测试以 `assertEqual` 冻结完整 12 项 `not_evaluable_fields` 元组及顺序，不再只检查其中 4 项。
- 直接断言 `gap_filter=True`、`gap_max=0.05`、`lu_filter=True`。

### Fix round 1 RED/GREEN 证据

RED 在修改预注册、JSON 和实现前执行：

```text
PYTHONPATH=src python3 -B -m unittest tests.test_legacy_config
```

实际结果：退出码 1，`Ran 18 tests`，`FAILED (failures=1)`；唯一失败为新增 `test_candidate_a_entry_veto_matches_preregistration`，因为预注册尚无 `skip_if_stop_at_or_above_entry=true`。

GREEN：

```text
PYTHONPATH=src python3 -B -m unittest tests.test_legacy_config
```

实际结果：退出码 0，`Ran 18 tests in 0.008s`，`OK`。

全套回归：

```text
PYTHONPATH=src python3 -B -m unittest discover -s tests
```

实际结果：退出码 0，`Ran 127 tests in 2.753s`，`OK`。127 = 既有 109 + 本任务 18。

本次 fix 未修改其他策略规则、阈值、数据库 manifest、账户引擎或结果，未运行真实候选收益，未执行 git，未派发子代理。

