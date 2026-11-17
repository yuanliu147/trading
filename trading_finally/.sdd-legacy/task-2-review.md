# Task 2 独立审查：旧候选统一复核

- 审查日期：2026-11-04
- 审查依据：任务简报、13 预注册、完整 diff、实际源码、真实审计 JSON 与 Parquet；未采信实现报告的结论。
- 审查方式：只读源码/SQL/元数据检查，独立重算逐日覆盖、入场分母与分子、主键和边界，并独立核验文件 SHA256。
- 总结：**Task quality = Needs fixes**。当前 A、B 均必须维持 `NOT_EVALUABLE`；不得运行候选收益。

## Spec Compliance

| 项目 | 结论 | 独立证据 |
| --- | --- | --- |
| 点时活跃宇宙与 IPO/退市边界 | 符合当前实际输出 | `sql/legacy_context.sql:19-31` 使用 `ipo_date <= date < out_date`，不以当前上市状态过滤历史。实际 Parquet 为 14,595,370 行、5,500 codes，`(date, code)` 全部唯一；pre-IPO、已退市仍残留、BJ 行均为 0。5,500 也与构建窗口内符合上市边界的 `stock_basic` 代码数一致。 |
| 全 A 与主板范围 | 符合 | SQL 保留 SH/SZ `type=1` 全 A；`src/trend_system/legacy_data.py:34,246` 仅以冻结的 600/601/603/605/000/001/002/003 前缀标识主板执行宇宙。独立源查询得到主板主窗口分母 7,334,981，与审计一致。 |
| 历史行业与当前快照 | 符合 fail-closed | 两个实际源库没有行业字段；行业 Parquet 为 0 行但 schema 完整，审计给出 0 覆盖。`point_in_time_industry` 排除无 `valid_from` 和未来生效记录；没有用当前行业回填。 |
| 逐日 99% 覆盖 | 分母/分子真实，闸门失败 | 独立从 Parquet 重算 2,440 个交易日，所有日期的 denominator/numerator/coverage 与审计逐项一致；18 日低于 99%，最低为 2022-05-05 的 4,627/4,715 = 98.133616%。没有用 pooled 99.832571% 掩盖单日失败。 |
| 100% 入场覆盖分母 | 分母符合，分子有 Important 缺陷 | `src/trend_system/legacy_data.py:437-441` 使用主窗口全部点时活跃主板 stock-day，未缩窄为已触发信号。独立重算 denominator=7,334,981、字段非空 numerator=7,322,998，field coverage=99.836632%，故未达 100%。但 `valid_numerator` 错计不可交易/执行冲突行，详见 Issues。 |
| NULL、零成交、复权缺失 | 部分符合 | NULL/负量额和非正 OHLC 被排除，复权缺失不会由未来行回填；但正常字段完整的停牌/零成交行被错误计入 A 的 `valid_numerator`。11,983 个不完整主板行中，10,964 行是 `trade_status=0` 且量额 NULL，1,019 行是 `trade_status=1` 但后复权信号缺失。 |
| SQL、字段映射与复权方向 | 符合 | SQL 只有 SELECT；SQLite 使用 URI `mode=ro`，DuckDB 使用 `read_only=True`。`adjustflag=3` 映射实际未复权 OHLC，`adjustflag=1` 映射后复权信号 OHLC；同花顺 `v_daily_hfq` 也使用 `backward_factor`，方向一致。SQL 文件实际 SHA256 与审计内嵌文本及哈希一致。 |
| 冲突继承 | 符合既有冻结定义 | `src/trend_system/legacy_data.py:247-257` 直接按 `(date,code)` 继承主面板三类标志；实际输出 1 / 1,079,751 / 1,080,277 个 execution/adjustment/source-conflict 行与既有面板审计一致。复权差异没有冒充执行冲突。 |
| 源库 SHA256 | **当前不符合可复现要求** | 实现确实在构建前后计算哈希，但本次独立计算的两个当前源库哈希均不等于预注册/审计；Baostock 在两次只读哈希之间还发生变化。详见 Critical 1。 |
| 输出、状态与 NOT_EVALUABLE 传播 | 符合 | 实际 full-A/industry 行数、窗口、主键与审计一致；2440 个主窗口日期与独立交易日历一致。A/B 均为 `NOT_EVALUABLE`，B 的 12 个规则级缺失项完整保留，`restored=false`，未实现信号、未运行收益。 |

## Strengths

1. **点时宇宙构造正确且失败关闭。** 采用交易日历与上市/退市日期生成活跃网格，缺失行情通过 LEFT JOIN 留空而不是删除分母；实际主键、边界和代码范围均通过独立重算。
2. **逐日覆盖没有 pooled 替代。** 每日分母固定为活跃全 A，2,440 条日记录可逐项复算；98.99%/99% 边界测试存在。
3. **100% 分母遵守冻结口径。** 分母明确是全部主窗口点时活跃主板 stock-day，没有事后改成信号触发行或实际买入行。
4. **行业处理保守。** 无有效日期行业数据时输出真实空表和零覆盖；B 的规则级 `NOT_EVALUABLE` 不会因数据构建而恢复。
5. **只读和复权映射清晰。** 源 SQL 只读，原始价与后复权信号价分列；同花顺后复权视图方向与 Baostock 映射一致。
6. **现有冻结面板身份可验证。** `main_panel.parquet`、`index_panel.parquet`、`hithink_signal_panel.parquet` 的独立 SHA256 均与预注册一致；审计中的 SQL 哈希也与实际文件一致。

## Issues

### Critical 1 — 当前源库已不匹配冻结 manifest，构建前后哈希记录现在不可独立复现

- **位置：** `src/trend_system/legacy_data.py:35-45,178-194,368,508-513`；`results/legacy_context_audit.json` 的 `databases.before/after`。
- **证据：** 本次用 Python hashlib 与系统 `shasum -a 256` 独立只读计算：
  - Hithink 登记值为 `67561be6…f61`，当前为 `9498e7b7…647`。
  - Baostock 登记值为 `77a12ced…a6e`；第一次当前值为 `1584cd8a…b2a2`，随后 `shasum` 得到 `1687d01a…c515`，说明文件在审查期间仍发生变化。
  - 两者当前字节数仍等于登记值，不能据此认定内容未变。
- **影响：** 代码的前后哈希检查本身是正确的，且现在重新构建会 fail-closed；但现有输出无法用当前源库独立重建。该发现不证明原构建记录虚假，却使“冻结输入可复现”当前不成立。
- **最小修法：** 恢复或挂接与预注册完全一致的只读源库快照，独立重算 before/after SHA256 后重建并复核；优先使用不可变快照，隔离实时写入。**不得直接把 manifest 改成当前活库哈希来冒充原冻结版本。** 若旧快照无法恢复，应明确标记当前构建不可复现，而不是声称哈希核验通过。

### Important 1 — A 的 `valid_numerator` 把停牌、零成交和执行冲突行计为有效

- **位置：** `src/trend_system/legacy_data.py:258-266,437-441,480-495`；缺失测试位于 `tests/test_legacy_data.py:50-92`。
- **证据：** `context_complete` 只要求量额 `>=0`，`a_entry_valid_rows` 只要求字段完整及 `context_complete`，没有要求 `trade_status=1`、正成交量/成交额，也没有排除 `execution_conflict=true`。实际被当前审计计入 valid、但按冻结规则应阻断的并集为 **97,727 行**：其中 97,726 行 `trade_status!=1`，97,720 行零成交量，97,722 行零成交额，1 行执行冲突。仅剔除这些已阻断行后，现有 valid 上限为 7,225,271/7,334,981 = **98.504291%**，不是审计的 99.836632%。
- **影响：** 违反 13 预注册第 6.2 条“不得把不可交易当成字段完整”及第 6.4 条执行冲突阻断。当前 A 因其他缺失仍然失败，所以没有造成虚假 `EVALUABLE`，但有效覆盖证据被高估，后续补数据时可能产生错误通过。
- **最小修法：** 保持 denominator=7,334,981 和 100% 门槛不变；把入场有效分子单独定义为字段完整、价格/复权有效、正常交易、量额有效且非执行冲突的行。量额字段非空覆盖可保留为单独诊断，但不能代替可评价性。增加停牌、零成交、NULL 量额和 execution conflict 的合成测试。

### Important 2 — A 尚未穷尽授权数据内的“同一字段补齐”

- **位置：** `src/trend_system/legacy_data.py:368-415,431-441`；构建行情完全来自 `sql/legacy_context.sql` 的 Baostock 表，同花顺仅用于行业 schema 检查。
- **证据：** 在 SHA256 与预注册一致的 `data/hithink_signal_panel.parquet` 中，按精确 `(date,code)` 只读匹配：1,019 个“正常交易但 Baostock 后复权信号缺失”行里，有 **12 行**存在四价均正的同花顺后复权同字段候选值。当前实现没有尝试或核销这 12 行，因此不能把现有 `NOT_EVALUABLE` 描述为已穷尽授权数据。
- **影响：** 数据闸门结论方向目前仍是失败，但 Task 2 的授权源利用和同字段补齐审计不完整。12 行只是待冲突审计的候选，**不代表补齐后可通过**：其余 1,007 个正常交易复权缺失行、10,964 个停牌 NULL 量额行、既存不可交易行以及 18 个逐日 99% 失败日仍须按冻结规则处理。
- **最小修法：** 增加一次严格受限的同字段补齐/核销审计，输出补齐前后逐字段 numerator 和剩余原因；不得改分母、覆盖率门槛或实现新策略。具体要求见 Assessment。

### Minor 1 — 整日缺失不会由预期交易日集合独立检出

- **位置：** `src/trend_system/legacy_data.py:428-435,459-473`。
- **证据：** `expected_dates` 来自实际 full-A 行里的 `performance_dates`，而不是独立交易日历；若某交易日整体缺失，该日会从覆盖表和指数分母一起消失。当前实际输出为 2,440 日，与独立 `trade_dates` 的 2,440 日完全一致，因此这是稳健性缺口，不是当前日期虚报。
- **最小修法：** 从冻结 Baostock `trade_dates` 生成主窗口预期日期并断言集合完全相等；增加“整日缺失必须失败”的测试。

## Assessment

### Task quality

**Needs fixes。** 存在未解决的 Critical 1 与 Important 1/2，不能 Approved。当前输出的大部分行数、主键、日期和覆盖记录是真实可重算的，但不能绕过上述问题。

### 是否应做 A 的同字段补齐

**应做；当前尚未穷尽。** 但仅允许在既有授权数据内补同一字段，并遵守以下约束：

1. 优先使用哈希仍匹配预注册的 `hithink_signal_panel.parquet`；若直接查询 Hithink 数据库，必须先恢复 manifest 一致的冻结快照。当前哈希不匹配的活库不能静默替代授权输入。
2. 只允许精确 `(date,code)` 同键匹配；不得前值填充、未来值回填、当前快照回填或跨日期插值。后复权信号只能使用同方向 hfq/backward 序列，实际成交仍使用未复权价。
3. 只补 NULL 字段，不覆盖已有非空值。实际未复权 OHLC 的任一侧缺失或绝对差超过冻结阈值仍属 `execution_conflict`；量额差异仅进入审计冲突，复权差异仅披露，均须沿用 13 预注册，不得借补齐清除冲突。
4. 同花顺没有点时 `trade_status/is_st`，不得推断或伪造这些字段。停牌、零成交及执行冲突行不得进入有效入场分子；尤其不能把停牌 NULL 量额补成 0 后视为可入场。
5. 补齐后仍须用原来的全部点时活跃主板 stock-day 分母重算 100%，并逐日重算全 A 99%；任何不足都继续维持 A=`NOT_EVALUABLE`。

### B 的状态

B 的 12 项规则/溯源缺失没有因本任务消失，历史行业仍为空且覆盖失败。**B 必须保持 `NOT_EVALUABLE`，不得因任何 A 字段补齐或数据构建恢复。**
