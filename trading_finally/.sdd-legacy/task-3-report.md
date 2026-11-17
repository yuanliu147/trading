# Task 3：候选 A 三法取二 + 低波活跃点时信号报告

- 执行日期：2026-11-04
- 候选：`three_of_two_low_vol`
- 最终状态：**`EVALUABLE`（仅表示 Task 2 全 A 闸门及本任务实际拟入场键 100% 数据闸门通过）**
- 主窗口：2016-10-13 至 2026-10-29
- 范围边界：只实现 T 日收盘信号、冷却、排序分及信号日元数据；没有实现退出、仓位或账户策略，没有运行任何候选账户收益，也没有恢复候选 B。

## 1. 交付物

- 实现：`/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/legacy_a.py`
  - SHA256：`aff1653eb3d3d4586181aaea02c5dddd2ee133c4d3884519c2c4849811f2ca52`
- 合成测试：`/Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_a.py`
  - SHA256：`d3cd230545faea5d4279b7b5e9fca2179c0e272792a100e4bc694b1ec569097b`
- 冻结实算入口：`/Users/admin/WorkSpace/trading/trading_finally/scripts/run_legacy_a.py`
  - SHA256：`8c9f46f76bbd8dbf400c6c60c8ff07b54fbd9e65346e9dcf419bce2dcfea9a3d`
- 配置：`/Users/admin/WorkSpace/trading/trading_finally/config/legacy_candidates.json`
  - SHA256：`09096fc0cd796e0cfa0fd603a898f7b307b2a216ca771c4ae0500c55515e5e74`

所有新交付均位于 `trading_finally/`；未修改 `.sdd-legacy/snapshots/task3-base`，未执行 git，未派发子代理。

## 2. TDD：RED / GREEN 证据

### 2.1 首轮 RED

先建 `tests/test_legacy_a.py`，使用三只股票、330 个连续工作日的**日期主序**合成面板；目标行索引为 300，之前保留完整热身历史。三只股票分别为低波活跃主板、较高波动主板和非主板全 A 上下文股票。

首轮覆盖公式、三法取二、全 A 百分位横截面、严格阈值、未来行业/除权截断、原始行索引冷却、low10 含 T、ST/冲突/非主板/窗口/停牌/零成交权限、无有效横截面、缺失布尔 fail-closed 和逐键入场覆盖。

实际执行：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_a.py
```

实际 RED：

```text
ModuleNotFoundError: No module named 'trend_system.legacy_a'
Ran 1 test in 0.000s
FAILED (errors=1)
```

失败原因是被测接口不存在。最小实现后又实际观察到公式/夹具行为失败并逐项修正，没有修改冻结阈值或规则参数；当前 pandas 的 `pct_change()` 默认口径被保留，未另造历史填充规则。

### 2.2 补充反例的 RED

1. **completed 后复权同键对齐**：主板输入目标键的 `signal_open=NULL`，completed 上下文同键为有效值。实现前测试得到 `np.isfinite(row["signal_open"]) == False`，即 RED。随后按 `(date,code)` 显式取 completed 四价，测试转 GREEN；没有位置补齐。
2. **主板键不在点时全 A 活跃网格**：实现前抛出 `ValueError: Main-panel keys missing from full-A context`，即 RED。修正后保留主板行、标记 `context_key_present=false`，无有效横截面因而无信号，不伪造上下文。

### 2.3 最终 GREEN

```text
Ran 11 tests in 0.370s
OK
```

与 Task 1/2 配置、上下文联合回归：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B -m unittest \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_a.py \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_config.py \
  /Users/admin/WorkSpace/trading/trading_finally/tests/test_legacy_data.py
```

实际结果：

```text
Ran 49 tests in 0.448s
OK
```

未运行包含账户/收益测试的项目全套件，以遵守本任务“不得运行任何账户收益”的边界；这不是对未运行测试的通过声明。

## 3. 公式与 13 预注册逐项对应

| 预注册项目 | 本实现 |
| --- | --- |
| 市场闸门 | 仅上证综指 `000001.SH`；`close>MA20` 且 `MA20>MA20.shift(1)`；含 T 的 20 日简单均值，完整指数历史热身。指数按显式 `(date,market_code)` 连接。 |
| 个股趋势 | 点时 `signal_close>MA20` 且 MA20 高于前一行。 |
| 流动性 | 含 T 的 20 日平均成交额 `>=30,000,000`。 |
| M1 | `MA5>MA20` 且前一行 `MA5<=MA20`。 |
| M2 | `wdif=EMA34(close)-EMA200(close)`，`wdea=EMA13(wdif)`，`whist=2*(wdif-wdea)`；`whist>0` 且 `wdea` 上升；全部 `ewm(adjust=False)`。 |
| M3 | `close>high.rolling(20).max().shift(1)`，不含 T；`volume>=1.3*volume.rolling(20).mean()`，量均值含 T。 |
| 三法取二 | 三个显式布尔值求真和，命中数 `>=2`；`pd.NA/NaN` 先 fail-closed，不计命中。 |
| 股性历史 | 每股全历史用于信号指标；股性特征等价于快照日 `tail(300)` 内固定窗口，至少 260 行且当前快照键存在。`amt60>=20,000,000` 后才可进入排名。 |
| `vol120` | `close.pct_change()` 后最近 120 项总体标准差 `ddof=0`，乘 `sqrt(252)`；缺失处理与 `nanstd` 的跳过缺失语义一致。 |
| `range250` | 最近 250 行 `max(signal_high)/min(signal_low)-1`。 |
| `act` | 最近 20 日平均成交额 / 最近 120 日平均成交额 - 1；长窗均值必须大于 0。 |
| `limfrac` | 最近 250 个收益绝对值大于等于阈值的比例；300/301/688 为 0.195，其余 0.095。 |
| `gapfrac` | 最近 250 个 `abs(open_t/close_(t-1)-1)>0.02` 的比例，分子分母错位一行，与 `open[-250:]/close[-251:-1]` 对应。 |
| 横截面 | 全 A 每个快照日分别对 `vol120/range250` 和 `act/limfrac/gapfrac` 使用默认平均秩 `rank(pct=True)`，再求均值；沿用均值跳过缺失，最终 `s_vol/s_heat` 缺失者无信号。不是仅主板排名。 |
| 停牌 | 当前快照 `rank_eligible=false`，正常停牌不进入横截面；没有用 NULL 量额构造排名。 |
| 股性过滤 | 严格 `s_vol<0.5` 且 `s_heat>=0.5`；无有效横截面时全无信号。 |
| 冷却 | 对每只股票的原始接受信号按全历史 `source_row_index` 顺序处理，仅接受与上一接受索引差 `>=5` 的信号；不是持仓退出冷却，也不按主窗口重新起算。 |
| 排序 | 保存 `hit_count*2 + volume/volume_ma20*0.2`；兼容字段 `rank_return_120` 在此处就是该原排序分，不是 120 日收益。代码和唯一键保留给 Task 5 确定性排序。 |
| `low10` | `signal_low.rolling(10).min()`，包含信号日、不 shift。 |
| 止损元数据 | `stop_ref=low10`、`stop_pct=0.08` 随信号保存；未计算实际买入价，未提前执行 `stop>=entry` 否决。 |
| 执行权限 | `proposed_entry` 保留实际触发且拟入场分母；最终 `entry_signal` 另要求主板、主窗口、非 ST 及 Task 2 严格有效谓词。停牌、零/负量额、NULL 必需字段和执行冲突均 fail-closed。 |

不同股票面板之间只按 `(date,code)` 做一对一显式连接；没有用位置掩码对齐不同表。同一股票内部的 `.iloc` 仅用于计算该股票自身的历史指标。

## 4. 点时、横截面、截断与权限反例

1. **未来行业/除权截断**：把 T 之后的行业标记、后复权 OHLC、量额改写后，再与截至 T 的截断输入比较；所有历史诊断列按 `(date,code)` 外连接后逐项完全相等。
2. **全 A 排名而非主板排名**：目标主板在含非主板股票的横截面中 `s_vol=1/3`、`s_heat=1`，可以通过；移除该非主板股票后目标 `s_vol=0.5`，严格 `<0.5` 使其无信号。非主板本身不能产生最终入场。
3. **冷却**：连续原始行 300—305 均触发时，只接受 300 和 305；301—304 标为冷却拒绝，305 的前接受索引明确为 300。
4. **low10 含 T**：把信号日低价改为此前 9 日最低价以下，`low10/stop_ref` 等于信号日新低价；没有 shift。
5. **权限**：ST 和执行冲突反例仍保留原接受/拟入场审计记录，但最终不可入场；非主板、窗口外、停牌、零成交同样不可入场。
6. **缺失与空横截面**：三法布尔缺失不计命中；所有股票当前不可排名时无信号；任一拟入场键缺 `signal_open` 时覆盖率降为 0、状态为 `NOT_EVALUABLE`，没有先删除该键再伪称 100%。
7. **completed 同键补齐**：主板原面板 NULL 不覆盖 completed 的有效点后复权值，按精确键进入信号和权限判断。
8. **无点时上下文键**：保留主板行但无横截面、无信号；不构造替代数据。

## 5. 真实冻结输入与执行边界

实算前后逐一计算 SHA256，四组结果完全相同：

| 输入 | 行数/用途 | SHA256 |
| --- | --- | --- |
| `/Users/admin/WorkSpace/trading/trading_finally/data/legacy_full_a_panel_completed.parquet` | 14,595,370；全 A 指标和横截面 | `12e92da7600f59c72851da194ac13fca237d593d1cb8c2a0c329e4c28c23d9fb` |
| `/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet` | 10,776,532；执行键和实际未复权字段 | `6a51ed0e7958aa1549dcf156e4387badcfced0236fd812d2bc1fd894aa3acc1a` |
| `/Users/admin/WorkSpace/trading/trading_finally/data/index_panel.parquet` | 上证综指市场闸门 | `8b8763e0a552894eee84407f1b0c5e6efa4ff7eb8640ddb149b637539bef387a` |
| `/Users/admin/WorkSpace/trading/trading_finally/results/legacy_context_audit.json` | Task 2 闸门与状态 | `d2e375ae169f078522d8b979dbea13f457b678d5325d3c77f2a8877d37d56c98` |

运行前确认 Task 2 `full_a_context_gate=true` 且 A 为 `PENDING_DATA_GATE`。实算命令：

```bash
PYTHONPATH=/Users/admin/WorkSpace/trading/trading_finally/src python3 -B \
  /Users/admin/WorkSpace/trading/trading_finally/scripts/run_legacy_a.py \
  --config /Users/admin/WorkSpace/trading/trading_finally/config/legacy_candidates.json
```

只读上述冻结文件；没有连接、查询或更新活源库，没有运行 SQL 源库重建。独立复核使用内存 DuckDB 直接读取结果 Parquet，也不连接源库。

## 6. 真实信号、键、窗口与权限证据

| 指标 | 真实结果 |
| --- | ---: |
| 全 A 上下文行数 | 14,595,370 |
| 输出信号行数 | 10,776,532 |
| 唯一 `(date,code)` 键数 | 10,776,532 |
| 主板键无点时全 A 上下文 | 198 |
| 上述 198 键的 raw/accepted/proposed/entry | 0 / 0 / 0 / 0 |
| 全历史原始规则信号 | 42,999 |
| 全历史冷却后接受信号 | 30,079 |
| 主窗口实际触发且拟入场候选 | 21,554 |
| 拟入场但因 ST 权限拒绝 | 317 |
| 最终 `entry_signal=true` | 21,237 |
| 最终入场股票数 | 3,094 |
| 最终首日 / 末日 | 2016-10-18 / 2026-09-23 |

198 个差异键来自主板执行面板未进入 Task 2 点时活跃网格的行；首次实算因此在键闸门中止且没有生成信号。修正不是删除这些主板行或补造排名，而是保留并 fail-closed。最终独立查询确认它们没有任何层级信号。

真实断言及独立进程复核：

| 检查 | 违规/不匹配数 |
| --- | ---: |
| 非主板最终入场 | 0 |
| ST 最终入场 | 0 |
| 执行冲突最终入场 | 0 |
| 窗口外最终入场 | 0 |
| 停牌或非正量额最终入场 | 0 |
| 冷却原始行索引差 `<5` | 0 |
| 最终入场 `low10` 与含 T 的 10 日最低价不一致 | 0 |
| `stop_ref!=low10` 或 `stop_pct!=0.08` | 0 |
| 最终入场排序分非有限值 | 0 |

317 个 ST 候选仍属于规则实际触发且拟入场集合，并进入 100% 字段审计；它们只是被最终权限层拒绝，没有从分母中隐藏。

## 7. 实际拟入场候选 100% 覆盖

逐键审计文件只含 21,554 个 `proposed_entry=true` 键，不采用全活跃 stock-day 分母，也不只检查实际成交样本。

每个键检查 Task 2 固定的 15 个必需字段：原始 OHLC、前收、点时后复权 OHLC、成交量、成交额、交易状态、ST、IPO 日期、执行冲突。除字段完整外，还必须满足原始/复权价格为正、正常交易、量额为正、`execution_conflict=false`。

| 覆盖指标 | 结果 |
| --- | ---: |
| 分母 | 21,554 |
| 唯一键 | 21,554 |
| 必需字段完整键 | 21,554 |
| 严格有效键 | 21,554 |
| 覆盖率 | **100%** |
| 与信号拟入场键外连接差异 | 0 |
| 门槛 | 100% |
| 结果 | **通过，A=`EVALUABLE`** |

因此冻结信号可交给 Task 5/6；这只是后续账户计算的数据许可，本任务没有据此运行账户收益或作任何有效性/收益结论。

## 8. 冻结输出

| 产物 | 行数 | 字节数 | SHA256 |
| --- | ---: | ---: | --- |
| `/Users/admin/WorkSpace/trading/trading_finally/results/legacy_a_signals.parquet` | 10,776,532 | 1,985,889,439 | `a3b274ad470f0aff9ce268e4cf172a437f864b963eda3703969aecbfde5fc390` |
| `/Users/admin/WorkSpace/trading/trading_finally/results/legacy_a_entry_coverage.parquet` | 21,554 | 1,577,181 | `b91342891360a87dccfbe4c7bc82c0caeff7ac62fd0d7dd069fba2fbd493129c` |
| `/Users/admin/WorkSpace/trading/trading_finally/results/legacy_a_summary.json` | 不适用 | 6,792 | `805204cac8c9ddc0d130bccf001fb9361072aa412427a01ea2b073ad2788bd4b` |

信号保留完整主板键、`entry_signal`、原排序分、三法布尔与命中数、市场/个股指标、股性特征和百分位、原始行索引、冷却状态、前接受索引、`low10/stop_ref/stop_pct` 以及覆盖与权限诊断字段。

## 9. 限制与后续边界

1. 未运行账户、成交、退出或收益；本报告不证明候选 A 的盈利能力、稳健性或可实盘性。
2. Task 5 仍须按冻结配置计算后续首个可执行开盘、实际买入价、`max(low10,实际买入价*0.92)`、`stop>=entry` 否决、退出和仓位。本任务没有提前删除止损冲突信号。
3. `legacy_rank_score` 是候选 A 原排序分的权威字段；`rank_return_120` 仅为现有账户兼容保留的等值别名，不应解释为 120 日收益。Task 5/6 必须消费 `legacy_rank_score`。
4. 点时上下文仍依赖 Task 2 冻结面板的 IPO/退市、ST、交易状态和字段质量；本任务不能超越源数据本身。
5. 198 个主板执行键没有点时全 A 上下文，采取无信号处理并保留审计；没有用当前名单或未来数据补齐。
6. 当前仅保证冻结输入可复现；Task 2 已记录的 `source_level_rebuild_reproducible=false` 不变。
7. 候选 B 仍为 `NOT_EVALUABLE`，未实现、未恢复。
8. 未修改任何冻结策略参数，未根据收益选择规则；事实上本任务根本没有产生或查看候选收益。

## 10. Fix round 1：独立排序字段

- 审查 Important：原排序分只写入 `rank_return_120`，与 Donchian 真实 120 日收益的字段语义冲突。
- 修复：`legacy_rank_score` 成为权威排序字段；`rank_return_120` 仅保留为等值兼容别名。新增 `order_legacy_a_candidates` 的“原分降序、代码升序”代码级测试，并增加与 Donchian 120 日收益不同的反例。Task 5 简报已改为强制消费权威字段。
- 真实输出采用从既有冻结信号 Parquet 派生新增等值列，不做全量信号重算。原信号 SHA256 为 `7571903ec0d4daec89cd1a598e535944ddcd70160b1ba844836e1aa07152ab02`，新信号为 `a3b274ad470f0aff9ce268e4cf172a437f864b963eda3703969aecbfde5fc390`；10,776,532 行及键不变，原列逐行哈希差异、排序别名差异均为 0。21,237 个最终入场信号的原公式复核不一致为 0。
- 控制方在实现代理因 API 429 中断后独立运行全项目测试：**159/159 PASS**，并核对当前源码、测试、脚本、信号与摘要哈希。未连接活源库，未运行真实候选账户收益。
