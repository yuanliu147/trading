# 旧候选统一复核最终独立审查报告

- 审查日期：2026-11-05
- 审查范围：`11_设计_旧候选统一复核.md`、`12_实施计划_旧候选统一复核.md`、`13_预注册_旧候选统一复核.md`、`14_预注册_候选B恢复.md`、`.sdd-legacy/progress.md`、`.sdd-legacy/final-legacy-review.diff`、`93—97`、主版本/稳健性/信号结果与冻结输入。
- 审查方式：只读代码、文档、结果和哈希；未运行 git、未修改任何结果、未派发子代理、未重跑完整矩阵；仅做聚焦数值/哈希核验。
- 结论：**Needs fixes（不批准 Approved）**。存在 2 项 Critical、6 项 Important；已发现的问题中，B 候选的已冻结主结果不能直接作为最终 B 裁决依据，A 的 REJECTED 裁决独立成立。

## 1. 总体结论

1. **A 的机械裁决独立成立：REJECTED。** 主窗口 -79.2002%、MDD 81.3216%、滚动三年 0/85、双倍费用 -87.0519%、随机排序已完成路径全负、剔除最佳年份后剩余年份为负；数据/执行完整性与盘后可复现两项通过，其余门槛失败。A 没有 ST 买入，未发现改变裁决的前视、T+1、整手、现金或复权锚错误。
2. **B 的恢复来源本身合法，但当前 B 结果不合格。** `10b_构建全A面板_含退市.py` 确实是 `30b` 所加载 `全A面板_2016_2026_含退市.pkl` 的唯一写入者，时间/版本早于母版，不是后续优化版；12 项公式可在 `10b` 中逐项还原，行业 CSV 带 `start_date`、无空值/重复键，99.7325% 的逐日行业覆盖和 100% 入场行业覆盖经独立复算成立。**但 B 实际买入了 46 只入场时已经 ST 的主板股票，并在次日以 `st_exit` 卖出，形成 46 笔一日往返；B 的机器配置/上下文审计又仍声明 B 为 `NOT_EVALUABLE`。** 因此 B 的主结果、双倍费用和随机路径均被污染，不能在修正并重跑前作为最终 B 结论。
3. **`97_停止边界与后续.md` 的停止方向正确，但尚不是符合设计 §8 的完整停止交付。** 它正确停止新策略探索、不生成日报/持仓/前瞻台账、不提供参数回调建议；但缺少设计要求的规模/波动状态分组、滚动窗口分布和共同失败原因分析，且仍沿用 A“前 5 条”随机路径写法。
4. 未发现日报、持仓清单、前瞻台账、实盘建议或连接券商的交付；默认 Donchian 基线与 A 主结果的复现证据成立。

## 2. Critical

### C1. B 买入了 46 只入场时已 ST 的股票，随后次日 `st_exit`，污染 B 主结果与失败归因

- 证据：
  - `src/trend_system/legacy_b.py:133-148` 的 `valid` 只包含趋势、波动、宽度、行业、流动性等条件，没有 `is_st==0`；`:179-183` 的 `proposed/accepted` 与 `:209-215` 的 `tradeable` 也只检查 `trade_status/volume/amount/execution_conflict`，不检查 ST。
  - `src/trend_system/legacy_b_policy.py:137-141` 在每次 `exit_reason` 中只要当前 `bar.is_st==1` 就返回 `st_exit`；入场日收盘后即触发，次日开盘卖出。`src/trend_system/legacy_b_runner.py:29-45` 的账户检查也没有“ST 不可买入”断言。
  - 实测 `results/legacy_candidates/early_trend_breadth/round_trips.parquet`：451 笔闭合往返中 **46 笔**入场日 `is_st=1`，46 笔全部以 `st_exit` 退出；45 笔持有 1 个交易日、1 笔持有 2 个交易日；净盈亏 **-611.03 元**、费用 **495.01 元**、毛盈亏 **-116.02 元**；占全部往返 **10.2%**、占总费用 **9.9%**。对应买入记录见 `results/legacy_candidates/early_trend_breadth/trades.parquet`（46 笔 ST 买入，日期 2019-12-09 至 2026-09-01）。
  - `96_候选B失败与场景.md:14` 把该 46 笔列为 `st_exit` 风险退出；`97_停止边界与后续.md:32` 同样把 `st_exit 46 笔净 -611.03` 当作退出原因。实际上它们是“买入时已 ST、次日立即卖出”的入场资格缺陷，不是“持仓变为 ST”的风险退出。
- 影响：B 的主窗口、双倍费用、随机路径和 `96/97` 的失败归因均包含不合格/不应发生的交易；当前 B 的 `REJECTED` 结论方向大概率仍成立（B 主收益 -13.87%，仅移除这些交易最多改善约 3 个百分点，毛优势只有 +852.62 元），但**当前 B 数值不是冻结规则的忠实执行结果，不能作为最终裁决与归因证据**。设计 §4.3/§5 与 `13` 的 ST 规则写的是“持仓变为 ST”的新增风险退出；A 也明确将 `is_st==0` 作为最终入场条件（`src/trend_system/legacy_a.py:323`），B 与 A 的资格口径不一致。
- 最小修法：在 B 入场资格层加入 `is_st==0`（信号生成/审计与 `entry_state` 双重 fail-closed），把 ST 从拟入场分母中显式记录并加入 `data_execution_integrity` 断言；不得删除 ST 退出规则。随后重跑 B 主版本、双倍费用、随机路径，重写 `95/96/97`。若产品口径坚持允许 ST 买入，则必须改为记录“入场前 ST 状态”，只在持仓由非 ST 变为 ST 时触发 `st_exit`，并同样重跑。

### C2. B 已在“机器配置仍为 NOT_EVALUABLE”的状态下运行，且没有按 `14` 另立恢复配置/审计；恢复治理是 fail-open

- 证据：
  - `14_预注册_候选B恢复.md:6` 明确写“本恢复另立配置与预注册，不回溯修改 A”；但 `config/` 下只有 `baseline.json` 与 `legacy_candidates.json`，没有 B 恢复配置。
  - `config/legacy_candidates.json:76` 的 B `evaluation_status` 仍是 `"NOT_EVALUABLE"`，`:83-95` 仍列 12 个 `not_evaluable_fields`（含 `amount_ranking_tie_break`）；`results/legacy_context_audit.json:58929-58949` 仍写 B `NOT_EVALUABLE`、`restored:false`。
  - `results/legacy_b_summary.json:3` 却写 `"status":"EVALUABLE"`，其 `inputs.config` 哈希 `09096f...` 正指向上述仍声明 B 不可评价的配置。
  - `scripts/build_legacy_b_signals.py:11,20,49-58` 只用数据覆盖条件计算 `evaluable`，不检查 `not_evaluable_fields`、不校验 `10b`/行业源哈希；`src/trend_system/legacy_b_runner.py:49-53` 只检查 B 摘要 `status`，不检查配置状态。也就是说，机器配置的 `NOT_EVALUABLE` 硬闸门被绕过，B 的“恢复”只存在于 Markdown 预注册和临时数据审计中。
- 影响：B 的恢复缺少可冻结、可机器校验的授权记录；未来任何公式/阈值改动都可在配置仍写 12 项不可还原的情况下被当作 `EVALUABLE` 运行。B 主结果的配置来源和失败关闭性不满足 `13 §5` 的“唯一机器配置”要求。
- 最小修法：新增独立的 B 恢复配置/审计（例如 `config/legacy_b_recovery.json` 或等价的冻结 recovery block），写入 B 的 `EVALUABLE`、空 `not_evaluable_fields`、`10b`/行业 CSV/30b 哈希和数据闸门摘要；让 B 构建/运行在配置未解除 `NOT_EVALUABLE` 时直接拒绝；更新 `results/legacy_context_audit.json` 或另建 B recovery audit，并把配置哈希写进 `results/legacy_b_summary.json` 与 B `run_manifest.json`。若配置只作展示、不进入运行，需在 `14` 中明确声明并给出哈希，不能继续让两个官方状态互相矛盾。

## 3. Important

### I1. A 稳健性摘要只记录 5/20 条随机路径，实际磁盘有 12 条已完成，且 `early_stop_reason` 的最大正路径数不是 12 条口径

- 证据：`results/legacy_robustness/summary.json:44-45` 写 `paths_completed=5`、`maximum possible positives 15 < 16`；但 `results/legacy_robustness/random_order/path_00..path_11/performance_summary.json` 实际有 **12 条**已完成路径，独立检查 12 条全部为负（-0.4313 至 -0.8978）。`93_旧候选复核结论.md:9-10` 一边写“0/5 条已完成”，一边写“结果目录实际保留 12 条已完成随机路径摘要”；`97_停止边界与后续.md:23` 仍写“已完成的前 5 条”。进度台账自己的 Ruling 也写“12 completed A paths all negative ... maximum 8”。
- 成因：`src/trend_system/legacy_robustness.py:310-321` 在检查已有 `path_XX/performance_summary.json` **之前**就执行 `completed - random_positive > 4` 的提前停止判断；断点续跑时只装载 path_00..04 就停止，未把 path_05..11 计入 `random_records`。
- 影响：最终门槛结论不变（5 条负已足以证明最多 15<16；12 条负更强，最多 8），但 `paths_completed` 字段、`early_stop_reason` 的最大值、93/97 的文字均不是对磁盘完成路径的如实记录，违反任务书“`paths_completed` 与 `early_stop_reason` 应如实记录”的要求。B 的 6 条与磁盘一致，无此问题。
- 最小修法：在提前停止判断前先按 path 顺序装载所有已存在的已完成摘要；或将字段改名为 `paths_considered` 并新增 `paths_on_disk`/`all_completed_paths_negative`。用 12 条重算 `early_stop_reason`（最多 8 条正），同步更新 `93/97/README`。无需重跑路径，只需修正装载/汇总逻辑并重生成摘要。

### I2. B 的 `run_manifest` 不含代码/配置哈希，且没有类似 A 的当前代码重跑证据；B 主结果无法用哈希绑定到实现

- 证据：`src/trend_system/legacy_b_runner.py:84-127` 构造的 manifest 只有 `signals`、`candidate_b_summary`、`main_panel`、`corporate_actions`、`policy`、`checks`、`summary`，没有 A manifest 中的 `code` 和 `config` 哈希。A 的 manifest 含 `code`（`broker.py/legacy_policy.py/legacy_a.py/legacy_runner.py/execution.py`）和 `config`，并有 `a_reproduction_addendum.json` 证明当前代码在 `/tmp/legacy-a-repro` 的 7 类核心输出（另含 `annual_and_rolling.csv`，本审查已核对 8/8）与冻结值一致。
- B 源码 mtime（`legacy_b_policy.py` 00:04、`legacy_b_runner.py` 00:04、`broker.py` 00:03，早于 B 主运行 00:09）只能作为旁证，不能替代代码哈希；B 稳健性代码 01:49 的改动也未在 B 主 manifest 中体现。
- 影响：B 主结果、双倍费用和随机路径不能仅凭现有冻结摘要证明与当前代码行为等价；这直接违反“输入/代码/结果哈希”的复现要求。
- 最小修法：给 B manifest 增加 `config` 与代码哈希（`legacy_b.py`、`legacy_b_policy.py`、`legacy_b_runner.py`、`broker.py`、`execution.py`、`legacy_runner.py`），在临时根目录用当前代码重跑 B 主版本并逐项比对新旧 7 类核心输出 SHA256（若因 C1/C2 需要重跑，则把修正后的运行作为新冻结版本）；把结果写入 B addendum 或 `task-b-main-report`。

### I3. `93_旧候选复核结论.md` 仍称 B `NOT_EVALUABLE`、未运行账户收益，与 95/97/README/B 摘要矛盾

- 证据：`93_旧候选复核结论.md:42` 写“候选 B 因母版 12 项公式/溯源无法唯一还原且无带有效日期的历史行业来源，保持 `NOT_EVALUABLE`，未运行账户收益。”；而 `95_候选B复核结论.md`、`97`、`README` 和 `results/legacy_b_summary.json` 均为 B 已恢复、已运行并 `REJECTED`。该句由 `src/trend_system/legacy_robustness.py:469` 在 A 报告写入器中硬编码，重跑 A 稳健性会再次生成同样的错误句子。
- 影响：最终交付物内部自相矛盾；读者无法判断 B 是否运行。
- 最小修法：删除该句或改为“候选 B 按 `14` 恢复并另行复核，见 95/96”；修正 `legacy_robustness.py` 的 B 段落，避免后续重跑再生。

### I4. `97/94/96` 缺少设计 §8 要求的部分失败场景归因

- 证据：`11_设计_旧候选统一复核.md:153-162` 要求两个候选失败后交付：信号/选择/仓位/退出/费用/执行各自贡献、旧报告口径差，以及按年度、滚动窗口、指数趋势/震荡/下跌、大小盘和波动状态的**事后分组**，并明确哪些环境不适用/样本不足。当前 `93/95` 有年度；`94/96` 只有“上证 above/below MA20”和退出原因；`97:39-49` 只有旧报告差异；没有滚动窗口分布、大小盘分组、波动状态分组，也没有共同失败原因分析。`94/96` 末尾甚至只写“分组仅作描述”，未给出设计要求的全部分组。
- 影响：停止边界本身正确，但失败归因交付不完整；无法回答“共同失败原因、各候选失败环节、适用/不适用场景”。
- 最小修法：只基于已冻结主路径，补充年度/滚动三年分布、指数趋势/震荡/下跌、大小盘、波动状态的分组统计（描述性，不作为新闸门），并增加 A/B 共同失败原因段落；不得新增消融或改参回测。

### I5. `.sdd-legacy/task-8-report.md` 缺失，Task 8 的正式交付/未交付记录和复现哈希没有落盘

- 证据：`.sdd-legacy/task-8-brief.md:52-54` 明确要求“完整报告写 `.sdd-legacy/task-8-report.md`，列状态、交付/未交付文件、测试、复现哈希、限制和最终停止/模拟边界”；该文件不存在。`progress.md` 末尾只记到“97 finalized; full suite 187/187 PASS; final review dispatched”，没有 Task 8 报告/交付清单。`.sdd-legacy/task-7-report.md` 也不存在（Task 7 无显式报告要求，列为 Minor）。
- 影响：最终停止/模拟边界缺少一份完整的交付审计记录；复现哈希和“未生成哪些文件”的证明只散落在 97/README/测试中。
- 最小修法：补写 `task-8-report.md`，纳入最终状态、未交付文件证明、187/187 测试、A/B 主结果与哈希、停止边界和复现命令；在 progress 追加 Task 8 完成记录。

### I6. A 在入场层排除了 ST（317 个拟入场信号），但该规则不在绑定的 `13` 预注册进入条件中

- 证据：`src/trend_system/legacy_a.py:323` 用 `result["is_st"].eq(0)` 作为最终 `entry_signal` 条件；`tests/test_legacy_a.py:296` 覆盖 ST 拒绝；Task 3 报告记录 317 个 ST 拟入场被拒、最终 ST 入场 0。`12_实施计划_旧候选统一复核.md:89` 要求断言 ST 入场为 0，但 `13_预注册_旧候选统一复核.md:103` 只规定“ST 叠加退出”，`13` 的 A 进入条件（§3.1）没有 ST 入场排除；`11` §4.2 也没有。ST 拒绝先于最终入场，但 `_apply_cooldown` 在 ST 过滤之前运行，ST 信号仍会消耗 5 日冷却，可能压掉后续非 ST 信号。
- 影响：A 的结果不是完全按 `13` 冻结进入规则生成；虽然 A 的 REJECTED 裁决不受 317/21,554 的显著影响，但规则可复现性与 A/B 资格一致性受影响。
- 最小修法：若“仅主板普通 A 股”本意排除 ST，在 `13` §3.1 明确写入 ST 入场排除及其与冷却的先后；若不允许排除，删除该过滤并重跑 A。B 也必须使用同一口径（见 C1）。

## 4. Minor

1. **B gap 过滤使用执行日原始 `open/prev_close`，而 `14` §3 与 `10b:343` 定义 `NEXT_GAP = O.shift(-1)/C - 1` 用的是点时复权 OHLC**（`src/trend_system/legacy_b_policy.py:105`）。盘后限价语义本身可执行、未偷用次日信息；实测 B 主路径 0 次 `gap_filter` 取消，唯一除权日入场 `603701.SH 2021-06-23` 的原始/复权 gap 均为 0.0044，无观察到的结果差异，但公式等价性未被证明。建议补一个除权日反例或明确披露“原始价限价语义”与 `10b` 的差异。
2. **`14` §1 声称 12 项全部唯一还原，但 `amount_ranking_tie_break` 仍无母版定义**（`config/legacy_candidates.json:95`；`14:50` 自行规定代码升序）。该 tie-break 已披露，不是隐藏猜测，但“12 项全部唯一还原”的措辞不准确，应改为“11 项公式 + 1 项已披露实现层 tie-break”。
3. **B heat 的 `risk_ps` 使用了含滑点的 `fill.price`（`legacy_b_policy.py:117`），而 sizing 的 `risk_ps` 使用 `open_price`（`:68`），`14:51` 写的是 `risk_ps=entry*sd_size`**。差异只有约 0.1%，且 `heat+0.005>0.0301` 是窄边界；未发现其单独改变 B 裁决，但建议统一为 open 价或披露。
4. **`97:21-22` 的 A 归因措辞“主要亏损来自固定止损被频繁触发”是退出原因会计，不是因果识别**；止损退出记录了 -24,291 元毛亏损（115 笔），但报告没有无止损反事实，不能证明是“止损规则”造成亏损。建议改为“止损类退出记录了主要已实现亏损”，并明确不是反事实归因。
5. **B 与旧母版信号对照只报告“旧主板信号复现 97.9%”，未披露新实现独有的 6,008 个信号和 Jaccard 0.679**（`.sdd-legacy/b_signal_comparison.json`；`97:37`、`task-b-main-report:49-55`）。这不影响 B 的负结果，但“高度一致”应同时列出双向差异和可能来源。
6. **随机排序日志没有 seed 列**：`random_order_log.csv` 只有 `path_id/date/code/random_rank`；种子来自 `config.random_seed=20261101` 和代码中的字符串种子。日志的排列和路径唯一性均已验证（A 12 条 × 1055 日、B 6 条 × 857 日；path_00/01 同日顺序不同），但建议在日志或 `input_manifest.json` 记录 seed，便于脱离代码审计。
7. **A/B 稳健性摘要和 `input_manifest.json` 没有对 per-path 输出/最终 `summary.json` 做 SHA256**；主版本有 freeze，稳健性只有摘要。建议在 `input_manifest` 增加路径摘要哈希或输出 manifest。
8. **`evaluate_legacy_gates(None)` 抛 `AttributeError` 而非返回 `REJECTED`**（`legacy_robustness.py:47`）；不会误放行，但 fail-closed 接口可更明确。建议先做 `isinstance(dict)` 检查。
9. **`97` 的旧报告口径差异对 A 使用了泛化表述“旧报告使用全 A（含创业板/科创板）”**；旧 A 源码实际使用 2025 中证500 快照/缓存（`backtest.py` 头部与 `ZZ500`），不是全 A。该差异不影响 B 的结论，但应按候选分别陈述。
10. **`task-7-report.md` 缺失**（Task 7 计划没有显式报告文件要求，但 SDD 交付一般应有）；列为 Minor。

## 5. A/B 机械裁决独立复核

### A：REJECTED（独立成立）

| 门槛 | 值/证据 | 结论 |
|---|---|---|
| 主窗口净收益 | -0.7920015；final 4,159.97 | 失败 |
| 滚动三年 | 85 个月末窗口，0 正（0.0） | 失败 |
| 双倍费用+20bp | -0.8705185；MDD 0.8835142 | 失败 |
| 随机排序 ≥16 正 | 摘要 0/5；磁盘 12 条全负（-0.4313 至 -0.8978） | 失败（数学上 5 负已足够，12 负更强） |
| 剔除最佳年份 | 最佳 2024 +898.48；剩余 -16,738.51 | 失败 |
| 数据/执行完整性 | 主 manifest checks 全 true；entry coverage 100% | 通过 |
| 完全盘后可复现 | `no_signal_day_trade` + execution integrity | 通过 |
| MDD≤20% | 0.8132163 | 失败 |
| 最终状态 | `evaluate_legacy_gates` 独立返回 `REJECTED` | **REJECTED** |

A 的 417 笔闭合往返、win rate 0.2446、2010—2015 压力 -0.302305/MDD 0.423616、年度 PnL 合计 -15,840.03（= final-20,000）均核对一致。A 买入 417 笔全部主板、全部 100 股整手、全部 T+1 次日开盘（日历差 1）；`max_hold` 退出日历差 40，与“入场日计第 1 日、第 40 日收盘确认、次日开盘卖出”一致。A 无 ST 买入。

### B：当前摘要计算为 REJECTED，但结果需修正后重跑

| 门槛 | 值/证据 | 结论 |
|---|---|---|
| 主窗口净收益 | -0.138745；final 17,225.10 | 失败 |
| 滚动三年 | 85 窗口，38 正（0.4470588 <0.5） | 失败 |
| 双倍费用+20bp | -0.4414945；MDD 0.4414945 | 失败 |
| 随机排序 ≥16 正 | 1/6 正（path_04 +0.01924）；5 负，max 15<16 | 失败 |
| 剔除最佳年份 | 最佳 2021 +1,192.88；剩余 -3,967.78 | 失败 |
| 数据/执行完整性 | manifest checks 全 true；但未检查 ST 买入（C1） | 形式上通过，实质不合格 |
| 完全盘后可复现 | execution integrity + trades_after_signal_date | 通过 |
| MDD≤20% | 0.166678 | 通过 |
| 最终状态 | 当前 `evaluate_legacy_gates` 返回 `REJECTED` | **REJECTED（待 C1/C2 修正后重算）** |

B 的 451 笔闭合往返、win rate 0.4102、年度 PnL 合计 -2,774.90、`time_exit` 402 笔日历差 20、`catastrophe_stop` 3 笔、`st_exit` 46 笔（缺陷，见 C1）均核对一致。B 买入 451 笔全部主板代码前缀（无 300/301/688/8/4/BJ）、全部 100 股整手、全部 T+1 次日开盘。B 的 46 笔 ST 买入使主结果、费用、双倍费用和随机路径均不可直接作为最终证据；从数量级看，修正后 B 大概率仍为 `REJECTED`（当前净亏 4,141.68 元，46 笔 ST 仅贡献约 611 元净改善），但必须重跑后才能定稿。

### 机械门槛与停止逻辑

- `evaluate_legacy_gates` 要求 status 严格为 `EVALUABLE`、8 个 gate 全部为 bool 且 true、MDD 有限非负；未知/缺失/非 bool 返回 `REJECTED`，`NOT_EVALUABLE` 单独返回；未发现 fail-open。仅 `None` 入参会抛异常（Minor）。
- MDD 状态机 A >20% -> REJECTED；B 16.67% 仅表示“若其他门槛全过则 BOUNDARY”，B 其他门槛失败，仍 REJECTED。B 的 `drawdown_within_limit` 通过、其余 5 个收益类 gate 失败，表/摘要一致。
- 双倍费用通过 `execution.transaction_costs` 的 `multiplier=2.0` 同时翻倍佣金（含最低佣金）、印花税、过户费，并把滑点设为 20bp；A/B 双倍成本复算 MDD 与摘要一致，未发现只翻倍部分费种。
- 随机提前停止的数学条件 `completed - positive > 4` 正确；A/B 的 `random_order_log.csv` 每个日期都是完整排列、path 间顺序不同；A 的问题是“已装载路径”而非数学条件，见 I1。

## 6. `97_停止边界与后续.md` 是否符合设计停止条件

- **符合的部分**：A/B 均为 REJECTED；不进入 30 笔纯模拟；不生成盘后日报、持仓清单、前瞻台账或实盘建议；不通过改持有期/止损/价格带/成交额/仓位/市场闸门寻找偶然正收益；明确停止新策略探索，并要求新假设另立预注册。`tests/test_legacy_delivery.py` 也断言不存在 `97_纯模拟观察协议.md`、`legacy_daily.py`、`legacy_daily_plan.py`、`state/legacy_forward_ledger.csv`，未发现禁止交付。
- **不符合/未完成的部分**：
  1. 设计 §8 要求的“年度、滚动窗口、指数趋势/震荡/下跌、大小盘、波动状态”分组未完整交付（I4）；`94/96` 只有指数 above/below MA20 和退出原因。
  2. A 的随机路径行仍写“前 5 条”，与 12 条已完成路径及 93 的说明不一致（I1）；97 应写 12 条全负、最大 8 条正。
  3. B 的 `st_exit 46 笔 -611.03` 在 C1 修正后应重算；当前 97 把它作为风险退出归因，未披露“入场即 ST”的资格缺陷。
  4. `97` 的旧报告差异对 A 的“全 A”表述不准确（Minor）。
- 结论：**97 的“停止/不模拟/不调参”方向符合设计停止边界，但作为最终停止交付尚不完整；在 C1/C2 和 I1/I4 修正后需要重生成/重写相关段落，才能视为符合设计 §8 的完整停止报告。**

## 7. 已独立核验的关键数值与哈希

- A：final 4,159.97；total -0.7920015；CAGR -0.1447478；MDD 0.8132163；RT 417；win 0.2446；年度 PnL 合计 -15,840.03；双倍 -0.8705185/MDD 0.8835142；压力 -0.302305/MDD 0.4236156/final 13,953.9；随机磁盘 12 条全负。
- B：final 17,225.10；total -0.138745；CAGR -0.0147633；MDD 0.166678；RT 451；win 0.4102；年度 PnL 合计 -2,774.90；双倍 -0.4414945/MDD 0.4414945；随机 1/6（path_04 +0.0192435）；剩余年份 -3,967.78；ST 46 笔/净 -611.03/费用 495.01/毛 -116.02，45 笔持有 1 日、1 笔 2 日。
- 滚动三年：A 0/85；B 38/85=0.4470588；均按月末窗口核对。
- T+1/持有：A/B 全部买入日历差 1；A `max_hold` 日历差 40；B `time_exit` 日历差 20；B `st_exit` 日历差 1（缺陷证据）。
- 行业覆盖：独立复算 2016-10-13 至 2026-10-29 逐日 active 宇宙，最低 **0.9973253**，0 天低于 99%；B 最终 116,340 个入场行业缺失 0。
- B 信号对照：旧母版 156,370；主板 112,654；复现 110,332（97.9%）；非主板 43,716（28.0%）；新实现独有 6,008；Jaccard 0.6794763。
- 哈希核对（全部 MATCH）：
  - A 信号 `a3b274ad470f0aff9ce268e4cf172a437f864b963eda3703969aecbfde5fc390`；B 信号 `b1e699a4f6d5b5f9411739de944c795d194124e15607f00b9f3912882d88de5a`。
  - `data/legacy_full_a_panel_completed.parquet` `12e92da7600f59c72851da194ac13fca237d593d1cb8c2a0c329e4c28c23d9fb`；`data/main_panel.parquet` `6a51ed0e7958aa1549dcf156e4387badcfced0236fd812d2bc1fd894aa3acc1a`；`data/index_panel.parquet` `8b8763e0a552894eee84407f1b0c5e6efa4ff7eb8640ddb149b637539bef387a`；`data/corporate_actions.parquet` `f7a924a63091af282b8d37ed2120ae9bae0598846d5d414edc7191a9769bb575`。
  - A 上下文审计 `d2e375ae169f078522d8b979dbea13f457b678d5325d3c77f2a8877d37d56c98`；A 摘要 `805204cac8c9ddc0d130bccf001fb9361072aa412427a01ea2b073ad2788bd4b`；A 权威源码 `73b5eb05…`/`52400a17…`；配置 `09096fc0…`。
  - B 行业 CSV `3ebc1a6c22c1b17f137470bb0c040c31b055b5d805fe29f4b55ce53d8ee62872`；10b `d45892c9…`；30b `8406120a…`；B 摘要 `98fde0d6…`。
  - `.sdd-legacy/main_freeze.json` 9 个文件、`.sdd-legacy/b_main_freeze.json` 9 个文件逐项字节/哈希 MATCH。
  - A 临时重跑 `/tmp/legacy-a-repro/results/legacy_candidates/three_of_two_low_vol` 的 `daily_account/daily_positions/trades/orders/round_trips/action_log/performance_summary/annual_and_rolling` 8 项哈希全部与 `main_freeze.json` 一致；`a_reproduction_addendum.json` 的当前 broker/legacy_policy/legacy_runner 哈希也实测一致。历史 run_manifest 的 broker/policy 哈希与当前不同，属于已披露的 behavior-preserving 代码演进。
  - 默认 Donchian 基线 `results/baseline/daily_account.parquet` `3f29fcfc…`、`round_trips.parquet` `3a9d3538…`、`performance_summary.json` `fd9ba012…` 与 Task 5 报告一致；默认路径核心输出未变。
- 交付边界：未发现 `97_纯模拟观察协议.md`、`legacy_daily.py`、`legacy_daily_plan.py`、`state/legacy_forward_ledger.csv`、`daily_plan.csv`、`live_orders.csv`；无券商连接。

## 8. 最小修复与复审批次

1. **C1（阻断）**：B 入场加 `is_st==0`（或改为 ST 状态转换检测），加入 B 审计/`data_execution_integrity` 的 ST/主板断言；重跑 B 主版本、双倍费用、20 随机路径；更新 `95/96/97` 和 B 冻结/摘要。预期仍 REJECTED，但必须以重跑数值为准。
2. **C2（阻断）**：建立 B 恢复机器配置/审计，解除 `NOT_EVALUABLE`、清空 `not_evaluable_fields`、写入 10b/行业哈希；B 构建/运行在配置未恢复时必须拒绝；更新 context audit/B summary/B manifest。
3. **I1**：修正 A 随机汇总的断点装载顺序，按 12 条重生成 `paths_completed=12`/最大 8；更新 93/97/README。
4. **I2**：给 B manifest 增加 config/code 哈希，并用当前代码做一次临时根目录 B 主版本重跑哈希对照（或把 C1 重跑作为新冻结）。
5. **I3/I4/I5/I6**：删除/修正 93 的 B 旧句；补 97/94/96 的规模/波动/滚动/共同失败分组；补 `task-8-report.md`；把 A 的 ST 入场规则写入 `13` 或删除。
6. Minor 按第 4 节处理；完成后重跑 187/187 测试、重生成 A/B 摘要与 93—97，再做 scoped re-review。

在 C1/C2 未解决前，本阶段不能标记 Approved；I1/I4 未完成前，`97` 不能视为符合设计 §8 的完整停止交付。
