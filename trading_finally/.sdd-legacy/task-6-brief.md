# Task 6：运行候选 A 冻结主版本

> 先读本简报，再读 13 预注册、Task 3 冻结信号摘要、Task 5 策略接口与默认回归报告。候选 B 已 NOT_EVALUABLE，不得调用账户引擎。

## 全局约束

- 所有新内容仅在 `trading_finally/`；SDD 仅 `.sdd-legacy/`；不 git、不派发子代理、不连接活源库。
- 初始资金 20,000 元；仅沪深主板；主窗口 2016-10-13 至 2026-10-29；T 日收盘信号，后续交易日开盘执行。
- 真实 100 股整手、最低佣金、历史税费、10bp 滑点、T+1、公司行动、涨跌停、停牌、退市均使用统一引擎。
- 只运行候选 A 一个冻结主版本；不得改参数、删过滤器、覆盖上一阶段 `results/baseline` 或其他结果。
- 大结果只落盘，测试使用 `PYTHONPATH=src python3 -B -m unittest ...`。

## Files

- Create: `src/trend_system/legacy_runner.py`
- Create: `scripts/run_legacy_candidates.py`
- Test: `tests/test_legacy_integration.py`

## Interfaces

- Produces: `run_legacy_candidate(candidate_id, context, config) -> LegacyRunResult`；结果含状态、六个账户表、绩效和 manifest。
- 输入为 Task 2/3 哈希核验的冻结面板与信号、公司行动、冻结配置和 Task 5 `LegacyPolicy`。候选 A 权威排序只能消费 `legacy_rank_score`/`order_legacy_a_candidates`。
- 面板与信号必须显式按 `(date,code)` 键选择和一对一断言，不得使用位置掩码。冻结主板面板不含 `adjustment_factor` 列；运行前必须按同一冻结口径新增 `adjustment_factor = signal_close / close`（仅当两者均>0，否则缺失并 fail-closed），断言候选入场及持仓涉及键的因子为正且有限，不得连接活源库重算。
- 候选 B 或任何 `NOT_EVALUABLE` 输入必须在调用账户引擎前返回/中止，不得生成虚假空账户结果。

## Steps

1. **写合成集成与不可评价测试**：重复运行输出哈希一致；无 T 日收盘成交；买入严格次一账户日、开盘成交与费用可核验；现金/持股/权益恒等式；`NOT_EVALUABLE` 不调用账户引擎；不得覆盖 `results/baseline` 或上一阶段结果。
2. **运行测试确认 RED，再实现至 GREEN**。
3. **运行真实冻结主版本**：`PYTHONPATH=src python3 -B scripts/run_legacy_candidates.py --config config/legacy_candidates.json`；只运行可评价候选 A，结果写 `results/legacy_candidates/three_of_two_low_vol/`，保存规则/代码/数据哈希、完整六表、绩效和 manifest。
4. **只读核验**：检查现金/持股/权益恒等式、成交时序、行数、窗口、配置和任务3输入哈希；无论盈亏不改规则。真实收益只在冻结主版本完全实现后生成一次，不得反复试参。

## 报告

完整报告写 `.sdd-legacy/task-6-report.md`，列 RED/GREEN、真实运行命令/状态、六表行数与哈希、总收益/CAGR/MDD/年度/暴露/费用/闭合交易、完整性与限制。不得运行稳健性或根据结果新增版本。
