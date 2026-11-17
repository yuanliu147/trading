# Task 3 fix round 1 scoped re-review

- 审查日期：2026-11-04
- 范围：只复核原 Important（候选 A 排序字段语义与下游契约）是否关闭，并检查真实 Parquet 派生结果及是否出现新的越范围改动。
- 方法：读取更新 diff、原审查、报告第 10 节、Task 5 契约、真实摘要和当前源码；使用 PyArrow 元数据及内存 DuckDB 聚焦聚合。未运行 git、未派发子代理、未连接活源库、未运行账户收益。控制方已独立报告 159/159 unittest OK，本轮未重复运行测试。

## Spec Compliance

| 核查项 | 结论 | 证据 |
| --- | --- | --- |
| 权威排序字段 | **已关闭** | `src/trend_system/legacy_a.py:310-315` 先按 `hit_count*2 + volume/volume_ma20*0.2` 计算 `legacy_rank_score`，再令 `rank_return_120=legacy_rank_score`；权威原分与兼容别名已在代码层分离。 |
| 排序顺序 | **符合** | `src/trend_system/legacy_a.py:52-58` 的 `order_legacy_a_candidates` 明确按 `date ASC, legacy_rank_score DESC, code ASC` 稳定排序；不消费 Donchian 收益。`tests/test_legacy_a.py:182-196` 验证原分降序及同分时代码升序。 |
| Donchian 反例 | **符合** | `tests/test_legacy_a.py:175-179` 独立计算 120 日收益，并断言其不等于 `legacy_rank_score`；原分公式和别名等值也在同一测试中验证。 |
| Task 5 下游契约 | **已关闭** | `.sdd-legacy/task-5-brief.md:20,27` 强制 Task 5/6 消费 `legacy_rank_score` 及排序 helper，禁止重算 Donchian 120 日收益；`rank_return_120` 被限定为等值兼容别名，不得作为权威来源。 |
| 真实 Parquet 结构 | **符合** | 当前文件为 10,776,532 行、62 列、11 个 row groups；`legacy_rank_score` 位于最后新增的 index 61，原 `rank_return_120` 位于 index 53，两者均为 double。该结构与“从既有冻结 Parquet 追加等值列、不做全量信号重算”一致。 |
| 行数、主键及原信号计数 | **符合** | 独立聚合得到 rows=10,776,532、unique `(date,code)`=10,776,532；raw=42,999、accepted=30,079、proposed=21,554、entry=21,237，均与修复前结果一致。fix 摘要另记录 `key_mismatches=0`、`original_columns_row_hash_mismatches=0`。 |
| 全量等值别名 | **符合** | 对全部 10,776,532 行独立聚合，`legacy_rank_score IS DISTINCT FROM rank_return_120` 为 **0**。 |
| 21,237 条最终入场原分公式 | **符合** | 对全部 21,237 个 `entry_signal=true` 键重算 `hit_count*2 + volume/volume_ma20*0.2`，公式不一致 **0**；NULL、非有限分数及零/缺失均量窗口均为 **0**。 |
| 真实输出身份 | **符合** | 独立 SHA256：信号 `a3b274ad470f0aff9ce268e4cf172a437f864b963eda3703969aecbfde5fc390`；覆盖审计仍为 `b91342891360a87dccfbe4c7bc82c0caeff7ac62fd0d7dd069fba2fbd493129c`；摘要为 `805204cac8c9ddc0d130bccf001fb9361072aa412427a01ea2b073ad2788bd4b`，均与报告/摘要登记一致。源码、测试和运行脚本哈希也与报告第 1 节一致。 |
| 范围与收益边界 | **符合** | 更新内容限于权威排序字段、等值别名、排序 helper、相应测试/契约和派生产物；未改冻结参数，未恢复候选 B，未加入账户收益路径。摘要仍记录 `returns_run=false`、`account_returns_run=false`。 |

## Strengths

1. 修复不仅增加文档说明，而是建立了代码级权威字段、等值别名和可复用排序接口，直接关闭原语义歧义。
2. 真实文件采用追加列方式保留原产物；新列位于 schema 末尾，主键、信号层级计数和覆盖审计哈希均保持不变。
3. 全量别名等值与 21,237 条最终入场公式均有独立聚合证据，不依赖报告自述。
4. Task 5/6 的下游消费契约已明确，不允许回退到 Donchian `rank_return_120`。

## Issues

### 原 Important 1：排序字段语义冲突

**已关闭。** `legacy_rank_score` 已成为原排序分权威字段，`rank_return_120` 仅作全量等值兼容别名；排序 helper、Donchian 反例及 Task 5/6 消费契约均已落实。真实 Parquet 的追加列结构、行数、主键、原信号计数、公式和哈希均吻合。

### 新增问题

**无新增 Critical、Important 或 Minor。**

## Assessment

### Task quality

**Approved**（仅限 Task 3 fix round 1 的排序字段语义修复及其真实产物复核范围）。

原 Important 已完整关闭，没有未解决 Critical/Important。该结论不构成候选 A 的收益、稳健性或实盘授权；候选 B 仍不得恢复，账户收益不在本轮审查范围内。
