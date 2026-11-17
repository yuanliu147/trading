# Task 2：构建全 A 上下文与历史行业可行性面板

> 先读本简报，再读 `13_预注册_旧候选统一复核.md` 和 Task 1 生成的 `src/trend_system/legacy_config.py`。本任务只做数据构建与可行性审计，不实现候选信号、不运行账户收益。

## 前置状态

- Task 1 已冻结两个候选。候选 A `three_of_two_low_vol` 为 `PENDING_DATA_GATE`；候选 B `early_trend_breadth` 因 12 项母版公式/溯源不能唯一还原而为 `NOT_EVALUABLE`。不得通过本任务补数据后擅自改写 B 的规则状态。
- 两个数据库只允许严格只读查询；不得同步、迁移、修复、建表或写库。其文件路径、字节数和 SHA256 已登记在 13 预注册。
- 全 A 数据仅用于原规则要求的横截面股性、市场宽度和行业上下文；实际可交易宇宙仍为沪深主板。

## 全局约束

- 所有新内容仅在 `trading_finally/`，SDD 文件仅 `.sdd-legacy/`；不执行任何 git 操作，不派发子代理。
- 主窗口 2016-10-13 至 2026-10-29，指标可使用此前历史热身。
- 实际未复权价用于成交、整手、真实价格过滤和费用；点时复权仅用于信号与收益。
- 不得使用当前名单、当前 ST 或当前行业回填历史；缺失必需字段必须 fail-closed。
- 大结果只写 Parquet/CSV/JSON，不回显全表；测试使用 `PYTHONPATH=src python3 -B -m unittest ...`。
- 用户新标准：MDD≤10% 且其他门槛通过为 UNREJECTED，10%<MDD≤20% 为较高风险 BOUNDARY，>20% 否决。本任务不运行或裁决收益。

## Files

- Create: `src/trend_system/legacy_data.py`
- Create: `scripts/build_legacy_context.py`
- Create: `sql/legacy_context.sql`
- Test: `tests/test_legacy_data.py`

## Interfaces

- Consumes: `LegacyResearchConfig`、只读源数据库和既有主面板。
- Produces: `build_legacy_context(config) -> LegacyContextResult`；`data/legacy_full_a_panel.parquet`、`data/legacy_industry_panel.parquet`、`results/legacy_context_audit.json`。

## Steps

1. **写点时和覆盖率失败测试**：覆盖未来行业记录不可用于截断日、当前行业不能回填、每日分母固定、98.99% 与 99% 边界、实际候选字段缺失必失败；缺少有效日期行业源时候选 B 返回 `NOT_EVALUABLE`。
2. **运行测试确认接口缺失 RED**。
3. **实现只读上下文构建**：全 A 仅作排名/宽度/行业上下文，执行宇宙仍为主板；保留原始与点时复权 OHLC、量额、状态、上市退市日期和行业有效起止日。若本地只有当前行业快照，保留审计证据并判 B 不可评价，不构造伪历史。
4. **真实构建并验证**：检查主键、日期、行数、覆盖率、数据库 SHA256；候选 A/B 分别给出 `EVALUABLE` 或 `NOT_EVALUABLE`，大表只落盘。
5. 数据闸门严格按 13 预注册：全 A 上下文逐日覆盖点时活跃宇宙至少 99%；实际候选入场所需字段覆盖 100%；历史行业需有效日期且逐日覆盖至少 99%、实际候选入场覆盖 100%；不足即对应候选不可评价。执行冲突沿用既有定义，辅助复权差异只披露，不冒充执行冲突。

## 报告

完整报告写 `.sdd-legacy/task-2-report.md`，列 RED/GREEN、只读 SQL/输入输出路径、行数/窗口/主键、逐日覆盖分母和边界、数据库哈希核验、A/B 可行性、限制和测试。不得运行候选收益或改规则。
