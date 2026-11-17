# Task 1：提取唯一规则并冻结预注册与配置

> 本简报是唯一任务要求来源。设计权威：`/Users/admin/WorkSpace/trading/trading_finally/11_设计_旧候选统一复核.md`。只执行本任务，不实现数据构建、策略信号或账户回测。

## 全局约束

- 所有新内容仅在 `trading_finally/`；本阶段 SDD 工作区为 `trading_finally/.sdd-legacy/`。
- 不执行 git add/commit/push/merge；旧研究及同花顺、baostock 数据库严格只读，不同步、不迁移、不修复、不下载外部数据、不连接券商。
- 主窗口固定 2016-10-13 至 2026-10-29；2 万元、仅主板、T 日收盘确认、后续交易日执行。
- 不复权实际价用于成交、整手、真实价格过滤和费用；点时复权仅用于信号与收益。
- 每个候选只有一个主版本；禁止网格、参数邻域搜索、从后续版本择优或删除不通过的过滤器。
- 测试统一 `PYTHONPATH=src python3 -B -m unittest ...`。
- 用户于 2026-11-03 明确：最终目标是在硬约束内尽量高的扣费后年化收益，同时关注收益回撤比和跨期稳定性；最大回撤硬上限改为 20%。主路径 MDD≤10% 且其他门槛通过为 `UNREJECTED`；10%<MDD≤20% 且其他门槛通过为较高风险 `BOUNDARY`，两者均可进入 30 笔纯模拟；MDD>20% 否决。高年化目标不授权调参或按结果择优。

## Files

- Create: `13_预注册_旧候选统一复核.md`
- Create: `config/legacy_candidates.json`
- Create: `src/trend_system/legacy_config.py`
- Test: `tests/test_legacy_config.py`

## Interfaces

- Produces: `load_legacy_config(path) -> LegacyResearchConfig`；不可变配置含两个候选、主窗口、成本、随机种子和数据验收阈值。

## Steps

1. **写配置失败测试**：断言两个且仅两个候选 `three_of_two_low_vol`、`early_trend_breadth`；初始资金 20000、种子 20261101；拒绝未知字段、非标准 JSON、额外参数和 CLI/环境覆盖。
2. **运行测试确认模块缺失 RED**。
3. **从权威旧源码提取并冻结规则**：候选 A 以 `/Users/admin/WorkSpace/trading/2026-10-18_最优策略_原版三法取二_低波活跃/04_复现与代码/backtest.py`、同目录 `股性分类.py` 为准；候选 B 以 `/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/02_代码/30b_组合回测_含退市.py` 为准。逐项记录 `low10` 是否含当日、持有日/停牌计数、退出优先级、趋势年龄、宽度、ATR、DD60、风险股数、heat、C>=2 和所有源文件 SHA256；不能唯一还原的规则标 `NOT_EVALUABLE`，不得猜测。
4. 数据闸门固定：全 A 上下文沿用旧源码的沪深 A 股范围并排除 BJ，逐日覆盖点时活跃宇宙至少 99%；实际候选入场所需字段覆盖 100%；历史行业需有效日期且逐日覆盖至少 99%、实际候选入场覆盖 100%；不足即对应候选不可评价。执行冲突沿用既有定义，辅助复权差异只披露不冒充执行冲突。
5. **实现严格加载并跑通测试**：配置值与预注册逐项一致；记录任务报告，不做 git 操作。

## 报告要求

完整报告写入 `/Users/admin/WorkSpace/trading/trading_finally/.sdd-legacy/task-1-report.md`。报告需列 RED/GREEN 命令与结果、提取的规则对照及旧源码精确路径/SHA256、无法唯一还原项、配置与预注册一致性、测试证据和顾虑。不得运行真实候选收益，不得访问或修改源数据库。
