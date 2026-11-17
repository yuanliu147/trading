# Task 5：为账户账本增加候选 A 可选策略对象

> 先读本简报，再读 13 预注册第 2/3 节、Task 3 信号接口和现有 `broker.py`。候选 B 已因 12 项公式/溯源缺失停止，本任务只实现候选 A，不创建 B 策略。

## 全局约束

- 所有新内容仅在 `trading_finally/`；SDD 仅 `.sdd-legacy/`；不 git、不派发子代理、不连接源库。
- 复用既有 `open_fill`、统一费用、现金、T+1、公司行动、停牌、跌停顺延和退市账本；不得另写成交引擎。
- T 日收盘信号，后续首个交易日开盘执行；实际价用于成交、整手、费用和真实价格阈值；点时复权只用于信号和收益。
- 不修改冻结规则、参数或数据门槛，不运行真实候选收益。

## Files

- Modify: `src/trend_system/broker.py`
- Create: `src/trend_system/legacy_policy.py`
- Test: `tests/test_legacy_policy.py`

## Interfaces

- Consumes: 既有 `run_backtest`、Task 3 信号和 `LegacyResearchConfig` 中候选 A 冻结规则。Task 3 的权威排序字段为 `legacy_rank_score`；Task 5/6 必须消费该字段及 `order_legacy_a_candidates` 的“原分降序、代码 tie-break”顺序，不得重算 Donchian 120 日收益。`rank_return_120` 仅是现有账户兼容的等值别名，不得作为候选 A 的权威排序来源。
- Produces: `LegacyPolicy`；`run_backtest(..., policy=None)` 可选参数支持候选 A 的每日开仓上限、原排序、sizer、固定入场止损锚和持仓相关收盘退出。
- `policy=None` 必须完全保留默认 Donchian 行为，不增加新字段依赖。

## 候选 A 必须固定的账户语义

- 最多 4 只、单票目标不超过权益 20%、每日实际新开最多 2 只；拒单/止损锚否决不耗槽，按 T 日冻结顺序递补；已持有不加仓。
- 原排序分数：权威字段 `legacy_rank_score` = 三法命中数×2 + 成交量/20 日均量×0.2，降序，代码最终确定性 tie-break；Task 5/6 不得使用或重算 Donchian 的 `rank_return_120` 替代。
- 信号日 low10 含 T，作为本次入场的固定结构锚，不是每日重算。实际买入时锚为 `max(low10 转换到实际价格尺度, 实际买入价×0.92)`；若锚 >= 实际买入价，放弃该次买入并有序递补，保留 `>=`。
- 公司行动后按 13 预注册的点时因子调整买入价锚和结构锚，保持经济尺度一致；不得除权后混比原始买入价与当前实际价。
- 入场日计第 1 日；仅有个股行情且成交量>0才增加持有计数，停牌/零成交不计；`held>=40` 在收盘确认，后续首个可执行开盘卖出。
- 收盘退出：固定止损锚、40日到期、MA5<MA20、魏氏柱<0、收盘<MA20，以及统一新增 ST。同日记录全部原因但只发一笔卖单；旧原因顺序为止损>到期>MA5<MA20>魏氏柱<0>破MA20，ST单列披露，不伪造旧优先级。
- 停牌、跌停、缺 bar 顺延继续使用既有订单/pending证据；买入委托不顺延。

## Steps

1. **写账户时序失败测试**：覆盖每日最多2开、4槽/20%、拒单递补、不加仓；固定low10、公司行动后锚、止损锚>=买入价放弃；40日计数及停牌不计；多退出原因/单笔卖单；跌停顺延；所有T日信号不得在T日成交。
2. **运行测试确认策略接口缺失 RED**。
3. **实现最小策略钩子**：复用统一成交与账本，候选 sizer 仍受实际现金、100股和组合上限约束；默认分支不增加字段依赖。
4. **验证默认回归**：跑全项目测试及既有固定基线复现；`daily_account/daily_positions/trades/orders/round_trips/action_log/performance_summary` SHA256必须一致，否则停止并修复，不能进入真实候选运行。

## 报告

完整报告写 `.sdd-legacy/task-5-report.md`，列RED/GREEN、接口、账户时序/公司行动/费用/现金反例、默认基线前后哈希、测试和限制。不得运行真实候选A/B收益。
