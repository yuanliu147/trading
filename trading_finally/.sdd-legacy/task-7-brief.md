# Task 7：候选 A 冻结稳健性、机械裁决与场景归因

> 先读本简报，再读 13 预注册第 6/7 节、Task 6 主版本结果与 `legacy_runner.py`。候选 B 已 NOT_EVALUABLE，不进入任何试验。

## 前提

- Task 6 候选 A 主版本已运行且六表/绩效/manifest 已冻结。先记录主结果全部哈希，再进行任何变体。
- 只允许预注册的有限复核：一个联合成本压力、20 个固定种子同信号随机排序、滚动三年、剔除最佳年份、主结果冻结后的 2010—2015 场景压力。
- 不新增参数邻域、不更换排序分数、不删除过滤器、不根据变体结果替换主版本。
- 所有新内容仅在 `trading_finally/`；SDD 仅 `.sdd-legacy/`；不 git、不连接活源库。

## Files

- Create: `src/trend_system/legacy_robustness.py`
- Create: `scripts/run_legacy_robustness.py`
- Create: `93_旧候选复核结论.md`
- Create: `94_旧候选失败与场景.md`
- Test: `tests/test_legacy_robustness.py`

## Interfaces

- Produces: `run_legacy_robustness(config) -> LegacyRobustnessResult`；`evaluate_legacy_gates(summary) -> str`，状态严格为 `UNREJECTED/BOUNDARY/REJECTED/NOT_EVALUABLE`。

## 冻结试验

1. **联合成本压力**：全部费用双倍且单边滑点 20bp，作为一个联合路径；费用/滑点必须实际进入 sizing 与成交，不能事后调整收益；不设额外 MDD 门槛。
2. **20 个随机排序路径**：种子固定 20261101；每个信号日内对同信号候选无放回随机排序，其他规则不变；至少 16 个扣费后为正。日志逐日保存 `(path_id,date,code,random_rank)` 和输入 SHA256。
3. **滚动三年**：实际月末、asof 三年，正收益比例严格超过 0.5。
4. **剔除最佳年份**：按账户权益年度盈亏剔除贡献最高年份，剩余年份合计为正。
5. **主路径收益**：扣费后净收益为正；主路径 MDD≤10% 为 UNREJECTED，10%<MDD≤20% 为高风险 BOUNDARY，>20% 或任一门槛失败为 REJECTED。
6. **2010—2015 场景压力**：主结果冻结后，使用同一冻结规则和全 A 冻结上下文重新计算该窗口信号并运行账户；只作场景分析，不反向调参、不进入主裁决。
7. **执行完整性**：主路径必须通过键集合/窗口、现金非负、整手、合法开盘成交/费用、权益恒等式、次一账户日与合法顺延证据。

## Steps

1. **写门槛和停止条件失败测试**：覆盖主路径收益、滚动三年、联合双倍费用+20bp、20 随机排序至少 16 正、剔除最佳年份、MDD 10%/20% 边界；NOT_EVALUABLE 不伪装为 REJECTED 或通过；随机排序固定种子可复现且不重用 Donchian 分数。
2. **运行测试确认 RED，再实现至 GREEN。**
3. **冻结主结果哈希后运行矩阵**：只对候选 A；结果写 `results/legacy_robustness/` 与 `results/legacy_stress_2010_2015/`。大表只落盘。
4. **生成结论和场景报告**：逐项列门槛；若候选 A 不可模拟，停止新策略探索，基于冻结路径做会计/交易归因、年度/滚动/指数状态/大小盘/波动分组；不新增消融或改参版本。
5. **条件交付**：仅当状态为 UNREJECTED 或“净收益为正且其他门槛通过”的 BOUNDARY，才在 Task 8 生成 30 笔纯模拟协议；否则只交付失败与停止边界。

## 报告

完整报告写 `.sdd-legacy/task-7-report.md`，列 RED/GREEN、主结果前后哈希、每个路径的真实结果、门槛逐项、随机排序日志、场景归因、限制。不得运行新的参数变体。
