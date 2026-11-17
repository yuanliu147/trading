# 最终整体审查任务（旧候选复核阶段）

请对 `/Users/admin/WorkSpace/trading/trading_finally` 的旧候选统一复核阶段做独立整体审查。只读，不修改文件、不运行git、不派发子代理；明确允许写审查报告 `.sdd-legacy/final-legacy-review.md`。不要评价上一阶段 Donchian 的既有设计，只检查其对复用模块的影响。

## 材料

- 设计：`11_设计_旧候选统一复核.md`
- 计划：`12_实施计划_旧候选统一复核.md`
- 预注册：`13_预注册_旧候选统一复核.md`、`14_预注册_候选B恢复.md`
- 台账：`.sdd-legacy/progress.md`（全部 Ruling 必须纳入）
- 完整快照 diff：`.sdd-legacy/final-legacy-review.diff`
- 结果：`93_旧候选复核结论.md`、`94_旧候选失败与场景.md`、`95_候选B复核结论.md`、`96_候选B失败与场景.md`、`97_停止边界与后续.md`
- 主版本：`results/legacy_candidates/three_of_two_low_vol/`、`results/legacy_candidates/early_trend_breadth/`
- 稳健性：`results/legacy_robustness/summary.json`、`results/legacy_b_robustness/summary.json`

补充：两个稳健性矩阵在运行中因主机内存压力暂停并断点续跑；已完成的路径摘要和双倍成本结果从磁盘跳过，未重算或改动种子/定义。请核对每条路径的日志/摘要齐全、随机排序种子与日期唯一，以及断点续跑没有改变门槛或路径含义。
随机门槛提前停止：A 已完成路径 12/20 全部为负；B 已完成 6/20 中 5 条为负。门槛要求至少 16/20 为正，因此最大可能正路径分别为 8 和 15，均低于 16；剩余路径无法改变门槛。摘要中的 `paths_completed` 与 `early_stop_reason` 应如实记录，不能虚报 20 条全部完成。

- 信号：`results/legacy_a_signals.parquet`、`results/legacy_b_signals.parquet`、`results/legacy_a_summary.json`、`results/legacy_b_summary.json`

## 必须核查

1. **规则真实性**：A 是否严格按 `backtest.py`/`股性分类.py`；B 是否严格按 `30b` 主方案 + `10b` 特征定义 + 时点行业 CSV。B 恢复是否属于同源补齐而非从后续优化版择优；不得猜测公式或删除过滤器。
2. **点时与前视**：全 A 排名/宽度/行业是否只用当日及以前；未来行业、除权、量额、停牌是否正确处理；B 的 `next_gap` 是否以盘后可执行的限价语义实现且未偷用次日信息。
3. **账户引擎**：默认 Donchian 路径是否字节级不变；A/B 的排序、每日开仓、风险 sizing、heat、止损锚、40/20 日计数、ST、跌停顺延、公司行动因子方向和 T+1 是否正确；是否存在位置掩码、现金透支、整手或费用错误。
4. **机械裁决**：门槛 ID/布尔是否 fail-closed；MDD 10%/20% 状态机；双倍费用是否实际进入 sizing 和成交（含印花税/过户费）；20 随机排序日志与种子；滚动三年、剔除最佳年份；B 的 2010—2015 结构性空集是否如实披露而非伪造。
5. **交付边界**：A、B 的最终状态与主结果是否一致；是否错误生成日报/持仓/前瞻台账/实盘建议；负结果归因是否超出证据（尤其 A 的止损、B 的费用与非主板信号）。
6. **复现**：输入/代码/结果哈希；A 主版本运行后 broker/policy 的可选参数改动是否影响 A 结果（应以重跑核心哈希证明）；B 主版本与旧母版信号对照是否真实。

## 输出

写入 `.sdd-legacy/final-legacy-review.md`：

- 整体 verdict（Approved/Needs fixes）；
- Critical/Important/Minor findings（file:line、证据、影响、最小修法）；
- 对 A/B 机械裁决的独立复核结论；
- 对 `97_停止边界与后续.md` 是否符合设计停止条件的结论；
- 已独立验证的关键数值/哈希。

只有无未解决 Critical/Important 才能 Approved。不要重跑完整矩阵；可做聚焦数值/哈希核验，大表不回显。
