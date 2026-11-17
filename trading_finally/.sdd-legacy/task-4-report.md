# Task 4：候选 B 不可评价停止诊断

日期：2026-11-04

## 结论

候选 B `early_trend_breadth` 保持 **NOT_EVALUABLE**。不创建 `legacy_b.py`，不生成信号、风险元数据或账户结果。

## 依据

Task 1 对唯一授权母版 `2026-10-20_涨幅趋势_全维度定制/02_代码/30b_组合回测_含退市.py` 的逐项审查已确认：该文件加载多个预计算矩阵，却没有定义其构造公式。以下 12 项不能唯一还原：

1. `up_struct_formula`
2. `trend_age_formula`
3. `atr_pct_formula`
4. `dd60_formula`
5. `lu_count20_formula`
6. `market_breadth20_formula`
7. `sector_breadth20_formula`
8. `sector_ret20_formula`
9. `amt20_formula`
10. `historical_industry_membership`
11. `next_gap_formula_and_timestamp`
12. `amount_ranking_tie_break`

Task 2 还确认两个授权数据源没有带有效起止日的历史行业记录，行业面板为 0 行。即使补数据，也不能解决母版公式缺失。

## 计划冲突处置

原计划要求“即使真实 B 不可评价也保留合成测试”。但合成实现必须先选择上述未知公式，会直接违反设计权威中的“公式不能唯一还原时不得猜测、不得删过滤器继续回测”。因此计划已修订为记录停止诊断，不创建猜测实现。该处置不修改候选 A 或任何收益门槛。

## 验证

- `tests/test_legacy_config.py`：18/18 PASS，完整 12 项及 gap/LU 开关均受精确回归保护。
- `src/trend_system/legacy_b.py` 不存在。
- `results/legacy_candidates/early_trend_breadth/` 不存在。
- 未运行候选 B 收益，未删除行业、LU、gap 或其他过滤器，未从后续版本补公式，未执行 git 或访问活源库。

候选 B 不进入 Task 5—8 的账户实现、回测、稳健性或模拟交付。
