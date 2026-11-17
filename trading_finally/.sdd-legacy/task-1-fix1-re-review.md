# Task 1 Fix Round 1 Scoped Re-review

复审仅覆盖原审查的 Important 1 与 Minor 1，并检查本轮 diff 是否越范围。依据更新后的完整 diff 和实际文件；实现报告不作为事实依据。独立复跑 `test_legacy_config.py`：18/18 OK。控制方所述 127/127 未在本复审中重复计算。未运行真实候选收益、未查询数据库、未执行 git、未派发子代理。

### Spec Compliance

- **通过：候选 A 实际入场否决已完整冻结。** `13_预注册_旧候选统一复核.md:94` 明确止损锚 `>=` 实际买入价时放弃买入、不建立仓位，同时保留“信号仍按原冷却规则收集”，并准确引用 `backtest.py:616-618`。`config/legacy_candidates.json:52` 固定 `skip_if_stop_at_or_above_entry: true`；`src/trend_system/legacy_config.py:37` 在 frozen dataclass 中声明同名 `bool` 必填字段。既有 `_check_fields` 与 `_scalar_value` 自动对该字段执行缺失/未知字段及严格布尔类型校验，没有新增默认值或旁路。`tests/test_legacy_config.py:80-86` 直接断言配置值、预注册键及权威源码行号，和文档语义一致。
- **通过：候选 B 精确回归断言已补齐。** `tests/test_legacy_config.py:118-120` 直接冻结 `lu_filter=True`、`gap_filter=True`、`gap_max=0.05`；第 121-137 行用 `assertEqual` 冻结完整且有序的 12 项 `not_evaluable_fields` 元组，与 JSON 和预注册一致，不再只检查其中 4 项。
- **通过：修改范围受控。** 本轮完整 diff 仍仅包含原来的四个 Task 1 文件；相对原审查版本的变化仅为上述入场否决字段、相应文档/测试及精确断言。没有策略信号或账户实现、其他参数调整、数据库变更或结果导向修改。

### Issues

无未解决的 Critical、Important 或 Minor。原 Important 1 和 Minor 1 均已关闭，本轮未发现新的 Critical/Important。

### Assessment

- **Spec Compliance：Approved**。
- **代码质量：Approved**。
- **Task quality：Approved**。两项原审查问题均已按要求修复，并有 18/18 配置测试通过。
