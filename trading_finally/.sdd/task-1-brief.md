### Task 1: 冻结预注册和唯一配置

**Files:**
- Create: `trading_finally/02_预注册_极简趋势消融.md`
- Create: `trading_finally/config/baseline.json`
- Create: `trading_finally/src/trend_system/config.py`
- Create: `trading_finally/src/trend_system/__init__.py`
- Create: `trading_finally/tests/__init__.py`
- Test: `trading_finally/tests/test_config.py`

**Interfaces:**
- Produces: `ResearchConfig` 不可变 dataclass，以及 `load_config(path: str | Path) -> ResearchConfig`。
- 配置字段覆盖数据路径、窗口、120/200/20/20 参数、3 槽位、20%/60% 暴露、万 2.5 最低 5 元佣金、10bp 滑点和固定随机种子 `20261101`；后续任务只读取该对象。

- [ ] **Step 1: 写配置失败测试**

断言默认值精确为 `initial_cash == 20000`、`slots == 3`、`max_weight == 0.20`、`max_exposure == 0.60`、`entry_lookback == 120`、`trend_window == 200`、`trend_slope_window == 20`、`exit_lookback == 20`、`start_date == "2016-10-13"`、`end_date == "2026-10-29"`；未知字段、额外策略参数和缺失数据库路径必须抛 `ValueError`。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_config -v`
Expected: FAIL，提示 `trend_system.config` 不存在。

- [ ] **Step 3: 写预注册和最小配置实现**

预注册必须列出旧 0/2,449、733 配置、28 组结构和 F2 的覆盖关系，明确本轮只新增极简归因与完全盘后执行证据；冻结验收线、8 个邻域、20 个随机路径和“失败即交付负结果”。`load_config` 严格解析 JSON，不提供环境变量覆盖或秘密第二套参数。

- [ ] **Step 4: 运行测试并确认通过**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_config -v`
Expected: PASS；配置中不存在可被命令行临时覆盖的策略字段。

---

