# Task 1 实施报告：冻结预注册和唯一配置

## 状态

DONE

## 实现内容

- 新建不可变 `@dataclass(frozen=True)` 配置对象 `ResearchConfig`，后续任务可统一读取数据路径、业绩窗口、趋势参数、组合约束、费用、滑点和随机种子。
- 实现 `load_config(path: str | Path) -> ResearchConfig`：从 JSON 读取唯一配置，拒绝非对象根节点、非标准 JSON 常量、未知字段、额外策略参数及缺失/空数据库路径；文件和 JSON 解析错误统一转为 `ValueError`。
- 未实现环境变量、命令行或其他第二套策略参数覆盖。
- 冻结 `config/baseline.json`：本金 20,000、3 槽位、20%/60% 暴露、120/200/20/20 参数、2016-10-13 至 2026-10-29 窗口、万 2.5 且最低 5 元佣金、10bp 滑点、随机种子 20261101，以及两个数据库的仅路径配置。
- 新建预注册，列明旧 0/2,449、733 配置、28 组结构和 F2 的覆盖关系；明确本轮只新增极简归因与完全盘后执行证据。
- 冻结 8 个单因子邻域、20 个随机排序路径、机械验收/否决线和“失败即交付负结果”。

## 文件变更

本任务新建：

- `/Users/admin/WorkSpace/trading/trading_finally/02_预注册_极简趋势消融.md`
- `/Users/admin/WorkSpace/trading/trading_finally/config/baseline.json`
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/config.py`
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/__init__.py`
- `/Users/admin/WorkSpace/trading/trading_finally/tests/__init__.py`
- `/Users/admin/WorkSpace/trading/trading_finally/tests/test_config.py`
- `/Users/admin/WorkSpace/trading/trading_finally/.sdd/task-1-report.md`（本报告）

未修改设计文档或实施计划。

## TDD 记录

### Step 1：先写失败测试

`tests/test_config.py` 覆盖：

1. 基线全部冻结值与两个数据库路径；
2. 配置对象不可变；
3. 未知字段抛 `ValueError`；
4. 额外策略参数抛 `ValueError`；
5. 分别缺失两个数据库路径时抛 `ValueError`。

### Step 2：RED

从 `trading_finally/` 执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_config -v
```

结果：预期失败，退出码 1。关键输出：

```text
ImportError: Failed to import test module: test_config
ModuleNotFoundError: No module named 'trend_system.config'
FAILED (errors=1)
```

失败原因确为实现模块不存在，与简报预期一致。

### Step 3：最小实现

仅增加唯一 JSON 配置、不可变 dataclass、严格加载逻辑和预注册，没有添加数据库访问、策略回测或命令行参数。

### Step 4：GREEN

从 `trading_finally/` 执行：

```bash
PYTHONPATH=src python3 -B -m unittest tests.test_config -v
```

最终输出：

```text
test_additional_strategy_parameter_is_rejected ... ok
test_baseline_loads_frozen_values ... ok
test_config_is_immutable ... ok
test_each_missing_database_path_is_rejected ... ok
test_unknown_field_is_rejected ... ok

Ran 5 tests in 0.002s
OK
```

### 全量测试发现

从 `trading_finally/` 执行：

```bash
PYTHONPATH=src python3 -B -m unittest discover -v
```

结果：发现并运行同一组 5 项测试，全部通过，无错误或警告。

## 自审

- 逐项核对简报的 10 个显式默认值，以及佣金、最低佣金、滑点和随机种子，代码、JSON 与测试一致。
- `ResearchConfig` 使用 `frozen=True`；测试通过尝试修改 `slots` 验证会抛 `FrozenInstanceError`。
- 未知字段和额外策略参数共用明确的允许字段校验，均抛 `ValueError`；两个数据库路径分别验证存在、非空且为字符串。
- JSON 使用标准库严格加载，无环境变量读取、无策略参数 CLI 覆盖、无隐藏第二套配置。
- 预注册中的 8 个邻域均为单因子变化：90/150、180/220、10/30、15/25；验收线包含 6/8 邻域、16/20 随机路径及完整回撤分级。
- 文件范围符合简报；未读取或写入两个源数据库，未访问行情数据。
- 未执行 `git add`、`git commit` 或其他 Git 状态修改操作；未创建任何提交。

## Concerns

无。
