# Task 1—8 最终修复第二轮 scoped re-review

日期：2026-11-03  
项目：`/Users/admin/WorkSpace/trading/trading_finally`

## 1. 整体结论

**复审结论：Approved。**

- 上轮 3 个 Important 已全部关闭。
- 本轮新增 Critical：0 项。
- 本轮新增 Important：0 项。
- Minor：0 项。

本次仅核查上轮三个必要条件及相关输出，没有评价新阶段，没有运行 git、没有派发子代理、没有重跑完整真实矩阵、没有回显大表。唯一写入为本报告。

这里的 Approved 是修复交付的复审结论，不是策略裁决。策略仍为 **REJECTED**：主基线总收益 -0.9265055，MDD 0.926955672192279，10 个冻结门槛仍 8 个失败，停止边界不变。

## 2. I1：严格布尔校验已关闭

**实现位置：**

- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/robustness.py:119-149`
- `robustness.py:142-144` 先要求所有 `passed` 满足 `type(value) is bool`，再检查实际布尔值，不再使用真值转换。
- `/Users/admin/WorkSpace/trading/trading_finally/tests/test_robustness.py:116-129` 覆盖字符串、0、1、None、`np.bool_` 及合法 True/False。

**独立反例结果：**

| 输入 | 裁决 |
| --- | --- |
| `"False"` / `"True"` | REJECTED |
| 0 / 1 | REJECTED |
| None | REJECTED |
| `np.bool_(True)` / `np.bool_(False)` | REJECTED |

合法状态机保持不变：

- MDD=0.10，其他门槛全为 Python True：UNREJECTED。
- MDD=0.100001、0.20：BOUNDARY。
- MDD=0.200001：REJECTED。
- 任一合法 Python False：REJECTED。

上轮字符串 `"False"` 可误判为 UNREJECTED 的路径已不存在。

## 3. I2：执行完整性证据已关闭

### 3.1 实现覆盖所需口径

**实现位置：**

- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/broker.py:495-696`
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/robustness.py:892-906`
- `/Users/admin/WorkSpace/trading/trading_finally/tests/test_broker.py:221-301`

核验内容如下：

1. `broker.py:617-649` 从真实账户日期序列建立次一账户日；买入必须严格等于该日，卖出不得早于该日。
2. `broker.py:650-659` 每笔成交必须有同订单、同代码、同方向、同信号日、同日期且同数量的唯一 filled 记录。
3. `broker.py:660-690` 延迟卖出逐日检查同订单记录；pending 必须具有允许的顺延原因，部分卖出期间的 filled 记录必须有对应真实成交，不能用孤立 filled 日志代替证据。
4. `broker.py:526-585` 使用实际成交日 bar 和对应 config 调用 `open_fill`，重算开盘价、方向滑点和合法范围夹取。
5. 成交额、佣金、印花税、过户费、总费用及现金变化逐项比较，`rtol=0.0`、`atol=1e-8`。
6. 双倍成本调用使用 20bp 配置和 `fee_multiplier=2.0`，没有误用基线费用口径。

### 3.2 负例与合法顺延

既有聚焦测试全部通过：

- delayed buy：`next_open_after_signal=False`、`all_passed=False`。
- non-open fill：`legal_fill_prices=False`、`all_passed=False`。
- 无逐日 pending 证据的 delayed sell：价格检查仍可独立为 True，但时序和总证据为 False。
- 真实跌停、停牌和缺失 bar 顺延对照均通过，没有把合法延迟一概拒绝。

另行逐字段修改买入成交的价格、成交额、佣金、印花税、过户费、总费用和现金变化，七类修改均使 `legal_fill_prices=False`、`all_passed=False`。

另以开盘价 10.00、买入滑点后 10.01、当日高价 10.005 的夹取场景完整运行账户和验收器，实际夹取成交价为 10.005，`all_passed=True`，证明新增价格核验没有破坏合法夹取路径。

### 3.3 真实冻结输出独立重算

从既有 `data/main_panel.parquet` 重算信号并选择主窗口，只读取冻结账户/成交/订单 Parquet，没有调用 `run_backtest`：

- 基线六项检查及 `all_passed` 全部 True。
- 双倍成本六项检查及 `all_passed` 全部 True。
- 重算结果与 `results/robustness/summary.json`、`main_summary.json` 中保存的证据一致。

因此，上轮“只检查晚于信号日且不核验既定开盘成交”的 Important 已实质关闭。

## 4. I3：episode 相关性已关闭

**实现与测试位置：**

- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/risk.py:95-130`
- `risk.py:99-106` 保留 `entry_date`，按 `(code,entry_date,date)` 排序，并按 `(code,entry_date)` 分组计算 `pct_change(fill_method=None)`。
- `/Users/admin/WorkSpace/trading/trading_finally/tests/test_risk.py:109-137` 构造两次独立建仓及中间价格跳变，明确断言第二次建仓首日收益为 NaN。

**真实输出独立重算：**

使用冻结基线 `daily_positions.parquet` 和主窗口后复权 `signal_close` 重算：

| 指标 | 重算结果 | 已保存结果 |
| --- | ---: | ---: |
| 股票对 | 275 | 275 |
| 均值 | 0.1990378071531245 | 0.1990378071531245 |
| 中位数 | 0.1881276815485074 | 0.1881276815485074 |
| 最大值 | 1.0 | 1.0 |

275 行股票对键完全一致，逐行相关系数最大差为 1.11e-16，仅是 CSV 浮点读写误差。`risk_report.json` 的完整相关性摘要与重算字典一致。

以下文档和输出均已刷新：

- `/Users/admin/WorkSpace/trading/trading_finally/results/robustness/risk/holding_return_correlations.csv`
- `/Users/admin/WorkSpace/trading/trading_finally/results/robustness/risk/risk_report.json`
- `/Users/admin/WorkSpace/trading/trading_finally/results/robustness/risk/risk_report.md`
- `/Users/admin/WorkSpace/trading/trading_finally/90_研究结论.md:166`
- `/Users/admin/WorkSpace/trading/trading_finally/92_负结果与后续边界.md:66`

90/92 均披露 275、0.199038，并明确 `(code,entry_date)` 隔离；旧的“277 个”及 0.179096 口径已不存在。风险诊断仍标记为 diagnostic_only、不生成订单、不进入机械门槛。

## 5. 核心文件与冻结哈希

对 `/Users/admin/WorkSpace/trading/trading_finally/results/robustness/main_freeze.json` 做独立流式 SHA256 重算：

- `files` 共 **543** 项，当前文件哈希不匹配数为 **0**。
- `round2_refresh.core_sha256` 共 **21** 项，当前文件哈希不匹配数为 **0**。
- 21 项完整覆盖独立基线、稳健性基线、双倍成本三个目录的 `daily_account`、`daily_positions`、`trades`、`orders`、`round_trips`、`action_log`、`performance_summary`。
- 21 个核心文件最晚修改时间为 2026-11-03 11:50:44 UTC，早于本轮 `round2_refresh` 的 13:50:45 UTC，与“没有重写核心账户结果”一致。
- `round2_refresh.changed_files` 精确列出 4 项：`main_summary.json`、相关性 CSV、风险 JSON、风险 Markdown。
- 独立基线 manifest 的 14 个源码/设计文档哈希也全部匹配。

`summary.json` 与 `main_summary.json` 的 verdict、baseline、gates、execution_integrity 关键字段一致；基线收益、MDD 和 REJECTED 未变。

本轮修改限于门槛验收、执行验收和非门槛风险统计。核心账户/成交/往返/绩效文件保持不变，543 项冻结清单全部可验证，因此**无需重跑完整 54 路径矩阵的结论成立**。

## 6. 实际验证范围

- 聚焦运行 6 项既有 unittest，全部 OK：严格布尔与状态机、执行三类负例、合法 pending 顺延、独立执行字段、episode 首日 NaN。
- 额外运行畸形门槛、七个成交字段及合法夹取的内存反例。
- 独立重算真实基线/双倍成本执行证据、275 行相关性、543 项冻结哈希、21 项核心哈希和 14 项源码/设计文档哈希。
- 控制方报告的 109 项完整 unittest 未在本复审中重复运行；本次聚焦验证未发现与其声明冲突的结果。

三个必要条件均已满足，且无新 Critical/Important，故本轮 Approved。
