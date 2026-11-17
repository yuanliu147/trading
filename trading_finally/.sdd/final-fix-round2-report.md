# 最终修复第二轮报告

日期：2026-11-03  
依据：`.sdd/final-fix-re-review.md`  
范围：仅 I1 门槛布尔校验、I2 执行完整性证据、I3 持仓 episode 相关性。

## 1. 结果摘要

三个未解决 Important 均已按复审给定的最小方向修复。最终全项目 **109 项 unittest 全部通过**。冻结基线与双倍成本的真实执行证据重新计算后全部通过，episode 隔离后的风险报告、90/92、两份汇总和受影响哈希已刷新。

没有改策略、成本、配置、信号或账户成交行为；没有派发子代理、执行 git 或访问源数据库。没有重跑完整 54 路径矩阵。主基线总收益、MDD 和 REJECTED 裁决均未改变。

## 2. 严格 RED 记录

先修改测试并实际运行以下聚焦用例，再修改生产代码：

```bash
PYTHONPATH=src:tests python3 -B -m unittest -v \
  test_robustness.RobustnessUnitTest.test_gate_passed_values_must_be_actual_booleans \
  test_broker.BrokerTest.test_execution_integrity_rejects_delayed_buy_non_open_fill_and_unproven_delayed_sell \
  test_broker.BrokerTest.test_execution_integrity_accepts_real_pending_delay_evidence \
  test_risk.RiskReportTest.test_holding_returns_reset_at_each_entry_episode
```

修复前结果为 4 项测试、5 个失败点：

| 反例 | 修复前实际结果 |
| --- | --- |
| 10 个正确 ID 且所有 `passed="False"` | 错误返回 UNREJECTED |
| 买入从次一账户日延后一日 | `next_open_after_signal=True`，`all_passed=True` |
| 买价 10.02，既定开盘 10.00 加 10bp 应为 10.01 | `legal_fill_prices=True` |
| 卖出延后且没有同订单逐日 pending 证据 | `next_open_after_signal=True` |
| 同一股票第二次建仓且中间价格跳变 | 第二次建仓首日收益不是 NaN |

合法跌停、停牌和缺 bar 顺延对照在 RED 运行中通过，证明反例没有把所有延迟一概视为错误。

## 3. 窄修复内容

### I1：严格布尔类型

`src/trend_system/robustness.py` 的 `evaluate_gates` 现在先要求每个 `passed` 满足 `type(value) is bool`，再检查实际布尔值。`"False"`、`"True"`、0、1、None 和 `np.bool_` 全部 fail-closed 为 REJECTED；合法 True/False 继续按既有 10%/20% 状态机裁决。

### I2：次一账户日、顺延与开盘成交规则

`src/trend_system/broker.py` 的 `execution_integrity` 现在：

1. 从账户日期序列建立每个信号日的次一账户日；买入成交日必须严格等于该日。
2. 卖出不得早于该日；更晚成交必须具有同一 `order_id` 的逐日 pending 合法原因。部分卖出允许以同一订单此前实际成交记录作为连续执行证据，不能用没有实际成交支持的 filled 日志替代 pending。
3. 每笔成交必须有同订单、同日期、同方向、同数量的 filled 记录。
4. 用实际成交日 bar 和对应 config 调用 `open_fill`，重算开盘价、滑点与合法夹取；再用 `transaction_costs` 独立核对佣金、印花税、过户费、成交额、总费用和现金变化。
5. 双倍成本使用其 20bp 滑点配置及 2 倍费用口径，不再误用基线 config 做价格验证。

只修改了验收函数，没有修改 `run_backtest`、订单生成、仓位、价格或成交逻辑。

### I3：隔离持仓 episode

`src/trend_system/risk.py` 保留 `entry_date`，按 `(code,entry_date)` 排序分组并计算 `signal_close.pct_change(fill_method=None)`。第二次建仓首日为 NaN，不再与上一持仓区间末尾跨空仓期计算收益。

## 4. GREEN 与真实冻结输出验证

新增反例全部通过后，运行完整测试：

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests
```

结果：**Ran 109 tests，OK**。

仅从既有冻结 Parquet 读取账户、成交、订单和持仓，重算执行证据及风险，没有调用账户回测：

| 检查 | 基线 | 双倍成本 |
| --- | --- | --- |
| 键集合、行数与窗口 | 通过 | 通过 |
| 现金非负 | 通过 | 通过 |
| 买入整手 | 通过 | 通过 |
| 既定开盘价、滑点、夹取、费用与金额 | 通过 | 通过 |
| 权益与持仓市值恒等式 | 通过 | 通过 |
| 买入次一账户日、卖出合法顺延 | 通过 | 通过 |
| `all_passed` | true | true |

真实相关性修正为：

| 指标 | 旧风险输出 | episode 隔离后 |
| --- | ---: | ---: |
| 可计算股票对 | 277 | 275 |
| 相关系数均值 | 0.17909568599766565 | 0.1990378071531245 |
| 相关系数中位数 | 0.18053774535329964 | 0.1881276815485074 |
| 最大值 | 1.0 | 1.0 |

该诊断仍为非门槛描述统计，不生成订单，不进入机械裁决。

## 5. 输出与哈希刷新

已刷新：

- `results/robustness/risk/` 风险报告与相关性明细。
- `results/robustness/main_summary.json` 和 `summary.json` 中的执行证据、风险统计与相应门槛说明。
- `90_研究结论.md`、`92_负结果与后续边界.md`。
- `results/baseline/run_manifest.json` 中受影响的源码哈希。
- `results/robustness/main_freeze.json` 中受影响文件哈希及本轮核心不变证据。

543 个冻结文件哈希全部重新计算并匹配。冻结清单中只有以下 4 个文件的哈希改变：

1. `main_summary.json`
2. `risk/holding_return_correlations.csv`
3. `risk/risk_report.json`
4. `risk/risk_report.md`

其余受保护冻结输出均保持原哈希。独立基线、稳健性基线和双倍成本目录的 `daily_account`、`daily_positions`、`trades`、`orders`、`round_trips`、`action_log`、`performance_summary` 共 **21 个核心文件 SHA256 前后完全一致**。因此无需且没有重跑完整 54 路径矩阵。

## 6. 裁决与顾虑

- 主基线总收益仍为 **-0.9265055**，MDD 仍为 **0.926955672192279**。
- 10 个冻结门槛仍 8 个失败，裁决仍为 **REJECTED**。
- 本轮修正的是未来回归的验收保护和风险统计口径，不把历史负结果改写为正，也不改变停止边界。
- 当前系统仍不得进入实盘观察；非门槛相关性变化不构成任何挽救证据。
