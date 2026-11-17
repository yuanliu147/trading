# Task 6：候选 A 冻结主版本真实运行报告

日期：2026-11-04
候选：`three_of_two_low_vol`（候选 B 为 `NOT_EVALUABLE`，未进入账户）
状态：**COMPLETED**（唯一一次主版本运行；未按结果改参数）

## 1. 交付与边界

- 新增 `src/trend_system/legacy_runner.py`、`scripts/run_legacy_candidates.py`、`tests/test_legacy_integration.py`。
- 真实输入只使用哈希核验的冻结 Parquet；未连接活源库，未覆盖 `results/baseline` 或上一阶段结果，未执行 git。
- 主窗口 2016-10-13 至 2026-10-29，初始资金 20,000 元，仅主板，T 日收盘信号、后续交易日开盘执行。

## 2. TDD 与回归

- 先写集成测试并取得模块缺失 RED：`ModuleNotFoundError: trend_system.legacy_runner`。
- 实现后集成测试 3/3 PASS；全项目 unittest **171/171 PASS**。
- 集成测试覆盖：按 `(date,code)` 一对一选窗与键漂移拒绝；重复运行输出一致；不存在信号日成交；`NOT_EVALUABLE` 在调用账户引擎前中止。
- 面板不含 `adjustment_factor` 时，运行器按冻结口径新增 `signal_close/close`，并要求所有入场键因子为正；本次真实运行入场键检查通过。

## 3. 真实运行

命令：

```bash
PYTHONPATH=src python3 -B scripts/run_legacy_candidates.py --config config/legacy_candidates.json
```

退出码 0，输出目录 `results/legacy_candidates/three_of_two_low_vol/`。

| 指标 | 值 |
| --- | ---: |
| 初始权益 | 20,000.00 |
| 期末权益 | 4,159.97 |
| 总收益 | -0.7920015 |
| CAGR | -0.14474781959117322 |
| 最大回撤 | 0.8132163141193354 |
| 闭合往返 | 417 |
| 胜率 | 0.2446043165467626 |
| 滚动三年正收益占比 | 0.0 |
| 账户日 | 2,440 |
| 成交 | 834 |
| 订单 | 21,579 |

年度权益收益（`performance_summary.json`）：2016 +3.33%、2017 -25.17%、2018 -12.55%、2019 -7.81%、2020 +6.42%、2021 -4.93%、2022 -21.61%、2023 -20.89%、2024 +11.49%、2025 -29.57%、2026 -32.27%。2016 与 2026 为部分年度。

## 4. 完整性与数据证据

主版本 `run_manifest.json` 的独立检查：

| 检查 | 结果 |
| --- | --- |
| 键集合与窗口 | 通过 |
| 现金非负 | 通过 |
| 买入整手 | 通过 |
| 合法开盘成交价与费用 | 通过 |
| 权益恒等式 | 通过 |
| 次一账户日买入、卖出合法顺延 | 通过 |
| 无信号日成交 | 通过 |

输入哈希：

- 主板面板 `6a51ed0e…acc1a`，7,335,154 行；
- 冻结信号 `a3b274ad…fc390`，7,335,154 行；
- 入场覆盖审计 `b9134289…3129c`；
- 公司行动 `f7a924a6…bb575`；
- 候选 A 摘要 `805204ca…8bd4b`；
- 配置 `09096fc0…5e74`。

主版本全部输出已另存 `.sdd-legacy/main_freeze.json`，在任何稳健性变体前冻结。

## 5. 顾虑

1. 主版本净亏损 79.20%、MDD 81.32%，远差于旧报告口径；该差异只能通过 Task 7 的冻结归因解释，不能据此改参。
2. 候选 B 保持 `NOT_EVALUABLE`，当前只有候选 A 一个可运行假设。
3. 数据层面仍有 Task 2 已披露的 `source_level_rebuild_reproducible=false`；冻结输入和本结果可复现。
4. 本报告不构成实盘授权；后续先完成 Task 7 冻结稳健性与机械裁决。

## 6. 后续代码演进的可复现性核验

为支持候选 B，`broker.py` 和 `legacy_policy.py` 后续只增加了可选关键字参数（`positions`、`date_index`、可配置拒绝原因），默认 Donchian 路径与候选 A 规则未变。2026-11-05 使用当前代码在临时根目录 `/tmp/legacy-a-repro` 重跑候选 A 主版本，未覆盖冻结结果；七类核心输出 SHA256 全部与 `.sdd-legacy/main_freeze.json` 一致：

`daily_account`、`daily_positions`、`trades`、`orders`、`round_trips`、`action_log`、`performance_summary` 均 MATCH。

因此 A 的业绩结论不因后续可选接口改动而改变；原 manifest 中的历史代码哈希保留为运行当时记录，重跑哈希作为行为等价的补充证据。
