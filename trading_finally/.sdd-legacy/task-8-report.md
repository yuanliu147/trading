# Task 8：负结果交付与停止边界报告（含最终审查修复）

日期：2026-11-05

## 1. 最终状态

| 候选 | 数据闸门 | 主窗口总收益 | CAGR | MDD | 闭合往返 | 最终状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A `three_of_two_low_vol` | EVALUABLE | -79.200% | -14.475% | 81.322% | 417 | REJECTED |
| B `early_trend_breadth` | EVALUABLE（同源恢复） | -12.115% | -1.278% | 16.990% | 408 | REJECTED |

两个候选均未达到主窗口扣费后正收益等机械门槛，不进入 30 笔纯模拟。

## 2. 交付文件

- `93_旧候选复核结论.md`：A 的机械门槛、年度盈亏与随机披露。
- `94_旧候选失败与场景.md`：A 的退出/费用归因、指数状态、规模与波动分组、滚动三年。
- `95_候选B复核结论.md`：B 的机械门槛、年度盈亏与随机披露。
- `96_候选B失败与场景.md`：B 的退出/费用归因、指数状态、规模与波动分组、滚动三年、共同失败原因。
- `97_停止边界与后续.md`：最终状态、亏损归因、旧报告口径差、停止边界、复现命令。
- `results/legacy_robustness/`、`results/legacy_b_robustness/`：主版本之外的双倍成本、随机排序与摘要。
- `.sdd-legacy/`：任务报告、台账、审查包、无效结果归档。

没有创建 `97_纯模拟观察协议.md`、`src/trend_system/legacy_daily.py`、`scripts/legacy_daily_plan.py`、`state/legacy_forward_ledger.csv`、日报、持仓清单或实盘订单。

## 3. 最终审查修复

- **B 已 ST 入场**：旧 B 买入 46 只入场即 ST 的股票，次日全部 ST 退出，污染结果。修复为入场要求 `is_st==0`，排除 1,965 个拟入场 ST 候选；旧结果归档 `.sdd-legacy/invalid_b_st/`，B 主版本/双倍/随机全部重跑。
- **B 恢复治理**：新增 `config/legacy_b_recovery.json`、`legacy_b_config.py`、`results/legacy_b_recovery_audit.json`；B 构建校验 30b/10b/行业 CSV 哈希，运行前校验恢复审计和配置哈希。
- **A 随机披露**：A 的 12 条已完成随机路径全部计入，`0/12`、最多 8 条正 <16，不再写成 5/20。
- **A 的 ST 入场排除**：13 预注册补充明确 `is_st==0`；A 的 317 个拟入场 ST 候选被权限层排除。
- **B manifest**：补入配置、恢复审计、六个代码文件哈希；B 稳健性 `input_manifest.json` 同样绑定 config/recovery/code。
- **B 配置闸门 fail-closed（复审 C2-residual）**：`legacy_b_config.py` 固定官方配置哈希、30b/10b 与行业 CSV 哈希，拒绝非空 `not_evaluable_fields` 与非法 `evaluation_status`；`run_legacy_b`/`run_legacy_b_robustness` 校验实际传入的 `--config` 哈希，稳健性运行同时校验恢复审计、主 manifest 与代码哈希。新增 `tests/test_legacy_b_config.py` 覆盖替代配置、非空字段、错误来源哈希、缺失/错误审计与缺失 config_path。
- **场景分组**：94/96 增加入场金额、持有期波动、滚动三年和共同失败原因；`scripts/legacy_scenario_groups.py` 可从冻结结果重建。A 的 38 笔持有不足 3 日交易单列为 `short_hold_lt3`，使 417 笔、净 -16,269.53、毛 -11,375.15 闭合。
- **最终旧数字清理**：94 的 B 毛/费用改为 +715.78/4,526.32；README 的 B 主窗口改为 -12.115%/16.990%；97 与 `.sdd-legacy/b_signal_comparison.json` 用当前 114,424 个入场重算，主板旧信号复现 96.8%、双向 Jaccard 0.675。
- **93 旧句**：不再把 B 称为 NOT_EVALUABLE/未运行。

## 4. 测试与复现

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests
```

控制方最终运行：**199 项测试全部通过**（最终审查修复后）。

主版本复现：

```bash
PYTHONPATH=src python3 -B scripts/run_legacy_candidates.py --config config/legacy_candidates.json
PYTHONPATH=src python3 -B scripts/run_legacy_b.py --config config/legacy_b_recovery.json
PYTHONPATH=src python3 -B scripts/run_legacy_robustness.py --config config/legacy_candidates.json
PYTHONPATH=src python3 -B scripts/run_legacy_b_robustness.py --config config/legacy_b_recovery.json
```

A 主版本已用当前代码在 `/tmp/legacy-a-repro` 重跑，七类核心输出哈希与冻结值一致；证据 `.sdd-legacy/a_reproduction_addendum.json`。B 修复后主版本哈希冻结在 `.sdd-legacy/b_main_freeze.json`。

## 5. 限制

- 源库级重建不可复现（`source_level_rebuild_reproducible=false`）；下游冻结 Parquet 输入可复现。
- B 未运行 2010—2015 压力：原规则资格日期为 2017-10-13，该窗口结构性无入场。
- 规模分组用入场金额作代理、波动分组用持有期日收益标准差；没有历史市值数据，不能冒充精确大小盘分类。
- 所有分组只作描述，不参与门槛，不用于调参。
- 本报告不构成实盘授权或收益保证。
