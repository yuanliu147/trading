# 旧候选最终审查修复 2（scoped re-review）审查包

- 日期：2026-11-05
- 范围：只复核第二轮 scoped re-review 的 1 Critical（C2-residual）、1 Important（I4-residual）和 1 新 Important（旧 B 数字/旧信号对照），以及这些修复是否引入新的 Critical/Important。
- 材料：
  - 修复 diff：`.sdd-legacy/final-legacy-review-fix2.diff`（相对 `.sdd-legacy/final-legacy-review-fix1.diff` 的 fix1 状态，12 个文件）
  - 机械证据：`.sdd-legacy/fix2-evidence.txt`
  - 测试日志：`.sdd-legacy/test-run-final.log`（199/199 PASS）
  - B 主 manifest：`results/legacy_candidates/early_trend_breadth/run_manifest.json`
  - B 主冻结：`.sdd-legacy/b_main_freeze.json`
  - B 稳健性：`results/legacy_b_robustness/summary.json`、`input_manifest.json`
  - 信号对照：`.sdd-legacy/b_signal_comparison.json`
  - 场景分组：`.sdd-legacy/scenario_groups_three_of_two_low_vol.json`、`.sdd-legacy/scenario_groups_early_trend_breadth.json`
  - 交付文档：`README.md`、`93`—`97`
  - 任务报告：`.sdd-legacy/task-8-report.md`、`.sdd-legacy/task-b-main-report.md`
- 约束：只读，不修改文件、不运行 git、不重跑完整回测、不派发子代理。可运行小范围只读校验或单测。
- 输出：把审查写入 `.sdd-legacy/final-legacy-fix2-re-review.md`，给出 **Approved** 或 **Needs fixes**，并逐项列出 C2-residual、I4-residual、新 Important 的关闭状态；如发现新的 Critical/Important，写明证据与最小修法。

## 本轮修复声明（需逐项核验，不要仅采信）

1. **C2-residual（B 运行前哈希闸门绑定实际 config）**
   - `src/trend_system/legacy_b_config.py` 固定 `OFFICIAL_CONFIG_SHA256`、30b/10b 与行业 CSV 期望哈希；加载器要求 candidate `evaluation_status ∈ {PENDING_DATA_GATE, EVALUABLE}` 且 `not_evaluable_fields == []`。
   - `src/trend_system/legacy_b_runner.py` 的 `run_legacy_b` 接收 `config_path`，比较实际传入 config 的 SHA256 与恢复审计；官方路径文件仍单独校验。
   - `src/trend_system/legacy_b_robustness.py` 的 `run_legacy_b_robustness` 同样要求 `config_path`，并校验恢复审计、B 主 manifest 的 config/recovery/code 哈希。
   - `scripts/run_legacy_b.py`、`scripts/run_legacy_b_robustness.py` 把 CLI 的 `--config` 传入。
   - 新增 `tests/test_legacy_b_config.py`（11 项），覆盖官方配置、替代配置、非空 `not_evaluable_fields`、错误来源哈希、非法状态、缺失/错误 config_path、缺失审计、审计哈希不匹配。
   - 需重点找：是否仍有路径可绕过实际 `--config` 绑定；稳健性是否仍可在不校验审计/代码的情况下重跑；loader 的哈希/字段校验是否有 fail-open 分支。

2. **I4-residual（A 分组闭合）**
   - `scripts/legacy_scenario_groups.py` 将持有期 <3 日的交易单列 `short_hold_lt3`。
   - A 的四组：low 127 / mid 126 / high 126 / short_hold 38；合计 417 笔，净 -16,269.53，毛 -11,375.15，与 `round_trips.parquet` 和 94 的总额闭合。
   - `94_旧候选失败与场景.md` 已加入该行与合计，并把结论限定为“三组可计算波动中低波动组亏损最重；另有 38 笔短持有全部亏损；这是会计不是反事实因果”。
   - 需重点找：脚本是否真的把 1 日/2 日交易全部纳入；是否存在重复/漏计；94 的表达是否仍把不完整样本当全样本结论。

3. **新 Important（旧 B 数字/旧信号对照）**
   - `94:58` 的 B 已改为毛 +715.78、费用 4,526.32。
   - `README` 的 B 已改为 -12.115%/16.990%。
   - `97` 的旧母版对照已用当前 114,424 个入场重算：旧母版主板 112,654，复现 109,090（96.8%），当前独有 5,334，Jaccard 0.675；`b_signal_comparison.json` 同步。
   - `95`/`96` 由 B 稳健性重新生成；`96` 的规模/波动/滚动/共同失败段落已恢复。
   - 需重点找：交付文档是否仍有修复前 B 数字（-13.87%、16.67%、+853、4,994、97.9%）或自相矛盾；97 的 96.8% 是否基于当前信号而非旧口径。

4. **附带需核验的机械事实**
   - B 主版本用最终代码重跑：8 类核心输出与旧冻结逐字节一致，仅 `run_manifest.json` 因代码哈希变化而更新；`b_main_freeze.json` 已更新。
   - B 稳健性从保留的双倍成本与 6 条随机路径重新 finalize；`input_manifest.json` 现有 config/recovery/code，且与 B 主 manifest 一致；裁决仍 REJECTED。
   - 全量测试 199/199 PASS。
   - 不得把 A/B 从 REJECTED 改为其他状态；不得生成纯模拟/实盘文件。

## 建议的最小核验命令（只读）

```bash
cd /Users/admin/WorkSpace/trading/trading_finally
PYTHONPATH=src python3 -B -m unittest tests.test_legacy_b_config -v
python3 - <<'PY'
# 可直接读取 .sdd-legacy/fix2-evidence.txt，无需重算全部
PY
```

不要运行 `run_legacy_b.py`、`run_legacy_b_robustness.py` 或任何会改写结果/报告的命令。
