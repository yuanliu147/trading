# 旧候选最终审查修复（scoped re-review）

- 审查日期：2026-11-05
- 范围：只复核上轮 2 Critical + 6 Important 的修复，以及修复是否引入新的 Critical/Important；材料为 `final-legacy-review-fix1.diff`、`task-8-report.md`、`task-b-main-report.md`、更新后的 93—97、两份稳健性摘要、`config/legacy_b_recovery.json`、`results/legacy_b_recovery_audit.json`、B `run_manifest.json`。
- 方式：只读代码/文档/结果，未运行 git、未修改文件、未派发子代理、未重跑完整矩阵；聚焦数值/哈希核验。
- 结论：**Needs fixes（不批准 Approved）**。C1 已关闭，C2 仍有未关闭的 Critical 残余；I1/I2（主 manifest）/I3/I5/I6 已关闭，I4 仅部分关闭并引入新的 Important 数值/文档问题。

## 1. 上轮 findings 关闭状态

| 上轮编号 | 状态 | 说明 |
|---|---|---|
| C1 B 买入已 ST 股票 | **已关闭** | 新 B 信号入场要求 `is_st==0`；`st_rejected_proposed=1965`；新 B 主/双倍均无 ST 买入、无 `st_exit`；旧结果归档 `.sdd-legacy/invalid_b_st/`。 |
| C2 B 恢复机器治理 | **未完全关闭（Critical 残余）** | 配置/审计/主 manifest 哈希已建立，但运行/稳健性 CLI 的实际 `--config` 未与恢复配置绑定；稳健性运行完全不校验恢复审计/配置；严格加载器仍接受 PENDING/NOT_EVALUABLE/非空 `not_evaluable_fields`。 |
| I1 A 随机 5 vs 12 | **已关闭** | `results/legacy_robustness/summary.json:44-45` 现为 12 条全负、max 8<16；93/97 已同步。 |
| I2 B manifest 无 config/code | **主版本已关闭；稳健性残余为 Minor** | B 主 manifest 现有 `config`、`recovery_audit`、6 个代码哈希，本审查逐项 MATCH；`results/legacy_b_robustness/input_manifest.json` 仍只有 signals/freeze，无 config/recovery/code。 |
| I3 93 仍称 B NOT_EVALUABLE | **已关闭** | `93:44` 已改为 B 恢复并指向 95/96。 |
| I4 场景分组缺失 | **部分关闭；新 Important** | 94/96 已加入规模/波动/滚动/共同失败；但 A 的波动分组漏掉 38 笔/净 -8,129.69，94 的共同失败仍用旧 B 数字。 |
| I5 task-8 report 缺失 | **已关闭** | `.sdd-legacy/task-8-report.md` 存在，状态/交付/测试/哈希/限制齐全。 |
| I6 A ST 入场未入预注册 | **已关闭** | `13:58` 已明确“已在 ST 状态的股票不得新买入”，记录 A 317、B 1,965 个拒绝。 |

## 2. Critical（仍有 1 项）

### C2-residual. B 的“运行前哈希闸门”没有绑定实际传入的 `--config`，稳健性运行完全没有恢复审计/配置闸门

- 证据：
  - `scripts/run_legacy_b.py:10-12` 接受任意 `--config` 并 `load_legacy_b_config(args.config)`，把得到的 config 对象传入 `run_legacy_b`。
  - `src/trend_system/legacy_b_runner.py:56-58` 只检查 `recovery["config"]["sha256"] == _sha256(root / "config" / "legacy_b_recovery.json")`，即只验证**官方路径文件**与恢复审计一致；从未比较 `args.config` 的路径/哈希，从未检查传入 `config` 的字段是否等于官方配置。因此传入一个复制后修改了阈值/seed/风险参数的 recovery 配置，只要官方文件未变，runner 仍会通过审计闸门并用**另一份 config**跑账户。
  - `scripts/run_legacy_b_robustness.py:10-12` 同样接受任意 `--config`；`src/trend_system/legacy_b_robustness.py:67-79` 只校验 B 主输出 freeze 和 `results/legacy_b_summary.json` 的 `status`，完全不读 `legacy_b_recovery_audit.json`，也不比较配置哈希/代码哈希；双倍费用和随机路径会直接用传入的 config 重跑。
  - `src/trend_system/legacy_b_config.py:86-100` 只限制 `status ∈ {PENDING_DATA_GATE, EVALUABLE}`；不校验 `candidate.evaluation_status`，不要求 `not_evaluable_fields` 为空，也不固定 30b/10b 的期望哈希。`config/legacy_b_recovery.json:15,23,34` 当前是 `PENDING_DATA_GATE/PENDING_DATA_GATE/[]`，而 `results/legacy_b_recovery_audit.json` 是 `EVALUABLE`；两步状态本身可以接受，但 loader 未把“字段必须已恢复”变成硬条件。
  - 构建脚本 `scripts/build_legacy_b_signals.py:19-23` 校验的是配置内声明的 source 哈希；它没有把 source 路径/哈希与 `14` 的固定值或代码常量比对，配置自身被改写后仍可能通过（当前 config 的 30b/10b/行业哈希是对的，见第 5 节）。
  - `tests/` 中没有 `legacy_b_config`/恢复审计/替代配置拒绝的测试（grep 无 `legacy_b_config`）；188/188 只证明现有测试通过，不证明该闸门失败关闭。
- 影响：**已经冻结的 B 主版本结果仍可由其 manifest 证明来源于官方 config/recovery/code**（本审查逐项核验 MATCH），因此现有 B 数值可以保留；但“运行前哈希闸门真实 fail-closed”这一修复声明不成立。用 `--config` 指向另一份 recovery 配置即可绕过，且 B 稳健性路径没有任何配置/恢复审计绑定。这不改变 A/B 的 REJECTED 方向，但属于未关闭的 Critical 治理问题。
- 最小修法：
  1. `run_legacy_b`/`run_legacy_b_robustness` 接收 `config_path`，比较 `sha256(config_path)` 与 `legacy_b_recovery_audit.json`/B 主 manifest 的配置哈希；或干脆忽略 CLI 路径，固定加载 `config/legacy_b_recovery.json`。
  2. 稳健性运行也读取 recovery audit，要求 `restored=true/status=EVALUABLE`，并校验 config/recovery/code 哈希与 B 主 manifest 一致。
  3. 严格 loader 增加：candidate `evaluation_status` 仅允许 `PENDING_DATA_GATE/EVALUABLE`；恢复配置进入运行时要求 `not_evaluable_fields == []`；在构建/运行中把 30b/10b/行业 CSV 的期望哈希固定为常量或与 `14` 的冻结值比对。
  4. 增加替代 config、非空 not_evaluable、缺失/错误恢复审计、错误哈希的失败测试。

## 3. Important（2 项）

### I4-residual. A 的“持有期波动分组”漏掉 38 笔交易和 -8,129.69 元净盈亏，94 的分组表不闭合，且“低波动组亏损最重”结论建立在不完整样本上

- 证据：`scripts/legacy_scenario_groups.py:28` 只有 `len(held) >= 3` 才计算持有期波动，否则置 NaN；`94_旧候选失败与场景.md:45-49` 的 A 波动组表为 127/126/126 笔，合计 **379 笔**，而 A 实际有 **417 笔**闭合往返。被排除的 38 笔（16 笔 1 日、22 笔 2 日）净盈亏 **-8,129.69 元**、毛盈亏 **-7,691.42 元**；三组表净盈亏合计 **-8,139.84 元**，仅覆盖 A 总净盈亏 **-16,269.53 元** 的约 50%，毛盈亏合计 **-3,683.73 元** vs 总 **-11,375.15 元**。报告没有“不可分组/漏计”行或披露。
- 影响：`94:57` 的“低波动组亏损最重”不是全样本结论；若把 38 笔短期止损交易归入低波动或单列，组间排序可能改变。它不会改变 A 的 REJECTED，但属于设计 §8 场景分组的数值完整性问题。
- 最小修法：将 2 日持仓纳入（用可定义的 1 个收益或明确 NaN），对 1 日持仓单列“持有<3日/不可分组”组，披露笔数、净/毛盈亏、胜率，并让三组+不可分组的总和与 `round_trips` 总数及总 PnL 对齐；再更新 `94:57` 结论或明确限定。不要用不完整分组得出“最重”结论。

### 新 Important. 最终交付仍保留多组“修复前 B”旧数字/旧信号对照，与 95/96/97 的新 B 结果矛盾

- 证据：
  - `94_旧候选失败与场景.md:58` 仍写“B 的毛收益仅 **+853** 元，显式费用 **4,994** 元”；修复后 B 是毛 **+715.78**、费用 **4,526.32**（95/96/97 已更新）。
  - `README.md:50` 仍写 B“主窗口 **-13.87%**、MDD **16.67%**”；修复后 B 是 **-12.115%**、MDD **16.990%**。
  - `97_停止边界与后续.md:43` 仍写“恢复实现复现旧母版主板信号的 **97.9%**”；该 97.9% 来自修复前 `our_keys=116,340` 的 `.sdd-legacy/b_signal_comparison.json:2`，当前 B 入场为 **114,424** 且排除了 ST。`task-b-main-report.md:59` 已注明“修复前口径”，但 97 未注明，读者会把旧口径当作当前 B 结果。
- 影响：最终交付中的 B 亏损归因、README 状态和信号对照并非同一版本；尤其 94 的共同失败原因仍引用被归档的旧 B 污染结果。B 的 REJECTED 方向不变，但最终文档一致性不满足“更新后的 93—97/交付一致”要求。
- 最小修法：把 94:58 改为 +715.78/4,526.32；README:50 改为 -12.115%/16.990%；对当前 B 信号（114,424，排除 ST）重算旧母版对照并更新 97/`.sdd-legacy/b_signal_comparison.json`，或至少在 97 标注“修复前口径，仅用于旧母版对照”。task-b-main-report 的 pre-fix 说明可保留。

## 4. Minor

1. **B 稳健性 `input_manifest.json` 仍只有 signals 与 b_main_freeze**，没有 config/recovery/code 哈希；B 主 manifest 已补齐，但稳健性路径仍无法从 manifest 独立绑定配置/代码。建议补 `config`、`recovery_audit`、`code` 哈希。
2. **B 恢复配置状态为 PENDING_DATA_GATE、audit 为 EVALUABLE**：两步状态可以接受，但 loader 不要求运行时配置已恢复；建议在 loader/runner 中显式声明并检查 `not_evaluable_fields==[]`，并将 config status 的语义写清楚。
3. **`legacy_b_config.py` 无测试**：188 测试未覆盖替代 config、非空 `not_evaluable_fields`、错误恢复审计/哈希；这是 C2 闸门无法被证明 fail-closed 的直接原因。
4. **`97:31` 的 A 归因“主要亏损来自固定止损”仍是退出原因会计，不是反事实因果**；建议改为“止损类退出记录了主要已实现亏损”，与上轮 Minor 一致。
5. 上轮 Minor（B gap 原始价 vs 10b hfq、tie-break 表述、B heat 的 fill/open `risk_ps`、随机日志 seed 列、稳健性输出哈希）仍未处理；不影响本轮裁决方向，但建议在下一轮一并清理。
6. **B 96/94 的“共同失败原因”引用旧 B 数字**已列入 Important；修复时同时检查 `94` 的 A 低波动组结论。

## 5. A/B 更新后的独立机械裁决

### A：REJECTED（不变，已核验）

- 主 -0.7920015、MDD 0.8132163、417 笔、win 0.2446、final 4,159.97；A 信号/摘要/配置哈希未变（`a3b274…`、`805204…`、`09096f…`）。
- A 随机摘要现为 **12 条全负**，`paths_completed=12`、`maximum possible positives 8<16`；93/97 一致。
- 双倍 -0.8705185/MDD 0.8835142；2010—2015 压力 -0.302305/MDD 0.4236156；A 的 REJECTED 独立成立。
- A 的 94 波动分组有 I4-residual 的 38 笔缺口，但不改变 A 裁决。

### B：REJECTED（修复后，数值已更新；结果本身可保留）

- 新 B 主：-0.121148、MDD 0.1699045、408 笔、win 0.4314、rolling 0.4117647、final 17,577.04；双倍 -0.3914465/MDD 0.3914465；随机 1/6（path_01 仅 +0.00785%），5 负，max 15<16；仍 REJECTED。
- `st_rejected_proposed=1965`、entry_count **114,424**；B 信号入场 ST=0；B 主/双倍买入 ST=0、`st_exit=0`；旧 451 笔/46 ST 结果已归档 `.sdd-legacy/invalid_b_st/`，当前结果与旧污染结果隔离。该部分 C1 修复成立。
- B 主 freeze 9/9 文件 MATCH；B 信号 `831e8460…`、B 摘要 `aba454ca…`、B manifest `config`/`recovery_audit`/6 个代码哈希均 MATCH；恢复配置 `d77b88c7…`、恢复审计 `9f0a3a28…` 与 manifest 一致。
- 两道剩余问题不影响当前 B 的 REJECTED 方向（C2-residual 是运行闸门，非数值；新 Important 是分组/文档/旧对照），但在修复前不能宣称 B 的复核流程 fail-closed。

## 6. 对 `97_停止边界与后续.md` 的复审结论

- **停止方向仍正确**：A/B 均 REJECTED，不进入 30 笔模拟；不生成日报/持仓/前瞻台账/实盘建议；不通过参数回调寻找偶然收益；A 随机已改为 12/12、最多 8；B 主/双倍/随机和 ST 修复后的数字已更新。
- **仍未最终闭环**：97:43 的 97.9% 是修复前口径（I-新），94:58/README:50 仍是旧 B 数字（I-新）；I4-residual 的 A 分组表不闭合。报告方向正确，但最终交付一致性不足。
- B 的 2010—2015 结构性空集仍如实披露；无日报/台账；无虚假 20 条完成。

## 7. 已核验的关键数值/哈希

- A：12/12 随机负；0.7920015；MDD 0.8132163；417；double -0.8705185；stress -0.302305；A 信号/摘要/配置哈希未变。
- B：主 -0.121148、MDD 0.1699045、408 笔、win 0.4314、rolling 38/85=0.4117647、final 17,577.04；double -0.3914465；random 1/6；remaining -3,242.0；总费用 4,526.32、毛 715.78、净 -3,810.54；`time_exit` 404、`catastrophe_stop` 4、`st_exit` 0。
- B 信号：proposed 116,568；ST 拒绝 1,965；entry 114,424；codes 3,198；ST 入场 0；B 主/双倍买入 408/408、ST 0、st_exit 0；旧 archive 主 451/46 ST、double 465/61 ST。
- 哈希：B 信号 `831e8460837c046b054e715cbdb1ccfc87bfc7aa28a76a2f70a6149a8ff33833`；B 摘要 `aba454ca9d65abaefd8be29ce234917f549f63e4a40256fd495f4afbeb014acc`；B 配置 `d77b88c7303b1567406e0f396b242a7b991d8998186ee704ba0f45f766552c1e`；B 恢复审计 `9f0a3a287d68360d6ea0bc668de0ee22029e936b0c72d866c2c1b651aa198573`；B freeze 9/9 MATCH；B manifest 代码 6/6 MATCH。
- A 94 波动分组：379/417 笔；组表净 -8,139.84、毛 -3,683.73；漏 38 笔净 -8,129.69、毛 -7,691.42；B 96 分组 408/408 且净/毛合计与总一致。
- 行业覆盖/恢复来源：B 行业 CSV/30b/10b 哈希与 14 一致；行业每日最低 99.7325%、0 天<99%、入场行业覆盖 100%（上轮已核验，本轮未变）。

## 8. 最小修复顺序与批准条件

1. **C2-residual（阻断）**：修 B runner/robustness 的 config 绑定；loader 增加 `evaluation_status`/`not_evaluable_fields` 检查；固定源哈希；补替代 config/错误审计/非空 fields 的测试。
2. **A 分组（阻断）**：修 `legacy_scenario_groups.py`，把 38 笔短持有交易纳入或单列“不可分组”，确保组表与 417 笔、-16,269.53 净、-11,375.15 毛对齐；更新 94 结论。
3. **旧 B 数字（阻断）**：更新 94:58、README:50、97:43；用当前 B 114,424 信号重算旧母版对照或标注修复前口径。
4. Minor：B robustness manifest、loader tests、A 归因措辞、上轮余留 Minor。

在上述 1—3 完成前，**不批准 Approved**；C1、I1、I3、I5、I6 已关闭，C2 与 I4 仍未完全闭环，且新增 Important 旧口径问题。
