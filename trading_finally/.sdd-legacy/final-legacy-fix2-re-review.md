# 旧候选最终审查修复 2（scoped re-review）

- 审查日期：2026-11-05
- 范围：只复核第二轮 scoped re-review 的 C2-residual（B 配置/审计/代码哈希闸门）、I4-residual（A 场景分组闭合）和新 Important（旧 B 数字/旧信号对照），以及这些修复是否引入新的 Critical/Important。
- 方式：只读代码、文档、冻结结果与哈希；未运行 git，未修改仓库文件，未重跑完整回测，未派发子代理。运行了 `tests.test_legacy_b_config`（11/11）和全量 unittest（199/199）。
- 结论：**Approved**。三个 scoped 项目均已关闭；本轮未发现新的 Critical 或 Important；有 8 项 Minor，均不改变 REJECTED 裁决，也不影响当前冻结数值。

## 1. 逐项关闭状态

| 编号 | 项目 | 状态 | 独立核验要点 |
| --- | --- | --- | --- |
| C2-residual | B 运行前配置/审计/代码哈希闸门绑定实际 `--config` | **已关闭**（CLI/脚本路径） | `scripts/run_legacy_b.py:12` 与 `scripts/run_legacy_b_robustness.py:12` 均把 `args.config` 传入；`run_legacy_b` 比较 `sha256(config_path)` 与恢复审计（`legacy_b_runner.py:56-60`）；稳健性还比较审计/主 manifest/config/code（`legacy_b_robustness.py:80-94`）；loader 固定官方 config、30b/10b/行业 CSV 哈希，拒绝非空 `not_evaluable_fields` 与非法 `evaluation_status`（`legacy_b_config.py:32-42,99-128`）；11 项新测试全部通过。未发现 CLI 路径可绕过。 |
| I4-residual | A 场景分组闭合到 417 笔与总额 | **已关闭** | 从 `results/legacy_candidates/three_of_two_low_vol/round_trips.parquet` 与 `daily_positions.parquet` 重算：417 笔、净 -16,269.53、毛 -11,375.15；分组 low 127/-11,090.60/-9,601.72、mid 126/-7,530.38/-6,043.79、high 126/+10,481.14/+11,961.78、short_hold_lt3 38/-8,129.69/-7,691.42；38 笔为 16 笔 1 日、22 笔 2 日，全部净亏损；无重复 `(code, entry_date)`；94:47-53 与 61 已按完整样本限定结论。 |
| 新 Important | 旧 B 数字/旧信号对照清理 | **已关闭** | README:50 为 -12.115%/16.990%；94:62 为毛 +715.78/费用 4,526.32；97:43 为旧母版 156,370、主板 112,654、当前 114,424、共同 109,090（96.8%）、当前独有 5,334、Jaccard 0.675，均与 `.sdd-legacy/b_signal_comparison.json` 一致；交付文档中未检索到 -13.87、16.67、+853、4,994、97.9%、116,340、110,332、6,008、0.679 等旧值；95 与 `_write_reports` 输出逐字节一致，96 的数值段落与当前 B 结果一致。 |

## 2. 机械事实核验

| 项目 | 结果 |
| --- | --- |
| B 主 freeze 9 个文件与当前输出 | 9/9 哈希与字节数 MATCH |
| B 主 manifest `config`/`recovery_audit`/6 个代码哈希 | 全部与当前文件 MATCH |
| B 稳健性 `input_manifest.json` | 含 `signals/config/recovery_audit/code/b_main_freeze`；config/audit 与主 manifest 一致，code 6/6 一致，嵌套 freeze 与 `.sdd-legacy/b_main_freeze.json` 相同 |
| B 主结果数值 | 总收益 -0.12114799999999992、CAGR -0.012776978618361023、MDD 0.1699045、期末 17,577.04、408 笔、胜率 0.431372549、滚动正占比 0.411764706 |
| B 稳健性 | REJECTED；双倍 -0.3914465/MDD 0.3914465；随机 1/6、5 负、最大可能 15<16；remaining -3,242.0 |
| B 场景/退出 | 费用 4,526.32、净 -3,810.54、毛 +715.78；time_exit 404/-2,828.74/+1,654.17/4,482.91；catastrophe_stop 4/-981.80/-938.39/43.41 |
| A 主 freeze | 9/9 与当前 A 输出 MATCH；A 裁决仍 REJECTED |
| 全量测试 | `Ran 199 tests ... OK`；新增 `test_legacy_b_config` 11/11 OK |
| 交付边界 | A/B 均 REJECTED；未生成纯模拟/日报/持仓台账/实盘文件 |

## 3. Critical

无。

## 4. Important

无。

## 5. Minor（不阻断）

1. **函数级 config 对象未被绑定。** `run_legacy_b`/`run_legacy_b_robustness` 只比较 `config_path` 的哈希，不比较传入的 config 对象。实际复现：以官方 `config_path` 加 `replace(config, initial_cash=999999)` 调用 `run_legacy_b`，闸门通过，随后才在读取 `/nonexistent` 信号时抛 `FileNotFoundError`；稳健性同样通过闸门后在临时根目录 `data/main_panel.parquet` 处失败。该路径不经过两个 CLI 脚本（CLI 先调用固定哈希的 loader），因此不构成本轮 CLI 闸门绕过，但建议 runner 内重新加载官方 config 并比较 dataclass 相等，或直接只接受 `config_path` 并在内部加载。
2. **稳健性未重新哈希输入数据。** `run_legacy_b_robustness` 读取 `data/main_panel.parquet`、`results/legacy_b_signals.parquet`、`data/corporate_actions.parquet` 前未比较主 manifest/frozen summary 中的哈希；`input_manifest.json` 的 `signals.sha256` 抄自 frozen 声明值而非实际文件哈希。若信号或面板在稳健性运行前被替换，双倍/随机结果可基于新数据，而 manifest 仍显示旧哈希。这是既有缺口，建议与主 runner 一样先校验实际信号哈希，并记录面板/公司行动的实际哈希。
3. **代码哈希只覆盖 6 个文件。** 主 manifest 注册 `legacy_b_runner.py`、`legacy_b_policy.py`、`legacy_b.py`、`legacy_b_config.py`、`broker.py`、`execution.py`；稳健性实际还依赖 `legacy_b_robustness.py`、`legacy_robustness.py`、`metrics.py`、`data.py`、`legacy_runner.py`，这些未在代码闸门内。策略核心文件已覆盖，且 `legacy_b_robustness.py` 自身无法自证；建议扩展 manifest 或明确文档化覆盖范围。
4. **96 的规模/波动/滚动/共同失败段落不是 `_write_reports` 生成的。** 当前 `_write_reports`（`legacy_b_robustness.py:289-322`）只生成退出原因、指数状态与“分组仅作描述”一句；交付的 `96_候选B失败与场景.md:25-48` 多出的四段是 19:03:54 人工追加的（95/96 基础生成于 19:03:34）。内容与 `.sdd-legacy/scenario_groups_early_trend_breadth.json` 及当前 parquet 重算一致，但重跑文档中的 `run_legacy_b_robustness.py` 会丢掉这四段。建议把这四段并入 `_write_reports`，或在报告中明确由场景 JSON 组装。
5. **“B 主版本 8 类核心输出与旧冻结逐字节一致”无法独立复核。** `.sdd-legacy/b_main_freeze.json` 已在 19:03 更新，旧 freeze 未保留；`fix2-evidence.txt` 的 9/9 MATCH 是对当前 freeze 的自校验，不是旧 freeze 对比。当前可确认的是：当前 9/9 与当前 freeze MATCH，且主窗口/CAGR/MDD/笔数/胜率/滚动占比与 fix1 复审记录完全一致；逐字节声明仅有 mtime（输出均 19:03:07）和 progress/evidence 叙述支持。建议未来更新 freeze 前先保存旧哈希清单。
6. **保留的双倍成本/6 条随机路径结果早于最终闸门代码。** 其 `performance_summary.json` mtime 为 05:29—06:14，而 `legacy_b_config.py`/`legacy_b_robustness.py` 的 fix2 修改 mtime 为 07:30；最终 finalize 在 19:03。fix2 对这两个文件的改动是闸门/报告层，策略文件 `legacy_b.py`、`legacy_b_policy.py`、`broker.py`、`execution.py` 未在 fix2 diff 中变化，因此数值仍成立；但 `input_manifest` 记录的是最终代码哈希，不是产出保留结果时的确切 harness 哈希，属于 provenance 说明项。
7. **新增测试未覆盖稳健性侧的审计/代码负例。** 11 项测试覆盖了官方配置、替代配置、非空 `not_evaluable_fields`、错误来源哈希、非法候选状态、缺失 config_path、主运行缺失审计/审计配置哈希不符；未覆盖稳健性审计文件被篡改、稳健性缺失审计、主 manifest config/code 不符、代码漂移。loader 侧也未专门测试行业 CSV 哈希错误和顶层 status 非法。建议补 3—4 个负例。
8. **97:31 的 A 归因仍是退出原因会计措辞。** “主要亏损来自固定止损被频繁触发，而不是佣金或滑点单独造成”不是反事实因果；94 已改为“退出原因与分组会计，不是反事实因果”，97 建议同步。另，`config/legacy_b_recovery.json` 顶层 `status=PENDING_DATA_GATE`、恢复审计为 `EVALUABLE`，loader 两者都接受、运行时依赖审计；建议在配置/文档中写清两步状态语义（延续上轮 Minor）。

## 6. 独立核验的数值/哈希

### B 主/稳健性

- 主：-0.12114799999999992；CAGR -0.012776978618361023；MDD 0.1699045；期末 17,577.04；408 笔；胜率 0.43137254901960786；滚动正占比 0.4117647058823529。
- 双倍：-0.39144650000000003/MDD 0.39144650000000003。
- 随机：positive_paths=1、paths_completed=6、early_stop_reason=`6 completed paths include 5 negatives; maximum possible positives 15 < 16`；各路径 total_return 为 -0.258388、+0.0000785、-0.1521565、-0.1605575、-0.0647135、-0.0300005。
- 场景：total_trade_costs=4,526.32；closed_net_pnl=-3,810.54；closed_gross_pnl=715.78；time_exit 404 笔 net -2,828.74/gross +1,654.17/costs 4,482.91；catastrophe_stop 4 笔 net -981.80/gross -938.39/costs 43.41。
- 分组：入场金额 low 137/+1,702.35/+3,183.02、mid 135/-1,806.83/-313.06、high 136/-3,706.06/-2,154.18；持有期波动 low 136/-6,579.02/-5,058.73、mid 136/-1,494.22/+18.46、high 136/+4,262.70/+5,756.05；滚动 85 窗口、35 正（41.18%）、min -16.44%、median -0.95%、max 9.41%。
- 信号：当前 `entry_signal=True` 行数 114,424，唯一 key 数 114,424，日期 2017-10-13 至 2026-09-22；summary 的 proposed 116,568、ST 拒绝 1,965、entry 114,424、codes 3,198。
- 信号对照 JSON：mother_keys=156,370、mother_mainboard=112,654、mother_non_mainboard=43,716、our_keys=114,424、common=mainboard_common=109,090、share=0.9683633071、mother_only=47,280、our_only=5,334、jaccard=0.6746277148。上述内部算术（47,280、5,334、Jaccard、96.836%）全部重算一致；`our_keys` 已由当前信号独立计数确认。母版 4.1GB pickle 的 156,370/112,654/43,716 本轮未重新解包，沿用脚本既有读法，但数值与上轮审查记录一致。

### A 场景

- 417 笔；净 -16,269.53；毛 -11,375.15；费用 4,894.38。
- 持有期波动 low 127/-11,090.60/-9,601.72/胜率 0.110236；mid 126/-7,530.38/-6,043.79/0.277778；high 126/+10,481.14/+11,961.78/0.420635；short_hold_lt3 38/-8,129.69/-7,691.42/0.0。
- 38 笔 short_hold：16 笔持仓行数 1、22 笔持仓行数 2；全部 net_pnl<0；无重复 `(code, entry_date)`。
- 入场金额组：139/139/139；net -3,786.28/-233.71/-12,249.54；gross -2,299.30/+1,398.34/-10,474.19。
- 滚动 85 窗口、0 正、min -47.91%、median -29.51%、max -2.81%。
- 止损相关 115 笔毛 -24,291.14；非止损毛 +12,915.99（主要盈利结构约 +14,554.04）。

### 哈希

- `config/legacy_b_recovery.json`：`d77b88c7303b1567406e0f396b242a7b991d8998186ee704ba0f45f766552c1e`
- `results/legacy_b_recovery_audit.json`：`9f0a3a287d68360d6ea0bc668de0ee22029e936b0c72d866c2c1b651aa198573`
- `results/legacy_b_signals.parquet`：`831e8460837c046b054e715cbdb1ccfc87bfc7aa28a76a2f70a6149a8ff33833`
- `results/legacy_b_summary.json`：`aba454ca9d65abaefd8be29ce234917f549f63e4a40256fd495f4afbeb014acc`
- B 主 `run_manifest.json`：`add6520e4d7f3aa69e7f384fe75030ab71207b2290a2f7d03f8d6d59507c1331`
- B 稳健性 `summary.json`：`250110dc98ac008e1e805e447631fa8e67d7a69a8ff805f4be665f2a85999599`
- B 稳健性 `input_manifest.json`：`c5e392071cc51bb555cc3be736c3ab611c9ce22c00c2185ed1b2fc84107ad188`
- `.sdd-legacy/b_main_freeze.json`：`dbee99823da3ce67a78693d1010ce8c4204fc7cea017b7724d381a3938d8c39c`
- `.sdd-legacy/b_signal_comparison.json`：`723e56b2ed44c5da74b33cb544465cb5e88ae7f530ade7f048f83279a8f1eadd`
- 代码哈希（主 manifest 与稳健性 input_manifest 一致，且与当前文件 MATCH）：`legacy_b_runner.py e16c0339a7e270e68d21090e5279b1b2d67308eaf74f8a9f6c888e6d639a726c`、`legacy_b_policy.py 22a26be18b6cb5a9f41d83dda45f45dec1d1f2b974f7bc552ba8cfc4cbb71103`、`legacy_b.py 0ab18ab66053d6bd2556f225e57807dc4ae79c8224e10c6e8e9930b9e7b20f08`、`legacy_b_config.py 70088d3023384d1e2dea1dbef3fac048bc24155657dc6374c4cd0c517de9d7ed`、`broker.py 57cf7e3276a768c567417756d6c123efcf225fb5eb62bceaf71a59845c53d1dd`、`execution.py 5461e2d28f2460f4392c00148645c3b4cf3f064d7b9bb52d946ccd3ce8f864df`。
- 30b/10b/行业 CSV 的固定哈希与 `legacy_b_config.py` 常量、恢复审计及实际文件三方一致：30b `8406120a...`、10b `d45892c9...`、行业 `3ebc1a6c...`。

## 7. 结论

- C2-residual、I4-residual、新 Important 三项均关闭。
- 无新 Critical，无新 Important；8 项 Minor 不阻断本轮批准。
- 建议下一轮优先处理 Minor 1（函数级 config 对象绑定）、Minor 2（稳健性输入哈希）、Minor 4（96 场景段落并入生成器），并在更新 freeze 前保留旧哈希清单以消除 Minor 5 的不可复核性。
