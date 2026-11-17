# SDD ledger — plan: /Users/admin/WorkSpace/trading/trading_finally/01_实施计划_主板趋势波段交易系统.md

Base git HEAD: bb3fe53c1f925b560c0044854849f8984d86c0a1
Workspace: /Users/admin/WorkSpace/trading/trading_finally/.sdd
Execution branch: research/2026-10-22-all-signal-timing（用户选择当前分支新目录）

## Setup rulings

- Ruling: 所有 SDD 过程文件放在 `trading_finally/.sdd/`，不调用会写 repo 根目录 `.superpowers/` 的默认 workspace — 用户明确要求本次全部内容放在 `trading_finally/` — 若错会把流程文件混入交付目录；通过目录隔离和最终保留审查记录降低影响。
- Ruling: 不执行 git commit，任务审查使用 `.sdd/snapshots/` 的基线快照与 `git diff --no-index` 生成完整 diff — 用户未授权提交，且当前工作区有大量无关修改 — 若快照遗漏文件会造成审查范围不完整；每次快照覆盖计划内源码、测试、配置、脚本和文档，排除大结果与快照自身。
- Ruling: 当前分支加唯一新目录作为隔离边界，不创建 worktree — 用户明确选择当前分支新目录 — 若错会增加当前工作区文件，但所有写入严格限制在 `trading_finally/`。
- Baseline: 新项目尚无源码或测试；依赖版本已在计划阶段确认，无根级项目测试可运行，未安装新依赖。

## Preflight interface and task scan

| 范围 | 生产方 / 消费方 | 检查结果 |
|---|---|---|
| Task 1 | 预注册、JSON、ResearchConfig / 全部后续任务 | 数值、路径和严格加载一致；无秘密参数覆盖 |
| Task 2 | 统一面板字段 / Task 3、5、6、7、8 | 原价、后复权、ST、状态和公司行动字段均有消费方 |
| Task 3 | entry/exit/ST 信号 / Task 5、6、8 | T 日信号供 T+1 执行；热身只计算指标，不产生主窗口前入场 |
| Task 4 | price_limits/costs/open_fill/size_buy / Task 5 | 原价、滑点、整手和费用接口一致 |
| Task 5 | BacktestResult / Task 6、7 | daily_account 与 round_trips 满足指标计算 |
| Task 6 | 固定基线与 manifest / Task 7 | 不允许参数覆盖，稳健性复用冻结入口 |
| Task 7 | UNREJECTED/BOUNDARY/REJECTED 与 90_研究结论 / Task 8 | 条件交付一致；历史不标 OOS_PASS |
| Task 8 | 日报与前瞻台账 / 用户盘后使用 | 仅 UNREJECTED 构建；边界/否决只交付负结果 |
| Task 1 自检 | 测试值 vs JSON/dataclass | 一致 |
| Task 2 自检 | 双源只读 vs 数据输出 | 一致；大结果落盘 |
| Task 3 自检 | shift(1) 与截断测试 vs 信号实现 | 一致 |
| Task 4 自检 | 400 股测试 vs 仓位口径 | 目标市值按未滑点开盘价，现金按滑点成交价，无冲突 |
| Task 5 自检 | 同日卖买、顺延、公司行动测试 vs 引擎 | 一致；分红到账近似已明确 |
| Task 6 自检 | 合成测试 vs 真实基线 | 真实盈亏不作为测试通过条件 |
| Task 7 自检 | 门槛与裁决命名 | 一致；`90_研究结论.md` 已补入文件清单 |
| Task 8 自检 | 条件文件与测试 | 一致；日报函数独立在 `src/trend_system/daily.py` |

Preflight plan corrections: Task 7 文件清单补入 `90_研究结论.md`；Task 3 真实面板断言明确为“入场信号”不早于主窗口。

## Task progress

Task 1: complete (no commits by user constraint; base `.sdd/snapshots/task1-base`, review clean; 5/5 tests PASS; two package marker ⚠️ items independently verified to exist)

Task 2: Ruling: 将 `source_conflict` 拆为执行冲突与复权模型冲突；原价 OHLC 冲突继续阻断，1,079,751 行复权模型差异不再等同数据无效，新增同花顺复权信号面板和共同窗口敏感性对照 — 原行情仅 1 行价格冲突，而复权差异既包含精度也包含真实模型差，全部阻断会改变宇宙且混淆执行与模型风险 — 若错会让 baostock 主信号承载更多第三方模型风险；成本是必须增加双源结果对照，若两源结论不同则不得形成未否决裁决。

Task 2 review 1: Important 2项（预注册仍要求全部源冲突阻断；计划新增双源机械门槛与预注册声称门槛不变矛盾）。
Task 2: Ruling: 不采纳“恢复全部 source_conflict 阻断/删除双源门槛”的表面一致性修法；改为在任何策略回测前修订预注册，使其与已批准设计和控制方数据裁决一致：仅 execution_conflict 阻断，双源共同窗口正收益且回撤<=10% 成为正式机械门槛，并修正 exclusions 文案 — 全部阻断会把模型不确定性误当执行不可行，而删除门槛会留下两源相反仍可未否决的漏洞 — 若错会更严格地否决部分候选或保留第三方基线风险；成本是全部规则需重新审查与披露。
Task 2: fix round 1/5 dispatched（resume original implementer）。
Task 2: fix round 1/5 (2 addressed, 0 open; scoped re-review approved; 20/20 tests PASS; no commits)
Task 2: complete (base `.sdd/snapshots/task2-base`, final re-review clean; remaining 40,042 rows >1% adjustment difference carried into mandatory Task 7 sensitivity, not parked)

Task 3: complete (base `.sdd/snapshots/task3-base`, review approved; controller independently reran 22/22 signal+data tests PASS from project directory; initial controller invocation from repo root failed import due wrong cwd, not code)
Task 3: minor (deferred): add independent negative cases for price/MA/channel gates and missing state fields; final review to triage.

Task 4: dispatched (base `.sdd/snapshots/task4-base`; implementer agent a864b41965beee861; no commits).
Task 4 review 1: Important 1项（size_buy 使用未夹取滑点价，与 open_fill 最终成交价不一致；联合边界测试缺失）。
Task 4: fix round 1/5 dispatched（resume original implementer）。
Task 4: fix round 1/5 (1 addressed, 0 open; scoped re-review approved; 19/19 execution and 46/46 full tests reported PASS).
Task 4: complete (base `.sdd/snapshots/task4-base`; final re-review clean; no commits).

Task 5: dispatched (base `.sdd/snapshots/task5-base`; implementer agent a062c33195f0072fd; no commits).
Task 5 review 1: Important 2项（T日预截断候选导致拒单不递补；负卖出净额可破坏现金非负及冻结买预算）；Minor 3项（往返数量核对、预检原子性、边界测试不足）。
Task 5: Ruling: 按简报“排序只决定同日有限预算下的先后顺序”，冻结全部合格候选及空槽资源，实际成交才耗槽，拒单后允许有序递补 — 通用的“合格信号多于空槽时排序”未明确预截断，而具体执行句更明确 — 若错会改变原预期的候选范围；成本是需保留该解释并验证没有使用T+1未来信息重排。
Task 5: fix round 1/5 dispatched（resume original implementer）。
Task 5: fix round 1 interrupted once by API 429 after targeted RED; user said 继续; original implementer resumed from existing workspace without restart or scope change.
Task 5: fix round 1/5 (5 addressed, 0 open; scoped re-review approved; controller independently reran 71/71 tests PASS).
Task 5: complete (base `.sdd/snapshots/task5-base`; final re-review clean; no commits).

Task 6: dispatched (base `.sdd/snapshots/task6-base`; implementer agent ac934224d00526010; no commits).
Task 6 review 1: Important 1项（闭合交易换手/贡献字段易被误读为全账户口径）；Minor 3项（滚动窗口长缺口、部分年度判断、manifest配置来源）。异常亏损已独立桥接：闭合净盈亏-17,860.06、未闭合-170.83、分红+332.70，合计-17,698.19，与权益差一致。
Task 6: Ruling: 保持 performance_summary(daily_account, round_trips) 冻结接口，将换手与贡献明确重命名为 closed/realized 口径并在摘要披露，不伪造成全账户归因 — 当前签名没有全成交和每日持仓，直接扩展成所谓全账户贡献会混淆数据口径 — 若错会少一个全账户换手字段；成本是Task7年度门槛应使用账户年度权益收益而非已实现贡献。
Task 6: minor (deferred): annual_and_rolling 仅凭账户序列无法无误判断任意日历的部分年度；当前2016/2026标记正确，交最终审查分类，避免引入任意日期阈值或新日历接口。
Task 6: fix round 1/5 dispatched（resume original implementer）。
Task 6: fix round 1/5 (3 addressed, 0 open; scoped re-review approved; controller independently reran 81/81 tests PASS; frozen account/trade tables unchanged, only summary/manifest regenerated).
Task 6: complete (base `.sdd/snapshots/task6-base`; final re-review clean; one deferred Minor on arbitrary-calendar partial-year detection).

Task 7: Ruling: 基线已明确为负且回撤超限，仍执行全部预注册稳健性矩阵，不以提前已知失败跳过试验 — 设计承诺失败也展示全部冻结试验，跳过会破坏负结果完整性 — 成本是计算时间，但不允许据变体结果救回基线。
Task 7: Ruling: 随机入场对照的“其他合格股票”定义为同一信号日满足非ST、可交易、data_valid、无execution_conflict、rank_return_120及必要信号历史完整的非entry_signal股票；按当日原entry信号数量无放回替换，保留替换股票真实rank_return_120供既有排序 — 若只在entry_signal内同数抽样会退化成随机排序，无法提供独立对照 — 若错会扩大随机入场的宇宙含义；成本是需明确它检验的是突破信号相对可交易宇宙的增量，不是纯随机无约束交易。
Task 7: Ruling: 去掉贡献最高年份的机械门槛使用账户年度权益盈亏（与年度收益同口径），不使用仅闭合交易的realized_annual_contribution — Task 6已明确后者不含未平仓和分红 — 若错会与预注册的“年份贡献”解释不同，但避免遗漏账户损益。
Task 7: Ruling: 市场状态与大小盘分组没有冻结阈值，只能作为事后描述性切片，公式与数据完整披露且不参与机械门槛、不反向形成交易闸门 — 防止借分组继续调参 — 成本是这些切片只具探索性，不能作为确认性证据。
Task 7: dispatched (base `.sdd/snapshots/task7-base`; implementer agent a7edfb7b073eb7cb0; no commits).
Task 7: implementation interrupted once by API weekly-limit 429 before RED; user said 继续; original implementer resumed without scope change.
Task 7: implementation and real matrix completed (55 full account runs, 1000 return bootstraps; 513 frozen-file hashes and 20 random-entry logs checked; final REJECTED with 8 failed gates).
Task 7: review dispatched (full snapshot diff `.sdd/task-7-review.diff`; controller independently reran 90/90 tests PASS).
Task 7: complete (review approved; no Critical/Important; final REJECTED independently reproduced; no commits).
Task 7: minor (deferred): persist random-order audit log/input hash; add direct double-cost sizing/fill and dual-source initialization regressions; itemize execution-integrity evidence instead of one broad boolean. Final review to triage; numeric verdict unaffected.

Task 8: negative-delivery path selected from reviewed REJECTED; no daily plan, live manual or forward ledger to be created.
Task 8: dispatched (base `.sdd/snapshots/task8-base`; implementer agent a1e2f4a2b541f370d; no commits).
Task 8 review 1: Important 1项（gross_pnl含滑点，负结果措辞却排除了滑点主因可能性）。
Task 8: fix round 1/5 dispatched（resume original implementer; wording/tests only, no real rerun）。
Task 8: fix round 1/5 (1 addressed, 0 open; scoped re-review approved; 95/95 tests reported PASS).
Task 8: complete (base `.sdd/snapshots/task8-base`; final re-review clean; no commits).

Final whole-project review: dispatched (strongest available model; `.sdd/final-review.diff`; controller independently reran 95/95 tests PASS; all deferred Minor items explicitly assigned for triage).
Final review: Needs fixes — Critical C1（compute_signals重排后，调用方用原面板位置掩码筛选，真实主窗口漏选/误选各2,065,437键；此前-88.49%结果及54个主窗口运行全部无效，必须撤回并重跑）；Important I1（门槛集合fail-open）、I2（完整性未含键集合）、I3（设计承诺的风险披露和停牌折价压力缺失）。
Final review: Ruling: 停牌估值折价压力固定为陈旧估值持仓市值额外下调10%和20%的非交易会计压力，只披露权益/回撤影响，不生成订单、不加入机械门槛、不用于救系统 — 设计已承诺但原任务未给假设，需在看修正结果前冻结 — 若错只是压力假设不同；成本是同时保留原始未折价结果和两种折价敏感性。
Final review: one fix dispatch authorized by SDD; fresh implementer, scoped re-review required; no new-stage strategy implementation until resolved.
Final fix: interrupted once by API 429 after root-cause confirmation and before RED; user said 继续 on 2026-11-03; original fix implementer resumed without scope change.

Final fix: user subsequently changed the hard MDD limit from 10% to 20% and prioritized higher after-cost CAGR within hard constraints; implementer was notified before corrected results were accepted. Frozen classification: MDD≤10% with all other gates passed = UNREJECTED; 10%<MDD≤20% = high-risk BOUNDARY eligible for pure simulation; MDD>20% = REJECTED.
Final fix: implementation and full rerun complete; report `.sdd/final-fix-report.md`; controller independently reran 105/105 tests PASS. Corrected baseline remains REJECTED (total return -0.9265055, MDD 0.926955672192279); old alignment results archived and invalid.
Final fix: scoped re-review dispatched to original final reviewer; package `.sdd/final-fix-review.diff`; report target `.sdd/final-fix-re-review.md`; no new-stage strategy implementation until verdict.
Final fix scoped re-review 1: Needs fixes — 0 Critical, 3 Important (non-bool `passed` fail-open; next-account-day/open-fill evidence too weak; correlation crosses holding episodes). C1, 20% policy, hashes, random logs, costs, dual-source initialization and delisting narrow exception passed. Numeric REJECTED unchanged.
Final fix: Ruling: authorize one additional narrow validation/reporting fix round and same-reviewer scoped re-review — all three findings are load-bearing for the reusable candidate gate/account evidence, and concrete minimal reproductions exist, so proceeding would knowingly reuse a fail-open validator — no strategy, cost, configuration or account behavior changes permitted; only validators, regression tests, episode-based risk statistics and affected reports/hashes may change. If account behavior changes, stop and rerun affected results.
Final fix round 2: implementation complete; report `.sdd/final-fix-round2-report.md`; controller independently reran 109/109 tests PASS. Numeric REJECTED and account results unchanged; risk correlation refreshed to 275 pairs. Scoped re-review 2 dispatched to original reviewer with `.sdd/final-fix-round2-review.diff`.
Final fix scoped re-review 2: Approved — 0 Critical/Important/Minor; report `.sdd/final-fix-round2-re-review.md`. Prior plan closed as verified research infrastructure and rejected Donchian candidate; no live delivery. New-stage implementation gate lifted.
