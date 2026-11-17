# Task 2 fix round 1 scoped re-review

- 审查日期：2026-11-04
- 范围：仅复核两项已批准的数据语义修正、原审查问题闭环、冻结输入可复现性及负面状态传播；不评价策略收益。
- 方法：读取更新 diff、设计/计划/预注册、报告、两份审计及冻结 Parquet；独立重算覆盖和补齐差异。未连接活源库、未运行 git、未派发子代理。

## Spec Compliance

| 核查项 | 结论 | 独立证据 |
| --- | --- | --- |
| 设计 11、计划 12、预注册 13 的一致性 | 符合 | 三处均记录 2026-11-04 修正：100% 仅用于 Task 3 实际触发且拟入场候选；正常停牌仅豁免 NULL 量额且不进入排名；活跃分母和逐日 99% 保留；原失败证据保留。 |
| 活跃分母不变 | 符合 | 独立按日比较原/completed Parquet，2,440 个主窗口日期全部匹配，逐日 denominator 不一致数为 0。行数均为 14,595,370，主键均唯一。 |
| 停牌豁免与排名资格 | 符合 | `refresh_context_flags` 只对 `trade_status=0` 豁免 NULL volume/amount，其他原始价、前收、hfq、ST、IPO 仍须完整有效；`rank_eligible=context_complete AND trade_status=1`。独立重算两个标志与存储值的全面板不一致数均为 0；所有停牌行 `rank_eligible=true` 为 0。 |
| 正常交易缺失仍失败 | 符合 | 正常交易 NULL 量额却 `context_complete=true` 的实际行数为 0。全窗口 16,146 个停牌 NULL 量额行中，15,672 行转为完整，474 行仍因其他字段无效而不完整；主窗口仍有 3,280 个上下文不完整行，没有伪装成全完整。 |
| 修正后逐日 99% | 符合 | 独立从 completed Parquet 重算 2,440 日，日期、denominator、numerator、coverage 与新 audit 逐项一致；低于 99% 为 **0 日**。最低值为 2022-01-18 的 4,614/4,620 = **99.87012987%**。未使用 pooled 比率。 |
| 原失败证据 | 符合 | `legacy_context_audit_pre_amendment.json` SHA256 为登记的 `c3037509…899`；新 audit 的 before 2,440 日记录与原审计逐字段完全相同，仍为 18 日失败、最低 98.133616%。原 full-A Parquet 哈希也保持 `8183acce…a85`。 |
| Task 3 严格入场谓词 | 符合 | `candidate_entry_valid` 同时要求必需字段非 NULL、原始/hfq 价格为正、`trade_status=1`、volume/amount 均为正、`execution_conflict=false`；NULL 冲突标志 fail-closed。独立 SQL 重算全活跃主板诊断 denominator=7,334,981、strict numerator=7,225,283、coverage=98.504454%，与审计一致。该值明确不是正式闸门。 |
| 100% 正式门槛的时点与分母 | 符合本轮批准 | 新 audit 的正式入场 `denominator/coverage/passes` 均为 NULL，`deferred_to_task3=true`；A 仅因全 A 上下文通过而保持 `PENDING_DATA_GATE`，没有判为 `EVALUABLE`。 |
| 12 键 hfq 同字段补齐 | 符合 | 全局比较原/completed 面板，四个 signal 字段实际发生变化恰好 **12 行**；原非空值被覆盖 0 行，非正常交易行变化 0，其他字段变化 0，三类冲突标志变化 0。实际变化键集合与 audit 的 12 键完全一致。 |
| 精确键、方向和剩余缺失 | 符合 | 12 键均与同花顺冻结面板同 `(date,code)` 的四价完全一致，原四价均 NULL，同花顺四价均正；未发生跨日期填充。冻结输入本身是同方向 hfq 信号面板。正常交易主板信号不完整行由 **1,019 降为 1,007**；10,964 个主板停牌 NULL 量额行前后不变。 |
| completed Parquet 身份与结构 | 符合 | 实际 SHA256 为 `12e92da7600f59c72851da194ac13fca237d593d1cb8c2a0c329e4c28c23d9fb`，与 audit/报告一致；14,595,370 行、5,500 codes、2010-01-04 至 2026-10-29、`(date,code)` 唯一，IPO/退市/BJ 违规均为 0。 |
| 整日缺失保护 | 符合 | 预期日期取哈希匹配的冻结 `main_panel.parquet`，不再由 full-A 实际行自证。预期/实际均为 2,440 日，missing=0、unexpected=0；合成测试覆盖整日缺失抛错。 |
| 冻结输入与源级限制披露 | 符合，可作为已知限制接受 | 本轮实现只读 Parquet，不含活库连接。原 full-A、main、hithink signal、industry、pre-amendment audit 的独立 SHA256 全部匹配 manifest。报告和 audit 均明确 `source_level_rebuild_reproducible=false`、`frozen_input_reproducible=true`，且未更新旧 DB 哈希。 |
| A/B 状态与收益边界 | 符合 | A=`PENDING_DATA_GATE`；B=`NOT_EVALUABLE`，12 项规则/溯源缺失完整保留，行业仍为 0 行、覆盖失败，`restored=false`。`signals_implemented=false`、`returns_run=false`。 |

## Strengths

1. 修正严格限制在批准语义内，没有借停牌豁免删除活跃分母或放宽正常交易字段要求。
2. 排名资格与入场有效性分离，停牌可保留上下文但不会进入排名或候选入场。
3. hfq 补齐有逐键证据，并通过全局差异证明只改变 12 键的 NULL signal 字段，没有覆盖、跨期填充或冲突清除。
4. 原失败审计、原面板及所有冻结输入均有可独立核验的 SHA256；修正前后逐日证据并列保留。
5. 对旧源库不可用作了明确负面披露，没有把冻结 Parquet 可复现冒充源库级重建可复现。

## Issues

### 原 Critical 1：活源库变化、旧源级构建不可复现

**在本轮限定范围内作为已如实披露的已知限制接受。** 外部 `bs_sync` 写入及 Hithink 外部变化使旧源库快照不可用，本轮没有伪称恢复源级重建；所有下游输入均锚定哈希匹配的冻结 Parquet。该接受仅适用于当前负面/数据审查，不构成源级重建通过或收益放行。

### 原 Important 1：不可交易行进入有效分子

**已关闭。** 新公共谓词排除停牌、零成交、NULL 量额、执行冲突及 NULL 冲突标志；正式 100% 检查移至 Task 3 实际候选集合，旧全活跃数字只保留为明确标记的诊断。

### 原 Important 2：授权内同字段补齐未穷尽

**已关闭。** 12 个可用精确键均已补齐并保留逐键审计，剩余 1,007 行及停牌 NULL 量额原因得到保留，没有宣称补齐本身使 A 可评价。

### 原 Minor 1：整日缺失可能静默消失

**已关闭。** 使用独立冻结主板日期集合并有合成反例。

### 未解决问题

**无未解决 Critical、Important 或新增 Minor。**

## Assessment

### Task quality

**Approved**（仅限 Task 2 fix round 1 的数据构建、负面状态与可复现性审查范围）。

理由：两项用户批准语义已一致落入设计、计划、预注册和实现；独立重算支持 0 日低于 99%、12 键精确补齐、1,007 行剩余缺失及严格入场谓词。旧源库不可重建已明确披露，冻结输入本身可复现。

该 Approved **不允许直接运行候选收益**：A 仍只是 `PENDING_DATA_GATE`，Task 3 必须对实际触发且拟入场候选完成严格 100% 检查；B 仍为 `NOT_EVALUABLE`。任何后续工作不得连接当前活库来伪装旧版本，也不得改写原失败证据或旧数据库 manifest。
