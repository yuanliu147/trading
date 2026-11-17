# Task 1—8 最终修复 scoped re-review

日期：2026-11-03  
项目：`/Users/admin/WorkSpace/trading/trading_finally`

## 1. 整体结论

**复审结论：Needs fixes。**

- Critical：0 项。
- Important：3 项，分别为 I1 的门槛布尔值 fail-open、I2 的次日/开盘价证据不足、I3 的持仓相关性跨持仓区间计算。
- Minor：0 项。
- 因仍有未解决 Important，不能 Approved。

这项复审结论针对交付与验收实现，不是策略裁决。现有真实结果仍为 **REJECTED**，且本次发现不会改变其方向：主基线总收益 -0.9265055、MDD 0.926955672192279，10 个冻结门槛中 8 个失败。当前基线仍不得进入实盘观察。

本次仅审查指定修复范围，没有评价新阶段 10/11/12，没有运行 git、没有派发子代理、没有重跑完整真实矩阵，也没有修改项目源码、输入或结果。唯一写入为本报告。

## 2. 未解决 Important

### I1-1：`passed` 使用真值转换，畸形门槛仍可得到 UNREJECTED

**位置：**

- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/robustness.py:119-148`
- 核心问题在 `robustness.py:142`：`bool(gate.get("passed", False))`。

**已修复部分：**

10 个冻结 ID 的精确集合检查已经生效。列表长度不是 10、元素不是 dict、ID 非字符串、重复、缺失、未知或仅有无关门槛，均返回 REJECTED。对应测试 `tests/test_robustness.py:102-114` 通过。

**反例证据：**

构造 10 个 ID 完全正确、MDD=0.05，但每个门槛均为 `{"passed": "False"}` 的摘要，实际返回：

```text
string_false_gate_verdict UNREJECTED
```

原因是非空字符串 `"False"` 经 `bool(...)` 后成为 True。相同的真值问题也适用于其他非布尔对象。

**影响：**

精确 ID 集合已 fail-closed，但整个机械裁决尚未对畸形 `passed` 值 fail-closed。内部 `_gate` 在 `robustness.py:597-605` 会生成真正布尔值，所以现有 `summary.json` 没有由该问题造成错误裁决；但验收函数本身仍可把全部字符串形式的失败门槛误判为通过。

**最小修法：**

在裁决前要求每个 `passed` 都是真正的布尔值，再直接检查其值；例如严格接受 `type(value) is bool`，不要用真值转换。补充 `"False"`、`"True"`、0、1、null 等反例，确保畸形摘要只能 REJECTED。

### I2-1：`next_open_after_signal` 只检查晚于信号日，且不验证开盘价

**位置：**

- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/broker.py:526-551`：价格只落在当日高低价及涨跌停交集内。
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/broker.py:583-586`：时序只检查 `trade.date > trade.signal_date`。
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/robustness.py:891-896,905-910,928-933,952-958`：上述证据进入基线、双倍成本及收盘后可复现门槛。
- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/report.py:72-89`：报告把该布尔值表述为“信号后次日成交时序”。
- `/Users/admin/WorkSpace/trading/trading_finally/90_研究结论.md:18,21,25,42` 与 `92_负结果与后续边界.md:24`：使用了“次日合法开盘”或同等强表述。

**聚焦反例：**

使用项目现有合成账户夹具，正常买入应为信号日 2024-01-02 后的次一账户日 2024-01-03。仅把买入成交日期改为 2024-01-04 后：

```text
delayed_buy_all_passed True
key_set_and_window       True
cash_nonnegative         True
whole_lot_buys           True
legal_fill_prices        True
equity_identity          True
next_open_after_signal   True
```

另一个反例把买价从 10.01 改为 10.02；当日开盘为 10.00，冻结 10bp 买入滑点下应为 10.01，10.02 虽处于合法日内范围却不是既定开盘成交规则的结果。检查仍返回：

```text
non_open_price_all_passed True
legal_fill_prices         True
next_open_after_signal    True
```

**实际输出的独立补验：**

现有真实账户没有发现上述日期错误。直接按账户日历重算：

- 基线买入 169/169 均在信号后的次一账户日；卖出 168/168 均不早于次一账户日。
- 双倍成本买入 164/164 均在次一账户日；卖出 163/163 均不早于次一账户日。
- 基线/双倍成本最低现金分别为 667.82/620.07 元，非整手买入均为 0。
- 基线权益恒等式及逐日持仓市值汇总误差均为 0；双倍成本最大权益组成误差为 0，持仓市值汇总浮点误差约 4.55e-13。

因此，本项是**验收证据不能识别未来执行回归**，不是指控现有真实成交日期已经错位。报告目前把弱检查写成强证据，仍须修正。

**影响：**

延迟一天的买入、没有既定开盘价依据的成交价，仍可能使 `data_execution_integrity`、`double_cost_positive_and_executable` 和 `fully_after_close_reproducible` 的执行部分通过。这不改变当前 REJECTED，但会削弱门槛对未来候选或代码回归的保护。

**最小修法：**

1. 按账户日期序列建立每个信号日的次一账户日；买入必须严格等于该日。
2. 卖出允许因停牌、跌停、T+1 或缺失 bar 顺延，但不得早于次一账户日；晚于该日时，应结合同一订单的逐日 pending 记录证明顺延原因，而不是接受任意更晚日期。
3. 用对应配置的 `open_fill` 规则或独立等价公式重算开盘滑点、合法范围钳制、费用及成交金额，至少核对成交价确由实际成交日开盘规则产生。
4. 新增 delayed buy、non-open fill、无 pending 证据的延迟卖出反例。若暂时只保留弱检查，则必须降低报告措辞，但仅改措辞不能满足“次日合法开盘”的冻结门槛语义。

### I3-1：持仓相关性跨不同持仓 episode 计算收益

**位置：**

- `/Users/admin/WorkSpace/trading/trading_finally/src/trend_system/risk.py:95-130`。
- `risk.py:99-100` 只保留 `date,code`，丢弃 `entry_date`。
- `risk.py:105-108` 随后按 `code` 做 `pct_change`，因此同一股票第二次及以后建仓的首日，会与上一次平仓前的最后持仓日直接计算“日收益”，中间真实的空仓间隔被忽略。
- 错误汇总进入 `90_研究结论.md:165-169`、`92_负结果与后续边界.md:64-70` 及 `results/robustness/risk/`。

**真实输出量化证据：**

基线持仓表有 6,662 行、161 只股票，其中 8 只股票存在多个 `entry_date` 持仓区间。保留 `entry_date` 并改按 `(code,entry_date)` 计算后，其他合并、重叠日和 Pearson 方法保持不变，得到：

| 指标 | 当前报告 | 隔离持仓区间后 |
| --- | ---: | ---: |
| 可计算股票对 | 277 | 275 |
| 相关系数均值 | 0.1790956860 | 0.1990378072 |
| 相关系数中位数 | 0.1805377454 | 0.1881276815 |

275 个共同股票对中有 13 个数值改变，另有 2 个当前股票对在正确隔离后不再满足至少两个重叠收益的条件。

**影响：**

已披露的持仓相关性不是报告所称的单纯“同期持仓股票日收益”，混入了跨空仓期价格变化。这是实际风险交付数值错误。该诊断不参与机械门槛，因此不改变收益、MDD 或 REJECTED，但 I3 的风险披露仍未完整修复。

**最小修法：**

保留 `entry_date`，按 `(code,entry_date)` 分组计算 `signal_close.pct_change(fill_method=None)`，再构造日期×股票收益矩阵。增加同一股票两次独立建仓且中间价格跳变的反例，断言第二次建仓首日为 NaN，不与上一区间形成收益。随后从已冻结基线 Parquet 重算风险表，更新 90/92 和受影响哈希。

## 3. 已通过的 scoped 修复项

### 3.1 C1：多股票、日期主序和热身期错位已修复

- `signals.py:20-46` 对非空、唯一 `(date,code)`、行数、outer merge 键集合及日期集合做硬断言。
- `signals.py:49-71` 分别按面板和信号自身日期筛选，再以 `(date,code)` 一对一连接；不依赖原行位置。
- `broker.py:78-80` 在真实账户入口再次断言。
- 主基线 `metrics.py:475-477`、邻域 `robustness.py:677-680`、双源 `robustness.py:473-482`、历史压力 `robustness.py:1022-1027` 均走统一键选择；主窗口、双倍成本、随机路径和组合变体复用这些结果或入口。
- 真实 `results/baseline/run_manifest.json` 记录主窗口 panel/signals 均为 7,335,154 行，键差异 0，窗口 2016-10-13 至 2026-10-29。
- 多股票、日期主序、含热身期测试和双源同类测试均通过。

本项没有发现残留的位置掩码路径。

### 3.2 MDD 新政策实现与文档一致

`robustness.py:140-148` 的边界正确：

- MDD≤0.10 且其他门槛全通过：UNREJECTED。
- 0.10<MDD≤0.20 且其他门槛全通过：BOUNDARY。
- MDD>0.20 或任一其他门槛失败：REJECTED。

`tests/test_robustness.py:88-100` 精确覆盖 0.10、0.100001、0.20、0.200001。双源 MDD 上限和 `drawdown_boundary` 在 `robustness.py:960-978` 均为 20%。设计、计划、预注册、90、92 和报告模板中的主路径政策一致；残留的 8% 仅以“原内部分界已取消”出现，不是现行阈值。

### 3.3 退市日终止停牌窄例外及下一日零残值正确

- `broker.py:188-208` 仅在 `data_valid=False`、`trade_status=0`、当日等于 `out_date`、原价 OHLC/prev_close 全缺失且 `execution_conflict=False` 时允许继续。
- `broker.py:224-243` 保留下一账户日零残值注销，不把退市日当成提前可成交信息。
- 真实 000043.SZ：2019-12-16 是 out_date，停牌、无效且五个原价字段全缺失；当日 100 股按最后实际收盘 20.25 元作陈旧估值，市值 2,025 元。
- 2019-12-17 记录 `delisting_zero_residual`，100 股、金额 0；闭合交易净盈亏 -1,906.94 元、收益率 -100%。
- 额外近似负例 `trade_status=1`、错误 out_date、仅一个价格字段存在、`execution_conflict=True` 均中止；其他无效数据没有放宽。

### 3.4 I3 中其余风险披露及停牌折价口径通过

- 169 笔入场中 158 笔 signed risk 非正，初始风险定义和信号日权益分母一致。
- 同日集中退出、跳空、跌停顺延和最低佣金均有明细；337 笔基线成交全部触发 5 元最低佣金。
- 166 个账户日有陈旧估值，最大陈旧市值 7,098 元。
- `risk.py:213-234` 只对当日陈旧持仓市值作 10%/20% 会计折价；没有生成订单或改变交易路径。
- 两档折价后期末权益均 1,469.89 元，点估计 MDD 均 0.926955672192279；90/92 明确说明这不代表停牌无风险，也不进入门槛。

本节的通过不排除上述 I3-1 相关性统计缺陷。

## 4. deferred Minor 逐项处置

| 项目 | 复审结论 | 证据 |
| --- | --- | --- |
| Task 3 信号反例 | 已解决 | `tests/test_signals.py:109-147` 分别覆盖价格/MA、斜率、通道和缺失状态；聚焦运行通过，缺失状态 fail-closed。 |
| Task 6 任意日历年限制 | 可保留，不构成未修复问题 | `metrics.py:58-96` 与 `robustness.py:277-310` 均按实际日历年份分组，没有硬编码 2016；首末年及部分年度有标记，2016/2026 在 92 中明确为部分年度。账户年度权益门槛没有退回 realized PnL。 |
| 随机排序逐日日志 | 已解决 | 20 份 `random_order_log.csv` 均存在；每份 74,492 行，字段为 `path_id,date,code,random_rank`，键非空且 `(date,code)` 无重复，各路径入场键集合一致。 |
| 5 个输入 Parquet SHA256 | 已解决 | `results/robustness/input_manifest.json` 中 main_panel、hithink_signal_panel、index_panel、corporate_actions、stress_panel 五项，独立流式重算 SHA256 全部匹配。 |
| double-cost sizing/fill | 已解决 | `robustness.py:382-398` 同一路径覆盖 sizing 和 fill；`tests/test_execution.py:63-94` 直接验证双倍最低佣金与 20bp。真实基线 337 笔佣金均 5 元，双倍成本 327 笔均 10 元。 |
| 双源初始化 | 已解决 | 两源账户均为 2017-08-31 至 2026-10-29、2,221 个账户日；首日现金/权益均 20,000 元，应收、持仓市值和 position_count 均 0。信号差异 154 行已落盘。 |
| 执行完整性单一布尔 | 部分解决，仍有 I2-1 | 键、现金、整手、合法范围和权益已拆成独立字段且真实输出通过；但次日时序和开盘价反例仍误过，不能标记完成。 |

双源 318 笔经济成交逐字段相同，与两源收益和 MDD 相同一致；154 行逐键信号差异另行保存。当前没有把“经济结果相同”误写成“信号完全相同”。

## 5. 关键结果与报告一致性

独立基线与稳健性基线的小汇总一致：期末权益 1,469.89 元、总收益 -0.9265055、MDD 0.926955672192279、169 个闭合往返、胜率 0.23076923076923078。

实际 `results/robustness/summary.json` 有精确 10 个门槛，8 个失败；基线和双倍成本保存的执行证据字段齐全。20 个随机排序日志和 5 个输入哈希均与实现报告一致。90/92 对负结果、毛盈亏仍含滑点、单条随机入场正路径、短共同窗口、单源历史压力及停止边界的措辞没有越界。

## 6. 本次实际验证

- 聚焦运行 9 项既有 unittest，全部通过：C1 主窗口、双源键对齐、MDD 边界、门槛 ID、Task 3 两组反例、双倍成本 sizing/fill、退市终止停牌及无效数据中止。
- 另运行内存中的 delayed-buy、non-open-fill、畸形门槛和退市近似负例；前三项暴露上述 Important，退市负例全部按预期中止。
- 只读取小 JSON/CSV、账户 Parquet、相关风险表及单个真实股票事件行；没有回显大表。
- 没有重跑完整真实基线、53 个变体、历史压力或 1000 次自助。

## 7. 重新 Approved 的必要条件

1. 修复 I1-1，严格验证 `passed` 布尔类型并加入畸形值反例。
2. 修复 I2-1，使证据名实相符地验证买入次一账户日、卖出合法顺延及实际开盘成交规则，并加入负例。
3. 修复 I3-1，按持仓区间计算收益相关性，刷新风险明细、90/92 和受影响哈希。
4. 运行聚焦回归并做一次同范围复审。只有上述 Important 全部关闭，才可改为 Approved。
