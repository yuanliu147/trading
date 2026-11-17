### Task 8: 按裁决交付负结果或盘后观察工具

**Files:**
- Create: `trading_finally/README.md`
- Create: `trading_finally/91_系统手册.md`（仅 `UNREJECTED`）
- Create: `trading_finally/92_负结果与后续边界.md`（仅 `BOUNDARY/REJECTED`）
- Create: `trading_finally/03_盘后模拟协议.md`（仅 `UNREJECTED`）
- Create: `trading_finally/scripts/daily_plan.py`（仅 `UNREJECTED`）
- Create: `trading_finally/src/trend_system/daily.py`（仅 `UNREJECTED`）
- Create: `trading_finally/state/positions.csv`（仅 `UNREJECTED`）
- Create: `trading_finally/state/forward_ledger.csv`（仅 `UNREJECTED`）
- Test: `trading_finally/tests/test_delivery.py`
- Test: `trading_finally/tests/test_daily_plan.py`（仅 `UNREJECTED`）

**Interfaces:**
- Consumes: Task 7 裁决、冻结面板、信号和账户规则。
- Produces（仅未否决）: `src/trend_system/daily.py` 中的 `generate_daily_plan(date, positions, panel, signals, config) -> DailyPlan`；`scripts/daily_plan.py` 只解析参数并写文件。输出数据健康、现金/槽位、持仓正常/预警/失效、次日订单和失败处理，不连接券商。

- [ ] **Step 1: 为裁决对应交付写失败测试**

若未否决，测试数据日期缺失或未来日期会拒绝出单、持仓退出优先、新买不超过预算、输出不含上涨概率；预警仅当收盘价距 20 日退出通道不足 2%，不影响交易。若边界或否决，测试交付物只解释失败门槛和不可继续调参的边界，不生成实盘候选手册。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_daily_plan -v`（未否决时）
Expected: FAIL，提示日报模块不存在。

- [ ] **Step 3: 实现对应交付**

未否决时，手册完整记录规则、费用、无法成交、公司行动、暂停条件和 30 笔前瞻协议。`forward_ledger.csv` 预建字段，30 笔只用于执行一致性验收；未解释信号差异和越规则订单必须为 0，前瞻回撤超过 10% 提前停止，真实成交与滑点不得从纸面模拟推断。

边界或否决时，只写负结果和后续研究限制，不通过新增指标、市场闸门或参数搜索继续“救系统”。

- [ ] **Step 4: 完成全量复现检查**

Run: `PYTHONPATH=src python3 -B -m unittest discover -s tests -v`
Expected: 全部 PASS。

Run: `PYTHONPATH=src python3 -B scripts/run_baseline.py --config config/baseline.json`
Expected: 在相同输入和代码下，基线 manifest 的数据哈希、配置和核心指标与 Task 6 一致；README 给出从面板构建到最终报告的全部命令。

- [ ] **Step 5: 做一次独立整体审查**

审查范围仅限 `trading_finally/`：核对前视、复权与原价、账户现金、退市/跌停、历史污染声明、报告与原始结果是否一致。任何关键问题先修复并重跑受影响任务，不宣称完成直到验证输出真实通过。
