### Task 2: 构建双源审计和点时行情面板

**Files:**
- Create: `trading_finally/sql/baostock_panel.sql`
- Create: `trading_finally/sql/hithink_overlap.sql`
- Create: `trading_finally/src/trend_system/data.py`
- Create: `trading_finally/scripts/build_panel.py`
- Test: `trading_finally/tests/test_data.py`
- Modify: `trading_finally/02_预注册_极简趋势消融.md`（记录任何策略运行前的执行冲突/复权模型冲突拆分与双源敏感性修订）

**Interfaces:**
- Consumes: `ResearchConfig`；baostock `stock_kline_d/stock_basic/index_kline_d`；同花顺 `v_daily/v_daily_hfq/raw_adjustment_events`。
- Produces: `build_panel(config) -> PanelBuildResult`；`load_panel(path) -> pd.DataFrame`。
- 面板唯一键为 `(date, code)`，字段至少为：`open/high/low/close/prev_close`（原价）、`signal_open/signal_high/signal_low/signal_close`（后复权）、`volume/amount/trade_status/is_st/ipo_date/out_date/data_valid/execution_conflict/adjustment_source_conflict/source_conflict`。
- 输出 `data/main_panel.parquet`、`data/stress_panel.parquet`、`data/corporate_actions.parquet`、`data/hithink_signal_panel.parquet`、`results/data_audit.json`、`results/data_conflicts.parquet` 和 `results/data_exclusions.csv`。

- [ ] **Step 1: 写代码转换、复权缺口和双源冲突测试**

测试 `sh.600000 -> 600000.SH`、`sz.000001 -> 000001.SZ`；原价存在而后复权缺失时保留该行且 `data_valid=False`，不得用未来日期的复权行回填；同键重复、负价格和无法转换的板块代码抛错。公司行动连接只允许使用当日已经生效的记录。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_data -v`
Expected: FAIL，提示数据模块不存在。

- [ ] **Step 3: 实现只读面板构建**

baostock 的 `adjustflag=3` 提供原行情，`adjustflag=1` 提供后复权信号行情；同花顺用于重叠段审计，不把两套复权绝对价格直接拼接。复权因子由同日 `hfq_close/raw_close` 得到，原价收盘无效时整行标记无效。

重叠段原价差异阈值为价格 `0.011` 元、成交量相对差 `1e-5`、成交额相对差 `1e-6`；超阈值设 `source_conflict=True`，不静默择值。原价 OHLC 差异另设 `execution_conflict=True`，复权序列差异另设 `adjustment_source_conflict=True`，二者不得混为同一种可交易性判断。复权序列按股票归一化后比较，缺失或冲突不得通过删除整只股票掩盖。

另存全部重叠行的同花顺后复权 OHLC 面板，不只保存冲突行，供 Task 7 在共同可计算窗口做独立信号敏感性。审计需包含归一化复权相对差分布，区分绝对尺度、精度和实质序列差异。

公司行动输出保留现金分红、送转、配股字段和来源日期；指数另存用于基准和状态报告。SQL 与行数、窗口、数据库 SHA256 写入审计 JSON。

- [ ] **Step 4: 运行单元测试并确认通过**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_data -v`
Expected: PASS。

- [ ] **Step 5: 构建真实面板并验收**

Run: `PYTHONPATH=src python3 -B scripts/build_panel.py --config config/baseline.json`
Expected: 主面板最大日期为 `2026-10-29`，主键无重复；审计文件包含两源行数、2016-10-13 起重叠行数、58 只复权不完整股票的处理计数、全部冲突计数和主板/退市样本计数。大结果只落盘。

---

