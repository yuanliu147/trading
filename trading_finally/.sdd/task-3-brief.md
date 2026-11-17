### Task 3: 实现无未来函数的唯一基线信号

**Files:**
- Create: `trading_finally/src/trend_system/signals.py`
- Test: `trading_finally/tests/test_signals.py`

**Interfaces:**
- Consumes: Task 2 面板与 `ResearchConfig`。
- Produces: `compute_signals(panel: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame`，保留原键并增加 `ma200/channel_high_120/channel_low_120/rank_return_120/entry_signal/exit_signal/st_exit`。
- 信号调用方只能取得每个交易日收盘后的行；不得向该函数传入账户未来状态。

- [ ] **Step 1: 写通道移位和点时截断测试**

构造至少 240 个交易日的单调序列，断言 120 日高点和 20 日低点都由 `shift(1)` 后的窗口计算；当日突然创最高价格只影响当日信号，不改写昨日信号。分别在除权日前截断面板，比较截断前后该日前所有信号，结果必须完全一致。历史 `is_st` 只影响当日及之后，不得使用当前状态回填。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_signals -v`
Expected: FAIL，提示 `compute_signals` 不存在。

- [ ] **Step 3: 实现固定信号**

按股票、日期排序后计算：`ma200=signal_close.rolling(200).mean()`；斜率比较 `ma200 > ma200.shift(20)`；`channel_high_120=signal_high.shift(1).rolling(120).max()`；`channel_low_120=signal_low.shift(1).rolling(20).min()`；排名收益为 `signal_close/signal_close.shift(120)-1`。

入场要求 `signal_close > ma200`、均线斜率向上、`signal_close > channel_high_120`、非 ST、可交易、`data_valid=True` 且无 `execution_conflict`，并且日期位于配置的主业绩窗口；复权模型差异单独保留用于 Task 7 对照，不作为入场阻断。热身日期可以计算指标但不生成入场。退出为 `signal_close < channel_low_120`；ST 退出单独标记。任何所需窗口或字段缺失时不得生成入场。

- [ ] **Step 4: 运行信号与数据回归测试**

Run: `PYTHONPATH=src python3 -B -m unittest tests.test_signals tests.test_data -v`
Expected: PASS；另在真实面板上断言入场信号日期不早于主窗口、同键无重复、入场行的排名收益非空。

---

