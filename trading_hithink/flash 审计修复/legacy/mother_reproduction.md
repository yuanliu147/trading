# 数据与复现

## 1. 数据来源

| 项目 | 内容 |
|---|---|
| 数据库 | `/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb`（同花顺本地库，约 1.86 GB） |
| 原始表/视图 | `raw_kline_daily`、`calc_adjust_factor_daily`、`raw_adjustment_events`、`dim_symbol`、`v_daily`、`v_daily_hfq`、`v_symbol` |
| 窗口 | 2016-10-13 至 2026-10-23 |
| 股票池 | 沪深主板 600/601/603/605/000/001/002/003；3,197 只 |
| 面板 | `01_数据/research_panel.parquet`，6,133,972 行，约 2.8 GB |
| 外部数据 | **无**。未使用 baostock/Tushare/远程行业指数；全部字段来自本地 DuckDB 及其派生特征 |

> `01_数据/features.parquet` 是指向既有特征工程的软链接（`trading_hithink/data/work/features.parquet`，4.1 GB），仅用于追溯；最终回测只读 `research_panel.parquet`，不依赖它。

## 2. 研究面板怎么来的

1. 既有特征文件 `features.parquet` 由本地 DuckDB 构建：日线 + 后复权 + 公司行为 + 全市场宽度/情绪/周月聚合，共 184 列。
2. `02_代码/build_research_panel.py` 从中截取 90+ 列，限定主板、上市≥250 日，并用窗口函数生成：
   - `ret_o3/5/10/20/30`：次日开盘买入、持有 N 日收盘的收益；
   - `mfe5/10/20`、`mae5/10/20`：持有期最大有利/不利变动；
   - `close_3/5/...`：未来收盘。
3. 面板落盘为 `01_数据/research_panel.parquet`。

### 核心字段定义

| 字段 | 定义 |
|---|---|
| `return_250d` | 后复权收盘 / 250 日前收盘 − 1 |
| `drawdown250` | 后复权收盘 / 过去 250 日最高后复权收盘 − 1 |
| `rsi14` | 14 日 RSI（后复权收盘） |
| `close_location` | `(close-low)/(high-low)`，当日收盘在振幅中的位置 |
| `market_return20` | 全主板等权日收益累乘 20 日 |
| `breadth20` | 全主板收盘在 MA20 上方的比例 |
| `amount20` | 20 日平均成交额（元） |
| `up_limit_05` | 5% 涨跌停样态（ST 代理） |

## 3. 脚本顺序

| 顺序 | 脚本 | 作用 |
|---:|---|---|
| 1 | `02_代码/build_research_panel.py` | 从 features 生成研究面板（已运行，产物在 `01_数据/`） |
| 2 | `02_代码/scan_events.py` | 多角度事件扫描（MACD/趋势/突破/涨停/跳空/K线/反转） |
| 3 | `02_代码/scan_reversal.py` | 反转族扩展（恐慌、深跌、跌停、连跌、强势股洗盘） |
| 4 | `02_代码/panic_lib.py` | 数据加载、信号、逐日组合模拟（T+1、涨跌停、止盈、费用） |
| 5 | `02_代码/robustness.py` | 参数悬崖、成本、延迟、止损、随机排序 |
| 6 | `02_代码/tp_grid.py` | 止盈×持有×仓位×延迟×止损网格 |
| 7 | `02_代码/walkforward.py` | 走步选参（2017–21→22–23；2017–23→24–26） |
| 8 | `02_代码/grid_all_periods.py` | 576 组全参数三段结果 |
| 9 | `02_代码/audit_c.py` | 候选逐笔、回撤、剔除最赚、随机选股、阈值扰动 |
| 10 | `02_代码/ablation.py` | 条件消融、确认入场、剔除盈利笔 |
| 11 | `02_代码/tail_check.py` | 灾难止损、跌停到期 |
| 12 | `02_代码/final_backtest.py` | 复权最终回测（99 笔） |
| 13 | `02_代码/actual_engine.py` | **旧版实际价引擎，已废弃**（送转/成本 bug，不要使用） |
| 13a | `02_代码/actual_engine_fixed.py` | **修正版实际价引擎**：100 股整手、送转/分红/成本基准修正 |
| 13b | `02_代码/make_actual_fixed_excel.py` | 修正版实际价 Excel |
| 14 | `02_代码/final_metrics2.py` | 复权/实际价年化、回撤、年度、月度、回撤区间 |
| 15 | `02_代码/current_signal.py` | 2026-10-23 当前状态 |
| 16 | `02_代码/postfix_audit.py` | 修正后分段/悬崖/剔除/随机 |
| 17 | `02_代码/verify_delivery.py` | 交付数字校验 |
| 18 | `02_代码/make_charts.py` | 图表 |
| 19 | `02_代码/make_excel.py` | 中文 Excel |

## 4. 复现命令

```bash
cd "/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30"
python3 02_代码/scan_events.py
python3 02_代码/scan_reversal.py
python3 02_代码/robustness.py
python3 02_代码/tp_grid.py
python3 02_代码/walkforward.py
python3 02_代码/grid_all_periods.py
python3 02_代码/audit_c.py
python3 02_代码/ablation.py
python3 02_代码/tail_check.py
python3 02_代码/final_backtest.py
python3 02_代码/actual_engine.py
python3 02_代码/final_metrics2.py
python3 02_代码/current_signal.py
python3 02_代码/postfix_audit.py
python3 02_代码/verify_delivery.py
python3 02_代码/make_charts.py
python3 02_代码/make_excel.py
```

依赖：Python 3.13、pandas、numpy、pyarrow、duckdb、matplotlib、openpyxl。

## 5. 已做的检查

| 检查 | 结果 |
|---|---|
| 面板行数 | 6,133,972 |
| 主板标的 | 3,197（原始 5,579 只 A 股中） |
| 退市股 | **无**（所有标的 2026-08-31 后仍有行情） |
| ST | 5% 涨跌停样态代理；非历史 ST 名称 |
| T+1 | 买入当日不可卖；止盈/止损仅从次日起 |
| 一字涨停 | 入场日实际开盘 ≥ 前收×1.098 不买 |
| 跌停到期 | 99 笔中 1 笔（000595，2025-01-27 一字跌停）；回测按跌停价，严格顺延影响约 0.3pp 年化 |
| 成本 | 佣金万 2.5（最低 5）+ 印花千 0.5 + 滑点千 1；0.5%/边压力仍 35.1% |
| 随机排序 | 20 种子：均值 28.6%、中位 30.0% |
| 剔除最赚 5 笔 | 32.6%（复权） |
| 走步 | 验证 26.0%；2024–2026 检验 48.5% |
| 576 组参数 | 三段≥30% 共 7 组；本策略最低段最高 |

## 6. 关键文件哈希

见 `05_证据/关键文件sha256.txt`。

## 7. 限制

- 本地库 10 年、当前上市口径；无退市股、无历史 ST 名称、无盘中数据。
- 行业分类/行业指数缺失；板块映射只做沪深主板等权/宽度/涨跌停占比/情绪，不是真正的行业轮动。
- 复权收益含分红；修正版实际价整手已加现金分红，与复权差异很小。
- **旧版实际价 42.6% / 236.2 万不可用**：002365 十转五被调成 3,868.95 股（应 3,750），并把调整后股数×原始买入价当成本，出现 -24,533 假亏损；002607 漏 4,095 元分红。修正版 **40.6% / 208.4 万 / 71.7% 胜率 / 6.72% 平均单笔 / 4.08 利润因子**；详见 `03_结果/实际价_修正说明.md` 与 `03_结果/实际价_修正版.xlsx`。
- 止盈按日线高低点近似；一字板/跌停只能日线近似。
- 2024 年 +95.1% 是异常大年；2021 年 +1.5% 是常态年份参考。

## 8. 部分区间标签

- `2017-10（10-19起）`、`2017（10-19起）`：数据从 2017-10-19 开始，是部分区间。
- `2026-10（截至10-23）`、`2026（截至10-23）`：数据到 2026-10-23，是月/年至今，不是 10-31 或 12-31 的未来数据。
- 月度表用 `resample('ME').last()` 分桶，桶标签是自然月末，但桶内最后一条权益是 10-23；数值是 10-01 至 10-23 的月内收益。
