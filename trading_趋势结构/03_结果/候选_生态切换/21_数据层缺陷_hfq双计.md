# 冻结层 `hfq_*` 列的复权口径缺陷（本族实测）

> 发现者：候选_生态切换（不同思维研究员）
> 发现时间：2026-10-28，**在本族第一次跑组合收益时**（净值出现单日 -17.7%，
> 逐笔核对后发现不是策略问题，而是价格口径）
> 影响范围：任何直接使用 `01_数据/frozen/frozen.duckdb` 的 `hfq_open/high/low/close`
> 做**收益率、均线、突破、ATR、波动**的研究。本族已改用 `factor` 本身，不受影响。
> 本文件只报告事实与可复现证据，**不修改数据层**（由数据负责人独占）。

## 1. 一句话

冻结层的 `hfq_close = close × factor`，其中 `factor = Π(1+pct_chg)` 已经是**累计总收益指数**。
两者相乘会把每一天的涨跌幅**再算一遍**，导致 `hfq` 的日收益约等于 `pct_chg` 的 **2 倍**。
`factor` 本身是正确的；`hfq_*` 作为“复权价”不可用于收益计算。

## 2. 实测（冻结层，主板，2010-01-01..2020-12-31，200 只抽样）

复现脚本（只读）：

```python
import duckdb
con = duckdb.connect('01_数据/frozen/frozen.duckdb', read_only=True)
df = con.execute("""
with x as (
 select symbol,date,close,preclose,pct_chg,factor,hfq_close,
        lag(hfq_close) over (partition by symbol order by date) h0,
        lag(factor) over (partition by symbol order by date) f0
 from daily
 where board='主板' and date between '2010-01-01' and '2020-12-31'
   and symbol in (select symbol from daily where board='主板' group by 1 order by 1 limit 200)
)
select symbol,date,pct_chg, factor/f0-1 as factor_ret, hfq_close/h0-1 as hfq_ret
from x where f0>0 and h0>0 and pct_chg is not null
""").df()
```

| 指标 | 实测 | 应有 |
|---|---:|---:|
| 样本证券日 | 492,949 | — |
| `factor` 日收益 vs `pct_chg/100` 最大绝对差 | **2.1e-15** | 0 |
| `hfq` 日收益 / `pct_chg` 中位数 | **2.0009** | 1.0 |
| `hfq` 日收益与 `pct_chg` 相关系数 | 0.722 | 1.0 |
| `hfq` 日收益与 `pct_chg` 差 >1bp 的占比 | **89.15%** | 0% |
| 最大绝对差 | **75.96**（即 7596 个百分点） | 0 |

## 3. 单笔可核对案例

`002386.SZ`，2020-02-03（春节后首个交易日，全市场跌停）：

| 字段 | 值 |
|---|---:|
| close / preclose | 5.18 / 5.75 |
| pct_chg | -9.913% |
| factor 前一日 → 当日 | 0.345194 → 0.310975（-9.91%） |
| hfq_close 前一日 → 当日 | 1.9849 → 1.6109（**-18.84%**） |
| close × factor | 5.18 × 0.310975 = 1.6109 ✓（乘法本身没错） |

同一天同一只股票，官方跌幅 -9.91%，`hfq` 跌幅 -18.84%。这不是除权，是重复计算。

## 4. 为什么会这样

数据字典 §5.1 写的是：

```
r_t = pct_chg/100
factor_t = Π_{s<=t}(1+r_s)
hfq_px_t = raw_px_t × factor_t
```

`factor_t` 是**总收益指数**（相当于把分红再投资的累计净值），不是“除权调整因子”。
正确的复权价应是 `factor_t × 该股首日原始价`（同股常数缩放），或直接把 `factor_t`
当作价格序列使用（均线、突破、ATR 对同股常数缩放不变）。
`raw_t × factor_t` 相当于 `价格 × 累计收益`，量纲上就是“价格 × 指数”，其收益必然双计。

## 5. 对本族的影响与处理

- 本族 `02_代码/候选_生态切换.py` 的 `compute_features()` 已改用：
  - 收益/均线/突破/ATR：`factor` 本身（记为 `hc`）；
  - 日内价：`factor × open/close`、`factor × high/close`、`factor × low/close`；
  - 事件收益：`(factor_exit × open_exit/close_exit) / (factor_entry × open_entry/close_entry) - 1`。
- 修正前后对比（C3 双 sleeve，2008-2020）：修正前 CAGR -1.01%、MDD 84.3%；
  修正 hfq 后（但尚未修正信号单位混用）为 -14.15%/87.2%；
  最终修正信号单位后为 **-16.13%/91.6%**。修正前的结果是**价格口径错误**，已作废。
- 本族所有落盘结果均为最终口径。修正前的中间文件已删除/覆盖。

## 6. 对其他研究者的提示（只提示，不修改他人文件）

1. `03_结果/趋势族_开发验证规范.md` 声明使用 Baostock `adjustflag=1` 后复权，**不是**
   冻结层 `hfq_*`，因此不受本缺陷影响（但需自行确认缓存来源）。
2. `02_代码/退出_数据层.py` 的 `close_hfq` 来自 `01_数据/exit_cache.duckdb`，若该缓存
   由冻结层 `hfq_close` 构建，则同样双计；若由 Baostock flag1 构建，则不受影响。
   本族未读取该缓存，无法替其背书。
3. 任何用 `frozen.duckdb` 的 `daily_qfq` 视图做收益的代码都会中招（同一乘法）。

## 7. 建议的最小修复（供数据负责人决策）

```sql
-- 正确：以首日原始价为锚，factor 只做总收益指数
first_close = first_value(close) over (partition by symbol order by date)
hfq_close_correct = factor * first_close
hfq_open_correct  = factor * open / close * first_close
-- 或：直接暴露 factor 作为 total_return_index，不提供 hfq_* 价格
```

修复后必须重跑 `01_自检_数据.py` 中“hfq = close × factor”的检查（该检查只验证乘法，
不验证收益），并新增一项：`factor` 日收益 vs `pct_chg` 的逐日一致性。
