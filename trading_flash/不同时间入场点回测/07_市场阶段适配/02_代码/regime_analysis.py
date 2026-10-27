# -*- coding: utf-8 -*-
"""
市场阶段适配分析：策略在什么环境下赚钱，2026 为什么难受。

输入（只读）：
- 2026-10-24_主板恐慌反转_年化30/03_结果/最终_每日权益.csv、最终_年度.csv、最终_逐笔.csv
- 06_不同信号日起点/03_结果/逐日首笔_逐只.csv、起点汇总.csv、起点固定持有期.csv
- trading_hithink/初始研究/data/work/market_daily.parquet（沪深主板等权市场）

输出：
- 03_结果/*.csv、指标.json
- 04_图表/*.png（由 make_charts.py 生成）
- 市场阶段适配.xlsx（由 make_excel.py 生成）
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
FLASH = ROOT.parent
MOTHER = FLASH / '2026-10-24_主板恐慌反转_年化30'
SRC06 = ROOT / '06_不同信号日起点' / '03_结果'
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

MARKET = FLASH.parent / 'trading_hithink' / '初始研究' / 'data' / 'work' / 'market_daily.parquet'

# ------------------------------------------------------------------ 市场
m = pd.read_parquet(MARKET, columns=[
    'date', 'equal_weight_index', 'market_return', 'market_return20',
    'market_return60', 'breadth20', 'market_volatility20'])
m['date'] = pd.to_datetime(m['date'])
m = m.sort_values('date').reset_index(drop=True)
m['ma250'] = m['equal_weight_index'].rolling(250).mean()
m['mkt_ret250'] = m['equal_weight_index'].pct_change(250)
m['mkt_ret120'] = m['equal_weight_index'].pct_change(120)
m['mkt_dd250'] = m['equal_weight_index'] / m['equal_weight_index'].rolling(250).max() - 1
m['mkt_above250'] = (m['equal_weight_index'] > m['ma250']).astype(int)


def phase_of(row):
    if pd.isna(row['mkt_ret250']):
        return 'NA'
    if row['mkt_above250'] == 1 and row['mkt_ret250'] > 0:
        return 'A 牛市上升段急跌'
    if row['mkt_above250'] == 0 and row['mkt_ret250'] > 0:
        return 'B 高位转弱/跌破年线'
    if row['mkt_above250'] == 1 and row['mkt_ret250'] <= 0:
        return 'C 熊市反弹'
    return 'D 熊市下跌'


m['阶段'] = m.apply(phase_of, axis=1)
m['是否信号日'] = 0

# ------------------------------------------------------------------ 信号日逐只
d = pd.read_csv(SRC06 / '逐日首笔_逐只.csv',
                parse_dates=['signal_date', 'entry_date', 'exit_date'])
d['complete'] = d['complete'].astype(bool)
d['y'] = d['signal_date'].dt.year
mcols = ['date', 'equal_weight_index', 'ma250', 'mkt_ret250', 'mkt_ret120',
         'mkt_dd250', 'mkt_above250', '阶段', 'market_return20',
         'market_return60', 'breadth20', 'market_volatility20']
d = d.merge(m[mcols], left_on='signal_date', right_on='date', how='left')

# 信号日到 15 个交易日后的市场收益（用于描述反弹是否到来；事后变量）
md = m.set_index('date')['equal_weight_index']


def fwd_market(start, n):
    dates = m['date'].values
    pos = int(np.searchsorted(dates, np.datetime64(start)))
    if pos + n >= len(dates):
        return np.nan
    return float(md.iloc[pos + n] / md.iloc[pos] - 1)


def fwd_between(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    if a not in md.index or b not in md.index:
        return np.nan
    return float(md.loc[b] / md.loc[a] - 1)


d['信号后15日市场'] = d['signal_date'].apply(lambda x: fwd_market(x, 15))
d['持有期市场'] = [fwd_between(a, b) for a, b in zip(d['entry_date'], d['exit_date'])]
d['是否完整'] = np.where(d['complete'], '是', '否')
d['阶段'] = d['阶段'].fillna('NA')
vol_med = d.loc[d['complete'], 'market_volatility20'].median()
d['波动分组'] = np.where(
    d['market_volatility20'] >= vol_med, '高波动', '低波动')

# 全部信号日（含未完成）落到市场日表
sig_days = set(pd.to_datetime(d['signal_date']).dt.normalize())
m['是否信号日'] = m['date'].isin(sig_days).astype(int)

# ------------------------------------------------------------------ 年度
mf = pd.read_csv(RES.parent / '06_不同信号日起点' / '03_结果' / '起点汇总.csv',
                 parse_dates=['signal_date']) if False else None
cohort_all = pd.read_csv(SRC06 / '起点汇总.csv', parse_dates=['signal_date'])
cohort_all['y'] = cohort_all['signal_date'].dt.year

annual_strategy = pd.read_csv(MOTHER / '03_结果' / '最终_年度.csv',
                              parse_dates=['date'])
annual_strategy['y'] = annual_strategy['date'].dt.year
annual_strategy = annual_strategy.set_index('y')['return']

# 市场年度：与账户年度对齐（上年最后一个交易日 -> 本年最后一个交易日）
mkt_year = {}
for y in range(2017, 2027):
    if y == 2017:
        a = m[(m['date'] >= '2017-11-23') & (m['date'].dt.year == 2017)]
    elif y == 2026:
        a = m[(m['date'] >= '2025-12-31') & (m['date'] <= '2026-10-23')]
    else:
        start = m[m['date'] < f'{y}-01-01']['date'].max()
        a = m[(m['date'] >= start) & (m['date'] <= f'{y}-12-31')]
    if len(a):
        mkt_year[y] = float(a['equal_weight_index'].iloc[-1] / a['equal_weight_index'].iloc[0] - 1)

annual_rows = []
for y in range(2017, 2027):
    z = d[d['y'] == y]
    zc = z[z['complete']]
    annual_rows.append(dict(
        年份=y,
        策略收益=float(annual_strategy.get(y, np.nan)),
        市场等权=float(mkt_year.get(y, np.nan)),
        超额百分点=float((annual_strategy.get(y, np.nan) - mkt_year.get(y, np.nan)) * 100),
        信号日数=int(cohort_all[cohort_all['y'] == y].shape[0]),
        完整单笔=int(len(zc)),
        单笔均值=float(zc['ret'].mean()) if len(zc) else np.nan,
        胜率=float((zc['ret'] > 0).mean()) if len(zc) else np.nan,
        止盈占比=float((zc['exit_reason'] == 'target').mean()) if len(zc) else np.nan,
        最差单笔=float(zc['ret'].min()) if len(zc) else np.nan,
    ))
annual = pd.DataFrame(annual_rows)
annual['策略收益'] = annual['策略收益'].fillna(0.0)

# ------------------------------------------------------------------ 阶段统计
def summary(z):
    z = z.dropna(subset=['ret'])
    if len(z) == 0:
        return dict(笔数=0)
    return dict(
        笔数=int(len(z)),
        平均=float(z['ret'].mean()),
        中位=float(z['ret'].median()),
        胜率=float((z['ret'] > 0).mean()),
        p10=float(z['ret'].quantile(.10)),
        p90=float(z['ret'].quantile(.90)),
        止盈占比=float((z['exit_reason'] == 'target').mean()),
        持有期市场=float(z['持有期市场'].mean()),
    )


def add_ex2026(z, rec):
    z2 = z[(z['complete']) & (z['y'] < 2026)].dropna(subset=['ret'])
    if len(z2):
        rec['剔除2026平均'] = float(z2['ret'].mean())
        rec['剔除2026胜率'] = float((z2['ret'] > 0).mean())
        rec['剔除2026笔数'] = int(len(z2))
    return rec


phase_rows = []
for name, z in [('全部', d[d['complete']]),
                ('A 牛市上升段急跌', d[(d['complete']) & (d['阶段'].str.startswith('A'))]),
                ('B 高位转弱/跌破年线', d[(d['complete']) & (d['阶段'].str.startswith('B'))]),
                ('C 熊市反弹', d[(d['complete']) & (d['阶段'].str.startswith('C'))]),
                ('D 熊市下跌', d[(d['complete']) & (d['阶段'].str.startswith('D'))]),
                ('年线上方', d[(d['complete']) & (d['mkt_above250'] == 1)]),
                ('年线下方', d[(d['complete']) & (d['mkt_above250'] == 0)]),
                ('高波动', d[(d['complete']) & (d['波动分组'] == '高波动')]),
                ('低波动', d[(d['complete']) & (d['波动分组'] == '低波动')]),
                ('市场250日回撤<-20%', d[(d['complete']) & (d['mkt_dd250'] < -.20)]),
                ('市场250日回撤-20~-10%', d[(d['complete']) & (d['mkt_dd250'].between(-.20, -.10, inclusive='left'))]),
                ('市场250日回撤>-10%', d[(d['complete']) & (d['mkt_dd250'] > -.10)]),
                ('持有期市场上涨', d[(d['complete']) & (d['持有期市场'] > 0)]),
                ('持有期市场下跌', d[(d['complete']) & (d['持有期市场'] <= 0)]),
                ]:
    rec = dict(分类=name)
    rec.update(summary(z))
    rec = add_ex2026(z, rec)
    phase_rows.append(rec)
phase = pd.DataFrame(phase_rows)

# ------------------------------------------------------------------ 波动 × 250日涨幅
z = d[d['complete']].copy()
z['250日涨幅分组'] = pd.cut(z['mkt_ret250'], [-1, 0, .10, 2],
                       labels=['<0', '0~10%', '>10%'])
vol_rows = []
for vb in ['低波动', '高波动']:
    for rb in ['<0', '0~10%', '>10%']:
        zz = z[(z['波动分组'] == vb) & (z['250日涨幅分组'] == rb)]
        rec = dict(波动=vb, 市场250日涨幅=rb)
        rec.update(summary(zz))
        rec = add_ex2026(zz, rec)
        vol_rows.append(rec)
vol = pd.DataFrame(vol_rows)

# 年线位置 × 波动
vol2_rows = []
for ab in [1, 0]:
    for vb in ['低波动', '高波动']:
        zz = z[(z['mkt_above250'] == ab) & (z['波动分组'] == vb)]
        rec = dict(年线位置='年线上方' if ab else '年线下方', 波动=vb)
        rec.update(summary(zz))
        rec = add_ex2026(zz, rec)
        vol2_rows.append(rec)
vol2 = pd.DataFrame(vol2_rows)

# ------------------------------------------------------------------ 持有期市场分位 + 回归
zz = d[d['complete']].dropna(subset=['ret', '持有期市场']).copy()
zz['五分位'] = pd.qcut(zz['持有期市场'], 5, labels=False, duplicates='drop') + 1
q_rows = []
for q, g in zz.groupby('五分位'):
    q_rows.append(dict(
        分位=int(q), 笔数=int(len(g)),
        持有期市场均值=float(g['持有期市场'].mean()),
        平均=float(g['ret'].mean()),
        胜率=float((g['ret'] > 0).mean()),
    ))
quint = pd.DataFrame(q_rows)
beta, alpha = np.polyfit(zz['持有期市场'], zz['ret'], 1)
corr = float(np.corrcoef(zz['持有期市场'], zz['ret'])[0, 1])
reg = dict(beta=float(beta), alpha=float(alpha), corr=corr, r2=float(corr ** 2), n=int(len(zz)))

# ------------------------------------------------------------------ 2026 信号逐只
x26 = d[d['y'] == 2026].copy()
x26['月份'] = x26['signal_date'].dt.strftime('%Y-%m')
m26 = x26[x26['complete']].copy()
mon_rows = []
for mo, g in m26.groupby('月份'):
    mon_rows.append(dict(
        月份=mo, 笔数=int(len(g)), 平均=float(g['ret'].mean()),
        中位=float(g['ret'].median()), 胜率=float((g['ret'] > 0).mean()),
        止盈占比=float((g['exit_reason'] == 'target').mean()),
        最差=float(g['ret'].min()), 最好=float(g['ret'].max()),
    ))
mon26 = pd.DataFrame(mon_rows)

# ------------------------------------------------------------------ 2026 起点
f = pd.read_csv(SRC06.parent / '03_结果' / '起点固定持有期.csv',
                parse_dates=['signal_date']) if False else pd.read_csv(
    OUT / '03_结果' / '起点固定持有期.csv', parse_dates=['signal_date'])
f['y'] = f['signal_date'].dt.year
c26 = cohort_all[cohort_all['y'] == 2026].merge(
    f, on='signal_date', how='left', suffixes=('', '_fh'))
c26['是否亏损'] = np.where(c26['total_ret'] < 0, '是', '否')
c26 = c26.sort_values('signal_date')

# 固定持有期对比
fh_rows = []
for h in [20, 60, 120, 250]:
    col = f'ret_{h}'
    ddcol = f'dd_{h}'
    hist = f.loc[f['y'] < 2026, col].dropna()
    cur = f.loc[f['y'] == 2026, col].dropna()
    for label, s, dd in [('历史 2017-2025', hist, f.loc[f['y'] < 2026, ddcol].dropna()),
                         ('2026', cur, f.loc[f['y'] == 2026, ddcol].dropna())]:
        fh_rows.append(dict(
            持有交易日=h, 样本=label, 起点数=int(len(s)),
            平均=float(s.mean()) if len(s) else np.nan,
            中位=float(s.median()) if len(s) else np.nan,
            p5=float(s.quantile(.05)) if len(s) else np.nan,
            p95=float(s.quantile(.95)) if len(s) else np.nan,
            亏损占比=float((s < 0).mean()) if len(s) else np.nan,
            最差=float(s.min()) if len(s) else np.nan,
            最差回撤=float(dd.min()) if len(dd) else np.nan,
        ))
fh = pd.DataFrame(fh_rows)
fh['2026中位在历史分位'] = np.nan
for h in [20, 60, 120, 250]:
    hist = f.loc[f['y'] < 2026, f'ret_{h}'].dropna()
    cur = f.loc[f['y'] == 2026, f'ret_{h}'].dropna()
    if len(hist) and len(cur):
        fh.loc[(fh['持有交易日'] == h) & (fh['样本'] == '2026'), '2026中位在历史分位'] = float((hist < cur.median()).mean())

# ------------------------------------------------------------------ 回撤事实
eq = pd.read_csv(MOTHER / '03_结果' / '最终_每日权益.csv', parse_dates=['date']).set_index('date')
eq = eq.sort_index()
e = eq['equity']
dd = e / e.cummax() - 1
trough = dd.idxmin()
peak = e.loc[:trough].idxmax()
rec = e.loc[trough:][e.loc[trough:] >= e.loc[peak]].index.min()
e26 = e[e.index >= '2025-12-31']
dd26 = e26 / e26.cummax() - 1
trough26 = dd26.idxmin()
peak26 = e26.loc[:trough26].idxmax()
rec26 = e26.loc[trough26:][e26.loc[trough26:] >= e26.loc[peak26]].index.min()
e26m = e26.resample('ME').last().pct_change()
mother_trades = pd.read_csv(MOTHER / '03_结果' / '最终_逐笔.csv',
                            parse_dates=['signal_date', 'entry_date', 'exit_date'])
mother_worst = mother_trades.nsmallest(1, 'ret').iloc[0]
mother_2026 = mother_trades[mother_trades['signal_date'].dt.year == 2026]
mother_2026_worst = mother_2026.nsmallest(1, 'ret').iloc[0]
worst60 = f.dropna(subset=['ret_60']).nsmallest(1, 'ret_60').iloc[0]
worst60dd = f.dropna(subset=['dd_60']).nsmallest(1, 'dd_60').iloc[0]
worst_trade = d[d['complete']].nsmallest(1, 'ret').iloc[0]
dd_rows = [
    dict(项目='连续账户全期最大回撤', 数值=float(dd.min()), 日期=f'{peak.date()} → {trough.date()}',
         修复日期=str(rec.date()) if pd.notna(rec) else '',
         说明='2017-11-23 起，10 万本金，2 仓 50%，后复权；全期最大回撤就发生在 2026 年'),
    dict(项目='2026 年账户收益（至 10-23）', 数值=float(e26.iloc[-1] / e26.iloc[0] - 1), 日期='2025-12-31 → 2026-10-23',
         修复日期='', 说明='不是亏损；但 4/16 到 6/30 有一段 -16.0% 的回撤'),
    dict(项目='2026 年最大回撤', 数值=float(dd26.min()), 日期=f'{peak26.date()} → {trough26.date()}',
         修复日期=str(rec26.date()) if pd.notna(rec26) else '', 说明='8/6 修复，10/20 创出新高'),
    dict(项目='2026 单笔最差（全部信号）', 数值=float(worst_trade['ret']),
         日期=f"{worst_trade['signal_date'].date()} {worst_trade['code']}", 修复日期='',
         说明='002342.SZ，15 日持有到期，期间无普通止损'),
    dict(项目='母路径单笔最差（99 笔）', 数值=float(mother_worst['ret']),
         日期=f"{mother_worst['signal_date'].date()} {mother_worst['thscode']}", 修复日期='',
         说明='全期最差；母路径的 99 笔没有覆盖所有信号日'),
    dict(项目='母路径 2026 单笔最差', 数值=float(mother_2026_worst['ret']),
         日期=f"{mother_2026_worst['signal_date'].date()} {mother_2026_worst['thscode']}", 修复日期='',
         说明='母路径 2026 年只买到 11 笔，没有买到 2026-06-26 的 002342；路径不同，风险不同'),
    dict(项目='2026 起点 60 日最差', 数值=float(worst60['ret_60']),
         日期=f"{worst60['signal_date'].date()}", 修复日期='',
         说明=f"60 日最大回撤 {worst60dd['dd_60']:.1%}，全样本 60 日最差回撤"),
    dict(项目='2026 起点 60 日最差回撤', 数值=float(worst60dd['dd_60']),
         日期=f"{worst60dd['signal_date'].date()}", 修复日期='', 说明='同一笔起点'),
]
ddtab = pd.DataFrame(dd_rows)

# 2026 月度账户
m26rows = [dict(月份=str(idx.strftime('%Y-%m')), 账户收益=float(v))
           for idx, v in e26m.dropna().items()]

# ------------------------------------------------------------------ 长期对比
long_rows = []
for label, n in [('1年', 252), ('3年', 756), ('5年', 1260)]:
    s = cohort_all.dropna(subset=[f'ret_{label}']).copy()
    s['市场'] = s['signal_date'].apply(lambda x: fwd_market(x, n))
    s = s.dropna(subset=['市场'])
    long_rows.append(dict(
        期限=label, 起点数=int(len(s)),
        策略均值=float(s[f'ret_{label}'].mean()),
        策略中位=float(s[f'ret_{label}'].median()),
        市场均值=float(s['市场'].mean()),
        市场中位=float(s['市场'].median()),
        超额均值=float((s[f'ret_{label}'] - s['市场']).mean()),
        超额中位=float((s[f'ret_{label}'] - s['市场']).median()),
        跑赢占比=float((s[f'ret_{label}'] > s['市场']).mean()),
        相关性=float(np.corrcoef(s[f'ret_{label}'], s['市场'])[0, 1]),
    ))
long = pd.DataFrame(long_rows)

# ------------------------------------------------------------------ 写 CSV
annual.to_csv(RES / '年度对比.csv', index=False, encoding='utf-8-sig')
phase.to_csv(RES / '阶段统计.csv', index=False, encoding='utf-8-sig')
vol.to_csv(RES / '波动分组.csv', index=False, encoding='utf-8-sig')
vol2.to_csv(RES / '年线波动分组.csv', index=False, encoding='utf-8-sig')
quint.to_csv(RES / '持有期市场分位.csv', index=False, encoding='utf-8-sig')
mon26.to_csv(RES / '2026月度信号.csv', index=False, encoding='utf-8-sig')
c26.to_csv(RES / '2026起点.csv', index=False, encoding='utf-8-sig')
fh.to_csv(RES / '固定持有期对比.csv', index=False, encoding='utf-8-sig')
ddtab.to_csv(RES / '回撤事实.csv', index=False, encoding='utf-8-sig')
long.to_csv(RES / '长期对比.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(m26rows).to_csv(RES / '2026账户月度.csv', index=False, encoding='utf-8-sig')

# 信号逐只（含未完成）
out_cols = ['signal_date', 'n_signals', 'rank', 'code', 'dd250', 'rsi14',
            'close_loc', 'status', 'complete', 'entry_date', 'exit_date',
            'exit_reason', 'ret', '阶段', 'mkt_ret250', 'mkt_dd250',
            'market_volatility20', 'breadth20', '信号后15日市场', '持有期市场',
            '是否完整']
out_cols = [c for c in out_cols if c in d.columns]
x26[out_cols].to_csv(RES / '2026信号逐只.csv', index=False, encoding='utf-8-sig')
d[out_cols].to_csv(RES / '全部信号日逐只_带阶段.csv', index=False, encoding='utf-8-sig')

worst = d[d['complete']].nsmallest(12, 'ret')[['signal_date', 'code', 'entry_date',
                                                'exit_date', 'ret', 'exit_reason',
                                                '阶段', 'mkt_ret250', 'mkt_dd250',
                                                '持有期市场']]
worst.to_csv(RES / '最差交易.csv', index=False, encoding='utf-8-sig')
best = d[d['complete']].nlargest(8, 'ret')[['signal_date', 'code', 'entry_date',
                                            'exit_date', 'ret', 'exit_reason',
                                            '阶段', 'mkt_ret250', 'mkt_dd250',
                                            '持有期市场']]
best.to_csv(RES / '最好交易.csv', index=False, encoding='utf-8-sig')

market_out = m[['date', 'equal_weight_index', 'ma250', 'mkt_ret250', 'mkt_ret120',
                'mkt_dd250', 'mkt_above250', 'market_return20', 'market_return60',
                'breadth20', 'market_volatility20', '阶段', '是否信号日']].copy()
market_out.to_csv(RES / '市场阶段日度.csv', index=False, encoding='utf-8-sig')

# ------------------------------------------------------------------ 指标
metrics = dict(
    样本=dict(
        信号日总数=int(cohort_all.shape[0]),
        逐只总数=int(len(d)),
        完整逐只=int(len(d[d['complete']])),
        未完成=int(len(d[~d['complete']])),
        二零二六信号日=int(len(cohort_all[cohort_all['y'] == 2026])),
        二零二六逐只=int(len(x26)),
        二零二六完整=int(len(m26)),
    ),
    阶段={r['分类']: {k: (None if pd.isna(v) else float(v)) if k not in ('笔数',) else int(v)
                     for k, v in r.items() if k != '分类'} for _, r in phase.iterrows()},
    回归=reg,
    年度={int(r['年份']): dict(策略=float(r['策略收益']), 市场=float(r['市场等权']),
                             单笔=int(r['完整单笔']), 均值=float(r['单笔均值']),
                             胜率=float(r['胜率'])) for _, r in annual.iterrows()},
    二零二六=dict(
        账户收益=float(e26.iloc[-1] / e26.iloc[0] - 1),
        账户最大回撤=float(dd26.min()),
        回撤峰=f'{peak26.date()}', 回撤谷=f'{trough26.date()}',
        修复=f'{rec26.date()}',
        全期最大回撤=float(dd.min()),
        全期峰=f'{peak.date()}', 全期谷=f'{trough.date()}',
        起点数=int(len(c26)), 亏损起点=int((c26['total_ret'] < 0).sum()),
        起点中位=float(c26['total_ret'].median()),
        起点最差=float(c26['total_ret'].min()),
        起点最差日=str(c26.loc[c26['total_ret'].idxmin(), 'signal_date'].date()),
        起点最好=float(c26['total_ret'].max()),
        起点最好日=str(c26.loc[c26['total_ret'].idxmax(), 'signal_date'].date()),
        六月笔数=int(len(m26[m26['月份'] == '2026-06'])),
        六月均值=float(m26[m26['月份'] == '2026-06']['ret'].mean()),
        六月胜率=float((m26[m26['月份'] == '2026-06']['ret'] > 0).mean()),
        七月笔数=int(len(m26[m26['月份'] == '2026-07'])),
        七月均值=float(m26[m26['月份'] == '2026-07']['ret'].mean()),
        七月胜率=float((m26[m26['月份'] == '2026-07']['ret'] > 0).mean()),
        持有期市场=float(m26['持有期市场'].mean()),
        持有期市场历史=float(d[d['complete'] & (d['y'] < 2026)]['持有期市场'].mean()),
    ),
    固定期={f'{r["持有交易日"]}日-{r["样本"]}': dict(
        起点=int(r['起点数']), 中位=float(r['中位']), 亏损占比=float(r['亏损占比']),
        最差=float(r['最差']), 最差回撤=float(r['最差回撤']))
        for _, r in fh.iterrows()},
)
(RES / '指标.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

print('年度对比'); print(annual.round(4).to_string(index=False))
print('阶段统计'); print(phase.round(4).to_string(index=False))
print('波动分组'); print(vol.round(4).to_string(index=False))
print('回归', reg)
print('2026月度'); print(mon26.round(4).to_string(index=False))
print('2026起点 亏损', int((c26['total_ret'] < 0).sum()), '中位', round(c26['total_ret'].median(), 4))
print('固定持有期'); print(fh.round(4).to_string(index=False))
print('回撤事实'); print(ddtab.round(4).to_string(index=False))
print('长期对比'); print(long.round(4).to_string(index=False))
print('done')
