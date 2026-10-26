# -*- coding: utf-8 -*-
"""603778.SH：K 线、信号日特征与横截面条件分析。

只读：
- 2026-10-24_主板恐慌反转_年化30/01_数据/research_panel.parquet
- 06_不同信号日起点/03_结果/逐日首笔_逐只.csv
- 07_市场阶段适配/03_结果/全部信号日逐只_带阶段.csv
- 2026-10-24_主板恐慌反转_年化30/03_结果/当前信号_2026-10-26.csv

输出：03_结果/*.csv、关键数字。
口径：
- K 线用后复权；图中按 2026-10-23 的实际价/后复权价折算成“当前价等价”，
  只用于可读性，历史价格不是当时成交价。
- 信号收益 = 15 日、T+3 开盘、+20% 止盈、0.175%/边。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
FLASH = HERE.parents[3]
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

PANEL = FLASH / '2026-10-24_主板恐慌反转_年化30' / '01_数据' / 'research_panel.parquet'
SIGNAL = ROOT / '06_不同信号日起点' / '03_结果' / '逐日首笔_逐只.csv'
ALLSIG = ROOT / '07_市场阶段适配' / '03_结果' / '全部信号日逐只_带阶段.csv'
SIG26 = FLASH / '2026-10-24_主板恐慌反转_年化30' / '03_结果' / '当前信号_2026-10-26.csv'

FEATS = [
    'thscode', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'hfq_open', 'hfq_high', 'hfq_low', 'hfq_close',
    'rsi14', 'macd_dif_pct', 'macd_dea_pct', 'macd_hist_pct', 'macd_hist_delta',
    'macd_golden_cross', 'macd_dead_cross', 'close_location',
    'drawdown250', 'return_250d', 'return_120d', 'return_60d', 'return_20d',
    'volatility20', 'atr20_pct', 'range_pct', 'candle_body', 'upper_shadow',
    'lower_shadow', 'hammer_shape', 'bullish_engulfing', 'inside_bar', 'narrow_range7',
    'amount_ratio1_20', 'amount_ratio5_20', 'amount_ratio20_60',
    'close_ma5', 'close_ma10', 'close_ma20', 'close_ma50', 'close_ma60',
    'close_ma120', 'close_ma250', 'market_return20', 'breadth20', 'sentiment_score',
]
df = pd.read_parquet(PANEL, columns=FEATS)
df['date'] = pd.to_datetime(df['date'])
g = df[df['thscode'] == '603778.SH'].sort_values('date').reset_index(drop=True).copy()

# 当前价等价折算
CUR_ACT = float(g.loc[g['date'] == '2026-10-23', 'close'].iloc[0])
CUR_HFQ = float(g.loc[g['date'] == '2026-10-23', 'hfq_close'].iloc[0])
FACTOR = CUR_ACT / CUR_HFQ
for c in ['hfq_open', 'hfq_high', 'hfq_low', 'hfq_close']:
    g[c.replace('hfq_', 'px_')] = g[c] * FACTOR
g['prior_low'] = g['low'].shift(1)
g['prior_close'] = g['close'].shift(1)
g['BOR'] = (g['low'] < g['prior_low']) & (g['close'] > g['prior_close']) & (g['close'] > g['open'])
g['ma5'] = g['px_close'].rolling(5).mean()
g['ma10'] = g['px_close'].rolling(10).mean()
g['ma20'] = g['px_close'].rolling(20).mean()
g['ma60'] = g['px_close'].rolling(60).mean()
g['ma250'] = g['px_close'].rolling(250).mean()
g['vol_ma20'] = g['volume'].rolling(20).mean()
g['vol_ma5'] = g['volume'].rolling(5).mean()
# RSI 用涨跌幅计算
chg = g['px_close'].diff()
up = chg.clip(lower=0).rolling(14).mean()
dn = (-chg.clip(upper=0)).rolling(14).mean()
g['rsi_calc'] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

# 关键价位
peak = g.loc[g['px_high'].idxmax()]
sub = g[(g['date'] >= '2025-01-01') & (g['date'] <= '2025-06-30')]
run_low = sub.loc[sub['px_low'].idxmin()]
jul = g[(g['date'] >= '2026-07-01') & (g['date'] <= '2026-07-31')]
jul_low = jul.loc[jul['px_low'].idxmin()]
aug = g[(g['date'] >= '2026-08-01') & (g['date'] <= '2026-08-31')]
aug_high = aug.loc[aug['px_high'].idxmax()]
sep = g[(g['date'] >= '2026-09-01') & (g['date'] <= '2026-09-30')]
sep_high = sep.loc[sep['px_high'].idxmax()]
sep_low = sep.loc[sep['px_low'].idxmin()]
oct = g[(g['date'] >= '2026-10-01') & (g['date'] <= '2026-10-23')]
oct_low = oct.loc[oct['px_low'].idxmin()]
last = g.loc[g['date'] == '2026-10-23'].iloc[0]

fib_rows = []
for level, name in [(0.618, '61.8%'), (0.786, '78.6%'), (0.886, '88.6%'), (1.0, '100%')]:
    price = peak['px_high'] - level * (peak['px_high'] - run_low['px_low'])
    fib_rows.append({'类型': f'回撤 {name}', '价格': price,
                     '距现价': price / last['px_close'] - 1})
key_rows = [
    ('历史高点', peak['date'].date().isoformat(), peak['px_high'], peak['px_high'] / last['px_close'] - 1),
    ('本轮起涨低点', run_low['date'].date().isoformat(), run_low['px_low'], run_low['px_low'] / last['px_close'] - 1),
    ('7月低点', jul_low['date'].date().isoformat(), jul_low['px_low'], jul_low['px_low'] / last['px_close'] - 1),
    ('8月反弹高点', aug_high['date'].date().isoformat(), aug_high['px_high'], aug_high['px_high'] / last['px_close'] - 1),
    ('9月反弹高点', sep_high['date'].date().isoformat(), sep_high['px_high'], sep_high['px_high'] / last['px_close'] - 1),
    ('9月低点', sep_low['date'].date().isoformat(), sep_low['px_low'], sep_low['px_low'] / last['px_close'] - 1),
    ('10月低点', oct_low['date'].date().isoformat(), oct_low['px_low'], oct_low['px_low'] / last['px_close'] - 1),
    ('当前收盘', '2026-10-23', last['px_close'], 0.0),
    ('MA5', '2026-10-23', last['ma5'], last['ma5'] / last['px_close'] - 1),
    ('MA10', '2026-10-23', last['ma10'], last['ma10'] / last['px_close'] - 1),
    ('MA20', '2026-10-23', last['ma20'], last['ma20'] / last['px_close'] - 1),
    ('MA60', '2026-10-23', last['ma60'], last['ma60'] / last['px_close'] - 1),
    ('MA250', '2026-10-23', last['ma250'], last['ma250'] / last['px_close'] - 1),
    ('+20%目标（按10/23收盘）', '—', last['px_close'] * 1.2, 0.2),
]
keys = pd.DataFrame(key_rows, columns=['类型', '日期', '价格', '距现价'])
keys = pd.concat([keys, pd.DataFrame(fib_rows)], ignore_index=True)
keys['价格'] = keys['价格'].round(3)
keys['距现价'] = keys['距现价'].round(4)
keys.to_csv(RES / '关键价位.csv', index=False, encoding='utf-8-sig', float_format='%.4f')

# 信号日表
s = pd.read_csv(SIGNAL, parse_dates=['signal_date', 'entry_date', 'exit_date'])
s = s[s['code'] == '603778.SH'].copy()
s['rank'] = s['rank'].astype(int)
s = s.merge(g[FEATS + ['BOR', 'px_close', 'px_low', 'px_high', 'ma5', 'ma10', 'ma20', 'ma60', 'ma250']],
            left_on='signal_date', right_on='date', how='left', suffixes=('', '_k'))
s['案例'] = np.where(s['ret'] > 0, '赢', np.where(s['ret'].notna(), '输', '待定'))
s.loc[s['signal_date'] == '2026-10-15', '案例'] = '浮亏/未完成'
s.loc[s['signal_date'] == '2026-10-23', '案例'] = '待入场'
s = s.sort_values('signal_date')
signal_cols = ['signal_date', 'rank', 'n_signals', '案例', 'entry_date', 'exit_date', 'exit_reason', 'ret',
               'drawdown250', 'return_250d', 'rsi14', 'close_location', 'macd_hist_pct',
               'macd_hist_delta', 'amount_ratio1_20', 'amount_ratio5_20', 'amount_ratio20_60',
               'close_ma5', 'close_ma10', 'close_ma20', 'close_ma60', 'close_ma250',
               'volatility20', 'atr20_pct', 'range_pct', 'candle_body', 'upper_shadow',
               'lower_shadow', 'hammer_shape', 'bullish_engulfing', 'inside_bar',
               'narrow_range7', 'BOR', 'market_return20', 'breadth20']
S = s[signal_cols].copy()
S.columns = ['信号日', '排名', '当日信号数', '案例', '买入日', '卖出日', '退出原因', '15日收益',
             '250日回撤', '250日涨幅', 'RSI14', '收盘位置', 'MACD柱', 'MACD柱变化',
             '量比1/20', '量比5/20', '量比20/60', '距MA5', '距MA10', '距MA20', '距MA60', '距MA250',
             '20日波动率', 'ATR20', '振幅', '实体', '上影', '下影', '锤子线', '吞没', '内包', '窄幅7',
             'BOR', '市场20日', '市场宽度20']
S.to_csv(RES / '信号日特征.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 10/26 部分快照
sig26 = pd.read_csv(SIG26)
r26 = sig26[sig26['thscode'] == '603778.SH'].iloc[0]
cur_rows = []
for label, r in [('2026-10-23', last)]:
    cur_rows.append({
        '日期': label, '排名': 1, '实际收盘': r['close'], '250日回撤': r['drawdown250'],
        '250日涨幅': r['return_250d'], 'RSI14': r['rsi14'], '收盘位置': r['close_location'],
        'MACD柱': r['macd_hist_pct'], 'MACD柱变化': r['macd_hist_delta'],
        '量比1/20': r['amount_ratio1_20'], '量比5/20': r['amount_ratio5_20'],
        '距MA5': r['close_ma5'], '距MA10': r['close_ma10'], '距MA20': r['close_ma20'],
        '距MA60': r['close_ma60'], '距MA250': r['close_ma250'],
        '20日波动率': r['volatility20'], 'ATR20': r['atr20_pct'], '振幅': r['range_pct'],
        '下影': r['lower_shadow'], '上影': r['upper_shadow'], '实体': r['candle_body'],
        '市场20日': r['market_return20'], '市场宽度20': r['breadth20'],
    })
cur_rows.append({
    '日期': '2026-10-26', '排名': int(r26['rank']), '实际收盘': r26['actual_close'],
    '250日回撤': r26['drawdown250'], '250日涨幅': np.nan, 'RSI14': r26['rsi14'],
    '收盘位置': r26['close_location'], 'MACD柱': np.nan, 'MACD柱变化': np.nan,
    '量比1/20': np.nan, '量比5/20': np.nan, '距MA5': np.nan, '距MA10': np.nan,
    '距MA20': np.nan, '距MA60': np.nan, '距MA250': np.nan, '20日波动率': np.nan,
    'ATR20': np.nan, '振幅': np.nan, '下影': np.nan, '上影': np.nan, '实体': np.nan,
    '市场20日': r26['market_return20'], '市场宽度20': r26['breadth20'],
})
cur = pd.DataFrame(cur_rows)
cur.to_csv(RES / '当前快照.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 横截面条件
allsig = pd.read_csv(ALLSIG, parse_dates=['signal_date'])
allsig = allsig[allsig['complete'] == True].copy()  # noqa: E712
allf = df.copy()
allf['date'] = pd.to_datetime(allf['date'])
allf = allf.sort_values(['thscode', 'date'])
allf['prior_low'] = allf.groupby('thscode')['low'].shift(1)
allf['prior_close'] = allf.groupby('thscode')['close'].shift(1)
allf['BOR'] = (allf['low'] < allf['prior_low']) & (allf['close'] > allf['prior_close']) & (allf['close'] > allf['open'])
d = allsig.merge(allf, left_on=['code', 'signal_date'], right_on=['thscode', 'date'], how='left', suffixes=('', '_f'))
d['cl'] = d['close_location'].fillna(d['close_loc'])
d['mh'] = d['macd_hist_pct']
d['vr'] = d['amount_ratio1_20']
d['vr5'] = d['amount_ratio5_20']
d['ma20d'] = d['close_ma20']
d['dd'] = d['dd250']
d['r250'] = d['return_250d']

buckets = [
    ('全部信号', pd.Series(True, index=d.index)),
    ('深回撤 ≤-65%', d['dd'] <= -0.65),
    ('深回撤 + 收盘位置>0.8', (d['dd'] <= -0.65) & (d['cl'] > 0.8)),
    ('深回撤 + 收盘位置≤0.8', (d['dd'] <= -0.65) & (d['cl'] <= 0.8)),
    ('深回撤 + BOR', (d['dd'] <= -0.65) & (d['BOR'] == True)),
    ('深回撤 + BOR + 高收盘', (d['dd'] <= -0.65) & (d['BOR'] == True) & (d['cl'] > 0.8)),
    ('深回撤 + 高收盘 + MACD≤0', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['mh'] <= 0)),
    ('深回撤 + 高收盘 + MACD>0', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['mh'] > 0)),
    ('深回撤 + 高收盘 + 量比<1', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['vr'] < 1)),
    ('深回撤 + 高收盘 + 量比≥1', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['vr'] >= 1)),
    ('深回撤 + 高收盘 + RSI≥30', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['rsi14'] >= 30)),
    ('深回撤 + 高收盘 + MA20≥-10%', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['ma20d'] >= -0.10)),
    ('深回撤 + 高收盘 + r250>100%', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['r250'] > 1)),
    ('当前近似组合*', (d['dd'] <= -0.65) & (d['cl'] > 0.8) & (d['mh'] <= 0) & (d['vr'] < 1) & (d['rsi14'] >= 30) & (d['ma20d'] >= -0.10)),
]
rows = []
for name, mask in buckets:
    q = d[mask & d['ret'].notna()]
    rows.append({
        '条件': name, '样本': len(q), '均值': q['ret'].mean(), '中位': q['ret'].median(),
        '胜率': (q['ret'] > 0).mean(), '最差': q['ret'].min(), '最好': q['ret'].max(),
        'p10': q['ret'].quantile(.10),
    })
conds = pd.DataFrame(rows)
conds.to_csv(RES / '条件统计.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 信号日对比简表（7月 + 10月）
comp_cols = ['signal_date', 'rank', 'ret', 'drawdown250', 'close_location', 'rsi14',
             'macd_hist_pct', 'macd_hist_delta', 'amount_ratio1_20', 'close_ma20',
             'return_250d', 'lower_shadow', 'hammer_shape']
comp = s[comp_cols].copy()
comp.to_csv(RES / '信号日对比.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 指标 JSON
metrics = {
    '当前日期': '2026-10-23',
    '折算因子': round(FACTOR, 6),
    '峰值': {'日期': peak['date'].date().isoformat(), '价格': round(float(peak['px_high']), 3)},
    '起涨低点': {'日期': run_low['date'].date().isoformat(), '价格': round(float(run_low['px_low']), 3)},
    '7月低点': {'日期': jul_low['date'].date().isoformat(), '价格': round(float(jul_low['px_low']), 3)},
    '10月低点': {'日期': oct_low['date'].date().isoformat(), '价格': round(float(oct_low['px_low']), 3)},
    '当前收盘': round(float(last['px_close']), 3),
    '当前回撤': round(float(last['drawdown250']), 4),
    '当前RSI': round(float(last['rsi14']), 2),
    '当前收盘位置': round(float(last['close_location']), 4),
    '当前MACD柱': round(float(last['macd_hist_pct']), 5),
    '当前量比1/20': round(float(last['amount_ratio1_20']), 3),
    '当前MA20': round(float(last['ma20']), 3),
    '当前MA60': round(float(last['ma60']), 3),
    '当前MA250': round(float(last['ma250']), 3),
    '条件统计': conds.round(4).to_dict('records'),
    '口径': '后复权价格按 2026-10-23 实际价/后复权价折算为当前价等价；历史价格不是当时成交价。',
}
with open(RES / '指标.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print('factor', round(FACTOR, 4), 'current', CUR_ACT, 'hfq', round(CUR_HFQ, 3))
print('\n信号日特征')
print(S.round(3).to_string(index=False))
print('\n当前快照')
print(cur.round(4).to_string(index=False))
print('\n关键价位')
print(keys.round(3).to_string(index=False))
print('\n条件统计')
print(conds.round(4).to_string(index=False))
print('\n指标')
print(json.dumps(metrics, ensure_ascii=False, indent=2)[:2000])
