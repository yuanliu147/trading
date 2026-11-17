# -*- coding: utf-8 -*-
"""70/15 策略的今日信号（默认取本地 DuckDB 最新交易日）。

规则（与 2026-10-22 研究目录一致）：
- 基础信号：上升结构（C>MA60、MA20>MA60、MA60>20日前）；
  趋势年龄 5–30；ATR/收盘≤4.4%；20日无涨停；行业宽度≥60%；
  行业20日收益>5%；距60日高点>-8%；20日均成交额≥3000万；收盘≥2；市场宽度>50%。
- 70/15 子池：20日均成交额≤7000万；后复权收盘≤15（注意：回测口径是后复权，不是实际价）。
- 买入：信号次日开盘；高开>5%、一字涨停、停牌不买；最多2只；成交额升序优先；
  单票 45% × min(1, 20%/市场20日波动)；持有20日收盘卖；-20%灾难止损。
- 板块：全板块文件保留研究口径；“主板可买”文件只保留 600/601/603/605/000/001/002/003，
  排除创业板 300/301、科创板 688/689、北交所 8/4/920，并排除名称含 ST。
"""
import os
import sys
import pickle
import duckdb
import numpy as np
import pandas as pd

BASE = '/Users/admin/WorkSpace/trading'
DIR = os.path.join(BASE, '2026-10-22_持仓天数_涨跌路径')
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
NAMES = os.path.join(DIR, '股票名称_2026-10-22.csv')

def board_of(c):
    c = str(c)
    if c.startswith(('600', '601', '603', '605')):
        return '沪主板'
    if c.startswith(('000', '001', '002', '003')):
        return '深主板'
    if c.startswith(('300', '301')):
        return '创业板'
    if c.startswith(('688', '689')):
        return '科创板'
    if c.startswith(('8', '4', '920')):
        return '北交所'
    return '其他'


con = duckdb.connect(DB, read_only=True)
end = con.execute("select max(date) from raw_kline_daily").fetchone()[0]
if len(sys.argv) > 1:
    end = pd.Timestamp(sys.argv[1]).date()
start = pd.Timestamp(end) - pd.Timedelta(days=160)
d = con.execute("""
    select thscode, date, open, high, low, close, amount
    from v_daily_hfq
    where date >= ? and date <= ?
""", [str(start), str(end)]).df()
q = con.execute("select thscode, close from v_daily_qfq where date = ?", [str(end)]).df()
con.close()
d['date'] = pd.to_datetime(d['date'])
print('date', end, 'rows', len(d), 'codes', d.thscode.nunique())

C = d.pivot(index='date', columns='thscode', values='close').sort_index()
H = d.pivot(index='date', columns='thscode', values='high').sort_index()
L = d.pivot(index='date', columns='thscode', values='low').sort_index()
A = d.pivot(index='date', columns='thscode', values='amount').sort_index()
codes = C.columns
dates = C.index
last = dates[-1]
boards = pd.Series([board_of(c) for c in codes], index=codes)
main_ok = boards.isin(['沪主板', '深主板'])

ma20 = C.rolling(20, min_periods=20).mean()
ma60 = C.rolling(60, min_periods=60).mean()
up = ((C > ma60) & (ma20 > ma60) & (ma60 > ma60.shift(20))).fillna(False)
up_t = up.iloc[-1]
cs = up.cumsum()
reset = cs.where(~up).ffill().fillna(0)
age = (cs - reset).iloc[-1]
r20 = (C / C.shift(20) - 1).iloc[-1]

prevC = C.shift(1)
tr = pd.concat([(H - L).stack(), (H - prevC).abs().stack(), (L - prevC).abs().stack()],
               axis=1).max(axis=1).unstack()
atr = (tr.rolling(14, min_periods=10).mean() / C).iloc[-1]
amt20 = A.rolling(20, min_periods=15).mean().iloc[-1]
dd60 = (C / C.rolling(60, min_periods=30).max() - 1).iloc[-1]

r1 = C.pct_change()
lim = pd.Series(0.098, index=codes)
lim[[c.startswith(('300', '301')) for c in codes]] = 0.196
lim[[c.startswith('688') for c in codes]] = 0.196
lim[[c.startswith(('8', '4')) for c in codes]] = 0.296
limit_up = r1.ge(lim, axis=1).fillna(False)
lu20 = limit_up.astype(float).rolling(20, min_periods=10).sum().iloc[-1]

# 行业（时点申万，取面板最后一日；7天内不变）
P = pickle.load(open(PANEL, 'rb'))
inds = P['meta']['industries']
ind_last = P['industry_id'].iloc[-1]
ind = pd.Series(0, index=codes, dtype=int)
for c in codes:
    if c in ind_last.index and pd.notna(ind_last[c]):
        ind[c] = int(ind_last[c])
del P

above = (C.iloc[-1] > ma20.iloc[-1]).where(C.iloc[-1].notna())
mb = above.mean()
sbr = pd.Series(np.nan, index=codes)
sr20 = pd.Series(np.nan, index=codes)
for k in range(1, len(inds) + 1):
    m = (ind == k)
    if m.any():
        sbr[m] = above[m].mean()
        sr20[m] = r20[m].mean()

# 条件
valid = up_t & (amt20 >= 3e7) & (C.iloc[-1] >= 2) & (age >= 5)
base = valid & (age <= 30) & (atr <= 0.044) & (lu20 == 0) & (sbr >= 0.6) & (sr20 > 0.05) & (dd60 > -0.08)
sig = base & (mb > 0.5) & (amt20 <= 70e6) & (C.iloc[-1] <= 15)
stock = valid & (age <= 30) & (atr <= 0.044) & (lu20 == 0) & (dd60 > -0.08)
near = stock & (amt20 <= 70e6) & (C.iloc[-1] <= 15)

qfq = q.set_index('thscode')['close']
names = pd.read_csv(NAMES, encoding='utf-8-sig').set_index('thscode')['name']
st = pd.Series([('ST' in str(names.get(c, '')).upper()) for c in codes], index=codes)
main_buy = main_ok & ~st
R = pd.DataFrame({'代码': codes, '名称': [names.get(c, '') for c in codes],
                  '板块': [boards.get(c, '') for c in codes],
                  '主板可买': main_buy.values,
                  '实际收盘': [qfq.get(c, np.nan) for c in codes],
                  '后复权收盘': C.iloc[-1].values,
                  '20日均成交额_万': amt20.values / 1e4,
                  'ATR占收盘': atr.values, '距60日高点': dd60.values,
                  '趋势年龄': age.values, '20日涨停次数': lu20.values,
                  '20日收益': r20.values,
                  '行业': [inds[int(ind[c]) - 1] if ind[c] > 0 else '' for c in codes],
                  '行业宽度': sbr.values, '行业20日收益': sr20.values})
R['行业宽度≥60%'] = R['行业宽度'] >= 0.6
R['行业20日>5%'] = R['行业20日收益'] > 0.05
R['有效信号'] = sig.values
R['70_15近信号'] = near.values

cols = ['代码', '名称', '板块', '主板可买', '实际收盘', '后复权收盘', '20日均成交额_万', 'ATR占收盘',
        '距60日高点', '趋势年龄', '20日涨停次数', '20日收益', '行业', '行业宽度', '行业20日收益',
        '行业宽度≥60%', '行业20日>5%', '有效信号']
N = R[near.fillna(False).values][cols].sort_values('20日均成交额_万')
V = R[sig.fillna(False).values][cols].sort_values('20日均成交额_万')
Nm = R[(near & main_buy).fillna(False).values][cols].sort_values('20日均成交额_万')
Vm = R[(sig & main_buy).fillna(False).values][cols].sort_values('20日均成交额_万')
N.to_csv(os.path.join(DIR, '今日70_15近信号_2026-10-22.csv'), index=False,
         encoding='utf-8-sig', float_format='%.4f')
V.to_csv(os.path.join(DIR, '今日70_15有效信号_2026-10-22.csv'), index=False,
         encoding='utf-8-sig', float_format='%.4f')
Nm.to_csv(os.path.join(DIR, '今日70_15主板近信号_2026-10-22.csv'), index=False,
          encoding='utf-8-sig', float_format='%.4f')
Vm.to_csv(os.path.join(DIR, '今日70_15主板有效信号_2026-10-22.csv'), index=False,
          encoding='utf-8-sig', float_format='%.4f')

# 行业闸门
rows = []
for k in range(1, len(inds) + 1):
    m = (ind == k)
    if m.sum() >= 5:
        rows.append(dict(行业=inds[k - 1], 股票数=int(m.sum()),
                         行业宽度=sbr[m].iloc[0], 行业20日收益=sr20[m].iloc[0],
                         宽度达标=bool(sbr[m].iloc[0] >= 0.6),
                         收益达标=bool(sr20[m].iloc[0] > 0.05)))
IND = pd.DataFrame(rows).sort_values('行业20日收益', ascending=False)
IND.to_csv(os.path.join(DIR, '今日行业闸门_2026-10-22.csv'), index=False,
           encoding='utf-8-sig', float_format='%.4f')

status = pd.DataFrame([dict(日期=str(last), 市场宽度=mb, 市场闸门=bool(mb > 0.5),
                            行业宽度达标数=int(IND['宽度达标'].sum()),
                            行业收益达标数=int(IND['收益达标'].sum()),
                            个股条件数=int(stock.sum()), 近信号数=int(near.sum()),
                            有效信号数=int(sig.sum()),
                            主板近信号数=int((near & main_buy).sum()),
                            主板可买数=int((sig & main_buy).sum()),
                            可买数=int((sig & main_buy).sum()))])
status.to_csv(os.path.join(DIR, '今日闸门_2026-10-22.csv'), index=False,
              encoding='utf-8-sig', float_format='%.4f')

print('\n市场宽度 %.4f，闸门 %s' % (mb, '开' if mb > 0.5 else '关'))
print('行业宽度达标 %d，行业收益达标 %d' % (IND['宽度达标'].sum(), IND['收益达标'].sum()))
print('个股条件（忽略行业/市场闸门） %d；70/15 近信号 %d；有效信号 %d' % (stock.sum(), near.sum(), sig.sum()))
print('其中主板近信号 %d；主板可买 %d' % ((near & main_buy).sum(), (sig & main_buy).sum()))
print('\n行业前10：')
print(IND.head(10).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n70/15 近信号（不是有效信号，全板块）：')
print(N.to_string(index=False, float_format=lambda v: f'{v:.3f}') if len(N) else 'NONE')
print('\n70/15 主板近信号（账户只能买主板时看这个）：')
print(Nm.to_string(index=False, float_format=lambda v: f'{v:.3f}') if len(Nm) else 'NONE')
print('\n全板块有效信号：')
print(V.to_string(index=False, float_format=lambda v: f'{v:.3f}') if len(V) else 'NONE')
print('\n主板可买有效信号：')
print(Vm.to_string(index=False, float_format=lambda v: f'{v:.3f}') if len(Vm) else 'NONE')
