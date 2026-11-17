# -*- coding: utf-8 -*-
"""F2 主板专用策略的今日信号（默认本地 DuckDB 最新交易日）。

F2 规则：
- 市场：全A宽度20 > 50%（否则 0 买入）。
- 主板：沪 600/601/603/605；深 000/001/002/003；排除 ST（名称 + 5%触板代理）。
- 个股：上升结构；年龄 10–30；ATR/收盘<=4.4%；20日无涨停；
  DD60>=-15%；20日均成交额 3000–7000 万；后复权收盘<=15 且>=2；
  20日年化波动>=20%。
- 行业：申万一级宽度>=60%，行业20日收益>5%。
- 排序：市值升序（面板最后一日市值，按 qfq 价格变化粗调）。
- 买入：次日开盘；高开>5%、一字涨停、停牌不买；2 槽，单票 45% × 波动系数。
- 卖出：20日收盘；涨25%卖一半；盘中最高浮盈>=10% 后止损上移到成本；
  盘中最高浮盈>=25% 后止损改为最高价×0.90（10%移动止盈）；-20% 灾难止损。
"""
import os
import sys
import pickle
import duckdb
import numpy as np
import pandas as pd

BASE = '/Users/admin/WorkSpace/trading'
DIR = os.path.join(BASE, '2026-10-23_主板专用_策略优化')
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl')
NAMES = os.path.join(BASE, '2026-10-22_持仓天数_涨跌路径/股票名称_2026-10-22.csv')
ASOF = '2026-10-22'


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
end = pd.Timestamp(sys.argv[1]).date() if len(sys.argv) > 1 else pd.Timestamp(ASOF).date()
start = pd.Timestamp(end) - pd.Timedelta(days=180)
d = con.execute("""
    select thscode, date, open, high, low, close, amount
    from v_daily_hfq where date >= ? and date <= ?
""", [str(start), str(end)]).df()
q = con.execute("select thscode, close from v_daily_qfq where date = ?", [str(end)]).df()
q0 = con.execute("select thscode, close from v_daily_qfq where date = ?",
                 ['2026-10-13']).df()
con.close()
d['date'] = pd.to_datetime(d['date'])
C = d.pivot(index='date', columns='thscode', values='close').sort_index()
H = d.pivot(index='date', columns='thscode', values='high').sort_index()
L = d.pivot(index='date', columns='thscode', values='low').sort_index()
A = d.pivot(index='date', columns='thscode', values='amount').sort_index()
codes = C.columns
last = C.index[-1]
boards = pd.Series([board_of(c) for c in codes], index=codes)
main_ok = boards.isin(['沪主板', '深主板'])

ma20 = C.rolling(20, min_periods=20).mean()
ma60 = C.rolling(60, min_periods=60).mean()
up = ((C > ma60) & (ma20 > ma60) & (ma60 > ma60.shift(20))).fillna(False)
up_t = up.iloc[-1]
cs = up.cumsum()
age = (cs - cs.where(~up).ffill().fillna(0)).iloc[-1]
r20 = (C / C.shift(20) - 1).iloc[-1]

prevC = C.shift(1)
tr = pd.concat([(H - L).stack(), (H - prevC).abs().stack(), (L - prevC).abs().stack()],
               axis=1).max(axis=1).unstack()
atr = (tr.rolling(14, min_periods=10).mean() / C).iloc[-1]
amt20 = A.rolling(20, min_periods=15).mean().iloc[-1]
dd60 = (C / C.rolling(60, min_periods=30).max() - 1).iloc[-1]
r1 = C.pct_change()
vol20 = r1.rolling(20, min_periods=15).std().iloc[-1] * np.sqrt(252)

# 主板 10% 涨停；20日无涨停
lim = 0.098
limit_up = r1.ge(lim).fillna(False)
lu20 = limit_up.astype(float).rolling(20, min_periods=10).sum().iloc[-1]

# ST 代理：过去60日 5% 触板次数 >=2，或 1 次且60日最大绝对日收益<6%
lu5 = ((r1 > 0.045) & (r1 < 0.055) & ((C == H) | (H == L)))
ld5 = ((r1 < -0.045) & (r1 > -0.055) & ((C == L) | (H == L)))
c5 = (lu5.astype(float) + ld5.astype(float)).rolling(60, min_periods=1).sum().iloc[-1]
maxabs60 = r1.abs().rolling(60, min_periods=20).max().iloc[-1]
stx = (c5 >= 2) | ((c5 >= 1) & (maxabs60 < 0.060))

# 行业：面板最后一日
P = pickle.load(open(PANEL, 'rb'))
inds = P['meta']['industries']
ind_last = P['industry_id'].iloc[-1]
del P
# 2026-10-24 修正：市值改用补抓后的全覆盖矩阵（面板自带 mcap 只有约 1,280 只）
SIDE = os.path.join(DIR, '市值矩阵_全覆盖.pkl')
if os.path.exists(SIDE):
    mcap_last = pd.read_pickle(SIDE).iloc[-1]
else:
    P = pickle.load(open(PANEL, 'rb'))
    mcap_last = P['mcap'].iloc[-1]
    del P
ind = pd.Series(0, index=codes, dtype=int)
for c in codes:
    if c in ind_last.index and pd.notna(ind_last[c]):
        ind[c] = int(ind_last[c])
above = (C.iloc[-1] > ma20.iloc[-1]).where(C.iloc[-1].notna())
mb = above.mean()
sbr = pd.Series(np.nan, index=codes)
sr20 = pd.Series(np.nan, index=codes)
for k in range(1, len(inds) + 1):
    m = (ind == k)
    if m.any():
        sbr[m] = above[m].mean()
        sr20[m] = r20[m].mean()

# 市值粗调：面板最后一日 mcap × qfq变化
qnow = q.set_index('thscode')['close']
qold = q0.set_index('thscode')['close']
mcap = pd.Series(np.nan, index=codes)
for c in codes:
    if c in mcap_last.index and pd.notna(mcap_last[c]):
        fac = qnow.get(c, np.nan) / qold.get(c, np.nan) if qold.get(c, 0) else 1.0
        mcap[c] = float(mcap_last[c]) * (fac if np.isfinite(fac) else 1.0)

names = pd.read_csv(NAMES, encoding='utf-8-sig').set_index('thscode')['name']
st_name = pd.Series([('ST' in str(names.get(c, '')).upper()) for c in codes], index=codes)
main_buy = main_ok & ~st_name & ~stx

stock_ok = (up_t & (age >= 10) & (age <= 30) & (atr <= 0.044) & (lu20 == 0) &
            (dd60 >= -0.15) & (amt20 >= 3e7) & (amt20 <= 70e6) &
            (C.iloc[-1] >= 2) & (C.iloc[-1] <= 15) & (vol20 >= 0.20) &
            main_ok & ~st_name & ~stx)
ind_ok = (sbr >= 0.60) & (sr20 > 0.05)
near = stock_ok & ind_ok
sig = near & (mb > 0.50)

R = pd.DataFrame({
    '代码': codes, '名称': [names.get(c, '') for c in codes],
    '板块': [boards.get(c, '') for c in codes],
    '实际收盘': [qnow.get(c, np.nan) for c in codes],
    '后复权收盘': C.iloc[-1].values,
    '20日均成交额_万': amt20.values / 1e4,
    'ATR占收盘': atr.values, '20日波动': vol20.values,
    '距60日高点': dd60.values, '趋势年龄': age.values,
    '20日涨停次数': lu20.values, '20日收益': r20.values,
    '市值_亿': mcap.values / 1e8,
    '行业': [inds[int(ind[c]) - 1] if ind[c] > 0 else '' for c in codes],
    '行业宽度': sbr.values, '行业20日收益': sr20.values,
    '5%触板次数60': c5.values, '名称ST': st_name.values, 'ST代理': stx.values,
})
R['行业宽度≥60%'] = R['行业宽度'] >= 0.60
R['行业20日>5%'] = R['行业20日收益'] > 0.05
R['F2个股条件'] = stock_ok.values
R['F2近信号'] = near.values
R['F2可买'] = (sig & main_buy).values

cols = ['代码', '名称', '板块', '实际收盘', '后复权收盘', '20日均成交额_万', 'ATR占收盘',
        '20日波动', '距60日高点', '趋势年龄', '20日涨停次数', '20日收益', '市值_亿', '行业',
        '行业宽度', '行业20日收益', '行业宽度≥60%', '行业20日>5%', '5%触板次数60',
        '名称ST', 'ST代理', 'F2个股条件', 'F2近信号', 'F2可买']
N = R[R['F2近信号']].copy()
N = N.sort_values('市值_亿', na_position='last')
V = R[R['F2可买']].copy().sort_values('市值_亿', na_position='last')
N[cols].to_csv(os.path.join(DIR, '15_今日F2近信号_2026-10-22.csv'), index=False,
               encoding='utf-8-sig', float_format='%.4f')
V[cols].to_csv(os.path.join(DIR, '15_今日F2可买_2026-10-22.csv'), index=False,
               encoding='utf-8-sig', float_format='%.4f')

rows = []
for k in range(1, len(inds) + 1):
    m = (ind == k)
    if m.sum() >= 5:
        rows.append(dict(行业=inds[k - 1], 股票数=int(m.sum()),
                         行业宽度=sbr[m].iloc[0], 行业20日收益=sr20[m].iloc[0],
                         宽度达标=bool(sbr[m].iloc[0] >= 0.60),
                         收益达标=bool(sr20[m].iloc[0] > 0.05)))
IND = pd.DataFrame(rows).sort_values('行业20日收益', ascending=False)
IND.to_csv(os.path.join(DIR, '15_今日F2_行业闸门.csv'), index=False,
           encoding='utf-8-sig', float_format='%.4f')

status = pd.DataFrame([dict(
    日期=str(last), 市场宽度=mb, 市场闸门=bool(mb > 0.50),
    行业宽度达标数=int(IND['宽度达标'].sum()), 行业收益达标数=int(IND['收益达标'].sum()),
    个股条件数=int(stock_ok.sum()), F2近信号数=int(near.sum()),
    F2可买数=int(sig.sum()), 主板数=int(main_ok.sum()),
    名称ST数=int(st_name.sum()), ST代理数=int(stx.sum()))])
status.to_csv(os.path.join(DIR, '15_今日F2状态_2026-10-22.csv'), index=False,
              encoding='utf-8-sig', float_format='%.4f')

print('日期', last, '市场宽度 %.4f' % mb, '闸门', '开' if mb > 0.5 else '关')
print('行业宽度达标 %d，行业收益达标 %d' % (IND['宽度达标'].sum(), IND['收益达标'].sum()))
print('个股条件 %d；F2近信号 %d；F2可买 %d' % (stock_ok.sum(), near.sum(), sig.sum()))
print('\nF2近信号（按市值升序）')
print(N[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}') if len(N) else 'NONE')
print('\nF2可买')
print(V[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}') if len(V) else 'NONE')
print('\n行业前10')
print(IND.head(10).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
