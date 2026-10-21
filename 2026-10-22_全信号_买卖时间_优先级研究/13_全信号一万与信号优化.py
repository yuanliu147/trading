# -*- coding: utf-8 -*-
"""全部信号按每笔1万元、无仓位上限的逐笔与信号优化。

- 信号段来自 独立信号段.csv（40,533段，同一股票连续信号日合并）。
- 每个信号段取首个信号日，次日开盘买，持有20个交易日；可选 -20% 灾难止损。
- 每笔名义本金 10,000 元；不考虑整手和资金占用，只用于比较信号质量。
- 费用：单边0.15%，最低5元。
- 同时输出“不止损”和“-20%止损”两套结果。
- 特征分析只用信号日及以前的数据；训练2018-2021，检验2022-2026。
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
DIR = '/Users/admin/WorkSpace/trading/2026-10-22_全信号_买卖时间_优先级研究'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 面板 ----------------
P = pickle.load(open(os.path.join(BASE, '01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
codes = np.array(P['meta']['codes'])
code_idx = {c: i for i, c in enumerate(codes)}
C = P['close'].values; O = P['open'].values; H = P['high'].values; L = P['low'].values
ATR = P['atr_pct'].values; AMT = P['amt20'].values
LU = P['lu_count20'].values; NG = P['next_gap'].values
OWU = P['one_word_up'].values; OWD = P['one_word_down'].values
AGE = P['trend_age'].values; DD60 = P['dd60'].values
DMA20 = P['dist_ma20'].values; DMA60 = P['dist_ma60'].values
R20 = P['ret20'].values; R60 = P['ret60'].values; VOL20 = P['vol20'].values
TURNZ = P['turn_z250'].values; AMTR = P['amt_ratio20'].values
GAP = P['gap'].values; UPPER = P['upper'].values; CLOSEPOS = P['close_pos'].values
SBR = P['sector_breadth20'].values; SR20 = P['sector_ret20'].values
SSZ = P['stock_sector_z'].values; SRS = P['sector_rs'].values
OWNB = P['own_base20'].values; OWNW = P['own_win20'].values
MCAP = P['mcap'].values; PRICE = P['close'].values
MB20 = P['market_breadth20'].values; MB60 = P['market_breadth60'].values
MKT_R20 = P['market_ret20'].values; MKT_VOL = P['market_index'].astype(float)
REGIME = P['regime_id'].values
del P

mkt_vol = pd.Series(MKT_VOL).pct_change().rolling(20, min_periods=15).std().values * np.sqrt(252)
_mkt_ma60 = pd.Series(MKT_VOL).rolling(60, min_periods=40).mean()
MKT_SLOPE = (_mkt_ma60 / _mkt_ma60.shift(20) - 1).values

# ---------------- 信号段 ----------------
EP = pd.read_csv(os.path.join(DIR, '独立信号段.csv'), encoding='utf-8-sig',
                 parse_dates=['首个信号日', '最后信号日'])
EP = EP.sort_values('首个信号日').reset_index(drop=True)
EP = EP[EP['代码'].isin(code_idx)].copy()
EP['j'] = EP['代码'].map(code_idx).astype(int)
EP['ts'] = dates.searchsorted(EP['首个信号日'])
EP['te'] = EP['ts'] + 1
T = len(dates)
ei = dates.searchsorted(pd.Timestamp('2026-10-13'), side='right') - 1

# 入场可执行性（t+1开盘、非一字涨停、非高开>5%、有开盘）
j = EP['j'].values; ts = EP['ts'].values; te = EP['te'].values
valid_idx = (te <= ei) & (ts >= 0)
te_safe = np.clip(te, 0, T - 1)
Oe = O[te_safe, j]; He = H[te_safe, j]
ng = NG[ts, j]; owu = OWU[te_safe, j]; lu_e = LU[te_safe, j]
entry_ok = valid_idx & np.isfinite(Oe) & (Oe > 0) & ~(np.isfinite(ng) & (ng > 0.05))
entry_ok &= ~(owu | (lu_e > 0) & np.isfinite(He) & (Oe >= He - 1e-9))
EP['入场可执行'] = entry_ok
EP['入场过滤'] = np.where(~valid_idx, '无次日',
                    np.where(~np.isfinite(Oe) | (Oe <= 0), '停牌/无开盘',
                    np.where(np.isfinite(ng) & (ng > 0.05), '高开>5%',
                    np.where(owu | ((lu_e > 0) & np.isfinite(He) & (Oe >= He - 1e-9)), '一字涨停', '可买'))))

# ---------------- 逐笔模拟 ----------------
def sim_one(t_entry, jj, use_stop):
    entry = O[t_entry, jj]
    if not np.isfinite(entry) or entry <= 0:
        return None
    stop = entry * 0.8
    if use_stop:
        for tt in range(t_entry, min(t_entry + 20, T)):
            if np.isfinite(L[tt, jj]) and L[tt, jj] <= stop:
                if OWD[tt, jj] or not np.isfinite(O[tt, jj]):
                    continue
                fill = min(O[tt, jj], stop)
                return tt, fill, '止损'
    tt = t_entry + 19
    while tt < T:
        if OWD[tt, jj]:
            tt += 1
            continue
        if np.isfinite(C[tt, jj]):
            return tt, C[tt, jj], '到期'
        tt += 1
    # 数据末端：用最后一个有效收盘
    future = np.where(np.isfinite(C[t_entry:, jj]))[0]
    if len(future):
        tt = t_entry + future[-1]
        return tt, C[tt, jj], '数据终止'
    return None

rows = []
for idx, r in EP[EP['入场可执行']].iterrows():
    t_entry = int(r['te']); jj = int(r['j'])
    no = sim_one(t_entry, jj, False)
    st = sim_one(t_entry, jj, True)
    if no is None:
        continue
    ent = O[t_entry, jj]
    def pnl(exit_t, exit_px):
        gross = 10000 * (exit_px / ent - 1)
        sell_amt = 10000 * exit_px / ent
        fee = 15.0 + max(5.0, sell_amt * 0.0015)
        net = gross - fee
        return sell_amt, fee, net
    no_amt, no_fee, no_net = pnl(no[0], no[1])
    st_amt, st_fee, st_net = pnl(st[0], st[1]) if st is not None else (np.nan, np.nan, np.nan)
    rows.append(dict(
        代码=r['代码'], 首个信号日=r['首个信号日'], 买入日=dates[t_entry],
        不止损卖出日=dates[no[0]], 不止损卖出价=no[1], 不止损退出=no[2],
        不止损卖出金额=no_amt, 不止损费用=no_fee, 不止损净盈亏=no_net,
        不止损收益率=no_net / 10000,
        止损后卖出日=dates[st[0]] if st is not None else pd.NaT,
        止损后卖出价=st[1] if st is not None else np.nan,
        止损后退出=st[2] if st is not None else '',
        止损后卖出金额=st_amt, 止损后费用=st_fee, 止损后净盈亏=st_net,
        止损后收益率=st_net / 10000 if st is not None else np.nan,
        持有天数=no[0] - t_entry + 1,
        买入价=ent, 股数=10000 / ent,
        年份=dates[t_entry].year,
        系统买入=r['是否买入'],
        # 特征（信号日收盘，全部为当时可见）
        趋势年龄=AGE[ts[idx], jj], 真实波幅=ATR[ts[idx], jj], 成交额=AMT[ts[idx], jj],
        距60日高点=DD60[ts[idx], jj], 距20日均线=DMA20[ts[idx], jj], 距60日均线=DMA60[ts[idx], jj],
        涨幅20=R20[ts[idx], jj], 涨幅60=R60[ts[idx], jj], 波动20=VOL20[ts[idx], jj],
        换手异常值=TURNZ[ts[idx], jj], 成交额比=AMTR[ts[idx], jj], 信号日跳空=GAP[ts[idx], jj],
        上影线=UPPER[ts[idx], jj], 收盘位置=CLOSEPOS[ts[idx], jj],
        行业宽度=SBR[ts[idx], jj], 行业20日收益=SR20[ts[idx], jj],
        个股行业相对强度=SSZ[ts[idx], jj], 行业相对强度=SRS[ts[idx], jj],
        自身基准=OWNB[ts[idx], jj], 自身胜率=OWNW[ts[idx], jj],
        市值=MCAP[ts[idx], jj], 股价=PRICE[ts[idx], jj],
        市场宽度20=MB20[ts[idx]], 市场宽度60=MB60[ts[idx]],
        市场20日收益=MKT_R20[ts[idx]], 市场波动=mkt_vol[ts[idx]],
        市场60日斜率=MKT_SLOPE[ts[idx]], 市场状态=REGIME[ts[idx]],
    ))
TR = pd.DataFrame(rows)
print('episodes', len(EP), 'entry executable', EP['入场可执行'].sum(), 'trades', len(TR))
print('entry filter counts')
print(EP['入场过滤'].value_counts().to_string())

# 汇总
def summary(x, name):
    x = pd.Series(x).dropna()
    return dict(口径=name, 笔数=len(x), 总净盈亏=x.sum(), 平均每笔=x.mean(), 中位每笔=x.median(),
                胜率=(x > 0).mean(), 盈亏比=x[x > 0].sum() / max(1e-9, -x[x < 0].sum()),
                最好=x.max(), 最差=x.min(), 正收益笔数=(x > 0).sum())

SUM = pd.DataFrame([
    summary(TR['不止损净盈亏'], '不止损'),
    summary(TR['止损后净盈亏'], '止损20'),
])
print('\nsummary')
print(SUM.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
print('notional total %.0f' % (len(TR) * 10000))

# 按年
Y = TR.groupby('年份').agg(笔数=('代码', 'size'), 不止损总净盈亏=('不止损净盈亏', 'sum'),
                          不止损平均=('不止损净盈亏', 'mean'), 不止损胜率=('不止损净盈亏', lambda s: (s > 0).mean()),
                          止损后总净盈亏=('止损后净盈亏', 'sum'), 止损后平均=('止损后净盈亏', 'mean'),
                          止损后胜率=('止损后净盈亏', lambda s: (s > 0).mean())).reset_index()
print('\nyearly')
print(Y.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

# 每股首笔（最早可执行）
FIRST = TR.sort_values(['代码', '买入日']).groupby('代码', as_index=False).first()
print('\nstocks with trades', len(FIRST), 'profitable (stop)', (FIRST['止损后净盈亏'] > 0).sum(),
      'share', (FIRST['止损后净盈亏'] > 0).mean())
print('first-trade mean stop', FIRST['止损后净盈亏'].mean(), 'total', FIRST['止损后净盈亏'].sum())
SUM = pd.concat([SUM, pd.DataFrame([
    summary(FIRST['不止损净盈亏'], '每股首笔-不止损'),
    summary(FIRST['止损后净盈亏'], '每股首笔-止损20'),
])], ignore_index=True)

# 保存逐笔
TR_OUT = TR.copy()
TR_OUT.to_csv(os.path.join(DIR, '全信号_一万_逐笔.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f', date_format='%Y-%m-%d')
FIRST.to_csv(os.path.join(DIR, '全信号_一万_每股首笔.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
SUM.to_csv(os.path.join(DIR, '全信号_一万_汇总.csv'), index=False, encoding='utf-8-sig',
           float_format='%.2f')
Y.to_csv(os.path.join(DIR, '全信号_一万_年度.csv'), index=False, encoding='utf-8-sig',
         float_format='%.2f')

# 按股票
BYSTOCK = TR.groupby('代码').agg(信号段数=('代码', 'size'), 不止损总净盈亏=('不止损净盈亏', 'sum'),
                                不止损胜率=('不止损净盈亏', lambda s: (s > 0).mean()),
                                止损后总净盈亏=('止损后净盈亏', 'sum'),
                                止损后平均=('止损后净盈亏', 'mean'),
                                止损后胜率=('止损后净盈亏', lambda s: (s > 0).mean())).reset_index()
BYSTOCK = BYSTOCK.sort_values('止损后总净盈亏', ascending=False)
BYSTOCK.to_csv(os.path.join(DIR, '全信号_一万_按股票.csv'), index=False, encoding='utf-8-sig',
               float_format='%.2f')
print('\nby stock top10')
print(BYSTOCK.head(10).to_string(index=False, float_format=lambda v: f'{v:.0f}'))
print('\nby stock bottom10')
print(BYSTOCK.tail(10).to_string(index=False, float_format=lambda v: f'{v:.0f}'))

# 训练/检验特征分箱
TR['期'] = np.where(TR['年份'] <= 2021, '训练2018-2021', '检验2022-2026')
features = ['趋势年龄', '真实波幅', '成交额', '距60日高点', '距20日均线', '距60日均线',
            '涨幅20', '涨幅60', '波动20', '换手异常值', '成交额比', '信号日跳空',
            '上影线', '收盘位置', '行业宽度', '行业20日收益', '个股行业相对强度', '行业相对强度',
            '自身基准', '自身胜率', '市值', '股价', '市场宽度20', '市场宽度60',
            '市场20日收益', '市场波动', '市场60日斜率']
rows_bin = []
for f in features:
    x = TR[[f, '期', '止损后收益率']].dropna()
    tr = x[x['期'] == '训练2018-2021'][f]
    if len(tr) < 500 or tr.nunique() < 5:
        continue
    try:
        bins = pd.qcut(tr, 5, duplicates='drop', retbins=True)[1]
    except Exception:
        continue
    for b in range(len(bins) - 1):
        lo, hi = bins[b], bins[b + 1]
        mask = (x[f] >= lo) & (x[f] <= hi if b == len(bins) - 2 else x[f] < hi)
        for per in ['训练2018-2021', '检验2022-2026']:
            z = x[mask & (x['期'] == per)]
            if len(z):
                rows_bin.append(dict(特征=f, 分箱=f'{lo:.4g}~{hi:.4g}', 期=per, 笔数=len(z),
                                     平均收益率=z['止损后收益率'].mean(),
                                     胜率=(z['止损后收益率'] > 0).mean()))
BINS = pd.DataFrame(rows_bin)
BINS.to_csv(os.path.join(DIR, '全信号_特征分箱.csv'), index=False, encoding='utf-8-sig',
            float_format='%.6f')
# 训练/检验相关性
corr_rows = []
for f in features:
    x = TR[[f, '期', '止损后收益率']].dropna()
    tr = x[x['期'] == '训练2018-2021']; te = x[x['期'] == '检验2022-2026']
    if len(tr) > 500 and len(te) > 500:
        corr_rows.append(dict(特征=f, 训练相关=tr[f].corr(tr['止损后收益率'], method='spearman'),
                              检验相关=te[f].corr(te['止损后收益率'], method='spearman')))
CORR = pd.DataFrame(corr_rows).sort_values('训练相关')
CORR.to_csv(os.path.join(DIR, '全信号_特征相关.csv'), index=False, encoding='utf-8-sig',
            float_format='%.4f')
print('\nfeature correlations')
print(CORR.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\nsaved trades/summary')
