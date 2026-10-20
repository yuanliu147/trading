# -*- coding: utf-8 -*-
"""
构建 2016-10 至 2026-10 全 A 股日线特征面板（矩阵形式）。
数据：本地 DuckDB v_daily_hfq（后复权），时点申万行业，申万一级行业指数，换手率缓存。
所有特征只用当日及以前数据；前瞻收益仅用于研究标签，不进入入场条件。
"""
import os, sys, time, pickle, glob, warnings
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
OUT = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制', '01_数据')
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

# ---------------- 1. 日线 ----------------
DB = '/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb'
con = duckdb.connect(DB, read_only=True)
log('读取 DuckDB 日线')
d = con.execute("""
    select thscode, date, open, high, low, close, volume, amount
    from v_daily_hfq
    where date >= '2016-10-13' and date <= '2026-10-13'
""").df()
con.close()
d['date'] = pd.to_datetime(d['date'])
d = d.sort_values(['thscode', 'date'])
log(f'日线 {len(d):,} 行, {d.thscode.nunique()} 只, {d.date.min().date()} ~ {d.date.max().date()}')

fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
wide = d.set_index(['date', 'thscode'])[fields].unstack('thscode')
wide = wide.sort_index(axis=1, level=1)
codes = wide.columns.get_level_values(1).unique().tolist()
dates = wide.index
M = {f: wide[f].astype('float32') for f in fields}
del wide, d

O, H, L, C, V, A = [M[f] for f in fields]
log(f'矩阵 {C.shape[0]} 日 × {C.shape[1]} 股')

# ---------------- 2. 换手率 / 流通市值（仅缓存覆盖的约1280只） ----------------
log('读取换手率缓存')
turn_list, share_list = {}, {}
for p in glob.glob(os.path.join(BASE, '2026-10-18_成交量融合_2010-2025/01_数据/turnover_akshare/*.pkl')):
    code = os.path.basename(p).replace('.pkl', '')
    code = code.split('.')[1] + '.' + code.split('.')[0].upper()
    try:
        x = pd.read_pickle(p)
        x['date'] = pd.to_datetime(x['date'])
        x = x.drop_duplicates('date').set_index('date')
        if code in set(codes):
            turn_list[code] = x['turn']
            if 'float_share' in x.columns:
                share_list[code] = x['float_share']
    except Exception:
        pass
TURN = pd.DataFrame(turn_list).reindex(dates).reindex(columns=codes)
FSHARE = pd.DataFrame(share_list).reindex(dates).reindex(columns=codes)
TURN = TURN.astype('float32'); FSHARE = FSHARE.astype('float32')
# 流通市值：成交额 / (换手率/100)
MCAP = (A / (TURN / 100.0)).astype('float32')
log(f'换手率覆盖 {TURN.notna().any().sum()} 只')

# ---------------- 3. 基础收益与均线 ----------------
log('计算收益/均线/波动')
R1 = C.pct_change(fill_method=None).astype('float32')
R5 = (C / C.shift(5) - 1).astype('float32')
R20 = (C / C.shift(20) - 1).astype('float32')
R60 = (C / C.shift(60) - 1).astype('float32')
R120 = (C / C.shift(120) - 1).astype('float32')
R250 = (C / C.shift(250) - 1).astype('float32')

MA5 = C.rolling(5, min_periods=5).mean().astype('float32')
MA10 = C.rolling(10, min_periods=10).mean().astype('float32')
MA20 = C.rolling(20, min_periods=20).mean().astype('float32')
MA60 = C.rolling(60, min_periods=60).mean().astype('float32')
MA120 = C.rolling(120, min_periods=120).mean().astype('float32')
MA20_SLOPE = (MA20 / MA20.shift(10) - 1).astype('float32')
MA60_SLOPE = (MA60 / MA60.shift(20) - 1).astype('float32')

VOL20 = R1.rolling(20, min_periods=15).std() * np.sqrt(252)
VOL60 = R1.rolling(60, min_periods=40).std() * np.sqrt(252)
VOL20 = VOL20.astype('float32'); VOL60 = VOL60.astype('float32')

TR_arr = np.maximum.reduce([(H - L).values, np.abs(H.values - C.shift(1).values),
                            np.abs(L.values - C.shift(1).values)])
TR = pd.DataFrame(TR_arr, index=dates, columns=codes)
ATR14 = TR.rolling(14, min_periods=10).mean() / C
ATR_PCT = ATR14.astype('float32')
del TR, TR_arr

AMT20 = A.rolling(20, min_periods=15).mean()
AMT60 = A.rolling(60, min_periods=40).mean()
AMT_RATIO20 = (A / AMT20).astype('float32')
VR5_20 = (V.rolling(5, min_periods=3).mean() / V.rolling(20, min_periods=15).mean()).astype('float32')
AMT_TREND = (AMT20 / AMT60).astype('float32')

# 位置/结构
HIGH20_PRIOR = H.shift(1).rolling(20, min_periods=10).max()
HIGH60_PRIOR = H.shift(1).rolling(60, min_periods=30).max()
HIGH250_PRIOR = H.shift(1).rolling(250, min_periods=120).max()
DIST_HIGH20 = (C / HIGH20_PRIOR - 1).astype('float32')
DIST_HIGH60 = (C / HIGH60_PRIOR - 1).astype('float32')
DIST_HIGH250 = (C / HIGH250_PRIOR - 1).astype('float32')
DIST_HIGH500 = (C / H.shift(1).rolling(500, min_periods=250).max() - 1).astype('float32')
DD60 = (C / C.rolling(60, min_periods=30).max() - 1).astype('float32')
DD250 = (C / C.rolling(250, min_periods=120).max() - 1).astype('float32')
DIST_MA20 = (C / MA20 - 1).astype('float32')
DIST_MA60 = (C / MA60 - 1).astype('float32')

UP_STRUCT = ((C > MA60) & (MA20 > MA60) & (MA60 > MA60.shift(20))).fillna(False)
# 趋势年龄：连续处于上升结构的天数
cs = UP_STRUCT.cumsum()
reset = cs.where(~UP_STRUCT).ffill().fillna(0)
TREND_AGE = (cs - reset).astype('float32')
EP_START = (UP_STRUCT & ~UP_STRUCT.shift(1).fillna(False))
EP_START_PRICE = C.where(EP_START).ffill()
EP_GAIN = (C / EP_START_PRICE - 1).astype('float32')
# 效率系数：20日净位移 / 20日路径
ER20 = ((C / C.shift(20) - 1).abs() / R1.abs().rolling(20, min_periods=15).sum()).astype('float32')
UP_RATIO20 = (R1 > 0).astype('float32').rolling(20, min_periods=15).mean().astype('float32')
NEWHIGH20 = (C >= C.rolling(20, min_periods=10).max())
NEWHIGH60 = (C >= C.rolling(60, min_periods=30).max())
NEWHIGH250 = (C >= C.rolling(250, min_periods=120).max())
NH20_COUNT = NEWHIGH20.astype('float32').rolling(20, min_periods=10).sum().astype('float32')
_up_cs = (R1 > 0).cumsum()
_up_reset = _up_cs.where(~(R1 > 0)).ffill().fillna(0)
CONSEC_UP = (_up_cs - _up_reset).astype('float32')
# 阳线/影线
BODY = (C - O) / C.shift(1)
UPPER = (H - np.maximum(O, C)) / C.shift(1)
LOWER = (np.minimum(O, C) - L) / C.shift(1)
CLOSE_POS = ((C - L) / (H - L + 1e-9)).astype('float32')
GAP = (O / C.shift(1) - 1).astype('float32')
UPPER = UPPER.astype('float32'); LOWER = LOWER.astype('float32'); BODY = BODY.astype('float32')

# RSI14
gain = R1.clip(lower=0).rolling(14, min_periods=10).mean()
loss = (-R1.clip(upper=0)).rolling(14, min_periods=10).mean()
RSI14 = (100 - 100 / (1 + gain / (loss + 1e-9))).astype('float32')
# MACD
EMA12 = C.ewm(span=12, adjust=False).mean()
EMA26 = C.ewm(span=26, adjust=False).mean()
DIFF = EMA12 - EMA26
DEA = DIFF.ewm(span=9, adjust=False).mean()
MACD_HIST = ((DIFF - DEA) * 2 / C).astype('float32')

# ---------------- 4. 涨停/跌停 ----------------
log('计算涨跌停')
LIM = pd.DataFrame(0.098, index=dates, columns=codes)
cols_300 = [c for c in codes if c.startswith('300') or c.startswith('301')]
if cols_300:
    LIM.loc[dates >= '2020-08-24', cols_300] = 0.196
cols_688 = [c for c in codes if c.startswith('688')]
if cols_688:
    LIM.loc[:, cols_688] = 0.196
cols_bj = [c for c in codes if c.startswith(('8', '4'))]
if cols_bj:
    LIM.loc[:, cols_bj] = 0.296
LIM = LIM.astype('float32')
LIMIT_UP = (R1 >= LIM)
LIMIT_DOWN = (R1 <= -LIM)
ONE_WORD_UP = (LIMIT_UP & (H == L))
ONE_WORD_DOWN = (LIMIT_DOWN & (H == L))

# ---------------- 5. 自身历史 ----------------
log('计算自身历史位置/历史趋势基准率')
R20_MEAN750 = R20.rolling(750, min_periods=250).mean()
R20_STD750 = R20.rolling(750, min_periods=250).std()
R20_HIST_Z = ((R20 - R20_MEAN750) / (R20_STD750 + 1e-9)).astype('float32')
TURN_Z250 = ((TURN - TURN.rolling(250, min_periods=60).mean()) / (TURN.rolling(250, min_periods=60).std() + 1e-9)).astype('float32')
VOL_HIST_Z = ((VOL20 - VOL20.rolling(500, min_periods=120).mean()) / (VOL20.rolling(500, min_periods=120).std() + 1e-9)).astype('float32')

# ----------------- 6. 行业（时点） -----------------
log('构建时点行业矩阵')
ind = pd.read_csv(os.path.join(BASE, '2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv'))
ind['code'] = ind.code.str.split('.').str[1] + '.' + ind.code.str.split('.').str[0].str.upper()
ind['start_date'] = pd.to_datetime(ind.start_date)
ind = ind.sort_values(['code', 'start_date'])
industries = sorted(ind.industry.dropna().unique().tolist())
ind2id = {x: i + 1 for i, x in enumerate(industries)}
IND = np.zeros((len(dates), len(codes)), dtype='int16')
date_arr = dates.values
code_pos = {c: j for j, c in enumerate(codes)}
for code, g in ind.groupby('code', sort=False):
    j = code_pos.get(code)
    if j is None:
        continue
    starts = g.start_date.values.astype('datetime64[ns]')
    ids = g.industry.map(ind2id).fillna(0).astype('int16').values
    pos = np.searchsorted(starts, date_arr, side='right') - 1
    col = np.where(pos >= 0, ids[np.maximum(pos, 0)], 0)
    IND[:, j] = col
IND = pd.DataFrame(IND, index=dates, columns=codes)
log(f'行业矩阵覆盖 {np.mean(IND.values>0):.1%}')

# 行业聚合（从全A自身算，PIT）
log('计算行业宽度/动量/相对强弱')
above20_np = np.where(C.notna().values, (C > MA20).values, np.nan)
above60_np = np.where(C.notna().values, (C > MA60).values, np.nan)
R20_np = R20.values
R60_np = R60.values
AMT_RATIO_np = AMT_RATIO20.values
IND_np = IND.values
sector_feat = {}
for name, arr, fn in [
    ('sector_breadth20', above20_np, np.nanmean),
    ('sector_breadth60', above60_np, np.nanmean),
    ('sector_ret20', R20_np, np.nanmean),
    ('sector_ret60', R60_np, np.nanmean),
    ('sector_amt_ratio', AMT_RATIO_np, np.nanmean),
]:
    out = np.full((len(dates), len(industries) + 1), np.nan, dtype='float32')
    for k in range(1, len(industries) + 1):
        mask = (IND_np == k)
        if not mask.any():
            continue
        sub = np.where(mask, arr, np.nan)
        with np.errstate(all='ignore'):
            out[:, k] = fn(sub, axis=1)
    sector_feat[name] = pd.DataFrame(out, index=dates).astype('float32')
# 行业内标准化 (ret20 - 行业均值)/行业标准差
sector_mean_r20 = sector_feat['sector_ret20'].values
sector_std_r20 = np.full_like(sector_mean_r20, np.nan)
for k in range(1, len(industries) + 1):
    mask = (IND_np == k)
    if not mask.any():
        continue
    sub = np.where(mask, R20_np, np.nan)
    with np.errstate(all='ignore'):
        sector_std_r20[:, k] = np.nanstd(sub, axis=1)
# 映射回个股
def map_sector(df):
    arr = df.values
    return pd.DataFrame(arr[np.arange(len(dates))[:, None], IND_np], index=dates, columns=codes).astype('float32')
SECTOR_BREADTH20 = map_sector(sector_feat['sector_breadth20'])
SECTOR_BREADTH60 = map_sector(sector_feat['sector_breadth60'])
SECTOR_RET20 = map_sector(sector_feat['sector_ret20'])
SECTOR_RET60 = map_sector(sector_feat['sector_ret60'])
SECTOR_AMT = map_sector(sector_feat['sector_amt_ratio'])
_mean = sector_mean_r20[np.arange(len(dates))[:, None], IND_np]
_std = sector_std_r20[np.arange(len(dates))[:, None], IND_np]
STOCK_SECTOR_Z = pd.DataFrame(((R20_np - _mean) / (_std + 1e-9)), index=dates, columns=codes).astype('float32')
SECTOR_RS = (R20 - SECTOR_RET20).astype('float32')

# 申万行业指数（板块自身趋势）
sw = pd.read_csv(os.path.join(BASE, '2026-10-18_成交量融合_2010-2025/01_数据/申万一级行业指数.csv'))
sw['date'] = pd.to_datetime(sw['date'])
sw = sw[sw.industry.isin(industries)]
SWC = sw.pivot_table(index='date', columns='industry', values='close', aggfunc='last').sort_index()
SWC = SWC.reindex(dates).ffill()
SW_R20 = (SWC / SWC.shift(20) - 1)
SW_R60 = (SWC / SWC.shift(60) - 1)
SW_MA20 = SWC.rolling(20, min_periods=15).mean()
SW_MA60 = SWC.rolling(60, min_periods=40).mean()
SW_ABOVE60 = (SWC > SW_MA60).astype('float32')
SW_SLOPE60 = (SW_MA60 / SW_MA60.shift(20) - 1)
SW_R20 = SW_R20.reindex(columns=industries).astype('float32')
SW_R60 = SW_R60.reindex(columns=industries).astype('float32')
SW_ABOVE60 = SW_ABOVE60.reindex(columns=industries).astype('float32')
SW_SLOPE60 = SW_SLOPE60.reindex(columns=industries).astype('float32')
# 映射
def map_sw(df):
    arr = np.full((len(dates), len(industries) + 1), np.nan, dtype='float32')
    arr[:, 1:] = df.reindex(columns=industries).values
    return pd.DataFrame(arr[np.arange(len(dates))[:, None], IND_np], index=dates, columns=codes).astype('float32')
SW_R20_M = map_sw(SW_R20); SW_R60_M = map_sw(SW_R60)
SW_ABOVE60_M = map_sw(SW_ABOVE60); SW_SLOPE60_M = map_sw(SW_SLOPE60)

# ---------------- 7. 市场状态/情绪 ----------------
log('计算市场状态/情绪')
mkt_ret = R1.clip(-0.2, 0.2).mean(axis=1, skipna=True)
MKT_INDEX = (1 + mkt_ret.fillna(0)).cumprod()
MKT_MA20 = MKT_INDEX.rolling(20, min_periods=15).mean()
MKT_MA60 = MKT_INDEX.rolling(60, min_periods=40).mean()
MKT_MA120 = MKT_INDEX.rolling(120, min_periods=80).mean()
MKT_SLOPE60 = MKT_MA60 / MKT_MA60.shift(20) - 1
MKT_BREADTH20 = (C > MA20).astype('float32').where(C.notna()).mean(axis=1)
MKT_BREADTH60 = (C > MA60).astype('float32').where(C.notna()).mean(axis=1)
MKT_BREADTH250 = (C > MA120).astype('float32').where(C.notna()).mean(axis=1)
MKT_DISPERSION = R20.std(axis=1, skipna=True)
MKT_LIMITUP = LIMIT_UP.astype('float32').where(C.notna()).mean(axis=1)
MKT_LIMITDOWN = LIMIT_DOWN.astype('float32').where(C.notna()).mean(axis=1)
MKT_GAPUP = (GAP > 0.02).astype('float32').where(C.notna()).mean(axis=1)
MKT_AMT_RATIO = A.sum(axis=1) / A.sum(axis=1).rolling(20, min_periods=15).mean()
MKT_NEWHIGH = NEWHIGH250.astype('float32').where(C.notna()).mean(axis=1)
MKT_RET20 = MKT_INDEX / MKT_INDEX.shift(20) - 1
MKT_RET60 = MKT_INDEX / MKT_INDEX.shift(60) - 1
# 行情状态
regime = pd.Series('震荡', index=dates)
regime[(MKT_INDEX > MKT_MA60) & (MKT_SLOPE60 > 0) & (MKT_BREADTH20 > 0.50)] = '上行'
regime[(MKT_INDEX < MKT_MA60) & (MKT_SLOPE60 < 0) & (MKT_BREADTH20 < 0.50)] = '下行'
# 强上行/强下行
regime[(MKT_INDEX > MKT_MA60) & (MKT_INDEX > MKT_MA120) & (MKT_SLOPE60 > 0.005) & (MKT_BREADTH20 > 0.60)] = '强上行'
regime[(MKT_INDEX < MKT_MA60) & (MKT_INDEX < MKT_MA120) & (MKT_SLOPE60 < -0.005) & (MKT_BREADTH20 < 0.40)] = '强下行'
REGIME_ID = pd.Series(regime.map({'强下行': 1, '下行': 2, '震荡': 3, '上行': 4, '强上行': 5}).fillna(3).astype('int8'), index=dates)
# 市场情绪综合分：涨停占比、成交额热度、新高占比、换手（用成交额代理）标准化
def zs(s):
    return (s - s.rolling(250, min_periods=60).mean()) / (s.rolling(250, min_periods=60).std() + 1e-9)
MKT_SENT = (zs(MKT_LIMITUP).fillna(0) * 0.35 + zs(MKT_AMT_RATIO).fillna(0) * 0.30 +
            zs(MKT_NEWHIGH).fillna(0) * 0.20 + zs(MKT_BREADTH20).fillna(0) * 0.15)
MKT_SENT = MKT_SENT.astype('float32')

# ---------------- 8. 个股情绪/拥挤 ----------------
log('计算个股情绪代理')
# 20日量价同步性
VOLCHG = V.pct_change(fill_method=None).clip(-1, 3)
VP_CORR20 = R1.rolling(20, min_periods=15).corr(VOLCHG).astype('float32')
# 近20日涨停次数
LU_COUNT20 = LIMIT_UP.astype('float32').rolling(20, min_periods=10).sum().astype('float32')
GAP_COUNT20 = (GAP.abs() > 0.02).astype('float32').rolling(20, min_periods=10).sum().astype('float32')

# ---------------- 9. 前瞻收益 ----------------
log('计算前瞻收益')
FWD5 = (C.shift(-5) / O.shift(-1) - 1).astype('float32')
FWD10 = (C.shift(-10) / O.shift(-1) - 1).astype('float32')
FWD20 = (C.shift(-20) / O.shift(-1) - 1).astype('float32')
FWD60 = (C.shift(-60) / O.shift(-1) - 1).astype('float32')
MKT_FWD20 = FWD20.median(axis=1, skipna=True)
EXCESS20 = FWD20.sub(MKT_FWD20, axis=0).astype('float32')
# 次日开盘可买性
NEXT_OPEN = O.shift(-1)
NEXT_GAP = (O.shift(-1) / C - 1).astype('float32')
NEXT_LIMIT_UP = (H.shift(-1) == L.shift(-1)) & (O.shift(-1) >= C * (1 + LIM - 0.005))
NEXT_GAP_DAYS = pd.DataFrame(np.nan, index=dates, columns=codes, dtype='float32')
dd = pd.Series(dates, index=dates)
dd_next = dd.shift(-1)
gap_days = (dd_next - dd).dt.days
NEXT_GAP_DAYS = pd.DataFrame(np.tile(gap_days.values[:, None], (1, len(codes))), index=dates, columns=codes).astype('float32')

# 自身历史：过去上升结构日的20日胜率/平均收益（只用已实现标签）
log('计算自身历史趋势基准率（PIT）')
base_mask = UP_STRUCT & FWD20.notna()
own_base = FWD20.where(base_mask).shift(21).expanding(min_periods=50).mean().astype('float32')
own_win = (FWD20 > 0).where(base_mask).shift(21).expanding(min_periods=50).mean().astype('float32')

# 20日路径内的最大回撤/最大上涨（用于事件研究，仅在需要时另行计算；这里先算20日最低/最高）
log('计算20日路径指标')
# 未来20日最低价/最高价（从次日开始），用反向rolling实现
FWD_MIN20 = L.shift(-1).iloc[::-1].rolling(20, min_periods=1).min().iloc[::-1].astype('float32')
FWD_MAX20 = H.shift(-1).iloc[::-1].rolling(20, min_periods=1).max().iloc[::-1].astype('float32')
FWD_MIN20_ENTRY = (FWD_MIN20 / O.shift(-1) - 1).astype('float32')
FWD_MAX20_ENTRY = (FWD_MAX20 / O.shift(-1) - 1).astype('float32')

# ---------------- 10. 保存 ----------------
log('保存面板')
panel = {
    'open': O, 'high': H, 'low': L, 'close': C, 'volume': V, 'amount': A,
    'ret1': R1, 'ret5': R5, 'ret20': R20, 'ret60': R60, 'ret120': R120, 'ret250': R250,
    'ma5': MA5, 'ma10': MA10, 'ma20': MA20, 'ma60': MA60, 'ma120': MA120,
    'ma20_slope': MA20_SLOPE, 'ma60_slope': MA60_SLOPE,
    'vol20': VOL20, 'vol60': VOL60, 'atr_pct': ATR_PCT,
    'amt20': AMT20.astype('float32'), 'amt_ratio20': AMT_RATIO20, 'vr5_20': VR5_20, 'amt_trend': AMT_TREND,
    'dist_high20': DIST_HIGH20, 'dist_high60': DIST_HIGH60, 'dist_high250': DIST_HIGH250, 'dist_high500': DIST_HIGH500,
    'dd60': DD60, 'dd250': DD250, 'dist_ma20': DIST_MA20, 'dist_ma60': DIST_MA60,
    'up_struct': UP_STRUCT, 'trend_age': TREND_AGE, 'ep_gain': EP_GAIN, 'er20': ER20,
    'up_ratio20': UP_RATIO20, 'newhigh20': NEWHIGH20, 'newhigh60': NEWHIGH60, 'newhigh250': NEWHIGH250,
    'nh20_count': NH20_COUNT, 'consec_up': CONSEC_UP,
    'body': BODY, 'upper': UPPER, 'lower': LOWER, 'close_pos': CLOSE_POS, 'gap': GAP,
    'rsi14': RSI14, 'macd_hist': MACD_HIST,
    'limit_up': LIMIT_UP, 'limit_down': LIMIT_DOWN, 'one_word_up': ONE_WORD_UP, 'one_word_down': ONE_WORD_DOWN,
    'turn': TURN, 'float_share': FSHARE, 'mcap': MCAP, 'turn_z250': TURN_Z250,
    'ret20_hist_z': R20_HIST_Z, 'vol_hist_z': VOL_HIST_Z,
    'own_base20': own_base, 'own_win20': own_win,
    'vp_corr20': VP_CORR20, 'lu_count20': LU_COUNT20, 'gap_count20': GAP_COUNT20,
    'sector_breadth20': SECTOR_BREADTH20, 'sector_breadth60': SECTOR_BREADTH60,
    'sector_ret20': SECTOR_RET20, 'sector_ret60': SECTOR_RET60, 'sector_amt_ratio': SECTOR_AMT,
    'stock_sector_z': STOCK_SECTOR_Z, 'sector_rs': SECTOR_RS,
    'sw_ret20': SW_R20_M, 'sw_ret60': SW_R60_M, 'sw_above60': SW_ABOVE60_M, 'sw_slope60': SW_SLOPE60_M,
    'fwd5': FWD5, 'fwd10': FWD10, 'fwd20': FWD20, 'fwd60': FWD60, 'excess20': EXCESS20,
    'next_gap': NEXT_GAP, 'next_limit_up': NEXT_LIMIT_UP, 'next_gap_days': NEXT_GAP_DAYS,
    'fwd_min20_entry': FWD_MIN20_ENTRY, 'fwd_max20_entry': FWD_MAX20_ENTRY,
    'industry_id': IND,
    'market_index': MKT_INDEX.astype('float32'), 'market_ma20': MKT_MA20.astype('float32'),
    'market_ma60': MKT_MA60.astype('float32'), 'market_ma120': MKT_MA120.astype('float32'),
    'market_slope60': MKT_SLOPE60.astype('float32'), 'market_breadth20': MKT_BREADTH20.astype('float32'),
    'market_breadth60': MKT_BREADTH60.astype('float32'), 'market_breadth250': MKT_BREADTH250.astype('float32'),
    'market_dispersion': MKT_DISPERSION.astype('float32'), 'market_limitup': MKT_LIMITUP.astype('float32'),
    'market_limitdown': MKT_LIMITDOWN.astype('float32'), 'market_gapup': MKT_GAPUP.astype('float32'),
    'market_amt_ratio': MKT_AMT_RATIO.astype('float32'), 'market_newhigh': MKT_NEWHIGH.astype('float32'),
    'market_ret20': MKT_RET20.astype('float32'), 'market_ret60': MKT_RET60.astype('float32'),
    'market_sent': MKT_SENT, 'regime_id': REGIME_ID,
    'meta': {'dates': dates, 'codes': codes, 'industries': industries, 'ind2id': ind2id,
             'regime_names': {1: '强下行', 2: '下行', 3: '震荡', 4: '上行', 5: '强上行'}},
}
with open(os.path.join(OUT, '全A面板_2016_2026.pkl'), 'wb') as f:
    pickle.dump(panel, f, protocol=4)
log(f'保存完成: {os.path.getsize(os.path.join(OUT, "全A面板_2016_2026.pkl"))/1e9:.2f} GB')

# 市场状态日表
mkt = pd.DataFrame({'date': dates, 'market_index': MKT_INDEX.values, 'breadth20': MKT_BREADTH20.values,
                    'breadth60': MKT_BREADTH60.values, 'limitup': MKT_LIMITUP.values,
                    'limitdown': MKT_LIMITDOWN.values, 'newhigh': MKT_NEWHIGH.values,
                    'amt_ratio': MKT_AMT_RATIO.values, 'dispersion': MKT_DISPERSION.values,
                    'sent': MKT_SENT.values, 'regime': regime.values, 'mkt_ret20': MKT_RET20.values})
mkt.to_csv(os.path.join(OUT, '市场状态日表.csv'), index=False, encoding='utf-8-sig')
log('完成')
