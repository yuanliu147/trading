# -*- coding: utf-8 -*-
"""
构建“T-1 特征 -> T 日跌停”分析面板。

输入：
  01_数据/跌停事件表_主.parquet
  01_数据/跌停事件表_补充.parquet
  2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl

输出：
  01_数据/分析面板.parquet      每行 = 一个可评估股票-日(T)，特征是 T-1 收盘
  01_数据/代码表.csv
  01_数据/市场日表.csv
  01_数据/行业日表.csv
  01_数据/跌停事件清单.csv
  01_数据/基准率.csv
  01_数据/字段说明.csv
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '01_数据')
PANEL = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制', '01_数据', '全A面板_2016_2026_含退市.pkl')
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

# ---------------- 1. 面板 ----------------
log('加载全A面板（4.1 GB）')
P = pickle.load(open(PANEL, 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
codes = list(P['meta']['codes'])
D, C = len(dates), len(codes)
code_pos = {c: j for j, c in enumerate(codes)}
log(f'面板 {D} 日 × {C} 股')

def M(key):
    x = P[key]
    if isinstance(x, pd.DataFrame):
        return x.reindex(index=dates, columns=codes)
    return x.reindex(dates)

C_ = M('close'); O_ = M('open'); H_ = M('high'); L_ = M('low'); V_ = M('volume'); A_ = M('amount')
R1 = M('ret1')
valid = (C_.notna() & (V_ > 0) & (C_ > 0))
valid_prev = valid.shift(1, fill_value=False)
log('基础矩阵就绪')

def board_of(code):
    if code.startswith(('300', '301')):
        return '创业板'
    if code.startswith('688'):
        return '科创板'
    if code.startswith(('8', '4', '920')):
        return '北交所'
    return '主板'
board = np.array([board_of(c) for c in codes])
board_id = pd.Series(board).map({'主板': 1, '创业板': 2, '科创板': 3, '北交所': 4}).values.astype('int8')
main_board = (board == '主板')

# ---------------- 2. 事件矩阵 ----------------
log('读取跌停事件表')
ev_cols = ['thscode', 'date', 'board', 'open', 'high', 'low', 'close', 'volume', 'amount',
           'ld_seal5', 'ld_seal10', 'ld_seal20', 'ld_seal30',
           'ld_touch5', 'ld_touch10', 'ld_touch20', 'ld_touch30',
           'lu_seal5', 'lu_seal10', 'lu_seal20', 'lu_seal30',
           'lu_touch5', 'lu_touch10', 'lu_touch20', 'lu_touch30',
           'ld_ratio', 'ldt_ratio', 'lu_ratio', 'lut_ratio', 'approx']
ev = pd.concat([
    pd.read_parquet(os.path.join(OUT, '跌停事件表_主.parquet'), columns=ev_cols),
    pd.read_parquet(os.path.join(OUT, '跌停事件表_补充.parquet'), columns=ev_cols),
], ignore_index=True)
ev['date'] = pd.to_datetime(ev['date'])
log(f'事件表 {len(ev):,} 行')

di = dates.get_indexer(ev['date'])
ci = np.array([code_pos.get(x, -1) for x in ev['thscode']])
ok = (di >= 0) & (ci >= 0)

def ev_num(col, dtype='int8'):
    a = np.zeros((D, C), dtype=dtype)
    vals = pd.to_numeric(ev[col], errors='coerce').fillna(0).values[ok]
    a[di[ok], ci[ok]] = vals.astype(dtype)
    return pd.DataFrame(a, index=dates, columns=codes)

def ev_bool(mask, dtype='int8'):
    a = np.zeros((D, C), dtype=dtype)
    m = np.asarray(mask, dtype=bool)
    a[di[ok], ci[ok]] = m[ok].astype(dtype)
    return pd.DataFrame(a, index=dates, columns=codes)

ld_ratio_m = ev_num('ld_ratio')
ldt_ratio_m = ev_num('ldt_ratio')
lu_ratio_m = ev_num('lu_ratio')
ld_seal = (ld_ratio_m > 0).astype('int8')
ld_touch = (ldt_ratio_m > 0).astype('int8')
lu_seal = (lu_ratio_m > 0).astype('int8')
ld5_seal = ev_num('ld_seal5')
lu5_seal = ev_num('lu_seal5')
ev_ow = (ev['ld_ratio'] > 0) & (np.abs(ev['high'] - ev['low']) < 1e-9)
ev_open = (ev['ld_ratio'].fillna(0) == 0) & (ev['ldt_ratio'].fillna(0) > 0)
ev_lu_ow = (ev['lu_ratio'] > 0) & (np.abs(ev['high'] - ev['low']) < 1e-9)
ev_lu_open = (ev['lu_ratio'].fillna(0) == 0) & (ev['lut_ratio'].fillna(0) > 0)
one_word_ld = ev_bool(ev_ow)
opened_ld = ev_bool(ev_open)
one_word_lu = ev_bool(ev_lu_ow)
opened_lu = ev_bool(ev_lu_open)
approx_codes = set(ev.loc[ev['approx'] == 1, 'thscode'].unique())
approx_flag = np.array([1 if c in approx_codes else 0 for c in codes], dtype='int8')
log(f'收盘跌停 {int(ld_seal.values.sum()):,}；触及 {int(ld_touch.values.sum()):,}；'
    f'一字 {int(one_word_ld.values.sum()):,}；打开 {int(opened_ld.values.sum()):,}')

# ---------------- 3. 新股年龄 ----------------
first_idx = np.argmax(valid.values, axis=0)
age = np.arange(D)[:, None] - first_idx[None, :] + 1
age[:, first_idx == 0] = 999
is_new = ((first_idx > 0)[None, :]) & (age <= 60)

# ---------------- 4. 事件相关特征 ----------------
log('计算跌停/涨停历史与连续天数')
def consec(flag):
    f = flag.where(valid).fillna(0)
    cs = f.cumsum()
    reset = cs.where(f == 0).ffill().fillna(0)
    return (cs - reset).astype('float32')

def days_since(flag):
    arr = np.where(flag.values > 0, dates.values[:, None], np.datetime64('NaT'))
    s = pd.DataFrame(arr, index=dates, columns=codes).ffill()
    d = (dates.values[:, None] - s.values) / np.timedelta64(1, 'D')
    return pd.DataFrame(d, index=dates, columns=codes).astype('float32')

consec_ld = consec(ld_seal)
consec_lu = consec(lu_seal)
days_since_ld = days_since(ld_seal).clip(upper=999)
days_since_lu = days_since(lu_seal).clip(upper=999)

def roll_sum(flag, n, minp=1):
    return flag.where(valid).rolling(n, min_periods=minp).sum().astype('float32')

ld_count20 = roll_sum(ld_seal, 20, 10)
ld_count60 = roll_sum(ld_seal, 60, 30)
ld_touch_count20 = roll_sum(ld_touch, 20, 10)
ld_touch_count60 = roll_sum(ld_touch, 60, 30)
ld_open_count20 = roll_sum(opened_ld, 20, 10)
lu_count5 = roll_sum(lu_seal, 5, 3)
lu_count20 = roll_sum(lu_seal, 20, 10)
lu_open_count20 = roll_sum(opened_lu, 20, 10)
lu_touch_count20 = roll_sum((lu_touch := ((ev_num('lut_ratio') > 0).astype('int8'))), 20, 10)

# ST 代理：主板过去 60 日出现过 5% 涨/跌停
st_raw = ((ld5_seal + lu5_seal) > 0).astype('float32')
st_raw = st_raw.where(np.tile(main_board, (D, 1)))
st_proxy = st_raw.rolling(60, min_periods=20).max().astype('float32')

# ---------------- 5. 价量衍生特征 ----------------
log('计算价量特征')
C_ma5, C_ma10, C_ma20, C_ma60, C_ma120 = [M(k) for k in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120']]
ret3 = (C_ / C_.shift(3) - 1).astype('float32')
ret10 = (C_ / C_.shift(10) - 1).astype('float32')
dist_ma5 = (C_ / C_ma5 - 1).astype('float32')
dist_ma10 = (C_ / C_ma10 - 1).astype('float32')
dist_ma120 = (C_ / C_ma120 - 1).astype('float32')
dd20 = (C_ / C_.rolling(20, min_periods=10).max() - 1).astype('float32')
max_drop20 = R1.rolling(20, min_periods=10).min().astype('float32')
drop5_raw = (R1 <= -0.05).astype('float32').where(R1.notna())
drop5_count20 = drop5_raw.rolling(20, min_periods=10).sum().astype('float32')
drop5_count60 = drop5_raw.rolling(60, min_periods=30).sum().astype('float32')
log_price = np.log(C_).astype('float32')
log_amt20 = np.log1p(M('amt20')).astype('float32')
down_vol_ratio20 = (V_.where(R1 < 0).rolling(20, min_periods=10).mean() /
                    V_.where(R1 > 0).rolling(20, min_periods=10).mean()).clip(0, 20).astype('float32')
# 60 日 beta（对全 A 等权收益）
mkt_r1 = R1.where(valid).mean(axis=1)
_mkt_mean60 = mkt_r1.rolling(60, min_periods=40).mean()
_cov60 = (R1.mul(mkt_r1, axis=0)).rolling(60, min_periods=40).mean() - \
         R1.rolling(60, min_periods=40).mean().mul(_mkt_mean60, axis=0)
_var60 = (mkt_r1 ** 2).rolling(60, min_periods=40).mean() - _mkt_mean60 ** 2
beta60 = _cov60.div(_var60, axis=0).clip(-3, 5).astype('float32')

# ---------------- 6. 市场状态（精确事件口径） ----------------
log('计算市场/行业状态')
mkt_ld_share = ld_seal.where(valid).mean(axis=1).astype('float32')
mkt_ld_touch_share = ld_touch.where(valid).mean(axis=1).astype('float32')
mkt_lu_share = lu_seal.where(valid).mean(axis=1).astype('float32')
mkt_ld_count = ld_seal.where(valid).sum(axis=1).astype('float32')
mkt_ld_rank250 = mkt_ld_share.rolling(250, min_periods=60).apply(
    lambda x: float((x[-1] >= x).mean()), raw=True).astype('float32')
mkt_ret5 = (1 + mkt_r1.fillna(0)).cumprod().pct_change(5).astype('float32')
mkt_ret20 = (1 + mkt_r1.fillna(0)).cumprod().pct_change(20).astype('float32')
mkt_above_ma20 = (M('market_index') > M('market_ma20')).astype('float32')
mkt_below_ma60 = (M('market_index') < M('market_ma60')).astype('float32')

industry_id = M('industry_id').fillna(0).astype('int16')
IND = industry_id.values
n_ind = len(P['meta']['industries'])
ind_ld = np.full((D, n_ind + 1), np.nan, dtype='float32')
ind_ld_touch = np.full((D, n_ind + 1), np.nan, dtype='float32')
ind_lu = np.full((D, n_ind + 1), np.nan, dtype='float32')
ind_ret1 = np.full((D, n_ind + 1), np.nan, dtype='float32')
ind_ret5 = np.full((D, n_ind + 1), np.nan, dtype='float32')
valid_np = valid.values
ld_np = ld_seal.values.astype('float32')
ldt_np = ld_touch.values.astype('float32')
lu_np = lu_seal.values.astype('float32')
r1_np = R1.values.astype('float32')
r5_np = M('ret5').values.astype('float32')
for k in range(1, n_ind + 1):
    m = (IND == k) & valid_np
    cnt = m.sum(axis=1)
    pos = cnt > 0
    ind_ld[pos, k] = (ld_np * m).sum(axis=1)[pos] / cnt[pos]
    ind_ld_touch[pos, k] = (ldt_np * m).sum(axis=1)[pos] / cnt[pos]
    ind_lu[pos, k] = (lu_np * m).sum(axis=1)[pos] / cnt[pos]
    ind_ret1[pos, k] = np.where(m, r1_np, 0).sum(axis=1)[pos] / cnt[pos]
    ind_ret5[pos, k] = np.where(m, r5_np, 0).sum(axis=1)[pos] / cnt[pos]
def map_ind(arr):
    return pd.DataFrame(arr[np.arange(D)[:, None], IND], index=dates, columns=codes).astype('float32')
sector_ld_share = map_ind(ind_ld)
sector_ld_touch_share = map_ind(ind_ld_touch)
sector_lu_share = map_ind(ind_lu)
sector_ret1 = map_ind(ind_ret1)
sector_ret5 = map_ind(ind_ret5)
log('市场/行业状态完成')

# ---------------- 7. 标签 ----------------
log('构造标签')
ld1 = ld_seal.values.astype('int8')
ldt1 = ld_touch.values.astype('int8')
one_word1 = one_word_ld.values.astype('int8')
ld5 = ld_seal.iloc[::-1].rolling(5, min_periods=5).max().iloc[::-1]
ld_count5 = ld_seal.iloc[::-1].rolling(5, min_periods=5).sum().iloc[::-1]
ld5_np = ld5.values.astype('float32')
ld_count5_np = ld_count5.values.astype('float32')

# ---------------- 8. 特征矩阵清单 ----------------
feat = {
    'ret1': R1, 'ret3': ret3, 'ret5': M('ret5'), 'ret10': ret10,
    'ret20': M('ret20'), 'ret60': M('ret60'),
    'dist_ma5': dist_ma5, 'dist_ma10': dist_ma10, 'dist_ma20': M('dist_ma20'),
    'dist_ma60': M('dist_ma60'), 'dist_ma120': dist_ma120,
    'ma20_slope': M('ma20_slope'), 'ma60_slope': M('ma60_slope'),
    'up_struct': M('up_struct'), 'trend_age': M('trend_age'),
    'dist_high20': M('dist_high20'), 'dist_high60': M('dist_high60'),
    'dist_high250': M('dist_high250'), 'dd20': dd20, 'dd60': M('dd60'), 'dd250': M('dd250'),
    'newhigh20': M('newhigh20'), 'newhigh60': M('newhigh60'), 'newhigh250': M('newhigh250'),
    'nh20_count': M('nh20_count'), 'rsi14': M('rsi14'), 'macd_hist': M('macd_hist'),
    'vol20': M('vol20'), 'vol60': M('vol60'), 'atr_pct': M('atr_pct'),
    'vol_hist_z': M('vol_hist_z'), 'er20': M('er20'),
    'max_drop20': max_drop20, 'drop5_count20': drop5_count20, 'drop5_count60': drop5_count60,
    'ld_count20': ld_count20, 'ld_count60': ld_count60,
    'ld_touch_count20': ld_touch_count20, 'ld_touch_count60': ld_touch_count60,
    'ld_open_count20': ld_open_count20,
    'lu_count5': lu_count5, 'lu_count20': lu_count20,
    'lu_open_count20': lu_open_count20, 'lu_touch_count20': lu_touch_count20,
    'ld_yesterday': ld_seal, 'lu_yesterday': lu_seal,
    'one_word_ld_yesterday': one_word_ld, 'opened_ld_yesterday': opened_ld,
    'one_word_lu_yesterday': one_word_lu, 'opened_lu_yesterday': opened_lu,
    'consec_ld': consec_ld, 'consec_lu': consec_lu,
    'days_since_ld': days_since_ld, 'days_since_lu': days_since_lu,
    'amt_ratio20': M('amt_ratio20'), 'vr5_20': M('vr5_20'), 'amt_trend': M('amt_trend'),
    'log_amt20': log_amt20, 'log_price': log_price,
    'vp_corr20': M('vp_corr20'), 'gap_count20': M('gap_count20'),
    'down_vol_ratio20': down_vol_ratio20,
    'close_pos': M('close_pos'), 'upper': M('upper'), 'lower': M('lower'),
    'body': M('body'), 'gap': M('gap'),
    'st_proxy': st_proxy, 'beta60': beta60,
    'market_breadth20': M('market_breadth20'), 'market_breadth60': M('market_breadth60'),
    'market_ret20': M('market_ret20'), 'market_ret60': M('market_ret60'),
    'market_slope60': M('market_slope60'), 'market_sent': M('market_sent'),
    'market_amt_ratio': M('market_amt_ratio'),
    'mkt_ret1': mkt_r1.astype('float32'), 'mkt_ret5': mkt_ret5,
    'mkt_ret20': mkt_ret20,
    'mkt_ld_share': mkt_ld_share, 'mkt_ld_touch_share': mkt_ld_touch_share,
    'mkt_lu_share': mkt_lu_share, 'mkt_ld_rank250': mkt_ld_rank250,
    'mkt_above_ma20': mkt_above_ma20, 'mkt_below_ma60': mkt_below_ma60,
    'sector_breadth20': M('sector_breadth20'), 'sector_breadth60': M('sector_breadth60'),
    'sector_ret20': M('sector_ret20'), 'sector_ret60': M('sector_ret60'),
    'sector_rs': M('sector_rs'), 'sw_ret20': M('sw_ret20'), 'sw_ret60': M('sw_ret60'),
    'sw_above60': M('sw_above60'), 'sw_slope60': M('sw_slope60'),
    'sector_ld_share': sector_ld_share, 'sector_ld_touch_share': sector_ld_touch_share,
    'sector_lu_share': sector_lu_share, 'sector_ret1': sector_ret1, 'sector_ret5': sector_ret5,
    'age': pd.DataFrame(np.clip(age, 0, 999), index=dates, columns=codes),
    'is_new': pd.DataFrame(is_new.astype('int8'), index=dates, columns=codes),
}
log(f'特征数 {len(feat)}')

# ---------------- 9. 样本与扁平化 ----------------
sample = valid.values & valid_prev.values & (age > 60) & (board_id[None, :] != 4)
rows, cols = np.where(sample)
log(f'主样本股票-日 {len(rows):,}')
out = pd.DataFrame({
    'date': dates.values[rows],
    'code_id': cols.astype('int32'),
    'board_id': board_id[cols],
    'industry_id': IND[rows, cols].astype('int16'),
    'approx': approx_flag[cols],
    'ld1': ld1[rows, cols],
    'ldt1': ldt1[rows, cols],
    'one_word1': one_word1[rows, cols],
    'ld5': ld5_np[rows, cols],
    'ld_count5': ld_count5_np[rows, cols],
})
for name in list(feat.keys()):
    df = feat.pop(name)
    if isinstance(df, pd.Series):
        vals = df.shift(1).values[rows]
    else:
        vals = df.shift(1).values[rows, cols]
    out[name] = vals.astype('float32')
    if len(out.columns) % 10 == 0:
        log(f'  特征 {len(out.columns)}')
log(f'扁平表 {out.shape}，内存 {out.memory_usage(deep=True).sum()/1e9:.1f} GB')
out.to_parquet(os.path.join(OUT, '分析面板.parquet'), index=False, compression='zstd')
log('分析面板已保存')

# 代码表
pd.DataFrame({'code_id': np.arange(C), 'code': codes, 'board': board,
              'approx': approx_flag}).to_csv(os.path.join(OUT, '代码表.csv'), index=False, encoding='utf-8-sig')

# 基准率
def masked_sum(x):
    return (x.values * sample).sum(axis=1)
base = pd.DataFrame({
    'date': dates,
    '样本数': sample.sum(axis=1),
    '收盘跌停数': masked_sum(ld_seal),
    '盘中触及数': masked_sum(ld_touch),
    '一字跌停数': masked_sum(one_word_ld),
    '打开跌停数': masked_sum(opened_ld),
    '收盘涨停数': masked_sum(lu_seal),
    '市场等权收益': mkt_r1.values,
    '市场宽度20': M('market_breadth20').values,
    '市场跌停占比': mkt_ld_share.values,
})
base['收盘跌停率'] = base['收盘跌停数'] / base['样本数']
base['盘中触及率'] = base['盘中触及数'] / base['样本数']
base.to_csv(os.path.join(OUT, '基准率.csv'), index=False, encoding='utf-8-sig')

# 市场日表
mkt = pd.DataFrame({
    'date': dates, 'market_index': M('market_index'), 'market_ma20': M('market_ma20'),
    'market_ma60': M('market_ma60'), 'market_breadth20': M('market_breadth20'),
    'market_breadth60': M('market_breadth60'), 'market_ret20': M('market_ret20'),
    'market_ret60': M('market_ret60'), 'market_slope60': M('market_slope60'),
    'market_sent': M('market_sent'), 'market_amt_ratio': M('market_amt_ratio'),
    'regime_id': M('regime_id'), 'mkt_ret1': mkt_r1, 'mkt_ret5': mkt_ret5,
    'mkt_ret20': mkt_ret20, 'mkt_ld_share': mkt_ld_share,
    'mkt_ld_touch_share': mkt_ld_touch_share, 'mkt_lu_share': mkt_lu_share,
    'mkt_ld_count': mkt_ld_count, 'mkt_ld_rank250': mkt_ld_rank250,
    'mkt_above_ma20': mkt_above_ma20, 'mkt_below_ma60': mkt_below_ma60,
})
mkt.to_csv(os.path.join(OUT, '市场日表.csv'), index=False, encoding='utf-8-sig')

# 行业日表
ind_rows = []
for k in range(1, n_ind + 1):
    ind_rows.append(pd.DataFrame({
        'date': dates, 'industry_id': k, 'industry': P['meta']['industries'][k - 1],
        '跌停占比': ind_ld[:, k], '触及跌停占比': ind_ld_touch[:, k],
        '涨停占比': ind_lu[:, k], '行业收益1': ind_ret1[:, k], '行业收益5': ind_ret5[:, k],
    }))
pd.concat(ind_rows, ignore_index=True).to_csv(os.path.join(OUT, '行业日表.csv'), index=False, encoding='utf-8-sig')

# 事件清单
ev_main = ev[(ev['ld_ratio'] > 0)].copy()
ev_main['one_word'] = ev_ow.values[ev['ld_ratio'] > 0]
# 用矩阵取值，避免逐行循环
ev_main['consec_ld'] = consec_ld.values[di[ev['ld_ratio'] > 0], ci[ev['ld_ratio'] > 0]]
ev_main['approx'] = ev['approx'].values[ev['ld_ratio'] > 0]
ev_main = ev_main[['date', 'thscode', 'board', 'ld_ratio', 'one_word', 'high', 'low', 'close',
                   'volume', 'amount', 'approx', 'consec_ld']]
ev_main.to_csv(os.path.join(OUT, '跌停事件清单.csv'), index=False, encoding='utf-8-sig')
ev_lu = ev[(ev['lu_ratio'] > 0)].copy()
ev_lu['one_word'] = ev_lu_ow.values[ev['lu_ratio'] > 0]
ev_lu = ev_lu[['date', 'thscode', 'board', 'lu_ratio', 'one_word', 'close', 'volume', 'amount', 'approx']]
ev_lu.to_csv(os.path.join(OUT, '涨停事件清单.csv'), index=False, encoding='utf-8-sig')

# 字段说明
desc = {
    'ret1': '前1日收益', 'ret3': '前3日收益', 'ret5': '前5日收益', 'ret10': '前10日收益',
    'ret20': '前20日收益', 'ret60': '前60日收益', 'dist_ma5': '距5日均线', 'dist_ma10': '距10日均线',
    'dist_ma20': '距20日均线', 'dist_ma60': '距60日均线', 'dist_ma120': '距120日均线',
    'ma20_slope': '20日均线10日斜率', 'ma60_slope': '60日均线20日斜率',
    'up_struct': '上涨结构', 'trend_age': '趋势年龄', 'dist_high20': '距20日高点',
    'dist_high60': '距60日高点', 'dist_high250': '距250日高点', 'dd20': '20日回撤',
    'dd60': '60日回撤', 'dd250': '250日回撤', 'newhigh20': '创20日新高',
    'newhigh60': '创60日新高', 'newhigh250': '创250日新高', 'nh20_count': '20日新高次数',
    'rsi14': 'RSI14', 'macd_hist': 'MACD柱', 'vol20': '20日年化波动', 'vol60': '60日年化波动',
    'atr_pct': 'ATR14/收盘', 'vol_hist_z': '波动自身历史z', 'er20': '20日效率系数',
    'max_drop20': '20日最大单日跌幅', 'drop5_count20': '20日跌超5%次数',
    'drop5_count60': '60日跌超5%次数', 'ld_count20': '20日收盘跌停次数',
    'ld_count60': '60日收盘跌停次数', 'ld_touch_count20': '20日触及跌停次数',
    'ld_touch_count60': '60日触及跌停次数', 'ld_open_count20': '20日打开跌停次数',
    'lu_count5': '5日收盘涨停次数', 'lu_count20': '20日收盘涨停次数',
    'lu_open_count20': '20日炸板次数', 'lu_touch_count20': '20日触及涨停次数',
    'ld_yesterday': '昨日收盘跌停', 'lu_yesterday': '昨日收盘涨停',
    'one_word_ld_yesterday': '昨日一字跌停', 'opened_ld_yesterday': '昨日打开跌停',
    'one_word_lu_yesterday': '昨日一字涨停', 'opened_lu_yesterday': '昨日炸板',
    'consec_ld': '连续跌停天数', 'consec_lu': '连续涨停天数',
    'days_since_ld': '距上次跌停天数', 'days_since_lu': '距上次涨停天数',
    'amt_ratio20': '成交额/20日均额', 'vr5_20': '5日/20日量比', 'amt_trend': '20日/60日成交额',
    'log_amt20': '20日均额对数', 'log_price': '收盘价对数', 'vp_corr20': '20日量价相关',
    'gap_count20': '20日跳空次数', 'down_vol_ratio20': '跌日/涨日成交量比',
    'close_pos': '收盘位置', 'upper': '上影线', 'lower': '下影线', 'body': '实体', 'gap': '跳空',
    'st_proxy': 'ST代理（60日5%涨跌停）', 'beta60': '60日市场beta',
    'market_breadth20': '市场宽度20', 'market_breadth60': '市场宽度60',
    'market_ret20': '市场20日收益', 'market_ret60': '市场60日收益',
    'market_slope60': '市场60日均线斜率', 'market_sent': '市场情绪分',
    'market_amt_ratio': '市场成交额量比', 'mkt_ret1': '全A等权前1日收益',
    'mkt_ret5': '全A等权前5日收益', 'mkt_ret20': '全A等权前20日收益',
    'mkt_ld_share': '全市场收盘跌停占比', 'mkt_ld_touch_share': '全市场触及跌停占比',
    'mkt_lu_share': '全市场收盘涨停占比', 'mkt_ld_rank250': '市场跌停占比250日分位',
    'mkt_above_ma20': '市场指数在MA20上方', 'mkt_below_ma60': '市场指数在MA60下方',
    'sector_breadth20': '行业宽度20', 'sector_breadth60': '行业宽度60',
    'sector_ret20': '行业20日收益', 'sector_ret60': '行业60日收益', 'sector_rs': '行业相对强弱',
    'sw_ret20': '申万行业指数20日收益', 'sw_ret60': '申万行业指数60日收益',
    'sw_above60': '行业指数在MA60上方', 'sw_slope60': '行业指数60日均线斜率',
    'sector_ld_share': '行业收盘跌停占比', 'sector_ld_touch_share': '行业触及跌停占比',
    'sector_lu_share': '行业收盘涨停占比', 'sector_ret1': '行业前1日收益', 'sector_ret5': '行业前5日收益',
    'age': '样本内年龄（交易日）', 'is_new': '是否次新（<=60日）',
}
pd.DataFrame({'字段': list(desc.keys()), '中文说明': list(desc.values())}).to_csv(
    os.path.join(OUT, '字段说明.csv'), index=False, encoding='utf-8-sig')
log('全部输出完成')
