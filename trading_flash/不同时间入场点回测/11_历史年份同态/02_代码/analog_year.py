# -*- coding: utf-8 -*-
"""历史年份同态：当前（2026-10-23）最像历史上哪一年？

只读输入：
- trading_hithink/初始研究/data/work/market_daily.parquet（沪深主板等权市场日频；
  若不存在，回退到 trading_hithink/data/work/market_daily.parquet）
- 07_市场阶段适配/03_结果/起点固定持有期.csv（251 个信号日起点的 20/60/120 日账户路径）
- 2026-10-24_主板恐慌反转_年化30/03_结果/最终_年度.csv、最终_逐笔.csv
- 07_市场阶段适配/03_结果/全部信号日逐只_带阶段.csv

输出：03_结果/*.csv、指标.json
口径：
- “当前”= 数据最后一根日线（应为 2026-10-23），只用当日及以前的数据。
- 相似度 = 特征稳健标准化（中位数/IQR）后的欧氏距离；全样本标准化仅用于描述，
  不构成交易信号。2026 年不参加“历史年份”排名。
- 未来收益 = 等权市场指数未来 20/60/120 个交易日收益；不是策略收益。
- 策略起点收益 = 同态区间内、代表日后 15 个交易日内第一个信号日起点，2 仓 50% 的账户路径；
  另给区间内全部起点的中位数，避免只挑一个起点。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
FLASH = HERE.parents[3]
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

MARKET = FLASH.parent / 'trading_hithink' / '初始研究' / 'data' / 'work' / 'market_daily.parquet'
if not MARKET.exists():
    MARKET = FLASH.parent / 'trading_hithink' / 'data' / 'work' / 'market_daily.parquet'
if not MARKET.exists():
    raise FileNotFoundError('找不到 market_daily.parquet；请检查 trading_hithink 数据目录')
BASE07 = FLASH / '不同时间入场点回测' / '07_市场阶段适配' / '03_结果'
MOTHER = FLASH / '2026-10-24_主板恐慌反转_年化30' / '03_结果'
START_FILE = BASE07 / '起点固定持有期.csv'
SIGNAL_FILE = BASE07 / '全部信号日逐只_带阶段.csv'
ANNUAL_FILE = MOTHER / '最终_年度.csv'

SAMPLE_START = pd.Timestamp('2018-05-01')
FUTURE_HORIZONS = (20, 60, 120)


# ------------------------------------------------------------------ 数据
m = pd.read_parquet(MARKET)
m['date'] = pd.to_datetime(m['date'])
m = m.sort_values('date').drop_duplicates('date').reset_index(drop=True)

idx = m['equal_weight_index']
m['ma250'] = idx.rolling(250).mean()
m['ret20'] = idx.pct_change(20)
m['ret60'] = idx.pct_change(60)
m['ret120'] = idx.pct_change(120)
m['ret250'] = idx.pct_change(250)
m['dd250'] = idx / idx.rolling(250).max() - 1
m['log_ma250'] = np.log(idx / m['ma250'])
m['vol_ratio'] = m['market_volatility20'] / m['market_volatility20'].rolling(60).mean()
# 跌停占比是厚尾比例；先 log1p 压缩，避免 2018 年 3.4% 的单日把距离完全主导。
m['log_limitdown'] = np.log1p(m['limit_down10_share'] * 1000)

for h in FUTURE_HORIZONS:
    m[f'f{h}'] = idx.shift(-h) / idx - 1
    m[f'min{h}'] = idx.iloc[::-1].rolling(h, min_periods=1).min().iloc[::-1] / idx - 1
    m[f'max{h}'] = idx.iloc[::-1].rolling(h, min_periods=1).max().iloc[::-1] / idx - 1

# 宽度回到 50%、指数收复年线的交易日数（只看未来 250 日）
br = m['breadth20'].to_numpy()
ix = idx.to_numpy()
ma250 = m['ma250'].to_numpy()


def days_until(cond: np.ndarray, i: int, limit: int = 250) -> float:
    j = i + 1
    while j < len(cond) and j <= i + limit:
        if bool(cond[j]):
            return float(j - i)
        j += 1
    return np.nan


m['d_breadth50'] = [days_until(br > 0.5, i) for i in range(len(m))]
m['d_ma250'] = [days_until(ix > ma250, i) for i in range(len(m))]


def phase_of(r: pd.Series) -> str:
    if pd.isna(r['ret250']):
        return 'NA'
    if r['equal_weight_index'] > r['ma250'] and r['ret250'] > 0:
        return 'A 牛市上升段急跌'
    if r['equal_weight_index'] <= r['ma250'] and r['ret250'] > 0:
        return 'B 高位转弱/跌破年线'
    if r['equal_weight_index'] > r['ma250'] and r['ret250'] <= 0:
        return 'C 熊市反弹'
    return 'D 熊市下跌'


m['阶段'] = m.apply(phase_of, axis=1)

cur = m.iloc[-1]
CURRENT_DATE = cur['date']
assert CURRENT_DATE == pd.Timestamp('2026-10-23'), f'当前日期变了：{CURRENT_DATE}'

# ------------------------------------------------------------------ 当前状态
signal = pd.read_csv(SIGNAL_FILE, parse_dates=['signal_date'])
signal_complete = signal[signal['complete'] == True].copy()  # noqa: E712

cur_rows = [
    ('等权指数', cur['equal_weight_index'], '点', '沪深主板等权，2016-10-13=1'),
    ('250日均线', cur['ma250'], '点', '等权指数 250 日均线'),
    ('距250日高', cur['dd250'], '%', '250 日最高点回撤；当前高点 2026-02-27'),
    ('250日涨幅', cur['ret250'], '%', '过去 250 个交易日收益'),
    ('120日涨幅', cur['ret120'], '%', '过去 120 个交易日收益'),
    ('60日涨幅', cur['ret60'], '%', '过去 60 个交易日收益'),
    ('20日涨幅', cur['ret20'], '%', '过去 20 个交易日收益'),
    ('20日波动率', cur['market_volatility20'], '比值', '等权指数日收益 20 日年化波动'),
    ('宽度20', cur['breadth20'], '比例', '站上 20 日线的股票占比'),
    ('宽度60', cur['breadth60'], '比例', '站上 60 日线的股票占比'),
    ('情绪分', cur['sentiment_score'], '分', '本地情绪综合分'),
    ('跌停占比', cur['limit_down10_share'], '比例', '跌停/接近跌停股票占比'),
    ('20/60日成交额比', cur['amount_ratio20_60'], '比值', '缩量/放量'),
]
state_rows = []
for name, value, unit, note in cur_rows:
    col = {
        '等权指数': 'equal_weight_index', '250日均线': 'ma250', '距250日高': 'dd250',
        '250日涨幅': 'ret250', '120日涨幅': 'ret120', '60日涨幅': 'ret60',
        '20日涨幅': 'ret20', '20日波动率': 'market_volatility20', '宽度20': 'breadth20',
        '宽度60': 'breadth60', '情绪分': 'sentiment_score', '跌停占比': 'limit_down10_share',
        '20/60日成交额比': 'amount_ratio20_60',
    }[name]
    full = m[m['date'] >= SAMPLE_START]
    sig = pd.Series(dtype=float)
    if name == '250日涨幅':
        sig = signal_complete['mkt_ret250']
    elif name == '距250日高':
        sig = signal_complete['mkt_dd250']
    elif name == '20日波动率':
        sig = signal_complete['market_volatility20']
    elif name == '宽度20':
        sig = signal_complete['breadth20']
    # 指数/均线是水平量，跨年份比较会混入趋势，不给百分位。
    is_level = name in ('等权指数', '250日均线')
    state_rows.append({
        '指标': name, '当前值': value, '单位': unit,
        '历史百分位': np.nan if is_level else float((full[col] < value).mean()),
        '信号日百分位': float((sig < value).mean()) if len(sig.dropna()) else np.nan,
        '历史中位': np.nan if is_level else float(full[col].median()),
        '历史P10': np.nan if is_level else float(full[col].quantile(.10)),
        '历史P90': np.nan if is_level else float(full[col].quantile(.90)),
        '说明': note,
    })
state = pd.DataFrame(state_rows)
state.to_csv(RES / '当前状态.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# ------------------------------------------------------------------ 相似度口径
SPECS = {
    '全状态13': ['log_ma250', 'ret250', 'dd250', 'ret120', 'ret60', 'ret20',
                'market_volatility20', 'vol_ratio', 'breadth20', 'breadth60',
                'sentiment_score', 'log_limitdown', 'amount_ratio20_60'],
    '核心8': ['log_ma250', 'ret250', 'dd250', 'ret120', 'market_volatility20',
             'breadth20', 'breadth60', 'sentiment_score'],
    '状态5': ['ret250', 'dd250', 'market_volatility20', 'breadth20', 'breadth60'],
    '路径6': ['ret250', 'dd250', 'ret60', 'market_volatility20', 'breadth20', 'breadth60'],
}

hist = m[(m['date'] >= SAMPLE_START)].copy()
for name, feats in SPECS.items():
    X = m[feats].copy()
    lo, hi = X.quantile(.01), X.quantile(.99)
    X = X.clip(lo, hi, axis=1)
    iqr = (X.quantile(.75) - X.quantile(.25)).replace(0, np.nan)
    Z = (X - X.median()) / iqr
    Z = Z.replace([np.inf, -np.inf], np.nan)
    cz = Z.iloc[-1]
    dist = np.sqrt(((Z - cz) ** 2).mean(axis=1))
    m[f'd_{name}'] = dist
    hist[f'd_{name}'] = dist.loc[hist.index]

# 只保留历史年（2026 不参与年度排名）
hist_year = hist[hist['date'] < '2026-01-01'].copy()

year_rows = []
for y, g in hist_year.groupby(hist_year['date'].dt.year):
    row = {'年份': int(y)}
    for name in SPECS:
        g2 = g.dropna(subset=[f'd_{name}'])
        row[f'{name}_距离'] = float(g2[f'd_{name}'].nsmallest(10).mean()) if len(g2) else np.nan
        row[f'{name}_最近'] = float(g2[f'd_{name}'].min()) if len(g2) else np.nan
    gs = g.dropna(subset=['d_核心8']).sort_values('d_核心8')
    row['核心8最近日期'] = gs['date'].iloc[0].date().isoformat() if len(gs) else ''
    row['核心8最近日未来60日'] = float(gs['f60'].iloc[0]) if len(gs) and pd.notna(gs['f60'].iloc[0]) else np.nan
    row['核心8最近日未来120日'] = float(gs['f120'].iloc[0]) if len(gs) and pd.notna(gs['f120'].iloc[0]) else np.nan
    row['核心8_top100天数'] = int((hist_year.dropna(subset=['d_核心8']).nsmallest(100, 'd_核心8')['date'].dt.year == y).sum())
    year_rows.append(row)

years = pd.DataFrame(year_rows)
for name in SPECS:
    years[f'{name}_排名'] = years[f'{name}_距离'].rank(method='min').astype(int)
years['综合平均排名'] = years[[f'{name}_排名' for name in SPECS]].mean(axis=1)
years = years.sort_values('综合平均排名')

# 年度策略收益、市场收益、单笔统计
annual = pd.read_csv(ANNUAL_FILE, parse_dates=['date'])
annual['年份'] = annual['date'].dt.year
annual = annual.rename(columns={'return': '策略年度收益'})
annual['市场年度收益'] = np.nan
for y in annual['年份']:
    if y == 2017:
        a = m[(m['date'] >= '2017-11-23') & (m['date'].dt.year == 2017)]
        b = m[(m['date'] < '2017-11-23') & (m['date'].dt.year == 2017)]
        annual.loc[annual['年份'] == y, '市场年度收益'] = a['equal_weight_index'].iloc[-1] / b['equal_weight_index'].iloc[-1] - 1 if len(a) and len(b) else np.nan
    elif y == 2026:
        a = m[(m['date'] >= '2025-12-31') & (m['date'] <= CURRENT_DATE)]
        annual.loc[annual['年份'] == y, '市场年度收益'] = a['equal_weight_index'].iloc[-1] / a['equal_weight_index'].iloc[0] - 1
    else:
        a = m[m['date'].dt.year == y]
        prev = m[(m['date'] < f'{y}-01-01')]
        if len(a) and len(prev):
            annual.loc[annual['年份'] == y, '市场年度收益'] = a['equal_weight_index'].iloc[-1] / prev['equal_weight_index'].iloc[-1] - 1

trades = signal_complete.copy()
trades['年份'] = trades['signal_date'].dt.year
tr = trades.groupby('年份').agg(单笔数=('ret', 'size'), 单笔均值=('ret', 'mean'),
                                胜率=('ret', lambda x: (x > 0).mean()))
years2 = years.merge(annual[['年份', '策略年度收益', '市场年度收益']], on='年份', how='left')
years2 = years2.merge(tr, on='年份', how='left')
years2 = years2[[
    '年份', '全状态13_距离', '全状态13_排名', '核心8_距离', '核心8_排名',
    '状态5_距离', '状态5_排名', '路径6_距离', '路径6_排名', '综合平均排名',
    '核心8最近日期', '核心8最近日未来60日', '核心8最近日未来120日', '核心8_top100天数',
    '策略年度收益', '市场年度收益', '单笔数', '单笔均值', '胜率',
]]
years2.to_csv(RES / '年度相似度.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 最近邻明细
nn_cols = ['date', 'd_全状态13', 'd_核心8', 'd_状态5', 'd_路径6',
           'equal_weight_index', 'ret250', 'dd250', 'ret120', 'ret60', 'ret20',
           'market_volatility20', 'breadth20', 'breadth60', 'sentiment_score',
           'limit_down10_share', 'amount_ratio20_60',
           'f20', 'f60', 'f120', 'min60', 'max60', 'd_breadth50', 'd_ma250', '阶段']
nn = hist_year.sort_values('d_核心8').head(40)[nn_cols].copy()
nn.to_csv(RES / '最近邻_核心8.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
nn2 = hist_year.sort_values('d_全状态13').head(40)[nn_cols].copy()
nn2.to_csv(RES / '最近邻_全状态13.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

all_hist = hist_year[['date', 'd_核心8', 'f20', 'f60', 'f120', 'ret250', 'dd250',
                      'breadth20', 'market_volatility20']].copy()
all_hist.to_csv(RES / '全部历史日_距离.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 距离有没有预测力：全样本相关 + 最近 10% 对比
pred_rows = []
q10 = hist_year['d_核心8'].quantile(.10)
close = hist_year[hist_year['d_核心8'] <= q10]
rest = hist_year[hist_year['d_核心8'] > q10]
for h in FUTURE_HORIZONS:
    sub = hist_year[['d_核心8', f'f{h}']].dropna()
    pred_rows.append({
        '口径': f'未来{h}日',
        'Pearson': sub['d_核心8'].corr(sub[f'f{h}']),
        'Spearman': sub['d_核心8'].corr(sub[f'f{h}'], method='spearman'),
        '最近10%均值': close[f'f{h}'].mean(),
        '最近10%中位': close[f'f{h}'].median(),
        '最近10%胜率': (close[f'f{h}'] > 0).mean(),
        '其余均值': rest[f'f{h}'].mean(),
        '其余中位': rest[f'f{h}'].median(),
        '其余胜率': (rest[f'f{h}'] > 0).mean(),
        '最近10%样本': len(close),
    })
pred = pd.DataFrame(pred_rows)
pred.to_csv(RES / '距离预测力.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# ------------------------------------------------------------------ 距离阈值事件
EPISODE_THRESHOLDS = (0.25, 0.35, 0.45, 0.55)


def make_episodes(threshold: float, sample: pd.DataFrame, gap: int = 20) -> pd.DataFrame:
    sel = sample[(sample['d_核心8'] <= threshold) & sample['f120'].notna()].copy()
    episodes = []
    for i, r in sel.iterrows():
        if not episodes or (i - episodes[-1][-1]) > gap:
            episodes.append([i])
        else:
            episodes[-1].append(i)
    rows = []
    for ep in episodes:
        sub = m.loc[ep]
        rep = sub.loc[sub['d_核心8'].idxmin()]
        pos = int(m.index[m['date'] == rep['date']][0])
        win = m.iloc[pos:pos + 251]
        low = win.loc[win['equal_weight_index'].idxmin()]
        rows.append({
            '阈值': threshold,
            '起点': m.loc[ep[0], 'date'].date().isoformat(),
            '终点': m.loc[ep[-1], 'date'].date().isoformat(),
            '天数': len(ep),
            '代表日': rep['date'].date().isoformat(),
            '最小距离': rep['d_核心8'],
            '未来20日': rep['f20'],
            '未来60日': rep['f60'],
            '未来120日': rep['f120'],
            '60日内最大回撤': rep['min60'],
            '60日内最大反弹': rep['max60'],
            '后续最低点日期': low['date'].date().isoformat(),
            '后续最低点回撤': float(low['equal_weight_index'] / rep['equal_weight_index'] - 1),
            '后续最低点交易日': int(m.index.get_loc(low.name) - pos),
            '宽度回50%天数': rep['d_breadth50'],
            '收复年线天数': rep['d_ma250'],
        })
    return pd.DataFrame(rows)


summary_rows = []
for th in EPISODE_THRESHOLDS:
    E = make_episodes(th, hist_year)
    if not len(E):
        continue
    summary_rows.append({
        '阈值': th, '事件数': len(E),
        '未来20日均值': E['未来20日'].mean(), '未来20日中位': E['未来20日'].median(),
        '未来20日胜率': (E['未来20日'] > 0).mean(),
        '未来60日均值': E['未来60日'].mean(), '未来60日中位': E['未来60日'].median(),
        '未来60日胜率': (E['未来60日'] > 0).mean(),
        '未来120日均值': E['未来120日'].mean(), '未来120日中位': E['未来120日'].median(),
        '未来120日胜率': (E['未来120日'] > 0).mean(),
        '最差60日': E['未来60日'].min(),
        '最好60日': E['未来60日'].max(),
        '最差120日': E['未来120日'].min(),
        '最好120日': E['未来120日'].max(),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(RES / '同态阈值汇总.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

episodes = make_episodes(0.35, hist_year)
episodes['主同态'] = np.where(episodes['最小距离'] <= 0.25, '是', '否')

# ------------------------------------------------------------------ 规则状态（不依赖距离）
starts = pd.read_csv(START_FILE, parse_dates=['signal_date'])
calendar = m[['date']].reset_index(drop=True)
rule_mask = (
    (m['date'] >= SAMPLE_START)
    & (m['ret250'] < 0)
    & (m['dd250'].between(-0.22, -0.12))
    & (m['breadth20'].between(0.12, 0.25))
    & (m['market_volatility20'].between(0.16, 0.23))
    & (m['date'] < '2026-01-01')
    & m['f120'].notna()
)
rule_sel = m[rule_mask]
rule_eps = []
for i, r in rule_sel.iterrows():
    if not rule_eps or (i - rule_eps[-1][-1]) > 20:
        rule_eps.append([i])
    else:
        rule_eps[-1].append(i)


def trading_pos(day: pd.Timestamp) -> int:
    return int(calendar.index[calendar['date'] == day][0])


def trading_date(pos: int) -> pd.Timestamp:
    pos = max(0, min(len(calendar) - 1, pos))
    return calendar.loc[pos, 'date']


rule_rows = []
for ep in rule_eps:
    r = m.loc[ep[0]]
    end = m.loc[ep[-1], 'date']
    hi_date = trading_date(trading_pos(end) + 15)
    g = starts[(starts['signal_date'] >= r['date']) & (starts['signal_date'] <= hi_date)]
    g = g.sort_values('signal_date')
    nxt = g.iloc[0] if len(g) else None
    rule_rows.append({
        '起点': r['date'].date().isoformat(), '终点': m.loc[ep[-1], 'date'].date().isoformat(),
        '天数': len(ep), '250日涨幅': r['ret250'], '250日回撤': r['dd250'],
        '20日波动率': r['market_volatility20'], '宽度20': r['breadth20'],
        '未来20日': r['f20'], '未来60日': r['f60'], '未来120日': r['f120'],
        '60日内最大回撤': r['min60'], '宽度回50%天数': r['d_breadth50'],
        '收复年线天数': r['d_ma250'],
        '下一可执行起点': nxt['signal_date'].date().isoformat() if nxt is not None else '',
        '策略起点60日': nxt['ret_60'] if nxt is not None else np.nan,
        '策略起点120日': nxt['ret_120'] if nxt is not None else np.nan,
    })
rule = pd.DataFrame(rule_rows)
rule.to_csv(RES / '规则状态_事件.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# ------------------------------------------------------------------ 策略起点
start_rows = []
for _, e in episodes.iterrows():
    rep = pd.Timestamp(e['代表日'])
    pos = trading_pos(rep)
    ep_lo = pd.Timestamp(e['起点'])
    hi_date = trading_date(pos + 15)
    g = starts[(starts['signal_date'] >= ep_lo) & (starts['signal_date'] <= hi_date)].copy()
    if not len(g):
        start_rows.append({
            '代表日': e['代表日'], '最小距离': e['最小距离'], '主同态': e['主同态'],
            '起点数': 0, '下一可执行起点': '', '下一可执行起点20日': np.nan,
            '下一可执行起点60日': np.nan, '下一可执行起点120日': np.nan,
            '区间起点20日中位': np.nan, '区间起点60日中位': np.nan,
            '区间起点120日中位': np.nan, '区间起点60日最差': np.nan,
            '区间起点60日最好': np.nan, '区间起点120日回撤中位': np.nan,
        })
        continue
    g = g.sort_values('signal_date')
    nxt = g[g['signal_date'] >= rep]
    nxt = nxt.iloc[0] if len(nxt) else g.iloc[-1]
    start_rows.append({
        '代表日': e['代表日'], '最小距离': e['最小距离'], '主同态': e['主同态'],
        '起点数': len(g), '下一可执行起点': nxt['signal_date'].date().isoformat(),
        '下一可执行起点20日': nxt['ret_20'], '下一可执行起点60日': nxt['ret_60'],
        '下一可执行起点120日': nxt['ret_120'],
        '区间起点20日中位': g['ret_20'].median(), '区间起点60日中位': g['ret_60'].median(),
        '区间起点120日中位': g['ret_120'].median(),
        '区间起点60日最差': g['ret_60'].min(), '区间起点60日最好': g['ret_60'].max(),
        '区间起点120日回撤中位': g['dd_120'].median(),
    })
start_tbl = pd.DataFrame(start_rows)
start_tbl.to_csv(RES / '同态期_策略起点.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 报告/Excel 用总表
event_tbl = episodes.merge(
    start_tbl.drop(columns=['最小距离', '主同态']),
    on='代表日', how='left', suffixes=('', '_策略')
)
event_tbl['年份'] = pd.to_datetime(event_tbl['代表日']).dt.year
event_tbl['阶段'] = [
    m.loc[m['date'] == pd.Timestamp(d), '阶段'].iloc[0] for d in event_tbl['代表日']
]
event_tbl = event_tbl.rename(columns={
    '未来20日': '市场20日', '未来60日': '市场60日', '未来120日': '市场120日',
    '60日内最大回撤': '市场60日最大回撤', '60日内最大反弹': '市场60日最大反弹',
    '下一可执行起点60日': '策略下一开始点60日', '下一可执行起点120日': '策略下一开始点120日',
    '区间起点60日中位': '策略区间起点60日中位', '区间起点120日中位': '策略区间起点120日中位',
    '区间起点120日回撤中位': '策略区间起点120日回撤中位',
    '宽度回50%天数': '宽度回50%交易日', '收复年线天数': '收复年线交易日',
})
event_tbl = event_tbl[[
    '年份', '代表日', '最小距离', '主同态', '起点', '终点', '天数', '阶段',
    '市场20日', '市场60日', '市场120日', '市场60日最大回撤', '市场60日最大反弹',
    '后续最低点日期', '后续最低点回撤', '后续最低点交易日',
    '宽度回50%交易日', '收复年线交易日',
    '下一可执行起点', '策略下一开始点60日', '策略下一开始点120日',
    '起点数', '策略区间起点60日中位', '策略区间起点120日中位', '策略区间起点120日回撤中位',
]]
event_tbl.to_csv(RES / '同态事件_总表.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# 全部起点逐笔明细（同态区间内），用于画起点离散度
detail_rows = []
for _, e in episodes.iterrows():
    rep = pd.Timestamp(e['代表日'])
    pos = trading_pos(rep)
    ep_lo = pd.Timestamp(e['起点'])
    hi_date = trading_date(pos + 15)
    g = starts[(starts['signal_date'] >= ep_lo) & (starts['signal_date'] <= hi_date)].copy()
    for _, r in g.iterrows():
        detail_rows.append({
            '代表日': e['代表日'], '最小距离': e['最小距离'], '主同态': e['主同态'],
            'signal_date': r['signal_date'].date().isoformat(),
            'ret_20': r['ret_20'], 'ret_60': r['ret_60'], 'ret_120': r['ret_120'],
            'dd_60': r['dd_60'], 'dd_120': r['dd_120'],
        })
pd.DataFrame(detail_rows).to_csv(RES / '同态期_逐起点.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# ------------------------------------------------------------------ 当前同态与同年回声
cur_dist = {name: float(hist_year[f'd_{name}'].min()) for name in SPECS}
echo_2026 = m[(m['date'] >= '2026-01-01')].sort_values('d_核心8').head(5)[
    ['date', 'd_核心8', 'ret250', 'dd250', 'breadth20', 'market_volatility20', '阶段']
].copy()
echo_2026.to_csv(RES / '2026同年回声.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

# ------------------------------------------------------------------ 指标 JSON
metrics = {
    '当前日期': CURRENT_DATE.date().isoformat(),
    '当前等权指数': round(float(cur['equal_weight_index']), 6),
    '当前250日均线': round(float(cur['ma250']), 6),
    '当前阶段': cur['阶段'],
    '当前250日涨幅': round(float(cur['ret250']), 6),
    '当前250日回撤': round(float(cur['dd250']), 6),
    '当前20日波动率': round(float(cur['market_volatility20']), 6),
    '当前宽度20': round(float(cur['breadth20']), 6),
    '当前宽度60': round(float(cur['breadth60']), 6),
    '当前情绪分': round(float(cur['sentiment_score']), 6),
    '最近历史距离': {k: round(v, 6) for k, v in cur_dist.items()},
    '距离小于0.2天数': int((hist_year['d_核心8'] < 0.2).sum()),
    '距离小于0.25天数': int((hist_year['d_核心8'] < 0.25).sum()),
    '距离预测力': pred.to_dict('records'),
    '最近邻_核心8': nn.head(5)[['date', 'd_核心8', 'f60', 'f120']].astype(str).to_dict('records'),
    '年度排名_核心8': years2[['年份', '核心8_距离', '核心8_排名']].head(6).to_dict('records'),
    '主同态事件': episodes[episodes['主同态'] == '是'][['代表日', '最小距离', '未来60日', '未来120日']].to_dict('records'),
    '策略起点': start_tbl[start_tbl['主同态'] == '是'][['代表日', '下一可执行起点', '下一可执行起点60日', '下一可执行起点120日']].to_dict('records'),
    '样本说明': {
        '市场日频起点': m['date'].min().date().isoformat(),
        '市场日频终点': m['date'].max().date().isoformat(),
        '历史年份候选': '2018-2025（2026 不参加排名）',
        '起点文件': START_FILE.name,
        '相似度标准化': '中位数/IQR，全样本 1%/99% 缩尾；描述性，不作交易信号',
    },
}
with open(RES / '指标.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------ 控制台
print('当前', CURRENT_DATE.date(), cur['阶段'])
print('距离', {k: round(v, 4) for k, v in cur_dist.items()})
print('\n年度相似度（按综合平均排名）')
print(years2.head(8).round(3).to_string(index=False))
print('\n核心8 最近邻 top 10')
print(nn.head(10)[['date', 'd_核心8', 'ret250', 'dd250', 'market_volatility20', 'breadth20',
                   'sentiment_score', 'f20', 'f60', 'f120', 'min60', 'd_breadth50', 'd_ma250']].round(3).to_string(index=False))
print('\n阈值汇总')
print(summary.round(3).to_string(index=False))
print('\n同态事件')
print(episodes.round(3).to_string(index=False))
print('\n策略起点')
print(start_tbl.round(3).to_string(index=False))
print('\n规则状态事件')
print(rule.round(3).to_string(index=False))
print('\n2026 同年回声')
print(echo_2026.round(3).to_string(index=False))
