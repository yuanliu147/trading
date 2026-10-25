# -*- coding: utf-8 -*-
"""
重复信号与买入时点：同一股票连续几天出信号，什么时候买更好？

口径与母策略一致：
- 信号日收盘确认五条件；T+3 开盘买；2 仓 50%；+20% 止盈；
- 15 个交易日到期；成本 0.175%/边；一字涨停/停牌不买、不递补。

本脚本输出：
- 全部信号池的重复序列（4195 条）、前二重复序列（459 条）
- 首笔 vs 第二笔配对、位置统计、入场延迟 N=1..5
- 买入日前一天 K 线/量能/MACD/均线/市场特征分组
- 当前 2026-10-22/23 信号的 250 日回撤明细与执行方案
- 603778.SH 连续信号案例
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
MOTHER = ROOT.parent / '2026-10-24_主板恐慌反转_年化30'
SRC06 = ROOT / '06_不同信号日起点' / '03_结果'
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / '02_代码'))
import timing_lib as tl  # noqa: E402

PANEL = MOTHER / '01_数据' / 'research_panel.parquet'
MARKET = Path('/Users/admin/WorkSpace/trading/trading_hithink/data/work/market_daily.parquet')
NTH, HOLD, TP, COST = 3, 15, 0.20, 0.00175
GAP = 3  # 同一股票两次信号间隔 <=3 个交易日，算同一段“持续信号”

D = tl.load(str(PANEL))
tl.add_signal(D)
b = D['base']
dates = pd.DatetimeIndex(D['dates'])
n_days = len(dates)

ev = b[b['ev_custom']].copy()
ev['di'] = ev['date'].map({d: i for i, d in enumerate(dates)})
ev['rk'] = ev.groupby('date')['drawdown250'].rank(ascending=True, method='first')
ev['top2'] = ev['rk'] <= 2
si = {x: i for i, x in enumerate(D['syms'])}
print('信号池', len(ev), '信号日', ev.date.nunique(), '股票', ev.thscode.nunique())


def assign_episodes(frame, gap=GAP):
    """按股票分组；相邻信号日间隔 <= gap 个交易日算同一段。"""
    out = []
    eid = 0
    for s, g in frame.sort_values('di').groupby('thscode'):
        g = g.sort_values('di')
        cur = [g.iloc[0]]
        for i in range(1, len(g)):
            if g.iloc[i]['di'] - g.iloc[i - 1]['di'] <= gap:
                cur.append(g.iloc[i])
            else:
                out.append((eid, s, pd.DataFrame(cur)))
                eid += 1
                cur = [g.iloc[i]]
        out.append((eid, s, pd.DataFrame(cur)))
        eid += 1
    rows = []
    for eid, s, g in out:
        g = g.reset_index(drop=True)
        first = g.iloc[0]['di']
        for pos, r in enumerate(g.itertuples(), 1):
            rows.append(dict(eid=eid, code=s, signal_date=r.date, pos=pos,
                             ep_len=len(g), gap=int(r.di - first),
                             di=int(r.di), rk=int(r.rk), top2=bool(r.rk <= 2),
                             drawdown250=float(r.drawdown250)))
    return pd.DataFrame(rows)


def trade_status(D, s, j, nth):
    st = tl.standalone_trade(D, int(s), int(j), nth, 'open', HOLD, TP, COST)
    status = st.get('status', '')
    complete = False
    if status == 'ok':
        if st.get('exit_reason') == 'target':
            complete = True
        elif int(st['entry_i']) + HOLD - 1 <= n_days - 1:
            complete = True
        else:
            status = '持有未满'
    return st, status, complete


# ---------------------------------------------------------------- 全部信号池
E_all = assign_episodes(ev, GAP)
rows = []
for r in E_all.itertuples():
    st, status, complete = trade_status(D, si[r.code], r.di, NTH)
    rows.append(dict(eid=r.eid, code=r.code, signal_date=r.signal_date,
                     pos=r.pos, ep_len=r.ep_len, gap=r.gap, di=r.di,
                     rk=r.rk, top2=r.top2, dd250=float(r.drawdown250), status=status, complete=complete,
                     entry_date=st.get('entry_date', pd.NaT),
                     exit_date=st.get('exit_date', pd.NaT),
                     exit_reason=st.get('exit_reason', ''),
                     ret=st.get('ret', np.nan)))
A = pd.DataFrame(rows)
A_raw = A.copy()
n_ep_all = A.eid.nunique()
print('全部序列', len(A), '完整', int(A.complete.sum()), '段', n_ep_all)

# ---------------------------------------------------------------- 前二
E_top = assign_episodes(ev[ev.top2], GAP)
det = pd.read_csv(SRC06 / '逐日首笔_逐只.csv',
                  parse_dates=['signal_date', 'entry_date', 'exit_date'])
det = det[det['complete']].copy()
T = det.merge(E_top[['eid', 'pos', 'ep_len', 'gap', 'signal_date', 'code']],
              on=['signal_date', 'code'], how='left', validate='one_to_one')
T['top2'] = True
T['di'] = T.signal_date.map({d: i for i, d in enumerate(dates)})
print('前二完整', len(T), '段', T.eid.nunique())

# ---------------------------------------------------------------- 特征面板
feat_cols = ['ret1', 'amount_ratio1_20', 'macd_hist_pct', 'macd_golden_cross',
             'macd_dif_pct', 'macd_dea_pct', 'close_ma5', 'close_ma10',
             'close_ma20', 'rsi14', 'close_location', 'candle_body',
             'upper_shadow', 'lower_shadow', 'gap_open', 'range_pct',
             'amount20']
df = pd.read_parquet(PANEL, columns=['thscode', 'date'] + feat_cols)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['thscode', 'date'])
prev = df[['thscode', 'date']].copy()
for c in feat_cols:
    prev[c + '_prev'] = df.groupby('thscode')[c].shift(1)
prev = prev.rename(columns={'date': 'entry_date'})

# 市场
m = pd.read_parquet(MARKET, columns=['date', 'equal_weight_index', 'market_return',
                                     'market_volatility20', 'breadth20',
                                     'market_return20'])
m['date'] = pd.to_datetime(m['date'])
m = m.sort_values('date').reset_index(drop=True)
m['ma250'] = m['equal_weight_index'].rolling(250).mean()
m['ma5'] = m['equal_weight_index'].rolling(5).mean()
m['ma10'] = m['equal_weight_index'].rolling(10).mean()
m['mkt_ret250'] = m['equal_weight_index'].pct_change(250)
m['mkt_dd250'] = m['equal_weight_index'] / m['equal_weight_index'].rolling(250).max() - 1
m['above250'] = (m['equal_weight_index'] > m['ma250']).astype(int)
m['above5'] = (m['equal_weight_index'] > m['ma5']).astype(int)
m['above10'] = (m['equal_weight_index'] > m['ma10']).astype(int)
m['breadth_chg5'] = m['breadth20'].diff(5)


def phase(r):
    if pd.isna(r['mkt_ret250']):
        return 'NA'
    if r['above250'] == 1 and r['mkt_ret250'] > 0:
        return 'A 牛市上升段急跌'
    if r['above250'] == 0 and r['mkt_ret250'] > 0:
        return 'B 高位转弱/跌破年线'
    if r['above250'] == 1 and r['mkt_ret250'] <= 0:
        return 'C 熊市反弹'
    return 'D 熊市下跌'


m['阶段'] = m.apply(phase, axis=1)
mprev = m[['date']].copy()
for c in ['market_return', 'above5', 'above10', 'breadth20', 'breadth_chg5']:
    mprev[c + '_prev'] = m[c].shift(1)
mprev = mprev.rename(columns={'date': 'entry_date'})

msig = m[['date', 'mkt_ret250', 'mkt_dd250', 'above250', 'market_volatility20',
          'breadth20', 'market_return20', '阶段']].rename(columns={'date': 'signal_date'})

# 合并特征
def add_features(frame):
    z = frame.merge(prev, left_on=['code', 'entry_date'],
                    right_on=['thscode', 'entry_date'], how='left')
    z = z.merge(mprev, on='entry_date', how='left')
    z = z.merge(msig, on='signal_date', how='left')
    z['prev_up'] = z['ret1_prev'] > 0
    z['market_prev_up'] = z['market_return_prev'] > 0
    z['macd_hist_up'] = z['macd_hist_pct_prev'] > 0
    z['golden'] = z['macd_golden_cross_prev'] > 0
    z['ma5_up'] = z['close_ma5_prev'] > 0
    z['ma10_up'] = z['close_ma10_prev'] > 0
    z['ma20_up'] = z['close_ma20_prev'] > 0
    z['vol_hi'] = z['amount_ratio1_20_prev'] >= z['amount_ratio1_20_prev'].median()
    return z


T = add_features(T)
A = add_features(A[A.status.eq('ok') & A.complete].copy())
A['y'] = A.signal_date.dt.year
n_ep_all_complete = A.eid.nunique()
T['y'] = T.signal_date.dt.year

# ---------------------------------------------------------------- 统计函数
def stat(g):
    g = g.dropna(subset=['ret'])
    if len(g) == 0:
        return dict(笔数=0)
    ex = g[g.y < 2026]
    return dict(笔数=int(len(g)), 平均=float(g.ret.mean()),
                中位=float(g.ret.median()),
                胜率=float((g.ret > 0).mean()),
                止盈占比=float((g.exit_reason == 'target').mean()) if 'exit_reason' in g else np.nan,
                剔除2026平均=float(ex.ret.mean()) if len(ex) else np.nan,
                剔除2026胜率=float((ex.ret > 0).mean()) if len(ex) else np.nan)


def pos_table(z, label):
    rows = []
    for p in [1, 2, 3, 4]:
        g = z[z.pos == p] if p < 4 else z[z.pos >= 4]
        rec = dict(样本=label, 位置='1' if p == 1 else ('2' if p == 2 else ('3' if p == 3 else '4+')),
                   回撤中位=float(g.dd250.median()) if len(g) else np.nan)
        rec.update(stat(g))
        rows.append(rec)
    return pd.DataFrame(rows)


pos_top = pos_table(T, '前二')
pos_all = pos_table(A, '全部信号')
pos_both = pd.concat([pos_top, pos_all], ignore_index=True)
pos_both.to_csv(RES / '位置统计.csv', index=False, encoding='utf-8-sig')

# 严格连续 3 天（不要求整个段连续，只找连续 run）
def strict_third(df, label):
    rows = []
    for (eid, code), g in df.sort_values('di').groupby(['eid', 'code']):
        g = g.reset_index(drop=True)
        start = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or g.loc[i, 'di'] - g.loc[i - 1, 'di'] != 1:
                run = g.iloc[start:i]
                if len(run) >= 3:
                    r = run.iloc[2]
                    rows.append(dict(样本=label, eid=eid, 代码=code,
                                     信号日=r.signal_date, 段内位置=int(r.pos),
                                     连续天数=int(len(run)), 收益=float(r.ret),
                                     回撤250=float(r.dd250)))
                start = i
    return pd.DataFrame(rows)


strict_top = strict_third(T, '前二')
strict_all = strict_third(A, '全部信号')
strict_detail = pd.concat([strict_top, strict_all], ignore_index=True)
strict_detail.to_csv(RES / '严格连续3天_明细.csv', index=False, encoding='utf-8-sig')
strict_summary = []
for label, z in [('前二', strict_top), ('全部信号', strict_all)]:
    strict_summary.append(dict(样本=label, 笔数=len(z), 段数=z.eid.nunique(),
                               平均=z.收益.mean(), 中位=z.收益.median(),
                              胜率=(z.收益 > 0).mean()))
strict_summary = pd.DataFrame(strict_summary)
strict_summary.to_csv(RES / '严格连续3天.csv', index=False, encoding='utf-8-sig')

# 首笔 vs 第二笔（前二，配对）
pairs = []
for eid, g in T.sort_values('pos').groupby('eid'):
    if len(g) >= 2:
        r1 = g[g.pos == 1].iloc[0]
        r2 = g[g.pos == 2].iloc[0]
        pairs.append(dict(eid=eid, code=r1.code, first_date=r1.signal_date,
                          second_date=r2.signal_date,
                          first=r1.ret, second=r2.ret,
                          first_dd=r1.dd250, second_dd=r2.dd250,
                          y=r1.y))
P = pd.DataFrame(pairs)
P.to_csv(RES / '首笔vs第二笔_明细.csv', index=False, encoding='utf-8-sig')

pair_rows = []
for label, g in [('全部', P), ('剔除2026', P[P.y < 2026]), ('2026', P[P.y == 2026])]:
    if len(g) == 0:
        continue
    pair_rows.append(dict(
        样本=label, 配对数=int(len(g)),
        首笔均值=float(g['first'].mean()), 第二笔均值=float(g['second'].mean()),
        差值均值=float((g['second'] - g['first']).mean()),
        差值中位=float((g['second'] - g['first']).median()),
        第二笔更好占比=float((g['second'] > g['first']).mean()),
        相关性=float(np.corrcoef(g['first'], g['second'])[0, 1]),
        首笔胜率=float((g['first'] > 0).mean()),
        第二笔胜率=float((g['second'] > 0).mean()),
    ))
pair = pd.DataFrame(pair_rows)
pair.to_csv(RES / '首笔vs第二笔.csv', index=False, encoding='utf-8-sig')

# 最佳位置（事后，仅用于说明“第一次通常不差”）
best_rows = []
for eid, g in T[T.eid.isin(P.eid)].sort_values('pos').groupby('eid'):
    best_rows.append(dict(eid=eid, code=g.iloc[0].code, ep_len=len(g),
                          best_pos=int(g.loc[g.ret.idxmax(), 'pos']),
                          best_ret=float(g.ret.max())))
B = pd.DataFrame(best_rows)
best_tab = B.groupby('best_pos').size().rename('次数').reset_index()
best_tab['占比'] = best_tab['次数'] / best_tab['次数'].sum()
best_tab.to_csv(RES / '最佳位置.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 入场延迟
n_rows = []
for eid, g in T.sort_values('pos').groupby('eid'):
    for _, r in g.iterrows():
        for n in [1, 2, 3, 4, 5]:
            st, status, complete = trade_status(D, si[r.code], dates.get_loc(r.signal_date), n)
            if complete:
                n_rows.append(dict(eid=eid, code=r.code, signal_date=r.signal_date,
                                   pos=r.pos, n=n, ret=st['ret']))
NT = pd.DataFrame(n_rows)
piv = NT.pivot_table(index=['eid', 'code', 'signal_date', 'pos'],
                     columns='n', values='ret')
piv = piv.dropna()
delay_rows = []
for label, z in [('前二全部', piv),
                 ('前二首笔', piv[piv.index.get_level_values('pos') == 1]),
                 ('前二第二笔', piv[piv.index.get_level_values('pos') == 2])]:
    for n in [1, 2, 3, 4, 5]:
        delay_rows.append(dict(样本=label, N=n, 笔数=int(len(z)),
                               平均=float(z[n].mean()),
                               中位=float(z[n].median()),
                               胜率=float((z[n] > 0).mean())))
delay = pd.DataFrame(delay_rows)
delay.to_csv(RES / '入场延迟.csv', index=False, encoding='utf-8-sig')

# 母策略组合口径 N=1..5（来自父目录主结果，仅作对照）
portfolio = pd.DataFrame([
    dict(N=1, 组合年化=0.236816, 最大回撤=-0.230501, 交易笔数=95, 单笔均值=0.044610),
    dict(N=2, 组合年化=0.302583, 最大回撤=-0.166284, 交易笔数=97, 单笔均值=0.054265),
    dict(N=3, 组合年化=0.400100, 最大回撤=-0.160003, 交易笔数=99, 单笔均值=0.066793),
    dict(N=4, 组合年化=0.377478, 最大回撤=-0.212828, 交易笔数=98, 单笔均值=0.064554),
    dict(N=5, 组合年化=0.279083, 最大回撤=-0.279366, 交易笔数=97, 单笔均值=0.051400),
])
portfolio.to_csv(RES / '组合入场延迟对照.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 特征分组
feat_rows = []


def add_group(z, sample, feature, group, g):
    rec = dict(样本=sample, 特征=feature, 分组=group)
    rec.update(stat(g))
    feat_rows.append(rec)


for sample, z in [('前二', T), ('全部信号', A)]:
    # 前一日涨跌
    for v, lab in [(True, '前一日上涨'), (False, '前一日下跌')]:
        add_group(z, sample, '买入日前一天涨跌', lab, z[z.prev_up == v])
    # 量能五分位
    zz = z.dropna(subset=['amount_ratio1_20_prev']).copy()
    zz['vq'] = pd.qcut(zz['amount_ratio1_20_prev'], 5, labels=False, duplicates='drop')
    for q in sorted(zz.vq.dropna().unique()):
        g = zz[zz.vq == q]
        add_group(z, sample, '买入日前一天量比(20日)',
                  f'Q{int(q)+1} {g.amount_ratio1_20_prev.min():.2f}-{g.amount_ratio1_20_prev.max():.2f}', g)
    # MACD
    for v, lab in [(True, 'MACD柱为正'), (False, 'MACD柱为负')]:
        add_group(z, sample, '买入日前一天MACD柱', lab, z[z.macd_hist_up == v])
    for v, lab in [(True, '前一日金叉'), (False, '前一日非金叉')]:
        add_group(z, sample, '买入日前一天MACD金叉', lab, z[z.golden == v])
    # 均线
    for v, lab in [(True, '站上MA5'), (False, 'MA5下方')]:
        add_group(z, sample, '前一日收盘 vs MA5', lab, z[z.ma5_up == v])
    for v, lab in [(True, '站上MA10'), (False, 'MA10下方')]:
        add_group(z, sample, '前一日收盘 vs MA10', lab, z[z.ma10_up == v])
    for v, lab in [(True, '站上MA20'), (False, 'MA20下方')]:
        add_group(z, sample, '前一日收盘 vs MA20', lab, z[z.ma20_up == v])
    # 市场
    for v, lab in [(True, '市场前一日上涨'), (False, '市场前一日下跌')]:
        add_group(z, sample, '市场前一日涨跌', lab, z[z.market_prev_up == v])
    for ph, g in z.groupby('阶段'):
        add_group(z, sample, '市场阶段', ph, g)
    # 2x2
    for a, b_, lab in [(True, True, '个股涨+量高'), (True, False, '个股涨+量低'),
                       (False, True, '个股跌+量高'), (False, False, '个股跌+量低')]:
        add_group(z, sample, '个股涨跌 × 量能高低', lab,
                  z[(z.prev_up == a) & (z.vol_hi == b_)])
    for a, b_, lab in [(True, True, '个股涨+MACD正'), (True, False, '个股涨+MACD负'),
                       (False, True, '个股跌+MACD正'), (False, False, '个股跌+MACD负')]:
        add_group(z, sample, '个股涨跌 × MACD柱', lab,
                  z[(z.prev_up == a) & (z.macd_hist_up == b_)])
    for a, b_, lab in [(True, True, '个股涨+市场涨'), (True, False, '个股涨+市场跌'),
                       (False, True, '个股跌+市场涨'), (False, False, '个股跌+市场跌')]:
        add_group(z, sample, '个股涨跌 × 市场涨跌', lab,
                  z[(z.prev_up == a) & (z.market_prev_up == b_)])

feat = pd.DataFrame(feat_rows)
feat.to_csv(RES / '特征分组.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 当前信号 + 250日回撤
cur = b[(b.date.isin([pd.Timestamp('2026-10-22'), pd.Timestamp('2026-10-23')])) & b.ev_custom].copy()
cur['rk'] = cur.groupby('date')['drawdown250'].rank(ascending=True, method='first')
cur = cur[cur.rk <= 5].sort_values(['date', 'rk'])
cur_rows = []
for r in cur.itertuples():
    z = D['df'][D['df'].thscode == r.thscode].sort_values('date').tail(250)
    hh = z.loc[z.hfq_high.idxmax()]
    ah = z.loc[z.high.idxmax()]
    cur_rows.append(dict(
        信号日=r.date, 排名=int(r.rk), 代码=r.thscode,
        当日收盘_hfq=float(r.hfq_close), 最高250_hfq=float(hh.hfq_high),
        最高日_hfq=hh.date, 回撤_hfq=float(r.hfq_close / hh.hfq_high - 1),
        跌价_hfq=float(hh.hfq_high - r.hfq_close),
        当日收盘_实际=float(r.close), 最高250_实际=float(ah.high),
        最高日_实际=ah.date, 回撤_实际=float(r.close / ah.high - 1),
        跌价_实际=float(ah.high - r.close),
        深于45阈值_pp=float((-.45 - (r.hfq_close / hh.hfq_high - 1)) * 100),
        当日RSI=float(r.rsi14), 收盘位置=float(r.close_location),
        量比=float(r.amount_ratio1_20), 涨幅250=float(r.return_250d),
        当日信号数=int((b[(b.date == r.date) & b.ev_custom]).shape[0]),
    ))
curtab = pd.DataFrame(cur_rows)
curtab.to_csv(RES / '当前信号_250日回撤.csv', index=False, encoding='utf-8-sig')

# 当前持仓
mt = pd.read_csv(MOTHER / '03_结果' / '最终_逐笔.csv',
                 parse_dates=['signal_date', 'entry_date', 'exit_date'])
mt = mt[mt.exit_reason == 'eod'][['thscode', 'signal_date', 'entry_date', 'entry_px',
                                 'last_px', 'ret', 'exit_reason']]
mt.to_csv(RES / '当前持仓.csv', index=False, encoding='utf-8-sig')

plan = pd.DataFrame([
    ('已持有 600783 + 603778', '不加。两仓占满，10/22/23 的新信号等到现有仓位到期后再看。'),
    ('空仓，按顺序都做', '10/22 的 605117 在 10/27 开盘买半仓；10/23 的 603778 在 10/28 开盘买半仓；605376/002969 因名额不足跳过。'),
    ('空仓，放弃 10/22 的 605117', '10/23 买 603778 + 605376，各半仓；002969 跳过。'),
    ('已持有 600783，另有一个空仓', '10/22 的 605117 在 10/27 先占这个空位；10/23 的 603778 因无名额跳过。若放弃 605117，则 10/23 买 603778。'),
    ('已持有 603778，另有一个空仓', '10/23 买 605376 半仓；不要重复买 603778。'),
    ('已持有 603778，有两个空仓', '10/23 买 605376 + 002969，各半仓（跳过已持有的 603778）。'),
    ('只有 605117 成交、之后没有名额', '10/23 不再买；等 605117 到期。'),
], columns=['场景', '执行'])
plan.to_csv(RES / '当前执行方案.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 603778 案例
case = ev[ev.thscode == '603778.SH'].copy()
case_rows = []
for r in case.itertuples():
    st, status, complete = trade_status(D, si[r.thscode], r.di, NTH)
    mrow = E_all[E_all.code == r.thscode]
    pos = int(mrow[mrow.signal_date == r.date]['pos'].iloc[0]) if len(mrow[mrow.signal_date == r.date]) else np.nan
    case_rows.append(dict(信号日=r.date, 排名=int(r.rk), 是否前二=bool(r.rk <= 2),
                          段内位置=pos, 回撤250=float(r.drawdown250), RSI=float(r.rsi14),
                          收盘位置=float(r.close_location),
                          入场日=st.get('entry_date', pd.NaT),
                          退出日=st.get('exit_date', pd.NaT),
                          收益=st.get('ret', np.nan), 状态=status))
case_tab = pd.DataFrame(case_rows)
case_tab.to_csv(RES / '603778案例.csv', index=False, encoding='utf-8-sig')

# 多发信号案例（满足“看一些股票”的需求）
CASE_CODES = ['603778.SH', '002097.SZ', '002104.SZ', '002759.SZ', '000603.SZ']
multi_rows = []
for code in CASE_CODES:
    g = ev[ev.thscode == code].sort_values('di')
    for r in g.itertuples():
        st, status, complete = trade_status(D, si[r.thscode], r.di, NTH)
        mrow = E_all[E_all.code == r.thscode]
        pos = int(mrow[mrow.signal_date == r.date]['pos'].iloc[0]) if len(mrow[mrow.signal_date == r.date]) else np.nan
        multi_rows.append(dict(代码=code, 信号日=r.date, 排名=int(r.rk),
                               是否前二=bool(r.rk <= 2), 段内位置=pos,
                               回撤250=float(r.drawdown250),
                               入场日=st.get('entry_date', pd.NaT),
                               退出日=st.get('exit_date', pd.NaT),
                               收益=st.get('ret', np.nan), 状态=status))
multi = pd.DataFrame(multi_rows)
multi.to_csv(RES / '多发信号案例.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 指标
metrics = dict(
    样本=dict(信号池=int(len(ev)), 信号日=int(ev.date.nunique()), 股票=int(ev.thscode.nunique()),
             全部信号=int(len(A_raw)), 全部序列=int(len(A)), 全部段=int(n_ep_all),
             全部完整段=int(n_ep_all_complete), 前二完整=int(len(T)), 前二段=int(T.eid.nunique()),
             配对数=int(len(P)), 最佳位置样本=int(len(B)),
             严格连续3天前二=int(len(strict_top)), 严格连续3天全部=int(len(strict_all))),
    位置={r['样本'] + '-' + r['位置']: dict(笔数=int(r['笔数']), 平均=float(r['平均']),
                                              胜率=float(r['胜率']), 回撤中位=float(r['回撤中位']))
          for _, r in pos_both.iterrows()},
    配对={r['样本']: dict(配对数=int(r['配对数']), 首笔均值=float(r['首笔均值']),
                          第二笔均值=float(r['第二笔均值']), 差值均值=float(r['差值均值']),
                          第二笔更好占比=float(r['第二笔更好占比']), 相关性=float(r['相关性']))
          for _, r in pair.iterrows()},
    延迟={r['样本'] + f"-N{int(r['N'])}": dict(笔数=int(r['笔数']), 平均=float(r['平均']),
                                               胜率=float(r['胜率'])) for _, r in delay.iterrows()},
    特征=feat.to_dict(orient='records'),
    当前=curtab.to_dict(orient='records'),
    案例=case_tab.to_dict(orient='records'),
)
(RES / '指标.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

# 保存底表
A_raw.to_csv(RES / '全部信号池_原始.csv', index=False, encoding='utf-8-sig')
A.to_csv(RES / '全部信号重复序列_完整.csv', index=False, encoding='utf-8-sig')
T.to_csv(RES / '前二重复序列.csv', index=False, encoding='utf-8-sig')
P.to_csv(RES / '首笔vs第二笔_明细.csv', index=False, encoding='utf-8-sig')
feat.to_csv(RES / '特征分组.csv', index=False, encoding='utf-8-sig')
curtab.to_csv(RES / '当前信号_250日回撤.csv', index=False, encoding='utf-8-sig')
case_tab.to_csv(RES / '603778案例.csv', index=False, encoding='utf-8-sig')

# 多发信号案例（满足“看一些股票”的需求）
CASE_CODES = ['603778.SH', '002097.SZ', '002104.SZ', '002759.SZ', '000603.SZ']
multi_rows = []
for code in CASE_CODES:
    g = ev[ev.thscode == code].sort_values('di')
    for r in g.itertuples():
        st, status, complete = trade_status(D, si[r.thscode], r.di, NTH)
        mrow = E_all[E_all.code == r.thscode]
        pos = int(mrow[mrow.signal_date == r.date]['pos'].iloc[0]) if len(mrow[mrow.signal_date == r.date]) else np.nan
        multi_rows.append(dict(代码=code, 信号日=r.date, 排名=int(r.rk),
                               是否前二=bool(r.rk <= 2), 段内位置=pos,
                               回撤250=float(r.drawdown250),
                               入场日=st.get('entry_date', pd.NaT),
                               退出日=st.get('exit_date', pd.NaT),
                               收益=st.get('ret', np.nan), 状态=status))
multi = pd.DataFrame(multi_rows)
multi.to_csv(RES / '多发信号案例.csv', index=False, encoding='utf-8-sig')

print('位置统计'); print(pos_both.round(4).to_string(index=False))
print('配对'); print(pair.round(4).to_string(index=False))
print('最佳位置'); print(best_tab.round(4).to_string(index=False))
print('入场延迟'); print(delay.round(4).to_string(index=False))
print('当前信号'); print(curtab.round(4).to_string(index=False))
print('603778'); print(case_tab.round(4).to_string(index=False))
print('done')
