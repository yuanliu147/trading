# -*- coding: utf-8 -*-
"""信号覆盖、漏买原因与涨停/龙虎榜关系。

- 复刻 45%+波动20 系统：2 个仓位、按20日成交额升序、t-1信号、t开盘买。
- 记录每个信号候选为什么没买：仓位已满、已持有、高开>5%、一字涨停、资金/整手不足、风险预算不足等。
- 把连续信号合并成“独立信号段”，比较买到的信号段与漏掉的信号段之后20日收益。
- 涨停：统计“其他条件都满足、仅因过去20日有涨停被排除”的股票，及其之后20日收益。
- 龙虎榜：当前面板没有龙虎榜数据，不能直接检验；用换手异常、成交额突增做价量代理，明确标注不是龙虎榜。
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
DIR = '/Users/admin/WorkSpace/trading/2026-10-21_45仓位_波动20_交易系统'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 面板 ----------------
P = pickle.load(open(os.path.join(BASE, '01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
codes = np.array(P['meta']['codes'])
C = P['close'].values
O = P['open'].values
H = P['high'].values
L = P['low'].values
U = P['up_struct'].values
AGE = P['trend_age'].values
ATR = P['atr_pct'].values
AMT = P['amt20'].values
LU = P['lu_count20'].values
SBR = P['sector_breadth20'].values
SR20 = P['sector_ret20'].values
DD60 = P['dd60'].values
MB = P['market_breadth20'].values
NG = P['next_gap'].values
OWU = P['one_word_up'].values
OWD = P['one_word_down'].values
F20 = P['fwd20'].values
TURNZ = P['turn_z250'].values
AMTR = P['amt_ratio20'].values
GAP = P['gap'].values
MKT = P['market_index'].values
del P

date_ok = (dates >= pd.Timestamp('2017-10-13'))[:, None]
valid = U & (AMT >= 3e7) & (C >= 2) & (AGE >= 5) & date_ok
base = valid & (AGE <= 30) & (ATR <= 0.044) & (SBR >= 0.6) & (SR20 > 0.05) & (DD60 > -0.08)
gate = MB[:, None] > 0.5
pre_lu = base & gate                 # 其他条件都满足，尚未做涨停过滤
sig = pre_lu & (LU == 0)             # 系统最终信号
mkt_vol = pd.Series(MKT).pct_change().rolling(20, min_periods=15).std().values * np.sqrt(252)

# ---------------- 精确复刻 ----------------
slots = 2
maxpos = 0.45
heat_cap = 0.10
vol_target = 0.20
vol_floor = 0.30
cost_rate = 0.0015
min_cost = 5.0
hold = 20
stop_pct = 0.20
si = dates.searchsorted(pd.Timestamp('2018-01-01'))
ei = dates.searchsorted(pd.Timestamp('2026-10-13'), side='right') - 1

cash = 20000.0
pos = []
trades = []
logs = []
eq_hist = [cash]

for t in range(si, ei + 1):
    # 1) 盘中止损
    for p in list(pos):
        j = p['j']
        if t < p['entry_i']:
            continue
        if np.isfinite(L[t, j]) and L[t, j] <= p['stop']:
            if OWD[t, j] or not np.isfinite(O[t, j]):
                continue
            fill = min(O[t, j], p['stop'])
            gross = p['sh'] * fill
            fee = max(min_cost, gross * cost_rate)
            cash += gross - fee
            trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                               entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason='止损',
                               hold=t - p['entry_i'] + 1, shares=p['sh'],
                               pnl=(fill - p['entry']) * p['sh'] - fee - p['entry_fee']))
            pos.remove(p)
    # 2) 到期
    for p in list(pos):
        j = p['j']
        if t < p['exit_i']:
            continue
        if OWD[t, j]:
            continue
        if not np.isfinite(C[t, j]):
            future = np.where(np.isfinite(C[t:, j]))[0]
            if len(future) > 0:
                continue
            fill = p['last'] * 0.5
            reason = '数据终止'
        else:
            fill = C[t, j]
            reason = '到期'
        gross = p['sh'] * fill
        fee = max(min_cost, gross * cost_rate)
        cash += gross - fee
        trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                           entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason=reason,
                           hold=t - p['entry_i'] + 1, shares=p['sh'],
                           pnl=(fill - p['entry']) * p['sh'] - fee - p['entry_fee']))
        pos.remove(p)
    # 3) 开盘买入（信号 t-1）
    if t > si:
        cand = np.where(sig[t - 1])[0]
        if len(cand):
            cand = sorted(cand, key=lambda j: AMT[t - 1, j])
            held = {p['j'] for p in pos}
            for rank, j in enumerate(cand, 1):
                reason = ''
                selected = False
                if j in held:
                    reason = '已持有'
                elif len(pos) >= slots:
                    reason = '仓位已满'
                elif not np.isfinite(O[t, j]) or O[t, j] <= 0:
                    reason = '停牌/无开盘'
                elif np.isfinite(NG[t - 1, j]) and NG[t - 1, j] > 0.05:
                    reason = '次日高开>5%'
                elif OWU[t, j] or (LU[t, j] and np.isfinite(H[t, j]) and O[t, j] >= H[t, j] - 1e-9):
                    reason = '一字涨停'
                else:
                    entry = O[t, j]
                    eq = cash + sum(p['sh'] * (C[t - 1, p['j']] if np.isfinite(C[t - 1, p['j']]) else p['last'])
                                    for p in pos)
                    mult = 1.0
                    if np.isfinite(mkt_vol[t - 1]):
                        mult = float(np.clip(vol_target / mkt_vol[t - 1], vol_floor, 1.0))
                    maxpos_eff = maxpos * mult
                    if entry * 100 > eq * maxpos_eff or entry * 100 > cash:
                        reason = '资金/整手不足'
                    else:
                        target = int((eq * maxpos_eff / entry) // 100) * 100
                        if target < 100:
                            target = 100 if entry * 100 <= eq * maxpos_eff and entry * 100 <= cash else 0
                        if target <= 0:
                            reason = '资金/整手不足'
                        else:
                            sd = float(np.clip(2 * ATR[t - 1, j], 0.04, 0.12))
                            heat = sum(p['sh'] * p['entry'] * p['sd'] for p in pos) / max(eq, 1e-9)
                            sh = target
                            while sh >= 100 and heat + sh * entry * sd / max(eq, 1e-9) > heat_cap:
                                sh -= 100
                            if sh < 100:
                                reason = '风险预算不足'
                            else:
                                gross = sh * entry
                                fee = max(min_cost, gross * cost_rate)
                                if gross + fee > cash:
                                    sh = int((cash / (entry * (1 + cost_rate))) // 100) * 100
                                    if sh < 100:
                                        reason = '现金不足'
                                    else:
                                        gross = sh * entry
                                        fee = max(min_cost, gross * cost_rate)
                                        if gross + fee > cash:
                                            reason = '现金不足'
                                if not reason:
                                    cash -= gross + fee
                                    pos.append(dict(j=j, entry_i=t, entry=entry, sh=sh, init_sh=sh, sd=sd,
                                                    stop=entry * (1 - stop_pct), last=entry, exit_i=t + hold - 1,
                                                    entry_fee=fee))
                                    held.add(j)
                                    selected = True
                                    reason = '买入'
                logs.append(dict(
                    signal_date=dates[t - 1], entry_date=dates[t], code=codes[j], rank=rank,
                    reason=reason, selected=selected,
                    amt20=AMT[t - 1, j], atr=ATR[t - 1, j], lu_count20=LU[t - 1, j],
                    breadth20=MB[t - 1], sector_breadth=SBR[t - 1, j], sector_ret20=SR20[t - 1, j],
                    next_gap=NG[t - 1, j], fwd20=F20[t - 1, j],
                    turn_z=TURNZ[t - 1, j], amt_ratio=AMTR[t - 1, j], gap=GAP[t - 1, j],
                ))
    # 4) 收盘估值
    val = 0.0
    for p in pos:
        j = p['j']
        px = C[t, j] if np.isfinite(C[t, j]) else p['last']
        p['last'] = px
        val += p['sh'] * px
    eq_hist.append(cash + val)

T = pd.DataFrame(logs)
TR = pd.DataFrame(trades)
print('trades', len(TR), 'final eq %.2f' % (cash + sum(p['sh'] * (C[ei, p['j']] if np.isfinite(C[ei, p['j']]) else p['last']) for p in pos)))
print('candidate-days', len(T), 'selected logs', T['selected'].sum())
print('reasons')
print(T['reason'].value_counts().to_string())
print('selected trades sum pnl', TR['pnl'].sum())

# 与既有逐笔核对
old = pd.read_csv(os.path.join(DIR, '逐笔交易记录.csv'), encoding='utf-8-sig', parse_dates=['买入日', '卖出日'])
print('old trades', len(old), 'old net', old['净盈亏'].sum())
old_key = set(zip(old['代码'], old['买入日']))
new_key = set(zip(TR['code'], TR['entry_date']))
print('entry keys match', len(old_key & new_key), 'old-only', len(old_key - new_key), 'new-only', len(new_key - old_key))

# ---------------- 独立信号段 ----------------
# 每个股票的连续信号日合并为一段（按交易日索引连续）
sig_days = T[['signal_date', 'code', 'selected', 'reason', 'fwd20']].copy()
sig_days['t'] = dates.searchsorted(sig_days['signal_date'])
sig_days = sig_days.sort_values(['code', 't']).reset_index(drop=True)
sig_days['new_ep'] = (sig_days['code'] != sig_days['code'].shift(1)) | (sig_days['t'] - sig_days['t'].shift(1) != 1)
sig_days['ep_id'] = sig_days['new_ep'].cumsum()

ep_rows = []
for ep, d in sig_days.groupby('ep_id'):
    first = d.iloc[0]
    ep_rows.append(dict(
        ep_id=ep, code=first['code'], first_signal=first['signal_date'],
        last_signal=d['signal_date'].iloc[-1], days=len(d),
        bought=d['selected'].any(),
        first_reason=first['reason'],
        reasons=' / '.join(d['reason'].drop_duplicates().tolist()),
        fwd20_first=first['fwd20'], fwd20_mean=d['fwd20'].mean(),
        fwd20_max=d['fwd20'].max(),
    ))
EP = pd.DataFrame(ep_rows)
print('\nepisodes', len(EP), 'bought', EP['bought'].sum(), 'missed', (~EP['bought']).sum())
print('episode first reasons')
print(EP.loc[~EP['bought'], 'first_reason'].value_counts().to_string())
print('\nepisode fwd20 bought/missed')
print(EP.groupby('bought')['fwd20_first'].agg(['count', 'mean', 'median', lambda s: (s > 0).mean()]).to_string())

# ---------------- 涨停分组 ----------------
lu_rows = []
for lo, hi, name in [(0, 0, '0 次'), (1, 1, '1 次'), (2, 2, '2 次'), (3, 999, '3 次及以上')]:
    m = pre_lu & (LU >= lo) & (LU <= hi) & np.isfinite(F20)
    x = F20[m]
    lu_rows.append(dict(涨停次数=name, 股票日数=len(x), 平均20日收益=x.mean(), 中位20日收益=np.median(x),
                        胜率=(x > 0).mean(), 盈亏比=x[x > 0].sum() / max(1e-9, -x[x < 0].sum())))
LU_TAB = pd.DataFrame(lu_rows)
print('\nlimit-up buckets')
print(LU_TAB.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# 仅因涨停被排除的股票-日/独立段
excl = pre_lu & (LU > 0) & np.isfinite(F20)
incl = pre_lu & (LU == 0) & np.isfinite(F20)
print('excluded lu>0 stock-days', excl.sum(), 'mean', F20[excl].mean(), 'win', (F20[excl] > 0).mean())
print('included lu0 stock-days', incl.sum(), 'mean', F20[incl].mean(), 'win', (F20[incl] > 0).mean())

# ---------------- 龙虎榜代理 ----------------
proxy_rows = []
for name, arr, thrs in [('换手异常值', TURNZ, [2.0]), ('成交额/20日均额', AMTR, [2.0]), ('信号日跳空', GAP, [0.02])]:
    for thr in thrs:
        for label, mask in [('高于阈值', arr >= thr), ('低于阈值', arr < thr)]:
            m = mask & sig & np.isfinite(F20)
            x = F20[m]
            proxy_rows.append(dict(代理=name, 分组=f'{label}{thr:g}', 股票日数=len(x),
                                   平均20日收益=x.mean(), 中位20日收益=np.median(x), 胜率=(x > 0).mean()))
PROXY = pd.DataFrame(proxy_rows)
print('\nLHB proxies')
print(PROXY.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# ---------------- 补充统计 ----------------
print('\nunique signaled codes', T['code'].nunique(), 'bought codes', old['代码'].nunique())
print('selected rank counts')
print(T.loc[T['selected'], 'rank'].value_counts().sort_index().to_string())
T['rank_group'] = np.minimum(T['rank'], 4)
rank_tab = T.groupby('rank_group').agg(候选日=('code', 'size'), 平均20日收益=('fwd20', 'mean'),
                                       中位20日收益=('fwd20', 'median'),
                                       胜率=('fwd20', lambda s: (s > 0).mean())).reset_index()
rank_tab['排序'] = rank_tab['rank_group'].map({1: '第1名', 2: '第2名', 3: '第3名', 4: '第4名及以后'})
print('\nrank performance')
print(rank_tab[['排序', '候选日', '平均20日收益', '中位20日收益', '胜率']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# 次日开盘的硬过滤：在所有候选日上统计，不受“仓位已满”遮挡
def shift1(a, fill):
    out = np.full_like(a, fill)
    out[:-1] = a[1:]
    return out
O_next = shift1(O, np.nan); H_next = shift1(H, np.nan)
OWU_next = shift1(OWU, False); LU_next = shift1(LU, False)
oneword_next = OWU_next | ((LU_next > 0) & np.isfinite(O_next) & np.isfinite(H_next) & (O_next >= H_next - 1e-9))
bad_open = ~np.isfinite(O_next) | (O_next <= 0)
gap_high = np.isfinite(NG) & (NG > 0.05)
sig_eval = sig.copy()
sig_eval[:si] = False   # 回测从2018-01-01开始，之前的信号不进入组合
sig_eval[ei:] = False   # 最后一天没有次日，不能算入场候选
entry_filter = pd.DataFrame([
    ('次日高开>5%', int((sig_eval & gap_high).sum())),
    ('次日一字涨停', int((sig_eval & oneword_next).sum())),
    ('次日停牌/无开盘', int((sig_eval & bad_open).sum())),
    ('全部候选日', int(sig_eval.sum())),
], columns=['项目', '候选日数'])
print('\nentry filters among all candidates')
print(entry_filter.to_string(index=False))

# 涨停组尾部：均值更差，但有没有漏掉大牛？
tail_rows = []
for name, m in [('涨停0次', pre_lu & (LU == 0)), ('涨停1次及以上', pre_lu & (LU > 0))]:
    x = F20[m & np.isfinite(F20)]
    tail_rows.append(dict(分组=name, 样本=len(x), 均值=x.mean(), 中位=np.median(x), 胜率=(x > 0).mean(),
                          九十分位=np.quantile(x, .90), 九十五分位=np.quantile(x, .95),
                          九十九分位=np.quantile(x, .99), 最大值=x.max()))
TAIL = pd.DataFrame(tail_rows)
print('\nlimit-up tails')
print(TAIL.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# 漏买原因：独立信号段
miss_reason = EP[~EP['bought']].groupby('first_reason').agg(
    独立信号段=('ep_id', 'size'), 平均首日20日收益=('fwd20_first', 'mean'),
    中位首日20日收益=('fwd20_first', 'median'),
    胜率=('fwd20_first', lambda s: (s > 0).mean())).reset_index().rename(columns={'first_reason': '首个原因'})
miss_reason['占漏买段'] = miss_reason['独立信号段'] / (~EP['bought']).sum()
print('\nmiss reasons (episodes)')
print(miss_reason.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# 买入 vs 漏买（剔除已持有）
comp_rows = []
for name, d in [('买入的信号段', EP[EP['bought']]),
                ('漏买：仓位已满', EP[(~EP['bought']) & (EP['first_reason'] == '仓位已满')]),
                ('漏买：其他原因', EP[(~EP['bought']) & (EP['first_reason'] != '仓位已满')])]:
    x = d['fwd20_first'].dropna()
    comp_rows.append(dict(分组=name, 信号段=len(d), 平均20日收益=x.mean(), 中位20日收益=np.median(x),
                          胜率=(x > 0).mean(), 盈亏比=x[x > 0].sum() / max(1e-9, -x[x < 0].sum())))
COMP = pd.DataFrame(comp_rows)
print('\nbought vs missed')
print(COMP.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# ---------------- 保存 ----------------
out = T.copy()
out['结果'] = out['reason']
out = out.rename(columns={
    'signal_date': '信号日', 'entry_date': '买入日', 'code': '代码', 'rank': '排序',
    'selected': '是否买入', 'amt20': '20日成交额', 'atr': '真实波幅', 'lu_count20': '20日涨停次数',
    'breadth20': '市场宽度', 'sector_breadth': '行业宽度', 'sector_ret20': '行业20日收益',
    'next_gap': '次日跳空', 'fwd20': '未来20日收益', 'turn_z': '换手异常值',
    'amt_ratio': '成交额比', 'gap': '信号日跳空'})
out = out[['信号日', '买入日', '代码', '排序', '结果', '是否买入', '20日成交额', '真实波幅',
           '20日涨停次数', '市场宽度', '行业宽度', '行业20日收益', '次日跳空', '未来20日收益',
           '换手异常值', '成交额比', '信号日跳空']]
out.to_csv(os.path.join(DIR, '信号候选明细.csv'), index=False, encoding='utf-8-sig',
           float_format='%.6f', date_format='%Y-%m-%d')

ep_out = EP.rename(columns={
    'ep_id': '段编号', 'code': '代码', 'first_signal': '首个信号日', 'last_signal': '最后信号日',
    'days': '信号天数', 'bought': '是否买入', 'first_reason': '首个原因', 'reasons': '全部原因',
    'fwd20_first': '首日未来20日收益', 'fwd20_mean': '段内平均20日收益', 'fwd20_max': '段内最高20日收益'})
ep_out.to_csv(os.path.join(DIR, '独立信号段.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f', date_format='%Y-%m-%d')

reason_out = T.groupby('reason').agg(候选日数=('code', 'size')).reset_index().rename(columns={'reason': '原因'})
reason_out['候选日占比'] = reason_out['候选日数'] / len(T)
reason_out = reason_out.sort_values('候选日数', ascending=False)
reason_out.to_csv(os.path.join(DIR, '信号覆盖_原因汇总.csv'), index=False, encoding='utf-8-sig',
                  float_format='%.6f')
LU_TAB.to_csv(os.path.join(DIR, '涨停过滤_分组表现.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f')
TAIL.to_csv(os.path.join(DIR, '涨停过滤_尾部.csv'), index=False, encoding='utf-8-sig',
            float_format='%.6f')
PROXY.to_csv(os.path.join(DIR, '龙虎榜代理_分组表现.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f')
COMP.to_csv(os.path.join(DIR, '信号覆盖_买入与漏买.csv'), index=False, encoding='utf-8-sig',
            float_format='%.6f')
rank_tab[['排序', '候选日', '平均20日收益', '中位20日收益', '胜率']].to_csv(
    os.path.join(DIR, '信号覆盖_排序表现.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
entry_filter.to_csv(os.path.join(DIR, '信号覆盖_入场过滤.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame([
    ('候选日数', f'{len(T)}'),
    ('独立信号段', f'{len(EP)}'),
    ('买入交易', f'{len(TR)}'),
    ('买入信号段', f"{int(EP['bought'].sum())}"),
    ('信号股票数', f"{T['code'].nunique()}"),
    ('买入股票数', f"{old['代码'].nunique()}"),
    ('候选日/买入', f"{len(T)/max(1,len(TR)):.0f} : 1"),
    ('买入段平均20日收益', f"{EP.loc[EP['bought'],'fwd20_first'].mean():.2%}"),
    ('漏买段平均20日收益', f"{EP.loc[~EP['bought'],'fwd20_first'].mean():.2%}"),
    ('仓位已满漏买段', f"{int(((~EP['bought']) & (EP['first_reason']=='仓位已满')).sum())}"),
    ('涨停>0被排除股票日', f'{int((pre_lu & (LU>0)).sum())}'),
    ('涨停>0未来20日均值', f'{F20[pre_lu & (LU>0) & np.isfinite(F20)].mean():.2%}'),
    ('涨停0未来20日均值', f'{F20[pre_lu & (LU==0) & np.isfinite(F20)].mean():.2%}'),
], columns=['指标', '数值']).to_csv(os.path.join(DIR, '信号覆盖_统计摘要.csv'), index=False, encoding='utf-8-sig')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
fig.subplots_adjust(top=0.86, bottom=0.20, left=0.16, right=0.97, wspace=0.32)

ax = axes[0]
miss = miss_reason.sort_values('独立信号段').copy()
miss['首个原因'] = miss['首个原因'].replace({'已持有': '已持有（非新机会）'})
ax.barh(miss['首个原因'], miss['独立信号段'], color='#F58518')
ax.set_xscale('log')
ax.set_xlabel('独立信号段数（对数轴）')
ax.set_title('漏买的信号段：几乎全是仓位已满', loc='left', fontsize=12, weight='bold')
for i, v in enumerate(miss['独立信号段']):
    ax.text(v * 1.12, i, f'{int(v):,}', va='center', fontsize=9)
ax.text(0.02, 0.04, f"买入 {int(EP['bought'].sum()):,} 段 / 漏买 {int((~EP['bought']).sum()):,} 段",
        transform=ax.transAxes, fontsize=9, color='#555555')
ax.grid(axis='x', alpha=.22)

ax = axes[1]
colors = ['#4C78A8', '#72B7B2', '#F58518', '#E45756']
ax.bar(LU_TAB['涨停次数'], LU_TAB['平均20日收益'] * 100, color=colors)
ax.axhline(0, color='black', lw=.6, alpha=.4)
ax.set_ylabel('未来20日平均收益（%）')
ax.set_title('过去20日涨停次数越多，后续越差', loc='left', fontsize=12, weight='bold')
ax.set_ylim(-4.2, 3.0)
for i, r in LU_TAB.iterrows():
    ax.text(i, r['平均20日收益'] * 100 + (0.12 if r['平均20日收益'] >= 0 else -0.55),
            f"{r['平均20日收益']*100:.1f}% / 胜率{r['胜率']*100:.0f}%", ha='center', fontsize=8.5,
            va='bottom' if r['平均20日收益'] >= 0 else 'top')
ax.grid(axis='y', alpha=.22)

fig.suptitle('信号覆盖与涨停过滤：2 个仓位装不下所有信号', fontsize=16, y=0.955)
fig.text(0.5, 0.045,
         '口径：候选日=每天通过全部信号条件的股票；独立信号段=同一股票连续信号日合并。'
         '漏买统计为首个信号日的原因；未来20日收益=次日开盘买、20个交易日后收盘卖，不含-20%止损。'
         '龙虎榜数据不在当前面板，图中只涉及涨停。',
         ha='center', fontsize=8, color='#555555')
plt.savefig(os.path.join(DIR, '信号覆盖与涨停.png'), dpi=200)
plt.close()
print('\nsaved outputs')
