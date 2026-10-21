# -*- coding: utf-8 -*-
"""年度收益差异与建仓时大盘状态。

口径：
- 系统年度收益 = 自然年最后一日权益 / 上年最后一日权益 - 1；2026 截至 10-13。
- 大盘 = 全部A股等权指数（与系统市场宽度同一套数据），不是沪深300。
- 建仓时状态 = 买入日前一个交易日收盘的市场状态（信号日），不使用未来数据。
- 逐笔净盈亏按卖出日归属；跨年持仓会造成“已实现净盈亏”和“权益变动”不一致。
"""
import os
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

# ---------------- 数据 ----------------
E = pd.read_csv(os.path.join(DIR, '权益曲线.csv'), parse_dates=['日期']).set_index('日期')
T = pd.read_csv(os.path.join(DIR, '逐笔交易记录.csv'), encoding='utf-8-sig',
                parse_dates=['买入日', '卖出日'])
M = pd.read_csv(os.path.join(BASE, '01_数据/市场状态日表.csv'), parse_dates=['date']).set_index('date')

M['market_ret20'] = M['market_index'] / M['market_index'].shift(20) - 1
M['market_ma60'] = M['market_index'].rolling(60, min_periods=40).mean()
M['market_ma120'] = M['market_index'].rolling(120, min_periods=80).mean()
M['market_above60'] = M['market_index'] > M['market_ma60']
M['market_slope60'] = M['market_ma60'] / M['market_ma60'].shift(20) - 1
M['market_vol20'] = M['market_index'].pct_change().rolling(20, min_periods=15).std() * np.sqrt(252)
M['market_dd'] = M['market_index'] / M['market_index'].cummax() - 1

# 建仓时：买入日前一个交易日（信号日）的状态
sig_i = M.index.searchsorted(T['买入日'], side='left') - 1
S = M.iloc[sig_i][['breadth20', 'breadth60', 'market_above60', 'market_ret20',
                   'market_vol20', 'market_slope60', 'regime']].copy()
S.index = T.index
S.columns = ['建仓时市场宽度', '建仓时市场宽度60', '建仓时指数在60日线上',
             '建仓时大盘20日收益', '建仓时大盘年化波动', '建仓时大盘60日斜率',
             '建仓时市场状态']
T = pd.concat([T, S], axis=1, sort=False)

# 逐笔表附上建仓时的大盘状态，便于单笔核查
out_cols = ['序号', '代码', '买入日', '卖出日', '持有天数', '买入价', '卖出价', '股数',
            '买入金额', '卖出金额', '毛盈亏', '费用', '净盈亏', '收益率', '买入仓位',
            '账户贡献', '退出原因', '盈亏', '年份',
            '建仓时市场宽度', '建仓时市场宽度60', '建仓时指数在60日线上',
            '建仓时大盘20日收益', '建仓时大盘年化波动', '建仓时大盘60日斜率', '建仓时市场状态']
out = T[out_cols].copy()
out['建仓时指数在60日线上'] = np.where(out['建仓时指数在60日线上'], '是', '否')
out.to_csv(os.path.join(DIR, '逐笔交易记录_含建仓时大盘.csv'), index=False,
           encoding='utf-8-sig', float_format='%.6f', date_format='%Y-%m-%d')

# ---------------- 年度表 ----------------
rows_a, rows_b = [], []
for y in range(2018, 2027):
    de = E[E.index.year == y]
    if de.empty:
        continue
    prev_e = E[E.index < de.index[0]]
    base_e = prev_e['权益'].iloc[-1] if len(prev_e) else de['权益'].iloc[0]
    sys_ret = de['权益'].iloc[-1] / base_e - 1
    sys_dd_year = (de['权益'] / de['权益'].cummax() - 1).min()
    sys_dd_hist = (de['权益'] / E[E.index <= de.index[-1]]['权益'].cummax() - 1).min()

    dm = M[M.index.year == y]
    prev_m = M[M.index < dm.index[0]]
    base_m = prev_m['market_index'].iloc[-1] if len(prev_m) else dm['market_index'].iloc[0]
    mkt_ret = dm['market_index'].iloc[-1] / base_m - 1
    mkt_dd_year = (dm['market_index'] / dm['market_index'].cummax() - 1).min()
    mkt_dd_hist = (dm['market_index'] / M[M.index <= dm.index[-1]]['market_index'].cummax() - 1).min()

    rows_a.append(dict(
        年份=y,
        系统收益=sys_ret,
        等权全市场收益=mkt_ret,
        相对大盘=sys_ret - mkt_ret,
        系统年内回撤=sys_dd_year,
        系统对历史峰值回撤=sys_dd_hist,
        大盘年内回撤=mkt_dd_year,
        大盘对历史峰值回撤=mkt_dd_hist,
        平均市场宽度20日=dm['breadth20'].mean(),
        宽度大于50天数占比=(dm['breadth20'] > 0.5).mean(),
        平均年化波动=dm['market_vol20'].mean(),
        上行或强上行天数占比=dm['regime'].isin(['上行', '强上行']).mean(),
        下行或强下行天数占比=dm['regime'].isin(['下行', '强下行']).mean(),
        平均总仓位=de['总仓位'].mean(),
    ))

    dt = T[T['年份'] == y]
    pos_sum = dt.loc[dt['净盈亏'] > 0, '净盈亏'].sum()
    top1 = dt['净盈亏'].max() if len(dt) else np.nan
    top2 = dt.nlargest(2, '净盈亏')['净盈亏'].sum() if len(dt) else np.nan
    top1_row = dt.loc[dt['净盈亏'].idxmax()] if len(dt) else None
    rows_b.append(dict(
        年份=y,
        交易笔数=len(dt),
        胜率=(dt['净盈亏'] > 0).mean(),
        平均单笔收益率=dt['收益率'].mean(),
        已实现净盈亏=dt['净盈亏'].sum(),
        权益变动=de['权益'].iloc[-1] - base_e,
        最大单笔净盈亏=top1,
        最大一笔占正收益=top1 / pos_sum if pos_sum > 0 else np.nan,
        前两大占正收益=top2 / pos_sum if pos_sum > 0 else np.nan,
        去掉最大一笔净盈亏=dt['净盈亏'].sum() - top1,
        去掉前两大净盈亏=dt['净盈亏'].sum() - top2,
        最差单笔净盈亏=dt['净盈亏'].min() if len(dt) else np.nan,
        平均建仓仓位=dt['买入仓位'].mean(),
        建仓时平均市场宽度=dt['建仓时市场宽度'].mean(),
        建仓时平均市场宽度60=dt['建仓时市场宽度60'].mean(),
        建仓时指数在60日线上占比=dt['建仓时指数在60日线上'].mean(),
        建仓时大盘20日收益=dt['建仓时大盘20日收益'].mean(),
        建仓时大盘年化波动=dt['建仓时大盘年化波动'].mean(),
        建仓时上行或强上行占比=dt['建仓时市场状态'].isin(['上行', '强上行']).mean(),
        建仓时强上行占比=(dt['建仓时市场状态'] == '强上行').mean(),
        最大一笔代码=top1_row['代码'] if top1_row is not None else '',
    ))

A = pd.DataFrame(rows_a)
B = pd.DataFrame(rows_b)
A.to_csv(os.path.join(DIR, '年度_收益与大盘.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')
B.to_csv(os.path.join(DIR, '年度_交易与建仓状态.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

# ---------------- 建仓状态分组 ----------------
T['指数相对60日均线'] = np.where(T['建仓时指数在60日线上'], '60日均线上方', '60日均线下方')
T['市场宽度分组'] = pd.cut(T['建仓时市场宽度'], [0.5, 0.55, 0.60, 0.65, 0.70, 1.0],
                           labels=['50%–55%', '55%–60%', '60%–65%', '65%–70%', '70%以上'],
                           include_lowest=True)
T['市场宽度60分组'] = pd.cut(T['建仓时市场宽度60'], [0, 0.30, 0.40, 0.50, 0.60, 1.0],
                            labels=['低于30%', '30%–40%', '40%–50%', '50%–60%', '60%以上'],
                            include_lowest=True)
T['大盘20日收益分组'] = pd.cut(T['建仓时大盘20日收益'], [-1, -0.05, 0, 0.05, 0.10, 1],
                               labels=['低于-5%', '-5%–0%', '0%–5%', '5%–10%', '高于10%'])
T['大盘波动分组'] = pd.cut(T['建仓时大盘年化波动'], [0, 0.15, 0.20, 0.25, 2],
                           labels=['低于15%', '15%–20%', '20%–25%', '高于25%'])

groups = []
for dim, col in [('指数相对60日均线', '指数相对60日均线'),
                 ('建仓时市场宽度', '市场宽度分组'),
                 ('建仓时市场宽度60', '市场宽度60分组'),
                 ('建仓时大盘20日收益', '大盘20日收益分组'),
                 ('建仓时大盘年化波动', '大盘波动分组'),
                 ('建仓时市场状态', '建仓时市场状态')]:
    for name, d in T.groupby(col, observed=True):
        groups.append(dict(维度=dim, 分组=str(name), 笔数=len(d), 占比=len(d) / len(T),
                           胜率=(d['净盈亏'] > 0).mean(), 平均单笔收益率=d['收益率'].mean(),
                           平均账户贡献=d['账户贡献'].mean(), 净盈亏=d['净盈亏'].sum(),
                           平均持有天数=d['持有天数'].mean()))
C = pd.DataFrame(groups)
C.to_csv(os.path.join(DIR, '建仓状态_分组表现.csv'), index=False, encoding='utf-8-sig', float_format='%.6f')

# ---------------- 统计摘要 ----------------
months = pd.concat([E['权益'].resample('ME').last().pct_change(),
                    M['market_index'].resample('ME').last().pct_change()],
                   axis=1, keys=['系统', '大盘'], sort=True).dropna()
X = np.column_stack([np.ones(len(months)), months['大盘']])
coef = np.linalg.lstsq(X, months['系统'], rcond=None)[0]
resid = months['系统'] - X @ coef
inv = np.linalg.inv(X.T @ X)
se_alpha = np.sqrt(np.sum(resid ** 2) / (len(months) - 2) * inv[0, 0])
r2 = np.corrcoef(months['系统'], months['大盘'])[0, 1] ** 2
up = months[months['大盘'] > 0]
down = months[months['大盘'] <= 0]

year_ret = pd.to_numeric(A['系统收益'], errors='coerce').values
mkt_year_ret = pd.to_numeric(A['等权全市场收益'], errors='coerce').values
cal_days = (E.index[-1] - E.index[0]).days / 365.25
cal_cagr = (E['权益'].iloc[-1] / E['权益'].iloc[0]) ** (1 / cal_days) - 1
trade_cagr = (E['权益'].iloc[-1] / E['权益'].iloc[0]) ** (252 / len(E)) - 1
above = T[T['建仓时指数在60日线上']]
below = T[~T['建仓时指数在60日线上']]
above_ex = above[~above['年份'].isin([2022, 2026])]

summary = [
    ('年度收益均值', f'{year_ret.mean():.1%}'),
    ('年度收益标准差', f'{year_ret.std(ddof=1):.1%}'),
    ('最好年度', f"2020 +{A.loc[A['年份'] == 2020, '系统收益'].iloc[0]:.1%}"),
    ('最差年度', f"2023 {A.loc[A['年份'] == 2023, '系统收益'].iloc[0]:.1%}"),
    ('正收益年份', f"{(year_ret > 0).sum()}/{len(year_ret)}"),
    ('年度收益与大盘相关性', f'{np.corrcoef(year_ret, mkt_year_ret)[0, 1]:.2f}'),
    ('月度贝塔', f'{coef[1]:.2f}'),
    ('月度阿尔法', f'{coef[0]:.2%}/月'),
    ('阿尔法检验值', f'{coef[0] / se_alpha:.2f}'),
    ('上涨月捕获', f'{up["系统"].mean() / up["大盘"].mean():.2f}'),
    ('下跌月捕获', f'{down["系统"].mean() / down["大盘"].mean():.2f}'),
    ('全期年化（252交易日）', f'{trade_cagr:.1%}'),
    ('全期年化（日历日）', f'{cal_cagr:.1%}'),
    ('60日均线上方交易', f"{len(above)}笔，净盈亏{above['净盈亏'].sum():,.0f}元"),
    ('60日均线下方交易', f"{len(below)}笔，净盈亏{below['净盈亏'].sum():,.0f}元"),
    ('上方交易去掉2022/2026', f"{len(above_ex)}笔，净盈亏{above_ex['净盈亏'].sum():,.0f}元"),
    ('上方交易平均单笔收益', f"{above['收益率'].mean():.1%}"),
    ('下方交易平均单笔收益', f"{below['收益率'].mean():.1%}"),
]
pd.DataFrame(summary, columns=['指标', '数值']).to_csv(
    os.path.join(DIR, '年度差异_统计摘要.csv'), index=False, encoding='utf-8-sig')

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))
fig.subplots_adjust(top=0.90, bottom=0.13, left=0.07, right=0.98, hspace=0.38, wspace=0.22)

# (a) 年度收益
ax = axes[0, 0]
x = np.arange(len(A))
w = 0.38
ax.bar(x - w / 2, A['系统收益'] * 100, w, color='#F58518', label='系统')
ax.bar(x + w / 2, A['等权全市场收益'] * 100, w, color='#4C78A8', label='等权全市场')
ax.axhline(0, color='black', lw=.6, alpha=.4)
ax.set_xticks(x); ax.set_xticklabels(A['年份'])
ax.set_ylabel('年度收益（%）')
ax.set_title('每年收益：系统 vs 等权全市场', loc='left', fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=9)
ax.grid(axis='y', alpha=.22)
for i, v in enumerate(A['系统收益'] * 100):
    ax.text(i - w / 2, v + (2.2 if v >= 0 else -4.2), f'{v:.0f}%', ha='center',
            fontsize=8.5, color='#B35A00',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=1.2))

# (b) 净盈亏与去掉最大一笔
ax = axes[0, 1]
net = B['已实现净盈亏'] / 10000
ex1 = B['去掉最大一笔净盈亏'] / 10000
ax.bar(x - w / 2, net, w, color='#54A24B', label='全部交易')
ax.bar(x + w / 2, ex1, w, color='#B8B8B8', label='去掉最大一笔')
ax.axhline(0, color='black', lw=.6, alpha=.4)
ax.set_xticks(x); ax.set_xticklabels(B['年份'])
ax.set_ylabel('净盈亏（万元）')
ax.set_title('年度利润高度依赖最大一笔', loc='left', fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=9)
ax.grid(axis='y', alpha=.22)
for i, (v, s) in enumerate(zip(net, B['最大一笔占正收益'])):
    ax.text(i, v + (0.12 if v >= 0 else -0.28), f'{s:.0%}', ha='center', fontsize=8, color='#555555')

# (c) 市场宽度：全年 vs 建仓时
ax = axes[1, 0]
ax.plot(A['年份'], A['平均市场宽度20日'] * 100, marker='o', color='#4C78A8', label='全年平均宽度')
ax.plot(A['年份'], B['建仓时平均市场宽度'] * 100, marker='s', color='#F58518', label='建仓时平均宽度')
ax.plot(A['年份'], A['宽度大于50天数占比'] * 100, marker='^', color='#54A24B', label='宽度>50%天数占比')
ax.axhline(50, color='black', lw=.7, ls='--', alpha=.5)
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_xticks(A['年份'])
ax.set_ylabel('百分比（%）')
ax.set_title('系统只在“局部开门”时交易', loc='left', fontsize=12, weight='bold')
ax.legend(frameon=False, fontsize=9, loc='lower left')
ax.grid(alpha=.22)

# (d) 散点
ax = axes[1, 1]
ax.scatter(A['等权全市场收益'] * 100, A['系统收益'] * 100, s=55, color='#7B61A8', zorder=3)
for _, r in A.iterrows():
    ax.annotate(str(int(r['年份'])), (r['等权全市场收益'] * 100, r['系统收益'] * 100),
                textcoords='offset points', xytext=(6, 4), fontsize=8.5)
lim = [-40, 90]
ax.plot(lim, lim, color='grey', lw=.8, ls='--', alpha=.6)
ax.axhline(0, color='black', lw=.5, alpha=.3); ax.axvline(0, color='black', lw=.5, alpha=.3)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('等权全市场年度收益（%）'); ax.set_ylabel('系统年度收益（%）')
ax.set_title('年度收益与大盘几乎不同步', loc='left', fontsize=12, weight='bold')
ax.text(0.03, 0.94, f"相关系数 {np.corrcoef(year_ret, mkt_year_ret)[0, 1]:.2f}",
        transform=ax.transAxes, fontsize=10, color='#7B61A8')
ax.grid(alpha=.22)

fig.suptitle('45% + 波动20：年度收益为什么差别这么大', fontsize=17, y=0.965)
fig.text(0.5, 0.035,
         '口径：年度收益按自然年权益首尾；2026截至10-13；大盘为全部A股等权指数。'
         '逐笔净盈亏按卖出日归属，跨年持仓会与权益变动不一致。右上角数字为最大一笔占全部盈利比例。'
         '全期21.6%为252交易日年化，日历口径20.7%。',
         ha='center', fontsize=8.5, color='#555555')
plt.savefig(os.path.join(DIR, '年度收益与大盘状态.png'), dpi=200)
plt.close()

# ---------------- 控制台 ----------------
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
print('=== 年度收益与大盘 ===')
print(A.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 年度交易与建仓状态 ===')
print(B.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 建仓状态分组 ===')
print(C.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 摘要 ===')
print(pd.DataFrame(summary, columns=['指标', '数值']).to_string(index=False))
print(f'\n年度收益均值 {year_ret.mean():.1%}  标准差 {year_ret.std(ddof=1):.1%}')
print(f'月度贝塔 {coef[1]:.3f}  阿尔法 {coef[0]:.3%}  t={coef[0]/se_alpha:.2f}  R2={r2:.3f}')
print(f'上方交易 {len(above)} 笔净 {above["净盈亏"].sum():,.0f}；去掉2022/2026后 {len(above_ex)} 笔净 {above_ex["净盈亏"].sum():,.0f}')
print(f'下方交易 {len(below)} 笔净 {below["净盈亏"].sum():,.0f}')
print('图已保存：年度收益与大盘状态.png')
