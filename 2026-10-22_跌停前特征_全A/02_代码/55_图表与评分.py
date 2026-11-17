# -*- coding: utf-8 -*-
"""
生成重点概率表、六条风险评分、特征族汇总、11 张中文图。
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
FIG = os.path.join(ROOT, '04_图表')
os.makedirs(FIG, exist_ok=True)
t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
COL = {'红': '#C0392B', '蓝': '#2E6DA4', '绿': '#2E8B57', '橙': '#D68910', '灰': '#7F8C8D', '紫': '#7D3C98'}
FEATURE_CN = {
    'one_word_ld_yesterday': '昨日一字跌停', 'days_since_ld': '距上次跌停', 'ld_yesterday': '昨日收盘跌停',
    'consec_lu': '连续涨停天数', 'lu_count20': '20日涨停次数', 'ret1': '前1日收益', 'ret20': '前20日收益',
    'amt_ratio20': '成交额/20日均额', 'opened_ld_yesterday': '昨日打开跌停', 'dist_ma20': '距20日线',
    'opened_lu_yesterday': '昨日炸板', 'ret5': '前5日收益', 'ld_count20': '20日跌停次数',
    'ret60': '前60日收益', 'days_since_lu': '距上次涨停', 'vr5_20': '5/20日量比', 'gap': '跳空',
    'dist_high250': '距250日高点', 'mkt_ld_share': '全市场跌停占比', 'lu_yesterday': '昨日收盘涨停',
    'one_word_lu_yesterday': '昨日一字涨停', 'mkt_ret1': '全市场前1日收益',
    'vol20': '20日波动', 'atr_pct': '真实波幅/收盘', 'dist_ma60': '距60日线', 'dist_high60': '距60日高点',
    'ld_count60': '60日跌停次数', 'ld_touch_count20': '20日触及跌停次数', 'lu_count5': '5日涨停次数',
    'lu_open_count20': '20日炸板次数', 'rsi14': '相对强弱指标', 'market_breadth20': '市场宽度20',
    'mkt_ld_rank250': '市场跌停占比250日分位', 'mkt_below_ma60': '市场指数低于60日线',
    'sector_ld_share': '行业跌停占比', 'sector_ret20': '行业20日收益', 'sector_rs': '行业相对强弱',
    'sector_ret5': '行业5日收益', 'close_pos': '收盘位置', 'upper': '上影线', 'body': '实体',
    'lower': '下影线', 'macd_hist': '异同移动平均柱', 'sw_above60': '行业指数高于60日线',
}

# ---------------- 1. 重点条件表 ----------------
log('整理重点条件')
u = pd.read_csv(os.path.join(OUT, '单特征概率表.csv'))
b = pd.read_csv(os.path.join(OUT, '单特征概率表_布尔.csv'))
# 每个连续特征选 n>=1000 且 lift 最大的一组
u = u[(u['样本数'] >= 1000) & u['P(次日跌停)'].notna()]
idx = u.groupby('特征')['lift'].idxmax()
top = u.loc[idx].copy()
# 布尔特征：选“是”
b = b[(b['分组'] == '是') & (b['样本数'] >= 1000)].copy()
top = pd.concat([top, b], ignore_index=True)
top = top.sort_values('lift', ascending=False)
top['样本占比'] = top['样本数'] / 9569164
top['事件覆盖率'] = top['跌停数'] / 37412
cols_cn = ['特征', '分组', '样本数', '跌停数', 'P(次日跌停)', 'lift', 'P(5日内跌停)',
           'P(盘中触及)', '事件侧占比', '样本占比', '事件覆盖率', '训练期lift', '检验期lift', '稳定性']
top[cols_cn].to_csv(os.path.join(OUT, '重点条件概率表.csv'), index=False, encoding='utf-8-sig')

# 特征族
family = {
    '跌停惯性': ['days_since_ld', 'ld_yesterday', 'ld_count20', 'ld_count60', 'ld_touch_count20', 'ld_touch_count60', 'ld_open_count20'],
    '涨停过热': ['days_since_lu', 'lu_yesterday', 'consec_lu', 'lu_count5', 'lu_count20', 'lu_open_count20', 'lu_touch_count20', 'one_word_lu_yesterday', 'opened_lu_yesterday'],
    '位置动量': ['ret1', 'ret5', 'ret20', 'ret60', 'dist_ma20', 'dist_ma60', 'dist_high60', 'dist_high250', 'dd20', 'dd60', 'dd250', 'rsi14'],
    '波动尾部': ['vol20', 'vol60', 'atr_pct', 'vol_hist_z', 'max_drop20', 'drop5_count20', 'drop5_count60'],
    '量能流动性': ['amt_ratio20', 'vr5_20', 'amt_trend', 'log_amt20', 'vp_corr20', 'down_vol_ratio20'],
    '市场状态': ['market_breadth20', 'mkt_ld_share', 'mkt_ld_rank250', 'mkt_ret1', 'mkt_ret5', 'mkt_below_ma60', 'market_sent', 'mkt_above_ma20'],
    '行业状态': ['sector_ret20', 'sector_ret5', 'sector_ld_share', 'sector_breadth20', 'sector_rs', 'sw_above60'],
    '盘口技术': ['close_pos', 'upper', 'gap', 'macd_hist', 'body', 'lower'],
}
rows = []
for fam, fs in family.items():
    x = top[top['特征'].isin(fs)]
    if len(x) == 0:
        continue
    r = x.iloc[0]
    rows.append({'特征族': fam, '最强特征': r['特征'], '最强分组': r['分组'],
                 '最强组概率': r['P(次日跌停)'], '最强组lift': r['lift'],
                 '非稳定强组数': int((x['稳定性'] == '稳定高').sum()),
                 '特征数': len(fs)})
pd.DataFrame(rows).sort_values('最强组lift', ascending=False).to_csv(
    os.path.join(OUT, '特征族汇总.csv'), index=False, encoding='utf-8-sig')

# ---------------- 2. 六条风险评分 ----------------
log('六条风险评分')
cols = ['date', 'ld1', 'ld5', 'ldt1', 'ld_yesterday', 'consec_lu', 'vol20', 'ret20',
        'amt_ratio20', 'mkt_ld_share', 'market_breadth20', 'mkt_below_ma60',
        'st_proxy', 'approx', 'is_new']
df = pd.read_parquet(os.path.join(ROOT, '01_数据', '分析面板.parquet'), columns=cols)
df['date'] = pd.to_datetime(df['date'])
main = (df['st_proxy'].fillna(0) == 0) & (df['approx'] == 0) & (df['is_new'] == 0)
df = df.loc[main].copy()
df['评分'] = ((df['ld_yesterday'] >= 1).astype(int) + (df['consec_lu'] >= 2).astype(int) +
             (df['vol20'] >= 0.50).astype(int) + (df['mkt_ld_share'] >= 0.005).astype(int) +
             (df['ret20'] >= 0.20).astype(int) + (df['amt_ratio20'] >= 2.0).astype(int))
base = df['ld1'].mean()
g = df.groupby('评分').agg(样本数=('ld1', 'size'), 跌停数=('ld1', 'sum'),
                            P次日=('ld1', 'mean'), P五日内=('ld5', 'mean'),
                            P触及=('ldt1', 'mean')).reset_index()
g['lift'] = g['P次日'] / base
g['事件覆盖率'] = g['跌停数'] / df['ld1'].sum()
g['样本占比'] = g['样本数'] / len(df)
# 训练/检验
df['段'] = np.where(df['date'] < '2022-01-01', '训练期', '检验期')
for seg in ['训练期', '检验期']:
    x = df[df['段'] == seg].groupby('评分')['ld1'].mean()
    g[seg + '概率'] = g['评分'].map(x)
g['训练期lift'] = g['训练期概率'] / df.loc[df['段'] == '训练期', 'ld1'].mean()
g['检验期lift'] = g['检验期概率'] / df.loc[df['段'] == '检验期', 'ld1'].mean()
# cluster CI
def cluster_ci(mask, B=1000):
    sub = df.loc[mask, ['date', 'ld1']]
    gg = sub.groupby('date').agg(n=('ld1', 'size'), k=('ld1', 'sum'))
    n = gg['n'].values.astype(float); k = gg['k'].values.astype(float)
    rng = np.random.default_rng(20261022); nd = len(n); rates = np.empty(B)
    for i in range(B):
        j = rng.integers(0, nd, nd)
        rates[i] = k[j].sum() / max(n[j].sum(), 1)
    return np.percentile(rates, 2.5), np.percentile(rates, 97.5)
lo, hi = zip(*[cluster_ci(df['评分'] == s) for s in g['评分']])
g['cluster下限'] = lo; g['cluster上限'] = hi
g.to_csv(os.path.join(OUT, '风险评分_六条清单.csv'), index=False, encoding='utf-8-sig')
print(g.to_string(index=False, float_format=lambda z: f'{z:.4g}'))

# ---------------- 3. 图1 年度 ----------------
log('图1')
yr = pd.read_csv(os.path.join(OUT, '分年基准率.csv'))
mkt = pd.read_csv(os.path.join(ROOT, '01_数据', '市场日表.csv'), parse_dates=['date'])
mkt['year'] = mkt['date'].dt.year
mb = mkt.groupby('year')['market_breadth20'].mean()
fig, ax = plt.subplots(figsize=(9.2, 4.8))
bars = ax.bar(yr['年份'].astype(str), yr['跌停率'] * 100, color=COL['红'], alpha=0.78, label='收盘跌停率')
ax.set_ylabel('收盘跌停率（%）')
ax.set_xlabel('年份')
for i, v in enumerate(yr['跌停率'] * 100):
    ax.text(i, v + 0.012, f'{v:.2f}%', ha='center', fontsize=8)
ax2 = ax.twinx()
ax2.plot(yr['年份'].astype(str), mb.reindex(yr['年份']).values * 100, color=COL['蓝'],
         marker='o', lw=1.8, label='年均市场宽度')
ax2.set_ylabel('市场宽度（%）', color=COL['蓝'])
ax2.tick_params(axis='y', colors=COL['蓝'])
ax.set_title('年度收盘跌停率与市场宽度（2016、2026 为不完整年度）', fontsize=12, pad=10)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', frameon=False, fontsize=9)
plt.savefig(os.path.join(FIG, '图1_年度跌停率与市场宽度.png')); plt.close()

# ---------------- 4. 图2 单条件 lift ----------------
log('图2')
x = top[(top['样本数'] >= 2000) & (top['特征'] != 'days_since_ld')].head(22).iloc[::-1]
labels = [f"{FEATURE_CN.get(r['特征'], r['特征'])}：{r['分组']}" for _, r in x.iterrows()]
fig, ax = plt.subplots(figsize=(10.5, 7.6))
colors = [COL['红'] if r['lift'] >= 10 else (COL['橙'] if r['lift'] >= 4 else COL['蓝']) for _, r in x.iterrows()]
ax.barh(labels, x['lift'], color=colors, alpha=0.85)
for i, (_, r) in enumerate(x.iterrows()):
    ax.text(r['lift'] + 0.3, i, f"{r['P(次日跌停)']*100:.2f}%（{r['lift']:.1f}×）", va='center', fontsize=8)
ax.axvline(1, color=COL['灰'], ls='--', lw=1)
ax.set_xlabel('相对全样本基准（0.391%）的倍数')
ax.set_title('单条件次日跌停概率：提升倍数最大的 22 组', fontsize=12, pad=10)
ax.set_xlim(0, max(x['lift']) * 1.22)
plt.savefig(os.path.join(FIG, '图2_单条件提升倍数.png')); plt.close()

# ---------------- 5. 图3 事件窗口 ----------------
log('图3')
path = pd.read_csv(os.path.join(OUT, '事件窗口_特征路径.csv'))
fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
for ax, f, title in [
    (axes[0, 0], 'ret20', '前20日收益中位数'),
    (axes[0, 1], 'vol20', '20日年化波动中位数'),
    (axes[1, 0], 'dist_ma20', '距20日均线中位数'),
    (axes[1, 1], 'market_breadth20', '市场宽度中位数')]:
    for gname, color, ls in [('全部', COL['红'], '-'), ('首次', COL['蓝'], '--'), ('连续', COL['橙'], '-.')]:
        z = path[(path.feature == f) & (path.group == gname)].sort_values('offset')
        ax.plot(z['offset'], z['median'], color=color, ls=ls, lw=1.8, label=gname)
    ax.axvline(0, color=COL['灰'], ls=':', lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('相对跌停日（0=T-1收盘，1=跌停日收盘）')
    if f in ('ret20', 'dist_ma20', 'vol20', 'market_breadth20'):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('跌停前 10 日至跌停后 6 日：特征中位数路径', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG, '图3_事件窗口路径.png')); plt.close()

# ---------------- 6. 图4/5 热力图 ----------------
log('图4/5')
def heat(name, title, fname, row_order, col_order, xlabel, ylabel):
    d = pd.read_csv(os.path.join(OUT, name))
    pv = d.pivot(index='x1', columns='x2', values='rate').reindex(index=row_order, columns=col_order)
    nv = d.pivot(index='x1', columns='x2', values='n').reindex(index=row_order, columns=col_order)
    pv = pv.where(nv >= 500)
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    im = ax.imshow(pv.values * 100, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index, fontsize=9)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v*100:.2f}%', ha='center', va='center', fontsize=8,
                        color='white' if v * 100 > 3 else 'black')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, label='次日跌停率（%）', shrink=0.85)
    plt.savefig(os.path.join(FIG, fname)); plt.close()
heat('二维_20日收益与距60日线.csv', '20日收益 × 距60日线：次日跌停概率', '图4_收益位置热力图.png',
     ['≤-30%','-30~-20%','-20~-10%','-10~0%','0~10%','10~20%','20~30%','30~50%','≥50%'],
     ['≤-30%','-30~-15%','-15~-5%','-5~0%','0~10%','10~20%','≥20%'], '距60日线', '前20日收益')
heat('二维_市场跌停占比与个股波动.csv', '市场跌停占比 × 个股20日波动：次日跌停概率', '图5_市场波动热力图.png',
     ['0','0~0.2%','0.2~0.5%','0.5~1%','1~2%','2~4%','≥4%'],
     ['≤20%','20~30%','30~40%','40~50%','50~70%','≥70%'], '个股20日波动', '全市场跌停占比')

# ---------------- 7. 图6 形态 ----------------
log('图6')
pat = pd.read_csv(os.path.join(OUT, '形态组合概率.csv'))
fig, ax = plt.subplots(figsize=(9.8, 6.2))
sc = ax.scatter(pat['事件覆盖率'] * 100, pat['P(次日跌停)'] * 100,
                s=25 + 220 * (pat['样本数'] / pat['样本数'].max()) ** 0.5, c=pat['lift'], cmap='YlOrRd',
                edgecolor='#333', alpha=0.85)
short = {'连板+弱市场': '连板+弱市', '连板+高波动': '连板+高波', '连板：昨日≥2连板': '≥2连板',
         '昨日炸板：触涨停未封': '昨日炸板', '市场恐慌+高波动+前期跌停': '恐慌+高波+前期跌停',
         '高位加速+弱市场': '高位加速+弱市', '昨日涨停': '昨日涨停', '前期跌停+弱市场': '前期跌停+弱市'}
offsets = {'连板+弱市场': (8, 3), '连板+高波动': (8, -14), '连板：昨日≥2连板': (8, 4),
           '昨日炸板：触涨停未封': (-12, 12), '市场恐慌+高波动+前期跌停': (8, -16),
           '高位加速+弱市场': (8, 4), '昨日涨停': (8, -12), '前期跌停+弱市场': (8, 4)}
for _, r in pat.head(8).iterrows():
    ax.annotate(short.get(r['形态'], r['形态']), (r['事件覆盖率'] * 100, r['P(次日跌停)'] * 100),
                xytext=offsets.get(r['形态'], (7, 4)), textcoords='offset points', fontsize=8)
ax.set_xlabel('事件覆盖率（%）：该形态覆盖了多少跌停事件')
ax.set_ylabel('次日跌停概率（%）')
ax.set_title('形态组合：概率与覆盖率（颜色越深 lift 越高）', fontsize=12, pad=10)
fig.colorbar(sc, ax=ax, label='提升倍数', shrink=0.8)
plt.savefig(os.path.join(FIG, '图6_形态概率与覆盖率.png')); plt.close()

# ---------------- 8. 图7 校准 ----------------
log('图7')
cal = pd.read_csv(os.path.join(OUT, '模型校准表_校准后.csv'))
cmp_ = pd.read_csv(os.path.join(OUT, '模型校准后对比.csv'))
simple = pd.read_csv(os.path.join(OUT, '模型简单清单.csv'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.plot([0, 5], [0, 5], color=COL['灰'], ls='--', lw=1)
ax.plot(cal['预测均值'] * 100, cal['实际跌停率'] * 100, marker='o', color=COL['红'], lw=2)
ax.set_xlabel('预测概率（%）'); ax.set_ylabel('实际跌停率（%）')
ax.set_xlim(0, 2.6); ax.set_ylim(0, 2.6)
ax.set_title('十分位校准（2022–2026 检验期）', fontsize=10)
ax = axes[1]
names = ['top1%', 'top5%', 'top10%']
colors = [COL['红'], COL['蓝']]
for i, (dfx, label) in enumerate([(cmp_.iloc[1], '市场+个股（校准）'), (simple.iloc[0], '简单清单（校准）')]):
    vals = [dfx['top1%lift'], dfx['top5%lift'], dfx['top10%lift']]
    ax.bar(np.arange(3) + i * 0.36, vals, width=0.34, color=colors[i], alpha=0.85, label=label)
    for j, v in enumerate(vals):
        ax.text(j + i * 0.36, v + 0.4, f'{v:.1f}×', ha='center', fontsize=8)
ax.set_xticks(np.arange(3) + 0.18); ax.set_xticklabels(names)
ax.set_ylabel('相对基准的提升倍数')
ax.set_title('头部风险名单的提升倍数', fontsize=10)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, '图7_模型校准与头部提升.png')); plt.close()

# ---------------- 9. 图8 模型 AUC ----------------
log('图8')
mc = pd.read_csv(os.path.join(OUT, '模型对比.csv'))
labels = ['仅市场', '仅个股', '市场+个股']
vals = [mc.loc[mc['模型'] == x, 'AUC'].iloc[0] for x in labels]
simple_auc = simple['AUC'].iloc[0]
within = pd.read_csv(os.path.join(OUT, '模型日内排序能力_市场+个股.csv'))['日内AUC'].iloc[0]
within_s = pd.read_csv(os.path.join(OUT, '模型日内排序能力_简单清单.csv'))['日内AUC'].iloc[0]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
xpos = np.arange(5)
vals_all = vals + [simple_auc, within]
labels_all = labels + ['简单清单', '市场+个股\n日内曲线下面积']
colors = [COL['灰'], COL['蓝'], COL['红'], COL['橙'], COL['紫']]
bars = ax.bar(xpos, vals_all, color=colors, alpha=0.85)
for i, v in enumerate(vals_all):
    ax.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=9)
ax.axhline(0.5, color='k', ls=':', lw=1)
ax.set_ylim(0.45, 1.0); ax.set_ylabel('曲线下面积')
ax.set_title('排序能力：整体曲线下面积与日内曲线下面积（2022–2026 样本外）', fontsize=12, pad=10)
ax.set_xticks(xpos); ax.set_xticklabels(labels_all, fontsize=9)
plt.savefig(os.path.join(FIG, '图8_模型排序能力对比.png')); plt.close()

# ---------------- 10. 图9 分板块 ----------------
log('图9')
bb = pd.read_csv(os.path.join(OUT, '稳健性_分板块.csv'))
conds = ['基准', '昨收跌停', '≥2连板', '20日涨≥20%', '20日波动≥50%', '市场跌停占比1~2%']
segs = ['主板', '创业板-10%时代', '创业板-20%时代', '科创板']
fig, ax = plt.subplots(figsize=(10.2, 5.0))
w = 0.2
for i, seg in enumerate(segs):
    vals = [bb[(bb['板块期间'] == seg) & (bb['条件'] == c)]['概率'].iloc[0] * 100 if len(bb[(bb['板块期间'] == seg) & (bb['条件'] == c)]) else np.nan for c in conds]
    ax.bar(np.arange(len(conds)) + i * w, vals, width=w, label=seg, alpha=0.85)
ax.set_yscale('log')
ax.set_ylabel('概率（%，对数轴）')
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:g}%'))
ax.set_xticks(np.arange(len(conds)) + 1.5 * w); ax.set_xticklabels(conds, fontsize=9)
ax.set_title('分板块：基准率与关键条件概率（对数轴）', fontsize=12, pad=10)
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.savefig(os.path.join(FIG, '图9_分板块概率.png')); plt.close()

# ---------------- 11. 图10 原型 ----------------
log('图10')
ar = pd.read_csv(os.path.join(OUT, '跌停原型.csv'))
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
ax = axes[0]
ax.barh(ar['原型'], ar['事件侧占比'] * 100, color=COL['蓝'], alpha=0.85)
for i, v in enumerate(ar['事件侧占比'] * 100):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('占全部跌停事件（%）'); ax.set_title('事件侧：跌停前原型占比', fontsize=10)
ax = axes[1]
ax.barh(ar['原型'], ar['P(次日跌停)'] * 100, color=COL['红'], alpha=0.85)
for i, v in enumerate(ar['P(次日跌停)'] * 100):
    ax.text(v + 0.02, i, f'{v:.2f}%', va='center', fontsize=9)
ax.axvline(0.391, color=COL['灰'], ls='--', lw=1, label='全样本基准 0.39%')
ax.set_xlabel('次日跌停概率（%）'); ax.set_title('预测侧：不同原型的条件概率', fontsize=10)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, '图10_跌停原型.png')); plt.close()

# ---------------- 12. 图11 风险评分 ----------------
log('图11')
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
ax = axes[0]
ax.bar(g['评分'].astype(str), g['P次日'] * 100, color=COL['红'], alpha=0.85)
for i, (_, r) in enumerate(g.iterrows()):
    ax.text(i, r['P次日'] * 100 + 0.2, f"{r['P次日']*100:.2f}%\n(n={int(r['样本数']):,})",
            ha='center', fontsize=7.5)
ax.set_xlabel('满足的风险条数（0–5）'); ax.set_ylabel('次日跌停概率（%）')
ax.set_title('六条风险清单：评分与次日跌停率', fontsize=10)
ax = axes[1]
ax.bar(g['评分'].astype(str), g['事件覆盖率'] * 100, color=COL['蓝'], alpha=0.85)
for i, v in enumerate(g['事件覆盖率'] * 100):
    ax.text(i, v + 0.4, f'{v:.1f}%', ha='center', fontsize=8)
ax.set_xlabel('满足的风险条数（0–5）'); ax.set_ylabel('覆盖的跌停事件（%）')
ax.set_title('各评分覆盖的跌停事件比例', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG, '图11_六条风险评分.png')); plt.close()

# ---------------- 13. 图12 行业 ----------------
log('图12')
ind = pd.read_csv(os.path.join(OUT, '分行业基准率.csv'))
ind = ind[ind['行业'].notna()].sort_values('跌停率')
fig, ax = plt.subplots(figsize=(8.4, 7.0))
colors = [COL['红'] if v >= 0.006 else (COL['橙'] if v >= 0.004 else COL['蓝']) for v in ind['跌停率']]
ax.barh(ind['行业'], ind['跌停率'] * 100, color=colors, alpha=0.85)
for i, (_, r) in enumerate(ind.iterrows()):
    ax.text(r['跌停率'] * 100 + 0.01, i, f"{r['跌停率']*100:.2f}%", va='center', fontsize=7)
ax.axvline(0.391, color=COL['灰'], ls='--', lw=1, label='全样本 0.39%')
ax.set_xlabel('次日收盘跌停率（%）'); ax.set_title('分行业跌停率（2016–2026）', fontsize=11)
ax.legend(frameon=False, fontsize=8)
plt.savefig(os.path.join(FIG, '图12_分行业跌停率.png')); plt.close()

log('完成')
