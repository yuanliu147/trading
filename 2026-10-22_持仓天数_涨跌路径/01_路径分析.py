# -*- coding: utf-8 -*-
"""买入后第1–60个交易日的涨跌路径。

回答问题：
1) 买入后，持有到第1、2、…、20天，上涨/下跌的概率各是多少？
2) 上涨时一般涨多少、下跌时一般跌多少（条件分布）？
3) 最终赚钱的交易，路径上是否先亏过？最终亏钱的，是否先赚过？
4) 最大浮盈/最大浮亏通常出现在第几天？20日卖出回吐/反弹了多少？
5) 如果固定持有N天卖出，期望收益、胜率、盈亏比怎么变？

口径：
- 买入价 = 信号次日开盘（t+1 open）；与原系统一致。
- 第d天 = 买入日起第d个交易日收盘（第1天=买入当日收盘）；
  第20天 = 原系统“持有20日”的卖点（买入日+19个交易日）。
- 路径收益为毛收益（未扣费）；第20天另算扣费后净收益并和原逐笔对账。
- 停牌日收盘用前收盘填充；盘中最高/最低用窗口内可得数据累计。
- 全信号/70-15 的逐笔高度重叠，概率是“按笔加权”的描述统计，不是独立样本。
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/admin/WorkSpace/trading'
SRC = os.path.join(BASE, '2026-10-20_涨幅趋势_全维度定制')
DIR = os.path.join(BASE, '2026-10-22_持仓天数_涨跌路径')
SYS_DIR = os.path.join(BASE, '2026-10-21_45仓位_波动20_交易系统')
ALL_DIR = os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究')

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

DMAX = 60
D20 = 20
KEY_DAYS = [1, 2, 3, 5, 7, 10, 12, 15, 18, 20]

# ---------------- 面板 ----------------
P = pickle.load(open(os.path.join(SRC, '01_数据/全A面板_2016_2026_含退市.pkl'), 'rb'))
dates = pd.DatetimeIndex(P['meta']['dates'])
codes = np.array(P['meta']['codes'])
code_idx = {c: i for i, c in enumerate(codes)}
O = P['open'].values.astype(float)
H = P['high'].values.astype(float)
L = P['low'].values.astype(float)
C = P['close'].values.astype(float)
T = len(dates)
del P


def prep(df, name):
    df = df.copy()
    df['样本'] = name
    df['j'] = df['代码'].map(code_idx)
    df = df[df['j'].notna()].copy()
    df['j'] = df['j'].astype(int)
    df['i'] = dates.searchsorted(pd.DatetimeIndex(df['买入日']))
    df = df[(df['i'] >= 0) & (df['i'] < T)].copy()
    df['买入价_面板'] = O[df['i'].values, df['j'].values]
    return df.reset_index(drop=True)


def windows(df):
    n = len(df)
    i0 = df['i'].values
    j = df['j'].values
    rows = i0[:, None] + np.arange(DMAX)[None, :]
    valid = rows < T
    rc = np.clip(rows, 0, T - 1)
    Cw = C[rc, j[:, None]].astype(float)
    Hw = H[rc, j[:, None]].astype(float)
    Lw = L[rc, j[:, None]].astype(float)
    Cw[~valid] = np.nan
    Hw[~valid] = np.nan
    Lw[~valid] = np.nan
    ent = df['买入价'].values.astype(float)
    ent = np.where(np.isfinite(ent) & (ent > 0), ent, O[i0, j])
    # 买入日收盘缺失时用买入价，之后停牌用前收盘
    Cw[:, 0] = np.where(np.isfinite(Cw[:, 0]), Cw[:, 0], ent)
    ret = pd.DataFrame(Cw).ffill(axis=1).values / ent[:, None] - 1
    ret[~valid] = np.nan
    # 盘中累计最高/最低
    hi = pd.DataFrame(Hw).cummax(axis=1, skipna=True).values / ent[:, None] - 1
    lo = pd.DataFrame(Lw).cummin(axis=1, skipna=True).values / ent[:, None] - 1
    hi[~valid] = np.nan
    lo[~valid] = np.nan
    ret_cummax = pd.DataFrame(ret).cummax(axis=1, skipna=True).values.copy()
    ret_cummin = pd.DataFrame(ret).cummin(axis=1, skipna=True).values.copy()
    ret_cummax[~valid] = np.nan
    ret_cummin[~valid] = np.nan
    return dict(ret=ret, hi=hi, lo=lo, ent=ent,
                cumnmax=ret_cummax, cummin=ret_cummin)


def argday(a):
    """最大值首次出现的第几天；全 NaN 返回 -1。"""
    finite = np.isfinite(a)
    z = np.where(finite, a, -np.inf)
    k = z.argmax(axis=1) + 1
    k[~finite.any(axis=1)] = -1
    return k


def argday_min(a):
    finite = np.isfinite(a)
    z = np.where(finite, a, np.inf)
    k = z.argmin(axis=1) + 1
    k[~finite.any(axis=1)] = -1
    return k


def peak_trough(ret20):
    finite = np.isfinite(ret20)
    peak = np.where(finite, ret20, -np.inf).max(axis=1)
    trough = np.where(finite, ret20, np.inf).min(axis=1)
    peak[~finite.any(axis=1)] = np.nan
    trough[~finite.any(axis=1)] = np.nan
    return peak, trough


def path_table(df, W, name):
    rows = []
    ret = W['ret']; hi = W['hi']; lo = W['lo']
    for d in range(1, DMAX + 1):
        x = ret[:, d - 1]
        m = np.isfinite(x)
        xv = x[m]
        up = xv > 0
        dn = xv < 0
        # 盘中“曾触及”按同一批有效样本统计
        hm = hi[:, :d][m]
        lm = lo[:, :d][m]
        hmax = np.where(np.isfinite(hm), hm, -np.inf).max(axis=1)
        lmin = np.where(np.isfinite(lm), lm, np.inf).min(axis=1)
        row = dict(样本=name, 持有天数=d, 样本数=int(m.sum()),
                   上涨概率=up.mean(),
                   上涨平均涨幅=xv[up].mean() if up.any() else np.nan,
                   上涨中位涨幅=np.median(xv[up]) if up.any() else np.nan,
                   下跌概率=dn.mean(),
                   下跌平均跌幅=xv[dn].mean() if dn.any() else np.nan,
                   下跌中位跌幅=np.median(xv[dn]) if dn.any() else np.nan,
                   平均收益=xv.mean(), 中位收益=np.median(xv),
                   涨超5概率=(xv >= .05).mean(), 涨超10概率=(xv >= .10).mean(),
                   涨超20概率=(xv >= .20).mean(), 涨超30概率=(xv >= .30).mean(),
                   跌超5概率=(xv <= -.05).mean(), 跌超10概率=(xv <= -.10).mean(),
                   跌超20概率=(xv <= -.20).mean(), 跌超30概率=(xv <= -.30).mean(),
                   分位5=np.quantile(xv, .05), 分位25=np.quantile(xv, .25),
                   分位75=np.quantile(xv, .75), 分位95=np.quantile(xv, .95),
                   盘中曾涨5概率=np.nanmean(hmax >= .05),
                   盘中曾涨10概率=np.nanmean(hmax >= .10),
                   盘中曾涨20概率=np.nanmean(hmax >= .20),
                   盘中曾跌5概率=np.nanmean(lmin <= -.05),
                   盘中曾跌10概率=np.nanmean(lmin <= -.10),
                   盘中曾跌20概率=np.nanmean(lmin <= -.20))
        rows.append(row)
    return pd.DataFrame(rows)


def cond_table(df, W, name):
    """上涨/下跌条件下的分位数（只看条件后的幅度）。"""
    ret = W['ret']
    rows = []
    for d in KEY_DAYS + [30, 40, 60]:
        x = ret[:, d - 1]
        x = x[np.isfinite(x)]
        up = x[x > 0]
        dn = x[x < 0]
        def qs(a):
            if len(a) == 0:
                return [np.nan] * 5
            return [np.quantile(a, p) for p in (.1, .25, .5, .75, .9)]
        uq = qs(up); dq = qs(dn)
        rows.append(dict(样本=name, 持有天数=d, 上涨笔数=len(up), 下跌笔数=len(dn),
                         涨幅P10=uq[0], 涨幅P25=uq[1], 涨幅中位=uq[2], 涨幅P75=uq[3], 涨幅P90=uq[4],
                         跌幅P10=dq[0], 跌幅P25=dq[1], 跌幅中位=dq[2], 跌幅P75=dq[3], 跌幅P90=dq[4]))
    return pd.DataFrame(rows)


def cond_threshold_table(df, W, name):
    """上涨组里涨超X%的比例；下跌组里跌超X%的比例。"""
    ret = W['ret']
    rows = []
    for d in range(1, DMAX + 1):
        x = ret[:, d - 1]
        x = x[np.isfinite(x)]
        up = x[x > 0]
        dn = x[x < 0]
        rows.append(dict(样本=name, 持有天数=d, 上涨笔数=len(up), 下跌笔数=len(dn),
                         涨超5=(up >= .05).mean(), 涨超10=(up >= .10).mean(),
                         涨超20=(up >= .20).mean(), 涨超30=(up >= .30).mean(),
                         跌超5=(dn <= -.05).mean(), 跌超10=(dn <= -.10).mean(),
                         跌超20=(dn <= -.20).mean(), 跌超30=(dn <= -.30).mean()))
    return pd.DataFrame(rows)


def group_table(df, W, name):
    ret20 = W['ret'][:, :D20]
    final = ret20[:, -1]
    peak, trough = peak_trough(ret20)
    pk_day = argday(ret20)
    tr_day = argday_min(ret20)
    hi20 = W['hi'][:, :D20]
    lo20 = W['lo'][:, :D20]
    hit_hi = np.where(np.isfinite(hi20), hi20, -np.inf).max(axis=1)
    hit_lo = np.where(np.isfinite(lo20), lo20, np.inf).min(axis=1)
    base = np.isfinite(final)
    groups = [('最终上涨', base & (final > 0)), ('最终下跌', base & (final < 0)),
              ('最终持平', base & (final == 0)), ('全部', base)]
    rows = []
    for gname, m in groups:
        if m.sum() == 0:
            continue
        r = dict(样本=name, 组别=gname, 笔数=int(m.sum()), 占比=m.sum() / base.sum(),
                 期末平均=final[m].mean(), 期末中位=np.median(final[m]),
                 收盘最大浮盈均值=peak[m].mean(), 收盘最大浮盈中位=np.median(peak[m]),
                 收盘最大浮亏均值=trough[m].mean(), 收盘最大浮亏中位=np.median(trough[m]),
                 盘中最大浮盈均值=hit_hi[m].mean(), 盘中最大浮亏均值=hit_lo[m].mean(),
                 曾浮亏比例=(trough[m] < 0).mean(), 曾浮盈比例=(peak[m] > 0).mean(),
                 曾盘中涨10比例=(hit_hi[m] >= .10).mean(), 曾盘中跌10比例=(hit_lo[m] <= -.10).mean(),
                 峰值日中位=np.median(pk_day[m][pk_day[m] > 0]),
                 谷值日中位=np.median(tr_day[m][tr_day[m] > 0]),
                 峰值日在5日内=(pk_day[m] <= 5).mean(), 峰值日在10日内=(pk_day[m] <= 10).mean(),
                 谷值日在5日内=(tr_day[m] <= 5).mean(), 谷值日在10日内=(tr_day[m] <= 10).mean())
        if gname == '最终上涨':
            ok = peak[m] > 0
            r['峰值回吐均值'] = (peak[m] - final[m]).mean()
            r['期末保留峰值比例中位'] = np.median(final[m][ok] / peak[m][ok])
            r['浮亏超5比例'] = (trough[m] <= -.05).mean()
            r['浮亏超10比例'] = (trough[m] <= -.10).mean()
        if gname == '最终下跌':
            r['谷值反弹均值'] = (final[m] - trough[m]).mean()
            r['浮盈超5比例'] = (peak[m] >= .05).mean()
            r['浮盈超10比例'] = (peak[m] >= .10).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def group_path(df, W, name):
    ret = W['ret'][:, :D20]
    final = ret[:, -1]
    valid = np.isfinite(final)
    rows = []
    for gname, m in [('最终上涨', valid & (final > 0)), ('最终下跌', valid & (final < 0)),
                     ('全部', valid)]:
        for d in range(1, D20 + 1):
            x = ret[m, d - 1]
            x = x[np.isfinite(x)]
            rows.append(dict(样本=name, 组别=gname, 持有天数=d, 笔数=len(x),
                             平均收益=x.mean(), 中位收益=np.median(x),
                             上涨概率=(x > 0).mean()))
    return pd.DataFrame(rows)


def tail_table(df, W, name):
    ret20 = W['ret'][:, :D20]
    peak, trough = peak_trough(ret20)
    pk_day = argday(ret20)
    tr_day = argday_min(ret20)
    hi20 = W['hi'][:, :D20]
    lo20 = W['lo'][:, :D20]
    hit_hi = np.where(np.isfinite(hi20), hi20, -np.inf).max(axis=1)
    hit_lo = np.where(np.isfinite(lo20), lo20, np.inf).min(axis=1)
    m = np.isfinite(peak) & np.isfinite(trough)
    r = dict(样本=name, 笔数=int(m.sum()),
             收盘最大浮盈均值=peak[m].mean(), 收盘最大浮盈中位=np.median(peak[m]),
             收盘最大浮亏均值=trough[m].mean(), 收盘最大浮亏中位=np.median(trough[m]),
             盘中最大浮盈均值=hit_hi[m].mean(), 盘中最大浮亏均值=hit_lo[m].mean(),
             峰值日中位=np.median(pk_day[m][pk_day[m] > 0]),
             谷值日中位=np.median(tr_day[m][tr_day[m] > 0]))
    for th in [.05, .10, .20, .30, .50]:
        r[f'收盘曾涨{int(th*100)}'] = (peak[m] >= th).mean()
        r[f'盘中曾涨{int(th*100)}'] = (hit_hi[m] >= th).mean()
    for th in [.05, .10, .20, .30]:
        r[f'收盘曾跌{int(th*100)}'] = (trough[m] <= -th).mean()
        r[f'盘中曾跌{int(th*100)}'] = (hit_lo[m] <= -th).mean()
    return pd.DataFrame([r])


def exit_table(df, W, name):
    ret = W['ret']
    rows = []
    for d in range(1, DMAX + 1):
        x = ret[:, d - 1]
        m = np.isfinite(x)
        xv = x[m]
        if len(xv) == 0:
            continue
        sell = 10000 * (1 + xv)
        fee = 15.0 + np.maximum(5.0, sell * 0.0015)
        net = (sell - 10000 - fee) / 10000
        pos = net[net > 0]; neg = net[net < 0]
        rows.append(dict(样本=name, 持有天数=d, 样本数=int(m.sum()),
                         毛平均=xv.mean(), 毛中位=np.median(xv), 毛胜率=(xv > 0).mean(),
                         毛盈亏比=pos.sum() / max(1e-9, -neg.sum()) if len(neg) else np.nan,
                         毛总盈亏=xv.sum() * 10000,
                         净平均=net.mean(), 净中位=np.median(net), 净胜率=(net > 0).mean(),
                         净总盈亏=net.sum() * 10000))
    return pd.DataFrame(rows)


def cont_table(df, W, name):
    ret = W['ret']
    rows = []
    for d0 in [3, 5, 10, 15]:
        x0 = ret[:, d0 - 1]
        x20 = ret[:, D20 - 1]
        m0 = np.isfinite(x0) & np.isfinite(x20)
        for side, sel in [('第%d天上涨' % d0, m0 & (x0 > 0)),
                          ('第%d天下跌' % d0, m0 & (x0 < 0))]:
            if sel.sum() == 0:
                continue
            inc = ret[sel, D20 - 1] - ret[sel, d0 - 1]
            after = ret[sel, d0:D20]
            fmax = np.where(np.isfinite(after), after, -np.inf).max(axis=1) - ret[sel, d0 - 1]
            fmin = np.where(np.isfinite(after), after, np.inf).min(axis=1) - ret[sel, d0 - 1]
            rows.append(dict(样本=name, 分组=side, 笔数=int(sel.sum()),
                             第20天上涨概率=(ret[sel, D20 - 1] > 0).mean(),
                             第20天平均收益=ret[sel, D20 - 1].mean(),
                             第20天中位收益=np.median(ret[sel, D20 - 1]),
                             之后平均增量=inc.mean(), 之后中位增量=np.median(inc),
                             之后增量为正概率=(inc > 0).mean(),
                             之后最大再涨均值=fmax.mean(), 之后最大再跌均值=fmin.mean()))
    return pd.DataFrame(rows)


# ---------------- 样本 ----------------
sys_df = pd.read_csv(os.path.join(SYS_DIR, '逐笔交易记录.csv'), encoding='utf-8-sig',
                     parse_dates=['买入日', '卖出日'])
sys_df = sys_df.rename(columns={'买入价': '买入价'})
sys_df['代码'] = sys_df['代码'].astype(str)
sys_df['年份'] = sys_df['买入日'].dt.year
SYS = prep(sys_df[['代码', '买入日', '买入价', '持有天数', '退出原因', '年份',
                   '收益率', '净盈亏', '买入金额']], '主系统150')

all_df = pd.read_csv(os.path.join(ALL_DIR, '全信号_一万_逐笔.csv'), encoding='utf-8-sig',
                     usecols=['代码', '买入日', '买入价', '年份', '成交额', '股价',
                              '不止损净盈亏', '不止损收益率', '持有天数', '系统买入', '不止损退出'],
                     parse_dates=['买入日'])
ALL = prep(all_df, '全信号')
SUB = prep(all_df[(all_df['成交额'] <= 70e6) & (all_df['股价'] <= 15)].copy(), '70/15子池')

samples = [('主系统150', SYS), ('全信号', ALL), ('70/15子池', SUB)]
W = {}
for name, d in samples:
    W[name] = windows(d)
    print(name, 'trades', len(d), 'entry price diff max',
          np.nanmax(np.abs(d['买入价'].values - d['买入价_面板'].values)))

# ---------------- 汇总表 ----------------
PATH = pd.concat([path_table(d, W[n], n) for n, d in samples], ignore_index=True)
COND = pd.concat([cond_table(d, W[n], n) for n, d in samples], ignore_index=True)
CTHR = pd.concat([cond_threshold_table(d, W[n], n) for n, d in samples], ignore_index=True)
GROUP = pd.concat([group_table(d, W[n], n) for n, d in samples], ignore_index=True)
GPATH = pd.concat([group_path(d, W[n], n) for n, d in samples], ignore_index=True)
TAIL = pd.concat([tail_table(d, W[n], n) for n, d in samples], ignore_index=True)
EXIT = pd.concat([exit_table(d, W[n], n) for n, d in samples], ignore_index=True)
CONT = pd.concat([cont_table(d, W[n], n) for n, d in samples], ignore_index=True)


def yearly_table(df, W, name):
    ret = W['ret']
    rows = []
    for y, idx in df.groupby('年份').groups.items():
        idx = np.array(list(idx))
        r = dict(样本=name, 年份=int(y), 笔数=len(idx))
        for d in [5, 10, 20]:
            x = ret[idx, d - 1]
            x = x[np.isfinite(x)]
            r[f'第{d}天平均'] = x.mean() if len(x) else np.nan
            r[f'第{d}天胜率'] = (x > 0).mean() if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


YEAR = pd.concat([yearly_table(d, W[n], n) for n, d in samples], ignore_index=True)

# 逐笔摘要
def trade_summary(df, W, name):
    ret = W['ret']
    r20 = ret[:, :D20]
    peak, trough = peak_trough(r20)
    hi20 = W['hi'][:, :D20]
    lo20 = W['lo'][:, :D20]
    hit_hi = np.where(np.isfinite(hi20), hi20, -np.inf).max(axis=1)
    hit_lo = np.where(np.isfinite(lo20), lo20, np.inf).min(axis=1)
    out = pd.DataFrame({
        '样本': name, '代码': df['代码'].values, '买入日': df['买入日'].values,
        '买入价': W['ent'],
        '第1天收益': ret[:, 0], '第3天收益': ret[:, 2], '第5天收益': ret[:, 4],
        '第10天收益': ret[:, 9], '第15天收益': ret[:, 14], '第20天收益': ret[:, 19],
        '收盘最大浮盈': peak, '收盘最大浮盈日': argday(r20),
        '收盘最大浮亏': trough, '收盘最大浮亏日': argday_min(r20),
        '盘中最大浮盈': hit_hi, '盘中最大浮亏': hit_lo,
        '20日有效收盘数': np.isfinite(r20).sum(axis=1),
    })
    if '不止损收益率' in df.columns:
        out['原记录净收益率'] = df['不止损收益率'].values
        out['原记录净盈亏'] = df['不止损净盈亏'].values
    elif '收益率' in df.columns:
        out['原记录毛收益率'] = df['收益率'].values
        out['原记录净收益率'] = df['净盈亏'].values / df['买入金额'].values
        out['原记录净盈亏'] = df['净盈亏'].values
    if '持有天数' in df.columns:
        out['原记录持有天数'] = df['持有天数'].values
    if '退出原因' in df.columns:
        out['原记录退出原因'] = df['退出原因'].values
    elif '不止损退出' in df.columns:
        out['原记录退出原因'] = df['不止损退出'].values
    # 扣费后第20天净收益
    g = out['第20天收益'].values
    sell = 10000 * (1 + g)
    fee = 15.0 + np.maximum(5.0, sell * 0.0015)
    out['第20天净收益'] = (sell - 10000 - fee) / 10000
    if '原记录净收益率' in out.columns:
        out['对账差异'] = out['第20天净收益'] - out['原记录净收益率']
    return out


SUMMARY = pd.concat([trade_summary(d, W[n], n) for n, d in samples], ignore_index=True)

# 对账
print('\n=== 第20天净收益 vs 原记录 ===')
for n in ['全信号', '70/15子池']:
    z = SUMMARY[SUMMARY['样本'] == n].dropna(subset=['原记录净收益率', '第20天净收益'])
    diff = (z['第20天净收益'] - z['原记录净收益率']).abs()
    print(n, 'n', len(z), 'mean diff %.6f' % diff.mean(), 'p99 %.6f' % diff.quantile(.99),
          'max %.6f' % diff.max(), '>1pct', int((diff > .01).sum()))
    big = z[diff > .01]
    if len(big):
        print(big.nlargest(5, '对账差异')[['代码', '买入日', '原记录持有天数', '原记录退出原因',
                                            '第20天净收益', '原记录净收益率', '对账差异']]
              .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
z = SUMMARY[SUMMARY['样本'] == '主系统150'].dropna(subset=['原记录净收益率'])
# 主系统有4笔止损，20日路径是反事实
print('主系统150 20日净收益均值 %.4f vs 原记录净均值 %.4f (含止损)' %
      (z['第20天净收益'].mean(), z['原记录净收益率'].mean()))

# ---------------- 输出 ----------------
for obj, fn in [(PATH, '路径统计.csv'), (COND, '条件分布.csv'), (CTHR, '条件阈值概率.csv'), (GROUP, '赢家输家.csv'),
                (GPATH, '赢家输家_路径.csv'), (TAIL, '极值触达.csv'),
                (EXIT, '固定天数退出.csv'), (CONT, '续持条件.csv'),
                (YEAR, '分年_关键天数.csv'), (SUMMARY, '逐笔路径摘要.csv')]:
    obj.to_csv(os.path.join(DIR, fn), index=False, encoding='utf-8-sig', float_format='%.6f')

print('\n=== 全体：关键天数 ===')
show = PATH[PATH['持有天数'].isin([1, 2, 3, 5, 10, 15, 20])]
print(show[['样本', '持有天数', '样本数', '上涨概率', '上涨平均涨幅', '下跌概率', '下跌平均跌幅',
            '平均收益', '中位收益']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 固定持有天数退出（全信号） ===')
print(EXIT[(EXIT['样本'] == '全信号') & EXIT['持有天数'].isin([1, 3, 5, 10, 15, 20, 30, 40, 60])]
      [['持有天数', '毛平均', '毛中位', '毛胜率', '毛盈亏比', '净平均', '净胜率']]
      .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 赢家输家 ===')
print(GROUP.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 极值 ===')
print(TAIL.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 续持 ===')
print(CONT.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\n=== 分年 ===')
print(YEAR.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
