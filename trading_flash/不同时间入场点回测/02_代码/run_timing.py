# -*- coding: utf-8 -*-
"""
运行全部入场时点回测，输出组合级、信号级、信号集合差异、执行差异。

用法：
    cd "/Users/admin/WorkSpace/trading/trading_flash/不同时间入场点回测"
    python3 02_代码/run_timing.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, '02_代码')
import timing_lib as T

OUT = '03_结果'
os.makedirs(OUT, exist_ok=True)

START = '2017-01-01'
END = '2026-10-23'
COST = 0.00175
SLOTS = 2
HOLD = 15
TP = 0.20

# ---------------------------------------------------------------- cases
CASES = []
CASES.append(dict(case_id='C00', mode='收盘', nth=0, fill='close',
                  label='同日收盘（前视参照）', kind='参照'))
for n in range(1, 16):
    CASES.append(dict(case_id=f'O{n:02d}', mode='开盘', nth=n, fill='open',
                      label=f'信号后第{n}个交易日开盘', kind='主'))
for n in range(1, 16):
    CASES.append(dict(case_id=f'C{n:02d}', mode='收盘', nth=n, fill='close',
                      label=f'信号后第{n}个交易日收盘', kind='主'))
for n in range(1, 11):
    CASES.append(dict(case_id=f'L{n:02d}', mode='最低', nth=n, fill='low',
                      label=f'第{n}个交易日最低价（乐观边界）', kind='边界'))
for n in range(1, 11):
    CASES.append(dict(case_id=f'H{n:02d}', mode='最高', nth=n, fill='high',
                      label=f'第{n}个交易日最高价（悲观边界）', kind='边界'))

MAIN_IDS = [c['case_id'] for c in CASES if c['kind'] in ('主', '参照')]
BOUND_IDS = [c['case_id'] for c in CASES if c['kind'] == '边界']
BASE_ID = 'O03'

print('加载面板 ...', flush=True)
t0 = time.time()
D = T.load()
T.add_signal(D)
B = D['base']
print(f'加载完成 {time.time()-t0:.1f}s, 行数 {len(B):,}, 日期 {D["dates"][0]} .. {D["dates"][-1]}', flush=True)

# 信号表（全期，含排名）
sig = B[B['ev_custom'] & (B.date >= pd.Timestamp(START)) & (B.date <= pd.Timestamp(END))].copy()
sig['rk'] = sig.groupby('date')['drawdown250'].rank(ascending=True, method='first')
sig = sig.sort_values(['date', 'rk']).reset_index(drop=True)
sig['j'] = sig['d'].astype(int)
sig['_key'] = list(zip(sig.date, sig.thscode))
print(f'信号事件 {len(sig):,} 条，信号日 {sig.date.nunique()} 天，标的 {sig.thscode.nunique()} 只', flush=True)

# ---------------------------------------------------------------- portfolio
rows, annual_rows, trade_rows, eq_wide, skip_rows = [], [], [], [], []
eq_store = {}
tr_store = {}
for k, c in enumerate(CASES, 1):
    print(f"[{k}/{len(CASES)}] 组合 {c['case_id']} {c['label']}", flush=True)
    eq, tr, sk = T.simulate_timing(
        D, ev='custom', nth=c['nth'], fill=c['fill'], hold=HOLD,
        slots=SLOTS, score='drawdown250', tp=TP, cost=COST,
        start=START, end=END)
    if eq is None:
        print('  无结果', flush=True)
        continue
    m = T.metrics(eq, tr)
    ann = T.annual(eq)
    eq_store[c['case_id']] = eq
    tr_store[c['case_id']] = tr

    row = dict(case_id=c['case_id'], 类型=c['kind'], 入场方式=c['mode'],
               第N个交易日=c['nth'], 说明=c['label'], **m)
    row['首笔入场'] = tr.entry_date.min().date() if len(tr) else None
    row['末笔入场'] = tr.entry_date.max().date() if len(tr) else None
    rows.append(row)

    arow = dict(case_id=c['case_id'], 类型=c['kind'], 入场方式=c['mode'],
                第N个交易日=c['nth'], 说明=c['label'])
    for d, v in ann.items():
        y = int(pd.Timestamp(d).year)
        arow[f'{y}'] = float(v)
    annual_rows.append(arow)

    for reason, cnt in sk.items():
        skip_rows.append(dict(case_id=c['case_id'], 入场方式=c['mode'],
                              第N个交易日=c['nth'], 跳过原因=reason, 次数=cnt))

    if len(tr):
        t = tr.copy()
        t['case_id'] = c['case_id']
        t['类型'] = c['kind']
        t['入场方式'] = c['mode']
        t['第N个交易日'] = c['nth']
        t['说明'] = c['label']
        trade_rows.append(t)

summary = pd.DataFrame(rows)
# 相对基准
base_row = summary[summary.case_id == BASE_ID].iloc[0]
summary['基准年化'] = base_row['cagr']
summary['年化差_vs基准'] = summary['cagr'] - base_row['cagr']
summary['基准回撤'] = base_row['dd']
summary['回撤差_vs基准'] = summary['dd'] - base_row['dd']
summary['期末权益_元'] = summary['final'] * 100000
summary['总利润_元'] = summary['profit_100k']
summary['是否基准'] = (summary.case_id == BASE_ID).astype(int)

# 中文列顺序
col_order = ['case_id', '类型', '入场方式', '第N个交易日', '说明',
             'cagr', 'dd', 'calmar', 'sharpe', 'final', '期末权益_元',
             '总利润_元', 'total_ret', 'n', 'win', 'avg', 'median',
             'best', 'worst', 'pf', 'avg_win', 'avg_loss',
             'n_target', 'n_time', 'n_eod', 'expo',
             '基准年化', '年化差_vs基准', '基准回撤', '回撤差_vs基准',
             '是否基准', '首笔入场', '末笔入场']
summary = summary[col_order].sort_values(['类型', '入场方式', '第N个交易日'],
                                         key=lambda x: x.map({'参照': 0, '主': 1, '边界': 2}) if x.name == '类型' else x)
summary.to_csv(f'{OUT}/组合汇总_全部时点.csv', index=False, encoding='utf-8-sig')

annual = pd.DataFrame(annual_rows)
year_cols = [str(y) for y in range(2017, 2027)]
for y in year_cols:
    if y not in annual.columns:
        annual[y] = np.nan
annual = annual[['case_id', '类型', '入场方式', '第N个交易日', '说明'] + year_cols]
annual.to_csv(f'{OUT}/年度收益_全部时点.csv', index=False, encoding='utf-8-sig')

skips = pd.DataFrame(skip_rows)
if not skips.empty:
    skips.to_csv(f'{OUT}/跳过统计_全部时点.csv', index=False, encoding='utf-8-sig')

trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
if not trades.empty:
    trades = trades.rename(columns={
        'signal_date': '信号日', 'entry_date': '入场日', 'exit_date': '退出日',
        'thscode': '代码', 'rk': '信号排名', 'drawdown250': '250日回撤',
        'rsi14': 'RSI14', 'close_location': '收盘位置',
        'return_250d': '250日收益', 'amount20': '20日均额',
        'market_return20': '市场20日', 'entry_px': '入场价_后复权',
        'exit_reason': '退出原因', 'ret': '单笔收益',
        'days': '持有交易日', 'value': '期末市值',
        'entry_i': '入场序号', 's': '股票序号', 'fill': '价格口径',
        'nth': '信号后第N日'
    })
    trades['退出价_后复权'] = trades['exit_fill'].fillna(trades['last_px'])
    trades['20日均额_亿'] = trades['20日均额'] / 1e8
    trades['250日回撤'] = trades['250日回撤'] * 100
    trades['250日收益'] = trades['250日收益'] * 100
    trades['单笔收益'] = trades['单笔收益'] * 100
    trades['分配额_初始本金%'] = trades['alloc'] * 100
    trades['仓位占当时权益%'] = trades['alloc_frac'] * 100
    trades['盈亏贡献_初始本金%'] = trades['alloc'] * trades['单笔收益']
    trades['市场20日'] = trades['市场20日'] * 100
    tcols = ['case_id', '类型', '入场方式', '第N个交易日', '说明',
             '信号日', '入场日', '退出日', '代码', '信号排名',
             '250日回撤', '250日收益', 'RSI14', '收盘位置', '20日均额_亿',
             '入场价_后复权', '退出价_后复权', '退出原因', '单笔收益',
             '分配额_初始本金%', '仓位占当时权益%', '盈亏贡献_初始本金%',
             '持有交易日', '市场20日']
    for c in tcols:
        if c not in trades.columns:
            trades[c] = np.nan
    trades = trades[tcols].sort_values(['case_id', '入场日', '代码'])
    trades.to_csv(f'{OUT}/逐笔_组合_全部时点.csv', index=False, encoding='utf-8-sig')

# 权益曲线宽表
if eq_store:
    eqw = None
    for cid, eq in eq_store.items():
        s = eq.set_index('date')[['equity']].rename(columns={'equity': cid})
        eqw = s if eqw is None else eqw.join(s, how='outer')
    eqw = eqw.sort_index().reset_index()
    eqw.to_csv(f'{OUT}/权益曲线_全部时点.csv', index=False, encoding='utf-8-sig')

print('组合完成', flush=True)

# ---------------------------------------------------------------- signal level
print('信号级独立收益 ...', flush=True)
sig_arrays = list(zip(sig['s'].astype(int), sig['j'].astype(int),
                      sig['rk'].astype(float), sig['thscode'].tolist(),
                      sig['date'].tolist()))
base_keys = set(zip(tr_store[BASE_ID]['signal_date'], tr_store[BASE_ID]['thscode']))
base_ret_map = {(r.signal_date, r.thscode): r.ret for r in tr_store[BASE_ID].itertuples()}
base_entries = {(r.signal_date, r.thscode): r for r in tr_store[BASE_ID].itertuples()}
base_sig = sig[sig['_key'].isin(base_keys)].copy()

sig_rows, sig_detail_rows, base_rows = [], [], []
for k, c in enumerate(CASES, 1):
    if c['kind'] == '边界':
        continue
    print(f"  信号级 [{k}/{len(CASES)}] {c['case_id']}", flush=True)
    recs = []
    for s, j, rk, code, sigdate in sig_arrays:
        res = T.standalone_trade(D, s, j, c['nth'], c['fill'],
                                 hold=HOLD, tp=TP, cost=COST)
        if res['status'] != 'ok':
            recs.append(dict(code=code, signal_date=sigdate, rk=rk,
                             status=res['status'], ret=np.nan,
                             pre_ret=np.nan, entry_date=None,
                             exit_date=None, exit_reason=None))
        else:
            recs.append(dict(code=code, signal_date=sigdate, rk=rk,
                             status='ok', ret=res['ret'],
                             pre_ret=res['pre_ret'],
                             entry_date=res['entry_date'],
                             exit_date=res['exit_date'],
                             exit_reason=res['exit_reason']))
    rec = pd.DataFrame(recs)
    rec['_key'] = list(zip(rec.signal_date, rec.code))
    ok = rec[rec.status == 'ok'].copy()
    top2 = ok[ok.rk <= 2].copy()
    top5 = ok[ok.rk <= 5].copy()
    rank1 = ok[ok.rk == 1].copy()
    rank2 = ok[ok.rk == 2].copy()

    def agg(x):
        if len(x) == 0:
            return dict(n=0, avg=np.nan, median=np.nan, win=np.nan, pf=np.nan,
                        avg_win=np.nan, avg_loss=np.nan, pre_avg=np.nan,
                        pre_med=np.nan, chase=np.nan)
        pos = x.loc[x.ret > 0, 'ret'].sum()
        neg = x.loc[x.ret < 0, 'ret'].sum()
        return dict(
            n=int(len(x)), avg=float(x.ret.mean()), median=float(x.ret.median()),
            win=float((x.ret > 0).mean()),
            pf=float(pos / abs(neg)) if neg < 0 else np.inf,
            avg_win=float(x.loc[x.ret > 0, 'ret'].mean()) if (x.ret > 0).any() else np.nan,
            avg_loss=float(x.loc[x.ret < 0, 'ret'].mean()) if (x.ret < 0).any() else np.nan,
            pre_avg=float(x.pre_ret.mean()), pre_med=float(x.pre_ret.median()),
            chase=float((x.pre_ret > 0.001).mean()))

    a_all, a2, a5 = agg(ok), agg(top2), agg(top5)
    a1, a2r = agg(rank1), agg(rank2)

    # 基准 99 笔同信号
    br = rec[(rec.status == 'ok') & rec._key.isin(base_keys)]
    ab = agg(br)

    row = dict(case_id=c['case_id'], 类型=c['kind'], 入场方式=c['mode'],
               第N个交易日=c['nth'], 说明=c['label'],
               信号总数=len(rec),
               全部_有效=a_all['n'], 全部_平均=a_all['avg'], 全部_中位=a_all['median'],
               全部_胜率=a_all['win'], 全部_利润因子=a_all['pf'],
               全部_平均盈利=a_all['avg_win'], 全部_平均亏损=a_all['avg_loss'],
               全部_入场前涨幅均值=a_all['pre_avg'], 全部_入场前涨幅中位=a_all['pre_med'],
               全部_追高比例=a_all['chase'],
               前二_有效=a2['n'], 前二_平均=a2['avg'], 前二_中位=a2['median'],
               前二_胜率=a2['win'], 前二_利润因子=a2['pf'],
               前二_入场前涨幅均值=a2['pre_avg'], 前二_追高比例=a2['chase'],
               前五_有效=a5['n'], 前五_平均=a5['avg'], 前五_胜率=a5['win'],
               排名1_有效=a1['n'], 排名1_平均=a1['avg'], 排名1_胜率=a1['win'],
               排名2_有效=a2r['n'], 排名2_平均=a2r['avg'], 排名2_胜率=a2r['win'],
               基准99_有效=ab['n'], 基准99_平均=ab['avg'], 基准99_中位=ab['median'],
               基准99_胜率=ab['win'], 基准99_利润因子=ab['pf'],
               基准99_入场前涨幅均值=ab['pre_avg'])
    sig_rows.append(row)

    # 每日前二明细（主/参照），边界不落明细
    d2 = rec[rec.rk <= 2].copy()
    d2['case_id'] = c['case_id']
    d2['入场方式'] = c['mode']
    d2['第N个交易日'] = c['nth']
    d2['说明'] = c['label']
    sig_detail_rows.append(d2)

    # 基准 99 笔明细：每个 case 的独立收益
    rec_map = {(r.signal_date, r.code): r for r in rec.itertuples()}
    for sr in base_sig.itertuples():
        rr = rec_map.get((sr.date, sr.thscode))
        val = rr.ret if (rr is not None and rr.status == 'ok') else np.nan
        st = rr.status if rr is not None else '无'
        base_rows.append(dict(case_id=c['case_id'], 入场方式=c['mode'],
                              第N个交易日=c['nth'],
                              信号日=sr.date, 代码=sr.thscode,
                              信号排名=sr.rk,
                              本档收益=val, 本档状态=st))

sig_sum = pd.DataFrame(sig_rows)
sig_sum['基准99_较基准差'] = sig_sum['基准99_平均'] - sig_sum.loc[
    sig_sum.case_id == BASE_ID, '基准99_平均'].iloc[0]
sig_sum.to_csv(f'{OUT}/信号级汇总_全部信号.csv', index=False, encoding='utf-8-sig')

sig_detail = pd.concat(sig_detail_rows, ignore_index=True) if sig_detail_rows else pd.DataFrame()
if not sig_detail.empty:
    sig_detail = sig_detail.rename(columns={
        'code': '代码', 'signal_date': '信号日', 'rk': '信号排名',
        'status': '状态', 'ret': '单笔收益', 'pre_ret': '入场前涨幅',
        'entry_date': '入场日', 'exit_date': '退出日', 'exit_reason': '退出原因'
    })
    sig_detail['单笔收益'] = sig_detail['单笔收益'] * 100
    sig_detail['入场前涨幅'] = sig_detail['入场前涨幅'] * 100
    sig_detail = sig_detail[['case_id', '入场方式', '第N个交易日', '说明',
                             '信号日', '代码', '信号排名', '状态',
                             '入场日', '退出日', '单笔收益', '入场前涨幅', '退出原因']]
    sig_detail.sort_values(['case_id', '信号日', '信号排名']).to_csv(
        f'{OUT}/信号级明细_每日前二.csv', index=False, encoding='utf-8-sig')

# 基准 99 笔明细宽表
brdf = pd.DataFrame(base_rows)
if not brdf.empty:
    keycols = ['信号日', '代码', '信号排名']
    base_meta = sig[['date', 'thscode', 'rk', 'drawdown250', 'rsi14',
                     'close_location', 'return_250d', '_key']].rename(
        columns={'date': '信号日', 'thscode': '代码', 'rk': '信号排名',
                 'drawdown250': '250日回撤', 'rsi14': 'RSI14',
                 'close_location': '收盘位置', 'return_250d': '250日收益'})
    base_meta = base_meta[base_meta['_key'].isin(base_keys)]
    base_meta['250日回撤'] = base_meta['250日回撤'] * 100
    base_meta['250日收益'] = base_meta['250日收益'] * 100
    wide = brdf.pivot_table(index=keycols, columns='case_id', values='本档收益', aggfunc='first').reset_index()
    wide = wide.merge(base_meta, on=keycols, how='left')
    # 基准组合实际收益
    btr = tr_store[BASE_ID][['signal_date', 'thscode', 'entry_date', 'exit_date', 'ret', 'exit_reason']].rename(
        columns={'signal_date': '信号日', 'thscode': '代码', 'entry_date': '基准入场日',
                 'exit_date': '基准退出日', 'ret': '基准组合收益', 'exit_reason': '基准退出原因'})
    wide = wide.merge(btr, on=['信号日', '代码'], how='left')
    wide['基准组合收益'] = wide['基准组合收益'] * 100
    for c in MAIN_IDS:
        if c in wide.columns:
            wide[c] = wide[c] * 100
    cols = ['信号日', '代码', '信号排名', '250日回撤', '250日收益', 'RSI14', '收盘位置',
            '基准入场日', '基准退出日', '基准退出原因', '基准组合收益'] + MAIN_IDS
    for c in cols:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[cols].sort_values(['信号日', '信号排名'])
    wide.to_csv(f'{OUT}/基准99笔_不同时点收益明细.csv', index=False, encoding='utf-8-sig')

print('信号级完成', flush=True)

# ---------------------------------------------------------------- signal set overlap
print('信号集合重叠 ...', flush=True)
date_list = list(D['dates'])
date_pos = {pd.Timestamp(d): i for i, d in enumerate(date_list)}
sig_sets = {}
sig_top2 = {}
for d, g in sig.groupby('date'):
    sig_sets[pd.Timestamp(d)] = set(g.thscode)
    sig_top2[pd.Timestamp(d)] = list(g.sort_values('rk').head(2).thscode)

ov_rows = []
for n in range(1, 16):
    inter_n, union_n, jac, top2_overlap, top2_survive, days = [], [], [], [], [], 0
    top2_full = []
    for d in sig_sets:
        i = date_pos.get(d)
        if i is None or i + n >= len(date_list):
            continue
        dn = pd.Timestamp(date_list[i + n])
        A, Bs = sig_sets[d], sig_sets.get(dn, set())
        inter = A & Bs
        union = A | Bs
        inter_n.append(len(inter))
        union_n.append(len(union))
        jac.append(len(inter) / len(union) if union else np.nan)
        ta, tb = sig_top2[d], sig_top2.get(dn, [])
        top2_overlap.append(len(set(ta) & set(tb)))
        top2_full.append(len(set(ta) & set(tb)) == len(ta))
        top2_survive.append(len(set(ta) & Bs) / len(ta) if ta else np.nan)
        days += 1
    ov_rows.append(dict(第N个交易日=n, 信号日数=days,
                        平均交集=np.mean(inter_n) if inter_n else np.nan,
                        平均并集=np.mean(union_n) if union_n else np.nan,
                        平均Jaccard=np.nanmean(jac) if jac else np.nan,
                        前二平均重合只数=np.mean(top2_overlap) if top2_overlap else np.nan,
                        前二完全重合比例=np.mean(top2_full) if top2_full else np.nan,
                        前二存活率=np.nanmean(top2_survive) if top2_survive else np.nan))
ov = pd.DataFrame(ov_rows)
ov.to_csv(f'{OUT}/信号集合变化_按档位.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- execution difference
print('执行差异 ...', flush=True)
base_tr = tr_store[BASE_ID]
diff_rows = []
for c in CASES:
    cid = c['case_id']
    if cid not in tr_store:
        continue
    tr = tr_store[cid]
    k_case = set(zip(tr.signal_date, tr.thscode))
    k_base = set(zip(base_tr.signal_date, base_tr.thscode))
    common = k_case & k_base
    only_case = k_case - k_base
    only_base = k_base - k_case
    cm = tr[tr.apply(lambda r: (r.signal_date, r.thscode) in common, axis=1)] if len(tr) else tr
    cb = base_tr[base_tr.apply(lambda r: (r.signal_date, r.thscode) in common, axis=1)]
    co = tr[tr.apply(lambda r: (r.signal_date, r.thscode) in only_case, axis=1)] if len(tr) else tr
    bo = base_tr[base_tr.apply(lambda r: (r.signal_date, r.thscode) in only_base, axis=1)]
    diff_rows.append(dict(
        case_id=cid, 类型=c['kind'], 入场方式=c['mode'], 第N个交易日=c['nth'],
        交易数=len(k_case), 共同信号数=len(common),
        本档独有数=len(only_case), 基准独有数=len(only_base),
        交易集合Jaccard=len(common) / len(k_case | k_base) if (k_case | k_base) else np.nan,
        共同信号_本档平均=cm.ret.mean() if len(cm) else np.nan,
        共同信号_基准平均=cb.ret.mean() if len(cb) else np.nan,
        共同信号_差=(cm.ret.mean() - cb.ret.mean()) if len(cm) and len(cb) else np.nan,
        本档独有_平均=co.ret.mean() if len(co) else np.nan,
        基准独有_平均=bo.ret.mean() if len(bo) else np.nan,
        共同信号_本档贡献=(cm.alloc * cm.ret).sum() if len(cm) else np.nan,
        共同信号_基准贡献=(cb.alloc * cb.ret).sum() if len(cb) else np.nan,
        本档独有_贡献=(co.alloc * co.ret).sum() if len(co) else np.nan,
        基准独有_贡献=(bo.alloc * bo.ret).sum() if len(bo) else np.nan,
    ))
diff = pd.DataFrame(diff_rows)
for c in ['共同信号_本档平均', '共同信号_基准平均', '共同信号_差',
          '本档独有_平均', '基准独有_平均', '共同信号_本档贡献',
          '共同信号_基准贡献', '本档独有_贡献', '基准独有_贡献']:
    if c in diff.columns:
        diff[c] = diff[c] * 100
diff.to_csv(f'{OUT}/执行差异_对比基准.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- current signal
cur_date = pd.Timestamp('2026-10-23')
cur = B[(B.date == cur_date) & B.ev_custom].copy()
cur = cur.sort_values('drawdown250').reset_index(drop=True)
cur['rk'] = np.arange(1, len(cur) + 1)
cur_out = cur.head(2)[['thscode', 'close', 'return_250d', 'drawdown250',
                       'rsi14', 'close_location', 'amount20', 'rk']].copy()
cur_out['return_250d'] = cur_out['return_250d'] * 100
cur_out['drawdown250'] = cur_out['drawdown250'] * 100
cur_out['amount20'] = cur_out['amount20'] / 1e8
cur_out = cur_out.rename(columns={'thscode': '代码', 'close': '收盘',
                                  'return_250d': '250日收益', 'drawdown250': '250日回撤',
                                  'rsi14': 'RSI14', 'close_location': '收盘位置',
                                  'amount20': '20日均额_亿', 'rk': '排名'})
cur_out['信号日'] = cur_date.date()
cur_out.to_csv(f'{OUT}/当前信号_2026-10-23.csv', index=False, encoding='utf-8-sig')

# 当前信号在不同档位下的入场日期（数据只到 10-23，价格未知）
fut = pd.bdate_range('2026-10-26', periods=15)
cur_dates = pd.DataFrame([
    dict(第N个交易日=n, 入场日期=fut[n - 1].date(),
         备注='按工作日推算，未核验节假日；数据未覆盖，无法回填收益')
    for n in range(1, 16)
])
cur_dates.to_csv(f'{OUT}/当前信号_入场日期.csv', index=False, encoding='utf-8-sig')

# 便于阅读的主结果摘要
keep = ['C00', 'O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07', 'O08',
        'O09', 'O10', 'C01', 'C02', 'C03', 'C04', 'C05', 'L03', 'H03']
summary[summary.case_id.isin(keep)].to_csv(
    f'{OUT}/主结果_摘要.csv', index=False, encoding='utf-8-sig')

# 当前信号候选数
cur_count = int(len(cur))

# ---------------------------------------------------------------- json
def slim(m):
    keys = ['cagr', 'dd', 'calmar', 'sharpe', 'n', 'win', 'avg',
            'median', 'pf', 'expo', 'final', 'profit_100k',
            'n_target', 'n_time', 'n_eod']
    return {k: (float(m[k]) if isinstance(m[k], (int, float, np.integer, np.floating)) else m[k])
            for k in keys if k in m}

best = summary.sort_values('cagr', ascending=False).iloc[0]
best_main = summary[summary.类型 == '主'].sort_values('cagr', ascending=False).iloc[0]
worst = summary[summary.类型 == '主'].sort_values('cagr').iloc[0]
base_m = summary[summary.case_id == BASE_ID].iloc[0]
main = summary[summary.类型 == '主']
meta = dict(
    data_start=str(pd.Timestamp(D['dates'][0]).date()),
    data_end=str(pd.Timestamp(D['dates'][-1]).date()),
    signal_events=int(len(sig)), signal_days=int(sig.date.nunique()),
    signal_symbols=int(sig.thscode.nunique()),
    signal_events_2026_10_23=cur_count,
    cost_per_side=COST, slots=SLOTS, hold=HOLD, take_profit=TP,
    base=slim(base_m.to_dict()),
    best_all=dict(case_id=best.case_id, label=best.说明, **slim(best.to_dict())),
    best_main=dict(case_id=best_main.case_id, label=best_main.说明, **slim(best_main.to_dict())),
    worst_main=dict(case_id=worst.case_id, label=worst.说明, **slim(worst.to_dict())),
    main_cagr_min=float(main.cagr.min()), main_cagr_median=float(main.cagr.median()),
    main_cagr_max=float(main.cagr.max()),
    main_dd_worst=float(main.dd.min()),
    n_main=int(len(main)),
)
json.dump(meta, open(f'{OUT}/指标.json', 'w'), ensure_ascii=False, indent=1)

print('\n=== 主结果（开盘/收盘/参照）===')
show = ['case_id', '入场方式', '第N个交易日', 'cagr', 'dd', 'n', 'win',
        'avg', 'pf', 'final', '年化差_vs基准']
print(summary[summary.类型 != '边界'][show].to_string(index=False))
print('\n=== 边界（最高/最低）===')
print(summary[summary.类型 == '边界'][show].to_string(index=False))
print('\n=== 信号集合变化 ===')
print(ov.to_string(index=False))
print('\n信号事件', len(sig), '信号日', sig.date.nunique(), '当前信号', cur_count)
print('完成')
