# -*- coding: utf-8 -*-
"""
不同信号日起点：交易员 A 在 m 日看到信号，交易员 B 在 m+1 日看到新信号。

口径：
- 信号条件、排序、2 仓、15 日持有、+20% 止盈、0.175%/边、一字涨停/停牌不买，均与母策略一致；
- 入场固定为信号后第 3 个交易日开盘（N=3），本分析只改变“第一个信号日”，不改变入场延迟；
- 每个信号日作为一个起点：当天用 10 万本金开始，买当天前二；之后按同一规则运行到 2026-10-23；
- 另做“首笔批次”独立收益（两只各 50%，未计后续加仓），用于剥离后续占位影响；
- 固定持有期 1/3/5 年按起点后 252/756/1260 个交易日收盘市值计算，未强制平仓。

输出均在 06_不同信号日起点/03_结果/。
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / '02_代码'))
import timing_lib as tl  # noqa: E402

PANEL = ROOT / '01_数据' / 'research_panel.parquet'
RES = OUT / '03_结果'
RES.mkdir(parents=True, exist_ok=True)

NTH = 3
HOLD = 15
SLOTS = 2
TP = 0.20
COST = 0.00175
END = '2026-10-23'
HORIZONS = {'1年': 252, '3年': 756, '5年': 1260}

D = tl.load(str(PANEL))
tl.add_signal(D)
base = D['base']
ev_all = base[base['ev_custom']].copy()
ev_all['rk'] = ev_all.groupby('date')['drawdown250'].rank(
    ascending=True, method='first')
top2 = ev_all[ev_all['rk'] <= SLOTS].sort_values(['date', 'rk']).copy()
signal_days = pd.DatetimeIndex(sorted(ev_all['date'].unique()))
n_by_day = ev_all.groupby('date').size().to_dict()
last_date = pd.Timestamp(D['dates'][-1])


def _scheduled_entry(j):
    """信号后第 NTH 个交易日；超出数据末端时按工作日推算。"""
    k = j + NTH
    if k < len(D['dates']):
        return D['dates'][k]
    extra = pd.bdate_range(
        pd.Timestamp(D['dates'][-1]) + pd.Timedelta(days=1),
        periods=k - len(D['dates']) + 5)
    return extra[k - len(D['dates'])]
print('信号日', len(signal_days), '信号数', len(ev_all), '前二行', len(top2),
      '数据截止', last_date.date(), flush=True)

# ---------------------------------------------------------------- 逐日首笔
batch_rows = []
for d, g in top2.groupby('date'):
    g = g.sort_values('rk')
    n_sig = int(n_by_day[d])
    j = int(g.iloc[0]['d'])
    recs = []
    for r in g.itertuples():
        st = tl.standalone_trade(D, int(r.s), j, NTH, fill='open',
                                 hold=HOLD, tp=TP, cost=COST)
        complete = False
        status = st['status']
        if status == 'ok':
            entry_i, exit_i = int(st['entry_i']), int(st['exit_i'])
            if st['exit_reason'] == 'target':
                complete = True
            elif entry_i + HOLD - 1 <= len(D['dates']) - 1:
                complete = True
            else:
                status = '持有未满'
        recs.append(dict(
            signal_date=d, n_signals=n_sig, rank=int(r.rk),
            code=r.thscode, dd250=float(r.drawdown250),
            rsi14=float(r.rsi14), close_loc=float(r.close_location),
            status=status, complete=complete,
            entry_date=st.get('entry_date', pd.NaT),
            exit_date=st.get('exit_date', pd.NaT),
            exit_reason=st.get('exit_reason', ''),
            entry_px=st.get('entry_px', np.nan),
            exit_px=st.get('exit_px', np.nan),
            ret=st.get('ret', np.nan),
            pre_ret=st.get('pre_ret', np.nan),
        ))
    executed = [x for x in recs if x['status'] in ('ok', '持有未满')]
    n_complete = sum(1 for x in executed if x['complete'])
    batch_complete = (n_complete == len(executed) and len(executed) > 0)
    batch_float = (sum(0.5 * x['ret'] for x in executed)
                   if executed else np.nan)  # 两只各半仓；仅一只则半仓
    batch_port = batch_float if batch_complete else np.nan
    batch_avg = (float(np.mean([x['ret'] for x in executed]))
                 if batch_complete else np.nan)
    batch_rows.append(dict(
        signal_date=d, n_signals=n_sig,
        first_code=recs[0]['code'] if len(recs) > 0 else '',
        first_dd250=recs[0]['dd250'] if len(recs) > 0 else np.nan,
        second_code=recs[1]['code'] if len(recs) > 1 else '',
        second_dd250=recs[1]['dd250'] if len(recs) > 1 else np.nan,
        scheduled_entry=_scheduled_entry(j),
        first_entry=min([x['entry_date'] for x in executed], default=pd.NaT),
        last_exit=max([x['exit_date'] for x in executed], default=pd.NaT),
        n_buy=len(executed), n_complete=n_complete,
        batch_complete=batch_complete,
        batch_ret=batch_port, batch_float=batch_float, batch_avg=batch_avg,
        codes='、'.join([x['code'] for x in executed]),
        rets='、'.join([f"{x['ret']*100:.2f}%" for x in executed]),
        status='、'.join(sorted(set(x['status'] for x in recs))),
    ))
batch = pd.DataFrame(batch_rows)
batch.to_csv(RES / '逐日首笔.csv', index=False, encoding='utf-8-sig')
# 逐只明细
det_rows = []
for d, g in top2.groupby('date'):
    g = g.sort_values('rk')
    j = int(g.iloc[0]['d'])
    n_sig = int(n_by_day[d])
    for r in g.itertuples():
        st = tl.standalone_trade(D, int(r.s), j, NTH, fill='open',
                                 hold=HOLD, tp=TP, cost=COST)
        status = st['status']
        complete = False
        if status == 'ok':
            entry_i, exit_i = int(st['entry_i']), int(st['exit_i'])
            if st['exit_reason'] == 'target':
                complete = True
            elif entry_i + HOLD - 1 <= len(D['dates']) - 1:
                complete = True
            else:
                status = '持有未满'
        det_rows.append(dict(
            signal_date=d, n_signals=n_sig, rank=int(r.rk), code=r.thscode,
            dd250=float(r.drawdown250), rsi14=float(r.rsi14),
            close_loc=float(r.close_location), status=status,
            complete=complete,
            entry_date=st.get('entry_date', pd.NaT),
            exit_date=st.get('exit_date', pd.NaT),
            exit_reason=st.get('exit_reason', ''),
            ret=st.get('ret', np.nan), pre_ret=st.get('pre_ret', np.nan),
        ))
det = pd.DataFrame(det_rows)
det.to_csv(RES / '逐日首笔_逐只.csv', index=False, encoding='utf-8-sig')
print('逐日首笔完成', len(batch), '天', len(det), '只', flush=True)

# ---------------------------------------------------------------- 起点 cohort
rows, trade_sets, curves, all_trades = [], {}, {}, []
t0 = time.time()
for k, d in enumerate(signal_days):
    eq, tr, sk = tl.simulate_timing(
        D, start=pd.Timestamp(d), end=END, nth=NTH, fill='open',
        hold=HOLD, slots=SLOTS, tp=TP, cost=COST)
    m = tl.metrics(eq, tr) if eq is not None else {}
    b = batch[batch.signal_date == d].iloc[0]
    rec = dict(
        signal_date=d, n_signals=int(n_by_day[d]),
        first_code=b['first_code'], first_dd250=b['first_dd250'],
        second_code=b['second_code'], second_dd250=b['second_dd250'],
        scheduled_entry=b['scheduled_entry'],
        first_entry=b['first_entry'], last_exit=b['last_exit'],
        n_buy=int(b['n_buy']), batch_complete=bool(b['batch_complete']),
        batch_ret=float(b['batch_ret']) if pd.notna(b['batch_ret']) else np.nan,
        batch_float=float(b['batch_float']) if pd.notna(b['batch_float']) else np.nan,
        batch_avg=float(b['batch_avg']) if pd.notna(b['batch_avg']) else np.nan,
        final=1.0, total_ret=0.0, cagr=np.nan, dd=np.nan,
        n=0, win=np.nan, avg=np.nan, median=np.nan, best=np.nan,
        worst=np.nan, pf=np.nan, expo=np.nan,
        n_target=0, n_time=0, n_eod=0, profit_100k=0.0,
    )
    if m:
        rec.update(
            final=m['final'], total_ret=m['total_ret'], cagr=m['cagr'],
            dd=m['dd'], n=m['n'], win=m['win'], avg=m['avg'],
            median=m['median'], best=m['best'], worst=m['worst'],
            pf=m['pf'], expo=m['expo'], n_target=m['n_target'],
            n_time=m['n_time'], n_eod=m['n_eod'],
            profit_100k=m['profit_100k'],
        )
    elif eq is not None and len(eq) > 0:
        # 样本不足10个交易日：metrics 会返回空；仍要记录期末市值与未平仓
        e = eq['equity']
        rec['final'] = float(e.iloc[-1])
        rec['total_ret'] = float(e.iloc[-1] - 1.0)
        rec['profit_100k'] = float((e.iloc[-1] - 1.0) * 100000)
        rec['expo'] = float((eq['npos'] > 0).mean())
        rec['dd'] = (float((e / e.cummax() - 1.0).min())
                     if len(e) > 1 else 0.0)
        if tr is not None and len(tr):
            rec['n'] = int(len(tr))
            rec['win'] = float((tr['ret'] > 0).mean())
            rec['avg'] = float(tr['ret'].mean())
            rec['median'] = float(tr['ret'].median())
            rec['best'] = float(tr['ret'].max())
            rec['worst'] = float(tr['ret'].min())
            pos_sum = tr.loc[tr['ret'] > 0, 'ret'].sum()
            neg_sum = tr.loc[tr['ret'] < 0, 'ret'].sum()
            rec['pf'] = (float(pos_sum / abs(neg_sum))
                         if neg_sum < 0 else np.inf)
            rec['n_target'] = int((tr['exit_reason'] == 'target').sum())
            rec['n_time'] = int((tr['exit_reason'] == 'time').sum())
            rec['n_eod'] = int((tr['exit_reason'] == 'eod').sum())
    # 固定持有期（按起点后交易日数；未强制平仓）
    if eq is not None:
        e = eq['equity'].to_numpy()
        for hname, h in HORIZONS.items():
            if len(e) > h:
                er = e[h] / e[0] - 1.0
                sub = e[:h + 1]
                hdd = float((sub / np.maximum.accumulate(sub) - 1.0).min())
                rec[f'ret_{hname}'] = float(er)
                rec[f'dd_{hname}'] = hdd
            else:
                rec[f'ret_{hname}'] = np.nan
                rec[f'dd_{hname}'] = np.nan
        # 给图表留少量代表曲线
        if k in (0, len(signal_days) - 1) or d.strftime('%Y-%m-%d') in (
                '2018-10-18', '2020-02-03', '2022-04-26', '2024-02-05',
                '2026-10-23'):
            curves[d] = eq[['date', 'equity', 'npos']].copy()
    else:
        for hname in HORIZONS:
            rec[f'ret_{hname}'] = np.nan
            rec[f'dd_{hname}'] = np.nan
    rows.append(rec)
    if tr is not None and len(tr):
        trade_sets[d] = set(zip(tr['signal_date'].astype(str), tr['thscode']))
        all_trades.append(tr.assign(起点=d))
    else:
        trade_sets[d] = set()
    if (k + 1) % 25 == 0:
        print(f'  cohort {k+1}/{len(signal_days)}  {time.time()-t0:.0f}s', flush=True)

if all_trades:
    all_tr = pd.concat(all_trades, ignore_index=True)
    all_tr.to_csv(RES / '全部起点_逐笔.csv', index=False,
                  encoding='utf-8-sig')
    print('全部起点逐笔', len(all_tr), flush=True)
coh = pd.DataFrame(rows)
coh['样本'] = np.select(
    [coh['ret_5年'].notna(), coh['ret_3年'].notna(), coh['ret_1年'].notna()],
    ['满5年', '满3年', '满1年'], default='不足1年')
coh.to_csv(RES / '起点汇总.csv', index=False, encoding='utf-8-sig')
for d, e in curves.items():
    e.assign(起点=d).to_csv(
        RES / f'净值_起点_{pd.Timestamp(d).date()}.csv', index=False,
        encoding='utf-8-sig')
print('cohort 完成', len(coh), f'{time.time()-t0:.0f}s', flush=True)

# ---------------------------------------------------------------- 起点分布
def dist(s):
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return dict(n=0)
    return dict(
        n=int(len(s)), mean=float(s.mean()),
        p5=float(s.quantile(.05)), p25=float(s.quantile(.25)),
        median=float(s.median()), p75=float(s.quantile(.75)),
        p95=float(s.quantile(.95)), min=float(s.min()), max=float(s.max()),
        neg=float((s < 0).mean()),
        ge30=float((s >= .30).mean()), ge40=float((s >= .40).mean()),
    )

samples = {
    '全部信号日': coh,
    '信号≥2只': coh[coh.n_signals >= 2],
    '满1年': coh[coh['ret_1年'].notna()],
    '满3年': coh[coh['ret_3年'].notna()],
    '满5年': coh[coh['ret_5年'].notna()],
    '满3年且信号≥2只': coh[(coh['ret_3年'].notna()) & (coh.n_signals >= 2)],
    '满5年且信号≥2只': coh[(coh['ret_5年'].notna()) & (coh.n_signals >= 2)],
}
metric_map = {
    '全期年化': 'cagr', '全期总收益': 'total_ret', '全期最大回撤': 'dd',
    '1年收益': 'ret_1年', '1年最大回撤': 'dd_1年',
    '3年收益': 'ret_3年', '3年最大回撤': 'dd_3年',
    '5年收益': 'ret_5年', '5年最大回撤': 'dd_5年',
    '首笔批次收益': 'batch_ret',
}
dist_rows = []
for sname, sdf in samples.items():
    for mname, col in metric_map.items():
        dd = dist(sdf[col])
        if dd['n'] == 0:
            continue
        dist_rows.append(dict(样本=sname, 指标=mname, **dd))
distdf = pd.DataFrame(dist_rows)
distdf.to_csv(RES / '起点分布.csv', index=False, encoding='utf-8-sig')

# 年度
coh['start_year'] = pd.to_datetime(coh.signal_date).dt.year
yr_rows = []
for y, g in coh.groupby('start_year'):
    yr_rows.append({
        '起点年份': int(y), '起点数': int(len(g)),
        '满1年数': int(g['ret_1年'].notna().sum()),
        '满3年数': int(g['ret_3年'].notna().sum()),
        '满5年数': int(g['ret_5年'].notna().sum()),
        '年化中位数': g.cagr.median(), '年化均值': g.cagr.mean(),
        '年化负收益占比': (g.cagr < 0).mean(),
        '全期总收益中位数': g.total_ret.median(),
        '1年收益中位数': g['ret_1年'].median(),
        '3年收益中位数': g['ret_3年'].median(),
        '5年收益中位数': g['ret_5年'].median(),
        '首笔批次收益中位数': g.batch_ret.median(),
    })
yrs = pd.DataFrame(yr_rows)
yrs.to_csv(RES / '起点年度.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 相邻信号日
by_date = coh.set_index('signal_date')
pair_rows = []
for a, b in zip(signal_days[:-1], signal_days[1:]):
    ra, rb = by_date.loc[a], by_date.loc[b]
    set_a = {ra['first_code'], ra['second_code']} - {''}
    set_b = {rb['first_code'], rb['second_code']} - {''}
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jac = inter / union if union else np.nan
    ta, tb = trade_sets[a], trade_sets[b]
    ti = len(ta & tb)
    tu = len(ta | tb)
    tjac = ti / tu if tu else np.nan
    ia = int(np.searchsorted(D['dates'], np.datetime64(a)))
    ib = int(np.searchsorted(D['dates'], np.datetime64(b)))
    pair_rows.append(dict(
        date_a=a, date_b=b, gap_td=int(ib - ia),
        gap_cd=int((b - a).days),
        n_a=int(ra['n_signals']), n_b=int(rb['n_signals']),
        n_buy_a=int(ra['n_buy']), n_buy_b=int(rb['n_buy']),
        both_buy2=(int(ra['n_buy']) == 2 and int(rb['n_buy']) == 2),
        top_a='、'.join(sorted(set_a)), top_b='、'.join(sorted(set_b)),
        overlap=inter, top_jac=jac, same_top=(set_a == set_b),
        batch_a=ra['batch_ret'], batch_b=rb['batch_ret'],
        batch_diff=rb['batch_ret'] - ra['batch_ret'],
        cagr_a=ra['cagr'], cagr_b=rb['cagr'],
        cagr_diff=rb['cagr'] - ra['cagr'],
        y1_a=ra['ret_1年'], y1_b=rb['ret_1年'],
        y1_diff=rb['ret_1年'] - ra['ret_1年'],
        y3_a=ra['ret_3年'], y3_b=rb['ret_3年'],
        y3_diff=rb['ret_3年'] - ra['ret_3年'],
        y5_a=ra['ret_5年'], y5_b=rb['ret_5年'],
        y5_diff=rb['ret_5年'] - ra['ret_5年'],
        trade_overlap=ti, trade_jac=tjac,
        both_ge2=(ra['n_signals'] >= 2 and rb['n_signals'] >= 2),
    ))
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(RES / '相邻信号日.csv', index=False, encoding='utf-8-sig')

# 次日配对汇总
next_day = pairs[pairs.gap_td == 1]
next_day_ge2 = next_day[next_day.both_ge2]

def pair_dist(s):
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return dict(n=0)
    return dict(n=int(len(s)), mean=float(s.mean()),
                median=float(s.median()), p5=float(s.quantile(.05)),
                p95=float(s.quantile(.95)), min=float(s.min()),
                max=float(s.max()), neg=float((s < 0).mean()))

pair_summary = []
next_day_buy2 = next_day[next_day.both_buy2]
for label, sdf in [('全部相邻信号日', pairs), ('次日信号日', next_day),
                   ('次日且两日都≥2只', next_day_ge2),
                   ('次日且两日各买满2只', next_day_buy2)]:
    for mcol, mname in [('batch_diff', '首笔批次差'), ('y1_diff', '1年收益差'),
                        ('cagr_diff', '全期年化差'), ('trade_jac', '交易重合度')]:
        pair_summary.append(dict(样本=label, 指标=mname, **pair_dist(sdf[mcol])))
pd.DataFrame(pair_summary).to_csv(
    RES / '相邻信号日_汇总.csv', index=False, encoding='utf-8-sig')

# 最近案例（最后 10 个信号日）
recent = coh.tail(10).copy()
recent['batch_status'] = np.where(
    recent['first_entry'].isna(), '尚未入场/数据不足',
    np.where(recent['batch_complete'], '已完成', '持有未满'))
recent.to_csv(RES / '最近信号日案例.csv', index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------- 元数据
meta = dict(
    start=pd.Timestamp(D['dates'][0]).date().isoformat(),
    end=pd.Timestamp(D['dates'][-1]).date().isoformat(),
    n_days=int(len(D['dates'])), n_syms=int(len(D['syms'])),
    n_signals=int(len(ev_all)), n_signal_days=int(len(signal_days)),
    n_signal_days_ge2=int((coh.n_signals >= 2).sum()),
    n_pairs=int(len(pairs)), n_next_day=int(len(next_day)),
    n_next_day_ge2=int(len(next_day_ge2)),
    n_next_day_buy2=int(len(next_day_buy2)),
    n_complete_batches=int(batch.batch_complete.sum()),
    n_cohorts_with_trades=int((coh.n > 0).sum()),
    n_cohorts_1y=int(coh['ret_1年'].notna().sum()),
    n_cohorts_3y=int(coh['ret_3年'].notna().sum()),
    n_cohorts_5y=int(coh['ret_5年'].notna().sum()),
    params=dict(nth=NTH, hold=HOLD, slots=SLOTS, tp=TP, cost=COST,
                end=END, score='drawdown250'),
    first_signal=pd.Timestamp(signal_days[0]).date().isoformat(),
    last_signal=pd.Timestamp(signal_days[-1]).date().isoformat(),
    base_first_signal_matches=tl.metrics(
        *tl.simulate_timing(D, start='2017-01-01', end=END,
                            nth=NTH, fill='open', hold=HOLD,
                            slots=SLOTS, tp=TP, cost=COST)[:2]),
)
with open(RES / '指标.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps({k: v for k, v in meta.items() if k != 'base_first_signal_matches'},
                 ensure_ascii=False, indent=2, default=str))
print('done', flush=True)
