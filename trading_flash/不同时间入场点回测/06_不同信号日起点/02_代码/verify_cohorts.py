# -*- coding: utf-8 -*-
"""交付校验：251 个信号日是否逐一有结果、首笔与组合是否对得上、与母策略是否一致。"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / '02_代码'))
import timing_lib as tl  # noqa: E402

RES = OUT / '03_结果'
EVI = OUT / '05_证据'
EVI.mkdir(parents=True, exist_ok=True)

C = pd.read_csv(RES / '起点汇总.csv', parse_dates=['signal_date'])
B = pd.read_csv(RES / '逐日首笔.csv', parse_dates=['signal_date'])
DET = pd.read_csv(RES / '逐日首笔_逐只.csv', parse_dates=['signal_date'])
P = pd.read_csv(RES / '相邻信号日.csv', parse_dates=['date_a', 'date_b'])
TR = pd.read_csv(RES / '全部起点_逐笔.csv',
                 parse_dates=['起点', 'signal_date', 'entry_date', 'exit_date'])
D = pd.read_csv(RES / '起点分布.csv')
M = json.load(open(RES / '指标.json', encoding='utf-8'))

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok), detail))


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ---------------- 基础行数
check('信号日汇总 251 行', len(C) == 251, str(len(C)))
check('逐日首笔 251 行', len(B) == 251, str(len(B)))
check('逐只首笔 459 行', len(DET) == 459, str(len(DET)))
check('相邻信号日 250 行', len(P) == 250, str(len(P)))
check('信号日无重复', C.signal_date.is_unique)
check('逐日首笔信号日无重复', B.signal_date.is_unique)
check('逐只首笔 (信号日,代码) 无重复',
      not DET.duplicated(['signal_date', 'code']).any())
check('起点汇总与逐日首笔信号日一致',
      set(C.signal_date) == set(B.signal_date))
check('相邻信号日首尾覆盖 251 个信号日',
      set(P.date_a) | set(P.date_b) == set(C.signal_date))

# ---------------- 组合表口径
check('期末倍数=1+总收益', np.allclose(C.final, 1 + C.total_ret, atol=1e-9))
check('总利润=总收益×10万',
      np.allclose(C.profit_100k, C.total_ret * 100000, atol=1e-4))
check('最大回撤≤0', (C.dd.dropna() <= 0).all())
check('逐笔退出笔数=总笔数',
      (C.loc[C.n > 0, 'n'].values ==
       (C.loc[C.n > 0, 'n_target'] + C.loc[C.n > 0, 'n_time'] +
        C.loc[C.n > 0, 'n_eod']).values).all())
check('有交易起点 249 个', int((C.n > 0).sum()) == 249,
      str(int((C.n > 0).sum())))
check('无交易起点 2 个（2026-10-22/23）',
      set(C.loc[C.n == 0, 'signal_date'].dt.strftime('%Y-%m-%d')) ==
      {'2026-10-22', '2026-10-23'})
check('样本列计数 101+45+56+49',
      C['样本'].value_counts().to_dict() ==
      {'满5年': 101, '满1年': 56, '不足1年': 49, '满3年': 45},
      str(C['样本'].value_counts().to_dict()))

# ---------------- 首笔批次
check('首笔完整批次 246 个', int(B.batch_complete.sum()) == 246,
      str(int(B.batch_complete.sum())))
check('首笔组合收益非空 246 个', int(B.batch_ret.notna().sum()) == 246,
      str(int(B.batch_ret.notna().sum())))
check('首笔浮动收益非空 249 个', int(B.batch_float.notna().sum()) == 249,
      str(int(B.batch_float.notna().sum())))
check('首笔买入只数 0–2', B.n_buy.between(0, 2).all())
check('首笔完整时买入只数≥1',
      (B.loc[B.batch_complete, 'n_buy'] >= 1).all())
check('首笔不完整时组合收益为空',
      B.loc[~B.batch_complete, 'batch_ret'].isna().all())

# ---------------- 最近案例
recent = C.set_index(C.signal_date.dt.strftime('%Y-%m-%d'))
check('2026-10-23 前二 603778/605376',
      recent.loc['2026-10-23', 'first_code'] == '603778.SH' and
      recent.loc['2026-10-23', 'second_code'] == '605376.SH')
check('2026-10-22 只有 605117',
      recent.loc['2026-10-22', 'first_code'] == '605117.SH' and
      pd.isna(recent.loc['2026-10-22', 'second_code']))
check('2026-10-23 计划入场 10-28',
      pd.Timestamp(recent.loc['2026-10-23', 'scheduled_entry']) ==
      pd.Timestamp('2026-10-28'))
check('2026-10-22 计划入场 10-27',
      pd.Timestamp(recent.loc['2026-10-22', 'scheduled_entry']) ==
      pd.Timestamp('2026-10-27'))
check('2026-10-14/15/16 期末市值已记录（非1.0）',
      all(not close(recent.loc[d, 'final'], 1.0, 1e-9)
          for d in ['2026-10-14', '2026-10-15', '2026-10-16']),
      str({d: round(float(recent.loc[d, 'final']), 6)
           for d in ['2026-10-14', '2026-10-15', '2026-10-16']}))
check('2026-10-14/15/16 都有未平仓笔数',
      all(recent.loc[d, 'n_eod'] >= 1
          for d in ['2026-10-14', '2026-10-15', '2026-10-16']))
check('2026-10-22/23 期末倍数=1、笔数=0',
      all(close(recent.loc[d, 'final'], 1.0, 1e-9) and
          recent.loc[d, 'n'] == 0 for d in ['2026-10-22', '2026-10-23']))

# ---------------- 相邻配对
check('次日配对 158 对', int((P.gap_td == 1).sum()) == 158,
      str(int((P.gap_td == 1).sum())))
check('次日且两日≥2只 114 对',
      int(((P.gap_td == 1) & P.both_ge2).sum()) == 114,
      str(int(((P.gap_td == 1) & P.both_ge2).sum())))
check('次日且两日各买满2只 113 对',
      int(((P.gap_td == 1) & P.both_buy2).sum()) == 113,
      str(int(((P.gap_td == 1) & P.both_buy2).sum())))
X = P[(P.gap_td == 1) & P.both_ge2]
check('前二零重合 63 对', int((X.overlap == 0).sum()) == 63,
      str(int((X.overlap == 0).sum())))
check('前二重合1只 38 对', int((X.overlap == 1).sum()) == 38,
      str(int((X.overlap == 1).sum())))
check('前二完全相同 13 对', int((X.overlap == 2).sum()) == 13,
      str(int((X.overlap == 2).sum())))
check('配对的 B 日晚于 A 日', (P.date_b > P.date_a).all())
check('配对间隔≥1个交易日', (P.gap_td >= 1).all())
check('配对日期都在信号日集合',
      (set(P.date_a) | set(P.date_b)).issubset(set(C.signal_date)))

# ---------------- 分布表
check('分布表首笔批次 n=246',
      int(D.query("样本=='全部信号日' and 指标=='首笔批次收益'").n.iloc[0]) == 246)
check('分布表满3年 n=146',
      int(D.query("样本=='满3年' and 指标=='1年收益'").n.iloc[0]) == 146)
check('分布表满5年 n=101',
      int(D.query("样本=='满5年' and 指标=='5年收益'").n.iloc[0]) == 101)

# ---------------- 元数据
check('元数据 251 个信号日', M['n_signal_days'] == 251)
check('元数据 208 个信号日≥2只', M['n_signal_days_ge2'] == 208)
check('元数据 158 对次日', M['n_next_day'] == 158)
check('元数据 114 对次日≥2只', M['n_next_day_ge2'] == 114)
check('元数据 113 对次日买满2只', M['n_next_day_buy2'] == 113)
check('元数据 246 个完整首笔', M['n_complete_batches'] == 246)
check('元数据 249 个有交易起点', M['n_cohorts_with_trades'] == 249)

# ---------------- 重新跑：最早起点 + 母策略
Dp = tl.load(str(ROOT / '01_数据' / 'research_panel.parquet'))
tl.add_signal(Dp)
eq0, tr0, _ = tl.simulate_timing(
    Dp, start='2017-11-20', end='2026-10-23', nth=3, fill='open',
    hold=15, slots=2, tp=0.20, cost=0.00175)
check('最早起点重跑 99 笔', len(tr0) == 99, str(len(tr0)))
check('最早起点期末 20.745834', close(eq0.equity.iloc[-1], 20.745834, 1e-6),
      f'{eq0.equity.iloc[-1]:.6f}')
check('最早起点与起点表终值一致',
      close(eq0.equity.iloc[-1], C.iloc[0]['final'], 1e-6))

base_eq, base_tr, _ = tl.simulate_timing(
    Dp, start='2017-01-01', end='2026-10-23', nth=3, fill='open',
    hold=15, slots=2, tp=0.20, cost=0.00175)
check('母口径 99 笔', len(base_tr) == 99)
bm = tl.metrics(base_eq, base_tr)
check('母口径年化 40.0%', close(bm['cagr'], 0.4001, 1e-3), f"{bm['cagr']:.6f}")

old_path = ROOT.parent / '2026-10-24_主板恐慌反转_年化30' / '03_结果' / '最终_逐笔.csv'
if old_path.exists():
    old = pd.read_csv(old_path, parse_dates=['entry_date', 'signal_date', 'exit_date'])
    old = old.sort_values(['signal_date', 'thscode']).reset_index(drop=True)
    new = tr0.sort_values(['signal_date', 'thscode']).reset_index(drop=True)
    check('最早起点与母策略键一致',
          set(zip(old.signal_date, old.thscode)) ==
          set(zip(new.signal_date, new.thscode)))
    check('最早起点与母策略日期一致',
          (old.entry_date.values == new.entry_date.values).all() and
          (old.exit_date.values == new.exit_date.values).all())
    check('最早起点与母策略收益差异<1e-6',
          np.abs(old.ret.values - new.ret.values).max() < 1e-6,
          f'{np.abs(old.ret.values - new.ret.values).max():.2e}')
else:
    check('母策略逐笔文件存在', False, str(old_path))

# ---------------- 抽查若干信号日：首笔与独立模拟一致
sample_days = ['2017-11-20', '2018-06-14', '2019-08-06', '2020-02-04',
               '2021-02-02', '2022-04-26', '2023-10-24', '2024-06-11',
               '2025-04-09', '2026-07-31', '2026-10-15']
max_ret_diff = 0.0
all_ok = True
for d in sample_days:
    dts = pd.Timestamp(d)
    eq, tr, _ = tl.simulate_timing(
        Dp, start=dts, end='2026-10-23', nth=3, fill='open',
        hold=15, slots=2, tp=0.20, cost=0.00175)
    first = tr[tr.signal_date == dts].sort_values('thscode')
    det = DET[(DET.signal_date == dts) &
              (DET.status.isin(['ok', '持有未满']))].sort_values('code')
    if len(first) != len(det) or set(first.thscode) != set(det.code):
        all_ok = False
        continue
    diff = float(np.abs(first.ret.values - det.ret.values).max())
    max_ret_diff = max(max_ret_diff, diff)
    c_row = C[C.signal_date == dts]
    if len(c_row) != 1 or not close(eq.equity.iloc[-1], c_row.iloc[0]['final'], 1e-6):
        all_ok = False
check('抽查11个信号日：首笔键一致', all_ok)
check('抽查11个信号日：首笔收益差异<1e-6', max_ret_diff < 1e-6,
      f'{max_ret_diff:.2e}')

# ---------------- 全部起点逐笔
check('全部起点逐笔 12,090 行', len(TR) == 12090, str(len(TR)))
check('全部起点逐笔行数=汇总笔数和', len(TR) == int(C.n.sum()),
      f'{len(TR)} vs {int(C.n.sum())}')
check('全部起点逐笔覆盖 249 个起点', TR['起点'].nunique() == 249,
      str(TR['起点'].nunique()))
grp = TR.groupby('起点').size().rename('rows')
cmp = pd.DataFrame(grp).join(
    C.set_index('signal_date')['n'].rename('n'), how='outer').fillna(0)
check('每个起点逐笔数=汇总笔数', (cmp.rows == cmp.n).all(),
      str(cmp[cmp.rows != cmp.n].shape[0]))
check('全部起点逐笔无重复键',
      not TR.duplicated(['起点', 'signal_date', 'thscode', 'entry_date']).any())
check('2026-10-22/23 无逐笔',
      not TR['起点'].isin([pd.Timestamp('2026-10-22'),
                           pd.Timestamp('2026-10-23')]).any())
check('逐笔分配额>0（初始本金倍数）', (TR['alloc'] > 0).all())
check('逐笔仓位占当时权益 0–0.5',
      TR['alloc_frac'].between(0, 0.5 + 1e-9).all())
check('逐笔持有 1–15 个交易日', TR['days'].between(1, 15).all())

# ---------------- 每信号日交易记录
DAY = RES / '每信号日交易记录'
day_files = sorted(DAY.glob('20*.csv'))
check('每信号日交易记录 251 个文件', len(day_files) == 251, str(len(day_files)))
check('每信号日索引存在', (DAY / '00_索引.csv').exists())
IDX = pd.read_csv(DAY / '00_索引.csv')
check('每信号日索引 251 行', len(IDX) == 251, str(len(IDX)))
check('每信号日索引日期与起点汇总一致',
      set(IDX['起点信号日']) == set(C.signal_date.dt.strftime('%Y-%m-%d')))
day_bad = []
day_sum = 0
for _, r in C.iterrows():
    d = pd.Timestamp(r.signal_date).strftime('%Y-%m-%d')
    f = DAY / f'{d}.csv'
    if not f.exists():
        day_bad.append(d); continue
    q = pd.read_csv(f)
    if int(r.n) > 0:
        day_sum += len(q)
        if len(q) != int(r.n):
            day_bad.append(d)
    else:
        if '状态' not in q.columns:
            day_bad.append(d)
check('每个信号日文件与汇总笔数一致', not day_bad,
      f'{len(day_bad)} bad: {day_bad[:5]}')
check('每信号日逐笔合计=12,090', day_sum == 12090, str(day_sum))
check('2026-10-22/23 无交易文件有状态列',
      all('状态' in pd.read_csv(DAY / f'{d}.csv').columns
          for d in ['2026-10-22', '2026-10-23']))
check('2018-04-23 交易记录 93 行',
      len(pd.read_csv(DAY / '2018-04-23.csv')) == 93)
check('每信号日交易记录 zip 存在',
      (RES / '每信号日交易记录.zip').exists())

# ---------------- 写证据
lines = ['不同信号日起点 — 交付校验',
         f'时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}',
         '']
for name, ok, detail in results:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" +
                 (f' | {detail}' if detail else ''))
n_fail = sum(1 for _, ok, _ in results if not ok)
lines += ['', f'通过 {len(results)-n_fail} / {len(results)}；失败 {n_fail}']
(EVI / '交付校验.txt').write_text('\n'.join(lines), encoding='utf-8')

# 哈希
files = [OUT / '不同信号日起点回测.xlsx',
         OUT / 'README.md', OUT / '00_一页纸_不同信号日起点.md',
         OUT / '01_完整报告_不同信号日起点.md', OUT / '03_数据与复现.md',
         RES / '起点汇总.csv', RES / '逐日首笔.csv', RES / '逐只首笔.csv',
         RES / '相邻信号日.csv', RES / '起点分布.csv',
         RES / '起点年度.csv', RES / '相邻信号日_汇总.csv',
         RES / '最近信号日案例.csv', RES / '全部起点_逐笔.csv',
         RES / '每信号日交易记录' / '00_索引.csv',
         RES / '每信号日交易记录' / 'README_每信号日交易记录.md',
         RES / '每信号日交易记录.zip',
         RES / '指标.json'] + sorted((OUT / '04_图表').glob('*.png'))
hlines = []
for f in files:
    if f.exists():
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        hlines.append(f'{h}  {f.relative_to(OUT)}')
(EVI / '关键文件sha256.txt').write_text('\n'.join(hlines), encoding='utf-8')

print('\n'.join(lines))
print('evidence written')
