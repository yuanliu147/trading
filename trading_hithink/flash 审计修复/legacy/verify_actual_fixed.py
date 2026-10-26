# -*- coding: utf-8 -*-
"""修正版实际价交付校验。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('03_结果')
EVI = Path('05_证据')
EVI.mkdir(exist_ok=True)
res = []


def check(name, ok, detail=''):
    res.append((name, bool(ok), detail))


tr = pd.read_csv(OUT / '实际价_修正版_逐笔.csv',
                 parse_dates=['signal_date', 'entry_date', 'exit_date'])
eq = pd.read_csv(OUT / '实际价_修正版_每日权益.csv', parse_dates=['date'])
M = json.load(open(OUT / '实际价_修正版_指标.json', encoding='utf-8'))
old = json.load(open(OUT / '最终_指标.json', encoding='utf-8'))['actual']
old_tr = pd.read_csv(OUT / '旧版_不可靠/实际价_逐笔_旧版.csv')

check('99 笔', len(tr) == 99, str(len(tr)))
check('期末=10万+净盈亏合计', abs(eq.equity.iloc[-1] - (100000 + tr.pnl.sum())) < 1,
      f'{eq.equity.iloc[-1]:.2f} vs {100000+tr.pnl.sum():.2f}')
check('期末 2,083,521', abs(eq.equity.iloc[-1] - 2083521.12) < 1,
      f'{eq.equity.iloc[-1]:.2f}')
check('年化 40.59%', abs(M['cagr'] - 0.40585159034050045) < 1e-9,
      f"{M['cagr']:.6f}")
check('最大回撤 -15.92%', abs(M['dd'] + 0.1591521865928841) < 1e-9,
      f"{M['dd']:.6f}")
check('胜率 71.7%', abs(M['win'] - 0.7171717171717171) < 1e-9)
check('平均单笔 6.72%', abs(M['avg'] - 0.06722396764185641) < 1e-9)
check('实际价 vs 复权平均绝对差<0.01',
      M['mean_abs_diff'] < 0.01, f"{M['mean_abs_diff']:.6f}")
check('实际价 vs 复权最大差<0.01',
      float((tr.ret - tr.hfq_ret).abs().max()) < 0.01,
      f"{float((tr.ret - tr.hfq_ret).abs().max()):.6f}")

r365 = tr[(tr.thscode == '002365.SZ') & (tr.signal_date == '2018-04-23')].iloc[0]
check('002365 股数 3,750', abs(r365.shares - 3750) < 1e-9,
      f'{r365.shares}')
check('002365 净盈亏 +6,793.76', abs(r365.pnl - 6793.759375) < 0.01,
      f'{r365.pnl:.2f}')
check('002365 净收益率 +11.19%', abs(r365.ret - 0.1118759732527252) < 1e-9,
      f'{r365.ret:.6f}')
check('002365 复权收益 +11.12%', abs(r365.hfq_ret - 0.111245155) < 1e-6,
      f'{r365.hfq_ret:.6f}')
r607 = tr[(tr.thscode == '002607.SZ') & (tr.signal_date == '2018-12-17')].iloc[0]
check('002607 股数 10,500', abs(r607.shares - 10500) < 1e-9,
      f'{r607.shares}')
check('002607 净盈亏 +3,974.28', abs(r607.pnl - 3974.27625) < 0.01,
      f'{r607.pnl:.2f}')
check('002607 净收益率 +5.28%', abs(r607.ret - 0.0527974808026281) < 1e-9,
      f'{r607.ret:.6f}')
check('002607 现金分红 4,095', abs(r607.dividend - 4095) < 1e-6,
      f'{r607.dividend}')

o365 = old_tr[(old_tr.thscode == '002365.SZ') &
              (old_tr.signal_date == '2018-04-23')].iloc[0]
check('旧版 002365 为 -24,533（假亏损）', abs(o365.pnl + 24533.15719) < 0.01,
      f'{o365.pnl:.2f}')
check('旧版 002365 股数 3,868.95（错）', abs(o365.shares - 3868.946897) < 1e-6,
      f'{o365.shares}')
check('旧版实际价年化 42.58%（不可用）', abs(old['cagr'] - 0.42579370149169615) < 1e-9,
      f"{old['cagr']:.6f}")
check('旧版期末 236.2 万（不可用）', abs(old['final'] - 2362269.6456) < 1,
      f"{old['final']:.2f}")

check('旧版 .DEPRECATED 标记存在',
      (OUT / '实际价_逐笔.csv.DEPRECATED').exists() and
      (OUT / '交易记录_中文.csv.DEPRECATED').exists())
check('修正版 Excel 存在', (OUT / '实际价_修正版.xlsx').exists())
check('修正说明存在', (OUT / '实际价_修正说明.md').exists())

lines = ['实际价修正版 — 交付校验',
         f"时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}", '']
for name, ok, detail in res:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' | {detail}' if detail else ''))
n_fail = sum(1 for _, ok, _ in res if not ok)
lines += ['', f'通过 {len(res)-n_fail} / {len(res)}；失败 {n_fail}']
(EVI / '实际价_修正校验.txt').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))
