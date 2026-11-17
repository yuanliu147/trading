# -*- coding: utf-8 -*-
"""回撤上限：把波动目标继续压低，量化“控回撤”的收益代价。"""
import os, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.45, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 '45%_波动20（主推荐）': dict(vol_target=.20),
 '45%_波动15': dict(vol_target=.15),
 '45%_波动12': dict(vol_target=.12),
 '45%_波动10': dict(vol_target=.10),
 '45%_权益波动15': dict(eq_vol_target=.15),
 '45%_权益波动12': dict(eq_vol_target=.12),
 '45%_权益波动10': dict(eq_vol_target=.10),
 '45%_波动10_分批20': dict(vol_target=.10, partial_trigger=.20),
 '45%_波动10_暂停': dict(vol_target=.10, loss_pause_n=2, loss_pause_days=5),
}
rows = []
for name, kw in V.items():
    a = B.copy(); a.update(kw)
    e, t, m = E.run('2018-01-01', '2026-10-13', **a)
    tr = E.run('2018-01-01', '2021-12-31', **a)[2]
    te = E.run('2022-01-01', '2026-10-13', **a)[2]
    m.update({'方案': name, '训练年化': tr['cagr'], '训练回撤': tr['maxdd'],
              '检验年化': te['cagr'], '检验回撤': te['maxdd']})
    rows.append(m)
    print(f'{name:24s} CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  '
          f'训练 {tr["cagr"]*100:5.1f}%/{tr["maxdd"]*100:6.1f}%  检验 {te["cagr"]*100:5.1f}%/{te["maxdd"]*100:6.1f}%  '
          f'交易 {m["trades"]:3d}  暴露 {m["avg_exposure"]*100:3.0f}%  期末 {m["final"]:.0f}')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/回撤上限_波动目标.csv'), index=False, encoding='utf-8-sig')
print('\n', out[['方案','cagr','maxdd','训练年化','训练回撤','检验年化','检验回撤','avg_exposure']].round(3).to_string(index=False))
