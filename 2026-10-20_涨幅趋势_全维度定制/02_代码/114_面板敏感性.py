# -*- coding: utf-8 -*-
"""面板敏感性：含102只退市/缺失股 vs 仅当前股票面板。"""
import sys, os, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, '01_数据/全A面板_2016_2026_含退市.pkl')
E = mod.Engine(path)
print('面板', os.path.basename(path), '股票数', len(E.codes))
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 '原25.8%': {},
 '收益优先_40_波动20': dict(maxpos=.40, vol_target=.20),
 '均衡_40_波动20_分批20': dict(maxpos=.40, vol_target=.20, partial_trigger=.20),
 '高收益_40_波动20_暂停': dict(maxpos=.40, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
}
for name, kw in V.items():
    a = B.copy(); a.update(kw)
    e, t, m = E.run('2018-01-01', '2026-10-13', **a)
    print(f'{name:24s} CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  交易 {m["trades"]:3d}  '
          f'期末 {m["final"]:.0f}  前5 {m["top5_share"]*100:3.0f}%')
