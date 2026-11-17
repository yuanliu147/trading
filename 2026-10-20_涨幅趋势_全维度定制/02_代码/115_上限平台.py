# -*- coding: utf-8 -*-
"""最后一小组：在40%上限附近确认收益/回撤平台，不新增规则类型。"""
import os, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 'W1_42.5_波动20': dict(maxpos=.425, vol_target=.20),
 'W2_45_波动20': dict(maxpos=.45, vol_target=.20),
 'W3_42.5_波动20_分批20': dict(maxpos=.425, vol_target=.20, partial_trigger=.20),
 'W4_45_波动20_分批20': dict(maxpos=.45, vol_target=.20, partial_trigger=.20),
 'W5_42.5_波动20_暂停': dict(maxpos=.425, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
 'W6_45_波动20_暂停': dict(maxpos=.45, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
}
rows = []
for name, kw in V.items():
    a = B.copy(); a.update(kw)
    for start, end, per in [('2018-01-01', '2021-12-31', '训练'),
                            ('2022-01-01', '2026-10-13', '检验'),
                            ('2018-01-01', '2026-10-13', '全期')]:
        e, t, m = E.run(start, end, **a)
        m.update({'方案': name, '期': per}); rows.append(m)
        if per == '全期':
            print(f'{name:24s} CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  '
                  f'交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  PF {m["pf"]:.2f}  '
                  f'最差 {m["worst_trade"]*100:5.1f}%  前5 {m["top5_share"]*100:5.0f}%  暴露 {m["avg_exposure"]*100:4.0f}%')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/回撤优化_上限平台.csv'), index=False, encoding='utf-8-sig')
print('\n===== 训练/检验 =====')
print(out.pivot_table(index='方案', columns='期', values=['cagr', 'maxdd', 'calmar', 'worst_trade', 'top5_share']).round(3).to_string())
