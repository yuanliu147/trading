# -*- coding: utf-8 -*-
"""第二轮：波动目标+亏损暂停+分批/宽度；组合权益波动目标。"""
import os, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 'Y1_40_波动20_连亏暂停': dict(maxpos=.40, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
 'Y2_40_波动20_分批20_暂停': dict(maxpos=.40, vol_target=.20, partial_trigger=.20,
                                  loss_pause_n=2, loss_pause_days=5),
 'Y3_40_波动20_宽55': dict(maxpos=.40, vol_target=.20, gate_thr=.55),
 'Y4_40_波动20_宽60': dict(maxpos=.40, vol_target=.20, gate_thr=.60),
 'Y5_40_波动20_分批20_宽55': dict(maxpos=.40, vol_target=.20, partial_trigger=.20, gate_thr=.55),
 'Y6_40_波动20_分批20_宽60': dict(maxpos=.40, vol_target=.20, partial_trigger=.20, gate_thr=.60),
 'Y7_45_波动20_暂停': dict(maxpos=.45, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
 'Y8_40_波动20_行业': dict(maxpos=.40, vol_target=.20, sector_cap=True),
 'Y9_40_波动20_宽55_暂停': dict(maxpos=.40, vol_target=.20, gate_thr=.55,
                                loss_pause_n=2, loss_pause_days=5),
 'Y10_40_波动20_宽60_暂停': dict(maxpos=.40, vol_target=.20, gate_thr=.60,
                                 loss_pause_n=2, loss_pause_days=5),
 'Z1_40_权益波动25': dict(maxpos=.40, eq_vol_target=.25),
 'Z2_40_权益波动20': dict(maxpos=.40, eq_vol_target=.20),
 'Z3_40_权益波动30': dict(maxpos=.40, eq_vol_target=.30),
 'Z4_40_权益波动25_分批20': dict(maxpos=.40, eq_vol_target=.25, partial_trigger=.20),
 'Z5_40_权益波动25_暂停': dict(maxpos=.40, eq_vol_target=.25, loss_pause_n=2, loss_pause_days=5),
 'Z6_45_权益波动25': dict(maxpos=.45, eq_vol_target=.25),
 'Z7_40_权益波动20_分批20': dict(maxpos=.40, eq_vol_target=.20, partial_trigger=.20),
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
                  f'低点 {m["maxdd_low"]*100:6.1f}%  交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  '
                  f'PF {m["pf"]:.2f}  最差 {m["worst_trade"]*100:5.1f}%  前5 {m["top5_share"]*100:5.0f}%  '
                  f'暴露 {m["avg_exposure"]*100:4.0f}%')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/回撤优化_第二轮.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 340); pd.set_option('display.max_rows', 300)
print('\n===== 训练/检验 =====')
print(out.pivot_table(index='方案', columns='期', values=['cagr', 'maxdd', 'calmar', 'worst_trade', 'top5_share']).round(3).to_string())
