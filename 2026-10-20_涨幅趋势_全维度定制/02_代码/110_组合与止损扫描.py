# -*- coding: utf-8 -*-
"""探索性组合：单票上限、分批止盈、波动目标、亏损暂停、更近固定止损。"""
import os, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 'X0_基准复核': {},
 'X1_40_分批20': dict(maxpos=.40, partial_trigger=.20),
 'X2_40_宽60_分批20': dict(maxpos=.40, gate_thr=.60, partial_trigger=.20),
 'X3_40_波动20': dict(maxpos=.40, vol_target=.20),
 'X4_40_波动20_分批20': dict(maxpos=.40, vol_target=.20, partial_trigger=.20),
 'X5_宽60_分批20': dict(gate_thr=.60, partial_trigger=.20),
 'X6_40_行业_分批20': dict(maxpos=.40, sector_cap=True, partial_trigger=.20),
 'X7_35_分批20': dict(maxpos=.35, partial_trigger=.20),
 'X8_40_宽55_分批20': dict(maxpos=.40, gate_thr=.55, partial_trigger=.20),
 'X9_40_连亏暂停': dict(maxpos=.40, loss_pause_n=2, loss_pause_days=5),
 'X10_40_回撤节流252': dict(maxpos=.40, dd_half=.08, dd_stop=.12, dd_factor=.5, dd_peak_window=252),
 'X11_固定50_止损6': dict(stop_pct=.06),
 'X12_固定40_止损7.5': dict(maxpos=.40, stop_pct=.075),
 'X13_固定35_止损10': dict(maxpos=.35, stop_pct=.10),
 'X14_固定50_止损10': dict(stop_pct=.10),
 'X15_固定50_止损15': dict(stop_pct=.15),
 'X16_风险4_止损10': dict(sizing='risk', risk=.04, risk_basis='stop', stop_pct=.10, heat_cap=.20),
 'X17_风险5_止损10': dict(sizing='risk', risk=.05, risk_basis='stop', stop_pct=.10, heat_cap=.20),
 'X18_风险3_止损12': dict(sizing='risk', risk=.03, risk_basis='stop', stop_pct=.12, heat_cap=.20),
 'X19_40_宽度仓位': dict(maxpos=.40, size_breadth=True),
 'X20_40_分批20_连亏暂停': dict(maxpos=.40, partial_trigger=.20, loss_pause_n=2, loss_pause_days=5),
 'X21_40_分批20_行业': dict(maxpos=.40, partial_trigger=.20, sector_cap=True),
 'X22_45_分批20': dict(maxpos=.45, partial_trigger=.20),
 'X23_40_分批15': dict(maxpos=.40, partial_trigger=.15),
 'X24_40_分批25': dict(maxpos=.40, partial_trigger=.25),
}
rows = []
for name, kw in V.items():
    a = B.copy(); a.update(kw)
    for start, end, per in [('2018-01-01', '2021-12-31', '训练'),
                            ('2022-01-01', '2026-10-13', '检验'),
                            ('2018-01-01', '2026-10-13', '全期')]:
        e, t, m = E.run(start, end, **a)
        m.update({'方案': name, '期': per})
        rows.append(m)
        if per == '全期':
            print(f'{name:20s} CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  '
                  f'低点 {m["maxdd_low"]*100:6.1f}%  交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  '
                  f'PF {m["pf"]:.2f}  最差 {m["worst_trade"]*100:5.1f}%  前5 {m["top5_share"]*100:5.0f}%  '
                  f'暴露 {m["avg_exposure"]*100:4.0f}%')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/回撤优化_探索组合.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 340); pd.set_option('display.max_rows', 300)
piv = out.pivot_table(index='方案', columns='期', values=['cagr', 'maxdd', 'calmar', 'worst_trade', 'top5_share'])
print('\n===== 训练/检验 =====')
print(piv.round(3).to_string())
