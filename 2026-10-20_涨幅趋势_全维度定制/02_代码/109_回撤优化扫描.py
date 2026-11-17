# -*- coding: utf-8 -*-
"""预注册的20个回撤优化变体：训练期/检验期/全期。"""
import os, pandas as pd, numpy as np
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()

BASE_KW = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
               rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 'V0_基准': {},
 'V1_宽度55': dict(gate_thr=0.55),
 'V2_宽度60': dict(gate_thr=0.60),
 'V3_指数MA60': dict(gate_mode='mb_mkt60'),
 'V4_指数MA20': dict(gate_mode='mb_mkt20'),
 'V5_宽度55_指数MA60': dict(gate_thr=0.55, gate_mode='mb_mkt60'),
 'V6_波动目标20': dict(vol_target=0.20),
 'V7_回撤节流': dict(dd_half=0.08, dd_stop=0.12, dd_factor=0.5),
 'V8_盈利保护': dict(profit_lock=True, lock_trigger=0.15, lock_gain=0.0,
                     trail_trigger=0.25, trail_pct=0.10),
 'V9_加20卖半': dict(partial_trigger=0.20),
 'V10_单票40': dict(maxpos=0.40),
 'V11_宽度60_单票40': dict(gate_thr=0.60, maxpos=0.40),
 'V12_行业一只': dict(sector_cap=True),
 'V13_低波排序': dict(rank='lowvol'),
 'V14_贴近MA20排序': dict(rank='pullback'),
 'V15_持有15': dict(hold=15),
 'V16_持有25': dict(hold=25),
 'V17_宽度60_盈利保护': dict(gate_thr=0.60, profit_lock=True, lock_trigger=0.15,
                              lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10),
 'V18_真实3%风险_止损20': dict(sizing='risk', risk=0.03, risk_basis='stop', stop_pct=0.20,
                                heat_cap=0.20),
 'V19_真实3%风险_止损10': dict(sizing='risk', risk=0.03, risk_basis='stop', stop_pct=0.10,
                                heat_cap=0.20),
 'V20_真实3%风险_止损8': dict(sizing='risk', risk=0.03, risk_basis='stop', stop_pct=0.08,
                               heat_cap=0.20),
}
rows = []
for name, kw in V.items():
    a = BASE_KW.copy(); a.update(kw)
    for start, end, per in [('2018-01-01', '2021-12-31', '训练'),
                            ('2022-01-01', '2026-10-13', '检验'),
                            ('2018-01-01', '2026-10-13', '全期')]:
        e, t, m = E.run(start, end, **a)
        m.update({'方案': name, '期': per})
        rows.append(m)
        if per == '全期':
            print(f'{name:22s} 全期 CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  '
                  f'低点DD {m["maxdd_low"]*100:6.1f}%  交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  '
                  f'PF {m["pf"]:.2f}  最差单笔 {m["worst_trade"]*100:5.1f}%  前5占比 {m["top5_share"]*100:5.0f}%')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/回撤优化_20变体.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 320); pd.set_option('display.max_rows', 200)
piv = out.pivot_table(index='方案', columns='期', values=['cagr', 'maxdd', 'calmar', 'win', 'pf'])
print('\n===== 训练/检验 =====')
print(piv.round(3).to_string())
