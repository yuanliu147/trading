# -*- coding: utf-8 -*-
"""第三轮：市场转弱时只砍亏损仓（宽度/指数MA60），保留强势赢家。"""
import os, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
V = {
 'R1_基准_宽度退出': dict(regime_exit='breadth'),
 'R2_基准_MA60退出': dict(regime_exit='ma60'),
 'R3_基准_双退出': dict(regime_exit='both'),
 'R4_40_宽度退出': dict(maxpos=.40, regime_exit='breadth'),
 'R5_40_MA60退出': dict(maxpos=.40, regime_exit='ma60'),
 'R6_40_双退出': dict(maxpos=.40, regime_exit='both'),
 'R7_40_权益波动25_宽度退出': dict(maxpos=.40, eq_vol_target=.25, regime_exit='breadth'),
 'R8_40_权益波动25_分批20_宽度退出': dict(maxpos=.40, eq_vol_target=.25, partial_trigger=.20,
                                           regime_exit='breadth'),
 'R9_40_市场波动20_宽度退出': dict(maxpos=.40, vol_target=.20, regime_exit='breadth'),
 'R10_40_分批20_宽度退出': dict(maxpos=.40, partial_trigger=.20, regime_exit='breadth'),
 'R11_40_权益波动25_分批20_双退出': dict(maxpos=.40, eq_vol_target=.25, partial_trigger=.20,
                                         regime_exit='both'),
 'R12_40_权益波动25_分批20_MA60': dict(maxpos=.40, eq_vol_target=.25, partial_trigger=.20,
                                       regime_exit='ma60'),
 'R13_40_市场波动20_暂停_宽度': dict(maxpos=.40, vol_target=.20, loss_pause_n=2, loss_pause_days=5,
                                     regime_exit='breadth'),
 'R14_40_市场波动20_暂停_双': dict(maxpos=.40, vol_target=.20, loss_pause_n=2, loss_pause_days=5,
                                   regime_exit='both'),
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
            print(f'{name:30s} CAGR {m["cagr"]*100:6.1f}%  DD {m["maxdd"]*100:6.1f}%  '
                  f'低点 {m["maxdd_low"]*100:6.1f}%  交易 {m["trades"]:3d}  胜率 {m["win"]*100:4.1f}%  '
                  f'PF {m["pf"]:.2f}  最差 {m["worst_trade"]*100:5.1f}%  前5 {m["top5_share"]*100:5.0f}%  '
                  f'暴露 {m["avg_exposure"]*100:4.0f}%')
out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/回撤优化_市场退出.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 340); pd.set_option('display.max_rows', 300)
print('\n===== 训练/检验 =====')
print(out.pivot_table(index='方案', columns='期', values=['cagr', 'maxdd', 'calmar', 'worst_trade', 'top5_share']).round(3).to_string())
