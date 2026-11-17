# -*- coding: utf-8 -*-
"""2万账户 25.8% 年化的严谨性审计：参数邻域、成本、持有期、单票上限、真实风险口径。"""
import importlib.util, os, pandas as pd

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = importlib.util.spec_from_file_location('m102', os.path.join(BASE, '02_代码/102_两万组合.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BASE_KW = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10,
               gate_thr=0.50, rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
START, END = '2018-01-01', '2026-10-13'


def one(cat, name, note, **kw):
    a = BASE_KW.copy()
    a.update(kw)
    E, T, met = m.run(START, END, **a)
    T['账户亏损'] = T.weight * T.ret
    return dict(
        类别=cat, 方案=name,
        年化收益=met['cagr'], 最大回撤=met['maxdd'], 最低价回撤=met['maxdd_low'],
        交易笔数=met['trades'], 胜率=met['win'], 盈亏比=met['pf'],
        最大单票仓位=T.weight.max(), 最差单笔账户亏损=T['账户亏损'].min(),
        期末资金=met['final'], 说明=note,
    )


rows = []
rows.append(one('基准', '固定50%·宽度50%·持有20日',
                '44组参数中的最高值；同族邻居19.7%–24.7%；2022和2026两年贡献大部分利润'))
for c in [0.0025, 0.0035, 0.005]:
    rows.append(one('成本', f'单边成本{c * 100:.2f}%', '基准为0.15%；成本越高，25.8%越难成立',
                    cost_rate=c))
for mp in [0.35, 0.40, 0.45, 0.55]:
    rows.append(one('单票上限', f'单票上限{mp * 100:.0f}%', '收益随仓位上升，回撤同步放大',
                    maxpos=mp))
for h in [10, 15, 25, 30]:
    rows.append(one('持有期', f'持有{h}日', '持有20日是尖峰，15日和25日差异极大', hold=h))
for g in [0.45, 0.55, 0.60]:
    rows.append(one('市场宽度', f'宽度{g * 100:.0f}%', '宽度50%是尖峰，45%和55%差别很大',
                    gate_thr=g))
for risk, gate, label in [(0.02, 0.55, '严格2%单笔风险·宽度55%'),
                          (0.03, 0.55, '严格3%单笔风险·宽度55%'),
                          (0.03, 0.50, '严格3%单笔风险·宽度50%')]:
    rows.append(one('真实风险口径', label,
                    '按-20%灾难止损反推股数，3%风险对应约15%仓位；一手超过预算的股票直接跳过，最差单笔约-3%',
                    sizing='risk', risk=risk, gate_thr=gate, risk_basis='stop',
                    heat_cap=0.20))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(BASE, '03_结果/两万_严谨性审计.csv'), index=False, encoding='utf-8-sig')
print(out.round(4).to_string(index=False))
