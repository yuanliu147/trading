# -*- coding: utf-8 -*-
"""主板 70/15：单因素网格。

每个变体只改一个维度，其他保持主板基线。输出全期/训练/检验/年度/尾部，
先看方向，不直接选“全期最好”的参数。

基准：2 槽 × 45%，宽度>50%，成交额 3000-7000 万，股价<=15，ATR<=4.4%，
年龄 5-30，行业宽度>=60%，行业 20 日>5%，DD60>-8%，20 日持有，-20% 止损。
"""
import os
import numpy as np
import pandas as pd
import mb_lib as L

OUT = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
BASE = dict(name='主板基线（非ST）', st_mode='x')

CFG = []


def add(cat, name, **kw):
    c = dict(BASE)
    c.update(kw)
    c['name'] = name
    c['类别'] = cat
    CFG.append(c)


# 1) 市场闸门
for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    add('闸门', f'宽度>{thr:.0%}', gate_thr=thr)
for mode, label in [('mb_mkt60', '宽度>50%+指数MA60/斜率'),
                    ('mb_mkt20', '宽度>50%+指数MA20'),
                    ('mb_slope', '宽度>50%+指数60日斜率>0'),
                    ('mb_b60', '宽度>50%+宽度60>50%')]:
    add('闸门', label, gate_mode=mode)

# 2) 股票池：成交额 / 股价
for x in [30, 40, 50, 60, 70, 80, 100, 150]:
    add('成交额上限', f'成交额<={x}00万', amt_max=x * 1e6)
for x in [10, 20, 30, 40, 50]:
    add('成交额下限', f'成交额>={x}00万', amt_min=x * 1e6)
for x in [6, 8, 10, 12, 15, 20, 30]:
    add('股价上限', f'股价<={x}元', price_max=x)
for x in [2, 3, 5, 8]:
    add('股价下限', f'股价>={x}元', price_min=x)

# 3) 波动 / 趋势年龄
for x in [0.025, 0.030, 0.035, 0.040, 0.044, 0.050]:
    add('ATR上限', f'ATR<={x:.1%}', atr_max=x)
for x in [0.020, 0.025, 0.030]:
    add('ATR下限', f'ATR>={x:.1%}', atr_min=x)
for x in [10, 15, 20, 25, 30, 40]:
    add('年龄上限', f'年龄<={x}', age_max=x)
for x in [5, 8, 10, 12]:
    add('年龄下限', f'年龄>={x}', age_min=x)
for x in [-0.15, -0.12, -0.10, -0.08, -0.05, 0.0]:
    add('DD60下限', f'DD60>={x:.0%}', dd60_min=x)

# 4) 行业条件
for x in [0.50, 0.60, 0.70, 0.80, 0.90]:
    add('行业宽度', f'行业宽度>={x:.0%}', sbr_min=x)
for x in [0.0, 0.05, 0.08, 0.10, 0.15]:
    add('行业20日', f'行业20日>{x:.0%}', sr20_min=x)

# 5) 个股动量 / 延伸 / RSI
for x in [0.0, 0.05, 0.08, 0.10, 0.15, 0.20]:
    add('20日涨幅下限', f'个股20日>{x:.0%}', ret20_min=x)
for x in [0.20, 0.25, 0.30, 0.40]:
    add('20日涨幅上限', f'个股20日<={x:.0%}', ret20_max=x)
for x in [-0.02, 0.0, 0.02, 0.05, 0.08]:
    add('距MA20下限', f'距MA20>={x:.0%}', dist20_min=x)
for x in [0.08, 0.10, 0.12, 0.15, 0.20]:
    add('距MA20上限', f'距MA20<={x:.0%}', dist20_max=x)
for x in [50, 55, 60, 65]:
    add('RSI下限', f'RSI>={x}', rsi_min=x)
for x in [70, 75, 80, 85]:
    add('RSI上限', f'RSI<={x}', rsi_max=x)
for x in [0.18, 0.20, 0.25, 0.30, 0.35, 0.40]:
    add('20日波动下限', f'20日波动>={x:.0%}', vol20_min=x)
for x in [0.35, 0.40, 0.50]:
    add('20日波动上限', f'20日波动<={x:.0%}', vol20_max=x)

# 6) 市值 / 换手
for x in [30, 50, 80, 120, 200]:
    add('市值下限', f'市值>={x}亿', mcap_min=x * 1e8)
for x in [50, 100, 200, 500]:
    add('市值上限', f'市值<={x}亿', mcap_max=x * 1e8)
for x in [0.3, 0.5, 0.8, 1.0, 1.5]:
    add('换手下限', f'换手>={x}%', turn_min=x)
for x in [1.0, 2.0, 3.0, 5.0]:
    add('换手上限', f'换手<={x}%', turn_max=x)

# 7) 持有 / 止损 / 均线退出
for x in [10, 15, 20, 25, 30, 40, 60]:
    add('持有天数', f'持有{x}日', hold=x)
for x in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    add('止损', f'止损{x:.0%}', stop_pct=x)
for x in [5, 10, 15]:
    add('破MA20', f'破MA20退出（{x}日后）', exit_ma20=True, exit_min_hold=x)

# 8) 排序
for x in ['amt', 'lowvol', 'price', 'mcap', 'mcap_desc', 'age', 'dd60', 'sbr', 'sbr_mid', 'sr20', 'pullback']:
    add('排序', f'排序={x}', rank=x)
for seed in range(10):
    add('随机排序', f'随机排序 seed={seed}', rank='random', rank_seed=seed)

# 9) 仓位结构
for x in [1, 2, 3, 4, 5, 6]:
    add('仓位数', f'{x}槽', slots=x)
for x in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    add('单票上限', f'单票{x:.0%}', maxpos=x)
for x in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
    add('热预算', f'热预算{x:.0%}', heat_cap=x)
for x in [None, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]:
    add('波动目标', f'波动目标={x}', vol_target=x)
for x in [0.0, 0.3, 0.5, 0.7, 1.0]:
    add('波动下限', f'波动系数下限={x}', vol_floor=x)
for x in ['fixed', 'risk']:
    for r in ([0.01, 0.02, 0.03] if x == 'risk' else [None]):
        add('仓位算法', f'{x}' + (f' risk={r:.0%}' if r else ''),
            sizing=x, risk=r if r else 0.03)

# 10) 回撤节流 / 盈利保护 / 分批 / 市场转弱
for x in [0.06, 0.08, 0.10, 0.12, 0.15]:
    add('回撤减半', f'回撤{x:.0%}减半', dd_half=x)
for x in [0.12, 0.15, 0.18, 0.20, 0.25]:
    add('回撤停手', f'回撤{x:.0%}停手', dd_half=x * 0.6, dd_stop=x)
for x in [0.10, 0.15, 0.20]:
    add('盈利保护', f'浮盈{x:.0%}后保本', profit_lock=True, lock_trigger=x, lock_gain=0.0)
for x in [0.15, 0.20, 0.25, 0.30]:
    for tp in [0.05, 0.08, 0.10]:
        add('移动止盈', f'浮盈{x:.0%}后回撤{tp:.0%}卖', profit_lock=True,
            trail_trigger=x, trail_pct=tp)
for x in [0.10, 0.15, 0.20]:
    add('分批止盈', f'涨{x:.0%}卖一半', partial_trigger=x, partial_frac=0.5)
for x in ['breadth', 'ma60', 'both']:
    add('市场转弱', f'转弱退出={x}', regime_exit=x)

# 11) 超跌卫星
for mb in [0.05, 0.08, 0.10, 0.12]:
    add('超跌卫星', f'宽度<{mb:.0%}+RSI<30', oversold=True, ov_mb=mb, gate_thr=None,
        slots=3, maxpos=0.33)
for rsi in [20, 25, 30, 35]:
    add('超跌卫星', f'宽度<10%+RSI<{rsi}', oversold=True, ov_rsi=rsi, gate_thr=None,
        slots=3, maxpos=0.33)
for hold in [10, 20, 30]:
    add('超跌卫星', f'超跌持有{hold}日', oversold=True, ov_mb=0.10, gate_thr=None,
        slots=3, maxpos=0.33, hold=hold)
for stop in [0.10, 0.15, 0.20]:
    add('超跌卫星', f'超跌止损{stop:.0%}', oversold=True, ov_mb=0.10, gate_thr=None,
        slots=3, maxpos=0.33, stop_pct=stop)


def main():
    e = L.fresh_engine()
    rows = []
    for i, c in enumerate(CFG):
        try:
            E, T, row = L.run_one(e, c, oversold=c.get('oversold', False))
            row['类别'] = c['类别']
            row['变体'] = c['name']
            rows.append(row)
        except Exception as ex:
            print('FAIL', c['name'], ex)
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(CFG)}', flush=True)
    R = pd.DataFrame(rows)
    front = ['类别', '变体', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win',
             '全期_pf', '训练_cagr', '训练_maxdd', '训练_trades', '检验_cagr',
             '检验_maxdd', '检验_trades', '全期_最大单笔亏损', '全期_前5笔占比',
             '负年份数', '最差年']
    R = R[front + [c for c in R.columns if c not in front]]
    R.to_csv(os.path.join(OUT, '02_单因素网格_非ST.csv'), index=False, encoding='utf-8-sig',
             float_format='%.4f')
    print('done', len(R), 'rows')


if __name__ == '__main__':
    main()
