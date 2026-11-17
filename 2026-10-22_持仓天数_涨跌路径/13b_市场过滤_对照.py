# -*- coding: utf-8 -*-
"""事后对照：给 70/15 再加“市场均线/斜率/宽度60”过滤器，能否消除 2018、2023 的负收益？
这不是推荐规则，只用于回答“为什么不额外加市场择时”。
输出 13_市场过滤_对照.csv。
"""
import os
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

BASE = '/Users/admin/WorkSpace/trading'
eng = SourceFileLoader(
    'eng', os.path.join(BASE, '2026-10-22_全信号_买卖时间_优先级研究', '108_稳健引擎.py')
).load_module()


class Base(eng.Engine):
    def __init__(self):
        super().__init__()
        self.base_nogate = self.base_nogate & (self.AMT <= 70e6) & (self.C <= 15)


def annual_returns(E):
    out = {}
    prev = E.equity.iloc[0]
    for y in range(2018, 2027):
        x = E.equity.loc[str(y)]
        if len(x) == 0:
            out[f'{y}年'] = np.nan
            continue
        out[f'{y}年'] = x.iloc[-1] / prev - 1
        prev = x.iloc[-1]
    return out


def main():
    modes = [
        ('mb', '宽度>50%（基准）'),
        ('mb_mkt60', '+市场>MA60且斜率>0'),
        ('mb_slope', '+市场斜率>0'),
        ('mb_b60', '宽度60>50%'),
    ]
    rows = []
    for mode, label in modes:
        b = Base()
        E, T, m = b.run(
            '2018-01-01', '2026-10-13', slots=2, sizing='fixed', maxpos=.45, heat_cap=.10,
            gate_thr=.50, gate_mode=mode, rank='amt', hold=20, cost_rate=.0015,
            vol_target=.20, stop_pct=.20, skip_entry_lu=True,
        )
        row = {'闸门口径': label, '年化': m['cagr'], '最大回撤': m['maxdd'],
               '交易数': m['trades'], '胜率': m['win'], '盈亏比': m['pf']}
        row.update(annual_returns(E))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv('13_市场过滤_对照.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
