# -*- coding: utf-8 -*-
"""三套策略的统一定义与回测：
1) 70/15 基准：市场宽度>50%，2仓45%，20日持有，-20%止损；
2) 70/15 + 极端超跌：宽度<10%时，额外允许 2 个仓位买“RSI<30、ATR<=6%、AMT20>=1亿”的超跌股；
3) 增强 + 回撤节流：在 2) 上，权益回撤>=10% 时新仓减半，>=18% 停止新开仓。
输出 08_*_权益.csv / 08_*_逐笔.csv，供 10_回撤与稳健性.py 审计。
"""
import os, numpy as np, pandas as pd
from importlib.machinery import SourceFileLoader
BASE='/Users/admin/WorkSpace/trading'
SRC=os.path.join(BASE,'2026-10-22_全信号_买卖时间_优先级研究','108_稳健引擎.py')
WEAK=os.path.join(BASE,'2026-10-22_持仓天数_涨跌路径','07_弱势引擎.py')
eng=SourceFileLoader('eng',SRC).load_module()
weak=SourceFileLoader('weak',WEAK).load_module()

class Base(eng.Engine):
    def __init__(self):
        super().__init__()
        self.base_nogate=self.base_nogate & (self.AMT<=70e6) & (self.C<=15)

class Enhanced(weak.Engine):
    def __init__(self):
        super().__init__()
        g=lambda k:self.P[k].values
        self.MB0=g('market_breadth20').copy()
        dates=self.dates
        self.valid=(dates>=pd.Timestamp('2017-10-13'))[:,None]&(self.AMT>=3e7)&(self.C>=2)
        self.trend_t=self.base_nogate&(self.AMT<=70e6)&(self.C<=15)&(self.MB0>0.5)[:,None]
        self.ov=(self.MB0<.10)[:,None]&(self.RSI<30)&(self.ATR<=.06)&(self.AMT>=1e8)
        self.MB=np.ones_like(self.MB0)
        self.base_nogate=self.trend_t|(self.valid&self.ov)

def save(name,e,t,met):
    e.to_csv(f'08_{name}_权益.csv',encoding='utf-8-sig',float_format='%.4f',date_format='%Y-%m-%d')
    t.to_csv(f'08_{name}_逐笔.csv',index=False,encoding='utf-8-sig',float_format='%.6f',date_format='%Y-%m-%d')
    print(name,{k:round(v,4) for k,v in met.items() if k!='daily'})

if __name__=='__main__':
    b=Base(); e,t,m=b.run('2018-01-01','2026-10-13',slots=2,sizing='fixed',maxpos=.45,heat_cap=.10,
        gate_thr=.50,rank='amt',hold=20,cost_rate=.0015,vol_target=.20,stop_pct=.20,skip_entry_lu=True)
    save('trend',e,t,m)
    en=Enhanced(); e,t,m=en.run('2018-01-01','2026-10-13',slots=2,sizing='fixed',maxpos=.45,heat_cap=.10,
        gate_thr=-1,rank='amt',hold=20,cost_rate=.0015,vol_target=.20,stop_pct=.20,skip_entry_lu=True)
    save('liq1e8',e,t,m)
    en2=Enhanced(); e,t,m=en2.run('2018-01-01','2026-10-13',slots=2,sizing='fixed',maxpos=.45,heat_cap=.10,
        gate_thr=-1,rank='amt',hold=20,cost_rate=.0015,vol_target=.20,stop_pct=.20,skip_entry_lu=True,
        dd_half=.10,dd_stop=.18,dd_factor=.5)
    save('liq1e8_dd',e,t,m)
