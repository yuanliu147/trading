import pandas as pd, numpy as np, sys
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
PER={'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}
def run(mask,**kw):
    base['ev_custom']=mask.values; out={}
    for p,(s,e) in PER.items():
        eq,tr=simulate2(D,'custom',start=s,end=e,**kw); out[p]=metrics(eq,tr)
    return out
def show(label,o):
    print(f"{label}: train {o['train'].get('cagr',0):.3f} val {o['val'].get('cagr',0):.3f} test {o['test'].get('cagr',0):.3f} full {o['full'].get('cagr',0):.3f} dd {o['full'].get('dd',0):.3f} n {o['full'].get('n',0)}")
kw=dict(hold=15,slots=2,delay=2,take_profit=0.20,score='drawdown250',dyn=False)
base0=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
print('=== ablation ===')
show('base',run(base0,**kw))
show('no_rsi',run((base.drawdown250<-0.45)&(base.return_250d>0)&(base.close_location>0.45)&(base.market_return20<-0.05),**kw))
show('no_r250',run((base.drawdown250<-0.45)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05),**kw))
show('no_cl',run((base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.market_return20<-0.05),**kw))
show('no_mkt',run((base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45),**kw))
print('\n=== delay2 + confirmation ===')
for c in ['conf_close2','conf_breadth2','conf_mkt2']:
    show(c,run(base0&base[c],**kw))
show('all3',run(base0&base.conf_close2&base.conf_breadth2&base.conf_mkt2,**kw))
show('close+breadth',run(base0&base.conf_close2&base.conf_breadth2,**kw))
print('\n=== delay2 without RSI + confirmation ===')
m=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.close_location>0.45)&(base.market_return20<-0.05)
show('noRSI+close',run(m&base.conf_close2,**kw))
show('noRSI+all3',run(m&base.conf_close2&base.conf_breadth2&base.conf_mkt2,**kw))
print('\n=== delay / confirmation interplay ===')
for d in [0,1,2,3]:
    show(f'd{d}',run(base0,**{**kw,'delay':d}))
print('\n=== remove top winners (correct signal_date) ===')
def runfull(over={}):
    eq,tr=simulate2(D,'custom',start='2017-01-01',end='2026-10-23',**{**kw,**over}); return eq,tr,metrics(eq,tr)
eq,tr,mm=runfull(); tr2=tr.sort_values('ret',ascending=False).reset_index(drop=True)
print('baseline',mm['cagr'],mm['dd'],mm['n'])
for n in [1,3,5,10,20]:
    drop=tr2.head(n)
    mask=base0.copy()
    for _,r in drop.iterrows():
        mask &= ~((base.date==r.signal_date)&(base.thscode==r.thscode))
    base['ev_custom']=mask.values
    eq2,tr3,mm2=runfull(); print(f'drop{n}',round(mm2['cagr'],3),round(mm2['dd'],3),mm2['n'])
