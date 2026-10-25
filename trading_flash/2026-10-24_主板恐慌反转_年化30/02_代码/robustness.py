import pandas as pd, numpy as np, sys, itertools, json
sys.path.insert(0,'02_代码')
from panic_lib import load, simulate2, metrics, annual
D=load(); base=D['base']
PERIODS={'train':('2017-01-01','2021-12-31'),'val':('2022-01-01','2023-12-31'),'test':('2024-01-01','2026-10-23'),'full':('2017-01-01','2026-10-23')}

def run(mask, **kw):
    base['ev_custom']=mask.values
    out={}
    for pname,(s,e) in PERIODS.items():
        eq,tr=simulate2(D,'custom',start=s,end=e,**kw)
        out[pname]=metrics(eq,tr)
    return out

def fmt(m): return m.get('cagr',np.nan),m.get('dd',np.nan),m.get('n',0),m.get('win',np.nan),m.get('expo',np.nan)

rows=[]
def add(label,mask,**kw):
    st=run(mask,**kw)
    row=dict(variant=label,param=kw.get('param',''),**{f'{p}_{k}':v for p in st for k,v in st[p].items()})
    rows.append(row)

# baseline mask
m0=(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05)
basekw=dict(hold=15,slots=1,score='drawdown250',dyn=False)
add('baseline',m0,**basekw)
for v in [-0.30,-0.35,-0.40,-0.45,-0.50,-0.55,-0.60]:
    add(f'dd{v}',(base.drawdown250<v)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05),**basekw)
for v in [20,25,30,35,40,45]:
    add(f'rsi{v}',(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<v)&(base.close_location>0.45)&(base.market_return20<-0.05),**basekw)
for v in [0.2,0.3,0.4,0.45,0.5,0.6,0.7]:
    add(f'cl{v}',(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>v)&(base.market_return20<-0.05),**basekw)
for v in [-0.02,-0.03,-0.04,-0.05,-0.06,-0.08,-0.10]:
    add(f'mkt{v}',(base.drawdown250<-0.45)&(base.return_250d>0)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<v),**basekw)
for v in [-0.2,0,0.05,0.1,0.2]:
    add(f'r250_{v}',(base.drawdown250<-0.45)&(base.return_250d>v)&(base.rsi14<35)&(base.close_location>0.45)&(base.market_return20<-0.05),**basekw)
for v in [5,8,10,12,15,18,20,25,30]:
    add(f'hold{v}',m0,**{**basekw,'hold':v})
for v in [0,1,2,3]:
    add(f'delay{v}',m0,**{**basekw,'delay':v})
for v in [0.0010,0.0015,0.0020,0.0025,0.0030,0.0040]:
    add(f'cost{v}',m0,**{**basekw,'cost':v})
for v in [0.05,0.07,0.10,0.12,0.15]:
    add(f'stop{v}',m0,**{**basekw,'stop_loss':v})
for v in [0.10,0.15,0.20,0.25,0.30,0.50]:
    add(f'tp{v}',m0,**{**basekw,'take_profit':v})
for v in ['drawdown250','rsi14','amount20','return_250d','close_location','amount_ratio1_20']:
    add(f'score{v}',m0,**{**basekw,'score':v})
for v in [1,2,3]:
    add(f'slots{v}',m0,**{**basekw,'slots':v})
# composite score: deeper dd + stronger reversal + smaller amount
base['comp']=(-base.drawdown250*1.0 + base.close_location*0.5 + base.amount_ratio1_20*0.2 - base.rsi14/100*0.5)
add('score_comp',m0,**{**basekw,'score':'comp'})
r=pd.DataFrame(rows)
r.to_csv('03_结果/robustness_strong.csv',index=False)
show=['variant']
for p in ['train','val','test','full']:
    show += [f'{p}_cagr',f'{p}_dd',f'{p}_n',f'{p}_win',f'{p}_expo']
print(r[show].to_string(index=False))
