import pandas as pd, numpy as np, json
out=[]
def chk(name,cond,detail=''):
    out.append((name,'PASS' if cond else 'FAIL',detail))
hfq=pd.read_csv('03_结果/最终_每日权益.csv',parse_dates=['date']).set_index('date').equity
act=pd.read_csv('03_结果/实际价_每日权益.csv',parse_dates=['date']).set_index('date').equity
ht=pd.read_csv('03_结果/最终_逐笔.csv'); at=pd.read_csv('03_结果/实际价_逐笔.csv')
def cagr(e,cap): return (e.iloc[-1]/cap)**(365.25/(e.index[-1]-e.index[0]).days)-1
chk('hfq CAGR ~40.0%',abs(cagr(hfq,1)-0.400)<0.005,f'{cagr(hfq,1)*100:.2f}%')
chk('hfq maxDD ~-16.0%',abs((hfq/hfq.cummax()-1).min()+0.160)<0.005,f'{(hfq/hfq.cummax()-1).min()*100:.2f}%')
chk('hfq trades 99',len(ht)==99,len(ht))
chk('actual CAGR ~42.6%',abs(cagr(act,100000)-0.426)<0.006,f'{cagr(act,100000)*100:.2f}%')
chk('actual maxDD ~-15.9%',abs((act/act.cummax()-1).min()+0.159)<0.005,f'{(act/act.cummax()-1).min()*100:.2f}%')
chk('actual trades 99',len(at)==99,len(at))
chk('no same-day exit',(pd.to_datetime(ht.entry_date)!=pd.to_datetime(ht.exit_date)).all())
chk('entry after signal >=3 sessions',(pd.to_datetime(ht.entry_date)>pd.to_datetime(ht.signal_date)).all())
chk('annual all positive',(pd.read_csv('03_结果/最终_年度.csv',index_col=0).iloc[:,0]>0).all())
s=pd.read_csv('03_结果/当前信号_2026-10-23.csv')
chk('current top2',s.iloc[0].thscode=='603778.SH' and s.iloc[1].thscode=='605376.SH',str(s.iloc[:2].thscode.tolist()))
print('\n'.join(f'{a:28s} {b:4s} {c}' for a,b,c in out))
print('\nALL PASS' if all(b=='PASS' for _,b,_ in out) else '\nSOME FAIL')
