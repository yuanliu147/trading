# -*- coding: utf-8 -*-
"""对最终系统逐笔交易做首周、持有路径、卖出后20日复盘。"""
import pickle,numpy as np,pandas as pd,os
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; codes=P['meta']['codes']; ci={c:i for i,c in enumerate(codes)}; di={d:i for i,d in enumerate(dates)}
C=P['close'].values; O=P['open'].values; H=P['high'].values; L=P['low'].values
rows=[]
for per,f in [('训练','逐笔_含退市_训练.csv'),('检验','逐笔_含退市_检验.csv')]:
    T=pd.read_csv(os.path.join(BASE,'03_结果',f))
    for _,r in T.iterrows():
        j=ci.get(r.code); 
        if j is None: continue
        i=di.get(pd.Timestamp(r.entry_date)); x=di.get(pd.Timestamp(r.exit_date))
        if i is None or x is None: continue
        seg=C[i:x+1,j]
        first5=C[min(i+4,x),j]/r.entry-1 if x>=i+4 else np.nan
        first10=C[min(i+9,x),j]/r.entry-1 if x>=i+9 else np.nan
        mae=np.nanmin(L[i:x+1,j]/r.entry-1); mfe=np.nanmax(H[i:x+1,j]/r.entry-1)
        post20=C[min(x+20,len(dates)-1),j]/C[x,j]-1 if x+1<len(dates) and np.isfinite(C[x,j]) else np.nan
        rows.append(dict(期=per,代码=r.code,买入日=r.entry_date,卖出日=r.exit_date,收益=r.ret,原因=r.reason,持有=r.hold,
            首5日=first5,首10日=first10,最大不利=mae,最大有利=mfe,卖出后20日=post20,仓位=r.weight))
R=pd.DataFrame(rows)
R.to_csv(os.path.join(BASE,'03_结果/交易复盘_明细.csv'),index=False,encoding='utf-8-sig')
print('交易数',len(R))
# 首周分组
def grp(col,bins,labels):
    return pd.cut(R[col],bins=bins,labels=labels,include_lowest=True)
R['首周组']=grp('首5日',[-1,-0.05,-0.02,0.02,0.05,1],['<-5%','-5~-2%','-2~+2%','+2~+5%','>+5%'])
summ=R.groupby(['期','首周组'],observed=True).agg(笔数=('收益','size'),最终均值=('收益','mean'),最终中位=('收益','median'),
    最终胜率=('收益',lambda s:(s>0).mean()),平均最大不利=('最大不利','mean'),平均最大有利=('最大有利','mean'))
summ.to_csv(os.path.join(BASE,'03_结果/交易复盘_首周.csv'),encoding='utf-8-sig')
print('\n首周'); print(summ.round(4).to_string())
# 卖出后
R['卖后组']=pd.cut(R['卖出后20日'],[-1,-0.1,-0.03,0.03,0.1,2],labels=['<-10%','-10~-3%','-3~+3%','+3~+10%','>+10%'])
post=R.groupby(['期','卖后组'],observed=True).agg(笔数=('收益','size'),卖后均值=('卖出后20日','mean'),卖后胜率=('卖出后20日',lambda s:(s>0).mean()))
post.to_csv(os.path.join(BASE,'03_结果/交易复盘_卖出后.csv'),encoding='utf-8-sig')
print('\n卖出后'); print(post.round(4).to_string())
print('\n卖出后20日>5%占比',round((R['卖出后20日']>0.05).mean(),3),'均值',round(R['卖出后20日'].mean(),4))
print('卖出后20日<-5%占比',round((R['卖出后20日']<-0.05).mean(),3))
# 原因
print('\n原因'); print(R.groupby(['期','原因']).agg(笔数=('收益','size'),均值=('收益','mean'),胜率=('收益',lambda s:(s>0).mean())).round(4).to_string())
# 不同持有期反事实
print('\n不同持有期（按实际入场日）')
for h in [5,10,15,20,25,30]:
    vals=[]
    T=pd.read_csv(os.path.join(BASE,'03_结果/逐笔_含退市_检验.csv'))
    for _,r in T.iterrows():
        j=ci.get(r.code); i=di.get(pd.Timestamp(r.entry_date))
        if j is None or i is None: continue
        xx=min(i+h-1,len(dates)-1)
        vals.append(C[xx,j]/r.entry-1-0.003)
    print(h,round(np.mean(vals),4),round(np.median(vals),4),round(np.mean(np.array(vals)>0),4))
