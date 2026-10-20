# -*- coding: utf-8 -*-
"""梯度提升树交叉检验：哪些上涨趋势特征有增量信息。训练2017-2021，检验2022-2026。"""
import pickle,numpy as np,pandas as pd,os,time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
P=pickle.load(open(os.path.join(BASE,'01_数据/全A面板_2016_2026_含退市.pkl'),'rb'))
dates=P['meta']['dates']; U=P['up_struct'].values; F20=P['fwd20'].values
AMT=P['amt20'].values; AGE=P['trend_age'].values; C=P['close'].values
bench=np.nanmean(F20,axis=1)[:,None]; EXC=F20-bench
date_ok=(dates>=pd.Timestamp('2017-10-13'))[:,None]
yr=np.array([d.year for d in dates]); period=np.where(yr<=2021,0,1)
valid=U&np.isfinite(F20)&(AMT>=3e7)&(C>=2)&(AGE>=5)&date_ok
feats=['trend_age','dist_ma20','dd60','er20','ma60_slope','vol20','atr_pct','amt_trend','vr5_20','rsi14',
 'ret20_hist_z','dist_high250','own_base20','sector_breadth20','sector_ret20','sector_rs','stock_sector_z',
 'sw_above60','market_breadth20','market_sent','amt_ratio20','gap','upper','close_pos','lu_count20','vp_corr20','gap_count20']
X=np.column_stack([np.broadcast_to(P[f].values[:,None],F20.shape) if P[f].values.ndim==1 else P[f].values for f in feats]).astype('float32')
y=(EXC>0).astype('int8')
rng=np.random.default_rng(42)
tr=np.where(valid&(period[:,None]==0))
te=np.where(valid&(period[:,None]==1))
def sample(idx,n):
    if len(idx)>n: idx=rng.choice(idx,n,replace=False)
    return idx
tr_i=sample(tr[0]*0+tr[0],0) if False else None
# flatten indices
tr_flat=np.where((valid&(period[:,None]==0)).ravel())[0]
te_flat=np.where((valid&(period[:,None]==1)).ravel())[0]
tr_flat=sample(tr_flat,900000); te_flat=sample(te_flat,900000)
Xtr=X.reshape(-1,len(feats))[tr_flat]; ytr=y.ravel()[tr_flat]
Xte=X.reshape(-1,len(feats))[te_flat]; yte=y.ravel()[te_flat]
print('train',Xtr.shape,'test',Xte.shape,'pos',ytr.mean(),yte.mean(),flush=True)
m=HistGradientBoostingClassifier(max_iter=200,max_depth=6,learning_rate=0.05,random_state=42,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20)
t0=time.time(); m.fit(Xtr,ytr); print('fit',time.time()-t0,flush=True)
for name,XX,yy in [('训练',Xtr,ytr),('检验',Xte,yte)]:
    p=m.predict_proba(XX)[:,1]; print(name,'AUC',roc_auc_score(yy,p),flush=True)
# 检验期十分位
p=m.predict_proba(Xte)[:,1]
# reconstruct excess for test sample
exc_flat=EXC.ravel()[te_flat]
q=pd.qcut(p,10,labels=False,duplicates='drop')
rows=[]
for g in range(q.max()+1):
    sel=(q==g); rows.append(dict(十分位=g+1,样本=int(sel.sum()),预测概率=p[sel].mean(),超额均值=exc_flat[sel].mean(),
        绝对收益均值=F20.ravel()[te_flat][sel].mean(),胜率=(exc_flat[sel]>0).mean()))
DEC=pd.DataFrame(rows); DEC.to_csv(os.path.join(BASE,'03_结果/模型_十分位.csv'),index=False,encoding='utf-8-sig')
print(DEC.round(4).to_string(index=False))
# 置换重要性（子样本）
sub=sample(np.arange(len(Xte)),120000); Xs=Xte[sub]; ys=yte[sub]
pi=permutation_importance(m,Xs,ys,n_repeats=4,random_state=42,scoring='roc_auc',n_jobs=-1)
IMP=pd.DataFrame({'特征':feats,'重要性':pi.importances_mean,'标准差':pi.importances_std}).sort_values('重要性',ascending=False)
IMP.to_csv(os.path.join(BASE,'03_结果/模型_特征重要性.csv'),index=False,encoding='utf-8-sig')
print(IMP.round(5).to_string(index=False))
