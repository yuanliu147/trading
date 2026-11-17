"""追加登记：同一指数的大方向与小周期组合，不用旧事件oracle。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BIG=[120,250]; SMALL=[10,20,60]; COSTS=[.0005,.001,.002]
NBOOT=1000; BLOCK=60; RNGSEED=20261030


def simulate(open_,close_,w,cost):
    o=np.asarray(open_,float);c=np.asarray(close_,float);w=np.asarray(w,float)
    factors=np.ones(len(w));prev_close_weight=0.0
    for t in range(len(w)):
        gap=0 if t==0 else o[t]/c[t-1]-1
        overnight_factor=1+prev_close_weight*gap
        before=prev_close_weight*(1+gap)/overnight_factor
        delta=w[t]-before
        # 成交费按交易额计，求解调仓后权重；恒定分数仓位也须漂移再平衡。
        fee=cost*abs(delta)/(1+cost*w[t] if delta>=0 else 1-cost*w[t])
        intra=c[t]/o[t]-1
        factors[t]=overnight_factor*(1-fee)*(1+w[t]*intra)
        prev_close_weight=w[t]*(1+intra)/(1+w[t]*intra)
    factors[-1]*=1-cost*prev_close_weight
    assert np.all(factors>0)
    return factors-1


def drawdown(r):
    v=np.r_[1,np.cumprod(1+r)]
    return float((v/np.maximum.accumulate(v)-1).min())


def bootstrap(x):
    x=np.asarray(x,float);n=len(x)
    if n<2*BLOCK:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(RNGSEED)
    k=int(np.ceil(n/BLOCK));starts=rng.integers(0,n,(NBOOT,k))
    cs=np.r_[0,np.cumsum(np.r_[x,x])]
    # 循环移动块让每个日历位置边际抽样概率相同，末块截断至恰好n日。
    sizes=np.full(k,BLOCK);sizes[-1]=n-BLOCK*(k-1)
    means=(cs[starts+sizes]-cs[starts]).sum(axis=1)/n
    lo,hi=np.quantile(means,[.025,.975])
    p=(1+np.sum(np.abs(means-x.mean())>=abs(x.mean())))/(NBOOT+1)
    return float(lo),float(hi),float(p)


def fdr(ps):
    ps=np.asarray(ps,float);out=np.full(len(ps),np.nan);valid=np.flatnonzero(np.isfinite(ps))
    order=valid[np.argsort(ps[valid])]
    if len(order):out[order]=np.minimum(1,np.minimum.accumulate((ps[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return out


def metrics(r,dates):
    years=max((pd.Timestamp(dates[-1])-pd.Timestamp(dates[0])).days/365.25,1/365.25)
    return float(np.prod(1+r)**(1/years)-1),drawdown(r)


def tests():
    o=np.array([100.,120.,100.]);c=np.array([110.,100.,105.])
    assert np.all(simulate(o,c,np.zeros(3),.001)==0)
    hold=simulate(o,c,np.ones(3),.001)
    assert abs(np.prod(1+hold)-c[-1]/o[0]*(1-.001)/(1+.001))<1e-12
    w=np.array([0.,1.,0.]);r=simulate(o,c,w,0)
    assert abs(r[1]-(100/120-1))<1e-12 # 新入仓不领取隔夜涨幅
    assert abs(r[2]-(100/100-1))<1e-12 # 昨日仓承担隔夜后开盘退出
    assert abs(np.prod(1+simulate(o,c,w,.001))-np.prod(1+r)*(1-.001)/(1+.001))<1e-12
    fractional=simulate([100,100],[110,110],[.5,.5],0)
    assert abs(np.prod(1+fractional)-1.05)<1e-12
    return ['现金0','全持有与价格比等价','隔夜归旧仓','进入不领取隔夜','退出费用复算','分数仓位按真实漂移再平衡']


def main():
    check=tests()
    assets=pd.read_parquet(ROOT/'数据/资产开收盘.parquet')
    assets=assets[assets.kind.ne('行业价格指数')]
    rows=[];curves=[]
    for sym,g in assets.groupby('symbol'):
        g=g.sort_values('date').reset_index(drop=True)
        cal=g.date; c=g.close
        valid=(g.date>=g.live_start.iloc[0])&(g.date>='2010-01-01')&(g.index>=250)
        g2=g[valid];dates=g2.date.to_numpy();o=g2.open.to_numpy();cl=g2.close.to_numpy()
        if len(g2)<250:continue
        endpos=valid.to_numpy()
        for b in BIG:
            direction=c.gt(c.rolling(b).mean())
            single=direction.shift(1).fillna(False).to_numpy()[endpos].astype(float)
            for s in SMALL:
                short=c.gt(c.rolling(s).mean())
                for mode,signal in [('共振追随',direction&short),('大多小弱',direction&~short)]:
                    weights=signal.shift(1).fillna(False).to_numpy()[endpos].astype(float)
                    for cost in COSTS:
                        r=simulate(o,cl,weights,cost)
                        large_r=simulate(o,cl,single,cost)
                        const=simulate(o,cl,np.full(len(cl),weights.mean()),cost)
                        gate_matched=simulate(o,cl,single*(weights.mean()/single.mean() if single.mean() else 0),cost)
                        delta=np.log1p(r)-np.log1p(const)
                        delta_gate=np.log1p(r)-np.log1p(gate_matched)
                        lo,hi,p=bootstrap(delta_gate)
                        annual=len(dates)/max((pd.Timestamp(dates[-1])-pd.Timestamp(dates[0])).days/365.25,1/365.25)
                        cagr,mdd=metrics(r,dates);bagr,bdd=metrics(large_r,dates)
                        signs={}
                        for name,a,z in [('2010-2015','2010-01-01','2015-12-31'),('2016-2020','2016-01-01','2020-12-31'),
                                          ('2015-2017','2015-01-01','2017-12-31'),('2018-2020','2018-01-01','2020-12-31')]:
                            m=(g2.date>=a)&(g2.date<=z)
                            signs[name+'超大周期同暴露']=float(delta_gate[m].mean()*annual) if m.sum()>=100 else np.nan
                        rows.append({'标的代码':sym,'标的':g.name.iloc[0],'类型':g.kind.iloc[0],'大周期':b,'小周期':s,
                          '组合动作':mode,'单边费用':cost,'评价起日':str(pd.Timestamp(dates[0]).date()),
                          '评价末日':str(pd.Timestamp(dates[-1]).date()),'日数':len(dates),'平均仓位':weights.mean(),
                          '净年化':cagr,'最大回撤':mdd,'单独大周期净年化':bagr,'单独大周期最大回撤':bdd,
                          '超单独大周期年化差':cagr-bagr,'超恒定同暴露年化对数差':delta.mean()*annual,
                          '超大周期同暴露年化对数差':delta_gate.mean()*annual,
                          '超大周期同暴露95下界':lo*annual,'超大周期同暴露95上界':hi*annual,
                          '超大周期同暴露双侧p':p,'买入次数':int(((weights>0)&(np.r_[0,weights[:-1]]==0)).sum()),**signs})
                        if cost==.0005:
                            curves.append(pd.DataFrame({'date':dates,'标的':g.name.iloc[0],'大周期':b,'小周期':s,'动作':mode,
                              '净值':np.cumprod(1+r),'大周期同暴露净值':np.cumprod(1+gate_matched),'平均仓位':weights}))
    df=pd.DataFrame(rows);df['统一FDR_q']=fdr(df['超大周期同暴露双侧p'].to_numpy())
    df.to_csv(ROOT/'结果/指数大小周期网格.csv',index=False,encoding='utf-8-sig')
    pd.concat(curves).to_parquet(ROOT/'数据/指数大小周期净值.parquet',index=False,compression='zstd')
    note={'测试':check,'配置数':len(df),'预登记252格是否一致':len(df)==252,'费用假设':'价格指数单边5/10/20bp代理，不是真ETF收益',
      '大多小弱定义':'大均线上但小均线下，立即持有，并非自主画线或未来确认',
      '比较口径':'恒定同仓位与按大均线开关降低权重后同仓位，两种基准。对数年化差不同于CAGR差',
      '2021以后的收益':'未读取未评估','统计':'60日连续移动块bootstrap1000次，双侧中心化p，全部252格FDR；不可称独立验证',
      '冻结追加文件':'证据/补充登记与前轮更正.json'}
    (ROOT/'证据/指数大小周期自检.json').write_text(json.dumps(note,ensure_ascii=False,indent=2))
    print('组合格',len(df),'通过FDR',df.统一FDR_q.le(.1).sum())
    print(df[df.单边费用.eq(.0005)].groupby('组合动作')[['超大周期同暴露年化对数差','超单独大周期年化差']].median().to_string())

if __name__=='__main__':main()
