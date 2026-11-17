"""Lo-MacKinlay有限样本修正、异方差稳健方差比；不是策略回测。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import duckdb
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[1]
SCALES=[2,5,10,20,60,120,250]
PERIODS={'全窗2010-2020':('2010-01-01','2020-12-31'),
 '2010-2012':('2010-01-01','2012-12-31'), '2013-2015':('2013-01-01','2015-12-31'),
 '2016-2017':('2016-01-01','2017-12-31'), '2018-2020':('2018-01-01','2020-12-31'),
 '行业主窗2015-2020':('2015-01-01','2020-12-31')}

def longest_segment(x):
    x=np.asarray(x,float)
    good=np.isfinite(x)
    edges=np.diff(np.r_[False,good,False].astype(int))
    starts,ends=np.where(edges==1)[0],np.where(edges==-1)[0]
    if len(starts)==0: return x[:0]
    i=np.argmax(ends-starts)
    return x[starts[i]:ends[i]]

def variance_ratios(x, scales=SCALES):
    x=np.asarray(x,float); n=len(x); x=x-x.mean() if n else x
    s2=np.dot(x,x)
    res=[]
    for q in scales:
        if n<5*q or s2<=1e-16:
            res.append((q,np.nan,np.nan,np.nan)); continue
        cs=np.r_[0,np.cumsum(x)]
        qsum=cs[q:]-cs[:-q]
        denominator=q*(n-q+1)*(1-q/n)
        vr=np.dot(qsum,qsum)/denominator/(s2/(n-1))
        square=x*x
        theta=sum((2*(q-j)/q)**2*np.dot(square[j:],square[:-j])/(s2*s2)
                  for j in range(1,q))
        z=(vr-1)/np.sqrt(theta) if theta>0 else np.nan
        p=2*ndtr(-abs(z)) if np.isfinite(z) else np.nan
        res.append((q,vr,z,p))
    return res

def fdr(p):
    out=np.full(len(p),np.nan); good=np.flatnonzero(np.isfinite(p))
    order=good[np.argsort(p[good])]
    if len(order):
        adjust=np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
        out[order]=np.minimum(adjust,1)
    return out

def check():
    rng=np.random.default_rng(908)
    x=rng.normal(size=20000)
    white=variance_ratios(x,[2,5,20])
    assert all(abs(v-1)<0.12 for q,v,z,p in white)
    y=x[1:]+0.8*x[:-1]; negative=x[1:]-0.8*x[:-1]
    assert variance_ratios(y,[5])[0][1]>1.4
    assert variance_ratios(negative,[5])[0][1]<0.5
    arr=np.r_[np.ones(7),np.nan,np.arange(12)]
    assert len(longest_segment(arr))==12
    assert np.isnan(variance_ratios(x[:300],[120])[0][1])
    # 独立直接循环实现核对公式，不比较期望随机输出。
    short=rng.normal(size=600); mu=short.mean(); q=20
    direct=sum((sum(short[i:i+q])-q*mu)**2 for i in range(len(short)-q+1))
    direct/=q*(len(short)-q+1)*(1-q/len(short))
    direct/=np.var(short,ddof=1)
    assert abs(direct-variance_ratios(short,[q])[0][1])<1e-12
    return {'白噪声方差比':[float(v) for q,v,z,p in white], '独立逐窗公式偏差':abs(direct-variance_ratios(short,[q])[0][1]),
      '测试':['白噪声接近1','正相关方向>1','负相关方向<1','缺失日不压缩','小样本NA','独立朴素实现相符']}

def add(rows,name,level,series,dates,valid_start='2010-01-01'):
    dates=pd.DatetimeIndex(dates)
    for period,(lo,hi) in PERIODS.items():
        lo=max(lo,valid_start)
        mask=(dates>=lo)&(dates<=hi)
        sample=np.asarray(series)[mask]; longest=longest_segment(sample)
        if len(sample)==0:continue
        good=np.isfinite(sample); edges=np.diff(np.r_[False,good,False].astype(int))
        starts,ends=np.where(edges==1)[0],np.where(edges==-1)[0]
        if len(starts):
            i=np.argmax(ends-starts); actual_start=str(dates[mask][starts[i]].date())
            actual_end=str(dates[mask][ends[i]-1].date())
        else: actual_start=actual_end=None
        for q,vr,z,p in variance_ratios(longest):
            rows.append((level,name,period,q,len(sample),len(longest),vr,z,p,actual_start,actual_end))

def main():
    tests=check()
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='3GB'")
    path=ROOT/'数据/个股与行业.parquet'
    dates=con.execute(f"SELECT DISTINCT date FROM read_parquet('{path}') ORDER BY date").fetch_df().date
    calendar=pd.DatetimeIndex(dates)
    names=con.execute(f"SELECT DISTINCT symbol FROM read_parquet('{path}') ORDER BY symbol").fetch_df().symbol.tolist()
    rows=[]
    for b in range(0,len(names),500):
        group=names[b:b+500];codes=','.join("'"+s+"'" for s in group)
        df=con.execute(f"""SELECT symbol,date,hfq_close,日收益,行业残差,td_since_list
          FROM read_parquet('{path}') WHERE symbol IN ({codes}) ORDER BY symbol,date""").fetch_df()
        for sym,g in df.groupby('symbol',sort=False):
            g=g.set_index('date').reindex(calendar)
            # 只用于路径描述的成分成熟度过滤，不形成交易选股规则。
            raw=np.log(g.hfq_close.where(g.hfq_close>0)).diff().where(g.td_since_list>=60)
            residual=g['行业残差'].where(g.td_since_list>=60)
            add(rows,sym,'个股原始',raw,calendar)
            add(rows,sym,'个股行业残差',residual,calendar)
        print('完成股票',min(b+500,len(names)),'诊断单元',len(rows),flush=True)
    assets=pd.read_parquet(ROOT/'数据/资产开收盘.parquet')
    for sym,g in assets.groupby('symbol'):
        g=g.sort_values('date').set_index('date')
        # 缺失行情在日历中保留NA，不串成连续市场日。
        r=np.log(g.close.reindex(calendar)).diff()
        lev={'宽基指数':'宽基指数','行业价格指数':'官方行业指数','全A合成':'全A等权合成'}[g.kind.iloc[0]]
        add(rows,g.name.iloc[0],lev,r,calendar,str(pd.Timestamp(g.live_start.iloc[0]).date()))
    synth=pd.read_parquet(ROOT/'数据/行业合成开收盘.parquet')
    for sym,g in synth.groupby('symbol'):
        g=g.sort_values('date').set_index('date')
        r=np.log(g.close.reindex(calendar)).diff()
        add(rows,g.name.iloc[0],'行业等权合成',r,calendar,'2015-01-01')
    df=pd.DataFrame(rows,columns=['层级','标的','窗口','聚合日数','日历样本数','连续可用日数','方差比','异方差稳健z','双侧p','实测段起','实测段止'])
    df['统一FDR_q']=fdr(df['双侧p'].to_numpy())
    df.to_csv(ROOT/'结果/尺度图逐标的.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (level,period,q),g in df.groupby(['层级','窗口','聚合日数']):
        finite=g[g.方差比.notna()]
        # 常样本只用于跨q可比性，不把长期存活作为部署选股条件。
        common=finite[finite.连续可用日数>=1250]
        summary.append({'层级':level,'窗口':period,'聚合日数':q,'标的总数':len(g),
         '可评估标的数':len(finite),'方差比中位数':finite.方差比.median(),
         '方差比P25':finite.方差比.quantile(.25),'方差比P75':finite.方差比.quantile(.75),
         '方差比大于1比例':(finite.方差比>1).mean() if len(finite) else np.nan,
         'FDR显著正比例':((finite.方差比>1)&finite.统一FDR_q.le(.1)).mean() if len(finite) else np.nan,
         'FDR显著负比例':((finite.方差比<1)&finite.统一FDR_q.le(.1)).mean() if len(finite) else np.nan,
         '1250日共同样本数':len(common),'共同样本方差比中位数':common.方差比.median()})
    sm=pd.DataFrame(summary)
    sm.to_csv(ROOT/'结果/层级尺度地图.csv',index=False,encoding='utf-8-sig')
    tests.update({'全部诊断单元':len(df),'有效方差比检验数':int(df.方差比.notna().sum()),
      '最大数据日':str(calendar.max().date()),'是否研究2021以后':False,
      '缺失处理':'每股在共同日历中取最长完整可用段，停牌0收益保留，缺失不压缩。n>=5q。',
      '不能得出的结论':['VR>1不等于策略扣费能赚钱','中位VR不是群组置信区间','异方差z未对跨股票共同冲击给出群组独立性','1250共同样本有存续选择，仅路径诊断不能变为选股']})
    (ROOT/'证据/尺度图公式自检.json').write_text(json.dumps(tests,ensure_ascii=False,indent=2))
    print(sm[sm.窗口.isin(['全窗2010-2020','行业主窗2015-2020'])].to_string(index=False))

if __name__=='__main__':main()
