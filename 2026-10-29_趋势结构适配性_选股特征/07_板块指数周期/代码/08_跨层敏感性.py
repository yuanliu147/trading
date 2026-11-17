"""预先限定的跨层事件主格：块长20/60/120与固定摩擦四档敏感性。"""
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
COSTS=[.00154,.00254,.00354,.00554]
BLOCKS=[20,60,120]
NBOOT=1999

def main():
    x=pd.read_csv(ROOT/'结果/04_全部格检验.csv')
    selected=x[x.研究.eq('跨层事件')&x.半区.eq('全部')&x.大周期状态.eq('红')
      &x.时段.isin(['主窗2015-2020','2015-2017','2018-2020'])
      &x.指标.isin(['相对全A','相对留一行业'])]
    con=duckdb.connect();con.execute('SET threads=2')
    ids=','.join(str(int(i)) for i in selected.格编号)
    daily=con.execute(f"SELECT * FROM read_parquet('{ROOT}/结果/04_日级聚合.parquet') WHERE 格编号 IN ({ids}) ORDER BY 格编号,日期").fetch_df()
    rows=[]
    for _,rec in selected.iterrows():
        d=daily[daily.格编号.eq(rec.格编号)]
        n=len(d);s=d.收益和.to_numpy(float);count=d.有效事件数.to_numpy(float)
        if count.sum()==0:continue
        base=s.sum()/count.sum()
        for block in BLOCKS:
            if n<2*block:continue
            rng=np.random.default_rng(20261030+block)
            k=int(np.ceil(n/block));starts=rng.integers(0,n,(NBOOT,k));lengths=np.full(k,block);lengths[-1]=n-block*(k-1)
            ss=np.r_[0,np.cumsum(np.r_[s,s])];nn=np.r_[0,np.cumsum(np.r_[count,count])]
            den=(nn[starts+lengths]-nn[starts]).sum(axis=1)
            sampled=(ss[starts+lengths]-ss[starts]).sum(axis=1)
            means=np.divide(sampled,den,out=np.full(NBOOT,np.nan),where=den>0)
            errors=means-base;errors=errors[np.isfinite(errors)]
            if len(errors)<NBOOT*.8:continue
            qlo,qhi=np.quantile(errors,[.025,.975])
            for cost in COSTS:
                mean=base-(cost-.00254)
                p=(1+(np.abs(errors)>=abs(mean)).sum())/(len(errors)+1)
                rows.append({'事件':rec.事件,'大周期':rec.大周期,'持有日':rec.持有日,'时段':rec.时段,'指标':rec.指标,
                  '块长':block,'往返摩擦':cost,'均值':mean,'95下界':mean-qhi,'95上界':mean-qlo,
                  '双侧p':p,'有效事件数':int(count.sum()),'有效信号日数':int((count>0).sum())})
    df=pd.DataFrame(rows);p=df.双侧p.to_numpy();idx=np.argsort(p)
    q=np.minimum(1,np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    df['本敏感性族FDR_q']=np.nan;df.loc[idx,'本敏感性族FDR_q']=q
    df.to_csv(ROOT/'结果/跨层事件块长与成本敏感性.csv',index=False,encoding='utf-8-sig')
    note={'格数':len(df),'输入主格数':len(selected),'限定方式':'全部股票，红状态，全部三事件/四周期/三持有/三主分段/二超额指标，没有看结果后择优',
      '循环块长':BLOCKS,'bootstrap次数':NBOOT,'成本往返假设':COSTS,'公式':'改变固定摩擦只平移事件净收益，无法替代逐股冲击成本/最低佣金/组合部署',
      '正边际下界大于0格数':int(df['95下界'].gt(0).sum()),'缺数据处理':'有事件分母则算，无事件日count0不是填零收益；不足块长/抽样标不可判',
      '检验范围':'这里只做预定主事件格敏感性，其余排序格的敏感性未完成，不能借此给它们验证声明'}
    (ROOT/'证据/跨层敏感性核验.json').write_text(json.dumps(note,ensure_ascii=False,indent=2))
    print('敏感性格',len(df),'95%正下界格',note['正边际下界大于0格数'])

if __name__=='__main__':main()
