#!/usr/bin/env python3
"""14 周期汇总：股票/参数/多周期中文表"""
import os, pandas as pd, numpy as np
ROOT="/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"; RES=os.path.join(ROOT,"03_结果")
# stock
s=pd.read_csv(os.path.join(RES,"周期_汇总.csv"))
s["周期"]=s["周期"].replace({"周线":"周线","月线":"月线"})
s["持有期"]=s["持有期"].astype(str)
# daily row
d=pd.read_csv(os.path.join(RES,"最终_总体.csv")); r=d[d.周期==20].iloc[0]
daily=pd.DataFrame([dict(周期="日线",持有期="20日",信号数=r.信号数,信号日数=r.信号日数,
                         平均分位=0.497902,分位大于50=0.498125,平均超额=r.平均超额,中位超额=r.中位超额,
                         日度平均超额=r.日度平均超额,日度t=r.日度t,聚类t=r.聚类t,
                         分位_2010_2017=0.498756,分位_2018_2026=0.497513,
                         超额_2010_2017=-0.000797,超额_2018_2026=-0.000417)])
s=pd.concat([daily,s],ignore_index=True)
s["结论"]=np.select(
    [(s.周期=="日线"),(s.周期=="周线")&(s.持有期.isin(["4","8","12"])),(s.周期=="周线")&(s.持有期=="26"),
     (s.周期=="月线")&(s.持有期=="3"),(s.周期=="月线")&(s.持有期=="6")],
    ["无效；扣费组合 -5.7%/年","无效；4/8/12 周分位 49.4%–49.5%","均值微正但中位 -5.3%；无实用 edge",
     "无效；3 个月分位 49.1%","约 50%；无 edge"], default="弱正但不稳：12 个月分位 50.7%，中位 -8.4%，前段负/后段正")
s.to_csv(os.path.join(RES,"最终_周期股票.csv"),index=False,encoding="utf-8-sig")
# parameter best
g=pd.read_csv(os.path.join(RES,"参数网格_含q.csv")).rename(columns={"pct20":"pct","ex20":"ex","t20":"t","median20":"med"})
g["参数"]=g.fast.astype(int).astype(str)+"/"+g.slow.astype(int).astype(str)+"/"+g.signal.astype(int).astype(str)
w4=pd.read_csv(os.path.join(RES,"周期_参数网格.csv")); w4=w4[w4.周期=="周线"].copy()
m3=pd.read_csv(os.path.join(RES,"周期_参数网格.csv")); m3=m3[m3.周期=="月线"].copy()
w26=pd.read_csv(os.path.join(RES,"周期_参数网格_周线_26.csv")).copy()
m12=pd.read_csv(os.path.join(RES,"周期_参数网格_月线_12.csv")).copy()
for x in [w4,m3,w26,m12]:
    x["参数"]=x.fast.astype(int).astype(str)+"/"+x.slow.astype(int).astype(str)+"/"+x.signal.astype(int).astype(str)
rows=[]
for lab,x,H,note in [("日线",g,"20日","最优也不够 0.15% 成本"),("周线",w4,"4周","无 edge"),
                     ("周线",w26,"26周","均值正、中位负"),("月线",m3,"3月","无 edge"),
                     ("月线",m12,"12月","仅后期正、中位负")]:
    r=x.loc[x.pct.idxmax()]
    rows.append(dict(周期=lab,持有期=H,参数组合数=len(x),最优参数=r.参数,最优平均分位=r.pct,
                     最优平均超额=r.ex,最优聚类t=r.t,最优中位超额=r.med,
                     分位大于50_5组合数=int((x.pct>0.505).sum()),中位大于0组合数=int((x.med>0).sum()),
                     结论=note))
pd.DataFrame(rows).to_csv(os.path.join(RES,"最终_周期参数.csv"),index=False,encoding="utf-8-sig")
# multi timeframe
parts=[]
for fn,lev in [("多周期_股票_日线.csv","日线+周/月"),("多周期_股票_周线.csv","周线+月"),("多周期_股票_月线.csv","月线+周")]:
    x=pd.read_csv(os.path.join(RES,fn)); x["层级"]=lev
    if "持有周" in x.columns: x["持有期"]=x["持有周"].astype(int).astype(str)+"周"
    elif "持有月" in x.columns: x["持有期"]=x["持有月"].astype(int).astype(str)+"月"
    else: x["持有期"]="20日"
    parts.append(x)
m=pd.concat(parts,ignore_index=True)
m["结论"]=np.select(
    [m.组合.str.contains("周线多头\\+月线多头"),m.组合.str.contains("\\+周线多头$")&(m.层级=="日线+周/月"),
     m.组合.str.contains("基准")&(m.层级=="日线+周/月")],
    ["最差：三重多头共振","更差","基准：无 edge"], default="弱正/不稳定，不能改善")
m.to_csv(os.path.join(RES,"最终_多周期.csv"),index=False,encoding="utf-8-sig")
print(s.to_string(index=False)); print("\n",pd.DataFrame(rows).to_string(index=False)); print("\n",m[["层级","组合","信号数","pct","ex","t_cl","early","late","结论"]].to_string(index=False))
