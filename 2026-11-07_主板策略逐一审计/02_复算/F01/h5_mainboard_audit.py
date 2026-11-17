#!/usr/bin/env python3
# F01 月度轮动独立复算：主板×含退市 四变体
# 规则镜像 analysis/h5_defensive_ew.py（全池等权、20交易日调仓、F1/F2负向过滤、age>=250、
# 往返0.15%×换手、f+2起计收益），数据改用 trading_finally 冻结面板（baostock含退市，后复权signal_close）。
# 调仓相位锚定2016-10-13（与原DuckDB日历一致），rn%20==0。
import duckdb, numpy as np, pandas as pd, json, os

PANEL = '/Users/admin/WorkSpace/trading/trading_finally/data/legacy_full_a_panel.parquet'
OUT = os.path.dirname(os.path.abspath(__file__))
COST_RT = 0.0015
con = duckdb.connect()

# 交易日历（面板全局），相位锚2016-10-13
cal = con.execute(f"SELECT DISTINCT date FROM '{PANEL}' WHERE date>='2016-10-13' ORDER BY date").df()['date']
cal = pd.to_datetime(cal)
rebal = cal[(np.arange(1, len(cal)+1) % 20) == 0]
rebal_str = "','".join(d.strftime('%Y-%m-%d') for d in rebal)

# 形成日指标（仅调仓日），含主板标记/退市日/ST
SQL_FORM = f"""
WITH px AS (
  SELECT code, date, is_main_board, out_date, is_st, amount,
         signal_close / LAG(signal_close,20) OVER w - 1.0 AS r20,
         signal_close / MAX(signal_high) OVER w20 - 1.0 AS dd,
         AVG(volume) OVER w5 / NULLIF(AVG(volume) OVER w20v,0) AS vratio,
         COUNT(*) OVER (PARTITION BY code ORDER BY date ROWS UNBOUNDED PRECEDING) AS age
  FROM '{PANEL}'
  WINDOW w AS (PARTITION BY code ORDER BY date),
         w20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
         w5 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
         w20v AS (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
SELECT date, code, is_main_board, out_date, is_st, amount, r20, dd, vratio
FROM px WHERE date IN ('{rebal_str}') AND age>=250 AND r20 IS NOT NULL
"""
form = con.execute(SQL_FORM).df()
form['date'] = pd.to_datetime(form['date'])

# 日收益（全窗，后复权），age>=250
SQL_DAILY = f"""
WITH px AS (
  SELECT code, date, is_main_board, out_date,
         signal_close / LAG(signal_close) OVER w - 1.0 AS ret1,
         COUNT(*) OVER (PARTITION BY code ORDER BY date ROWS UNBOUNDED PRECEDING) AS age
  FROM '{PANEL}'
  WINDOW w AS (PARTITION BY code ORDER BY date))
SELECT date, code, is_main_board, out_date, ret1 FROM px WHERE age>=250 AND date>='2017-10-01'
"""
daily = con.execute(SQL_DAILY).df()
daily['date'] = pd.to_datetime(daily['date'])
con.close()

# 股票级属性（是否曾退市、是否主板）
meta = daily.groupby('code').agg(mb=('is_main_board','max'), out=('out_date','first')).reset_index()
meta['delisted'] = meta['out'].notna() & (meta['out'].astype(str).str.len()>0) & (meta['out'].astype(str)!='None')
delisted_codes = set(meta.loc[meta['delisted'],'code'])
mb_codes = set(meta.loc[meta['mb']==True,'code']) if meta['mb'].dtype==bool else set(meta.loc[meta['mb']==1,'code'])

ret_all = daily.pivot(index='date', columns='code', values='ret1').sort_index()

def run_variant(name, code_filter):
    codes = [c for c in ret_all.columns if code_filter(c)]
    ret = ret_all[codes]
    ret = ret[ret.index >= '2017-11-01']
    calv = ret.index; n=len(calv)
    fdates = np.sort(form['date'].unique())
    fpos = [p for p in calv.get_indexer(pd.DatetimeIndex(fdates)) if 0<=p<n-2]
    col_idx = {c:i for i,c in enumerate(ret.columns)}
    fset = set(codes)
    members=[]; turns=[]; prev=None
    for f in fpos:
        g = form[(form['date']==calv[f]) & (form['code'].isin(fset))]
        if len(g)==0: members.append(set()); turns.append(0.5); continue
        thr = g['r20'].quantile(0.8)
        excl = (g['r20']>=thr) | ((g['dd']>=-0.15)&(g['dd']<=-0.05)&(g['vratio']>=1.6))
        cur = set(g.loc[~excl,'code'])
        turn = 0.5 if prev is None else 1-len(prev&cur)/max(len(cur),1)
        members.append(cur); turns.append(turn); prev=cur
    R = ret.values; port=np.zeros(n); period_ret=[]
    for k,f in enumerate(fpos):
        if len(members[k])<50: continue
        end = (fpos[k+1]+1) if k+1<len(fpos) else n-1
        end=min(end,n-1)
        if f+2>end: continue
        mi=np.array([col_idx[c] for c in members[k] if c in col_idx])
        if len(mi)==0: continue
        seg=R[f+2:end+1][:,mi]
        port[f+2:end+1]=np.nanmean(seg,axis=1)
        port[f+1]-=COST_RT*turns[k]
        period_ret.append(float((1+port[f+1:end+1]).prod()-1))
    start=fpos[0]+1
    dates=calv[start:]; port=pd.Series(port[start:],index=dates)
    eq=(1+port).cumprod()
    def mets(s):
        r=s.dropna(); nn=len(r); e=(1+r).cumprod()
        dd=float((e/e.cummax()-1).min()); ann=float(e.iloc[-1]**(252/nn)-1)
        return ann,dd
    res={}
    for tag,mask in [('train',(dates>='2017-11-01')&(dates<='2022-12-31')),
                     ('oos',(dates>='2023-01-01')&(dates<='2026-10-29')),
                     ('full',dates>='2017-11-01')]:
        a,d=mets(port[mask]); res[tag]={'ann':round(a,4),'mdd':round(d,4),'calmar':round(a/abs(d),3) if d<0 else None}
    pr=np.array(period_ret)
    res['period_win']=round(float((pr>0).mean()),4)
    res['ann_turn']=round(float(np.mean(turns)*252/20),2)
    res['mean_members']=round(float(np.mean([len(m) for m in members])),0)
    res['n_codes']=len(codes)
    eq.to_csv(os.path.join(OUT,f'h5_audit_{name}_equity.csv'))
    return res

variants = {
 '全A含退市': lambda c: not c.endswith('.BJ'),
 '全A无退市': lambda c: (not c.endswith('.BJ')) and c not in delisted_codes,
 '主板含退市': lambda c: c in mb_codes,
 '主板无退市': lambda c: c in mb_codes and c not in delisted_codes,
}
out={}
for name,filt in variants.items():
    out[name]=run_variant(name,filt)
    print(name, json.dumps(out[name],ensure_ascii=False))
json.dump(out, open(os.path.join(OUT,'h5_audit_summary.json'),'w'), ensure_ascii=False, indent=1)
print('delisted_codes:',len(delisted_codes),'mb_codes:',len(mb_codes))