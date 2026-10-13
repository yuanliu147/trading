"""板块动量轮动回测：R1纯轮动(永远满仓) vs R2(轮动+系统闸门)
每月(20交易日)调仓: 板块按60日等权涨幅排名取前5, 每板块取60日动量最强且流动性合格的2只
"""
import duckdb, json, pandas as pd, numpy as np

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'

WINDOWS = [('2018-04-13','2019-04-15'),('2018-06-15','2019-06-18'),('2018-12-14','2019-12-16'),
           ('2020-03-20','2021-03-22'),('2022-01-07','2023-01-09'),('2022-06-17','2023-06-19'),
           ('2022-09-09','2023-09-12'),('2023-06-30','2024-07-01'),('2023-10-27','2024-10-28'),
           ('2024-03-01','2025-03-03'),('2024-09-20','2025-09-22'),('2025-01-03','2026-01-05')]

def load_data():
    con = duckdb.connect(DB, read_only=True)
    smap = json.load(open('sector_map.json'))
    df = con.execute("""
        WITH w AS (
            SELECT thscode, date, close, amount,
                   lag(close,60) OVER (PARTITION BY thscode ORDER BY date) AS c60,
                   avg(amount) OVER (PARTITION BY thscode ORDER BY date ROWS 60 PRECEDING) AS amt60,
                   count(*) OVER (PARTITION BY thscode ORDER BY date) AS nbar,
                   lag(close) OVER (PARTITION BY thscode ORDER BY date) AS pc
            FROM v_daily_qfq)
        SELECT thscode, date, close, c60, amt60, nbar, close/pc-1 AS ret1
        FROM w""").df()
    df['date'] = pd.to_datetime(df['date'])
    df['sector'] = df['thscode'].map(smap)
    df['r60'] = df['close']/df['c60'] - 1
    hs = pd.read_csv('hs300_daily.csv', parse_dates=['date'])
    gate = pd.read_csv('gate_weekly.csv', parse_dates=['week_end'])
    return df, hs, gate

def exposure_map(hs, gate):
    g = gate.set_index('week_end')['gate']
    em = {}
    for d in hs['date']:
        we = (d - pd.Timedelta(days=d.weekday())) - pd.Timedelta(days=3)
        past = g[g.index <= we]
        gg = past.iloc[-1] if len(past) else 'unknown'
        em[d] = {'green':1.0,'yellow':0.5}.get(gg, 0.0)
    return em

def run_window(df, cal, start, end, expo=None, sel='r60'):
    """expo=None → R1满仓; 否则R2按闸门敞口"""
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    days = [d for d in cal if start <= d <= end]
    rebal = set(days[::20]) | {days[0]}
    by_date = {d: g for d, g in df.groupby('date')}
    holdings = []   # list of thscode
    pending = []
    rebal_next = None
    eq = 1.0; curve = []
    for d in days:
        g = by_date.get(d)
        if g is None:
            curve.append((d, eq)); continue
        if rebal_next:                      # 前一日选出的组合今日生效(避免前视)
            holdings = pending
            rebal_next = None
        if d in rebal:
            el = g[g['sector'].notna() & (g['nbar']>=250) & (g['close']>=2)
                   & (g['amt60']>=2e8) & g['r60'].notna()
                   & (~g['thscode'].str.endswith('.BJ'))]
            sec = el.groupby('sector')['r60'].mean()
            top5 = sec.nlargest(5).index
            picks = []
            for s in top5:
                picks += el[el['sector']==s].nlargest(2, sel)['thscode'].tolist()
            pending = picks
            rebal_next = 'next'
        if holdings:
            h = g[g['thscode'].isin(holdings)]
            day_ret = h['ret1'].mean()
            if np.isnan(day_ret): day_ret = 0.0
        else:
            day_ret = 0.0
        e = 1.0 if expo is None else expo.get(d, 0.0)
        eq *= (1 + day_ret * e)
        curve.append((d, eq))
    cur = pd.DataFrame(curve, columns=['date','equity'])
    ret = eq - 1
    mdd = (cur['equity']/cur['equity'].cummax()-1).min()
    return ret, mdd, cur

def main():
    df, hs, gate = load_data()
    cal = sorted(hs['date'].tolist())
    expo = exposure_map(hs, gate)
    hsx = hs.set_index('date')
    print(f"{'窗口':<12} {'R1纯轮动':>9} {'R2轮动+闸门':>11} {'系统(形态)':>10} {'沪深300':>9}  R1回撤")
    sysres = pd.read_csv('analog_results.csv').set_index('analog')
    rows = []
    for s, e in WINDOWS:
        r1, mdd1, c1 = run_window(df, cal, s, e)
        r2, mdd2, c2 = run_window(df, cal, s, e, expo)
        r1b, _, _ = run_window(df, cal, s, e, sel='amt60')
        r2b, _, _ = run_window(df, cal, s, e, expo, sel='amt60')
        b = hsx[(hsx.index>=s)&(hsx.index<=e)]
        bret = b['close'].iloc[-1]/b['close'].iloc[0]-1
        sysret = sysres.loc[s,'ret12m'] if s in sysres.index else np.nan
        rows.append({'window':s,'R1':r1,'R2':r2,'R1b':r1b,'R2b':r2b,'sys':sysret,'hs300':bret,'mdd1':mdd1,'mdd2':mdd2})
        print(f"{s:<12} {r1*100:>8.1f}% {r2*100:>10.1f}% | 龙头R1 {r1b*100:>7.1f}% 龙头R2 {r2b*100:>7.1f}% | 系统 {sysret*100:>6.1f}% 沪深300 {bret*100:>7.1f}%")
    res = pd.DataFrame(rows)
    res.to_csv('rotation_results.csv', index=False)
    print("\n=== 汇总 ===")
    for c in ['R1','R2','R1b','R2b','sys','hs300']:
        print(f"{c}: 胜率 {(res[c]>0).mean()*100:.0f}%  中位 {res[c].median()*100:+.1f}%  均值 {res[c].mean()*100:+.1f}%")

if __name__ == '__main__':
    main()
