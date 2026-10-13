"""在5个历史相似时段运行回测：每个时段从模拟日起跑12个月"""
import pandas as pd, numpy as np
from engine import Engine, EQUITY0

ANALOGS = ['2018-06-15', '2020-03-20', '2022-06-17', '2023-10-27', '2024-09-20',  # 特征匹配
           '2018-04-13', '2018-12-14', '2022-01-07', '2022-09-09', '2023-06-30', '2024-03-01', '2025-01-03']  # 深红扩充

def main():
    gate = pd.read_csv('gate_weekly.csv', parse_dates=['week_end'])
    sig = pd.read_csv('signals.csv')
    pool = pd.read_csv('pool.csv')
    hs = pd.read_csv('hs300_daily.csv', parse_dates=['date'])
    cal = hs['date'].tolist()

    results = []
    all_trades = []
    for a in ANALOGS:
        start = pd.Timestamp(a)
        # 从下一个交易日开始
        start = min(d for d in cal if d > start)
        end = start + pd.Timedelta(days=365)
        end = min(d for d in cal if d >= end)
        eng = Engine(start, end, gate, sig, pool, cal)
        r = eng.run()
        r['trades'].to_csv(f'trades_{a}.csv', index=False)
        r['curve'].to_csv(f'curve_{a}.csv', index=False)
        t6 = start + pd.Timedelta(days=182)
        c6 = r['curve'][r['curve']['date']<=t6]
        ret6 = (c6['equity'].iloc[-1]/EQUITY0 - 1) if len(c6) else np.nan
        results.append({'analog': a, 'start': str(start)[:10], 'end': str(end)[:10],
                        'ret12m': r['return'], 'ret6m': ret6, 'n_trades': r['n_trades'],
                        'win_rate': r['win_rate'], 'avg_win%': r['avg_win'],
                        'avg_loss%': r['avg_loss'], 'max_dd': r['max_dd']})
        tt = r['trades'].copy(); tt['analog'] = a
        all_trades.append(tt)
        print(f"=== {a} → {str(end)[:10]} | 12月收益 {r['return']*100:.1f}% | 6月 {ret6*100:.1f}% | "
              f"交易 {r['n_trades']} 笔 | 胜率 {r['win_rate']*100 if r['n_trades'] else 0:.0f}% | 回撤 {r['max_dd']*100:.1f}%")
    res = pd.DataFrame(results)
    res.to_csv('analog_results.csv', index=False)
    pd.concat(all_trades).to_csv('all_trades.csv', index=False)
    print("\n=== 汇总 ===")
    print(res[['analog','ret12m','ret6m','n_trades','win_rate','max_dd']].to_string())
    pos = (res['ret12m']>0).mean()
    print(f"\n12个月赚钱概率: {pos*100:.0f}%  中位收益: {res['ret12m'].median()*100:.1f}%")
    allt = pd.concat(all_trades)
    if len(allt):
        print(f"合并交易 {len(allt)} 笔, 交易级胜率 {(allt['pnl']>0).mean()*100:.0f}%, "
              f"盈亏比 {allt[allt['pnl']>0]['pnl'].mean() / abs(allt[allt['pnl']<=0]['pnl'].mean()):.2f}")

if __name__ == '__main__':
    main()
