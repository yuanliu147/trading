"""月度股票池构建：剔除 次新/低价/低流动性/北交所，按120日波动率取前40"""
import duckdb, pandas as pd, numpy as np

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'

def build_pool():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("""
        SELECT thscode, date, close, amount,
               avg(amount) OVER (PARTITION BY thscode ORDER BY date ROWS 60 PRECEDING) AS amt60,
               count(*) OVER (PARTITION BY thscode ORDER BY date) AS nbar
        FROM v_daily_qfq
    """).df()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['thscode','date'])
    # 修正收益率波动率: 用日收益std
    df['ret'] = df.groupby('thscode')['close'].pct_change()
    df['vol120'] = df.groupby('thscode')['ret'].transform(lambda s: s.rolling(120).std())
    # 250日振幅: "走得远"的股性
    df['hi250'] = df.groupby('thscode')['close'].transform(lambda s: s.rolling(250, min_periods=120).max())
    df['lo250'] = df.groupby('thscode')['close'].transform(lambda s: s.rolling(250, min_periods=120).min())
    df['range250'] = df['hi250'] / df['lo250'] - 1
    df['month'] = df['date'].dt.to_period('M')

    def board(code):
        pre, ex = code.split('.')
        if ex == 'BJ': return 'bj'
        if pre[:3] in ('300','301') or pre[:3] == '688': return 'cm20'
        return 'cm10'
    df['board'] = df['thscode'].map(board)

    # 每月最后一个交易日做筛选（次月生效）
    rows = []
    for mth, g in df.groupby('month'):
        last = g[g['date'] == g['date'].max()]
        ok = last[(last['nbar'] >= 250) & (last['close'] >= 2.0) &
                  (last['amt60'] >= 2e8) & (last['board'] != 'bj') &
                  last['vol120'].notna() & last['range250'].notna()]
        # 波动率与振幅等权排序(名次相加)
        ok = ok.assign(score=ok['range250'].rank())
        ok = ok.sort_values('score', ascending=False).head(50)
        for _, r in ok.iterrows():
            rows.append({'month': str(mth + 1), 'thscode': r['thscode'],
                         'board': r['board'], 'vol120': r['vol120'],
                         'close': r['close'], 'amt60': r['amt60']})
    pool = pd.DataFrame(rows)
    pool.to_csv('pool.csv', index=False)
    print(pool.groupby('month').size().describe())
    print(pool.head(8).to_string())
    return pool

if __name__ == '__main__':
    build_pool()
