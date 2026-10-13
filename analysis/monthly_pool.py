#!/usr/bin/env python3
# monthly_pool.py — SYSTEM.md v1.0 的每月选股工具（操作层，不涉及任何规则修改）
# 用法: python3 monthly_pool.py [--date YYYY-MM-DD] [--n 50]
# 口径冻结于 FINDINGS.md「H5 预注册」与 SYSTEM.md：宇宙 = age≥250、剔 BJ（无价格/成交额地板）。
# 注意：dim_symbol.name 列 100% 为 NULL（2026-10-12 复核），本地无名称；需要名称请用
#       hithink-finance symbol 命令逐只查询。amt60 仅用于分层抽样，不作过滤。
import argparse
import duckdb
import numpy as np
import pandas as pd

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'

SQL = """
  WITH px AS (
    SELECT v.thscode, v.date, v.close,
           v.close / LAG(v.close, 20) OVER w - 1.0 AS r20,
           v.close / MAX(v.high) OVER w20 - 1.0 AS dd,
           AVG(v.volume) OVER w5 / NULLIF(AVG(v.volume) OVER w20v, 0) AS vratio,
           AVG(v.amount) OVER w60 AS amt60,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{d}' AND s.exchange <> 'BJ'
    WINDOW w   AS (PARTITION BY v.thscode ORDER BY v.date),
           w20 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           w5  AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
           w20v AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
           w60 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW))
  SELECT date, thscode, close, r20, dd, vratio, amt60
  FROM px
  WHERE date = '{d}' AND age >= 250 AND r20 IS NOT NULL
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='定员日 T（默认库内最新交易日）')
    ap.add_argument('--n', type=int, default=50, help='抽样只数（分层随机，默认 50）')
    args = ap.parse_args()

    con = duckdb.connect(DB, read_only=True)
    if args.date is None:
        d = con.execute('SELECT MAX(date) FROM v_daily').fetchone()[0]
    else:
        d = args.date
        ok = con.execute("SELECT COUNT(*) FROM v_daily WHERE date = ?", [d]).fetchone()[0]
        if not ok:
            latest = con.execute('SELECT MAX(date) FROM v_daily WHERE date <= ?', [d]).fetchone()[0]
            con.close()
            raise SystemExit(f'{d} 非交易日；最近可用交易日为 {latest}')
    df = con.execute(SQL.format(d=d)).df()
    con.close()

    n_univ = len(df)
    thr = df['r20'].quantile(0.8)
    f1 = df['r20'] >= thr
    f2 = (df['dd'] >= -0.15) & (df['dd'] <= -0.05) & (df['vratio'] >= 1.6)
    pool = df[~f1 & ~f2].copy()
    print(f'定员日 T = {d}')
    print(f'宇宙 {n_univ} 只 → F1(强势前20%, r20≥{thr:.1%}) 剔 {int(f1.sum())} → '
          f'F2(放量回调) 剔 {int((f2 & ~f1).sum())} → 池 {len(pool)} 只')
    if len(pool) < 500:
        print('警告：池规模 < 500，触及 SYSTEM.md 第 7 节失效红线第 2 条')

    pool['dec'] = pool['amt60'].rank(pct=True).apply(lambda x: min(int(x * 10), 9))
    seed = int(pd.Timestamp(d).strftime('%Y%m%d'))
    rng = np.random.default_rng(seed)
    per = int(np.ceil(args.n / 10))
    picks = []
    for _, g in pool.groupby('dec'):
        k = min(per, len(g))
        picks.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    sel = pd.concat(picks).head(args.n).sort_values('amt60', ascending=False)
    out = sel[['thscode', 'close', 'amt60']].reset_index(drop=True)
    path = f'/Users/xianyi/trading/analysis/pool_{d}.csv'
    out.to_csv(path, index=False)
    med = out['amt60'].median()
    print(f'抽样 N={len(out)}（分层随机，种子 {seed}）→ {path}')
    print(f'抽样流动性画像: 60日均额中位数 {med / 1e8:.2f} 亿（min {out["amt60"].min() / 1e8:.2f} 亿）——'
          f'低额票的实盘冲击成本自行评估')
    print(out.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
