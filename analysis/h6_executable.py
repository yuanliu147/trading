#!/usr/bin/env python3
# h6_executable.py
# H6（预备预算第2席）：可执行容量版负向过滤组合（N=50 主判据 / N=200 衰减参照），仅训练集。
# 口径冻结于 FINDINGS.md「H6 预注册」(2026-10-12)。纪律：本检验不读 2023+ 数据。
import sys
import duckdb
import json
import datetime as dt
import numpy as np
import pandas as pd

if len(sys.argv) > 1 and sys.argv[1] == 'oos':
    sys.exit('纪律禁止：样本外揭封已在 H5 用尽，H6 只允许训练集。')

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
D1 = '2022-12-31'
COST_RT = 0.0015
COST_SIDE = 0.00075

SQL_FORM = """
  WITH cal AS (SELECT DISTINCT date FROM v_daily WHERE date <= '{D1}'),
  ci AS (SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS rn FROM cal),
  rebal AS (SELECT date FROM ci WHERE rn % 20 = 0),
  px AS (
    SELECT v.thscode, v.date, v.close,
           v.close / LAG(v.close, 20) OVER w - 1.0 AS r20,
           v.close / MAX(v.high) OVER w20 - 1.0 AS dd,
           AVG(v.volume) OVER w5 / NULLIF(AVG(v.volume) OVER w20v, 0) AS vratio,
           AVG(v.amount) OVER w60 AS amt60,
           LEAD(v.close, 1) OVER w AS c_in,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{D1}' AND s.exchange <> 'BJ'
    WINDOW w   AS (PARTITION BY v.thscode ORDER BY v.date),
           w20 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           w5  AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
           w20v AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
           w60 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW))
  SELECT p.date, p.thscode, p.r20, p.dd, p.vratio, p.amt60, p.c_in / p.close - 1.0 AS gap1
  FROM px p JOIN rebal r ON p.date = r.date
  WHERE p.age >= 250 AND p.r20 IS NOT NULL AND p.amt60 >= 2e8 AND p.close >= 2.0
"""

SQL_DAILY = """
  WITH px AS (
    SELECT v.thscode, v.date,
           v.close / LAG(v.close) OVER w - 1.0 AS ret1,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{D1}' AND s.exchange <> 'BJ'
    WINDOW w AS (PARTITION BY v.thscode ORDER BY v.date))
  SELECT date, thscode, ret1 FROM px WHERE age >= 250
"""


def limit_pct(thscode, date):
    t = thscode.split('.')[0]
    if t[:3] in ('300', '301'):
        return 0.20 if str(date) >= '2020-08-24' else 0.10
    if t[:3] == '688':
        return 0.20
    return 0.10


def load_index():
    wu = json.load(open('/tmp/hs300_warmup.json'))['data']['item']
    fix = lambda ms: (dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc) + dt.timedelta(days=1)).date()
    wu_df = pd.DataFrame([{'date': pd.Timestamp(fix(it['date_ms'])), 'close': it['close_price']} for it in wu])
    loc = pd.read_csv('/Users/xianyi/trading/analysis/hs300_daily.csv', parse_dates=['date'])[['date', 'close']]
    idx = (pd.concat([wu_df, loc]).drop_duplicates('date').sort_values('date').reset_index(drop=True))
    idx['ma200'] = idx['close'].rolling(200).mean()
    return idx.set_index('date')


def mets(daily_ret):
    r = daily_ret.dropna()
    n = len(r)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    ann = float(eq.iloc[-1] ** (252 / n) - 1)
    return dict(days=n, ann=ann, vol=float(r.std(ddof=1) * np.sqrt(252)), maxdd=dd,
                calmar=float(ann / abs(dd)) if dd < 0 else np.nan)


def run_n(ret, cal, fpos, picks, R, col_idx, n_label):
    n = len(cal)
    port = np.zeros(n)
    period_ret = []
    turns = []
    prev = None
    touch = 0
    touch_tot = 0
    for k, f in enumerate(fpos):
        cur = picks.get(f)
        if not cur or len(cur) < 20:
            continue
        end = (fpos[k + 1] + 1) if k + 1 < len(fpos) else n - 1
        end = min(end, n - 1)
        if f + 2 > end:
            continue
        turn = 0.5 if prev is None else 1 - len(prev & cur) / len(cur)
        turns.append(turn)
        mi = np.array([col_idx[c] for c in cur if c in col_idx])
        seg = R[f + 2:end + 1][:, mi]
        port[f + 2:end + 1] = np.nanmean(seg, axis=1)
        port[f + 1] -= COST_RT * turn
        period_ret.append(float((1 + port[f + 1:end + 1]).prod() - 1))
        prev = cur
    start = fpos[0] + 1
    return port[start:], period_ret, float(np.mean(turns) * 252 / 20) if turns else np.nan, start


def main():
    con = duckdb.connect(DB, read_only=True)
    form = con.execute(SQL_FORM.format(D1=D1)).df()
    daily = con.execute(SQL_DAILY.format(D1=D1)).df()
    con.close()
    form['date'] = pd.to_datetime(form['date'])
    daily['date'] = pd.to_datetime(daily['date'])

    ret = daily.pivot(index='date', columns='thscode', values='ret1').sort_index()
    cal = ret.index
    n = len(cal)
    fdates = np.sort(form['date'].unique())
    fpos = [p for p in cal.get_indexer(pd.DatetimeIndex(fdates)) if 0 <= p < n - 2]
    R = ret.values
    col_idx = {c: i for i, c in enumerate(ret.columns)}

    # 双过滤（同 H5）+ 换仓日涨跌停触及统计
    picks50, picks200 = {}, {}
    touch = tot = 0
    for f in fpos:
        g = form[form['date'] == cal[f]].copy()
        thr = g['r20'].quantile(0.8)
        keep = (g['r20'] < thr) & ~((g['dd'] >= -0.15) & (g['dd'] <= -0.05) & (g['vratio'] >= 1.6))
        g = g[keep]
        lim = np.array([limit_pct(t, cal[f]) for t in g['thscode']])
        sealed = np.abs(g['gap1'].values) >= (lim - 0.001)
        touch += int(sealed.sum())
        tot += len(g)
        g['dec'] = g['amt60'].rank(pct=True).apply(lambda x: min(int(x * 10), 9))
        top = g.sort_values('amt60', ascending=False).groupby('dec').head(5)
        picks50[f] = set(top['thscode'])
        top2 = g.sort_values('amt60', ascending=False).groupby('dec').head(20)
        picks200[f] = set(top2['thscode'])
    print(f"换仓执行日（t+1）涨跌停封死触及率: {touch}/{tot} = {touch / tot:.2%}")

    idx = load_index()
    idx = idx[idx.index <= pd.Timestamp(D1)]

    for label, picks in [('N=50', picks50), ('N=200', picks200)]:
        port, period_ret, ann_turn, start = run_n(ret, cal, fpos, picks, R, col_idx, label)
        dates = cal[start:]
        b = idx.reindex(dates).ffill()
        bh = b['close'].pct_change()
        sig = (b['close'] > b['ma200']).astype(float)
        pos = sig.shift(2).fillna(0.0)
        t200 = pos * bh - COST_SIDE * pos.diff().fillna(0.0).abs()
        pm, bm, tm = mets(pd.Series(port, index=dates)), mets(bh), mets(t200)
        pr = np.array(period_ret)
        win = float((pr > 0).mean())
        pl = float(pr[pr > 0].mean() / abs(pr[pr < 0].mean()))
        sizes = [len(picks[f]) for f in fpos if picks.get(f)]
        print(f"\n===== H6 {label} 训练集（{dates[0].date()} ~ {dates[-1].date()}） =====")
        print(f"组合: ann={pm['ann']:+.2%} vol={pm['vol']:.2%} maxDD={pm['maxdd']:.2%} Calmar={pm['calmar']:+.2f}")
        print(f"300B&H Calmar={bm['calmar']:+.2f} | 200dma择时 Calmar={tm['calmar']:+.2f}")
        print(f"期胜率 {win:.1%}，盈亏比 {pl:.2f}，年化换手 {ann_turn:.1f}x，成员均值 {np.mean(sizes):.0f} 只")
        pd.DataFrame({'date': dates, 'port': (1 + pd.Series(port, index=dates)).cumprod(),
                      'bh300': (1 + bh).cumprod(), 't200': (1 + t200).cumprod()}
                     ).to_csv(f'/Users/xianyi/trading/analysis/h6_equity_{label.replace("=", "")}.csv', index=False)
        if label == 'N=50':
            print("\n===== H6 判决（对照预注册判据线，N=50） =====")
            ok = pm['calmar'] > bm['calmar'] and pm['calmar'] > tm['calmar']
            print(f"  Calmar: 组合 {pm['calmar']:+.2f} vs B&H {bm['calmar']:+.2f} vs 200dma {tm['calmar']:+.2f}")
            print(f"  判决: {'成立 → 系统蓝图 v1 定型' if ok else '不成立 → 交付物回退为 H5 配置型规则 + 披露'}")


if __name__ == '__main__':
    main()
