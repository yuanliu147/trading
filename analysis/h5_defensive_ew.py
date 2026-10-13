#!/usr/bin/env python3
# h5_defensive_ew.py
# H5（预备预算第1席）：负向过滤等权组合 vs 双基准（沪深300 B&H、200dma择时）。
# 口径冻结于 FINDINGS.md「H5 预注册」(2026-10-12)。
# 用法: python3 h5_defensive_ew.py train   —— 训练集（<=2022-12-31）
#       python3 h5_defensive_ew.py oos     —— 样本外（2023-01-01起，揭封时只跑一次）
import sys
import duckdb
import json
import datetime as dt
import numpy as np
import pandas as pd

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'train'
D1 = '2022-12-31' if MODE == 'train' else '2026-10-09'
D0 = None if MODE == 'train' else '2023-01-01'
COST_RT = 0.0015       # 往返 0.15% × 单边换手
COST_SIDE = 0.00075    # 基准择时每边

SQL_FORM = """
  WITH cal AS (SELECT DISTINCT date FROM v_daily WHERE date <= '{D1}'),
  ci AS (SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS rn FROM cal),
  rebal AS (SELECT date FROM ci WHERE rn % 20 = 0),
  px AS (
    SELECT v.thscode, v.date,
           v.close / LAG(v.close, 20) OVER w - 1.0 AS r20,
           v.close / MAX(v.high) OVER w20 - 1.0 AS dd,
           AVG(v.volume) OVER w5 / NULLIF(AVG(v.volume) OVER w20v, 0) AS vratio,
           COUNT(*) OVER (PARTITION BY v.thscode ORDER BY v.date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS age
    FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
    WHERE v.date <= '{D1}' AND s.exchange <> 'BJ'
    WINDOW w   AS (PARTITION BY v.thscode ORDER BY v.date),
           w20 AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
           w5  AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
           w20v AS (PARTITION BY v.thscode ORDER BY v.date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
  SELECT p.date, p.thscode, p.r20, p.dd, p.vratio
  FROM px p JOIN rebal r ON p.date = r.date
  WHERE p.age >= 250 AND p.r20 IS NOT NULL
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


def main():
    con = duckdb.connect(DB, read_only=True)
    form = con.execute(SQL_FORM.format(D1=D1)).df()
    daily = con.execute(SQL_DAILY.format(D1=D1)).df()
    con.close()
    form['date'] = pd.to_datetime(form['date'])
    daily['date'] = pd.to_datetime(daily['date'])

    ret = daily.pivot(index='date', columns='thscode', values='ret1').sort_index()
    if D0:
        ret = ret[ret.index >= D0]
    cal = ret.index
    n = len(cal)

    # 形成日 → 日历位置（相位由 SQL 全局日历决定，与模式无关）
    fdates = np.sort(form['date'].unique())
    fpos_all = cal.get_indexer(pd.DatetimeIndex(fdates))
    fpos = [p for p in fpos_all if 0 <= p < n - 2]

    members, turns = [], []
    prev = None
    col_idx = {c: i for i, c in enumerate(ret.columns)}
    for f in fpos:
        g = form[form['date'] == cal[f]]
        thr = g['r20'].quantile(0.8)
        excl = (g['r20'] >= thr) | ((g['dd'] >= -0.15) & (g['dd'] <= -0.05) & (g['vratio'] >= 1.6))
        cur = set(g.loc[~excl, 'thscode'])
        turn = 0.5 if prev is None else 1 - len(prev & cur) / max(len(cur), 1)
        members.append(cur)
        turns.append(turn)
        prev = cur

    R = ret.values  # n_days × n_stocks
    port = np.zeros(n)
    period_ret = []
    for k, f in enumerate(fpos):
        if len(members[k]) < 50:
            continue
        end = (fpos[k + 1] + 1) if k + 1 < len(fpos) else n - 1
        end = min(end, n - 1)
        if f + 2 > end:
            continue
        mi = np.array([col_idx[c] for c in members[k] if c in col_idx])
        seg = R[f + 2:end + 1][:, mi]
        port[f + 2:end + 1] = np.nanmean(seg, axis=1)
        port[f + 1] -= COST_RT * turns[k]          # 执行日扣成本
        period_ret.append(float((1 + port[f + 1:end + 1]).prod() - 1))

    start = fpos[0] + 1
    dates = cal[start:]
    port = port[start:]

    idx = load_index()
    idx = idx[(idx.index <= pd.Timestamp(D1))]
    if D0:
        idx = idx[idx.index >= pd.Timestamp(D0)]
    b = idx.reindex(dates).ffill()
    bh = b['close'].pct_change()
    sig = (b['close'] > b['ma200']).astype(float)
    pos = sig.shift(2).fillna(0.0)                    # 信号日+2 起计收益
    dpos = pos.diff().fillna(0.0).abs()
    t200 = pos * bh - COST_SIDE * dpos

    pm, bm, tm = mets(pd.Series(port, index=dates)), mets(bh), mets(t200)
    pr = np.array(period_ret)
    win = float((pr > 0).mean())
    pl = float(pr[pr > 0].mean() / abs(pr[pr < 0].mean()))
    ann_turn = float(np.mean(turns) * 252 / 20)

    tag = '训练集' if MODE == 'train' else '样本外'
    print(f"===== H5 {tag}（{dates[0].date()} ~ {dates[-1].date()}） =====")
    print(f"组合: ann={pm['ann']:+.2%} vol={pm['vol']:.2%} maxDD={pm['maxdd']:.2%} Calmar={pm['calmar']:+.2f}")
    print(f"300B&H: ann={bm['ann']:+.2%} maxDD={bm['maxdd']:.2%} Calmar={bm['calmar']:+.2f}")
    print(f"300择时: ann={tm['ann']:+.2%} maxDD={tm['maxdd']:.2%} Calmar={tm['calmar']:+.2f}")
    print(f"组合期胜率 {win:.1%}，盈亏比 {pl:.2f}，年化换手 {ann_turn:.1f}x，成员均值 "
          f"{np.mean([len(m) for m in members]):.0f} 只")
    out = pd.DataFrame({'date': dates, 'port': (1 + pd.Series(port, index=dates)).cumprod(),
                        'bh300': (1 + bh).cumprod(), 't200': (1 + t200).cumprod()})
    out.to_csv(f'/Users/xianyi/trading/analysis/h5_equity_{MODE}.csv', index=False)

    print("\n===== H5 判决（对照预注册判据线） =====")
    if pm['calmar'] > bm['calmar'] and pm['calmar'] > tm['calmar']:
        v = '成立（训练集）→ 允许一次性揭封样本外' if MODE == 'train' else '样本外通过 → 系统成立'
    else:
        v = '不成立 → 记录"未找到站得住的规律"，空手而归，不揭封' if MODE == 'train' else '样本外失败 → 假设作废，不许改参数重封'
    print(f"  Calmar: 组合 {pm['calmar']:+.2f} vs B&H {bm['calmar']:+.2f} vs 200dma {tm['calmar']:+.2f}")
    print(f"  判决: {v}")


if __name__ == '__main__':
    main()
