"""大盘闸门状态机：沪深300 + 全A等权指数（中证全指代理）周线基准点法"""
import json, datetime as dt
import duckdb, pandas as pd, numpy as np

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'

def load_index(path, code):
    d = json.load(open(path))
    df = pd.DataFrame(d['data']['item'])
    df['date'] = pd.to_datetime(df['date_ms'], unit='ms').dt.date
    # date_ms 是北京时间交易日0点的UTC表示(前一日16:00 UTC), naive读数晚一天 → +1天还原真实交易日
    df['date'] = (pd.to_datetime(df['date']) + pd.Timedelta(days=1)).dt.date
    df = df.rename(columns={'open_price':'open','high_price':'high','low_price':'low',
                            'close_price':'close','turnover':'amount'})
    df['thscode'] = code
    return df[['thscode','date','open','high','low','close','volume','amount']]

def build_all_a(con):
    """全A等权日收益指数 + 全市场成交额"""
    df = con.execute("""
        SELECT date,
               count(*) AS n,
               exp(avg(ln(close / prev_close))) - 1 AS eq_ret,
               sum(amount) AS total_amount
        FROM (
            SELECT thscode, date, close, amount,
                   lag(close) OVER (PARTITION BY thscode ORDER BY date) AS prev_close
            FROM v_daily_qfq
        ) WHERE prev_close IS NOT NULL AND prev_close > 0
        GROUP BY date ORDER BY date
    """).df()
    df['close'] = (1 + df['eq_ret']).cumprod() * 1000
    df['open'] = df['close'] / (1 + df['eq_ret'])  # 近似
    df['high'] = df[['open','close']].max(axis=1)
    df['low'] = df[['open','close']].min(axis=1)
    df['thscode'] = 'ALLA.EQ'
    return df[['thscode','date','open','high','low','close','total_amount']].rename(
        columns={'total_amount':'amount'})

def weekly_bars(df):
    """周五结尾的周线；周末不完整周也保留（用isoweek分组）"""
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    w = d.resample('W-FRI').agg(open=('open','first'), high=('high','max'),
                                 low=('low','min'), close=('close','last'),
                                 amount=('amount','sum'))
    w = w.dropna(subset=['close'])
    return w.reset_index().rename(columns={'date':'week_end'})

def detect_base_points(w):
    """三周法则：候选低点周W后，连续3周 low > W.high → W为有效基准点。
    候选低点 = 近6周（含当周）最低low。
    满足条件的周会在大底附近成簇出现，同簇（间隔<4周）只保留low最低的一周。
    返回基准点列表 [(week_idx, low, confirm_idx)]"""
    lows = w['low'].values; highs = w['high'].values; n = len(w)
    cands = []
    for i in range(n - 3):
        if lows[i] == min(lows[max(0, i-5):i+1]) and all(lows[i+1:i+4] > highs[i]):
            cands.append((i, lows[i], i+3))
    bps = []
    for c in cands:
        if bps and c[0] - bps[-1][0] < 4:
            if c[1] < bps[-1][1]:
                bps[-1] = c
        else:
            bps.append(c)
    return bps

def gate_series(df):
    """输出每周闸门判定所需的完整状态"""
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date').reset_index(drop=True)
    d['ma150'] = d['close'].rolling(150).mean()
    daily = d
    w = weekly_bars(df)
    bps = detect_base_points(w)
    # 每周状态: 最近两个基准点
    rows = []
    for i in range(len(w)):
        we = w.loc[i, 'week_end']
        past = [b for b in bps if b[2] <= i]  # 已确认
        above_ma = None
        day = daily[daily['date'] <= we]
        if len(day) and not np.isnan(day['ma150'].iloc[-1]):
            above_ma = bool(day['close'].iloc[-1] > day['ma150'].iloc[-1])
        rising = None
        if len(past) >= 2:
            rising = past[-1][1] > past[-2][1]
        rows.append({'week_idx': i, 'week_end': we,
                     'close': w.loc[i,'close'],
                     'above_ma150': above_ma, 'lows_rising': rising,
                     'last_bp_low': past[-1][1] if past else None,
                     'last_bp_week': past[-1][0] if past else None,
                     'prev_bp_low': past[-2][1] if len(past)>=2 else None,
                     'n_bps': len(past)})
    return pd.DataFrame(rows), w, bps

def combined_gate(g300, gall):
    """合并两指数 → 绿/黄/红灯（未含量能补丁）"""
    m = g300.merge(gall, on='week_end', suffixes=('_300','_all'))
    def color(r):
        ok300 = r['lows_rising_300'] and r['above_ma150_300']
        okall = r['lows_rising_all'] and r['above_ma150_all']
        if r['lows_rising_300'] is None or r['lows_rising_all'] is None:
            return 'unknown'
        if ok300 and okall: return 'green'
        if not ok300 and not okall: return 'red'
        return 'yellow'
    m['gate_raw'] = m.apply(color, axis=1)
    return m

if __name__ == '__main__':
    con = duckdb.connect(DB, read_only=True)
    hs300 = load_index('/tmp/idx_000300b.json', '000300.SH')
    shc   = load_index('/tmp/idx_000001.SH.json', '000001.SH')
    alla  = build_all_a(con)
    hs300.to_csv('hs300_daily.csv', index=False)
    shc.to_csv('shcomposite_daily.csv', index=False)
    alla.to_csv('alla_eq_daily.csv', index=False)
    g300, w300, bp300 = gate_series(hs300)
    gall, wall, bpall = gate_series(alla)
    m = combined_gate(g300, gall)
    m.to_csv('gate_weekly.csv', index=False)
    g300.to_csv('gate_300_detail.csv', index=False)
    gall.to_csv('gate_all_detail.csv', index=False)
    print("300基准点:", [(str(w300.loc[i,'week_end'].date()), round(l,1)) for i,l,_ in bp300][-8:])
    print("全A基准点:", [(str(wall.loc[i,'week_end'].date()), round(l,1)) for i,l,_ in bpall][-8:])
    print("\n最近12周闸门:")
    print(m.tail(12)[['week_end','above_ma150_300','lows_rising_300','above_ma150_all','lows_rising_all','gate_raw']].to_string())
