"""买点信号引擎：对池内股票逐日检测买点A/B/C（全因果，只用t日及之前数据）
A 底部放量突破箱体; B 突破后回调承接(三天法则); C 上升通道下轨
止损: 基准点下方6% 与 2×ATR(14) 取宽者(更低者); 赔率≥3:1 才放行
"""
import duckdb, pandas as pd, numpy as np

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'

def load_stock(con, code):
    df = con.execute("""
        SELECT date, open, high, low, close, volume, amount
        FROM v_daily_qfq WHERE thscode = ? ORDER BY date
    """, [code]).df()
    return df

def add_indicators(df):
    df = df.copy()
    pc = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['vol20'] = df['volume'].shift(1).rolling(20).mean()
    # 已确认日级摆动低点: low[i]=min(low[i-3..i+3]), 于i+3确认
    lows = df['low'].values
    sw = np.full(len(df), np.nan)
    for i in range(3, len(df)-3):
        if lows[i] == lows[i-3:i+4].min():
            sw[i+3] = lows[i]   # 在确认日可见
    df['swing_low_confirmed'] = pd.Series(sw, index=df.index).ffill()
    return df

def last_swing_low(df, t):
    """t日(含)之前最近确认摆动低点"""
    s = df['swing_low_confirmed'].iloc[:t+1].dropna()
    return s.iloc[-1] if len(s) else None

def stop_price(entry, base, atr):
    """基准点下6% 与 2ATR 取宽(低)"""
    return min(base * 0.94, entry - 2.0 * atr)

def sig_A(df, t):
    """底部箱体放量突破: 箱体长40/60/80/120, 高度≤30%, 处于250日低位区"""
    if t < 260: return None
    c = df['close'].iloc[t]; v = df['volume'].iloc[t]; v20 = df['vol20'].iloc[t]
    h = df['high'].iloc[t]; l = df['low'].iloc[t]
    if h == l or np.isnan(v20) or v < 2.0 * v20: return None
    if df['amount'].iloc[t] < 2e8: return None
    hi250 = df['high'].iloc[max(0,t-250):t].max()
    for L in (120, 80, 60, 40):
        box = df.iloc[t-L:t]
        bh, bl = box['high'].max(), box['low'].min()
        if bl <= 0: continue
        height = bh / bl - 1
        if height > 0.30: continue
        if bl > hi250 * 0.75: continue          # 底部区域
        if c > bh * 1.003 and c < bh * 1.08:     # 放量突破且不追高
            atr = df['atr14'].iloc[t]
            if np.isnan(atr): return None
            base = last_swing_low(df, t)          # 最近确认基准点(箱体内最后的收缩低点)
            if base is None or base > c: base = bl
            stp = stop_price(c, base, atr)
            risk = c - stp
            if risk <= 0 or risk / c > 0.15: return None
            # 目标: 箱体1:1 与 箱底前历史高点(前一轮下跌起点) 取高
            tgt = bh + (bh - bl)
            r_pre = df['high'].iloc[max(0,t-260):t-L].max() if t-L > 0 else 0
            if r_pre > tgt: tgt = r_pre
            elif c < r_pre < tgt: tgt = r_pre
            if (tgt - c) / risk >= 3.0:
                return {'type':'A','entry':c,'stop':stp,'target':tgt,'base':base}
    return None

def sig_B(df, t):
    """回调承接: 45日内的≥25%脉冲 → 回撤40%~65% → 三天法则 → 放量拐头"""
    if t < 80: return None
    c = df['close'].iloc[t]; v = df['volume'].iloc[t]; v20 = df['vol20'].iloc[t]
    if np.isnan(v20) or v < 1.5 * v20: return None
    if df['close'].iloc[t] <= df['close'].iloc[t-1]: return None   # 拐头
    hi = df['high'].values; lo = df['low'].values; cl = df['close'].values
    # 脉冲高点PH: 过去5~30日内的最高
    seg_hi = hi[t-30:t-4]
    if len(seg_hi) == 0: return None
    ph = seg_hi.max(); ph_i = t-30 + int(seg_hi.argmax())
    # 脉冲低点PL: PH之前45日内最低
    lo_seg = lo[max(0,ph_i-45):ph_i+1]
    if len(lo_seg) == 0: return None
    pl = lo_seg.min(); pl_i = max(0,ph_i-45) + int(lo_seg.argmin())
    if pl_i >= ph_i or ph / pl < 1.25: return None
    # 回撤低点PB: PH之后
    pb_seg = lo[ph_i+1:t]
    if len(pb_seg) < 3: return None
    pb = pb_seg.min(); pb_i = ph_i+1 + int(pb_seg.argmin())
    retr = (ph - pb) / (ph - pl)
    if not (0.40 <= retr <= 0.65): return None
    if pb_i > t - 4: return None                    # 三天法则需要至少3天确认
    # 三天法则: PB后连续3日低点不破PB低点 且 缩量
    if not all(lo[pb_i+1:pb_i+4] > lo[pb_i]): return None
    vol_pull = df['volume'].iloc[pb_i:t].mean()
    vol_imp = df['volume'].iloc[pl_i:ph_i+1].mean()
    if not (vol_pull < 0.9 * vol_imp): return None   # 回调缩量
    # 今日放量重新拐头: 收盘突破PB以来小平台
    if c <= max(hi[pb_i+1:t]): return None
    atr = df['atr14'].iloc[t]
    if np.isnan(atr): return None
    stp = stop_price(c, pb, atr)
    risk = c - stp
    if risk <= 0 or risk / c > 0.15: return None
    tgt = ph + (ph - pl) * 0.382
    if (tgt - c) / risk >= 3.0:
        return {'type':'B','entry':c,'stop':stp,'target':tgt,'base':pb}
    return None

def sig_C(df, t):
    """通道下轨: 取最近两个已确认摆动低点连线(上升趋势线), 触线缩量企稳买入
    目标=过两点间最高点的平行线(通道上轨)"""
    if t < 80: return None
    lows = df['low'].values
    # 已确认摆动低点(确认日≤t)
    sw = []
    for i in range(max(3, t-80), min(len(df)-3, t+1)):
        if lows[i] == lows[i-3:i+4].min() and i+3 <= t:
            sw.append((i, lows[i]))
    if len(sw) < 2: return None
    (i1, s1), (i2, s2) = sw[-2], sw[-1]
    if s2 <= s1 or i2 - i1 < 8: return None      # 上升趋势且相距≥8日
    slope = (s2 - s1) / (i2 - i1)
    if slope / s1 < 0.002: return None           # 日均斜率≥0.2%
    line_t = s2 + slope * (t - i2)
    c = df['close'].iloc[t]; l = df['low'].iloc[t]
    if l > line_t * 1.02 or c < line_t * 0.99: return None   # 触线且收回
    v = df['volume'].iloc[t]; v20 = df['vol20'].iloc[t]
    if np.isnan(v20) or v > v20: return None                 # 缩量
    if c <= df['close'].iloc[t-1]: return None               # 拐头
    atr = df['atr14'].iloc[t]
    if np.isnan(atr): return None
    stp = stop_price(c, s2, atr)
    risk = c - stp
    if risk <= 0 or risk / c > 0.15: return None
    # 通道上轨: 过两点间最高高点的平行线
    hh = df['high'].iloc[i1:i2+1].max(); hh_i = i1 + int(df['high'].iloc[i1:i2+1].argmax())
    upper_t = hh + slope * (t - hh_i)
    if upper_t <= c: return None
    if (upper_t - c) / risk >= 3.0:
        return {'type':'C','entry':c,'stop':stp,'target':upper_t,'base':s2}
    return None

def compute_signals():
    con = duckdb.connect(DB, read_only=True)
    pool = pd.read_csv('pool.csv')
    codes = sorted(pool['thscode'].unique())
    print("池内独立股票数:", len(codes))
    out = []
    for k, code in enumerate(codes):
        df = load_stock(con, code)
        if len(df) < 300: continue
        df = add_indicators(df)
        # 该股在哪些月份入池
        months = pool[pool['thscode']==code]['month'].tolist()
        dmonths = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
        in_pool = dmonths.isin(months).values
        for t in range(260, len(df)):
            if not in_pool[t]: continue
            s = sig_A(df, t)
            if s is None: s = sig_B(df, t)
            if s is None: s = sig_C(df, t)
            if s:
                s['thscode'] = code
                s['date'] = str(df['date'].iloc[t])[:10]
                out.append(s)
        if (k+1) % 200 == 0: print(f"{k+1}/{len(codes)} 信号数={len(out)}")
    sig = pd.DataFrame(out)
    sig.to_csv('signals.csv', index=False)
    print("总信号:", len(sig))
    if len(sig): print(sig['type'].value_counts())
    return sig

if __name__ == '__main__':
    compute_signals()
