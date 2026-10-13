"""大盘闸门 v2：周线摆动低点序列 + 基准点破位 + 150日均线 + 量能补丁"""
import pandas as pd, numpy as np

def swing_lows(w, k=3):
    """周线摆动低点: low[i] 为前后k周内最低; 确认时点为 i+k 周"""
    lows = w['low'].values; n = len(w)
    pts = []
    for i in range(k, n - k):
        if lows[i] == min(lows[i-k:i+k+1]):
            pts.append((i, lows[i], i + k))  # (周idx, 低点值, 确认周idx)
    # 相距太近(<3周)的去重保留更低
    out = []
    for p in pts:
        if out and p[0] - out[-1][0] < 3:
            if p[1] < out[-1][1]:
                out[-1] = p
        else:
            out.append(p)
    return out

def gate_series(df, vol_df=None):
    """每周闸门明细:
    - swings: 截至当周已确认的摆动低点序列
    - broken: 当周收盘价 < 最近确认摆动低点 (结构破位)
    - lows_rising: 最近确认低点 > 前一确认低点
    - above_ma150: 周末日线收盘 > MA150
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date').reset_index(drop=True)
    d['ma150'] = d['close'].rolling(150).mean()
    w = d.set_index('date').resample('W-FRI').agg(
        open=('open','first'), high=('high','max'), low=('low','min'),
        close=('close','last'), amount=('amount','sum')).dropna(subset=['close']).reset_index()
    w = w.rename(columns={'date':'week_end'})
    sw = swing_lows(w)
    rows = []
    for i in range(len(w)):
        we = w.loc[i, 'week_end']
        past = [s for s in sw if s[2] <= i]
        day = d[d['date'] <= we]
        above_ma = bool(day['close'].iloc[-1] > day['ma150'].iloc[-1]) if len(day) and not np.isnan(day['ma150'].iloc[-1]) else None
        last_low = past[-1][1] if past else None
        prev_low = past[-2][1] if len(past) >= 2 else None
        broken = bool(w.loc[i,'close'] < last_low) if last_low else None
        rising = bool(last_low > prev_low) if prev_low else None
        rows.append({'week_idx': i, 'week_end': we, 'close': w.loc[i,'close'],
                     'above_ma150': above_ma, 'broken': broken, 'lows_rising': rising,
                     'last_sw_low': last_low, 'prev_sw_low': prev_low,
                     'last_sw_week': past[-1][0] if past else None})
    g = pd.DataFrame(rows)
    return g, w, sw

def per_index_ok(r):
    """单指数"破坏"判定: 结构破位 或 跌破150日线 或 低点下移"""
    if r['broken'] is None or pd.isna(r['broken']):
        return None
    return (not r['broken']) and r['above_ma150'] and r['lows_rising']

def combined_gate(g300, gall, mkt_amount_daily):
    """评分制闸门(补丁规则: 结构为准, MA150辅助):
    单指数: 结构破位=0; 结构OK但低于MA150=0.5; 结构OK且站上MA150=1
    合计: 2→绿; 1~1.5→黄; ≤0.5→红"""
    m = g300.merge(gall, on='week_end', suffixes=('_300', '_all'))
    for tag in ['300','all']:
        sc = []
        for _, r in m.iterrows():
            broken = r[f'broken_{tag}']; rising = r[f'lows_rising_{tag}']; ama = r[f'above_ma150_{tag}']
            if pd.isna(broken) or broken is None:
                sc.append(None); continue
            struct_ok = (not broken) and bool(rising)
            if not struct_ok:
                sc.append(0.0)
            else:
                sc.append(1.0 if ama else 0.5)
        m[f'score_{tag}'] = sc
    def color(r):
        if r['score_300'] is None or r['score_all'] is None:
            return 'unknown'
        t = r['score_300'] + r['score_all']
        if t >= 2: return 'green'
        if t >= 1: return 'yellow'
        return 'red'
    m['gate_raw'] = m.apply(color, axis=1)
    # 量能补丁: 全市场5日均额 < 0.8 × 60日均额 → 降一档
    a = mkt_amount_daily.copy()
    a['date'] = pd.to_datetime(a['date'])
    a['ma5'] = a['total_amount'].rolling(5).mean()
    a['ma60'] = a['total_amount'].rolling(60).mean()
    a = a.set_index('date')
    shrink = []
    for _, r in m.iterrows():
        sub = a[a.index <= r['week_end']]
        if len(sub) and not np.isnan(sub['ma60'].iloc[-1]):
            shrink.append(bool(sub['ma5'].iloc[-1] < 0.7 * sub['ma60'].iloc[-1]))
        else:
            shrink.append(False)
    m['vol_shrink'] = shrink
    order = {'red':0, 'yellow':1, 'green':2, 'unknown':-1}
    inv = {0:'red',1:'yellow',2:'green'}
    m['gate'] = [inv[max(0, order[g]-1)] if s and g in ('green','yellow') else g
                 for g, s in zip(m['gate_raw'], m['vol_shrink'])]
    return m
