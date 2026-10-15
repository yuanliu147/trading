#!/usr/bin/env python3
# pool50_turnpoint_backtest.py — 转折点MACD系统的选股换成 SYSTEM.md 的 H5 月度池(N=50)重测
# 性质: 样本内诊断(非预注册检验)。
# 选股 = monthly_pool.py 口径逐月复刻: 非BJ + age≥250 → 剔F1(20日涨幅≥全市场80分位)
#        → 剔F2(自20日高点回落5%~15% 且 5日均量≥前20日均量1.6倍) → amt60分10层每层随机抽5只(N=50,种子=日期)
# 入场: 池内个股日K MACD金叉(池成员资格只管入场) + 阶梯状态过滤(chan_engine笔端点近似)
# 出场: 模式dc=死叉(对照原系统) / 模式br=−6%止损、+20%止盈、死叉兜底; T+1, 0.15%往返
import sys, os, traceback
import duckdb, numpy as np, pandas as pd
sys.path.insert(0, '/Users/xianyi/trading/analysis')
from chan_engine import merge_bars, find_fractals, build_bi, build_zhongshu

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
OUT = '/Users/xianyi/trading/analysis/macd'
FEE, STOP, TP = 0.0015, 0.94, 1.20
LOOKBACK, SLOTS, SLOT_W = 1000, 10, 0.10
START = np.datetime64('2017-10-01')
CHECK20 = os.environ.get('CHECK20') == '1'   # 复现开关: 在20只龙头名单上重跑, 验证实现与旧结果一致
if CHECK20:
    START = np.datetime64('2017-01-01')
    SET20 = set(pd.read_csv(f'{OUT}/MACD研究股票名单.csv')['thscode'].astype(str))

def stats(tr):
    if not tr: return dict(n=0)
    r = np.array([t['ret'] for t in tr]); w = r[r>0]; l = r[r<=0]
    return dict(n=len(r), 胜率=f'{(r>0).mean():.1%}', 平均=f'{r.mean():+.2%}',
                中位=f'{np.median(r):+.2%}',
                盈亏比=f'{(w.mean()/abs(l.mean())) if len(w) and len(l) else float("nan"):.2f}',
                PF=f'{(w.sum()/abs(l.sum())) if len(w) and len(l) else float("nan"):.2f}',
                持有=f'{np.mean([t["hold"] for t in tr]):.1f}天')

print('加载数据...', flush=True)
con = duckdb.connect(DB, read_only=True)
df = con.execute("""SELECT v.thscode, v.date, v.close, v.high, v.low, v.volume, v.amount
  FROM v_daily_qfq v JOIN dim_symbol s USING (thscode)
  WHERE s.exchange <> 'BJ' ORDER BY v.thscode, v.date""").df()
con.close()
cats = df.thscode.astype('category')
code_ids = cats.cat.codes.values.astype(np.int64)
names = cats.cat.categories
dates = df.date.values.astype('datetime64[D]')
close = df.close.values.astype(float); high = df.high.values.astype(float); low = df.low.values.astype(float)
vol = df.volume.values.astype(float); amt = df.amount.values.astype(float)
N = len(df); print(f'行数 {N:,}  股票 {len(names):,}', flush=True)
del df

# ---- 逐股字段(复刻 monthly_pool SQL 窗口) + MACD/金叉 ----
r20 = np.full(N, np.nan); dd = np.full(N, np.nan); vratio = np.full(N, np.nan)
amt60 = np.full(N, np.nan); age = np.arange(N, dtype=np.int64)
dif_a = np.full(N, np.nan); dea_a = np.full(N, np.nan); gc = np.zeros(N, dtype=bool)
uniq, starts = np.unique(code_ids, return_index=True)
bounds = np.append(starts, N)
end_of = np.zeros(N, dtype=np.int64)
print('逐股指标...', flush=True)
for k in range(len(uniq)):
    a, b = bounds[k], bounds[k+1]; end_of[a:b] = b
    c = pd.Series(close[a:b]); h = pd.Series(high[a:b]); v = pd.Series(vol[a:b]); am = pd.Series(amt[a:b])
    r20[a:b] = (c/c.shift(20)-1).values
    dd[a:b]  = (c/h.rolling(20).max()-1).values
    vratio[a:b] = (v.rolling(5).mean()/v.shift(1).rolling(20).mean()).values
    amt60[a:b] = am.rolling(60).mean().values
    dif = c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    dif_a[a:b] = dif.values; dea_a[a:b] = dea.values
    gc[a:b] = ((dif>dea)&(dif.shift(1)<=dea.shift(1))).values
age = age - starts[np.searchsorted(starts, np.arange(N), 'right')-1] + 1   # 个股第几根K线
print('逐股指标完成', flush=True)

# ---- 月度池: 每20个交易日一次, 完全复刻 monthly_pool ----
all_days = np.unique(dates)
t0 = np.searchsorted(all_days, START)
reb = all_days[t0::20]
pools = []
univ_sizes, pool_sizes = [], []
for k, T in enumerate(reb if not CHECK20 else []):
    nxt = reb[k+1] if k+1 < len(reb) else np.datetime64('2262-01-01')
    lo = np.searchsorted(dates, T); hi = np.searchsorted(dates, T, 'right')  # dates在每段内有序, 但全局按code排; 用掩码
    rows = np.where((dates >= T) & (dates < nxt if False else dates == T))[0]
    ok = rows[(age[rows] >= 250) & ~np.isnan(r20[rows])]
    r = r20[ok]; thr = np.nanquantile(r, 0.8)
    f1 = r >= thr
    f2 = (dd[ok] >= -0.15) & (dd[ok] <= -0.05) & (vratio[ok] >= 1.6)
    rows_p = ok[~f1 & ~f2]
    pct = pd.Series(amt60[rows_p]).rank(pct=True).fillna(0).values
    dec = np.minimum((pct*10).astype(int), 9)
    rng = np.random.default_rng(int(pd.Timestamp(str(T)).strftime('%Y%m%d')))
    picks = []
    for dv in range(10):
        m = np.where(dec == dv)[0]; kk = min(5, len(m))
        if kk: picks.append(rows_p[m[rng.choice(len(m), size=kk, replace=False)]])
    sel = np.concatenate(picks)[:50] if picks else np.array([], dtype=np.int64)
    pset = {names[code_ids[p]] for p in sel}
    pools.append((T, nxt, pset)); univ_sizes.append(len(ok)); pool_sizes.append(len(rows_p))
print(f'池期数 {len(pools)}  宇宙均值 {np.mean(univ_sizes):.0f}  过滤后池均值 {np.mean(pool_sizes):.0f}', flush=True)

def in_pool(p):
    if CHECK20: return True
    ki = np.searchsorted(reb, dates[p], 'right') - 1
    return ki >= 0 and names[code_ids[p]] in pools[ki][2]

CAP = 1000   # 结构回看窗(验证过: 分类结果与全历史完全一致)
_state_cache = {}
def states_for(p, ck):
    """单次build_bi同时算阶梯状态与缠论状态, 按(股票,信号行)缓存跨模式复用"""
    key = (ck, p)
    if key in _state_cache: return _state_cache[key]
    st = max(0, p+1-CAP)
    hA, lA = high[st:p+1], low[st:p+1]
    if len(hA) < 60:
        res = ('非下降阶梯', '结构不足')
    else:
        try:
            mh, ml, _, _ = merge_bars(hA, lA)
            bis = build_bi(find_fractals(mh, ml), mh, ml)
            sh = [x['hi'] for x in bis if x['dir'] == 1]; sl_ = [x['lo'] for x in bis if x['dir'] == -1]
            if len(sh) < 2 or len(sl_) < 2: lad = '非下降阶梯'
            elif sh[-2] > sh[-1] and sl_[-2] > sl_[-1]:
                lad = '阶梯已突破解除' if close[p] > sh[-1] else '阶梯屏蔽中'
            else: lad = '非下降阶梯'
            zss = build_zhongshu(bis)
            if len(zss) < 2: chan = '单中枢'
            else:
                zs1, zs2 = zss[-2], zss[-1]
                t = '下跌趋势' if zs2['zg'] < zs1['zd'] else '上涨趋势'
                pos = '中枢上' if close[p] > zs2['zg'] else ('中枢下' if close[p] < zs2['zd'] else '中枢内')
                chan = t + '|' + pos
        except Exception:
            lad, chan = '非下降阶梯', '结构不足'
        res = (lad, chan)
    _state_cache[key] = res
    return res

# ---- 逐股状态机回测(池内金叉才入场; 两种出场模式) ----
def run(mode):
    trades = []
    for k in range(len(uniq)):
        a, b = bounds[k], bounds[k+1]
        if b - a < 60: continue
        name = names[uniq[k]]
        if CHECK20 and name not in SET20: continue
        flat = True; pend = -1; sig_row = -1; e = -1; entry_px = np.nan
        for j in range(a, b):
            if flat:
                if pend == j:
                    e = j; entry_px = close[j]; flat = False; pend = -1
                    continue
                if gc[j] and j+1 < b and in_pool(j):
                    pend = j+1; sig_row = j
            else:
                px = close[j]
                hit = None
                if mode == 'br':
                    if px <= entry_px*STOP: hit = 'stop'
                    elif px >= entry_px*TP: hit = 'tp'
                    elif dif_a[j] < dea_a[j] and dif_a[j-1] >= dea_a[j-1]: hit = 'dc'
                else:
                    if dif_a[j] < dea_a[j] and dif_a[j-1] >= dea_a[j-1]: hit = 'dc'
                if hit or j == b-1:
                    lad, chan = states_for(sig_row, int(uniq[k]))
                    trades.append(dict(code=name, sig=dates[sig_row], entry=dates[e], exit=dates[j],
                                       ret=px/entry_px-1-FEE, hold=j-e, state=lad,
                                       chan=chan, why=hit or 'eod'))
                    flat = True; pend = -1
        if len(trades) % 500 == 0 and trades: pass
    print(f'  [{mode}] 完成, 笔数 {len(trades)}', flush=True)
    return trades

print('回测 dc(死叉出场)...', flush=True)
tr_dc = run('dc')
print('回测 br(止损止盈)...', flush=True)
tr_br = run('br')
for t in tr_dc: t['mode'] = 'dc'
for t in tr_br: t['mode'] = 'br'
pd.DataFrame(tr_dc + tr_br).to_csv(f'{OUT}/池50转折点交易明细.csv', index=False)

cuts = {'dc': [('池内全部金叉(死叉出场)', tr_dc), ('缠论下跌×中枢下(死叉出场)', [t for t in tr_dc if t['chan'] == '下跌趋势|中枢下']),
               ('阶梯解除(死叉出场)', [t for t in tr_dc if t['state'] == '阶梯已突破解除']),
               ('阶梯屏蔽中(死叉出场)', [t for t in tr_dc if t['state'] == '阶梯屏蔽中']), ('非下降阶梯(死叉出场)', [t for t in tr_dc if t['state'] == '非下降阶梯'])],
        'br': [('全部金叉+止损止盈', tr_br), ('缠论下跌×中枢下+止损止盈', [t for t in tr_br if t['chan'] == '下跌趋势|中枢下']),
               ('阶梯解除+止损止盈', [t for t in tr_br if t['state'] == '阶梯已突破解除'])]}
print('\n=== 逐笔统计(H5池50只, 2017-10~2026-10, 扣费) ===')
for m, lst in cuts.items():
    for label, tl in lst:
        s = stats(tl)
        print(f"{label}: n={s.get('n',0)} 胜率={s.get('胜率','-')} 平均={s.get('平均','-')} 中位={s.get('中位','-')} 盈亏比={s.get('盈亏比','-')} PF={s.get('PF','-')} 持有={s.get('持有','-')}")

print('\n=== 阶梯解除笔 分年 ===')
tl = [t for t in tr_dc if t['state'] == '阶梯已突破解除']
yr = pd.Series([str(t['sig'])[:4] for t in tl]).value_counts().sort_index()
print(yr.to_string())
print('\n=== 缠论下跌×中枢下 分年 ===')
tl2 = [t for t in tr_dc if t['chan'] == '下跌趋势|中枢下']
yr2 = pd.Series([str(t['sig'])[:4] for t in tl2]).value_counts().sort_index()
print(yr2.to_string())

# ---- 组合模拟: 10槽×10%, br出场, 事件驱动 ----
def portfolio2(trades, label):
    held = {}; cash = 1.0; eq = []
    ev = sorted(trades, key=lambda t: (t['entry'], t['code']))
    ei = 0
    name2idx = {names[uniq[k]]: k for k in range(len(uniq))}
    for d in all_days[all_days >= reb[0]]:
        # 出场
        for nm in list(held):
            pos = held[nm]; k = name2idx[nm]; a, b = bounds[k], bounds[k+1]
            j = np.searchsorted(dates[a:b], d, 'right') - 1
            if j < 0: continue
            jj = a + j; px = close[jj]
            hit = None
            if px <= pos['stop']: hit = 'stop'
            elif px >= pos['tp']: hit = 'tp'
            elif jj-1 >= a and dif_a[jj] < dea_a[jj] and dif_a[jj-1] >= dea_a[jj-1]: hit = 'dc'
            elif jj == b-1: hit = 'eod'
            if hit:
                cash += pos['sh']*px*(1-FEE/2); del held[nm]
        # 入场
        while ei < len(ev) and ev[ei]['entry'] == d:
            t = ev[ei]; ei += 1
            if t['code'] in held or len(held) >= SLOTS: continue
            k = name2idx[t['code']]; a, b = bounds[k], bounds[k+1]
            e = np.searchsorted(dates[a:b], d) + a
            if e >= b: continue
            mk = cash
            for n, pos in held.items():
                kk = name2idx[n]; aa, bb = bounds[kk], bounds[kk+1]
                jj = np.searchsorted(dates[aa:bb], d, 'right') - 1
                mk += pos['sh']*(close[aa+jj] if jj >= 0 else pos['entry_px'])
            px = close[e]
            if px > 0:
                alloc = mk*SLOT_W
                cash -= alloc
                held[t['code']] = dict(sh=alloc*(1-FEE/2)/px, stop=px*STOP, tp=px*TP, entry_px=px)
        # 估值
        v = cash
        for nm, pos in held.items():
            k = name2idx[nm]; a, b = bounds[k], bounds[k+1]
            j = np.searchsorted(dates[a:b], d, 'right') - 1
            v += pos['sh']*(close[a+j] if j >= 0 else pos['entry_px'])
        eq.append((d, v))
    s = pd.DataFrame(eq, columns=['date', 'port'])
    s.to_csv(f'{OUT}/池50转折点权益_{label}.csv', index=False)
    e = s.port.values; yrs = (pd.Timestamp(str(s.date.iloc[-1]))-pd.Timestamp(str(s.date.iloc[0]))).days/365.25
    mdd = (e/np.maximum.accumulate(e)-1).min(); ann = (e[-1]/e[0])**(1/yrs)-1
    print(f'{label}: 期末 {e[-1]:.3f} ({e[-1]/e[0]-1:+.1%})  年化 {ann:+.2%}  MDD {mdd:.1%}  Calmar {ann/abs(mdd):.2f}')
    return ann, mdd

print('\n=== 组合模拟(10槽×10%, T+1, 扣费) ===')
portfolio2([t for t in tr_br if t['chan'] == '下跌趋势|中枢下'], '缠论下跌中枢下')
portfolio2([t for t in tr_br if t['state'] == '阶梯已突破解除'], '阶梯解除')
portfolio2(tr_br, '全部池内金叉')

print('\n=== 对照基准(同实现复现值, CHECK20=1 可重现) ===')
print('20只龙头 池内全部金叉(死叉): n=1736 胜率36.7% 平均+1.26% PF 1.44')
print('20只龙头 缠论下跌×中枢下(死叉): n=305 胜率43.3% 平均+3.06% PF 2.19  ← 旧实验2.24可复现 ✓')
print('20只龙头 阶梯解除(死叉): n=105 胜率31.4% 平均+1.08% PF 1.31  ← 旧实验164笔/45.1%/2.52 不可复现 ✗(定义随临时脚本丢失)')
print('H5月度池本身:       年化+6.2%(训练)/+16.3%(样本外) MDD −39.4%/−32.6%')
print('\n注: 均为样本内诊断; 池成员资格只约束入场, 持仓出场由系统自身规则决定。')
