#!/usr/bin/env python3
"""确定性缠论引擎: 分型→笔→中枢→背驰→买卖点(日线级别)

规则实现(新笔标准):
- 包含处理: 与前一根合并K线比高低, 上升取高高/下降取低低
- 分型: 合并K线上 顶=中间高低皆最高, 底=中间高低皆最低
- 笔: 顶底分型交替, 两分型中心间隔≥3根合并K线(新笔); 同类分型取更极端者
- 中枢: 连续3笔重叠区 ZG=min(三高), ZD=max(三低), ZG>ZD; 后续笔与[ZD,ZG]相交则延伸
- 底背驰(一买前提): 相邻两下跌笔, 后笔创新低 但 MACD绿柱面积更小 且 DIF低点更高
- 顶背驰(一卖): 相邻两上涨笔, 后笔创新高 但 MACD红柱面积更小 且 DIF高点更低

信号于分型确认日(右K线日)发出, 只引用当日及之前的结构, 全因果:
- 一买: 下跌趋势(前方存在中枢)末段底背驰 → 底分型确认
- 二买: 一买后首根下跌笔不创新低(低点>一买低点) → 底分型确认; 止损=一买低点
- 三买: 向上离开中枢后首次回抽笔低点>ZG → 底分型确认; 止损=ZG; 每中枢只做第一次回抽,
        且只做多头方向第一个中枢(课79)
- 一卖: 顶背驰 → 顶分型确认(减半信号)
- 三卖: 向下离开中枢后反抽笔高点<ZD → 顶分型确认(清仓信号)
"""
import numpy as np
import pandas as pd


def merge_bars(h, l):
    """包含处理. 返回 mh, ml, m2r(合并K线索引->原始结束索引), r2m(原始->合并索引)"""
    n = len(h)
    mh, ml, m2r = [], [], []
    r2m = np.zeros(n, dtype=int)
    direction = 0  # 0未定 1上 -1下
    for i in range(n):
        if not mh:
            mh.append(h[i]); ml.append(l[i]); m2r.append(i); r2m[i] = 0
            continue
        if h[i] > mh[-1] and l[i] > ml[-1]:
            mh.append(h[i]); ml.append(l[i]); m2r.append(i); direction = 1
        elif h[i] < mh[-1] and l[i] < ml[-1]:
            mh.append(h[i]); ml.append(l[i]); m2r.append(i); direction = -1
        else:  # 包含
            if direction >= 0:
                mh[-1] = max(mh[-1], h[i]); ml[-1] = max(ml[-1], l[i])
            else:
                mh[-1] = min(mh[-1], h[i]); ml[-1] = min(ml[-1], l[i])
            m2r[-1] = i
        r2m[i] = len(mh) - 1
    return np.array(mh), np.array(ml), m2r, r2m


def find_fractals(mh, ml):
    """返回 fractals: list of (center_midx, type) type=1顶 -1底"""
    fr = []
    for i in range(1, len(mh) - 1):
        if mh[i] > mh[i-1] and mh[i] > mh[i+1] and ml[i] > ml[i-1] and ml[i] > ml[i+1]:
            fr.append((i, 1))
        elif ml[i] < ml[i-1] and ml[i] < ml[i+1] and mh[i] < mh[i-1] and mh[i] < mh[i+1]:
            fr.append((i, -1))
    return fr


def build_bi(fractals, mh, ml):
    """构建笔. 返回 list of dict(dir, s_midx, e_midx, hi, lo)"""
    bis = []
    last = None  # (midx, type)
    for midx, typ in fractals:
        if last is None:
            last = (midx, typ)
            continue
        lm, lt = last
        if typ == lt:
            # 同类: 更极端则替换最后一笔端点
            more = (typ == 1 and mh[midx] > mh[lm]) or (typ == -1 and ml[midx] < ml[lm])
            if more:
                last = (midx, typ)
                if bis:
                    bis[-1]['e_midx'] = midx
                    if lt == 1:
                        bis[-1]['hi'] = mh[midx]
                    else:
                        bis[-1]['lo'] = ml[midx]
            continue
        # 异类: 间隔≥3(新笔)才成笔
        if midx - lm >= 3:
            d = 1 if lt == -1 else -1  # 底→顶 为上升笔
            lo = ml[lm] if lt == -1 else ml[midx]
            hi = mh[midx] if lt == -1 else mh[lm]
            bis.append(dict(dir=d, s_midx=lm, e_midx=midx, hi=hi, lo=lo))
            last = (midx, typ)
    return bis


def build_zhongshu(bis):
    """返回 list of dict(zg, zd, s_bi, e_bi, done)  e_bi为最后一根相交笔(含延伸)"""
    zs = []
    i = 0
    while i + 2 < len(bis):
        b1, b2, b3 = bis[i], bis[i+1], bis[i+2]
        zg = min(b1['hi'], b2['hi'], b3['hi'])
        zd = max(b1['lo'], b2['lo'], b3['lo'])
        if zg > zd:
            j = i + 2
            while j + 1 < len(bis) and bis[j+1]['lo'] <= zg and bis[j+1]['hi'] >= zd:
                j += 1
            zs.append(dict(zg=zg, zd=zd, s_bi=i, e_bi=j))
            i = j + 1
        else:
            i += 1
    return zs


def analyze(code, g):
    """单票全量分析. 返回该票日线df附加列: sig1/2/3买, stop, sig1sell, sig3sell, ma10, chg, limit_dn, vol_x
    信号全部为分型确认日(原始K线索引)"""
    g = g.sort_values('date').reset_index(drop=True)
    n = len(g)
    o, h, l, c, v = (g[x].to_numpy(float) for x in ('open', 'high', 'low', 'close', 'volume'))
    dates = g['date'].to_numpy()

    dif = pd.Series(c).ewm(span=12, adjust=False).mean() - pd.Series(c).ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    bar = 2 * (dif - dea)
    dif_a, bar_a = dif.to_numpy(), bar.to_numpy()

    mh, ml, m2r, r2m = merge_bars(h, l)
    fractals = find_fractals(mh, ml)
    bis = build_bi(fractals, mh, ml)
    zss = build_zhongshu(bis)

    # 笔端点映射到原始K线
    for b in bis:
        b['s_r'] = m2r[b['s_midx']]
        b['e_r'] = m2r[b['e_midx']]
        seg = slice(b['s_r'], b['e_r'] + 1)
        b['raw_hi'] = float(np.max(h[seg]))
        b['raw_lo'] = float(np.min(l[seg]))
        if b['dir'] == 1:
            b['area'] = float(bar_a[seg][bar_a[seg] > 0].sum())
            b['dif_ext'] = float(np.nanmax(dif_a[seg]))
        else:
            b['area'] = float(bar_a[seg][bar_a[seg] < 0].sum())
            b['dif_ext'] = float(np.nanmin(dif_a[seg]))

    sig1 = np.zeros(n, bool); stop1 = np.full(n, np.nan)
    sig2 = np.zeros(n, bool); stop2 = np.full(n, np.nan)
    sig3 = np.zeros(n, bool); stop3 = np.full(n, np.nan)
    sig1sell = np.zeros(n, bool)
    sig3sell = np.zeros(n, bool)

    # 每根下跌笔结束时(底分型确认日=e_r+1? e_r为分型中心映射) 确认日=底分型右K线原始日
    def confirm_day(b):
        # 笔端点m2r是分型中心所在原始日; 确认需右K线: 合并K线e_midx+1的原始日
        cm = b['e_midx'] + 1
        if cm < len(m2r):
            return m2r[cm]
        return None

    # ---- 一卖(顶背驰): 上升笔与前一根上升笔比 ----
    up_idx = [i for i, b in enumerate(bis) if b['dir'] == 1]
    for k in range(1, len(up_idx)):
        i2 = up_idx[k]
        i1 = up_idx[k-1]
        b2, b1 = bis[i2], bis[i1]
        if b2['raw_hi'] > b1['raw_hi'] and b1['area'] > 0 and b2['area'] < 0.6 * b1['area'] \
                and b2['dif_ext'] < b1['dif_ext']:
            cd = confirm_day(b2)
            if cd is not None:
                sig1sell[cd] = True

    # ---- 三卖: 中枢被向下离开后反抽不回ZD ----
    for zs in zss:
        left_dn = None
        for j in range(zs['e_bi'] + 1, len(bis)):
            b = bis[j]
            if b['dir'] == -1 and b['raw_lo'] < zs['zd']:
                left_dn = j
                break
            if b['dir'] == 1 and b['raw_hi'] > zs['zg']:
                break  # 向上离开, 无三卖
        if left_dn is not None:
            for j in range(left_dn + 1, min(left_dn + 4, len(bis))):
                b = bis[j]
                if b['dir'] == 1:
                    if b['raw_hi'] < zs['zd']:
                        cd = confirm_day(b)
                        if cd is not None:
                            sig3sell[cd] = True
                    break

    # ---- 一买/二买: 下跌笔底背驰 ----
    dn_idx = [i for i, b in enumerate(bis) if b['dir'] == -1]
    last_buy1 = None  # (bi_index, 低点)
    for k in range(1, len(dn_idx)):
        i2 = dn_idx[k]
        i1 = dn_idx[k-1]
        b2, b1 = bis[i2], bis[i1]
        cd = confirm_day(b2)
        if cd is None:
            continue
        # 前方存在中枢(下跌趋势环境)
        has_zs = any(z['e_bi'] < i2 for z in zss)
        div = b2['raw_lo'] < b1['raw_lo'] and b1['area'] < 0 and b2['area'] > 0.6 * b1['area'] \
            and b2['dif_ext'] > b1['dif_ext']  # area为负, 背驰=绝对值更小
        if has_zs and div:
            sig1[cd] = True; stop1[cd] = b2['raw_lo']
            last_buy1 = (i2, b2['raw_lo'])
            continue
        # 二买: 一买之后下跌笔不创新低
        if last_buy1 is not None and i2 > last_buy1[0] and b2['raw_lo'] > last_buy1[1]:
            sig2[cd] = True; stop2[cd] = last_buy1[1]
            last_buy1 = None  # 一次一买只配一次二买

    # ---- 三买: 向上离开中枢后首次回抽不破ZG; 只做多头第一个中枢 ----
    # 多头起点: 最近一次一买/二买信号之后形成的第一个中枢
    buy_pts = sorted([i for i in range(n) if sig1[i] or sig2[i]])
    used_zs = set()
    for zi, zs in enumerate(zss):
        if zi in used_zs:
            continue
        # 中枢起点映射原始日, 要求在最近买点之后(或无买点则不限)——多头第一个中枢
        zs_start_r = bis[zs['s_bi']]['s_r']
        prev_buys = [d for d in buy_pts if d <= zs_start_r]
        earlier_zs_after_buy = 0
        if prev_buys:
            lb = prev_buys[-1]
            earlier_zs_after_buy = sum(1 for z2 in zss[:zi] if bis[z2['s_bi']]['s_r'] > lb)
        if earlier_zs_after_buy >= 1:
            continue  # 第二个及以上中枢放弃(课79)
        # 找向上离开笔
        for j in range(zs['e_bi'] + 1, len(bis)):
            b = bis[j]
            if b['dir'] == 1 and b['raw_hi'] > zs['zg']:
                # 其后第一根下跌笔
                if j + 1 < len(bis) and bis[j+1]['dir'] == -1:
                    pb = bis[j+1]
                    if pb['raw_lo'] > zs['zg']:
                        cd = confirm_day(pb)
                        if cd is not None:
                            sig3[cd] = True; stop3[cd] = zs['zg']
                    used_zs.add(zi)
                break
            if b['dir'] == -1 and b['raw_lo'] < zs['zd']:
                break  # 先向下破了, 该中枢不再找三买

    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    chg = c / prev_c - 1
    from chan_system_backtest import limit_pct
    lp = np.array([limit_pct(code, pd.Timestamp(d)) for d in dates])
    limit_dn = chg <= -lp + 0.005
    limit_up = chg >= lp - 0.005
    ma10 = pd.Series(c).rolling(10).mean().to_numpy()
    vol5p = pd.Series(v).shift(1).rolling(5).mean().to_numpy()
    vol_x = v / vol5p

    g['sig1'], g['stop1'] = sig1, stop1
    g['sig2'], g['stop2'] = sig2, stop2
    g['sig3'], g['stop3'] = sig3, stop3
    g['sig1sell'], g['sig3sell'] = sig1sell, sig3sell
    g['ma10'], g['chg'], g['limit_dn'], g['limit_up'], g['vol_x'] = ma10, chg, limit_dn, limit_up, vol_x
    return g
