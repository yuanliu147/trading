# -*- coding: utf-8 -*-
"""可复用的2万元组合引擎：在102基础上加入市场趋势、波动目标、回撤节流、盈利保护、
分批止盈、行业分散、MA20退出等可解释叠加。所有信号只用t-1及以前，t开盘成交。"""
import pickle, os, numpy as np, pandas as pd

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'


class Engine:
    def __init__(self, panel_path=None):
        if panel_path is None:
            panel_path = os.path.join(BASE, '01_数据/全A面板_2016_2026_含退市.pkl')
        P = pickle.load(open(panel_path, 'rb'))
        self.P = P
        self.dates = pd.DatetimeIndex(P['meta']['dates'])
        self.codes = P['meta']['codes']
        self.code_idx = {c: i for i, c in enumerate(self.codes)}
        g = lambda k: P[k].values
        self.O, self.H, self.L, self.C = g('open'), g('high'), g('low'), g('close')
        self.ATR, self.AMT = g('atr_pct'), g('amt20')
        self.U, self.AGE, self.DD60 = g('up_struct'), g('trend_age'), g('dd60')
        self.LU, self.SBR, self.SR20 = g('lu_count20'), g('sector_breadth20'), g('sector_ret20')
        self.MB, self.NG = g('market_breadth20'), g('next_gap')
        self.OWD, self.OWU = g('one_word_down'), g('one_word_up')
        self.LD, self.MCAP = g('limit_down'), g('mcap')
        self.MA20, self.MA60 = g('ma20'), g('ma60')
        self.IND = g('industry_id')
        self.MKT, self.MKT_MA20, self.MKT_MA60 = g('market_index'), g('market_ma20'), g('market_ma60')
        self.MKT_SLOPE60, self.MKT_BREADTH60 = g('market_slope60'), g('market_breadth60')
        self.mkt_vol = (pd.Series(self.MKT).pct_change().rolling(20, min_periods=15).std()
                        .values * np.sqrt(252))
        self.msig = (self.MKT > self.MKT_MA60) & (self.MKT_SLOPE60 > 0)
        self.msig20 = self.MKT > self.MKT_MA20
        date_ok = (self.dates >= pd.Timestamp('2017-10-13'))[:, None]
        valid = self.U & (self.AMT >= 3e7) & (self.C >= 2) & (self.AGE >= 5) & date_ok
        self.base_nogate = (valid & (self.AGE <= 30) & (self.ATR <= 0.044) & (self.LU == 0) &
                            (self.SBR >= 0.6) & (self.SR20 > 0.05) & (self.DD60 > -0.08))

    def run(self, start, end, slots=2, sizing='fixed', risk=0.03, maxpos=0.5, heat_cap=0.10,
            gate_thr=0.5, gate_mode='mb', rank='amt', hold=20, cost_rate=0.0015, min_cost=5.0, rank_seed=None,
            start_cash=20000.0, risk_basis='atr', stop_pct=0.20,
            vol_target=None, vol_floor=0.3, size_breadth=False,
            dd_half=None, dd_stop=None, dd_factor=0.5,
            profit_lock=False, lock_trigger=0.15, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
            partial_trigger=None, partial_frac=0.5,
            exit_ma20=False, exit_min_hold=5,
            sector_cap=False, skip_entry_lu=True, force_close=False, ret_daily=False,
            loss_pause_n=None, loss_pause_days=5, dd_peak_window=None,
            eq_vol_target=None, eq_vol_floor=0.3, regime_exit=None, skip_entries=None):
        dates = self.dates; codes = self.codes
        O, H, L, C = self.O, self.H, self.L, self.C
        ATR, AMT, SBR, SR20, MB = self.ATR, self.AMT, self.SBR, self.SR20, self.MB
        NG, OWD, OWU, LU, MA20, IND = self.NG, self.OWD, self.OWU, self.LU, self.MA20, self.IND
        si = dates.searchsorted(pd.Timestamp(start)); ei = dates.searchsorted(pd.Timestamp(end), side='right') - 1
        gate = MB[:, None] > gate_thr
        if gate_mode == 'mb':
            pass
        elif gate_mode == 'mb_mkt60':
            gate = gate & self.msig[:, None]
        elif gate_mode == 'mb_mkt20':
            gate = gate & self.msig20[:, None]
        elif gate_mode == 'mb_slope':
            gate = gate & (self.MKT_SLOPE60 > 0)[:, None]
        elif gate_mode == 'mb_b60':
            gate = gate & (self.MKT_BREADTH60[:, None] > 0.5)
        else:
            raise ValueError(gate_mode)
        sig = self.base_nogate & gate

        cash = start_cash; pos = []; eqc = []; trades = []; peak_eq = start_cash
        eq_hist = [start_cash]; loss_streak = 0; pause_until = -1
        for t in range(si, ei + 1):
            # 0) 市场转弱退出：t-1收盘判断，t开盘卖出（只砍亏损仓）
            if regime_exit is not None:
                for p in list(pos):
                    j = p['j']
                    if t <= p['entry_i'] or not np.isfinite(O[t, j]):
                        continue
                    prev = C[t - 1, j] if np.isfinite(C[t - 1, j]) else p['last']
                    losing = prev < p['entry']
                    cond = False
                    if regime_exit in ('breadth', 'both') and np.isfinite(MB[t - 1]):
                        cond = cond or (MB[t - 1] < 0.35 and losing)
                    if regime_exit in ('ma60', 'both') and np.isfinite(self.MKT_MA60[t - 1]):
                        cond = cond or (self.MKT[t - 1] < self.MKT_MA60[t - 1] and prev < p['entry'] * 0.97)
                    if not cond or OWD[t, j]:
                        continue
                    fill = O[t, j]
                    gross = p['sh'] * fill; fee = max(min_cost, gross * cost_rate); cash += gross - fee
                    trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                                       entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason='市场转弱',
                                       hold=t - p['entry_i'] + 1, shares=p['sh'],
                                       pnl=(fill - p['entry']) * p['sh'] - fee - p['entry_fee'] * p['sh'] / p['init_sh'],
                                       weight=p['w'] * p['sh'] / p['init_sh']))
                    if loss_pause_n:
                        if fill / p['entry'] - 1 <= 0:
                            loss_streak += 1
                        else:
                            loss_streak = 0
                        if loss_streak >= loss_pause_n:
                            pause_until = t + loss_pause_days; loss_streak = 0
                    pos.remove(p)
            # 1) 盘中止损（含盈利保护后的止损）
            for p in list(pos):
                j = p['j']
                if t < p['entry_i']:
                    continue
                if np.isfinite(L[t, j]) and L[t, j] <= p['stop']:
                    if OWD[t, j]:
                        continue
                    if not np.isfinite(O[t, j]):
                        continue
                    fill = min(O[t, j], p['stop'])
                    gross = p['sh'] * fill; fee = max(min_cost, gross * cost_rate); cash += gross - fee
                    trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                                       entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason='止损',
                                       hold=t - p['entry_i'] + 1, shares=p['sh'],
                                       pnl=(fill - p['entry']) * p['sh'] - fee - p['entry_fee'] * p['sh'] / p['init_sh'],
                                       weight=p['w'] * p['sh'] / p['init_sh']))
                    if loss_pause_n:
                        if fill / p['entry'] - 1 <= 0:
                            loss_streak += 1
                        else:
                            loss_streak = 0
                        if loss_streak >= loss_pause_n:
                            pause_until = t + loss_pause_days; loss_streak = 0
                    pos.remove(p)
            # 2) 分批止盈（盘中触价）
            if partial_trigger is not None:
                for p in list(pos):
                    j = p['j']
                    if t < p['entry_i'] or p.get('partial_done') or p['sh'] < 200:
                        continue
                    if not np.isfinite(H[t, j]) or H[t, j] < p['entry'] * (1 + partial_trigger):
                        continue
                    if not np.isfinite(O[t, j]):
                        continue
                    trig = p['entry'] * (1 + partial_trigger)
                    fill = O[t, j] if O[t, j] >= trig else trig
                    sell_sh = int(p['sh'] * partial_frac / 100) * 100
                    if sell_sh < 100:
                        continue
                    gross = sell_sh * fill; fee = max(min_cost, gross * cost_rate); cash += gross - fee
                    p['sh'] -= sell_sh
                    trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                                       entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason='分批',
                                       hold=t - p['entry_i'] + 1, shares=sell_sh,
                                       pnl=(fill - p['entry']) * sell_sh - fee - p['entry_fee'] * sell_sh / p['init_sh'],
                                       weight=p['w'] * sell_sh / p['init_sh']))
                    p['partial_done'] = True
            # 3) 到期 / 破MA20
            for p in list(pos):
                j = p['j']
                if t < p['entry_i']:
                    continue
                reason = None
                if t >= p['exit_i']:
                    reason = '到期'
                elif exit_ma20 and t >= p['entry_i'] + exit_min_hold:
                    if np.isfinite(C[t, j]) and np.isfinite(MA20[t, j]) and C[t, j] < MA20[t, j]:
                        reason = '破MA20'
                if reason is None:
                    continue
                if OWD[t, j]:
                    continue
                if not np.isfinite(C[t, j]):
                    future = np.where(np.isfinite(C[t:, j]))[0]
                    if len(future) > 0:
                        continue
                    fill = p['last'] * 0.5; reason = '数据终止'
                else:
                    fill = C[t, j]
                gross = p['sh'] * fill; fee = max(min_cost, gross * cost_rate); cash += gross - fee
                trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                                   entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason=reason,
                                   hold=t - p['entry_i'] + 1, shares=p['sh'],
                                   pnl=(fill - p['entry']) * p['sh'] - fee - p['entry_fee'] * p['sh'] / p['init_sh'],
                                   weight=p['w'] * p['sh'] / p['init_sh']))
                if loss_pause_n and reason in ('到期', '破MA20', '数据终止'):
                    if fill / p['entry'] - 1 <= 0:
                        loss_streak += 1
                    else:
                        loss_streak = 0
                    if loss_streak >= loss_pause_n:
                        pause_until = t + loss_pause_days; loss_streak = 0
                pos.remove(p)
            # 4) 收盘后更新盈利保护止损（次日生效）
            if profit_lock:
                for p in pos:
                    j = p['j']
                    if t < p['entry_i'] or not np.isfinite(H[t, j]):
                        continue
                    gain = H[t, j] / p['entry'] - 1
                    if gain >= trail_trigger:
                        p['stop'] = max(p['stop'], H[t, j] * (1 - trail_pct))
                    elif gain >= lock_trigger:
                        p['stop'] = max(p['stop'], p['entry'] * (1 + lock_gain))
            # 5) 开盘买入（信号t-1）
            if t > si:
                cand = np.where(sig[t - 1])[0]
                if len(cand) and not (loss_pause_n and t < pause_until):
                    if rank == 'amt':
                        cand = sorted(cand, key=lambda j: AMT[t - 1, j])
                    elif rank == 'lowvol':
                        cand = sorted(cand, key=lambda j: (ATR[t - 1, j], AMT[t - 1, j]))
                    elif rank == 'pullback':
                        cand = sorted(cand, key=lambda j: (abs(C[t - 1, j] / MA20[t - 1, j] - 1)
                                                           if np.isfinite(MA20[t - 1, j]) else 9, AMT[t - 1, j]))
                    elif rank == 'sbr':
                        cand = sorted(cand, key=lambda j: -SBR[t - 1, j])
                    elif rank in ('sbr_mid', 'sbr_mid65', 'sbr_mid85'):
                        target = {'sbr_mid': 0.75, 'sbr_mid65': 0.65, 'sbr_mid85': 0.85}[rank]
                        cand = sorted(cand, key=lambda j: (abs(SBR[t - 1, j] - target)
                                                           if np.isfinite(SBR[t - 1, j]) else 9,
                                                           AMT[t - 1, j]))
                    elif rank == 'sr20':
                        cand = sorted(cand, key=lambda j: (-SR20[t - 1, j]
                                                           if np.isfinite(SR20[t - 1, j]) else 9,
                                                           AMT[t - 1, j]))
                    elif rank == 'age':
                        cand = sorted(cand, key=lambda j: (self.AGE[t - 1, j]
                                                           if np.isfinite(self.AGE[t - 1, j]) else 999,
                                                           AMT[t - 1, j]))
                    elif rank == 'dd60':
                        cand = sorted(cand, key=lambda j: (-self.DD60[t - 1, j]
                                                           if np.isfinite(self.DD60[t - 1, j]) else 9,
                                                           AMT[t - 1, j]))
                    elif rank == 'price':
                        cand = sorted(cand, key=lambda j: (C[t - 1, j]
                                                           if np.isfinite(C[t - 1, j]) else 9,
                                                           AMT[t - 1, j]))
                    elif rank == 'mcap':
                        cand = sorted(cand, key=lambda j: (self.MCAP[t - 1, j]
                                                           if np.isfinite(self.MCAP[t - 1, j]) else 9,
                                                           AMT[t - 1, j]))
                    elif rank == 'mcap_desc':
                        cand = sorted(cand, key=lambda j: (-self.MCAP[t - 1, j]
                                                           if np.isfinite(self.MCAP[t - 1, j]) else -9,
                                                           AMT[t - 1, j]))
                    elif rank == 'random':
                        rng = np.random.default_rng(rank_seed)
                        cand = list(cand)
                        rng.shuffle(cand)
                    else:
                        raise ValueError(rank)
                    held = {p['j'] for p in pos}
                    inds = {int(IND[t - 1, p['j']]) for p in pos if int(IND[t - 1, p['j']]) >= 0}
                    for j in cand:
                        if len(pos) >= slots:
                            break
                        if j in held:
                            continue
                        if skip_entries and (codes[j], dates[t]) in skip_entries:
                            continue
                        if not np.isfinite(O[t, j]) or O[t, j] <= 0:
                            continue
                        if np.isfinite(NG[t - 1, j]) and NG[t - 1, j] > 0.05:
                            continue
                        if skip_entry_lu and (OWU[t, j] or (LU[t, j] and np.isfinite(H[t, j]) and O[t, j] >= H[t, j] - 1e-9)):
                            continue
                        if sector_cap and int(IND[t - 1, j]) >= 0 and int(IND[t - 1, j]) in inds:
                            continue
                        entry = O[t, j]
                        eq = cash + sum(p['sh'] * (C[t - 1, p['j']] if np.isfinite(C[t - 1, p['j']]) else p['last'])
                                        for p in pos)
                        mult = 1.0
                        if vol_target is not None and np.isfinite(self.mkt_vol[t - 1]):
                            mult *= float(np.clip(vol_target / self.mkt_vol[t - 1], vol_floor, 1.0))
                        if eq_vol_target is not None and len(eq_hist) >= 21:
                            arr = np.asarray(eq_hist[-21:])
                            rr = np.diff(arr) / arr[:-1]
                            ev = rr.std() * np.sqrt(252)
                            if ev > 1e-9:
                                mult *= float(np.clip(eq_vol_target / ev, eq_vol_floor, 1.0))
                        if size_breadth:
                            mult *= float(np.clip((MB[t - 1] - gate_thr) / 0.15 + 0.4, 0.4, 1.0))
                        if dd_half is not None:
                            peak_ref = max(eq_hist[-dd_peak_window:]) if dd_peak_window else peak_eq
                            ddprev = eq / peak_ref - 1
                            if dd_stop is not None and ddprev <= -dd_stop:
                                mult = 0.0
                            elif ddprev <= -dd_half:
                                mult *= dd_factor
                        if mult <= 0:
                            continue
                        maxpos_eff = maxpos * mult
                        if entry * 100 > eq * maxpos_eff or entry * 100 > cash:
                            continue
                        sd = stop_pct if risk_basis == 'stop' else float(np.clip(2 * ATR[t - 1, j], 0.04, 0.12))
                        if sizing == 'risk':
                            target = int((eq * risk * mult / (entry * sd)) // 100) * 100
                            target = min(target, int((eq * maxpos_eff / entry) // 100) * 100)
                        else:
                            target = int((eq * maxpos_eff / entry) // 100) * 100
                        if target < 100:
                            target = 100 if entry * 100 <= eq * maxpos_eff and entry * 100 <= cash else 0
                        if target <= 0:
                            continue
                        heat = sum(p['sh'] * p['entry'] * p['sd'] for p in pos) / max(eq, 1e-9)
                        sh = target
                        while sh >= 100 and heat + sh * entry * sd / max(eq, 1e-9) > heat_cap:
                            sh -= 100
                        if sh < 100:
                            continue
                        gross = sh * entry; fee = max(min_cost, gross * cost_rate)
                        if gross + fee > cash:
                            sh = int((cash / (entry * (1 + cost_rate))) // 100) * 100
                            if sh < 100:
                                continue
                            gross = sh * entry; fee = max(min_cost, gross * cost_rate)
                            if gross + fee > cash:
                                continue
                        cash -= gross + fee
                        pos.append(dict(j=j, entry_i=t, entry=entry, sh=sh, init_sh=sh, sd=sd,
                                        stop=entry * (1 - stop_pct), last=entry, exit_i=t + hold - 1,
                                        w=gross / eq, entry_fee=fee, partial_done=False))
                        held.add(j)
                        if sector_cap and int(IND[t - 1, j]) >= 0:
                            inds.add(int(IND[t - 1, j]))
            # 6) 收盘估值
            val = 0.0; val_low = 0.0
            for p in pos:
                j = p['j']
                px = C[t, j] if np.isfinite(C[t, j]) else p['last']
                lpx = L[t, j] if np.isfinite(L[t, j]) else px
                p['last'] = px
                val += p['sh'] * px; val_low += p['sh'] * lpx
            eqnow = cash + val
            peak_eq = max(peak_eq, eqnow)
            eq_hist.append(eqnow)
            eqc.append((dates[t], eqnow, cash, len(pos), cash + val_low))
        # 期末强制平仓（可选，仅用于已实现口径）
        if force_close and pos:
            t = ei
            for p in list(pos):
                j = p['j']
                fill = C[t, j] if np.isfinite(C[t, j]) else p['last']
                gross = p['sh'] * fill; fee = max(min_cost, gross * cost_rate); cash += gross - fee
                trades.append(dict(code=codes[j], entry_date=dates[p['entry_i']], exit_date=dates[t],
                                   entry=p['entry'], exit=fill, ret=fill / p['entry'] - 1, reason='期末',
                                   hold=t - p['entry_i'] + 1, shares=p['sh'],
                                   pnl=(fill - p['entry']) * p['sh'] - fee - p['entry_fee'] * p['sh'] / p['init_sh'],
                                   weight=p['w'] * p['sh'] / p['init_sh']))
                pos.remove(p)
            eqc[-1] = (dates[t], cash, 0, 0, cash)
        E = pd.DataFrame(eqc, columns=['date', 'equity', 'cash', 'npos', 'equity_low']).set_index('date')
        E['ret'] = E.equity.pct_change().fillna(0)
        E['exposure'] = 1 - E.cash / E.equity
        T = pd.DataFrame(trades)
        total = E.equity.iloc[-1] / start_cash - 1
        cagr = (1 + total) ** (252 / len(E)) - 1 if len(E) else np.nan
        dd = (E.equity / E.equity.cummax() - 1).min()
        ddl = (E.equity_low / E.equity.cummax() - 1).min()
        win = (T.ret > 0).mean() if len(T) else np.nan
        pf = T.pnl[T.pnl > 0].sum() / max(1e-9, -T.pnl[T.pnl < 0].sum()) if len(T) else np.nan
        contrib = (T.weight * T.ret) if len(T) else pd.Series(dtype=float)
        met = dict(total=total, cagr=cagr, maxdd=dd, maxdd_low=ddl,
                   sharpe=E.ret.mean() / (E.ret.std() + 1e-9) * np.sqrt(252),
                   calmar=cagr / (abs(dd) + 1e-9), trades=len(T), win=win, pf=pf,
                   avg_hold=T.hold.mean() if len(T) else np.nan, avg_pos=E.npos.mean(),
                   final=E.equity.iloc[-1], avg_weight=T.weight.mean() if len(T) else np.nan,
                   avg_exposure=E.exposure.mean(), worst_trade=(contrib.min() if len(T) else np.nan),
                   top5_share=(contrib.nlargest(5).sum() / max(1e-9, np.log(E.equity.iloc[-1] / start_cash))
                               if len(T) else np.nan),
                   max_weight=(T.weight.max() if len(T) else np.nan))
        if ret_daily:
            met['daily'] = E.ret
        return E, T, met
