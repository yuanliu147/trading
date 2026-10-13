"""组合回测引擎：严格执行交易系统规则
- 闸门: 红=清仓且禁新开; 黄=风险减半,总仓≤50%; 绿=正常
- 仓位: 股数=floor(权益×2%/(买入-止损),100); 单票市值≤20%; 组合总风险≤6%; 20cm减半
- 止损: 只上移; 棘轮(15%涨幅+40%回撤+三天法则); 横盘3周抬升; 天量换档跟进止损
- T+1; 跌停封死则次日开盘价离场; 连亏3笔停5日
"""
import duckdb, pandas as pd, numpy as np
from signals import load_stock, add_indicators

DB = '/Users/xianyi/Library/Application Support/hithink-finance/data/market.duckdb'
EQUITY0 = 1_000_000.0

class StockCache:
    def __init__(self):
        self.con = duckdb.connect(DB, read_only=True)
        self.cache = {}
    def get(self, code):
        if code not in self.cache:
            df = add_indicators(load_stock(self.con, code))
            df['date'] = pd.to_datetime(df['date'])
            self.cache[code] = df.reset_index(drop=True)
        return self.cache[code]

def limit_pct(code, date):
    pre, ex = code.split('.')
    if pre[:3] in ('300','301'):
        return 0.20 if date >= pd.Timestamp('2020-08-24') else 0.10
    if pre[:3] == '688': return 0.20
    return 0.10

class Engine:
    def __init__(self, start, end, gate_df, sig_df, pool_df, calendar, apprentice=False):
        self.start = pd.Timestamp(start); self.end = pd.Timestamp(end)
        self.gate = gate_df; self.sig = sig_df; self.pool = pool_df
        self.cal = [pd.Timestamp(d) for d in calendar]
        self.apprentice = apprentice
        self.sc = StockCache()
        self.equity = EQUITY0; self.cash = EQUITY0
        self.pos = {}   # code -> dict
        self.trades = []
        self.curve = []
        self.consec_losses = 0; self.halt_until = None
        # 预处理: 每日闸门
        g = gate_df.set_index('week_end')['gate']
        self.gate_by_day = {}
        for d in self.cal:
            monday = d - pd.Timedelta(days=d.weekday())
            we = monday - pd.Timedelta(days=3)
            past = g[g.index <= we]
            self.gate_by_day[d] = past.iloc[-1] if len(past) else 'unknown'
        # 池: 月份 -> codes
        self.pool_by_month = pool_df.groupby('month')['thscode'].apply(set).to_dict()
        self.board_by_code = pool_df.drop_duplicates('thscode').set_index('thscode')['board'].to_dict()
        # 信号按日期
        sig_df = sig_df.copy(); sig_df['date'] = pd.to_datetime(sig_df['date'])
        self.sig_by_day = {d: g.to_dict('records') for d, g in sig_df.groupby('date')}

    def in_pool(self, code, d):
        m = str(d.to_period('M'))
        return code in self.pool_by_month.get(m, set())

    def heat(self):
        h = 0.0
        for c, p in self.pos.items():
            h += p['shares'] * max(0.0, p['last_close'] - p['stop'])
        return h / self.equity

    def mv_total(self):
        return sum(p['shares'] * p['last_close'] for p in self.pos.values())

    def run(self):
        days = [d for d in self.cal if self.start <= d <= self.end]
        for d in days:
            self.step(d)
        # 期末按最后收盘平仓
        for code in list(self.pos):
            p = self.pos[code]
            self.close_pos(code, p['last_close'], days[-1], '期末')
        return self.report()

    def step(self, d):
        g = self.gate_by_day.get(d, 'unknown')
        prev_g = getattr(self, '_prev_g', None)
        # 闸门切换处理(开盘执行)
        if g == 'red' and prev_g != 'red' and self.pos:
            for code in list(self.pos):
                df = self.sc.get(code); row = df[df['date']==d]
                px = row['open'].iloc[0] if len(row) else self.pos[code]['last_close']
                self.close_pos(code, px, d, '红灯清仓')
        elif g == 'yellow' and prev_g == 'green':
            # 降至≤50%市值: 平掉浮盈最少的
            while self.mv_total() > 0.5 * self.equity and self.pos:
                weakest = min(self.pos, key=lambda c: self.pos[c]['last_close']/self.pos[c]['entry'])
                df = self.sc.get(weakest); row = df[df['date']==d]
                px = row['open'].iloc[0] if len(row) else self.pos[weakest]['last_close']
                self.close_pos(weakest, px, d, '黄灯降仓')
        self._prev_g = g

        # 持仓管理: 更新last_close, 检查止损(盘中), 收盘后棘轮/天量
        for code in list(self.pos):
            p = self.pos[code]
            df = self.sc.get(code)
            idx = df.index[df['date']==d]
            if not len(idx): continue
            i = idx[0]; row = df.iloc[i]
            p['last_close'] = row['close']
            if d == p['entry_date']:  # T+1
                continue
            # 止损检查
            if row['low'] <= p['stop']:
                lp = limit_pct(code, d)
                sealed = (row['low']==row['close']) and (row['close'] <= row['open']*(1-lp)*1.001 or
                          (i>0 and row['close'] <= df['close'].iloc[i-1]*(1-lp)*1.001))
                if sealed:
                    pass  # 跌停封死, 明日再走
                else:
                    fill = row['open'] if row['open'] < p['stop'] else p['stop']
                    self.close_pos(code, fill, d, '止损')
                    continue
            else:
                self.after_close(p, row, i, df, code)

        # 入场(尾盘=收盘价成交)
        if g in ('green','yellow'):
            if self.halt_until is None or d > self.halt_until:
                for s in self.sig_by_day.get(d, []):
                    self.try_entry(s, d, g)

        # 记权益
        self.curve.append({'date': d, 'equity': self.cash + self.mv_total(), 'gate': g})
        self.equity = self.cash + self.mv_total()

    def after_close(self, p, row, i, df, code):
        """收盘后: 更新最高收盘/天量换档/棘轮/横盘"""
        d = row['date']
        p['hi_close'] = max(p['hi_close'], row['close'])
        lp = limit_pct(code, d)
        trail_f = 0.995 if lp == 0.10 else 0.99
        # 天量换档: 入场≥3日后, 当日量为60日最大, 且浮盈≥10%
        if not p['trailing'] and (d - p['entry_date']).days >= 4 and row['close'] >= p['entry']*1.10:
            v60 = df['volume'].iloc[max(0,i-60):i].max()
            if i > 0 and row['volume'] > v60:
                p['trailing'] = True
                p['stop'] = max(p['stop'], row['close']*trail_f)
        if p['trailing']:
            if row['close'] >= p['hi_close']:
                p['stop'] = max(p['stop'], row['close']*trail_f)
            return
        # 横盘3周抬升
        if i >= 15:
            seg = df.iloc[i-14:i+1]
            if seg['high'].max()/seg['low'].min() - 1 <= 0.06:
                p['stop'] = max(p['stop'], seg['low'].min()*0.94)
        # 棘轮: 新确认摆动低点
        sw = df['swing_low_confirmed'].iloc[i]
        if not np.isnan(sw) and sw > p['base']:
            gain = p['hi_close']/p['base'] - 1
            if gain >= 0.15:
                pull = (p['hi_close'] - sw)/(p['hi_close'] - p['base']) if p['hi_close'] > p['base'] else 0
                if pull >= 0.40:
                    atr = df['atr14'].iloc[i]
                    new_stop = min(sw*0.94, sw - 1.75*atr)
                    if new_stop > p['stop']:
                        p['stop'] = new_stop; p['base'] = sw

    def try_entry(self, s, d, g):
        code = s['thscode']
        if code in self.pos: return
        if not self.in_pool(code, d): return
        entry, stop = s['entry'], s['stop']
        risk_ps = entry - stop
        if risk_ps <= 0: return
        risk_pct = 0.02 if g == 'green' else 0.01
        if self.apprentice and d < self.start + pd.Timedelta(days=92):
            risk_pct /= 2
        budget = self.equity * risk_pct
        heat_cap = 0.06 if g == 'green' else 0.03
        if self.heat() + budget/self.equity > heat_cap: return
        shares = int(budget / risk_ps / 100) * 100
        mv_cap = (0.20 if g == 'green' else 0.10) * self.equity
        shares = min(shares, int(mv_cap / entry / 100) * 100)
        if self.board_by_code.get(code) == 'cm20': shares //= 2
        if shares < 100: return
        # 跌停测试: 假设明日跌停的亏损 ≤ 4% 权益
        lp = limit_pct(code, d)
        while shares >= 100 and shares*entry*lp > 0.04*self.equity:
            shares -= 100
        if shares < 100 or shares*entry > self.cash: 
            shares = min(shares, int(self.cash/entry/100)*100)
        if shares < 100: return
        self.cash -= shares*entry
        self.pos[code] = {'shares': shares, 'entry': entry, 'stop': stop, 'base': s['base'],
                          'entry_date': d, 'type': s['type'], 'hi_close': entry,
                          'last_close': entry, 'trailing': False, 'target': s['target']}

    def close_pos(self, code, px, d, reason):
        p = self.pos.pop(code)
        self.cash += p['shares']*px
        pnl = p['shares']*(px - p['entry'])
        pnl_pct = px/p['entry'] - 1
        self.trades.append({'code': code, 'type': p['type'], 'entry_date': str(p['entry_date'])[:10],
                            'exit_date': str(d)[:10], 'entry': round(p['entry'],3), 'exit': round(px,3),
                            'shares': p['shares'], 'pnl': round(pnl,0), 'pnl_pct': round(pnl_pct*100,2),
                            'reason': reason})
        if pnl < 0:
            self.consec_losses += 1
            if self.consec_losses >= 3:
                self.halt_until = d + pd.Timedelta(days=7)
                self.consec_losses = 0
        else:
            self.consec_losses = 0

    def report(self):
        tr = pd.DataFrame(self.trades)
        cur = pd.DataFrame(self.curve)
        ret = self.equity/EQUITY0 - 1
        wins = tr[tr['pnl']>0] if len(tr) else tr
        out = {'return': ret, 'n_trades': len(tr),
               'win_rate': len(wins)/len(tr) if len(tr) else np.nan,
               'avg_win': wins['pnl_pct'].mean() if len(wins) else np.nan,
               'avg_loss': tr[tr['pnl']<=0]['pnl_pct'].mean() if len(tr) and len(tr)>len(wins) else np.nan,
               'max_dd': (cur['equity']/cur['equity'].cummax()-1).min() if len(cur) else np.nan,
               'trades': tr, 'curve': cur}
        return out
