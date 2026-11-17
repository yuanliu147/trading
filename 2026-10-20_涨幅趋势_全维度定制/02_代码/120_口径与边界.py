# -*- coding: utf-8 -*-
"""口径差异：全期连续资金 vs 训练/检验分段；跨2021-12-31持仓的边界效应。"""
import os, numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()
KW = dict(slots=2, sizing='fixed', risk=0, maxpos=.45, heat_cap=.10, gate_thr=.50,
          rank='amt', hold=20, start_cash=20000, cost_rate=.0015, vol_target=.20)

def run(a, b):
    return E.run(a, b, **KW)

full, ft, fm = run('2018-01-01', '2026-10-13')
tr, tt, tm = run('2018-01-01', '2021-12-31')
te, tet, tem = run('2022-01-01', '2026-10-13')
print('全期 %.2f%% / DD %.2f%% / 期末 %.0f' % (fm['cagr']*100, fm['maxdd']*100, fm['final']))
print('训练 %.2f%% / DD %.2f%% / 期末 %.0f' % (tm['cagr']*100, tm['maxdd']*100, tm['final']))
print('检验 %.2f%% / DD %.2f%% / 期末 %.0f' % (tem['cagr']*100, tem['maxdd']*100, tem['final']))
print('训练×检验 复合年化 %.2f%%（理论分段拼接）' % (
    (((tm['final']/20000)*(tem['final']/20000))**(252/(len(tr)+len(te)))-1)*100))
print('全期/分段乘积 差额 %.2f%%' % ((fm['final']/(tm['final']*tem['final']/20000)-1)*100))

# 跨边界交易
ft2 = ft.copy()
ft2['entry_date'] = pd.to_datetime(ft2['entry_date']); ft2['exit_date'] = pd.to_datetime(ft2['exit_date'])
cross = ft2[(ft2.entry_date <= '2021-12-31') & (ft2.exit_date > '2021-12-31')]
cross['acct'] = cross.weight * cross.ret
print('\n跨2021-12-31持仓：')
print(cross[['code','entry_date','exit_date','ret','weight','acct','reason']].round(4).to_string(index=False))
print('跨界贡献合计 %.4f（对数收益）' % cross.acct.sum())
print('\n002694 在哪些分段出现：')
for nm, t in [('全期', ft), ('训练', tt), ('检验', tet)]:
    x = t[t.code == '002694.SZ'] if len(t) and 'code' in t else pd.DataFrame()
    print(nm, x[['code','entry_date','exit_date','ret','weight']].to_string(index=False) if len(x) else '无')

# 分段拼接：训练权益 + 检验权益按训练期末缩放
tr_e = tr.equity.copy(); te_e = te.equity.copy() * (tm['final']/20000)
stitch = pd.concat([tr_e, te_e])
stitch = stitch[~stitch.index.duplicated(keep='first')]
ret = stitch.pct_change().fillna(0)
print('\n拼接（2021年末清仓/重新起算，检验按训练期末缩放）：')
print('总收益 %.2f%%  CAGR %.2f%%  DD %.2f%%' % (
    (stitch.iloc[-1]/20000-1)*100, ((stitch.iloc[-1]/20000)**(252/len(stitch))-1)*100,
    ((stitch/stitch.cummax()-1).min())*100))
# 全期各年
y = full.equity.resample('YE').last().pct_change(); y.iloc[0] = full.equity.loc[full.index.year==2018].iloc[-1]/20000-1
print('\n全期年度：', (y*100).round(1).to_dict())
# 保存拼接权益
pd.DataFrame({'日期': stitch.index, '权益': stitch.values}).to_csv(
    os.path.join(BASE, '03_结果/回撤优化_权益_分段拼接.csv'), index=False, encoding='utf-8-sig')
