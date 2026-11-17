# -*- coding: utf-8 -*-
"""复现门槛：在【原面板 + 原 mb_lib/mb_engine + 原区间 2018-01-01→2026-10-13】上
重跑进攻版（成交额升序、10 日、2×50%、0.25% 成本），必须复现 27.69% / -24.59% / 222 笔。
不复现 => 环境/代码不一致，停止后续工作。
"""
import os
import sys

SRC = '/Users/admin/WorkSpace/trading/2026-10-24_结构交易_主板40系统/06_代码'
sys.path.insert(0, SRC)
import mb_lib as L  # noqa: E402

ATTACK = dict(
    name='进攻版', st_mode='x', age_min=10, age_max=30, vol20_min=0.20, dd60_min=-0.15,
    hold=10, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
    lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
    rank='amt', slots=2, maxpos=0.50, vol_target=None, heat_cap=1.0,
    cost_rate=0.0025,
)

if __name__ == '__main__':
    e = L.fresh_engine()
    E, T, r = L.run_one(e, ATTACK)
    keys = ['全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
            '全期_期末权益', '负年份数', '最差年']
    print({k: round(r[k], 6) if isinstance(r[k], float) else r[k] for k in keys})
    print('2018年:', r['2018年'])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '03_结果')
    os.makedirs(out, exist_ok=True)
    T.to_csv(os.path.join(out, '复现_进攻版逐笔_原区间.csv'), index=False,
             encoding='utf-8-sig', float_format='%.6f')
    expect = dict(cagr=0.2769, maxdd=-0.2459, trades=222)
    got = dict(cagr=r['全期_cagr'], maxdd=r['全期_maxdd'], trades=r['全期_trades'])
    ok = (abs(got['cagr'] - expect['cagr']) < 0.002
          and abs(got['maxdd'] - expect['maxdd']) < 0.002
          and got['trades'] == expect['trades'])
    print('复现判定:', 'PASS' if ok else 'FAIL', got)
