# -*- coding: utf-8 -*-
"""结构交易·主板40%系统：生成核心版/进攻版权益、逐笔、年度、压力测试与图表。

运行：
    cd "2026-10-24_结构交易_主板40系统/06_代码"
    python3 run_final.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '05_证据')
FIG = os.path.join(ROOT, '07_图表')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, HERE)
import mb_lib as L
from mb_engine import Engine

L.eng = None  # 不再使用旧的 108 引擎

# 中文字体
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ---------------- 配置 ----------------
CORE = dict(
    name='核心版', st_mode='x', age_min=10, age_max=30, vol20_min=0.20, dd60_min=-0.15,
    hold=20, partial_trigger=0.25, partial_frac=0.5, profit_lock=True,
    lock_trigger=0.10, lock_gain=0.0, trail_trigger=0.25, trail_pct=0.10,
    rank='mcap', slots=2, maxpos=0.45, vol_target=0.20, heat_cap=0.10,
    cost_rate=0.0025,
)
ATTACK = dict(
    CORE, name='进攻版', hold=10, maxpos=0.50, vol_target=None, heat_cap=1.0,
)


def run_cfg(cfg):
    e = L.fresh_engine()
    E, T, r = L.run_one(e, cfg)
    return e, E, T, r


def yearly(E, start=20000.0):
    rows = []
    prev = start
    for y in range(2018, 2027):
        x = E.loc[str(y)]
        if len(x) == 0:
            continue
        rows.append(dict(年份=y, 收益=x.equity.iloc[-1] / prev - 1,
                         年末权益=x.equity.iloc[-1]))
        prev = x.equity.iloc[-1]
    return pd.DataFrame(rows)


def leverage_curve(E, Lv=1.4, rate=0.065, freq='Q', start=20000.0):
    r0 = E.equity.pct_change().fillna(0).values
    n = len(r0)
    dates = E.index
    eq = start
    debt = 0.0
    assets = start
    rows = []
    for i in range(n):
        if i > 0:
            assets *= (1 + r0[i])
            debt *= (1 + rate / 252)
            eq = assets - debt
        rebal = (i == 0) or (freq == 'D') or \
                (freq == 'Q' and dates[i].quarter != dates[i - 1].quarter) or \
                (freq == 'M' and dates[i].month != dates[i - 1].month)
        if rebal:
            debt = (Lv - 1) * eq
            assets = Lv * eq
        rows.append(dict(date=dates[i], equity=eq, debt=debt, assets=assets,
                         collateral=(eq / debt if debt > 1e-9 else np.nan)))
    D = pd.DataFrame(rows).set_index('date')
    D['ret'] = D.equity.pct_change().fillna(0)
    return D


def lev_metrics(D, start=20000.0):
    cagr = (D.equity.iloc[-1] / start) ** (252 / len(D)) - 1
    dd = (D.equity / D.equity.cummax() - 1).min()
    return dict(cagr=cagr, maxdd=dd, min_collateral=D.collateral.min(),
                final=D.equity.iloc[-1])


def main():
    summary = []
    # 核心版
    e, E, T, r = run_cfg(CORE)
    E.index.name = 'date'
    E.to_csv(os.path.join(OUT, '核心_权益.csv'), encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    T.to_csv(os.path.join(OUT, '核心_逐笔.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f', date_format='%Y-%m-%d')
    yc = yearly(E)
    yc.to_csv(os.path.join(OUT, '核心_年度.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f')
    summary.append(dict(版本='核心版', 持有=20, 单票='45%×波动', 杠杆=1.0,
                        年化=r['全期_cagr'], 最大回撤=r['全期_maxdd'],
                        交易=r['全期_trades'], 胜率=r['全期_win'], 盈亏比=r['全期_pf'],
                        平均仓位=r['全期_平均仓位'], 前5笔占比=r['全期_前5笔占比'],
                        最差单笔=r['全期_最大单笔亏损'], y2018=yc.loc[yc.年份 == 2018, '收益'].iloc[0]))

    # 进攻版
    e2, E2, T2, r2 = run_cfg(ATTACK)
    E2.index.name = 'date'
    T2.to_csv(os.path.join(OUT, '进攻_逐笔_无杠杆.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f', date_format='%Y-%m-%d')
    D = leverage_curve(E2, 1.4, 0.065, 'Q')
    D.to_csv(os.path.join(OUT, '进攻_权益_1.4倍.csv'), encoding='utf-8-sig',
             float_format='%.4f', date_format='%Y-%m-%d')
    ya = yearly(D)
    ya.to_csv(os.path.join(OUT, '进攻_年度_1.4倍.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f')
    lm = lev_metrics(D)
    summary.append(dict(版本='进攻版', 持有=10, 单票='50%', 杠杆=1.4,
                        年化=lm['cagr'], 最大回撤=lm['maxdd'],
                        交易=r2['全期_trades'], 胜率=r2['全期_win'], 盈亏比=r2['全期_pf'],
                        平均仓位=r2['全期_平均仓位'], 前5笔占比=r2['全期_前5笔占比'],
                        最差单笔=r2['全期_最大单笔亏损'],
                        y2018=ya.loc[ya.年份 == 2018, '收益'].iloc[0]))

    # 杠杆表
    lev_rows = []
    for Lv in [1.2, 1.3, 1.4, 1.45, 1.5, 1.6, 1.7]:
        d = leverage_curve(E2, Lv, 0.065, 'Q')
        m = lev_metrics(d)
        lev_rows.append(dict(杠杆=Lv, 年化=m['cagr'], 最大回撤=m['maxdd'],
                             最低抵押率=m['min_collateral'], 期末权益=m['final']))
    lev = pd.DataFrame(lev_rows)
    lev.to_csv(os.path.join(OUT, '融资杠杆表.csv'), index=False, encoding='utf-8-sig',
               float_format='%.6f')

    # 成本敏感性
    cost_rows = []
    for cr in [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]:
        c = dict(ATTACK); c['cost_rate'] = cr
        _, Ex, Tx, rx = run_cfg(c)
        dx = leverage_curve(Ex, 1.4, 0.065, 'Q')
        mx = lev_metrics(dx)
        cost_rows.append(dict(单边成本=cr, 无杠杆年化=rx['全期_cagr'],
                              无杠杆回撤=rx['全期_maxdd'],
                              杠杆年化=mx['cagr'], 杠杆回撤=mx['maxdd'],
                              最低抵押率=mx['min_collateral']))
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(os.path.join(OUT, '成本敏感性.csv'), index=False, encoding='utf-8-sig',
                float_format='%.6f')

    # 持有期参数邻域
    hold_rows = []
    for h in [5, 7, 10, 12, 15, 20]:
        c = dict(ATTACK); c['hold'] = h
        _, Eh, Th, rh = run_cfg(c)
        hold_rows.append(dict(持有天数=h, 年化=rh['全期_cagr'], 最大回撤=rh['全期_maxdd'],
                              交易=rh['全期_trades'], 胜率=rh['全期_win'], 盈亏比=rh['全期_pf']))
    hold = pd.DataFrame(hold_rows)
    hold.to_csv(os.path.join(OUT, '持有期邻域.csv'), index=False, encoding='utf-8-sig',
                float_format='%.6f')

    # 持有期抖动
    jit_rows = []
    for h in [10, 20]:
        for j in [1, 2]:
            vals = []
            for seed in range(1, 11):
                c = dict(ATTACK); c['hold'] = h; c['hold_jitter'] = j; c['hold_seed'] = seed
                _, Ej, Tj, rj = run_cfg(c)
                vals.append((rj['全期_cagr'], rj['全期_maxdd']))
            a = np.array(vals)
            jit_rows.append(dict(持有天数=h, 抖动天数=j, 年化均值=a[:, 0].mean(),
                                 年化最差=a[:, 0].min(), 年化最好=a[:, 0].max(),
                                 回撤均值=a[:, 1].mean(), 回撤最差=a[:, 1].min()))
    jit = pd.DataFrame(jit_rows)
    jit.to_csv(os.path.join(OUT, '持有期抖动.csv'), index=False, encoding='utf-8-sig',
               float_format='%.6f')

    # ST 口径
    st_rows = []
    for st, label in [('x', '基准STX'), ('one', '1次5%触板'), ('two', '2次5%触板'), ('y', '更严STY')]:
        c = dict(ATTACK); c['st_mode'] = st
        _, Es, Ts, rs = run_cfg(c)
        ds = leverage_curve(Es, 1.4, 0.065, 'Q')
        ms = lev_metrics(ds)
        st_rows.append(dict(ST口径=label, 无杠杆年化=rs['全期_cagr'],
                            无杠杆回撤=rs['全期_maxdd'],
                            杠杆年化=ms['cagr'], 杠杆回撤=ms['maxdd'],
                            交易=rs['全期_trades']))
    st = pd.DataFrame(st_rows)
    st.to_csv(os.path.join(OUT, 'ST口径.csv'), index=False, encoding='utf-8-sig',
              float_format='%.6f')

    # 起点敏感性
    start_rows = []
    for st0 in ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']:
        x = E2.loc[st0:]
        c = (x.equity.iloc[-1] / x.equity.iloc[0]) ** (252 / len(x)) - 1
        d = (x.equity / x.equity.cummax() - 1).min()
        start_rows.append(dict(起点=st0, 年化=c, 最大回撤=d))
    start = pd.DataFrame(start_rows)
    start.to_csv(os.path.join(OUT, '起点敏感性.csv'), index=False, encoding='utf-8-sig',
                 float_format='%.6f')

    # 汇总
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(OUT, '汇总.csv'), index=False, encoding='utf-8-sig',
             float_format='%.6f')
    json.dump(summary, open(os.path.join(OUT, '汇总.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=float)

    # ---------------- 图 ----------------
    # 1) 净值与回撤
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0, 0]
    ax.plot(E.index, E.equity / 20000, label='核心版（无杠杆）', color='#1f4e79', lw=1.4)
    ax.plot(D.index, D.equity / 20000, label='进攻版（1.4倍）', color='#c0504d', lw=1.4)
    ax.set_yscale('log')
    ax.set_title('① 净值（对数轴，初始=1）')
    ax.legend()
    ax.grid(alpha=.25)
    ax = axes[0, 1]
    ax.fill_between(E.index, E.equity / E.equity.cummax() - 1, 0, color='#1f4e79', alpha=.25, label='核心回撤')
    ax.fill_between(D.index, D.equity / D.equity.cummax() - 1, 0, color='#c0504d', alpha=.25, label='进攻回撤')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('② 回撤')
    ax.legend()
    ax.grid(alpha=.25)
    ax = axes[1, 0]
    ax.bar(yc['年份'].astype(str), yc['收益'], color=['#c0504d' if v < 0 else '#4f81bd' for v in yc['收益']])
    ax.axhline(0, color='black', lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('③ 核心版年度收益')
    ax.grid(alpha=.25, axis='y')
    ax = axes[1, 1]
    ax.bar(ya['年份'].astype(str), ya['收益'], color=['#c0504d' if v < 0 else '#c0504d' for v in ya['收益']])
    ax.axhline(0, color='black', lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('④ 进攻版（1.4倍）年度收益')
    ax.grid(alpha=.25, axis='y')
    fig.suptitle('结构交易·主板40%系统：核心版 vs 进攻版（单边0.25%成本）', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, '净值与回撤.png'), dpi=170)
    plt.close()

    # 2) 成本敏感性
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost['单边成本'] * 100, cost['杠杆年化'] * 100, marker='o', color='#c0504d', label='1.4倍融资')
    ax.plot(cost['单边成本'] * 100, cost['无杠杆年化'] * 100, marker='o', color='#1f4e79', label='无杠杆')
    ax.axhline(40, color='gray', ls='--', lw=1, label='40%')
    ax.set_xlabel('单边成本（%）')
    ax.set_ylabel('年化（%）')
    ax.set_title('成本敏感性：40%只在单边≤0.25%时成立')
    ax.legend()
    ax.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, '成本敏感性.png'), dpi=170)
    plt.close()

    # 3) 四屏结构示意
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    boxes = [
        (0.02, 0.70, 0.22, 0.22, '① 框架\n季/月\n市场宽度>50%\n行业宽度≥60%\n行业20日>5%', '#dbe5f1'),
        (0.27, 0.70, 0.22, 0.22, '② 结构\n周\nMA20周>MA60周\nMA60周向上\n价格在结构线上方', '#e2efda'),
        (0.52, 0.70, 0.22, 0.22, '③ 趋势\n日\n收盘>MA60\n年龄10-30\nATR≤4.4%\n20日无涨停', '#fff2cc'),
        (0.77, 0.70, 0.22, 0.22, '④ 风险\n日\n离结构线≤15%\n止损≤20%\nheat≤10%', '#fce4d6'),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#666', lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=10)
    ax.annotate('', xy=(0.27, 0.81), xytext=(0.24, 0.81), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(0.52, 0.81), xytext=(0.49, 0.81), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(0.77, 0.81), xytext=(0.74, 0.81), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.text(0.5, 0.55, '四屏全过 → 次日开盘买；任一屏不过 → 空仓/放弃', ha='center', fontsize=13, weight='bold')
    ax.text(0.5, 0.40, '结构线：日线 MA60 + 最近有效日线低点；周线 MA20 + 最近周线低点', ha='center', fontsize=10)
    ax.text(0.5, 0.30, '驱动段：更高的高点 + 更高的低点；调整段：回调不破结构线；反转：跌破结构线并创新低', ha='center', fontsize=10)
    ax.text(0.5, 0.18, '核心：20日持有，2×45%；进攻：10日持有，2×50%+1.4倍融资', ha='center', fontsize=11, color='#c0504d')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, '结构四屏.png'), dpi=170)
    plt.close()

    print('核心版:', {k: round(v, 4) for k, v in r.items() if isinstance(v, (int, float))})
    print('进攻版:', {k: round(v, 4) for k, v in r2.items() if isinstance(v, (int, float))})
    print('杠杆:', lev.round(4).to_string(index=False))
    print('\n成本:\n', cost.round(4).to_string(index=False))
    print('\n持有:\n', hold.round(4).to_string(index=False))
    print('\n抖动:\n', jit.round(4).to_string(index=False))
    print('\nST:\n', st.round(4).to_string(index=False))
    print('\n起点:\n', start.round(4).to_string(index=False))
    print('\n汇总:\n', S.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
