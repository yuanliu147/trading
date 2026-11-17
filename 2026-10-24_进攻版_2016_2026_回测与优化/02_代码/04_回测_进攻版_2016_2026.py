# -*- coding: utf-8 -*-
"""进攻版 2016-10-13→2026-10-23 主回测 + 稳健性电池。

流程：
  [门槛A] v2 面板在原窗口(2018-01-01→2026-10-13)必须复现 27.6852%/-24.5937%/222 笔
          （与 02_复现门槛_原口径.py 的原面板结果逐位一致）；
  [R1] 进攻版，新窗口 2017-03-01→2026-10-23（信号起点，预热 80+ 交易日）；
  [R2] 进攻版，原窗口起点 + 新尾部（隔离"多出 2017 年 + 尾部 8 天"两种效应）；
  [R3] 核心版（20 日、45%×波动）对照；
  [稳健性] 持有邻域{5,7,9,10,11,12,15,20}、持有±1抖动(10种子)、成本、ST口径、
           纯主板闸门、起点偏移、随机排序(5种子)。
输出：03_结果/*.csv + 04_图表/*.png + 控制台摘要。
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bt2016 as B  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '03_结果')
FIG = os.path.join(ROOT, '04_图表')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def save(df, name, float_format='%.6f'):
    df.to_csv(os.path.join(OUT, name), index=False, encoding='utf-8-sig',
              float_format=float_format)


def monthly(E):
    m = E.equity.resample('ME').last()
    first = E.equity.iloc[0]
    rows = []
    prev = first
    for dt, v in m.items():
        rows.append(dict(month=str(dt)[:7], ret=v / prev - 1, equity=v))
        prev = v
    return pd.DataFrame(rows)


def main():
    summary = {}

    # ---------- 门槛A：v2 面板 + 原窗口 ----------
    # 背景说明：v2 面板相对原面板的原始数据差异 = 同花顺 2026-10-22 同步时补录了
    # 4 个此前遗漏的现金分红除息事件（601198.SH 2017-06-29 / 002731.SZ 2017-07-06 /
    # 000016.SZ 2018-06-06 / 601059.SH 2023-08-18），已用 baostock 分红记录逐条核对
    # （除息日与缩放比例全部吻合，见 01_数据/核对_复权因子修订_股票分布.csv）。
    # 因此 v2 面板在原窗口的期望值是修正口径 29.5394% / -24.5937% / 223 笔
    # （原面板为 27.6852% / -24.5937% / 222 笔，由 02_复现门槛_原口径.py 复现）。
    print('[门槛A] v2 面板原窗口复现（修正口径）...', flush=True)
    e = B.fresh_engine()
    cfgA = dict(B.ATTACK, name='门槛A', date_start='2017-10-13')
    EA, TA, rA = B.run_one(e, cfgA, window=('2018-01-01', '2026-10-13'))
    TA.to_csv(os.path.join(OUT, '门槛A_v2面板原窗口_逐笔.csv'), index=False,
              encoding='utf-8-sig', float_format='%.6f')
    okA = (abs(rA['全期_cagr'] - 0.295394) < 1e-3
           and abs(rA['全期_maxdd'] - (-0.245937)) < 1e-3
           and rA['全期_trades'] == 223)
    print(f"  v2面板原窗口: cagr={rA['全期_cagr']:.6f} maxdd={rA['全期_maxdd']:.6f} "
          f"trades={rA['全期_trades']} -> {'PASS' if okA else 'FAIL'}")
    # 与原面板逐笔差异定位
    try:
        T0 = pd.read_csv(os.path.join(OUT, '复现_进攻版逐笔_原区间.csv'))
        k0 = list(zip(T0.code, T0.entry_date.astype(str)))
        k1 = list(zip(TA.code, TA.entry_date.astype(str)))
        only0 = [x for x in k0 if x not in k1][:5]
        only1 = [x for x in k1 if x not in k0][:5]
        print(f'  逐笔差异: 原有v2无 {only0} | v2有原无 {only1}')
    except Exception as ex:
        print('  逐笔差异定位失败:', ex)
    if not okA:
        print('  !! v2 面板与预期修正口径不一致，停止。')
        sys.exit(1)

    # ---------- R1 进攻版 新窗口 ----------
    print('[R1] 进攻版 2017-03-01→2026-10-23 ...', flush=True)
    E1, T1, r1 = B.run_one(e, B.ATTACK)
    E1.index.name = 'date'
    E1.to_csv(os.path.join(OUT, 'R1_进攻版_权益.csv'), encoding='utf-8-sig',
              float_format='%.4f', date_format='%Y-%m-%d')
    save(T1, 'R1_进攻版_逐笔.csv')
    save(pd.DataFrame([r1]), 'R1_进攻版_指标.csv')
    save(monthly(E1), 'R1_进攻版_月度.csv')

    # ---------- R2 原窗口起点 + 新尾部 ----------
    print('[R2] 进攻版 2018-01-01→2026-10-23 ...', flush=True)
    cfgR2 = dict(B.ATTACK, name='R2', date_start='2017-10-13')
    E2, T2, r2 = B.run_one(e, cfgR2, window=('2018-01-01', '2026-10-23'))
    E2.index.name = 'date'
    E2.to_csv(os.path.join(OUT, 'R2_进攻版_2018起_权益.csv'), encoding='utf-8-sig',
              float_format='%.4f', date_format='%Y-%m-%d')
    save(T2, 'R2_进攻版_2018起_逐笔.csv')
    save(pd.DataFrame([r2]), 'R2_进攻版_2018起_指标.csv')

    # ---------- R3 核心版 ----------
    print('[R3] 核心版 2017-03-01→2026-10-23 ...', flush=True)
    E3, T3, r3 = B.run_one(e, B.CORE)
    E3.index.name = 'date'
    E3.to_csv(os.path.join(OUT, 'R3_核心版_权益.csv'), encoding='utf-8-sig',
              float_format='%.4f', date_format='%Y-%m-%d')
    save(T3, 'R3_核心版_逐笔.csv')
    save(pd.DataFrame([r3]), 'R3_核心版_指标.csv')

    summary.update(门槛A_v2面板复现=dict(cagr=rA['全期_cagr'], maxdd=rA['全期_maxdd'],
                                    trades=rA['全期_trades'], ok=bool(okA)),
                   R1_进攻版_新窗口={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in r1.items()},
                   R2_进攻版_2018起新尾={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                   for k, v in r2.items()},
                   R3_核心版_新窗口={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in r3.items()})

    # ---------- 稳健性电池 ----------
    print('[稳健性] 持有邻域...', flush=True)
    rows = []
    for h in [5, 7, 9, 10, 11, 12, 15, 20]:
        cfg = dict(B.ATTACK, name=f'hold{h}', hold=h)
        _, _, r = B.run_one(e, cfg)
        rows.append(dict(持有天数=h, 年化=r['全期_cagr'], 最大回撤=r['全期_maxdd'],
                         交易=r['全期_trades'], 胜率=r['全期_win'], 盈亏比=r['全期_pf'],
                         负年份数=r['负年份数']))
    hold_nb = pd.DataFrame(rows)
    save(hold_nb, '稳健_持有邻域.csv')

    print('[稳健性] 持有±1抖动...', flush=True)
    rows = []
    for h in [10, 20]:
        vals = []
        for seed in range(1, 11):
            cfg = dict(B.ATTACK, name=f'jit{h}_{seed}', hold=h, hold_jitter=1, hold_seed=seed)
            _, _, r = B.run_one(e, cfg)
            vals.append((r['全期_cagr'], r['全期_maxdd']))
        a = np.array(vals)
        rows.append(dict(持有天数=h, 年化均值=a[:, 0].mean(), 年化最差=a[:, 0].min(),
                         年化最好=a[:, 0].max(), 回撤均值=a[:, 1].mean(), 回撤最差=a[:, 1].min()))
    jit = pd.DataFrame(rows)
    save(jit, '稳健_持有抖动.csv')

    print('[稳健性] 成本...', flush=True)
    rows = []
    for cr in [0.0015, 0.0025, 0.0035, 0.0040]:
        cfg = dict(B.ATTACK, name=f'cost{cr}', cost_rate=cr)
        _, _, r = B.run_one(e, cfg)
        rows.append(dict(单边成本=cr, 年化=r['全期_cagr'], 最大回撤=r['全期_maxdd'],
                         交易=r['全期_trades']))
    cost = pd.DataFrame(rows)
    save(cost, '稳健_成本.csv')

    print('[稳健性] ST 口径...', flush=True)
    rows = []
    for st in ['x', 'one', 'y']:
        cfg = dict(B.ATTACK, name=f'st{st}', st_mode=st)
        _, _, r = B.run_one(e, cfg)
        rows.append(dict(ST口径=st, 年化=r['全期_cagr'], 最大回撤=r['全期_maxdd'],
                         交易=r['全期_trades']))
    stt = pd.DataFrame(rows)
    save(stt, '稳健_ST口径.csv')

    print('[稳健性] 纯主板闸门...', flush=True)
    em = B.fresh_engine(gate='main')
    _, _, rm = B.run_one(em, B.ATTACK)
    save(pd.DataFrame([rm]), '稳健_纯主板闸门.csv')
    del em

    print('[稳健性] 起点偏移...', flush=True)
    rows = []
    for s0 in ['2017-03-01', '2017-06-01', '2018-01-01', '2019-01-01', '2020-01-01']:
        x = E1.loc[s0:]
        c = (x.equity.iloc[-1] / x.equity.iloc[0]) ** (252 / len(x)) - 1
        d = (x.equity / x.equity.cummax() - 1).min()
        rows.append(dict(起点=s0, 年化=c, 最大回撤=d))
    starts = pd.DataFrame(rows)
    save(starts, '稳健_起点.csv')

    print('[稳健性] 随机排序...', flush=True)
    rows = []
    for seed in range(1, 6):
        cfg = dict(B.ATTACK, name=f'rnd{seed}', rank='random', rank_seed=seed)
        _, _, r = B.run_one(e, cfg)
        rows.append(dict(种子=seed, 年化=r['全期_cagr'], 最大回撤=r['全期_maxdd'],
                         交易=r['全期_trades']))
    rnd = pd.DataFrame(rows)
    save(rnd, '稳健_随机排序.csv')

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0, 0]
    ax.plot(E1.index, E1.equity / 20000, label='进攻版 2017-03→2026-10', color='#c0504d', lw=1.3)
    ax.plot(E3.index, E3.equity / 20000, label='核心版（对照）', color='#1f4e79', lw=1.2)
    ax.set_yscale('log')
    ax.set_title('净值（对数轴，初始=1，2万本金）')
    ax.legend(); ax.grid(alpha=.25)
    ax = axes[0, 1]
    ax.fill_between(E1.index, E1.equity / E1.equity.cummax() - 1, 0, color='#c0504d', alpha=.3, label='进攻回撤')
    ax.fill_between(E3.index, E3.equity / E3.equity.cummax() - 1, 0, color='#1f4e79', alpha=.3, label='核心回撤')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('回撤'); ax.legend(); ax.grid(alpha=.25)
    ax = axes[1, 0]
    yrs = [y for y in range(2017, 2027) if f'{y}年' in r1]
    v1 = [r1[f'{y}年'] for y in yrs]
    v3 = [r3.get(f'{y}年', np.nan) for y in yrs]
    x = np.arange(len(yrs))
    ax.bar(x - 0.2, v1, 0.4, color='#c0504d', label='进攻版')
    ax.bar(x + 0.2, v3, 0.4, color='#1f4e79', label='核心版')
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in yrs])
    ax.axhline(0, color='black', lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('年度收益'); ax.legend(); ax.grid(alpha=.25, axis='y')
    ax = axes[1, 1]
    ax.plot(hold_nb['持有天数'], hold_nb['年化'] * 100, marker='o', color='#c0504d')
    for _, row in hold_nb.iterrows():
        ax.annotate(f"{row['年化']*100:.1f}", (row['持有天数'], row['年化'] * 100),
                    textcoords='offset points', xytext=(0, 6), fontsize=9)
    ax.set_xlabel('持有天数'); ax.set_ylabel('年化（%）')
    ax.set_title('持有天数邻域（执行日敏感性）')
    ax.grid(alpha=.25)
    fig.suptitle('进攻版（主板·成交额升序·10日·2×50%）2017-03→2026-10-23', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, '主图_净值回撤年度.png'), dpi=170)
    plt.close()

    with open(os.path.join(OUT, 'R1_摘要.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print('\n==== 摘要 ====')
    for k in ['全期_cagr', '全期_maxdd', '全期_trades', '全期_win', '全期_pf',
              '负年份数', '最差年', '全期_前5笔占比', '全期_平均仓位']:
        print(f"R1 {k}: {r1.get(k)}")
    print('R1 年度:', {y: round(r1[f'{y}年'], 4) for y in range(2017, 2027) if f'{y}年' in r1})
    print('\n持有邻域:\n', hold_nb.round(4).to_string(index=False))
    print('\n抖动:\n', jit.round(4).to_string(index=False))
    print('\n成本:\n', cost.round(4).to_string(index=False))
    print('\nST:\n', stt.round(4).to_string(index=False))
    print('\n纯主板闸门: cagr=%.4f dd=%.4f trades=%d' % (rm['全期_cagr'], rm['全期_maxdd'], rm['全期_trades']))
    print('\n随机排序均值年化: %.4f' % rnd['年化'].mean())
    print('完成')


if __name__ == '__main__':
    main()
