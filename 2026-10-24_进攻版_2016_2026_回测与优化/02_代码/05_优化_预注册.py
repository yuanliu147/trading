# -*- coding: utf-8 -*-
"""针对性优化（预注册，先定变体与选择标准，后看结果）。

背景诊断（来自既有研究与本次回测）：
  D1 10 日到期是执行日尖峰：持有 9/11/12 天年化大幅低于 10 天，±1 天抖动均值 ~16%；
  D2 2018 年深度亏损（-22%）；既有研究证明：收紧闸门/回撤节流/破位清仓都会砍掉右尾，不采用；
  D3 右尾依赖（前 5 笔占利润 ~40%），结构性，用指标披露而不是调参解决。

预注册变体（全部只在"新窗口 2017-03-01→2026-10-23"上评估）：
  V0 attack10   基线：进攻版（10 日、2×50%、成交额升序、0.25%）
  V1 core20     核心版：20 日、2×45%×波动目标
  V2 blend1020  各半资金：attack10 + core20 两个子账本（各 1 万）——退出时间分散
  V3 stagger911 各半资金：hold=9 + hold=11 两个子账本——到期日永不在第 10 天
  V4 extend10   attack10 + 盈利≥10% 到期顺延 10 日（封顶 20 日）
  V5 trail8     attack10，移动止盈 25% 后改为最高×0.92
  V6 partial20  attack10，+20% 卖一半（原 +25%）
  V7 attack10mb attack10，纯主板宽度闸门（对照）
  V8 mcaprank   attack10，市值升序排序（v2 面板市值覆盖已补全，可检验）

预注册评估（每个变体同一口径）：
  全期 / 训练(2017-03→2021-12) / 检验(2022→2026-10) 年化与回撤；
  持有±1 抖动 10 种子均值年化（子账本变体对两个子账本同种子抖动后合并）；
  单边 0.35% 成本压力；
  最差年、交易数、前5笔利润占比。

预注册选择规则：
  可行 = 交易≥120 且 全期回撤≥-30% 且 最差年≥-25%；
  在可行变体中选  min(全期, 检验, 抖动均值, 成本0.35%)  年化最大者（maximin）。
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
os.makedirs(OUT, exist_ok=True)


def sub_run(e, cfg, window=None):
    E, T, r = B.run_one(e, cfg, window=window)
    return E, T, r


def blend(e, cfgA, cfgB, half=10000.0):
    """两个子账本各 half 元，合并权益与逐笔。"""
    a = dict(cfgA, start_cash=half)
    b = dict(cfgB, start_cash=half)
    EA, TA, _ = B.run_one(e, a)
    EB, TB, _ = B.run_one(e, b)
    idx = EA.index.union(EB.index)
    eq = EA.equity.reindex(idx).ffill() + EB.equity.reindex(idx).ffill()
    eql = EA.equity_low.reindex(idx).ffill() + EB.equity_low.reindex(idx).ffill()
    E = pd.DataFrame({'equity': eq, 'equity_low': eql})
    E['ret'] = E.equity.pct_change().fillna(0)
    T = pd.concat([TA.assign(子账本='A'), TB.assign(子账本='B')], ignore_index=True)
    return E, T


def met(E, T, start_cash=20000.0):
    total = E.equity.iloc[-1] / start_cash - 1
    cagr = (1 + total) ** (252 / len(E)) - 1
    dd = (E.equity / E.equity.cummax() - 1).min()
    out = dict(cagr=cagr, maxdd=dd, trades=len(T),
               win=(T.net_ret > 0).mean() if len(T) else np.nan,
               pf=(T.pnl[T.pnl > 0].sum() / max(1e-9, -T.pnl[T.pnl < 0].sum())) if len(T) else np.nan)
    # 年度
    prev = E.equity.iloc[0]
    for y in range(2017, 2027):
        x = E.equity.loc[f'{y}-01-01':f'{y}-12-31']
        if len(x) == 0:
            continue
        out[f'{y}年'] = x.iloc[-1] / prev - 1
        prev = x.iloc[-1]
    yrs = [y for y in range(2017, 2027) if f'{y}年' in out]
    out['最差年'] = min(out[f'{y}年'] for y in yrs)
    out['负年份数'] = sum(1 for y in yrs if out[f'{y}年'] < 0)
    out['前5笔占比'] = (T.nlargest(5, 'pnl').pnl.sum() / T.pnl.sum()
                      if len(T) and abs(T.pnl.sum()) > 1e-9 else np.nan)
    return out


def seg(E, a, b):
    x = E.loc[a:b]
    if len(x) < 2:
        return np.nan, np.nan
    c = (x.equity.iloc[-1] / x.equity.iloc[0]) ** (252 / len(x)) - 1
    d = (x.equity / x.equity.cummax() - 1).min()
    return c, d


def eval_variant(name, runner, e):
    """runner(e, **kw) -> (E, T)；成本/抖动通过 cfg 覆盖复跑。"""
    E, T = runner(e)
    m = met(E, T)
    c_tr, d_tr = seg(E, '2017-03-01', '2021-12-31')
    c_te, d_te = seg(E, '2022-01-01', '2026-10-23')
    row = dict(变体=name, 全期年化=m['cagr'], 全期回撤=m['maxdd'], 交易=m['trades'],
               胜率=m['win'], 盈亏比=m['pf'], 最差年=m['最差年'], 负年份数=m['负年份数'],
               前五笔占比=m['前5笔占比'], 训练年化=c_tr, 训练回撤=d_tr,
               检验年化=c_te, 检验回撤=d_te)
    for y in range(2017, 2027):
        if f'{y}年' in m:
            row[f'{y}年'] = m[f'{y}年']

    # 成本 0.35%（通过全局覆盖注入 cfg）
    global _COST_OVERRIDE
    _COST_OVERRIDE = 0.0035
    Ec, Tc = runner(e)
    _COST_OVERRIDE = None
    mc = met(Ec, Tc)
    row['成本035年化'] = mc['cagr']
    row['成本035回撤'] = mc['maxdd']

    # 抖动 ±1（10 种子）
    vals = []
    for seed in range(1, 11):
        Ej, Tj = runner(e, hold_jitter=1, hold_seed=seed)
        mj = met(Ej, Tj)
        vals.append(mj['cagr'])
    row['抖动均值年化'] = float(np.mean(vals))
    row['抖动最差年化'] = float(np.min(vals))
    return row, E, T


_COST_OVERRIDE = None


def make_runner(kind, cfgA, cfgB=None):
    if kind == 'single':
        def runner(e, **kw):
            cfg = dict(cfgA)
            if _COST_OVERRIDE:
                cfg['cost_rate'] = _COST_OVERRIDE
            cfg.update(kw)
            E, T, _ = B.run_one(e, cfg)
            return E, T
    else:
        def runner(e, **kw):
            a = dict(cfgA)
            b = dict(cfgB)
            if _COST_OVERRIDE:
                a['cost_rate'] = _COST_OVERRIDE
                b['cost_rate'] = _COST_OVERRIDE
            a.update(kw); b.update(kw)
            return blend(e, a, b)
    return runner


def main():
    e = B.fresh_engine()
    rows = []
    curves = {}

    A10 = dict(B.ATTACK)
    C20 = dict(B.CORE)
    A9 = dict(B.ATTACK, hold=9)
    A11 = dict(B.ATTACK, hold=11)
    A10mb = dict(B.ATTACK)  # 主板闸门需独立引擎
    A10ext = dict(B.ATTACK, extend_if_gain=0.10, extend_days=10, extend_max=20)
    A10t8 = dict(B.ATTACK, trail_pct=0.08)
    A10p20 = dict(B.ATTACK, partial_trigger=0.20)
    A10mc = dict(B.ATTACK, rank='mcap')

    for name, runner in [
        ('V0_attack10', make_runner('single', A10)),
        ('V1_core20', make_runner('single', C20)),
        ('V2_blend1020', make_runner('blend', A10, C20)),
        ('V3_stagger911', make_runner('blend', A9, A11)),
        ('V4_extend10', make_runner('single', A10ext)),
        ('V5_trail8', make_runner('single', A10t8)),
        ('V6_partial20', make_runner('single', A10p20)),
        ('V8_mcaprank', make_runner('single', A10mc)),
    ]:
        print(f'评估 {name} ...', flush=True)
        row, E, T = eval_variant(name, runner, e)
        rows.append(row)
        curves[name] = (E, T)
        print('  全期 %.4f / dd %.4f / trades %d / 抖动均值 %.4f' %
              (row['全期年化'], row['全期回撤'], row['交易'], row['抖动均值年化']), flush=True)
        T.to_csv(os.path.join(OUT, f'优化_{name}_逐笔.csv'), index=False,
                 encoding='utf-8-sig', float_format='%.6f')
        E.to_csv(os.path.join(OUT, f'优化_{name}_权益.csv'), encoding='utf-8-sig',
                 float_format='%.4f', date_format='%Y-%m-%d')

    # V7 纯主板闸门（独立引擎）
    print('评估 V7_attack10mb ...', flush=True)
    em = B.fresh_engine(gate='main')
    row7, E7, T7 = eval_variant('V7_attack10mb', make_runner('single', A10mb), em)
    rows.append(row7)
    T7.to_csv(os.path.join(OUT, '优化_V7_attack10mb_逐笔.csv'), index=False,
              encoding='utf-8-sig', float_format='%.6f')
    print('  全期 %.4f / dd %.4f' % (row7['全期年化'], row7['全期回撤']), flush=True)
    del em

    R = pd.DataFrame(rows)

    # 预注册选择
    R['可行'] = (R['交易'] >= 120) & (R['全期回撤'] >= -0.30) & (R['最差年'] >= -0.25)
    R['maximin'] = R[['全期年化', '检验年化', '抖动均值年化', '成本035年化']].min(axis=1)
    R = R.sort_values('maximin', ascending=False)
    save_path = os.path.join(OUT, '优化_预注册对比.csv')
    R.to_csv(save_path, index=False, encoding='utf-8-sig', float_format='%.6f')

    cols = ['变体', '全期年化', '全期回撤', '交易', '胜率', '盈亏比', '最差年', '负年份数',
            '训练年化', '检验年化', '抖动均值年化', '抖动最差年化', '成本035年化', '前五笔占比',
            '可行', 'maximin']
    print('\n', R[cols].round(4).to_string(index=False))
    print('\n预注册规则下的排序（maximin）：')
    print(R[['变体', 'maximin', '可行']].round(4).to_string(index=False))
    R.to_json(os.path.join(OUT, '优化_预注册对比.json'), orient='records',
              force_ascii=False, indent=2)
    print('完成')


if __name__ == '__main__':
    main()
