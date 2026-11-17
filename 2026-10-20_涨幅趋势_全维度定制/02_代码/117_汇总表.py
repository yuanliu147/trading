# -*- coding: utf-8 -*-
"""汇总所有回撤优化尝试，生成中文总表与最终候选表。"""
import os, numpy as np, pandas as pd
from scipy.stats import norm
from importlib.util import spec_from_file_location, module_from_spec

BASE = '/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
R = os.path.join(BASE, '03_结果')
spec = spec_from_file_location('eng', os.path.join(BASE, '02_代码/108_稳健引擎.py'))
mod = module_from_spec(spec); spec.loader.exec_module(mod)
E = mod.Engine()

files = [
 ('预注册20变体', '回撤优化_20变体.csv'),
 ('探索组合', '回撤优化_探索组合.csv'),
 ('第二轮', '回撤优化_第二轮.csv'),
 ('市场退出', '回撤优化_市场退出.csv'),
 ('上限平台', '回撤优化_上限平台.csv'),
]
frames = []
for batch, f in files:
    d = pd.read_csv(os.path.join(R, f)); d['批次'] = batch; frames.append(d)
allr = pd.concat(frames, ignore_index=True, sort=False)
allr = allr.rename(columns={
    '方案': '方案', '期': '期', 'cagr': '年化收益', 'maxdd': '最大回撤', 'maxdd_low': '最低价回撤',
    'sharpe': '夏普比率', 'calmar': '卡玛比率', 'trades': '交易笔数', 'win': '胜率', 'pf': '盈亏比',
    'avg_hold': '平均持有天数', 'avg_pos': '平均持仓数', 'avg_weight': '平均单笔仓位',
    'avg_exposure': '平均总仓位', 'worst_trade': '最差单笔账户亏损', 'top5_share': '前五笔利润占比',
    'final': '期末资金', 'max_weight': '最大单笔仓位', 'total': '累计收益',
})
keep = ['批次', '方案', '期', '年化收益', '最大回撤', '最低价回撤', '夏普比率', '卡玛比率',
        '交易笔数', '胜率', '盈亏比', '最差单笔账户亏损', '前五笔利润占比', '平均总仓位', '期末资金']
allr = allr[[c for c in keep if c in allr.columns]].copy()
# 前五笔占比只在全期有意义；总收益太小时比值会超过100%，置空避免误读
allr.loc[allr['期'] != '全期', '前五笔利润占比'] = np.nan
allr.loc[allr['前五笔利润占比'] > 1.0, '前五笔利润占比'] = np.nan
allr['是否通过硬约束'] = ((allr['期'] == '全期') & (allr['年化收益'] >= .20) &
                        (allr['最大回撤'] >= -.25) & (allr['最差单笔账户亏损'] >= -.06) &
                        (allr['前五笔利润占比'] < .60))
allr.to_csv(os.path.join(R, '回撤优化_总表.csv'), index=False, encoding='utf-8-sig')

# 最终候选
B = dict(slots=2, sizing='fixed', risk=0, maxpos=0.5, heat_cap=0.10, gate_thr=0.50,
         rank='amt', hold=20, start_cash=20000, cost_rate=0.0015)
C = {
 '旧25.8%': {},
 '回撤优先_42.5_波动20': dict(maxpos=.425, vol_target=.20),
 '收益优先_45_波动20': dict(maxpos=.45, vol_target=.20),
 '均衡_45_波动20_分批20': dict(maxpos=.45, vol_target=.20, partial_trigger=.20),
 '高收益_45_波动20_暂停': dict(maxpos=.45, vol_target=.20, loss_pause_n=2, loss_pause_days=5),
 '严格3%风险_止损20': dict(sizing='risk', risk=.03, risk_basis='stop', stop_pct=.20, heat_cap=.20),
}
def run(name, start='2018-01-01', end='2026-10-13', **over):
    a = B.copy(); a.update(C[name]); a.update(over)
    return E.run(start, end, **a)
var_sr = .000884; N = 119; G = .5772156649015329
sr0 = np.sqrt(var_sr) * ((1 - G) * norm.ppf(1 - 1 / N) + G * norm.ppf(1 - 1 / (N * np.e)))
rows = []
for name in C:
    e, t, m = run(name)
    r = e.ret.values[1:]; sr = r.mean() / r.std(); T = len(r)
    sk = pd.Series(r).skew(); ku = pd.Series(r).kurtosis() + 3
    dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    tr = run(name, '2018-01-01', '2021-12-31')[2]
    te = run(name, '2022-01-01', '2026-10-13')[2]
    c35 = run(name, cost_rate=.0035)[2]
    g = t.groupby(['code', 'entry_date']).apply(lambda x: (x.weight * x.ret).sum(), include_groups=False)
    skip = {(c, pd.Timestamp(d)) for c, d in g.nlargest(5).index}
    ex = run(name, skip_entries=skip)[2]
    s = e.equity
    blocks = []
    for y0 in [2018, 2021, 2024]:
        x = s[(s.index.year >= y0) & (s.index.year <= y0 + 2)]
        cg = (x.iloc[-1] / x.iloc[0]) ** (252 / len(x)) - 1; dd = (x / x.cummax() - 1).min()
        blocks.append(f'{y0}-{y0+2} {cg*100:.1f}%/{dd*100:.1f}%')
    rows.append({
        '方案': name, '全期年化': m['cagr'], '最大回撤': m['maxdd'], '最低价回撤': m['maxdd_low'],
        '夏普比率': m['sharpe'], '卡玛比率': m['calmar'], '交易笔数': m['trades'], '胜率': m['win'],
        '盈亏比': m['pf'], '最差单笔账户亏损': m['worst_trade'], '前五笔利润占比': m['top5_share'],
        '平均总仓位': m['avg_exposure'], '期末资金': m['final'],
        '训练年化': tr['cagr'], '训练回撤': tr['maxdd'], '检验年化': te['cagr'], '检验回撤': te['maxdd'],
        '成本0.35%年化': c35['cagr'], '成本0.35%回撤': c35['maxdd'],
        '去前五笔年化': ex['cagr'], '去前五笔回撤': ex['maxdd'], '多重检验调整夏普': dsr,
        '三年块': '；'.join(blocks),
    })
out = pd.DataFrame(rows)
out.to_csv(os.path.join(R, '回撤优化_候选总表.csv'), index=False, encoding='utf-8-sig')
# 主推荐的中文表头副本，便于直接看
e2, t2, _ = run('收益优先_45_波动20')
e2 = e2.rename_axis('日期')
e2.rename(columns={'equity': '权益', 'cash': '现金', 'npos': '持仓数',
                   'equity_low': '最低价权益', 'ret': '日收益', 'exposure': '总仓位'}).to_csv(
    os.path.join(R, '回撤优化_权益_收益优先_中文.csv'), encoding='utf-8-sig')
t2.rename(columns={'code': '代码', 'entry_date': '买入日', 'exit_date': '卖出日', 'entry': '买入价',
                   'exit': '卖出价', 'ret': '收益率', 'reason': '原因', 'hold': '持有天数',
                   'shares': '股数', 'pnl': '盈亏', 'weight': '仓位'}).to_csv(
    os.path.join(R, '回撤优化_逐笔_收益优先_中文.csv'), index=False, encoding='utf-8-sig')
print(out.round(4).to_string(index=False))

# 25.8审计摘要
diag = pd.read_csv(os.path.join(R, '二十五点八_诊断摘要.csv'), index_col=0).iloc[:, 0].to_dict()
audit = pd.DataFrame([
 ('全期年化/回撤', f"{diag['cagr']*100:.1f}% / {diag['maxdd']*100:.1f}%", '样本内最高值，不是预期'),
 ('前五笔利润占比', f"{diag['top5_contrib']/np.log(1+diag['total'])*100:.0f}%", '去掉前五笔，全期年化约10.4%'),
 ('最大连续亏损', f"{int(diag['max_consec_loss'])}笔", '连续亏损会放大回撤；暂停规则需单独测试'),
 ('最差单笔账户亏损', '-10.4%', '固定50%仓位 × -20%灾难止损，违反3%单笔风险规则'),
 ('44组DSR', '0.415', '按44次尝试修正后仍不显著'),
 ('119次DSR', '0.236', '按全部尝试修正后更不显著'),
 ('持有期', '15日10.9% / 20日25.8% / 25日-2.2%', '20日是尖峰，路径依赖强'),
 ('市场宽度', '45%10.7% / 50%25.8% / 55%19.7% / 60%20.6%', '50%也是尖峰'),
 ('成本', '0.15%25.8% / 0.25%23.6% / 0.35%20.9% / 0.50%18.2%', '成本每+0.1pp，年化约-2pp'),
 ('幸存者偏差', '含102只退市股25.8%/-36.4%；不含28.2%/-30.7%', '补退市股后收益下降、回撤加深，方向符合预期'),
 ('严格3%风险', '8.4%/-14.9%，最差-4.5%', '这是2万+3%规则的真实上限，不是25.8%'),
], columns=['检查项', '结果', '含义'])
audit.to_csv(os.path.join(R, '二十五点八_审计摘要.csv'), index=False, encoding='utf-8-sig')
print('\n', audit.to_string(index=False))
