# -*- coding: utf-8 -*-
"""F08 独立抽查② + 各家族交付数字核验（只读）。
抽查②：从旧交付 涨停事件_含前瞻.parquet 重算 4板+/一字板 的 5 日收益
（T+1开盘买→T+5收盘卖，已剔除次日不可买/开盘跌停/跨日缺口），
按原报告分期：设计期 2016-10-14..2021-12-31，样本外 2024-01-01..2026-10-13。
对照原称：4板+ 设计期+15.5%/样本外-4.8%；一字板 +19.8%/-3.9%；全期 4板+ n=8587 mean=11.65%。
其余：跌停六条/流动性反转/龙虎榜前视审计/股价分层/万元账户 关键数字对交付CSV的复核。
"""
import os, json
import duckdb
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/admin/WorkSpace/trading'
rows = []

def check(family, item, claimed, found, ok):
    rows.append(dict(家族=family, 项目=item, 原称=claimed, 核验值=found, 一致=bool(ok)))

# ---------- 抽查② 连板/一字板 分期重算 ----------
EV = f'{ROOT}/2026-10-22_龙虎榜涨停_行业影响与策略/01_数据/涨停事件_含前瞻.parquet'
con = duckdb.connect()
con.execute(f"CREATE TEMP TABLE u AS SELECT * FROM read_parquet('{EV}')")
split_rows = []
for name, cond in [('4板+', "consec_bucket='4板+'"), ('一字板', 'lu_one=1')]:
    for pname, d0, d1 in [('全期', '2016-10-14', '2026-10-13'),
                          ('设计期', '2016-10-14', '2021-12-31'),
                          ('样本外', '2024-01-01', '2026-10-13')]:
        r = con.execute(f"""
          SELECT count(stock_fwd5) AS n, avg(stock_fwd5) AS mean5, median(stock_fwd5) AS med5
          FROM u WHERE {cond} AND date BETWEEN '{d0}' AND '{d1}'
        """).df().iloc[0]
        split_rows.append(dict(分组=name, 分期=pname, n=int(r.n), 五日均值=r.mean5, 五日中位=r.med5))
split = pd.DataFrame(split_rows)
split.to_csv(os.path.join(OUT, 'F08_连板一字板分期复算.csv'), index=False)
print(split.to_string(index=False))
g = {(r.分组, r.分期): r for r in split.itertuples()}
check('打板连板', '4板+全期5日(n/均值)', 'n=8587, +11.65%',
      f"n={g[('4板+','全期')].n}, {g[('4板+','全期')].五日均值*100:.2f}%",
      g[('4板+','全期')].n == 8587 and abs(g[('4板+','全期')].五日均值 - 0.116452) < 0.001)
check('打板连板', '4板+设计期5日', '+15.5%', f"{g[('4板+','设计期')].五日均值*100:.2f}%",
      abs(g[('4板+','设计期')].五日均值 - 0.155) < 0.005)
check('打板连板', '4板+样本外5日', '-4.8%', f"{g[('4板+','样本外')].五日均值*100:.2f}%",
      abs(g[('4板+','样本外')].五日均值 + 0.048) < 0.005)
check('打板连板', '一字板设计期5日', '+19.8%', f"{g[('一字板','设计期')].五日均值*100:.2f}%",
      abs(g[('一字板','设计期')].五日均值 - 0.198) < 0.005)
check('打板连板', '一字板样本外5日', '-3.9%', f"{g[('一字板','样本外')].五日均值*100:.2f}%",
      abs(g[('一字板','样本外')].五日均值 + 0.039) < 0.005)

# ---------- 跌停六条 ----------
D = f'{ROOT}/2026-10-22_跌停前特征_全A/03_结果'
s = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{D}/风险评分_六条清单.csv')
tot_n, tot_ld = int(s.样本数.sum()), int(s.跌停数.sum())
check('跌停六条', '主样本股票-日/跌停数', '956.9万 / 37,412', f'{tot_n:,} / {tot_ld:,}',
      tot_n == 9569164 and tot_ld == 37412)
p0, p3, p4, p5 = [float(s.loc[s.评分 == k, 'P次日'].iloc[0]) for k in (0, 3, 4, 5)]
check('跌停六条', '0/3/4/5条次日跌停', '0.069%/4.03%/9.19%/17.03%',
      f'{p0*100:.3f}%/{p3*100:.2f}%/{p4*100:.2f}%/{p5*100:.2f}%',
      abs(p0-0.00069) < 1e-5 and abs(p3-0.0403) < 5e-4 and abs(p4-0.0919) < 5e-4 and abs(p5-0.1703) < 5e-4)
ge3 = s[s.评分 >= 3]
share, cover = float(ge3.样本占比.sum()), float(ge3.事件覆盖率.sum())
check('跌停六条', '评分>=3占比/覆盖', '2.2% / 27.8%', f'{share*100:.2f}% / {cover*100:.1f}%',
      abs(share-0.022) < 0.001 and abs(cover-0.278) < 0.002)
w5 = float((s.样本数 * s.P五日内).sum() / tot_n)
check('跌停六条', '5日内跌停基准率', '1.718%', f'{w5*100:.3f}%', abs(w5-0.01718) < 0.0005)
t3 = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{D}/昨日跌停三态_次日风险.csv')
yz = t3[t3.昨日状态 == '一字跌停'].iloc[0]
sealed = t3[t3.昨日状态.isin(['一字跌停', '非一字封死'])]
p_sealed = sealed.跌停数.sum() / sealed.样本数.sum()
check('跌停六条', '一字跌停后再跌停', '28.97%', f'{yz["P(次日跌停)"]*100:.2f}%', abs(yz['P(次日跌停)']-0.2897) < 0.001)
check('跌停六条', '收盘跌停后再跌停', '12.34%', f'{p_sealed*100:.2f}%', abs(p_sealed-0.1234) < 0.001)

# ---------- 龙虎榜/涨停行业 ----------
L = f'{ROOT}/2026-10-22_龙虎榜涨停_行业影响与策略/03_结果'
ie = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{L}/行业预测_事件与非事件.csv')
for ev, h, cl, tcl in [('涨停', 5, -0.0037, -13.5), ('涨停', 10, -0.0068, -17.2), ('龙虎榜', 5, -0.0031, -13.4)]:
    r = ie[(ie.事件 == ev) & (ie.口径 == '剔除两者') & (ie.h == h)].iloc[0]
    check('龙虎榜行业', f'{ev}非事件股{h}日差/t', f'{cl*100:.2f}% (t={tcl})',
          f'{r["差"]*100:.3f}% (t={r.t:.1f})', abs(r['差']-cl) < 0.0003 and abs(r.t-tcl) < 0.3)
la = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{L}/前视审计_理想vs实盘.csv')
ideal, real = la.iloc[0], la.iloc[1]
check('龙虎榜行业', '理想版年化(前视)', '+57.5%', f'{ideal.年化*100:.1f}%', abs(ideal.年化-0.575) < 0.002)
check('龙虎榜行业', '实盘版年化/回撤', '-32.4% / -98.0%',
      f'{real.年化*100:.1f}% / {real.最大回撤*100:.1f}%',
      abs(real.年化+0.324) < 0.002 and abs(real.最大回撤+0.98) < 0.005)
check('龙虎榜行业', '跌停卖不出占比', '5.24%', f'{real.跌停卖不出占比*100:.2f}%', abs(real.跌停卖不出占比-0.0524) < 0.001)

# ---------- 流动性反转（主板PIT，直接核验交付物） ----------
Q = f'{ROOT}/2026-10-30_流动性供给_短期反转/结果'
g12 = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{Q}/事件层_12格.csv')
neg = int((g12.净均值 < 0).sum())
check('流动性反转', '12格净负个数', '12/12', f'{neg}/12', neg == 12)
main = g12[g12.主候选 == True].iloc[0]
check('流动性反转', 'R5H3毛/净每笔', '+0.192% / -0.263%',
      f'{main.毛均值*100:.3f}% / {main.净均值*100:.3f}%',
      abs(main.毛均值-0.00192) < 5e-5 and abs(main.净均值+0.00263) < 5e-5)
check('流动性反转', 'ADV匹配超额/q', '+0.176% / q=0',
      f'{main.C2差*100:.3f}% / q={main.C2差q值}', abs(main.C2差-0.00176) < 5e-5 and main.C2差q值 < 0.001)
era = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{Q}/事件层_分时代.csv')
e16 = era[(era.k == 5) & (era.h == 3) & (era.时期 == '2016-2020')].iloc[0]
check('流动性反转', '2016-2020净每笔', '-0.442%', f'{e16.输家净均值*100:.3f}%', abs(e16.输家净均值+0.00442) < 5e-5)
acc = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{Q}/账户_汇总.csv')
a0 = acc[(acc.种子 == 20261030) & (acc.组合 == '输家')].iloc[0]
check('流动性反转', '主账户CAGR/MDD', '-26.32% / -96.86%',
      f'{a0.CAGR*100:.2f}% / {a0.最大回撤*100:.2f}%', abs(a0.CAGR+0.2632) < 0.001 and abs(a0.最大回撤+0.9686) < 0.002)

# ---------- 股价分层 ----------
P = f'{ROOT}/trading_flash/2026-10-27_股价分层_盈利差异'
pair = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{P}/同日配对_检验.csv')
diff = pair['差(便宜-贵)'].astype(float)
tt = stats.ttest_1samp(diff, 0)
check('股价分层', '同日配对差/p', '+2.50pp / p=0.343',
      f'{diff.mean()*100:.2f}pp / p={tt.pvalue:.3f} (n={len(diff)})',
      abs(diff.mean()-0.025) < 0.002 and abs(tt.pvalue-0.343) < 0.01)
hb = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{P}/换基准对照.csv')
r0 = hb[hb.基准 == '实际成交价'].iloc[0]
check('股价分层', '<15 vs >=15每笔/p', '6.68% vs 6.53% / p=0.949',
      f'{r0.低于15平均*100:.2f}% vs {r0["15以上平均"]*100:.2f}% / p={r0.p值:.3f}',
      abs(r0.低于15平均-0.0668) < 0.001 and abs(r0['15以上平均']-0.0653) < 0.001 and abs(r0.p值-0.949) < 0.01)

# ---------- 万元账户 ----------
W1 = f'{ROOT}/2026-10-30_万元账户_低换手估值/结果'
c1 = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{W1}/核心系统中文比较.csv')
b20 = c1[c1.配置编号 == 'B20_M3'].iloc[0]
check('万元低换手', '最高配置中位年化/MDD', '+5.81% / -57.07%',
      f'{b20.净年化中位数*100:.2f}% / {b20.最大回撤中位数*100:.2f}%',
      abs(b20.净年化中位数-0.0581) < 0.001 and abs(b20.最大回撤中位数+0.5707) < 0.002)
holm_pass = int((c1.相对差霍尔姆p < 0.05).sum())
check('万元低换手', '24项Holm通过数', '0', f'{holm_pass} (12配置x对零/对控制)', holm_pass == 0)
j1 = json.load(open(f'{W1}/结论.json'))
check('万元低换手', '沪深300价格基准CAGR', '+5.24%', f'{j1["沪深300价格基准"]["CAGR"]*100:.2f}%',
      abs(j1['沪深300价格基准']['CAGR']-0.0524) < 0.001)
W2 = f'{ROOT}/2026-10-31_万元账户_持仓数与审视频率/结果'
c2 = pd.read_csv(encoding='utf-8-sig', filepath_or_buffer=f'{W2}/九配置中文比较.csv')
s8 = c2[c2.配置 == 'S8_M1'].iloc[0]
check('万元持仓数', '8槽每月买不起一手', '95.5次', f'{s8.买入失败中位}', abs(float(s8.买入失败中位)-95.5) < 0.05)
cagr = c2.中位年化.str.replace('%', '').str.replace('+', '').astype(float)
check('万元持仓数', '9配置中位CAGR范围', '2.13%-3.33%', f'{cagr.min():.2f}%-{cagr.max():.2f}%',
      abs(cagr.min()-2.13) < 0.01 and abs(cagr.max()-3.33) < 0.01)
check('万元持仓数', '候选/交付通过数', '0', f"候选线通过={int((c2.候选线=='通过').sum())}, 交付线通过={int((c2.交付线=='通过').sum())}",
      (c2.候选线 != '通过').all() and (c2.交付线 != '通过').all())

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, 'F08_交付数字核验.csv'), index=False)
print(out.to_string(index=False))
print('不一致项:', out[~out.一致.astype(bool)][['家族', '项目']].to_dict('records'))