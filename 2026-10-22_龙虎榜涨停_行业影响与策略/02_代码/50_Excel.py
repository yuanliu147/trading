# -*- coding: utf-8 -*-
"""生成中文 Excel 数据表。"""
import os
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
OUT = os.path.join(BASE, '03_结果')
XLSX = os.path.join(BASE, '龙虎榜涨停_数据表.xlsx')

read = lambda f, **kw: pd.read_csv(os.path.join(OUT, f), **kw)

metrics = read('最终策略_指标.csv')
yearly = read('最终策略_分年.csv', index_col=0)
yearly.index = yearly.index.astype(str)
period = read('最终策略_分期.csv', index_col=0)
period.index = period.index.astype(str)
cost = read('最终策略_成本敏感.csv')
param = read('最终策略_参数邻域.csv')
trades = read('最终策略_逐笔_均衡_N10_45bp.csv')
monthly = read('最终策略_月度收益.csv', index_col=0)
monthly.index = monthly.index.astype(str)
ev = read('行业预测_事件与非事件.csv')
ic = read('行业预测_RankIC.csv')
grp = read('行业特征_分组.csv')
stockev = read('事件研究_汇总.csv')
lhb_year = read('龙虎榜_分年分类汇总.csv')
buy = read('龙虎榜_可买率.csv')

summary = pd.DataFrame([
    ['研究问题', '龙虎榜/涨停板对同行业板块的外溢；能否构建年化30%/40%/50%的策略'],
    ['样本', '2016-10-14 至 2026-10-13，全 A（含退市补漏），日频'],
    ['龙虎榜', '东方财富历史龙虎榜 + 机构买卖统计；153,944 个股票-日事件'],
    ['涨停', '收盘封板 133,959 次；非 ST 116,738 次进入事件研究'],
    ['行业', '时点申万一级 32 个行业，71,702 个行业-日'],
    ['核心结论 1', '涨停/龙虎榜对同行业非事件股是负外溢（虹吸/挤出），不是扩散'],
    ['核心结论 2', '涨停行业非事件股 5 日跑输非事件行业 0.37%（t=-13.5），10 日 0.68%（t=-17.2）'],
    ['核心结论 3', '龙虎榜行业非事件股 5 日跑输 0.31%（t=-13.4），10 日 0.57%（t=-17.4）'],
    ['核心结论 4', '机构龙虎榜净买的连续 IC 为正，但分组差≈0、样本外不稳；不能做行业轮动'],
    ['核心结论 5', '连板高度在 2016-2021 有强动量，2024-2026 反向；不稳健'],
    ['核心结论 6', '看起来57.5%的“龙虎榜+低开”策略是前视偏差；实盘建模后-32.4%/-98.0%'],
    ['策略', '看起来的“龙虎榜事件+次日低开”策略：实盘建模后为负，不能做'],
    ['主方案（实盘版）', 'N=10，每只 10% 权益，45bp；T+1 收盘跌停顺延 T+2 开盘卖'],
    ['主方案年化（实盘版）', f"{metrics.loc[metrics.策略=='均衡_N10_45bp','年化'].iloc[0]*100:.1f}%"],
    ['主方案最大回撤（实盘版）', f"{metrics.loc[metrics.策略=='均衡_N10_45bp','最大回撤'].iloc[0]*100:.1f}%"],
    ['主方案 Sharpe（实盘版）', f"{metrics.loc[metrics.策略=='均衡_N10_45bp','Sharpe'].iloc[0]:.2f}"],
    ['样本外（实盘版）', f"{period.loc['OOS'].iloc[0]*100:.1f}%（2024-01 至 2026-10）"],
    ['基准', f"全 A 等权年化 {metrics.loc[0,'基准年化']*100:.1f}%，最大回撤 {metrics.loc[0,'基准最大回撤']*100:.1f}%"],
    ['关键边界', '理想版57.5%依赖剔除T+1收盘跌停（5.24%交易）；实盘版-32.4%，止损/过滤救不回来'],
    ['关键边界', '实盘版成本敏感：20bp +30.7%，30bp +0.4%，45bp -32.4%，60bp -54.4%'],
    ['关键边界', '40 组参数在实盘建模下全部为负；不是某一组没调好'],
    ['关键边界', '容量：股票池要求 20 日均额≥3000 万；小资金可行，大资金冲击成本上升'],
    ['声明', '历史回测，不构成投资建议'],
], columns=['项目', '内容'])

with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='摘要', index=False)
    metrics.to_excel(writer, sheet_name='策略指标', index=False)
    yearly.to_excel(writer, sheet_name='分年收益')
    period.to_excel(writer, sheet_name='分期收益')
    monthly.to_excel(writer, sheet_name='月度收益')
    cost.to_excel(writer, sheet_name='成本敏感', index=False)
    param.to_excel(writer, sheet_name='参数邻域', index=False)
    trades.to_excel(writer, sheet_name='逐笔交易', index=False)
    ev[ev['口径'] == '剔除两者'].to_excel(writer, sheet_name='行业事件研究', index=False)
    ick = ic[(ic['期'].isin(['全样本', '设计', '验证', 'OOS'])) &
             (ic['特征'].isin(['lhb_org_ratio', 'lhb_net_ratio', 'lhb_share', 'lu_share', 'lu2_share', 'lu_open_share'])) &
             (ic['目标'].isin(['行业全部收益_h1', '行业全部收益_h3', '行业全部收益_h5',
                               '行业剔除LHB_h3', '行业剔除LHB_h5', '行业剔除涨停_h3', '行业剔除涨停_h5',
                               '行业剔除两者_h3', '行业剔除两者_h5']))]
    ick.to_excel(writer, sheet_name='行业RankIC', index=False)
    grp.to_excel(writer, sheet_name='行业分组收益', index=False)
    sev = stockev[stockev['分组'].isin(['全部', 'net_bucket', 'org_bucket', 'chg_bucket', 'consec_bucket',
                                        'lu_one', 'lu_share_bucket', 'mkt_regime'])]
    sev.to_excel(writer, sheet_name='个股事件研究', index=False)
    lhb_year.to_excel(writer, sheet_name='龙虎榜结构', index=False)
    pd.read_csv(os.path.join(OUT, '龙虎榜_三日榜研究.csv')).to_excel(writer, sheet_name='三日榜研究', index=False)
    pd.read_csv(os.path.join(OUT, '行业异质性.csv')).to_excel(writer, sheet_name='行业异质性', index=False)
    pd.read_csv(os.path.join(OUT, '行业分状态.csv')).to_excel(writer, sheet_name='行业分状态', index=False)
    pd.read_csv(os.path.join(OUT, '前视审计_理想vs实盘.csv')).to_excel(writer, sheet_name='前视审计', index=False)
    buy.to_excel(writer, sheet_name='次日可买率', index=False)
    pd.DataFrame([
        ['龙虎榜明细', 'AkShare stock_lhb_detail_em（东方财富）2016-2026'],
        ['机构统计', 'AkShare stock_lhb_jgmmtj_em（东方财富）2016-2026'],
        ['个股日线', '本地 DuckDB v_daily_hfq，后复权；2016-10-14 至 2026-10-13'],
        ['涨停事件', '本地涨停事件清单 + 日线涨跌停价交叉核对'],
        ['行业', '跌停研究项目分析面板的时点申万一级 industry_id'],
        ['龙虎榜去重', '同一股票-日多个上榜原因时，优先单日榜（涨幅>换手>振幅>跌幅），连续三日榜单独列'],
        ['收益口径', 'T 日收盘信号；T+1 开盘买入；T+1 收盘卖出；后复权'],
        ['成本', '单边 45bp（佣金+滑点），卖出另加印花税 2023-08-28 前 10bp、之后 5bp'],
        ['过滤', 'ST、上市<60日、北交所、20日均额<3000万、停牌/跨越交易日、T+1 一字/涨停/跌停开盘、T+1 收盘跌停'],
        ['复现', '01_下载龙虎榜.py → 05_整理龙虎榜.py → 10_构建日频面板.py → 20_事件研究.py → 25_行业预测.py → 30/35_策略 → 40_图表 → 50_Excel'],
    ], columns=['项目', '说明']).to_excel(writer, sheet_name='数据说明', index=False)

# 格式
from openpyxl import load_workbook
wb = load_workbook(XLSX)
head_fill = PatternFill('solid', fgColor='1F4E5F')
head_font = Font(color='FFFFFF', bold=True, size=10)
thin = Side(style='thin', color='D9D9D9')
for ws in wb.worksheets:
    for c in ws[1]:
        c.fill = head_fill; c.font = head_font
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = 'A2'
    for col in ws.columns:
        maxlen = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        width = min(max(10, maxlen * 1.2 + 2), 48)
        ws.column_dimensions[get_column_letter(col[0].column)].width = width
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 300)):
        for c in row:
            c.border = Border(bottom=thin)
            c.alignment = Alignment(vertical='center', wrap_text=False)
wb.save(XLSX)
print('saved', XLSX, os.path.getsize(XLSX) / 1024 / 1024, 'MB')
