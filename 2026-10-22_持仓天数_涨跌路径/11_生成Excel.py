# -*- coding: utf-8 -*-
"""把当前时段策略的汇总表写入一个中文 Excel，便于筛选和审计。"""
import pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

OUT='09_当前时段_策略.xlsx'
wb=Workbook(); wb.remove(wb.active)
head_fill=PatternFill('solid',fgColor='1F4E78'); head_font=Font(color='FFFFFF',bold=True)
thin=Side(style='thin',color='D9D9D9'); border=Border(left=thin,right=thin,top=thin,bottom=thin)

def add_sheet(name,df,widths=None,pct_cols=None):
    ws=wb.create_sheet(name)
    ws.append(list(df.columns))
    for c in ws[1]:
        c.fill=head_fill; c.font=head_font; c.alignment=Alignment(horizontal='center',vertical='center'); c.border=border
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v,float) and np.isnan(v)) else (v.item() if hasattr(v,'item') else v) for v in row])
    for r in ws.iter_rows(min_row=2,max_row=ws.max_row):
        for c in r:
            c.border=border
    if pct_cols:
        for j,col in enumerate(df.columns,1):
            if col in pct_cols:
                for r in range(2,ws.max_row+1):
                    ws.cell(r,j).number_format='0.0%'
    if widths:
        for j,w in enumerate(widths,1):
            ws.column_dimensions[get_column_letter(j)].width=w
    else:
        for j,col in enumerate(df.columns,1):
            ws.column_dimensions[get_column_letter(j)].width=max(10,min(28,len(str(col))*2+2))
    ws.freeze_panes='A2'

# 一页纸
ws=wb.create_sheet('一页纸')
lines=[
 ('当前时段策略一页纸（2026-10-22 收盘）',''),
 ('一句话','今天宽度 11.7%、0 个有效信号；最优动作是空仓等触发，不是买股票。'),
 ('三档开关','宽度>50%：70/15 核心；10–50%：空仓/货基；<10%：极端超跌卫星（可选）。'),
 ('今天','0 买入；13 个近信号全部卡行业闸门。'),
 ('70/15 核心','+30.3% 年化 / -19.6% 最大回撤 / 62.1% 胜率 / 盈亏比 2.97 / 140 笔。'),
 ('增强版','+37.4% / -19.7% / 62.9%，但只多 24 笔超跌；去掉最好一笔 +31.8%，延迟 1 天 +29.2%。'),
 ('回撤尾部','1 年自举 P95 -24%、3 年 P95 -30%；不要按 -19.6% 准备。'),
 ('最大集中度','前五笔净盈亏占比 68.7%（基准）/ 62.8%（增强）。'),
 ('结论','现在的最优策略是“等”；空仓是策略的一部分。'),
 ('边界','样本内、后复权、未计滑点/冲击/ST 5% 跌停；不构成投资建议。'),
]
ws.append(['项目','内容'])
for c in ws[1]:
    c.fill=head_fill; c.font=head_font; c.alignment=Alignment(horizontal='center'); c.border=border
for a,b in lines:
    ws.append([a,b])
ws.column_dimensions['A'].width=18; ws.column_dimensions['B'].width=110
for r in ws.iter_rows(min_row=2,max_row=ws.max_row):
    for c in r: c.alignment=Alignment(wrap_text=True,vertical='top'); c.border=border
ws['A2'].font=Font(bold=True,size=13); ws.merge_cells('A2:B2')

# 当前状态
cur=pd.DataFrame([
 ('市场宽度20','11.7%','闸门关（阈值50%）'),
 ('市场宽度60','15.2%','中期结构弱'),
 ('市场宽度120','13.2%','长期结构弱'),
 ('等权全A距9/22高点','-10.7%','一个月快速回撤'),
 ('20日收益','-7.3%','下跌加速'),
 ('20日年化波动','21.8%','高波动'),
 ('MA20/60/120','低于三线','空头排列'),
 ('行业','31个全部20日负收益','无避风港'),
 ('最大行业宽度','28.6%（银行）','远低于60%'),
 ('最大行业20日收益','+1.6%（传媒）','远低于5%'),
 ('70/15有效信号','0','0买入'),
 ('70/15近信号','13','全部卡行业闸门'),
 ('超跌卫星触发','否','要求宽度<10%'),
],columns=['指标','读数','说明'])
add_sheet('当前状态',cur,[28,30,40])

# 策略对比
M=pd.read_csv('10_策略对比.csv').rename(columns={'Ulcer':'回撤持续指数','最大回撤_盘中':'盘中最大回撤'})
add_sheet('策略对比',M,[16,10,10,10,10,12,12,10,10,12,12,12,12,12,12,12],
          pct_cols=['年化','年化波动','最大回撤','盘中最大回撤','最差20日','最差月','平均仓位'])

# 年度
Y=pd.read_csv('10_年度表现.csv')
add_sheet('年度表现',Y,[16,8,10,12,12],pct_cols=['收益','年内回撤','年内波动'])

# 压力
S=pd.read_csv('10_压力测试.csv')
add_sheet('压力测试',S,[16,18,12,14],pct_cols=['区间收益','区间最大回撤'])

# 自举
B=pd.read_csv('10_自举回撤.csv').rename(columns={'P95回撤':'95分位回撤','P99回撤':'99分位回撤'})
add_sheet('自举回撤',B,[16,8,8,12,12,12,12,12,12],
          pct_cols=['中位回撤','95分位回撤','99分位回撤','跌破20概率','跌破30概率','最差回撤'])

# 逐笔统计
T=pd.read_csv('10_逐笔统计.csv')
add_sheet('逐笔统计',T,[16,8,12,12,10,10,14,12,12,12],
          pct_cols=['单笔净均值','单笔净中位','净胜率','前五笔净盈亏占比','最差单笔','最好单笔'])

# 参数敏感性
P=pd.read_csv('08_参数敏感性.csv').rename(columns={'param':'参数','value':'取值','cagr':'年化','dd':'最大回撤','ddlow':'盘中最大回撤','calmar':'卡玛','n':'交易数','win':'胜率','pf':'盈亏比','final':'期末权益'})
add_sheet('参数敏感性',P,[14,10,10,12,12,10,10,10,12],
          pct_cols=['年化','最大回撤','盘中最大回撤'])

# 延迟
D=pd.read_csv('08_超跌信号延迟_安慰剂.csv').rename(columns={'cagr':'年化','dd':'最大回撤','calmar':'卡玛','n':'交易数','win':'胜率','pf':'盈亏比','final':'期末权益'})
add_sheet('延迟安慰剂',D,[12,10,12,12,10,10,10,12],
          pct_cols=['年化','最大回撤'])

# 宽度分桶
W=pd.read_csv('07_宽度分桶_前瞻.csv').rename(columns={'mb_bucket':'宽度区间'})
add_sheet('宽度分桶',W,[12,8,10,10,10,10,10,10,14,14],
          pct_cols=['未来5日','未来10日','未来20日','未来60日','未来20日胜率','未来60日胜率','未来20日路径最低均值','未来20日路径最低最差'])

# 近信号分宽度
N=pd.read_csv('07_近信号_分宽度.csv').rename(columns={'P10':'10分位'})
add_sheet('近信号分宽度',N,[14,22,10,10,10,10,10,10,12,12],
          pct_cols=['均值','中位','胜率','10分位','路径最低均值','路径最低最差'])

# 13 近信号
G=pd.read_csv('今日70_15近信号_2026-10-22.csv').rename(columns={'ATR占收盘':'真实波幅占收盘'})
add_sheet('今日近信号',G,[12,10,10,10,14,10,10,10,10,10,10,10],
          pct_cols=['真实波幅占收盘','距60日高点','20日收益','行业宽度','行业20日收益'])

# 今日超跌观察 top 50
O=pd.read_csv('09_今日超跌观察_2026-10-22.csv').head(50).rename(columns={'thscode':'代码','RSI14':'相对强弱指标14','ATR占收盘':'真实波幅占收盘','距MA20':'距20日均线','ST':'是否特别处理'})
add_sheet('今日超跌观察',O,[10,14,10,10,10,10,14,10,10,8],
          pct_cols=['真实波幅占收盘','20日收益','距20日均线'])

wb.save(OUT)
print('saved',OUT)
