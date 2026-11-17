# -*- coding: utf-8 -*-
"""生成中文 Excel 数据表（工作表名、表头均为中文）。"""
import os, time, warnings
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')
BASE = '/Users/admin/WorkSpace/trading'
ROOT = os.path.join(BASE, '2026-10-22_跌停前特征_全A')
OUT = os.path.join(ROOT, '03_结果')
XLSX = os.path.join(ROOT, '跌停前特征_数据表.xlsx')
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

FEATURE_CN = {
    'one_word_ld_yesterday': '昨日一字跌停', 'days_since_ld': '距上次跌停', 'ld_yesterday': '昨日收盘跌停',
    'consec_lu': '连续涨停天数', 'lu_count20': '20日涨停次数', 'ret1': '前1日收益', 'ret20': '前20日收益',
    'amt_ratio20': '成交额/20日均额', 'opened_ld_yesterday': '昨日打开跌停', 'dist_ma20': '距20日线',
    'opened_lu_yesterday': '昨日炸板', 'ret5': '前5日收益', 'ld_count20': '20日跌停次数',
    'ret60': '前60日收益', 'days_since_lu': '距上次涨停', 'vr5_20': '5/20日量比', 'gap': '跳空',
    'dist_high250': '距250日高点', 'mkt_ld_share': '全市场跌停占比', 'lu_yesterday': '昨日收盘涨停',
    'one_word_lu_yesterday': '昨日一字涨停', 'mkt_ret1': '全市场前1日收益',
    'vol20': '20日波动', 'atr_pct': '真实波幅/收盘', 'dist_ma60': '距60日线', 'dist_high60': '距60日高点',
    'ld_count60': '60日跌停次数', 'ld_touch_count20': '20日触及跌停次数', 'lu_count5': '5日涨停次数',
    'lu_open_count20': '20日炸板次数', 'rsi14': '相对强弱指标', 'market_breadth20': '市场宽度20',
    'mkt_ld_rank250': '市场跌停占比250日分位', 'mkt_below_ma60': '市场指数低于60日线',
    'sector_ld_share': '行业跌停占比', 'sector_ret20': '行业20日收益', 'sector_rs': '行业相对强弱',
    'sector_ret5': '行业5日收益', 'close_pos': '收盘位置', 'upper': '上影线', 'body': '实体',
    'lower': '下影线', 'macd_hist': '异同移动平均柱', 'sw_above60': '行业指数高于60日线',
}
HEAD = {
    'P(次日跌停)': '次日跌停概率', 'P(5日内跌停)': '5日内跌停概率', 'P(盘中触及)': '盘中触及跌停概率',
    'lift': '提升倍数', '95%下限': '95%置信下限', '95%上限': '95%置信上限',
    '事件侧差(pp)': '事件侧差（百分点）', 'p值': '显著性概率值', 'q值': '多重检验校正值',
    '样本充足': '样本是否充足', '训练期lift': '训练期提升倍数', '检验期lift': '检验期提升倍数',
    '最强组lift': '最强组提升倍数', '最弱组lift': '最弱组提升倍数',
    '最强组事件侧占比': '最强组事件侧占比', '最小q值': '最小校正值',
    'cluster下限': '按日聚类下限', 'cluster上限': '按日聚类上限',
    'P次日': '次日跌停概率', 'P五日内': '5日内跌停概率', 'P触及': '盘中触及概率',
    'AUC': '排序能力曲线下面积', 'PR-AUC': '平均精确率', 'Brier': '布里尔分数',
    'Brier基准': '布里尔分数基准', 'top1%捕获': '前1%捕获率', 'top1%精度': '前1%精度',
    'top1%lift': '前1%提升倍数', 'top5%捕获': '前5%捕获率', 'top5%精度': '前5%精度',
    'top5%lift': '前5%提升倍数', 'top10%捕获': '前10%捕获率', 'top10%精度': '前10%精度',
    'top10%lift': '前10%提升倍数', 'AUC下降均值': '排序能力下降均值', 'AUC下降标准差': '排序能力下降标准差',
    '5%跌停数': '5%跌停数', '跌停日类型': '跌停日类型',
}
def cn(df, extra=None):
    m = dict(HEAD)
    if extra:
        m.update(extra)
    return df.rename(columns={k: v for k, v in m.items() if k in df.columns})

def map_feat(df, cols=('特征', '最强特征', 'feature')):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(FEATURE_CN).fillna(df[c])
    return df

def read(name, **kw):
    return pd.read_csv(os.path.join(OUT, name), **kw)

# ---------------- 准备各表 ----------------
sheets = {}
sheets['年度基准率'] = read('分年基准率.csv')
sheets['分板块基准率'] = read('分年分板块基准率.csv').rename(columns={'year': '年份'})
x = read('重点条件概率表.csv'); x = map_feat(x)
sheets['重点条件概率'] = cn(x)
x = read('单特征概率表.csv'); x = map_feat(x)
sheets['单特征概率_连续'] = cn(x)
x = read('单特征概率表_布尔.csv'); x = map_feat(x)
sheets['单特征概率_布尔'] = cn(x)
x = read('单特征汇总.csv'); x = map_feat(x)
x = x.rename(columns={'事件侧最大差(pp)': '事件侧最大差（百分点）'})
sheets['特征汇总'] = cn(x)
x = read('特征族汇总.csv'); x = map_feat(x, cols=('最强特征',))
sheets['特征族汇总'] = cn(x)
x = read('形态组合概率.csv')
sheets['形态组合'] = cn(x)
x = read('风险评分_六条清单.csv')
sheets['六条风险评分'] = cn(x)
x = read('事件侧分组画像.csv')
x = x.rename(columns={
    'ret20_中位数':'前20日收益中位数','ret60_中位数':'前60日收益中位数','vol20_中位数':'20日波动中位数',
    'atr_pct_中位数':'真实波幅中位数','dist_ma20_中位数':'距20日线中位数','dist_ma60_中位数':'距60日线中位数',
    'rsi14_中位数':'相对强弱指标中位数','amt_ratio20_中位数':'成交额量比中位数','close_pos_中位数':'收盘位置中位数',
    'upper_中位数':'上影线中位数','gap_中位数':'跳空中位数','ld_count20_中位数':'20日跌停次数中位数',
    'lu_count20_中位数':'20日涨停次数中位数','market_breadth20_中位数':'市场宽度中位数','mkt_ld_share_中位数':'市场跌停占比中位数',
    'ret20>20%占比':'前20日涨>20%占比','ret20<-10%占比':'前20日跌>10%占比','vol20>50%占比':'20日波动>50%占比',
    'amt_ratio20>1.5占比':'成交额量比>1.5占比','dist_ma20>10%占比':'距20日线>10%占比',
    'dist_ma20<-10%占比':'距20日线<-10%占比','昨涨停占比':'昨日涨停占比','20日有跌停占比':'20日内有跌停占比',
    '弱市场占比':'弱市场占比'})
sheets['事件侧画像'] = x
# 事件窗口宽表
p = read('事件窗口_特征路径.csv')
p['feature'] = p['feature'].map(FEATURE_CN).fillna(p['feature'])
wide = p.pivot_table(index=['offset', 'group'], columns='feature', values='median').reset_index()
wide = wide.rename(columns={'offset': '相对跌停日', 'group': '分组'})
sheets['事件窗口中位数'] = wide
for name, f in [('二维_收益位置', '二维_20日收益与距60日线.csv'),
                ('二维_市场波动', '二维_市场跌停占比与个股波动.csv')]:
    x = read(f).rename(columns={'x1': '横轴分组', 'x2': '纵轴分组', 'n': '样本数', 'ld': '跌停数', 'rate': '次日跌停概率'})
    sheets[name] = x
x = read('分行业基准率.csv').dropna(subset=['行业']).drop(columns=['industry_id'], errors='ignore')
sheets['分行业'] = cn(x)
x = read('模型对比.csv')
sheets['模型对比'] = cn(x)
x = read('模型校准后对比.csv')
sheets['模型校准对比'] = cn(x)
x = read('模型校准表_校准后.csv').rename(columns={'十分位': '预测概率十分位', '预测均值': '预测概率均值',
                                                    '实际跌停率': '实际跌停率', 'lift': '提升倍数'})
sheets['模型校准表'] = x
x = read('模型置换重要性.csv'); x = map_feat(x)
sheets['模型重要性'] = cn(x)
x = read('模型分年排序能力.csv').rename(columns={'AUC': '排序能力曲线下面积', 'top5%lift': '前5%提升倍数'})
sheets['模型分年'] = x
x = read('事件侧特征对比.csv'); x = map_feat(x)
x = x.rename(columns={'标准化差(SMD)': '标准化差', 'KS统计量': '分布差异统计量', '单变量AUC': '单变量排序能力'})
sheets['事件侧对比'] = x
x = read('模型前1%构成.csv').T.reset_index()
x.columns = ['项目', '数值']
sheets['前1%构成'] = x
x = read('模型分市场状态.csv').rename(columns={'AUC': '排序能力曲线下面积', 'top5%精度': '前5%精度', 'top5%lift': '前5%提升倍数'})
sheets['模型分市场状态'] = x
x = read('稳健性_关键条件分年.csv')
sheets['稳健性_分年'] = cn(x)
x = read('稳健性_分板块.csv')
sheets['稳健性_分板块'] = cn(x)
x = read('稳健性_危机年平静年.csv')
sheets['稳健性_危机平静'] = cn(x)
x = read('稳健性_退市近似样本.csv')
sheets['退市近似样本'] = cn(x)
x = read('特别处理股5%跌停概率.csv').rename(columns={'5%跌停数': '5%跌停数', 'lift': '提升倍数'})
sheets['特别处理股5%跌停'] = x
x = read('昨日跌停三态_次日风险.csv')
sheets['昨日跌停三态'] = cn(x)
x = read('跌停日三态_次日风险.csv')
sheets['跌停日三态'] = x
x = pd.read_csv(os.path.join(ROOT, '01_数据', '字段说明.csv')).rename(columns={'字段': '字段名', '中文说明': '中文说明'})
x['中文说明'] = x['中文说明'].str.replace('ST代理', '特别处理股代理', regex=False)
sheets['字段说明'] = x

# ---------------- 核心结论 ----------------
core = pd.DataFrame([
    ('主样本股票-日', '9,569,164', '2016-10-14～2026-10-13；非北交所、非特别处理股代理、上市超过60个交易日、剔除退市近似股'),
    ('收盘跌停事件', '37,412', '收盘价=最低价=按板块规则计算的跌停价'),
    ('次日收盘跌停基准率', '0.391%', '主样本股票-日口径'),
    ('次日盘中触及跌停率', '0.716%', '最低=跌停价，包括收盘封死和打开'),
    ('5日内收盘跌停率', '1.718%', '当日～第4个交易日至少一次收盘跌停'),
    ('昨日一字跌停→次日跌停', '28.97%', '样本4,626；提升74.1倍；按日聚类区间15.1%～49.3%'),
    ('昨日非一字封死→次日跌停', '10.03%', '样本33,316；提升25.7倍'),
    ('昨日收盘打开跌停→次日跌停', '5.49%', '样本31,352；提升14.0倍'),
    ('昨日收盘涨停→次日跌停', '2.68%', '样本116,219；提升6.9倍；5日内跌停15.7%'),
    ('昨日≥2连板→次日跌停', '7.05%', '样本25,085；提升18.0倍；5日内跌停35.9%'),
    ('20日涨幅≥20%→次日跌停', '1.86%', '样本592,952；提升4.7倍；事件覆盖率29.4%'),
    ('20日波动≥50%→次日跌停', '0.85%', '样本2,718,093；提升2.2倍；事件覆盖率62.1%'),
    ('全市场跌停占比1%～2%→次日跌停', '2.99%', '样本264,985；提升7.6倍；极端恐慌≥4%反而降至1.45%'),
    ('六条评分0条→次日跌停', '0.07%', '样本5,601,675'),
    ('六条评分3条→次日跌停', '4.03%', '样本181,335；覆盖19.5%的跌停事件'),
    ('六条评分4条→次日跌停', '9.19%', '样本29,321；覆盖7.2%的跌停事件'),
    ('六条评分5条→次日跌停', '17.03%', '样本2,314；覆盖1.1%的跌停事件'),
    ('模型样本外排序能力（曲线下面积）', '0.881', '2022-01～2026-10；训练2016-10～2020-12，2021校准'),
    ('模型前1%风险名单精度', '8.76%', '前1%中实际跌停比例；基准0.343%'),
    ('模型前1%风险名单捕获率', '25.6%', '前1%名单覆盖了25.6%的样本外跌停事件'),
    ('模型日内排序能力（曲线下面积）', '0.822', '同一天内比较个股；剔除市场择时'),
    ('事件侧：高位强势型占比', '37.9%', '20日涨>10%且距60日线>0；条件概率0.95%'),
    ('事件侧：下跌延续型占比', '18.6%', '20日涨<-10%且距60日线<0；条件概率0.48%'),
    ('事件侧：中性型占比', '41.4%', '条件概率0.24%'),
], columns=['指标', '数值', '说明'])

# ---------------- 说明 ----------------
notes = pd.DataFrame([
    ('文件', '个股跌停前的特征与概率：全A股 2016-10～2026-10'),
    ('研究问题', '发生收盘跌停的股票，在前1～20个交易日通常有什么特征？某个特征出现后，未来1日/5日跌停概率是多少？'),
    ('数据', '本地行情数据库未复权日线+后复权日线；时点申万行业；市场/行业宽度从全市场面板实时计算。'),
    ('主样本', '9,569,164个股票-日；剔除北交所、上市不足60个交易日、特别处理股代理（过去60日出现5%涨跌停）、102只退市近似股。'),
    ('跌停定义', '跌停价=四舍五入(调整前收×(1-限制比例),2)；收盘跌停=收盘=最低=跌停价。主板10%，创业板2020-08-24前10%/后20%，科创板20%，北交所30%。'),
    ('精确度校验', '主板精确口径34,456个事件；与“收益≤-9.8%且收盘=最低”的阈值口径相比，精确率99.7%、召回率97.7%。'),
    ('特别处理股处理', '本地无历史股票名称；用“收盘=最低=5%跌停价”识别5%跌停，单独分析。主口径排除过去60日出现5%涨跌停的股票。'),
    ('特征时点', '所有特征取前一日收盘及以前；标签为当日收盘跌停。行业按公告起始日匹配；市场/行业宽度只用当日横截面。'),
    ('训练/检验', '主模型训练2016-10～2020-12，2021年做等渗校准，2022-01～2026-10为样本外检验。单特征表另给2016-2021/2022-2026两段。'),
    ('模型', '直方图梯度提升；训练集负样本10:1下采样并加权还原；市场特征、个股特征、市场+个股三组对照。'),
    ('重要限制', '日线数据，无分时、新闻、公告、龙虎榜、融资融券；无法知道盘中封死时间或突发利空。'),
    ('重要限制', '历史特别处理股名称缺失，5%跌停用价格规则代理；退市股只补了102只，仍可能有幸存者偏差。'),
    ('重要限制', '跌停高度聚集于少数交易日；单特征表中的普通置信区间偏窄，关键结论另给按日聚类区间。'),
    ('重要限制', '这是历史描述与风险排序，不是“预测明天哪只股票一定跌停”的交易系统，不保证未来收益。'),
    ('代码', '02_代码/10构建事件表；20构建分析面板；30单特征；35形态与事件研究；40模型；45校准与日内；50稳健性与特别处理股；55图表与评分；60三态补充；70数据表。'),
], columns=['项目', '内容'])

# ---------------- 写入 Excel ----------------
log('写入 Excel')
with pd.ExcelWriter(XLSX, engine='openpyxl') as writer:
    notes.to_excel(writer, sheet_name='说明', index=False)
    core.to_excel(writer, sheet_name='核心结论', index=False)
    for name, data in sheets.items():
        data.to_excel(writer, sheet_name=name[:31], index=False)

wb = load_workbook(XLSX)
head_fill = PatternFill('solid', fgColor='1F4E79')
head_font = Font(color='FFFFFF', bold=True, size=10)
thin = Side(style='thin', color='D9D9D9')
border = Border(left=thin, right=thin, top=thin, bottom=thin)
for ws in wb.worksheets:
    max_row = ws.max_row; max_col = ws.max_column
    for c in range(1, max_col + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = head_fill; cell.font = head_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border
    ws.freeze_panes = 'A2'
    if max_row > 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(max_col)}{max_row}"
    # 列宽
    for c in range(1, max_col + 1):
        vals = [str(ws.cell(row=r, column=c).value or '') for r in range(1, min(max_row, 200) + 1)]
        width = min(max(len(v) for v in vals) * 1.15 + 2, 42)
        ws.column_dimensions[get_column_letter(c)].width = max(width, 8)
    # 数字格式
    for c in range(1, max_col + 1):
        head = str(ws.cell(row=1, column=c).value or '')
        pct = any(k in head for k in ['概率', '率', '占比', '基准', '下限', '上限', '捕获', '精度', '覆盖', '实际跌停']) and '布里尔' not in head
        if pct and not any(k in head for k in ['样本数', '事件数', '跌停数', '数量']):
            for r in range(2, max_row + 1):
                ws.cell(row=r, column=c).number_format = '0.00%' if '预测概率均值' not in head else '0.0000%'
        if '提升倍数' in head:
            for r in range(2, max_row + 1):
                ws.cell(row=r, column=c).number_format = '0.00'
        if head in ('样本数', '跌停数', '事件数', '5%跌停数', '盘中触及数', '收盘跌停数', '样本占比'):
            for r in range(2, max_row + 1):
                ws.cell(row=r, column=c).number_format = '#,##0'
        for r in range(2, max_row + 1):
            ws.cell(row=r, column=c).alignment = Alignment(vertical='top', wrap_text=True)
    ws.row_dimensions[1].height = 32

wb.save(XLSX)
log(f'完成：{XLSX}（{os.path.getsize(XLSX)/1e6:.1f} MB，{len(wb.sheetnames)}个工作表）')
print(wb.sheetnames)
