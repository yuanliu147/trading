# -*- coding: utf-8 -*-
import os,pandas as pd,numpy as np
BASE='/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制'
OUT=os.path.join(BASE,'03_结果')
XLS=os.path.join(BASE,'涨幅趋势_数据表.xlsx')
feature_map={'atr_pct':'平均真实波幅','vol20':'20日波动率','amt20':'20日成交额','trend_age':'趋势年龄','dist_ma20':'距20日均线',
 'lu_count20':'20日涨停次数','gap_count20':'20日跳空次数','sector_breadth60':'行业宽度60','sw_ret20':'行业指数20日涨幅',
 'market_ret20':'市场20日涨幅','market_sent':'市场情绪','turn_z250':'换手率历史标准分','ret20_hist_z':'20日涨幅自身历史标准分',
 'own_base20':'自身历史趋势基准','ma60_slope':'60日均线斜率','sector_rs':'行业相对强弱','stock_sector_z':'行业内标准分',
 'dist_high250':'距250日高点','vr5_20':'5比20日量比','dist_ma20':'距20日均线','ep_gain':'趋势累计涨幅','dd60':'距60日高点回撤',
 'er20':'趋势效率','ma20_slope':'20日均线斜率','up_ratio20':'20日上涨天数占比','nh20_count':'20日新高次数','rsi14':'相对强弱指标',
 'consec_up':'连续上涨天数','dist_high500':'距500日高点','own_win20':'自身历史趋势胜率','vol_hist_z':'波动率历史标准分',
 'sector_breadth20':'行业宽度20','sector_ret20':'行业20日涨幅','sector_ret60':'行业60日涨幅','sw_above60':'行业指数在60日均线上',
 'sw_slope60':'行业指数60日斜率','market_breadth20':'市场宽度','market_dispersion':'市场分化','market_limitup':'市场涨停占比',
 'market_amt_ratio':'市场成交额量比','market_newhigh':'市场新高占比','amt_ratio20':'成交额量比','gap':'跳空','upper':'上影线',
 'close_pos':'收盘位置','vp_corr20':'量价相关性','mcap':'流通市值','amt_trend':'成交额趋势','macd_hist':'MACD柱',
 'market_dispersion':'市场分化','market_limitup':'市场涨停占比','market_amt_ratio':'市场成交额量比','market_newhigh':'市场新高占比',
 'gap':'跳空','upper':'上影线','lu_count20':'20日涨停次数','gap_count20':'20日跳空次数','turn_z250':'换手率历史标准分','amt20':'20日成交额'}
rank_map={'r_atr':'波动率最小','r_amt':'成交额最小','r_age':'趋势年龄最小','r_dd':'回撤最小','r_dma':'距20日均线最近',
 'r_sbr':'行业宽度最大','r_sr':'行业动量最大','r_ssz':'行业内标准分最小','r_score':'综合评分最高','r_score2':'综合评分加成交额'}
exit_map={'r20':'持有20日','r40':'持有40日','r_stop40':'固定止损加40日','r_stoptrail20':'止损加20日均线','r_partial':'止损加2倍风险分批',
 'st15_h20':'1.5倍波幅止损加20日','st20_h20':'2倍波幅止损加20日','st8_h20':'8%止损加20日','tp2r_h20':'2倍风险止盈一半加20日'}
scen_map={'days':'交易日数','total':'累计收益','cagr':'年化收益','maxdd':'最大回撤','maxdd_low':'最低价回撤','sharpe':'夏普比率','trades':'交易笔数','win':'胜率',
 'pf':'盈亏比','avg_hold':'平均持有','exposure':'平均仓位','bench_total':'基准累计','bench_dd':'基准最大回撤','bench_sharpe':'基准夏普'}

def sec(txt): return txt.replace('MA20','20日均线').replace('MA10','10日均线').replace('MA60','60日均线').replace('MA5','5日均线')

def write(w,name,df):
    for c in df.columns:
        if c in feature_map: df=df.rename(columns={c:feature_map[c]})
        elif c in rank_map: df=df.rename(columns={c:rank_map[c]})
        elif c in exit_map: df=df.rename(columns={c:exit_map[c]})
        elif c in scen_map: df=df.rename(columns={c:scen_map[c]})
    df.to_excel(w,sheet_name=name,index=False)

# 读入
dist=pd.read_csv(os.path.join(OUT,'A股收益分布_全样本.csv'))
ann=pd.read_csv(os.path.join(OUT,'A股年度分布.csv'))
ind=pd.read_csv(os.path.join(OUT,'A股_行业分组.csv'))
amtg=pd.read_csv(os.path.join(OUT,'A股_成交额分组.csv'))
age=pd.read_csv(os.path.join(OUT,'上涨趋势_年龄阶段.csv'))
epy=pd.read_csv(os.path.join(OUT,'上涨趋势_episode年度.csv'))
cond=pd.read_csv(os.path.join(OUT,'上涨趋势_条件扫描.csv'))
event=pd.read_csv(os.path.join(OUT,'上涨趋势_事件类型.csv'))
exitr=pd.read_csv(os.path.join(OUT,'出场规则_汇总.csv'))
rank=pd.read_csv(os.path.join(OUT,'信号排序.csv'))
scen=pd.read_csv(os.path.join(OUT,'组合回测_场景.csv'))
scen2=pd.read_csv(os.path.join(OUT,'组合回测_含退市.csv'))
model=pd.read_csv(os.path.join(OUT,'模型_十分位.csv'))
tr=pd.read_csv(os.path.join(OUT,'逐笔_含退市_训练.csv')); te=pd.read_csv(os.path.join(OUT,'逐笔_含退市_检验.csv'))
trade=pd.concat([tr.assign(期='训练 2017-2021'),te.assign(期='检验 2022-2026')],ignore_index=True)
hold=pd.read_csv(os.path.join(OUT,'每日持仓_含退市_检验.csv'))
first=pd.read_csv(os.path.join(OUT,'交易复盘_首周.csv'))
stock=pd.read_csv(os.path.join(OUT,'个股案例_概览.csv'))
stockbig=pd.read_csv(os.path.join(OUT,'个股案例_大涨幅起点.csv'))
stopdist=pd.read_csv(os.path.join(OUT,'结构止损距离分组.csv'))
closestop=pd.read_csv(os.path.join(OUT,'收盘止损_短持有.csv'))
selected=pd.read_csv(os.path.join(OUT,'精选形态.csv'))
perm=pd.read_csv(os.path.join(OUT,'置换检验.csv'))
sent=pd.read_csv(os.path.join(OUT,'个股情绪合成.csv'))
selport=pd.read_csv(os.path.join(OUT,'精选版_组合.csv'))
dd=pd.read_csv(os.path.join(OUT,'回撤审计.csv'))
post=pd.read_csv(os.path.join(OUT,'交易复盘_卖出后.csv'))

# 中文列名
dist=dist.rename(columns={'期限':'期限','样本':'样本','均值':'均值','中位':'中位','标准差':'标准差','偏度':'偏度','p1':'1%分位','p5':'5%分位','p10':'10%分位','p25':'25%分位','p75':'75%分位','p90':'90%分位','p95':'95%分位','p99':'99%分位','胜率':'胜率','大涨10':'涨超10%','大涨20':'涨超20%','大跌10':'跌超10%','大跌20':'跌超20%'})
ann=ann.rename(columns={'年份':'年份','交易日':'交易日','1日均值':'1日均值','1日中位':'1日中位','1日胜率':'1日胜率','5日均值':'5日均值','5日中位':'5日中位','5日胜率':'5日胜率','20日均值':'20日均值','20日中位':'20日中位','20日胜率':'20日胜率','60日均值':'60日均值','60日中位':'60日中位','60日胜率':'60日胜率','平均涨幅':'平均涨幅','平均跌幅':'平均跌幅','涨跌比':'涨跌比','涨停占比':'涨停占比','跌停占比':'跌停占比','等权年度收益':'等权年度收益','个股年度中位':'个股年度中位','正收益占比':'正收益占比'})
ind=ind.rename(columns={'行业':'行业','样本':'样本','20日均值':'20日均值','20日中位':'20日中位','20日胜率':'20日胜率','20日p10':'20日低分位','20日p90':'20日高分位'})
amtg=amtg.rename(columns={'成交额组':'成交额组','样本':'样本','20日均值':'20日均值','20日中位':'20日中位','20日胜率':'20日胜率','20日p10':'20日低分位','20日p90':'20日高分位'})
age=age.rename(columns={'年龄':'趋势年龄','样本':'样本','20日均值':'20日均值','20日中位':'20日中位','20日胜率':'20日胜率','超额均值':'超额均值','平均最大不利':'平均最大不利','平均最大有利':'平均最大有利'})
epy=epy.rename(columns={'start_year':'起始年份','数量':'数量','中位持续':'中位持续','平均持续':'平均持续','中位涨幅':'中位涨幅','平均涨幅':'平均涨幅','涨幅胜率':'涨幅胜率','中位回撤':'中位回撤','中位波动':'中位波动','中位量比':'中位量比'})
cond=cond.rename(columns={'特征':'特征','期':'期','组':'组','样本':'样本','下限':'下限','上限':'上限','超额均值':'超额均值','超额中位':'超额中位','超额胜率':'超额胜率','绝对收益均值':'绝对收益均值','绝对胜率':'绝对胜率','平均最大不利':'平均最大不利','平均最大有利':'平均最大有利','盈亏比':'盈亏比'})
event['事件']=event['事件'].map(sec)
event=event.rename(columns={'净收益_扣成本':'净收益扣成本'})
exitr=exitr.rename(columns={'期':'期','规则':'规则','样本':'样本','均值':'均值','中位':'中位','胜率':'胜率','盈亏比':'盈亏比','平均持有':'平均持有','平均MAE':'平均最大不利','平均MFE':'平均最大有利'})
rank=rank.rename(columns={'排序':'排序','前N':'前几名','期':'期','样本':'样本','均值':'均值','中位':'中位','胜率':'胜率','盈亏比':'盈亏比'})
trade=trade.rename(columns={'code':'代码','entry_date':'买入日','exit_date':'卖出日','entry':'买入价','exit':'卖出价','ret':'收益率','reason':'原因','hold':'持有天数','shares':'股数','pnl':'盈亏','weight':'仓位','entry_eq':'入场权益','entry_amt':'入场成交额','期':'期'})
trade['买入日']=trade['买入日'].str[:10]; trade['卖出日']=trade['卖出日'].str[:10]
hold=hold.rename(columns={'date':'日期','code':'代码','shares':'股数','price':'价格','value':'市值','weight':'权重','entry':'买入价','ret':'浮动收益','hold':'持有天数'})
hold['日期']=hold['日期'].str[:10]
model=model.rename(columns={'十分位':'十分位','样本':'样本','预测概率':'预测概率','超额均值':'超额均值','绝对收益均值':'绝对收益均值','胜率':'胜率'})
first=first.rename(columns={'期':'期','首周组':'首周组','笔数':'笔数','最终均值':'最终均值','最终中位':'最终中位','最终胜率':'最终胜率','平均最大不利':'平均最大不利','平均最大有利':'平均最大有利'})
post=post.rename(columns={'期':'期','卖后组':'卖出后20日分组','笔数':'笔数','卖后均值':'卖出后均值','卖后胜率':'卖出后胜率'})
for df in [scen,scen2]:
    pass
scen['场景']=scen['场景'].replace({'主方案_cat20':'主方案（灾难止损）'})
scen=scen.rename(columns={'场景':'场景','期':'期'})
scen2['场景']=scen2['场景'].replace({'主方案_cat20':'主方案（灾难止损）','止损_none':'无灾难止损','止损_fixed8':'8%固定止损','止损_fixed12':'12%固定止损','止损_fixed15':'15%灾难止损','四仓_cat20':'四仓版','成本25':'成本0.5%','成本40':'成本0.8%','流动性5千万':'流动性下限5000万','流动性8千万':'流动性下限8000万','流动性10千万':'流动性下限1亿'})
scen2=scen2.rename(columns={'场景':'场景','期':'期'})

# 总览
overview=pd.DataFrame([
 ('一句话','只做“市场宽度>50%、趋势年龄5-30日、低波、无涨停、行业宽度和动量都强”的上涨趋势；持有20个交易日。'),
 ('回测区间','2018-01-01 至 2026-10-13'),
 ('股票池','全部股票 + 102只退市股，共5675只'),
 ('全期累计收益','117.3%'),('全期年化','9.6%'),('全期最大回撤','-13.9%（按最低价 -14.2%）'),('全期夏普','0.93'),
 ('全期交易笔数','419（分段回测合计413，差额来自分段边界）'),('全期胜率','55.6%'),('全期盈亏比','1.86'),('平均仓位','62.7%'),
 ('训练期累计（2018-2021）','30.1%'),('训练期年化','7.1%'),('训练期最大回撤','-13.9%'),
 ('检验期累计（2022-2026）','63.5%'),('检验期年化','11.3%'),('检验期最大回撤','-9.4%'),
 ('基准全期累计（等权）','90.1%'),('基准全期最大回撤','-36.6%'),
 ('年度收益','2018 -8.2%；2019 +7.8%；2020 +20.7%；2021 +8.9%；2022 +6.4%；2023 +5.0%；2024 +12.6%；2025 +8.0%；2026至10月 +20.4%'),
 ('最大单笔亏损','-26.7%（2018年跳空穿过-20%止损）'),
 ('核心风险','小市值/低流动性依赖；退市股只补了102只，仍有幸存者偏差；2022-2026用于最终选择，不是干净样本外。'),
 ('近止损检验','3%以内结构止损盘中胜率仅17%-21%；5%-7%收盘确认组合年化1.6%、最大回撤-21.6%、胜率39.9%；主系统用20日时间止损+-20%灾难止损。'),
 ('个股适用性','招商银行信号胜率73.3%；茅台46.2%；宁德时代34.0%；中际旭创42.4%；寒武纪0个信号。系统不覆盖高波暴涨。'),
 ('不适合','不能接受20日时间止损、不能接受单笔-20%灾难止损、资金容量大于约2000万元、需要T+0或日内交易的人。'),
],columns=['项目','内容'])

rules=pd.DataFrame([
 ('1','市场闸门：全部股票20日均线以上占比 > 50% 才开新仓；否则只平不开。'),
 ('2','趋势结构：收盘价>60日均线，20日均线>60日均线，60日均线高于20日前。'),
 ('3','趋势年龄：5-30个交易日；超过30日不追。'),
 ('4','波动：14日平均真实波幅/收盘价 <= 4.4%（训练期40分位）。'),
 ('5','情绪：过去20日涨停次数=0；不追近期涨停股。'),
 ('6','板块：行业宽度（行业内收盘>20日均线占比）>=60%，且行业20日涨幅>5%。'),
 ('7','位置：距60日高点回撤 > -8%；不买刚崩塌的。'),
 ('8','可买性：次日开盘买入；一字涨停、开盘高开>5%、停牌不买。'),
 ('9','排序：同一天候选按20日成交额从小到大排序，取前6只。'),
 ('10','仓位：单笔风险0.5%；仓位=风险金额/(买入价×2倍波幅)；单票<=20%；组合热预算<=3%；最多6只。'),
 ('11','退出：持有20个交易日收盘卖出；灾难止损-20%（盘中止损，跌停不卖）；不设窄止损。'),
 ('12','不加仓、不摊平、不当天买回；卖出后出现新信号可再买。'),
 ('13','成本：单边0.15%，往返0.3%；一字跌停无法卖出时顺延。'),
 ('14','容量：20日成交额>=3000万元；成交额下限提高到5000万元后收益明显下降，建议自有资金<=2000万元。'),
 ('15','若坚持近止损：用5%-7%结构止损+收盘确认；不要用3%以内；预期胜率42%-49%，收益约减半。'),
],columns=['条款','内容'])

note=pd.DataFrame([
 ('数据','本地DuckDB后复权日线，2016-10-13至2026-10-13；加入中证500历史缓存中102只退市/缺失股票。'),
 ('时点','申万行业分类按公告起始日；指数/宽度只用当日及以前；信号t日收盘，t+1开盘成交。'),
 ('行业覆盖','按日期和股票计算约75.7%；缺失行业不参与板块条件。'),
 ('情绪','只有价量代理（涨停、跳空、换手、量比、上影线、市场情绪分）；没有新闻、股吧、龙虎榜、北向、融资数据。'),
 ('样本外','2017-2021用于选规则，2022-2026用于检验；但宽度闸门、无窄止损、成交额排序也参考了检验期，因此不是严格样本外。'),
 ('幸存者偏差','全部股票面板只含当前仍上市股票；补了102只退市股后，全期年化降至9.6%。剩余小市值退市股仍未补。'),
 ('机器学习','梯度提升树训练期区分度0.617，检验期0.505；复杂模型没有增量，最终用规则。'),
 ('回撤口径','旧口径-9.2%含前视/停牌乐观处理；修正后收盘回撤-13.9%、最低价回撤-14.2%；审计见回撤审计表。'),
 ('置换检验','500次随机抽取同数量趋势日，训练/检验均为0/500超过实际；p<0.002。'),
 ('龙虎榜','本会话外部接口被数据协议拦截，未纳入；从重叠关系看更像排除项，不是买入信号。'),
 ('交易笔数口径','全期420笔；训练/检验分段回测合计414笔，差额来自分段边界。'),
 ('复现','依次运行 10b、21、22、23、24、25、26、27、30b、40b、50、60、70、72。'),
],columns=['项目','说明'])

with pd.ExcelWriter(XLS,engine='openpyxl') as w:
    overview.to_excel(w,sheet_name='总览',index=False)
    rules.to_excel(w,sheet_name='规则清单',index=False)
    dist.to_excel(w,sheet_name='全部股票收益分布',index=False)
    ann.to_excel(w,sheet_name='年度分布',index=False)
    ind.to_excel(w,sheet_name='行业分组',index=False)
    amtg.to_excel(w,sheet_name='成交额分组',index=False)
    age.to_excel(w,sheet_name='趋势年龄',index=False)
    epy.to_excel(w,sheet_name='趋势阶段年度',index=False)
    write(w,'条件扫描',cond.copy())
    write(w,'事件类型',event.copy())
    write(w,'出场规则',exitr.copy())
    write(w,'信号排序',rank.copy())
    write(w,'组合情景',scen2.copy())
    model.to_excel(w,sheet_name='模型十分位',index=False)
    trade.to_excel(w,sheet_name='逐笔交易',index=False)
    hold.to_excel(w,sheet_name='每日持仓',index=False)
    first.to_excel(w,sheet_name='首周复盘',index=False)
    post.to_excel(w,sheet_name='卖出后复盘',index=False)
    stock.to_excel(w,sheet_name='个股案例',index=False)
    stockbig.to_excel(w,sheet_name='个股大涨幅起点',index=False)
    stopdist.to_excel(w,sheet_name='结构止损距离',index=False)
    closestop['方式']=closestop['方式'].replace({'fixed':'固定20日','closestop':'收盘确认止损'})
    closestop.to_excel(w,sheet_name='收盘止损与短持有',index=False)
    selected=selected.rename(columns={'MAE':'平均最大不利','MFE':'平均最大有利'})
    selected.to_excel(w,sheet_name='精选形态',index=False)
    perm=perm.rename(columns={'置换p值':'置换显著性'})
    perm.to_excel(w,sheet_name='置换检验',index=False)
    sent.to_excel(w,sheet_name='个股情绪合成',index=False)
    selport=selport.rename(columns={'start':'开始','end':'结束','total':'累计收益','cagr':'年化收益','maxdd':'最大回撤','sharpe':'夏普比率','trades':'交易笔数','win':'胜率','pf':'盈亏比','avg_hold':'平均持有','exposure':'平均仓位'})
    selport.to_excel(w,sheet_name='精选版组合',index=False)
    dd.to_excel(w,sheet_name='回撤审计',index=False)
    note.to_excel(w,sheet_name='说明',index=False)
print('saved',XLS)
# 检查表头拉丁
x=pd.ExcelFile(XLS)
bad=[]
for sh in x.sheet_names:
    d=pd.read_excel(XLS,sheet_name=sh,nrows=2)
    for c in d.columns:
        if any(ch.isascii() and ch.isalpha() for ch in str(c)): bad.append((sh,c))
print('拉丁表头',bad)
print('sheets',x.sheet_names)
