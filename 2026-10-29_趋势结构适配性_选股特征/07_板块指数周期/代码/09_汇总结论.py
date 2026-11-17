"""只在各子任务完成、整改重跑后调用；读取实际结果生成结论和验收映射。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def records(x):
    return json.loads(x.to_json(orient='records',force_ascii=False))


def main():
    timeline=json.loads((ROOT/'证据/02_时序与行业轮动核验.json').read_text())
    assert timeline['完成']
    for path,value in timeline['输入哈希'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,'时序子任务源数据版本不一致'
    timing=pd.read_csv(ROOT/'结果/02_时序趋势网格.csv')
    rotation=pd.read_csv(ROOT/'结果/02_行业轮动网格.csv')
    cross=pd.read_csv(ROOT/'结果/04_跨层事件.csv')
    sorting=pd.read_csv(ROOT/'结果/04_双重排序.csv')
    scale=pd.read_csv(ROOT/'结果/层级尺度地图.csv')
    combo=pd.read_csv(ROOT/'结果/指数大小周期网格.csv')
    ci=json.loads((ROOT/'结果/04_审计与分段.json').read_text())
    for path,info in ci['源文件'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==info['SHA256'],'跨层源数据版本不一致'
    timing_core=timing[timing['单边成本(bp)'].eq(10)&timing['最长连续段'].eq(True)]
    qcol='超同暴露_块60_全块长统一BH_FDR_q'
    alpha='超同暴露_年化对数差(%)'
    assert qcol in timing_core and alpha in timing_core, '实际时序字段缺失，不能补默认0'
    n_q=int((timing_core[qcol].le(.1)&timing_core[alpha].gt(0)).sum())
    pos=int(timing_core[alpha].gt(0).sum());total=len(timing_core)
    principal=timing_core[timing_core.资产类别.eq('宽基指数')&timing_core.规则.eq('均线')&timing_core.回看.eq(120)]
    cols=['资产名称','有效段首日','有效段末日','净年化(%)','最大回撤(%)','平均开盘仓位(%)',
          '买入持有净年化(%)','买入持有最大回撤(%)',alpha,qcol]
    cols=[c for c in cols if c in principal]
    principal[cols].to_csv(ROOT/'结果/预登记指数120日均线案例.csv',index=False,encoding='utf-8-sig')
    rot_core=rotation[rotation['单边成本(bp)'].eq(10)&rotation['最长连续段'].eq(True)]
    ralpha='超行业等权_年化对数差(%)';rq='超行业等权_块60_全块长统一BH_FDR_q'
    assert rq in rot_core and ralpha in rot_core, '实际轮动字段缺失，不能补默认0'
    rpass=int((rot_core[rq].le(.1)&rot_core[ralpha].gt(0)).sum())
    rmed=rot_core.groupby(['数据组','规则'])[ralpha].median().round(3).to_dict()
    cost5=combo[combo.单边费用.eq(.0005)]
    cm=cost5.groupby('组合动作')['超大周期同暴露年化对数差'].median()
    cpass=int(combo.统一FDR_q.le(.1).sum())
    ev=cross[cross.时段.eq('主窗2015-2020')&cross.半区.eq('全部')&cross.大周期状态.eq('红')
      &cross.事件.isin(['20日新高','55日新高'])&cross.持有日.eq(5)&cross.指标.eq('相对留一行业')]
    pair=sorting[sorting.研究.str.startswith('强行业弱股')&sorting.时段.eq('主窗2015-2020')&sorting.半区.eq('全部')
      &sorting.流动性.eq('全部')&sorting.指标.eq('相对留一行业')]
    sector=timing_core[timing_core['数据组'].str.contains('完整行业合成')]
    assert len(sector)==280, '完整行业合成10bp时序应有280格，禁止静默丢失'
    s_q=int((sector[qcol].le(.1)&sector[alpha].gt(0)).sum())
    current=scale[scale.窗口.eq('行业主窗2015-2020')]
    vals={(x.层级,int(x.聚合日数)):x.方差比中位数 for _,x in current.iterrows()}
    d20=vals['个股行业残差',20];sec20=vals['行业等权合成',20];idx20=vals['宽基指数',20]
    statements=[
      f'层级确实不同：2015—2020个股行业残差20日方差比中位{d20:.3f}，行业等权{sec20:.3f}，宽基{idx20:.3f}。这只是路径诊断，不是策略收益。',
      f'指数与行业时序主成本10bp、最长完整段共{total}格，其中{pos}格超同暴露为正；统一FDR(q≤0.10)通过{n_q}格。价格代理、未含股息/真实ETF追踪成本。',
      f'行业轮动最长段主成本格的统一FDR通过{rpass}格；官方缓存缺口与完整行业等权合成分别报告，不能跨缺口拼接净值。',
      f'大小周期252格：5bp下共振追随超大周期同暴露中位{cm.get("共振追随",np.nan):+.2%}，大多小弱{cm.get("大多小弱",np.nan):+.2%}；统一FDR通过{cpass}格。',
      f'行业红状态里的20/55日新高，5日扣费后相对留一行业均值范围{ev.均值.min():+.2%}至{ev.均值.max():+.2%}；大周期红不能替个股买点创造边际。',
      '强行业里的弱股相对强股有提示，但只有71个有效月末信号日；分段与流动性控制不构成确认，不能把相对差当买入策略。'
    ]
    table=[{'问题':'板块趋势','当前答案':'行业等权中短期路径有延续倾向；行业时序及买强/买弱轮动需用完整资金和同集合基准判定，不能把行业涨当个股alpha。'},
      {'问题':'指数趋势','当前答案':f'可作为风险开关的候选；本轮时序主成本{total}格中FDR通过{n_q}格，不能给时间样本外声明。'},
      {'问题':'大小周期','当前答案':f'同向组合整体优于大多小弱，但252格FDR通过{cpass}格；用交易日回看尺度，不等于已完成月周K线信号。'},
      {'问题':'强板块选什么股','当前答案':'行业红状态并不救新高买点；行业内弱股相对更好只是线索，71月末样本日不足。'},
      {'问题':'证据边界','当前答案':'2010—2020描述性研究；2021以后未评估；价格指数和合成基准非ETF实盘收益。'}]
    limits=[
      '旧轮oracle、空仓超额和A/B验证口径有缺陷，相关强结论已撤回；本轮独立重算，不沿用旧上限数字。',
      '本轮同暴露基准显式计算隔夜和日内仓位漂移及交易再平衡费；置信区间/p改循环移动块，以免端点权重不足。',
      '行业分类2010—2013股票日覆盖约68%—72%，2015后约99.8%以上；早期只有17行业，主行业窗口2015—2020，不能反向回填新行业。',
      '官方行业价格缓存缺若干交易日，严格分段；段末退出只是数据边界截断，不是事前知道缺价的交易策略。完整合成行业作补充但不是可投资产品。',
      '股票日缺失先补日历再差分，缺日后的跨日收益不能充当单日收益；行业成分数和全部均值使用同一有效样本，留一后至少10只。',
      '指数价格未含股息、ETF费率及追踪差；行业与全A等权底层日再平衡费用未计入，不能宣称可部署收益。',
      '所有网格是样本内描述，不从最好的一格倒推未来；多重性校正只覆盖各明确统计族，不能称全球所有工作区试验的校正。',
      '月末双重排序最长71个有效信号日，预设底线100日；统计不足不等于被证伪。相对强股更好不等于绝对收益为正。',
      '这里的大小周期是10—250交易日回看，不是另一次月线/周线已完成K线或分钟级执行研究。'
    ]
    explanations={
      '板块趋势：先分行业共同运动和个股相对运动':f'行业等权20/60日方差比中位分别{sec20:.3f}/{vals["行业等权合成",60]:.3f}，个股行业残差约{d20:.3f}/{vals["个股行业残差",60]:.3f}。同一段里行业可以共同走趋势、个股之间又相对回复，因此不能用个股失败推断行业失败。官方价格指数与等权成分序列权重不同，也不应择优混用。',
      '指数趋势：收益、降暴露与风险保险分开':f'按次开交易/完整资金曲线，在所有成本和参数下并列净CAGR、回撤、仓位及两种基准差。主成本最长段FDR通过{n_q}格，仍须看分时代和邻域；即使降回撤稳定，也不能自动说有alpha。完整行业合成主成本{len(sector)}格，FDR通过{s_q}格。',
      '大小周期：只看方向还不够':f'同指数大均线120/250×小均线10/20/60，两种动作各三成本，共252格。同向持有的超大周期同暴露中位为{cm.get("共振追随",np.nan):+.2%}，大多小弱为{cm.get("大多小弱",np.nan):+.2%}。统一FDR通过{cpass}格，因此不要把共振当买点保证，也不要把小周期弱机械当便宜。',
      '跨层选股：行业上涨不保证追新高有边际':f'在行业前日上行状态下，20/55日新高5日净留一超额范围{ev.均值.min():+.2%}到{ev.均值.max():+.2%}，统一校正后仍为负的格须明确拒绝。强行业弱股减强股约{pair.均值.min():+.2%}到{pair.均值.max():+.2%}，仅71月末日；应继续做未来冻结验证，不能把这变成今日买入列表。'
    }
    result={'研究范围':'板块趋势/指数趋势/大小周期/跨层联动','总判断':'趋势是层级与尺度相关的现象；更高层共同运动不等于个股买点有效，更不等于所有周期都能赚钱。',
      '核心说明':statements,'结论表':table,'限制':limits,'研究解释':explanations,
      'FDR阳性计数口径':'只计净超同暴露或行业基准为正且q≤0.10的格；负向显著不算策略通过',
      '对立假说裁决':{'H1_趋势更偏高层级':'中期路径有描述性支持，不能泛称中长期所有尺度有效',
        'H2_全部聚合无可交易趋势':'当前网格未建立稳定盈利证据，不等于证明一切趋势策略不存在',
        'H3_主要提供保险':'须逐指数看回撤与同暴露差，风险开关不能自动冒称alpha',
        'H4_强行业弱股互补':'全窗相对差有线索，只有71月末日，分段与流动性未确认'},
      '时序参数计数':timeline['试验数'],'时序主成本最长段数':total,'时序主成本FDR通过':n_q,
      '完整行业时序FDR通过':s_q,'行业轮动FDR通过':rpass,'行业轮动分组中位数':{str(k):v for k,v in rmed.items()},
      '指数120日案例':records(principal[cols]),'红行业新高事件':records(ev),'强行业弱股配对':records(pair),
      '跨层检验数':ci['检验'],'大小周期配置数':len(combo),'大小周期FDR通过':cpass,
      '前轮更正':'证据/补充登记与前轮更正.json','独立审查整改':'证据/数据整改实查.json',
      '验收映射':{'板块趋势':['层级尺度地图.csv','02_时序趋势网格.csv','02_行业轮动网格.csv'],
                  '指数趋势':['02_时序趋势网格.csv','02_时序趋势主成本资金曲线.csv'],
                  '大小周期':['层级尺度地图.csv','指数大小周期网格.csv'],
                  '跨层大小周期':['04_跨层事件.csv','04_双重排序.csv','跨层事件块长与成本敏感性.csv'],
                  'PIT与数据质量':['数据核验.json','数据整改实查.json'],
                  '资金与成本检查':['02_时序与行业轮动核验.json','指数大小周期自检.json'],
                  '统计及完整试验数':['尺度图公式自检.json','04_审计与分段.json']}}
    (ROOT/'证据/本轮结论与验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=str))
    print('\n'.join(statements))

if __name__=='__main__':main()
