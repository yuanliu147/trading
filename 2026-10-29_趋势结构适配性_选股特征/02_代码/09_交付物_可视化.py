#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化报告（单文件 HTML，无外部依赖）"""
from __future__ import annotations
import pandas as pd, numpy as np, html
from pathlib import Path

本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
结果 = 本项目 / "03_结果"
交付 = 本项目 / "06_交付"; 交付.mkdir(parents=True, exist_ok=True)

等权 = pd.read_csv(结果/"条件分层_全表.csv")
沪深 = pd.read_csv(结果/"条件分层_全表_沪深300.csv")
平台 = pd.read_csv(结果/"容量与稳健性_参数平台.csv")
去尾 = pd.read_csv(结果/"容量与稳健性_去尾.csv")
容量 = pd.read_csv(结果/"容量与稳健性_容量.csv")
画像 = pd.read_csv(结果/"容量与稳健性_子集画像.csv")
排除 = pd.read_csv(结果/"容量与稳健性_排除清单.csv")
上限 = pd.read_csv(结果/"市场状态_全知上限.csv")
持续 = pd.read_csv(结果/"市场状态_状态持续性.csv")

W = 760
def 条形(项: list[tuple[str, float]], 单位="%", 放大=100, 高=26, 正色="#2f855a", 负色="#c53030"):
    """水平条形图：零轴居中，正右负左。"""
    vals = [v*放大 for _, v in 项]
    m = max(1e-9, max(abs(v) for v in vals))
    零 = W/2
    刻度 = (W/2 - 90)/m
    out = [f'<svg viewBox="0 0 {W} {len(项)*高+16}" class="chart">']
    out.append(f'<line x1="{零}" y1="6" x2="{零}" y2="{len(项)*高+6}" stroke="#cbd5e0" stroke-width="1"/>')
    for i,((名,_),v) in enumerate(zip(项, vals)):
        y = i*高+8; w = abs(v)*刻度
        x = 零 if v>=0 else 零-w
        c = 正色 if v>=0 else 负色
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w,0.6):.1f}" height="{高-10}" fill="{c}" rx="2"/>')
        tx = 零+w+6 if v>=0 else 零-w-6
        anc = "start" if v>=0 else "end"
        out.append(f'<text x="{tx:.1f}" y="{y+高-15}" class="val" text-anchor="{anc}">{v:+.2f}{单位}</text>')
        out.append(f'<text x="4" y="{y+高-15}" class="lab">{html.escape(名)}</text>')
    out.append('</svg>')
    return "".join(out)

def 折线(系列: dict[str, list[tuple[str,float]]], 放大=100, 高=230, 单位="%"):
    xs = [k for k,_ in next(iter(系列.values()))]
    allv = [v*放大 for s in 系列.values() for _,v in s]
    lo, hi = min(allv+[0]), max(allv+[0])
    pad = (hi-lo)*0.15 or 1
    lo -= pad; hi += pad
    L, R, T, B = 70, 20, 14, 30
    pw, ph = W-L-R, 高-T-B
    def px(i): return L + pw*i/max(len(xs)-1,1)
    def py(v): return T + ph*(hi-v)/(hi-lo)
    out=[f'<svg viewBox="0 0 {W} {高}" class="chart">']
    out.append(f'<line x1="{L}" y1="{py(0):.1f}" x2="{W-R}" y2="{py(0):.1f}" stroke="#a0aec0" stroke-dasharray="3 3"/>')
    for v in np.linspace(lo,hi,4):
        out.append(f'<text x="{L-8}" y="{py(v)+4:.1f}" class="ax" text-anchor="end">{v:+.2f}</text>')
    for i,x in enumerate(xs):
        out.append(f'<text x="{px(i):.1f}" y="{高-10}" class="ax" text-anchor="middle">{html.escape(str(x))}</text>')
    cols=["#2b6cb0","#c53030","#2f855a","#b7791f","#6b46c1","#319795"]
    for j,(名,s) in enumerate(系列.items()):
        pts=" ".join(f"{px(i):.1f},{py(v*放大):.1f}" for i,(_,v) in enumerate(s))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{cols[j%len(cols)]}" stroke-width="2.2"/>')
        for i,(_,v) in enumerate(s):
            out.append(f'<circle cx="{px(i):.1f}" cy="{py(v*放大):.1f}" r="3" fill="{cols[j%len(cols)]}"/>')
    out.append('</svg>')
    图例 = " ".join(f'<span class="lg"><i style="background:{cols[j%len(cols)]}"></i>{html.escape(n)}</span>'
                    for j,n in enumerate(系列))
    return "".join(out)+f'<div class="legend">{图例}</div>'

# ---- 数据准备
ok = 等权[等权["样本判定"].isna()]
A族系列 = {}
for f,nm in [("ER_60","效率比60日"),("ER_120","效率比120日"),("VR_20","方差比20日"),
             ("R2_60","趋势拟合R²"),("AC1_120","自相关120日")]:
    s = ok[(ok["特征"]==f)&(ok["信号代码"]=="S1_BREAK_N55")].sort_values("分位")
    if len(s)==5: A族系列[nm]=[(f"Q{int(float(r['分位']))}", r["超额均值"]) for _,r in s.iterrows()]

B族系列={}
for f,nm in [("ADV20","成交额均值20日"),("turn20","换手率20日"),("AMIHUD","非流动性"),("float_mktcap","流通市值")]:
    s = ok[(ok["特征"]==f)&(ok["信号代码"]=="S1_BREAK_N55")].sort_values("分位")
    if len(s)==5: B族系列[nm]=[(f"Q{int(float(r['分位']))}", r["超额均值"]) for _,r in s.iterrows()]

平台5 = 平台[(平台["信号"]=="R1_NEWLOW_N55")&(平台["滑点"]=="5bp")]
平台系列={}
for 子,g in 平台5.groupby("子集"):
    g=g.sort_values("持有期")
    平台系列[子]=[(f"{int(h)}日",v) for h,v in zip(g["持有期"],g["超额均值"])]

去尾系列={}
for _,r in 去尾.iterrows():
    去尾系列[r["子集"]]=[("原始",r["原始均值"]),("去前1%",r["去前1%均值"]),("去前5%",r["去前5%均值"])]

信号排序 = ok.groupby("信号代码").apply(
    lambda g: np.average(g["超额均值"],weights=g["事件数"]), include_groups=False).sort_values()
信号名={"S1_BREAK_N20":"趋势突破20日","S1_BREAK_N55":"趋势突破55日","S1_BREAK_N120":"趋势突破120日",
       "S2_VCB_N20_P20":"收缩突破20日低波20%","S2_VCB_N20_P30":"收缩突破20日低波30%",
       "S2_VCB_N55_P20":"收缩突破55日低波20%","S2_VCB_N55_P30":"收缩突破55日低波30%",
       "S3_PULL_K05":"趋势回调0.5ATR","S3_PULL_K10":"趋势回调1.0ATR",
       "S4_BRKPULL_W5":"突破回踩5日","S4_BRKPULL_W10":"突破回踩10日",
       "R1_NEWLOW_N20":"反向·20日新低","R1_NEWLOW_N55":"反向·55日新低",
       "R1_NEWLOW_N120":"反向·120日新低","R3_RALLY_K10":"反向·跌势反弹转弱"}
信号条=[(信号名.get(k,k),v) for k,v in 信号排序.items()]

状态名={"R1_指数均线":"R1 指数均线","R2_市场宽度":"R2 市场宽度","R3_波动率":"R3 波动率",
       "R4_离散度":"R4 横截面离散度","R5_风格差":"R5 风格差"}
上限条=[(状态名.get(r["状态定义"],r["状态定义"]),r["用户对角组合超额"]) for _,r in 上限.iterrows()]
持续系列={}
for 定义,g in 持续.groupby("状态定义"):
    g=g.sort_values("时代")
    持续系列[状态名.get(定义,定义)]=[(r["时代"],r["平均持续天数"]/100) for _,r in g.iterrows()]

排除系列={}
for 信号,g in 排除.groupby("信号"):
    行=[("不剔除",g["全样本均值"].iloc[0])]+[(f"剔{r['剔除ADV分位']}",r["剔除后均值"]) for _,r in g.iterrows()]
    排除系列[信号名.get(信号,信号)]=行

容量sel=容量[(容量["仓位数"]==10)&(容量["ADV参与率上限"]==0.05)]
容量系列={}
for 子,g in 容量sel.groupby("子集"):
    g=g.sort_values("资金")
    容量系列[子]=[({10000000:"1000万",50000000:"5000万",100000000:"1亿"}[int(z)],v)
                 for z,v in zip(g["资金"],g["可部署率"])]

HTML=f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>趋势结构策略的股票适配性与选股特征</title><style>
:root{{--ink:#1a202c;--mute:#718096;--line:#e2e8f0;--bg:#f7fafc;--neg:#c53030;--pos:#2f855a}}
*{{box-sizing:border-box}}
body{{margin:0;font:15px/1.75 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink);background:var(--bg)}}
.wrap{{max-width:900px;margin:0 auto;padding:36px 22px 80px}}
h1{{font-size:27px;margin:0 0 6px;letter-spacing:-.3px}}
.sub{{color:var(--mute);font-size:13.5px;margin-bottom:26px}}
h2{{font-size:19px;margin:38px 0 4px;padding-top:20px;border-top:2px solid var(--ink)}}
h3{{font-size:15.5px;margin:24px 0 8px;color:#2d3748}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0}}
.verdict{{background:#fff5f5;border:1px solid #fc8181;border-left:5px solid var(--neg)}}
.note{{background:#fffaf0;border:1px solid #f6ad55;border-left:5px solid #dd6b20;font-size:13.5px}}
.ok{{background:#f0fff4;border:1px solid #9ae6b4;border-left:5px solid var(--pos)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}}
th,td{{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#edf2f7;font-weight:600;font-size:12.5px}}
tr:hover td{{background:#f7fafc}}
.neg{{color:var(--neg);font-variant-numeric:tabular-nums}}
.pos{{color:var(--pos);font-variant-numeric:tabular-nums}}
.chart{{width:100%;height:auto;display:block;margin:6px 0}}
.lab{{font-size:11.5px;fill:#4a5568}} .val{{font-size:11px;fill:#2d3748;font-weight:600}}
.ax{{font-size:10.5px;fill:#718096}}
.legend{{font-size:12px;color:var(--mute);margin-top:4px}}
.lg{{margin-right:14px;white-space:nowrap}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.kpi div{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:12px 14px}}
.kpi b{{display:block;font-size:22px;letter-spacing:-.5px}}
.kpi span{{font-size:12px;color:var(--mute)}}
code{{background:#edf2f7;padding:1px 5px;border-radius:3px;font-size:12.5px}}
.foot{{color:var(--mute);font-size:12.5px;margin-top:30px;border-top:1px solid var(--line);padding-top:14px}}
</style></head><body><div class="wrap">

<h1>趋势结构策略的股票适配性与选股特征</h1>
<div class="sub">研究期 2010-01-01 至 2020-12-31 ｜ 7,658,304 行 / 4,205 只 / 2,674 个交易日 ｜
33 个事前特征 × 15 个信号配置 = 2,498 个条件格 ｜ 2021-2025 全程封存未触碰 ｜
证据等级上限：同期横截面留出（不是时间样本外）</div>

<div class="card verdict"><b>结论</b><br>
在 2,498 个条件子集里，<b>没有任何子集</b>能让趋势/结构类策略在扣费后跑赢等权市场
（事件层 95% 下界 &gt; 0 的格数 = <b>0 / 2,449</b>）。母项目从未测过的"趋势可交易性"特征族
（效率比 / 方差比 / 自相关 / 趋势 R²）<b>完全无效</b>。唯一接近正边际的是<b>反向信号</b>，
但它靠右尾 1% 事件支撑、参数平台不成立。市场状态切换的<b>全知上限是 0</b>。</div>

<div class="kpi">
<div><b>0 / 2,449</b><span>等权基准下通过全部门槛的格数</span></div>
<div><b class="neg">−2.23%</b><span>趋势突破55日 20日净超额（n=436,345）</span></div>
<div><b class="neg">−0.51%</b><span>反向·55日新低（同口径，更优）</span></div>
<div><b>0.7919</b><span>全市场方差比VR_20中位数（&lt;1 = 均值回复）</span></div>
<div><b>+0.0132%</b><span>市场状态切换全知上限（洗牌95分位 +0.0212%）</span></div>
<div><b>19.75×</b><span>多重检验超出预注册预算倍数</span></div>
</div>

<h2>一、15 个信号配置的全样本净超额：趋势类全负，反向类最优</h2>
<div class="card">{条形(信号条)}
<div class="legend">口径：持有 20 个交易日，买入 T+1 开盘，扣往返摩擦 0.254%（含 5bp 单边滑点），
相对等权市场超额。<b>4 个反向信号占据最优的 4 位</b>，11 个趋势类配置全部更差。</div></div>

<h2>二、Q2 的答案：哪些股适合？——没有</h2>
<h3>2.1 母项目从未测过的"趋势可交易性"特征，是平的</h3>
<div class="card">{折线(A族系列)}
<div class="legend">横轴为按日横截面五分位（Q1 最低 → Q5 最高），纵轴为趋势突破55日的 20 日净超额(%)。
效率比（路径走得有多直）从 Q1 到 Q5 几乎不动；<b>方差比与自相关的 Q5（最"动量型"的股票）反而最差</b>，
与直觉相反。</div></div>

<h3>2.2 流动性是唯一单调轴，但方向是"越不流动越不差"，且全程为负</h3>
<div class="card">{折线(B族系列)}
<div class="legend">成交额 Q1（最不流动）−0.82% → Q5 −3.09%，跨度 2.27pp，单调。
但<b>连最不流动分位都是负的</b>，所以不存在"换个流动性区间就能做"的子集。</div></div>

<div class="card note"><b>基准口径决定答案，必须并列读</b><br>
同一格「成交额最低分位 × 趋势突破」：对<b>沪深300</b> 是 <b class="pos">+1.68%</b>，
对<b>等权市场</b> 是 <b class="neg">−0.82%</b>。那 2.5pp 差额就是 2010-2020 的小盘溢价。<br>
沪深300 口径下通过门槛1 的 113 格里，流动性类特征的通过分位<b>全部集中在 Q1-Q2</b>、
非流动性指标集中在 Q4-Q5 —— 说明那是<b>风格 beta，不是选股能力</b>。
等权市场等于"随机选股"的结果，是选股问题的正确零点。</div>

<h2>三、唯一的候选，以及它怎么倒的</h2>
<div class="sub">候选 = 买入 55 日新低 × 高波动分位（5 格是波动的 5 种测法，同一个发现）</div>

<h3>3.1 门槛7 参数平台：正值只在 20-40 日出现，60 日全部转负</h3>
<div class="card">{折线(平台系列)}
<div class="legend">这是<b>窄尖峰，不是平台</b>。预注册门槛7 要求参数邻域内符号稳定，不过。</div></div>

<h3>3.2 去尾稳健性：去掉右尾 1% 即归零，去掉 5% 明确为负</h3>
<div class="card">{折线(去尾系列)}
<div class="legend">正均值<b>全部来自超额最大的前 1%-5% 事件</b>。10 仓组合抓不到这种右尾。</div></div>

<h3>3.3 容量：1,000 万尺度可行，1 亿塌陷</h3>
<div class="card">{折线(容量系列, 放大=100, 单位="%")}
<div class="legend">纵轴为可部署率(%)，10 仓 / 成交额参与率上限 5%。
意外发现：候选子集的<b>成交额中位 9,933 万，与对照组 10,010 万几乎相同</b>，
落在最低成交额分位的仅 5.5%（对照 16.7%）——<b>它不是微盘流动性尾巴</b>，所以容量幻觉假说不解释这个发现。</div></div>

<h2>四、逆向检验：排除清单有真实方向价值，但只能把亏损减半</h2>
<div class="card">{折线(排除系列)}
<div class="legend">剔除成交额最高的分位后，趋势信号<b>单调改善</b>：趋势突破55日从 −2.23% 改善到 −1.25%（+0.99pp），
收缩突破从 −1.19% 到 −0.50%（+0.69pp）。<b>但剔除后 95% 下界仍全部为负</b>，
所以"不要在高成交额的大票上做趋势突破"是有方向性的规则，不是能盈利的策略。</div></div>

<h2>五、Q4 市场状态切换：先算上限，上限就是 0</h2>
<div class="card">
<table><tr><th>状态定义</th><th>L1 全知上限<br>(不可实现)</th><th>最好单一动作</th>
<th>洗牌95分位</th><th>用户对角组合</th><th>L2 实时</th></tr>
{"".join(f'<tr><td>{状态名.get(r["状态定义"],r["状态定义"])}</td>'
         f'<td>{r["L1_ORACLE_上限_不可实现"]*100:+.4f}%</td>'
         f'<td>{r["最好单一动作"].replace("A4_","")} {r["最好单一动作超额"]*100:+.2f}%</td>'
         f'<td>{r["洗牌L1_95分位"]*100:+.4f}%</td>'
         f'<td class="neg">{r["用户对角组合超额"]*100:+.4f}%</td>'
         f'<td class="neg">{r["L2_实时可实现"]*100:+.4f}%</td></tr>'
         for _,r in 上限.iterrows())}
</table>
<div class="legend">三个动作的全样本加权超额：趋势跟随 <span class="neg">−2.2338%</span>、
波段 <span class="neg">−1.4500%</span>、反弹 <span class="neg">−0.5096%</span>、空仓 0。
因为都是负的，<b>全知最优在每个状态下都选空仓</b>，所以切换上限 = 0。
R1 的 +0.0132% 还<b>低于洗牌对照的 95 分位 +0.0212%</b>，连全知上限都打不过随机对齐。</div>
</div>
<div class="card">{条形(上限条)}
<div class="legend">你那套对角映射（牛→跟趋势、震荡→玩波段、熊→抓反弹）在 5 套状态定义下<b>全部为负</b>，
且全部劣于"一直空仓"。<b>失败原因不是"认不出状态"，是"认出来也没用"。</b></div></div>

<h3>5.1 "量化参与使状态难识别"——2010-2020 数据不支持</h3>
<div class="card">{折线(持续系列, 放大=100, 单位="天")}
<div class="legend">纵轴为状态平均持续天数。若该命题成立，应逐代下降。实测 <b>5 套定义里 4 套上升或持平</b>，
指数均线状态从 22.8 天拉长到 42.9 天。<br>
<b>边界声明</b>：中国量化规模的大幅扩张主要发生在 2021 年之后，而 2021-2025 在本研究中封存未看，
<b>所以这个结论不能外推到最近 5 年</b>。你的直觉可能恰好在封存段才成立。</div></div>

<h2>六、2,498 格的失败解剖</h2>
<div class="card"><table><tr><th>分类</th><th>格数</th><th>含义</th></tr>
{"".join(f'<tr><td>{k}</td><td>{v:,}</td><td style="text-align:left">{d}</td></tr>' for k,v,d in [
 ("F1_毛边际为零或负", int((等权["失败解剖"]=="F1_毛边际为零或负").sum()), "扣费前就没有正边际"),
 ("F7_样本不足", int((等权["失败解剖"]=="F7_样本不足").sum()), "格内少于200事件或少于100信号日"),
 ("F5_符号不稳", int((等权["失败解剖"]=="F5_符号不稳").sum()), "跨时代或跨半区符号翻转"),
 ("F4_替身可解释", int((等权["失败解剖"]=="F4_替身可解释").sum()), "超额为正但配对增量为零，是子集beta不是信号"),
 ("F5_符号不稳_留出未确认", int((等权["失败解剖"]=="F5_符号不稳_留出未确认").sum()), "检验半区未确认"),
])}</table></div>

<h2>七、必须一起读的方法局限</h2>
<div class="card note">
<b>1. 横截面留出不是时间样本外。</b>两半区等权日收益相关系数 <b>0.9982</b>——它们是同一个市场。
检验半区能证明"不是个股层面的运气"（两半区股票集完全不重叠：1,562 只 vs 1,583 只），
<b>不能</b>证明"不是风格/时代 artifact"。<br>
<b>2. 多重性超预算 19.75×</b>：实际 2,449 检验 vs 预注册 124。方向有利于负结论（检验越多越易出假阳性，而结果是 0 格通过）。<br>
<b>3. 配对增量 Δ 含入场价位混淆</b>：创新低天然买在月内低位，Δ 高达 +5~6.7% 有相当部分是算术优势。
故预注册规定 Δ 必须与事件层超额同向才能下结论。<br>
<b>4. 本研究自己犯过 4 个错并已修</b>：基准单位错（百分数当小数）、分位公式 off-by-one（Q5 每天只剩 1 只股）、
替身缺配对 ID、内存设计导致静默 OOM。前两个若未发现会直接伪造结论。详见审计文档。<br>
<b>5. 账户层回测未做</b>：预注册规定仅在通过门槛后才做；0 格通过，故不做，避免为负结果再造一批配置。
</div>

<div class="card ty ok"><b>与母项目的一处意外对账</b><br>
本研究 <code>趋势回调</code> 的毛超额（沪深300 基准）= <b>+0.0763%</b>，
母项目 TRENDCALLBACK L5 公布 <b>+0.077%</b>（t=0.78, n=67,118），<b>小数点后两位相符</b>。
这反推出母项目那个数是<b>毛口径 + 沪深300 类基准</b>；同一信号族换等权基准并扣费后是 <b class="neg">−1.45%</b>,
解释了母项目"事件层接近零、账户层深度为负"的落差。<br>
<b>保留</b>：两者样本差 8.3 倍（552,922 vs 67,118）、定义不同，<b>相符可能是巧合</b>，不能据此宣称复现了母项目实现。</div>

<div class="foot">
本页为研究结论可视化，<b>不构成交易建议</b>，不含任何当前买卖点、目标价或仓位建议。<br>
L1 全知上限使用未来信息，是刻意设置的可行性上限，<b>永远不得当策略绩效引用</b>。<br>
不得宣称样本外验证通过（2021-2025 未触碰）。不得外推到北交所、注册制全面推行后或含做空/杠杆的场景。<br>
数据表：<code>06_交付/趋势结构适配性_研究数据表.xlsx</code>（15 个工作表，中文表头）+ <code>06_交付/CSV副本/</code>
</div>
</div></body></html>"""

p = 交付/"趋势结构适配性_可视化报告.html"
p.write_text(HTML, encoding="utf-8")
print(f"HTML: {p}  ({p.stat().st_size/1e3:.0f} KB)")
