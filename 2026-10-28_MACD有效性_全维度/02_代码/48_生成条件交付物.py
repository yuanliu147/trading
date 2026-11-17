#!/usr/bin/env python3
"""48 生成条件有效性 HTML / Excel，并在主 HTML 中加入口"""
import base64
import os

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"
RES = os.path.join(ROOT, "03_结果")
FIG = os.path.join(ROOT, "04_图表")


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def img64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def fig_block(name, cap):
    return f'<figure><img src="data:image/png;base64,{img64(name)}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


def show(df, cols, pct=(), integer=(), tcol=(), qcol=()):
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in pct:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—")
    for c in integer:
        if c in d:
            if c == "年份":
                d[c] = d[c].map(lambda v: f"{int(v)}" if pd.notna(v) else "—")
            else:
                d[c] = d[c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    for c in tcol:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
    for c in qcol:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    d = d.astype(object).where(pd.notna(d), "—")
    return d


def table(df, cls="data"):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        body.append("<tr>" + "".join(f"<td>{v}</td>" for v in r.values) + "</tr>")
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


CSS = """
:root{--navy:#13233b;--ink:#1f2937;--muted:#667085;--line:#e5e7eb;--card:#fff;--bg:#f5f7fa;--blue:#2f6fed;--red:#c0392b;--green:#1f8a70;--warn:#b7791f}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.65}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px}
header{background:linear-gradient(135deg,#0e1b2d 0%,#17324f 55%,#1f4b5f 100%);color:#fff;padding:44px 0 34px}
header .kicker{font-size:12px;letter-spacing:.12em;color:#9fb3c8;text-transform:uppercase}
header h1{font-size:34px;margin:8px 0 10px;letter-spacing:-.02em}
header .sub{color:#c9d6e3;max-width:980px;font-size:15px}
.hero-stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:24px}
.kpi{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px 14px}
.kpi .v{font-size:23px;font-weight:700;letter-spacing:-.02em}.kpi .l{font-size:11.5px;color:#a9bacb;margin-top:3px}
.kpi .v.bad{color:#ff9b8f}.kpi .v.good{color:#7fe0c3}.kpi .v.warn{color:#ffd28a}
nav{position:sticky;top:0;z-index:10;background:rgba(245,247,250,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:16px;overflow:auto;padding-top:11px;padding-bottom:11px}
nav a{color:var(--muted);text-decoration:none;font-size:13px;white-space:nowrap}nav a:hover{color:var(--navy)}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 28px;margin:20px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
h2{font-size:24px;margin:0 0 8px;letter-spacing:-.01em}h3{font-size:17px;margin:22px 0 8px}
p{margin:9px 0;color:#344054}.lead{font-size:15.5px;color:#475467}
.callout{border-left:4px solid var(--blue);background:#f4f7ff;padding:13px 17px;border-radius:0 10px 10px 0;margin:16px 0;color:#20345c}
.callout.bad{border-left-color:var(--red);background:#fff6f5;color:#5b1d16}
.callout.good{border-left-color:var(--green);background:#f0faf7;color:#134e4a}
.callout.warn{border-left-color:var(--warn);background:#fffaf0;color:#5b4216}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}
.tw{overflow:auto;margin:12px 0;border:1px solid var(--line);border-radius:10px}
table.data{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
table.data th{background:#f8fafc;color:#475467;font-weight:600;text-align:right;padding:8px 9px;border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0}
table.data td{padding:7px 9px;border-bottom:1px solid #f0f2f5;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
table.data td:first-child,table.data th:first-child{text-align:left}
table.data.text td,table.data.text th{text-align:left}
table.data tr:hover td{background:#fbfcfe}
figure{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px}
figure img{width:100%;display:block;border-radius:6px}figcaption{font-size:12.5px;color:var(--muted);padding:8px 4px 2px}
ul{margin:8px 0 8px 20px;color:#344054}li{margin:5px 0}
.small{font-size:12.5px;color:var(--muted)}
.footer{color:var(--muted);font-size:12.5px;padding:30px 0 50px;text-align:center}
@media(max-width:900px){.hero-stats{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}header h1{font-size:27px}}
"""


def build_html():
    ld = read("条件_多头.csv")
    sd = read("条件_空头.csv")
    dd = read("条件_多空差.csv")
    sc = read("条件_证据分数.csv")
    ev = read("条件_事件.csv")
    cf = read("条件_经典过滤.csv")
    yr = read("条件_逐年.csv")
    hz = read("条件_期限.csv")
    fam = read("条件_族汇总.csv")
    mz = read("条件_市场x零轴.csv")

    # 表格展示
    ld_disp = show(ld, ["条件族", "条件", "股票日", "日均超额", "t值", "q值", "前期", "后期", "相对MA60增量", "增量t", "综合判定"],
                   pct=["日均超额", "前期", "后期", "相对MA60增量"], integer=["股票日"], tcol=["t值", "增量t"], qcol=["q值"])
    sd_disp = show(sd, ["条件族", "条件", "股票日", "日均超额", "t值", "前期", "后期", "判定"],
                   pct=["日均超额", "前期", "后期"], integer=["股票日"], tcol=["t值"])
    dd_disp = show(dd, ["条件族", "条件", "多头日均", "空头日均", "多空差", "多空差t", "MACD增量", "增量t"],
                   pct=["多头日均", "空头日均", "多空差", "MACD增量"], tcol=["多空差t", "增量t"])
    sc_disp = show(sc[sc["股票日"] > 0], ["条件", "股票日", "日均超额", "t值", "前期", "后期", "综合判定"],
                   pct=["日均超额", "前期", "后期"], integer=["股票日"], tcol=["t值"])
    ld2 = ld.merge(dd[["条件族", "条件", "多空差", "多空差t"]], on=["条件族", "条件"], how="left")
    # 零轴/周月：多头/空头用各自全样本日均，多空差/增量用同一批交易日
    def side_tab(fam):
        a = ld[ld["条件族"] == fam][["条件", "日均超额"]].rename(columns={"日均超额": "多头日均"})
        b = sd[sd["条件族"] == fam][["条件", "日均超额"]].rename(columns={"日均超额": "空头日均"})
        c = dd[dd["条件族"] == fam][["条件", "多空差", "多空差t", "MACD增量", "增量t"]]
        return a.merge(b, on="条件", how="outer").merge(c, on="条件", how="left")
    z_disp = show(side_tab("零轴四态").loc[lambda d: d["多空差"].notna()],
                  ["条件", "多头日均", "空头日均", "多空差", "多空差t", "MACD增量", "增量t"],
                  pct=["多头日均", "空头日均", "多空差", "MACD增量"], tcol=["多空差t", "增量t"])
    wk_disp = show(side_tab("周月状态"),
                   ["条件", "多头日均", "空头日均", "多空差", "多空差t", "MACD增量", "增量t"],
                   pct=["多头日均", "空头日均", "多空差", "MACD增量"], tcol=["多空差t", "增量t"])
    ev_disp = show(ev, ["条件族", "条件", "信号数", "交易日", "净超额", "t值", "前期", "后期", "判定"],
                   pct=["净超额", "前期", "后期"], integer=["信号数", "交易日"], tcol=["t值"])
    cf_disp = show(cf, ["过滤", "信号数", "毛超额", "净超额", "t值", "前期", "后期"],
                   pct=["毛超额", "净超额", "前期", "后期"], integer=["信号数"], tcol=["t值"])
    fam_disp = show(fam, ["条件族", "最优多头条件", "最优多头超额", "最差多头条件", "最差多头超额", "最大增量条件", "最大增量", "增量t"],
                    pct=["最优多头超额", "最差多头超额", "最大增量"], tcol=["增量t"])
    mz_disp = show(mz[mz["日均超额"].notna()], ["市场", "零轴", "股票日", "日均超额", "t值", "前期", "后期", "判定"],
                   pct=["日均超额", "前期", "后期"], integer=["股票日"], tcol=["t值"])
    hz_disp = show(hz[hz["条件"].isin(["全部多头", "零轴双上", "红柱放大", "柱体水平Q3", "DIF水平Q2", "证据分数<=1", "证据分数>=5", "顶背离", "底背离"])
                     & hz["日均超额"].notna()],
                   ["条件", "方向", "周期", "日均超额", "t值", "前期", "后期"],
                   pct=["日均超额", "前期", "后期"], tcol=["t值"])
    # 逐年表：只展示关键条件
    keyy = ["全部多头", "零轴双上", "红柱放大", "证据分数<=1", "证据分数>=5", "日周月共振", "顶背离"]
    yy = yr[yr["条件"].isin(keyy)].pivot_table(index="年份", columns="条件", values="日均超额").reset_index()
    yy_disp = show(yy, ["年份"] + [c for c in keyy if c in yy.columns], pct=[c for c in keyy if c in yy.columns], integer=["年份"])

    # 事件经典过滤排序
    cf_sorted = cf.sort_values("净超额")
    cf_sorted_disp = show(cf_sorted, ["过滤", "信号数", "净超额", "t值", "前期", "后期"],
                          pct=["净超额", "前期", "后期"], integer=["信号数"], tcol=["t值"])

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MACD 条件有效性：什么状态下判断有效？</title><style>{CSS}</style></head><body>
<header><div class="wrap">
<div class="kicker">MACD 有效性 · 条件层 · 2010–2026</div>
<h1>条件有效性：MACD 在什么状态下判断有效？</h1>
<p class="sub">问题不是“用 MACD 选哪些股票”，而是固定 MACD 的多/空判断，问它在每种 MACD 特征状态下对不对。结论：<b>55 个条件格，0 格同时通过“扣费后绝对为正”和“减去 MA60 后仍有增量”</b>。强确认状态显著跑输；唯一不亏的是弱确认，但也没有超额。</p>
<div class="hero-stats">
<div class="kpi"><div class="v bad">0 / 55</div><div class="l">同时过绝对与增量的格</div></div>
<div class="kpi"><div class="v bad">-0.90%</div><div class="l">零轴双上，20 日多头超额</div></div>
<div class="kpi"><div class="v bad">-0.69%</div><div class="l">红柱放大，20 日多头超额</div></div>
<div class="kpi"><div class="v bad">-0.87%</div><div class="l">证据≥5，20 日多头超额</div></div>
<div class="kpi"><div class="v bad">-1.41%</div><div class="l">零轴上金叉，扣费净超额</div></div>
<div class="kpi"><div class="v warn">+0.34%</div><div class="l">柱体 Q3 绝对正，但增量 t=1.2</div></div>
</div></div></header>
<nav><div class="wrap">
<a href="#conclusion">结论</a><a href="#all">总图</a><a href="#score">证据分数</a><a href="#zero">零轴柱体</a>
<a href="#inc">绝对vs增量</a><a href="#age">交叉年龄</a><a href="#event">事件层</a><a href="#regime">周月市场</a>
<a href="#year">逐年期限</a><a href="#action">判断表</a><a href="#method">方法</a>
</div></nav>
<main class="wrap">

<section id="conclusion"><h2>先看结论</h2>
<div class="callout bad"><b>没有一种 MACD 特征状态能让“买入判断”同时通过绝对与增量。</b> 55 个条件格里，绝对为正的 5 格全是温和区（柱体 Q3/Q4、DIF Q2、斜率 Q3/Q4，+0.16% ~ +0.34%/20 日），但同一批交易日多空差 -0.15% ~ +0.16%，减 MA60 后增量 t 绝对值只有 0.2–2.0；有 MACD 增量的 7 格全是深弱区（柱体 Q1、z Q1、零轴双下、周月空、指数下行），多头绝对仍 -0.73% ~ +0.00%。</div>
<div class="callout bad"><b>强确认 = 反向。</b> 零轴双上 -0.899%（t=-6.4）、红柱放大 -0.686%（t=-9.7）、DIF Q5 -1.164%、柱体 Q5 -0.747%、斜率 Q5 -0.660%、周线多+月线空 -1.030%、证据≥5 -0.872%。60 日更大：零轴双上 -1.382%、红柱放大 -1.068%、DIF Q5 -1.905%、证据≥5 -1.389%。证据≥5 为 <b>17/17 年负</b>。</div>
<div class="callout warn"><b>事件层也没有正格。</b> 519,111 个金叉，扣 0.15% 后全部金叉 -0.895%/20 日；零轴上金叉 -1.411%（方向性过滤里最差），周线+月线多头 -1.044%，顶背离 -1.560%，DIF 距离 Q5 -2.360%（更差）。唯一微正的是“证据≤1” +0.264%（t=1.4，后期仅 +0.074%），不显著。</div>
<div class="callout"><b>一句话：</b>MACD 是卖出/回避刻度，不是买入开关。强确认不是买入理由，是减仓理由；弱确认不是买入理由，只是“少亏”；中间地带的正收益属于价格趋势和低波，不属于 MACD。</div>
</section>

<section id="all"><h2>总图：55 格，0 格双门槛</h2>
<div class="grid2">
<div><h3>综合判定分布</h3>
{table(pd.DataFrame({"判定":["反向","无效","仅相对有效","仅价格趋势","样本不足","有效（双门槛）"],"格数":[26,12,7,5,5,0],"含义":["绝对为负、t<-2、前后同负","绝对不显著或前后不同号","减 MA60 后有增量，但多头绝对仍负/零","绝对为正，但减 MA60 后无增量","股票日/交易日不足","没有一格同时过绝对与增量"]}), cls="data text")}
<p class="small">绝对有效 = 多头 20 日超额 &gt;0.15%、t&gt;2、2010-17 与 2018-26 同为正；增量有效 = 同一批交易日 MACD 多空差减 MA60 多空差 &gt;0、t&gt;2。</p></div>
<div><h3>每族最优/最差/最大增量</h3>
{table(fam_disp)}</div>
</div>
{fig_block("图22_绝对与增量.png", "图22：横轴是全部有效日的多头 20 日超额，纵轴是同一批交易日 MACD 多空差减 MA60 多空差。右上“有效区”0 格；蓝点虽绝对为正，但增量 t<2；橙点有增量，但绝对为负。")}
</section>

<section id="score"><h2>证据分数：越确认越差</h2>
{fig_block("图20_证据分数.png", "图20：6 项多头证据（DIF&gt;0、DIF&gt;DEA、柱体上行、DIF 上行、周线多头、月线多头）相加。从 1 项到 5 项单调变差，6 项仍 -0.70%；没有一项为正。")}
{table(sc_disp)}
<div class="callout bad"><b>“确认”本身就是负向信息。</b> 证据从 1 项到 5 项，20 日超额从 -0.025% 降到 -0.872%；60 日 5 项 -1.389%。证据≥5 在 17 年里没有一年为正。证据≤1 均值 -0.105%，只有 6/13 年为正，是“少亏”，不是“赚”。</div>
</section>

<section id="zero"><h2>零轴与柱体：最强确认最差</h2>
{fig_block("图21_零轴与柱体.png", "图21：左图零轴四态的多空对比（同一批交易日）；右图柱体四态（红柱只有多、绿柱只有空，单边状态）。零轴双上多头 -0.90%、空头 -0.40%；红柱放大 -0.69%。")}
<div class="grid2">
<div><h3>零轴四态</h3>{table(z_disp)}<p class="small">多/空为各自全样本日均；多空差/增量为同一批交易日，不一定等于两列之差。</p></div>
<div><h3>柱体四态</h3>{table(show(pd.concat([ld[ld["条件族"]=="柱体四态"].assign(方向="多"), sd[sd["条件族"]=="柱体四态"].assign(方向="空")]).loc[lambda d: d["日均超额"].notna()], ["条件","方向","日均超额","t值","前期","后期"], pct=["日均超额","前期","后期"], tcol=["t值"]))}</div>
</div>
<div class="callout bad"><b>教科书说“只做零轴上的金叉”，这里正好相反。</b> 零轴双上 -0.899%（t=-6.41），零轴双下 -0.180%（t=-1.90）；双上的多空差 -0.383%（t=-3.59），MACD 的方向判断是反的。红柱放大是全部柱体态里最差的（-0.686%，t=-9.65）；空头侧只在绿柱放大时有效（-0.362%，t=-5.04），但那是“回避”有效，不是“做多”有效。</div>
</section>

<section id="inc"><h2>绝对 vs 增量：两个不相交的“好看”</h2>
<h3>绝对为正的 5 格：全是温和区</h3>
{table(show(ld[ld["判定"]=="有效"], ["条件族","条件","日均超额","t值","前期","后期","相对MA60增量","增量t","综合判定"], pct=["日均超额","前期","后期","相对MA60增量"], tcol=["t值","增量t"]))}
<div class="callout warn"><b>5 格都过绝对门槛，但都过不了增量门槛。</b> 同一批交易日多空差 -0.021%、-0.145%、+0.157%、-0.072%、-0.144%；增量 t 只有 1.20、-1.96、1.72、1.59、0.23。Q4 更露馅：全样本 +0.219%，共同日多头 -0.068%、空头 +0.077%，多空差 -0.145%（t=-2.38）——正收益来自空头样本稀少的日子。</div>
<h3>有 MACD 增量的 7 格：全是深弱区</h3>
{table(show(ld2[ld2["综合判定"]=="仅相对有效"], ["条件族","条件","日均超额","t值","多空差","多空差t","相对MA60增量","增量t"], pct=["日均超额","多空差","相对MA60增量"], tcol=["t值","多空差t","增量t"]))}
<div class="callout"><b>增量 +0.49% ~ +1.14%/20 日（t=2.1–4.3），但多头绝对 -0.73% ~ +0.00%。</b> MACD 在这些格子里比 MA60 更会分“谁更差”，不是更会挑“谁会涨”。“周线空+月线空”“指数下行”的多空差本身还是负的（-0.188%、-0.314%，t≈-1.7），增量来自“MACD 的空头差比 MA60 的空头差小”，是相对/对冲语言，不是 long-only 语言。</div>
</section>

<section id="age"><h2>交叉年龄：事件工具，不是状态工具</h2>
{fig_block("图23_交叉年龄.png", "图23：多头取金叉后的年龄，空头取死叉后的年龄。多头 0-5 日 -0.41% → 41-120 日 -0.98%；空头 0-5 日 -0.35% → 41-120 日 +1.32%。")}
{table(show(pd.concat([ld[ld["条件族"]=="交叉年龄"].assign(方向="多"), sd[sd["条件族"]=="交叉年龄"].assign(方向="空")]).loc[lambda d: d["日均超额"].notna()], ["条件","方向","股票日","日均超额","t值","前期","后期"], pct=["日均超额","前期","后期"], integer=["股票日"], tcol=["t值"]))}
<div class="callout"><b>拿得越久越差，空头晚期反转。</b> “DIF&gt;DEA 就持有”不是好规则；它只在刚交叉时最不难看。空头在死叉后 41-120 日转为 +1.32%，继续空/回避会错过反弹。41-120 日多头只有 5.8 万股票日，方向可参考，幅度不要精确外推。</div>
</section>

<section id="event"><h2>事件层：金叉买入，没有正格</h2>
{fig_block("图24_经典过滤.png", "图24：519,111 个 2010 年以来的日线金叉，扣 0.15% 往返成本后的 20 日超额。12 个经典过滤里 11 个为负；零轴上金叉 -1.41%，方向性过滤里最差；DIF 距离 Q5 -2.36% 更差，唯一微正的“证据≤1”+0.26%（t=1.4）。")}
{table(cf_sorted_disp)}
<div class="callout bad"><b>零轴上金叉是方向性过滤里最差的，不是最好的；但 DIF 距离 Q5 比它更差。</b> 零轴上 -1.411%，零轴下 -0.977%；周线+月线多头 -1.044%，比不确认 -0.895% 更差；顶背离 -1.560%；DIF 距离 Q5 -2.360%。“证据≤1”只有 14,806 个信号，前期 +0.917%、后期 +0.074%，不能当策略。</div>
</section>

<section id="regime"><h2>周月与市场：共振反向，弱市相对有效</h2>
<div class="grid2">
<div><h3>周/月状态</h3>{table(wk_disp)}
<p class="small">多/空为各自全样本日均；多空差/增量为同一批交易日，不一定等于两列之差。周线多头+月线空头 -1.030%（t=-8.70），是最差组合；周线空头+月线多头 -0.068%，只是不显著的不亏。</p></div>
<div><h3>市场 × 零轴（多头）</h3>{table(mz_disp)}
<p class="small">12 个组合（3 个 DIF下DEA上 无多头样本）里唯一为正的是“指数下行+零轴双下”+0.175%（t=1.29，前期 +0.318%、后期 +0.025%），是深跌反弹；同期“指数下行+零轴双上” -1.595%（t=-7.25）。</p></div>
</div>
</section>

<section id="year"><h2>逐年与期限：不是一年的事</h2>
{fig_block("图25_逐年.png", "图25：逐年日均 20 日超额。证据≥5 为 17/17 年负；零轴双上、红柱放大各只有 1 年不为负。")}
<h3>逐年日均 20 日超额（关键条件）</h3>
{table(yy_disp)}
<h3>5/20/60 日稳健性</h3>
{table(hz_disp)}
<div class="callout"><b>强确认在三个期限都负，且 60 日更负。</b> 温和区（柱体 Q3、DIF Q2）在三个期限都正，但同一批交易日多空差在 60 日也只有 -0.017%/-0.116%——“绝对正、判别零”不随期限改变。“证据≤1” 5 日 +0.144%、20 日 -0.025%、60 日 -0.229%，随期限翻负。</div>
</section>

<section id="action"><h2>判断表</h2>
{table(pd.DataFrame([
["零轴双上（DIF>0, DEA>0）","-0.899%","反向，t=-6.4","不买；持仓考虑减"],
["红柱放大","-0.686%","反向，t=-9.7","不追；止盈/减仓"],
["DIF Q5 / 柱体 Q5 / 斜率 Q5","-1.164% / -0.747% / -0.660%","反向，t=-7 ~ -7.4","不买；回避"],
["周线多头+月线空头","-1.030%","反向，t=-8.7","最差共振，不追反弹"],
["证据分数≥5","-0.872%","反向，t=-6.8","强确认=反向"],
["顶背离","-0.532%","反向，t=-3.1","有效的是“避”，不是“买”"],
["柱体 Q3 / Q4","+0.340% / +0.219%","仅价格趋势；多空差≈0","不要归功于 MACD"],
["DIF Q2 / 斜率 Q3 / Q4","+0.294% / +0.278% / +0.161%","仅价格趋势；增量 t<2","不要归功于 MACD"],
["零轴双下","-0.180%","仅相对有效；增量 t=2.58","不亏但不赚；只能减配/对冲"],
["柱体 Q1 / 柱体 z Q1","-0.726% / -0.718%","仅相对有效；增量 t=2.6–2.9","多头仍亏；只能“相对少配”"],
["周线空+月线空 / 指数下行","-0.357% / -0.496%","仅相对有效；增量 t=3.9–4.3","弱市相对判断，不是买入"],
["证据分数≤1","-0.025%","无效，t=-0.1","不要当买入信号"],
["底背离","+0.425%","无效，t=1.45","不显著；别抄底"],
["指数下行+零轴双下","+0.175%","无效，t=1.29；后期≈0","深跌反弹，不能当策略"],
], columns=["当前状态","未来20日多头超额","MACD判断","怎么用"]))}
<div class="callout bad"><b>如果你问“MACD 在什么情况下有效”，答案是：作为卖出/回避判断，在 MACD 自己最强的时候有效；作为买入判断，没有一种状态是稳定有效的。</b> 强确认不是买入理由，是减仓理由；弱确认不是买入理由，只是“少亏”；中间地带的正收益属于价格趋势和低波，不属于 MACD。</div>
</section>

<section id="method"><h2>方法与局限</h2>
<div class="grid2">
<div><h3>口径</h3><ul>
<li>2010-01-01 至 2026-10-28，全 A，剔除 ST、上市不足 120 日、T+1 不可成交、T+20 停牌。</li>
<li>多：T 日收盘 DIF&gt;DEA；空：否则。fwd20 = close[T+20]/open[T+1]-1；ex20 = fwd20 - 同日全市场等权（leave-one-out）。</li>
<li>t：日度横截面均值序列，Newey-West lag=20（60 日口径 lag=60）。事件层扣 0.15% 往返；状态层不扣，但绝对门槛要求 &gt;0.15%。</li>
<li>两道门槛：绝对（&gt;0.15%、t&gt;2、前后同号）；增量（同一批交易日 MACD 多空差减 MA60 多空差 &gt;0、t&gt;2）。</li>
<li>周/月状态用已收盘、滞后一期的周线/月线 MACD；月线映射到日线用上月末状态，无未来。</li>
</ul></div>
<div><h3>局限</h3><ul>
<li>20 日重叠窗口，t 值仍偏乐观；结论主要看方向、前后段和逐年一致性。</li>
<li>55 格是看完单维结果后扫出来的，没有做正式 FDR；但 0 格双门槛是“没有通过”，多重检验只会让“有效”更少。</li>
<li>状态层的“仅相对有效”是相对 MA60 的多空差，不是可实现 long-only 收益；A 股做空受限，只能用于减配/回避。</li>
<li>“证据≤1”只有 14,806 个信号，后期 +0.074%，不能因为它是唯一正数就当成策略。</li>
<li>0.15% 成本对低流动性股票可能偏低；状态层不扣每日成本。</li>
</ul></div></div>
<p class="small">完整报告：<a href="报告_条件有效性.md">报告_条件有效性.md</a>；数据表：<a href="条件有效性_数据表.xlsx">条件有效性_数据表.xlsx</a>；主报告：<a href="报告_MACD有效性.md">报告_MACD有效性.md</a>。</p>
</section>
<div class="footer"><p>2026-10-28_MACD有效性_全维度 · 条件有效性追加研究 · 数据截至 2026-10-28</p><p>本报告是统计研究，不构成投资建议。所有历史结果都不保证未来。</p></div>
</main></body></html>"""
    out = os.path.join(ROOT, "条件有效性_可视化.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("html", len(html))


def style_sheet(ws, df, pct=(), tcol=(), qcol=(), integer=()):
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    for j, col in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=j)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        width = max(10, min(34, max(len(str(x)) for x in [col] + df[col].astype(str).head(80).tolist()) + 2))
        ws.column_dimensions[get_column_letter(j)].width = width
        for row in range(2, len(df) + 2):
            c = ws.cell(row=row, column=j)
            if col in pct:
                c.number_format = "0.00%"
            elif col in tcol:
                c.number_format = "0.00"
            elif col in qcol:
                c.number_format = "0.000"
            elif col in integer:
                c.number_format = "0" if col == "年份" else "#,##0"
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def build_excel():
    ld = read("条件_多头.csv")
    sd = read("条件_空头.csv")
    dd = read("条件_多空差.csv")
    sc = read("条件_证据分数.csv")
    ev = read("条件_事件.csv")
    cf = read("条件_经典过滤.csv")
    yr = read("条件_逐年.csv")
    hz = read("条件_期限.csv")
    fam = read("条件_族汇总.csv")
    mz = read("条件_市场x零轴.csv")
    mzs = read("条件_零轴x分数.csv")

    notes = pd.DataFrame({
        "项目": ["研究", "数据", "判断", "结果", "超额", "成本", "绝对门槛", "增量门槛", "注意"],
        "说明": [
            "MACD 条件有效性：把 MACD 特征当判断条件，检验多/空判断在未来 20 日对不对",
            "2010-2026 全 A 日线，剔除 ST、上市<120 日、不可成交；5,507 只（含退市）",
            "T 日收盘 DIF>DEA 为多，否则为空；T+1 开盘执行",
            "55 个条件格；绝对有效 5 格、增量有效 7 格、交集 0 格",
            "fwd20 = close[T+20]/open[T+1]-1；ex20 = fwd20 - 同日全市场等权 leave-one-out",
            "事件层扣 0.15% 往返；状态层不扣，但绝对门槛要求 >0.15%",
            "多头 20 日超额 >0.15%，t>2，2010-17 与 2018-26 同为正",
            "同一批交易日 MACD 多空差减 MA60 多空差 >0，t>2",
            "20 日重叠窗口，t 仍偏乐观；55 格未做正式 FDR；仅相对有效不是 long-only 收益",
        ],
    })
    out = os.path.join(ROOT, "条件有效性_数据表.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        notes.to_excel(w, sheet_name="00_说明", index=False)
        ld.to_excel(w, sheet_name="01_多头", index=False)
        sd.to_excel(w, sheet_name="02_空头", index=False)
        dd.to_excel(w, sheet_name="03_多空差", index=False)
        sc.to_excel(w, sheet_name="04_证据分数", index=False)
        ev.to_excel(w, sheet_name="05_事件", index=False)
        cf.to_excel(w, sheet_name="06_经典过滤", index=False)
        yr.to_excel(w, sheet_name="07_逐年", index=False)
        hz.to_excel(w, sheet_name="08_期限", index=False)
        fam.to_excel(w, sheet_name="09_族汇总", index=False)
        mz.to_excel(w, sheet_name="10_市场x零轴", index=False)
        mzs.to_excel(w, sheet_name="11_零轴x分数", index=False)

    from openpyxl import load_workbook
    wb = load_workbook(out)
    pct = ["日均超额", "中位超额", "中位", "正日占比", "前期", "后期", "相对MA60增量", "多空差", "MACD增量",
           "MA60多空差", "毛超额", "净超额", "多头日均", "空头日均", "最优多头超额",
           "最差多头超额", "最大增量"]
    tcol = ["t值", "增量t", "多空差t"]
    for name, df in [("01_多头", ld), ("02_空头", sd), ("03_多空差", dd), ("04_证据分数", sc),
                     ("05_事件", ev), ("06_经典过滤", cf), ("07_逐年", yr), ("08_期限", hz),
                     ("09_族汇总", fam), ("10_市场x零轴", mz), ("11_零轴x分数", mzs)]:
        style_sheet(wb[name], df, pct=pct, tcol=tcol, qcol=["q值"], integer=["股票日", "交易日", "信号数", "年份", "绝对有效格数", "有增量格数", "综合有效格数"])
    wb.save(out)
    print("excel", out)


def add_nav():
    p = os.path.join(ROOT, "MACD有效性_可视化.html")
    if not os.path.exists(p):
        return
    s = open(p, encoding="utf-8").read()
    if "条件有效性_可视化.html" in s:
        return
    s = s.replace('<a href="#stocks">个股适配</a>',
                  '<a href="#stocks">个股适配</a><a href="条件有效性_可视化.html">条件判断</a>')
    open(p, "w", encoding="utf-8").write(s)
    print("nav linked")


if __name__ == "__main__":
    build_html()
    build_excel()
    add_nav()
