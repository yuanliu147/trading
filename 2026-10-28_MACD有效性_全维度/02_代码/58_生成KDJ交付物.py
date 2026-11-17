#!/usr/bin/env python3
"""58 生成 MACD×KDJ 配合 HTML / Excel，并在主 HTML 中加入口"""
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
            d[c] = d[c].map(lambda v: f"{v:.4f}" if pd.notna(v) and abs(v) < 0.001 else (f"{v:.3f}" if pd.notna(v) else "—"))
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
html{scroll-padding-top:72px}*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.65}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px}
header{background:linear-gradient(135deg,#0e1b2d 0%,#17324f 55%,#1f4b5f 100%);color:#fff;padding:44px 0 34px}
header .kicker{font-size:12px;letter-spacing:.12em;color:#9fb3c8;text-transform:uppercase}
header h1{font-size:34px;margin:8px 0 10px;letter-spacing:-.02em}
header .sub{color:#c9d6e3;max-width:1000px;font-size:15px}
.hero-stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:24px}
.kpi{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:12px 14px}
.kpi .v{font-size:23px;font-weight:700;letter-spacing:-.02em}.kpi .l{font-size:11.5px;color:#a9bacb;margin-top:3px}
.kpi .v.bad{color:#ff9b8f}.kpi .v.good{color:#7fe0c3}.kpi .v.warn{color:#ffd28a}
nav{position:sticky;top:0;z-index:10;background:rgba(245,247,250,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:16px;overflow:auto;padding-top:11px;padding-bottom:11px}
nav a{color:var(--muted);text-decoration:none;font-size:13px;white-space:nowrap}nav a:hover{color:var(--navy)}
section{scroll-margin-top:72px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:26px 28px;margin:20px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
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
    four = read("KDJ_状态_四态.csv")
    cond = read("KDJ_状态_条件.csv").rename(columns={
        "多头t": "多头t值", "空头t": "空头t值", "多空差t": "多空差t值",
        "MACD增量t": "MACD增量t值", "KDJ增量_多t": "KDJ增量多t值", "KDJ增量_空t": "KDJ增量空t值",
        "DiDt": "交互t值", "多头q": "多头q值", "MACD增量q": "MACD增量q值",
        "KDJ增量_多q": "KDJ增量多q值", "DiDq": "交互q值", "DiD": "交互项",
        "KDJ增量_多": "KDJ增量多", "KDJ增量_空": "KDJ增量空",
        "KDJ增量_多判定": "KDJ增量多判定", "KDJ增量_空判定": "KDJ增量空判定"})
    did = read("KDJ_状态_四态DiD.csv").rename(columns={"t": "t值"})
    em = read("KDJ_事件_MACD金叉_q.csv")
    emi = read("KDJ_事件_MACD金叉增量_q.csv").rename(columns={"A毛": "条件组毛超额", "B毛": "对照组毛超额"})
    ek = read("KDJ_事件_KDJ金叉_q.csv")
    eki = read("KDJ_事件_KDJ金叉增量_q.csv").rename(columns={"A毛": "条件组毛超额", "B毛": "对照组毛超额"})
    hz = read("KDJ_事件_期限.csv").rename(columns={"t": "t值"})
    yr = read("KDJ_事件_逐年差异.csv").rename(columns={"t": "t值"})
    param = read("KDJ_参数稳健.csv").rename(columns={"A": "条件组均值", "B": "对照组均值", "t": "t值"})
    wkp = read("KDJ_周线极值_参数.csv").rename(columns={"early": "前期", "late": "后期", "n": "信号数", "days": "交易日", "mean": "日均超额", "t": "t值", "KDJ参数": "周线KDJ参数"})
    wky = read("KDJ_周线极值_逐年.csv").rename(columns={"t": "t值"})
    k20 = read("KDJ_K20解剖.csv").rename(columns={"t": "t值"})
    k20y = read("KDJ_K20逐年.csv").rename(columns={"t": "t值"})
    k20c = read("KDJ_K20控制.csv").rename(columns={"A": "条件组均值", "B": "对照组均值", "t": "t值"})
    k20m = read("KDJ_K20匹配.csv").rename(columns={"t": "t值"})
    fm = read("探索_FamaMacBeth.csv").rename(columns={"平均R2": "平均R方"})

    # 四态
    four_disp = show(four, ["KDJ", "MACD", "股票日", "交易日", "日均超额", "中位", "t值", "前期", "后期", "有效年数", "有效正年数", "判定"],
                     pct=["日均超额", "中位", "前期", "后期"], integer=["股票日", "交易日", "有效年数", "有效正年数"],
                     tcol=["t值"])
    did_disp = show(did, ["对比", "均值", "t", "共同日"], pct=["均值"], tcol=["t"], integer=["共同日"])
    # 条件
    key = ["KQ1", "KQ2", "KQ3", "KQ4", "KQ5", "K>D", "K>70", "K>80", "K<20", "5日内金叉", "顶背离", "底背离", "周线KDJ多"]
    ck = cond[cond.条件.isin(key)].copy()
    ck["order"] = ck.条件.apply(lambda x: key.index(x))
    ck = ck.sort_values("order")
    ck = ck.rename(columns={"多头t": "多头t值", "空头t": "空头t值", "KDJ增量_多t": "KDJ增量t值",
                             "KDJ增量_多q": "KDJ增量q值", "DiDt": "DiDt值"})
    cond_disp = show(ck, ["条件", "多头日均", "多头t值", "多头中位", "KDJ增量_多", "KDJ增量t值",
                          "空头日均", "空头t值", "DiD", "绝对判定"],
                     pct=["多头日均", "多头中位", "空头日均", "KDJ增量_多", "DiD"],
                     tcol=["多头t值", "空头t值", "KDJ增量t值"])
    # Fama-MacBeth
    fm2 = fm[fm.设定.isin(["20日_B_加趋势控制", "20日_D_常用状态", "20日_C_连续K二次"])].copy()
    fm_disp = show(fm2, ["设定", "变量", "均值", "t值", "交易日", "平均R方"],
                   pct=["均值"], tcol=["t值"], integer=["交易日"], qcol=[])
    # K20
    k20_disp = show(k20, ["口径", "均值", "中位", "修剪均值", "股票日", "交易日", "t值"],
                    pct=["均值", "中位", "修剪均值"], tcol=["t值"], integer=["股票日", "交易日"])
    k20y_disp = show(k20y, ["年份", "交易日", "日均超额", "中位", "t值", "股票日"],
                     pct=["日均超额", "中位"], tcol=["t值"], integer=["年份", "交易日", "股票日"])
    # 事件
    em_key = ["全部MACD金叉", "KQ1(最低)", "KQ2", "KQ3", "KQ4", "KQ5(最高)", "KDJ多(K>D)",
              "K>70", "K>80", "K<20", "同日KDJ金叉", "KDJ顶背离", "KDJ底背离", "周线KDJ多", "月线KDJ多"]
    em2 = em[em.过滤.isin(em_key)].copy()
    em2["order"] = em2.过滤.apply(lambda x: em_key.index(x))
    em2 = em2.sort_values("order")
    em_disp = show(em2, ["过滤", "信号数", "毛超额", "净超额", "t值", "q值", "前期", "后期", "正年数"],
                   pct=["毛超额", "净超额", "前期", "后期"], integer=["信号数", "正年数"], tcol=["t值"], qcol=["q值"])
    emi = emi.rename(columns={"A毛": "条件组毛超额", "B毛": "对照组毛超额"})
    emi_disp = show(emi, ["对比", "条件组毛超额", "对照组毛超额", "差值", "t值", "q值", "共同日"],
                    pct=["条件组毛超额", "对照组毛超额", "差值"], tcol=["t值"], qcol=["q值"], integer=["共同日"])
    ek_key = ["全部KDJ金叉", "MACD多", "MACD空", "MACD双上", "MACD双下", "价格>MA60",
              "MACD多+价格>MA60", "前5日内MACD金叉", "前5日内无MACD金叉", "KQ5(最高)", "KDJ顶背离", "KDJ底背离"]
    ek2 = ek[ek.过滤.isin(ek_key)].copy()
    ek2["order"] = ek2.过滤.apply(lambda x: ek_key.index(x))
    ek2 = ek2.sort_values("order")
    ek_disp = show(ek2, ["过滤", "信号数", "毛超额", "净超额", "t值", "q值", "前期", "后期", "正年数"],
                   pct=["毛超额", "净超额", "前期", "后期"], integer=["信号数", "正年数"], tcol=["t值"], qcol=["q值"])
    eki_disp = show(eki, ["对比", "条件组毛超额", "对照组毛超额", "差值", "t值", "q值", "共同日"],
                    pct=["条件组毛超额", "对照组毛超额", "差值"], tcol=["t值"], qcol=["q值"], integer=["共同日"])
    hz_disp = show(hz[hz.类别.isin(["MACD金叉全部", "MACD金叉+KQ2-4", "MACD金叉+KQ5", "MACD金叉+同日KDJ金叉", "MACD金叉+K>70"])],
                   ["类别", "周期", "信号数", "毛超额", "净超额", "t值"],
                   pct=["毛超额", "净超额"], tcol=["t值"], integer=["信号数"])
    # 参数
    kp = param[param.网格 == "KDJ"].copy()
    kp_sum = []
    for (grid, ind), g in param.groupby(["网格", "指标"], sort=False):
        x = g.dropna(subset=["差值"])
        if len(x) == 0:
            continue
        kp_sum.append(dict(网格=grid, 指标=ind, 组数=len(x),
                           负组数=int((x.差值 < 0).sum()), 正组数=int((x.差值 > 0).sum()),
                           均值=x.差值.mean(), 最小=x.差值.min(), 最大=x.差值.max(),
                           t负2=int((x["t值"] < -2).sum()), t正2=int((x["t值"] > 2).sum())))
    kp_sum = pd.DataFrame(kp_sum)
    kp_disp = show(kp_sum, ["网格", "指标", "组数", "负组数", "正组数", "均值", "最小", "最大", "t负2", "t正2"],
                   pct=["均值", "最小", "最大"], integer=["组数", "负组数", "正组数", "t负2", "t正2"])
    wkp2 = wkp[(wkp.周期 == 20) & (wkp.过滤.str.contains("周K<20增量"))].copy()
    wkp2["N"] = wkp2.周线KDJ参数.astype(int)
    wkp2 = wkp2.sort_values("N")
    wkp_disp = show(wkp2, ["N", "信号数", "交易日", "日均超额", "t值", "前期", "后期"],
                    pct=["日均超额", "前期", "后期"], tcol=["t值"], integer=["N", "信号数", "交易日"])
    # 逐年差异
    yd = yr.pivot_table(index="年份", columns="对比", values="差值").reset_index()
    ycols = ["同日双金叉 vs 非", "KQ5 vs KQ2-4", "KQ1 vs KQ2-4", "KDJ多 vs 空", "K>70 vs <="]
    for c in ycols:
        if c not in yd:
            yd[c] = np.nan
    yd = yd[["年份"] + ycols]
    yd_disp = show(yd, ["年份"] + ycols, pct=ycols, integer=["年份"])

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MACD × KDJ：能相互配合吗？</title><style>{CSS}</style></head><body>
<header><div class="wrap">
<div class="kicker">MACD × KDJ · 条件与事件 · 2010–2026</div>
<h1>MACD 与 KDJ 能相互配合吗？</h1>
<p class="sub">结论：<b>能，但不是“双金叉确认”，而是“MACD 定方向、KDJ 定是否过热”。</b> 把 KDJ 金叉当作 MACD 金叉的确认，是本研究里最差的一类用法：同日双金叉 20 日净超额 -1.25%，17 年 0 年为正；KDJ 金叉在 MACD 多头里比在 MACD 空头里差 0.58 个百分点。KDJ 真正有增量的地方，是告诉你“这波多头已经走到什么位置”：中间区比两端好，但绝对仍不足以覆盖成本。</p>
<div class="hero-stats">
<div class="kpi"><div class="v bad">0 / 43</div><div class="l">稳健有效的 KDJ 条件格</div></div>
<div class="kpi"><div class="v bad">-0.65%</div><div class="l">MACD多+KDJ多，20 日超额</div></div>
<div class="kpi"><div class="v bad">-1.25%</div><div class="l">同日双金叉，20 日净超额</div></div>
<div class="kpi"><div class="v bad">-0.58pp</div><div class="l">KDJ金叉：MACD多 - MACD空</div></div>
<div class="kpi"><div class="v warn">+0.58% / -0.99%</div><div class="l">K&lt;20 的均值 / 中位</div></div>
<div class="kpi"><div class="v good">15 / 15</div><div class="l">KQ5 惩罚在 KDJ 参数下为负</div></div>
</div></div></header>
<nav><div class="wrap">
<a href="#answer">结论</a><a href="#four">四态</a><a href="#level">KDJ水平</a><a href="#k20">K&lt;20解剖</a>
<a href="#macd">MACD金叉</a><a href="#kdj">KDJ金叉</a><a href="#timing">双金叉时序</a>
<a href="#param">参数周期</a><a href="#judge">判断表</a><a href="#method">方法边界</a>
</div></nav>
<main class="wrap">

<section id="answer"><h2>先看结论</h2>
<div class="callout bad"><b>1. 两个指标同多，是最差的四态格。</b> MACD多+KDJ多 -0.65%（t=-8.6，17 年 1 年正）；MACD空+KDJ多 -0.18%（四态里最好，但仍负）；KDJ 与 MACD 同向的交互 DiD = -0.176pp（t=-3.48）。<b>同向不是共振，是拥挤。</b></div>
<div class="callout bad"><b>2. “双金叉”是负选择。</b> 同日 MACD 金叉 + KDJ 金叉：20 日净 -1.25%，0/17 年正，t=-9.58；比非同日 KDJ 金叉的 MACD 金叉差 0.479pp（t=-4.46），比全部 MACD 金叉差 0.356pp。KDJ 金叉在 MACD 多头里比在 MACD 空头里差 0.58pp（t=-4.17）；在 MACD 金叉后 5 日内差 0.47pp（t=-5.41）。</div>
<div class="callout warn"><b>3. KDJ 的增量在“位置”，不在“方向”。</b> K 五分位是 U 形：Q1 -0.30%、Q5 -0.52%，Q3 +0.14%、Q4 +0.01%（MACD 多）。Q3/Q4 相对补集 +0.30/+0.26pp（t=3.49/4.81，q&lt;0.01），但绝对 t 只有 1.94/0.12，中位 -1.30%/-1.41%。<b>中间只是相对少亏，不是买点。</b></div>
<div class="callout bad"><b>4. K&lt;20 是均值陷阱。</b> 股票层均值 +0.58%，中位 -0.99%，5% 修剪 +0.18%；只有当日 ≥50 只股票同时 K&lt;20（市场级恐慌）时，日均才 +1.53%。匹配对照 +0.51%（t=2.18），但 2020/2022 为负。<b>这是暴跌反弹的广度选择，不是个股 KDJ 买点。</b></div>
<div class="callout"><b>5. 唯一稳健的方向是“别买太晚”。</b> KQ5、K&gt;70、KDJ 金叉/双金叉都是显著负向过滤；它们不能让你赚钱，但能让你少亏。事件层里 KDJ 底背离是唯一毛超额为正的过滤（+0.12%），但净 -0.03%、t=0.26；状态层的相对正增量是 KQ3/Q4（+0.30/+0.26pp，t=3.49/4.81），但绝对也不显著。</div>
<div class="callout warn"><b>6. 周线 K&lt;20 + 日线 MACD 金叉</b>是参数与年份上较稳的弱正口袋：+0.32%（t=1.54），相对周 K≥20 +0.69pp（t=2.05），5 组周线参数全为正、约 11/15 年正。但绝对 t 不足 2，只能观察，不能当策略。</div>
</section>

<section id="four"><h2>一、四态：同向是坏消息</h2>
<p>MACD 判断 = DIF&gt;DEA；KDJ 判断 = K&gt;D。四态按各自全样本日均；多空差/增量按同一批交易日配对。</p>
{table(four_disp)}
{table(did_disp)}
<div class="callout bad">KDJ 多对 MACD 多的“确认”是负增量：MACD 多内部 KDJ多-空 -0.181pp（t=-2.94）；KDJ 空对 MACD 空也不是好的“确认”（+0.089pp，t=+1.24）。两个指标同向时，新增信息最少，反而把“已经涨了”再确认一遍。</div>
{fig_block("图26_KDJ四态.png", "图26　四态全部为负；MACD多+KDJ多最差，MACD空+KDJ多反而少亏")}
</section>

<section id="level"><h2>二、KDJ 水平：U 形，不是越高越强</h2>
<p>把 K 值按当日横截面分五组。原始均值是 U 形：Q1、Q5 差，Q3/Q4 好；但 Q3 的绝对 t 只有 1.94，Q4 几乎为 0。相对补集，Q3/Q4 的增量是正的且 q&lt;0.01；Q1/Q5 是负的。</p>
{table(cond_disp)}
<p class="small">K&lt;20、K&gt;80 的条件层口径要求当天该格 ≥50 只股票；这是市场级极端日的均值，个股层与全部有信号日见第三节。K&lt;20 的 +1.53% 不是个股买入信号。</p>
<div class="callout warn">控制先期收益、波动、成交额、MA60 后（Fama-MacBeth），Q2-Q4 仍比 Q1 好 0.27-0.33pp（t 3.6-6.4）；但 K&lt;20 的系数是 -0.686%（t=-6.13），MACD多×K&gt;D 是 -0.316%（t=-4.01）。<b>位置有信息，方向是噪音。</b></div>
{fig_block("图27_KDJ水平U形.png", "图27　KDJ 水平是 U 形：Q1、Q5 差，Q3/Q4 较好；但 Q3 也只有 +0.14%（MACD多）")}
<h3>控制后的系数（20 日，Fama-MacBeth）</h3>
{table(fm_disp)}
</section>

<section id="k20"><h2>三、K&lt;20 解剖：均值陷阱</h2>
<p>MACD 多 + K&lt;20 的 45,327 个股票日：均值 +0.58%，中位 <b>-0.99%</b>，5% 修剪 +0.18%。典型股票继续跌；均值靠少数暴跌反弹日拉高（2024-09-23 +18.3%、2021-02-09/10 +14~16%、2026-07 多日 +10-12%）。</p>
{table(k20_disp)}
<div class="callout bad">“日度 ≥50 只”才转正，是因为这个门槛只留下市场级恐慌日；全部有信号的交易日（≥1 只）均值是 -0.86%。这不是个股 KDJ 买点，而是“市场恐慌反弹”。事件层 MACD 金叉当天 K&lt;20 也只有 -0.41% 净、t=-0.72。</div>
<div class="grid2"><div>{fig_block("图32_K20解剖.png", "图32　同一格三种口径：中位始终为负；≥50 只/日才转正")}</div>
<div><h3>逐年日度均值</h3>{table(k20y_disp)}</div></div>
<h3>先期收益/波动控制</h3>
{table(show(k20c, ["维度", "分位", "A股日", "B股日", "条件组均值", "对照组均值", "差值", "t值", "共同日"], pct=["条件组均值", "对照组均值", "差值"], tcol=["t值"], integer=["A股日", "B股日", "共同日"]))}
<p class="small">匹配对照：同一天、同 20 日收益分位、同波动分位，随机抽 K≥20 的股票；差值 +0.51%（t=2.18），2020 -1.66%（t=-3.03）、2022 -0.97%，方向不稳。</p>
</section>

<section id="macd"><h2>四、事件层：MACD 金叉 × KDJ</h2>
<p>519,111 个 MACD 金叉，20 日毛超额 -0.745%、净 -0.895%，17 年 0 年正。<b>没有一种 KDJ 过滤能把 MACD 金叉变成正净超额。</b>最好的 KQ2/Q3 也只有 -0.50%/-0.48%；最差是 KDJ 顶背离 -1.35%、同日双金叉 -1.25%、K&gt;70 -1.21%、KQ5 -1.12%。</p>
{table(em_disp)}
{fig_block("图28_事件过滤_MACD金叉.png", "图28　MACD 金叉的 KDJ 过滤：没有一种能转正；同日双金叉、KQ5、K&gt;70 最差")}
<h3>过滤增量：两端都比中间差</h3>
{table(emi_disp)}
<div class="callout warn">KDJ 方向 (K&gt;D) 的增量只有 -0.096pp（t=-0.79，q=0.48），不显著；真正有增量的是“位置极端”（KQ1/KQ5）和“同日双金叉”。KDJ 底背离是唯一毛超额为正的过滤（+0.12%），但净 -0.03%、t=0.26、n=1,028，不显著。</div>
</section>

<section id="kdj"><h2>五、事件层：KDJ 金叉 × MACD</h2>
<p>1,131,381 个 KDJ 金叉，20 日毛 -0.664%、净 -0.814%，17 年 0 年正。KDJ 金叉在 MACD 多头里 -1.36%，在 MACD 空头里 -0.76%，差 -0.58pp（t=-4.17）；在价格&gt;MA60 时差 -0.83pp（t=-5.72）。</p>
{table(ek_disp)}
{fig_block("图29_事件过滤_KDJ金叉.png", "图29　KDJ 金叉：在 MACD 多头里、在 MACD 金叉后 5 日内更差")}
<h3>KDJ 金叉内部：MACD 过滤增量</h3>
{table(eki_disp)}
<div class="callout bad">“KDJ 金叉确认 MACD”是负选择：MACD 多-空 -0.577pp（t=-4.17，q&lt;0.001）；前 5 日 MACD 金叉-无 -0.473pp（t=-5.41，q&lt;1e-6）；KQ5-Q2-4 -0.727pp（t=-4.46）。在价格&gt;MA60 时，MACD 多-空 -0.827pp（t=-5.72）；价格≤MA60 时不显著。</div>
</section>

<section id="timing"><h2>六、双金叉与时序：等确认=晚买</h2>
<div class="callout bad"><b>同日双金叉</b>：98,384 个信号，20 日净 -1.25%，0/17 年正，t=-9.58；5/20/60 日 -0.50% / -1.10% / -1.69%。逐年差异（同日-非同日）16/17 年为负；同日双金叉本身的绝对超额 17/17 年为负。参数（同日-非同日差值）：15 组 KDJ 中 12 组负；5 组 MACD 中 2 组负——标准参数下很强，但不是所有参数都成立。</div>
<div class="grid2">
<div>{fig_block("图31_逐年.png", "图31　逐年差异（同日双金叉 - 非同日）：16/17 年为负")}</div>
<div>{fig_block("图33_期限.png", "图33　期限：确认惩罚不是一日噪声，20-60 日仍为负")}</div>
</div>
<h3>谁先谁后</h3>
<table class="data text"><thead><tr><th>时序</th><th>20 日净超额</th><th>t</th><th>说明</th></tr></thead><tbody>
<tr><td>MACD 金叉 + 同日 KDJ 金叉</td><td>-1.251%</td><td>-9.58</td><td>最差</td></tr>
<tr><td>MACD 金叉 + 前 5 日 KDJ 金叉（不含当日）</td><td>-0.909%</td><td>-7.42</td><td>与非同日基本相同（差 +0.04pp，t=0.86）</td></tr>
<tr><td>MACD 金叉 + 前 5 日无 KDJ 金叉</td><td>-0.948%</td><td>-8.67</td><td>略差，差 -0.04pp</td></tr>
<tr><td>KDJ 金叉 + 前 5 日 MACD 金叉</td><td>-1.212%</td><td>-9.70</td><td>差</td></tr>
<tr><td>KDJ 金叉 + 前 5 日无 MACD 金叉</td><td>-0.782%</td><td>-11.01</td><td>相对好，但仍负</td></tr>
</tbody></table>
<p>两个方向都指向：<b>“等另一个指标确认”不创造超额，只是把买点推后。</b>唯一相对好的是“KDJ 金叉、但前 5 日没有 MACD 金叉”（-0.78%），也就是 KDJ 单独先动；即使如此，绝对仍负。</p>
</section>

<section id="param"><h2>七、参数与周期稳健性</h2>
<h3>KDJ/MACD 参数网格</h3>
{table(kp_disp)}
<p class="small">KDJ 15 组：N=5/9/14/19/26 × 平滑 3,3 / 5,3 / 3,5；MACD 5 组：8/17/9、12/26/9、16/32/9、20/40/9、6/13/5。最稳的两条：KQ5 在 MACD 金叉时更差（15/15，12 组 t&lt;-2）；KDJ 金叉在 MACD 多头里更差（15/15，所有 t&lt;-2）。不稳的两条：K&gt;70 的事件过滤（10/15，0 组 t&lt;-2）和 KQ1 的事件过滤（9/15）。</p>
{fig_block("图30_参数稳健.png", "图30　参数稳健：KQ5 惩罚与“KDJ金叉在MACD多头更差”在 15 组 KDJ 参数下全部为负")}
<h3>周线 KDJ 超卖 + 日线 MACD 金叉</h3>
{table(wkp_disp)}
<div class="grid2"><div>{fig_block("图34_周线极值.png", "图34　周线 K&lt;20 + 日线 MACD 金叉：方向为正但不稳")}</div>
<div><h3>周期结论</h3><ul>
<li>日线 MACD × 周线 KDJ：周 KDJ 多确认同样负；DiD -0.242pp（t=-3.26）。</li>
<li>周线同周期：KDJ 金叉 + MACD 多 -0.475%，+MACD 空 -0.053%，差 -0.517pp（t=-2.59）。</li>
<li>月线同周期样本小，方向不稳；月线 KDJ 在日线 MACD 上无增量（+0.02pp，t=0.14）。</li>
<li>周 K&lt;20 + 日线 MACD 金叉是参数与年份上较稳的弱正口袋：+0.32%（t=1.54），增量 +0.69pp（t=2.05），5 组参数全正、约 11/15 年正；不能单独当策略。</li>
</ul></div></div>
</section>

<section id="judge"><h2>八、最终判断表</h2>
<table class="data text"><thead><tr><th>用法</th><th>判断</th><th>关键数字</th><th>证据等级</th><th>可执行性</th></tr></thead><tbody>
<tr><td>同日 MACD+KDJ 双金叉</td><td><b>反向，最差</b></td><td>净 -1.25%，0/17 年正，t=-9.58</td><td>A（标准参数）/ B（跨参数）</td><td>不要用</td></tr>
<tr><td>KDJ 金叉在 MACD 多头</td><td><b>反向</b></td><td>-1.36% vs -0.76%，差 -0.58pp，t=-4.17</td><td>A</td><td>不要用</td></tr>
<tr><td>KDJ 金叉在 MACD 金叉后 5 日内</td><td><b>反向</b></td><td>-1.21% vs -0.78%，差 -0.47pp，t=-5.41</td><td>A</td><td>不要用</td></tr>
<tr><td>MACD 金叉 + KQ5 / K&gt;70</td><td><b>不买</b></td><td>KQ5 净 -1.12%；K&gt;70 净 -1.21%</td><td>A（KQ5）/ B（K&gt;70）</td><td>不买/减仓</td></tr>
<tr><td>MACD 金叉 + KQ1</td><td><b>不买（接飞刀）</b></td><td>净 -0.84%，4/17 年正</td><td>B</td><td>不买</td></tr>
<tr><td>MACD 多 + KQ3/Q4</td><td><b>相对少亏，不是买点</b></td><td>绝对 +0.14%/+0.01%；相对补集 +0.30/+0.26pp</td><td>B</td><td>只用于减仓排序</td></tr>
<tr><td>MACD 多 + K&lt;20</td><td><b>均值陷阱</b></td><td>均值 +0.58%，中位 -0.99%；≥50 只/日才 +1.53%</td><td>C</td><td>不能买</td></tr>
<tr><td>周线 K&lt;20 + 日线 MACD 金叉</td><td><b>弱正，不稳</b></td><td>+0.32%，t=1.54；增量 +0.69pp，t=2.05</td><td>C</td><td>观察，不可单独</td></tr>
<tr><td>KDJ 底背离（MACD 多）</td><td><b>相对正，绝对不显著</b></td><td>+0.95pp，t=2.57；绝对 +0.52%，t=1.71</td><td C</td><td>观察</td></tr>
<tr><td>月线 KDJ 配合</td><td><b>无增量</b></td><td>差 +0.02pp，t=0.14</td><td>D</td><td>不用</td></tr>
<tr><td>任何 MACD+KDJ 组合</td><td><b>无稳健绝对多头</b></td><td>全部条件格绝对为负或 t&lt;2；稳健有效 0 格</td><td>A</td><td>不能当买入系统</td></tr>
</tbody></table>
<div class="callout bad"><b>执行版：</b>MACD 多 + KDJ 多 = 不买/减仓；MACD 金叉 + 同日/近日 KDJ 金叉 = 不追；MACD 金叉 + KQ5/K&gt;70 = 不追高；MACD 金叉 + KQ1 = 不接飞刀；MACD 多 + KQ3/Q4 = 只用于相对排序；K&lt;20 不能买；周线 K&lt;20 + 日线 MACD 金叉只能观察。</div>
</section>

<section id="method"><h2>九、方法、口径与边界</h2>
<div class="grid2"><div>
<h3>定义</h3>
<ul>
<li>KDJ(9,3,3)：RSV=(C-LLV(L,9))/(HHV(H,9)-LLV(L,9))×100；K=2/3K+1/3RSV；D=2/3D+1/3K；J=3K-2D；K/D 初值 50，区间为 0 时 RSV=50。</li>
<li>MACD(12,26,9)：DIF=EMA12-EMA26，DEA=EMA9(DIF)。</li>
<li>信号 T 日收盘确认，T+1 开盘买入，20 日持有；0.15% 往返成本。</li>
<li>超额 = 同日全 A 等权同口径收益（leave-one-out）。</li>
</ul>
<h3>判定</h3>
<ul>
<li>绝对：扣成本后 &gt;0 且 t&gt;2 且 2010-17/2018-26 同号。</li>
<li>相对 MA60：MACD 多空差 - MA60 多空差 &gt;0 且 t&gt;2。</li>
<li>相对单指标：格子 vs 补集 &gt;0 且 t&gt;2。</li>
<li>稳定性：中位同号、有效年份≥6、同号年份≥60%。</li>
<li>t：日度横截面均值序列，Newey-West lag=20；BH-FDR。</li>
</ul>
</div><div>
<h3>边界</h3>
<ul>
<li>只做信号层，不做组合/仓位/滑点/涨跌停排队；不做空。</li>
<li>日线，不含盘中判断；KDJ 路径依赖，不同软件初值/平滑可能不同。</li>
<li>稀有格（K&lt;20、K&gt;80）有广度选择；报告用多口径对照，不把 ≥50 只/日的数字当个股结论。</li>
<li>2026 数据到 10-28，最后一段 20 日收益有部分未完成；K&lt;20 的 2026 +6.16% 对均值影响大。</li>
<li>参数只扫 15 组 KDJ、5 组 MACD；不排除其他参数有不同结果。</li>
<li>多重检验按族内 BH-FDR；跨表未合并，结论按“族内可重复”理解。</li>
</ul>
<div class="callout warn">原条件研究（55 格，0 格同时过绝对与 MA60 增量）与本报告（43 个 KDJ 条件格，0 个稳健有效）方向一致：<b>MACD 是卖出/回避刻度，不是买入开关；KDJ 的加入没有把它变成买入开关，只是把“什么时候不该买”做得更细。</b></div>
</div></div>
<p class="small">完整推导、表格与复现见 <b>报告_MACD_KDJ配合.md</b>；数据表见 <b>MACD_KDJ配合_数据表.xlsx</b>。图表：图26-图34。</p>
</section>

</main><div class="footer wrap">数据：baostock 本地 SQLite（后复权，含退市）；2010-01-01 至 2026-10-28。研究口径见报告第 1 节。</div>
</body></html>"""
    out = os.path.join(ROOT, "MACD_KDJ配合_可视化.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("html", out, len(html) / 1024, "KB")
    return html


def style_sheet(ws, df, pct_cols=(), int_cols=(), t_cols=(), q_cols=(), year_cols=()):
    fill = PatternFill("solid", fgColor="13233B")
    for c in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for i, col in enumerate(df.columns, 1):
        letter = get_column_letter(i)
        width = max(len(str(col)) * 1.6, 10)
        for v in df[col].head(80):
            width = max(width, len(str(v)) * 1.05)
        ws.column_dimensions[letter].width = min(max(width, 9), 40)
        if col in pct_cols:
            for r in range(2, len(df) + 2):
                ws.cell(row=r, column=i).number_format = "0.00%"
        elif col in t_cols:
            for r in range(2, len(df) + 2):
                ws.cell(row=r, column=i).number_format = "0.00"
        elif col in q_cols:
            for r in range(2, len(df) + 2):
                ws.cell(row=r, column=i).number_format = "0.0000"
        elif col in int_cols or col in year_cols:
            fmt = "0" if col in year_cols else "#,##0"
            for r in range(2, len(df) + 2):
                ws.cell(row=r, column=i).number_format = fmt
        for r in range(2, len(df) + 2):
            ws.cell(row=r, column=i).alignment = Alignment(vertical="center")


def build_excel():
    four = read("KDJ_状态_四态.csv")
    cond = read("KDJ_状态_条件.csv").rename(columns={
        "多头t": "多头t值", "空头t": "空头t值", "多空差t": "多空差t值",
        "MACD增量t": "MACD增量t值", "KDJ增量_多t": "KDJ增量多t值", "KDJ增量_空t": "KDJ增量空t值",
        "DiDt": "交互t值", "多头q": "多头q值", "MACD增量q": "MACD增量q值",
        "KDJ增量_多q": "KDJ增量多q值", "DiDq": "交互q值", "DiD": "交互项",
        "KDJ增量_多": "KDJ增量多", "KDJ增量_空": "KDJ增量空",
        "KDJ增量_多判定": "KDJ增量多判定", "KDJ增量_空判定": "KDJ增量空判定"})
    did = read("KDJ_状态_四态DiD.csv").rename(columns={"t": "t值"})
    em = read("KDJ_事件_MACD金叉_q.csv")
    emi = read("KDJ_事件_MACD金叉增量_q.csv").rename(columns={"A毛": "条件组毛超额", "B毛": "对照组毛超额"})
    ek = read("KDJ_事件_KDJ金叉_q.csv")
    eki = read("KDJ_事件_KDJ金叉增量_q.csv").rename(columns={"A毛": "条件组毛超额", "B毛": "对照组毛超额"})
    hz = read("KDJ_事件_期限.csv").rename(columns={"t": "t值"})
    yr = read("KDJ_事件_逐年差异.csv").rename(columns={"t": "t值"})
    param = read("KDJ_参数稳健.csv").rename(columns={"A": "条件组均值", "B": "对照组均值", "t": "t值"})
    wkp = read("KDJ_周线极值_参数.csv").rename(columns={"early": "前期", "late": "后期", "n": "信号数", "days": "交易日", "mean": "日均超额", "t": "t值", "KDJ参数": "周线KDJ参数"})
    wky = read("KDJ_周线极值_逐年.csv").rename(columns={"t": "t值"})
    k20 = read("KDJ_K20解剖.csv").rename(columns={"t": "t值"})
    k20y = read("KDJ_K20逐年.csv").rename(columns={"t": "t值"})
    k20c = read("KDJ_K20控制.csv").rename(columns={"A": "条件组均值", "B": "对照组均值", "t": "t值"})
    k20m = read("KDJ_K20匹配.csv").rename(columns={"t": "t值"})
    fm = read("探索_FamaMacBeth.csv").rename(columns={"平均R2": "平均R方"})
    hz2 = hz.copy()

    out = os.path.join(ROOT, "MACD_KDJ配合_数据表.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        four.to_excel(w, sheet_name="四态", index=False)
        style_sheet(w.sheets["四态"], four,
                    pct_cols=["日均超额", "中位", "前期", "后期", "正日占比"],
                    int_cols=["股票日", "交易日", "年数", "正年数", "有效年数", "有效正年数"],
                    t_cols=["t值"])
        cond.to_excel(w, sheet_name="KDJ条件全表", index=False)
        style_sheet(w.sheets["KDJ条件全表"], cond,
                    pct_cols=["多头日均", "多头中位", "多头前期", "多头后期", "空头日均", "空头中位",
                              "空头前期", "空头后期", "多空差", "MA60多空差", "MACD增量",
                              "KDJ增量多", "KDJ增量空", "交互项"],
                    int_cols=["多头股票日", "空头股票日", "共同日", "多头年数", "多头正年数",
                              "多头有效年数", "多头有效正年数"],
                    t_cols=["多头t值", "空头t值", "多空差t值", "MACD增量t值", "KDJ增量多t值", "KDJ增量空t值", "交互t值"],
                    q_cols=["多头q值", "MACD增量q值", "KDJ增量多q值", "交互q值"])
        did.to_excel(w, sheet_name="四态交互", index=False)
        style_sheet(w.sheets["四态交互"], did, pct_cols=["均值"], t_cols=["t值"], int_cols=["共同日"])
        em.to_excel(w, sheet_name="MACD金叉过滤", index=False)
        style_sheet(w.sheets["MACD金叉过滤"], em,
                    pct_cols=["毛超额", "净超额", "中位", "前期", "后期", "正日占比"],
                    int_cols=["信号数", "交易日", "年数", "正年数"], t_cols=["t值"], q_cols=["q值"])
        emi.to_excel(w, sheet_name="MACD金叉增量", index=False)
        style_sheet(w.sheets["MACD金叉增量"], emi, pct_cols=["条件组毛超额", "对照组毛超额", "差值"],
                    t_cols=["t值"], q_cols=["q值"], int_cols=["共同日"])
        ek.to_excel(w, sheet_name="KDJ金叉过滤", index=False)
        style_sheet(w.sheets["KDJ金叉过滤"], ek,
                    pct_cols=["毛超额", "净超额", "中位", "前期", "后期", "正日占比"],
                    int_cols=["信号数", "交易日", "年数", "正年数"], t_cols=["t值"], q_cols=["q值"])
        eki.to_excel(w, sheet_name="KDJ金叉增量", index=False)
        style_sheet(w.sheets["KDJ金叉增量"], eki, pct_cols=["条件组毛超额", "对照组毛超额", "差值"],
                    t_cols=["t值"], q_cols=["q值"], int_cols=["共同日"])
        hz2.to_excel(w, sheet_name="事件期限", index=False)
        style_sheet(w.sheets["事件期限"], hz2, pct_cols=["毛超额", "净超额"],
                    t_cols=["t值"], int_cols=["信号数"])
        yr.to_excel(w, sheet_name="逐年差异", index=False)
        style_sheet(w.sheets["逐年差异"], yr, pct_cols=["差值"], t_cols=["t值"],
                    int_cols=["年份", "共同日"], year_cols=["年份"])
        param.to_excel(w, sheet_name="参数稳健", index=False)
        style_sheet(w.sheets["参数稳健"], param, pct_cols=["条件组均值", "对照组均值", "差值"],
                    t_cols=["t值"], int_cols=["共同日"])
        wkp.to_excel(w, sheet_name="周线极值参数", index=False)
        style_sheet(w.sheets["周线极值参数"], wkp, pct_cols=["日均超额", "前期", "后期"],
                    t_cols=["t值"], int_cols=["周线KDJ参数", "周期", "信号数", "交易日"])
        wky.to_excel(w, sheet_name="周线极值逐年", index=False)
        style_sheet(w.sheets["周线极值逐年"], wky, pct_cols=["日均超额"],
                    t_cols=["t值"], int_cols=["年份", "股票日", "交易日"], year_cols=["年份"])
        k20.to_excel(w, sheet_name="K20解剖", index=False)
        style_sheet(w.sheets["K20解剖"], k20, pct_cols=["均值", "中位", "修剪均值"],
                    t_cols=["t值"], int_cols=["股票日", "交易日"])
        k20y.to_excel(w, sheet_name="K20逐年", index=False)
        style_sheet(w.sheets["K20逐年"], k20y, pct_cols=["日均超额", "中位"],
                    t_cols=["t值"], int_cols=["年份", "交易日", "股票日"], year_cols=["年份"])
        k20c.to_excel(w, sheet_name="K20控制", index=False)
        style_sheet(w.sheets["K20控制"], k20c, pct_cols=["条件组均值", "对照组均值", "差值"],
                    t_cols=["t值"], int_cols=["A股日", "B股日", "共同日"])
        k20m.to_excel(w, sheet_name="K20匹配", index=False)
        style_sheet(w.sheets["K20匹配"], k20m, pct_cols=["差值", "低K均值", "匹配均值"],
                    t_cols=["t值"], int_cols=["年份", "交易日"], year_cols=["年份"])
        fm.to_excel(w, sheet_name="FamaMacBeth", index=False)
        style_sheet(w.sheets["FamaMacBeth"], fm, pct_cols=["均值"], t_cols=["t值"],
                    int_cols=["交易日"], q_cols=["平均R方"])
    print("excel", out)
    return out


def add_nav():
    path = os.path.join(ROOT, "MACD有效性_可视化.html")
    s = open(path, encoding="utf-8").read()
    old = '<a href="条件有效性_可视化.html">条件判断</a>'
    new = '<a href="条件有效性_可视化.html">条件判断</a><a href="MACD_KDJ配合_可视化.html">MACD×KDJ</a>'
    if "MACD_KDJ配合_可视化.html" not in s:
        s = s.replace(old, new)
        open(path, "w", encoding="utf-8").write(s)
        print("nav added")
    else:
        print("nav exists")


if __name__ == "__main__":
    build_html()
    build_excel()
    add_nav()
