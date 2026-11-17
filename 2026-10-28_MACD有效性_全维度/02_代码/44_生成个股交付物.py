#!/usr/bin/env python3
"""44 生成个股适配性 HTML / Excel，并在主 HTML 中加入口"""
import base64
import os
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"
RES = os.path.join(ROOT, "03_结果")
FIG = os.path.join(ROOT, "04_图表")

KEY_EVENT = ["换手20", "特质波动250", "规模_log", "60日涨幅", "波动20", "ER60", "VR20", "AC1_60", "距250日高", "PE_TTM", "PB_MRQ"]
KEY_TIMING = ["VR20", "ER60", "AC1_60", "换手20", "规模_log", "波动20", "特质波动250", "MA60择时250", "动量12_1", "60日涨幅", "距250日高", "偏度250"]


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def img64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def fmt(v, col):
    if pd.isna(v):
        return "—"
    c = str(col)
    if any(x in c for x in ["日t", "股票t", " t", "t值"]) or c in ("t",):
        return f"{v:.2f}"
    if "q值" in c or "p值" in c:
        return f"{v:.3f}"
    if c in ("年换手", "换手"):
        return f"{v:.1f}"
    if "夏普" in c:
        return f"{v:.3f}"
    if "单调性" in c:
        return f"{v:.2f}"
    if any(x in c for x in ["年份", "事件数", "股票数", "期数", "天数", "分位", "类型"]):
        return f"{v:.0f}"
    if any(x in c for x in ["%", "率", "超额", "年化", "回撤", "暴露", "Q5", "Q1", "Q5Q1", "均值", "中位", "系数", "实际", "零假", "每期差", "平均", "相位"]):
        return f"{v*100:+.2f}%"
    if "相位期数" in c:
        return f"{v:.1f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


def table(df, cols=None, max_rows=40, cls="data"):
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df = df.head(max_rows)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        body.append("<tr>" + "".join(f"<td>{fmt(v, c)}</td>" for c, v in zip(df.columns, r.values)) + "</tr>")
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def fig_block(name, cap):
    return f'<figure><img src="data:image/png;base64,{img64(name)}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


def overview(df):
    rows = []
    for _, r in df.iterrows():
        v, m = r["值"], str(r["指标"])
        if "夏普" in m:
            s = f"{v:.3f}"
        elif "股票数" in m:
            s = f"{v:.0f}"
        elif any(x in m for x in ["中位", "均值", "比例", "成本"]):
            s = f"{v*100:+.2f}%"
        else:
            s = f"{v:.4f}"
        rows.append(dict(期间=r["期间"], 指标=m, 数值=s))
    return pd.DataFrame(rows)


def translate(df):
    m = {
        "code": "股票代码", "t": "t值", "beta": "市场beta",
        "event_ex_mkt": "金叉超额_市场中性", "event_ex_style": "金叉超额_风格中性",
        "event_ex_ind": "金叉超额_行业中性", "event_ex_own": "金叉超额_自身基准",
        "event_median_style": "金叉中位_风格中性", "event_hit": "金叉胜率",
        "event_std": "金叉标准差", "event_t": "金叉t值", "event_fwd20": "金叉20日均值",
        "event_mfe20": "金叉20日MFE", "event_mae20": "金叉20日MAE",
        "event_first": "首个信号日", "event_last": "末个信号日",
    }
    return df.rename(columns={k: v for k, v in m.items() if k in df.columns})


def build_html():
    ev = read("最终_个股适配_事件价差.csv")
    ev = ev[ev["特征"].isin(KEY_EVENT)]
    tm = read("最终_个股适配_择时价差.csv")
    tm = tm[tm["特征"].isin(KEY_TIMING)]
    pers = read("最终_个股适配_持续性.csv")
    ex = read("最终_个股适配_事前价差.csv")
    oos = read("最终_个股适配_OOS.csv")
    phase = read("最终_个股适配_配对相位.csv")
    port = read("最终_个股适配_组合_可成交.csv")
    star = read("最终_个股适配_科创板.csv")
    stary = read("最终_个股适配_科创板逐年.csv")
    stary = stary[stary["年份"] >= 2020]
    ind = read("最终_个股适配_行业_择时_明细.csv")
    typ = read("最终_个股适配_类型择时.csv")
    nullp = read("最终_个股适配_零假_可成交.csv")

    port_all = port[(port["分组"] == "全市场")]
    star_p = star[(star["板块"] == "科创板")]
    ind_late = ind[ind["期间"] == "后期"].sort_values("净超同暴露", ascending=False).head(12)
    typ_late = typ[typ["期间"] == "后期"]

    css = """
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
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.tw{overflow:auto;margin:12px 0;border:1px solid var(--line);border-radius:10px}
table.data{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
table.data th{background:#f8fafc;color:#475467;font-weight:600;text-align:right;padding:8px 9px;border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0}
table.data td{padding:7px 9px;border-bottom:1px solid #f0f2f5;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
table.data td:first-child,table.data th:first-child{text-align:left}
table.data tr:hover td{background:#fbfcfe}
figure{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px}
figure img{width:100%;display:block;border-radius:6px}figcaption{font-size:12.5px;color:var(--muted);padding:8px 4px 2px}
ul{margin:8px 0 8px 20px;color:#344054}li{margin:5px 0}
.small{font-size:12.5px;color:var(--muted)}
.footer{color:var(--muted);font-size:12.5px;padding:30px 0 50px;text-align:center}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:2px 8px;border-radius:6px}
.badge.bad{background:#fdecea;color:#a93226}.badge.good{background:#e8f6f2;color:#1b6f5c}.badge.warn{background:#fdf3e3;color:#8a5a12}
@media(max-width:900px){.hero-stats{grid-template-columns:repeat(2,1fr)}.grid2,.grid3{grid-template-columns:1fr}header h1{font-size:27px}}
"""

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MACD 个股适配性：哪些股票适合？</title><style>{css}</style></head><body>
<header><div class="wrap">
<div class="kicker">MACD 有效性 · 追加研究 · 2010–2026</div>
<h1>个股适配性：MACD 到底适合哪些股票？</h1>
<p class="sub">把 MACD 从“全市场信号”下沉到“单只股票”，逐股检验金叉事件与 DIF&gt;DEA 状态择时。结论不是“某些股票适合”，而是：<b>事前可识别的正子集不存在</b>；看起来好的低 60 日/低波子集，是反转效应，金叉反而是拖累；唯一例外是科创板 2022–2026 的 regime 口袋。</p>
<div class="hero-stats">
<div class="kpi"><div class="v bad">41.7%</div><div class="l">逐股金叉超额为正（5,381 只）</div></div>
<div class="kpi"><div class="v bad">-1.46%</div><div class="l">逐股择时净超同暴露中位</div></div>
<div class="kpi"><div class="v bad">0.016</div><div class="l">金叉前期→后期相关</div></div>
<div class="kpi"><div class="v warn">+0.1%</div><div class="l">全市场择时 alpha（可成交）</div></div>
<div class="kpi"><div class="v bad">-1.2% vs +7.1%</div><div class="l">全 A 金叉 vs 全 A 等权</div></div>
<div class="kpi"><div class="v good">+8.0%</div><div class="l">科创板 2022–26 择时 alpha</div></div>
</div></div></header>
<nav><div class="wrap">
<a href="#conclusion">结论</a><a href="#dist">逐股分布</a><a href="#features">特征分层</a><a href="#persist">事前识别</a>
<a href="#pair">配对对照</a><a href="#portfolio">组合层</a><a href="#star">科创板例外</a><a href="#industry">行业/类型</a>
<a href="#action">操作建议</a><a href="#method">方法</a>
</div></nav>
<main class="wrap">

<section id="conclusion"><h2>先看结论</h2>
<div class="callout bad"><b>没有“适合 MACD 的股票子集”。</b> 逐股金叉超额中位 <b>-0.29%/20 日</b>，只有 41.7% 为正；逐股状态择时净超同暴露中位 <b>-1.46%/年</b>，只有 41.9% 为正（5,391 只可评估）。更关键的是不持久：事件前期→后期相关 0.016，择时 -0.044；事件劈半信度 0.237，择时前期 0.041、后期 -0.068。</div>
<div class="grid2">
<div>
<h3>看起来“好”的特征</h3>
<ul>
<li><b>金叉事件少亏</b>：低换手、低特质波动、低 60 日涨幅、非小市值、低波、低偏度。但只是“少亏”：Q5-Q1 最大 0.2%–0.4%/20 日，最好的 Q5 也在 0 附近，扣 0.15% 成本后为负。</li>
<li><b>状态择时同期强</b>：高 VR20、ER60、AC1、MA60 择时边际；Q5-Q1 早期 +45%/+37%/+17%/+14%，后期 +16%/+7%/+3%/+6%。</li>
<li><b>但趋势性不持久</b>：VR20/ER60 前期→后期稳定性仅 0.087/0.127；用前期 VR20 选后期，Q5-Q1 反而 -1.5%/年（t=-3.3），Q5 绝对 -2.5%/年。</li>
</ul>
</div>
<div>
<h3>配对对照：金叉是拖累</h3>
<ul>
<li>20 日 cohort、20 相位平均、可成交：全 A 等权 <b>7.10%</b>；全 A 金叉 <b>-1.17%</b>。</li>
<li>低 60 日全 A <b>11.01%</b>；低 60 日 + 金叉 <b>4.54%</b>；低 60 日全 A 中再要求金叉 <b>4.98%</b>。</li>
<li>低波全 A <b>7.59%</b>；低波 + 金叉 <b>4.23%</b>；低波全 A + 金叉 <b>2.70%</b>。</li>
</ul>
</div></div>
<div class="callout warn"><b>唯一例外：</b>科创板 2022–2026 DIF&gt;DEA 可成交 net 12.1%/年 vs 同暴露 4.0%，alpha <b>+8.0%</b>，夏普 0.611 vs 0.312，平移零假 p≈0.01；但 2019–2021 为 <b>-6.7%</b>，2023 年 -2.8%，MA60 同期只有 +0.2%。这是 regime/行业口袋，不是“科创板股票都适合 MACD”。</div>
<div class="callout"><b>一句话：</b>不要用 MACD 金叉选股；想赚低 60 日/低波反转，就直接做反转，不要等金叉。个股 MACD 择时只能当降暴露/降回撤的风险 overlay，对同暴露没有 alpha。</div>
</section>

<section id="dist"><h2>逐股分布：大多数为负</h2>
{fig_block("图13_个股分布.png", "图13：逐股金叉 20 日风格中性超额（左）与逐股 DIF>DEA 择时净超同暴露（右）。中位分别为 -0.29% 和 -1.46%。")}
<div class="grid2">
<div><h3>金叉事件</h3>{table(overview(read("最终_个股适配_总览.csv").query("项目=='金叉逐股超额'")), ["期间","指标","数值"], 20)}</div>
<div><h3>状态择时</h3>{table(overview(read("最终_个股适配_总览.csv").query("项目=='状态择时逐股'")), ["期间","指标","数值"], 20)}</div>
</div>
<div class="callout bad"><b>不是“少数股票很好、多数一般”，而是“大多数股票为负”。</b> 金叉逐股平均 -0.17%、中位 -0.29%；择时净超同暴露中位 -1.46%/年，成本中位 1.48%/年；前期 56.2% 为正（2,960 只可评估）、后期 36.5%（5,361 只）——这是 regime，不是个股属性。事件劈半信度 0.237；择时劈半信度前期 0.041、后期 -0.068。</div>
</section>

<section id="features"><h2>特征分层：哪些股票“看起来”适合？</h2>
<h3>金叉事件：Q5-Q1（风格中性，20 日）</h3>
{table(ev, ["特征","期间","Q5平均","Q1平均","Q5-Q1","日t","q值"], 40)}
<div class="callout"><b>稳定为负</b>：高换手、高特质波动、小市值、高 60 日涨幅、高偏度。<b>不稳定</b>：ER60、VR20、AC1、距高、PE/PB。最好的 Q5 也只有 0 附近（规模 +0.03%、换手 Q1 +0.07%、60 日 Q1 +0.04%），全部低于 0.15% 成本。</div>
<h3>状态择时：同期 Q5-Q1（毛超同暴露）</h3>
{table(tm, ["特征","期间","Q5毛超","Q1毛超","Q5-Q1毛超","股票t","Q5-Q1净超","q值"], 40)}
<div class="callout warn"><b>同期最强的是“趋势性”</b>：VR20、ER60、AC1、MA60 择时边际；高换手/高波/小盘/高动量在早期也强。但这是“这段时间走得有趋势”，不是“这只股票天生适合”。低 60 日、远离 250 日高（深跌）在前后期都更好，是反转/超跌，不是趋势。</div>
{fig_block("图15_稳定性与效果.png", "图15：横轴是特征的前期→后期稳定性，纵轴是择时/事件效应。同期最强的 VR20/ER60 几乎不持久；稳定的换手/规模/波动只给出“少亏”。")}
</section>

<section id="persist"><h2>事前识别：过去好 ≠ 未来好</h2>
{fig_block("图14_持续性.png", "图14：逐股金叉（左）与逐股择时（右）的前期 vs 后期。相关系数 0.016 / -0.044，几乎不相关。")}
<div class="grid2">
<div><h3>特征稳定性</h3>{table(pers.assign(值=pers["值"].map(lambda x: f"{x:.3f}")), ["对象","指标","值"], 30)}</div>
<div><h3>前期特征 → 后期效果（Q5-Q1）</h3>{table(ex, ["对象","特征","Q5Q1","t","单调性"], 40)}</div>
</div>
{fig_block("图16_VR20分位.png", "图16：左图同期 VR20 五分位，前期/后期都单调；右图用前期 VR20 选后期，Q5-Q1 反而 -1.5%/年、Q5 绝对 -2.5%/年。")}
<h3>OOS 线性评分</h3>
{table(oos, None, 20)}
<div class="callout bad"><b>“过去在这只股票上好用”几乎不预测未来。</b> 事件 0.016 / 择时 -0.044；劈半信度 0.24 / 0.04；VR20/ER60 稳定性 0.087/0.127。前期 VR20/ER60/换手/波动选后期，方向全部反转。唯一稳定正的是规模（大市值），但后期 Q5 毛超仍 -0.62%/年，只是“少亏”。</div>
</section>

<section id="pair"><h2>配对对照：金叉是反转/低波子集的拖累</h2>
{fig_block("图17_配对对照.png", "图17：20 日 cohort、20 相位平均、可成交口径。全 A 金叉 -1.2%；低 60 日全 A 11.0%，加金叉后 4.5%；低波全 A 7.6%，加金叉后 4.2%。")}
{table(phase, ["对象","相位均值","相位最小","相位最大","相位期数"], 20)}
<div class="callout bad"><b>低 60 日/低波是反转/低风险效应；MACD 金叉不是触发器，是拖累。</b> 相位 0 的配对差：全 A 金叉 vs 全 A -0.58%/20 日（t=-3.07）；低 60 日+金叉 vs 低 60 日 -0.40%/20 日（t=-1.61）；低波+金叉 vs 低波 -0.25%/20 日（t=-1.32）。随机抽同样只数的低 60 日组合全期 9.81%/年，远高于低 60 日金叉 4.54%。</div>
</section>

<section id="portfolio"><h2>组合层：降暴露，不是 alpha</h2>
{fig_block("图18_组合对比.png", "图18：全市场 DIF>DEA 可成交口径。对买入持有 6.3% vs 9.8%、回撤 -30.6% vs -62.3%；对同暴露 alpha 仅 +0.1%/年（p=0.099），前期 +1.0%（p=0.228），后期 -0.7%（p=0.168）。")}
{table(port_all, ["策略","区间","年化净","年化同暴露","净超同暴露","年化持有","夏普净","夏普同暴露","回撤净","回撤同暴露","年换手"], 20)}
{table(nullp, None, 10)}
<div class="callout"><b>动态择时没有跑赢“固定 52% 仓位”。</b> 全期回撤 -30.6% vs 同暴露 -34.8%（略好），后期 -22.2% vs -21.3%（略差）；夏普全期 0.448 vs 0.463、后期 0.366 vs 0.446。成本约 1.6%/年、年换手约 20 次。未剔除一字涨停的版本全期 +0.73%，可成交版本降到 +0.1%——一字涨停入场贡献了大部分表面 alpha。</div>
</section>

<section id="star"><h2>唯一例外：科创板 2022–2026</h2>
{fig_block("图19_科创板.png", "图19：左图科创板 DIF>DEA vs MA60 的净超同暴露；右图逐年（2020 起；2026 截至 10 月）。2022–2026 +8.0%，但 2019–2021 -6.7%，2023 -2.8%。")}
{table(star_p, ["区间","策略","年化净","年化同暴露","净超同暴露","年化持有","夏普净","夏普同暴露","回撤净","回撤同暴露"], 10)}
{table(stary, ["年份","年化净","年化同暴露","净超同暴露","年化持有","夏普净","夏普同暴露","回撤净","回撤同暴露"], 10)}
<div class="callout warn"><b>这是 regime，不是稳定股票属性。</b> 2022–2026 alpha +8.0%（p≈0.01），但 2019–2021 -6.7%，2023 年 -2.8%；MA60 同期只有 +0.2%，MA200 regime 过滤反而把 alpha 打到 -0.4%。行业集中在电子/半导体（158 只）与机械（96 只）。可作为研究线索/regime overlay，不能直接实盘。</div>
</section>

<section id="industry"><h2>行业与股票类型：没有稳定正轴</h2>
<div class="grid2">
<div><h3>行业择时（后期，净超同暴露）</h3>{table(ind_late, ["行业","股票数","净超同暴露","毛超同暴露","夏普差","t","q值"], 12)}
<p class="small">31 个行业 BH 后没有一个显著；综合 +2.52%（q=0.154）、电子 +0.81%（q=0.238）、房地产 +0.19%（q=0.775）为正但不显著。</p></div>
<div><h3>股票类型（前期聚类，后期择时）</h3>{table(typ_late, ["类型","股票数","净超同暴露","毛超同暴露","夏普差","回撤超同暴露"], 10)}
<p class="small">类型 1（小盘/高换手/高波/高 VR）前期 +13.8%、后期 -3.4%；类型 2（高 ER/VR/AC1）前期 +6.8%、后期 -1.4%；没有一类在后期净为正。</p></div>
</div>
</section>

<section id="action"><h2>结论与操作建议</h2>
<div class="callout bad"><b>直接回答：没有事前可识别、扣费后稳定为正的“MACD 适合股票”子集。</b></div>
<ul>
<li><b>不要用金叉选股。</b> 更不要在低 60 日/低波股票上等金叉：直接做低 60 日/低波反转，收益更高（11.0% vs 4.5%；7.6% vs 4.2%），成本更低。</li>
<li><b>个股 MACD 择时只应作为降暴露/降回撤的风险 overlay。</b> 它把回撤从 -62% 降到 -30%，但扣费后对同暴露没有 alpha；直接持有 52% 仓位效果差不多。</li>
<li><b>不要用“过去在这只股票上好用”来选股票。</b> 事件 0.016、择时 -0.044；劈半信度 0.24/0.04。过去好 = 噪声/regime。</li>
<li><b>不要用同期特征（VR20/ER60/换手/小盘）做事前选股。</b> 同期强、跨期不持久；用前期值选后期，方向反转。</li>
<li><b>唯一值得继续研究的是科创板 2022–2026 的择时口袋。</b> 需行业中性、regime 条件、2019–2021 失败解释、样本外确认；不能直接实盘。</li>
<li><b>参数/特征/周期不要调。</b> 个股层没有稳定子集；调参只会把噪声拟合得更漂亮。</li>
</ul>
</section>

<section id="method"><h2>方法与局限</h2>
<div class="grid2">
<div><h3>口径</h3><ul>
<li>2008–2026 全 A，5,507 只（含退市），信号评估 2010 起；金叉 519,111 个可评估事件。</li>
<li>事件：T+1 开盘买，持有 20 日，0.15% 往返；市场/行业/风格/自身四种中性超额；主表风格中性。</li>
<li>择时：DIF&gt;DEA，T+1 开盘建/平，open→open，0.075%/边；约 20 次/年，成本约 1.6%/年。</li>
<li>组合：可成交口径剔除一字涨停买不进/一字跌停卖不出；同暴露基准；循环平移零假。</li>
<li>cohort：每 20 日一个起点、20 个相位、持有 20 日；BH-FDR；FM；OOS 线性评分。</li>
</ul></div>
<div><h3>局限</h3><ul>
<li>逐股择时统计未剔除一字涨停入场；组合层已做可成交口径，结论以组合层为准。</li>
<li>ST 无法做风格中性；之前 ST 原始超额 +1.3%/年、t=0.32，不构成实用子集。</li>
<li>科创板 2019-07 起，2022–2026 只有 4 年多；2019–2021 失败说明 regime 不稳定。</li>
<li>行业/板块与 regime 混淆；科创板 alpha 集中在电子/半导体。</li>
<li>大量检验；正文只强调跨期同号、BH 显著、且幅度超过成本的结果。</li>
<li>容量：低 60 日全 A 约 640 只；科创板平均持仓约 237 只；未做冲击成本模型。</li>
</ul></div></div>
<p class="small">完整报告：<a href="报告_个股适配性.md">报告_个股适配性.md</a>；数据表：<a href="个股适配性_数据表.xlsx">个股适配性_数据表.xlsx</a>；主报告：<a href="报告_MACD有效性.md">报告_MACD有效性.md</a>。</p>
</section>
<div class="footer"><p>2026-10-28_MACD有效性_全维度 · 个股适配性追加研究 · 数据截至 2026-10-28</p><p>本报告是统计研究，不构成投资建议。所有历史结果都不保证未来。</p></div>
</main></body></html>"""
    with open(os.path.join(ROOT, "个股适配性_可视化.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return html


def style_sheet(ws, df):
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(color="FFFFFF", bold=True, size=10)
    for c in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        width = max(10, min(34, max(len(str(x)) for x in [df.columns[c - 1]] + df.iloc[:, c - 1].astype(str).head(80).tolist()) + 2))
        ws.column_dimensions[get_column_letter(c)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def build_excel():
    out = os.path.join(ROOT, "个股适配性_数据表.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        notes = pd.DataFrame({
            "项目": ["研究", "数据", "事件", "择时", "成本", "组合", "零假", "注意"],
            "说明": [
                "MACD 个股适配性：逐股金叉事件 + DIF>DEA 状态择时 + 特征分层 + 事前检验 + 配对对照 + 组合",
                "2008-2026 全 A，5,507 只（含退市），信号评估 2010-01-01 起；baostock 后复权 + 不复权",
                "T 日收盘确认，T+1 开盘买，持有 20 日，扣 0.15% 往返；剔除一字涨停/停牌",
                "DIF>DEA 时持有，T+1 开盘建/平，open→open，0.075%/边；约 20 次/年",
                "事件 0.15% 往返；择时 0.075%/边；组合层可成交口径剔除一字板",
                "每 20 日 cohort，20 个相位；日频状态组合；同暴露基准；买入持有",
                "循环平移仓位序列，保留换手/暴露，打乱与收益的对应；100/200 次",
                "逐股择时统计未剔除一字涨停入场；ST 无风格中性；科创板历史短；不构成投资建议",
            ]})
        notes.to_excel(w, sheet_name="00_说明", index=False)
        sheets = {
            "01_总览": read("最终_个股适配_总览.csv"),
            "02_事件分位": read("最终_个股适配_事件分位.csv"),
            "03_事件价差": read("最终_个股适配_事件价差.csv"),
            "04_择时分位": read("最终_个股适配_择时分位.csv"),
            "05_择时价差": read("最终_个股适配_择时价差.csv"),
            "06_事前事件": read("最终_个股适配_事前事件.csv"),
            "07_事前价差": read("最终_个股适配_事前价差.csv"),
            "08_持续性": read("最终_个股适配_持续性.csv"),
            "09_事件方差": read("最终_个股适配_事件方差分解.csv"),
            "10_择时信度": read("最终_个股适配_择时信度.csv"),
            "11_配对主表": read("最终_个股适配_配对.csv"),
            "12_配对相位": read("最终_个股适配_配对相位.csv"),
            "13_组合可成交": read("最终_个股适配_组合_可成交.csv"),
            "14_零假": read("最终_个股适配_零假_可成交.csv"),
            "15_科创板": read("最终_个股适配_科创板.csv"),
            "16_科创板逐年": read("最终_个股适配_科创板逐年.csv"),
            "17_科创板零假": read("最终_个股适配_科创板零假.csv"),
            "18_行业择时": read("最终_个股适配_行业_择时_明细.csv"),
            "19_类型画像": read("最终_个股适配_类型画像.csv"),
            "20_类型事件": read("最终_个股适配_类型事件.csv"),
            "21_类型择时": read("最终_个股适配_类型择时.csv"),
            "22_FM事件": read("最终_个股适配_FM事件.csv"),
            "23_FM择时": read("最终_个股适配_FM择时.csv"),
            "24_OOS": read("最终_个股适配_OOS.csv"),
            "25_逐股事件": read("个股_事件表现.csv"),
            "26_逐股择时": read("个股_择时表现_分期.csv"),
            "27_逐股特征": read("个股_特征.csv"),
        }
        for name, df in sheets.items():
            df = translate(df)
            df.to_excel(w, sheet_name=name, index=False)
            style_sheet(w.book[name], df)
    return out


def append_main_excel():
    path = os.path.join(ROOT, "MACD有效性_数据表.xlsx")
    wb = load_workbook(path)
    names = ["29_个股适配_总览", "30_个股适配_事件", "31_个股适配_择时",
             "32_个股适配_配对", "33_个股适配_组合", "34_个股适配_科创板"]
    for n in names:
        if n in wb.sheetnames:
            del wb[n]
    wb.save(path)
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        dfs = {
            names[0]: read("最终_个股适配_总览.csv"),
            names[1]: read("最终_个股适配_事件价差.csv"),
            names[2]: read("最终_个股适配_择时价差.csv"),
            names[3]: read("最终_个股适配_配对相位.csv"),
            names[4]: read("最终_个股适配_组合_可成交.csv"),
            names[5]: read("最终_个股适配_科创板.csv"),
        }
        for n, df in dfs.items():
            df = translate(df)
            df.to_excel(w, sheet_name=n, index=False)
            style_sheet(w.book[n], df)


def update_main_html():
    path = os.path.join(ROOT, "MACD有效性_可视化.html")
    s = open(path, encoding="utf-8").read()
    if 'href="#stocks"' not in s:
        s = s.replace('<a href="#features">特征与结构</a><a href="#exit">离场</a>',
                      '<a href="#features">特征与结构</a><a href="#stocks">个股适配</a><a href="#exit">离场</a>')
    if '<section id="stocks">' not in s:
        sec = """
<section id="stocks">
<h2>追加：个股适配性——MACD 适合哪些股票？</h2>
<div class="callout bad"><b>没有事前可识别的正子集。</b> 逐股金叉超额中位 -0.29%/20 日（41.7% 为正），逐股 DIF&gt;DEA 择时净超同暴露中位 -1.46%/年（41.9% 为正）；前期→后期事件相关 0.016、择时 -0.044，劈半信度 0.24/0.04。看起来好的低 60 日/低波子集，是反转效应：低 60 日全 A 11.0%/年，加金叉后 4.5%；低波全 A 7.6%，加金叉后 4.2%。</div>
<div class="grid2">
<div class="callout warn"><b>唯一例外：</b>科创板 2022–2026 的 DIF&gt;DEA 择时，可成交 alpha +8.0%/年 vs 同暴露，p≈0.01；但 2019–2021 为 -6.7%，是 regime/行业口袋，不是稳定股票属性。</div>
<div class="callout"><b>组合层：</b>全市场 DIF&gt;DEA 可成交净 6.3%/年 vs 同暴露 6.2%（+0.1%，p=0.099）；对买入持有 9.8%/年，但回撤 -30.6% vs -62.3%。这是降暴露/降回撤，不是 alpha。</div>
</div>
<p>完整逐股、特征分层、事前检验、配对对照与组合结果见 <a href="个股适配性_可视化.html">个股适配性_可视化.html</a>；文字版 <a href="报告_个股适配性.md">报告_个股适配性.md</a>；数据表 <a href="个股适配性_数据表.xlsx">个股适配性_数据表.xlsx</a>。</p>
</section>

<section id="method">"""
        s = s.replace('<section id="method">', sec, 1)
    open(path, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    build_html()
    build_excel()
    append_main_excel()
    update_main_html()
    print("deliverables done")
