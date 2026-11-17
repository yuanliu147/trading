# -*- coding: utf-8 -*-
"""生成单文件 HTML 报告（图表内嵌 base64）。"""
import base64
import os
import pandas as pd
import numpy as np

ROOT = "/Users/admin/WorkSpace/trading/2026-11-03_603778_换手率区间"
RES = f"{ROOT}/03_结果"
FIG = f"{ROOT}/04_图表"
OUT = f"{ROOT}/603778_换手率区间.html"

hist = pd.read_pickle(f"{ROOT}/01_数据/603778_hist.pkl")
full = pd.read_pickle(f"{ROOT}/01_数据/603778_full.pkl")
band = pd.read_csv(f"{RES}/换手区间统计.csv")
phase = pd.read_csv(f"{RES}/阶段换手.csv")
events = pd.read_csv(f"{RES}/极端换手事件.csv")
yearly = pd.read_csv(f"{RES}/年度换手.csv")

BAND_COLORS = {
    "<0.5": "#e8e1d2", "0.5-1": "#d9c9a8", "1-2": "#c9b077", "2-3.5": "#c69a4e",
    "3.5-5": "#b9772f", "5-8": "#b25a35", "8-12": "#a9442e", "12-20": "#93302a", ">20": "#6f1f1d",
}


def img(name):
    with open(f"{FIG}/{name}", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


def fmt(v, d=1, suffix=""):
    if pd.isna(v):
        return "—"
    return f"{v:,.{d}f}{suffix}"


def pct(v, d=1):
    if pd.isna(v):
        return "—"
    s = f"{v:+.{d}f}%"
    return f'<span class="{"pos" if v > 0 else "neg" if v < 0 else ""}">{s}</span>'


def win(v):
    if pd.isna(v):
        return "—"
    return f'<span class="{"pos" if v >= 50 else "neg"}">{v:.0f}%</span>'


def num(v, d=1):
    return "—" if pd.isna(v) else f"{v:,.{d}f}"


# 当前同类样本（三组对照）
conds = {
    "全部交易日": pd.Series(True, index=hist.index),
    "深度回撤>60% + 换手<5%": (hist.dd250 < -60) & (hist.turn < 5),
    "深度回撤>60% + 换手5-8%": (hist.dd250 < -60) & (hist.turn >= 5) & (hist.turn < 8),
    "250日低位<30 + 换手3.5-5%": (hist.pos250 < 30) & (hist.turn >= 3.5) & (hist.turn < 5),
    "250日低位<30 + 换手5-8%": (hist.pos250 < 30) & (hist.turn >= 5) & (hist.turn < 8),
    "换手5-8%（全部）": (hist.turn >= 5) & (hist.turn < 8),
    "换手3.5-5%（全部）": (hist.turn >= 3.5) & (hist.turn < 5),
}
analog_rows = []
for name, m in conds.items():
    for sample, sub in [("全样本", hist[m]),
                        ("剔除2026", hist[m & (hist.year < 2026)]),
                        ("2018-24", hist[m & (hist.year >= 2018) & (hist.year <= 2024)])]:
        analog_rows.append((name, sample, len(sub),
                            sub.f20.median(), (sub.f20 > 0).mean() * 100 if len(sub) else np.nan,
                            sub.f60.median(), (sub.f60 > 0).mean() * 100 if len(sub) else np.nan))

# ---------- 表格 HTML ----------
band_rows = []
for _, r in band.iterrows():
    b = str(r["区间(%)"])
    band_rows.append(f"""<tr>
<td><span class="swatch" style="background:{BAND_COLORS[b]}"></span><b>{b}%</b></td>
<td>{r['天数']:.0f}</td><td>{r['占比(%)']:.1f}%</td><td>{r['换手中位(%)']:.2f}%</td>
<td>{r['成交额中位(亿)']:.2f}</td><td>{pct(r['前20日涨跌中位(%)'])}</td>
<td>{r['250日位置中位(%)']:.0f}%</td><td>{r['站上均线20(%)']:.0f}%</td>
<td>{pct(r['T+20中位(%)'])}</td><td>{pct(r['T+20均值(%)'])}</td>
<td>{win(r['T+20胜率(%)'])}</td><td>{pct(r['20日最大浮亏中位(%)'])}</td>
<td class="left small">{r['主要年份']}</td></tr>""")

phase_rows = []
for _, r in phase.iterrows():
    phase_rows.append(f"""<tr><td class="left"><b>{r['阶段']}</b></td>
<td>{r['交易日']:.0f}</td><td>{r['换手中位(%)']:.1f}%</td><td>{r['换手P90(%)']:.1f}%</td>
<td>{r['换手最大(%)']:.1f}%</td><td>{pct(r['区间涨跌(%)'], 0)}</td>
<td>{r['成交额中位(亿)']:.2f}</td><td>{r['20日波动中位(%)']:.0f}%</td></tr>""")

event_rows = []
for _, r in events.iterrows():
    pos = "—" if pd.isna(r["结束250日位置(%)"]) else f"{r['结束250日位置(%)']:.0f}%"
    event_rows.append(f"""<tr><td class="left">{r['起始日']} → {r['结束日']}</td>
<td>{r['交易日']:.0f}</td><td>{r['换手中位(%)']:.1f}% / {r['换手最高(%)']:.1f}%</td>
<td>{pct(r['期间涨跌(%)'], 0)}</td><td>{pct(r['前20日涨跌(%)'], 0)}</td>
<td>{pos}</td><td>{pct(r['后20日(%)'], 0)}</td><td>{pct(r['后60日(%)'], 0)}</td></tr>""")

analog_rows_html = []
for name, sample, n, f20, w20, f60, w60 in analog_rows:
    analog_rows_html.append(f"""<tr><td class="left">{name}</td><td>{sample}</td><td>{n}</td>
<td>{pct(f20)}</td><td>{win(w20)}</td><td>{pct(f60)}</td><td>{win(w60)}</td></tr>""")

year_rows = []
for _, r in yearly.iterrows():
    year_rows.append(f"""<tr><td class="left">{int(r['年份'])}</td><td>{r['交易日']:.0f}</td>
<td>{r['换手中位(%)']:.1f}%</td><td>{r['换手P90(%)']:.1f}%</td><td>{r['换手最大(%)']:.1f}%</td>
<td>{r['成交额中位(亿)']:.2f}</td><td>{pct(r['全年涨跌(%)'], 0)}</td></tr>""")

last = full.iloc[-1]
factor = last.close_h / last.close

bar_cards = [
    ("<0.5", "地量 / 一字板", "长期下跌末端、无人问津；2016 年一字板也会缩量，需剔除。",
     "+2.9%", "60%", "唯一正期望区，但“地量复地量”。2018–20 持续了两年，不是买入信号，是等待区。"),
    ("0.5-1", "地量边缘", "2019–20 的常态，成交额仅 0.15 亿。",
     "+1.6%", "54%", "底部区域，但要等价格不再创新低、并放量站上均线20 才确认。"),
    ("1-2", "清淡", "下跌中继或底部构筑；2023–24 转型期常见。",
     "+0.2%", "50%", "中性区，方向由趋势决定，单看换手没有优势。"),
    ("2-3.5", "温和放量", "反弹初期、题材试盘，成交额约 0.6 亿。",
     "−2.4%", "42%", "“看起来活跃但不够强”，多数是反弹中继，不是主升。"),
    ("3.5-5", "弱活跃", "分歧加大、反弹失败；2023–24 为 −4.8%/38%，2018–20 为 −6.3%/30%。",
     "−3.7%", "34%", "全历史最差区间之一。没有站上均线20 和板块效应，容易继续阴跌。"),
    ("5-8", "高换手（当前）", "题材发酵、游资博弈，成交额约 1.6 亿；当前 6.4% 落在这里。",
     "−3.2%", "40%", "只有主升段才是“接力”，下跌/震荡段是“出货/失败反弹”。当前属于后者。"),
    ("8-12", "很高", "主升中继或顶部派发；涨停率 9.9%，跌停率 4.7%。",
     "−3.2%", "42%", "高波动博弈区：20 日最大浮盈中位 14.2%，最大浮亏 −12.9%。"),
    ("12-20", "天量", "主升加速、妖股、顶部；成交额约 4.6 亿。",
     "−2.7%", "45%", "高位（>70%）在 2025 主升段是接力（+36%/73%），2016–24 却是派发（−16.4%/24%）；低位多为失败反弹（−5.5%/24%）。"),
    (">20", "极端", "新股开板、高送转、妖股、恐慌/起爆；成交额约 7.6 亿。",
     "−3.9%", "42%", "最不可预测：2016/17 多为派发，2025–26 却是主升起点。只看换手会误判。"),
]
bar_cards_html = []
for b, title, scene, f20, w20, read in bar_cards:
    bar_cards_html.append(f"""<div class="bcard" style="--bc:{BAND_COLORS[b]}">
<div class="bcard-top"><span class="bchip">{b}%</span><span class="bname">{title}</span></div>
<p class="bscene">{scene}</p>
<div class="bstat"><span class="bv">{f20}</span><span class="bl">T+20 中位</span><span class="bv">{w20}</span><span class="bl">胜率</span></div>
<p class="bread">{read}</p></div>""")

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>603778 换手率区间全解</title>
<style>
:root{{
  --paper:#f2eee6; --panel:#fbf8f2; --card:#fffdf8; --ink:#211f1b; --ink2:#4d473d;
  --muted:#756d60; --faint:#9b9181; --line:#ded5c6; --line2:#c9bda9;
  --accent:#a93426; --accent-soft:#f3e3dc; --up:#b23a2b; --down:#2e6b55;
  --gold:#9a6a18; --blue:#3d5b72; --mono:"SFMono-Regular",Menlo,Consolas,monospace;
  --serif:"Songti SC","STSong","Noto Serif SC",Georgia,serif;
  --sans:"PingFang SC","Hiragino Sans GB","Noto Sans SC","Microsoft YaHei",sans-serif;
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:
  radial-gradient(circle at 10% 0%,rgba(169,52,38,.05),transparent 34%),
  radial-gradient(circle at 94% 6%,rgba(61,91,114,.05),transparent 30%),var(--paper);
  color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.78;
  -webkit-font-smoothing:antialiased}}
.page{{width:min(1280px,calc(100% - 40px));margin:26px auto 56px;background:var(--panel);
  border:1px solid var(--line2);box-shadow:0 28px 70px rgba(72,55,32,.11);position:relative}}
.masthead{{padding:46px 54px 30px;border-bottom:1px solid var(--line2);position:relative;overflow:hidden}}
.masthead::after{{content:"603778";position:absolute;right:10px;top:-46px;font-family:var(--serif);
  font-size:210px;line-height:1;letter-spacing:-10px;color:rgba(169,52,38,.045);pointer-events:none}}
.kicker{{display:flex;justify-content:space-between;gap:14px;font-family:var(--mono);
  font-size:10.5px;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);position:relative;z-index:1}}
h1{{font-family:var(--serif);font-size:46px;line-height:1.18;letter-spacing:-1px;margin:16px 0 12px;position:relative;z-index:1}}
.lede{{font-size:16px;color:var(--ink2);max-width:860px;position:relative;z-index:1}}
.meta{{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:20px;font-size:11.5px;color:var(--muted);position:relative;z-index:1}}
.meta b{{color:var(--ink2);font-weight:600}}
.kpis{{display:grid;grid-template-columns:repeat(6,1fr);background:var(--line);gap:1px;border-bottom:1px solid var(--line2)}}
.kpi{{background:var(--card);padding:16px 15px 14px}}
.kpi .lab{{font-family:var(--mono);font-size:9.5px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}}
.kpi .v{{font-family:var(--serif);font-size:25px;line-height:1.25;margin-top:4px;font-variant-numeric:tabular-nums}}
.kpi .s{{font-size:10.5px;color:var(--muted);margin-top:2px}}
.toc{{display:flex;flex-wrap:wrap;gap:6px 16px;padding:14px 54px;border-bottom:1px solid var(--line);font-size:11.5px}}
.toc a{{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}}
.toc a:hover{{color:var(--accent);border-color:var(--accent)}}
section{{padding:34px 54px 38px;border-bottom:1px solid var(--line)}}
.sec-head{{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:20px}}
.sec-num{{font-family:var(--mono);font-size:11px;letter-spacing:.12em;color:var(--accent)}}
h2{{font-family:var(--serif);font-size:27px;line-height:1.3;margin:0;letter-spacing:-.3px}}
h3{{font-family:var(--serif);font-size:19px;margin:26px 0 10px}}
p{{margin:9px 0}}
.lead{{font-size:15.5px;color:var(--ink2)}}
.figure{{margin:20px 0 6px;border:1px solid var(--line);background:var(--card);padding:9px}}
.figure img{{width:100%;display:block}}
.caption{{font-size:11.5px;color:var(--muted);margin-top:8px;line-height:1.6}}
.callout{{background:var(--card);border-left:3px solid var(--accent);padding:14px 18px;margin:18px 0;font-size:14px;color:var(--ink2)}}
.callout b{{color:var(--ink)}}
.callout.gold{{border-left-color:var(--gold)}}
.callout.green{{border-left-color:var(--down)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);background:var(--card);margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;min-width:760px}}
th{{background:#efe7d8;color:var(--ink2);font-weight:600;text-align:right;padding:9px 9px;
  border-bottom:1px solid var(--line2);white-space:nowrap;font-size:11.5px}}
td{{padding:7px 9px;border-bottom:1px solid #ece4d6;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;padding-left:12px}}
td.left,th.left{{text-align:left}}
td.small{{font-size:11px;color:var(--muted)}}
tr:nth-child(even) td{{background:rgba(255,255,255,.42)}}
tr:hover td{{background:#faf4e8}}
.pos{{color:var(--up);font-weight:600}}
.neg{{color:var(--down);font-weight:600}}
.swatch{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:1px}}
.bcard{{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--bc);padding:14px 15px 12px}}
.bcard-top{{display:flex;align-items:center;gap:9px}}
.bchip{{font-family:var(--mono);font-size:11px;font-weight:700;color:#fff;background:var(--bc);padding:2px 7px;border-radius:2px}}
.bname{{font-family:var(--serif);font-size:16px;font-weight:700}}
.bscene{{font-size:12px;color:var(--muted);line-height:1.55;margin:8px 0 10px;min-height:54px}}
.bstat{{display:flex;align-items:baseline;gap:6px;border-top:1px dashed var(--line);padding-top:8px}}
.bv{{font-family:var(--serif);font-size:19px;font-variant-numeric:tabular-nums}}
.bl{{font-size:9.5px;color:var(--muted);margin-right:8px}}
.bread{{font-size:12px;color:var(--ink2);line-height:1.6;margin:9px 0 0;border-top:1px dashed var(--line);padding-top:8px}}
.note{{font-size:11.5px;color:var(--muted);line-height:1.7}}
.pill{{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.04em;padding:1px 6px;border:1px solid var(--line2);color:var(--muted);border-radius:2px;margin-right:4px}}
.grade{{display:grid;grid-template-columns:54px 1fr;gap:0;border:1px solid var(--line);background:var(--card);margin:12px 0}}
.grade>div{{padding:10px 14px;border-bottom:1px solid var(--line);font-size:13px}}
.grade>div:nth-last-child(-n+2){{border-bottom:0}}
.grade .g{{font-family:var(--mono);font-weight:700;color:var(--accent);border-right:1px solid var(--line);text-align:center}}
.kv{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:14px 0}}
.kv>div{{background:var(--card);padding:12px 14px}}
.kv .k{{font-size:10px;color:var(--muted);font-family:var(--mono);letter-spacing:.05em}}
.kv .v{{font-family:var(--serif);font-size:19px;margin-top:2px}}
.trigger{{display:grid;grid-template-columns:24px 1fr;gap:10px;background:var(--card);border:1px solid var(--line);padding:12px 14px;margin:9px 0}}
.trigger .t{{font-family:var(--mono);font-size:12px;color:var(--accent);font-weight:700}}
.trigger b{{font-family:var(--serif);font-size:15px}}
footer{{padding:24px 54px 34px;font-size:11px;color:var(--muted);line-height:1.8}}
footer b{{color:var(--ink2)}}
@media(max-width:980px){{
  .kpis{{grid-template-columns:repeat(3,1fr)}} .grid3{{grid-template-columns:1fr 1fr}}
  .grid2{{grid-template-columns:1fr}} .kv{{grid-template-columns:1fr 1fr}}
  h1{{font-size:34px}} .masthead,section,footer,.toc{{padding-left:24px;padding-right:24px}}
  .masthead::after{{font-size:120px;top:-10px}}
}}
@media(max-width:620px){{
  .kpis{{grid-template-columns:1fr 1fr}} .grid3{{grid-template-columns:1fr}}
  .page{{width:100%;margin:0;border-left:0;border-right:0}}
  h1{{font-size:29px}} .lede{{font-size:14px}}
  .masthead,section,footer,.toc{{padding-left:16px;padding-right:16px}}
  section{{padding-top:24px;padding-bottom:28px}}
}}
</style>
</head>
<body>
<div class="page">

<div class="masthead">
  <div class="kicker"><span>个股研究 / 换手率结构</span><span>2026-11-03</span></div>
  <h1>603778 换手率区间全解</h1>
  <p class="lede">从单日 66.5% 到不足 0.5%，这只股票的历史换手率跨了近三个数量级。更关键的是：<b>同样的换手率，在主升段是接力，在崩塌段是出货</b>。本文把 2016 年以来的 2591 个交易日拆成九个区间，逐一对应价格位置、阶段与后续 20/60 日结果。</p>
  <div class="meta">
    <span><b>标的</b> 国晟科技（原乾景园林，603778.SH）</span>
    <span><b>数据</b> baostock 2015-12-31 → 2026-10-29；同花顺补 10-30、11-02</span>
    <span><b>收益</b> 后复权，T+1 开盘买入</span>
    <span><b>最新</b> 2026-11-02 收盘 6.92 元 / 换手 6.4%</span>
  </div>
</div>

<div class="kpis">
  <div class="kpi"><div class="lab">当前换手</div><div class="v">6.4%</div><div class="s">11-02，成交额 2.77 亿</div></div>
  <div class="kpi"><div class="lab">全历史分位</div><div class="v">73.5%</div><div class="s">历史中位仅 2.5%</div></div>
  <div class="kpi"><div class="lab">2026 年内分位</div><div class="v">18.9%</div><div class="s">本轮泡沫中的“缩量”</div></div>
  <div class="kpi"><div class="lab">250 日位置</div><div class="v">9%</div><div class="s">区间 4.11–34.27 元</div></div>
  <div class="kpi"><div class="lab">距 4/13 收盘高点</div><div class="v neg">−79.1%</div><div class="s">33.10 → 6.92；距盘中高 34.27 为 −79.8%</div></div>
  <div class="kpi"><div class="lab">估值</div><div class="v">PB 20.0</div><div class="s">PE(TTM) −8.3</div></div>
</div>

<div class="toc">
  <a href="#s1">01 全景</a><a href="#s2">02 分布</a><a href="#s3">03 九个区间</a>
  <a href="#s4">04 阶段决定含义</a><a href="#s5">05 极端事件</a><a href="#s6">06 当前定位</a>
  <a href="#s7">07 证据与口径</a>
</div>

<section id="s1">
  <div class="sec-head"><span class="sec-num">01</span><h2>先把价格和换手放在同一条时间轴上</h2></div>
  <p class="lead">603778 在 2016 年上市，经历两次“10 转 15”、2023 年转型 HJT、2025–26 年约 12 倍的行情和随后的 80% 崩塌。换手率的绝对水平随流通盘、题材和股东结构变化，但“放量—缩量”的节奏始终围绕价格位置展开。</p>
  <div class="figure"><img src="{img('01_价格与换手全景.png')}" alt="价格与换手全景">
  <div class="caption">上：后复权收盘价（对数），已还原 2016、2017 两次 10 转 15；下：日换手率，颜色对应九个区间。后复权口径下，2026-04 的 208.9 才是历史最高，约为 2016 年峰值的 2.6 倍——实际价 33.10 元看起来低于 2016 年的 75.19 元，是两次送转造成的错觉。</div></div>
  <div class="callout"><b>先记住三个坐标：</b>① 2016 年新股开板日换手 66.5%，是历史极值；② 2019–2020 年地量中位只有 0.9%，成交额 0.19 亿；③ 2025–26 主升段中位 13.5%，最高 35.5%。同一个指标，在三段行情里完全不是一回事。</div>
</section>

<section id="s2">
  <div class="sec-head"><span class="sec-num">02</span><h2>分布：一半交易日在 2.5% 以下，右尾来自泡沫</h2></div>
  <div class="grid2">
    <div>
      <p>2591 个交易日的中位数是 <b>2.5%</b>，均值 5.3%；P75 为 6.9%，P90 为 13.9%，P99 为 29.4%。也就是说，<b>6.4% 已经比历史上约四分之三的交易日更活跃</b>。</p>
      <p>但 2025–26 年的天量把分布右尾拉得很长：2026 年换手中位数 10.5%，P90 19.6%。所以在 2026 年内看，6.4% 反而只有 18.9 分位，是“泡沫期缩量”。</p>
      <div class="callout gold"><b>同一个 6.4%，两种叙事：</b>相对全历史是偏高（73.5 分位），相对本轮泡沫是偏低（18.9 分位）。判断见底时，后者不能替代前者——历史底部的地量：2019–2020 年 0.7%–1.3%，2024 年 1.4%–2.3%；当前仍是它的 3–9 倍。</div>
    </div>
    <div class="figure" style="margin-top:0"><img src="{img('02_换手率分布.png')}" alt="换手率分布">
    <div class="caption">对数刻度；虚线为 P50/P75/P90/P99，绿色为 2026-11-02 的 6.4%。</div></div>
  </div>
</section>

<section id="s3">
  <div class="sec-head"><span class="sec-num">03</span><h2>九个区间：每个区间对应什么场景、什么结果</h2></div>
  <p class="lead">下表是全样本统计。注意：T+20 是 T+1 开盘买入、持有 20 个交易日的后复权收益；胜率是 T+20 收盘价高于买入价的比例。样本高度重叠，n 天不等于 n 次独立交易。</p>
  <div class="tablewrap"><table>
    <thead><tr><th>区间</th><th>天数</th><th>占比</th><th>换手中位</th><th>成交额中位(亿)</th><th>前20日</th><th>250日位置</th><th>站上均线20</th><th>T+20中位</th><th>T+20均值</th><th>T+20胜率</th><th>20日最大浮亏</th><th class="left">主要年份</th></tr></thead>
    <tbody>{''.join(band_rows)}</tbody>
  </table></div>
  <div class="figure"><img src="{img('03_区间后续收益.png')}" alt="区间后续收益">
  <div class="caption">左：T+20 收益中位数；右：T+20 胜率。2%–8% 是这只股票最稳定的“负期望带”；<1% 是唯一正期望带；>20% 中位为负但均值高，属于右尾驱动。</div></div>

  <h3>逐区间解读</h3>
  <div class="grid3">{''.join(bar_cards_html)}</div>

  <div class="callout green"><b>反直觉但重要：</b>“换手率越高越危险”并不成立。>20% 区间的 T+20 均值是 +8.9%，高于 <1% 的 +4.7%；但它的中位数是 −3.9%，说明少数妖股行情把均值拉高。真正稳定的是：<b>2%–8% 的温和活跃区，长期跑输</b>；<b><1% 的地量区，长期跑赢</b>。</div>
</section>

<section id="s4">
  <div class="sec-head"><span class="sec-num">04</span><h2>阶段决定含义：同一换手率，主升段与崩塌段完全相反</h2></div>
  <p class="lead">把 2025-07 之后拆成两段：主升段（至 2026-04-13）与崩塌段（2026-04-14 起）。同样 8%–20% 的换手，主升段是资金接力，崩塌段是出货和失败反弹。</p>
  <div class="figure"><img src="{img('05_制度翻转.png')}" alt="制度翻转">
  <div class="caption">主升段 8–12% 换手：T+20 中位 +41%，胜率约 60%；12–20%：+36%，73%。崩塌段 8–12%：−25%，23%；12–20%：−22%，15%。样本 <5 天的区间未画。</div></div>
  <div class="tablewrap"><table>
    <thead><tr><th class="left">阶段</th><th>交易日</th><th>换手中位</th><th>换手P90</th><th>换手最大</th><th>区间涨跌(后复权)</th><th>成交额中位(亿)</th><th>20日波动中位</th></tr></thead>
    <tbody>{''.join(phase_rows)}</tbody>
  </table></div>
  <p class="note">阶段按行情切分，2016 段从 2015-12-31 起算；年度表按自然年，故 2016 年阶段 +126% 与年度 +106% 的起点不同。</p>
  <div class="grid2">
    <div class="callout"><b>2016–2017：高换手 = 派发。</b>新股开板 66.5%、高送转抢权 55.4%、除权后 45.1%，后续 20 日多为 −10% 到 −20%。当时流通盘只有 0.2–2.3 亿股，少量资金就能做出天量。</div>
    <div class="callout green"><b>2025–2026：高换手 = 接力，直到不再是。</b>2025-10-17 换手 31.8% 起步，后 20 日 +96%；2025-12-11 换手 35.5%，后 20 日 +98%。但 2026-03-11 至 04-17 的 13 天 ≥20% 换手集群，后 60 日 −60%；04-16 跌停后 27.3% 的换手，后 20 日仅 −4%，后 60 日 −60%。<b>拐点不在换手率本身，而在“高换手后能否继续创新高”。</b></div>
  </div>

  <h3>年度视角：换手率的“水位”随行情整体升降</h3>
  <div class="tablewrap"><table>
    <thead><tr><th class="left">年份</th><th>交易日</th><th>换手中位</th><th>换手P90</th><th>换手最大</th><th>成交额中位(亿)</th><th>全年涨跌(后复权)</th></tr></thead>
    <tbody>{''.join(year_rows)}</tbody>
  </table></div>
</section>

<section id="s5">
  <div class="sec-head"><span class="sec-num">05</span><h2>极端换手事件：>20% 之后发生了什么</h2></div>
  <p class="lead">把连续（间隔 ≤10 个自然日）的 ≥20% 换手日合并成事件。它们大致分三类：新股/高送转派发、题材主升起点、崩塌中的失败反弹。</p>
  <div class="tablewrap"><table>
    <thead><tr><th class="left">起止</th><th>天数</th><th>换手中位 / 最高</th><th>期间涨跌(后复权)</th><th>前20日</th><th>结束250日位置</th><th>后20日</th><th>后60日</th></tr></thead>
    <tbody>{''.join(event_rows)}</tbody>
  </table></div>
  <p class="note">口径提醒：2017-06 的 45.1% 换手发生在“10 转 15”除权前后，实际价格从 27.54 元跌到 13.69 元，但后复权总回报为正；表中“期间涨跌”与“后 20/60 日”均为后复权，已还原送转。2016 年部分事件尚无完整 250 日窗口，位置显示“—”。</p>
  <div class="callout"><b>三组对照：</b>2016-08 高送转（55.4%）后 20 日 −12%、后 60 日 −22%；2021-03 冲高（30.1%）后 20 日 −31%、后 60 日 −34%；2025-10 主升起点（31.8%）后 20 日 +96%、后 60 日 +257%。<b>同样是 30%+ 的换手，结果从 −34% 到 +257%。</b>差异不在换手率，而在价格位置、趋势和此后的价格确认。</div>
</section>

<section id="s6">
  <div class="sec-head"><span class="sec-num">06</span><h2>当前 6.4%：泡沫期缩量，不是历史地量</h2></div>
  <div class="kv">
    <div><div class="k">最新收盘</div><div class="v">6.92 元</div></div>
    <div><div class="k">换手 / 成交额</div><div class="v">6.4% / 2.77 亿</div></div>
    <div><div class="k">均线20（不复权）</div><div class="v">7.56 元</div></div>
    <div><div class="k">均线60（不复权）</div><div class="v">9.14 元</div></div>
    <div><div class="k">近5日换手中位</div><div class="v">4.5%</div></div>
    <div><div class="k">近20日换手中位</div><div class="v">5.2%</div></div>
    <div><div class="k">2025-10 起点</div><div class="v">4.11 元</div></div>
    <div><div class="k">2026-04 高点</div><div class="v">33.10 元</div></div>
  </div>
  <div class="figure"><img src="{img('04_2025_2026放大.png')}" alt="2025-2026 放大">
  <div class="caption">红底为主升段（2025-07 至 2026-04-13），绿底为崩塌段。当前 6.4% 处于崩塌段的中等换手区，低于 2026 年内 81% 的交易日，但高于全历史 73% 的交易日。</div></div>

  <h3>历史同类样本：把 2026 年自己剔除后，样本还剩多少？</h3>
  <p>这是本次研究最需要克制的地方。当前“深度回撤 + 换手 5–8%”的组合，在 2016–2025 年只出现过 <b>2 天</b>（2018-10-29、2024-02-07），且后 20 日都为正；而 2026 年 4 月以来出现了 <b>24 天</b>，它们是同一轮崩塌，不能当作 24 次独立验证。所以下表同时给出全样本、剔除 2026、2018–2024 三组。</p>
  <div class="tablewrap"><table>
    <thead><tr><th class="left">条件</th><th>样本</th><th>天数</th><th>T+20中位</th><th>T+20胜率</th><th>T+60中位</th><th>T+60胜率</th></tr></thead>
    <tbody>{''.join(analog_rows_html)}</tbody>
  </table></div>
  <div class="callout gold"><b>怎么读：</b>“换手 5–8%（全部）”剔除 2026 后仍有 239 天，T+20 −2.4%、T+60 −12.5%、60 日胜率 29%——这是相对可靠的历史基准。“深度回撤 + 5–8%”剔除 2026 只剩 2 天，不能单独下结论；它只能说明：<b>历史上这个组合既可能像 2024-02 那样见底，也可能像 2018-10 那样只是反弹</b>。当前崩塌段自身 24 天的 −20%/6% 是描述，不是验证。</div>

  <h3>三条路径与观察点</h3>
  <div class="trigger"><div class="t">A</div><div><b>地量底（历史最常见）</b><br>换手降到 1%–2%、至少 2–3 周不创新低，随后放量站上均线20。2019–2020 年用了两年；2024-02 的第一次触底只换来一段约 3 个月的反弹，随后再创新低，真正的起点在 2025 年下半年。<span class="pill">当前未满足</span><span class="pill">6.4% 仍偏高</span></div></div>
  <div class="trigger"><div class="t">B</div><div><b>恐慌底 / 破底翻</b><br>盘中跌破 6.72/6.48 后快速收回，且 3 日内不再破低；若伴随 12%–20% 的换手且次日不破低，可提高关注，但本股历史上此类样本很少，不能当高胜率信号。11-02 盘中 6.48 收回 6.92 是雏形，但换手只有 6.4%，且收盘仍在均线20 下方。<span class="pill">待确认</span></div></div>
  <div class="trigger"><div class="t">C</div><div><b>下跌中继（历史基准偏差）</b><br>5%–8% 弱反弹后再度放量跌破 6.48，下一参考是 2025-10 起点 4.11 元。历史上 5–8% 带（剔除 2026）T+60 中位 −12.5%、胜率 29%。<span class="pill">当前基准情形</span></div></div>

  <div class="callout green"><b>一句话结论：</b>6.4% 不是“地量见底”，而是“崩塌后的中等换手弱反弹区”。历史上这个区间（5–8%）在非主升阶段偏负；当前价格仍在均线20 下方、PB 20 倍、较 2025-10 起点仍高 68%，没有估值或筹码上的安全垫。<b>在放量站上 7.56 元（均线20）之前，把反弹当反弹看，不当反转看。</b></div>
</section>

<section id="s7">
  <div class="sec-head"><span class="sec-num">07</span><h2>证据等级与口径</h2></div>
  <div class="grade">
    <div class="g">A</div><div><b>全样本分档与阶段差异。</b>2591 个交易日，九个区间的中位/胜率差异稳定；2025 主升段与 2026 崩塌段方向相反，是最强的结构性证据。局限：样本重叠，且 2016、2025–26 两段极端行情权重很大。</div>
    <div class="g">B</div><div><b>5–8% 带在 2018–2024 各阶段的负偏。</b>2018–20、2021–22、2023–24 三段分别为 −8.8%/30%、−7.5%/28%、−4.2%/38%（T+20），T+60 均为 −13% 到 −17%。这是当前 6.4% 最直接的历史基准。</div>
    <div class="g">C</div><div><b>“深度回撤 + 5–8%”的独立样本。</b>2016–2025 仅 2 天，2026 年 24 天是同一轮行情。只能作为情景参考，不能当统计规律。</div>
    <div class="g">D</div><div><b>单日/单周触发信号。</b>“破底翻”“放量站上均线20”等需要逐日跟踪，历史样本不足；本文只给观察条件，不给买卖指令。</div>
  </div>
  <h3>数据与计算口径</h3>
  <p class="note">
  · 换手率：baostock 官方 turn = 成交量 / 流通股本；2015-12-31 至 2026-10-29 共 2591 个有效交易日，37 个停牌日剔除。<br>
  · 最新两日：同花顺 DuckDB 补充 2026-10-30、2026-11-02；流通股本 6.428562 亿股（2021-06 起稳定，逐日反推误差 <0.01%），换手率为估算值。<br>
  · 收益：后复权价（adjustflag=1），T+1 开盘买入，持有 N 个交易日到收盘；未扣佣金、印花税、滑点。T+20 均值受右尾影响大，故同时给中位数与胜率。<br>
  · 复权：2016-05、2017-06 两次 10 转 15，2019-01、2021-06 两次送转；所有均线、250 日位置、动量均用后复权价，避免除权假跳空。全样本 12 个“一字板缩量”样本（2016 年 8 个）保留在 <1% 区间，但对结论影响很小（剔除后 <0.5% 仍为 +2.9%/60%）。<br>
  · 估值：PB(MRQ)、PE(TTM) 来自 baostock，最新为 2026-10-29；当前 PB 20.0，历史低点 2018-10 为 2.15、2024-02 为 2.00、2025-10 为 3.70。<br>
  · 本文是单只股票的历史结构研究，不构成投资建议；历史统计不代表未来。
  </p>
</section>

<footer>
  <b>603778 换手率区间研究</b> · 数据截至 2026-11-02 · 生成于 2026-11-03<br>
  数据源：baostock 本地 SQLite（主）、同花顺本地 DuckDB（最新两日） · 图表与表格见同目录 Excel / CSV / PNG<br>
  提醒：单只股票、样本重叠、极端行情权重高；任何区间统计都不能替代对当下价格、成交和基本面的确认。
</footer>

</div>
</body>
</html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 已生成:", OUT, round(os.path.getsize(OUT) / 1024 / 1024, 2), "MB")
