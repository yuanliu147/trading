#!/usr/bin/env python3
"""生成自包含的可视化 HTML。"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "03_结果"
FIG = BASE / "04_图表"
OUT = BASE / "买入后证伪.html"


def img(name: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def pct(x, digits=1):
    return "—" if pd.isna(x) else f"{x*100:.{digits}f}%"


def ci(lo, hi, digits=1):
    if pd.isna(lo) or pd.isna(hi):
        return "—"
    return f"{lo*100:.{digits}f}–{hi*100:.{digits}f}%"


def num(x):
    return "—" if pd.isna(x) else f"{int(x):,}"


def read(name):
    return pd.read_csv(RES / name, encoding="utf-8-sig")


def main():
    th = read("收益阈值.csv")
    fe = read("结构特征.csv")
    sc = read("预警评分.csv")
    co = read("组合规则.csv")
    ea = read("早退反事实.csv")
    sa = read("基准_样本.csv")
    cb = read("候选数量_基准.csv")
    cw = read("候选数量_预警.csv")
    sb = read("路径三态_基准.csv")
    sw = read("路径三态_转弱.csv")
    sm = read("路径三态_市场环境.csv")

    def get_th(d, thr, col="最终亏损率"):
        r = th[(th["样本"] == "全部信号前二") & (th["期间"] == "全期") &
               (th["天数"] == d) & np.isclose(th["阈值"], thr)]
        return r[col].iloc[0] if len(r) else np.nan

    def get_th_n(d, thr):
        r = th[(th["样本"] == "全部信号前二") & (th["期间"] == "全期") &
               (th["天数"] == d) & np.isclose(th["阈值"], thr)]
        return int(r["N"].iloc[0]) if len(r) else 0

    def get_co(d, rule, col="最终亏损率", period="全期", sample="全部信号前二"):
        r = co[(co["样本"] == sample) & (co["期间"] == period) &
               (co["天数"] == d) & (co["规则"] == rule)]
        return r[col].iloc[0] if len(r) else np.nan

    def get_co_row(d, rule, period="全期", sample="全部信号前二"):
        r = co[(co["样本"] == sample) & (co["期间"] == period) &
               (co["天数"] == d) & (co["规则"] == rule)]
        return r.iloc[0] if len(r) else None

    def get_fe(d, rule, col="最终亏损率"):
        r = fe[(fe["样本"] == "全部信号前二") & (fe["期间"] == "全期") &
               (fe["天数"] == d) & (fe["规则"] == rule)]
        return r[col].iloc[0] if len(r) else np.nan

    # 核心逐日表
    day_rows = []
    for d in [1, 2, 3, 5, 10]:
        r = get_co_row(d, "收益<-2%")
        day_rows.append(f"""
        <tr class="{'hot' if d == 3 else ''}">
          <td class="day">D+{d}</td>
          <td>{num(r['N'])}</td>
          <td class="big">{pct(r['最终亏损率'])}</td>
          <td class="muted">{ci(r['亏损率_CI下'], r['亏损率_CI上'])}</td>
          <td>{pct(r['深度亏损率'])}</td>
          <td>{pct(r['次日下跌率'])}</td>
          <td>{pct(get_th(d, 0.03))}</td>
          <td>{pct(get_co(d, '跌破买入价+未创新高+破5日线'))}</td>
        </tr>""")

    # 早退表
    early_rules = ["收益<-2%", "未达+3%", "未创买入后新高", "预警分>=4", "跌破买入价+未创新高+破5日线"]
    early_rows = []
    for rule in early_rules:
        r = ea[(ea["样本"] == "全部信号前二") & (ea["期间"] == "全期") &
               (ea["天数"] == 5) & (ea["规则"] == rule)].iloc[0]
        name = "三项复合" if rule.startswith("跌破买入价") else rule
        early_rows.append(f"""
        <tr>
          <td>{name}</td><td>{num(r['笔数'])}</td>
          <td>{pct(r['持有均值'])}</td><td>{pct(r['早退均值'])}</td>
          <td class="neg">{r['早退-持有pp']:+.1f}pp</td>
          <td>{pct(r['持有<=-10%'])}</td><td>{pct(r['早退<=-10%'])}</td>
          <td>{pct(r['持有<=-15%'])}</td><td>{pct(r['早退<=-15%'])}</td>
        </tr>""")

    # 分期稳定性表
    stab_rows = []
    for d in [3, 5]:
        for rule in ["收益<-2%", "未达+3%", "未创买入后新高", "预警分>=4", "跌破买入价+未创新高+破5日线"]:
            a = get_co_row(d, rule, period="2005-2016")
            b = get_co_row(d, rule, period="2017-2026")
            if a is None or b is None:
                continue
            name = "三项复合" if rule.startswith("跌破买入价") else rule
            stab_rows.append(f"""
            <tr>
              <td>D+{d}</td><td>{name}</td>
              <td>{pct(a['最终亏损率'])}</td><td>{pct(b['最终亏损率'])}</td>
              <td class="{'neg' if b['最终亏损率'] > a['最终亏损率'] else ''}">{(b['最终亏损率']-a['最终亏损率'])*100:+.1f}pp</td>
              <td>{num(a['N'])} / {num(b['N'])}</td>
            </tr>""")

    # 实际成交复核
    actual_rows = []
    for d in [3, 5]:
        for rule in ["收益<-2%", "未创买入后新高", "跌破买入价+未创新高+破5日线"]:
            r = get_co_row(d, rule, period="2017-2026", sample="实际成交")
            if r is None:
                continue
            name = "三项复合" if rule.startswith("跌破买入价") else rule
            actual_rows.append(f"""
            <tr>
              <td>D+{d}</td><td>{name}</td><td>{num(r['N'])}</td>
              <td>{pct(r['最终亏损率'])}</td>
              <td class="muted">{ci(r['亏损率_CI下'], r['亏损率_CI上'])}</td>
              <td>{pct(r['未触发亏损率'])}</td>
            </tr>""")

    # 基准表
    base_rows = []
    for _, r in sa.iterrows():
        base_rows.append(f"""
        <tr>
          <td>{r['样本']}</td><td>{r['期间']}</td><td>{num(r['笔数'])}</td>
          <td>{pct(r['胜率'])}</td><td>{pct(r['最终亏损率'])}</td>
          <td>{pct(r['均值'])}</td><td>{pct(r['中位'])}</td>
          <td>{pct(r['止盈率'])}</td><td>{pct(r['平均最大浮亏'])}</td>
        </tr>""")

    # 候选数量表
    cand_rows = []
    for b in ["1-2", "3-9", "10-19", "20-49", "50-99", "100+"]:
        ba = cb[(cb["期间"] == "全期") & (cb["候选级别"] == b)].iloc[0]
        wa = cw[(cw["期间"] == "全期") & (cw["天数"] == 5) & (cw["规则"] == "收益<-2%") & (cw["候选级别"] == b)]
        if not len(wa):
            continue
        wa = wa.iloc[0]
        cand_rows.append(f"""
        <tr>
          <td>{b}</td><td>{num(ba['信号日数'])} / {num(ba['交易数'])}</td>
          <td>{pct(ba['最终亏损率'])}</td><td>{pct(ba['均值'])}</td><td>{pct(ba['止盈率'])}</td>
          <td>{num(wa['触发笔数'])} / {num(wa['笔数'])}</td><td>{pct(wa['触发率'])}</td>
          <td class="big">{pct(wa['最终亏损率'])}</td><td>{pct(wa['同桶基准亏损率'])}</td>
          <td>{wa['差值pp']:+.1f}pp</td><td>{wa['早退相对持有']*100:+.1f}pp</td>
        </tr>""")

    # D+3 三态
    state_base_rows = []
    for st in ["上涨", "中震荡", "下跌"]:
        r = sb[(sb["样本"] == "全部信号前二") & (sb["期间"] == "全期") &
               (sb["状态"] == st)].iloc[0]
        state_base_rows.append(f"""
        <tr>
          <td>{st}</td><td>{num(r['笔数'])}</td><td>{pct(r['占比'])}</td>
          <td class="big">{pct(r['最终亏损率'])}</td>
          <td class="muted">{ci(r['95%置信下'], r['95%置信上'])}</td>
          <td>{pct(r['深度亏损率'])}</td><td>{pct(r['均值'])}</td>
          <td>{pct(r['止盈率'])}</td><td>{pct(r['平均最大浮亏'])}</td>
          <td>{pct(r['D+3均值'])}</td><td>{pct(r['D+5均值'])}</td>
        </tr>""")

    state_rules = [
        ("上涨", "D+5 回吐到成本下"), ("上涨", "D+5 距 D+0–5 最高回撤≥10%"),
        ("上涨", "D+5 比 D+3 收盘回吐≥5pp"), ("上涨", "D+5 未创新高（相对 D+3）"),
        ("中震荡", "D+5 < -2%"), ("中震荡", "D+5 < -5%"),
        ("中震荡", "D+5 预警分≥4"), ("中震荡", "D+5 三项复合"),
        ("下跌", "D+5 预警分≥4"), ("下跌", "D+5 三项复合"),
        ("下跌", "D+5 比 D+3 再跌≥2pp"), ("下跌", "D+5 < -2%"),
    ]
    state_warn_rows = []
    for st, rule in state_rules:
        r = sw[(sw["样本"] == "全部信号前二") & (sw["期间"] == "全期") &
               (sw["状态"] == st) & (sw["规则"] == rule)].iloc[0]
        state_warn_rows.append(f"""
        <tr>
          <td>{st}</td><td>{rule}</td><td>{num(r['触发笔数'])}</td><td>{pct(r['触发率'])}</td>
          <td class="big">{pct(r['最终亏损率'])}</td><td>{pct(r['同态基准'])}</td>
          <td>{r['差值pp']:+.1f}pp</td><td>{pct(r['深度亏损率'])}</td>
          <td>{pct(r['剩余收益均值'])}</td><td>{r['早退相对持有']*100:+.1f}pp</td>
          <td>{pct(r['持有<=-15%'])}</td><td>{pct(r['早退<=-15%'])}</td>
        </tr>""")

    state_market_rows = []
    for period in ["全期", "2017-2026"]:
        for rg in ["上涨市", "中震荡市", "下跌市"]:
            allr = sm[(sm["期间"] == period) & (sm["市场环境"] == rg) & (sm["状态"] == "全部")]
            if not len(allr):
                continue
            allr = allr.iloc[0]
            vals = []
            for st in ["上涨", "中震荡", "下跌"]:
                r = sm[(sm["期间"] == period) & (sm["市场环境"] == rg) & (sm["状态"] == st)]
                vals.append(pct(r["最终亏损率"].iloc[0]) if len(r) else "—")
            state_market_rows.append(f"""
            <tr>
              <td>{period}</td><td>{rg}</td><td>{pct(allr['最终亏损率'])}</td>
              <td>{vals[0]}</td><td>{vals[1]}</td><td>{vals[2]}</td>
              <td>{num(allr['笔数'])}</td>
            </tr>""")

    # 预警分表
    score_rows = []
    for s in range(6):
        a = sc[(sc["样本"] == "全部信号前二") & (sc["期间"] == "全期") & (sc["天数"] == 3) & (sc["预警分"] == s)].iloc[0]
        b = sc[(sc["样本"] == "全部信号前二") & (sc["期间"] == "全期") & (sc["天数"] == 5) & (sc["预警分"] == s)].iloc[0]
        score_rows.append(f"""
        <tr>
          <td>{s} 个</td>
          <td>{num(a['N'])}</td><td>{pct(a['最终亏损率'])}</td><td>{pct(a['深度亏损率'])}</td>
          <td>{num(b['N'])}</td><td>{pct(b['最终亏损率'])}</td><td>{pct(b['深度亏损率'])}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>买入后证伪：哪天没达标，就该警惕了？</title>
<style>
:root {{
  --bg: #f7f4ee; --paper: #fffdf9; --ink: #1c1917; --muted: #6b6257;
  --line: #e7e1d8; --red: #b91c1c; --orange: #c2410c; --gold: #a16207;
  --green: #15803d; --blue: #1d4ed8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  line-height: 1.65; font-size: 16px;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 48px 28px 80px; }}
.kicker {{ color: var(--orange); font-weight: 700; letter-spacing: .12em; font-size: 12px; text-transform: uppercase; }}
h1 {{ font-family: Georgia, "Songti SC", serif; font-size: clamp(34px, 5vw, 54px); line-height: 1.12; margin: 12px 0 16px; letter-spacing: -.02em; }}
.lede {{ font-size: 19px; color: #3f3a34; max-width: 900px; }}
.lede strong {{ color: var(--red); }}
.meta {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 22px; color: var(--muted); font-size: 13px; }}
.meta span {{ border-left: 3px solid var(--line); padding-left: 10px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 34px 0 26px; }}
.card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 14px; padding: 18px 18px 16px; box-shadow: 0 1px 0 rgba(28,25,23,.04); }}
.card .label {{ font-size: 12px; color: var(--muted); font-weight: 600; }}
.card .num {{ font-family: Georgia, "Songti SC", serif; font-size: 36px; line-height: 1.1; margin: 8px 0 4px; color: var(--red); }}
.card .sub {{ font-size: 12.5px; color: var(--muted); }}
.card.amber .num {{ color: var(--orange); }}
.card.green .num {{ color: var(--green); }}
.card.blue .num {{ color: var(--blue); }}
.callout {{ background: #1c1917; color: #faf7f1; border-radius: 16px; padding: 24px 26px; margin: 26px 0 36px; }}
.callout .big {{ font-family: Georgia, "Songti SC", serif; font-size: 23px; line-height: 1.45; }}
.callout .big em {{ color: #fbbf24; font-style: normal; }}
.callout .small {{ color: #c7bfb3; font-size: 13.5px; margin-top: 8px; }}
section {{ margin: 54px 0; }}
h2 {{ font-family: Georgia, "Songti SC", serif; font-size: 30px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -.01em; }}
h3 {{ font-size: 18px; margin: 28px 0 8px; }}
p {{ margin: 10px 0; }}
.gray {{ color: var(--muted); }}
.small {{ font-size: 13px; }}
.figure {{ background: var(--paper); border: 1px solid var(--line); border-radius: 16px; padding: 14px; margin: 20px 0; }}
.figure img {{ display: block; width: 100%; height: auto; border-radius: 8px; }}
.caption {{ font-size: 12.5px; color: var(--muted); padding: 10px 6px 2px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; background: var(--paper); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f1ece3; font-size: 12px; color: #4a443c; font-weight: 700; }}
tr:last-child td {{ border-bottom: 0; }}
td.big {{ font-weight: 800; color: var(--red); }}
td.day {{ font-weight: 800; color: var(--ink); }}
td.neg {{ color: var(--red); font-weight: 700; }}
td.muted {{ color: var(--muted); }}
tr.hot td {{ background: #fff1ed; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.note {{ background: #fff8e8; border-left: 4px solid #d97706; padding: 14px 16px; border-radius: 0 10px 10px 0; font-size: 14px; margin: 18px 0; }}
ul {{ margin: 10px 0 10px 20px; padding: 0; }}
li {{ margin: 7px 0; }}
.rule {{ display: grid; grid-template-columns: 32px 1fr; gap: 12px; margin: 14px 0; }}
.rule .n {{ width: 32px; height: 32px; border-radius: 50%; background: #1c1917; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 14px; }}
.rule b {{ color: var(--red); }}
.footer {{ border-top: 1px solid var(--line); margin-top: 56px; padding-top: 22px; color: var(--muted); font-size: 12.5px; }}
code {{ background: #f1ece3; padding: 2px 6px; border-radius: 5px; font-size: 12.5px; }}
table.compact {{ font-size: 12.3px; }}
table.compact th, table.compact td {{ padding: 7px 6px; }}
@media (max-width: 860px) {{
  .cards {{ grid-template-columns: 1fr 1fr; }}
  .two {{ grid-template-columns: 1fr; }}
  .wrap {{ padding: 28px 16px 60px; }}
  table {{ font-size: 12px; }}
  th, td {{ padding: 7px 6px; }}
}}
</style>
</head>
<body>
<div class="wrap">

  <div class="kicker">买入后证伪研究 · 2026-10-28</div>
  <h1>买入后第几天没达标，<br>就该警惕了？</h1>
  <p class="lede">把“恐慌洗盘反转”的买入拆成逐日路径：<strong>D+3 收盘是第一道可靠的证伪线</strong>；D+5 是确认线；D+10 只是事后确认。但“大概率会亏”不等于“现在卖更划算”——提前卖平均少赚 2–5 个百分点。</p>
  <div class="meta">
    <span>样本：全部信号前二 1,256 笔 / 实际成交 219 笔（有效 1,251 / 217）</span>
    <span>期间：2005-2026</span>
    <span>买入：T+3 开盘；持有 15 日；+20% 止盈</span>
    <span>D+0=买入日收盘；D+3=买入后第3个交易日（旧报告第4天）</span>
    <span>数据：baostock 主板日线（含退市股）</span>
  </div>

  <div class="cards">
    <div class="card">
      <div class="label">D+3 收盘 &lt; -2%</div>
      <div class="num">62.0%</div>
      <div class="sub">最终亏损率（基准 35.0%）<br>N=382，95%CI 56.5–67.6%</div>
    </div>
    <div class="card amber">
      <div class="label">D+5 收盘 &lt; -2%</div>
      <div class="num">65.1%</div>
      <div class="sub">最终亏损率（基准 35.0%）<br>N=401，95%CI 59.7–70.5%</div>
    </div>
    <div class="card">
      <div class="label">D+5 三项复合</div>
      <div class="num">71.4%</div>
      <div class="sub">最终亏损率；最终≤-10% 42.2%<br>基准深度亏损仅 15.6%</div>
    </div>
    <div class="card blue">
      <div class="label">D+5 触发后 D+6 早退</div>
      <div class="num">-4.3pp</div>
      <div class="sub">相对持有到最终的平均差<br>早退不是免费的保险</div>
    </div>
  </div>

  <div class="callout">
    <div class="big">如果“预期”是“D+5 应该涨 3%”，没达到只有 <em>54.5%</em> 最终亏损；<br>如果“预期”是“别亏超过 2%”，已经跌破则有 <em>65.1%</em> 最终亏损。</div>
    <div class="small">前者是弱警告，后者才是强警告。真正有用的不是“没涨够”，而是“已经跌穿”。</div>
  </div>

  <section>
    <h2>1. 先看基准：买入后本来会发生什么</h2>
    <p>全样本平均路径是向上的：D+5 平均 +2.7%，D+10 平均 +4.2%。但这是“含已止盈”的平均——到 D+10，28.9% 的交易最终触及 +20% 止盈离场。<strong>如果只看还没止盈的持仓，平均路径走平甚至转负</strong>（D+5 +0.1%，D+10 -1.0%）。所以后面所有条件概率，分母都是“当天还没止盈的持仓”，不是全部交易。</p>
    <div class="figure"><img src="{img('01_基准路径.png')}" alt="基准路径"><div class="caption">蓝线=全样本平均；红线=仍在持仓的平均/中位；灰点线=持仓中高于买入价的概率（右轴）。2005-2026，全部信号前二。</div></div>
    <div class="figure"><img src="{img('02_条件概率热力图.png')}" alt="条件概率热力图"><div class="caption">每一格=该检查点触发后的最终亏损率；下方小字=样本数。右图是现行系统期 2017-2026。</div></div>
    <table>
      <thead><tr><th>样本</th><th>期间</th><th>笔数</th><th>胜率</th><th>最终亏损率</th><th>均值</th><th>中位</th><th>止盈率</th><th>平均最大浮亏</th></tr></thead>
      <tbody>{''.join(base_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>2. 哪天？D+3 是第一道可靠线</h2>
    <p>用最简单的规则“D 日收盘相对买入价 &lt; -2%”逐日看：D+1 只有 55.0%，还不够；<strong>D+2 到 60.2%，D+3 到 62.0%，D+5 到 65.1%</strong>；D+10 到 82.8%，但只剩 4 个交易日，已经没有操作价值。</p>
    <table>
      <thead><tr><th>检查点</th><th>触发笔数</th><th>最终亏损率</th><th>95%CI</th><th>最终≤-10%</th><th>次日仍下跌</th><th>未达+3%的亏损率</th><th>三项复合的亏损率</th></tr></thead>
      <tbody>{''.join(day_rows)}</tbody>
    </table>
    <div class="note"><b>怎么读：</b>D+0=买入日收盘，D+3=买入后第3个交易日（旧报告的“第4天”）。“D+3 收盘 &lt; -2%”的 382 笔里，最终 62.0% 亏损；没触发的 782 笔只有 25.6% 亏损。差值 36.4 个百分点。到 D+5，触发组 65.1%，未触发组 25.2%。</div>
    <div class="two">
      <div>
        <h3>哪天算“早”？</h3>
        <ul>
          <li><b>D+1：</b>55.0%，还不足以当证伪；第一天红绿很随机。</li>
          <li><b>D+2：</b>60.2%，是黄色预警；2005-2016 只有 58.4%，略低于 60。</li>
          <li><b>D+3：</b>62.0%；两个样本期分别 62.1% / 62.0%，<b>最稳</b>。</li>
          <li><b>D+5：</b>65.1%；确认线，适合做“不追加/降仓”的依据。</li>
          <li><b>D+10：</b>82.8%；已经太晚，只能解释，不能预防。</li>
        </ul>
      </div>
      <div>
        <h3>哪条“特征”最好？</h3>
        <ul>
          <li><b>未创买入后新高：</b>D+3 58.8%，D+5 64.8%，是最稳的单特征。</li>
          <li><b>收盘跌破买入日最低：</b>D+3 54.7%，D+5 55.5%，不如 -2%。</li>
          <li><b>收盘低于5日线：</b>D+3 51.9%，D+5 53.7%，更弱。</li>
          <li><b>三项复合：</b>D+3 63.3%，D+5 71.4%，是最强的特征组合。</li>
          <li><b>跑输等权市场：</b>单独看 D+3 44.1%、D+5 48.7%，不强；但在已经下跌的样本里，跑输组比跑赢组高约 9–10pp，是第二层信息。</li>
          <li><b>量价特征：</b>放量下跌、涨缩量/跌放量、高开低走、长上影也测了；方向不稳、样本小，没有一条比“已跌2%”更可靠。</li>
          <li><b>但 AUC 显示：</b>D+3 收益本身 0.74，预警分 0.72，三项复合只有 0.63——<b>特征大多是“已经跌了”的另一种说法</b>。</li>
        </ul>
      </div>
    </div>
  </section>

  <section>
    <h2>3. 预期没达到 vs 已经跌了</h2>
    <p>这是整份研究最反直觉的一张图：横轴越往右，预期越宽松；越往左，实际越弱。同样在 D+5，<strong>“没涨到 +3%”只有 54.5% 最终亏损，而“已经跌 2%”有 65.1%</strong>。市场不惩罚“没涨够”，它惩罚“已经破位”。</p>
    <div class="figure"><img src="{img('03_阈值曲线.png')}" alt="阈值曲线"><div class="caption">左=最终亏损率；右=最终≤-10%的概率。橙线 D+3，红线 D+5；蓝虚线=-2%，绿点线=+3%。</div></div>
    <table>
      <thead><tr><th>检查点</th><th>“未达+3%”触发数</th><th>“未达+3%”最终亏损</th><th>“已跌2%”触发数</th><th>“已跌2%”最终亏损</th><th>差值</th></tr></thead>
      <tbody>
        <tr><td>D+2</td><td>{num(get_th_n(2, .03))}</td><td>{pct(get_th(2, .03))}</td><td>{num(get_th_n(2, -.02))}</td><td class="big">{pct(get_th(2, -.02))}</td><td>+{(get_th(2,-.02)-get_th(2,.03))*100:.1f}pp</td></tr>
        <tr class="hot"><td>D+3</td><td>{num(get_th_n(3, .03))}</td><td>{pct(get_th(3, .03))}</td><td>{num(get_th_n(3, -.02))}</td><td class="big">{pct(get_th(3, -.02))}</td><td>+{(get_th(3,-.02)-get_th(3,.03))*100:.1f}pp</td></tr>
        <tr><td>D+5</td><td>{num(get_th_n(5, .03))}</td><td>{pct(get_th(5, .03))}</td><td>{num(get_th_n(5, -.02))}</td><td class="big">{pct(get_th(5, -.02))}</td><td>+{(get_th(5,-.02)-get_th(5,.03))*100:.1f}pp</td></tr>
        <tr><td>D+10</td><td>{num(get_th_n(10, .03))}</td><td>{pct(get_th(10, .03))}</td><td>{num(get_th_n(10, -.02))}</td><td class="big">{pct(get_th(10, -.02))}</td><td>+{(get_th(10,-.02)-get_th(10,.03))*100:.1f}pp</td></tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>4. 预警分：0–5 个特征，能分层</h2>
    <p>把五个收盘特征相加：低于买入价、未创新高、跌破买入日最低、低于5日线、连续两日低点下移。D+3 的 0 分只有 18.3% 最终亏损，4 分 58.1%，5 分 72.7%；D+5 的 0 分 13.0%，4–5 分 68–72%。</p>
    <div class="figure"><img src="{img('04_预警分.png')}" alt="预警分"><div class="caption">全部信号前二，2005-2026。竖排小字为样本数。</div></div>
    <table>
      <thead><tr><th>预警分</th><th>D+3 笔数</th><th>D+3 最终亏损</th><th>D+3 最终≤-10%</th><th>D+5 笔数</th><th>D+5 最终亏损</th><th>D+5 最终≤-10%</th></tr></thead>
      <tbody>{''.join(score_rows)}</tbody>
    </table>
    <p class="small gray">注意：五个特征高度相关，预警分不是独立信息的相加；它的 AUC 反而低于“D 日收益”本身。它的价值是直观、可盘中手工打勾，不是更强的统计模型。</p>
  </section>

  <section>
    <h2>5. 最关键的反转：能预测“会亏”，不能预测“明天跌”</h2>
    <p>这是最容易被误用的一点。D+5 收盘 &lt; -2% 的 401 笔，<strong>次日仍下跌的概率只有 42.4%</strong>；全样本基准是 44.3%。它同时把未来 +20% 止盈概率从 19.4% 砍到 8.5%（D+3 从 23.7% 到 9.4%）。也就是说，它并不告诉你“明天要跌”，它告诉你“这笔交易大概率最终是亏的”。真正剩下的 10 天，平均还有 <strong>+4.5%</strong> 的收益（因为反弹），所以 D+6 开盘早退平均比持有少 4.3 个百分点。</p>
    <div class="figure"><img src="{img('08_触发后走势.png')}" alt="触发后走势"><div class="caption">触发组不是一路下跌：D+5 触发组从 -8.4% 反弹到 -4.6%，未触发组从 +7.9% 到 +10.2%。反弹存在，但补不回买入价。两组从买入日起就有差异，不能把差距全归因于触发。</div></div>
    <p>这也是为什么“看到 D+5 触发就卖”平均更差：触发组后面确实会反弹，只是弹不回买入价。早退锁定了 D+6 开盘的亏损，放弃了这段反弹。</p>
    <div class="figure"><img src="{img('05_早退对比.png')}" alt="早退对比"><div class="caption">D+5 触发，D+6 开盘卖；左=平均收益，右=最终≤-15%的概率。全部信号前二，毛收益，未扣早退费用。</div></div>
    <table>
      <thead><tr><th>D+5 规则</th><th>笔数</th><th>持有均值</th><th>早退均值</th><th>早退-持有</th><th>持有≤-10%</th><th>早退≤-10%</th><th>持有≤-15%</th><th>早退≤-15%</th></tr></thead>
      <tbody>{''.join(early_rows)}</tbody>
    </table>
    <div class="note"><b>唯一像“保险”的用法：</b>不是等到 D+5 已经亏了再卖，而是更早用“D+2/D+3 没涨到 +3%”做尾部保险。D+2 未达 +3% 就早退，最终≤-15% 的概率从 11.4% 降到 3.0%，但平均少赚 4.1 个百分点；D+5 三项复合再早退，≤-15% 只从 29.2% 降到 23.4%，因为亏损已经发生了。</div>
  </section>

  <section>
    <h2>6. 两个样本期 + 实际成交复核</h2>
    <p>预警规则不是某一轮行情的产物：2005-2016 和 2017-2026 的最终亏损率落在 45 度线附近。实际成交 2017-2026 只有 107 笔，方向一致但区间很宽，不能把点估计当精确值。</p>
    <div class="figure"><img src="{img('06_分期稳定性.png')}" alt="分期稳定性"><div class="caption">橙点=D+3，红方=D+5；每个点是一条规则在两个样本期的最终亏损率。</div></div>
    <div class="two">
      <div>
        <h3>分期稳定性</h3>
        <table>
          <thead><tr><th>检查点</th><th>规则</th><th>2005-16</th><th>2017-26</th><th>差</th><th>N</th></tr></thead>
          <tbody>{''.join(stab_rows)}</tbody>
        </table>
      </div>
      <div>
        <h3>实际成交 2017-2026</h3>
        <table>
          <thead><tr><th>检查点</th><th>规则</th><th>N</th><th>最终亏损</th><th>95%CI</th><th>未触发组</th></tr></thead>
          <tbody>{''.join(actual_rows)}</tbody>
        </table>
        <p class="small gray">实际成交样本小（107 笔），且是 2 仓约束后的子集；它的胜率 71.0%，高于全部信号前二的 63.8%，所以预警触发后的亏损率会低一些。</p>
      </div>
    </div>
    <div class="figure"><img src="{img('07_AUC.png')}" alt="AUC"><div class="caption">D 日收益本身是最强单变量：D+3 AUC 0.744、D+5 0.779；预警分 0.718 / 0.753，三项复合只有 0.636 / 0.615。复杂特征不能替代“当前收益”。</div></div>
  </section>

  <section>
    <h2>6.5 信号日候选数量：宽恐慌 vs 窄信号</h2>
    <p>候选数 = 信号日收盘同时满足五条条件的股票数，是信号日收盘就知道的“市场宽度”，不是买入后的特征。它不是单调关系：20–49 只的基准最好（亏损 26.9%），50–99 只最差（42.7%），100+ 只基准 29.3% 但 2017–26 只有 6 笔，主要是 2008/2015/2016 的极端恐慌反弹。</p>
    <div class="figure"><img src="{img('09_候选数量分层.png')}" alt="候选数量分层"><div class="caption">左=基准平均路径；右=D+5 收盘<-2% 的触发组 vs 同桶基准。全部信号前二，2005-2026。</div></div>
    <table class="compact">
      <thead><tr><th>候选级别</th><th>信号日 / 交易</th><th>基准亏损</th><th>基准均值</th><th>基准止盈</th><th>触发 / 总</th><th>触发率</th><th>触发亏损</th><th>同桶基准</th><th>差值</th><th>早退-持有</th></tr></thead>
      <tbody>{''.join(cand_rows)}</tbody>
    </table>
    <div class="note"><b>怎么读：</b>1–9 只的触发后亏损 67–70%，早退只差 -2.6~-3.0pp，窄信号失败后更“真失败”；20–49 只触发后 52.7%，早退差 -8.7pp，反弹强；50–99 只触发后 78.1%，但只有 32 笔触发，且 2017–26 的 12 笔集中在 2026；100+ 只触发后 52.9%，早退差 -10.2pp，但 2017–26 只有 6 笔，不能外推。</div>
    <p class="small gray">当前 2026-10-23 的 123 只候选属于 100+。历史上这个桶的近期样本极少，不能因为“宽恐慌”就放松预警。候选数与年份/市场阶段高度相关，分桶是描述性，不是独立规则。</p>
  </section>

  <section id="state68">
    <h2>6.8 上涨 / 中震荡 / 下跌：D+3 之后怎么走</h2>
    <p>前面的规则都在问“跌了怎么办”。这一节换一个问法：D+3 收盘时，这笔交易处在哪种路径？<strong>上涨 ≥ +3%、中震荡 -2%~+3%、下跌 ≤ -2%</strong>。三态不是新的入场信号，而是持仓管理的分层；D+3 之前已止盈的 84 笔和 5 笔未满 15 日的交易不进入三态。</p>
    <div class="figure"><img src="{img('10_路径三态.png')}" alt="D+3 路径三态"><div class="caption">左=三态之后的平均路径；右=最终亏损率、最终≤-10% 与均值。全部信号前二，2005-2026；分态是事后分组，D+3 之前的线不能用于提前识别。</div></div>
    <table>
      <thead><tr><th>D+3 状态</th><th>笔数</th><th>占比</th><th>最终亏损</th><th>95%CI</th><th>最终≤-10%</th><th>均值</th><th>止盈率</th><th>平均最大浮亏</th><th>D+3 均值</th><th>D+5 均值</th></tr></thead>
      <tbody>{''.join(state_base_rows)}</tbody>
    </table>
    <p class="small gray">三态把同一批交易拉开 45.8pp：上涨 16.2% / 中震荡 37.4% / 下跌 62.0%；分期为 16.1/16.5、36.0/38.6、62.1/62.0（2005-2016 / 2017-2026）。实际成交 2017-2026 为 8.1% / 40.5% / 52.4%（N=37/42/21），方向一致但样本更小。</p>

    <h3>上涨组：没创新高不够，回吐到成本才是</h3>
    <p>上涨组 437 笔，最终亏损 16.2%，均值 +10.4%，止盈 39.8%。D+5 “未创新高”只有 25.0%，相对基准 18.8% 只高 6.2pp，几乎不够；真正跳升的是<strong>回吐到成本下 47.5%</strong>（N=40）和<strong>距 D+0–5 最高回撤≥10% 44.2%</strong>（N=43）。但这两条触发后 D+6 开盘早退平均比持有少 5.2 / 5.5pp——它们预测最终亏损，不是次日卖点。</p>
    <div class="figure"><img src="{img('11_三态转弱.png')}" alt="三态转弱规则"><div class="caption">灰=同态基准，彩色=触发；右侧为触发笔数与相对基准的差值。全部信号前二，2005-2026。</div></div>
    <table class="compact">
      <thead><tr><th>状态</th><th>D+5 规则</th><th>触发N</th><th>触发率</th><th>最终亏损</th><th>同态基准</th><th>差值</th><th>最终≤-10%</th><th>剩余收益</th><th>早退-持有</th><th>持有≤-15%</th><th>早退≤-15%</th></tr></thead>
      <tbody>{''.join(state_warn_rows)}</tbody>
    </table>

    <h3>中震荡组 / 下跌组：同一张表的不同读法</h3>
    <p><strong>中震荡组</strong> 348 笔，最终亏损 37.4%。D+5 &lt; -2% 升到 58.8%（N=80），&lt; -5% 升到 77.1%（N=35）；&lt; -5% 的最终≤-15% 从 11.4% 压到 5.7%，但早退平均少 4.4pp。<strong>下跌组</strong> 382 笔已经亏损 62.0%；D+5 预警分≥4 升到 75.6%，最终≤-15% 从 33.7% 压到 27.9%，但早退平均少 3.4pp。D+3 已经是失败，D+5 只是确认，不是新的出口。</p>
    <div class="note"><b>为什么“知道会亏”却不宜 D+6 清仓：</b>触发后的剩余 10 天仍有反弹，D+6 早退在三个状态里平均都少 3–6pp；它压的是深尾，不是提高平均收益。要保险，应该在 D+3 或更早用硬止损/仓位控制，而不是等 D+5 已经跌了再卖。</div>

    <h3>市场环境交叉验证</h3>
    <p>把信号日的市场环境分成上涨市/中震荡市/下跌市：三态效应在每个环境里都同向（上涨组最低、下跌组最高）。但<strong>市场环境本身的基线不稳</strong>：2005-2026 是 29.1%/36.5%/45.6%，2017-2026 变成 37.7%/38.8%/36.7%。所以“市场是涨是震荡”不如“个股 D+3 路径”有区分度。</p>
    <div class="figure"><img src="{img('12_市场环境稳健.png')}" alt="市场环境稳健性"><div class="caption">分组柱=三态；黑菱形=该市场环境的全部交易。左=2005-2026，右=2017-2026。</div></div>
    <table class="compact">
      <thead><tr><th>期间</th><th>市场环境</th><th>全部亏损</th><th>上涨组</th><th>中震荡组</th><th>下跌组</th><th>全部N</th></tr></thead>
      <tbody>{''.join(state_market_rows)}</tbody>
    </table>
    <p class="small gray">阈值敏感性：上涨线从 +2% 调到 +3%/+5%/+8%，上涨组亏损率 18.0%/16.2%/13.7%/9.0%；下跌线从 0 调到 -1%/-2%/-3%，下跌组 54.5%/58.4%/62.0%/64.9%。单调，没有阈值悬崖，但具体切点仍是事后选择。</p>
    <div class="note"><b>最大的限制：</b>三态是事后分层，不是提前预测；上涨组 D+5 回吐规则只有 40/43 笔，实际成交 2017-26 只有 4 笔触发且 0 亏损，方向没有验证。不要把它当成“上涨就安全”或“下跌就必卖”。</div>
  </section>

  <section>
    <h2>7. 怎么用：三条可执行、两条不要做</h2>
    <div class="rule"><div class="n">1</div><div><b>D+3 收盘做第一次证伪。</b>收盘 &lt; -2% 记为“未通过”；若同时未创新高 / 跌破买入日最低 / 低于5日线，预警分≥4。此时<b>不要追加</b>，也不要把“跌了”解释成“更便宜了”。</div></div>
    <div class="rule"><div class="n">2</div><div><b>D+5 收盘做确认。</b>仍 &lt; -2%，最终亏损概率约 2/3；三项复合则约 71%，最终≤-10% 约 42%。如果仓位或回撤预算紧张，这时应该<b>降仓</b>，而不是等 D+10。</div></div>
    <div class="rule"><div class="n">3</div><div><b>如果目标是尾部保险，用更早、更宽的“没涨到 +3%”规则，或直接设硬止损。</b>它会把 ≤-15% 的深尾从 ~11% 压到 ~3%，代价是平均少赚 ~4pp；这是一笔保险，不是 alpha。</div></div>
    <div class="rule"><div class="n">✕</div><div><b>不要把它当次日择时。</b>触发后次日下跌概率 40–45%，并不高于基准；短期反而容易反弹。</div></div>
    <div class="rule"><div class="n">✕</div><div><b>不要因为 D+5 触发就机械清仓。</b>历史上前述规则触发后，持有到最终的均值仍高于 D+6 开盘早退 2–5pp；策略的 edge 本来就是“急跌后的反弹”。</div></div>
  </section>

  <section>
    <h2>8. 限制</h2>
    <ul>
      <li><b>不是 1,256 个独立机会。</b>按信号段聚类后只有 925 段；同一股票连续信号高度重叠。置信区间按信号段聚类稳健标准误，仍不能消除重叠。</li>
      <li><b>实际成交样本小。</b>2017-2026 只有 107 笔，D+5 触发组 28 笔；95%CI 很宽。点估计只作方向参考。</li>
      <li><b>口径混合。</b>全部信号前二用模拟规则（+20% 止盈/15日）；实际成交 219 笔用逐笔净收益（含跳空止盈、跌停顺延、退市）。两者在基准上差 0.2–1.3pp。</li>
      <li><b>早退是近似。</b>按 d+1 开盘成交，不建模跌停卖不出、排队和滑点扩大；实际成交口径扣 0.2% 卖出费用，全部信号口径为毛收益。</li>
      <li><b>只用了日线 OHLC。</b>没有盘中路径、封单、盘口；+20% 止盈按盘中触价近似。</li>
      <li><b>多重检验。</b>本文看了约 20 条规则 × 5 个检查点，只保留两期方向一致、样本量足够的几条；没有做 Bonferroni/FDR 校正，仍存在数据挖掘风险。</li>
      <li><b>描述性统计，不是因果。</b>“触发后亏损率更高”不等于“触发导致亏损”；它主要是“已经弱”的同义词。历史不代表未来，不构成投资建议。</li>
    </ul>
  </section>

  <div class="footer">
    <p><b>数据与复现：</b>baostock 主板日线（后复权收益 + 不复权成交/涨跌停，含退市股），面板由上级目录 <code>02_代码/build_panel.py</code> 构建；本专题只读面板。信号 2005-01 至 2026-10-23。代码：<code>02_代码/build_path_data.py</code> → <code>analyze_post_entry.py</code> / <code>analyze_path_states.py</code> → <code>make_charts.py</code> → <code>make_excel.py</code>。完整表格见 <code>买入后证伪_数据表.xlsx</code>（18 个中文表）与 <code>03_结果/</code> 下 CSV；三态明细在 <code>路径三态_*.csv</code>。</p>
    <p>本页为历史回测研究，不构成投资建议。</p>
  </div>

</div>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print("HTML 已生成：", OUT, "大小 %.1f KB" % (OUT.stat().st_size / 1024))


if __name__ == "__main__":
    main()
