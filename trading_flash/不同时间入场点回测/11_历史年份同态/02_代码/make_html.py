# -*- coding: utf-8 -*-
"""历史年份同态：自包含 HTML。"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'
FIG = OUT / '04_图表'
HTML = OUT / '历史年份同态.html'

Y = pd.read_csv(RES / '年度相似度.csv')
E = pd.read_csv(RES / '同态事件_总表.csv')
S = pd.read_csv(RES / '同态期_策略起点.csv')
ST = pd.read_csv(RES / '当前状态.csv')
R = pd.read_csv(RES / '规则状态_事件.csv')
P = pd.read_csv(RES / '距离预测力.csv')

DATE = '2026-10-23'


def pct(x, d=1, sign=True):
    if pd.isna(x):
        return '—'
    return f'{x * 100:+.{d}f}%' if sign else f'{x * 100:.{d}f}%'


def num(x, d=3):
    if pd.isna(x):
        return '—'
    return f'{x:.{d}f}'


def row(year, col):
    return Y.loc[Y['年份'] == year, col].iloc[0]


def event(day, col):
    return E.loc[E['代表日'] == day, col].iloc[0]


def start(day, col):
    return S.loc[S['代表日'] == day, col].iloc[0]


def state(name, col='当前值'):
    return ST.loc[ST['指标'] == name, col].iloc[0]


def state_pct(name):
    v = ST.loc[ST['指标'] == name, '历史百分位']
    return float(v.iloc[0]) * 100 if len(v) and pd.notna(v.iloc[0]) else 0.0


def img64(name):
    b = (FIG / name).read_bytes()
    return 'data:image/png;base64,' + base64.b64encode(b).decode()


def cls(x):
    return 'neg' if (pd.notna(x) and x < 0) else 'pos'


# 当前状态卡
stats = [
    ('250 日涨幅', pct(state('250日涨幅')), state_pct('250日涨幅'),
     '过去一年仍为负；不是高位牛市回调', 'neg'),
    ('250 日回撤', pct(state('距250日高')), state_pct('距250日高'),
     '距 2026-02-27 高点 -17.5%', 'neg'),
    ('20 日波动率', pct(state('20日波动率'), 1, False), state_pct('20日波动率'),
     '全样本 61 分位；信号日仅 29 分位', 'warn'),
    ('宽度 20', pct(state('宽度20'), 1, False), state_pct('宽度20'),
     '全样本 11 分位；信号日 55 分位', 'cool'),
    ('宽度 60', pct(state('宽度60'), 1, False), state_pct('宽度60'),
     '中期宽度也很窄', 'cool'),
    ('情绪分', num(state('情绪分'), 2), state_pct('情绪分'),
     '仍为负，但没有 2018-05 那么极端', 'neg'),
    ('跌停占比', pct(state('跌停占比'), 2, False), state_pct('跌停占比'),
     '绝对水平 0.25%，2018-05-30 是 3.42%', 'cool'),
    ('20/60 成交额比', num(state('20/60日成交额比'), 3), state_pct('20/60日成交额比'),
     '缩量：历史 14 分位', 'cool'),
]
stat_html = ''
for label, value, p, note, tone in stats:
    stat_html += f'''
      <div class="stat {tone}">
        <div class="stat-label">{label}</div>
        <div class="stat-value {tone}">{value}</div>
        <div class="track"><i style="width:{min(max(p, 0), 100):.0f}%"></i></div>
        <div class="stat-foot"><span>历史分位 {p:.0f}</span><span>{note}</span></div>
      </div>'''

# 年度表
year_rows = ''
for _, r in Y.head(6).iterrows():
    y = int(r['年份'])
    hl = ' class="hl"' if y in (2018, 2024) else ''
    year_rows += f'''
      <tr{hl}>
        <td><b>{y}</b></td>
        <td>{num(r['核心8_距离'])} <span class="rank">#{int(r['核心8_排名'])}</span></td>
        <td>{num(r['全状态13_距离'])} <span class="rank">#{int(r['全状态13_排名'])}</span></td>
        <td>{r['核心8最近日期']}</td>
        <td class="{cls(r['核心8最近日未来60日'])}">{pct(r['核心8最近日未来60日'])}</td>
        <td class="{cls(r['核心8最近日未来120日'])}">{pct(r['核心8最近日未来120日'])}</td>
        <td class="{cls(r['策略年度收益'])}"><b>{pct(r['策略年度收益'])}</b></td>
        <td class="{cls(r['市场年度收益'])}">{pct(r['市场年度收益'])}</td>
      </tr>'''

# 主同态表
twin_rows = [
    ('事件区间', event('2018-05-30', '起点') + ' ~ ' + event('2018-05-30', '终点'),
     event('2024-06-14', '起点') + ' ~ ' + event('2024-06-14', '终点')),
    ('距离', num(event('2018-05-30', '最小距离')), num(event('2024-06-14', '最小距离'))),
    ('市场 20 日', pct(event('2018-05-30', '市场20日')), pct(event('2024-06-14', '市场20日'))),
    ('市场 60 日', pct(event('2018-05-30', '市场60日')), pct(event('2024-06-14', '市场60日'))),
    ('市场 120 日', pct(event('2018-05-30', '市场120日')), pct(event('2024-06-14', '市场120日'))),
    ('60 日内最大回撤', pct(event('2018-05-30', '市场60日最大回撤')),
     pct(event('2024-06-14', '市场60日最大回撤'))),
    ('后续最低点', event('2018-05-30', '后续最低点日期'), event('2024-06-14', '后续最低点日期')),
    ('后续最低点回撤', pct(event('2018-05-30', '后续最低点回撤')),
     pct(event('2024-06-14', '后续最低点回撤'))),
    ('后续最低点交易日', f"{int(event('2018-05-30', '后续最低点交易日'))} 日",
     f"{int(event('2024-06-14', '后续最低点交易日'))} 日"),
    ('宽度回到 50%', f"{int(event('2018-05-30', '宽度回50%交易日'))} 日",
     f"{int(event('2024-06-14', '宽度回50%交易日'))} 日"),
    ('收复年线', f"{int(event('2018-05-30', '收复年线交易日'))} 日",
     f"{int(event('2024-06-14', '收复年线交易日'))} 日"),
    ('策略下一开始点', event('2018-05-30', '下一可执行起点'), event('2024-06-14', '下一可执行起点')),
    ('策略 60 日', pct(start('2018-05-30', '下一可执行起点60日')),
     pct(start('2024-06-14', '下一可执行起点60日'))),
    ('策略 120 日', pct(start('2018-05-30', '下一可执行起点120日')),
     pct(start('2024-06-14', '下一可执行起点120日'))),
    ('策略 120 日回撤中位', pct(start('2018-05-30', '区间起点120日回撤中位')),
     pct(start('2024-06-14', '区间起点120日回撤中位'))),
]
twin_html = ''
for label, a, b in twin_rows:
    klass_a = cls(float(a.replace('%', '').replace('+', ''))) if isinstance(a, str) and a.endswith('%') else ''
    klass_b = cls(float(b.replace('%', '').replace('+', ''))) if isinstance(b, str) and b.endswith('%') else ''
    twin_html += f'<tr><td>{label}</td><td class="{klass_a}">{a}</td><td class="{klass_b}">{b}</td></tr>'

# 当前 vs 两个同态
compare = [
    ('250 日涨幅', pct(-0.05151), pct(-0.08154), pct(-0.11710), '2018'),
    ('250 日回撤', pct(-0.17464), pct(-0.18990), pct(-0.16580), '2024'),
    ('120 日涨幅', pct(-0.13017), pct(-0.11323), pct(-0.14238), '2024'),
    ('60 日涨幅', pct(0.02112), pct(-0.03708), pct(-0.05507), '2018'),
    ('20 日波动率', pct(0.19775, 1, False), pct(0.19317, 1, False), pct(0.20030, 1, False), '2024'),
    ('波动率/60 日均值', num(0.966, 2), num(0.873, 2), num(0.794, 2), '2018'),
    ('宽度 20', pct(0.18997, 1, False), pct(0.14577, 1, False), pct(0.15045, 1, False), '2024'),
    ('宽度 60', pct(0.19812, 1, False), pct(0.22108, 1, False), pct(0.21820, 1, False), '2024'),
    ('情绪分', num(-0.643, 2), num(-0.916, 2), num(-0.631, 2), '2024'),
    ('跌停占比', pct(0.00251, 2, False), pct(0.03424, 2, False), pct(0.00096, 2, False), '2024'),
    ('20/60 成交额比', num(0.853, 3), num(0.919, 3), num(0.870, 3), '2024'),
]
compare_html = ''
for name, cur, a, b, closer in compare:
    compare_html += f'''
      <tr><td>{name}</td><td>{cur}</td><td class="{'closer' if closer == '2018' else ''}">{a}</td>
      <td class="{'closer' if closer == '2024' else ''}">{b}</td><td><span class="chip">{closer}</span></td></tr>'''

# 策略表
strategy_rows = [
    ('同态区间内信号日起点数', f"{int(start('2018-05-30', '起点数'))}",
     f"{int(start('2024-06-14', '起点数'))}"),
    ('下一可执行起点', start('2018-05-30', '下一可执行起点'), start('2024-06-14', '下一可执行起点')),
    ('下一开始点 20 日', pct(start('2018-05-30', '下一可执行起点20日')),
     pct(start('2024-06-14', '下一可执行起点20日'))),
    ('下一开始点 60 日', pct(start('2018-05-30', '下一可执行起点60日')),
     pct(start('2024-06-14', '下一可执行起点60日'))),
    ('下一开始点 120 日', pct(start('2018-05-30', '下一可执行起点120日')),
     pct(start('2024-06-14', '下一可执行起点120日'))),
    ('区间起点 60 日中位', pct(start('2018-05-30', '区间起点60日中位')),
     pct(start('2024-06-14', '区间起点60日中位'))),
    ('区间起点 120 日中位', pct(start('2018-05-30', '区间起点120日中位')),
     pct(start('2024-06-14', '区间起点120日中位'))),
    ('区间起点 60 日最差 / 最好',
     pct(start('2018-05-30', '区间起点60日最差')) + ' / ' + pct(start('2018-05-30', '区间起点60日最好')),
     pct(start('2024-06-14', '区间起点60日最差')) + ' / ' + pct(start('2024-06-14', '区间起点60日最好'))),
    ('区间起点 120 日回撤中位', pct(start('2018-05-30', '区间起点120日回撤中位')),
     pct(start('2024-06-14', '区间起点120日回撤中位'))),
]
strategy_html = ''
for label, a, b in strategy_rows:
    strategy_html += f'<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>'

pred_html = ''
for _, r in P.iterrows():
    pred_html += f'''
      <tr><td>{r['口径']}</td>
      <td>{r['Pearson']:+.2f}</td>
      <td>{pct(r['最近10%均值'])}</td>
      <td>{pct(r['其余均值'])}</td>
      <td>{pct(r['最近10%胜率'], 1, False)}</td>
      <td>{pct(r['其余胜率'], 1, False)}</td></tr>'''

CSS = """
:root{
  --paper:#f4f1e9; --card:#fffdf7; --ink:#1b2a3a; --muted:#6b7780;
  --line:#d9dfe2; --red:#b03a2e; --teal:#1f6f8b; --gold:#b8892f;
  --soft-red:#f6e2dd; --soft-teal:#e2eef2; --navy:#1b2a3a;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;max-width:100%;overflow-x:hidden}
body{max-width:100%;overflow-x:hidden}
img{max-width:100%;height:auto}
body{
  margin:0; background:
    radial-gradient(900px 500px at 92% -8%, rgba(31,111,139,.08), transparent 60%),
    radial-gradient(700px 400px at -5% 12%, rgba(176,58,46,.06), transparent 55%),
    var(--paper);
  color:var(--ink); font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-size:16px; line-height:1.75; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1120px;margin:0 auto;padding:36px 28px 80px}
.topline{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid var(--ink);
  padding-bottom:10px;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.topline b{color:var(--ink);font-weight:600}
h1{font-family:"Songti SC","Noto Serif SC",Georgia,serif;font-size:clamp(42px,6.6vw,76px);
  line-height:1.06;letter-spacing:-.03em;margin:42px 0 18px;font-weight:700}
h1 em{font-style:italic;color:var(--red)}
.lede{font-size:clamp(17px,2vw,21px);line-height:1.7;color:#33414d;max-width:880px;margin:0 0 30px}
.lede b{color:var(--ink)}
.meta{display:flex;flex-wrap:wrap;gap:10px 22px;font-size:12px;color:var(--muted);
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:34px}
.meta b{color:var(--ink);font-weight:600}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);
  font-weight:700;margin-bottom:10px}
.answer{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:8px 0 16px}
.answer-card{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--line);
  padding:22px 24px 20px;position:relative;box-shadow:0 1px 0 rgba(27,42,58,.04)}
.answer-card.r{border-top-color:var(--red)}
.answer-card.t{border-top-color:var(--teal)}
.answer-card .tag{font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
  font-weight:700}
.answer-card .year{font-family:"Songti SC",Georgia,serif;font-size:58px;line-height:1.05;
  font-weight:700;margin:6px 0 2px}
.answer-card.r .year{color:var(--red)} .answer-card.t .year{color:var(--teal)}
.answer-card .date{font-size:12px;color:var(--muted);margin-bottom:14px}
.answer-card .date + .date{margin-top:-9px;font-size:11px}
.answer-card .row{display:flex;justify-content:space-between;gap:12px;border-top:1px dashed var(--line);
  padding:7px 0;font-size:13px}
.answer-card .row span{color:var(--muted)} .answer-card .row b{font-variant-numeric:tabular-nums}
.verdict{background:var(--navy);color:#f7f4ec;padding:20px 24px;margin:0 0 44px;
  font-size:14.5px;line-height:1.8}
.verdict b{color:#fff}
.verdict .em{color:#f0c27a;font-weight:700}
section{margin:54px 0 0}
.sec-head{display:flex;align-items:baseline;gap:14px;border-top:1px solid var(--ink);padding-top:13px;margin-bottom:18px}
.sec-no{font-family:Georgia,serif;font-size:13px;color:var(--gold);letter-spacing:.06em}
h2{font-family:"Songti SC","Noto Serif SC",Georgia,serif;font-size:clamp(25px,3.3vw,34px);
  line-height:1.25;margin:0;font-weight:700;letter-spacing:-.01em}
h3{font-size:15px;margin:26px 0 8px;letter-spacing:.01em}
p{margin:0 0 14px;color:#33414d}
p.note{font-size:12.5px;color:var(--muted);line-height:1.7}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
  border:1px solid var(--line);margin:20px 0 16px}
.stat{background:var(--card);padding:15px 15px 13px;min-height:138px;display:flex;flex-direction:column}
.stat-label{font-size:11.5px;color:var(--muted);letter-spacing:.04em}
.stat-value{font-family:Georgia,"Songti SC",serif;font-size:28px;line-height:1.25;margin:2px 0 9px;
  font-variant-numeric:tabular-nums}
.stat-value.cool{color:var(--teal)} .stat-value.warn{color:var(--red)}
.stat-value.neg{color:var(--red)} .stat-value.pos{color:var(--teal)}
.track{height:4px;background:#e8e4da;position:relative;margin-top:auto}
.track i{display:block;height:100%;background:var(--ink)}
.stat.cool .track i{background:var(--teal)}
.stat.warn .track i,.stat.neg .track i{background:var(--red)}
.stat-foot{display:flex;flex-direction:column;gap:2px;font-size:10.5px;color:var(--muted);margin-top:7px;line-height:1.5}
.caption{font-size:11.5px;color:var(--muted);margin:8px 0 0;line-height:1.6}
.figure{background:var(--card);border:1px solid var(--line);padding:12px;margin:18px 0}
.figure img{display:block;width:100%;height:auto}
.table-wrap{overflow-x:auto;border:1px solid var(--line);background:var(--card);margin:14px 0}
table{border-collapse:collapse;width:100%;min-width:660px;font-size:12.5px}
th{background:#eef1f2;color:var(--ink);font-weight:700;text-align:left;padding:10px 11px;
  border-bottom:1px solid var(--line);white-space:nowrap;font-size:11.5px;letter-spacing:.02em}
td{padding:9px 11px;border-bottom:1px solid #e8ecee;vertical-align:top;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
tr.hl td{background:#fbf5ed}
.rank{color:var(--muted);font-size:10.5px;margin-left:2px}
.neg{color:var(--red)} .pos{color:var(--teal)}
.closer{background:#f3f7f8;font-weight:700;color:var(--teal)}
.chip{display:inline-block;border:1px solid currentColor;color:var(--muted);font-size:10px;
  padding:0 6px;line-height:1.7;letter-spacing:.04em}
.callout{border-left:3px solid var(--gold);background:#fbf6e9;padding:15px 18px;margin:18px 0;
  font-size:13.5px;line-height:1.8}
.callout b{color:var(--ink)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}
.why{background:var(--card);border:1px solid var(--line);padding:18px 20px}
.why.r{border-left:3px solid var(--red)} .why.t{border-left:3px solid var(--teal)}
.why h4{margin:0 0 6px;font-size:14px} .why p{font-size:12.8px;margin:0;color:var(--muted);line-height:1.7}
.why .big{font-family:Georgia,serif;font-size:26px;line-height:1.2;margin:2px 0 6px}
.why.r .big{color:var(--red)} .why.t .big{color:var(--teal)}
.split{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 4px}
.pill{font-size:11.5px;padding:3px 9px;border:1px solid var(--line);background:#fff;color:var(--ink)}
.pill.r{border-color:#e3b9b2;background:var(--soft-red)}
.pill.t{border-color:#b9d5de;background:var(--soft-teal)}
ol.steps{counter-reset:s;list-style:none;padding:0;margin:16px 0}
ol.steps li{counter-increment:s;position:relative;padding:14px 0 14px 46px;border-top:1px solid var(--line);
  font-size:14px;line-height:1.8;color:#33414d}
ol.steps li:before{content:counter(s,decimal-leading-zero);position:absolute;left:0;top:14px;
  font-family:Georgia,serif;font-size:13px;color:var(--gold);letter-spacing:.05em}
ol.steps b{color:var(--ink)}
.limits{font-size:12.5px;color:var(--muted);line-height:1.8;border-top:1px solid var(--line);padding-top:14px}
.limits b{color:var(--ink)}
.footer{margin-top:54px;border-top:1px solid var(--ink);padding-top:14px;font-size:11.5px;
  color:var(--muted);line-height:1.8}
.footer a{color:var(--ink);text-decoration:underline;text-underline-offset:3px}
.footer .links{display:flex;flex-wrap:wrap;gap:8px 18px;margin:8px 0}
@media(max-width:860px){
  .wrap{padding:24px 18px 60px}
  .answer{grid-template-columns:1fr}
  .stats{grid-template-columns:1fr 1fr}
  .two{grid-template-columns:1fr}
  .topline{flex-direction:column;gap:2px}
  h1{margin-top:30px}
  .verdict{padding:16px 18px}
}
@media(max-width:520px){
  .stats{grid-template-columns:1fr}
  .stat{min-height:auto}
  .answer-card .year{font-size:48px}
  table{font-size:11.5px}
  th,td{padding:8px 8px}
}
"""

BODY = f"""
<div class="wrap">
  <div class="topline"><span><b>历史年份同态</b> · 研究笔记</span><span>{DATE} 收盘 · 沪深主板等权</span></div>
  <h1>历史上哪一年，<br><em>像现在？</em></h1>
  <p class="lede">按价量状态找最近邻：<b>2018 与 2024 几乎并列</b>。它们之后 120 天，市场一个 <b class="neg">−19.5%</b>，一个 <b class="pos">+29.7%</b>。所以年份不能回答方向，但能把两条路径和它们的代价摆在你面前。</p>
  <div class="meta"><span>市场：沪深主板等权，2016-10-13 = 1</span><span>策略：2 仓 × 50%，T+3，15 日，+20% 止盈</span><span>历史同态样本：1,865 个交易日</span><span>当前信号日：{DATE}</span></div>

  <div class="answer">
    <div class="answer-card r">
      <div class="tag">全状态 13 年度第一 · 压力测试版</div>
      <div class="year">2018</div>
      <div class="date">年度距离 0.299（全状态）· 0.172（核心）</div>
      <div class="date">核心 8 最近邻 2018-05-30</div>
      <div class="row"><span>市场 120 日</span><b class="neg">−19.5%</b></div>
      <div class="row"><span>策略下一开始点 120 日</span><b class="pos">+16.0%</b></div>
      <div class="row"><span>策略 120 日回撤中位</span><b class="neg">−13.2%</b></div>
    </div>
    <div class="answer-card t">
      <div class="tag">核心 8 年度第一 · 修复版</div>
      <div class="year">2024</div>
      <div class="date">年度距离 0.158（核心）· 0.322（全状态）</div>
      <div class="date">核心 8 最近邻 2024-06-14</div>
      <div class="row"><span>市场 120 日</span><b class="pos">+29.7%</b></div>
      <div class="row"><span>策略下一开始点 120 日</span><b class="pos">+11.4%</b></div>
      <div class="row"><span>策略 120 日回撤中位</span><b class="neg">−12.5%</b></div>
    </div>
  </div>
  <div class="verdict">如果只能写一个年份：<b>2018</b>（全状态年度距离第一）。如果按逐项投票：13 个特征里 <b>10 个更像 2024</b>。两者只差 <b>0.014（核心）/ 0.023（全状态）</b>，<span class="em">不足以二选一</span>；第三名 2019/2022 的核心 8 距离约 0.41–0.44，是前两名的 2.4–2.8 倍。</div>

  <section>
    <div class="sec-head"><span class="sec-no">01</span><h2>先把“现在”说清楚</h2></div>
    <p>2026-10-23：<b>D 熊市下跌</b>；指数 1.743，年线 1.908，低于年线 8.6%；250 日涨幅 −5.2%，250 日回撤 −17.5%。距 2026-07-24 低点反弹 4.8%，但距 2026-02-27 高点仍差 17.5%。</p>
    <div class="stats">{stat_html}</div>
    <p class="note">“低波”要分清参照系：波动 0.198 在全样本是 61 分位（不低），在信号日是 29 分位（偏低）；宽度 0.190 在全样本是 11 分位（极低），在信号日是 55 分位（中位）。所以这是“宽度极低、波动不低”的阴跌，不是全面低波，也不是高波恐慌。跌停占比只有 0.25%，2018-05-30 是 3.42%——这是当前与 2018 最大的不同。</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">02</span><h2>年度相似度：前两名稳居</h2></div>
    <div class="figure"><img src="{img64('01_年度相似度.png')}" alt="年度相似度：核心 8 与全状态 13 两套口径"></div>
    <p class="caption">左：核心 8，2024 第一；右：全状态 13，2018 第一。颜色只标 2018/2024；核心 8 第三名距离约为前两名的 2.5 倍，全状态差距较小。</p>
    <div class="table-wrap"><table>
      <thead><tr><th>年份</th><th>核心 8</th><th>全状态 13</th><th>核心8最近日期</th><th>市场 60 日</th><th>市场 120 日</th><th>策略年度</th><th>市场年度</th></tr></thead>
      <tbody>{year_rows}</tbody>
    </table></div>
    <p class="note">距离 = 1%/99% 缩尾 + 中位数/IQR 标准化后的欧氏距离；年度取该年最近 10 个交易日均值。表中的“核心8最近日期”和右侧市场 60/120 日从该日算；全状态最近日不同（2018-06-12 / 2024-06-07）。2024 的“像”更持久：核心 8 前 100 名里 2024 占 71 天，2018 占 17 天——2018 是更尖的单点匹配，2024 是更长的同态区间。</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">03</span><h2>同态之后：两条分叉</h2></div>
    <div class="figure"><img src="{img64('02_同态路径.png')}" alt="同态后 120 日市场路径：2018 与 2024"></div>
    <div class="table-wrap"><table>
      <thead><tr><th>项目</th><th>2018-05-30</th><th>2024-06-14</th></tr></thead>
      <tbody>{twin_html}</tbody>
    </table></div>
    <div class="callout"><b>两个近邻的 20/60 日市场都是负的，120 日才分叉。</b> 2018 的宽度 31 天回到 50%，但指数又跌了 <b>94 个交易日、-29.2%</b> 才见底；2024 的宽度 33 天回到 50%，也又跌了 <b>66 个交易日、-10.0%</b> 才见底，之后才在 74 天收复年线。年线收复速度是路径差异，但它是事后变量——站在 2018-05-30，你不知道要等 179 天。</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">04</span><h2>相似度不是方向信号</h2></div>
    <div class="figure"><img src="{img64('03_距离与未来60日.png')}" alt="距离与未来 60 日收益散点：相关很弱"></div>
    <div class="table-wrap"><table>
      <thead><tr><th>口径</th><th>Pearson</th><th>最近 10% 均值</th><th>其余均值</th><th>最近 10% 胜率</th><th>其余胜率</th></tr></thead>
      <tbody>{pred_html}</tbody>
    </table></div>
    <div class="callout"><b>最近 10% 的日子 60 日平均 +7.2%，其余 +2.9%；但相关只有 -0.14，且最近的两段（2018、2024-06）60 日都是负的。</b> 20 日几乎无差异（+1.05% vs +1.04%）。这说明相似度有一点中期正偏，但没有短线方向；左尾 -19.5% 仍在。它不是抄底信号，也不能替代策略的市场闸门。</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">05</span><h2>为什么分叉：价量看不到的东西</h2></div>
    <div class="two">
      <div class="why r"><h4>2018-05-30 · 压力测试版</h4><div class="big">−19.5%</div>
        <p>跌停 3.42%、情绪 −0.92、60 日 −3.7%、250 日 −8.2%。事后看是去杠杆/贸易摩擦的下跌中继，政策底要等四季度。宽度回到 50% 也没有止住指数。</p></div>
      <div class="why t"><h4>2024-06-14 · 修复版</h4><div class="big">+29.7%</div>
        <p>跌停 0.10%、情绪 −0.63、60 日 −5.5%、250 日 −11.7%。事后看是 2 月政策底后的第二次测试，9 月 24 日增量政策后快速反弹，年线 74 天收复。</p></div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>特征</th><th>当前</th><th>2018-05-30</th><th>2024-06-14</th><th>更像</th></tr></thead>
      <tbody>{compare_html}</tbody>
    </table></div>
    <p><b>当前是混合体：</b></p>
    <div class="split"><span class="pill r">250 日涨幅 · 像 2018</span><span class="pill r">60 日涨幅 · 像 2018</span><span class="pill r">波动率/60 日均值 · 像 2018</span><span class="pill t">250 日回撤 · 像 2024</span><span class="pill t">120 日涨幅 · 像 2024</span><span class="pill t">宽度 · 像 2024</span><span class="pill t">情绪 · 像 2024</span><span class="pill t">跌停 · 像 2024</span><span class="pill t">成交额 · 像 2024</span></div>
    <p>逐项看，13 个特征里 10 个更像 2024；2018 的全状态距离仍略近，但差距只有 0.023。<b>距离不是投票，少数大差异会主导；所以不要用“第一/第二”覆盖逐项事实。</b></p>
    <div class="callout">这是叙事，不是本文数据验证的因果。本地日频只能描述“跌了多少、多宽、多波动”，不能描述“为什么跌、谁在救”。<b>2018 与 2024 的分水岭很可能不在价量里。</b></div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">06</span><h2>对策略：市场分叉，账户没有分叉</h2></div>
    <div class="figure"><img src="{img64('04_市场与策略.png')}" alt="市场与策略：同态区间的 60/120 日收益，以及起点离散度"></div>
    <div class="table-wrap"><table>
      <thead><tr><th>项目</th><th>2018 同态</th><th>2024-06 同态</th></tr></thead>
      <tbody>{strategy_html}</tbody>
    </table></div>
    <p>2018 年：68 笔完整信号，均值 +4.3%，胜率 63.2%，连续账户 <b>+47.4%</b>，市场 −30.4%。2024 年：61 笔，均值 +6.5%，胜率 65.6%，连续账户 <b>+93.0%</b>，市场 +5.3%。<b>两个年份都是策略丰年</b>，但这不是“年份好所以策略好”，而是策略只持有 15 天，赚的是跌深之后的反弹，不是指数年度方向。</p>
    <div class="callout"><b>但“像”不等于“今天买”。</b> 2024-06 同态区间内，60 日从 <b>−13.0% 到 +18.9%</b>；2018 区间从 <b>−3.2% 到 +9.8%</b>。区间起点 60 日中位只有 +2.8%（2018）和 +2.3%（2024）。同一天的两个账户，可能因为差几天而完全不同。</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">07</span><h2>结论：把年份当压力测试，不当开关</h2></div>
    <ol class="steps">
      <li><b>不要因为像 2018 就空仓，也不要因为像 2024 就满仓。</b> 两个年份的 120 天相差 49 个百分点，但价量状态几乎一样；价量分不清的分水岭，靠猜年份也分不清。</li>
      <li><b>当前策略照原规则执行。</b> 10-23 信号：603778.SH + 605376.SH，计划 10-28 开盘；2 仓 × 50%，T+3，15 日，+20% 止盈。若 10-28 一字涨停/停牌，跳过该名并保留现金，不递补第 3 名。002969 是第 3 名，不买。</li>
      <li><b>用风险预算，不用方向押注。</b> 同类起点 120 日最大回撤中位 −12.5%~−13.2%；2026 已经有过 −16.0% 的回撤。若只能承受一半，降到 1 仓 50% 或 2×25% 是主观降风险，<b>不是回测规则</b>。</li>
      <li><b>观察“扩散”而不是猜年份。</b> 跌停占比是否从 0.25% 扩散、宽度能否回到 50%、指数能否收复 1.908 的年线。它们不是已验证的开关，但比“像 2018/2024”更接近可观察事实。</li>
    </ol>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">08</span><h2>口径与限制</h2></div>
    <div class="limits">
      <b>数据：</b>沪深主板等权，2016-10-13 至 2026-10-23；策略样本 2017-11 至 2026-10。市场日频从 2016-10 开始，<b>2011/2015 不在样本内</b>，无法量化更早的类比。<br>
      <b>相似度：</b>标准化用了全样本，严格说含未来分布；同态日高度重叠，不能当独立样本；年度排名对特征集敏感，所以同时给了 4 套口径。它是描述性工具，不是交易信号。<br>
      <b>策略：</b>2 仓 × 50%、T+3、15 日、+20% 止盈；起点收益来自 251 个信号日，不是任意一天。2018/2024 的年度收益是连续账户，不能和“6 月起点”直接比较。<br>
      <b>外生变量：</b>政策、信用、估值、行业结构、退市/停牌都没有进入相似度；2018 与 2024 的分水岭很可能在这些变量里。<br>
      <b>2026：</b>完整交易只覆盖 3–7 月，10 月 8 条未完成；10-26 新信号未做全特征重算，也不改变 10-23 已冻结的执行名单。<br>
      <b>一句话：</b>历史不代表未来；同态不是预测，是把两条可能路径和它们的代价摆出来。先纸面 50 笔。
    </div>
  </section>

  <div class="footer">
    <div><b>历史年份同态</b> · 2026-10-23 · 沪深主板等权 · 2 仓 × 50% 恐慌反转策略</div>
    <div class="links">
      <a href="00_一页纸_历史年份同态.md">一页纸</a>
      <a href="01_完整报告_历史年份同态.md">完整报告</a>
      <a href="03_数据与复现.md">数据与复现</a>
      <a href="历史年份同态.xlsx">Excel 数据表</a>
      <a href="03_结果/同态事件_总表.csv">同态事件总表</a>
      <a href="05_证据/校验.txt">校验</a>
    </div>
    <div>本页为研究记录，不构成投资建议。历史同态不是预测；策略收益受起点、成交、停牌和成本影响。</div>
  </div>
</div>
"""

html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">' \
       f'<meta name="viewport" content="width=device-width, initial-scale=1">' \
       f'<title>历史上哪一年像现在？2018 与 2024</title><style>{CSS}</style></head>' \
       f'<body>{BODY}</body></html>'
HTML.write_text(html, encoding='utf-8')
print('saved', HTML, len(html))
