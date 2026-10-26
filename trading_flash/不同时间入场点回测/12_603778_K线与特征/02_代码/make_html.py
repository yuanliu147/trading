# -*- coding: utf-8 -*-
"""603778 K 线与特征：自包含 HTML。"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'
FIG = OUT / '04_图表'
HTML = OUT / '603778_K线与特征.html'

S = pd.read_csv(RES / '信号日特征.csv', parse_dates=['信号日'])
C = pd.read_csv(RES / '条件统计.csv')
K = pd.read_csv(RES / '关键价位.csv')
CUR = pd.read_csv(RES / '当前快照.csv')
REL = pd.read_csv(RES / '相对强弱.csv', parse_dates=['date'])


def pct(x, d=1, sign=True):
    if pd.isna(x):
        return '—'
    return f'{x * 100:+.{d}f}%' if sign else f'{x * 100:.{d}f}%'


def num(x, d=2):
    if pd.isna(x):
        return '—'
    return f'{x:.{d}f}'


def img64(name):
    return 'data:image/png;base64,' + base64.b64encode((FIG / name).read_bytes()).decode()


def cls(x):
    return 'neg' if (pd.notna(x) and x < 0) else 'pos'


# 信号表
rows = ''
for _, r in S.iterrows():
    ret = pct(r['15日收益'])
    ret_cls = cls(r['15日收益'])
    if pd.isna(r['15日收益']):
        ret_cls = 'muted'
    bor = '是' if r['BOR'] == True else '否'  # noqa: E712
    rows += f'''
      <tr class="{'hl' if r['信号日'] == pd.Timestamp('2026-10-23') else ''}">
        <td>{r['信号日'].strftime('%m-%d')}</td>
        <td>{int(r['排名'])}</td>
        <td class="{ret_cls}">{ret}</td>
        <td>{pct(r['250日回撤'])}</td>
        <td>{num(r['收盘位置'], 3)}</td>
        <td>{num(r['RSI14'], 1)}</td>
        <td class="{'neg' if r['MACD柱'] < 0 else 'pos'}">{num(r['MACD柱'], 3)}</td>
        <td>{num(r['量比1/20'], 2)}</td>
        <td>{pct(r['距MA20'])}</td>
        <td>{bor}</td>
        <td>{r['案例']}</td>
      </tr>'''
# 10/26 partial
rows += '''
      <tr>
        <td>10-26</td><td>1</td><td class="muted">待定</td><td>−78.7%</td>
        <td>0.4545</td><td>31.8</td><td>—</td><td>—</td><td>—</td><td>否</td><td>待定</td>
      </tr>'''

# 条件表
cond_rows = ''
for _, r in C.iterrows():
    if '当前近似组合' in r['条件']:
        continue
    hl = 'hl' if ('BOR + 高收盘' in r['条件']) else ''
    cond_rows += f'''
      <tr class="{hl}"><td>{r['条件']}</td><td>{int(r['样本'])}</td>
      <td class="{cls(r['均值'])}">{pct(r['均值'])}</td><td>{pct(r['中位'])}</td>
      <td>{pct(r['胜率'], 1, False)}</td><td class="{cls(r['最差'])}">{pct(r['最差'])}</td>
      <td>{pct(r['p10'])}</td></tr>'''

# 关键价位
K2 = K.copy()
# 只保留接近现价的关键位 + 两端
keep = ['历史高点', '9月反弹高点', '8月反弹高点', 'MA60', '回撤 78.6%', '+20%目标（按10/23收盘）',
        'MA20', 'MA10', '7月低点', 'MA5', '当前收盘', '10月低点', '回撤 88.6%',
        '2025-11 低点附近', '回撤 100%']
# CSV 没有 2025-11；补一行
K2 = pd.concat([K2, pd.DataFrame([{'类型': '2025-11 低点附近', '日期': '—', '价格': 4.50, '距现价': 4.50 / 7.33 - 1}])], ignore_index=True)
K2 = K2[K2['类型'].isin(keep)].copy()
K2['price'] = K2['价格'].astype(float)
K2 = K2.sort_values('price', ascending=False)
level_rows = ''
for _, r in K2.iterrows():
    typ = r['类型']
    kind = '支撑' if r['price'] < 7.33 else ('压力' if r['price'] > 7.33 else '现价')
    if typ in ('10月低点', '回撤 88.6%', '2025-11 低点附近', '回撤 100%'):
        kind = '支撑'
    level_rows += f'''
      <tr><td>{typ}</td><td>{num(r['price'], 2)}</td>
      <td class="{'neg' if r['price'] < 7.33 else 'pos'}">{pct(r['price'] / 7.33 - 1)}</td>
      <td><span class="chip {'sup' if kind == '支撑' else 'res'}">{kind}</span></td></tr>'''

CSS = """
:root{--paper:#f5f2ea;--card:#fffdf8;--ink:#1b2a3a;--muted:#6b7780;--line:#d9dfe2;
--red:#b03a2e;--teal:#1f6f8b;--gold:#b8892f;--green:#2e7d5b;--navy:#1b2a3a;}
*{box-sizing:border-box}
html{scroll-behavior:smooth;max-width:100%;overflow-x:hidden}
body{margin:0;background:radial-gradient(900px 500px at 92% -8%,rgba(31,111,139,.08),transparent 60%),
radial-gradient(700px 400px at -5% 12%,rgba(176,58,46,.06),transparent 55%),var(--paper);
color:var(--ink);font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
font-size:16px;line-height:1.75;-webkit-font-smoothing:antialiased;max-width:100%;overflow-x:hidden}
img{max-width:100%;height:auto}
.wrap{max-width:1120px;margin:0 auto;padding:34px 28px 80px}
.topline{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid var(--ink);
padding-bottom:10px;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.topline b{color:var(--ink);font-weight:600}
h1{font-family:"Songti SC","Noto Serif SC",Georgia,serif;font-size:clamp(38px,6vw,68px);
line-height:1.08;letter-spacing:-.03em;margin:40px 0 16px;font-weight:700}
h1 em{font-style:italic;color:var(--red)}
.lede{font-size:clamp(16px,2vw,20px);line-height:1.7;color:#33414d;max-width:900px;margin:0 0 26px}
.lede b{color:var(--ink)}
.meta{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:12px;color:var(--muted);
border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:30px}
.meta b{color:var(--ink);font-weight:600}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--gold);font-weight:700;margin-bottom:8px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:6px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--line);padding:18px 19px}
.card.red{border-top-color:var(--red)} .card.teal{border-top-color:var(--teal)} .card.gold{border-top-color:var(--gold)}
.card .tag{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
.card .big{font-family:Georgia,"Songti SC",serif;font-size:34px;line-height:1.15;margin:4px 0 6px}
.card.red .big{color:var(--red)} .card.teal .big{color:var(--teal)} .card.gold .big{color:var(--gold)}
.card p{font-size:12.5px;line-height:1.65;color:var(--muted);margin:0}
.verdict{background:var(--navy);color:#f7f4ec;padding:19px 23px;margin:0 0 42px;font-size:14px;line-height:1.8}
.verdict b{color:#fff}.verdict .em{color:#f0c27a;font-weight:700}
section{margin:52px 0 0}
.sec-head{display:flex;align-items:baseline;gap:14px;border-top:1px solid var(--ink);padding-top:12px;margin-bottom:16px}
.sec-no{font-family:Georgia,serif;font-size:13px;color:var(--gold);letter-spacing:.06em}
h2{font-family:"Songti SC","Noto Serif SC",Georgia,serif;font-size:clamp(24px,3.2vw,33px);line-height:1.25;margin:0;font-weight:700}
h3{font-size:15px;margin:24px 0 8px}
p{margin:0 0 13px;color:#33414d}
.figure{background:var(--card);border:1px solid var(--line);padding:10px;margin:16px 0}
.figure img{display:block;width:100%}
.caption{font-size:11.5px;color:var(--muted);margin:6px 0 0;line-height:1.6}
.table-wrap{overflow-x:auto;border:1px solid var(--line);background:var(--card);margin:14px 0}
table{border-collapse:collapse;width:100%;min-width:680px;font-size:12.5px}
th{background:#eef1f2;color:var(--ink);font-weight:700;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;font-size:11.5px}
td{padding:8px 10px;border-bottom:1px solid #e8ecee;vertical-align:top;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
tr.hl td{background:#fbf3ed}
.neg{color:var(--red)} .pos{color:var(--teal)} .muted{color:var(--muted)}
.chip{display:inline-block;border:1px solid currentColor;font-size:10px;padding:0 6px;line-height:1.7;letter-spacing:.04em}
.chip.sup{color:var(--teal)} .chip.res{color:var(--red)}
.callout{border-left:3px solid var(--gold);background:#fbf6e9;padding:14px 17px;margin:16px 0;font-size:13.2px;line-height:1.8}
.callout b{color:var(--ink)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}
.box{background:var(--card);border:1px solid var(--line);padding:16px 18px}
.box.green{border-left:3px solid var(--green)} .box.red{border-left:3px solid var(--red)}
.box h4{margin:0 0 8px;font-size:13.5px}
.box ul{margin:0;padding-left:18px;font-size:12.5px;color:#4a5860;line-height:1.8}
.score{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);margin:14px 0}
.score div{background:var(--card);padding:11px 14px;font-size:12.5px}
.score b{font-family:Georgia,serif;font-size:20px}
.score .s1{color:var(--teal)} .score .s2{color:var(--red)}
.steps{counter-reset:s;list-style:none;padding:0;margin:12px 0}
.steps li{counter-increment:s;position:relative;padding:12px 0 12px 44px;border-top:1px solid var(--line);font-size:13.5px;line-height:1.8;color:#33414d}
.steps li:before{content:counter(s,decimal-leading-zero);position:absolute;left:0;top:12px;font-family:Georgia,serif;font-size:13px;color:var(--gold)}
.steps b{color:var(--ink)}
.limits{font-size:12.5px;color:var(--muted);line-height:1.85;border-top:1px solid var(--line);padding-top:12px}
.limits b{color:var(--ink)}
.footer{margin-top:50px;border-top:1px solid var(--ink);padding-top:14px;font-size:11.5px;color:var(--muted);line-height:1.8}
.footer a{color:var(--ink);text-decoration:underline;text-underline-offset:3px}
.footer .links{display:flex;flex-wrap:wrap;gap:8px 18px;margin:8px 0}
@media(max-width:860px){.wrap{padding:24px 18px 60px}.cards{grid-template-columns:1fr}.two{grid-template-columns:1fr}.score{grid-template-columns:1fr}.topline{flex-direction:column;gap:2px}h1{margin-top:28px}}
@media(max-width:520px){table{font-size:11.5px}th,td{padding:8px 8px}}
"""

BODY = f"""
<div class="wrap">
  <div class="topline"><span><b>603778 · K 线与特征</b> · 个股深挖</span><span>2026-10-23 收盘 · 10/26 仅部分数据</span></div>
  <h1>603778：<br><em>BOR 很好看，底部还没到</em></h1>
  <p class="lede">10/23 是“深回撤 + BOR + 高收盘”的强反转结构，历史 17 笔全为正；但 10/26 收盘位置只有 0.4545，<b>不是第二次确认</b>。这是一笔 15 日反弹，不是结构性底部。</p>
  <div class="meta"><span>价格：后复权折算当前价等价</span><span>现价 7.33</span><span>250日回撤 -78.6%</span><span>250日涨幅 +110.6%</span><span>样本：450 条完整信号</span></div>

  <div class="cards">
    <div class="card red"><div class="tag">10/23 · BOR</div><div class="big">0.924</div>
      <p>收盘位置 92.4%；低点 6.72 破前低、收盘 7.33 破前收。深回撤 + BOR + 高收盘 17 笔全部为正，均值 +14.5%。</p></div>
    <div class="card gold"><div class="tag">10/26 · 弱跟随</div><div class="big">0.4545</div>
      <p>收盘 7.29，收盘位置掉到 45.5%；不是 BOR，仍在 7 月低点 7.56 下方。深回撤 + 低收盘 73 笔均值 +8.8%/74%。</p></div>
    <div class="card teal"><div class="tag">结构 · 仍在空头</div><div class="big">-17.6pp</div>
      <p>7/21 以来股票 -14.5%，市场 +3.8%；4/09 以来 -77.7% vs 市场 -11.2%。破 7 月低点，低于所有均线。</p></div>
  </div>
  <div class="verdict">一句话：<b>10/23 是这轮下跌里最好的 K 线反转，10/26 不是第二次确认；BOR 支持 15 日反弹，结构仍不支持“底部”。</b> 按 2 仓规则做，不因榜一加仓；跌破 6.72 则 BOR 失败，下一道 6.11。</div>

  <section>
    <div class="sec-head"><span class="sec-no">01</span><h2>先看结构：14 倍泡沫后的 -79%</h2></div>
    <div class="figure"><img src="{img64('01_两年K线.png')}" alt="603778 两年 K 线：14 倍泡沫与 -79% 回撤"></div>
    <p class="caption">后复权价格按 2026-10-23 实际价/后复权价折算成当前价等价；历史价格不是当时成交价。红涨绿跌。</p>
    <p>2025-04-09 低点 2.49 → 2026-04-09 高点 34.27，约 <b>13.8 倍</b>；2026-10-23 收 7.33，回撤 <b>-78.6%</b>。月线是“泡沫 → 反抽 → 失败 → 再下台阶”：2026-01 冲高回落、3–4 月反抽、5–7 月主跌、8–9 月反弹、10 月新低。</p>
    <div class="callout"><b>250 日涨幅 +110.6% 不是“强”，是“基数低”。</b> 一年前价格还在 3–4 元附近；250 日过滤器在这只股票上已经旧了。14 倍泡沫的完全回归是 2.49（-66%），不是 7.33。</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">02</span><h2>近月 K 线：10/23 BOR vs 10/26 弱跟随</h2></div>
    <div class="figure"><img src="{img64('02_近月K线.png')}" alt="603778 近月 K 线：10/23 BOR 与 10/26 弱跟随"></div>
    <p class="caption">10/26 仅有收盘与部分特征，OHLC 未入库；图中以圆点标出收盘 7.29。</p>
    <div class="table-wrap"><table>
      <thead><tr><th>日期</th><th>收盘</th><th>收盘位置</th><th>RSI14</th><th>MACD柱</th><th>量比1/20</th><th>下影</th><th>结构</th></tr></thead>
      <tbody>
        <tr><td>10-22</td><td>7.15</td><td>0.054</td><td>28.0</td><td class="neg">-0.0090</td><td>0.640</td><td>0.4%</td><td>大阴线，收在低位</td></tr>
        <tr class="hl"><td><b>10-23</b></td><td><b>7.33</b></td><td><b>0.924</b></td><td>32.2</td><td class="neg">-0.0057（改善）</td><td>0.808</td><td><b>4.9%</b></td><td><b>BOR：低点 6.72 破前低，收盘破前收</b></td></tr>
        <tr><td>10-26</td><td>7.29</td><td class="neg"><b>0.4545</b></td><td>31.8</td><td>—</td><td>—</td><td>—</td><td>弱跟随，仍在 7.56 下方</td></tr>
      </tbody>
    </table></div>
    <p>10/23 的 K 线：开 7.07、高 7.38、低 6.72、收 7.33；低点破 10/22 低点 7.12，收盘高于 10/22 收盘 7.15，收盘高于开盘——<b>BOR</b>；下影 4.9%，收盘在当日区间 92.4% 位置。但它不是教科书锤子：下影/实体约 1.35，`hammer_shape=False`；量比 0.808、5/20 量比 0.572，<b>没有放量 capitulation</b>。</p>
    <p>10/26 收 7.29，收盘位置 0.4545，低于 10/23 收盘 7.33；没有收回 7.56，也没有收回 MA5 7.43/MA10 7.63。<b>10/26 的榜一不是第二次确认，只是“回撤最深”的状态在延续。</b></p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">03</span><h2>信号日特征：7 月赢家/输家与当前</h2></div>
    <div class="table-wrap"><table>
      <thead><tr><th>信号日</th><th>排名</th><th>15日收益</th><th>250日回撤</th><th>收盘位置</th><th>RSI</th><th>MACD柱</th><th>量比</th><th>距MA20</th><th>BOR</th><th>结果</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    <div class="figure"><img src="{img64('03_信号日散点.png')}" alt="603778 信号日散点：深度 vs 收盘位置"></div>
    <p class="caption">颜色=15日结果；形状=BOR/非BOR；浅色区域=深回撤（≤-65%）与高收盘（>0.8）。样本仅 603778 的 9 个信号日，10/26 为补充点。</p>
    <p><b>7 月输家不是“BOR 就赢”：</b>07-02 是 BOR，但回撤只有 -62.4%（浅）；07-08 回撤 -68.1% 但收盘位置只有 0.586（中）。<b>深度 + 收盘位置同时到位，才进入 10/23 的格子。</b>07-21 是完整版本：-75.0% + BOR + 收盘位置 0.971 + 锤子 + MACD 正，随后 +19.6%。</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">04</span><h2>BOR + 深度 + 高收盘：17 笔全为正</h2></div>
    <div class="table-wrap"><table>
      <thead><tr><th>条件</th><th>样本</th><th>均值</th><th>中位</th><th>胜率</th><th>最差</th><th>p10</th></tr></thead>
      <tbody>{cond_rows}</tbody>
    </table></div>
    <div class="figure"><img src="{img64('04_条件收益.png')}" alt="条件收益：深回撤、BOR、高收盘"></div>
    <p class="caption">450 条完整信号；分组重叠、样本非独立；描述性统计，不是新规则。</p>
    <div class="callout"><b>17 笔：10 笔止盈、7 笔到期小赚；均值 +14.5%、中位 +19.6%、最小 +1.4%。</b> 年份从 2017-12 到 2026-07，不是单一年份的产物；敏感性在 -65%/-70%、收盘位置 0.6–0.9 下仍为 100% 胜率。但 <b>17 笔、样本内、事后定义</b>，100% 不能当概率。</div>
    <p><b>当前不是 17 笔里的任何一笔：</b>把 MACD≤0、量比&lt;1、RSI≥30、MA20≥-10%、r250&gt;100% 全部叠上去，历史上 <b>0 笔</b>；去掉 r250 只剩 1 笔（2018-12-17 002907，+22.9%）。10/23 是“大格子里的形态”，不是历史样本的复制品。</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">05</span><h2>关键价位与相对强弱</h2></div>
    <div class="table-wrap"><table>
      <thead><tr><th>类型</th><th>价格</th><th>距现价</th><th>位置</th></tr></thead>
      <tbody>{level_rows}</tbody>
    </table></div>
    <div class="figure"><img src="{img64('05_相对强弱.png')}" alt="603778 与沪深主板等权的相对强弱"></div>
    <p class="caption">后复权；市场为沪深主板等权。7/21=100。</p>
    <p><b>上方：</b>7.56（7 月低点）→ 7.63（MA10）→ 8.08（MA20）→ <b>8.80（+20% 目标）</b>→ 9.29/9.31（78.6%/MA60）。<b>下方：</b>6.72（10/23 低点）→ <b>6.11（88.6% 回撤）</b>→ 4.50（2025-11 低点附近）→ 2.49（100% 回撤）。+20% 目标正好卡在 MA20 与 78.6% 之间，不是一条轻松的路。</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">06</span><h2>支持与反对：战术偏多，结构偏空</h2></div>
    <div class="two">
      <div class="box green"><h4>支持 15 日反弹</h4><ul>
        <li>深回撤 -78.6%，历史最好深度档（≤-75% 8 笔 +19.7%/100%）</li>
        <li>10/23 BOR + 收盘位置 0.924；深+BOR+高收盘 17/17 正</li>
        <li>RSI 28→32 底背离；MACD 柱整体改善</li>
        <li>市场 123 只大池子，深度排序更有效；88.6% 回撤 6.11 附近有支撑</li>
      </ul></div>
      <div class="box red"><h4>反对结构性底部</h4><ul>
        <li>已破 7 月低点 7.56，10 月是更低低点</li>
        <li>10/26 收盘位置 0.4545，弱跟随；仍在 7.56 下方</li>
        <li>价格低于所有均线；量比 0.808、5/20 0.572，无放量确认</li>
        <li>14 倍泡沫、250 日 +110%；7/21 以来相对市场 -17.6pp</li>
      </ul></div>
    </div>
    <div class="score">
      <div>15 日战术 <b class="s1">6.5/10</b>：BOR/深度/收盘位置强，但量能/相对弱/样本小</div>
      <div>结构性底部 <b class="s2">3/10</b>：破位、弱势、泡沫未清，BOR 不是底</div>
    </div>
    <div class="callout"><b>10/26 不是升级。</b> 它不是 BOR，收盘位置 0.4545，属于“深回撤 + 低收盘”格（+8.8%/74%），比 10/23 弱一档。10/23 名单已冻结：603778+605376（10/28）；10/26 新信号 603778+002240（10/29），不是补丁。已持有 603778 不加；若 10/28 一字板/停牌，跳过并保留现金。</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-no">07</span><h2>观察点与限制</h2></div>
    <ol class="steps">
      <li><b>10/27-10/29 能否收回 7.56：</b>收回 7 月低点是 BOR 的第一确认；收不回，仍是破位后的弱反抽。</li>
      <li><b>能否站上 8.08（MA20）：</b>站上才打开 8.80/9.29 的空间；站不上，+20% 目标难。</li>
      <li><b>量能是否放大：</b>量比 0.808、5/20 0.572；反弹要持续，需要量能回到 1 以上。</li>
      <li><b>6.72 是否失守：</b>失守则 BOR 失败，下一道 6.11；再下 4.50/2.49 是尾部。</li>
    </ol>
    <div class="limits">
      <b>限制：</b>10/26 OHLC 未入库，只有收盘/RSI/收盘位置/回撤/市场宽度；BOR 是事后定义，17 笔样本小、分组重叠，100% 胜率不能当概率；当前价等价按 10/23 复权因子折算，历史价格不是当时成交价；14 倍泡沫个股的截面规律不能覆盖基本面/流动性/退市风险；不是投资建议。<br>
      <b>一句话：</b>10/23 是这轮下跌里最好的 K 线反转，10/26 不是第二次确认；BOR 支持 15 日反弹，结构仍不支持“底部”。按 2 仓规则做，不因榜一加仓；跌破 6.72 则 BOR 失败，下一道 6.11。
    </div>
  </section>

  <div class="footer">
    <div><b>603778 · K 线与特征</b> · 2026-10-23 · 后复权 / 当前价等价 · 450 条完整信号</div>
    <div class="links">
      <a href="00_一页纸_603778_K线与特征.md">一页纸</a>
      <a href="01_完整报告_603778_K线与特征.md">完整报告</a>
      <a href="03_数据与复现.md">数据与复现</a>
      <a href="03_结果/信号日特征.csv">信号日特征</a>
      <a href="03_结果/条件统计.csv">条件统计</a>
      <a href="05_证据/校验.txt">校验</a>
    </div>
    <div>本页为研究记录，不构成投资建议。历史形态不代表未来；先纸面 50 笔。</div>
  </div>
</div>
"""

html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">' \
       f'<meta name="viewport" content="width=device-width, initial-scale=1">' \
       f'<title>603778：BOR 很好看，底部还没到</title><style>{CSS}</style></head>' \
       f'<body>{BODY}</body></html>'
HTML.write_text(html, encoding='utf-8')
print('saved', HTML, len(html))
