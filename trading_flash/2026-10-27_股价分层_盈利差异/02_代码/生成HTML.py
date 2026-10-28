# -*- coding: utf-8 -*-
"""把股价分层分析结果生成一页 HTML。"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
DATA = json.loads((OUT / "股价分层_数据.json").read_text(encoding="utf-8"))

HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>股价分层：低价真的更赚吗？｜当前策略 O03</title>
<style>
:root{
  --paper:#f2eee6; --card:#fffdf8; --ink:#211f1b; --ink2:#4d473d; --muted:#756d60;
  --faint:#9b9181; --line:#ded5c6; --line2:#c9bda9; --accent:#a93426; --soft:#f3e3dc;
  --green:#2e6b55; --green-soft:#e6efe9; --amber:#9a6a18; --amber-soft:#faf1dd;
  --blue:#3d5b72; --blue-soft:#edf3f7; --sage:#7d8f7a;
  --serif:"Songti SC","STSong","Noto Serif SC","Source Han Serif SC",Georgia,serif;
  --sans:"PingFang SC","Hiragino Sans GB","Noto Sans SC","Source Han Sans SC","Microsoft YaHei",sans-serif;
  --mono:"SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0; color:var(--ink); font-family:var(--sans); font-size:15px; line-height:1.65;
  background:
    radial-gradient(circle at 10% 0%, rgba(169,52,38,.05), transparent 30%),
    radial-gradient(circle at 94% 6%, rgba(61,91,114,.05), transparent 26%),
    var(--paper);
}
.wrap{max-width:1060px;margin:0 auto;padding:44px 28px 80px}
.kicker{font-size:12px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,4.6vw,46px);line-height:1.18;margin:10px 0 14px;letter-spacing:-.01em}
.lead{font-size:17px;color:var(--ink2);max-width:860px;margin:0}
.lead strong{color:var(--ink);font-weight:600}
.hero{padding-bottom:30px;border-bottom:2px solid var(--ink)}
.verdict{
  display:flex;align-items:center;gap:26px;flex-wrap:wrap;margin-top:26px;padding:18px 22px;
  background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 1px 0 rgba(33,31,27,.03)
}
.verdict .num{font-family:var(--serif);font-size:30px;line-height:1.1;display:block}
.verdict .lbl{font-size:12px;color:var(--muted);display:block;margin-top:2px}
.verdict .vs{font-family:var(--serif);color:var(--faint);font-size:18px}
.verdict .note{margin-left:auto;font-size:13px;color:var(--muted);border-left:1px solid var(--line);padding-left:22px}
.verdict .note b{color:var(--accent);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0 6px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-family:var(--serif);font-size:24px;line-height:1.2}
.kpi .k{font-size:12px;color:var(--muted);margin-top:2px}
section{margin-top:52px}
h2{font-family:var(--serif);font-size:26px;font-weight:600;margin:0 0 10px;letter-spacing:-.01em}
h3{font-size:16px;margin:26px 0 8px}
p{margin:0 0 12px;color:var(--ink2)}
.muted{color:var(--muted)}
.small{font-size:12.5px;color:var(--muted);line-height:1.6}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-top:16px}
.chart-head{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
svg{display:block;width:100%;height:auto;margin-top:10px}
.grid{stroke:#e7dfd1;stroke-width:1}
.zero{stroke:#b9ad99;stroke-width:1.2}
.axis{font-family:var(--mono);font-size:11px;fill:var(--muted)}
.axislab{font-family:var(--sans);font-size:11.5px;fill:var(--muted)}
.bar-label{font-family:var(--mono);font-size:10.5px;fill:var(--ink2);paint-order:stroke;stroke:var(--card);stroke-width:3px;stroke-linejoin:round}
.band-bad{fill:rgba(169,52,38,.045)}
.whisker{stroke:#5c5346;stroke-width:1.1}
.whisker-cap{stroke:#5c5346;stroke-width:1.1}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:12px}
th{font-weight:500;color:var(--muted);font-size:11.5px;letter-spacing:.04em;text-align:right;padding:8px 8px;border-bottom:1.5px solid var(--line2);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:hover{background:rgba(169,52,38,.03)}
tr.row-bad td{background:rgba(169,52,38,.055)}
tr.row-low td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.up{color:var(--accent)}
.down{color:var(--green)}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;line-height:1.7;background:var(--soft);color:var(--accent);white-space:nowrap}
.tag.gray{background:#eee8dc;color:var(--muted)}
.tag.green{background:var(--green-soft);color:var(--green)}
.table-wrap{overflow-x:auto;margin-top:4px}
.note{margin-top:10px;font-size:12.5px;color:var(--muted);line-height:1.65}
.quart{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.q{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.q .qt{font-size:12px;color:var(--muted)}
.q .qv{font-family:var(--serif);font-size:23px;line-height:1.3}
.q .qs{font-size:11.5px;color:var(--muted)}
.split-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px}
.mini{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.mini .v{font-family:var(--serif);font-size:26px;line-height:1.2}
.mini .k{font-size:12px;color:var(--muted)}
details{margin-top:14px;border-top:1px dashed var(--line);padding-top:12px}
summary{cursor:pointer;font-size:13px;color:var(--blue)}
.callout{border-left:3px solid var(--accent);background:var(--soft);padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0;font-size:14px;color:var(--ink2)}
.callout b{color:var(--ink)}
ul.clean{margin:8px 0 0;padding-left:20px;color:var(--ink2)}
ul.clean li{margin:8px 0}
ul.clean li b{color:var(--ink)}
.footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line2);font-size:12px;color:var(--muted);line-height:1.7}
@media (max-width:820px){
  .wrap{padding:28px 16px 60px}
  .kpis{grid-template-columns:repeat(2,1fr)}
  .quart{grid-template-columns:repeat(2,1fr)}
  .split-grid{grid-template-columns:1fr}
  .verdict .note{margin-left:0;border-left:none;padding-left:0}
  .verdict{gap:16px}
  th,td{padding:7px 6px;font-size:12.5px}
}
@media print{
  body{background:#fff}
  .wrap{max-width:none;padding:16px}
  section{break-inside:avoid}
}
</style>
</head>
<body>
<div class="wrap">

<header class="hero">
  <div class="kicker">当前策略 · 主板恐慌洗盘反转 O03 · 实际价账户 · 97 笔已平仓</div>
  <h1>股价分层：低价真的更赚吗？</h1>
  <p class="lead">把交易按<strong>实际成交价</strong>分成六档后，结论是：<strong>没有稳定的“低于 15 元更赚”</strong>。低于 15 元与 15 元以上的每笔平均净收益几乎一样；按投入资金加权，低价档反而更低。真正难看的是 &lt;5 元档，而它主要被 2026 年 6–7 月的三笔亏损拖累。</p>
  <p class="small" style="margin-top:10px">范围：本页只分析 <b>trading_flash</b> 的当前策略 O03（主板恐慌洗盘反转，实际价账户）。21 号 45 仓位主系统、70/15 子池是另一套信号与持有期，结论不可直接互推。</p>
  <div class="verdict">
    <div><span class="num up" id="vLt">+6.68%</span><span class="lbl">低于 15 元 · 63 笔 · 每笔平均</span></div>
    <div class="vs">vs</div>
    <div><span class="num up" id="vGe">+6.53%</span><span class="lbl">15 元以上 · 34 笔 · 每笔平均</span></div>
    <div class="note">差 <b>+0.16 个百分点</b><br>Welch p=0.95 · 统计上等于零<br><span class="muted">按资金加权：+5.57% vs +7.55%</span></div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="v">97</div><div class="k">笔已平仓交易（另有 2 笔未平仓）</div></div>
    <div class="kpi"><div class="v up">+31.79 万</div><div class="k">累计净盈亏（费用后）</div></div>
    <div class="kpi"><div class="v">70.1%</div><div class="k">胜率</div></div>
    <div class="kpi"><div class="v">12.9 日</div><div class="k">平均持有交易日</div></div>
  </div>
</header>

<section>
  <h2>一、六档：不是越低越好，是中间略好</h2>
  <p>按实际成交价切六档。每笔平均最高的是 10–15 元档（+8.6%），最低的是 &lt;5 元档（+2.6%）；但除 &lt;5 元外，其余五档都落在 +4% 到 +9% 之间，95% 区间大面积重叠。价格和收益的关系更像倒 U，而不是单调向下。</p>
  <div class="card">
    <div class="chart-head">
      <div class="legend">
        <span><i style="background:#a93426"></i>每笔平均（等权）</span>
        <span><i style="background:#3d5b72"></i>资金加权（总盈亏÷总成本）</span>
        <span><i style="background:#5c5346;height:2px;border-radius:0"></i>95% 区间</span>
      </div>
      <span class="small">横轴：实际成交价（元）</span>
    </div>
    <svg id="barChart" role="img" aria-label="按实际成交价分档的每笔平均收益与资金加权收益"></svg>
    <div class="note">条形上方的数字是每笔平均净收益率；蓝色条是资金加权收益。&lt;5 元档每笔平均仍为正，但资金加权为负——因为亏损发生在后期更大的仓位上。</div>
  </div>
  <div class="table-wrap">
    <table id="bucketTable">
      <thead><tr>
        <th>价格档</th><th>笔数</th><th>胜率</th><th>每笔平均</th><th>95% 区间</th>
        <th>中位数</th><th>资金加权</th><th>止盈占比</th><th>净盈亏</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="note">每笔平均 = 97 笔等权；资金加权 = 档内净盈亏合计 ÷ 档内买入成本合计，会被后期大仓位交易放大，不能当成“只做这一档”的收益。</div>
  <h3>换成四分位，形态一样：中间两档强，两端弱</h3>
  <div class="quart" id="quart"></div>
  <div class="note">四分位按 97 笔的实际成交价排序切分，每档约 24–25 笔。Q1 最便宜、Q4 最贵；资金加权同样受后期大仓位影响。</div>
  <div class="card" style="margin-top:18px">
    <div class="chart-head">
      <h3 style="margin:0">逐笔散点：价格横着走，收益竖着跳</h3>
      <div class="legend">
        <span><i style="background:#9a6a18"></i>2017–19</span>
        <span><i style="background:#3d5b72"></i>2020–22</span>
        <span><i style="background:#7d8f7a"></i>2023–25</span>
        <span><i style="background:#a93426"></i>2026</span>
      </div>
    </div>
    <svg id="scatter" role="img" aria-label="实际成交价与单笔净收益的散点图"></svg>
    <div class="note">每个点是一笔已平仓交易；横轴为对数刻度。低价区既有大赢家也有大亏损，价格本身分不开收益。红色虚线是全样本平均 +6.63%。</div>
  </div>
</section>

<section>
  <h2>二、同日对照：把市场涨跌拿掉</h2>
  <p>同一天买入的两只票，面对同一个市场窗口，价差不是选股逻辑，而是一个天然对照。31 个“双仓”买入日里，便宜的那只 17 次跑赢（54.8%），平均多赚 2.50 个百分点，p=0.34；只看一笔低于 15、一笔 15 元以上的 15 组，便宜的那只 8 次跑赢，平均多 3.74 个百分点。方向略偏低价，但远没到能当规则的程度。</p>
  <div class="kpis" style="grid-template-columns:repeat(3,1fr)">
    <div class="kpi"><div class="v">17 / 31</div><div class="k">便宜的那只跑赢次数</div></div>
    <div class="kpi"><div class="v up">+2.50pp</div><div class="k">平均收益差（便宜 − 贵）</div></div>
    <div class="kpi"><div class="v">−2.49 ~ +7.45pp</div><div class="k">95% 区间 · p=0.34</div></div>
  </div>
  <details>
    <summary>展开 15 组“跨 15 元档”的同日配对</summary>
    <div class="table-wrap">
      <table id="pairTable">
        <thead><tr><th>买入日</th><th>便宜（代码/价）</th><th>贵（代码/价）</th><th>便宜收益</th><th>贵收益</th><th>差</th><th>谁赢</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="note">只列一笔 &lt;15、一笔 ≥15 的配对；同日两笔都落在同一档的 16 组未列。差 = 便宜收益 − 贵收益。</div>
  </details>
</section>

<section>
  <h2>三、换一种价格基准，结论不变</h2>
  <p>实际价会被拆股/送转改写，所以再用信号日收盘价和后复权价各切一次。三套口径都没有出现“低于 15 元显著更好”。后复权口径的差值略大（+1.29pp），但 p=0.59，而且资金加权方向相反（+4.50% vs +8.02%）——等权说低价好、钱说高价好，说明差异只是噪声和仓位时点的混合。</p>
  <div class="table-wrap">
    <table id="baseTable">
      <thead><tr><th>价格基准</th><th>&lt;15 笔数</th><th>&lt;15 每笔平均</th><th>&lt;15 资金加权</th><th>≥15 笔数</th><th>≥15 每笔平均</th><th>≥15 资金加权</th><th>差值</th><th>p 值</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="note">p 值为 Welch t 检验；差值为“&lt;15 平均 − ≥15 平均”。后复权价使用买入日的 hfq_open，实际价与信号日收盘价来自同一面板。</div>
  <h3>分阶段看，方向会翻转</h3>
  <div class="table-wrap">
    <table id="eraTable">
      <thead><tr><th>阶段</th><th>&lt;15 笔数</th><th>&lt;15 每笔</th><th>&lt;15 资金加权</th><th>≥15 笔数</th><th>≥15 每笔</th><th>≥15 资金加权</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="note">2017–19 低价档更强，2020–22 反过来，2023–26 又是高价略强。方向不稳定，说明“价格低”本身不是一个可依赖的收益因子。</div>
</section>

<section>
  <h2>四、&lt;5 元档为什么难看：主要是 2026 年 6–7 月的三笔亏损</h2>
  <p>&lt;5 元档 10 笔，每笔平均还是 +2.6%，但资金加权 −3.4%、合计亏 3.34 万。拆开看，4 笔 2026 年交易平均 −9.3%（合计 −5.87 万），其余 6 笔平均 +10.6%（合计 +2.53 万）。亏损集中在 2026-06-08 和 2026-07-16 两批。6 月 8 日两笔伴随等权市场 −6.9%；7 月 16 日那笔市场 +2.3% 仍亏 12.1%，是个股自身问题。不管归因于市场窗口还是个股，这 4 笔都不是“低价”这个标签的稳定规律。</p>
  <div class="card">
    <div class="chart-head"><h3 style="margin:0">2026 年的 9 笔：低价档的亏损集中在 6–7 月</h3><span class="small">浅色底 = 实际成交价低于 15 元</span></div>
    <div class="table-wrap">
      <table id="y2026Table">
        <thead><tr><th>买入日</th><th>代码</th><th>成交价</th><th>净收益</th><th>净盈亏</th><th>退出</th><th>同期市场</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <div class="callout"><b>实际价不是稳定的股票特征。</b> 603007 实际 4.73 元、后复权 22.57 元；002777 实际 23.71 元、后复权 118.36 元。用实际价分档，混进了“曾经拆过股”这一层；用后复权价分档，也看不到“越低越好”。</div>
  <div class="table-wrap">
    <table id="splitTable">
      <thead><tr><th>买入日</th><th>代码</th><th>实际成交价</th><th>后复权价</th><th>倍数</th><th>该笔净收益</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
</section>

<section>
  <h2>五、结论与边界</h2>
  <ul class="clean">
    <li><b>不要因为“低于 15 更好”给当前策略加价格过滤。</b> 97 笔里没有这个证据：&lt;15 与 ≥15 的每笔平均差 +0.16pp（p=0.95），资金加权反而是 +5.57% vs +7.55%。</li>
    <li><b>子集统计不等于过滤器回测。</b> 真要加价格上限，必须改候选池、重新排队成交并重跑账户；不能从这 97 笔里直接推断“过滤后更好”。现在能说的是：没有看到值得加的理由。</li>
    <li><b>如果一定要设一条价格底线，&lt;5 元档是唯一资金加权为负的档。</b> 但它只有 10 笔，亏损集中在 2026 年 6–7 月，且拆股会污染实际价；不足以立规则。</li>
    <li><b>10–15 元档描述性最好</b>（+8.6%，止盈率 36.8%），但只有 19 笔，95% 区间 [+3.5%, +13.5%]，与 5–10、15–25 大量重叠，不能当成“甜点区”去优化。</li>
    <li><b>这套策略的收益更多来自市场 β 和个股反转。</b> 97 笔对同期等权市场的回归：β≈0.83、α≈+5.4%/笔，R²≈0.12；价格与收益的 Spearman 相关只有 0.03，分档解释不了差异。</li>
    <li><b>这是历史回测，不是独立样本外证明。</b> 2017-11 至 2026-08，2024–2026 已参与研究；不构成投资建议。</li>
  </ul>
</section>

<div class="footer">
  <b>数据口径</b>：O03_c20000 实际价账户，20,000 元起，2 仓，单仓约 50%，信号后第 3 个交易日开盘买入，持有 15 个交易日或 +20% 止盈；100 股整手，佣金/过户费/印花税/0.1% 滑点/分红税后。收益率 = 单笔净收益率。<br>
  <b>样本</b>：97 笔已平仓（2017-11-23 至 2026-08-10）；另有 2 笔未平仓（600783 13.27 元、603778 7.51 元，均低于 15 元，2026-10-23 合计浮动约 −1.0%），未计入。<br>
  <b>统计</b>：Welch t 检验、Mann–Whitney、20,000 次 bootstrap、同日配对；所有分档均为描述统计，未做多重比较校正。<br>
  <b>来源</b>：<span style="font-family:var(--mono)">flash 审计修复/results/accounts/O03_c20000/trades.csv</span>；<span style="font-family:var(--mono)">research_panel.parquet</span>。生成日期 2026-10-27。
</div>

</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const sgn = (v, d = 1) => (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + '%';
const money = v => (v >= 0 ? '+' : '') + (v / 10000).toFixed(2) + ' 万';
const cls = v => v >= 0 ? 'up' : 'down';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const dispBucket = b => b.replace('>=40', '≥40').replace(/^(\d+)-(\d+)$/, '$1–$2');
const reason = r => r === 'profit_target' ? '止盈' : (r === 'max_holding' ? '持有到期' : r);

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function put(parent, x, y, str, klass, anchor = 'middle') {
  const t = el('text', {x, y, class: klass, 'text-anchor': anchor}, parent);
  t.textContent = str;
  return t;
}

/* ---------- KPI / verdict ---------- */
document.getElementById('vLt').textContent = sgn(D.lt15.avg, 2);
document.getElementById('vGe').textContent = sgn(D.ge15.avg, 2);
document.querySelector('.verdict .note').innerHTML =
  `差 <b>${(D.lt15.avg - D.ge15.avg >= 0 ? '+' : '') + ((D.lt15.avg - D.ge15.avg) * 100).toFixed(2)} 个百分点</b><br>` +
  `Welch p=${D.welch_p.toFixed(2)} · 统计上等于零<br>` +
  `<span class="muted">按资金加权：${sgn(D.lt15.capw, 2)} vs ${sgn(D.ge15.capw, 2)}</span>`;

/* ---------- 六档表 ---------- */
const tb = document.querySelector('#bucketTable tbody');
D.buckets.forEach(b => {
  const tr = document.createElement('tr');
  if (b.bucket === '<5') tr.className = 'row-bad';
  tr.innerHTML = `<td>${esc(dispBucket(b.bucket))} 元</td><td>${b.n}</td><td>${pct(b.win)}</td>` +
    `<td class="${cls(b.avg)}">${sgn(b.avg, 2)}</td><td class="muted">[${sgn(b.ci_lo)}, ${sgn(b.ci_hi)}]</td>` +
    `<td>${sgn(b.median, 2)}</td><td class="${cls(b.cap_weighted)}">${sgn(b.cap_weighted, 2)}</td>` +
    `<td>${pct(b.tp_share, 0)}</td><td class="${cls(b.pnl)}">${money(b.pnl)}</td>`;
  tb.appendChild(tr);
});

/* ---------- 四分位 ---------- */
const qbox = document.getElementById('quart');
D.quartiles.forEach(q => {
  const d = document.createElement('div');
  d.className = 'q';
  d.innerHTML = `<div class="qt">${esc(q.quartile)}</div>` +
    `<div class="qv ${cls(q.avg)}">${sgn(q.avg)}</div>` +
    `<div class="qs">${q.n} 笔 · 胜率 ${pct(q.win, 0)} · 资金加权 ${sgn(q.cap_weighted)}</div>`;
  qbox.appendChild(d);
});

/* ---------- 柱状图 ---------- */
function drawBars() {
  const svg = document.getElementById('barChart');
  const W = 780, H = 390, m = {t: 34, r: 18, b: 66, l: 58};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const ymin = -0.09, ymax = 0.20;
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const y = v => m.t + (ymax - v) / (ymax - ymin) * ph;
  const band = pw / D.buckets.length;

  for (let v = -0.05; v <= 0.2001; v += 0.05) {
    el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: 'grid'}, svg);
    put(svg, m.l - 10, y(v) + 4, (v * 100).toFixed(0) + '%', 'axis', 'end');
  }
  el('line', {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: 'zero'}, svg);

  D.buckets.forEach((b, i) => {
    const x0 = m.l + band * i;
    const cx = x0 + band / 2;
    if (b.bucket === '<5') el('rect', {x: x0 + 3, y: m.t, width: band - 6, height: ph, class: 'band-bad'}, svg);

    const bw = band * 0.27, gap = band * 0.055;
    const x1 = cx - bw - gap / 2, x2 = cx + gap / 2;

    // 等权
    const y0 = y(0), y1 = y(b.avg);
    el('rect', {
      x: x1, y: Math.min(y0, y1), width: bw, height: Math.max(1.2, Math.abs(y1 - y0)),
      fill: '#a93426', rx: 1.5
    }, svg);
    // 资金加权
    const y2 = y(b.cap_weighted);
    el('rect', {
      x: x2, y: Math.min(y0, y2), width: bw, height: Math.max(1.2, Math.abs(y2 - y0)),
      fill: '#3d5b72', rx: 1.5
    }, svg);

    // 95% 区间
    const xc = x1 + bw / 2;
    el('line', {x1: xc, x2: xc, y1: y(b.ci_lo), y2: y(b.ci_hi), class: 'whisker'}, svg);
    el('line', {x1: xc - 5, x2: xc + 5, y1: y(b.ci_lo), y2: y(b.ci_lo), class: 'whisker-cap'}, svg);
    el('line', {x1: xc - 5, x2: xc + 5, y1: y(b.ci_hi), y2: y(b.ci_hi), class: 'whisker-cap'}, svg);

    // 数值
    const ly = b.avg >= 0 ? y(b.avg) - 7 : y(b.avg) + 14;
    put(svg, xc, ly, (b.avg >= 0 ? '+' : '') + (b.avg * 100).toFixed(1) + '%', 'bar-label');
    const ly2 = b.cap_weighted >= 0 ? y(b.cap_weighted) - 6 : y(b.cap_weighted) + 13;
    put(svg, x2 + bw / 2, ly2, (b.cap_weighted >= 0 ? '+' : '') + (b.cap_weighted * 100).toFixed(1) + '%', 'bar-label');

    // x 标签
    put(svg, cx, m.t + ph + 24, dispBucket(b.bucket), 'axis');
    put(svg, cx, m.t + ph + 42, `n=${b.n}`, 'axislab');
  });

  // 轴标题
  put(svg, m.l + pw / 2, H - 12, '实际成交价（元）', 'axislab');
  const yl = el('text', {x: 14, y: m.t + ph / 2, class: 'axislab', 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ph / 2})`}, svg);
  yl.textContent = '单笔净收益率';
}

/* ---------- 散点图 ---------- */
function drawScatter() {
  const svg = document.getElementById('scatter');
  const W = 780, H = 400, m = {t: 26, r: 22, b: 58, l: 58};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const xmin = 1.5, xmax = 120, ymin = -0.18, ymax = 0.25;
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const x = v => m.l + (Math.log10(v) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * pw;
  const y = v => m.t + (ymax - v) / (ymax - ymin) * ph;

  for (let v = -0.15; v <= 0.2501; v += 0.05) {
    el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: 'grid'}, svg);
    put(svg, m.l - 10, y(v) + 4, (v * 100).toFixed(0) + '%', 'axis', 'end');
  }
  [2, 5, 10, 15, 20, 50, 100].forEach(v => {
    el('line', {x1: x(v), x2: x(v), y1: m.t, y2: m.t + ph, class: 'grid'}, svg);
    put(svg, x(v), m.t + ph + 22, String(v), 'axis');
  });
  el('line', {x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: 'zero'}, svg);
  el('line', {x1: m.l, x2: W - m.r, y1: y(D.meta.avg), y2: y(D.meta.avg), stroke: '#a93426', 'stroke-width': 1.2, 'stroke-dasharray': '5 4', opacity: .65}, svg);
  put(svg, W - m.r - 4, y(D.meta.avg) - 6, `全样本 ${sgn(D.meta.avg)}`, 'axislab', 'end');

  const colors = {'2017-19': '#9a6a18', '2020-22': '#3d5b72', '2023-25': '#7d8f7a', '2026': '#a93426'};
  const era = t => t.era === '2023-26' ? (t.buy_date.slice(0, 4) === '2026' ? '2026' : '2023-25') : t.era;
  D.trades.forEach(t => {
    const c = el('circle', {cx: x(t.buy_price), cy: y(t.return), r: 4.3, fill: colors[era(t)] || '#756d60', opacity: .82}, svg);
    const title = el('title', {}, c);
    title.textContent = `${t.buy_date} ${t.thscode}\n成交价 ${t.buy_price.toFixed(2)} 元\n净收益 ${sgn(t.return)} · ${money(t.net_pnl)}`;
  });

  put(svg, m.l + pw / 2, H - 14, '实际成交价（元，对数轴）', 'axislab');
  const yl = el('text', {x: 14, y: m.t + ph / 2, class: 'axislab', 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ph / 2})`}, svg);
  yl.textContent = '单笔净收益率';
}

/* ---------- 同日配对 ---------- */
const pt = document.querySelector('#pairTable tbody');
D.pairs.filter(p => p['跨15元档']).forEach(p => {
  const tr = document.createElement('tr');
  const win = p['差(便宜-贵)'] > 0;
  tr.innerHTML = `<td>${p['买入日']}</td>` +
    `<td>${esc(p['便宜代码'])} · ${p['便宜价'].toFixed(2)}</td>` +
    `<td>${esc(p['贵代码'])} · ${p['贵价'].toFixed(2)}</td>` +
    `<td class="${cls(p['便宜收益'])}">${sgn(p['便宜收益'])}</td>` +
    `<td class="${cls(p['贵收益'])}">${sgn(p['贵收益'])}</td>` +
    `<td class="${cls(p['差(便宜-贵)'])}">${sgn(p['差(便宜-贵)'])}</td>` +
    `<td><span class="tag ${win ? '' : 'green'}">${win ? '便宜赢' : '贵赢'}</span></td>`;
  pt.appendChild(tr);
});

/* ---------- 换基准 ---------- */
const bt = document.querySelector('#baseTable tbody');
D.baselines.forEach(b => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${esc(b['基准'])}</td><td>${b['低于15笔数']}</td>` +
    `<td class="${cls(b['低于15平均'])}">${sgn(b['低于15平均'])}</td>` +
    `<td class="${cls(b['低于15资金加权'])}">${sgn(b['低于15资金加权'])}</td>` +
    `<td>${b['15以上笔数']}</td>` +
    `<td class="${cls(b['15以上平均'])}">${sgn(b['15以上平均'])}</td>` +
    `<td class="${cls(b['15以上资金加权'])}">${sgn(b['15以上资金加权'])}</td>` +
    `<td class="${cls(b['差值'])}">${sgn(b['差值'], 2)}</td><td>${b['p值'].toFixed(2)}</td>`;
  bt.appendChild(tr);
});

/* ---------- 分阶段 ---------- */
const et = document.querySelector('#eraTable tbody');
D.era_table.forEach(b => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${esc(b['阶段'])}</td><td>${b['低于15笔数']}</td>` +
    `<td class="${cls(b['低于15每笔平均'])}">${sgn(b['低于15每笔平均'])}</td>` +
    `<td class="${cls(b['低于15资金加权'])}">${sgn(b['低于15资金加权'])}</td>` +
    `<td>${b['15以上笔数']}</td>` +
    `<td class="${cls(b['15以上每笔平均'])}">${sgn(b['15以上每笔平均'])}</td>` +
    `<td class="${cls(b['15以上资金加权'])}">${sgn(b['15以上资金加权'])}</td>`;
  et.appendChild(tr);
});

/* ---------- 2026 表 ---------- */
const yt = document.querySelector('#y2026Table tbody');
D.y2026.slice().sort((a, b) => a.buy_date.localeCompare(b.buy_date)).forEach(r => {
  const tr = document.createElement('tr');
  if (r.buy_price < 15) tr.className = 'row-low';
  tr.innerHTML = `<td>${r.buy_date}</td><td>${esc(r.thscode)}</td><td>${r.buy_price.toFixed(2)}</td>` +
    `<td class="${cls(r.return)}">${sgn(r.return)}</td>` +
    `<td class="${cls(r.net_pnl)}">${money(r.net_pnl)}</td>` +
    `<td>${reason(r.exit_reason)}</td><td class="${cls(r.mkt)}">${sgn(r.mkt)}</td>`;
  yt.appendChild(tr);
});

/* ---------- 拆股例子 ---------- */
const st = document.querySelector('#splitTable tbody');
D.splits.forEach(r => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${r.buy_date}</td><td>${esc(r.thscode)}</td><td>${r.buy_price.toFixed(2)}</td>` +
    `<td>${r.buy_hfq_open.toFixed(2)}</td><td>${r.hfq_ratio.toFixed(2)}×</td>` +
    `<td class="${cls(r.return)}">${sgn(r.return)}</td>`;
  st.appendChild(tr);
});

drawBars();
drawScatter();
</script>
</body>
</html>
"""

html = HTML.replace("__DATA__", json.dumps(DATA, ensure_ascii=False, separators=(",", ":")))
path = OUT / "股价分层_盈利差异.html"
path.write_text(html, encoding="utf-8")
print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)")
