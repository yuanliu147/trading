#!/usr/bin/env python3
"""生成单文件可视化 HTML（图片 base64 内嵌，可离线打开）。"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "03_结果"
FIG = ROOT / "04_图表"
OUT = ROOT / "策略补充研究.html"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}">'


def fmt_pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)


def table(df: pd.DataFrame, pct_cols=(), cls="data") -> str:
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in pct_cols:
                v = fmt_pct(v)
            elif isinstance(v, float):
                v = f"{v:.2f}"
            cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


CSS = """
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#18222d; --muted:#63707e;
  --blue:#1f6fb2; --blue2:#e8f1fa; --red:#b83b2f; --red2:#fbecea;
  --amber:#b8860b; --amber2:#fdf6e3; --green:#2c7a4b; --line:#e3e7ec;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-size:15px;line-height:1.7;}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px 64px}
header{padding:56px 0 30px;border-bottom:1px solid var(--line)}
.kicker{color:var(--blue);font-weight:600;letter-spacing:.08em;font-size:12px;text-transform:uppercase}
h1{font-size:40px;line-height:1.2;margin:10px 0 12px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:16px;max-width:820px}
.banner{margin:26px 0 0;background:var(--red2);border-left:4px solid var(--red);
  padding:16px 20px;border-radius:0 10px 10px 0;color:#5a1f18;font-weight:600}
nav{position:sticky;top:0;background:rgba(246,247,249,.92);backdrop-filter:blur(10px);
  z-index:20;border-bottom:1px solid var(--line);margin-bottom:28px}
nav .inner{max-width:1120px;margin:0 auto;padding:10px 24px;display:flex;gap:18px;overflow:auto}
nav a{color:var(--muted);text-decoration:none;font-size:13px;white-space:nowrap}
nav a:hover{color:var(--blue)}
section{margin:38px 0}
h2{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h3{font-size:18px;margin:26px 0 8px}
p{margin:8px 0 12px}
.small{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 18px 14px}
.card .label{color:var(--muted);font-size:12px;letter-spacing:.02em}
.card .big{font-size:28px;font-weight:700;margin:4px 0 2px;letter-spacing:-.02em}
.card .note{color:var(--muted);font-size:12px}
.card.red .big{color:var(--red)} .card.blue .big{color:var(--blue)} .card.green .big{color:var(--green)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:14px 0}
.panel.blue{background:var(--blue2);border-color:#cfe2f3}
.panel.red{background:var(--red2);border-color:#f0cfca}
.panel.amber{background:var(--amber2);border-color:#efdfb8}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
img{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
figure{margin:14px 0 6px}
figcaption{color:var(--muted);font-size:12px;margin-top:6px}
table.data{width:100%;border-collapse:collapse;font-size:13px;background:#fff;
  border:1px solid var(--line);border-radius:10px;overflow:hidden}
table.data th{background:#eef2f6;text-align:right;padding:9px 10px;font-weight:600;color:#33414f}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data td{padding:8px 10px;border-top:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
table.data tr:hover td{background:#fafbfd}
.tablewrap{overflow:auto;border-radius:10px}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.tag.blue{background:var(--blue2);color:var(--blue)} .tag.red{background:var(--red2);color:var(--red)}
.tag.amber{background:var(--amber2);color:var(--amber)} .tag.green{background:#e7f4ec;color:var(--green)}
ul{margin:8px 0 12px 20px;padding:0} li{margin:5px 0}
.kv{display:grid;grid-template-columns:180px 1fr;gap:6px 14px;font-size:14px}
.kv .k{color:var(--muted)} .kv .v{font-weight:500}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px;margin-top:44px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}
  h1{font-size:30px}.wrap{padding:0 16px 48px}.kv{grid-template-columns:120px 1fr}}
"""


def main():
    core = pd.read_csv(R / "核心结论.csv")
    core = core.rename(columns={"期间": "期间", "口径": "口径", "年化": "年化", "最大回撤": "最大回撤",
                                "卡玛": "卡玛", "交易数": "交易", "胜率": "胜率", "期末倍数": "期末倍数"})
    core["口径"] = core["口径"].replace({
        "A_仅当前上市": "仅当前上市", "B_含退市_无ST过滤": "含退市股",
        "C_含退市_排除ST": "含退市+排除风险警示"})
    core_t = core[["期间", "口径", "年化", "最大回撤", "卡玛", "交易", "胜率", "期末倍数"]]
    core_html = table(core_t, pct_cols=("年化", "最大回撤", "胜率"))

    dec = pd.read_csv(R / "口径分解.csv")
    dec["标签"] = dec["标签"].replace({
        "A_仅上市_无ST过滤": "仅上市", "A_ST_仅上市_排除ST": "仅上市+排风险警示",
        "C_含退市_排除ST": "含退市+排风险警示"})
    dec_t = dec[["期间", "标签", "cagr", "max_dd", "n_trades"]].rename(
        columns={"标签": "口径", "cagr": "年化", "max_dd": "最大回撤", "n_trades": "交易"})
    dec_html = table(dec_t, pct_cols=("年化", "最大回撤"))

    risk = pd.read_csv(R / "风险叠加汇总.csv")
    risk["标签"] = risk["标签"].replace({
        "C_基础_无止损": "无止损", "C_止损15": "止损15%", "C_止损20": "止损20%",
        "C_止损25": "止损25%", "C_止损15_仓位35": "止损15%+仓位35%",
        "C_止损25_仓位35": "止损25%+仓位35%", "C_止损15_仓位25": "止损15%+仓位25%",
        "C_止损25_仓位25": "止损25%+仓位25%"})
    risk_t = risk[["期间", "标签", "cagr", "max_dd", "calmar", "n_trades"]].rename(
        columns={"标签": "方案", "cagr": "年化", "max_dd": "最大回撤", "calmar": "卡玛", "n_trades": "交易"})
    risk_html = table(risk_t, pct_cols=("年化", "最大回撤"))

    # 年度表：只取关键年份
    def annual(folder, label):
        s = pd.read_csv(R / folder / f"{label}_年度.csv", index_col=0).iloc[:, 0]
        s.index = s.index.astype(int).astype(str)
        return s
    A = annual("2005_2026", "A_仅当前上市")
    B = annual("2005_2026", "B_含退市股")
    C = annual("指标优化/2005-2026", "C_排除ST")
    years = ["2008", "2009", "2015", "2016", "2018", "2022", "2024", "2025", "2026"]
    ann = pd.DataFrame({"年份": years,
                        "仅上市": [A.get(y, None) for y in years],
                        "含退市": [B.get(y, None) for y in years],
                        "含退市+排风险警示": [C.get(y, None) for y in years]})
    ann_html = table(ann, pct_cols=("仅上市", "含退市", "含退市+排风险警示"))

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>退市股补充与指标检验 · baostock</title>
<style>{CSS}</style></head>
<body>
<div class="wrap">
<header>
  <div class="kicker">baostock 全主板 · 2005-2026 · 研究笔记</div>
  <h1>退市股补充与指标检验</h1>
  <div class="sub">把 295 只退市股、历史风险警示、换手率和 2016 年前数据补进“恐慌洗盘反转”策略。结论不是“又优化了 5 个点”，而是：<b>40% 是 2017-2026 顺风期的结果，完整样本里没有二十年 40% 这回事。</b></div>
  <div class="banner">补退市股后，2017-2026 从 43.2% / -15.8% 变成 29.8% / -54.3%；排除风险警示回到 35.0% / -27.4%；2005-2016 样本外只有 4.9% / -71.5%。</div>
</header>

<nav><div class="inner">
  <a href="#numbers">关键数字</a><a href="#survivor">退市股</a><a href="#case">002684</a>
  <a href="#oos">2016 前</a><a href="#indicators">指标检验</a><a href="#risk">止损/仓位</a>
  <a href="#current">当前信号</a><a href="#limits">限制</a>
</div></nav>

<section id="numbers">
  <h2>三套口径</h2>
  <p class="small">A 仅当前上市 = 旧研究的幸存者口径；B 含退市股 = 完整主板池；C 含退市+排除风险警示 = 完整池，但信号日 is_st=1 不买。收益为后复权、实际价成交、含费用。</p>
  <div class="cards">
    <div class="card blue"><div class="label">2017-2026 仅当前上市</div><div class="big">43.2%</div><div class="note">最大回撤 -15.8% · 100 笔</div></div>
    <div class="card red"><div class="label">2017-2026 含退市股</div><div class="big">29.8%</div><div class="note">最大回撤 -54.3% · 101 笔</div></div>
    <div class="card green"><div class="label">2017-2026 含退市+排风险警示</div><div class="big">35.0%</div><div class="note">最大回撤 -27.4% · 101 笔</div></div>
    <div class="card red"><div class="label">2005-2016 样本外</div><div class="big">4.9%</div><div class="note">最大回撤 -71.5% · 110 笔</div></div>
  </div>
  <div class="tablewrap">{core_html}</div>
  <p class="small">2005-2026 全周期：A 18.3% / -74.1%；B 13.2% / -75.5%；C 16.8% / -71.5%。同期主板等权指数 16.2% / -72.1%——全周期与等权指数打平。</p>
</section>

<section id="survivor">
  <h2>退市股不是小数</h2>
  <p>主板 3,489 只，其中 295 只退市；280 只有完整日线，缺失 15 只全部在 2004 年前退市。补全后，前二名单的配对重合度只有 0.862（A vs B）/0.805（A vs C），约 14%-20% 的前二配对不同——不只是多了几笔退市股交易，而是换了一个更完整的候选池。</p>
  <figure>{img(FIG/'01_权益对比_2017_2026.png','2017-2026 净值对比')}<figcaption>2017-2026 净值（对数轴）。灰 A 仅上市，红 B 含退市，蓝 C 含退市+排风险警示，金虚线为等权指数。</figcaption></figure>
  <figure>{img(FIG/'02_回撤对比_2017_2026.png','2017-2026 回撤对比')}<figcaption>2017-2026 回撤。含退市股后最大回撤从 -15.8% 扩大到 -54.3%；排风险警示后回到 -27.4%。</figcaption></figure>
  <div class="panel red"><b>2022 年为什么从 +50% 变 -36%？</b><br>
  A 池 2022 年 +50.6%，B 池 -35.8%，C 池 +2.3%。差异主要来自 002684 *ST猛狮：信号日已是风险警示，停牌后直接进入退市整理，15 日到期卖不出，-82.9%，一笔亏 21.85 万元。</div>
  <h3>口径分解：风险警示过滤 vs 完整池</h3>
  <div class="tablewrap">{dec_html}</div>
  <p class="small">在旧幸存者池里，排除风险警示反而把 2017-2026 从 43.2% 降到 38.4%，因为风险警示股里有 002592、000752、603007、002759 等赢家；只有在完整池里，排除风险警示才把 B 从 29.8% / -54.3% 拉回 35.0% / -27.4%。<b>它是尾部保险，不是 alpha。</b></p>
</section>

<section id="case">
  <h2>002684：一笔 -82.9% 的样本</h2>
  <figure>{img(FIG/'07_002684退市过程.png','002684 退市过程')}<figcaption>信号日 2022-04-21（风险警示=1，回撤 -61.4%，RSI 27.8）；4/26 开盘 2.69 买入；5/5 起停牌；6/6 退市整理首日开盘 0.46。</figcaption></figure>
  <div class="kv">
    <div class="k">信号日</div><div class="v">2022-04-21，风险警示，250 日回撤 -61.4%，RSI 27.8</div>
    <div class="k">买入</div><div class="v">2022-04-26 开盘 2.69</div>
    <div class="k">停牌</div><div class="v">2022-05-05 至 2022-06-02</div>
    <div class="k">卖出</div><div class="v">2022-06-06 开盘 0.46（15 日到期时仍在停牌）</div>
    <div class="k">单笔</div><div class="v"><b>-82.9% / -218,485 元</b>，约当时权益的 40%</div>
  </div>
  <p>旧数据没有这只股，所以旧回测 2022 年是 +50%；完整池 -35.8%；排风险警示 +2.3%。这不是参数问题，是样本缺失。退市整理期首日无涨跌幅，原逻辑把“开盘低于跌停价”误判为跌停并顺延一天，已修正为只在“正好等于跌停价”时顺延。</p>
</section>

<section id="oos">
  <h2>2016 年前：真正的样本外</h2>
  <div class="cards">
    <div class="card red"><div class="label">2005-2016 年化</div><div class="big">4.9%</div><div class="note">12 年只有 1.78x</div></div>
    <div class="card red"><div class="label">2005-2016 最大回撤</div><div class="big">-71.5%</div><div class="note">2008 年 -52.7%</div></div>
    <div class="card blue"><div class="label">2017-2026 年化</div><div class="big">35.0%</div><div class="note">震荡市，等权指数只有 4.1%</div></div>
    <div class="card amber"><div class="label">2005-2026 全周期</div><div class="big">16.8%</div><div class="note">等权指数 16.2%，打平</div></div>
  </div>
  <figure>{img(FIG/'04_权益对比_2005_2026.png','2005-2026 净值')}<figcaption>2005-2026 净值（对数轴）。策略在 2006-2007、2009、2014-2015 的趋势市明显跑输等权指数。</figcaption></figure>
  <figure>{img(FIG/'10_2008压力年.png','2008 压力年')}<figcaption>2008 年：策略 -52.7%，主板等权指数 -58.9%；策略最大回撤 -71.5%，指数 -72.1%。</figcaption></figure>
  <div class="tablewrap">{ann_html}</div>
  <p class="small">利润高度集中：2018 +99.8%、2024 +94.1%；2022 年 B -35.8%。2015 年 600485 连续一字跌停，-49.2%，说明止损失效的真实边界。策略是“震荡市急跌反弹”，不是全天候。</p>
  <figure>{img(FIG/'06_年度收益_2005_2026.png','2005-2026 年度收益')}<figcaption>2005-2026 年度收益：2008 是唯一的 -50% 级年份，2018/2024 是 +95% 级年份。</figcaption></figure>
</section>

<section id="indicators">
  <h2>指标检验：没有稳健的 alpha 过滤器</h2>
  <p>样本为所有满足冻结条件的信号（不只前二），T+3 开盘至 T+17 收盘的 15 日收益。信号日在同一天高度聚集，只做描述统计。</p>
  <figure>{img(FIG/'09_事件研究_换手与市场深度.png','换手与市场深度事件研究')}<figcaption>换手率越高越好；市场跌幅 -15%~-10% 是甜点区，但深水区在两个样本期方向不一致。</figcaption></figure>
  <div class="grid2">
    <div class="panel">
      <h3>逐项结论</h3>
      <ul>
        <li><b>风险警示</b>：2005-2016 平均 7.2% vs 非风险警示 2.8%；2017-2026 是 2.7% vs 6.1%。平均不差，差在 -83% 尾部。</li>
        <li><b>换手率</b>：7-10% 组 8.6%/8.5%，&gt;10% 组 6.4%/12.0%；&lt;2% 组只有 0.7%-5.3%。限制上限不成立。</li>
        <li><b>市净率（PB）/ 低价</b>：负市净率平均 10.3%/6.0%，但尾部肥；排风险警示后市净率&gt;0、股价≥2 对成交无影响。</li>
        <li><b>波动率 / 250 日涨幅</b>：高波动、高涨幅方向更好，但切阈值后 2017-2026 明显变差。</li>
        <li><b>成交额</b>：没有单调关系；≥2000 万全周期 20.4% / -63.9%，但近期 34.4% / -27.8%，略差。</li>
        <li><b>市场跌幅下限</b>：-18% 全周期 21.6% / -57.2%，-19% 立刻 18.8% / -61.7%，参数悬崖，不推荐。</li>
      </ul>
    </div>
    <div class="panel amber"><b>为什么都不采纳？</b> 这是一个只有 100-220 笔交易、收益高度集中的策略。任何过滤器都在改变“哪几笔大赢家/大输家被选中”，而不是稳定提高胜率。高换手、高波动、高涨幅只是方向性提示，不是可切的阈值。</div>
  </div>
</section>

<section id="risk">
  <h2>止损与仓位：唯一稳健的叠加</h2>
  <figure>{img(FIG/'08_止损与仓位前沿.png','止损与仓位前沿')}<figcaption>止损 15%/25% 在两个样本期都同时改善收益和回撤；仓位缩放按比例降低收益与回撤，卡玛基本不变。</figcaption></figure>
  <div class="tablewrap">{risk_html}</div>
  <div class="grid3">
    <div class="panel blue"><div class="small">2017-2026 · 2×35% + 止损 25%</div><div class="big" style="font-size:28px;font-weight:700">25.4%</div><div class="small">最大回撤 -18.1%</div></div>
    <div class="panel blue"><div class="small">2005-2026 · 2×35% + 止损 25%</div><div class="big" style="font-size:28px;font-weight:700">14.4%</div><div class="small">最大回撤 -47.6%</div></div>
    <div class="panel amber"><div class="small">长期预算</div><div class="big" style="font-size:28px;font-weight:700">10-15%</div><div class="small">压力回撤 -45%~-50%</div></div>
  </div>
  <p class="small">止损不是 alpha，是把单笔左尾截断；连续跌停时只能顺延到下一个可成交开盘，成交价可能远低于止损价（600485 2015 就是例子）。</p>
</section>

<section id="current">
  <h2>当前信号：不重排</h2>
  <div class="panel blue">
    <p><b>2026-10-23 冻结名单不变：603778、605376，计划 2026-10-28 开盘。</b>两者都不是风险警示、市净率（PB）为正、股价≥2，新过滤不影响它们。002969 仍是第 3，不递补。</p>
    <p><b>2026-10-26 是新信号日，不是补丁：</b>满足条件 26 只，603778 仍是第 1（收盘 7.29，收盘位置 0.4545，回撤 -78.73%），第 2 为 002240；605376 已不在前二。按执行冻结规则，不改变 10/28 的名单；若 10/28 已成交 603778，10/29 不再加。</p>
    <p class="small">把市场闸门收窄到“20 日跌幅 &lt; -10%”会跳过 10/23（当天 -5.21%）；该变体不推荐，所以不据此取消 10/28 的计划。</p>
  </div>
</section>

<section id="limits">
  <h2>限制</h2>
  <ul>
    <li>2024-2026 已在旧研究中使用，不是纯净留出；2005-2016 是冻结规则的真正样本外，但市场结构不同。</li>
    <li>盘中止盈/止损按触价成交，未建模排队、盘口深度、滑点扩大；大资金容量有限。</li>
    <li>15 只退市股缺失均在 2004 年前；2007-2009 重组复牌的超大单日收益保留未改。</li>
    <li>风险警示标记逐日可用，但退市整理期可能变为 0；等权指数不可投资。</li>
    <li>收益高度集中：2018、2024 两年贡献大部分利润；不要用 2017-2026 的 35% 做长期预算。</li>
  </ul>
</section>

<footer>研究日期 2026-10-27 · 数据截至 2026-10-26 · 信号截至 2026-10-23 · 历史回测不代表未来，不构成投资建议。<br>
完整报告：01_完整报告_退市股补充与指标优化.md · 数据表：baostock补充研究_数据表.xlsx · 校验：05_证据/校验报告.md</footer>
</div></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print("saved", OUT, f"{OUT.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
