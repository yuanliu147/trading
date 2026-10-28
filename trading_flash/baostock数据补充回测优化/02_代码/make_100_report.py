#!/usr/bin/env python3
"""生成“100+ 信号日：榜一榜二”图表、HTML 与摘要。"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "08_百信号日"
FIG = OUT / "图表"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.22
plt.rcParams["grid.linestyle"] = "--"

GREEN = "#2c7a4b"
BLUE = "#1f6fb2"
ORANGE = "#d39b1f"
RED = "#b83b2f"
GREY = "#8a8f98"


def pct(x: float, d: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{d}f}%"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}">'


def html_table(headers: list[str], rows: list[list[str]], cls: str = "data") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#18222d;--muted:#63707e;--blue:#1f6fb2;--blue2:#e8f1fa;
--red:#b83b2f;--red2:#fbecea;--amber:#b8860b;--amber2:#fdf6e3;--green:#2c7a4b;--line:#e3e7ec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
font-size:15px;line-height:1.75}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 64px}
header{padding:52px 0 28px;border-bottom:1px solid var(--line)}
.kicker{color:var(--blue);font-weight:600;letter-spacing:.08em;font-size:12px;text-transform:uppercase}
h1{font-size:37px;line-height:1.22;margin:10px 0 12px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:15px;max-width:850px}
.banner{margin:24px 0 0;background:var(--amber2);border-left:4px solid var(--amber);padding:15px 19px;border-radius:0 10px 10px 0;color:#5a4310;font-weight:600}
section{margin:36px 0}h2{font-size:25px;margin:0 0 6px}h3{font-size:17px;margin:22px 0 8px}
p{margin:8px 0 12px}.small{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 17px 13px}
.card .label{color:var(--muted);font-size:12px}.card .big{font-size:26px;font-weight:700;margin:3px 0 2px;letter-spacing:-.02em}
.card .note{color:var(--muted);font-size:12px}.card.blue .big{color:var(--blue)}.card.red .big{color:var(--red)}.card.green .big{color:var(--green)}.card.amber .big{color:var(--amber)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.panel.amber{background:var(--amber2);border-color:#efdfb8}.panel.blue{background:var(--blue2);border-color:#cfe2f3}
img{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
figure{margin:14px 0 8px}figcaption{color:var(--muted);font-size:12px;margin-top:6px}
table.data{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
table.data th{background:#eef2f6;text-align:right;padding:9px 8px;font-weight:600;color:#33414f}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data td{padding:8px 8px;border-top:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
table.data tr:hover td{background:#fafbfd}.tablewrap{overflow:auto;border-radius:10px}
ul{margin:8px 0 12px 20px;padding:0}li{margin:5px 0}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px;margin-top:42px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}.wrap{padding:0 16px 48px}}
"""


def main():
    detail = pd.read_csv(OUT / "百信号日_明细.csv", parse_dates=["信号日"])
    detail["当日持有"] = detail["当日持有"].fillna("")
    summary = pd.read_csv(OUT / "百信号日_汇总.csv")

    full = summary[summary["期间"] == "2005-2026"].iloc[0]
    old = summary[summary["期间"] == "2005-2016"].iloc[0]
    recent = summary[summary["期间"] == "2017-2026"].iloc[0]

    fd = detail[detail["期间"] == "2005-2026"].copy()
    rd = detail[detail["期间"] == "2017-2026"].copy()
    actual_buys = fd[fd["实际买入状态"] != "没买"].copy()
    both_buy = fd[fd["实际买入状态"] == "两笔都买"].copy()

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    cats = ["双盈", "仅榜一", "仅榜二", "双亏"]
    vals = [int(full["信号层双盈"]), int(full["仅榜一盈"]), int(full["仅榜二盈"]), int(full["双亏"])]
    axes[0].bar(cats, vals, color=[GREEN, BLUE, ORANGE, RED], width=0.62)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.4, str(v), ha="center", va="bottom", fontsize=10)
    axes[0].set_title(f"信号层：如果都买（n={sum(vals)}）", fontsize=12, loc="left")
    axes[0].set_ylim(0, max(vals) + 3)
    axes[0].set_ylabel("天数")

    status_counts = fd["实际买入状态"].value_counts()
    cats2 = ["没买", "只买榜一", "两笔都买"]
    vals2 = [int(status_counts.get(c, 0)) for c in cats2]
    axes[1].bar(cats2, vals2, color=[GREY, BLUE, RED], width=0.55)
    for i, v in enumerate(vals2):
        axes[1].text(i, v + 0.6, str(v), ha="center", va="bottom", fontsize=10)
    axes[1].set_title(f"组合层：实际买到（n={len(fd)}）", fontsize=12, loc="left")
    axes[1].set_ylim(0, max(vals2) + 4)
    axes[1].set_ylabel("天数")
    fig.tight_layout()
    fig.savefig(FIG / "01_百信号日_结果.png", bbox_inches="tight")
    plt.close(fig)

    # 汇总表
    sh = ["期间", "100+ 天数", "可观察", "双盈", "仅榜一", "仅榜二", "双亏", "榜一均值", "榜二均值", "实际有买入"]
    srows = []
    for r in [full, old, recent]:
        srows.append([r["期间"], str(int(r["百信号日"])), str(int(r["信号层可观察"])),
                      str(int(r["信号层双盈"])), str(int(r["仅榜一盈"])), str(int(r["仅榜二盈"])),
                      str(int(r["双亏"])), pct(r["榜一均值"]), pct(r["榜二均值"]),
                      str(int(r["实际有买入"]))])
    sum_html = html_table(sh, srows)
    sum_md = md_table(sh, srows)

    # 胜率口径：单笔 vs 双盈天数
    wh = ["期间", "可观察天数", "单笔胜率", "榜一胜率", "榜二胜率", "双盈天数", "双亏天数"]
    wrows = []
    for period in ["2005-2026", "2005-2016", "2017-2026"]:
        g = detail[detail["期间"] == period].dropna(subset=["榜一毛收益", "榜二毛收益"])
        n = len(g)
        w1 = int((g["榜一毛收益"] > 0).sum())
        w2 = int((g["榜二毛收益"] > 0).sum())
        wins = w1 + w2
        total = 2 * n
        both = int(((g["榜一毛收益"] > 0) & (g["榜二毛收益"] > 0)).sum())
        bl = int(((g["榜一毛收益"] <= 0) & (g["榜二毛收益"] <= 0)).sum())
        wrows.append([period, str(n),
                      f"{wins}/{total} ({wins / total * 100:.1f}%)",
                      f"{w1}/{n} ({w1 / n * 100:.1f}%)",
                      f"{w2}/{n} ({w2 / n * 100:.1f}%)",
                      f"{both}/{n} ({both / n * 100:.1f}%)",
                      f"{bl}/{n} ({bl / n * 100:.1f}%)"])
    win_html = html_table(wh, wrows)
    win_md = md_table(wh, wrows)

    # 亏损对比
    loss = pd.read_csv(OUT / "百信号日_亏损对比.csv")

    def lrow(kj, period, rank):
        return loss[(loss["口径"] == kj) & (loss["期间"] == period)
                    & (loss["排名"] == rank)].iloc[0]

    h100 = ["排名", "亏损笔数", "亏损均值", "亏损中位", "最差", "亏损合计", "≤-10%", "≤-20%"]
    r100 = []
    for rank, name in [(1, "榜一"), (2, "榜二")]:
        r = lrow("100+信号日", "2005-2026", rank)
        r100.append([name, f"{int(r['亏损笔数'])}/{int(r['笔数'])} ({r['亏损率'] * 100:.1f}%)",
                     pct(r["亏损均值"]), pct(r["亏损中位"]), pct(r["最差"]),
                     f"{r['亏损合计'] * 100:.1f}pp", str(int(r["跌10"])), str(int(r["跌20"]))])
    loss100_html = html_table(h100, r100)
    loss100_md = md_table(h100, r100)

    hfull = ["期间", "排名", "笔数", "亏损笔数", "亏损率", "亏损均值", "亏损中位", "最差", "≤-20%"]
    rfull = []
    for period in ["2005-2026", "2017-2026"]:
        for rank, name in [(1, "榜一"), (2, "榜二")]:
            r = lrow("全样本前二", period, rank)
            rfull.append([period, name, str(int(r["笔数"])), str(int(r["亏损笔数"])),
                          pct(r["亏损率"]), pct(r["亏损均值"]), pct(r["亏损中位"]),
                          pct(r["最差"]), str(int(r["跌20"]))])
    lossfull_html = html_table(hfull, rfull)
    lossfull_md = md_table(hfull, rfull)

    # 最近 4 个
    dh = ["信号日", "信号数", "榜一", "榜一毛收益", "榜二", "榜二毛收益", "组合实际"]
    drows = []
    for r in rd.sort_values("信号日").itertuples():
        if pd.isna(r.榜一毛收益):
            r1 = "未到买入日"
            r2 = "未到买入日"
            act = "没买（两仓满；603778 已在手）"
        else:
            r1 = pct(r.榜一毛收益)
            r2 = pct(r.榜二毛收益)
            act = "没买（两仓满）"
        drows.append([r.信号日.date().isoformat(), str(int(r.信号数)),
                      r.榜一.split(".")[-1], r1, r.榜二.split(".")[-1], r2, act])
    recent_html = html_table(dh, drows)
    recent_md = md_table(dh, drows)

    # 实际买入
    ah = ["信号日", "信号数", "当日持有", "实际买入", "榜一实际", "榜二实际"]
    arows = []
    for r in actual_buys.sort_values("信号日").itertuples():
        arows.append([r.信号日.date().isoformat(), str(int(r.信号数)),
                      r.当日持有 if r.当日持有 else "空仓",
                      r.实际买入状态,
                      f"{r.榜一.split('.')[-1]} {pct(r.实际榜一收益)}" if not pd.isna(r.实际榜一收益) else "—",
                      f"{r.榜二.split('.')[-1]} {pct(r.实际榜二收益)}" if not pd.isna(r.实际榜二收益) else "—"])
    actual_html = html_table(ah, arows)
    actual_md = md_table(ah, arows)

    # 个案 2016-01-05
    case = both_buy.iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>100+ 信号日：榜一榜二</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<div class="kicker">信号宽度 · 100+ 候选</div>
<h1>出现 100+ 信号时，榜一榜二都盈利了吗？</h1>
<p class="sub">C 池前二（含退市股、信号日排除风险警示），2×50%；T 日收盘出信号，T+3 开盘买入，+20% 盘中止盈，
15 个交易日收盘卖出。信号层收益按毛收益（未计成本）模拟，组合层是真实 2×50% 成交。</p>
<div class="banner">信号层：31 个可观察的 100+ 信号日里，18 天榜一榜二双盈、9 天只有一边盈、4 天双亏。
组合层：32 天里有 30 天两仓已满，实际只有 3 天买到；唯一一次两笔都买的那天，两笔都亏。</div>
</header>

<div class="cards">
<div class="card blue"><div class="label">2005-2026 的 100+ 信号日</div><div class="big">32 天</div><div class="note">2017-2026 只有 4 天</div></div>
<div class="card green"><div class="label">信号层双盈</div><div class="big">18/31</div><div class="note">58.1%；单笔胜率 72.6%</div></div>
<div class="card amber"><div class="label">两仓已满</div><div class="big">30/32</div><div class="note">93.8%，信号多但没空位</div></div>
<div class="card red"><div class="label">唯一一次两笔都买</div><div class="big">双亏</div><div class="note">2016-01-05：-15.0% / -14.0%</div></div>
</div>

<section>
<h2>一、信号层：大多数双盈，但不是“都”</h2>
<p>把每个 100+ 信号日的榜一、榜二各买一份，按 +20% 止盈 / 15 个交易日到期模拟：
全历史 31 个可观察日里，<b>18 天双盈（58.1%）</b>、4 天仅榜一盈、5 天仅榜二盈、4 天双亏。
榜一平均 {pct(full['榜一均值'])}，榜二平均 {pct(full['榜二均值'])}，榜一比榜二高约 4 个百分点。</p>
<p>2017-2026 的 4 个 100+ 信号日里，前 3 个都可观察，而且<b>三次都是双盈</b>：2022-04-27、2026-07-09、
2026-07-21，榜一全部 +20%，榜二分别 +7.9%、+11.9%、+20.0%。这是 3 天、6 笔全胜，所以那一行是 100%；
但样本只有 3 天，不是全历史 100%。</p>
<div class="tablewrap">{sum_html}</div>
<p class="small">“可观察”排除了 2026-10-23（买入日 10/28 在样本外）。“榜一/榜二均值”是毛收益，未计成本；
2017-2026 平均只基于 3 天，不能外推。</p>
<h3>胜率口径：100% 只出现在 3 天样本里</h3>
<p>把榜一、榜二各算一笔，全历史 31 个可观察日共 62 笔，只有 <b>45 笔盈利（72.6%）</b>；
按交易日算，<b>18/31 天双盈（58.1%）</b>，还有 4 天双亏（12.9%）。2017-2026 的 6/6 是 3 天样本，
不能外推到全历史。</p>
<div class="tablewrap">{win_html}</div>
<p class="small">“单笔胜率”= 盈利的榜一/榜二笔数 ÷ 2×可观察天数；“双盈天数”= 榜一榜二当天都盈利的交易日占比。</p>

<h3>谁亏得多？100+ 日看榜二，全样本看榜一</h3>
<p><b>100+ 信号日：榜二亏得更多。</b>榜二亏损天数略少（8 天对 9 天），但一旦亏，平均亏
{pct(lrow('100+信号日', '2005-2026', 2)['亏损均值'])}，榜一只有
{pct(lrow('100+信号日', '2005-2026', 1)['亏损均值'])}；榜二最差
{pct(lrow('100+信号日', '2005-2026', 2)['最差'])}，榜一最差
{pct(lrow('100+信号日', '2005-2026', 1)['最差'])}。榜二有 5 笔亏损 ≥10%，榜一只有 1 笔。
2008/2016 的危机日里，榜二接连吃到 -25.9%、-13.8%、-11.5%、-11.3%、-10.9%。</p>
<div class="tablewrap">{loss100_html}</div>
<p><b>全样本：反过来。</b>榜二亏损更频繁（39.7% 对 29.8%），但榜一单笔亏得更深
（-12.0% 对 -10.6%），尾部更重（最差 -49.2% 对 -29.5%；亏损 ≥20% 的有 8 笔对 4 笔）。
2017-2026 也是榜一亏得更深（-9.5% 对 -7.3%，最差 -27.9% 对 -15.5%）。</p>
<div class="tablewrap">{lossfull_html}</div>
<p class="small">100+ 信号日是毛收益；全样本是实际净收益。亏损均值/中位是“只算亏损笔”的条件均值，
不是全部交易的平均。</p>
</section>

<section>
<h2>二、最近 4 个 100+ 信号日</h2>
<div class="tablewrap">{recent_html}</div>
<p>最近三次榜一榜二都盈利，但真实组合<b>一次都没买到</b>：这三天的当日持仓数都是 2，两个仓位早被前面的信号占满。
2026-10-23 也一样：122 个信号，榜一 603778 已经在手，榜二 605376 没有空位。</p>
<figure>{img(FIG / '01_百信号日_结果.png', '100+ 信号日信号层与组合层结果')}
<figcaption>左边是“如果都买”的信号层，右边是真实 2×50% 组合实际买到的天数。信号多不等于买得到。</figcaption></figure>
</section>

<section>
<h2>三、组合层：100+ 信号日往往是“仓位已满日”</h2>
<p>32 个 100+ 信号日里，<b>30 天当日已经持有 2 个仓位</b>，只有 2 天是空仓；实际只有 3 天产生过买入：
2015-07-10 和 2015-07-16 只买到榜一，2016-01-05 是唯一一次两笔都买到。</p>
<div class="tablewrap">{actual_html}</div>
<p>2016-01-05 那次，000584 和 601766 各亏约 15% 和 14%，两笔各半仓，组合约 -14.5%。
2015-07-10 和 07-16 只买到榜一（002072、600571），都是 +20% 止盈；榜二因为另一仓还在手，没有买。
2008-03-19 更典型：当天空仓，但 3/17 的信号已经在 3/20 把两个仓位填满，3/19 的榜一榜二到 T+3 时已经没位置。</p>
<div class="panel amber"><b>为什么？</b> 100+ 信号是市场恐慌洗盘的“同步结果”，不是新的入场触发器。
策略按信号日先来后到，前面的信号先占住两个半仓；等到 100+ 那天，通常已经满仓。信号数量越多，
越可能说明“你已经在场”，而不是“又多了两个好机会”。</div>
<p>实际买入率也能看出来：100+ 信号日只有 3/32（9.4%）产生买入，而所有信号日是 151/693（21.8%）。
信号越宽，越买不到。</p>
</section>

<section>
<h2>四、对 2026-10-23 的含义</h2>
<p>10/23 的 C 池信号数是 <b>122</b>（A 池 123），榜一 603778、榜二 605376。但回测组合在 10/23 持有
<b>600783（10/19 买入）和 603778（10/20 买入）</b>，两个仓位都满，npos=2。
按 15 个交易日到期，600783 大约 11/6 才能卖，603778 大约 11/9；10/28 没有空位。</p>
<p>所以按冻结的 2×50% 规则：<b>10/28 不会新买 603778（已在手），也买不进 605376</b>。
10/26 的新信号日（603778 第 1、002240 第 2）到 10/29 同样没有空位。
除非你手动腾出一个仓位，或者实盘并没有这两个持仓，否则“122 个信号”不会变成新的榜二买入。</p>
<p class="small">这是回测口径下的结论；如果你实盘的持仓/现金与回测不同，以实盘为准。</p>
</section>

<section>
<h2>五、限制</h2>
<ul>
<li>信号层是毛收益模拟，未计成本、滑点、涨停/停牌拒买；组合层才是真实成交。</li>
<li>2017-2026 只有 4 个 100+ 信号日（3 个可观察），样本太小，三次双盈不能外推。</li>
<li>“100+”阈值是人为选的；50-99 个信号的日子没有单独统计，结论只适用于 100+。</li>
<li>组合层用了 2×50% 前二规则；如果你用 2×35%、加止损或只买第一，仓位占用会不同。</li>
<li>历史回测不代表未来，不构成投资建议。</li>
</ul>
</section>

<footer>
口径：C 池 = 全主板含退市股、信号日 is_st=1 不买；成交价不复权，信号/收益后复权；
成本含 0.1%/边滑点、佣金、过户费、卖出印花税；信号层图表为毛收益。
复现：<code>02_代码/analyze_100_signals.py</code> 生成统计与明细，<code>02_代码/make_100_report.py</code> 生成本页。
</footer>
</div></body></html>"""
    (OUT / "百信号日.html").write_text(html, encoding="utf-8")

    md = f"""# 100+ 信号日：榜一榜二都盈利了吗？

C 池前二（含退市股、信号日排除风险警示），2×50%；T 日收盘信号，T+3 开盘买入，+20% 盘中止盈，
15 个交易日收盘卖出。信号层为毛收益模拟，组合层为真实 2×50% 成交。

## 短答

不是“都盈利”。

- 信号层：31 个可观察的 100+ 信号日，18 天双盈（58.1%）、4 天仅榜一盈、5 天仅榜二盈、4 天双亏；
  单笔胜率 45/62（72.6%），榜一平均 {pct(full['榜一均值'])}，榜二平均 {pct(full['榜二均值'])}。
- 组合层：32 天里 30 天两仓已满，实际只有 3 天产生买入；唯一一次两笔都买的是 2016-01-05，
  000584 / 601766 各亏约 15% / 14%。
- 2017-2026 的 4 个 100+ 信号日，前 3 个可观察、6 笔全胜（3 天样本）；但组合一次都没买到。

## 汇总

{sum_md}

## 胜率口径

{win_md}

把榜一、榜二各算一笔，全历史 31 个可观察日共 62 笔，只有 45 笔盈利（72.6%）；按交易日算，
18/31 天双盈（58.1%），4 天双亏（12.9%）。2017-2026 的 6/6 是 3 天样本，不是全历史 100%。

## 谁亏得多

100+ 信号日：**榜二亏得更多**。榜二亏损天数略少（8 对 9），但一旦亏，平均亏 -11.3%，榜一只有 -3.6%；
榜二最差 -25.9%，榜一 -14.8%；榜二有 5 笔亏损 ≥10%，榜一只有 1 笔。2008/2016 的危机日里，
榜二接连吃到 -25.9%、-13.8%、-11.5%、-11.3%、-10.9%。

{loss100_md}

全样本：**反过来**。榜二亏损更频繁（39.7% 对 29.8%），但榜一单笔亏得更深（-12.0% 对 -10.6%），
尾部更重（最差 -49.2% 对 -29.5%；亏损 ≥20% 的有 8 笔对 4 笔）。2017-2026 也是榜一亏得更深
（-9.5% 对 -7.3%，最差 -27.9% 对 -15.5%）。

{lossfull_md}

100+ 信号日是毛收益；全样本是实际净收益。亏损均值/中位是“只算亏损笔”的条件均值。

## 最近 4 个

{recent_md}

## 实际买到

{actual_md}

2015-07-10 / 07-16 只买到榜一（002072、600571），都是 +20% 止盈；榜二因为另一仓还在手，没有买。
2016-01-05 是唯一一次两笔都买，000584 和 601766 各亏约 15% / 14%。
2008-03-19 空仓，但 3/17 的信号已经在 3/20 填满两个仓位，3/19 的榜一榜二到 T+3 时没位置。

## 为什么

100+ 信号是市场恐慌洗盘的同步结果，不是新的入场触发器。策略按信号日先来后到，前面的信号先占住两个半仓；
等到 100+ 那天，通常已经满仓。100+ 信号日实际买入率只有 3/32（9.4%），所有信号日是 151/693（21.8%）。

## 当前 10/23

C 池 122 个信号（A 池 123），榜一 603778、榜二 605376。但组合在 10/23 持有 600783（10/19 买入）
和 603778（10/20 买入），npos=2；15 个交易日到期分别约 11/6 和 11/9，10/28 没有空位。
按冻结的 2×50% 规则，10/28 不会新买 603778（已在手），也买不进 605376；10/26 的新信号到 10/29
同样没有空位。除非手动腾仓，或实盘持仓与回测不同。

## 限制

信号层是毛收益，未计成本/滑点/涨停停牌；2017-2026 只有 3 个可观察样本；100+ 是人为阈值；
组合层是 2×50% 前二规则，用 2×35%、止损或只买第一会不同；历史回测不代表未来，不构成投资建议。

图表与交互页：`百信号日.html`；明细：`百信号日_明细.csv`、`百信号日_汇总.csv`；
复现：`02_代码/analyze_100_signals.py`、`02_代码/make_100_report.py`。
"""
    (OUT / "百信号日_结论.md").write_text(md, encoding="utf-8")

    print(sum_md)
    print(recent_md)
    print(actual_md)


if __name__ == "__main__":
    main()
