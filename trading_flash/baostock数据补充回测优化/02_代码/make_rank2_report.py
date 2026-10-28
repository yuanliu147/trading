#!/usr/bin/env python3
"""生成“榜二大跌个案”图表、HTML 与摘要。"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_只买第一名"
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

BLUE = "#1f6fb2"
RED = "#c0392b"
GREY = "#63707e"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}">'


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def cnt(row: pd.Series, col: str) -> str:
    n = int(row["笔数"])
    k = int(row[col])
    return f"{k}/{n} ({k / n * 100:.1f}%)"


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
.card .note{color:var(--muted);font-size:12px}.card.red .big{color:var(--red)}.card.blue .big{color:var(--blue)}.card.amber .big{color:var(--amber)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.panel.amber{background:var(--amber2);border-color:#efdfb8}
img{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
figure{margin:14px 0 8px}figcaption{color:var(--muted);font-size:12px;margin-top:6px}
table.data{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
table.data th{background:#eef2f6;text-align:right;padding:9px 9px;font-weight:600;color:#33414f}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data td{padding:8px 9px;border-top:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
table.data tr:hover td{background:#fafbfd}.tablewrap{overflow:auto;border-radius:10px}
ul{margin:8px 0 12px 20px;padding:0}li{margin:5px 0}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px;margin-top:42px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}.wrap{padding:0 16px 48px}}
"""


def main():
    stat = pd.read_csv(OUT / "榜二尾部统计.csv")
    cases = pd.read_csv(OUT / "榜二大跌个案.csv")

    sel = cases.copy()

    # 图：收盘收益 vs 最大浮亏
    d = sel.sort_values("mae").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    y = list(range(len(d)))
    ax.barh([i - 0.19 for i in y], d.ret * 100, height=0.36, color=BLUE, label="收盘收益")
    ax.barh([i + 0.19 for i in y], d.mae * 100, height=0.36, color=RED, alpha=0.78, label="持有期最大浮亏")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.code.split('.')[-1]} {r.code_name}" for r in d.itertuples()])
    ax.invert_yaxis()
    ax.axvline(0, color="#333", lw=0.8)
    for i, r in d.iterrows():
        for val, yoff, color in [(r.ret, -0.19, BLUE), (r.mae, 0.19, RED)]:
            x = val * 100
            if x >= 0:
                ax.text(x + 0.6, i + yoff, f"{x:.1f}%", va="center", ha="left",
                        fontsize=7.5, color=color)
            else:
                ax.text(x - 0.6, i + yoff, f"{x:.1f}%", va="center", ha="right",
                        fontsize=7.5, color=color)
    ax.set_xlim(-43, 8)
    ax.set_xlabel("相对买入价（%）")
    ax.set_title("榜二买入后：收盘收益 vs 持有期最大浮亏", fontsize=13, loc="left", pad=10)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "04_榜二大跌个案.png", bbox_inches="tight")
    plt.close(fig)

    # 尾部统计表
    stat_rows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        for rk, lab in [(1, "榜一"), (2, "榜二")]:
            r = stat[(stat["期间"] == period) & (stat["排名"] == rk)].iloc[0]
            stat_rows.append([
                period, lab, str(int(r["笔数"])), cnt(r, "收盘亏损20"), cnt(r, "浮亏20"),
                cnt(r, "单日跌10"), pct(r["收盘最差"]), pct(r["浮亏最深"]),
                pct(r["单日最差"]), pct(r["平均单笔"]),
            ])
    stat_html = html_table(
        ["期间", "排名", "笔数", "收盘≤-20%", "浮亏≤-20%", "单日≤-10%",
         "收盘最差", "浮亏最深", "单日最差", "平均单笔"], stat_rows)
    stat_md = md_table(
        ["期间", "排名", "笔数", "收盘≤-20%", "浮亏≤-20%", "单日≤-10%",
         "收盘最差", "浮亏最深", "单日最差", "平均单笔"], stat_rows)

    # 个案表
    exit_map = {
        "max_holding": "15日到期",
        "pending_max_holding_limit_down": "到期遇跌停，顺延",
        "pending_max_holding_no_bar": "到期停牌，顺延",
        "take_profit_intraday": "盘中止盈",
        "take_profit_open": "跳空止盈",
        "eod": "期末持仓",
    }
    case_rows = []
    for r in sel.itertuples():
        case_rows.append([
            r.period, f"{r.code.split('.')[-1]} {r.code_name}",
            f"{r.signal_date[:10]} → {r.entry_date[:10]} → {r.exit_date[:10]}",
            pct(r.ret), pct(r.mae), pct(r.worst_day),
            f"{r.pnl / 1e4:.1f}", exit_map.get(r.exit_reason, r.exit_reason),
        ])
    case_html = html_table(
        ["期间", "股票", "信号 → 买入 → 卖出", "收盘", "最大浮亏", "单日最差", "净盈亏(万)", "退出"],
        case_rows)
    case_md = md_table(
        ["期间", "股票", "信号 → 买入 → 卖出", "收盘", "最大浮亏", "单日最差", "净盈亏(万)", "退出"],
        case_rows)

    # 榜一极端个案
    lead_rows = [
        ["603260 合盛硅业", "2017-2026", "榜一", "-27.9%", "-28.2%", "-6.9", "同一前二组合里，榜一的最大跌幅比榜二更深"],
        ["000572 海马汽车", "2017-2026", "榜一", "-16.1%", "-18.5%", "-12.1", "2017-2026 单笔最大金额亏损"],
        ["600485 *ST信威", "2005-2016", "榜一", "-49.2%", "-50.3%", "-4.6", "信号日不是风险警示，后来退市；C 池仍会买到"],
        ["002684 猛狮退", "2017-2026", "榜一（B 池）", "-82.9%", "-84.4%", "-21.9", "风险警示股；C 池会被过滤，但它是榜一，不是榜二"],
    ]
    lead_html = html_table(["股票", "期间", "排名", "收盘", "最大浮亏", "净盈亏(万)", "说明"], lead_rows)
    lead_md = md_table(["股票", "期间", "排名", "收盘", "最大浮亏", "净盈亏(万)", "说明"], lead_rows)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>榜二大跌个案</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<div class="kicker">前二组合 · 榜二尾部</div>
<h1>历史上买到的榜二，有没有大跌？</h1>
<p class="sub">C 池：全主板含退市股、信号日排除风险警示；前二组合 2×50%，T+3 开盘买入，15 个交易日或 +20% 止盈。
“最大浮亏”按买入后每日最低价 / 买入开盘价；“单日最差”买入日按收盘/开盘，之后按收盘/前收。</p>
<div class="banner">有。榜二在 2007-2008、2015 都出现过大跌，最深盘中浮亏 -36.4%；但榜一的尾部更极端，
不能靠“只买第一”来规避尾部风险。</div>
</header>

<div class="cards">
<div class="card amber"><div class="label">2005-2016 榜二浮亏 ≤ -20%</div><div class="big">12/41</div><div class="note">29.3%，高于同期榜一 20.3%</div></div>
<div class="card red"><div class="label">榜二最深浮亏</div><div class="big">-36.4%</div><div class="note">600862 中航高科，2008-09</div></div>
<div class="card blue"><div class="label">2017-2026 榜二最差收盘</div><div class="big">-15.4%</div><div class="note">000822 山东海化，2021-10</div></div>
<div class="card red"><div class="label">2017-2026 榜二最大金额亏损</div><div class="big">-10.6万</div><div class="note">000892 欢瑞世纪，2026-06</div></div>
</div>

<section>
<h2>一、三个口径</h2>
<p>只看收盘会漏掉“盘中砸坑又拉回”的票，所以拆成三个口径：收盘实现收益、持有期最大浮亏、单日最大跌幅。
下表是真正成交的前二组合，按信号日排名分榜一/榜二。</p>
<div class="tablewrap">{stat_html}</div>
<p class="small">“收盘≤-20%”是最终亏损超过 20% 的笔数；“浮亏≤-20%”是持有期内最低价相对买入价跌超过 20% 的笔数；
“单日≤-10%”是买入后至少有一天跌超过 10%。比例的分母是该组成交笔数。</p>
</section>

<section>
<h2>二、榜二的大跌个案</h2>
<p>2007-2008 是重灾区：2008-08-05 的 600655 豫园股份，15 个交易日后收盘 -29.5%，盘中最大浮亏 -33.1%；
2008-09-17 的 600862 中航高科盘中最大浮亏 -36.4%，最终 -22.8%。
2007-06-05 的 600617 国新能源收盘 -27.0%，而且 15 个交易日到期时撞上跌停，卖不出去，顺延后才成交。</p>
<p>2015 年的 000948 南天信息收盘只亏 -12.0%，但盘中一度 -34.1%，是“收盘看不出来”的典型。
最近九年没有 -20% 级别的榜二：最差收盘是 000822 山东海化 -15.4%；金额最大的是 2026 年 000892 欢瑞世纪，
-14.1%，半仓亏 10.6 万元。600633 浙数文化盘中 -15.2%，最后却 +6.7% 到期离场，说明盘中大跌不等于最终亏损。</p>
<figure>{img(FIG / '04_榜二大跌个案.png', '榜二大跌个案：收盘收益与最大浮亏')}
<figcaption>蓝条是收盘收益，红条是持有期最大浮亏。红条明显长于蓝条，说明只看收盘会低估持有过程中的压力。</figcaption></figure>
<div class="tablewrap">{case_html}</div>
<p class="small">净盈亏是该笔绝对金额，受当时权益基数影响；2005-2016 的几笔绝对金额较小，不代表百分比风险更小。</p>
</section>

<section>
<h2>三、但榜一并没有更安全</h2>
<p>在同一个 2017-2026 前二组合里，榜一的最大跌幅是 603260 合盛硅业 -27.9%（盘中 -28.2%），
比榜二最差的 -15.4% 更深；单笔最大金额亏损是榜一 000572 海马汽车 -12.1 万元，也高于榜二 000892 的 -10.6 万元。
全历史里，榜一 600485 *ST信威 -49.2%，信号日并不是风险警示，后来退市，C 池照样买到。</p>
<div class="tablewrap">{lead_html}</div>
<p>002684 猛狮退 -82.9% 是榜一，不是榜二；它在 C 池会被风险警示过滤掉，但它说明极端尾部在榜一。
所以“只买第一”不是尾部风险开关，它只是把暴露换了一组。</p>
</section>

<section>
<h2>四、这意味着什么</h2>
<ul>
<li><b>榜二的尾部是真实的，而且集中在系统性下跌。</b>2005-2016 有 12/41 笔榜二盘中浮亏 ≤ -20%，
比例甚至高于榜一；2008 年两个榜位一起跌，分散化在危机里几乎失效。</li>
<li><b>但榜一的极端值更深。</b>榜二最差收盘 -29.5%，榜一 -49.2%；只买第一不能替代止损和仓位控制。</li>
<li><b>最近九年榜二没有 -20% 级别的大跌，不等于以后没有。</b>2026 年 000892 仍亏 10.6 万元；
样本只有 33 笔，不能据此认为榜二安全。</li>
<li><b>止损是保险，不是免费午餐。</b>25% 止损通常能把 -30% 以上的尾部截断在 -25% 附近（跳空/跌停除外），
但 600862 这种盘中 -36.4%、收盘 -22.8% 的票会被止损在 -25%，反而比持有更差；是否叠加要接受这个 trade-off。</li>
<li><b>仓位上，2×50% 的单票 -30% 等于账户 -15%。</b>如果承受不了，应该降单票仓位，而不是只买第一名。</li>
</ul>
</section>

<section>
<h2>五、限制</h2>
<ul>
<li>“最大浮亏”按后复权日线最低价计算，未建模盘中止损的滑点、封单和排队；实际成交可能更差。</li>
<li>单日跌幅按后复权收盘计算，买入日按收盘/开盘；停牌日不计入。</li>
<li>样本小：榜二 2017-2026 只有 33 笔，2005-2016 只有 41 笔；极端个案不能当成稳定频率。</li>
<li>历史回测不代表未来，不构成投资建议。</li>
</ul>
</section>

<footer>
口径：C 池 = 全主板含退市股、信号日 is_st=1 不买；成交价不复权，信号/收益后复权；成本含 0.1%/边滑点、佣金、过户费、卖出印花税。
复现：<code>02_代码/analyze_rank2_tail.py</code> 生成统计与个案，<code>02_代码/make_rank2_report.py</code> 生成本页；
完整候选见 <code>榜二大跌明细.csv</code>。
</footer>
</div></body></html>"""
    (OUT / "榜二大跌个案.html").write_text(html, encoding="utf-8")

    md = f"""# 榜二大跌个案

C 池：全主板含退市股、信号日排除风险警示；前二组合 2×50%，T+3 开盘买入，15 个交易日或 +20% 止盈。
“最大浮亏”按买入后每日最低价 / 买入开盘价；“单日最差”买入日按收盘/开盘，之后按收盘/前收。

## 短答

有。榜二在 2007-2008、2015 都出现过大跌，最深盘中浮亏 -36.4%；但榜一的尾部更极端，
不能靠“只买第一”来规避尾部风险。

## 尾部统计

{stat_md}

- 2005-2016 榜二：12/41 笔浮亏 ≤ -20%，比例高于同期榜一 20.3%；最差收盘 -29.5%，最深浮亏 -36.4%。
- 2017-2026 榜二：33 笔里没有收盘 ≤ -20%、也没有浮亏 ≤ -20%；最差收盘 -15.4%，最深浮亏 -17.2%。
- 全历史榜二：4/78 收盘 ≤ -20%，12/78 浮亏 ≤ -20%，7/78 单日 ≤ -10%；最差收盘 -29.5%，最深浮亏 -36.4%。

## 个案

{case_md}

2008-08-05 的 600655 豫园股份，15 个交易日后收盘 -29.5%，盘中最大浮亏 -33.1%；
2008-09-17 的 600862 中航高科盘中最大浮亏 -36.4%，最终 -22.8%。
2007-06-05 的 600617 国新能源收盘 -27.0%，15 个交易日到期时撞上跌停，卖不出去，顺延后才成交。
2015 年的 000948 南天信息收盘只亏 -12.0%，但盘中一度 -34.1%。
最近九年最差收盘是 000822 山东海化 -15.4%；金额最大的是 2026 年 000892 欢瑞世纪 -14.1%，半仓亏 10.6 万元。
600633 浙数文化盘中 -15.2%，最后却 +6.7% 到期离场，说明盘中大跌不等于最终亏损。

## 但榜一没有更安全

{lead_md}

同一个 2017-2026 前二组合里，榜一的最大跌幅是 603260 -27.9%，比榜二最差的 -15.4% 更深；
单笔最大金额亏损是榜一 000572 -12.1 万元，也高于榜二 000892 的 -10.6 万元。
全历史里，榜一 600485 *ST信威 -49.2%，信号日不是风险警示，后来退市，C 池照样买到。
002684 猛狮退 -82.9% 是榜一，不是榜二。

## 含义

- 榜二的尾部是真实的，且集中在系统性下跌；2008 年两个榜位一起跌，分散化在危机里几乎失效。
- 但榜一的极端值更深；只买第一不能替代止损和仓位控制。
- 最近九年榜二没有 -20% 级别的大跌，不等于以后没有；2026 年 000892 仍亏 10.6 万元，样本只有 33 笔。
- 25% 止损通常能把 -30% 以上的尾部截断在 -25% 附近（跳空/跌停除外），但 600862 这种盘中 -36.4%、
  收盘 -22.8% 的票会被止损在 -25%，反而比持有更差；是否叠加要接受这个 trade-off。
- 2×50% 的单票 -30% 等于账户 -15%；如果承受不了，应该降单票仓位，而不是只买第一名。

## 限制

“最大浮亏”按后复权日线最低价计算，未建模盘中止损的滑点、封单和排队；单日跌幅按后复权收盘计算；
样本小（榜二 2017-2026 只有 33 笔，2005-2016 只有 41 笔）；历史回测不代表未来，不构成投资建议。

图表与交互页：`榜二大跌个案.html`；统计与个案：`榜二尾部统计.csv`、`榜二大跌个案.csv`；
完整候选：`榜二大跌明细.csv`；复现：`02_代码/analyze_rank2_tail.py`、`02_代码/make_rank2_report.py`。
"""
    (OUT / "榜二大跌个案.md").write_text(md, encoding="utf-8")

    print(stat_md)
    print(case_md)
    print(lead_md)


if __name__ == "__main__":
    main()
