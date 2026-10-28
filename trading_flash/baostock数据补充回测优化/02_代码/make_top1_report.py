#!/usr/bin/env python3
"""生成“只买第一名 vs 买前二”的图表、单文件 HTML 与结论摘要。"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_只买第一名"
RES = OUT / "结果"
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

C2 = "#1f6fb2"
C1 = "#d35400"
GREY = "#8a8f98"


def load_eq(period: str, label: str) -> pd.DataFrame:
    p = RES / period / f"C_含退市_排除风险警示_{label}_每日权益.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df["nav"] = df.equity / df.equity.iloc[0]
    df["dd"] = df.nav / df.nav.cummax() - 1
    return df


def eq_chart(period: str, tag: str, title: str, ticks: list[float], dd_floor: float = -80):
    a = load_eq(period, "买前二")
    b = load_eq(period, "只买第一名")
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 5.9), sharex=True,
        gridspec_kw={"height_ratios": [2.35, 1.0]})
    for d, color, label in [(a, C2, "买前二"), (b, C1, "只买第一名")]:
        ax.plot(d.date, d.nav, color=color, lw=1.9, label=label)
        ax.annotate(f"{d.nav.iloc[-1]:.1f}x", (d.date.iloc[-1], d.nav.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", color=color,
                    fontsize=9, va="center")
        ax2.fill_between(d.date, d.dd * 100, 0, color=color, alpha=0.12)
        ax2.plot(d.date, d.dd * 100, color=color, lw=1.2)
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.set_ylabel("净值（起点=1，对数轴）")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13, loc="left", pad=10)
    ax2.set_ylabel("回撤（%）")
    ax2.set_ylim(dd_floor, 5)
    ax2.set_xlim(a.date.min(), a.date.max())
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(FIG / tag, bbox_inches="tight")
    plt.close(fig)


def annual_chart():
    y = pd.read_csv(OUT / "只买第一名_年度收益.csv")
    p = y[y["期间"] == "2017-2026"].pivot(index="年份", columns="选股", values="收益")
    p = p[["买前二", "只买第一名"]]
    idx = [str(int(i)) + ("*" if int(i) in (2017, 2026) else "") for i in p.index]
    x = range(len(p))
    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    w = 0.38
    ax.bar([i - w / 2 for i in x], p["买前二"] * 100, width=w, color=C2, label="买前二")
    ax.bar([i + w / 2 for i in x], p["只买第一名"] * 100, width=w, color=C1, label="只买第一名")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(idx)
    ax.set_ylabel("年度收益（%）")
    ax.set_ylim(-20, 118)
    ax.set_title("2017-2026 年度收益：只买第一名 vs 买前二", fontsize=13, loc="left", pad=10)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    ax.text(0.995, 0.02, "* 2017 从 10/19 起，2026 截至 10/23", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "02_年度收益_2017_2026.png", bbox_inches="tight")
    plt.close(fig)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def pp(x: float, d: int = 1) -> str:
    return f"{x * 100:+.{d}f}pp"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def html_table(headers: list[str], rows: list[list[str]], cls: str = "data") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}">'


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#18222d;--muted:#63707e;--blue:#1f6fb2;--blue2:#e8f1fa;
--red:#b83b2f;--red2:#fbecea;--amber:#b8860b;--amber2:#fdf6e3;--green:#2c7a4b;--line:#e3e7ec}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;font-size:15px;line-height:1.75}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 64px}
header{padding:52px 0 28px;border-bottom:1px solid var(--line)}
.kicker{color:var(--blue);font-weight:600;letter-spacing:.08em;font-size:12px;text-transform:uppercase}
h1{font-size:38px;line-height:1.22;margin:10px 0 12px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:15px;max-width:850px}
.banner{margin:24px 0 0;background:var(--red2);border-left:4px solid var(--red);padding:15px 19px;border-radius:0 10px 10px 0;color:#5a1f18;font-weight:600}
section{margin:36px 0}h2{font-size:25px;margin:0 0 6px;letter-spacing:-.01em}h3{font-size:17px;margin:22px 0 8px}
p{margin:8px 0 12px}.small{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 17px 13px}
.card .label{color:var(--muted);font-size:12px}.card .big{font-size:27px;font-weight:700;margin:3px 0 2px;letter-spacing:-.02em}
.card .note{color:var(--muted);font-size:12px}.card.blue .big{color:var(--blue)}.card.red .big{color:var(--red)}.card.green .big{color:var(--green)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.panel.blue{background:var(--blue2);border-color:#cfe2f3}.panel.amber{background:var(--amber2);border-color:#efdfb8}
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
    cmp = pd.read_csv(OUT / "只买第一名_对比.csv")
    rank = pd.read_csv(OUT / "只买第一名_排名分解.csv")
    stop = pd.read_csv(OUT / "只买第一名_止损25.csv")

    eq_chart("2017-2026", "01_权益与回撤_2017_2026.png",
             "2017-2026：只买第一名 vs 买前二（2×50%）",
             [0.8, 1, 1.5, 2, 3, 5, 8, 12, 16, 20], dd_floor=-40)
    eq_chart("2005-2026", "03_权益与回撤_2005_2026.png",
             "2005-2026：只买第一名 vs 买前二（2×50%）",
             [0.3, 0.5, 1, 2, 5, 10, 20, 50], dd_floor=-80)
    annual_chart()

    # ---------- 主表（C 池） ----------
    c = cmp[cmp["股票池"].str.startswith("C")]
    headers = ["期间", "选股", "年化", "最大回撤", "卡玛", "交易", "胜率", "平均单笔", "平均仓位", "期末净值"]
    rows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        for lab in ["买前二", "只买第一名"]:
            r = c[(c["期间"] == period) & (c["选股"] == lab)].iloc[0]
            rows.append([period, lab, pct(r["年化收益"]), pct(r["最大回撤"]),
                         f"{r['卡玛']:.2f}", str(int(r["交易笔数"])), pct(r["胜率"]),
                         pct(r["平均单笔"]), pct(r["平均仓位"]), f"{r['期末净值']:.2f}x"])
    main_html = html_table(headers, rows)
    main_md = md_table(headers, rows)

    # 差值
    dh = ["期间", "年化变化", "回撤变化（正=更浅）", "卡玛变化", "交易变化", "期末净值变化"]
    drows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        a = c[(c["期间"] == period) & (c["选股"] == "买前二")].iloc[0]
        b = c[(c["期间"] == period) & (c["选股"] == "只买第一名")].iloc[0]
        drows.append([period, pp(b["年化收益"] - a["年化收益"]),
                      pp(b["最大回撤"] - a["最大回撤"]),
                      f"{b['卡玛'] - a['卡玛']:+.2f}",
                      f"{int(b['交易笔数'] - a['交易笔数']):+d}",
                      f"{b['期末净值'] - a['期末净值']:+.2f}x"])
    diff_html = html_table(dh, drows)
    diff_md = md_table(dh, drows)

    # ---------- 仓位使用 ----------
    ph = ["期间", "选股", "两仓天数", "一仓天数", "空仓天数", "平均仓位", "平均持仓数"]
    prows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        for lab in ["买前二", "只买第一名"]:
            e = load_eq(period, lab)
            prows.append([period, lab, pct((e.npos == 2).mean()), pct((e.npos == 1).mean()),
                          pct((e.npos == 0).mean()), pct((1 - e.cash / e.equity).mean()),
                          f"{e.npos.mean():.2f}"])
    pos_html = html_table(ph, prows)
    pos_md = md_table(ph, prows)

    # ---------- 交易分解（C，2017-2026 与 2005-2016） ----------
    rh = ["期间", "组", "笔数", "平均单笔", "中位单笔", "胜率", "净盈亏（万）"]
    rrows = []
    label_map = {
        (1, "买前二"): "前二中的第 1 名",
        (2, "买前二"): "前二中的第 2 名",
    }
    for period in ["2017-2026", "2005-2016"]:
        for rk, run in [(1, "买前二"), (2, "买前二")]:
            g = rank[(rank["期间"] == period) & (rank["运行"] == run)
                     & (rank["排名"].astype(str) == str(rk))]
            if g.empty:
                continue
            r = g.iloc[0]
            rrows.append([period, label_map[(rk, run)], str(int(r["笔数"])),
                          pct(r["平均单笔"]), pct(r["中位单笔"]), pct(r["胜率"]),
                          f"{r['净盈亏合计'] / 1e4:.1f}"])
        for tag, name in [("只买第一多出的交易", "只买第一多出的"),
                          ("仅前二买入的交易", "仅前二买入的")]:
            g = rank[(rank["期间"] == period) & (rank["运行"] == "对比") & (rank["排名"] == tag)]
            if g.empty:
                continue
            r = g.iloc[0]
            rrows.append([period, name, str(int(r["笔数"])), pct(r["平均单笔"]),
                          pct(r["中位单笔"]), pct(r["胜率"]),
                          f"{r['净盈亏合计'] / 1e4:.1f}"])
    rank_html = html_table(rh, rrows)
    rank_md = md_table(rh, rrows)

    # ---------- 25% 止损 ----------
    sh = ["期间", "选股", "年化", "最大回撤", "卡玛", "交易"]
    srows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        for lab in ["买前二", "只买第一名"]:
            r = stop[(stop["期间"] == period) & (stop["选股"] == lab)].iloc[0]
            srows.append([period, lab, pct(r["cagr"]), pct(r["max_dd"]),
                          f"{r['calmar']:.2f}", str(int(r["n_trades"]))])
    stop_html = html_table(sh, srows)
    stop_md = md_table(sh, srows)

    # ---------- A 池复核 ----------
    a = cmp[cmp["股票池"].str.startswith("A")]
    ah = ["期间", "选股", "年化", "最大回撤", "交易"]
    arows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        for lab in ["买前二", "只买第一名"]:
            r = a[(a["期间"] == period) & (a["选股"] == lab)].iloc[0]
            arows.append([period, lab, pct(r["年化收益"]), pct(r["最大回撤"]),
                          str(int(r["交易笔数"]))])
    a_html = html_table(ah, arows)
    a_md = md_table(ah, arows)

    # ---------- 2026 个案 ----------
    t1 = pd.read_csv(RES / "2017-2026" / "C_含退市_排除风险警示_只买第一名_逐笔.csv",
                     parse_dates=["signal_date", "entry_date", "exit_date"])
    worst = t1.nsmallest(1, "ret").iloc[0]
    worst_code = str(worst["code"]).split(".")[-1]
    y = pd.read_csv(OUT / "只买第一名_年度收益.csv")
    y26 = y[(y["期间"] == "2017-2026") & (y["年份"] == 2026)]
    r26_2 = float(y26[y26["选股"] == "买前二"]["收益"].iloc[0])
    r26_1 = float(y26[y26["选股"] == "只买第一名"]["收益"].iloc[0])

    # ---------- HTML ----------
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>只买第一名 vs 买前二</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<div class="kicker">策略变体 · 两个半仓</div>
<h1>每次信号日只买第一名，收益会更好吗？</h1>
<p class="sub">baostock 全主板（含退市股，排除风险警示），2005-2026。信号按后复权，成交按不复权实际价；
T 日收盘排名，T+3 开盘买入，+20% 盘中止盈，15 个交易日收盘卖出；2 个半仓。数据截至 2026-10-23。</p>
<div class="banner">短答：最近九年更差；2005-2016 更好；全周期略好，但优势全部来自 2005-2016。这不是一个稳定的改进。</div>
</header>

<div class="cards">
<div class="card blue"><div class="label">2017-2026 年化</div><div class="big">28.5%</div><div class="note">买前二 35.0%</div></div>
<div class="card red"><div class="label">2017-2026 最大回撤</div><div class="big">-29.5%</div><div class="note">买前二 -27.4%</div></div>
<div class="card green"><div class="label">2005-2026 年化</div><div class="big">18.7%</div><div class="note">买前二 16.8%</div></div>
<div class="card"><div class="label">平均仓位</div><div class="big">24.6%</div><div class="note">买前二 26.4%</div></div>
</div>

<section>
<h2>一、数字</h2>
<p>同一套规则，只改一件事：每个信号日最多买 rank≤1，第二名不递补；仓位仍是 2×50%。
下表是含退市股、排除风险警示的 C 池。</p>
<div class="tablewrap">{main_html}</div>
<h3>只买第一减去买前二</h3>
<div class="tablewrap">{diff_html}</div>
<p class="small">“平均仓位”是每日持仓市值占权益的平均比例，不是“有持仓的天数占比”。
回撤变化为正表示回撤更浅。2017-2026 的 10 年里有 9 年整年、2026 年为部分年度。</p>
</section>

<section>
<h2>二、不是“半仓空转”</h2>
<p>听起来只买第一会空出一半仓位，但实际平均仓位只低 1.8 个百分点。原因是策略大多数时间本来就不在场：
2017-2026 有 67% 的交易日空仓，两个仓位同时满着的日子只占 23.1%（买前二）和 16.9%（只买第一）。
第二仓位不是天天在用，它只在信号簇里起作用。</p>
<div class="tablewrap">{pos_html}</div>
<p class="small">所以收益差主要不是“现金拖累”，而是“买了哪些票、在什么顺序买”。</p>
</section>

<section>
<h2>三、交易分解：第一名单笔更强，但总账不一定更好</h2>
<p>2017-2026 里，前二中的第 1 名单笔平均 <b>+7.20%</b>，第 2 名 <b>+4.35%</b>，第一名确实更强。
但只买第一多出来的 29 笔只有 <b>+1.88%</b>，而买前二多出来的 37 笔有 <b>+4.48%</b>。
你为了等“下一天的第一名”，放弃的是“同一天的第二名”，后者反而更好。</p>
<p>2005-2016 正好反过来：只买第一多出的 38 笔 <b>+5.68%</b>，仅前二买入的 45 笔 <b>+1.53%</b>。
这不是“第一名一定更好”的稳定规律，而是市场阶段和路径差异。</p>
<div class="tablewrap">{rank_html}</div>
<p class="small">“净盈亏”是各笔绝对金额之和，受当时权益基数影响，不宜横向直接比较；看平均单笔和胜率更合适。
两组“多出/仅前二”的交易集合按“信号日 + 代码”取差集。</p>
<p>榜二的具体大跌个案（2008 年最差收盘 -29.5%、盘中浮亏 -36.4%）见
<a href="榜二大跌个案.html">榜二大跌个案</a>。</p>
</section>

<section>
<h2>四、2026 的个案：单票集中风险</h2>
<p>只买第一在 2026 年吃到一笔 <b>{worst_code}</b>：{worst['signal_date'].date()} 信号，
{worst['entry_date'].date()} 买入，{worst['exit_date'].date()} 到期，<b>{pct(worst['ret'])}</b>；
半仓一笔亏 <b>{abs(worst['proceeds'] - worst['invested']) / 1e4:.1f} 万元</b>。它不是风险警示股，风险警示过滤拦不住。</p>
<p>结果：2026 年（截至 10/23）只买第一 <b>{pct(r26_1)}</b>，买前二 <b>{pct(r26_2)}</b>。
只买第一的最大回撤发生在 2026-06-16 至 2026-07-30，<b>-29.5%</b>；买前二的最大回撤仍是 2022 年的 -27.4%。
少一个仓位，不等于少一份尾部风险，反而可能更集中。</p>
</section>

<section>
<h2>五、权益与年度收益</h2>
<p>2017-2026 的 10 个年度里，只买第一只在 2020、2022、2025 三年跑赢，其余 7 年落后；
不是“某一年拖累”，而是多数年份都不占优。</p>
<figure>{img(FIG / '01_权益与回撤_2017_2026.png', '2017-2026 权益与回撤对比')}
<figcaption>2017-2026：只买第一 9.59x，买前二 14.93x；下半部分是回撤。</figcaption></figure>
<figure>{img(FIG / '02_年度收益_2017_2026.png', '2017-2026 年度收益对比')}
<figcaption>年度收益。10 年只赢 3 年（2020、2022、2025）；2021、2026 只买第一明显更差。</figcaption></figure>
<figure>{img(FIG / '03_权益与回撤_2005_2026.png', '2005-2026 权益与回撤对比')}
<figcaption>2005-2026：只买第一 41.99x，买前二 29.58x；优势主要来自 2005-2016。</figcaption></figure>
</section>

<section>
<h2>六、叠加 25% 止损后，结论不变</h2>
<p>前面是纯 2×50% 无止损。把之前推荐过的 25% 止损叠上去，最近九年仍然是只买第一更差：
<b>29.6% / -26.7%</b> 对 <b>35.6% / -26.2%</b>；全周期 <b>20.1% / -55.9%</b> 对 <b>19.3% / -60.6%</b>，
优势依旧来自 2005-2016。</p>
<div class="tablewrap">{stop_html}</div>
</section>

<section>
<h2>七、旧池复核</h2>
<p>换成“仅当前上市”的 A 池，方向也一样：最近九年 <b>36.7%</b> 对 <b>43.2%</b>，回撤 <b>-29.6%</b> 对 <b>-15.8%</b>；
2005-2016 是 <b>11.0%</b> 对 <b>3.6%</b>。所以不是 C 池的口径造成的。</p>
<div class="tablewrap">{a_html}</div>
</section>

<section>
<h2>八、对当前信号的含义</h2>
<p>按 2026-10-23 的冻结名单，买前二是 <b>603778、605376</b>；只买第一只会买 <b>603778</b>，跳过 605376。
2026-10-26 是新信号日，603778 仍是第 1（002240 第 2）；如果 603778 已在 10/28 成交，10/29 不会再加。
原计划没有改；要不要把执行规则改成“只买第一”，是一个独立决策。</p>
</section>

<section>
<h2>九、限制</h2>
<ul>
<li>样本小：2017-2026 只有 101 / 93 笔，单年差异可能由一两笔决定；2005-2016 的 110 / 103 笔也不算多。</li>
<li>优势/劣势都集中在特定市场阶段。2005-2016 是趋势与危机市，2017-2026 是震荡市；不能把任一段外推成长期规律。</li>
<li>只买第一在 2026 的回撤更深，说明单票集中风险；实盘还应叠加止损、仓位上限和流动性约束。</li>
<li>盘中止盈/止损按触价成交，未建模排队、封单和滑点扩大；大资金另有容量问题。</li>
<li>2024-2026 已在旧研究中使用，不是纯净留出；2005-2016 对冻结规则是样本外，但市场结构与现在不同。</li>
<li>历史回测不代表未来，不构成投资建议。</li>
</ul>
</section>

<footer>
口径：C 池 = 全主板含退市股、信号日 is_st=1 不买；成交价不复权，信号/收益后复权；成本含 0.1%/边滑点、佣金、过户费、卖出印花税。
复现：<code>02_代码/run_top1.py</code> 生成结果，<code>02_代码/make_top1_report.py</code> 生成本页。
</footer>
</div></body></html>"""
    (OUT / "只买第一名.html").write_text(html, encoding="utf-8")

    # ---------- Markdown ----------
    md = f"""# 只买第一名 vs 买前二（2×50%）

数据：baostock 全主板，含退市股，排除风险警示（C 池）；2005-2026，2026-10-23 截止。
执行：T 日收盘排名，T+3 开盘买入；+20% 盘中止盈；15 个交易日收盘卖出；2 个半仓，无止损。

## 结论

- **2017-2026：只买第一更差。** 年化 28.5% 对 35.0%，最大回撤 -29.5% 对 -27.4%，期末 9.59x 对 14.93x。
- **2005-2016：只买第一更好。** 11.8% 对 4.9%，回撤 -67.9% 对 -71.5%，期末 3.80x 对 1.78x。
- **2005-2026：略好，但优势全部来自 2005-2016。** 18.7% 对 16.8%，回撤 -67.9% 对 -71.5%，期末 41.99x 对 29.58x。
- 这不是稳定的改进；最近九年反而更差。原因不是“半仓空转”：平均仓位只差 1.8 个百分点。

## 主表

{main_md}

差值：

{diff_md}

## 仓位使用

{pos_md}

2017-2026 有 67% 的交易日空仓；两仓同时满着只占 23.1%（买前二）和 16.9%（只买第一）。
第二仓位只在信号簇里起作用，所以“只买第一”并没有把资金占用减半。

## 交易分解

{rank_md}

2017-2026：第 1 名单笔 +7.20%，第 2 名 +4.35%；但只买第一多出的 29 笔只有 +1.88%，
买前二多出的 37 笔有 +4.48%。为了等下一天的第一名，放弃同一天的第二名，后者反而更好。
2005-2016 相反：多出的 38 笔 +5.68%，仅前二的 45 笔 +1.53%。差异是阶段性的，不是稳定规律。

榜二的具体大跌个案（2008 年最差收盘 -29.5%、盘中浮亏 -36.4%）见 `榜二大跌个案.html`。

## 2026 个案

{worst_code}：{worst['signal_date'].date()} 信号，{worst['entry_date'].date()} 买入，{worst['exit_date'].date()} 到期，收益 {pct(worst['ret'])}；
半仓一笔亏 {abs(worst['proceeds'] - worst['invested']) / 1e4:.1f} 万元。它不是风险警示股，风险警示过滤拦不住。
2026 年（截至 10/23）只买第一 {pct(r26_1)}，买前二 {pct(r26_2)}；只买第一最大回撤 -29.5%（2026-06-16 至 07-30）。
2017-2026 的 10 个年度里只赢 3 年（2020、2022、2025），其余 7 年落后。

## 25% 止损复核

{stop_md}

最近九年仍然更差（29.6% 对 35.6%），全周期略好（20.1% 对 19.3%），结论不变。

## 旧池复核（仅当前上市）

{a_md}

## 当前信号

2026-10-23 冻结名单：买前二为 603778、605376；只买第一只会买 603778，跳过 605376。
2026-10-26 新信号日 603778 仍是第 1（002240 第 2）；若 603778 已在 10/28 成交，10/29 不再加。
原计划未改。

## 限制

样本小（101 / 93 笔），单年差异可能由一两笔决定；两段市场结构不同，不能外推；只买第一在 2026 的回撤更深，
说明单票集中风险；未建模排队、封单和滑点扩大。历史回测不代表未来，不构成投资建议。

图表与交互页：`只买第一名.html`；逐笔与权益：`结果/`；复现：`02_代码/run_top1.py`、`02_代码/make_top1_report.py`。
"""
    (OUT / "只买第一名_结论.md").write_text(md, encoding="utf-8")

    print("done")
    print(main_md)
    print(diff_md)


if __name__ == "__main__":
    main()
