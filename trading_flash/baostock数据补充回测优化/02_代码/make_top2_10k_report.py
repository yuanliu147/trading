#!/usr/bin/env python3
"""生成“100+ 日、榜一+榜二各 1w”的图表、HTML 与摘要。"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "09_百信号日_榜一二_1w"
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
ORANGE = "#d35400"
GREEN = "#2c7a4b"
RED = "#b83b2f"
GREY = "#8a8f98"


def pct(x: float, d: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{d}f}%"


def money(x: float) -> str:
    return f"{x:,.0f}"


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
details{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 14px}
summary{cursor:pointer;color:var(--blue);font-weight:600}
ul{margin:8px 0 12px 20px;padding:0}li{margin:5px 0}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px;margin-top:42px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}.wrap{padding:0 16px 48px}}
"""


def main():
    trades = pd.read_csv(OUT / "榜一二_1w_逐笔.csv", parse_dates=["信号日", "entry_date", "exit_date"])
    summary = pd.read_csv(OUT / "榜一二_1w_汇总.csv")
    annual = pd.read_csv(OUT / "榜一二_1w_年度.csv")
    skipped = pd.read_csv(OUT / "榜一二_1w_跳过重复.csv",
                           parse_dates=["信号日", "entry_date", "exit_date"])

    full = trades[trades["期间"] == "2005-2026"]
    noadd = full[full["版本"] == "不重复"].sort_values("entry_date")
    add = full[full["版本"] == "每天加"].sort_values("entry_date")
    s_full = summary[(summary["期间"] == "2005-2026")]
    s_no = s_full[s_full["版本"] == "不重复"].iloc[0]
    s_add = s_full[s_full["版本"] == "每天加"].iloc[0]
    s_recent = summary[(summary["期间"] == "2017-2026") & (summary["版本"] == "不重复")].iloc[0]
    y2016 = annual[(annual["期间"] == "2005-2026") & (annual["年份"] == 2016)]["盈亏"].iloc[0]

    # 图：累计盈亏 + 年度盈亏
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for d, color, label in [(noadd, BLUE, "已持有就不加"), (add, ORANGE, "每天再加 1w")]:
        d = d.sort_values("exit_date").copy()
        x = [d["entry_date"].iloc[0]] + d["exit_date"].tolist()
        y = [0] + d["pnl"].cumsum().tolist()
        ax.plot(x, y, color=color, lw=1.9, label=label)
        ax.annotate(f"{y[-1] / 1e4:.2f}万", (x[-1], y[-1]), xytext=(5, 0),
                    textcoords="offset points", color=color, fontsize=8, va="center")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title("累计盈亏（2005-2026）", fontsize=12, loc="left")
    ax.set_ylabel("元")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    a = annual[annual["期间"] == "2005-2026"].sort_values("年份")
    colors = [GREEN if v > 0 else RED for v in a["盈亏"]]
    ax2.bar(a["年份"].astype(str), a["盈亏"], color=colors, width=0.6)
    for i, r in enumerate(a.itertuples()):
        ax2.text(i, r.盈亏 + (80 if r.盈亏 >= 0 else -80), f"{r.盈亏 / 1e4:.1f}",
                 ha="center", va="bottom" if r.盈亏 >= 0 else "top", fontsize=7.5)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_title("不重复版：年度盈亏（万元）", fontsize=12, loc="left")
    ax2.set_ylabel("元")
    fig.tight_layout()
    fig.savefig(FIG / "01_累计与年度盈亏.png", bbox_inches="tight")
    plt.close(fig)

    # 汇总表
    sh = ["版本", "期间", "笔数", "胜率", "总盈亏", "平均/笔", "中位/笔", "最差/笔", "峰值并发", "峰值资金", "投入回报率"]
    srows = []
    for version in ["不重复", "每天加"]:
        for period in ["2005-2026", "2005-2016", "2017-2026"]:
            g = summary[(summary["期间"] == period) & (summary["版本"] == version)]
            if g.empty:
                continue
            r = g.iloc[0]
            srows.append([version, period, str(int(r["笔数"])), pct(r["胜率"]),
                          f"{r['总盈亏'] / 1e4:.2f}万", f"{r['平均盈亏']:.0f}",
                          f"{r['中位盈亏']:.0f}", f"{r['最差']:.0f}",
                          str(int(r["峰值并发"])), f"{r['峰值资金'] / 1e4:.1f}万",
                          pct(r["投入回报率"])])
    sum_html = html_table(sh, srows)
    sum_md = md_table(sh, srows)

    # 年度表
    ah = ["年份", "笔数", "胜率", "盈亏（元）"]
    arows = []
    for r in annual[annual["期间"] == "2005-2026"].sort_values("年份").itertuples():
        arows.append([str(int(r.年份)), str(int(r.笔数)), pct(r.胜率), money(r.盈亏)])
    annual_html = html_table(ah, arows)
    annual_md = md_table(ah, arows)

    # 逐日盯市 vs 到期已实现
    mtm = pd.read_csv(OUT / "榜一二_1w_每日盯市.csv", parse_dates=["date"])

    def realized_dd(d):
        d = d.sort_values("exit_date")
        lvl = d["pnl"].cumsum()
        dd = lvl - lvl.cummax()
        i = dd.idxmin()
        return dd.min(), d.loc[i, "exit_date"]

    dd_rows = []
    dd_vals = {}
    for version, color in [("不重复", BLUE), ("每天加", ORANGE)]:
        d = full[full["版本"] == version]
        rdd, rdate = realized_dd(d)
        lv = mtm[version]
        mdd = (lv - lv.cummax())
        mi = mdd.idxmin()
        dd_vals[version] = (rdd, mdd.min(), mtm.loc[mi, "date"])
        peak_cap = s_no["峰值资金"] if version == "不重复" else s_add["峰值资金"]
        dd_rows.append([version, f"{d['pnl'].sum() / 1e4:.2f}万",
                        f"{rdd:,.0f}（{rdate.date()}）",
                        f"{mdd.min():,.0f}（{mtm.loc[mi, 'date'].date()}）",
                        pct(mdd.min() / peak_cap)])
    dd_html = html_table(["版本", "期末盈亏", "到期已实现最大回撤", "逐日盯市最大回撤", "盯市回撤/峰值资金"], dd_rows)
    dd_md = md_table(["版本", "期末盈亏", "到期已实现最大回撤", "逐日盯市最大回撤", "盯市回撤/峰值资金"], dd_rows)

    # 图：逐日盯市 vs 到期已实现
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for version, color in [("不重复", BLUE), ("每天加", ORANGE)]:
        ax.plot(mtm["date"], mtm[version], color=color, lw=1.8, label=f"{version}（逐日盯市）")
        d = full[full["版本"] == version].sort_values("exit_date")
        x = [d["entry_date"].iloc[0]] + d["exit_date"].tolist()
        y = [0] + d["pnl"].cumsum().tolist()
        ax.plot(x, y, color=color, lw=1.0, ls="--", alpha=0.75, label=f"{version}（到期已实现）")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title("逐日盯市 vs 到期已实现", fontsize=12, loc="left")
    ax.set_ylabel("累计盈亏（元）")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)

    labels = ["不重复\n已实现", "不重复\n盯市", "每天加\n已实现", "每天加\n盯市"]
    vals = [dd_vals["不重复"][0], dd_vals["不重复"][1], dd_vals["每天加"][0], dd_vals["每天加"][1]]
    bars = ax2.bar(labels, vals, color=[BLUE, BLUE, ORANGE, ORANGE], width=0.6)
    for b, alpha in zip(bars, [0.45, 0.9, 0.45, 0.9]):
        b.set_alpha(alpha)
    for i, v in enumerate(vals):
        ax2.text(i, v - 250, f"{v:,.0f}", ha="center", va="top", fontsize=8)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_title("最大回撤（元）", fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "02_逐日盯市.png", bbox_inches="tight")
    plt.close(fig)

    # 2008 个案
    y2008 = noadd[noadd["exit_date"].dt.year == 2008].sort_values("entry_date")
    y2008_add = add[add["exit_date"].dt.year == 2008]
    skip2008 = skipped[(skipped["期间"] == "2005-2026") &
                       (skipped["entry_date"].dt.year == 2008)].sort_values("entry_date")
    y2008_rows = [[r.信号日.date().isoformat(), f"榜{'一' if r.排名 == 1 else '二'}",
                   f"{r.代码.split('.')[-1]} {r.名称}",
                   f"{r.entry_date.date()} → {r.exit_date.date()}",
                   money(r.pnl), pct(r.ret)] for r in y2008.itertuples()]
    y2008_html = html_table(["信号日", "排名", "股票", "买入 → 卖出", "盈亏", "收益"], y2008_rows)
    y2008_md = md_table(["信号日", "排名", "股票", "买入 → 卖出", "盈亏", "收益"], y2008_rows)
    skip2008_rows = [[r.信号日.date().isoformat(), f"榜{'一' if r.排名 == 1 else '二'}",
                      f"{r.代码.split('.')[-1]} {r.名称}", r.entry_date.date().isoformat(),
                      money(r.pnl), pct(r.ret)] for r in skip2008.itertuples()]
    skip2008_html = html_table(["信号日", "排名", "股票", "买入日", "如果不跳过：盈亏", "收益"], skip2008_rows)
    skip2008_md = md_table(["信号日", "排名", "股票", "买入日", "如果不跳过：盈亏", "收益"], skip2008_rows)
    win2008 = y2008[y2008["pnl"] > 0]["pnl"].sum()
    loss2008 = y2008[y2008["pnl"] < 0]["pnl"].sum()
    big2008 = y2008.nsmallest(1, "pnl").iloc[0]

    # 最大盈亏
    def trade_rows(d: pd.DataFrame, n: int, largest: bool):
        g = d.nlargest(n, "pnl") if largest else d.nsmallest(n, "pnl")
        rows = []
        for r in g.itertuples():
            rows.append([r.信号日.date().isoformat(), f"榜{'一' if r.排名 == 1 else '二'}",
                         f"{r.代码.split('.')[-1]} {r.名称}",
                         f"{r.entry_date.date()} → {r.exit_date.date()}",
                         money(r.pnl), pct(r.ret), str(int(r.holding_sessions))])
        return rows

    win_rows = trade_rows(noadd, 5, True)
    loss_rows = trade_rows(noadd, 5, False)
    win_html = html_table(["信号日", "排名", "股票", "买入 → 卖出", "盈亏", "收益", "持有"], win_rows)
    loss_html = html_table(["信号日", "排名", "股票", "买入 → 卖出", "盈亏", "收益", "持有"], loss_rows)
    win_md = md_table(["信号日", "排名", "股票", "买入 → 卖出", "盈亏", "收益", "持有"], win_rows)
    loss_md = md_table(["信号日", "排名", "股票", "买入 → 卖出", "盈亏", "收益", "持有"], loss_rows)

    # 全部 37 笔
    all_rows = []
    reason_map = {"max_holding": "15日到期", "take_profit_intraday": "盘中止盈",
                  "take_profit_open": "跳空止盈", "delist_last_close": "退市最后收盘"}
    for r in noadd.itertuples():
        all_rows.append([r.信号日.date().isoformat(), f"榜{'一' if r.排名 == 1 else '二'}",
                         f"{r.代码.split('.')[-1]} {r.名称}",
                         f"{r.entry_date.date()} → {r.exit_date.date()}",
                         money(r.invested), money(r.pnl), pct(r.ret),
                         str(int(r.holding_sessions)), reason_map.get(r.reason, r.reason)])
    all_html = html_table(["信号日", "排名", "股票", "买入 → 卖出", "投入", "盈亏", "收益", "持有", "退出"], all_rows)
    all_md = md_table(["信号日", "排名", "股票", "买入 → 卖出", "投入", "盈亏", "收益", "持有", "退出"], all_rows)

    # 连续 100+ / 重复
    dup = skipped[skipped["期间"] == "2005-2026"].sort_values("entry_date")
    dup_html = html_table(
        ["信号日", "排名", "股票", "买入日", "如果不跳过：盈亏", "收益"],
        [[r.信号日.date().isoformat(), f"榜{'一' if r.排名 == 1 else '二'}",
          f"{r.代码.split('.')[-1]} {r.名称}", r.entry_date.date().isoformat(),
          money(r.pnl), pct(r.ret)] for r in dup.itertuples()])
    dup_md = md_table(
        ["信号日", "排名", "股票", "买入日", "如果不跳过：盈亏", "收益"],
        [[r.信号日.date().isoformat(), f"榜{'一' if r.排名 == 1 else '二'}",
          f"{r.代码.split('.')[-1]} {r.名称}", r.entry_date.date().isoformat(),
          money(r.pnl), pct(r.ret)] for r in dup.itertuples()])

    # 数字
    cagr_no = (1 + s_no["总盈亏"] / s_no["峰值资金"]) ** (1 / 21.8) - 1
    cagr_add = (1 + s_add["总盈亏"] / s_add["峰值资金"]) ** (1 / 21.8) - 1
    cagr_recent = (1 + s_recent["总盈亏"] / s_recent["峰值资金"]) ** (1 / 9.01) - 1

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>100+ 日，榜一+榜二各 1w</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<div class="kicker">100+ 信号日 · 固定 1w</div>
<h1>只在 100+ 信号日买榜一+榜二，各 1 万元</h1>
<p class="sub">C 池（含退市、信号日排除风险警示），32 个 100+ 信号日；T+3 开盘买，+20% 盘中止盈，
15 个交易日收盘卖出；每只 1 万元，不限总仓位，100 股整手，含滑点/佣金/印花税。</p>
<div class="banner">不重复持仓：37 笔，26 胜，总盈亏 +2.44 万元，峰值 7 个仓位 / 6.6 万元。
每天再加 1w：58 笔，总盈亏 +4.12 万元，但峰值 18 个仓位 / 16.4 万元。多出来的 21 笔几乎都是
2016 年初对 600571、000062、601010 等同一波反弹的重复加仓。</div>
</header>

<div class="cards">
<div class="card green"><div class="label">不重复：总盈亏</div><div class="big">+2.44万</div><div class="note">37 笔，26 胜（70.3%）</div></div>
<div class="card blue"><div class="label">不重复：平均每笔</div><div class="big">+659</div><div class="note">中位 +577，最差 -2,306</div></div>
<div class="card amber"><div class="label">不重复：峰值资金</div><div class="big">6.6万</div><div class="note">最多同时 7 个仓位</div></div>
<div class="card red"><div class="label">每天加：峰值资金</div><div class="big">16.4万</div><div class="note">18 个仓位，总盈亏 +4.12万</div></div>
</div>

<section>
<h2>一、口径</h2>
<p>100+ 信号日 = 当天满足冻结条件的股票数 ≥100。每个这样的信号日，取榜一和榜二，T+3 开盘各买 1 万元；
不限制同时持仓数量，也不限制总资金。买入按 100 股整手，1 万元可能买不满（高价股）；
+20% 盘中止盈，15 个交易日收盘卖出；滑点 0.1%/边，佣金最低 5 元，过户费，卖出印花税。</p>
<p>32 个 100+ 日共 64 个“信号日×榜位”候选。其中 58 个真正成交：3 个开盘涨停、1 个停牌无 K 线、
2 个是 2026-10-23（买入日 10/28 在样本外）。不重复持仓再跳过 21 个“同一只股票还在持仓期内”的重复信号，
剩 37 笔；每天加则 58 笔全买。</p>
</section>

<section>
<h2>二、盈利汇总</h2>
<div class="tablewrap">{sum_html}</div>
<p class="small">“峰值资金”按同时持仓的投入成本加总；“投入回报率”= 总盈亏 ÷ 所有交易的投入合计，
不是年化。2005-2026 不重复版按峰值 6.6 万算，21.8 年累计 +36.8%，年化约 {cagr_no * 100:.1f}%；
每天加版按 16.4 万算，年化约 {cagr_add * 100:.1f}%。</p>

<h3>盈亏口径：T+3 开盘买入，逐日盯市</h3>
<p>你问的这点很关键：每笔盈亏的买入价是 <b>T+3 开盘</b>，卖出价由路径决定（+20% 盘中触价/跳空、
15 个交易日到期收盘、跌停/停牌顺延），<b>不是简单用卖出日收盘价除以买入日开盘价</b>。
不过，上面的汇总/年度表是按“到期已实现”记账；未平仓的浮动盈亏没有逐日画出来。现在补了逐日盯市：
未平仓按每日后复权收盘估值，平仓日换成已实现盈亏。</p>
<div class="tablewrap">{dd_html}</div>
<figure>{img(FIG / '02_逐日盯市.png', '逐日盯市 vs 到期已实现')}
<figcaption>实线是逐日盯市，虚线是到期已实现。最终总额一样，但中途回撤更深。</figcaption></figure>
<p>最终总额不变（+2.44 万 / +4.12 万），但中途回撤更深：不重复版逐日最大回撤
{money(dd_vals['不重复'][1])} 元（2016-01-28），到期已实现口径只有 {money(dd_vals['不重复'][0])} 元
（2008-04-29）；每天加版 {money(dd_vals['每天加'][1])} 对 {money(dd_vals['每天加'][0])}。
2016 年年度仍是 {y2016:,.0f} 元，但 1 月路径一度 -8,719 元，年度表看不出来。</p>
<p class="small">所以：金额可以逐日更新；之前的汇总/年度是按到期日结算的。看真实持仓压力用逐日盯市，
看最终交易结果两者总额一致。</p>
</section>

<section>
<h2>三、年度盈亏（不重复版）</h2>
<div class="tablewrap">{annual_html}</div>
<p>只有 2008 是负年（7 笔，胜率 28.6%，{money(win2008 + loss2008)} 元）；2015 最好（11 笔，+10,832 元）。
2017-2021、2023-2025 没有 100+ 信号日，所以没有交易。2026 年（截至 10/23）4 笔全胜，+6,618 元；
10/23 的两笔要等 10/28，样本外未计。</p>
<figure>{img(FIG / '01_累计与年度盈亏.png', '累计与年度盈亏')}
<figcaption>左：累计盈亏；右：不重复版年度盈亏。2015 是主要利润来源，2008 是唯一亏损年。</figcaption></figure>

<h3>2008 为什么是负的？</h3>
<p>2008 年 7 笔不是全亏：2 笔盈利合计 {money(win2008)} 元，5 笔亏损合计 {money(loss2008)} 元，
净 {money(win2008 + loss2008)} 元。最大的一笔 {big2008.代码.split('.')[-1]} {big2008.名称}
亏 {money(big2008.pnl)} 元（{pct(big2008.ret)}），一笔就超过了全年的净亏损。</p>
<div class="tablewrap">{y2008_html}</div>
<p>{big2008.代码.split('.')[-1]} 在 {big2008.entry_date.date()} 以 {big2008.entry_px:.2f} 买入，
{big2008.exit_date.date()} 收 {big2008.exit_px:.2f}，买入后出现 -5.6%、-7.8%、-6.6%、-8.7%、-9.5% 等连续下跌；
同期主板等权指数从 3/19 到 4/14 约 -13.5%，4/1 单日 -8.2%、4/2 -5.5%。
这是持续熊市里的失败反弹，不是一次性洗盘。</p>
<p class="small">表中的 *ST 是当前名称；2008 年信号日 600811、000838 的 is_st 都是 0，
风险警示过滤没有拦住它们，是后来才变成 ST 的。</p>
<p>另外，不重复规则还跳过了 000838 在 3/20 的 +1,730、000952 在 4/3 的 +1,841 和 4/7 的 +2,388，
合计 +{skip2008['pnl'].sum():,.0f} 元；如果每天加 1w，2008 反而是 +{y2008_add['pnl'].sum():,.0f} 元。
所以 2008 负 = 一笔榜二大亏 + 不重复规则跳过同一只股票的后续盈利，不是“100+ 信号在 2008 全失效”。</p>
<div class="tablewrap">{skip2008_html}</div>
</section>

<section>
<h2>四、最大的 5 笔盈利与亏损</h2>
<h3>盈利</h3>
<div class="tablewrap">{win_html}</div>
<h3>亏损</h3>
<div class="tablewrap">{loss_html}</div>
<p>最大亏损是 2008-03-19 的榜二 600811（-26.2%，-2,306 元）；2016-01-05 的榜一 000584、榜二 601766
分别亏 -15.1%、-14.1%。最大盈利是 2016-01-19 的榜二 600288（+29.5%，+2,901 元）。</p>
</section>

<section>
<h2>五、交易记录（不重复版，37 笔）</h2>
<details><summary>展开全部 37 笔</summary>
<div class="tablewrap">{all_html}</div>
</details>
<p class="small">完整逐笔（含“每天加”版和候选/重复跳过）：`榜一二_1w_逐笔.csv`、`榜一二_1w_候选.csv`、
`榜一二_1w_跳过重复.csv`。持有日数是交易日，含买入日；600571 在 2016-02-01 那笔因停牌顺延到 6/14。</p>
</section>

<section>
<h2>六、连续 100+ 日与重复信号</h2>
<p>连续 100+ 日主要出现在 2008、2015、2016：2007-10-26/29、2008-03-19/20、2008-04-03/07、
2015-07-08/09/10、2015-08-27/28、2016-01-08/12/14/18/19、2016-01-29/02-01/02/03。
榜一经常重复：600571 在 2016-01-08/12/14/18/22/25/27 反复出现，002072 在 2015-07-09/10 连续出现。</p>
<p>不重复版跳过 21 个重复信号。如果这些信号也各加 1w，理论上多赚 +{dup['pnl'].sum() / 1e4:.2f} 万元，
但峰值资金从 6.6 万升到 16.4 万，且多出来的利润主要是 2016 年初对同一波反弹的重复加仓。</p>
<div class="tablewrap">{dup_html}</div>
<p class="small">“如果不跳过”是该信号单独按 1w 模拟的盈亏，不代表不重复版少赚了同样金额；
两条路径的持仓和资金占用不同。</p>
</section>

<section>
<h2>七、怎么理解</h2>
<ul>
<li><b>胜率不低，但总利润不大。</b>37 笔 70.3% 胜率、平均每笔 +659 元；因为每只只放 1w、21 年只有 37 笔，
总盈亏只有 +2.44 万。按峰值资金 6.6 万算，年化约 {cagr_no * 100:.1f}%，资金大部分时间闲置。</li>
<li><b>2017-2026 只有 6 笔，全胜，+9,220 元。</b>按峰值 2.9 万算，年化约 {cagr_recent * 100:.1f}%；
但只有 6 笔，不能当成稳定年化。</li>
<li><b>每天加 1w 的绝对利润更高，但资本效率更低。</b>多 21 笔只多 +1.68 万，峰值资金却多 9.7 万；
年化从约 {cagr_no * 100:.1f}% 降到约 {cagr_add * 100:.1f}%。它本质是对同一波反弹加仓，不是独立信号。</li>
<li><b>2008 是唯一亏损年。</b>7 笔里只有 2 笔盈利，榜二 600811 -26.2%；如果只做 2017-2026，
看不到这种尾部。</li>
<li><b>1w 固定金额不是复利。</b>盈亏按每笔 1w 累加，没有把盈利再投入；如果按资金曲线复利，结果会不同。</li>
</ul>
</section>

<section>
<h2>八、限制</h2>
<ul>
<li>每只 1w、100 股整手，高价股可能买不满 1w；实际投入略低于 1w。</li>
<li>4 个候选被开盘涨停/停牌拒买，2 个 2026-10-23 的信号在样本外；未计盘口排队和封单。</li>
<li>“不重复”按同一只股票在持仓期内不再加；如果你允许同一只股票多笔，见“每天加”版本。</li>
<li>100+ 是人为阈值；50-99 个信号的日子没有纳入。样本只有 32 个信号日、37 笔，不能外推。</li>
<li>历史回测不代表未来，不构成投资建议。</li>
</ul>
</section>

<footer>
口径：C 池 = 全主板含退市股、信号日 is_st=1 不买；成交价不复权，信号按后复权；
成本含 0.1%/边滑点、佣金 0.025%（最低 5 元）、过户费、卖出印花税。
复现：<code>02_代码/analyze_top2_10k.py</code> 生成交易与统计，<code>02_代码/make_top2_10k_report.py</code> 生成本页。
</footer>
</div></body></html>"""
    (OUT / "榜一二_1w.html").write_text(html, encoding="utf-8")

    md = f"""# 100+ 信号日，榜一+榜二各 1 万元

C 池（含退市、信号日排除风险警示），32 个 100+ 信号日；T+3 开盘买，+20% 盘中止盈，15 个交易日收盘卖出；
每只 1 万元，不限总仓位，100 股整手，含滑点/佣金/印花税。

## 短答

- 不重复持仓：37 笔，26 胜（70.3%），总盈亏 +2.44 万元，平均每笔 +659 元，峰值 7 个仓位 / 6.6 万元。
- 每天再加 1w：58 笔，41 胜（70.7%），总盈亏 +4.12 万元，峰值 18 个仓位 / 16.4 万元。
- 2017-2026：6 笔全胜，+9,220 元，峰值 3 个仓位 / 2.9 万元。
- 2008 是唯一亏损年：7 笔，胜率 28.6%，{money(win2008 + loss2008)} 元。

## 汇总

{sum_md}

### 盈亏口径：T+3 开盘买入，逐日盯市

每笔盈亏的买入价是 T+3 开盘，卖出价由路径决定（+20% 盘中触价/跳空、15 个交易日到期收盘、
跌停/停牌顺延），不是简单用卖出日收盘价除以买入日开盘价。上面的汇总/年度表按“到期已实现”记账；
逐日盯市则把未平仓按每日后复权收盘估值，平仓日换成已实现盈亏。最终总额不变，但中途回撤更深。

{dd_md}

不重复版逐日最大回撤 {money(dd_vals['不重复'][1])} 元（2016-01-28），到期已实现口径只有
{money(dd_vals['不重复'][0])} 元（2008-04-29）；每天加版 {money(dd_vals['每天加'][1])} 对
{money(dd_vals['每天加'][0])}。2016 年年度仍 {y2016:,.0f} 元，但 1 月路径一度 -8,719 元，年度表看不出来。

## 年度（不重复版）

{annual_md}

### 2008 为什么是负的

2008 年 7 笔不是全亏：2 笔盈利合计 {money(win2008)} 元，5 笔亏损合计 {money(loss2008)} 元，
净 {money(win2008 + loss2008)} 元。最大的一笔 {big2008.代码.split('.')[-1]} {big2008.名称}
亏 {money(big2008.pnl)} 元（{pct(big2008.ret)}），一笔就超过了全年的净亏损。

{y2008_md}

{big2008.代码.split('.')[-1]} 在 {big2008.entry_date.date()} 以 {big2008.entry_px:.2f} 买入，
{big2008.exit_date.date()} 收 {big2008.exit_px:.2f}，买入后出现 -5.6%、-7.8%、-6.6%、-8.7%、-9.5% 等连续下跌；
同期主板等权指数从 3/19 到 4/14 约 -13.5%，4/1 单日 -8.2%、4/2 -5.5%。这是持续熊市里的失败反弹。
表中的 *ST 是当前名称；2008 年信号日 600811、000838 的 is_st 都是 0，风险警示过滤没有拦住它们，是后来才变成 ST 的。

不重复规则还跳过了 000838 在 3/20 的 +1,730、000952 在 4/3 的 +1,841 和 4/7 的 +2,388，
合计 +{skip2008['pnl'].sum():,.0f} 元；如果每天加 1w，2008 反而是 +{y2008_add['pnl'].sum():,.0f} 元。
所以 2008 负 = 一笔榜二大亏 + 不重复规则跳过同一只股票的后续盈利，不是“100+ 信号在 2008 全失效”。

{skip2008_md}

## 最大盈利 5 笔

{win_md}

## 最大亏损 5 笔

{loss_md}

## 全部 37 笔

{all_md}

## 连续 100+ 与重复

连续 100+ 日：2007-10-26/29、2008-03-19/20、2008-04-03/07、2015-07-08/09/10、
2015-08-27/28、2016-01-08/12/14/18/19、2016-01-29/02-01/02/03。
600571 在 2016-01-08/12/14/18/22/25/27 反复出现，002072 在 2015-07-09/10 连续出现。

不重复版跳过 21 个重复信号；这些信号如果各加 1w，理论上多赚 +{dup['pnl'].sum() / 1e4:.2f} 万元，
但峰值资金从 6.6 万升到 16.4 万，且多出来的利润主要是 2016 年初对同一波反弹的重复加仓。

{dup_md}

## 怎么理解

- 胜率不低，但总利润不大：37 笔、平均 +659 元，21 年总盈亏 +2.44 万；按峰值 6.6 万年化约 {cagr_no * 100:.1f}%，资金大部分时间闲置。
- 2017-2026 只有 6 笔全胜，+9,220 元；按峰值 2.9 万年化约 {cagr_recent * 100:.1f}%，但只有 6 笔。
- 每天加 1w 绝对利润更高，但多 21 笔只多 +1.68 万，峰值资金多 9.7 万；年化从约 {cagr_no * 100:.1f}% 降到约 {cagr_add * 100:.1f}%。
- 2008 是唯一亏损年，榜二 600811 -26.2%；只做 2017-2026 看不到这种尾部。
- 1w 固定金额不是复利；按资金曲线复利会不同。

## 限制

每只 1w、100 股整手，高价股可能买不满；4 个候选被涨停/停牌拒买，2 个 10/23 信号在样本外；
“不重复”按持仓期内同一只股票不再加；100+ 是人为阈值；32 个信号日、37 笔，不能外推；
历史回测不代表未来，不构成投资建议。

图表与页面：`榜一二_1w.html`；逐笔：`榜一二_1w_逐笔.csv`；候选/重复：`榜一二_1w_候选.csv`、
`榜一二_1w_跳过重复.csv`；复现：`02_代码/analyze_top2_10k.py`、`02_代码/make_top2_10k_report.py`。
"""
    (OUT / "榜一二_1w_结论.md").write_text(md, encoding="utf-8")

    print(sum_md)
    print(annual_md)
    print(win_md)
    print(loss_md)


if __name__ == "__main__":
    main()
