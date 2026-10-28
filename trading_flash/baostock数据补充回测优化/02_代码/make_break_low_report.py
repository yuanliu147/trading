#!/usr/bin/env python3
"""生成“T+3 开盘跌破信号日低点”的图表、HTML 与摘要。"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "07_破信号低点"
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
GREY = "#63707e"


def pct(x: float, d: int = 1) -> str:
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
.banner{margin:24px 0 0;background:var(--blue2);border-left:4px solid var(--blue);padding:15px 19px;border-radius:0 10px 10px 0;color:#17415f;font-weight:600}
section{margin:36px 0}h2{font-size:25px;margin:0 0 6px}h3{font-size:17px;margin:22px 0 8px}
p{margin:8px 0 12px}.small{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 17px 13px}
.card .label{color:var(--muted);font-size:12px}.card .big{font-size:26px;font-weight:700;margin:3px 0 2px;letter-spacing:-.02em}
.card .note{color:var(--muted);font-size:12px}.card.blue .big{color:var(--blue)}.card.red .big{color:var(--red)}.card.green .big{color:var(--green)}.card.amber .big{color:var(--amber)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:14px 0}
.panel.amber{background:var(--amber2);border-color:#efdfb8}
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
    stat = pd.read_csv(OUT / "破信号低点_统计.csv")
    pool = pd.read_csv(OUT / "破信号低点_分池统计.csv")
    detail = pd.read_csv(OUT / "破信号低点_交易明细.csv", parse_dates=["信号日", "买入日"])
    stocks = pd.read_csv(ROOT / "05_证据/主板股票清单.csv")
    names = stocks.set_index("code")["code_name"].to_dict()

    c = stat.copy()
    order = ["未破", "破位", "破位+5日收盘收回", "破位+5日未收回"]
    c["组"] = pd.Categorical(c["组"], order, ordered=True)

    # 图：2017-2026 与 2005-2026 两组
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for ax, period, title in zip(axes, ["2017-2026", "2005-2026"], ["2017-2026", "2005-2026"]):
        g = c[c["期间"] == period].sort_values("组")
        x = range(len(g))
        avg = g["实际收益均值"] * 100
        med = g["实际收益中位"] * 100
        ax.bar([i - 0.18 for i in x], avg, width=0.36, color=BLUE, label="平均")
        ax.bar([i + 0.18 for i in x], med, width=0.36, color=ORANGE, label="中位")
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{r.组}\n(n={int(r.笔数)})" for r in g.itertuples()], fontsize=8)
        for i, r in enumerate(g.itertuples()):
            ax.text(i - 0.18, r.实际收益均值 * 100 + 0.5, f"{r.实际收益均值 * 100:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=BLUE)
            ax.text(i + 0.18, r.实际收益中位 * 100 + 0.5, f"{r.实际收益中位 * 100:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=ORANGE)
        ax.set_title(title, fontsize=12, loc="left", pad=8)
        ax.set_ylim(-6, 23)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("实际收益（%）")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("T+3 开盘破信号日低点：破位本身不是结论，5 日收回才是", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "01_破位分组收益.png", bbox_inches="tight")
    plt.close(fig)

    # 主表
    headers = ["期间", "组", "笔数", "实际均值", "中位", "胜率", "15日", "最大浮亏", "浮亏≤-10%", "止盈20%"]
    rows = []
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        for grp in order:
            g = c[(c["期间"] == period) & (c["组"] == grp)]
            if g.empty:
                continue
            r = g.iloc[0]
            rows.append([period, grp, str(int(r["笔数"])), pct(r["实际收益均值"]),
                         pct(r["实际收益中位"]), pct(r["胜率"]), pct(r["十五日"]),
                         pct(r["最大浮亏均值"]), pct(r["浮亏10"]), pct(r["止盈20"])])
    main_html = html_table(headers, rows)
    main_md = md_table(headers, rows)

    # A/B/C 复核
    ph = ["期间", "池", "未破均值", "未破中位", "破位均值", "破位中位", "未破浮亏", "破位浮亏"]
    prows = []
    for period in ["2017-2026", "2005-2026"]:
        for label in ["A_仅当前上市", "B_含退市_无ST过滤", "C_含退市_排除风险警示"]:
            a = pool[(pool["期间"] == period) & (pool["股票池"] == label) & (pool["组"] == "未破")]
            b = pool[(pool["期间"] == period) & (pool["股票池"] == label) & (pool["组"] == "破位")]
            if a.empty or b.empty:
                continue
            a = a.iloc[0]; b = b.iloc[0]
            prows.append([period, label.split("_")[0], pct(a["实际收益均值"]), pct(a["实际收益中位"]),
                          pct(b["实际收益均值"]), pct(b["实际收益中位"]),
                          pct(a["最大浮亏均值"]), pct(b["最大浮亏均值"])])
    pool_html = html_table(ph, prows)
    pool_md = md_table(ph, prows)

    # 个案
    cb = detail[(detail["股票池"].str.startswith("C")) & (detail["破位"])]
    recent = cb[cb["期间"] == "2017-2026"]
    best = recent.nlargest(3, "实际收益")
    worst = recent.nsmallest(4, "实际收益")
    exit_map = {
        "max_holding": "15日到期",
        "pending_max_holding_limit_down": "到期遇跌停，顺延",
        "pending_max_holding_no_bar": "到期停牌，顺延",
        "take_profit_open": "跳空止盈",
        "take_profit_intraday": "盘中止盈",
        "eod": "期末持仓",
    }
    case_rows = []
    for r in pd.concat([worst, best]).itertuples():
        case_rows.append([
            f"{r.代码.split('.')[-1]} {names.get(r.代码, '')}", f"{r.信号日.date()} → {r.买入日.date()}",
            pct(r.实际收益), pct(r.最大浮亏), pct(r.最大浮盈),
            "是" if r.五日收盘收回 else "否",
            exit_map.get(r.退出, r.退出),
        ])
    case_html = html_table(["股票", "信号 → 买入", "实际收益", "最大浮亏", "最大浮盈", "5日收回", "退出"], case_rows)
    case_md = md_table(["股票", "信号 → 买入", "实际收益", "最大浮亏", "最大浮盈", "5日收回", "退出"], case_rows)

    # 假破位表
    fake = detail[(detail["期间"].isin(["2005-2016", "2005-2026"])) &
                  (detail["股票池"].str.startswith("C")) &
                  (detail["破位"] != detail["破位_实际"])]
    fake = fake.drop_duplicates(subset=["代码", "信号日"])
    fake_rows = []
    for r in fake.itertuples():
        fake_rows.append([f"{r.代码.split('.')[-1]}", f"{r.信号日.date()}", f"{r.信号最低:.2f}",
                          f"{r.买入开盘:.2f}", f"{r.信号最低_复权:.2f}", f"{r.买入开盘_复权:.2f}",
                          pct(r.实际收益)])
    fake_html = html_table(["股票", "信号日", "实际低点", "实际开盘", "复权低点", "复权开盘", "实际收益"], fake_rows)
    fake_md = md_table(["股票", "信号日", "实际低点", "实际开盘", "复权低点", "复权开盘", "实际收益"], fake_rows)

    # 关键数字
    r17 = c[(c["期间"] == "2017-2026") & (c["组"] == "未破")].iloc[0]
    b17 = c[(c["期间"] == "2017-2026") & (c["组"] == "破位")].iloc[0]
    rec17 = c[(c["期间"] == "2017-2026") & (c["组"] == "破位+5日收盘收回")].iloc[0]
    no17 = c[(c["期间"] == "2017-2026") & (c["组"] == "破位+5日未收回")].iloc[0]
    bfull = c[(c["期间"] == "2005-2026") & (c["组"] == "破位")].iloc[0]
    rfull = c[(c["期间"] == "2005-2026") & (c["组"] == "未破")].iloc[0]
    recfull = c[(c["期间"] == "2005-2026") & (c["组"] == "破位+5日收盘收回")].iloc[0]
    nofull = c[(c["期间"] == "2005-2026") & (c["组"] == "破位+5日未收回")].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T+3 开盘破信号日低点</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<div class="kicker">预入场条件 · 信号低点</div>
<h1>T+3 开盘跌破信号日最低价，后来怎样？</h1>
<p class="sub">C 池前二（含退市股、信号日排除风险警示），2×50%；T 日收盘出信号，T+3 开盘买入，+20% 盘中止盈，
15 个交易日收盘卖出。破位按后复权价判定，避免除权除息造成的假破位。</p>
<div class="banner">破位本身不是坏信号，也不是好信号：中位数更高、左尾更肥。真正的分水岭是
5 日内能否收盘收回信号低点。</div>
</header>

<div class="cards">
<div class="card blue"><div class="label">2017-2026 破位占比</div><div class="big">26/101</div><div class="note">25.7%，T+3 开盘低于信号日低点</div></div>
<div class="card"><div class="label">破位 vs 未破 平均</div><div class="big">5.6% / 6.5%</div><div class="note">中位 8.2% / 7.0%，破位反而更高</div></div>
<div class="card green"><div class="label">破位 + 5日收盘收回</div><div class="big">+12.7%</div><div class="note">14 笔，胜率 92.9%，中位 19.7%</div></div>
<div class="card red"><div class="label">破位 + 5日未收回</div><div class="big">-2.7%</div><div class="note">12 笔，胜率 41.7%，中位 -3.0%</div></div>
</div>

<section>
<h2>一、口径与假破位</h2>
<p>信号日 T 的最低点、T+3 的开盘价，都用后复权价比较：<b>T+3 开盘 &lt; T 日最低</b> 记为破位。
这样不会把除权除息造成的价格跳空误判成破位。全历史 C 池实际价口径有 54 笔“破位”，复权口径 52 笔；
差的 2 笔是 2008 年的 600655 和 2016 年的 000062，都是除权/分红导致的假破位，已从主口径剔除。</p>
<div class="tablewrap">{fake_html}</div>
<p class="small">600655 的 2008-08-08 实际价从 18.35 跳到 15.21，是除权；复权后 1328.0 高于信号低点 1311.9，不是真破位。
如果只看实际价，会把这笔 -29.5% 的亏损误算进“破位”组。</p>
</section>

<section>
<h2>二、主表：破位不是单调的坏消息</h2>
<p>2017-2026 年，破位 26 笔的平均收益 <b>{pct(b17['实际收益均值'])}</b>，未破 75 笔 <b>{pct(r17['实际收益均值'])}</b>；
但破位的中位数 <b>{pct(b17['实际收益中位'])}</b> 高于未破 <b>{pct(r17['实际收益中位'])}</b>。
破位的最大浮亏更深（{pct(b17['最大浮亏均值'])} 对 {pct(r17['最大浮亏均值'])}），
浮亏 ≤ -10% 的比例更高（{pct(b17['浮亏10'])} 对 {pct(r17['浮亏10'])}）。</p>
<p>全历史也一样：破位平均 {pct(bfull['实际收益均值'])} 对未破 {pct(rfull['实际收益均值'])}，几乎打平；
中位 {pct(bfull['实际收益中位'])} 对 {pct(rfull['实际收益中位'])}，破位更高；最大浮亏
{pct(bfull['最大浮亏均值'])} 对 {pct(rfull['最大浮亏均值'])}，破位更深。</p>
<div class="tablewrap">{main_html}</div>
<p class="small">“实际均值/中位”是策略实际成交后的收益（含止盈/到期和成本）；“15日”是买入后第 15 个交易日收盘的远期收益；
“最大浮亏”是持有期最低价相对买入开盘价；“止盈20%”是持有期最高价是否触及 +20%。</p>
</section>

<section>
<h2>三、真正的分水岭：5 日内能否收盘收回</h2>
<p>口径先说清楚：以买入日（信号日 T+3）为第 1 天，买入日加后续 5 个交易日共 6 个收盘价，
最晚到信号日 T+8；其中任一天收盘价回到信号日低点之上，记为“收回”。
这是“买入后的 5 日”，不是相对信号日的 T+5。2017-2026 年，
26 笔破位里有 14 笔收回：平均 <b>{pct(rec17['实际收益均值'])}</b>，中位 <b>{pct(rec17['实际收益中位'])}</b>，
胜率 <b>{pct(rec17['胜率'])}</b>，最大浮亏只有 {pct(rec17['最大浮亏均值'])}。
另外 12 笔没收回：平均 <b>{pct(no17['实际收益均值'])}</b>，中位 {pct(no17['实际收益中位'])}，
胜率 {pct(no17['胜率'])}，最大浮亏 {pct(no17['最大浮亏均值'])}。</p>
<p>全历史同样分得很开：收回 25 笔平均 {pct(recfull['实际收益均值'])}、中位 {pct(recfull['实际收益中位'])}、
胜率 {pct(recfull['胜率'])}；未收回 27 笔平均 {pct(nofull['实际收益均值'])}、中位
{pct(nofull['实际收益中位'])}、胜率 {pct(nofull['胜率'])}。这不是“破位一定差”，而是“破位后收不回来才差”。</p>
<figure>{img(FIG / '01_破位分组收益.png', '破位分组收益')}
<figcaption>蓝条是平均，橙条是中位。破位组本身不差；拆成“收回/未收回”后差距才出来。</figcaption></figure>
<p class="small">换成更早的截止点，方向不变、差距变窄：截止到信号日 T+5（买入后第 3 个交易日），
2017-2026 收回 9 笔平均 13.9%、未收回 17 笔平均 1.1%；全历史收回 18 笔平均 11.0%、
未收回 34 笔平均 0.3%。截止点越早，“收回”的样本越少，分层越弱。</p>
</section>

<section>
<h2>四、但它不是确定性规则</h2>
<p>反例两边都有：2026-07-16 的 603778 破位后 5 日没有收盘收回，最后却 +20.9% 跳空止盈；
2025-01-02 的 000595 破位后 5 日收回了，但到期撞上跌停卖不出去，最终 -16.9%。
所以“5 日收回”是一个平均意义上的分层，不是每笔都成立的开关。</p>
<div class="tablewrap">{case_html}</div>
<p class="small">上表是 2017-2026 破位组里收盘最差 4 笔和最好 3 笔。</p>
</section>

<section>
<h2>五、A/B/C 复核：不要把它当成稳定 alpha</h2>
<p>同样的“破位 vs 未破”在三个池子里方向并不一致。2017-2026 年，A 池破位平均 8.6% 对未破 6.9%，
B 池 7.5% 对 5.0%，都更高；只有 C 池是 5.6% 对 6.5%，略低。全历史 A/B 也是破位更高，C 几乎打平。
但三个池子、两个期间，<b>破位的中位数都更高，最大浮亏都更深</b>。这说明破位更像是“波动更大”，
而不是“必然更差”。</p>
<div class="tablewrap">{pool_html}</div>
<p>因此，“T+3 开盘破信号低点”本身不适合当一刀切的入场否决条件；它会把很多后来的大赢家一起砍掉。
如果要用，应该用“收回确认”或“不收回就减仓”这种后验规则，并且单独回测净值，而不是在信号日直接过滤。</p>
</section>

<section>
<h2>六、限制</h2>
<ul>
<li>破位样本不大：2017-2026 只有 26 笔，2005-2026 只有 52 笔；收回/未收回再拆一半，单组只有 10-27 笔。</li>
<li>“5 日收回”是买入后才知道的，不能当 T+3 的预入场条件；把它改成“等收回再买”会改变成交价和错过 V 型反转，必须单独回测。</li>
<li>收回窗口是买入日起 6 个收盘价（T+3~T+8）；截止点提前到 T+5 时方向不变但差距缩小（全历史收回 18 笔 11.0% 对未收回 34 笔 0.3%），结论依赖窗口选择。</li>
<li>最大浮亏按后复权日线最低价计算，未建模盘中止损的滑点、封单和排队；实际成交可能更差。</li>
<li>破位深度没有单调关系，且样本很小，不建议用 -3%/-5% 之类的阈值做过滤器。</li>
<li>2024-2026 已在旧研究中使用，不是纯净留出；历史回测不代表未来，不构成投资建议。</li>
</ul>
</section>

<footer>
口径：C 池 = 全主板含退市股、信号日 is_st=1 不买；成交价不复权，信号/收益后复权；破位用后复权价判定；
成本含 0.1%/边滑点、佣金、过户费、卖出印花税。
复现：<code>02_代码/analyze_break_low.py</code> 生成统计与明细，<code>02_代码/make_break_low_report.py</code> 生成本页。
</footer>
</div></body></html>"""
    (OUT / "破信号低点.html").write_text(html, encoding="utf-8")

    md = f"""# T+3 开盘跌破信号日最低价

C 池前二（含退市股、信号日排除风险警示），2×50%；T 日收盘信号，T+3 开盘买入，+20% 盘中止盈，
15 个交易日收盘卖出。破位按后复权价判定，避免除权除息造成的假破位。

## 短答

破位本身不是坏信号，也不是好信号：中位数更高、左尾更肥。真正的分水岭是 5 日内能否收盘收回信号低点。

“5 日收回”以买入日（信号日 T+3）为第 1 天，含买入日共 6 个收盘价，最晚到信号日 T+8；不是相对信号日的 T+5。

- 2017-2026：破位 26/101（25.7%），平均 {pct(b17['实际收益均值'])} 对未破 {pct(r17['实际收益均值'])}，
  中位 {pct(b17['实际收益中位'])} 对 {pct(r17['实际收益中位'])}，最大浮亏
  {pct(b17['最大浮亏均值'])} 对 {pct(r17['最大浮亏均值'])}。
- 全历史：破位 52 笔平均 {pct(bfull['实际收益均值'])} 对未破 {pct(rfull['实际收益均值'])}，几乎打平；
  中位 {pct(bfull['实际收益中位'])} 对 {pct(rfull['实际收益中位'])}，破位更高；最大浮亏更深。
- 2017-2026 破位 + 5 日收盘收回：14 笔，平均 {pct(rec17['实际收益均值'])}、中位
  {pct(rec17['实际收益中位'])}、胜率 {pct(rec17['胜率'])}；未收回 12 笔，平均
  {pct(no17['实际收益均值'])}、中位 {pct(no17['实际收益中位'])}、胜率 {pct(no17['胜率'])}。

## 主表

{main_md}

## 假破位

全历史 C 池实际价口径 54 笔“破位”，复权口径 52 笔。差的是 2008 年的 600655 和 2016 年的 000062，
都是除权/分红导致的假破位。600655 实际价从 18.35 跳到 15.21，复权后 1328.0 高于信号低点 1311.9。

{fake_md}

## 个案

{case_md}

反例两边都有：2026-07-16 的 603778 破位后 5 日没有收盘收回，最后 +20.9% 跳空止盈；
2025-01-02 的 000595 破位后 5 日收回，但到期撞上跌停卖不出去，最终 -16.9%。

## A/B/C 复核

{pool_md}

2017-2026 年 A 池破位平均 8.6% 对 6.9%，B 池 7.5% 对 5.0%，都更高；只有 C 池 5.6% 对 6.5%，略低。
全历史 A/B 也是破位更高，C 几乎打平。但三个池子、两个期间，破位中位数都更高、最大浮亏都更深。
所以“破位”更像是“波动更大”，不是“必然更差”，不适合当一刀切的入场否决条件。

## 限制

样本小（2017-2026 破位 26 笔，2005-2026 52 笔）；“5 日收回”是买入后才知道的，不能当预入场条件，
改成“等收回再买”要单独回测；收回窗口是买入日起 6 个收盘价（T+3~T+8），截止点提前到 T+5 方向不变但差距缩小
（全历史 11.0% 对 0.3%）；最大浮亏按后复权日线最低价，未建模滑点/封单；破位深度没有单调关系，
不建议设阈值；历史回测不代表未来，不构成投资建议。

图表与交互页：`破信号低点.html`；统计与明细：`破信号低点_统计.csv`、`破信号低点_分池统计.csv`、
`破信号低点_交易明细.csv`；复现：`02_代码/analyze_break_low.py`、`02_代码/make_break_low_report.py`。
"""
    (OUT / "破信号低点_结论.md").write_text(md, encoding="utf-8")

    print(main_md)
    print(pool_md)
    print(case_md)


if __name__ == "__main__":
    main()
