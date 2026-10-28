#!/usr/bin/env python3
"""生成“下跌斜率：越陡越好还是越缓越好”的图表、HTML 与摘要。"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "11_下跌斜率_对数坐标"
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
PURPLE = "#6b4c9a"


def pct(x, d=1, sign=False):
    if pd.isna(x):
        return "—"
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return s


def pp(x, d=2):
    if pd.isna(x):
        return "—"
    return f"{x * 100:+.{d}f}pp"


def money(x, d=0):
    if pd.isna(x):
        return "—"
    return f"{x:+,.{d}f}"


def num(x, d=2):
    if pd.isna(x):
        return "—"
    return f"{x:.{d}f}"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64(path)}" alt="{alt}">'


def html_table(headers, rows, cls="data"):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
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
.sub{color:var(--muted);font-size:15px;max-width:880px}
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
table.data{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
table.data th{background:#eef2f6;text-align:right;padding:7px 6px;font-weight:600;color:#33414f;white-space:nowrap}
table.data th:first-child,table.data td:first-child{text-align:left}
table.data td{padding:7px 6px;border-top:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.data tr:hover td{background:#fafbfd}.tablewrap{overflow:auto;border-radius:10px}
details{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 14px}
summary{cursor:pointer;color:var(--blue);font-weight:600}
ul{margin:8px 0 12px 20px;padding:0}li{margin:5px 0}
footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px;margin-top:42px}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}.wrap{padding:0 16px 48px}}
"""


def main():
    b = pd.read_csv(OUT / "下跌斜率_分桶.csv")
    reg = pd.read_csv(OUT / "下跌斜率_回归.csv")
    within = pd.read_csv(OUT / "下跌斜率_同日对比.csv")
    corr = pd.read_csv(OUT / "下跌斜率_相关性.csv")
    tr = pd.read_csv(OUT / "下跌斜率_逐笔.csv", dtype={"代码": str, "代码6位": str},
                     parse_dates=["信号日", "高点日", "买入日"])

    q = b[b["样本"] == "前二_五分位"].copy()
    d10 = b[b["样本"] == "前二_十分位"].copy()
    a100 = b[b["样本"] == "100+全部候选_五分位"].copy()
    act = b[b["样本"] == "100+实际买入_三分位"].copy()

    # 同日两只前二：陡 vs 缓（用未四舍五入的全信号表，避免近似并列时排序互换）
    allsig = pd.read_csv(OUT / "下跌斜率_全信号.csv", parse_dates=["signal_date"])
    allsig = allsig[allsig.status == "filled"]
    t2 = allsig[allsig["rank"] <= 2].copy()
    t2["dec"] = -t2.slope250
    rows = []
    for day, g in t2.groupby("signal_date"):
        if len(g) != 2:
            continue
        s = g.sort_values("dec").iloc[-1]  # 日均跌幅大 = 陡
        f = g.sort_values("dec").iloc[0]
        rows.append(dict(date=day, s_r15=s["r15"], f_r15=f["r15"],
                         s_pnl=s["pnl"], f_pnl=f["pnl"],
                         s_slope=s["dec"], f_slope=f["dec"]))
    pair = pd.DataFrame(rows)
    pr15 = pair[["s_r15", "f_r15"]].dropna()
    ppnl = pair[["s_pnl", "f_pnl"]].dropna()
    r15_diff = (pr15.s_r15 - pr15.f_r15)
    pnl_diff = (ppnl.s_pnl - ppnl.f_pnl)
    r15_t = ttest_rel(pr15.s_r15, pr15.f_r15).statistic
    pnl_t = ttest_rel(ppnl.s_pnl, ppnl.f_pnl).statistic

    # ---------- 图 1：前二五分位 ----------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    x = np.arange(len(q))
    ax.bar(x, q["十五日收益"] * 100, color=BLUE, width=0.58, alpha=0.9)
    ax.plot(x, q["十五日中位"] * 100, color="#111", marker="o", lw=1.2, ms=4,
            label="中位")
    for i, r in enumerate(q.itertuples()):
        ax.text(i, r.十五日收益 * 100 + 0.35, f"{r.十五日收益 * 100:.1f}%",
                ha="center", fontsize=8.5, color=BLUE, fontweight="600")
    ax.set_xticks(x, q["组"], fontsize=9)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("15日收益（%）")
    ax.set_title("前二：15日持有收益", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax2.bar(x, q["平均盈亏"], color=GREEN, width=0.58, alpha=0.85)
    ax2b = ax2.twinx()
    ax2b.plot(x, q["止盈20率"] * 100, color=ORANGE, marker="o", lw=1.8, ms=5,
              label="止盈20率")
    for i, r in enumerate(q.itertuples()):
        ax2.text(i, r.平均盈亏 / 2, f"{r.平均盈亏:,.0f}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="600")
    ax2.set_xticks(x, q["组"], fontsize=9)
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_ylabel("1w 平均盈亏（元）")
    ax2b.set_ylabel("止盈20率（%）")
    ax2b.set_ylim(0, 55)
    ax2.set_title("前二：1w 策略盈亏与止盈率", fontsize=12, loc="left")
    ax2b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "01_前二_斜率五分位.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 2：100+ 全样本 vs 同日 ----------
    # 同日 Q1/Q2/Q5 for 100+
    def within_q(x, nmin=10):
        rec = []
        for day, g in x.groupby("signal_date"):
            if len(g) < nmin:
                continue
            g = g.sort_values("slope250")
            k = max(1, int(round(len(g) * 0.2)))
            rec.append(dict(date=day, q1=g.iloc[:k].r15.mean(),
                            q2=g.iloc[k:2*k].r15.mean(), q5=g.iloc[-k:].r15.mean()))
        return pd.DataFrame(rec)

    cnt = allsig.groupby("signal_date").size()
    a100sig = allsig[allsig.signal_date.isin(set(cnt[cnt >= 100].index))]
    w100 = within_q(a100sig)
    cats = ["Q1最陡", "Q2", "Q5最缓"]
    pooled = {k: a100.loc[a100["组"] == k, "十五日收益"].iloc[0] * 100 for k in cats}
    same = {"Q1最陡": w100.q1.mean() * 100, "Q2": w100.q2.mean() * 100,
            "Q5最缓": w100.q5.mean() * 100}
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    xx = np.arange(len(cats))
    v1 = [pooled[c] for c in cats]
    v2 = [same[c] for c in cats]
    b1 = ax.bar(xx - 0.19, v1, width=0.36, color=GREY, alpha=0.55, label="全样本分组")
    b2 = ax.bar(xx + 0.19, v2, width=0.36, color=BLUE, alpha=0.9, label="同一天内分组")
    for bars in [b1, b2]:
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + (0.12 if h >= 0 else -0.35),
                    f"{h:+.1f}%", ha="center", fontsize=8.5,
                    va="bottom" if h >= 0 else "top")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(xx, cats)
    ax.set_ylabel("15日收益（%）")
    ax.set_title("100+ 日：全样本 Q2 最好，同一天内 Q1 反而略高于 Q2", fontsize=12, loc="left")
    ax.set_ylim(min(0, min(v1 + v2)) * 1.35, max(v1 + v2) * 1.45)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.text(0.02, 0.96, "Q1-Q2 同日差 +0.70pp（t=1.00）；Q1-Q5 +2.73pp（t=3.49）",
            transform=ax.transAxes, fontsize=8, color=GREY, va="top")
    fig.tight_layout()
    fig.savefig(FIG / "02_100信号日_全样本与同日.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 3：固定效应 ----------
    rs = reg[reg["变量"] == "斜率(陡为正)"].copy()
    rs["结果"] = np.where(rs["检验"].str.contains("r15"), "15日收益", "1w盈亏")
    order = [("15日收益", "无"), ("15日收益", "year"), ("15日收益", "ym"),
             ("1w盈亏", "无"), ("1w盈亏", "year"), ("1w盈亏", "ym")]
    labels = ["无", "年固定", "月固定"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for j, (metric, axx) in enumerate([("15日收益", ax), ("1w盈亏", ax2)]):
        vals, errs, ts = [], [], []
        for _, fe in order:
            if _ != metric:
                continue
            row = rs[(rs["结果"] == metric) & (rs["模型"] == fe)].iloc[0]
            if metric == "15日收益":
                vals.append(row["系数"] * 100)
                errs.append(1.96 * row["标准误"] * 100)
            else:
                vals.append(row["系数"])
                errs.append(1.96 * row["标准误"])
            ts.append(row["t值"])
        colors = [BLUE, "#5a9bd4", "#a8c8e8"] if metric == "15日收益" else [GREEN, "#6fb08a", "#b8d8c5"]
        bars = axx.bar(labels, vals, color=colors, width=0.58)
        axx.errorbar(labels, vals, yerr=errs, fmt="none", ecolor="#333", capsize=3, lw=1)
        for i, (v, t) in enumerate(zip(vals, ts)):
            axx.text(i, v + (errs[i] + 0.15 if v >= 0 else -errs[i] - 0.15),
                     f"{v:+.2f}" + ("pp" if metric == "15日收益" else "元") + f"\nt={t:.2f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        axx.axhline(0, color="#333", lw=0.8)
        lo = min([v - e for v, e in zip(vals, errs)] + [0])
        hi = max([v + e for v, e in zip(vals, errs)] + [0])
        pad = (hi - lo) * 0.25
        axx.set_ylim(lo - pad, hi + pad * 1.6)
        axx.set_title(f"{metric}：斜率系数（每 1%/日）", fontsize=12, loc="left")
        axx.set_ylabel("系数" + ("（pp）" if metric == "15日收益" else "（元）"))
    fig.suptitle("斜率的解释力：加月份固定效应后接近 0", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "03_固定效应.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 4：同日两只前二 ----------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    labels = ["更陡的那只", "更缓的那只"]
    vals = [pr15.s_r15.mean() * 100, pr15.f_r15.mean() * 100]
    bars = ax.bar(labels, vals, color=[BLUE, GREY], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.2f}%", ha="center", fontsize=9, fontweight="600")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel("15日收益（%）")
    ax.set_title(f"同日两只前二：15日收益（n={len(pr15)} 天）", fontsize=11, loc="left")
    ax.text(0.5, 0.93, f"差 {r15_diff.mean()*100:+.2f}pp，t={r15_t:.2f}",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=GREY)

    vals2 = [ppnl.s_pnl.mean(), ppnl.f_pnl.mean()]
    ax2.bar(labels, vals2, color=[GREEN, GREY], width=0.55)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 12, f"{v:,.0f}", ha="center", fontsize=9, fontweight="600")
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.set_ylim(0, max(vals2) * 1.35)
    ax2.set_ylabel("1w 平均盈亏（元）")
    ax2.set_title(f"同日两只前二：1w 盈亏（n={len(ppnl)} 天）", fontsize=11, loc="left")
    ax2.text(0.5, 0.93, f"差 {pnl_diff.mean():+.0f} 元，t={pnl_t:.2f}",
             transform=ax2.transAxes, ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(FIG / "04_同日两只前二.png", bbox_inches="tight")
    plt.close(fig)

    ex_steep = tr[(tr["代码"] == "sh.600559") & (tr["信号日"] == pd.Timestamp("2008-04-01"))].iloc[0]
    ex_flat = tr[(tr["代码"] == "sz.000022") & (tr["信号日"] == pd.Timestamp("2018-09-10"))].iloc[0]
    ex_susp = tr[(tr["代码"] == "sz.000908") & (tr["信号日"] == pd.Timestamp("2008-06-13"))].iloc[0]

    # ---------- 图 5：两个例子的对数路径 ----------
    mktcal = pd.DatetimeIndex(sorted(pd.read_parquet(
        ROOT / "01_数据/market_daily.parquet", columns=["date"]).date.astype("datetime64[ns]")))
    cidx = {d: i for i, d in enumerate(mktcal)}
    bars = pd.read_parquet(
        ROOT / "01_数据/bars.parquet", columns=["code", "date", "hfq_close"],
        filters=[("code", "in", ["sh.600559", "sz.000022"])])
    bars["date"] = pd.to_datetime(bars.date)
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    for row, color, label in [(ex_steep, BLUE, "600559 老白干酒"),
                              (ex_flat, GREY, "000022 深赤湾A")]:
        g = bars[(bars.code == row.代码) & (bars.date >= row.高点日) &
                 (bars.date <= row.信号日)].sort_values("date")
        g = g.dropna(subset=["hfq_close"])
        if g.empty:
            continue
        x = [cidx[d] - cidx[row.高点日] for d in g.date if d in cidx]
        y = [v / row["250日高点"] * 100 for v in g.hfq_close]
        ax.plot(x, y, color=color, lw=1.8, label=label)
        ax.scatter([x[-1]], [y[-1]], color=color, s=22, zorder=5)
        ax.annotate(f"{row.名称}：{row.下跌交易日}日 · {row['日均跌幅%每日']:.2f}%/日\n"
                    f"15日 {row.十五日收益*100:+.1f}% · 1w {row.策略盈亏:+,.0f}元",
                    (x[-1], y[-1]), xytext=(-10, 80) if row.代码 == "sz.000022" else (-8, 12),
                    textcoords="offset points",
                    ha="right" if row.代码 == "sz.000022" else "left",
                    fontsize=8, color=color)
    ax.set_yscale("log")
    ax.set_yticks([30, 40, 50, 60, 80, 100])
    ax.set_yticklabels(["30", "40", "50", "60", "80", "100"])
    ax.set_xlabel("从 250 日高点起的市场交易日")
    ax.set_ylabel("后复权收盘 / 250日高点（对数）")
    ax.set_title("两个例子：同样接近 -45%，一个 11 天，一个 302 天", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "05_两个例子_对数路径.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- HTML 表格 ----------
    qh = ["组", "笔数", "日均跌幅", "回撤", "下跌天数", "15日收益", "15日胜率",
          "止盈20率", "平均盈亏", "最大浮亏"]
    qrows = []
    for r in q.itertuples():
        qrows.append([r.组, str(int(r.笔数)), f"{r.日均跌幅:.2f}%/日", pct(r.回撤均值, 1),
                      f"{r.下跌天数:.0f}日", pct(r.十五日收益, 1, True),
                      pct(r.十五日胜率, 1), pct(r.止盈20率, 1),
                      money(r.平均盈亏), pct(r.最大浮亏, 1)])
    q_html = html_table(qh, qrows)
    q_md = md_table(qh, qrows)

    dh = ["十分位", "笔数", "日均跌幅", "回撤", "下跌天数", "15日收益", "15日胜率",
          "止盈20率", "平均盈亏", "最大浮亏"]
    drows = []
    for r in d10.itertuples():
        drows.append([r.组, str(int(r.笔数)), f"{r.日均跌幅:.2f}%/日", pct(r.回撤均值, 1),
                      f"{r.下跌天数:.0f}日", pct(r.十五日收益, 1, True),
                      pct(r.十五日胜率, 1), pct(r.止盈20率, 1),
                      money(r.平均盈亏), pct(r.最大浮亏, 1)])
    d_html = html_table(dh, drows)
    d_md = md_table(dh, drows)

    ah = ["组", "笔数", "日均跌幅", "回撤", "下跌天数", "15日收益", "15日胜率",
          "止盈20率", "最大浮亏"]
    arows = []
    for r in a100.itertuples():
        arows.append([r.组, str(int(r.笔数)), f"{r.日均跌幅:.2f}%/日", pct(r.回撤均值, 1),
                      f"{r.下跌天数:.0f}日", pct(r.十五日收益, 1, True),
                      pct(r.十五日胜率, 1), pct(r.止盈20率, 1), pct(r.最大浮亏, 1)])
    a_html = html_table(ah, arows)
    a_md = md_table(ah, arows)

    ch = ["组", "笔数", "日均跌幅", "回撤", "15日收益", "策略收益", "平均盈亏",
          "胜率", "最差", "最好"]
    crows = []
    for r in act.itertuples():
        crows.append([r.组, str(int(r.笔数)), f"{r.日均跌幅:.2f}%/日", pct(r.回撤均值, 1),
                      pct(r.十五日收益, 1, True), pct(r.策略收益, 1, True),
                      money(r.平均盈亏), pct(r.盈亏胜率, 1),
                      money(r.最差), money(r.最好)])
    c_html = html_table(ch, crows)
    c_md = md_table(ch, crows)

    # 2026-10-23 两个信号在“当天”和“全样本前二”里的相对位置
    sig_all = pd.read_csv(OUT / "下跌斜率_全信号.csv", parse_dates=["signal_date"])
    d23 = sig_all[sig_all.signal_date == "2026-10-23"].copy().sort_values("slope250").reset_index(drop=True)
    d23["slope_rank"] = np.arange(1, len(d23) + 1)
    d23["slope_pct"] = (d23.slope_rank - 1) / (len(d23) - 1) * 100
    d23["Q"] = pd.qcut(d23.slope_rank, 5, labels=["Q1最陡", "Q2", "Q3", "Q4", "Q5最缓"])
    d23["D"] = pd.qcut(d23.slope_rank, 10, labels=[f"D{i}" for i in range(1, 11)])
    t2_all = sig_all[(sig_all.status == "filled") & (sig_all["rank"] <= 2)].copy()
    loc_rows = []
    for code in ["sh.605376", "sh.603778"]:
        r = d23[d23.code == code].iloc[0]
        slope = float(r.slope250)
        ar = int((t2_all.slope250 < slope).sum() + 1)
        ap = (ar - 1) / (len(t2_all) - 1) * 100
        aq = int(np.ceil(ar / (len(t2_all) / 5)))
        ad = int(np.ceil(ar / (len(t2_all) / 10)))
        loc_rows.append(dict(
            代码=code, 代码6位=code.split(".")[-1], 名称=r["name"], 信号日=r.signal_date.date(),
            回撤排名=int(r["rank"]), 日均跌幅=-slope, 回撤深度=float(r.depth250),
            当天斜率排名=int(r.slope_rank), 当天候选数=len(d23), 当天前百分比=float(r.slope_pct),
            当天五分位=str(r.Q), 当天十分位=str(r.D),
            全样本前二排名=ar, 全样本前二前百分比=ap,
            全样本五分位=f"Q{aq}", 全样本十分位=f"D{ad}",
        ))
    loc23 = pd.DataFrame(loc_rows)
    loc23.to_csv(OUT / "下跌斜率_20261023_定位.csv", index=False, encoding="utf-8-sig")
    lh = ["代码", "名称", "回撤排名", "日均跌幅", "回撤", "当天斜率排名", "当天前%", "当天组", "全样本前二组"]
    lrows = []
    for r in loc23.itertuples():
        lrows.append([r.代码6位, r.名称, str(r.回撤排名), f"{r.日均跌幅:.2f}%/日",
                      pct(r.回撤深度, 1), f"{r.当天斜率排名}/{r.当天候选数}",
                      f"{r.当天前百分比:.1f}%", f"{r.当天五分位}/{r.当天十分位}",
                      f"{r.全样本五分位}/{r.全样本十分位}（前 {r.全样本前二前百分比:.1f}%）"])
    loc_html = html_table(lh, lrows)
    loc_md = md_table(lh, lrows)

    # 回归表
    rh = ["结果", "模型", "系数（每 1%/日）", "t 值", "95% 区间", "样本/天数"]
    rrows = []
    for metric in ["15日收益", "1w盈亏"]:
        for fe, lab in [("无", "无固定效应"), ("year", "年固定效应"), ("ym", "月固定效应")]:
            row = rs[(rs["结果"] == metric) & (rs["模型"] == fe)].iloc[0]
            if metric == "15日收益":
                coef = row["系数"] * 100
                lo, hi = (row["系数"] - 1.96 * row["标准误"]) * 100, (row["系数"] + 1.96 * row["标准误"]) * 100
                unit = "pp"
            else:
                coef = row["系数"]
                lo, hi = row["系数"] - 1.96 * row["标准误"], row["系数"] + 1.96 * row["标准误"]
                unit = "元"
            rrows.append([metric, lab, f"{coef:+.2f} {unit}", f"{row['t值']:.2f}",
                          f"[{lo:+.2f}, {hi:+.2f}] {unit}",
                          f"{int(row['样本'])} / {int(row['天数'])}"])
    r_html = html_table(rh, rrows)
    r_md = md_table(rh, rrows)

    # 同日表
    wh = ["样本", "对比", "平均差", "t 值", "胜出天数", "天数"]
    wrows = []
    for r in within[within["对比"].isin(["Q1-Q5", "Q1-Q2", "Q2-Q5"])].itertuples():
        wrows.append([r.样本, r.对比, pp(r.平均差), f"{r.t值:.2f}",
                      pct(r.胜出天数占比, 1) if pd.notna(r.胜出天数占比) else "—",
                      str(int(r.天数))])
    w_html = html_table(wh, wrows)
    w_md = md_table(wh, wrows)

    # 相关性表
    xh = ["斜率口径", "5日收益", "10日收益", "15日收益", "策略收益"]
    xrows = []
    for xname in ["250日高点_信号日", "250日高点_买入日", "60日高点_信号日",
                  "60日高点_买入日", "20日高点_信号日", "20日高点_买入日"]:
        vals = []
        for y in ["5日收益", "10日收益", "15日收益", "策略收益"]:
            z = corr[(corr["斜率口径"] == xname) & (corr["结果"] == y)]
            vals.append(f"{z['相关系数'].iloc[0]:.3f}" if len(z) else "—")
        xrows.append([xname] + vals)
    x_html = html_table(xh, xrows)
    x_md = md_table(xh, xrows)

    # 数字
    q1, q5 = q.iloc[0], q.iloc[-1]
    steep = act[act["组"] == "陡"].iloc[0]
    flatc = act[act["组"] == "缓"].iloc[0]
    nofe = rs[(rs["结果"] == "15日收益") & (rs["模型"] == "无")].iloc[0]
    ymfe = rs[(rs["结果"] == "15日收益") & (rs["模型"] == "ym")].iloc[0]
    pnofe = rs[(rs["结果"] == "1w盈亏") & (rs["模型"] == "无")].iloc[0]
    pym = rs[(rs["结果"] == "1w盈亏") & (rs["模型"] == "ym")].iloc[0]
    q1q5 = within[(within["样本"] == "100+_同日") & (within["对比"] == "Q1-Q5")].iloc[0]
    q1q2 = within[(within["样本"] == "100+_同日") & (within["对比"] == "Q1-Q2")].iloc[0]

    # 示例
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>下跌斜率：越陡越好，还是越缓越好？</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
<div class="kicker">下跌斜率 · 对数坐标 · 2005-2026</div>
<h1>从最近高点到信号日/买入日：越陡越好，还是越缓越好？</h1>
<p class="sub">C 池（含退市、信号日排除风险警示），前二候选 1,250 个、100+ 日全部候选 7,561 个；
高点取信号日前 250 个个股成交日的最高后复权价，斜率 = ln(信号日收盘 / 高点) ÷ 全市场交易日数；
买入 T+3 开盘，15 日收益不含成本，1w 策略盈亏含 +20% 止盈与全部费用。</p>
<div class="banner">原始结果：陡明显好于缓。前二最陡 Q1 的 15 日收益 +{q1.十五日收益*100:.1f}%、
止盈20率 {q1.止盈20率*100:.1f}%、1w 平均盈亏 +{q1.平均盈亏:,.0f} 元；最缓 Q5 只有
+{q5.十五日收益*100:.1f}%、{q5.止盈20率*100:.1f}%、+{q5.平均盈亏:,.0f} 元。
但把“同一天/同一月”控制住后，差距基本消失：同日两只前二二选一，陡的那只没有更好
（15日 {r15_diff.mean()*100:+.2f}pp，t={r15_t:.2f}；1w {pnl_diff.mean():+.0f} 元，t={pnl_t:.2f}）。
陡的优势主要是“它出现在什么行情窗口”，不是“同一天选哪只”。</div>
</header>

<div class="cards">
<div class="card blue"><div class="label">前二最陡 Q1：15日收益</div><div class="big">+{q1.十五日收益*100:.1f}%</div><div class="note">止盈20率 {q1.止盈20率*100:.1f}%，1w 平均 +{q1.平均盈亏:,.0f}</div></div>
<div class="card amber"><div class="label">前二最缓 Q5：15日收益</div><div class="big">+{q5.十五日收益*100:.1f}%</div><div class="note">止盈20率 {q5.止盈20率*100:.1f}%，1w 平均 +{q5.平均盈亏:,.0f}</div></div>
<div class="card green"><div class="label">100+ 实际买入：陡 tercile</div><div class="big">+{steep.策略收益*100:.1f}%</div><div class="note">13 笔，平均 +{steep.平均盈亏:,.0f} 元；缓只有 +{flatc.策略收益*100:.1f}%</div></div>
<div class="card red"><div class="label">同日二选一：陡 - 缓</div><div class="big">+{r15_diff.mean()*100:.2f}pp</div><div class="note">t={r15_t:.2f}，不显著；1w 差 {pnl_diff.mean():+.0f} 元</div></div>
</div>

<section>
<h2>一、口径：这个“斜率”怎么算</h2>
<p>对每个信号，先找 <b>信号日前 250 个个股成交日内的最高后复权价</b>（与 drawdown250 同一个窗口），
记下高点日；再算</p>
<div class="panel blue"><b>对数斜率 = ln(信号日收盘 ÷ 250日高点) ÷ 从高点日到信号日的全市场交易日数 × 100</b><br>
单位是“%/日”，负数；越负越陡。报告里为了直观，统一写成 <b>日均跌幅 = -斜率</b>，数值越大越陡。</div>
<p>为什么用“全市场交易日”而不是个股成交日：长期停牌的股票，用个股成交日会把复牌的一次性下跌算成
“几天跌 70%”，斜率会假陡。例：000908 在 2008-06-13，250 日高点是 2007-08-23，个股只成交了 4 天，
按个股日算斜率是 <b>-30.4%/日</b>；按全市场交易日是 195 天，斜率 <b>-0.62%/日</b>。
在对数图上，横轴是市场时间，所以用后者。</p>
<p>“最近的高点”用 250 日高点，而不是 60 日/20 日高点：下面第五节会看到，250 日高点的斜率对后续收益的
解释力最强；60 日、20 日反而更弱。信号日算和买入日算几乎一样（相关系数 0.984），所以用信号日算即可，
T+3 不会改变排序。</p>
<p class="small">名称是当前名称；信号日是否风险警示按当时的 is_st 判断，不按当前名称排除后来变 ST 的股票。
15 日收益（后复权）是买入后第 15 个交易日收盘 / 买入开盘 - 1（第 15 日停牌则顺延到下一个可成交收盘），不含成本；策略盈亏是按 +20% 盘中止盈 / 15 日到期、
1w、100 股整手、滑点/佣金/过户费/印花税模拟的，按到期已实现记账，不是逐日盯市；中途压力看“最大浮亏”列。
胜率按笔，不是按信号日；“盈亏胜率”的分母是可买满 1 手的笔数。</p>
</section>

<section>
<h2>二、原始分层：陡的组明显更好</h2>
<p>前二候选按日均跌幅分五组（Q1 最陡，Q5 最缓）。Q1 的日均跌幅是 {q1.日均跌幅:.2f}%/日，
从高点到信号日只有 {q1.下跌天数:.0f} 个交易日；Q5 只有 {q5.日均跌幅:.2f}%/日，
拖了 {q5.下跌天数:.0f} 个交易日。两者回撤深度差不多（{q1.回撤均值*100:.1f}% 对 {q5.回撤均值*100:.1f}%），
所以这不是“跌得深不深”的差别，而是“跌得快不快”的差别。</p>
<div class="panel blue"><b>先说明 Q/D 是相对分组：</b>Q1–Q5 是五分位、D1–D10 是十分位；这里的 Q1/D1 是
<b>1,250 个前二候选放在一起</b>按斜率排序后的最陡 20%/10%，不是某一天的最陡 20%/10%。
第三节的“同日 Q1/Q2/Q5”是每个 100+ 日内部重新排序，标签含义不同；10/23 两个信号的定位见第三节末尾。</div>
<div class="tablewrap">{q_html}</div>
<figure>{img(FIG / '01_前二_斜率五分位.png', '前二斜率五分位')}
<figcaption>左：15日持有收益（柱）与中位（黑点）；右：1w 平均盈亏（柱）与止盈20率（橙线）。
Q1 的 15日收益比 Q5 高 {(q1.十五日收益-q5.十五日收益)*100:.1f}pp，止盈20率高
{(q1.止盈20率-q5.止盈20率)*100:.1f}pp，1w 平均盈亏高 {q1.平均盈亏-q5.平均盈亏:,.0f} 元。</figcaption></figure>
<p>但 Q1 也不是“越极端越好”：十分位里 D2（日均跌幅 {d10.iloc[1].日均跌幅:.2f}%/日）的 15日收益
+{d10.iloc[1].十五日收益*100:.1f}%、平均盈亏 +{d10.iloc[1].平均盈亏:,.0f} 元，反而略高于 D1
（+{d10.iloc[0].十五日收益*100:.1f}%、+{d10.iloc[0].平均盈亏:,.0f} 元）。最极端的一档波动最大：
D1 的最大浮亏 {d10.iloc[0].最大浮亏*100:.1f}%，比 D2 的 {d10.iloc[1].最大浮亏*100:.1f}% 更深。</p>
<details><summary>展开十分位</summary><div class="tablewrap">{d_html}</div></details>
<p>示例：600559 老白干酒在 2008-04-01，高点 2008-03-13 到信号日只有 {ex_steep.下跌交易日:.0f} 个交易日，
日均跌幅 {ex_steep["日均跌幅%每日"]:.2f}%/日，15日收益 +{ex_steep.十五日收益*100:.1f}%，1w 盈亏
+{ex_steep.策略盈亏:,.0f} 元。反例：000022 深赤湾A 在 2018-09-10，高点 2017-06-09 到信号日
{ex_flat.下跌交易日:.0f} 个交易日，日均跌幅只有 {ex_flat["日均跌幅%每日"]:.2f}%/日，15日收益
{ex_flat.十五日收益*100:+.1f}%，1w 盈亏 {ex_flat.策略盈亏:+,.0f} 元。一个是急跌后的反弹，
一个是慢慢阴跌、没有恐慌盘。</p>
<figure>{img(FIG / '05_两个例子_对数路径.png', '两个例子的对数路径')}
<figcaption>同一张对数图：600559 从高点到信号日只有 11 个市场交易日，几乎垂直；000022 用了 302 天，
缓慢阴跌。两者回撤深度都接近 -45%，但斜率差了一个数量级。000022 中间有一段长期停牌，
横轴按市场交易日，所以显示为水平线；复牌后补跌。若按个股成交日，这段会被压缩掉。</figcaption></figure>
</section>

<section>
<h2>三、100+ 信号日：全样本 Q2 最好，但同一天内 Q1 略高于 Q2</h2>
<p>只看 100+ 信号日，把当天所有候选按日均跌幅分五组：Q1（最陡）的 15日收益只有
+{a100.iloc[0].十五日收益*100:.1f}%，Q2 是 +{a100.iloc[1].十五日收益*100:.1f}%，Q5（最缓）是
{a100.iloc[-1].十五日收益*100:+.1f}%。看起来“Q2 最好、Q1 一般、Q5 最差”。</p>
<div class="tablewrap">{a_html}</div>
<p>但这是 <b>全样本分组</b>，不是同一天比较。把每个 100+ 日内部再分 Q1/Q2/Q5，Q1 的 15日收益是
{w100.q1.mean()*100:+.1f}%，Q2 是 {w100.q2.mean()*100:+.1f}%，Q5 是 {w100.q5.mean()*100:+.1f}%；
Q1 反而略高于 Q2（+{q1q2.平均差*100:.2f}pp，t={q1q2.t值:.2f}，不显著），Q1 比 Q5 高
+{q1q5.平均差*100:.2f}pp（t={q1q5.t值:.2f}，显著）。全样本里 Q2 更好，是因为 Q1 更集中在整体反弹更弱的几天，
不是 Q1 本身更差。</p>
<figure>{img(FIG / '02_100信号日_全样本与同日.png', '100+ 全样本与同日')}
<figcaption>灰柱是全样本分组，蓝柱是同一天内分组。全样本 Q2 > Q1；同一天内 Q1 > Q2，但差距不显著。
真正稳定的是 Q5（最缓）最差。</figcaption></figure>
<h3>100+ 日实际买入的 37 笔</h3>
<p>按 09 的“不重复”实际交易，37 笔再按斜率分三组：陡 13 笔、15日收益 +{steep.十五日收益*100:.1f}%、
策略收益 +{steep.策略收益*100:.1f}%、平均 +{steep.平均盈亏:,.0f} 元、胜率 {steep.盈亏胜率*100:.1f}%；
缓 12 笔只有 +{flatc.十五日收益*100:.1f}%、+{flatc.策略收益*100:.1f}%、+{flatc.平均盈亏:,.0f} 元、
胜率 {flatc.盈亏胜率*100:.1f}%。</p>
<div class="tablewrap">{c_html}</div>
<p>缓组也有单笔大赢家：600288 在 2016-01-19 赚 +{flatc.最好:,.0f} 元，是 37 笔里最好的；
但缓组整体胜率、平均收益都低，靠个别大赢家撑住。陡组更稳定地赢，但最大浮亏也更深
（{steep.最大浮亏*100:.1f}% 对 {flatc.最大浮亏*100:.1f}%）。</p>
<p class="small">37 笔的样本很小；而且这 37 笔是按“回撤最深”选出来的前二，不是随机样本。
下面的同日/同月检验更关键。</p>
<h3>附：2026-10-23 的两个信号在哪个组</h3>
<p>10/23 共 122 个候选。按当天斜率排序：605376 第 3 陡（前 1.7%），603778 第 25 陡（前 19.8%，
刚好在 Q1 边界）。放到 2005-2026 的 1,250 个前二里：605376 约前 11.9%（Q1/D2），
603778 约前 21.5%（Q2/D3）。</p>
<div class="tablewrap">{loc_html}</div>
<p class="small">Q1–Q5 / D1–D10 是相对分位：主表的 Q1 是 1,250 个前二放在一起的最陡 20%；
这里的“当天”是 122 个候选在 10/23 内部重新排序。同一个 603778，当天是 Q1 边界，放到全样本只是 Q2。
两者的回撤（-71.3% / -78.6%）比全样本 Q1/Q2 的平均（-58.5% / -63.9%）更深，历史档位平均只能作参考；
更接近的是上面 100+ 实际 37 笔（平均回撤 -71.6%）。10/23 的买入日在 10/28、在样本外，
且 10/28 没有空位，所以这里只做形态定位，不是买卖建议。</p>
</section>

<section>
<h2>四、关键：把“同一天/同一月”控制住，斜率的优势基本消失</h2>
<p>前二候选的 15日收益对日均跌幅做回归（控制回撤深度，按信号日聚类）：</p>
<div class="tablewrap">{r_html}</div>
<p>另外，同一回归里回撤深度的系数是 -0.20（t=-2.51）：同月内越深反而越差，和“按最深排序”的直觉相反。</p>
<p>不加固定效应时，日均跌幅每多 1%/日，15日收益高 {nofe.系数*100:.2f}pp（t={nofe.t值:.2f}），
1w 盈亏高 {pnofe.系数:,.0f} 元（t={pnofe.t值:.2f}），看起来很显著。加上<b>年固定效应</b>后，
15日系数降到 {rs[(rs['结果']=='15日收益')&(rs['模型']=='year')].iloc[0].系数*100:.2f}pp（t=1.97），
1w 盈亏降到 {rs[(rs['结果']=='1w盈亏')&(rs['模型']=='year')].iloc[0].系数:,.0f} 元（t=2.12）。
再加上<b>月固定效应</b>，15日系数只剩 {ymfe.系数*100:.2f}pp（t={ymfe.t值:.2f}），
1w 盈亏只剩 {pym.系数:,.0f} 元（t={pym.t值:.2f}）——基本是 0。</p>
<figure>{img(FIG / '03_固定效应.png', '固定效应')}
<figcaption>每 1%/日的斜率系数，误差棒是 95% 区间。无固定效应显著为正；加月份固定效应后，
15日收益和 1w 盈亏都接近 0。陡的信号大多出现在反弹更猛的月份，不是它本身有独立溢价。</figcaption></figure>
<p>最直接的同日检验：同一天有两只前二候选时，把更陡的那只和更缓的那只配对比较。</p>
<figure>{img(FIG / '04_同日两只前二.png', '同日两只前二')}
<figcaption>564 天配对：更陡的那只 15日收益 {pr15.s_r15.mean()*100:.2f}%，更缓的 {pr15.f_r15.mean()*100:.2f}%，
差 {r15_diff.mean()*100:+.2f}pp（t={r15_t:.2f}）；1w 盈亏 {ppnl.s_pnl.mean():,.0f} 对
{ppnl.f_pnl.mean():,.0f}，差 {pnl_diff.mean():+.0f} 元（t={pnl_t:.2f}）。都没有显著差异。</figcaption></figure>
<p>再看全部候选的同日横截面（当天候选数 ≥10 的日子）：</p>
<div class="tablewrap">{w_html}</div>
<p>全部信号：Q1-Q5 同日差 +{within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q5')].iloc[0].平均差*100:.2f}pp
（t={within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q5')].iloc[0].t值:.2f}），
但 Q1-Q2 只有 +{within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q2')].iloc[0].平均差*100:.2f}pp
（t={within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q2')].iloc[0].t值:.2f}）。
100+ 日：Q1-Q5 +{q1q5.平均差*100:.2f}pp（t={q1q5.t值:.2f}），Q1-Q2 +{q1q2.平均差*100:.2f}pp（t={q1q2.t值:.2f}）。
所以真正稳定的不是“陡的溢价”，而是“<b>最缓的那组更差</b>”：同日虚拟变量回归里，最缓 1/3 比中间组低
0.94pp（t=-3.67）；最陡 1/3 只比中间组高 0.53pp（t=2.01），而且加回撤深度控制后只剩 0.26pp（t=0.98）。
换句话说：<b>斜率的主要信息是“别买平缓阴跌”，不是“买最陡”。</b></p>
</section>

<section>
<h2>五、用哪个高点、在哪天算？</h2>
<p>把斜率分别按 250 日、60 日、20 日高点计算，再分别用信号日收盘和 T+3 买入开盘点算，和后续收益做
Spearman 相关（已把斜率取反成“陡峭度”，正=越陡越好）：</p>
<div class="tablewrap">{x_html}</div>
<p>250 日高点的 15日收益相关是 {corr[(corr['斜率口径']=='250日高点_信号日')&(corr['结果']=='15日收益')].iloc[0].相关系数:.3f}，
60 日只有 {corr[(corr['斜率口径']=='60日高点_信号日')&(corr['结果']=='15日收益')].iloc[0].相关系数:.3f}，
20 日只有 {corr[(corr['斜率口径']=='20日高点_信号日')&(corr['结果']=='15日收益')].iloc[0].相关系数:.3f}。
信号日算和买入日算几乎一样（0.166 对 0.164）——T+3 的 3 天不会改变排序。</p>
<p>所以：<b>看整段下跌的形状，不要只看最后 20 天/60 天。</b>最后一段加速下跌，可能只是下跌的尾声；
250 日高点到现在的整段斜率，才和反弹力度更相关。实际使用时在信号日算即可，不用等 T+3 开盘。</p>
</section>

<section>
<h2>六、结论：怎么用</h2>
<ul>
<li><b>平缓阴跌是最该回避的。</b>前二最缓 Q5 的 15日收益只有 +{q5.十五日收益*100:.1f}%、止盈20率
{q5.止盈20率*100:.1f}%、1w 平均 +{q5.平均盈亏:,.0f} 元；同日横截面里最缓 1/3 也比中间组低
0.94pp（t=-3.67）。它没有恐慌盘，反弹弱，最容易变成“慢慢阴跌”的价值陷阱。</li>
<li><b>陡峭急跌整体更好，但不是“越陡越神”。</b>前二 Q1 的 15日收益 +{q1.十五日收益*100:.1f}%、
1w 平均 +{q1.平均盈亏:,.0f} 元；但十分位 D1 不如 D2，100+ 日全部候选里 Q1 也只有 +{a100.iloc[0].十五日收益*100:.1f}%，
最大浮亏 {a100.iloc[0].最大浮亏*100:.1f}%。最极端的一档是“高波动彩票”，不是稳定 alpha。</li>
<li><b>同一天二选一，斜率不能帮你选。</b>564 天配对：更陡的那只 15日收益只高
{r15_diff.mean()*100:+.2f}pp（t={r15_t:.2f}），1w 盈亏反而低 {-pnl_diff.mean():.0f} 元（t={pnl_t:.2f}）。
两只都已经很陡时，不要再纠结谁更陡。</li>
<li><b>它主要是“行情温度计”，不是“选股因子”。</b>无固定效应时斜率系数 +{nofe.系数*100:.2f}pp/1%/日
（t={nofe.t值:.2f}），加月份固定效应只剩 +{ymfe.系数*100:.2f}pp（t={ymfe.t值:.2f}）。
陡的信号集中在 2007-2008、2015 这类反弹更猛的窗口；是“什么时候”而不是“选哪只”。</li>
<li><b>不要用它替代风险控制。</b>斜率与回撤深度相关；加回撤控制后，陡的正向基本消失。
2008 的 600811 -2,306 元这类尾部，止盈/斜率都解决不了，仍是止损/仓位的事。深度排序本身也值得再检验。</li>
</ul>
<div class="panel amber"><b>一句话：</b>对数图上“从最近高点到信号日”的下跌，<b>太平缓比太陡更糟</b>；
陡峭急跌可以偏好，但“最陡”不是核心排序因子。真正可操作的只有两条：<b>回避最缓的 1/3；不要把斜率当成同一天二选一的依据。</b>
如果要做成规则，先做“排除最缓”的过滤，再做 OOS；不要直接“只买最陡”。</div>
</section>

<section>
<h2>七、限制</h2>
<ul>
<li>斜率是“高点→信号日”的平均速度，不是曲线形状；同样斜率的 V 型反转和圆弧顶，后续可能不同。</li>
<li>只按斜率分组，没有做波动率/RSI/成交额/市值中性化；斜率和这些特征相关，分组差异不能全归因于斜率本身。</li>
<li>250 日高点是个股 250 个成交日内的最高价，长期停牌股的“高点”可能离信号日很远；本报告用全市场交易日做横轴，
但仍不能完全消除停牌复牌的一次性重估。已用 000908 的例子说明。</li>
<li>前二候选 1,250 个、100+ 日全部候选 7,561 个，但 100+ 实际买入只有 37 笔；小样本的 tercile 差异不能外推。</li>
<li>前二 1,250 个信号来自 543 只股票、608 个“股票×250日高点” episode；同一只股票在连续信号日会重复出现（最多 18 次）。回归按信号日聚类，但仍不是 1,250 个完全独立的样本。</li>
<li>15日收益不含成本；1w 策略盈亏含成本，但按 100 股整手，高价股可能买不满 1w（每组的“盈亏笔数”略少于“笔数”）。</li>
<li>这是历史回测，不是未来收益保证；不构成投资建议。</li>
</ul>
</section>

<footer>
口径：C 池 = 全主板含退市股，信号日 is_st=1 不买；成交价不复权，信号按后复权；成本含 0.1%/边滑点、
佣金 0.025%（最低 5 元）、过户费、卖出印花税。复现：<code>02_代码/analyze_slope.py</code> 生成数据，
<code>02_代码/make_slope_report.py</code> 生成本页。
</footer>
</div></body></html>"""
    (OUT / "下跌斜率_对数坐标.html").write_text(html, encoding="utf-8")

    md = f"""# 下跌斜率：越陡越好，还是越缓越好？

C 池（含退市、信号日排除风险警示）；高点 = 信号日前 250 个个股成交日的最高后复权价；
斜率 = ln(信号日收盘 / 高点) / 全市场交易日数 × 100（%/日，越负越陡）；
报告里用“日均跌幅 = -斜率”，数值越大越陡。T+3 开盘买，15日收益不含成本，1w 策略盈亏含 +20% 止盈与全部费用。

## 短答

- **平缓阴跌最差，陡峭急跌整体更好，但不是越陡越好。**
- 前二最陡 Q1：15日收益 +{q1.十五日收益*100:.1f}%、止盈20率 {q1.止盈20率*100:.1f}%、1w 平均 +{q1.平均盈亏:,.0f} 元；
  最缓 Q5 只有 +{q5.十五日收益*100:.1f}%、{q5.止盈20率*100:.1f}%、+{q5.平均盈亏:,.0f} 元。
- 100+ 日实际买入 37 笔：陡 13 笔平均 +{steep.平均盈亏:,.0f} 元（策略收益 +{steep.策略收益*100:.1f}%），
  缓 12 笔平均 +{flatc.平均盈亏:,.0f} 元（+{flatc.策略收益*100:.1f}%）。
- **但同日二选一没有优势**：564 天配对，更陡的那只 15日收益只高 {r15_diff.mean()*100:+.2f}pp（t={r15_t:.2f}），
  1w 盈亏反而低 {-pnl_diff.mean():.0f} 元（t={pnl_t:.2f}）。
- **它主要是行情温度计**：无固定效应斜率系数 +{nofe.系数*100:.2f}pp/1%/日（t={nofe.t值:.2f}），
  加月份固定效应只剩 +{ymfe.系数*100:.2f}pp（t={ymfe.t值:.2f}）；1w 盈亏从 +{pnofe.系数:,.0f} 元
  降到 +{pym.系数:,.0f} 元（t={pym.t值:.2f}）。

## 一、口径

- 高点：信号日前 250 个个股成交日内的最高后复权价（与 drawdown250 同一窗口）。
- 斜率：ln(信号日收盘 / 高点) / 从高点日到信号日的**全市场交易日数** × 100。用全市场交易日，
  避免长期停牌把斜率人为拉陡（000908 在 2008-06-13：个股只成交 4 天，按个股日 -30.4%/日；
  按全市场 195 天 -0.62%/日）。
- “最近的高点”用 250 日高点；60/20 日高点的预测力更弱（见第五节）。
- 信号日算和买入日算几乎一样（相关系数 0.984；15日收益 Spearman 0.166 对 0.164），用信号日即可。
- 名称是当前名称；信号日是否风险警示按当时的 is_st，不按当前名称排除后来变 ST 的股票。
- 15日收益（后复权） = 买入后第 15 个交易日收盘 / 买入开盘 - 1（第 15 日停牌顺延到下一个可成交收盘），不含成本；1w 盈亏含成本、按到期已实现记账，不是逐日盯市；中途压力看“最大浮亏”。胜率按笔，不是按信号日；“盈亏胜率”的分母是可买满 1 手的笔数。

## 二、前二分层：陡的组明显更好

{q_md}

Q1 日均跌幅 {q1.日均跌幅:.2f}%/日、下跌 {q1.下跌天数:.0f} 个交易日；Q5 只有 {q5.日均跌幅:.2f}%/日、
拖了 {q5.下跌天数:.0f} 个交易日。两者回撤深度差不多（{q1.回撤均值*100:.1f}% 对 {q5.回撤均值*100:.1f}%），
差别是“跌得快不快”，不是“跌得深不深”。

Q1–Q5 / D1–D10 是**样本内相对分组**：这里的 Q1 是 1,250 个前二候选放在一起的最陡 20%，
不是某一天的最陡 20%；D1 是最陡 10%。第三节的“同日 Q1/Q2/Q5”是每个 100+ 日内部重新排序，
标签含义不同；10/23 的定位见第三节末尾。

但最极端的一档不是最好：十分位 D2 的 15日收益 +{d10.iloc[1].十五日收益*100:.1f}%、平均盈亏
+{d10.iloc[1].平均盈亏:,.0f} 元，略高于 D1 的 +{d10.iloc[0].十五日收益*100:.1f}%、
+{d10.iloc[0].平均盈亏:,.0f} 元；D1 最大浮亏 {d10.iloc[0].最大浮亏*100:.1f}%，比 D2 的
{d10.iloc[1].最大浮亏*100:.1f}% 更深。

<details><summary>展开十分位</summary>

{d_md}

</details>

示例：600559 老白干酒 2008-04-01，高点 2008-03-13 到信号日 {ex_steep.下跌交易日:.0f} 个交易日，
日均跌幅 {ex_steep["日均跌幅%每日"]:.2f}%/日，15日收益 +{ex_steep.十五日收益*100:.1f}%，
1w 盈亏 +{ex_steep.策略盈亏:,.0f} 元。反例：000022 深赤湾A 2018-09-10，高点 2017-06-09 到信号日
{ex_flat.下跌交易日:.0f} 个交易日，日均跌幅只有 {ex_flat["日均跌幅%每日"]:.2f}%/日，15日收益
{ex_flat.十五日收益*100:+.1f}%，1w 盈亏 {ex_flat.策略盈亏:+,.0f} 元。

## 三、100+ 信号日：全样本 Q2 最好，同一天内 Q1 略高于 Q2

{a_md}

全样本看起来 Q2 最好、Q1 一般、Q5 最差；但把每个 100+ 日内部再分 Q1/Q2/Q5：
Q1 {w100.q1.mean()*100:+.1f}%、Q2 {w100.q2.mean()*100:+.1f}%、Q5 {w100.q5.mean()*100:+.1f}%。
Q1 反而略高于 Q2（+{q1q2.平均差*100:.2f}pp，t={q1q2.t值:.2f}），Q1 比 Q5 高
+{q1q5.平均差*100:.2f}pp（t={q1q5.t值:.2f}）。全样本 Q2 更好，是 Q1 更集中在整体反弹更弱几天的构成效应。

100+ 日实际买入 37 笔：

{c_md}

缓组也有单笔大赢家（600288 2016-01-19 +{flatc.最好:,.0f} 元），但整体胜率、平均收益都低；
陡组更稳定地赢，最大浮亏也更深（{steep.最大浮亏*100:.1f}% 对 {flatc.最大浮亏*100:.1f}%）。
37 笔样本小，不能外推。

### 附：2026-10-23 的两个信号在哪个组

10/23 共 122 个候选。按当天斜率排序：605376 第 3 陡（前 1.7%），603778 第 25 陡（前 19.8%，
刚好在 Q1 边界）。放到 2005-2026 的 1,250 个前二里：605376 约前 11.9%（Q1/D2），
603778 约前 21.5%（Q2/D3）。

{loc_md}

同一个 603778，当天是 Q1 边界，放到全样本只是 Q2；Q/D 标签是相对的，不是绝对档位。
两者的回撤（-71.3% / -78.6%）比全样本 Q1/Q2 的平均（-58.5% / -63.9%）更深，历史档位平均只能作参考；
更接近的是上面 100+ 实际 37 笔（平均回撤 -71.6%）。10/23 的买入日在 10/28、在样本外，
且 10/28 没有空位，所以这里只做形态定位，不是买卖建议。

## 四、同日/同月检验：斜率的优势基本消失

前二 15日收益 / 1w 盈亏对日均跌幅回归（控制回撤深度，按信号日聚类）：

{r_md}

无固定效应：15日系数 +{nofe.系数*100:.2f}pp/1%/日（t={nofe.t值:.2f}），1w +{pnofe.系数:,.0f} 元/1%/日
（t={pnofe.t值:.2f}）。年固定后 15日 +{rs[(rs['结果']=='15日收益')&(rs['模型']=='year')].iloc[0].系数*100:.2f}pp
（t=1.97）、1w +{rs[(rs['结果']=='1w盈亏')&(rs['模型']=='year')].iloc[0].系数:,.0f} 元（t=2.12）。
月固定后 15日 +{ymfe.系数*100:.2f}pp（t={ymfe.t值:.2f}）、1w +{pym.系数:,.0f} 元（t={pym.t值:.2f}）——基本是 0。另外，回撤深度的系数是 -0.20（t=-2.51）：同月内越深反而越差；这和“按最深排序”的直觉相反，深度本身也需要 OOS。

同日两只前二配对（564 天）：更陡 15日 {pr15.s_r15.mean()*100:.2f}% 对更缓 {pr15.f_r15.mean()*100:.2f}%，
差 {r15_diff.mean()*100:+.2f}pp（t={r15_t:.2f}）；1w {ppnl.s_pnl.mean():,.0f} 对
{ppnl.f_pnl.mean():,.0f}，差 {pnl_diff.mean():+.0f} 元（t={pnl_t:.2f}）。没有显著差异。

当天候选数 ≥10 的同日横截面：

{w_md}

全部信号 Q1-Q5 同日 +{within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q5')].iloc[0].平均差*100:.2f}pp
（t={within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q5')].iloc[0].t值:.2f}），
但 Q1-Q2 只有 +{within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q2')].iloc[0].平均差*100:.2f}pp
（t={within[(within['样本']=='全部信号_同日')&(within['对比']=='Q1-Q2')].iloc[0].t值:.2f}）。
100+ 日 Q1-Q5 +{q1q5.平均差*100:.2f}pp（t={q1q5.t值:.2f}），Q1-Q2 +{q1q2.平均差*100:.2f}pp
（t={q1q2.t值:.2f}）。

同日虚拟变量回归：最缓 1/3 比中间组低 0.94pp（t=-3.67），最陡 1/3 只高 0.53pp（t=2.01）；
加回撤深度控制后，最陡只剩 +0.26pp（t=0.98），最缓 -0.47pp（t=-1.75）。
所以主要信息是“别买平缓阴跌”，不是“买最陡”。

## 五、用哪个高点、在哪天算

{x_md}

250 日高点 15日相关 {corr[(corr['斜率口径']=='250日高点_信号日')&(corr['结果']=='15日收益')].iloc[0].相关系数:.3f}，
60 日 {corr[(corr['斜率口径']=='60日高点_信号日')&(corr['结果']=='15日收益')].iloc[0].相关系数:.3f}，
20 日 {corr[(corr['斜率口径']=='20日高点_信号日')&(corr['结果']=='15日收益')].iloc[0].相关系数:.3f}。
信号日算和买入日算几乎一样（0.166 对 0.164）。看整段下跌的形状，不要只看最后 20/60 天；
在信号日算即可，不用等 T+3。

## 六、结论

- 平缓阴跌最该回避：Q5 的 15日收益 +{q5.十五日收益*100:.1f}%、止盈20率 {q5.止盈20率*100:.1f}%、
  1w 平均 +{q5.平均盈亏:,.0f} 元；同日最缓 1/3 比中间组低 0.94pp（t=-3.67）。
- 陡峭急跌整体更好，但不是越陡越好：Q1 15日 +{q1.十五日收益*100:.1f}%、1w +{q1.平均盈亏:,.0f} 元；
  D1 不如 D2；100+ 全部候选 Q1 只有 +{a100.iloc[0].十五日收益*100:.1f}%，最大浮亏 {a100.iloc[0].最大浮亏*100:.1f}%。
- 同一天二选一，斜率没有区分度：564 天配对 {r15_diff.mean()*100:+.2f}pp（t={r15_t:.2f}），
  1w 差 {pnl_diff.mean():+.0f} 元（t={pnl_t:.2f}）。
- 它主要是行情温度计：月固定后系数 +{ymfe.系数*100:.2f}pp（t={ymfe.t值:.2f}），1w +{pym.系数:,.0f} 元（t={pym.t值:.2f}）。
- 不要替代风险控制；2008 的 600811 -2,306 元，止盈/斜率都解决不了。深度排序本身也值得再检验。

**一句话：太平缓比太陡更糟；陡峭急跌可以偏好，但“最陡”不是核心排序因子。
可操作的只有“回避最缓的 1/3”，而且要先 OOS。**

## 七、限制

- 斜率是平均速度，不是曲线形状；V 型反转和圆弧顶可能不同。
- 只按斜率分组，没有做波动率/RSI/成交额/市值中性化；斜率和这些特征相关，分组差异不能全归因于斜率本身。
- 250 日高点长期停牌股可能离信号日很远；已用全市场交易日做横轴，但仍不能完全消除停牌复牌的一次性重估。
- 前二 1,250 个、100+ 全部候选 7,561 个，但 100+ 实际买入只有 37 笔；小样本不能外推。
- 前二 1,250 个信号来自 543 只股票、608 个“股票×250日高点” episode；同一只股票在连续信号日会重复出现（最多 18 次）。回归按信号日聚类，但仍不是 1,250 个完全独立的样本。
- 15日收益不含成本；1w 盈亏含成本，但高价股可能买不满 1w（每组“盈亏笔数”略少于“笔数”）。
- 历史回测不代表未来，不构成投资建议。

图表与页面：`下跌斜率_对数坐标.html`；逐笔：`下跌斜率_逐笔.csv`；分桶：`下跌斜率_分桶.csv`；
回归：`下跌斜率_回归.csv`；同日：`下跌斜率_同日对比.csv`；相关性：`下跌斜率_相关性.csv`。
复现：`02_代码/analyze_slope.py`、`02_代码/make_slope_report.py`。
"""
    (OUT / "下跌斜率_结论.md").write_text(md, encoding="utf-8")

    print(q_md)
    print()
    print(c_md)
    print()
    print(r_md)
    print()
    print(w_md)


if __name__ == "__main__":
    main()
