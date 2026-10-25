# -*- coding: utf-8 -*-
"""生成《主板恐慌反转：策略与场景全景》单文件 HTML。

数据源：
- 07_市场阶段适配/03_结果/  （年度、ABCD、波动、宽度、信号数、市场回撤）
- 09_信号日回撤深度与收益/03_结果/（个股回撤深度分档、当前信号）
- 08_重复信号与买入时点/03_结果/（当前回撤口径，备用）

输出：不同时间入场点回测/策略与场景全景.html
"""
from __future__ import annotations

from pathlib import Path
import html as htmllib
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
R07 = BASE / "07_市场阶段适配" / "03_结果"
R09 = BASE / "09_信号日回撤深度与收益" / "03_结果"
OUT = BASE / "策略与场景全景.html"


# ----------------------------------------------------------------------------- 数据
d = pd.read_csv(R07 / "全部信号日逐只_带阶段.csv", parse_dates=["signal_date"])
c = d[d["complete"] == True].copy()
c["y"] = c["signal_date"].dt.year
c["month"] = c["signal_date"].dt.strftime("%Y-%m")

vol_median = float(c["market_volatility20"].median())
c["vol_group"] = np.where(c["market_volatility20"] < vol_median, "低波", "高波")
c["breadth_q"] = pd.qcut(c["breadth20"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
c["signal_bin"] = pd.cut(
    c["n_signals"],
    [0, 2, 5, 15, 40, 100, np.inf],
    labels=["1-2", "3-5", "6-15", "16-40", "41-100", ">100"],
)
c["mdd_bin"] = pd.cut(
    c["mkt_dd250"],
    [-1, -0.20, -0.10, 1],
    labels=["<-20%", "-20~-10%", ">-10%"],
)
c["ret250_bin"] = pd.cut(
    c["mkt_ret250"],
    [-1, 0, 0.10, 1],
    labels=["<0", "0-10%", ">10%"],
)
c["phase"] = c["阶段"].str[0]


def summarize(z: pd.DataFrame) -> dict:
    z = z.dropna(subset=["ret"])
    ex = z[z["y"] < 2026]
    return {
        "n": int(len(z)),
        "mean": float(z["ret"].mean()) if len(z) else np.nan,
        "median": float(z["ret"].median()) if len(z) else np.nan,
        "win": float((z["ret"] > 0).mean()) if len(z) else np.nan,
        "exn": int(len(ex)),
        "exmean": float(ex["ret"].mean()) if len(ex) else None,
        "exwin": float((ex["ret"] > 0).mean()) if len(ex) else None,
    }


phase = {k: summarize(c[c["phase"] == k]) for k in "ABCD"}
vol = {k: summarize(c[c["vol_group"] == k]) for k in ["低波", "高波"]}
breadth = {k: summarize(c[c["breadth_q"] == k]) for k in ["Q1", "Q2", "Q3", "Q4", "Q5"]}
signals = {
    k: summarize(c[c["signal_bin"] == k])
    for k in ["1-2", "3-5", "6-15", "16-40", "41-100", ">100"]
}
mdd = {k: summarize(c[c["mdd_bin"] == k]) for k in ["<-20%", "-20~-10%", ">-10%"]}
volret = {
    f"{a}/{b}": summarize(c[(c["vol_group"] == a) & (c["ret250_bin"] == b)])
    for a in ["低波", "高波"]
    for b in ["<0", "0-10%", ">10%"]
}
vol_below_178 = summarize(c[c["market_volatility20"] < 0.178])
vol_mid_band = summarize(c[(c["market_volatility20"] >= 0.178) & (c["market_volatility20"] < vol_median)])

annual = pd.read_csv(R07 / "年度对比.csv")
contrib = pd.read_csv(R07 / "阶段贡献.csv")
mother_contrib = contrib[contrib["口径"] == "母路径99笔"].copy()
depth = pd.read_csv(R09 / "前二深度分档.csv")
depth = depth[depth["样本"] == "前二"].copy()
current_sig = pd.read_csv(R09 / "当前信号_深度对照.csv")
R10 = BASE / "10_250日涨幅与收益" / "03_结果"
r250_bins = pd.read_csv(R10 / "六档成绩_前二.csv")
r250_corr = pd.read_csv(R10 / "相关性.csv").set_index("样本")
r250_reg = pd.read_csv(R10 / "回归.csv").set_index("模型")
r250_cur = pd.read_csv(R10 / "当前信号_250日涨幅.csv")
market = pd.read_csv(R07 / "市场阶段日度.csv", parse_dates=["date"])
cur = market.iloc[-1]
fh = pd.read_csv(R07 / "固定持有期对比.csv")
starts26 = pd.read_csv(R07 / "2026起点.csv", parse_dates=["signal_date"])

# 当前市场分位
all_vol = market["market_volatility20"].dropna()
sig_vol = market.loc[market["是否信号日"] == 1, "market_volatility20"].dropna()
all_breadth = market["breadth20"].dropna()
sig_breadth = market.loc[market["是否信号日"] == 1, "breadth20"].dropna()
vol_pct_all = float((all_vol < cur["market_volatility20"]).mean())
vol_pct_sig = float((sig_vol < cur["market_volatility20"]).mean())
breadth_pct_all = float((all_breadth < cur["breadth20"]).mean())
breadth_pct_sig = float((sig_breadth < cur["breadth20"]).mean())
mkt_after = market[market["date"] >= "2017-11-23"]
vol_med_after = float(mkt_after["market_volatility20"].median())
vol_pct_after = float((mkt_after["market_volatility20"] < cur["market_volatility20"]).mean())

# 2026 三口径
acct_2026 = 0.1974693801159288
st26_n = len(starts26)
st26_med = float(starts26["total_ret"].median())
st26_pos = int((starts26["total_ret"] > 0).sum())
st26_neg = int((starts26["total_ret"] < 0).sum())
st26_zero = int((starts26["total_ret"] == 0).sum())
st26_best = starts26.loc[starts26["total_ret"].idxmax()]
st26_worst = starts26.loc[starts26["total_ret"].idxmin()]
st26_best_date = f"{st26_best['signal_date'].month}/{st26_best['signal_date'].day}"
st26_worst_date = f"{st26_worst['signal_date'].month}/{st26_worst['signal_date'].day}"
st26_0603 = float(starts26.loc[starts26["signal_date"] == "2026-06-03", "total_ret"].iloc[0])
st26_0604 = float(starts26.loc[starts26["signal_date"] == "2026-06-04", "total_ret"].iloc[0])

c26 = c[c["y"] == 2026]
s26 = summarize(c26)
c26_phase = c26.groupby("phase").agg(n=("ret", "size"), mean=("ret", "mean"), win=("ret", lambda s: (s > 0).mean()))
june = c[c["month"] == "2026-06"]
july = c[c["month"] == "2026-07"]
june_s = summarize(june)
july_s = summarize(july)
june_hold = float(june["持有期市场"].mean())
june_hold_neg = float((june["持有期市场"] < 0).mean())
july_hold = float(july["持有期市场"].mean())
july_hold_neg = float((july["持有期市场"] < 0).mean())
c26_hold = float(c26["持有期市场"].mean())
c26_hold_neg = float((c26["持有期市场"] < 0).mean())
pre_hold = float(c[c["y"] < 2026]["持有期市场"].mean())
pre_hold_neg = float((c[c["y"] < 2026]["持有期市场"] < 0).mean())
june_vol = float(june["market_volatility20"].median())
pre_vol = float(c[c["y"] < 2026]["market_volatility20"].median())
june_nsig = float(june["n_signals"].median())
big100 = c[c["n_signals"] > 100]
big100_n = int(len(big100))
big100_target = int((big100["exit_reason"] == "target").sum())
big100_time = int(big100_n - big100_target)

fh60 = fh[fh["持有交易日"] == 60].set_index("样本")
fh120 = fh[fh["持有交易日"] == 120].set_index("样本")


# ----------------------------------------------------------------------------- 格式化
def pct(x, d=1, plus=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    val = (Decimal(str(float(x))) * 100).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{val:+.{d}f}%" if plus else f"{val:.{d}f}%"


def cls(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "flat"
    return "pos" if x > 0 else ("neg" if x < 0 else "flat")


def pct_class(x):
    return cls(x)


def bar_rows(rows: list[dict], maxv: float) -> str:
    head = """
    <div class="bar-head">
      <span>场景 / 样本</span><span>单笔均值</span><span>均值</span><span>胜率</span><span>剔除 2026</span>
    </div>"""
    body = []
    for r in rows:
        mean = r["mean"]
        w = max(0.0, mean) * 100 / maxv * 100
        tick = ""
        if r.get("exmean") is not None and abs(r["exmean"] - mean) > 0.02:
            tw = max(0.0, r["exmean"]) * 100 / maxv * 100
            tick = f'<i class="bar-tick" style="left:{tw:.1f}%" title="剔除2026：{pct(r["exmean"])}"></i>'
        sub = r.get("sub", "")
        note = r.get("note", "")
        if note:
            sub = f"{sub} · {note}" if sub else note
        sub_html = f"<small>{sub}</small>" if sub else ""
        current = " current" if r.get("current") else ""
        badge = '<span class="mini-badge">当前</span>' if r.get("current") else ""
        if r.get("exmean") is not None:
            ex = f'<b>{pct(r.get("exmean"))}</b>'
            if r.get("exwin") is not None:
                ex += f'<span>{r["exwin"]*100:.1f}%</span>'
        else:
            ex = "—"
        body.append(
            f"""
        <div class="bar-row{current}" title="{htmllib.escape(r.get('title', ''))}">
          <div class="bar-label"><span>{r['label']}{badge}</span>{sub_html}</div>
          <div class="bar-track"><div class="bar-fill {pct_class(mean)}" style="width:{w:.1f}%"></div>{tick}</div>
          <div class="bar-mean {pct_class(mean)}">{pct(mean)}</div>
          <div class="bar-win">{r['win']*100:.1f}%</div>
          <div class="bar-ex">{ex}</div>
        </div>"""
        )
    return f'<div class="bar-table">{head}{"".join(body)}</div>'


def annual_chart(df: pd.DataFrame) -> str:
    lo, hi = -40.0, 100.0
    span = hi - lo
    zero = (0 - lo) / span * 100
    axis = f"""
      <div class="annual-axis">
        <span style="left:0%">-40%</span>
        <span style="left:{zero:.1f}%">0</span>
        <span style="left:{(50-lo)/span*100:.1f}%">+50%</span>
        <span style="left:100%">+100%</span>
      </div>"""
    rows = []
    for _, r in df.iterrows():
        y = int(r["年份"])
        strat = float(r["策略收益"])
        mkt = float(r["市场等权"])
        strat_pp = strat * 100
        mkt_pp = mkt * 100
        w = strat_pp / span * 100
        mpos = (mkt_pp - lo) / span * 100
        star = "*" if y in (2017, 2026) else ""
        rows.append(
            f"""
        <div class="annual-row">
          <div class="annual-year">{y}{star}</div>
          <div class="annual-track">
            <div class="annual-zero" style="left:{zero:.2f}%"></div>
            <div class="annual-bar" style="left:{zero:.2f}%;width:{w:.2f}%"></div>
            <div class="annual-tick" style="left:{mpos:.2f}%" title="等权市场 {pct(mkt)}"></div>
          </div>
          <div class="annual-strat pos">{pct(strat)}</div>
          <div class="annual-mkt {pct_class(mkt)}">{pct(mkt)}</div>
        </div>"""
        )
    return f'<div class="annual-chart">{axis}{"".join(rows)}</div>'


def phase_cell(code: str, title: str, tone: str, extra_class: str = "") -> str:
    s = phase[code]
    contrib_row = mother_contrib[mother_contrib["阶段"].str.startswith(code)].iloc[0]
    current = '<span class="current-badge">当前在这里</span>' if code == "D" else ""
    return f"""
      <div class="matrix-cell {tone} {extra_class}">
        {current}
        <div class="cell-code">{code}</div>
        <div class="cell-name">{title}</div>
        <div class="cell-n">{s['n']} 笔 · 单笔均值 <b class="{pct_class(s['mean'])}">{pct(s['mean'])}</b></div>
        <div class="cell-win">胜率 {s['win']*100:.1f}%</div>
        <div class="cell-ex">剔除 2026：{pct(s['exmean'])} / {s['exwin']*100:.1f}%</div>
        <div class="cell-contrib">母路径贡献 {contrib_row['利润贡献x']:.2f}x · {contrib_row['贡献占比']*100:.1f}%</div>
      </div>"""


# ----------------------------------------------------------------------------- CSS
CSS = r"""
:root{
  --paper:#f3eee5;
  --paper-2:#faf5ec;
  --card:#fffdf8;
  --ink:#1c2521;
  --ink-2:#3f4943;
  --muted:#72786f;
  --line:#e1d7c7;
  --line-2:#ece4d7;
  --red:#b54430;
  --red-2:#d46f54;
  --red-soft:#f4dfd6;
  --green:#2f7665;
  --green-soft:#dcebe5;
  --gold:#a87832;
  --gold-soft:#f3e6cf;
  --navy:#17231f;
  --navy-2:#22322c;
  --serif:"Songti SC","Noto Serif SC","Source Han Serif SC","STSong",serif;
  --sans:"PingFang SC","Hiragino Sans GB","Noto Sans SC","Microsoft YaHei",sans-serif;
  --mono:"SFMono-Regular","Cascadia Code","Roboto Mono",Menlo,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;color:var(--ink);background:
    radial-gradient(900px 480px at 8% -6%, rgba(181,68,48,.07), transparent 62%),
    radial-gradient(760px 420px at 96% 8%, rgba(47,118,101,.06), transparent 58%),
    var(--paper);
  font-family:var(--sans);font-size:16px;line-height:1.68;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
::selection{background:#eccdbd;color:#241a15}
a{color:var(--red);text-decoration:none;border-bottom:1px solid rgba(181,68,48,.28)}
a:hover{border-bottom-color:var(--red)}
.wrap{max-width:1120px;margin:0 auto;padding:0 26px}
.serif{font-family:var(--serif)}
.mono{font-family:var(--mono)}

/* hero */
.hero{position:relative;overflow:hidden;background:var(--navy);color:#f7f0e4;padding:74px 0 0}
.hero:before{
  content:"";position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
  background-size:44px 44px;
  mask-image:linear-gradient(to bottom,rgba(0,0,0,.75),transparent 82%);
}
.hero:after{
  content:"";position:absolute;right:-160px;top:-170px;width:560px;height:560px;border-radius:50%;
  background:radial-gradient(circle,rgba(203,126,80,.22),transparent 66%);
}
.hero-inner{position:relative;z-index:1}
.eyebrow{font-size:12px;letter-spacing:.18em;color:#c8b9a3;font-weight:600}
.hero h1{font-family:var(--serif);font-weight:600;font-size:clamp(42px,6.4vw,78px);line-height:1.03;letter-spacing:-.015em;margin:18px 0 22px}
.hero h1 em{font-style:normal;color:#e0a176}
.hero .lede{max-width:870px;color:#e7ded0;font-size:clamp(16px,1.8vw,20px);line-height:1.72;margin:0 0 10px}
.hero .lede b{color:#fff;font-weight:600}
.hero .lede .accent{color:#f0b58b;font-weight:600}
.hero-note{margin-top:24px;color:#afa697;font-size:12.5px;line-height:1.65}
.hero-note b{color:#e7ded0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.14);border-top:0;margin:38px 0 0;position:relative;z-index:1}
.stat{background:rgba(19,31,27,.72);padding:22px 22px 20px;backdrop-filter:blur(4px)}
.stat .k{font-size:11.5px;letter-spacing:.1em;color:#bdb1a0;margin-bottom:8px}
.stat .v{font-family:var(--serif);font-size:clamp(30px,3.6vw,44px);line-height:1.02;font-variant-numeric:tabular-nums;color:#fff}
.stat .v.red{color:#f0a37d}
.stat .v.green{color:#8ec7b5}
.stat .s{font-size:11.5px;color:#a99f90;margin-top:7px}
.hero-bottom{padding:18px 0 0;border-top:1px solid rgba(255,255,255,.1);margin-top:1px;display:flex;justify-content:space-between;gap:16px;color:#a99f90;font-size:11.5px}
.hero-bottom span{max-width:780px}

/* nav */
.toc{position:sticky;top:0;z-index:50;background:rgba(243,238,229,.93);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.toc-inner{max-width:1120px;margin:0 auto;padding:9px 26px;display:flex;gap:4px;overflow-x:auto;scrollbar-width:none}
.toc-inner::-webkit-scrollbar{display:none}
.toc a{flex:none;border:0;color:var(--muted);font-size:12.5px;padding:6px 11px;border-radius:999px;white-space:nowrap;transition:.18s}
.toc a:hover{color:var(--ink);background:rgba(28,37,33,.06)}
.toc a.active{background:var(--navy);color:#f7f0e4}

/* sections */
main{padding-bottom:30px}
.section{padding:66px 0 12px;border-bottom:1px solid var(--line)}
.section:last-of-type{border-bottom:0}
.kicker{font-size:11.5px;font-weight:700;letter-spacing:.16em;color:var(--red);margin:0 0 8px}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(30px,4vw,46px);line-height:1.18;letter-spacing:-.01em;margin:0 0 14px}
h3{font-family:var(--serif);font-weight:600;font-size:clamp(21px,2.3vw,27px);line-height:1.3;margin:30px 0 10px}
h4{font-size:16px;margin:0 0 6px}
.intro{max-width:880px;color:var(--ink-2);font-size:16.5px;margin:0 0 20px}
.intro b{color:var(--ink)}
.small{font-size:12.5px;color:var(--muted);line-height:1.6}
.note{font-size:12.5px;color:var(--muted);line-height:1.65}
.note b{color:var(--ink-2)}
.quickref{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:16px 0}
.quickref div{background:var(--paper-2);border:1px solid var(--line);border-radius:11px;padding:12px 14px}
.quickref b{display:block;font-family:var(--serif);font-size:16px;color:var(--ink);margin-bottom:2px}
.quickref span{font-size:11.5px;color:var(--muted);line-height:1.55}

/* tldr */
.tldr{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--red);border-radius:0 16px 16px 0;padding:24px 26px;box-shadow:0 16px 40px rgba(43,33,21,.05);margin:26px 0 4px}
.tldr-label{display:inline-block;font-size:11px;letter-spacing:.14em;color:var(--red);font-weight:700;margin-bottom:8px}
.tldr ol{margin:0;padding-left:1.35em}
.tldr li{margin:9px 0;color:var(--ink-2);font-size:15.5px}
.tldr li b{color:var(--ink)}

/* cards / rules */
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;box-shadow:0 18px 46px rgba(43,33,21,.045)}
.rules{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:22px 0 14px}
.rule{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 15px 14px;position:relative;min-height:168px;box-shadow:0 10px 28px rgba(43,33,21,.035)}
.rule .n{font-family:var(--serif);font-size:13px;color:var(--red);border:1px solid rgba(181,68,48,.25);width:25px;height:25px;display:flex;align-items:center;justify-content:center;border-radius:50%;margin-bottom:12px}
.rule h4{font-size:15px;margin:0 0 4px}
.rule .cond{font-size:13px;font-weight:600;color:var(--red);line-height:1.4;margin-bottom:6px}
.rule p{font-size:12px;color:var(--muted);line-height:1.55;margin:0}
.pool{font-size:12.5px;color:var(--muted);border-top:1px solid var(--line-2);padding-top:12px;margin-top:4px}
.pool b{color:var(--ink-2)}

.exec{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}
.exec .item{background:var(--paper-2);border:1px solid var(--line-2);border-radius:12px;padding:13px 14px}
.exec .k{font-size:11px;color:var(--muted);letter-spacing:.08em;margin-bottom:3px}
.exec .v{font-size:13.5px;font-weight:600;line-height:1.45}
.exec .v small{font-weight:400;color:var(--muted);font-size:11.5px}

.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:20px 0 8px}
.flow .step{position:relative;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 12px;text-align:center}
.flow .step:not(:last-child):after{content:"→";position:absolute;right:-9px;top:50%;transform:translateY(-50%);color:var(--line-strong,#cbbda9);font-size:13px;z-index:1}
.flow .d{font-size:11px;color:var(--muted);letter-spacing:.05em}
.flow .a{font-size:13px;font-weight:600;margin-top:2px}
.flow .step.buy{background:var(--navy);border-color:var(--navy);color:#f7f0e4}
.flow .step.buy .d{color:#b8ab97}
.flow .step.hold{background:var(--gold-soft);border-color:#e3cfab}
.t3{display:flex;align-items:stretch;gap:0;margin:18px 0 4px;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--card)}
.t3 .lab{flex:none;padding:12px 14px;background:var(--paper-2);border-right:1px solid var(--line);font-size:11.5px;color:var(--muted);display:flex;align-items:center}
.t3 .vals{display:grid;grid-template-columns:repeat(5,1fr);flex:1}
.t3 .tv{padding:10px 8px;text-align:center;border-right:1px solid var(--line-2)}
.t3 .tv:last-child{border-right:0}
.t3 .tv .k{font-size:10.5px;color:var(--muted)}
.t3 .tv .v{font-family:var(--serif);font-size:17px;font-variant-numeric:tabular-nums;line-height:1.2}
.t3 .tv.best{background:#f7e8dd}
.t3 .tv.best .v{color:var(--red);font-weight:700}

.callout{background:var(--gold-soft);border:1px solid #e5d2b2;border-radius:13px;padding:14px 17px;font-size:13.5px;color:#5c4b34;margin:16px 0}
.callout b{color:#3d2f1e}
.callout.dark{background:var(--navy);border-color:var(--navy);color:#ded3c2}
.callout.dark b{color:#fff}

/* split */
.split{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}
.split-card{border-radius:16px;padding:20px 22px;border:1px solid var(--line);background:var(--card);position:relative;overflow:hidden}
.split-card:after{content:"";position:absolute;right:-38px;bottom:-50px;width:150px;height:150px;border-radius:50%;opacity:.5}
.split-card.up{background:linear-gradient(135deg,#fffaf5,#f9e6da);border-color:#eed3c4}
.split-card.up:after{background:radial-gradient(circle,rgba(181,68,48,.18),transparent 68%)}
.split-card.down{background:linear-gradient(135deg,#f8fcf9,#e5f0ea);border-color:#cfe2d9}
.split-card.down:after{background:radial-gradient(circle,rgba(47,118,101,.16),transparent 68%)}
.split-card .big{font-family:var(--serif);font-size:40px;line-height:1;font-variant-numeric:tabular-nums;margin-bottom:6px}
.split-card.up .big{color:var(--red)}
.split-card.down .big{color:var(--green)}
.split-card .lab{font-size:13px;font-weight:600;color:var(--ink-2)}
.split-card .meta{font-size:12px;color:var(--muted);margin-top:4px}

/* annual */
.annual-chart{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px 10px;margin:14px 0;box-shadow:0 14px 38px rgba(43,33,21,.035)}
.annual-axis{position:relative;height:24px;margin-left:50px;margin-right:140px;border-bottom:1px solid var(--line)}
.annual-axis span{position:absolute;top:0;transform:translateX(-50%);font-size:10.5px;color:var(--muted);white-space:nowrap}
.annual-axis span:first-child{transform:none}
.annual-axis span:last-child{transform:translateX(-100%)}
.annual-row{display:grid;grid-template-columns:46px 1fr 70px 70px;align-items:center;gap:9px;height:35px;border-bottom:1px dashed var(--line-2)}
.annual-row:last-child{border-bottom:0}
.annual-year{font-family:var(--serif);font-size:14px;color:var(--ink-2)}
.annual-track{position:relative;height:19px}
.annual-zero{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--line-strong,#c7b9a5)}
.annual-bar{position:absolute;top:4px;height:11px;border-radius:2px;background:linear-gradient(90deg,var(--red),var(--red-2))}
.annual-tick{position:absolute;top:0;height:19px;width:2px;background:var(--ink-2);border-radius:2px}
.annual-tick:after{content:"";position:absolute;left:-2px;top:-2px;width:6px;height:6px;border-radius:50%;background:var(--ink-2)}
.annual-strat,.annual-mkt{text-align:right;font-size:12px;font-variant-numeric:tabular-nums}
.annual-strat{font-weight:600}
.annual-mkt{color:var(--ink-2)}
.annual-legend{display:flex;gap:16px;align-items:center;font-size:11.5px;color:var(--muted);margin-bottom:6px;flex-wrap:wrap}
.annual-legend i{display:inline-block;width:16px;height:7px;border-radius:2px;margin-right:5px;vertical-align:middle;background:var(--red)}
.annual-legend .mk{width:2px;height:13px;background:var(--ink-2);border-radius:2px;margin-right:6px}
.longline{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.pill{background:var(--paper-2);border:1px solid var(--line-2);border-radius:999px;padding:7px 12px;font-size:12px;color:var(--ink-2)}
.pill b{color:var(--red);font-variant-numeric:tabular-nums}

/* scene */
.scene{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:26px 26px 24px;margin:18px 0;box-shadow:0 18px 48px rgba(43,33,21,.045)}
.scene-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.scene-num{font-family:var(--serif);font-size:30px;line-height:1;color:var(--red);opacity:.75}
.scene h3{margin:0;font-size:clamp(22px,2.5vw,29px)}
.tag{font-size:11px;letter-spacing:.04em;color:var(--gold);background:var(--gold-soft);border:1px solid #e6d2ad;border-radius:999px;padding:3px 9px;white-space:nowrap}
.formula{background:var(--paper-2);border-left:3px solid var(--gold);padding:11px 14px;border-radius:0 9px 9px 0;font-size:13px;color:var(--ink-2);margin:13px 0 14px;line-height:1.6}
.formula b{color:var(--ink)}
.scene p{font-size:14.5px;color:var(--ink-2);margin:9px 0}
.scene p b{color:var(--ink)}
.scene .takeaway{background:#f8f1e5;border:1px solid #eadbc2;border-radius:11px;padding:12px 14px;font-size:13px;color:#5a4a33;margin-top:14px;line-height:1.65}
.scene .takeaway b{color:#3e301f}
.scene .warn{background:#fbf0ea;border-color:#edd3c5;color:#6b3d2e}
.scene .warn b{color:#5e2f22}

/* matrix */
.matrix{display:grid;grid-template-columns:82px 1fr 1fr;grid-template-rows:36px auto auto;gap:7px;margin:16px 0 8px}
.matrix .corner{display:flex;align-items:flex-end;justify-content:flex-end;padding:0 4px 4px;font-size:10.5px;color:var(--muted);text-align:right}
.matrix .col-head{display:flex;align-items:flex-end;justify-content:center;text-align:center;font-size:11.5px;color:var(--muted);padding:0 6px 4px;border-bottom:1px solid var(--line)}
.matrix .row-head{display:flex;align-items:center;justify-content:flex-end;text-align:right;font-size:11.5px;color:var(--muted);padding-right:5px}
.matrix-cell{position:relative;border:1px solid var(--line);border-radius:13px;padding:14px 14px 12px;min-height:154px}
.matrix-cell .cell-code{position:absolute;right:12px;top:9px;font-family:var(--serif);font-size:23px;color:rgba(28,37,33,.16);line-height:1}
.matrix-cell .cell-name{font-size:13.5px;font-weight:700;color:var(--ink);padding-right:24px;line-height:1.35}
.matrix-cell .cell-n{font-size:11.5px;color:var(--muted);margin-top:7px}
.matrix-cell .cell-n b{font-family:var(--serif);font-size:18px;font-weight:600}
.matrix-cell .cell-win{font-size:12.5px;color:var(--ink-2);margin-top:2px}
.matrix-cell .cell-ex{font-size:11px;color:var(--muted);margin-top:4px}
.matrix-cell .cell-contrib{font-size:10.5px;color:var(--muted);margin-top:7px;border-top:1px dashed rgba(114,120,111,.22);padding-top:6px}
.matrix-cell.a{background:linear-gradient(135deg,#fffaf5,#f8e7de)}
.matrix-cell.b{background:linear-gradient(135deg,#fbefe8,#f2d4c5)}
.matrix-cell.c{background:linear-gradient(135deg,#f9e8dc,#efc6b0)}
.matrix-cell.d{background:linear-gradient(135deg,#f5e3d3,#ead0b8);border-color:#d9b995}
.matrix-cell.current{outline:2px solid var(--gold);outline-offset:2px}
.current-badge{position:absolute;right:9px;bottom:9px;background:var(--navy);color:#fff;font-size:9.5px;padding:3px 7px;border-radius:999px}
.matrix-legend{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:6px}
.matrix-legend .scale{display:flex;align-items:center;gap:5px}
.matrix-legend .scale i{width:34px;height:8px;border-radius:2px;background:linear-gradient(90deg,#fff5ed,#e6bfa7);display:inline-block}

/* bar table */
.bar-table{margin:14px 0 4px}
.bar-head,.bar-row{display:grid;grid-template-columns:minmax(160px,1.25fr) minmax(130px,1.9fr) 64px 58px 92px;gap:12px;align-items:center}
.bar-head{font-size:10.5px;color:var(--muted);padding-bottom:7px;border-bottom:1px solid var(--line)}
.bar-head span:nth-child(n+3){text-align:right}
.bar-row{padding:9px 0;border-bottom:1px dashed var(--line-2)}
.bar-row:last-child{border-bottom:0}
.bar-row.current{background:linear-gradient(90deg,rgba(168,120,50,.09),transparent 72%);border-radius:8px;padding-left:8px;margin-left:-8px}
.bar-label{font-size:13.5px;font-weight:600;line-height:1.4;display:flex;flex-direction:column;gap:1px}
.bar-label small{font-size:10.5px;font-weight:400;color:var(--muted);line-height:1.4}
.mini-badge{display:inline-block;font-size:9.5px;font-weight:600;color:var(--gold);background:var(--gold-soft);border:1px solid #e5d0ab;border-radius:999px;padding:1px 6px;margin-left:6px;vertical-align:1px}
.bar-track{position:relative;height:10px;background:#f0e8db;border-radius:999px}
.bar-fill{height:10px;border-radius:999px;background:linear-gradient(90deg,var(--red),var(--red-2))}
.bar-fill.neg{background:linear-gradient(90deg,var(--green),#66a493)}
.bar-fill.flat{background:#c9c2b6}
.bar-tick{position:absolute;top:-3px;width:2px;height:16px;background:var(--ink);border-radius:2px}
.bar-tick:after{content:"";position:absolute;left:-2px;top:-2px;width:6px;height:6px;border-radius:50%;background:var(--ink)}
.bar-mean{text-align:right;font-weight:700;font-size:13px;font-variant-numeric:tabular-nums}
.bar-win{text-align:right;font-size:12.5px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.bar-ex{text-align:right;font-size:11px;color:var(--muted);line-height:1.35;font-variant-numeric:tabular-nums}
.bar-ex b{display:block;font-size:12px;color:var(--ink-2)}
.pos{color:var(--red)}
.neg{color:var(--green)}
.flat{color:var(--muted)}

/* vol / breadth */
.vol-big{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}
.vol-card{border:1px solid var(--line);border-radius:13px;padding:16px;background:var(--paper-2)}
.vol-card .k{font-size:12px;color:var(--muted)}
.vol-card .v{font-family:var(--serif);font-size:32px;line-height:1.15;margin:2px 0}
.vol-card .m{font-size:11.5px;color:var(--muted)}
.vol-card.high{background:linear-gradient(135deg,#fff8f2,#f8e5da);border-color:#eed2c2}
.vol-card.low{background:linear-gradient(135deg,#f6fbf8,#e5f0ea);border-color:#cfe2d8}
.vol-scale{position:relative;height:52px;margin:18px 0 2px;border-bottom:1px solid var(--line)}
.vol-scale .mark{position:absolute;bottom:0;transform:translateX(-50%);text-align:center;font-size:10.5px;color:var(--muted)}
.vol-scale .mark b{display:block;font-size:11.5px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.vol-scale .mark:before{content:"";display:block;width:1px;height:18px;background:var(--line-strong,#c9bca9);margin:0 auto 5px}
.vol-scale .mark.cur{color:var(--red);font-weight:700}
.vol-scale .mark.cur b{color:var(--red)}
.vol-scale .mark.cur:before{background:var(--red);height:25px;width:2px}
.breadth-chart{background:var(--paper-2);border:1px solid var(--line-2);border-radius:13px;padding:14px 15px 8px;margin:14px 0}
.breadth-head,.breadth-row{display:grid;grid-template-columns:minmax(150px,1.1fr) minmax(130px,1.7fr) 66px 86px;gap:12px;align-items:center}
.breadth-head{font-size:10.5px;color:var(--muted);padding-bottom:7px;border-bottom:1px solid var(--line)}
.breadth-head span:nth-child(n+3){text-align:right}
.breadth-row{padding:8px 0;border-bottom:1px dashed var(--line-2)}
.breadth-row:last-child{border-bottom:0}
.breadth-row.current{background:linear-gradient(90deg,rgba(168,120,50,.1),transparent 72%);border-radius:7px;padding-left:7px;margin-left:-7px}
.breadth-label{font-size:13px;font-weight:600;line-height:1.35}
.breadth-label small{display:block;font-size:10.5px;color:var(--muted);font-weight:400}
.pair-track{display:flex;flex-direction:column;gap:3px;justify-content:center;height:26px}
.pair-bar{height:7px;border-radius:999px;background:linear-gradient(90deg,var(--red),var(--red-2))}
.pair-bar.ghost{height:4px;background:#b9a58f;opacity:.7}
.breadth-val{text-align:right;font-size:12px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.35}
.breadth-val small{display:block;font-size:9.5px;color:var(--muted);font-weight:400}
.breadth-ex{text-align:right;font-size:10.5px;color:var(--muted);font-variant-numeric:tabular-nums;line-height:1.4}
.legend-inline{display:flex;gap:14px;flex-wrap:wrap;font-size:10.5px;color:var(--muted);margin:4px 0 8px}
.legend-inline i{display:inline-block;width:13px;height:6px;border-radius:2px;background:var(--red);margin-right:4px;vertical-align:middle}
.legend-inline .ghost{height:3px;background:#b9a58f}

/* vr matrix */
.vr-grid{display:grid;grid-template-columns:74px repeat(3,1fr);gap:7px;margin:14px 0}
.vr-grid .vr-corner{font-size:10.5px;color:var(--muted);display:flex;align-items:flex-end;justify-content:flex-end;padding:0 3px 4px;text-align:right}
.vr-grid .vr-head{font-size:11px;color:var(--muted);text-align:center;padding-bottom:4px;border-bottom:1px solid var(--line)}
.vr-grid .vr-rowhead{font-size:11.5px;color:var(--muted);display:flex;align-items:center;justify-content:flex-end;padding-right:3px}
.vr-cell{border:1px solid var(--line);border-radius:11px;padding:12px 12px 10px;min-height:94px;position:relative}
.vr-cell .n{font-size:10.5px;color:var(--muted)}
.vr-cell .v{font-family:var(--serif);font-size:25px;line-height:1.2;font-variant-numeric:tabular-nums}
.vr-cell .w{font-size:11.5px;color:var(--ink-2)}
.vr-cell .ex{font-size:10px;color:var(--muted);margin-top:3px}
.vr-cell.current{outline:2px solid var(--gold);outline-offset:1px}
.vr-cell .badge{position:absolute;right:8px;top:8px;font-size:9px;background:var(--navy);color:#fff;border-radius:999px;padding:2px 6px}
.vr-cell.negative{background:linear-gradient(135deg,#eef7f2,#dcebe5);border-color:#c6ddd3}
.vr-cell.negative .v{color:var(--green)}

/* pipeline */
.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
.pipe{position:relative;background:var(--paper-2);border:1px solid var(--line);border-radius:13px;padding:14px 14px 12px}
.pipe:not(:last-child):after{content:"→";position:absolute;right:-10px;top:50%;transform:translateY(-50%);color:#b7a891;font-size:13px;z-index:1}
.pipe .k{font-size:10.5px;color:var(--muted);line-height:1.4;min-height:30px}
.pipe .n{font-family:var(--serif);font-size:30px;line-height:1.05;margin:2px 0}
.pipe .m{font-size:11.5px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.pipe.current{border-color:#d8b68e;background:linear-gradient(135deg,#fbf1e4,#f4e3cd)}
.pipe.thin{border-style:dashed;opacity:.85}
.pipe .tag2{position:absolute;right:8px;top:8px;font-size:8.5px;color:var(--gold);border:1px solid #dfc8a4;border-radius:999px;padding:1px 5px}

/* 2026 */
.lens-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
.lens{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:18px 18px 16px;position:relative;overflow:hidden}
.lens:before{content:"";position:absolute;left:0;top:0;width:100%;height:3px;background:var(--red)}
.lens:nth-child(2):before{background:var(--gold)}
.lens:nth-child(3):before{background:var(--green)}
.lens .lab{font-size:11px;color:var(--muted);letter-spacing:.06em}
.lens .big{font-family:var(--serif);font-size:38px;line-height:1.1;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.lens .sub{font-size:12.5px;color:var(--ink-2);line-height:1.6}
.lens .sub b{color:var(--ink)}
.lens .tiny{font-size:10.5px;color:var(--muted);margin-top:8px;border-top:1px dashed var(--line-2);padding-top:7px}
.reasons{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin:14px 0}
.reason{background:var(--paper-2);border:1px solid var(--line-2);border-radius:12px;padding:14px 15px}
.reason .rn{font-family:var(--serif);font-size:12px;color:var(--red);margin-bottom:3px}
.reason h4{font-size:14px;margin:0 0 4px}
.reason p{font-size:12px;color:var(--muted);line-height:1.6;margin:0}
.reason p b{color:var(--ink-2)}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:16px}

/* current */
.current-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:12px;margin:18px 0 12px}
.cur-card{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:18px 19px}
.cur-card h4{font-size:13px;color:var(--muted);font-weight:600;letter-spacing:.06em;margin-bottom:8px}
.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:7px}
.metric{background:var(--paper-2);border:1px solid var(--line-2);border-radius:9px;padding:8px 10px}
.metric .k{font-size:10px;color:var(--muted);line-height:1.3}
.metric .v{font-size:13.5px;font-weight:600;line-height:1.4;font-variant-numeric:tabular-nums}
.big-signal{display:flex;align-items:baseline;gap:10px}
.big-signal .n{font-family:var(--serif);font-size:56px;line-height:1;color:var(--red)}
.big-signal .t{font-size:12px;color:var(--muted);line-height:1.5}
.top2{display:grid;gap:7px;margin-top:10px}
.top2 .row{display:flex;justify-content:space-between;align-items:center;gap:8px;border-top:1px dashed var(--line-2);padding-top:7px;font-size:12px}
.top2 .row:first-child{border-top:0;padding-top:0}
.top2 code{font-family:var(--mono);font-size:12px}
.hist-list{display:grid;gap:6px;margin-top:2px}
.hist-list .row{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:baseline;font-size:12px;border-top:1px dashed var(--line-2);padding-top:6px}
.hist-list .row:first-child{border-top:0;padding-top:0}
.hist-list .v{font-weight:700;font-variant-numeric:tabular-nums}
.hist-list .w{font-size:10.5px;color:var(--muted);min-width:38px;text-align:right}
.verdict{display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start;background:var(--navy);color:#e8ddcd;border-radius:15px;padding:18px 20px;margin:10px 0 16px}
.verdict .label{font-size:10.5px;letter-spacing:.14em;color:#d8a477;border:1px solid rgba(216,164,119,.45);border-radius:999px;padding:4px 9px;white-space:nowrap}
.verdict p{margin:0;font-size:14px;line-height:1.7;color:#e8ddcd}
.verdict b{color:#fff}
.mobile-only{display:none}
.cur-cards{display:none}
.cur-cand{background:var(--paper-2);border:1px solid var(--line-2);border-radius:11px;padding:12px 13px}
.cc-top{display:flex;justify-content:space-between;gap:10px;font-size:10.5px;color:var(--muted)}
.cc-code{font-family:var(--mono);font-size:14px;margin:3px 0 2px}
.cc-meta{font-size:10.5px;color:var(--muted)}
.cc-note{font-size:11px;color:var(--ink-2);line-height:1.55;margin:6px 0 0}

/* table */
.table-wrap{border:1px solid var(--line);border-radius:14px;overflow:auto;background:var(--card);margin:12px 0}
.scroll-hint{display:none}
table{width:100%;border-collapse:collapse;font-size:12.5px;min-width:660px}
th{background:var(--paper-2);color:var(--muted);font-weight:600;text-align:left;padding:10px 11px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 11px;border-bottom:1px solid var(--line-2);vertical-align:middle}
tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td code{font-family:var(--mono);font-size:11.5px}
.rownote{font-size:10.5px;color:var(--muted);line-height:1.5}
.badge{display:inline-block;font-size:9.5px;border-radius:999px;padding:2px 7px;line-height:1.5;white-space:nowrap}
.badge.hold{background:var(--gold-soft);color:#7b5b2e;border:1px solid #e3cba6}
.badge.fresh{background:var(--green-soft);color:#2d6a5b;border:1px solid #c4dcd3}
.badge.wait{background:#f0ece5;color:var(--muted);border:1px solid var(--line)}
.badge.split{background:#f4e5dd;color:#88402f;border:1px solid #e6c7b9}

/* boundary */
.boundary{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}
.bcard{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:16px 17px}
.bcard .k{font-size:10.5px;color:var(--red);letter-spacing:.08em;font-weight:700;margin-bottom:4px}
.bcard h4{font-size:14.5px;margin:0 0 5px}
.bcard p{font-size:12px;color:var(--muted);line-height:1.65;margin:0}
.bcard p b{color:var(--ink-2)}
details{background:var(--paper-2);border:1px solid var(--line);border-radius:13px;padding:0 16px;margin:14px 0}
summary{cursor:pointer;padding:13px 0;font-size:13px;font-weight:600;color:var(--ink-2)}
details[open] summary{border-bottom:1px solid var(--line);margin-bottom:10px}
details .dl{font-size:12px;color:var(--muted);line-height:1.7;padding-bottom:14px}
details .dl b{color:var(--ink-2)}
details .dl p{margin:6px 0}
.links{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0}
.links a{font-size:11.5px;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--ink-2);border-bottom:1px solid var(--line)}
.links a:hover{border-color:var(--gold);color:var(--red)}
.disclaimer{font-size:11.5px;color:var(--muted);line-height:1.7;margin-top:18px}
footer{background:var(--navy);color:#a99f90;padding:28px 0;font-size:11.5px}
footer .wrap{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
footer b{color:#e8ddcd}

@media(max-width:980px){
  .rules{grid-template-columns:repeat(3,1fr)}
  .exec{grid-template-columns:repeat(2,1fr)}
  .bar-head,.bar-row{grid-template-columns:minmax(140px,1.2fr) minmax(100px,1.6fr) 56px 52px 82px}
  .current-grid{grid-template-columns:1fr}
  .matrix{grid-template-columns:64px 1fr 1fr}
}
@media(max-width:760px){
  body{font-size:15px}
  .table-wrap{position:relative}
  .table-wrap:after{content:"";position:absolute;right:0;top:0;bottom:0;width:22px;background:linear-gradient(90deg,transparent,rgba(255,253,248,.96));pointer-events:none;border-radius:0 13px 13px 0}
  .scroll-hint{display:block;font-size:9px;color:var(--muted);padding:5px 9px 0;letter-spacing:.04em}
  .wrap{padding:0 17px}
  .hero{padding-top:48px}
  .hero-bottom{flex-direction:column;gap:4px}
  .stats{grid-template-columns:1fr 1fr}
  .stat{padding:15px}
  .section{padding-top:48px}
  .rules{grid-template-columns:1fr 1fr}
  .rule{min-height:0}
  .flow{grid-template-columns:1fr 1fr;overflow:hidden}
  .flow .step:after{display:none}
  .flow .step:last-child{grid-column:1/-1}
  .t3{display:block}
  .t3 .lab{border-right:0;border-bottom:1px solid var(--line)}
  .t3 .vals{grid-template-columns:repeat(5,1fr)}
  .split,.twocol,.lens-grid,.reasons,.boundary,.quickref{grid-template-columns:1fr}
  .vol-big{grid-template-columns:1fr}
  .scene{padding:20px 17px}
  .matrix{grid-template-columns:46px minmax(0,1fr) minmax(0,1fr);grid-template-rows:32px auto auto;gap:5px}
  .matrix-cell{padding:11px 10px;min-height:150px;min-width:0}
  .matrix-cell .cell-name{font-size:12px;padding-right:16px}
  .matrix-cell .cell-code{display:none}
  .mobile-only{display:inline}
  .cur-table{display:none}
  .cur-cards{display:grid;gap:8px;margin:12px 0}
  .vr-cell .badge{top:auto;bottom:7px;right:7px}
  .vr-cell{padding-bottom:24px}
  .matrix-cell .cell-n{font-size:10.5px}
  .matrix-cell .cell-ex,.matrix-cell .cell-contrib{font-size:9.5px}
  .bar-head{grid-template-columns:minmax(120px,1fr) minmax(80px,1.25fr) 50px 46px}
  .bar-head span:last-child,.bar-row .bar-ex{display:none}
  .bar-row{grid-template-columns:minmax(120px,1fr) minmax(80px,1.25fr) 50px 46px;gap:8px}
  .bar-label{font-size:12px}
  .bar-label small{font-size:9.5px}
  .bar-mean{font-size:12px}
  .bar-win{font-size:11px}
  .breadth-head,.breadth-row{grid-template-columns:minmax(92px,1fr) minmax(56px,1fr) 46px 58px;gap:5px}
  .breadth-label small{font-size:9px}
  .breadth-ex{font-size:9.5px}
  .r250-table table{min-width:0}
  .r250-table th,.r250-table td{padding:8px 6px;font-size:11px}
  .r250-table th:nth-child(4),.r250-table td:nth-child(4),
  .r250-table th:nth-child(7),.r250-table td:nth-child(7){display:none}
  .r250-table .scroll-hint{display:none}
  .vr-grid{grid-template-columns:50px repeat(3,1fr);gap:5px}
  .vr-cell{padding:9px 8px;min-height:88px}
  .vr-cell .v{font-size:21px}
  .vr-cell .ex{display:none}
  .pipeline{grid-template-columns:1fr 1fr}
  .pipe:nth-child(2):after{display:none}
  .annual-axis{margin-left:42px;margin-right:112px}
  .annual-axis span{font-size:9px}
  .annual-axis span:first-child,.annual-axis span:nth-child(3){display:none}
  .annual-row{grid-template-columns:40px 1fr 58px 58px;gap:6px}
  .annual-strat,.annual-mkt{font-size:10.5px}
  .annual-year{font-size:12px}
}
@media(max-width:380px){
  .bar-head,.bar-row{grid-template-columns:minmax(88px,1fr) minmax(56px,1fr) 44px 40px;gap:6px}
  .bar-label,.bar-mean{font-size:11.5px}
  .bar-win{font-size:10.5px}
  .breadth-head,.breadth-row{grid-template-columns:minmax(72px,1fr) minmax(44px,1fr) 40px 48px;gap:4px}
  .breadth-label{font-size:11px}
  .breadth-label small{font-size:8.5px}
  .vr-grid{grid-template-columns:44px repeat(3,minmax(0,1fr));gap:4px}
  .vr-cell{min-width:0;padding:8px 7px 22px}
  .vr-cell .n{font-size:9.5px}
  .vr-cell .v{font-size:19px}
  .vr-cell .w{font-size:10px}
}
@media(max-width:480px){
  .rules{grid-template-columns:1fr}
  .exec{grid-template-columns:1fr 1fr}
  .stat .v{font-size:28px}
  .hero h1{font-size:40px}
  .matrix{grid-template-columns:38px minmax(0,1fr) minmax(0,1fr)}
  .matrix .row-head{font-size:10px;padding-right:2px}
  .matrix-cell{min-height:170px}
  .matrix-cell .cell-name{font-size:11.5px}
  .matrix-cell .cell-n b{font-size:16px}
  .pipeline{grid-template-columns:1fr}
  .pipe:not(:last-child):after{content:"↓";right:auto;left:50%;top:auto;bottom:-12px;transform:translateX(-50%)}
  .pipe:nth-child(2):after{display:block}
}
@media print{
  body{background:#fff}
  .toc{display:none}
  .hero{background:#fff;color:#111;padding:20px 0}
  .hero h1 em{color:#9a3d2a}
  .hero .lede,.hero .lede b,.hero .lede .accent{color:#222}
  .hero-note,footer{color:#555}
  .stat{background:#f6f2ea;border:1px solid #ddd}
  .stat .v,.stat .v.red,.stat .v.green{color:#111}
  .section{break-inside:avoid;padding-top:28px}
  .scene,.card,.lens,.cur-card{box-shadow:none}
  a{color:#111}
}
"""


# ----------------------------------------------------------------------------- 页面
phase_rows = "".join(
    [
        f"""<div class="matrix">
        <div class="corner"></div>
        <div class="col-head">过去一年涨<br><b>&gt; 0</b></div>
        <div class="col-head">过去一年跌<br><b>≤ 0</b></div>
        <div class="row-head">年线上</div>
        {phase_cell("A", "牛市上升段急跌", "a")}
        {phase_cell("C", "熊市反弹", "c")}
        <div class="row-head">年线下</div>
        {phase_cell("B", "高位转弱 / 跌破年线", "b")}
        {phase_cell("D", "熊市下跌", "d")}
      </div>"""
    ]
)

phase_contrib_rows = ""
for code in "ABCD":
    r = mother_contrib[mother_contrib["阶段"].str.startswith(code)].iloc[0]
    phase_contrib_rows += f"""
      <tr>
        <td><b>{code}</b> · {r['阶段'][2:]}</td>
        <td class="num">{int(r['笔数'])}</td>
        <td class="num {pct_class(r['平均'])}">{pct(r['平均'])}</td>
        <td class="num">{r['胜率']*100:.1f}%</td>
        <td class="num">{r['利润贡献x']:.2f}x</td>
        <td class="num">{r['贡献占比']*100:.1f}%</td>
      </tr>"""

breadth_rows_html = ""
breadth_specs = [
    ("Q1", "0.029–0.119", "最低 20%"),
    ("Q2", "0.120–0.157", "次低 20%"),
    ("Q3", "0.158–0.203", "中间 20%"),
    ("Q4", "0.203–0.276", "次高 20%"),
    ("Q5", "0.277–0.676", "最高 20%"),
]
for q, rng, label in breadth_specs:
    s = breadth[q]
    full_w = s["mean"] * 100 / 8 * 100
    ex_w = (s["exmean"] or 0) * 100 / 8 * 100
    current = " current" if q == "Q3" else ""
    badge = '<span class="mini-badge">当前</span>' if q == "Q3" else ""
    breadth_rows_html += f"""
      <div class="breadth-row{current}">
        <div class="breadth-label"><span>{q}{badge}</span><small>{rng} · {label} · {s['n']} 笔</small></div>
        <div class="pair-track">
          <div class="pair-bar" style="width:{full_w:.1f}%"></div>
          <div class="pair-bar ghost" style="width:{ex_w:.1f}%"></div>
        </div>
        <div class="breadth-val {pct_class(s['mean'])}">{pct(s['mean'])}<small>胜率 {s['win']*100:.1f}%</small></div>
        <div class="breadth-ex">剔26 {pct(s['exmean'])}<br>{s['exwin']*100:.1f}%</div>
      </div>"""

signal_rows = []
for k, lab, sub in [
    ("1-2", "1–2 只", "很稀疏"),
    ("3-5", "3–5 只", "偏少"),
    ("6-15", "6–15 只", "常态"),
    ("16-40", "16–40 只", "偏多"),
    ("41-100", "41–100 只", "拥挤"),
    (">100", "> 100 只", "极端；仅 6 笔"),
]:
    s = signals[k]
    signal_rows.append(
        {
            "label": lab,
            "sub": sub,
            "mean": s["mean"],
            "win": s["win"],
            "exmean": s["exmean"],
            "exwin": s["exwin"],
            "n": s["n"],
            "current": k == ">100",
            "title": f"{lab}: {s['n']}笔",
        }
    )

mdd_rows = []
for k, lab in [("<-20%", "深跌：< -20%"), ("-20~-10%", "中间：-20% ~ -10%"), (">-10%", "浅跌：> -10%")]:
    s = mdd[k]
    mdd_rows.append(
        {
            "label": lab,
            "sub": "市场 250 日回撤",
            "mean": s["mean"],
            "win": s["win"],
            "exmean": s["exmean"],
            "exwin": s["exwin"],
            "n": s["n"],
            "current": k == "-20~-10%",
            "title": f"{lab}: {s['n']}笔",
        }
    )

depth_rows = []
for _, r in depth.iterrows():
    label = str(r["深度档"])
    current = label in ("≤-75%", "-75~-70%")
    sub = f"{int(r['笔数'])} 笔"
    if label == "≤-75%":
        sub += " · 当前 603778"
    elif label == "-75~-70%":
        sub += " · 当前 605376 / 002969"
    depth_rows.append(
        {
            "label": label,
            "sub": sub,
            "mean": float(r["平均收益"]) / 100,
            "win": float(r["胜率"]) / 100,
            "exmean": float(r["剔除2026平均"]) / 100,
            "exwin": None,
            "n": int(r["笔数"]),
            "current": current,
            "title": f"{label}: {int(r['笔数'])}笔，止盈率 {r['止盈率']:.1f}%",
        }
    )

# 当前表 + 移动端卡片
cur_rows = ""
cur_cards = ""
for _, r in current_sig.iterrows():
    code = r["代码"]
    rank = int(r["排名"])
    dd = float(r["信号日回撤_后复权"]) / 100
    # 当前表只保留 10/23 前三 + 10/22 的 605117
    if not ((r["信号日"] == "2026-10-23" and rank <= 3) or code == "605117.SH"):
        continue
    badge = ""
    if code == "603778.SH":
        badge = '<span class="badge hold">已持有</span>'
    elif code == "605376.SH":
        badge = '<span class="badge fresh">首进前二</span>'
    elif code == "002969.SZ":
        badge = '<span class="badge wait">第 3 名</span>'
    elif code == "605117.SH":
        badge = '<span class="badge split">10/22 · 除权</span>'
    note = htmllib.escape(str(r["备注"]))
    cur_rows += f"""
      <tr>
        <td>{r['信号日']}</td>
        <td class="num">{rank}</td>
        <td><code>{code}</code> {badge}</td>
        <td class="num neg">{pct(dd)}</td>
        <td class="num">{r['实际价回撤']:.1f}%</td>
        <td class="rownote">{note}</td>
      </tr>"""

    cur_cards += f"""
      <article class="cur-cand">
        <div class="cc-top"><span>{r['信号日']} · 第 {rank} 名</span><span class="neg">{pct(dd)}</span></div>
        <div class="cc-code"><code>{code}</code> {badge}</div>
        <div class="cc-meta">实际价回撤 {r['实际价回撤']:.1f}%</div>
        <p class="cc-note">{note}</p>
      </article>"""

r250_rows = ""
for _, r in r250_bins.iterrows():
    r250_rows += f"""
      <tr>
        <td><b>{r['涨幅档']}</b></td>
        <td class="num">{int(r['笔数'])}</td>
        <td class="num pos">{pct(r['单笔均值'])}</td>
        <td class="num pos">{pct(r['单笔中位'])}</td>
        <td class="num">{pct(r['胜率'], plus=False)}</td>
        <td class="num">{pct(r['止盈率'], plus=False)}</td>
        <td class="num neg">{pct(r['p10'])}</td>
      </tr>"""


def r250_row(code):
    return r250_cur.loc[r250_cur["代码"] == code].iloc[0]


r250_pear = float(r250_corr.loc["前二全部", "Pearson相关"])
r250_pear_p = float(r250_corr.loc["前二全部", "Pearson p"])
r250_spear = float(r250_corr.loc["前二全部", "Spearman相关"])
r250_spear_p = float(r250_corr.loc["前二全部", "Spearman p"])
r250_ex_pear = float(r250_corr.loc["前二_剔除2026", "Pearson相关"])
r250_mean_coef = float(r250_reg.loc["前二_全部_均值_年度FE", "估计"])
r250_mean_lo = float(r250_reg.loc["前二_全部_均值_年度FE", "CI低"])
r250_mean_hi = float(r250_reg.loc["前二_全部_均值_年度FE", "CI高"])
r250_ex_coef = float(r250_reg.loc["前二_剔除2026_均值_年度FE", "估计"])
r250_target_or = float(r250_reg.loc["前二_全部_止盈_年度FE", "估计"])
r250_target_lo = float(r250_reg.loc["前二_全部_止盈_年度FE", "CI低"])
r250_target_hi = float(r250_reg.loc["前二_全部_止盈_年度FE", "CI高"])
r250_first = r250_bins.iloc[0]


def vr_cell_html(v, r, title, is_current=False, june_tag=False):
    s = volret[f"{v}/{r}"]
    negative = s["mean"] < 0
    bg = ""
    if not negative:
        alpha = min(0.16, max(0.025, s["mean"] / 100 * 0.9))
        bg = f' style="background:linear-gradient(135deg,rgba(181,68,48,{alpha:.3f}),rgba(255,253,248,.98))"'
    badge = '<span class="badge">当前</span>' if is_current else ""
    june = '<span class="badge" style="background:#a87832;color:#fff">2026-06</span>' if june_tag else ""
    return f"""
      <div class="vr-cell{' current' if is_current else ''}{' negative' if negative else ''}"{bg}>
        {badge}{june}
        <div class="n">{title} · {s['n']} 笔</div>
        <div class="v {pct_class(s['mean'])}">{pct(s['mean'])}</div>
        <div class="w">胜率 {s['win']*100:.1f}%</div>
        <div class="ex">剔26 {pct(s['exmean'])} / {s['exwin']*100:.1f}%</div>
      </div>"""


vr_low = "".join(
    [
        vr_cell_html("低波", "<0", "低波 · 过去一年跌", True, False),
        vr_cell_html("低波", "0-10%", "低波 · 过去一年涨 0–10%", False, False),
        vr_cell_html("低波", ">10%", "低波 · 过去一年涨 >10%", False, True),
    ]
)
vr_high = "".join(
    [
        vr_cell_html("高波", "<0", "高波 · 过去一年跌", False, False),
        vr_cell_html("高波", "0-10%", "高波 · 过去一年涨 0–10%", False, False),
        vr_cell_html("高波", ">10%", "高波 · 过去一年涨 >10%", False, False),
    ]
)


BODY = f"""
<header class="hero">
  <div class="wrap hero-inner">
    <div class="eyebrow">研究笔记 · 2026-10-25 · 数据截至 2026-10-23</div>
    <h1>主板恐慌反转<br><em>策略与场景全景</em></h1>
    <p class="lede">在市场恐慌时，买入“年内仍强、短期却被砸掉 45% 以上、当天有承接”的主板股；等两个交易日，<b>第 3 天开盘买</b>；15 个交易日内反弹 20% 就止盈，不反弹就到期卖。</p>
    <p class="lede"><span class="accent">它赚的不是“牛市”，而是急跌之后 15 个交易日内的反弹。</span></p>
    <div class="hero-note">10 万本金 → 20.75 倍；后复权主口径。实际价整手修正版 40.6% / -15.9%。<b>40% 是母路径，不是任意起点保证。</b></div>
    <div class="stats">
      <div class="stat"><div class="k">年化（母路径 99 笔）</div><div class="v red">40.0%</div><div class="s">后复权；实际价修正 40.6%</div></div>
      <div class="stat"><div class="k">最大回撤</div><div class="v green">-16.0%</div><div class="s">2026-04-16 → 06-30；8/6 修复</div></div>
      <div class="stat"><div class="k">胜率 / 单笔均值</div><div class="v">71.7%</div><div class="s">99 笔；平均单笔 +6.68%</div></div>
      <div class="stat"><div class="k">持仓时间</div><div class="v">31%</div><div class="s">69% 空仓；不是全天候</div></div>
    </div>
    <div class="hero-bottom"><span>口径：沪深主板；2 仓 × 50% 权益；T+3 开盘；+20% 止盈；15 个交易日到期；无普通止损；成本 0.175%/边。</span><span>不构成投资建议</span></div>
  </div>
</header>

<nav class="toc" aria-label="目录">
  <div class="toc-inner">
    <a href="#strategy">01 规则</a>
    <a href="#source">02 收益来源</a>
    <a href="#dictionary">03 场景字典</a>
    <a href="#combo">04 组合场景</a>
    <a href="#y2026">05 2026</a>
    <a href="#current">06 当前</a>
    <a href="#limits">07 边界与口径</a>
  </div>
</nav>

<main class="wrap">

  <section class="section" id="strategy">
    <div class="kicker">01 / 策略规则</div>
    <h2>它只做一件事：等恐慌，买深跌，等反弹</h2>
    <p class="intro">每天收盘后扫一遍沪深主板。五条同时满足才进入候选；候选按“距 250 日最高价跌得多深”排序，<b>只买最深的两只</b>。不是买所有满足条件的股票。</p>

    <div class="tldr">
      <span class="tldr-label">先给结论</span>
      <ol>
        <li><b>它不是牛市策略。</b> 2018 年等权市场 -30.4%，策略 +47.4%；2022 年 -3.5% / +49.6%；2021 年慢牛 +27.7% / 策略只 +1.3%。决定单笔的，是持有 15 天里市场有没有反弹。</li>
        <li><b>历史偏好“跌得深、波动大、随后反弹”。</b> 高波单笔 +7.8% vs 低波 +2.7%；个股回撤 ≤-65% 单笔 +9.2% vs 更浅 +4.0%。但 2026 年 6 月这种“低波阴跌”会失效。</li>
        <li><b>当前是 D + 低波 + 宽度 Q3 + 123 只信号。</b> 这个环境组合历史偏弱（22 笔 +0.2% / 45.5%），但当前个股更深、信号更拥挤，样本不能直接套用；只能算中性偏弱，不是高确信。</li>
      </ol>
    </div>

    <div class="rules">
      <div class="rule"><div class="n">1</div><h4>仍强</h4><div class="cond">250 日收益 &gt; 0</div><p>过去一年仍在涨。不是随便接飞刀，而是买“长期赢家”的短期崩跌。</p></div>
      <div class="rule"><div class="n">2</div><h4>够深</h4><div class="cond">250 日回撤 &lt; -45%</div><p>距 250 日最高价至少跌 45%。跌得越深，排序越靠前。</p></div>
      <div class="rule"><div class="n">3</div><h4>超卖</h4><div class="cond">RSI14 &lt; 35</div><p>14 日相对强弱指标进入超卖区。不是“跌了就买”，要有超卖确认。</p></div>
      <div class="rule"><div class="n">4</div><h4>有承接</h4><div class="cond">收盘位置 &gt; 0.45</div><p>收盘在当日振幅上半部。0 = 收在最低，1 = 收在最高；&gt;0.45 说明盘中有资金接。</p></div>
      <div class="rule"><div class="n">5</div><h4>市场恐慌</h4><div class="cond">主板等权 20 日 &lt; -5%</div><p>不是个股独跌，而是市场一起慌。用沪深主板等权指数，不用上证指数，也不是全A。</p></div>
    </div>
    <div class="pool"><b>股票池：</b>沪深主板 600/601/603/605/000/001/002/003；上市 ≥250 日；20 日均额 ≥1000 万；剔除 5% 涨跌停样态（ST 代理）。不含创业板、科创板、北交所。</div>

    <h3>排序与执行</h3>
    <div class="exec">
      <div class="item"><div class="k">排序</div><div class="v">按 250 日回撤深度<br><small>跌得越深越靠前，取前 2</small></div></div>
      <div class="item"><div class="k">仓位</div><div class="v">2 仓 × 50% 权益<br><small>不足 2 只就留现金；无信号空仓</small></div></div>
      <div class="item"><div class="k">入场</div><div class="v">T+3 开盘<br><small>一字涨停 / 停牌不买，不递补</small></div></div>
      <div class="item"><div class="k">止盈</div><div class="v">盘中 +20%<br><small>T+1 起生效；买入当天不触发</small></div></div>
      <div class="item"><div class="k">到期</div><div class="v">15 个交易日收盘卖<br><small>跌停顺延；不是 15 个自然日</small></div></div>
      <div class="item"><div class="k">止损</div><div class="v">无普通止损<br><small>12% 止损实测把回撤扩大到 -39%</small></div></div>
      <div class="item"><div class="k">成本</div><div class="v">0.175% / 边<br><small>佣金 + 印花税 + 滑点</small></div></div>
      <div class="item"><div class="k">空仓</div><div class="v">69% 的时间<br><small>只在信号出现时出手</small></div></div>
    </div>

    <div class="flow">
      <div class="step"><div class="d">T 日收盘</div><div class="a">确认信号</div></div>
      <div class="step"><div class="d">T+1</div><div class="a">等待</div></div>
      <div class="step"><div class="d">T+2</div><div class="a">等待</div></div>
      <div class="step buy"><div class="d">T+3 开盘</div><div class="a">买入前二</div></div>
      <div class="step hold"><div class="d">之后</div><div class="a">+20% 止盈 / 15 日到期</div></div>
    </div>

    <div class="t3">
      <div class="lab">延后 N 天入场的<br>组合年化</div>
      <div class="vals">
        <div class="tv"><div class="k">N=1</div><div class="v">23.7%</div></div>
        <div class="tv"><div class="k">N=2</div><div class="v">30.3%</div></div>
        <div class="tv best"><div class="k">N=3</div><div class="v">40.0%</div></div>
        <div class="tv"><div class="k">N=4</div><div class="v">37.7%</div></div>
        <div class="tv"><div class="k">N=5</div><div class="v">27.9%</div></div>
      </div>
    </div>
    <div class="callout"><b>T+3 不是随手定的。</b> 等两个交易日让第一波情绪释放，第 3 天开盘买；组合年化在 N=3 达到峰值。但 30 个主档位只有 4 个 ≥30%，40% 是时点峰值，不是“什么时候买都行”。</div>
  </section>

  <section class="section" id="source">
    <div class="kicker">02 / 收益来源</div>
    <h2>收益来自“持有期沪深主板等权指数涨没涨”</h2>
    <p class="intro">把 450 笔完整交易按入场日到退出日的沪深主板等权指数涨跌拆开：指数涨，策略平均 <b>+9.6%</b>、胜率 <b>82%</b>；指数跌，平均 <b>-2.6%</b>、胜率 <b>36%</b>。单笔对市场是 high beta（β≈1.28，R²≈0.27）。<b>这里的“市场”不是上证指数，也不是全A，更不是个股。</b></p>
    <div class="callout"><b>口径：</b>沪深主板等权指数（和策略的市场闸门同一个指数；不含创业板、科创板、北交所）；持有期 = T+3 入场日收盘 → 实际退出日收盘（+20% 止盈日或第 15 个交易日）。股票从 T+3 开盘买，指数只有日线，所以市场用收盘价，方向和量级够用，但不是同一条价格。</div>
    <div class="split">
      <div class="split-card up"><div class="big">+9.6%</div><div class="lab">持有期等权指数上涨 · 291 笔</div><div class="meta">胜率 81.8%；策略“顺风”</div></div>
      <div class="split-card down"><div class="big">-2.6%</div><div class="lab">持有期等权指数下跌 · 159 笔</div><div class="meta">胜率 35.8%；策略“逆风”</div></div>
    </div>
    <p class="note"><b>这是事后归因，不是预测，也不是“大盘会反弹”的技术判断。</b> 买的时候不知道未来 15 天指数涨跌；它解释 2026 年 6 月为什么难受，不是提前判断的方法。个股收益是另一件事：个股可以逆势涨，也可以顺势跌；这个拆分只描述环境。</p>

    <h3>十年年度：牛熊方向解释不了</h3>
    <p>策略年度全部为正，但强弱与市场方向并不同步。最差的一年是 <b>2021 年低波动慢牛（+1.3%）</b>，不是 2018 / 2022 的熊市。</p>
    <div class="annual-chart">
      <div class="annual-legend"><span><i></i>策略账户</span><span><i class="mk"></i>主板等权（竖线标记）</span><span>星号 = 不完整年度</span></div>
      {annual_chart(annual)}
    </div>
    <p class="note"><b>读法：</b>红条是策略账户年度收益；灰黑竖线是主板等权年度收益，按同一百分比轴定位——负年份在 0 左侧，正年份在右侧。竖线落在红条内部是正常的（例如 2024 年市场 +5.3% 落在策略 +93% 的红条里），不是漏画。</p>
    <p class="note">2017 从 11-23 起；2026 截至 10-23。2026 的“完整单笔”只含 3–7 月，10 月 8 条未完成。策略账户是连续复利路径；单笔均值是另一口径，见下表。</p>

    <div class="table-wrap">
      <table>
        <thead><tr><th>年份</th><th class="num">策略账户</th><th class="num">主板等权</th><th class="num">完整单笔</th><th class="num">单笔均值</th><th class="num">胜率</th></tr></thead>
        <tbody>
          {"".join(f"<tr><td>{int(r['年份'])}{'*' if int(r['年份']) in (2017,2026) else ''}</td><td class='num {pct_class(r['策略收益'])}'>{pct(r['策略收益'])}</td><td class='num {pct_class(r['市场等权'])}'>{pct(r['市场等权'])}</td><td class='num'>{int(r['完整单笔'])}</td><td class='num {pct_class(r['单笔均值'])}'>{pct(r['单笔均值'])}</td><td class='num'>{r['胜率']*100:.1f}%</td></tr>" for _, r in annual.iterrows())}
        </tbody>
      </table>
    </div>
    <div class="longline">
      <span class="pill">1 年起点 <b>73.3%</b> 跑赢等权</span>
      <span class="pill">3 年 <b>95.2%</b></span>
      <span class="pill">5 年 <b>100%</b></span>
      <span class="pill">5 年策略中位 <b>+386.9%</b> vs 市场 +71.9%</span>
    </div>
    <p class="note">长期跑赢是 alpha，但起点高度重叠，不是 101 次独立实验；15 天级别单笔仍是 high beta。</p>
  </section>

  <section class="section" id="dictionary">
    <div class="kicker">03 / 场景字典</div>
    <h2>先把“行话”翻译成人话</h2>
    <p class="intro">下面七组条件，都是<b>信号日当天就能看到</b>的市场 / 个股状态，不是未来函数。每组先讲怎么算、怎么读，再给历史成绩。成绩来自 <b>450 笔完整的前二交易</b>，和母路径 99 笔不是同一个样本。</p>
    <p class="note">统一图例：实心条 / 粗条 = 全样本；竖线 / 细条 = 剔除 2026 后再算，用来检查结论是不是只靠 2026。这不是样本外，只是敏感性检验。<span class="mobile-only"> 移动端表格可左右滑动查看完整列。</span></p>

    <div class="quickref">
      <div><b>D 格</b><span>年线下方 + 过去一年跌；只是格子名，不是评级 D。当前就在这里。</span></div>
      <div><b>高波 / 低波</b><span>看的是沪深主板等权市场的 20 日年化波动，不是个股。当前 0.198 &lt; 信号日中位 0.241 → 低波。</span></div>
      <div><b>宽度 Q3</b><span>19% 的主板股票在 MA20 上方；把历史信号日宽度五等分，当前落在中间档 Q3（0.158–0.203）。</span></div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">01</span><h3>四个格子：年线 × 250 日涨幅</h3><span class="tag">A / B / C / D</span></div>
      <div class="formula"><b>年线</b> = 沪深主板等权指数的 250 日均线；在线上方 / 下方。<b>250 日涨幅</b> = 指数今天比 250 个交易日前涨 / 跌。两个问题各两答案，得到四格。</div>
      {phase_rows}
      <div class="matrix-legend"><span>颜色越深 = 全样本单笔均值越高；右下角 D 是当前格。</span><span class="scale"><i></i>低 → 高</span></div>
      <p><b>“D 格”只是右下角这一格，不是评级 D。</b> 它的意思是：指数在年线下方，且比一年前低——所以叫“熊市下跌”。当前 2026-10-23：年线下方，250 日 -5.15%，落在 D。</p>
      <p><b>单看 ABCD，D 不是最好的格，但贡献最大。</b> D 有 202 笔，均值 +5.7%、胜率 66.3%；B 均值 +6.4%，C 均值 +9.8% 但只有 22 笔；A 最弱 +3.0%。</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>母路径 99 笔</th><th class="num">笔数</th><th class="num">平均</th><th class="num">胜率</th><th class="num">利润贡献</th><th class="num">占比</th></tr></thead>
          <tbody>{phase_contrib_rows}
            <tr><td><b>合计</b></td><td class="num">99</td><td class="num">{pct(mother_contrib[mother_contrib['阶段']=='合计']['平均'].iloc[0])}</td><td class="num">{mother_contrib[mother_contrib['阶段']=='合计']['胜率'].iloc[0]*100:.1f}%</td><td class="num">{mother_contrib[mother_contrib['阶段']=='合计']['利润贡献x'].iloc[0]:.2f}x</td><td class="num">100%</td></tr>
          </tbody>
        </table>
      </div>
      <p class="note">利润贡献 = 每笔 <span class="mono">alloc × ret</span> 求和，单位是初始资金倍数；99 笔合计 19.71x，与终值利润 19.75x 差 0.04x（现金 / 复利路径），可视为近似归因。<b>D 贡献最大是因为笔数最多（45 笔），不是均值最高；B 均值最高；C 只有 5 笔，100% 胜率不能外推；A 最弱。</b></p>
      <div class="takeaway"><b>怎么用：</b> 不要把“D 格”当成“第四档 / 差评”，也不要把 A 当成“牛市所以更好”。剔除 2026 后：A +4.9%、B +9.7%、D +5.6%；2026 把 A 和低波拖低了。阶段标签是描述，不是实时开关。</div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">02</span><h3>高波 / 低波：看的是“市场”，不是个股</h3><span class="tag">20 日年化波动率</span></div>
      <div class="formula"><b>算法：</b> 沪深主板等权指数过去 20 个交易日日收益的标准差 × √252，折算成年化。数值越大，市场日间波动越剧烈。<b>分界：</b> 历史 450 笔信号样本的中位数 {vol_median:.3f}；低于 = 低波，高于 = 高波。</div>
      <div class="vol-scale">
        <div class="mark" style="left:23.3%"><b>0.178</b>全部日中位</div>
        <div class="mark cur" style="left:40%"><b>0.198</b>当前</div>
        <div class="mark" style="left:75.8%"><b>0.241</b>样本中位 / 分界</div>
      </div>
      <p class="note"><b>0.178 的基准：</b>全部 {len(all_vol)} 个有 20 日数据的交易日（{market['date'].min().strftime('%Y-%m-%d')} 起）的 20 日波动率中位，仅作“普通日”参照，<b>不是策略阈值</b>。把 450 笔信号交易按 0.178 再切一刀（只作观察）：低于 0.178 的 {vol_below_178['n']} 笔，{pct(vol_below_178['mean'])} / {pct(vol_below_178['win'], plus=False)}；剔26 {pct(vol_below_178['exmean'])} / {pct(vol_below_178['exwin'], plus=False)}）；0.178–0.241 是中间带（{vol_mid_band['n']} 笔，{pct(vol_mid_band['mean'])} / {pct(vol_mid_band['win'], plus=False)}，全样本最弱）；≥0.241 是高波（{vol['高波']['n']} 笔，{pct(vol['高波']['mean'])} / {pct(vol['高波']['win'], plus=False)}）。当前 0.198 在中间带。若只算策略起点 2017-11-23 以后，普通日中位约 {vol_med_after:.3f}、当前约第 {vol_pct_after*100:.0f} 百分位，结论不变。报告按 0.241 二分，不把 0.178 以下单独当策略分组。</p>
      <div class="vol-big">
        <div class="vol-card high"><div class="k">高波 · 226 笔</div><div class="v pos">{pct(vol['高波']['mean'])}</div><div class="m">胜率 {pct(vol['高波']['win'], plus=False)} · 剔除 2026 {pct(vol['高波']['exmean'])} / {pct(vol['高波']['exwin'], plus=False)}</div></div>
        <div class="vol-card low"><div class="k">低波 · 224 笔</div><div class="v">{pct(vol['低波']['mean'])}</div><div class="m">胜率 {pct(vol['低波']['win'], plus=False)} · 剔除 2026 {pct(vol['低波']['exmean'])} / {pct(vol['低波']['exwin'], plus=False)}</div></div>
      </div>
      <p><b>高波明显更好，但“低波”不是禁用条件。</b> 低波全样本只有 {pct(vol['低波']['mean'])} / {pct(vol['低波']['win'], plus=False)}，剔除 2026 后是 {pct(vol['低波']['exmean'])} / {pct(vol['低波']['exwin'], plus=False)}。它更多是“反弹力度不够”的环境，不是“必亏”。</p>
      <div class="callout warn"><b>当前 {cur['market_volatility20']:.3f} 的“低波”只在信号日语境里成立。</b> 它在全部交易日中排第 <b>{vol_pct_all*100:.0f} 百分位</b>（普通日中位 {all_vol.median():.3f}，绝对不算低），在信号日中排第 <b>{vol_pct_sig*100:.0f} 百分位</b>（偏低）。所以说“当前是低波”，必须加限定：相对历史信号日偏低，不是市场绝对安静。</div>
      <div class="takeaway"><b>怎么用：</b> 波动率比“牛熊方向”更有信息量：高波 +7.8% vs 低波 +2.7%。但这是样本内观察，会砍掉一半信号；先当描述，不要直接改成实盘开关。2026 的低波格在剔除 2026 后并不差（+4.6%）。</div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">03</span><h3>宽度 0.19 / Q3：多少股票在 20 日线上</h3><span class="tag">Breadth20</span></div>
      <div class="formula"><b>算法：</b> 沪深主板股票中，收盘价 &gt; 20 日均线的只数 ÷ 有 20 日以上数据的主板股票数。<b>0.19</b> = 19% 在 MA20 上方，81% 在下方。<b>Q1–Q5：</b> 把历史信号日的宽度从低到高五等分；Q3 = 0.158–0.203，当前 0.190 落在 Q3。</div>
      <div class="breadth-chart">
        <div class="legend-inline"><span><i></i>全样本</span><span><i class="ghost"></i>剔除 2026</span><span>右列：全样本胜率 / 剔26 均值与胜率</span></div>
        <div class="breadth-head"><span>宽度档</span><span>条形对比</span><span>均值 / 胜率</span><span>剔26</span></div>
        {breadth_rows_html}
      </div>
      <p><b>Q3 只是“中间那一档”，不是质量评级。</b> 把当前 0.190 放进历史信号日：它确实在 Q3（0.158–0.203）；但宽度与收益不是单调关系——Q1 +5.8%、Q3 +3.3%、Q5 +7.5%，两头高、中间低。</p>
      <p><b>Q3 全样本最弱，剔除 2026 后却变成 +6.6% / 70.8%。</b> 这说明 2026 的交易把 Q3 拉低了；不是 Q3 本身“有毒”。当前既是 Q3、又是 2026，两个标签叠在一起，才需要谨慎。</p>
      <div class="callout warn"><b>宽度 {cur['breadth20']:.2f} 的两种读法。</b> 相对全部交易日：第 <b>{breadth_pct_all*100:.0f} 百分位</b>，很低；相对信号日：第 <b>{breadth_pct_sig*100:.0f} 百分位</b>，接近中位。因为策略本来就在市场弱的时候出手，信号日通常就是低宽度。不要用“宽度低”单独否定当前。</div>
      <div class="takeaway"><b>怎么用：</b> 宽度是“市场里有多少股票还在上升趋势”的温度计。它适合描述环境，不适合单独做买卖开关；Q1/Q5 两头高、Q3 中间低，说明它和收益不是直线关系。</div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">04</span><h3>市场 250 日回撤：跌得越深，历史上越好</h3><span class="tag">指数的回撤</span></div>
      <div class="formula"><b>算法：</b> 沪深主板等权指数距自己过去 250 日最高点的跌幅。和“个股回撤”不是一回事；它描述的是市场整体跌了多少。</div>
      {bar_rows(mdd_rows, 10)}
      <p><b>当前 -17.5%，落在中间档（-20% ~ -10%）。</b> 这一档历史 +5.3% / 65.0%，剔除 2026 +6.7% / 70.7%。它不差，但也不是 &lt;-20% 的“深度恐慌”档（+7.8% / 71.8%）。</p>
      <div class="takeaway"><b>怎么用：</b> 深回撤是“恐慌定价”的来源，越深历史均值越高；但当前只是中间，不是极端便宜。不要因为“跌了”就自动加仓，回撤深度要和波动、宽度、个股深度一起看。</div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">05</span><h3>当日信号数：门槛有多松，不是你买几只</h3><span class="tag">10/23 123 只</span></div>
      <div class="formula"><b>算法：</b> 当天有多少只股票同时满足五条。<b>不是买入数量</b>——组合永远最多买前 2 只。信号数越多，说明“五条门槛”越松、市场越拥挤。</div>
      {bar_rows(signal_rows, 20)}
      <p><b>当前 123 只，是极端值。</b> 历史上 &gt;100 只只有 {big100_n} 笔：2022-04-27 2 笔、2026-07 4 笔，{big100_target} 笔止盈、{big100_time} 笔到期，全部为正；但 n={big100_n}，没有统计意义。41–100 只全样本 +3.9% / 55.1%，剔除 2026 后 +9.9% / 73.9%——2026 把这一档拖低了。</p>
      <p><b>信号多不等于机会多。</b> 它只说明“最新信号日很容易筛出一堆票”；真正决定买哪两只的，还是回撤深度排序。</p>
      <div class="takeaway"><b>怎么用：</b> 把信号数当“拥挤度 / 门槛松紧”的描述，不要当收益预测。当前 123 只，最接近的历史样本只有 6 笔，任何“历史同档 +16%”的说法都不可靠。</div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">06</span><h3>个股回撤深度：策略真正用来排序的维度</h3><span class="tag">后复权 · 250 日</span></div>
      <div class="formula"><b>算法：</b> 个股现价（后复权）距自己过去 250 日最高价的跌幅，越深排名越前。<b>它不是额外过滤条件，而是选股排序键。</b> 用后复权，避免分红 / 送转把“实际价回撤”放大或缩小。</div>
      {bar_rows(depth_rows, 22)}
      <p><b>深组明显更好：</b> ≤-65% 共 109 笔，均值 +9.21%、胜率 77.1%；&gt;-65% 共 341 笔，均值 +3.99%、胜率 61.9%。控制年度后，每深 10 个百分点，单笔均值约多 <b>+3.95pp</b>（t=5.0）；再控制市场回撤，仍有 <b>+3.20pp</b>（t=4.0）。</p>
      <p><b>当前两只都在最深档：</b> 603778 -78.6%（≤-75%，历史 8 笔 +19.7% / 100%）；605376 -71.3%（-75~-70%，历史 29 笔 +9.7% / 86.2%）。002969 -70.7% 也在 -75~-70%，但它是第 3 名。</p>
      <div class="callout warn"><b>深度不是免死金牌。</b> 2026 年 6 月：深组 9 笔平均 +0.3%，浅组 29 笔 -9.3%；7 月：深组 24 笔 +12.2%，浅组 +6.1%。深度在反弹来时放大收益，在阴跌里也会让你买得更集中。≤-75% 只有 8 笔，+19.7% 不能当保证。</div>
      <div class="takeaway"><b>怎么用：</b> 这是所有维度里最“策略原生”的一个：它不是事后加的开关，而是每天排序的依据。历史关系稳定，但样本小、环境依赖强；当前两只深，只说明“排序分高”，不说明“必涨”。</div>
    </div>

    <div class="scene">
      <div class="scene-head"><span class="scene-num">07</span><h3>250 日涨幅：涨得越多越赚吗？</h3><span class="tag">规则已有：&gt; 0</span></div>
      <div class="formula"><b>算法：</b> 信号日后复权收盘 / 250 个交易日前收盘 - 1；250 个交易日约 12 个月，不是 365 个自然日。策略只要求 &gt;0，<b>不按涨幅大小排序</b>。</div>
      <div class="table-wrap r250-table">
        <table>
          <thead><tr><th>250日涨幅</th><th class="num">笔数</th><th class="num">单笔均值</th><th class="num">中位</th><th class="num">胜率</th><th class="num">止盈率</th><th class="num">p10</th></tr></thead>
          <tbody>{r250_rows}</tbody>
        </table>
      </div>
      <p><b>均值不单调，止盈率单调上升。</b> 前二 450 笔：Pearson {r250_pear:.3f}（p={r250_pear_p:.3f}），Spearman {r250_spear:.3f}（p={r250_spear_p:.3f}）；每 +100pp 的均值斜率 <b>{r250_mean_coef:+.2f}pp</b>（95%CI {r250_mean_lo:+.2f}~{r250_mean_hi:+.2f}），区间跨 0；止盈 OR <b>{r250_target_or:.2f}</b>（{r250_target_lo:.2f}~{r250_target_hi:.2f}）。</p>
      <p><b>剔除 2026 后，均值关系消失。</b> Pearson {r250_ex_pear:.3f}，均值斜率 {r250_ex_coef:+.2f}pp；0–10% 档的弱主要是 2026 年：2026 该档 <b>{pct(r250_first['2026均值'])} / {pct(r250_first['2026胜率'], plus=False)}</b>，剔除后 <b>{pct(r250_first['剔除2026均值'])} / {pct(r250_first['剔除2026胜率'], plus=False)}</b>。所以不要硬排除 0–10%。</p>
      <div class="callout warn"><b>怎么用：</b> 保留 r250 &gt; 0 的门槛，<b>不要加“涨得多就买 / 涨得少就排除”</b>；排序仍按 250 日回撤深度。高涨幅更容易摸 +20%，但 p10 从 -11.2% 到 -18.8%，左尾更深；个股波动率比 r250 更有区分度（Q1→Q5 均值 +0.9%→+8.3%）。</div>
      <p class="note">当前 603778 <b>+{r250_row('603778.SH')['250日涨幅(%)']:.1f}%</b>（{r250_row('603778.SH')['涨幅档']} 档，同档 +{r250_row('603778.SH')['同档均值(%)']:.1f}% / {r250_row('603778.SH')['同档胜率(%)']:.1f}% / {r250_row('603778.SH')['同档止盈率(%)']:.1f}%）；605376 <b>+{r250_row('605376.SH')['250日涨幅(%)']:.1f}%</b>（{r250_row('605376.SH')['涨幅档']} 档，同档 +{r250_row('605376.SH')['同档均值(%)']:.1f}% / {r250_row('605376.SH')['同档胜率(%)']:.1f}% / {r250_row('605376.SH')['同档止盈率(%)']:.1f}%）。前二都不在 0–10% 弱档，但不改变当前 D + 低波 + Q3 的环境判断。详见 <a href="10_250日涨幅与收益/00_一页纸_250日涨幅与收益.md">250日涨幅与收益</a>。</p>
    </div>
  </section>

  <section class="section" id="combo">
    <div class="kicker">04 / 组合场景</div>
    <h2>条件叠在一起，样本会迅速变小</h2>
    <p class="intro">单看每个条件都有“好 / 坏”的故事，但实盘是同时发生。下面两组组合更接近“当前”：一组是<b>波动 × 市场 250 日涨幅</b>，一组是<b>当前相似条件的逐层叠加</b>。</p>

    <h3>波动 × 市场 250 日涨幅</h3>
    <div class="vr-grid">
      <div class="vr-corner">行：波动<br>列：250日涨幅</div>
      <div class="vr-head">过去一年跌<br>&lt; 0</div>
      <div class="vr-head">涨 0–10%</div>
      <div class="vr-head">涨 &gt;10%</div>
      <div class="vr-rowhead">低波</div>
      {vr_low}
      <div class="vr-rowhead">高波</div>
      {vr_high}
    </div>
    <p class="note">表内为单笔均值 / 胜率；“剔26”为剔除 2026 后的均值 / 胜率。当前在“低波 / 过去一年跌”格：96 笔 +3.95% / 59.4%。</p>
    <p><b>最差的一格是“低波 + 市场过去一年仍在涨”（-0.8% / 44.6%）。</b> 这不是恐慌底，更像滚动做顶：市场还没跌够，但个股已经开始崩。2026 年 6 月正落在这里；剔除 2026 后该格 +4.5% / 65.2%，说明它是“特定年份的陷阱”，不是永远差。</p>
    <p><b>当前“低波 / 过去一年跌”不是最差格，但只是中等：</b> 96 笔 +3.95% / 59.4%；比高波 / 过去一年跌（+7.65% / 75.8%）弱一截。它和“D 格 + 低波”（90 笔 +3.07% / 56.7%）互相印证：环境分数不高。</p>

    <h3>当前相似组合：越叠越薄</h3>
    <div class="pipeline">
      <div class="pipe"><div class="k">D 格</div><div class="n">202</div><div class="m">+5.66% · 66.3%</div></div>
      <div class="pipe"><div class="k">+ 低波</div><div class="n">90</div><div class="m">+3.07% · 56.7%</div></div>
      <div class="pipe current"><span class="tag2">近似当前</span><div class="k">+ 宽度 Q3</div><div class="n">22</div><div class="m">+0.17% · 45.5%</div></div>
      <div class="pipe thin"><div class="k">+ 个股深度 &lt;-70%</div><div class="n">1</div><div class="m">+19.58% · 100%</div><div class="m">样本不足，不可用</div></div>
    </div>
    <p><b>22 笔“D + 低波 + Q3”分布在 2017 / 2018 / 2022 / 2024，没有 2026 样本；平均 +0.17%、中位 -1.52%、胜率 45.5%。</b> 再加“个股 &lt;-70%”只剩 1 笔（2018 年，+19.58%），不能拿来外推。</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>组合</th><th class="num">笔数</th><th class="num">均值</th><th class="num">中位</th><th class="num">胜率</th><th>年份</th></tr></thead>
        <tbody>
          <tr><td>D 格</td><td class="num">202</td><td class="num pos">+5.66%</td><td class="num">+5.03%</td><td class="num">66.3%</td><td>2017–2026</td></tr>
          <tr><td>D + 低波</td><td class="num">90</td><td class="num pos">+3.07%</td><td class="num">+1.68%</td><td class="num">56.7%</td><td>2017–2024</td></tr>
          <tr><td>D + 低波 + 宽度 Q3</td><td class="num">22</td><td class="num pos">+0.17%</td><td class="num neg">-1.52%</td><td class="num">45.5%</td><td>2017/2018/2022/2024</td></tr>
          <tr><td>D + 低波 + 宽度 0.15–0.22</td><td class="num">36</td><td class="num pos">+2.36%</td><td class="num">+0.23%</td><td class="num">50.0%</td><td>2017/2018/2022/2024</td></tr>
          <tr><td>D + 低波 + Q3 + 个股 &lt;-70%</td><td class="num">1</td><td class="num pos">+19.58%</td><td class="num">+19.58%</td><td class="num">100%</td><td>2018</td></tr>
        </tbody>
      </table>
    </div>
    <div class="callout warn"><b>当前和 22 笔“像”在 D / 低波 / Q3，“不像”在另外两件事：</b> 这 22 笔的信号数中位只有 3 只（最多 17），个股回撤中位 -60.2%（最深 -70.8%）；当前是 <b>123 只信号</b>、两只回撤 <b>-78.6% / -71.3%</b>。所以不能把 +0.17% 直接套到当前，也不能因为只有 1 笔深组合就反推会涨。<b>环境分数偏弱，个股深度分数偏高，合起来是中性偏弱，不是高确信。</b></div>
  </section>

  <section class="section" id="y2026">
    <div class="kicker">05 / 2026</div>
    <h2>2026 不是全年都差，是“6 月那一簇差 + 起点差异大”</h2>
    <p class="intro">同一个策略，三个口径会给出三种体感。先把它们分开，再说 6 月为什么特别差。</p>
    <div class="lens-grid">
      <div class="lens"><div class="lab">连续账户 · 2017 起</div><div class="big pos">{pct(acct_2026)}</div><div class="sub">2026 年至 10-23（精确 {pct(acct_2026,2)}）；11 笔，均 +4.0%，胜率 63.6%。</div><div class="tiny">4/16 → 6/30 回撤 <b>-16.0%</b>，9 年最大；8/6 修复，10/20 新高。</div></div>
      <div class="lens"><div class="lab">2026 年每个信号日起点</div><div class="big">{pct(st26_med)}</div><div class="sub">{st26_n} 个起点，中位收益；<b>{st26_pos} 正 / {st26_neg} 负 / {st26_zero} 未入场</b>。</div><div class="tiny">最好 {st26_best_date} <b>{pct(st26_best['total_ret'])}</b>；最差 {st26_worst_date} <b>{pct(st26_worst['total_ret'])}</b>。短窗口，不能和 40% 年化直接比。</div></div>
      <div class="lens"><div class="lab">2026 所有信号逐只</div><div class="big">{pct(s26['mean'])}</div><div class="sub">{s26['n']} 笔完整；胜率 <b>{pct(s26['win'], plus=False)}</b>。3–7 月。</div><div class="tiny">6 月 {june_s['n']} 笔 <b>{pct(june_s['mean'])} / {pct(june_s['win'], plus=False)}</b>；7 月 {july_s['n']} 笔 <b>{pct(july_s['mean'])} / {pct(july_s['win'], plus=False)}</b>。</div></div>
    </div>

    <h3>6 月为什么特别差</h3>
    <div class="reasons">
      <div class="reason"><div class="rn">01</div><h4>不是一次性恐慌，是滚动下跌</h4><p>等权指数 2/27 见顶 2.112，7/24 见底 1.663（<b>-21.3%</b>）；6/23 起跌破 250 日线并一直在下方。策略等的是 V 型反弹，不是慢慢阴跌。</p></div>
      <div class="reason"><div class="rn">02</div><h4>250 日涨幅过滤器“变旧”了</h4><p>6 月信号日的市场 250 日涨幅中位数仍有 <b>+19.7%</b>，因为 2025 年的大涨还在窗口里。策略仍在买“过去一年的赢家”，但这些股票已经进入下跌段。</p></div>
      <div class="reason"><div class="rn">03</div><h4>15 天内等权指数没有上涨</h4><p>6 月 {june_s['n']} 笔的持有期等权指数平均 <b>{pct(june_hold)}</b>，<b>{pct(june_hold_neg, plus=False)}</b> 为负；7 月是 <b>{pct(july_hold)}</b>，只有 {pct(july_hold_neg, plus=False)} 为负。15 日时间止损在反弹前就卖了。</p></div>
      <div class="reason"><div class="rn">04</div><h4>低波 + 过去一年涨：最差的一格</h4><p>6 月波动率中位只有 <b>{june_vol:.3f}</b>（2017–2025 信号日中位 {pre_vol:.3f}），信号数中位 <b>{june_nsig:.0f} 只</b>；对应“低波 / 市场 250 日涨幅 &gt;10%”格：<b>-0.8% / 44.6%</b>。是“宽而浅”的阴跌，不是高波恐慌。</p></div>
    </div>

    <div class="twocol">
      <div class="callout"><b>一天之差，差 {abs(st26_0603-st26_0604)*100:.0f} 个百分点。</b> 6/3 起点期末 <b>{pct(st26_0603)}</b>；6/4 起点 <b>{pct(st26_0604)}</b>。6/3 的两只在 6/29 收盘到期，6/30 开盘正好空出仓位，买到 6/25 信号的 600829/600408；6/4 的两只 6/30 收盘才到期，只能等 7/1 买 6/26 信号的 002342/603878。这不是“2026 年策略失效”，是 <b>2 仓、15 天、无递补</b> 的路径依赖。</div>
      <div class="callout warn"><b>没有普通止损。</b> 6/26 的 002342 在 15 个交易日里从 10.61 跌到 7.13（约 <b>-33%</b>），是全部 450 笔里最差的一笔。母路径没有买到它，但其他起点买到了。2 仓集中 + 无止损，是这套策略最大的尾部风险。</div>
    </div>

    <h3>同口径 60 日：2026 确实低于常态</h3>
    <div class="table-wrap">
      <table>
        <thead><tr><th>起点后 60 个交易日</th><th class="num">起点数</th><th class="num">中位</th><th class="num">亏损占比</th><th class="num">最差</th><th class="num">最差回撤</th></tr></thead>
        <tbody>
          <tr><td>历史 2017–2025</td><td class="num">{int(fh60.loc['历史 2017-2025','起点数'])}</td><td class="num">{pct(fh60.loc['历史 2017-2025','中位'])}</td><td class="num">{pct(fh60.loc['历史 2017-2025','亏损占比'], plus=False)}</td><td class="num">{pct(fh60.loc['历史 2017-2025','最差'])}</td><td class="num neg">{pct(fh60.loc['历史 2017-2025','最差回撤'])}</td></tr>
          <tr><td><b>2026</b></td><td class="num">{int(fh60.loc['2026','起点数'])}</td><td class="num"><b>{pct(fh60.loc['2026','中位'])}</b></td><td class="num"><b>{pct(fh60.loc['2026','亏损占比'], plus=False)}</b></td><td class="num neg"><b>{pct(fh60.loc['2026','最差'])}</b></td><td class="num neg"><b>{pct(fh60.loc['2026','最差回撤'])}</b></td></tr>
        </tbody>
      </table>
    </div>
    <p class="note">2026 的 60 日中位在历史第 <b>33</b> 百分位；20 日中位在第 27 百分位。但 3–4 月起点满 120 日的 {int(fh120.loc['2026','起点数'])} 个<b>全部为正</b>，中位 <b>{pct(fh120.loc['2026','中位'])}</b>。2026 是“前 1–3 个月差、4 个月后修复”的一年，不是全年单边亏损。</p>
    <p class="note">2026 全年：持有期等权指数平均 <b>{pct(c26_hold)}</b>（历史 {pct(pre_hold)}）；负收益占比 <b>{c26_hold_neg*100:.0f}%</b>（历史 {pre_hold_neg*100:.0f}%）。</p>
    <div class="callout"><b>2026 的 {s26['n']} 笔里：A {int(c26_phase.loc['A','n'])} 笔 {pct(c26_phase.loc['A','mean'])} / {pct(c26_phase.loc['A','win'], plus=False)}，B {int(c26_phase.loc['B','n'])} 笔 {pct(c26_phase.loc['B','mean'])} / {pct(c26_phase.loc['B','win'], plus=False)}，D 只有 {int(c26_phase.loc['D','n'])} 笔 {pct(c26_phase.loc['D','mean'])} / {pct(c26_phase.loc['D','win'], plus=False)}。</b> 6 月主要是 A/B，不是 D。所以“当前 D 格”不能直接类比 2026 年 6 月；反过来，当前的环境分数也不能只靠“D 格”判断。</div>
  </section>

  <section class="section" id="current">
    <div class="kicker">06 / 当前</div>
    <h2>最新数据日（2026-10-23）落在什么场景</h2>
    <p class="intro">市场：<b>D 格</b>；低于年线 8.7%；250 日 -5.2%；250 日回撤 -17.5%；20 日 -5.2%（闸门开）；波动 <b>0.198</b>；宽度 <b>19.0%</b>（Q3）。信号：<b>123 只</b>通过五条，前二为 <b>603778 / 605376</b>。</p>
    <div class="current-grid">
      <div class="cur-card">
        <h4>市场状态</h4>
        <div class="metrics">
          <div class="metric"><div class="k">等权指数</div><div class="v">{cur['equal_weight_index']:.4f}</div></div>
          <div class="metric"><div class="k">250 日线</div><div class="v">{cur['ma250']:.4f}</div></div>
          <div class="metric"><div class="k">年线位置</div><div class="v neg">{pct(cur['equal_weight_index']/cur['ma250']-1)}</div></div>
          <div class="metric"><div class="k">250 日涨幅</div><div class="v neg">{pct(cur['mkt_ret250'])}</div></div>
          <div class="metric"><div class="k">250 日回撤</div><div class="v neg">{pct(cur['mkt_dd250'])}</div></div>
          <div class="metric"><div class="k">20 日涨幅（闸门）</div><div class="v neg">{pct(cur['market_return20'])}</div></div>
          <div class="metric"><div class="k">20 日波动</div><div class="v">{cur['market_volatility20']:.4f}</div></div>
          <div class="metric"><div class="k">宽度 20</div><div class="v">{pct(cur['breadth20'], plus=False)}</div></div>
        </div>
        <p class="note">波动 {cur['market_volatility20']:.3f} 在全部交易日第 {vol_pct_all*100:.0f} 百分位，在信号日第 {vol_pct_sig*100:.0f} 百分位；宽度 {cur['breadth20']:.3f} 在全部交易日第 {breadth_pct_all*100:.0f} 百分位，在信号日第 {breadth_pct_sig*100:.0f} 百分位。两个“低”都只在信号日语境里成立。</p>
      </div>
      <div class="cur-card">
        <h4>最新信号日</h4>
        <div class="big-signal"><div class="n">123</div><div class="t">只通过五条<br>只买最深 2 只</div></div>
        <div class="top2">
          <div class="row"><span><code>603778.SH</code> <span class="badge hold">已持有</span></span><span class="neg">-78.6%</span></div>
          <div class="row"><span><code>605376.SH</code> <span class="badge fresh">首进前二</span></span><span class="neg">-71.3%</span></div>
          <div class="row"><span><code>002969.SZ</code> <span class="badge wait">第 3 名</span></span><span class="neg">-70.7%</span></div>
        </div>
        <p class="note">603778 已在 10/15 信号、10/20 入场；10/23 是新一轮第 1 天，不是第 2/3 次，但组合层面不重复买。605376 是首次进前二。按规则对应 10/28 开盘；一字涨停 / 停牌不买。</p>
      </div>
    </div>
    <div class="cur-card">
      <h4>历史同场景：环境弱、个股深</h4>
      <div class="hist-list">
        <div class="row"><span>D 格</span><span class="v pos">+5.66%</span><span class="w">66.3%</span></div>
        <div class="row"><span>D + 低波</span><span class="v pos">+3.07%</span><span class="w">56.7%</span></div>
        <div class="row"><span>D + 低波 + 宽度 Q3</span><span class="v pos">+0.17%</span><span class="w">45.5%</span></div>
        <div class="row"><span>个股回撤 ≤-65%（不分环境）</span><span class="v pos">+9.21%</span><span class="w">77.1%</span></div>
        <div class="row"><span>2026 年 D 格（仅 2 笔）</span><span class="v pos">+13.0%</span><span class="w">100%</span></div>
      </div>
      <p class="note">当前是“环境组合偏弱 + 个股深度偏强”的混合体。22 笔相似组合的 +0.17% 不能直接套用，因为当前信号数是 123（相似样本中位 3）、个股回撤 -78.6/-71.3（相似样本中位 -60.2）。</p>
    </div>
    <div class="verdict"><span class="label">判断</span><p><b>不是高确信。</b> 环境分数偏弱（D + 低波 + Q3 历史 +0.2% / 45.5%），个股深度分数偏高（≤-65% 历史 +9.2% / 77.1%），但两者叠加的历史样本不足。若参与，按规则轻仓 / 半仓，不因为“跌得深”加仓；若不想承担 2 仓集中 + 无止损的尾部风险，空仓等更清晰的环境也合理。</p></div>

    <h3>当前候选逐只</h3>
    <div class="table-wrap cur-table">
      <table>
        <thead><tr><th>信号日</th><th class="num">排名</th><th>代码</th><th class="num">250日回撤（后复权）</th><th class="num">实际价回撤</th><th>备注</th></tr></thead>
        <tbody>{cur_rows}</tbody>
      </table>
    </div>
    <div class="cur-cards">{cur_cards}</div>
    <p class="note">605117 是 10/22 第 1 名；10/23 未进前二。它的实际价回撤 -61.1% 受 10 转 4 除权影响，信号看后复权 -45.3%。这也是为什么口径必须区分“后复权信号”和“实际价成交”。第 4/5 名 603256、002796 也在 -70% 左右，但不在前二。</p>
  </section>

  <section class="section" id="limits">
    <div class="kicker">07 / 边界与口径</div>
    <h2>40% 是母路径的成绩，不是任意起点的保证</h2>
    <p class="intro">这套策略的历史很漂亮，但漂亮的数字有明确边界。下面四件事，比“今年赚多少”更重要。</p>
    <div class="boundary">
      <div class="bcard"><div class="k">01 / 路径</div><h4>40% 是特定路径的复利</h4><p>母路径 99 笔、10 万起、2 仓连续滚动；450 笔信号样本单笔均值 <b>+5.26%</b>、胜率 65.6%。同一段历史换不同起点，结果差很多；按 <b>25%–30%</b> 规划更稳。</p></div>
      <div class="bcard"><div class="k">02 / 幸存者</div><h4>本地库不含退市股</h4><p>策略专挑“年内仍强、短期暴跌”的股票，退市 / 长期停牌会被系统性漏掉。若 99 笔中 1 笔因退市变 -100%（半仓），年化约从 40% 降到 <b>29.5%</b>；2 笔约 <b>19.7%</b>。</p></div>
      <div class="bcard"><div class="k">03 / 尾部</div><h4>2 仓集中 + 无普通止损</h4><p>母路径最差单笔 -17.2%，但 450 笔全样本最差 <b>-33.0%</b>；251 个起点最差 60 日回撤 <b>-38.3%</b>。40% / -16% 是那条路径的成绩，不是任意起点的风险上限。</p></div>
      <div class="bcard"><div class="k">04 / 场景</div><h4>阶段是描述，不是实时开关</h4><p>ABCD、波动、宽度都是信号日截面；剔除 2026 只是敏感性检验，不是样本外。高波 / 深回撤的差异是样本内观察，直接改成实盘过滤会砍掉一半信号，且 2026 的低波格剔除后并不差。</p></div>
    </div>

    <details>
      <summary>数据口径与复现</summary>
      <div class="dl">
        <p><b>样本：</b>2017-11-23 至 2026-10-23，沪深主板；251 个信号日、459 条信号日逐只、450 条完整；2026 年 49 个信号日、92 条逐只、84 条完整（3–7 月），10 月 8 条未完成。</p>
        <p><b>价格：</b>信号与排序用后复权；实际价整手修正版另算（40.6% / -15.9%）。605117 的 10 转 4 会让实际价回撤显得更深，信号仍看后复权。</p>
        <p><b>执行：</b>信号日收盘确认；T+3 开盘买；2 仓各 50% 当前权益；+20% 盘中止盈（T+1 生效）；15 个交易日收盘到期；跌停顺延；一字涨停 / 停牌不买、不递补；无普通止损。</p>
        <p><b>成本：</b>佣金万 2.5（最低 5 元）+ 印花税千 0.5（卖）+ 滑点千 1（单边），约 0.175% / 边。</p>
        <p><b>市场：</b>沪深主板等权指数，来自本地同花顺 DuckDB 派生的 <span class="mono">market_daily.parquet</span>；市场闸门用同一指数的 20 日收益，不用上证指数。宽度 = 沪深主板股票中收盘 &gt; MA20 的占比（至少 20 日数据）。<b>持有期市场</b> = 沪深主板等权指数在入场日收盘 → 退出日收盘的收益；只用于事后归因，不是买入信号。</p>
        <p><b>场景：</b>ABCD 在信号日计算：年线上下 × 250 日涨幅正负。波动 = 20 日等权日收益标准差 × √252，以 450 笔信号样本中位数 0.241 分高低。宽度 Q1–Q5 为信号日宽度五等分。剔除 2026 = 只算 2017–2025，属于敏感性检验，不是 OOS。</p>
        <p><b>个股深度：</b>后复权收盘 / 250 日最高 -1；09 的深度分档使用 450 笔前二完整交易。回归为年度固定效应，括号内为 t 值；控制市场回撤后为 +3.20pp（t=4.0）。</p>
        <p><b>数据源：</b>本地 HiThink DuckDB → <span class="mono">research_panel.parquet</span>；不请求 baostock / Tushare 等第三方。历史不代表未来；先纸面 50 笔，再小资金。</p>
      </div>
    </details>

    <div class="links">
      <a href="07_市场阶段适配/00_一页纸_市场阶段适配.md">一页纸 · 市场阶段</a>
      <a href="07_市场阶段适配/01_完整报告_市场阶段适配.md">完整报告 · 市场阶段</a>
      <a href="07_市场阶段适配/03_结果/场景统计.csv">场景统计 CSV</a>
      <a href="07_市场阶段适配/03_结果/阶段贡献.csv">阶段贡献 CSV</a>
      <a href="07_市场阶段适配/市场阶段适配.xlsx">场景数据 Excel</a>
      <a href="09_信号日回撤深度与收益/信号日回撤深度与收益.xlsx">深度分析 Excel</a>
      <a href="09_信号日回撤深度与收益/03_结果/前二深度分档.csv">深度分档 CSV</a>
      <a href="09_信号日回撤深度与收益/README.md">深度分析说明</a>
      <a href="08_重复信号与买入时点/03_结果/当前信号_250日回撤.csv">当前回撤 CSV</a>
      <a href="../2026-10-24_主板恐慌反转_年化30/00_一页纸_恐慌洗盘反转.md">母策略一页纸</a>
      <a href="10_250日涨幅与收益/00_一页纸_250日涨幅与收益.md">250日涨幅专题</a>
      <a href="10_250日涨幅与收益/250日涨幅与收益.xlsx">250日涨幅 Excel</a>
      <a href="07_市场阶段适配/02_代码/make_html_report.py">本页生成脚本</a>
    </div>
    <p class="disclaimer">本文是研究记录，不是投资建议。场景统计是样本内描述；历史收益不代表未来。当前信号只说明“按规则最新数据日会选到谁”，不保证上涨。先纸面 50 笔，再考虑小资金。</p>
  </section>
</main>

<footer>
  <div class="wrap"><span><b>主板恐慌反转 · 策略与场景全景</b> · 2026-10-25</span><span>数据截至 2026-10-23 · 本地同花顺 DuckDB · 不构成投资建议</span></div>
</footer>
"""


BODY = BODY.replace(
    '<div class="table-wrap">',
    '<div class="table-wrap"><span class="scroll-hint">左右滑动查看完整表格 →</span>',
)

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>主板恐慌反转 · 策略与场景全景</title>
<style>{CSS}</style>
</head>
<body>
{BODY}
<script>
(() => {{
  const links = [...document.querySelectorAll('.toc a')];
  const ids = links.map(a => a.getAttribute('href').slice(1));
  const sections = ids.map(id => document.getElementById(id)).filter(Boolean);
  const io = new IntersectionObserver((entries) => {{
    const visible = entries.filter(e => e.isIntersecting).sort((a,b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + visible.target.id));
  }}, {{ rootMargin: '-20% 0px -65% 0px', threshold: [0, .1, .25, .5] }});
  sections.forEach(s => io.observe(s));
}})();
</script>
</body>
</html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
