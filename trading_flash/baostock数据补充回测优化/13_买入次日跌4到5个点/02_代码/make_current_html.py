#!/usr/bin/env python3
"""生成 603778 当前案例（2026-10-30）HTML。"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_结果"
HTML = BASE / "当前案例_603778_2026-10-30.html"

CSS = """
:root{--ink:#18202a;--muted:#667085;--line:#e7eaf0;--bg:#f6f7f9;--card:#fff;
--red:#d64545;--red-soft:#fbeaea;--green:#1f8a5b;--green-soft:#e9f6ef;
--blue:#2f5f9e;--blue-soft:#eaf1f9;--amber:#b7791f;--amber-soft:#fdf3e3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
line-height:1.65;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 56px}
header{background:linear-gradient(135deg,#2a1c1c,#4a2a2a);color:#fff;border-radius:16px;padding:30px 32px;margin-bottom:18px}
header .tag{display:inline-block;font-size:12px;letter-spacing:.08em;color:#f0c9c9;border:1px solid rgba(255,255,255,.25);padding:3px 9px;border-radius:999px;margin-bottom:14px}
h1{font-size:30px;line-height:1.3;margin:0 0 10px;font-weight:700}
header p{margin:0;color:#f0c9c9;font-size:14px}
.callout{background:#fff;border:1px solid var(--line);border-left:5px solid var(--red);border-radius:12px;padding:18px 20px;margin:16px 0 8px}
.callout b{color:var(--red)}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:16px 0 22px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}
.card .v{font-size:22px;font-weight:700}
.card .l{font-size:12px;color:var(--muted);margin-top:2px}
.card .s{font-size:11px;color:var(--muted);margin-top:3px}
section{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:16px 0}
h2{font-size:20px;margin:0 0 12px}
p{margin:8px 0}
.muted{color:var(--muted)}.small{font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600;background:#fbfcfd}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:#fafbfc}
.tbl-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}
.red{color:var(--red)}.green{color:var(--green)}.amber{color:var(--amber)}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11.5px;font-weight:600}
.pill.red{background:var(--red-soft);color:var(--red)}
.pill.green{background:var(--green-soft);color:var(--green)}
.pill.amber{background:var(--amber-soft);color:var(--amber)}
.note{background:var(--amber-soft);border:1px solid #f2dfbd;border-radius:10px;padding:12px 14px;font-size:13px;color:#7a5312;margin:12px 0}
.key{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:13px}
.two{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}
.ladder{display:grid;grid-template-columns:120px 1fr;gap:6px 12px;font-size:13px}
.ladder .p{font-weight:700;text-align:right}
.ladder .d{color:var(--muted)}
.ladder .row{display:contents}
.ladder .bar{height:26px;background:#f1f3f6;border-radius:4px;position:relative}
.ladder .bar>span{position:absolute;left:0;top:0;height:100%;border-radius:4px;background:#dce1e8}
.ladder .bar .mark{position:absolute;top:-4px;height:34px;width:2px;background:#18202a}
.now{background:#fff7f7 !important}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:22px}
@media(max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}.two{grid-template-columns:1fr}h1{font-size:25px}}
@media(max-width:560px){.grid{grid-template-columns:repeat(2,1fr)}header{padding:22px}.wrap{padding:16px 12px 40px}}
"""


def half_up(x: float, digits: int = 1) -> float:
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def fnum(v, digits=1, suffix=""):
    if v is None or pd.isna(v):
        return "—"
    return f"{half_up(float(v), digits):.{digits}f}{suffix}".replace("-", "−")


def fsign(v, digits=1):
    if v is None or pd.isna(v):
        return "—"
    return f"{half_up(float(v), digits):+.{digits}f}%".replace("-", "−")


def cls(v):
    if v is None or pd.isna(v):
        return ""
    return "red" if float(v) < 0 else "green"


def main() -> None:
    path = pd.read_csv(OUT / "当前案例_603778_路径.csv")
    comp = pd.read_csv(OUT / "当前案例_对比组.csv")
    own = pd.read_csv(OUT / "当前案例_603778_同形态.csv")

    # 当前路径表
    path_rows = []
    for _, r in path.iterrows():
        path_rows.append(
            "<tr>"
            f"<td>{r['日期']}</td><td>{r['D+n']}</td><td>{r['开盘']:.2f}</td><td>{r['最高']:.2f}</td>"
            f"<td>{r['最低']:.2f}</td><td>{r['收盘']:.2f}</td>"
            f"<td class=\"{cls(r['涨跌幅(%)'])}\">{fsign(r['涨跌幅(%)'])}</td>"
            f"<td class=\"{cls(r['相对买入价(%)'])}\">{fsign(r['相对买入价(%)'])}</td>"
            f"<td class=\"{cls(r['最低相对买入价(%)'])}\">{fsign(r['最低相对买入价(%)'])}</td>"
            f"<td>{r['量比20日']:.2f}</td><td>{r['收盘位置']:.3f}</td>"
            "</tr>"
        )

    # 对比表
    comp_rows = []
    for _, r in comp.iterrows():
        style = ' class="now"' if r["口径"] == "D+2 当日跌4-5% 且 累计 ≤ -5%" else ""
        comp_rows.append(
            f"<tr{style}>"
            f"<td>{r['口径']}</td><td>{int(r['笔数'])}</td>"
            f"<td class=\"{cls(r['均值(%)'])}\">{fsign(r['均值(%)'])}</td>"
            f"<td class=\"{cls(r['中位数(%)'])}\">{fsign(r['中位数(%)'])}</td>"
            f"<td class=\"{'red' if r['亏损率(%)']>=60 else ''}\">{fnum(r['亏损率(%)'])}%</td>"
            f"<td>{fnum(r['止盈率(%)'])}%</td><td>{fnum(r['亏损15(%)'])}%</td>"
            f"<td class=\"muted\" style=\"white-space:normal;text-align:left\">{r['备注']}</td>"
            "</tr>"
        )

    # 603778 同形态
    own_rows = []
    for _, r in own.iterrows():
        own_rows.append(
            "<tr>"
            f"<td>{r['信号日']}</td><td>{r['买入日']}</td><td>{r['后复权买入价']:.2f}</td>"
            f"<td class=\"{cls(r['D+0收盘相对买入价(%)'])}\">{fsign(r['D+0收盘相对买入价(%)'])}</td>"
            f"<td class=\"{cls(r['D+1(%)'])}\">{fsign(r['D+1(%)'])}</td>"
            f"<td class=\"{cls(r['D+2(%)'])}\">{fsign(r['D+2(%)'])}</td>"
            f"<td class=\"{cls(r['D+2累计(%)'])}\">{fsign(r['D+2累计(%)'])}</td>"
            f"<td class=\"{cls(r['D+3累计(%)'])}\">{fsign(r['D+3累计(%)'])}</td>"
            f"<td class=\"{cls(r['最终收益(%)'])}\">{fsign(r['最终收益(%)'])}</td>"
            f"<td class=\"{cls(r['截至10月30日收益(%)'])}\">{fsign(r['截至10月30日收益(%)'])}</td>"
            f"<td>{r['状态']}</td><td class=\"muted\" style=\"white-space:normal;text-align:left\">{r['相似度']}</td>"
            "</tr>"
        )

    # 价格梯子：从高到低
    levels = [
        (8.62, "+20% 止盈目标", ""),
        (7.18, "买入价", ""),
        (7.10, "MA5 / 10-29 收盘 7.08", ""),
        (6.82, "−5% 累计线", "基准位置"),
        (6.77, "当前收盘", "当前"),
        (6.72, "10/23 BOR 低点", "结构线"),
        (6.11, "88.6% 回撤位", ""),
        (6.10, "15% 硬止损", ""),
        (5.39, "25% 止损", ""),
    ]
    mx, mn = 8.62, 5.20
    ladder_rows = []
    for price, label, note in levels:
        width = (price - mn) / (mx - mn) * 100
        color = "#d64545" if "止损" in label or "BOR" in label else "#2f5f9e" if "买入" in label or "止盈" in label else "#667085"
        ladder_rows.append(
            f'<div class="p">{price:.2f}</div>'
            f'<div class="bar"><span style="width:{width:.1f}%;background:{color};opacity:.18"></span>'
            f'<span class="mark" style="left:{width:.1f}%;background:{color}"></span></div>'
            f'<div class="d">{label} {("· "+note) if note else ""}</div>'
        )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>603778 当前案例：D+2 收 -5.7%</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
<div class="tag">当前案例 · 2026-10-30 收盘</div>
<h1>603778：买入后 D+2 收 −5.7%，6.72 还在守</h1>
<p>10/28 开盘 7.18 买入｜10/30 收 6.77｜今日 −4.38%｜浮亏 −5.71%｜最低 6.76</p>
</header>

<div class="callout">
<b>今天不是“D+1 当日跌 4-5%”的 47.1% 组。</b>当前是 <b>D+2 累计 −5.71%</b>：D+2 当日跌 4-5% 的 33 笔里，累计仍在 −5% 以上的 21 笔最终均值 +11.2%、亏损率 23.8%；累计跌破 −5% 的 12 笔均值 −6.7%、亏损率 75.0%、止盈率 0%。<b>当前在坏的那一半。</b>
</div>

<div class="grid">
<div class="card"><div class="v">7.18</div><div class="l">买入价</div><div class="s">10/28 开盘</div></div>
<div class="card"><div class="v">6.77</div><div class="l">10/30 收盘</div><div class="s">最低 6.76</div></div>
<div class="card"><div class="v red">−4.38%</div><div class="l">今日涨跌</div><div class="s">D+2</div></div>
<div class="card"><div class="v red">−5.71%</div><div class="l">相对买入价</div><div class="s">最低 −5.85%</div></div>
<div class="card"><div class="v amber">0.6%</div><div class="l">距 6.72 BOR 低点</div><div class="s">还没破，但很近</div></div>
</div>

<section>
<h2>这三天的路径</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>日期</th><th>D+n</th><th>开盘</th><th>最高</th><th>最低</th><th>收盘</th><th>当日</th><th>相对买入价</th><th>最低相对买入价</th><th>量比20日</th><th>收盘位置</th></tr></thead>
<tbody>{''.join(path_rows)}</tbody></table></div>
<p class="small muted">10/30 盘中最高 7.22，已经高于买入价 7.18（+0.56%），但收盘 6.77，收在当日最低附近（收盘位置 0.02）。冲高失败，量比 1.09，不是放量恐慌低点。</p>
</section>

<section>
<h2>为什么 47.1% 不适用：累计线比日跌幅更重要</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>口径</th><th>笔数</th><th>均值</th><th>中位数</th><th>亏损率</th><th>止盈率</th><th>≤ −15%</th><th>备注</th></tr></thead>
<tbody>{''.join(comp_rows)}</tbody></table></div>
<div class="note"><b>读法：</b>“今天跌 4.38%”本身的历史组是 33 笔、均值 +4.7%、亏损率 42.4%；但这个均值被“累计还在 −5% 以上”的 21 笔拉高了。当前累计 −5.71%，落在“累计 ≤ −5%”的 12 笔里：均值 −6.7%、中位 −5.9%、亏损率 75%、止盈率 0%。再叠加“榜一 + 深回撤”，最窄组 10 笔亏损率 80%，但样本太小。</div>
</section>

<section>
<h2>603778 自己的同形态案例：一胜一负</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>信号日</th><th>买入日</th><th>后复权买入价</th><th>D+0</th><th>D+1</th><th>D+2</th><th>D+2累计</th><th>D+3累计</th><th>最终</th><th>截至10/30</th><th>状态</th><th>相似度</th></tr></thead>
<tbody>{''.join(own_rows)}</tbody></table></div>
<div class="two" style="margin-top:12px">
<div class="key"><b>7/23 那笔（相似且成功）：</b>首日 +1.12%、D2 −4.39%、累计 −4.71%，D3 +1.35%，最终 +20%。当时是 7 月反弹段。</div>
<div class="key"><b>10/15 那笔（相似但失败）：</b>首日 +1.33%、D2 −5.30%、累计 −4.67%，D3 反弹 +2.52%，但之后又跌；截至 10/30 相对 7.50 买入价 −9.73%，未满 15 日。当前更像 10 月下跌段。</div>
</div>
<p class="small muted">7/14 和 7/08 不算相似：它们买入当天收盘已经 −8.2%~−9.7%，是“买入即深套”的结构。当前买入当天还是 +1.25%。</p>
</section>

<section>
<h2>价格梯子：6.82 是 −5% 线，6.72 是结构线</h2>
<div class="ladder">{''.join(ladder_rows)}</div>
<p class="small muted">当前 6.77 在 6.82（−5% 累计线）和 6.72（BOR 低点）之间。收回 6.82/7.08 才算修复；收盘跌破 6.72，BOR 失败。</p>
</section>

<section>
<h2>怎么办：按规则 vs 按结构</h2>
<div class="two">
<div class="key">
<b>按系统规则：</b><br>
15% 硬止损在 <b>6.10</b>，当前 6.77 未触发；15 个交易日到期在 <b>11/18</b>；+20% 目标 <b>8.62</b>。机械规则下今天没有卖出信号。<br><br>
<b>按 BOR 结构：</b><br>
<b>6.72</b> 是 10/23 BOR 的低点；<b>11/2 收盘跌破 6.72</b>，BOR 失败，继续持有的结构理由消失。守住并收回 <b>6.82 / 7.08</b> 才算修复。<br><br>
<b>不要：</b>加仓、用“跌了更便宜”解释、用 7/23 的 +20% 安慰自己。当前不是 7 月反弹段。
</div>
<div class="key">
<b>明天/11/2 看三件事：</b><br>
1. 收盘是否守住 <b>6.72</b>；<br>
2. 是否能收回 <b>6.82</b>（−5% 线）和 <b>7.08/7.10</b>（10/29 收盘/MA5）；<br>
3. 量能是否放大到“恐慌 capitulation”级别（当前量比只有 1.09）。<br><br>
<b>历史同形态：</b>52 笔亏损率 69.2%、中位 −7.9%；榜一 26 笔亏损率 76.9%、止盈率 0%。这不是“必跌”，但胜率不支持死扛。<br><br>
<b>我的读数：</b>6.72 还没破，今天也不是放量恐慌低点；但收盘位置 0.02、冲高 7.22 失败，结构很弱。11/2 是确认日。
</div>
</div>
</section>

<section>
<h2>口径</h2>
<div class="key">
当前 OHLC/量：baostock 2026-10-30 收盘后快照；买入价 = 10/28 开盘 7.18。<br>
历史路径：<code>../12_买入后证伪_哪天没达标/03_结果/买入后逐日路径.parquet</code>（1,256 笔信号前二）。<br>
历史分组按同一路径数据重算；收益为 +20% 止盈或 15 日到期收盘，后复权，未扣成本。当前案例尚未进入 1,256 笔历史样本，是样本外个案。<br>
后复权买入价不可与当前实际价直接比较；同形态表只比较收益率。
</div>
</section>

<footer>
历史回测不代表未来，不构成投资建议。生成于 2026-10-30。
</footer>
</div></body></html>"""

    HTML.write_text(html, encoding="utf-8")
    print("written", HTML)


if __name__ == "__main__":
    main()
