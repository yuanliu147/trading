#!/usr/bin/env python3
"""生成 2026-11-02 更新 HTML。"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "03_结果"
HTML = BASE / "当前案例_603778_2026-11-02.html"

CSS = """
:root{--ink:#18202a;--muted:#667085;--line:#e7eaf0;--bg:#f6f7f9;--card:#fff;
--red:#d64545;--red-soft:#fbeaea;--green:#1f8a5b;--green-soft:#e9f6ef;--blue:#2f5f9e;--blue-soft:#eaf1f9;--amber:#b7791f;--amber-soft:#fdf3e3}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Noto Sans CJK SC","Microsoft YaHei",sans-serif;line-height:1.65;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 56px}
header{background:linear-gradient(135deg,#1a2b22,#2d4a3a);color:#fff;border-radius:16px;padding:30px 32px;margin-bottom:18px}
header .tag{display:inline-block;font-size:12px;letter-spacing:.08em;color:#c9e3d3;border:1px solid rgba(255,255,255,.25);padding:3px 9px;border-radius:999px;margin-bottom:14px}
h1{font-size:30px;line-height:1.3;margin:0 0 10px;font-weight:700}
header p{margin:0;color:#c9e3d3;font-size:14px}
.callout{background:#fff;border:1px solid var(--line);border-left:5px solid var(--green);border-radius:12px;padding:18px 20px;margin:16px 0 8px}
.callout b{color:var(--green)}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:16px 0 22px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}
.card .v{font-size:22px;font-weight:700}.card .l{font-size:12px;color:var(--muted);margin-top:2px}.card .s{font-size:11px;color:var(--muted);margin-top:3px}
section{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:16px 0}
h2{font-size:20px;margin:0 0 12px}
p{margin:8px 0}.muted{color:var(--muted)}.small{font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600;background:#fbfcfd}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:#fafbfc}
.tbl-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}
.red{color:var(--red)}.green{color:var(--green)}.amber{color:var(--amber)}
.note{background:var(--amber-soft);border:1px solid #f2dfbd;border-radius:10px;padding:12px 14px;font-size:13px;color:#7a5312;margin:12px 0}
.key{background:#f8fafc;border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:13px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.ladder{display:grid;grid-template-columns:90px 1fr 200px;gap:6px 10px;align-items:center;font-size:13px}
.ladder .p{font-weight:700;text-align:right}
.ladder .bar{height:18px;background:#f1f3f6;border-radius:4px;position:relative}
.ladder .bar>span{position:absolute;left:0;top:0;height:100%;border-radius:4px;background:#dce1e8}
.ladder .bar .mark{position:absolute;top:-3px;height:24px;width:2px}
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
    path = pd.read_csv(OUT / "当前案例_603778_2026-11-02_路径.csv", dtype={"代码": str})
    comp = pd.read_csv(OUT / "当前案例_2026-11-02_对比组.csv")
    third = pd.read_csv(OUT / "当前案例_2026-11-02_第三次同现.csv")

    def path_rows(code):
        rows = []
        for _, r in path[path["代码"] == code].iterrows():
            rows.append(
                "<tr>"
                f"<td>{r['日期']}</td><td>{r['D+n']}</td><td>{r['开盘']:.2f}</td><td>{r['最高']:.2f}</td>"
                f"<td>{r['最低']:.2f}</td><td>{r['收盘']:.2f}</td>"
                f"<td class=\"{cls(r['涨跌幅(%)'])}\">{fsign(r['涨跌幅(%)'])}</td>"
                f"<td class=\"{cls(r['相对买入价(%)'])}\">{fsign(r['相对买入价(%)'])}</td>"
                f"<td class=\"{cls(r['最低相对买入价(%)'])}\">{fsign(r['最低相对买入价(%)'])}</td>"
                f"<td>{r['量比20日']:.2f}</td><td>{r['收盘位置']:.3f}</td><td>{r['MA5']:.2f}</td><td>{r['RSI14']:.1f}</td>"
                "</tr>"
            )
        return "".join(rows)

    comp_rows = []
    for _, r in comp.iterrows():
        comp_rows.append(
            "<tr>"
            f"<td>{r['口径']}</td><td>{int(r['笔数'])}</td>"
            f"<td class=\"{cls(r['均值(%)'])}\">{fsign(r['均值(%)'])}</td>"
            f"<td class=\"{cls(r['中位数(%)'])}\">{fsign(r['中位数(%)'])}</td>"
            f"<td class=\"{'red' if r['亏损率(%)']>=60 else ''}\">{fnum(r['亏损率(%)'])}%</td>"
            f"<td>{fnum(r['止盈率(%)'])}%</td><td>{fnum(r['亏损15(%)'])}%</td>"
            "</tr>"
        )
    third_rows = []
    for _, r in third.iterrows():
        third_rows.append(
            "<tr>"
            f"<td>{r['口径']}</td><td>{int(r['笔数'])}</td>"
            f"<td class=\"{cls(r['均值(%)'])}\">{fsign(r['均值(%)'])}</td>"
            f"<td class=\"{cls(r['中位数(%)'])}\">{fsign(r['中位数(%)'])}</td>"
            f"<td class=\"{'red' if r['亏损率(%)']>=60 else ''}\">{fnum(r['亏损率(%)'])}%</td>"
            f"<td>{fnum(r['最差(%)'])}%</td><td>{fnum(r['最好(%)'])}%</td>"
            "</tr>"
        )

    levels = [
        (7.18, "买入价", "blue"), (7.10, "MA5 / 10-29 收盘 7.08", "gray"),
        (6.92, "11/2 收盘", "green"), (6.82, "−5% 累计线", "gray"),
        (6.72, "10/23 BOR 低点（11/2 盘中破、收盘收回）", "red"),
        (6.48, "11/2 盘中低点", "red"), (6.11, "88.6% 回撤位", "gray"),
        (6.10, "15% 硬止损", "red"), (5.39, "25% 止损", "gray"),
    ]
    mx, mn = 7.30, 5.20
    ladder = []
    for price, label, color in levels:
        w = (price - mn) / (mx - mn) * 100
        ladder.append(
            f'<div class="p">{price:.2f}</div>'
            f'<div class="bar"><span style="width:{w:.1f}%"></span>'
            f'<span class="mark" style="left:{w:.1f}%;background:{ {"blue":"#2f5f9e","green":"#1f8a5b","red":"#d64545","gray":"#667085"}[color] }"></span></div>'
            f'<div class="muted">{label}</div>'
        )

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>11/2 更新：603778 假破 6.72 收回</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
<div class="tag">11/2 更新 · 2026-11-02 收盘</div>
<h1>603778 假破 6.72 收回，前二第三次还是这俩</h1>
<p>603778 10/28 开盘 7.18 买入｜11/2 收 6.92（−3.62%）｜最低 6.48，收盘收回 6.72｜前二 11/2 再是 603778/605376</p>
</header>

<div class="callout">
<b>今天是一次“假破回收”：</b>最低 6.48 跌破 10/23 BOR 低点 6.72，但收盘 6.92 收回，收盘位置 0.917。D+3 收盘 −3.62%，仍低于 −2% 的时间证伪线，但已从 10/30 的 −5.71% 回到 −5% 上方。<b>最坏的 66.5% 亏损组降到了 45.5%</b>（当前形态+榜一，N=11）——改善，但不是转强确认。
</div>

<div class="grid">
<div class="card"><div class="v">6.92</div><div class="l">603778 11/2 收盘</div><div class="s">+2.22%</div></div>
<div class="card"><div class="v red">−3.62%</div><div class="l">相对买入价 7.18</div><div class="s">10/30 是 −5.71%</div></div>
<div class="card"><div class="v green">6.48 → 6.92</div><div class="l">盘中低点 → 收盘</div><div class="s">收回 6.72</div></div>
<div class="card"><div class="v amber">−7.88%</div><div class="l">605376 相对 77.80</div><div class="s">11/2 收 71.67</div></div>
<div class="card"><div class="v">第 3 次</div><div class="l">前二同现</div><div class="s">10/23、10/28、11/2</div></div>
</div>

<section>
<h2>603778：四天路径</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>日期</th><th>D+n</th><th>开盘</th><th>最高</th><th>最低</th><th>收盘</th><th>当日</th><th>相对买入价</th><th>最低相对买入价</th><th>量比</th><th>收盘位置</th><th>MA5</th><th>RSI14</th></tr></thead>
<tbody>{path_rows('603778')}</tbody></table></div>
<p class="small muted">11/2 最低 6.48，比 6.72 低 3.6%；收盘 6.92，比 6.72 高 3.0%。收盘位置 0.917，尾盘接住；量比 1.17，略高于 10/30，但不是爆量。价格仍在 MA5/10/20/60 下方。</p>
</section>

<section>
<h2>605376：还是弱</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>日期</th><th>D+n</th><th>开盘</th><th>最高</th><th>最低</th><th>收盘</th><th>当日</th><th>相对 77.80</th><th>最低相对 77.80</th><th>量比</th><th>收盘位置</th><th>MA5</th><th>RSI14</th></tr></thead>
<tbody>{path_rows('605376')}</tbody></table></div>
<p class="small muted">605376 从 10/23 信号买入 10/28 开盘 77.80，11/2 收 71.67，−7.88%；从 10/28 信号买入 11/2 开盘 72.02，收 71.67，−0.49%。收盘 71.67 仍在 MA5 74.40 下方，收盘位置 0.597，比 603778 的 0.917 弱。</p>
</section>

<section>
<h2>为什么说“改善，但不是转强”</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>历史口径</th><th>笔数</th><th>均值</th><th>中位数</th><th>亏损率</th><th>止盈率</th><th>≤ −15%</th></tr></thead>
<tbody>{''.join(comp_rows)}</tbody></table></div>
<div class="note"><b>读法：</b>10/30 的状态是“D+2 累计≤−5%”，191 笔、亏损率 66.5%。今天 D+3 收回 −5% 上方，37 笔、亏损率 56.8%；再叠加“D+3 当日>+2%、榜一”，11 笔、亏损率 45.5%、均值 +0.32%。<b>改善是真的，但 45.5% 仍是“接近一半会亏”，不是“转强确认”。</b> D+3 累计 −3.62% 仍在 −2% 以下，时间规则没有解除。</div>
</section>

<section>
<h2>前二第三次还是这俩：历史第三次同现</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>口径</th><th>笔数</th><th>均值</th><th>中位数</th><th>亏损率</th><th>最差</th><th>最好</th></tr></thead>
<tbody>{''.join(third_rows)}</tbody></table></div>
<div class="note"><b>18 次第三次同现整体不差</b>（均值 +0.81%、中位 +5.22%、亏损率 44.4%），但<b>间隔正好 3 日的 3 次很差</b>（−4.11%、−6.32%、66.7% 亏损）。当前是历史上第一次“连续两次间隔 3 日”，没有直接先例。<b>不要把“又是这俩”当成确认；11/2 信号买 11/5 是独立决策。</b></div>
</section>

<section>
<h2>价格梯子</h2>
<div class="ladder">{''.join(ladder)}</div>
<p class="small muted">6.72 今天盘中破了但收盘收回；6.92 站上 6.82 的 −5% 线。上方先看 7.08/7.10，再看 7.18；下方再破 6.72/6.48 才是结构失败。</p>
</section>

<section>
<h2>怎么办</h2>
<div class="two">
<div class="key">
<b>603778 当前持仓：</b><br>
15% 止损 <b>6.10</b>，25% 止损 <b>5.39</b>，当前 6.92 未触发；15 日到期 <b>11/18</b>，+20% 目标 <b>8.62</b>。<br><br>
<b>结构：</b>6.72 收回，假破回收成立；但 D+3 仍 −3.62%（<−2%），时间证伪未解除。<br><br>
<b>11/3 看：</b>守住 6.72，并收回 7.08/7.10，才算修复；再收盘跌破 6.72，BOR 失败。
</div>
<div class="key">
<b>605376：</b>相对 10/23 买入价 −7.88%，相对 11/2 开盘 −0.49%；仍在 MA5 74.40 下方，RSI 18.8 超卖但没有收回关键位。<br><br>
<b>前二第三次：</b>历史 18 次 +0.81%/44.4% 亏损；间隔 3 日 3 次 −4.11%/66.7% 亏损；连续两次间隔 3 日 0 次先例。<br><br>
<b>不要：</b>因为“又是这俩”加仓；603778 已有一笔，605376 仍弱。11/5 的新信号是独立决策。
</div>
</div>
</section>

<section>
<h2>口径</h2>
<div class="key">
当前 OHLC/量：baostock 2026-11-02 收盘后快照；603778 买入价 7.18（10/28 开盘），605376 买入价 77.80（10/28 开盘）。<br>
历史分组：<code>../12_买入后证伪_哪天没达标/03_结果/买入后逐日路径.parquet</code>（1,256 笔前二）；收益为 +20% 止盈或 15 日到期，后复权，未扣成本。<br>
前二第三次同现：同一对组合、同顺序，第 3 次出现；收益=两只最终收益等权平均。当前 11/2 信号尚未进入历史样本。
</div>
</section>

<footer>
历史回测不代表未来，不构成投资建议。生成于 2026-11-02。
</footer>
</div></body></html>"""

    HTML.write_text(html, encoding="utf-8")
    print("written", HTML)


if __name__ == "__main__":
    main()
