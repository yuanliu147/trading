#!/usr/bin/env python3
"""Render the selection report Markdown to a standalone HTML with embedded charts."""
from __future__ import annotations

import base64
import html
import os
import re

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/05_选股特征"
MD = f"{BASE}/报告_选股层.md"
OUT = f"{BASE}/选股层.html"

CSS = """
:root{--bg:#f6f7f9;--ink:#1b2430;--muted:#66717f;--line:#dce3ea;--accent:#2f5d8a;--warm:#a45f1f;--good:#1f7a57;--bad:#a53f3f;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:44px 34px 80px;}
.hero{padding:8px 0 22px;border-bottom:1px solid var(--line);margin-bottom:26px;}
.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.12em;font-weight:700;text-transform:uppercase;}
h1{font-size:34px;line-height:1.22;letter-spacing:-.6px;margin:10px 0 8px;}
.sub{color:var(--muted);font-size:14px;max-width:900px;}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0 8px;}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;}
.card .k{color:var(--muted);font-size:12px;}
.card .v{font-size:24px;font-weight:750;letter-spacing:-.4px;margin-top:4px;}
.card .n{color:var(--muted);font-size:12px;margin-top:2px;}
.callout{background:#fff8e8;border:1px solid #efd7a8;border-left:5px solid var(--warm);border-radius:8px;padding:12px 16px;margin:18px 0 8px;font-size:14px;}
.callout.good{background:#eef7f1;border-color:#c9e3d4;border-left-color:var(--good);}
h2{font-size:22px;margin:42px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line);letter-spacing:-.3px;}
h3{font-size:17px;margin:28px 0 8px;}
h4{font-size:15px;margin:20px 0 6px;}
p{margin:10px 0;}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(47,93,138,.25);}
a:hover{border-bottom-color:var(--accent);}
strong{font-weight:700;}
code{background:#eef2f6;border:1px solid #e3e9ef;border-radius:5px;padding:1px 5px;font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
pre{background:#111a24;color:#e7eef5;border-radius:9px;padding:14px 16px;overflow-x:auto;font-size:12.5px;line-height:1.6;}
pre code{background:transparent;border:0;color:inherit;padding:0;}
blockquote{margin:14px 0;padding:10px 16px;border-left:4px solid #b9c8d6;background:#fff;color:#3d4a58;border-radius:0 8px 8px 0;}
hr{border:0;border-top:1px solid var(--line);margin:30px 0;}
.table-wrap{overflow-x:auto;margin:12px 0 18px;border:1px solid var(--line);border-radius:9px;background:#fff;}
table{width:100%;border-collapse:collapse;font-size:12.5px;min-width:640px;}
th{background:#eef3f7;color:#2b3948;font-weight:650;text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}
td{padding:8px 10px;border-bottom:1px solid #edf1f5;vertical-align:top;}
tr:nth-child(even) td{background:#fbfcfd;}
tr:last-child td{border-bottom:0;}
figure{margin:16px 0 24px;}
figure img{width:100%;display:block;border:1px solid var(--line);border-radius:9px;background:#fff;}
figcaption{color:var(--muted);font-size:12px;margin-top:6px;text-align:center;}
ul,ol{margin:10px 0 14px;padding-left:24px;}
li{margin:5px 0;}
.small{color:var(--muted);font-size:12.5px;}
.footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12px;}
@media(max-width:860px){.wrap{padding:24px 16px 60px}.cards{grid-template-columns:1fr 1fr}h1{font-size:26px}h2{font-size:19px}}
"""


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def image_block(line: str) -> str | None:
    m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
    if not m:
        return None
    alt, src = m.group(1), m.group(2)
    path = os.path.join(BASE, src)
    if not os.path.exists(path):
        return f'<figure><figcaption>{html.escape(alt)}（图缺失：{html.escape(src)}）</figcaption></figure>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def parse_table(lines: list[str]) -> str:
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return ""
    header = rows[0]
    body = rows[2:] if len(rows) > 1 else []
    out = ['<div class="table-wrap"><table><thead><tr>']
    out += [f"<th>{inline(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def convert(md: str) -> str:
    lines = md.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        # fenced code
        if s.startswith("```"):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            continue
        # image
        if s.startswith("!["):
            img = image_block(s)
            if img:
                out.append(img)
                i += 1
                continue
        # heading
        if s.startswith("#"):
            m = re.match(r"^(#{1,4})\s+(.*)$", s)
            if m:
                level = len(m.group(1))
                text = inline(m.group(2))
                if level == 1:
                    out.append(f"<h1>{text}</h1>")
                else:
                    out.append(f"<h{level}>{text}</h{level}>")
                i += 1
                continue
        # table
        if s.startswith("|"):
            tbl = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.append(parse_table(tbl))
            continue
        # blockquote
        if s.startswith(">"):
            q = []
            while i < n and lines[i].strip().startswith(">"):
                q.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + "<br>".join(inline(x) for x in q) + "</blockquote>")
            continue
        # lists (group items and indented continuation lines)
        if re.match(r"^[-*]\s+", s):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                text = re.sub(r"^\s*[-*]\s+", "", lines[i]).strip()
                i += 1
                cont = []
                while i < n and lines[i].startswith(("   ", "\t")) and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                    cont.append(lines[i].strip()); i += 1
                items.append((text, cont))
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and re.match(r"^\s*[-*]\s+", lines[j]):
                    i = j
                else:
                    break
            out.append("<ul>" + "".join(
                f"<li>{inline(t)}" + "".join(f"<p>{inline(c)}</p>" for c in cont) + "</li>"
                for t, cont in items) + "</ul>")
            continue
        if re.match(r"^\d+\.\s+", s):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                text = re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()
                i += 1
                cont = []
                while i < n and lines[i].startswith(("   ", "\t")) and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                    cont.append(lines[i].strip()); i += 1
                items.append((text, cont))
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and re.match(r"^\s*\d+\.\s+", lines[j]):
                    i = j
                else:
                    break
            out.append("<ol>" + "".join(
                f"<li>{inline(t)}" + "".join(f"<p>{inline(c)}</p>" for c in cont) + "</li>"
                for t, cont in items) + "</ol>")
            continue
        # hr
        if s == "---":
            out.append("<hr>")
            i += 1
            continue
        # paragraph
        para = [s]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|\||>|```|!\[|[-*]\s|\d+\.\s|---)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + " ".join(inline(x) for x in para) + "</p>")
    return "\n".join(out)


def main() -> None:
    with open(MD, encoding="utf-8") as f:
        md = f.read()
    body_md = md.split("## 结论先行", 1)[1]
    body_md = "## 结论先行" + body_md
    body = convert(body_md)

    hero = """
<div class="hero">
  <div class="eyebrow">独立验证 · 2012-2026 · 2026-10-29</div>
  <h1>看长做短：哪些股票适合这套买点</h1>
  <div class="sub">买点固定为 V2 机械代理，只加信号日可观测的股票特征。开发期定方向，验证期只检验。结论不是“选牛股”，而是“把原策略的亏损和回撤压小”。</div>
  <div class="cards">
    <div class="card"><div class="k">评分 A 验证期 IC</div><div class="v">0.131</div><div class="n">t 16.4 · 行业中性 0.133</div></div>
    <div class="card"><div class="k">Q5-Q1 20 日超额</div><div class="v">+2.00pp</div><div class="n">t 5.64 · 行业中性 +1.88pp</div></div>
    <div class="card"><div class="k">前 30% 绝对超额</div><div class="v">+0.48%</div><div class="n">t 1.52 · 未过预注册 t≥2</div></div>
    <div class="card"><div class="k">最佳组合 vs 等权</div><div class="v">+2.12%</div><div class="n">等权 +6.74% · 仍跑输 4.6pp</div></div>
  </div>
  <div class="callout"><strong>总判断：</strong>选股层在信号层未达到预注册显著性；组合层主口径相对改善但绝对仍亏。它是<strong>风险控制/排序层</strong>，不是<strong>跑赢等权的 alpha 系统</strong>。中段 70-90%、闸门、固定 20 日是看结果后补充的探索性口径，不能当作已验证。</div>
</div>
"""

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>看长做短·选股层：哪些股票适合这套买点</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{hero}
{body}
<div class="footer">数据：baostock 原始行情 + frozen 时点市值/估值/板块 + 申万一级时点行业。所有收益口径、样本数、缺失和限制见正文。本文不构成投资建议。</div>
</div>
</body>
</html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("saved", OUT, len(html_doc), "chars")


if __name__ == "__main__":
    main()
