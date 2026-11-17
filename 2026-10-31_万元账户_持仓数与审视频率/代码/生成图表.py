#!/usr/bin/env python3
"""生成静态小多图：9个核心配置的8种子路径、中位数与沪深300价格指数。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "2026-10-30_万元账户_低换手估值"
OUT = ROOT / "结果"

z = np.load(OUT / "共同日历账户收益.npz", allow_pickle=True)
ids = [str(x) for x in z["ids"]]
logs, dates = z["logs"], [str(x) for x in z["dates"]]
paths = np.exp(np.cumsum(logs, axis=2)) * 10000
paths = np.concatenate([np.full((logs.shape[0], logs.shape[1], 1), 10000.0), paths], axis=2)
summary = pd.read_csv(OUT / "配置种子分布.csv")
index = pd.read_csv(PRIOR / "数据/日历指数.csv", dtype={"日期": str})
index = index[index["日期"] >= "2010-12-31"].reset_index(drop=True)
index_path = index["沪深300收盘"].to_numpy(float)
index_path = index_path / index_path[0] * 10000
n = len(dates)
assert len(index_path) == n

STEP = 25
idx = list(range(0, n, STEP))
if idx[-1] != n - 1:
    idx.append(n - 1)

VLO, VHI = 3000.0, 45000.0
W, H, L, R, T, B = 340, 190, 38, 6, 6, 20
PW, PH = W - L - R, H - T - B


def X(t):
    return L + PW * t / (n - 1)


def Y(v):
    v = min(max(v, VLO), VHI)
    return T + PH * (1 - (np.log(v) - np.log(VLO)) / (np.log(VHI) - np.log(VLO)))


def path(values):
    return "M" + "L".join(f"{X(t):.1f} {Y(values[t]):.1f}" for t in idx)


def pct(v):
    return ("+" if v >= 0 else "−") + f"{abs(v) * 100:.2f}%"


panels = []
for ident in [x for x in ids if x.startswith("S")]:
    j = ids.index(ident)
    rows = summary[(summary["配置"] == ident) & (summary["类别"] == "core")].iloc[0]
    med = np.median(paths[j], axis=0)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{ident}，{int(rows["槽位"])}个槽位，每{int(rows["审视频率月"])}个月审视，中位年化{pct(rows["CAGR中位"])}，中位最大回撤{pct(rows["回撤中位"])}">']
    for v in (3000, 5000, 10000, 20000, 40000):
        y = Y(v)
        parts.append(f'<line x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--border-subtle,#e3e6ea)" stroke-width="1"/>')
        lab = {3000: "3千", 5000: "5千", 10000: "1万", 20000: "2万", 40000: "4万"}[v]
        parts.append(f'<text x="{L-5}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="var(--seed-muted,#6b7280)">{lab}</text>')
    for yr in (2011, 2015, 2020):
        t = next(i for i, d in enumerate(dates) if d[:4] == str(yr))
        x = X(t)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{H-B}" y2="{H-B+3}" stroke="var(--border-standard,#c9ced6)"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-B+14}" text-anchor="middle" font-size="10" fill="var(--seed-muted,#6b7280)">{yr}</text>')
    parts.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(10000):.1f}" y2="{Y(10000):.1f}" stroke="var(--border-standard,#c9ced6)" stroke-dasharray="2 3"/>')
    for s in range(paths.shape[1]):
        parts.append(f'<path d="{path(paths[j, s])}" fill="none" stroke="var(--viz-series-1,#2563eb)" stroke-opacity="0.20" stroke-width="1"/>')
    parts.append(f'<path d="{path(index_path)}" fill="none" stroke="var(--viz-series-2,#c26a06)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<path d="{path(med)}" fill="none" stroke="var(--viz-series-1,#2563eb)" stroke-width="2"/>')
    parts.append('</svg>')
    meta = (f'中位年化 {pct(rows["CAGR中位"])} · 回撤 {pct(rows["回撤中位"])} · '
            f'正种子 {int(rows["正CAGR种子数"])}/8 · 2016–2020 {pct(rows["2016-2020_CAGR中位"])}')
    m = int(rows["审视频率月"])
    freq = "每月" if m == 1 else f"每{m}月"
    title = f'{ident} · {int(rows["槽位"])}槽 / {freq}审视'
    panels.append(f'<section class="wy-panel"><p class="wy-cap">{title}</p>'
                  f'<p class="wy-meta text-small text-muted tabular-nums">{meta}</p>{"".join(parts)}</section>')

fragment = f'''<div id="wy-grid-c4e8">
<style>
#wy-grid-c4e8{{font-family:var(--font-sans);color:var(--seed-fg)}}
#wy-grid-c4e8 .wy-title{{font-weight:500;margin:0 0 4px}}
#wy-grid-c4e8 .wy-sub{{margin:0 0 8px}}
#wy-grid-c4e8 .wy-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px 22px;margin-top:10px}}
#wy-grid-c4e8 .wy-panel{{min-width:0}}
#wy-grid-c4e8 .wy-cap{{font-size:13px;font-weight:500;margin:0 0 1px}}
#wy-grid-c4e8 .wy-meta{{margin:0 0 2px}}
#wy-grid-c4e8 svg{{display:block;width:100%;height:auto}}
#wy-grid-c4e8 .wy-note{{margin:10px 0 0}}
#wy-grid-c4e8 .wy-baseline{{display:inline-block;width:16px;height:0;border-top:2px dashed var(--border-standard,#c9ced6);vertical-align:middle;margin-right:6px}}
</style>
<p class="wy-title">9个核心配置 × 8个种子：净值路径（2010-12-31 = 10,000元）</p>
<p class="wy-sub text-small text-muted">9个配置的中位年化 +2.1% 至 +3.3%，全部低于沪深300价格指数 +5.24%；2016–2020 的8种子中位收益全部为负。</p>
<div class="viz-legend text-small">
<span><span class="viz-legend-swatch" style="background:var(--viz-series-1,#2563eb);opacity:.3"></span>8个种子</span>
<span><span class="viz-legend-swatch" style="background:var(--viz-series-1,#2563eb)"></span>8种子逐日中位数</span>
<span><span class="viz-legend-swatch" style="background:var(--viz-series-2,#c26a06)"></span>沪深300价格指数（不含分红）</span>
<span><span class="wy-baseline"></span>本金10,000元</span>
</div>
<div class="wy-grid">
{"".join(panels)}
</div>
<p class="wy-note text-small text-muted">纵轴为对数刻度，3,000–45,000元；每25个交易日取一点。指数为价格指数，不含分红，不是ETF；8个种子是持仓抽样，不是8段独立历史。</p>
</div>'''

vis = Path("/Users/admin/.qoder/app/chat-session-visualizations/d6166c1b-d881-4556-9e12-5d2f102af317/visualizations/wanyuan-grid-9configs-c4e8.html")
vis.write_text(fragment, encoding="utf-8")

css_vars = '''
:root{color-scheme:light dark;--seed-fg:#1f2328;--seed-muted:#6b7280;--border-standard:#c9ced6;--border-subtle:#e3e6ea;--card-bg:#fff;--viz-series-1:#2563eb;--viz-series-2:#c26a06;--font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
@media (prefers-color-scheme:dark){:root{--seed-fg:#e6edf3;--seed-muted:#9aa4b2;--border-standard:#3d444d;--border-subtle:#2a3138;--card-bg:#161b22;--viz-series-1:#6ea8fe;--viz-series-2:#f0a35e}}
body{margin:0;padding:24px;background:var(--card-bg);color:var(--seed-fg);font-family:var(--font-sans)}
main{max-width:1100px;margin:0 auto}
.viz-legend{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}
.viz-legend-swatch{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.text-small{font-size:12px}.text-muted{color:var(--seed-muted)}.tabular-nums{font-variant-numeric:tabular-nums}
'''
doc = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>9个配置的净值路径</title><style>{css_vars}</style></head><body><main>{fragment}</main></body></html>'
(OUT / "图表_持仓数与频率.html").write_text(doc, encoding="utf-8")
print("fragment", len(fragment.encode()), "doc", len(doc.encode()))
