#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K 线案例图谱：赢家/输家、b3 与死叉的买卖点对照。

每张图含：
- 日 K（红涨绿跌）、MA5/10/20/60、成交量、MACD(12,26,9)；
- 信号日、买入日、死叉卖出、b3 卖出标记；
- 信号日关键特征（前20日、距MA20、量比、股性类型）。

图表用 matplotlib 自绘，不额外依赖 mplfinance/plotly。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "01_数据"
FIG = HERE / "02_图表"
CASE = FIG / "案例"
CASE.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

for name in ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]:
    if name in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

RED = "#d62728"
GREEN = "#2ca02c"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
GRAY = "#7f7f7f"


def load_data():
    tr = pd.read_csv(DATA / "全信号交易_分规则.csv", encoding="utf-8-sig",
                     parse_dates=["信号日", "买入日", "卖出日"])
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", encoding="utf-8-sig",
                      parse_dates=["signal_date"])
    d = tr[tr["规则"] == "死叉"].merge(
        sig, left_on=["代码", "信号日"], right_on=["code", "signal_date"],
        how="left", suffixes=("", "_sig"))
    b = tr[tr["规则"] == "b3_10_8"][
        ["代码", "信号日", "净收益率", "卖出日", "离场原因"]].rename(
        columns={"净收益率": "b3收益", "卖出日": "b3卖出日",
                 "离场原因": "b3原因"})
    d = d.merge(b, on=["代码", "信号日"], how="left")
    d["年份"] = d["信号日"].dt.year
    d["entry_gap"] = d["入场跳空"]
    return d


def pick_examples(d: pd.DataFrame, mask: pd.Series, target: float,
                  n: int = 4) -> pd.DataFrame:
    """按年份分散，选最接近目标收益的代表样本。"""
    x = d[mask].copy()
    if x.empty:
        return x
    x["距离"] = (x["净收益率"] - target).abs()
    chosen = []
    years = sorted(x["年份"].unique())
    if len(years) > n:
        idx = np.linspace(0, len(years) - 1, n).round().astype(int)
        years = [years[i] for i in idx]
    for y in years:
        g = x[x["年份"] == y].sort_values("距离")
        if len(g):
            chosen.append(g.index[0])
    if len(chosen) < n:
        rest = x.drop(chosen).sort_values("距离")
        for i in rest.index:
            if len(chosen) >= n:
                break
            if i not in chosen:
                chosen.append(i)
    return x.loc[chosen]


def plot_case(row: pd.Series, df: pd.DataFrame, file: Path, archetype: str) -> None:
    sd = pd.Timestamp(row["信号日"]); ed = pd.Timestamp(row["买入日"])
    xd = pd.Timestamp(row["卖出日"]); bxd = pd.Timestamp(row["b3卖出日"])
    idx_map = {d: i for i, d in enumerate(df["date"].tolist())}
    if sd not in idx_map:
        return
    si = idx_map[sd]
    ei = idx_map.get(ed, si + 1)
    xi = idx_map.get(xd, ei + 1)
    bxi = idx_map.get(bxd, xi)
    pre = 90
    post = max(60, xi - si + 30)
    a = max(0, si - pre)
    b = min(len(df), max(xi + post, si + 120))
    sub = df.iloc[a:b].reset_index(drop=True)
    off = a
    n = len(sub)
    x = np.arange(n)
    o = sub["open"].to_numpy(float); h = sub["high"].to_numpy(float)
    l = sub["low"].to_numpy(float); c = sub["close"].to_numpy(float)
    v = sub["volume"].to_numpy(float)
    ma5 = sub["close"].rolling(5, min_periods=1).mean().to_numpy()
    ma10 = sub["close"].rolling(10, min_periods=1).mean().to_numpy()
    ma20 = sub["close"].rolling(20, min_periods=1).mean().to_numpy()
    ma60 = sub["close"].rolling(60, min_periods=1).mean().to_numpy()
    dif = sub["dif"].to_numpy(float); dea = sub["dea"].to_numpy(float)
    hist = sub["hist"].to_numpy(float)

    fig = plt.figure(figsize=(11, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.0, 1.25], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    for i in range(n):
        color = RED if c[i] >= o[i] else GREEN
        ax1.vlines(i, l[i], h[i], color=color, linewidth=0.8)
        bottom = min(o[i], c[i])
        height = max(abs(c[i] - o[i]), 1e-8)
        ax1.add_patch(Rectangle((i - 0.32, bottom), 0.64, height,
                                facecolor=color, edgecolor=color, linewidth=0.4))
    ax1.plot(x, ma5, color=BLUE, lw=0.9, label="MA5")
    ax1.plot(x, ma10, color=ORANGE, lw=0.8, label="MA10")
    ax1.plot(x, ma20, color=PURPLE, lw=1.0, label="MA20")
    ax1.plot(x, ma60, color="#8c564b", lw=1.1, label="MA60")
    ax1.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)

    xs = si - off; xe = ei - off; xx = xi - off; xb = bxi - off
    ax1.axvspan(xs, max(xx, xs + 1), color="#fff3cd", alpha=0.45, zorder=0)
    ax1.axvline(xs, color=GRAY, ls="--", lw=1.0)
    ax1.scatter([xe], [o[ei - off] if 0 <= ei - off < n else c[xe]],
                marker="^", s=80, color=BLUE, zorder=5, label="买入")
    if 0 <= xx < n:
        ax1.scatter([xx], [c[xx]], marker="v", s=80, color="black",
                    zorder=5, label="死叉卖")
    if 0 <= xb < n and bxd != xd:
        ax1.scatter([xb], [c[xb]], marker="x", s=70, color=ORANGE,
                    zorder=5, label="b3卖")
    ax1.set_ylabel("价格")
    ax1.grid(alpha=0.15)
    ax1.set_title(
        f"{archetype}  |  {row['名称']}（{row['代码']}）  |  信号 {sd.date()}",
        fontsize=12, loc="left")
    ax1.text(
        0.995, 0.98,
        f"死叉 {row['净收益率']*100:+.1f}% / {int(row['持有交易日'])}日\n"
        f"b3 {row['b3收益']*100:+.1f}%" if pd.notna(row.get("b3收益")) else
        f"死叉 {row['净收益率']*100:+.1f}% / {int(row['持有交易日'])}日",
        transform=ax1.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#cccccc"))

    colors = [RED if c[i] >= o[i] else GREEN for i in range(n)]
    ax2.bar(x, v, color=colors, width=0.7, alpha=0.8)
    ax2.set_ylabel("成交量")
    ax2.grid(alpha=0.15)

    hcolors = [RED if hist[i] >= 0 else GREEN for i in range(n)]
    ax3.bar(x, hist, color=hcolors, width=0.7, alpha=0.8)
    ax3.plot(x, dif, color=BLUE, lw=0.9, label="DIF")
    ax3.plot(x, dea, color=ORANGE, lw=0.9, label="DEA")
    ax3.axhline(0, color=GRAY, lw=0.7)
    ax3.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    ax3.set_ylabel("MACD")
    ax3.grid(alpha=0.15)

    ticks = np.linspace(0, n - 1, min(7, n)).astype(int)
    labels = [pd.Timestamp(sub["date"].iat[i]).strftime("%Y-%m") for i in ticks]
    ax3.set_xticks(ticks, labels, fontsize=8)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    feat = (f"前20日 {row['ret20']*100:+.1f}%  |  距MA20 {row['dist_ma20']*100:+.1f}%  |  "
            f"距MA60 {row['dist_ma60']*100:+.1f}%  |  量比 {row['amt5_60']*100:+.0f}%  |  "
            f"股性 {row['type']}  |  买入跳空 {row['entry_gap']*100:+.1f}%")
    fig.text(0.01, 0.005, feat, fontsize=8.5, color="#333333")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.06)
    fig.savefig(file, dpi=150)
    plt.close(fig)


def contact_sheet(files: list[Path], out: Path, cols: int = 2,
                  label: str = "") -> None:
    if not files:
        return
    imgs = [Image.open(f).convert("RGB") for f in files]
    w = 1400
    imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
    rows = int(np.ceil(len(imgs) / cols))
    h = max(im.height for im in imgs)
    pad = 20
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad,
                              rows * h + (rows + 1) * pad + 50), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 26)
    except Exception:
        font = ImageFont.load_default()
    draw.text((pad, 10), label, fill="black", font=font)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (pad + c * (w + pad), 50 + pad + r * (h + pad)))
    sheet.save(out, quality=95)


def main() -> None:
    d = load_data()
    print("加载行情…", flush=True)
    data = eng.load_pit_data()
    names = {c: "" for c in data}
    # 名称已在 d 里
    archetypes = {
        "赢家_低位反转": (
            (d["净收益率"] >= 0.20) & (d["dist_ma60"] < -0.05)
            & (d["ret20"] < 0.05) & (d["s_vol"] < 0.6), 0.28),
        "赢家_趋势延续": (
            (d["净收益率"] >= 0.20) & (d["dist_ma60"] > 0)
            & d["ret20"].between(0, 0.10)
            & d["dist_ma20"].between(-0.02, 0.05) & (d["s_vol"] < 0.6), 0.28),
        "输家_追高": (
            (d["净收益率"] <= -0.10)
            & ((d["ret20"] > 0.10) | (d["dist_ma20"] > 0.08))
            & (d["s_vol"] > 0.5) & (d["entry_gap"] > 0), -0.16),
        "输家_高位放量": (
            (d["净收益率"] <= -0.10) & (d["amt5_60"] > 0.2)
            & (d["signal_day_ret"] > 0.03) & (d["dist_ma20"] > 0.05), -0.16),
        "b3砍掉大赢家": (
            (d["净收益率"] >= 0.25) & (d["b3收益"] <= d["净收益率"] - 0.10), 0.35),
        "b3救下大输家": (
            (d["净收益率"] <= -0.15) & (d["b3收益"] >= d["净收益率"] + 0.05), -0.20),
        "极端情绪输家": (
            (d["subtype"] == "极端情绪型") & (d["净收益率"] <= -0.08), -0.15),
        "低波低量赢家": (
            (d["type"] == "低波低量型") & (d["净收益率"] >= 0.15), 0.22),
    }

    groups = {}
    for arch, (mask, target) in archetypes.items():
        ex = pick_examples(d, mask, target, 4)
        files = []
        for k, (_, r) in enumerate(ex.iterrows(), 1):
            df = data.get(r["代码"])
            if df is None:
                continue
            fn = CASE / f"{arch}_{k:02d}_{r['名称']}_{r['代码']}.png"
            try:
                plot_case(r, df, fn, arch)
                files.append(fn)
                print(f"  {fn.name}", flush=True)
            except Exception as e:
                print(f"  跳过 {r['代码']} {r['信号日']}: {e}", flush=True)
        groups[arch] = files

    # 对照拼图
    contact_sheet(groups.get("赢家_低位反转", []) + groups.get("赢家_趋势延续", []),
                  FIG / "03_赢家图谱_低位反转与趋势延续.png", cols=2,
                  label="赢家：低位反转（上两行） vs 趋势延续（下两行）")
    contact_sheet(groups.get("输家_追高", []) + groups.get("输家_高位放量", []),
                  FIG / "04_输家图谱_追高与高位放量.png", cols=2,
                  label="输家：追高（上两行） vs 高位放量（下两行）")
    contact_sheet(groups.get("b3砍掉大赢家", []) + groups.get("b3救下大输家", []),
                  FIG / "05_b3与死叉买卖点对照.png", cols=2,
                  label="b3 对照：砍掉的大赢家（上两行） vs 救下的大输家（下两行）")
    contact_sheet(groups.get("极端情绪输家", []) + groups.get("低波低量赢家", []),
                  FIG / "06_股性对照_极端情绪与低波低量.png", cols=2,
                  label="股性：极端情绪输家（上两行） vs 低波低量赢家（下两行）")

    # HTML 索引
    rows = []
    for arch, files in groups.items():
        for f in files:
            rel = f.relative_to(HERE)
            rows.append(f"<tr><td>{arch}</td><td><a href='{rel}'>{f.name}</a></td>"
                        f"<td><img src='{rel}' style='width:520px'></td></tr>")
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>MACD 全信号 K 线案例图谱</title>
<style>body{{font-family:-apple-system,'PingFang SC',sans-serif;margin:24px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:8px;vertical-align:top}}
h1{{font-size:20px}}</style></head><body>
<h1>MACD 金叉：K 线案例图谱</h1>
<p>红涨绿跌；阴影为持有期；蓝三角买入，黑倒三角死叉卖，橙叉为 b3 卖。每笔固定 1 万元，不计槽位。</p>
<table><tr><th>类型</th><th>文件</th><th>图</th></tr>{''.join(rows)}</table>
</body></html>"""
    (HERE / "K线案例图谱.html").write_text(html, encoding="utf-8")
    print(f"\n完成：{CASE}\n索引：{HERE/'K线案例图谱.html'}")


if __name__ == "__main__":
    main()
