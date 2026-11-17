#!/usr/bin/env python3
"""47 条件有效性图表（图20-图25）"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度/03_结果"
FIG = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度/04_图表"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linewidth"] = 0.8

C_MACD = "#d95f02"
C_BENCH = "#1f77b4"
C_NEG = "#c0392b"
C_POS = "#2e7d32"
C_GRAY = "#7f7f7f"


def load():
    ld = pd.read_csv(os.path.join(RES, "条件_多头.csv"))
    sd = pd.read_csv(os.path.join(RES, "条件_空头.csv"))
    dd = pd.read_csv(os.path.join(RES, "条件_多空差.csv"))
    ev = pd.read_csv(os.path.join(RES, "条件_事件.csv"))
    cf = pd.read_csv(os.path.join(RES, "条件_经典过滤.csv"))
    yr = pd.read_csv(os.path.join(RES, "条件_逐年.csv"))
    hz = pd.read_csv(os.path.join(RES, "条件_期限.csv"))
    return ld, sd, dd, ev, cf, yr, hz


def fig20(ld, hz):
    a = ld[ld["条件族"] == "证据分数"].copy()
    a["分数"] = a["条件"].str.replace("项", "", regex=False).astype(int)
    a = a.sort_values("分数")
    a = a[a["股票日"] > 0].copy()
    b = hz[(hz["条件"] == "证据分数>=5") & (hz["方向"] == "多")].set_index("周期")["日均超额"]
    # 60 日只算了 >=5，单独标注；20 日主线
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = a["分数"].to_numpy()
    y = a["日均超额"].to_numpy() * 100
    ax.plot(x, y, marker="o", color=C_MACD, lw=2.0, label="20 日超额")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:+.2f}%", (xi, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.axhline(0.15, color=C_POS, lw=0.9, ls="--")
    ax.text(1.05, 0.22, "往返成本 0.15%", color=C_POS, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xlabel("MACD 多头证据数（DIF>0 / DIF>DEA / 柱体上行 / DIF上行 / 周线多头 / 月线多头）")
    ax.set_ylabel("未来 20 日超额 %")
    ax.set_title("证据从 1 项到 5 项单调变差；6 项仍 -0.70%，没有一项为正", fontsize=11.5, loc="left")
    ax.set_ylim(np.nanmin(y) - 0.25, 0.35)
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.01, -0.02, "全部 A 股，2010-2026；T 日状态、T+1 开盘；超额为同日全市场等权 leave-one-out。"
                          "60 日口径：5 项 -1.39%、<=1 项 -0.23%（见条件_期限.csv）。",
             fontsize=7.8, color="#666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图20_证据分数.png"), bbox_inches="tight")
    plt.close(fig)


def fig21(ld, sd, dd):
    zb = ld[ld["条件族"] == "零轴四态"][["条件", "日均超额"]].rename(columns={"日均超额": "多头日均"})
    zs = sd[sd["条件族"] == "零轴四态"][["条件", "日均超额"]].rename(columns={"日均超额": "空头日均"})
    zsp = dd[dd["条件族"] == "零轴四态"][["条件", "多空差"]]
    z = zb.merge(zs, on="条件", how="outer").merge(zsp, on="条件", how="left")
    z = z.set_index("条件").reindex(["双上(DIF>0,DEA>0)", "双下(DIF<=0,DEA<=0)"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    x = np.arange(len(z)); w = 0.36
    axes[0].bar(x - w / 2, z["多头日均"] * 100, w, color=C_MACD, label="DIF>DEA（多）")
    axes[0].bar(x + w / 2, z["空头日均"] * 100, w, color=C_BENCH, alpha=0.85, label="DIF<=DEA（空）")
    axes[0].axhline(0, color="#555", lw=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["零轴双上\n(DIF>0, DEA>0)", "零轴双下\n(DIF<=0, DEA<=0)"])
    axes[0].set_ylabel("未来 20 日超额 %")
    axes[0].set_title("零轴双上：多 -0.90%、空 -0.40%（差 -0.38%, t=-3.6）；双下差≈0", fontsize=9.6, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)
    for i, (_, row) in enumerate(z.iterrows()):
        axes[0].text(i - w / 2, row["多头日均"] * 100 + 0.03, f"{row['多头日均']*100:+.2f}",
                     ha="center", va="bottom", fontsize=8)
        axes[0].text(i + w / 2, row["空头日均"] * 100 + 0.03, f"{row['空头日均']*100:+.2f}",
                     ha="center", va="bottom", fontsize=8)

    # 柱体四态：红柱只有多、绿柱只有空
    labs = ["红柱放大", "红柱缩小", "绿柱缩短", "绿柱放大"]
    vals, colors = [], []
    for lab in labs:
        if lab.startswith("红"):
            v = ld[(ld["条件族"] == "柱体四态") & (ld["条件"] == lab)]["日均超额"]
            vals.append(v.iloc[0] * 100 if len(v) else np.nan); colors.append(C_MACD)
        else:
            v = sd[(sd["条件族"] == "柱体四态") & (sd["条件"] == lab)]["日均超额"]
            vals.append(v.iloc[0] * 100 if len(v) else np.nan); colors.append(C_BENCH)
    x2 = np.arange(len(labs))
    axes[1].bar(x2, vals, color=colors, alpha=0.85)
    axes[1].axhline(0, color="#555", lw=0.9)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(labs)
    axes[1].set_ylabel("未来 20 日超额 %")
    axes[1].set_title("柱体四态：红柱态看多，绿柱态看空", fontsize=10, loc="left")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.03, f"{v:+.2f}%", ha="center", va="bottom", fontsize=8.5)
    fig.text(0.52, -0.03, "红柱态没有空头样本，绿柱态没有多头样本；柱体是单边状态，不是多空对比。",
             fontsize=7.8, color="#666")
    fig.text(0.01, -0.03, "左图：多/空为各自全样本日均，多空差为同一批交易日（不一定等于两柱之差）；右图：各自单边样本的全样本均值。",
             fontsize=7.8, color="#666")
    fig.suptitle("MACD 最强确认处，多头判断最差；绿柱/零轴下的空头判断才是有效的",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "图21_零轴与柱体.png"), bbox_inches="tight")
    plt.close(fig)


def fig22(ld, dd):
    m = ld.merge(dd[["条件族", "条件", "MACD增量", "增量t"]], on=["条件族", "条件"], how="left")
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    colmap = {"仅价格趋势": C_BENCH, "仅相对有效": C_MACD, "反向": C_NEG, "无效": C_GRAY, "样本不足": "#cccccc"}
    for k, g in m.groupby("综合判定"):
        ax.scatter(g["日均超额"] * 100, g["MACD增量"] * 100, s=52,
                   color=colmap.get(k, C_GRAY), label=k, alpha=0.85, edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="#555", lw=0.9)
    ax.axvline(0.15, color=C_POS, lw=0.9, ls="--")
    ax.axvline(0, color="#aaa", lw=0.6)
    ax.fill_between([0.15, 1.2], 0, 1.2, color=C_POS, alpha=0.06)
    ax.text(0.47, 0.97, "有效区：绝对 >0.15%\n且增量 t>2\n0 格", fontsize=9.2, color=C_POS,
            ha="center", va="top")
    labmap = {
        ("柱体水平", "Q3(3)"): ("柱体Q3", (8, 4)),
        ("柱体水平", "Q4(4)"): ("柱体Q4", (8, -12)),
        ("DIF水平", "Q2(2)"): ("DIF Q2", (8, 8)),
        ("DIF5日斜率", "Q3(3)"): ("斜率Q3", (-42, 10)),
        ("DIF5日斜率", "Q4(4)"): ("斜率Q4", (8, -14)),
        ("柱体水平", "Q1(最低)"): ("柱体Q1", (-38, -12)),
        ("柱体60日z", "Q1(最低)"): ("zQ1", (8, -12)),
        ("零轴四态", "双下(DIF<=0,DEA<=0)"): ("零轴双下", (8, 8)),
        ("周月状态", "周线空头+月线空头"): ("周空月空", (8, 8)),
        ("市场状态", "指数在MA200下且下行"): ("市场下行", (-10, -18)),
        ("零轴四态", "双上(DIF>0,DEA>0)"): ("零轴双上", (8, -14)),
        ("DIF水平", "Q5(最高)"): ("DIF Q5", (8, 8)),
        ("DIF5日斜率", "Q5(最高)"): ("斜率Q5", (8, 8)),
    }
    for (fam, cond), (txt, (dx, dy)) in labmap.items():
        row = m[(m["条件族"] == fam) & (m["条件"] == cond)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        ax.annotate(txt, (r["日均超额"] * 100, r["MACD增量"] * 100),
                    xytext=(dx, dy), textcoords="offset points", fontsize=7.8, color="#333",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#999"))
    ax.text(0.17, 0.40, "蓝点 y>0，\n但增量 t<2", fontsize=7.6, color=C_BENCH)
    ax.set_xlim(-1.3, 0.55)
    ax.set_ylim(-1.6, 1.2)
    ax.set_xlabel("多头判断：全样本日均 20 日超额 %")
    ax.set_ylabel("MACD 相对 MA60 的增量 %（同一批交易日）")
    ax.set_title("右上角为空：绝对为正的格子没有 MACD 增量；有增量的格子绝对为负", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, -0.02, "每个点是一个条件格子；x 为全部有效日的多头超额，y 为同一批交易日中 "
                          "MACD 多空差减 MA60 多空差。增量 t>2 才算 MACD 有独立判断力。",
             fontsize=7.8, color="#666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图22_绝对与增量.png"), bbox_inches="tight")
    plt.close(fig)


def fig23(ld, sd):
    order = ["距交叉0-5日", "6-15日", "16-40日", "41-120日"]
    b = ld[ld["条件族"] == "交叉年龄"].set_index("条件").reindex(order)
    s = sd[sd["条件族"] == "交叉年龄"].set_index("条件").reindex(order)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(x, b["日均超额"] * 100, marker="o", color=C_MACD, lw=2, label="DIF>DEA（多）")
    ax.plot(x, s["日均超额"] * 100, marker="s", color=C_BENCH, lw=2, label="DIF<=DEA（空）")
    ax.axhline(0, color="#555", lw=0.9)
    for i, (v1, v2) in enumerate(zip(b["日均超额"] * 100, s["日均超额"] * 100)):
        ax.annotate(f"{v1:+.2f}%", (i, v1), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=C_MACD)
        if np.isfinite(v2):
            ax.annotate(f"{v2:+.2f}%", (i, v2), xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=8.5, color=C_BENCH)
    ax.set_xticks(x); ax.set_xticklabels(["0-5 日", "6-15 日", "16-40 日", "41-120 日"])
    ax.set_xlabel("距最近一次交叉的交易日数")
    ax.set_ylabel("未来 20 日超额 %")
    ax.set_title("交叉后持有越久，多头判断越差；空头判断在 40 日后反转为正", fontsize=11.5, loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.text(0.01, -0.02, "年龄按绝对天数分桶；多头格是金叉后的年龄，空头格是死叉后的年龄。"
                          "41-120 日多头样本仅 5.8 万股票日，t=-2.5，方向可参考、幅度不作精确估计。",
             fontsize=7.8, color="#666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图23_交叉年龄.png"), bbox_inches="tight")
    plt.close(fig)


def fig24(cf):
    d = cf.copy()
    d["净超额"] = d["净超额"] * 100
    d = d.sort_values("净超额")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(d))
    colors = [C_POS if v > 0 else C_NEG for v in d["净超额"]]
    ax.barh(y, d["净超额"], color=colors, alpha=0.82)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(d["过滤"])
    for i, (v, t) in enumerate(zip(d["净超额"], d["t值"])):
        ax.text(v + (-0.12 if v < 0 else 0.12), i, f"{v:+.2f}% (t={t:+.1f})",
                va="center", ha="right" if v < 0 else "left", fontsize=8)
    ax.set_xlabel("扣 0.15% 往返成本后的 20 日超额 %")
    ax.set_title("金叉买入：只有“证据<=1”微正（+0.26%，t=1.4）；零轴上金叉是方向性过滤里最差，DIF 距离 Q5 更差",
                 fontsize=9.6, loc="left")
    ax.set_xlim(d["净超额"].min() - 0.8, 0.6)
    fig.text(0.01, -0.02, "519,111 个 2010 年以来的日线金叉；T 日收盘确认、T+1 开盘买入、持有 20 日；"
                          "超额为同日全市场等权 leave-one-out。", fontsize=7.8, color="#666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图24_经典过滤.png"), bbox_inches="tight")
    plt.close(fig)


def fig25(yr):
    conds = ["证据分数>=5", "红柱放大", "零轴双上", "全部多头", "日周月共振",
             "顶背离", "全部多头", "零轴双下", "绿柱缩短", "证据分数<=1", "底背离"]
    # 去重，保留有意义的顺序
    conds = ["证据分数>=5", "红柱放大", "零轴双上", "日周月共振", "顶背离",
             "全部多头", "零轴双下", "绿柱缩短", "证据分数<=1", "底背离"]
    p = yr.pivot_table(index="条件", columns="年份", values="日均超额")
    p = p.reindex(conds)
    years = [c for c in p.columns if c >= 2010]
    p = p[years]
    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    data = p.to_numpy(dtype=float) * 100
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-1.6, vmax=1.6)
    ax.set_xticks(np.arange(len(years))); ax.set_xticklabels([str(int(y)) for y in years], fontsize=8)
    ax.set_yticks(np.arange(len(conds))); ax.set_yticklabels(conds, fontsize=9)
    for i in range(len(conds)):
        for j in range(len(years)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.3,
                        color="white" if abs(v) > 1.0 else "#222")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=6.3, color="#999")
    ax.set_title("逐年日均 20 日超额（%）：强确认状态几乎年年为负，证据分数>=5 为 17/17 负",
                 fontsize=11.5, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% / 20 日")
    ax.grid(False)
    fig.text(0.01, -0.04, "每个格子是该条件在该年度的日度横截面均值；负=跑输全市场等权。"
                          "证据分数<=1 的 2010/2013/2014 年样本不足，标“—”。", fontsize=7.8, color="#666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图25_逐年.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ld, sd, dd, ev, cf, yr, hz = load()
    fig20(ld, hz)
    fig21(ld, sd, dd)
    fig22(ld, dd)
    fig23(ld, sd)
    fig24(cf)
    fig25(yr)
    print("figures 20-25 done")
