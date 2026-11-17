#!/usr/bin/env python3
"""09 汇总：中文数据表（xlsx）+ 图表 PNG + HTML 一页纸"""
from __future__ import annotations

import base64
import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(HERE, "02_代码")
DB = os.path.join(CODE, "work.duckdb")


def event_stats(con):
    q = """
    SELECT strftime(date,'%Y') AS 年份, count(*) AS 事件数,
           count(DISTINCT code) AS 股票数,
           avg(ex60) AS 平均超额60, median(ex60) AS 中位超额60,
           avg(ret60) AS 绝对收益60
    FROM ev
    WHERE box_prev >= 30 AND comp_prev <= 0.7 AND vshr_prev <= 0.8
      AND close > hh20_prev AND vr >= 1.5 AND pos250 >= 0.85
    GROUP BY 1 ORDER BY 1
    """
    return con.execute(q).df()


def make_charts(d: dict, outdir: str) -> dict:
    paths = {}

    # 1 涨跌腿中位时长
    s = d["sens"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for sign, label, color in [(1, "上涨腿", "#c0392b"), (-1, "下跌腿", "#2c7fb8")]:
        sub = s[s["方向"] == sign].sort_values("θ")
        ax.plot(sub["θ"] * 100, sub["中位天数"], marker="o", label=label, color=color)
    ax.set_xlabel("Zigzag 反转阈值（%）")
    ax.set_ylabel("中位持续天数")
    ax.set_title("涨跌腿中位时长：涨得久、跌得快")
    ax.grid(alpha=0.3)
    ax.legend()
    p = os.path.join(outdir, "c1_legs.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths["legs"] = p

    # 2 横盘时长 vs 未来60日
    b = d["boxcond"].copy()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(b))
    ax.bar(x - 0.2, b["平均超额60"] * 100, width=0.4, label="平均超额60日（%）", color="#2c7fb8")
    ax.bar(x + 0.2, b["中位未来60日最大涨幅"] * 100, width=0.4, label="中位未来60日最大涨幅（%）", color="#f39c12")
    ax.set_xticks(x); ax.set_xticklabels(b["横盘天数"])
    ax.set_xlabel("横盘游程（交易日）")
    ax.set_title("横盘越久，后面涨得越好？——没有")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    p = os.path.join(outdir, "c2_box.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths["box"] = p

    # 3 72 组 IS vs OOS
    g = d["grid"]
    is_ = g[g["period"] == "样本内06-15"].set_index(["N", "q", "v", "m", "p"])
    oos = g[g["period"] == "样本外16-26"].set_index(["N", "q", "v", "m", "p"])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(is_["ex60"] * 100, oos["ex60"] * 100, s=18, alpha=0.6, color="#7f8c8d")
    ax.axhline(0, color="black", linewidth=0.8); ax.axvline(0, color="black", linewidth=0.8)
    top = oos["ex60"].idxmax()
    ax.scatter([is_.loc[top, "ex60"] * 100], [oos.loc[top, "ex60"] * 100], s=60, color="#c0392b", zorder=5)
    ax.annotate("N=30,q=0.7,v=0.8,m=1.5,p=0.85", (is_.loc[top, "ex60"] * 100, oos.loc[top, "ex60"] * 100),
                xytext=(-8, -14), textcoords="offset points", fontsize=8, color="#c0392b", ha="right")
    ax.set_xlabel("样本内 2006-2015 超额60（%）")
    ax.set_ylabel("样本外 2016-2026 超额60（%）")
    ax.set_title("72 组参数：样本内与样本外方向不一致")
    ax.grid(alpha=0.3)
    p = os.path.join(outdir, "c3_grid.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths["grid"] = p

    # 4 年度超额
    y = d["yearly"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = ["#c0392b" if v > 0 else "#2c7fb8" for v in y["平均超额60"]]
    ax.bar(y["年份"].astype(str), y["平均超额60"] * 100, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("平均超额60（%）")
    ax.set_title("代表组合的年度超额：13/25 年为正")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.grid(alpha=0.3, axis="y")
    p = os.path.join(outdir, "c4_year.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths["year"] = p

    # 5 压缩×位置 热力图（N=30, v=0.8, m=1.5）
    g = d["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, per, title in [(axes[0], "样本内06-15", "样本内 2006-2015"),
                           (axes[1], "样本外16-26", "样本外 2016-2026")]:
        sub = g[(g["period"] == per) & (g["N"] == 30) & (g["v"] == 0.8) & (g["m"] == 1.5)]
        piv = sub.pivot_table(index="q", columns="p", values="ex60") * 100
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-5, vmax=2, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(["不限", "≥0.85"])
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xlabel("位置：收盘/250日最高"); ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=9,
                        color="black" if abs(v) < 3.5 else "white")
    axes[0].set_ylabel("压缩：ATR20/ATR250")
    fig.suptitle("压缩越紧越差，位置越高越好", fontsize=12)
    p = os.path.join(outdir, "c5_heat.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    paths["heat"] = p
    return paths


def img64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def main() -> None:
    con = duckdb.connect(DB)
    con.execute("PRAGMA threads=8")
    d = {
        "legs": pd.read_csv(os.path.join(CODE, "h1_legs.csv")),
        "sens": pd.read_csv(os.path.join(CODE, "h1_sensitivity.csv")),
        "box": pd.read_csv(os.path.join(CODE, "h1_box.csv")),
        "boxcond": pd.read_csv(os.path.join(CODE, "h5_box_condition.csv")),
        "dz": pd.read_csv(os.path.join(CODE, "h5_dz_condition.csv")),
        "leader": pd.read_csv(os.path.join(CODE, "h2_leader_summary.csv")),
        "leaderpct": pd.read_csv(os.path.join(CODE, "h2_leader_pct.csv")),
        "leadery": pd.read_csv(os.path.join(CODE, "h2_leader_by_year.csv")),
        "rally": pd.read_csv(os.path.join(CODE, "h2_rally_summary.csv")),
        "rallypct": pd.read_csv(os.path.join(CODE, "h2_rally_pct.csv")),
        "grid": pd.read_csv(os.path.join(CODE, "h3_grid.csv")),
        "ma": pd.read_csv(os.path.join(CODE, "h4_ma.csv")),
        "boot": pd.read_csv(os.path.join(CODE, "h3_bootstrap.csv")),
        "audit": con.execute("SELECT * FROM audit_year").df(),
        "yearly": event_stats(con),
    }
    d["yearly"].to_csv(os.path.join(CODE, "h3_best_yearly.csv"), index=False)

    grid_cn = d["grid"].rename(columns={
        "N": "横盘天数N", "q": "压缩阈值q", "v": "缩量阈值v", "m": "放量倍数m",
        "p": "位置阈值p", "period": "区间", "n": "样本数", "days": "事件天数",
        "ex20": "超额20", "ex60": "超额60", "ex120": "超额120",
        "med_ex60": "中位超额60", "win60": "胜率60", "abs60": "绝对收益60",
        "nw_t60": "t值60", "ex60_all": "全市场超额60", "nw_t60_all": "全市场t值60",
    })
    grid_cn = grid_cn[["横盘天数N", "压缩阈值q", "缩量阈值v", "放量倍数m", "位置阈值p", "区间",
                       "样本数", "事件天数", "超额20", "超额60", "超额120", "中位超额60",
                       "胜率60", "绝对收益60", "t值60", "全市场超额60", "全市场t值60"]]

    note = pd.DataFrame({
        "项目": ["数据源", "复权", "股票池", "样本区间", "时点处理", "基准",
                 "入场/出场", "成本", "多重检验", "结论口径"],
        "说明": [
            "baostock 本地 SQLite（只读），与 HiThink 主库抽样交叉核验（最大日收益偏差 0.29%，中位 0）",
            "后复权；除权不产生假跳空。价格水平与历史屏幕不同，形态比率不受影响",
            "沪深主板/创业板/科创板；含 341 只退市股，不含北交所与 B 股",
            "1999-01 至 2026-10；主结论以 2006 年起为准；样本内 2006-2015，样本外 2016-2026",
            "ST、停牌用当日字段；不用当前简称；信号只用 t 日及以前数据",
            "同筛选条件的全样本等权（主口径）；全市场等权为对照口径",
            "T+1 开盘入场；T+1 停牌或一字涨停视为不可成交；持有 20/60/120 日收盘出场",
            "固定持有；未计成本；表内另给 0.2% 双边成本后的绝对收益参考",
            "72 组参数全部报告；20 日块 bootstrap 最大统计量检验",
            "描述性解剖与可交易信号严格分开；不达预注册标准即判未通过",
        ],
    })

    xlsx = os.path.join(HERE, "03_数据表.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        note.to_excel(w, sheet_name="说明", index=False)
        d["legs"].to_excel(w, sheet_name="时长_涨跌腿", index=False)
        d["sens"].to_excel(w, sheet_name="时长_敏感性", index=False)
        d["box"].to_excel(w, sheet_name="时长_箱体", index=False)
        d["boxcond"].to_excel(w, sheet_name="时长_横盘条件", index=False)
        d["dz"].to_excel(w, sheet_name="时长_距高点条件", index=False)
        d["leader"].to_excel(w, sheet_name="龙头_摘要", index=False)
        d["leaderpct"].to_excel(w, sheet_name="龙头_分位", index=False)
        d["leadery"].to_excel(w, sheet_name="龙头_分年", index=False)
        d["rally"].to_excel(w, sheet_name="主升浪_解剖", index=False)
        d["rallypct"].to_excel(w, sheet_name="主升浪_分位", index=False)
        grid_cn.to_excel(w, sheet_name="形态_网格72", index=False)
        d["yearly"].to_excel(w, sheet_name="形态_代表组合年度", index=False)
        d["ma"].to_excel(w, sheet_name="均线对照", index=False)
        d["boot"].to_excel(w, sheet_name="多重检验", index=False)
        d["audit"].to_excel(w, sheet_name="数据审计", index=False)
    print("xlsx", xlsx)

    charts = make_charts(d, CODE)
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>盘感量化：裸K时长与形态</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;max-width:980px;margin:36px auto;padding:0 20px;color:#1f2937;line-height:1.75;background:#fbfbfa}}
h1{{font-size:26px;margin-bottom:4px}} h2{{font-size:19px;margin-top:34px;border-left:4px solid #c0392b;padding-left:10px}}
.sub{{color:#6b7280;font-size:13px;margin-bottom:24px}}
.kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.kpi div{{flex:1;min-width:180px;background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px}}
.kpi b{{display:block;font-size:22px;color:#c0392b}} .kpi span{{font-size:12px;color:#6b7280}}
img{{width:100%;border:1px solid #e5e7eb;border-radius:8px;background:#fff;margin:8px 0 4px}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
th,td{{border:1px solid #e5e7eb;padding:6px 8px;text-align:right}} th{{background:#f3f4f6}}
td:first-child,th:first-child{{text-align:left}}
.note{{font-size:12px;color:#6b7280}} .warn{{background:#fff7ed;border:1px solid #fed7aa;padding:12px;border-radius:8px}}
</style></head><body>
<h1>盘感量化：裸K时长、三阶段形态与前向收益</h1>
<div class="sub">数据 1999-2026 · 5,564 只（含 341 只退市） · 预注册见 00_预注册_研究设计.md · 生成日期 2026-10-30</div>

<div class="kpi">
<div><b>43 / 37 天</b><span>20% 阈值下，上涨腿 / 下跌腿中位时长；上涨中位 +47%，下跌 -47%</span></div>
<div><b>16 天</b><span>横盘箱体中位时长（箱宽≤15%），四分位 9–31 天</span></div>
<div><b>-1.4% / +0.9%</b><span>代表性形态组合的 60 日超额：样本内 / 样本外</span></div>
<div><b>p = 0.50</b><span>72 组参数最大统计量 bootstrap；观测 t=2.31，零分布中位 2.29</span></div>
</div>

<h2>一句话结论</h2>
<p>这套"看时长、看形态、看量能、去均线"的方法，作为<strong>训练眼睛的框架</strong>是成立的：它逼你建立时长分布、位置与量能的对照系。但把它机械化为可交易规则后，<strong>72 组参数没有一组同时满足样本内为正、样本外显著、参数邻域稳定、并通过多重检验</strong>。最接近的组合样本外 60 日超额 +0.88%（t=2.31），但样本内是 -1.42%，且 family-wise p=0.50——这正是"从 72 次试验里挑出最好看的一次"的典型噪声。</p>

<h2>① 时长：涨得久、跌得快</h2>
<img src="{img64(charts['legs'])}">
<p class="note">Zigzag 只统计已完成腿；θ=20% 时上涨腿中位 43 天、下跌腿 37 天；θ 越大差距越大（30%：80 天 vs 67 天）。横盘箱体中位 16 天，但分布很偏，10% 的箱体超过 58 天。</p>

<h2>② 横盘越久，后面涨得越好？没有</h2>
<img src="{img64(charts['box'])}">
<p class="note">横盘 80 天以上的样本，未来 60 日平均超额 -0.9%，未来 60 日最大涨幅中位仅 7.6%；横盘 10–19 天的样本反而最好（+1.0%、14.9%）。横盘越久，股票越偏低位波动型，后续弹性反而更小。</p>

<h2>③ 形态突破：样本内外方向相反</h2>
<img src="{img64(charts['grid'])}">
<p class="note">72 组参数中，样本外 60 日超额最大值 +1.06%，但样本内为负；统计上最"像"的组合（N=30, q=0.7, v=0.8, m=1.5, p=0.85）样本外 +0.88%（t=2.31）、样本内 -1.42%（t=-1.11）。胜率 44%，中位超额 -1.8%——靠右尾拉平均。</p>

<h2>④ 年度稳定性</h2>
<img src="{img64(charts['year'])}">
<p class="note">代表组合 25 个有事件的年份里 13 年为正；2025 年 -2.7%、2026 年 -1.5%。没有"年年有效"的证据。</p>

<h2>⑤ 去掉均线：不损失信息，但也没多出信息</h2>
<table>
<tr><th>样本</th><th>样本数</th><th>60日超额</th><th>日聚类t</th></tr>
<tr><td>形态 + 均线多头</td><td>1,513</td><td>-1.02%</td><td>-0.88</td></tr>
<tr><td>形态 + 非多头</td><td>2,247</td><td>+0.45%</td><td>0.80</td></tr>
<tr><td>均线金叉 + 放量（无形态）</td><td>29,432</td><td>-1.37%</td><td>-7.46</td></tr>
<tr><td>全样本均线多头（背景）</td><td>3,549,878</td><td>-0.33%</td><td>-9.47</td></tr>
</table>
<p class="note">叠加均线多头条件后超额反而低 1.2 个百分点，差值置信区间 [-2.8%, +0.6%] 跨 0；单独用均线金叉+放量是明显的负超额。支持"去掉均线"，但原因是均线没有增量，不是裸K本身有预测力。</p>

<h2>⑥ 一个稳定的反向发现：越紧越差</h2>
<img src="{img64(charts['heat'])}">
<p class="note">同一横轴位置下，压缩最紧（q=0.5）的箱体突破最差：样本内 -4.1%、样本外 -1.4%（p 不限）；靠近一年高点（p≥0.85）能把结果抬升约 0.7–1.4 个百分点，但抬不到稳定为正。q=0.5 的 24 组参数在样本内、样本外**全部为负**（样本内均值 -5.3%，样本外均值 -1.7%），说明"越紧越差"不是单点噪声——这与"缩量窄幅横盘后突破"的流行说法相反。</p>

<h2>⑦ 龙头解剖（描述性，不是信号）</h2>
<p>1,843 段幅度≥50%、≤250 日的主升浪，启动点前一日：距一年高点分位中位 <b>0.23</b>（低位启动）、成交额萎缩分位 <b>0.44</b>（略低于中位）、波动压缩分位 <b>0.41</b>（并不紧）、20 日箱宽分位 <b>0.73</b>（反而偏宽）。<strong>"长期缩量窄幅横盘再突破"不是大行情启动的典型画像；更常见的是深跌之后的低位置、宽幅反转。</strong>中位前一段下跌腿 59 天、-78%，但这是"已经发生大行情"条件下的解剖，存在选择偏差，不能倒推。</p>

<h2>方法与边界</h2>
<p class="note">后复权日线；时点 ST/停牌；T+1 开盘入场，一字涨停剔除；同筛选条件等权基准；72 组参数全报；20 日块 bootstrap 最大统计量。未计成本（另给 0.2% 双边参考）。日线无法还原盘中盘口，本研究只检验"裸K日线"这一层。完整定义与修订记录见预注册文档，全部数字见 03_数据表.xlsx。</p>
<div class="warn">本研究不构成投资建议。任何单一形态在 A 股都没有稳定到可以机械执行的程度；若要用，只作为缩小观察池的第一步，最终决策仍需仓位、流动性与大盘环境配合。</div>
</body></html>"""
    hp = os.path.join(HERE, "04_可视化.html")
    open(hp, "w", encoding="utf-8").write(html)
    print("html", hp)
    con.close()


if __name__ == "__main__":
    main()
