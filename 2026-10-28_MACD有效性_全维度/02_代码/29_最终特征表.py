#!/usr/bin/env python3
"""29 汇总特征层的中文表与图 9-12"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"
RES = os.path.join(ROOT, "03_结果")
FIG = os.path.join(ROOT, "04_图表")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#eeeeee"
C_POS, C_NEG, C_GRAY, C_BLUE, C_ORANGE = "#1f8a70", "#c0392b", "#95a5a6", "#2c6fbb", "#e67e22"


def main():
    fs = pd.read_csv(os.path.join(RES, "特征_全样本.csv"))
    st = pd.read_csv(os.path.join(RES, "特征_结构.csv"))
    sta = pd.read_csv(os.path.join(RES, "特征_状态_含q.csv"))
    strict = pd.read_csv(os.path.join(RES, "特征_严格增量.csv"))
    sst = pd.read_csv(os.path.join(RES, "特征_结构状态.csv"))
    md = pd.read_csv(os.path.join(RES, "月度下行_稳健.csv"))
    mdfm = pd.read_csv(os.path.join(RES, "月度下行_稳健FM.csv"))
    mdd = pd.read_csv(os.path.join(RES, "月度下行_分布.csv"))
    port = pd.read_csv(os.path.join(RES, "特征_组合.csv"))
    pf = pd.read_csv(os.path.join(RES, "周期_特征.csv"))
    ps = pd.read_csv(os.path.join(RES, "周期_特征结构.csv"))

    # 1) 全样本 20 日
    cols = ["特征", "股票日", "交易日数", "平均IC", "IC_t", "IC大于0占比", "多空5_1", "多空t",
            "Q1", "Q5", "单调性", "IC_2010_2017", "IC_2018_2026", "多空大于0年份", "年份数"]
    t1 = fs[fs.周期 == 20][cols].copy()
    t1.to_csv(os.path.join(RES, "最终_特征全样本.csv"), index=False, encoding="utf-8-sig")

    # 2) 结构热图：柱体水平
    order = [("长期趋势", ["上行", "下行", "过渡"]), ("中期排列", ["多头排列", "空头排列", "混合"]),
             ("250日位置", ["高位", "中位", "深位"]), ("短期结构", ["高高低低", "低高低低", "收敛/其他"]),
             ("波动", ["低波", "中波", "高波"]), ("换手", ["低换手", "中换手", "高换手"]),
             ("市场状态", ["指数上行", "指数下行", "过渡"])]
    rows = []
    for dim, labs in order:
        for lab in labs:
            x = st[(st.维度 == dim) & (st.分组 == lab) & (st.特征 == "柱体水平")]
            if len(x):
                r = x.iloc[0]
                rows.append(dict(维度=dim, 分组=lab, 平均IC=r.平均IC, IC_t=r.IC_t,
                                 多空5_1=r.多空5_1, 多空t=r.多空t, q_bh=r.q_bh))
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(RES, "最终_特征结构.csv"), index=False, encoding="utf-8-sig")

    # 3) 状态矩阵
    keep = ["红柱放大", "柱体上行", "DIF>DEA", "DIF上行"]
    t3 = sta[sta.状态.isin(keep)][["状态", "格子名称", "状态1占比", "原始差", "原始t",
                                   "控制涨幅后差", "控制t", "q_all",
                                   "原始差_2010_2017", "原始差_2018_2026", "交易日数"]]
    t3.to_csv(os.path.join(RES, "最终_状态矩阵.csv"), index=False, encoding="utf-8-sig")

    # 4) 严格增量
    t4 = strict[strict.项.isin(["state", "state_x_level"])][
        ["检验", "项", "系数", "t", "系数_2010_2017", "系数_2018_2026", "正系数年份", "年份数", "交易日数"]].copy()
    t4["项"] = t4.项.map({"state": "状态", "state_x_level": "状态×水平"})
    t4.to_csv(os.path.join(RES, "最终_特征严格增量.csv"), index=False, encoding="utf-8-sig")

    # 5) 结构状态（红柱放大 / DIF>DEA）
    t5 = sst[sst.状态.isin(["红柱放大", "DIF>DEA"])][
        ["维度", "分组", "状态", "控制后差", "控制t", "q_bh", "原始差", "原始t",
         "状态1绝对", "状态0绝对", "市场绝对", "交易日数"]].copy()
    t5.to_csv(os.path.join(RES, "最终_结构状态.csv"), index=False, encoding="utf-8-sig")

    # 6) 月度下行
    t6 = md[md.样本.isin(["下行_全", "下行_非ST_高价_高流动性", "下行_非ST_高价_低流动性",
                          "上行_非ST_高价_高流动性"])][
        ["周期", "样本", "月数", "平均IC", "IC_t12", "IC_2010_2017", "IC_2018_2026",
         "Q1绝对", "Q5绝对", "Q1超额", "Q5超额", "样本绝对"]].copy()
    t6.to_csv(os.path.join(RES, "最终_月度下行.csv"), index=False, encoding="utf-8-sig")
    t7 = mdfm[["周期", "检验", "项", "系数", "t", "系数_2010_2017", "系数_2018_2026", "正系数年份", "年份数"]].copy()
    t7["项"] = t7.项.map({"state": "状态", "state_x_level": "状态×水平", "level": "水平", "level_cont": "水平(连续)"})
    t7.to_csv(os.path.join(RES, "最终_月度下行FM.csv"), index=False, encoding="utf-8-sig")

    # 7) 周期特征
    t8 = pf[pf.特征.isin(["柱体水平", "DIF水平"])][
        ["周期", "持有期", "特征", "期数", "股票期", "平均IC", "IC_t", "多空5_1", "多空t",
         "IC_2010_2017", "IC_2018_2026"]].copy()
    t8.to_csv(os.path.join(RES, "最终_周期特征.csv"), index=False, encoding="utf-8-sig")
    t9 = ps[(ps.特征 == "柱体水平") & (ps.持有期.isin([12]))][
        ["周期", "持有期", "趋势", "期数", "平均IC", "IC_t", "多空5_1", "多空t"]].copy()
    t9.to_csv(os.path.join(RES, "最终_周期特征结构.csv"), index=False, encoding="utf-8-sig")

    # 8) 组合
    pick = ["全A等权", "剔除红柱放大", "剔除柱体上行", "剔除DIF>DEA", "只买DIF最低20%",
            "只买20日涨幅最低20%", "只买60日涨幅最低20%", "只买低波20%", "空头排列+绿柱缩短"]
    t10 = port[port.策略.isin(pick)][["策略", "区间", "年化", "回撤", "夏普", "胜率", "平均持仓"]].copy()
    t10.to_csv(os.path.join(RES, "最终_特征组合.csv"), index=False, encoding="utf-8-sig")

    # 图9 全样本 IC
    macd_names = ["柱体水平", "柱体3日变化", "柱体加速度", "20日柱体均值", "柱体60日z",
                  "DIF水平", "DIF-DEA距离", "DIF5日斜率", "DEA5日斜率", "DIF60日z",
                  "交叉后天数", "20日红柱占比"]
    d = t1[t1.特征.isin(macd_names)].sort_values("平均IC")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    colors = [C_POS if v >= 0 else C_NEG for v in d.平均IC]
    ax.barh(d.特征, d.平均IC * 100, color=colors, alpha=.85)
    ax.axvline(0, color="#333", lw=1)
    for i, (v, t) in enumerate(zip(d.平均IC * 100, d.IC_t)):
        ax.text(0.08 if v < 0 else v + 0.08, i, f"{v:+.2f}% (t={t:+.1f})",
                va="center", ha="left", fontsize=8)
    ax.set_xlabel("20 日超额收益的日度 Spearman IC（%）")
    ax.set_title("12 个 MACD 连续特征：全样本 IC 全部为负", loc="left", fontsize=11.5)
    ax.set_xlim(-8.4, 4.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图9_特征IC.png"))
    plt.close(fig)

    # 图10 结构热图
    mat = np.full((len(order), 3), np.nan)
    for i, (dim, labs) in enumerate(order):
        for j, lab in enumerate(labs):
            x = t2[(t2.维度 == dim) & (t2.分组 == lab)]
            if len(x):
                mat[i, j] = x.平均IC.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["第1桶", "第2桶", "第3桶"], fontsize=8)
    # 每行标签不同，改用文字标注
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{dim}：{labs[0]}/{labs[1]}/{labs[2]}" for dim, labs in order], fontsize=8)
    for i in range(len(order)):
        for j in range(3):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.2f}%", ha="center", va="center", fontsize=8.5)
    ax.set_title("柱体水平 IC（20 日，%）：只有中期空头排列弱正（t=1.4）", loc="left", fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图10_结构热图.png"))
    plt.close(fig)

    # 图11 月度下行
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))
    ax = axes[0]
    labels, vals3, vals12 = [], [], []
    for s in ["下行_全", "下行_非ST_高价_高流动性", "下行_非ST_高价_低流动性", "上行_非ST_高价_高流动性"]:
        labels.append(s.replace("下行_", "下/").replace("上行_", "上/").replace("非ST_高价_", ""))
        x3 = t6[(t6.周期 == 3) & (t6.样本 == s)]
        x12 = t6[(t6.周期 == 12) & (t6.样本 == s)]
        vals3.append(x3.平均IC.iloc[0] * 100 if len(x3) else np.nan)
        vals12.append(x12.平均IC.iloc[0] * 100 if len(x12) else np.nan)
    xx = np.arange(len(labels)); w = .36
    ax.bar(xx - w/2, vals3, w, label="3 个月", color=C_BLUE, alpha=.85)
    ax.bar(xx + w/2, vals12, w, label="12 个月", color=C_ORANGE, alpha=.85)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("IC（%）"); ax.set_title("月线柱体水平：下跌趋势正、上涨趋势负", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(vals3, vals12)):
        ax.text(i - w/2, a + (0.3 if a >= 0 else -0.6), f"{a:+.1f}", ha="center", fontsize=7.5)
        ax.text(i + w/2, b + (0.3 if b >= 0 else -0.6), f"{b:+.1f}", ha="center", fontsize=7.5)
    ax = axes[1]
    d3 = mdd[mdd.周期 == 3]; d12 = mdd[mdd.周期 == 12]
    labels2 = ["3m Q1", "3m Q5", "12m Q1", "12m Q5"]
    med = [d3[d3.分位 == "Q1"].中位绝对.iloc[0], d3[d3.分位 == "Q5"].中位绝对.iloc[0],
           d12[d12.分位 == "Q1"].中位绝对.iloc[0], d12[d12.分位 == "Q5"].中位绝对.iloc[0]]
    mean = [d3[d3.分位 == "Q1"].均值绝对.iloc[0], d3[d3.分位 == "Q5"].均值绝对.iloc[0],
            d12[d12.分位 == "Q1"].均值绝对.iloc[0], d12[d12.分位 == "Q5"].均值绝对.iloc[0]]
    xx = np.arange(4)
    ax.bar(xx - .18, mean, .36, label="均值", color=C_BLUE, alpha=.8)
    ax.bar(xx + .18, med, .36, label="中位", color=C_GRAY, alpha=.8)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(xx); ax.set_xticklabels(labels2, fontsize=8)
    ax.set_ylabel("绝对收益"); ax.set_title("高流动性下跌样本：Q5 均值正，但中位仍薄", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)
    for i, (a, b) in enumerate(zip(mean, med)):
        ax.text(i - .18, a + .005, f"{a*100:+.1f}%", ha="center", fontsize=7)
        ax.text(i + .18, b + .005, f"{b*100:+.1f}%", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图11_月度下行.png"))
    plt.close(fig)

    # 图12 组合
    d = port[(port.区间 == "2010-2026") & (port.策略.isin(pick))]
    d = d.set_index("策略").loc[[p for p in pick if p in d.策略.values]].reset_index()
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    colors = [C_GRAY if (s == "全A等权" or abs(r - 0.0609) < 0.0015) else (C_POS if r > 0.0609 else C_NEG)
              for s, r in zip(d.策略, d.年化)]
    ax.bar(d.策略, d.年化 * 100, color=colors, alpha=.85)
    ax.axhline(6.09, color="#333", lw=1, ls="--", label="全 A 等权 6.09%")
    for i, v in enumerate(d.年化 * 100):
        ax.text(i, v + .25, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("年化（%）"); ax.set_title("20 日非重叠组合：剔除 MACD 看多状态并没有改善", loc="left", fontsize=11)
    ax.tick_params(axis="x", labelrotation=22, labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "图12_特征组合.png"))
    plt.close(fig)
    print("saved tables and figures")


if __name__ == "__main__":
    main()
