"""把结果 CSV 渲染成 markdown 表，供 03_结果/退出_报告.md 引用。

只做搬运与格式化，不做任何筛选或择优——所有臂、所有窗口全量输出。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "03_结果")

ARM_ORDER = [
    "B0", "S_fix", "S_struct",
    "T2", "T3", "T45", "T6", "T_struct",
    "BE", "TS10", "TS20", "TS40",
    "SC", "TG2", "SE20", "MX60", "STK", "ALL",
]
FAMILY = {
    "B0": "基准", "S_fix": "初始止损", "S_struct": "初始止损",
    "T2": "追踪", "T3": "追踪", "T45": "追踪", "T6": "追踪", "T_struct": "追踪",
    "BE": "保本", "TS10": "时间", "TS20": "时间", "TS40": "时间",
    "SC": "止盈", "TG2": "止盈", "SE20": "结构离场", "MX60": "结构离场",
    "STK": "状态机", "ALL": "组合",
}


def f(x, n=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        return f"{x*100:+.{n-1}f}%"
    return f"{x:+.{n}f}"


def tab_trades():
    lines = []
    for tb, tbn in (("TB2", "TB-2 日历控制（零信号 alpha）"), ("TB1", "TB-1 突破（真实交易密度）")):
        p = os.path.join(OUT, f"退出_臂对照_交易级_{tb}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for w, wn in (("dev", "开发 2008–2015"), ("val", "验证 2016–2020")):
            x = d[d.window == w].set_index("arm").reindex(ARM_ORDER).dropna(subset=["n"])
            if x.empty:
                continue
            lines.append(f"\n**{tbn} · {wn}**\n")
            lines.append("| 臂 | 机制 | 笔数 | 胜率 | 平均R | 中位R | R 5%分位 | R 95%分位 | 平均持有(日) |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for a, r in x.iterrows():
                lines.append(
                    f"| {a} | {FAMILY.get(a,'')} | {int(r.n)} | {f(r.win,3,True)} | "
                    f"{f(r.avg_R)} | {f(r.med_R)} | {f(r.p05_R)} | {f(r.p95_R)} | {r.avg_hold:.1f} |")
    return "\n".join(lines)


def tab_portfolio():
    p = os.path.join(OUT, "退出_臂对照_组合级.csv")
    if not os.path.exists(p):
        return ""
    d = pd.read_csv(p)
    lines = []
    for w, wn in (("dev", "开发 2008–2015"), ("val", "验证 2016–2020")):
        x = d[d.window == w].set_index("arm").reindex(ARM_ORDER).dropna(subset=["cagr"])
        if x.empty:
            continue
        lines.append(f"\n**组合级 · {wn}**\n")
        lines.append("| 臂 | 机制 | 年化 | 最大回撤 | Calmar | Sharpe | 笔数 | 胜率 | 平均R | 平均持仓数 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a, r in x.iterrows():
            lines.append(
                f"| {a} | {FAMILY.get(a,'')} | {f(r.cagr,2,True)} | {f(r.mdd,2,True)} | "
                f"{f(r.calmar,2)} | {f(r.sharpe,2)} | {int(r.n_trades)} | {f(r.win,3,True)} | "
                f"{f(r.avg_R)} | {r.avg_pos:.1f} |")
    return "\n".join(lines)


def tab_reasons():
    lines = []
    for tb in ("TB1", "TB2"):
        p = os.path.join(OUT, f"退出_退出原因_{tb}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.window == "dev"]
        piv = d.pivot_table(index="arm", columns="reason", values="n", aggfunc="sum").fillna(0)
        piv = piv.reindex([a for a in ARM_ORDER if a in piv.index])
        share = piv.div(piv.sum(axis=1), axis=0)
        lines.append(f"\n**{tb} 开发窗口退出原因占比**\n")
        lines.append("| 臂 | " + " | ".join(share.columns) + " |")
        lines.append("|---|" + "---:|" * len(share.columns))
        for a, r in share.iterrows():
            lines.append(f"| {a} | " + " | ".join(f"{v*100:.0f}%" for v in r.values) + " |")
    return "\n".join(lines)


def tab_states():
    p = os.path.join(OUT, "退出_状态区分力.csv")
    if not os.path.exists(p):
        return ""
    d = pd.read_csv(p)
    order = ["趋势延伸", "预期内推进", "盘整待确认", "谨慎", "假设失效", "强制离场"]
    lines = []
    for tb in ("TB1", "TB2"):
        for w, wn in (("dev", "开发"), ("val", "验证")):
            x = d[(d.testbed == tb) & (d.window == w)].set_index("state").reindex(order).dropna(subset=["n"])
            if x.empty:
                continue
            lines.append(f"\n**{tb} · {wn}窗口持仓日状态 → 未来5日收益（t+1开盘到t+5收盘）**\n")
            lines.append("| 状态 | 持仓日数 | 未来5日均值 | 中位 | 上涨占比 |")
            lines.append("|---|---:|---:|---:|---:|")
            for s, r in x.iterrows():
                lines.append(f"| {s} | {int(r.n)} | {f(r.fwd5_mean*100,2)}% | "
                             f"{f(r.fwd5_median*100,2)}% | {f(r.fwd5_pos*100,1)}% |")
    return "\n".join(lines)


def tab_sizing():
    p = os.path.join(OUT, "退出_仓位敏感.csv")
    if not os.path.exists(p):
        return ""
    d = pd.read_csv(p)
    lines = []
    for study in d.study.unique():
        x = d[d.study == study]
        lines.append(f"\n**{study}**\n")
        lines.append("| 臂 | 窗口 | 口径 | 风险% | 波动% | 槽位 | 资金 | 风险上限 | 年化 | 回撤 | Calmar | 笔数 | 平均R | 平均持仓数 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in x.iterrows():
            cap = "无" if not np.isfinite(r.risk_cap) else f"{r.risk_cap*100:.0f}%"
            lines.append(
                f"| {r.arm} | {r.window} | {r.sizing} | {r.risk_pct*100:.1f} | {r.vol_pct*100:.1f} | "
                f"{int(r.slots)} | {int(r.capital)} | {cap} | {f(r.cagr,2,True)} | {f(r.mdd,2,True)} | "
                f"{f(r.calmar,2)} | {int(r.n_trades)} | {f(r.avg_R)} | {r.avg_pos:.1f} |")
    return "\n".join(lines)


def tab_concentration():
    """收益是否被极少数交易撑起来：去掉最好的 1% 后还剩多少。"""
    lines = []
    for tb in ("TB1", "TB2"):
        rows = []
        for arm in ("B0", "T3", "TG2", "ALL"):
            p = os.path.join(OUT, f"退出_明细_{tb}_{arm}.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            for w in ("dev", "val"):
                r = d[d.window == w].R_mult.replace([np.inf, -np.inf], np.nan).dropna()
                if len(r) < 100:
                    continue
                r = r.sort_values()
                tot = r.sum()
                k1 = max(int(len(r) * 0.01), 1)
                k5 = max(int(len(r) * 0.05), 1)
                rows.append(dict(arm=arm, window=w, n=len(r),
                                 top1_share=r.nlargest(k1).sum() / tot * 100 if tot else np.nan,
                                 top5_share=r.nlargest(k5).sum() / tot * 100 if tot else np.nan,
                                 mean_all=r.mean(),
                                 mean_ex_top1=(tot - r.nlargest(k1).sum()) / (len(r) - k1)))
        if not rows:
            continue
        x = pd.DataFrame(rows)
        lines.append(f"\n**{tb} 收益集中度**\n")
        lines.append("| 臂 | 窗口 | 笔数 | 平均R | 最好1%占总R比重 | 最好5%占总R比重 | 剔除最好1%后的平均R |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in x.iterrows():
            lines.append(f"| {r.arm} | {r.window} | {int(r.n)} | {f(r.mean_all)} | "
                         f"{r.top1_share:.0f}% | {r.top5_share:.0f}% | {f(r.mean_ex_top1)} |")
    return "\n".join(lines)


def tab_yearly():
    lines = []
    for tb in ("TB1", "TB2"):
        p = os.path.join(OUT, f"退出_分年_{tb}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        piv = d.pivot_table(index="arm", columns="year", values="avg_R", aggfunc="mean")
        piv = piv.reindex([a for a in ARM_ORDER if a in piv.index])
        lines.append(f"\n**{tb} 各年交易级平均 R**\n")
        lines.append("| 臂 | " + " | ".join(str(int(c)) for c in piv.columns) + " |")
        lines.append("|---|" + "---:|" * len(piv.columns))
        for a, r in piv.iterrows():
            lines.append(f"| {a} | " + " | ".join(f(v) for v in r.values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    md = ["## A. 交易级配对对照（同一入场集合，只换退出规则）", tab_trades(),
          "\n## B. 组合级（10 槽、100 万、TB-1）", tab_portfolio(),
          "\n## C. 退出原因分布", tab_reasons(),
          "\n## D. 状态标签区分力", tab_states(),
          "\n## E. 仓位与组合风险", tab_sizing(),
          "\n## F. 收益集中度", tab_concentration(),
          "\n## G. 分年稳定性", tab_yearly()]
    txt = "\n".join(md)
    path = os.path.join(OUT, "退出_报告_表格.md")
    open(path, "w").write(txt)
    print(f"写入 {path} ({len(txt)} 字符)")
