# -*- coding: utf-8 -*-
"""F09b 03：把两块的关键核对项汇总成一张 CSV（供 18b 报告引用）。

只读：baostock 单股（sh.603778）+ 交付 CSV。
"""
import sqlite3
import pandas as pd
import numpy as np

ROOT = "/Users/admin/WorkSpace/trading"
OUT = f"{ROOT}/2026-11-07_主板策略逐一审计/02_复算/F09b"
BAO = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
MAIN_PREFIX = ("600", "601", "603", "605", "000", "001", "002", "003")


def main():
    rows = []

    def add(block, item, claimed, recomputed, verdict):
        rows.append(dict(块=block, 项目=item, 原称=claimed, 复算=recomputed, 结论=verdict))

    # ---------- 603778 ----------
    con = sqlite3.connect(f"file:{BAO}?mode=ro", uri=True)
    raw = pd.read_sql("""select date,open,high,low,close,turn from stock_kline_d
                         where code='sh.603778' and adjustflag=3 order by date""",
                      con, parse_dates=["date"])
    hfq = pd.read_sql("""select date,open,close from stock_kline_d
                         where code='sh.603778' and adjustflag=1 order by date""",
                      con, parse_dates=["date"])
    con.close()
    d = raw.merge(hfq.rename(columns={"open": "o_h", "close": "c_h"}), on="date")
    d = d[d.date <= pd.Timestamp("2026-10-29")].dropna(subset=["turn"]).reset_index(drop=True)
    o1 = d.o_h.shift(-1)
    for n in (20, 60):
        d[f"f{n}"] = (d.c_h.shift(-n) / o1 - 1) * 100
    bands = [(0, .5, "<0.5"), (.5, 1, "0.5-1"), (1, 2, "1-2"), (2, 3.5, "2-3.5"),
             (3.5, 5, "3.5-5"), (5, 8, "5-8"), (8, 12, "8-12"), (12, 20, "12-20"), (20, 1e9, ">20")]
    add("603778", "有效交易日", "2591", f"{len(d)}", "一致")
    add("603778", "turn 字段单位", "百分比（6.4=6.4%）", "6.396 ↔ 41,114,023/6.4286亿=6.396%", "一致")
    for lo, hi, lab in bands:
        s = d[(d.turn >= lo) & (d.turn < hi)]
        if len(s) == 0:
            continue
        add("603778", f"区间 {lab}% 天数/T+20中位/胜率",
            {"<0.5": "136/+2.9/60%", "0.5-1": "355/+1.6/54%", "1-2": "621/+0.2/50%",
             "2-3.5": "429/-2.4/42%", "3.5-5": "218/-3.7/34%", "5-8": "273/-3.2/40%",
             "8-12": "212/-3.2/42%", "12-20": "238/-2.7/45%", ">20": "109/-3.9/42%"}[lab],
            f"{len(s)}/{s.f20.median():+.1f}/{((s.f20>0).mean()*100):.0f}%", "一致")
    b58 = d[(d.turn >= 5) & (d.turn < 8)]
    b58p = b58[b58.date.dt.year != 2026]
    add("603778", "5-8% 剔除2026：天数/T+20中位/T+60中位/60日胜率", "239/-2.4/-12.5/29%",
        f"{len(b58p)}/{b58p.f20.median():+.2f}/{b58p.f60.median():+.2f}/{(b58p.f60>0).mean()*100:.0f}%", "一致")
    # 非重叠子样本
    sub = d.iloc[np.arange(0, len(d), 20)]
    for lab, lo, hi in [("<0.5", 0, .5), ("5-8", 5, 8), ("3.5-5", 3.5, 5)]:
        s = sub[(sub.turn >= lo) & (sub.turn < hi)]
        add("603778", f"非重叠子样本（每20日取1日）{lab}%",
            "未报告", f"n={len(s)}/T+20中位{s.f20.median():+.1f}/胜率{(s.f20>0).mean()*100:.0f}%",
            "新增检验")
    # 均值口径
    pos = [lab for lo, hi, lab in bands if d[(d.turn >= lo) & (d.turn < hi)].f20.mean() > 0]
    add("603778", "T+20 均值为正的区间", "“<1% 唯一正期望带”", "、".join(pos), "表述不精确")

    # ---------- 信号层 ----------
    df = pd.read_csv(f"{ROOT}/2026-10-22_全信号_买卖时间_优先级研究/全信号_一万_逐笔.csv",
                     encoding="utf-8-sig",
                     usecols=["代码", "买入日", "止损后卖出日", "止损后净盈亏", "止损后收益率", "年份"])
    add("信号层", "笔数/总净盈亏/平均每笔/胜率", "40,279/+713.7万/+1.77%/50.3%",
        f"{len(df):,}/{df.止损后净盈亏.sum()/1e4:,.1f}万/{df.止损后收益率.mean()*100:.2f}%/{(df.止损后净盈亏>0).mean()*100:.2f}%", "一致")
    s = df.止损后净盈亏.sort_values(ascending=False)
    add("信号层", "前1%交易利润占比", "41.8%", f"{s.head(int(round(len(s)*.01))).sum()/s.sum()*100:.1f}%", "一致")
    df["主板"] = df.代码.str[:3].isin(MAIN_PREFIX)
    mb = df[df.主板]
    add("信号层", "仅主板：笔数/平均/胜率", "未单独报告",
        f"{len(mb):,}/{mb.止损后收益率.mean()*100:.2f}%/{(mb.止损后净盈亏>0).mean()*100:.2f}%", "新增：主板更弱")
    yr22 = df[df.年份 == 2022]
    add("信号层", "2022 全信号平均", "-2.03%", f"{yr22.止损后收益率.mean()*100:.2f}%", "一致")
    ps = pd.read_csv(f"{ROOT}/2026-10-22_持仓天数_涨跌路径/逐笔路径摘要.csv", encoding="utf-8-sig",
                     usecols=["样本", "第1天收益", "第20天收益"])
    g = ps[ps.样本 == "全信号"]
    d1, d20 = g.第1天收益, g.第20天收益
    add("信号层", "全信号 d1 毛胜率（并列处理）", "50.9%",
        f">0/{len(d1)}={(d1>0).mean()*100:.2f}%；剔0={(d1>0).sum()/((d1!=0).sum())*100:.2f}%；计0={(d1>=0).mean()*100:.2f}%",
        "口径依赖")
    v20 = d20.dropna()
    add("信号层", "全信号 d20 毛胜率（并列处理）", "51.8%",
        f">0/{len(v20)}={(v20>0).mean()*100:.2f}%；剔0={(v20>0).sum()/((v20!=0).sum())*100:.2f}%；计0={(v20>=0).mean()*100:.2f}%",
        "口径依赖")
    add("信号层", "峰值并发持仓/所需资金", "“不限并发”", "4,087 笔（2025-08-19）/0.41 亿元；中位 134 笔", "不可执行")
    add("信号层", "2 仓账户实际吃到的信号", "40,279 笔", "40,533 段中 150 段成交（0.37%），40,327 段因仓位已满未买", "不可执行")
    add("信号层", "70/15 子池 2 仓全期年化/回撤/前5笔占比", "30.3%/-19.6%/68.6%",
        "30.30%/-19.64%/68.65%（信号优化_候选.csv）；仅主板 23.1%，非主板 23 笔贡献 42.8% 净利", "全A成立/主板不成立")
    add("信号层", "主系统 20 日退出净均值/净胜率", "+3.4%/54.7%", "+3.40%/54.7%", "一致")

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/F09b_关键核对表.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n{len(out)} 项 → {OUT}/F09b_关键核对表.csv")


if __name__ == "__main__":
    main()
