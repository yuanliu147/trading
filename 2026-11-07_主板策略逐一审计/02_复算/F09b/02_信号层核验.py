# -*- coding: utf-8 -*-
"""F09b 信号层核验：复核 B07-B09 / S01/S02/S08/S09 的交付 CSV。

只读原交付 CSV，不重跑信号引擎；全部为 pandas 汇总。
输出：
- F09b_全信号_复核汇总.csv
- F09b_全信号_分年_主板vs全A.csv
- F09b_并发与资金占用.csv
- F09b_7015子池_复核.csv
- F09b_持仓天数_复核.csv
"""
import numpy as np
import pandas as pd

ROOT = "/Users/admin/WorkSpace/trading"
SIG = f"{ROOT}/2026-10-22_全信号_买卖时间_优先级研究"
DAY = f"{ROOT}/2026-10-22_持仓天数_涨跌路径"
OUT = f"{ROOT}/2026-11-07_主板策略逐一审计/02_复算/F09b"

MAIN_PREFIX = ("600", "601", "603", "605", "000", "001", "002", "003")


def is_main(code: str) -> bool:
    return code[:3] in MAIN_PREFIX


def main():
    lines = []

    def p(s=""):
        print(s)
        lines.append(str(s))

    # ---------- 1. 全信号逐笔 ----------
    df = pd.read_csv(f"{SIG}/全信号_一万_逐笔.csv", encoding="utf-8-sig",
                     usecols=["代码", "首个信号日", "买入日", "止损后卖出日", "止损后退出",
                              "不止损净盈亏", "不止损收益率", "止损后净盈亏", "止损后收益率",
                              "持有天数", "买入价", "股数", "年份", "成交额", "股价",
                              "换手异常值", "系统买入"])
    p("=" * 78)
    p("【1】全信号 40,279 笔 基础复核")
    p("=" * 78)
    n = len(df)
    p(f"笔数 {n:,}｜股票数 {df.代码.nunique():,}｜买入日跨度 {df.买入日.min()} → {df.买入日.max()}")
    p(f"止损后：总净盈亏 {df.止损后净盈亏.sum():,.0f} 元｜平均 {df.止损后净盈亏.mean():.2f} 元"
      f"｜中位 {df.止损后净盈亏.median():.2f} 元｜胜率 {(df.止损后净盈亏>0).mean()*100:.2f}%")
    p(f"止损后：平均收益率 {df.止损后收益率.mean()*100:.2f}%｜不止损平均 {df.不止损收益率.mean()*100:.2f}%"
      f"｜不止损总盈亏 {df.不止损净盈亏.sum():,.0f} 元")
    win = df.loc[df.止损后净盈亏 > 0, "止损后净盈亏"].mean()
    loss = -df.loc[df.止损后净盈亏 <= 0, "止损后净盈亏"].mean()
    p(f"盈亏比 {win/loss:.2f}｜中位收益率 {df.止损后收益率.median()*100:.2f}%")
    s = df.止损后净盈亏.sort_values(ascending=False)
    k = max(1, int(round(n * 0.01)))
    p(f"前1%({k}笔)利润占比 {s.head(k).sum()/s.sum()*100:.1f}%｜前5%占比 {s.head(k*5).sum()/s.sum()*100:.1f}%")
    p(f"名义本金合计 {n*1e4/1e8:.2f} 亿元（每笔1万，不限并发、允许小数股）")

    # 主板 vs 全A
    df["主板"] = df.代码.map(is_main)
    p("")
    p("主板口径（600/601/603/605/000/001/002/003）:")
    mb = df[df.主板]
    p(f"  主板 {len(mb):,} 笔（{len(mb)/n*100:.1f}%）｜股票 {mb.代码.nunique():,} 只"
      f"｜总净 {mb.止损后净盈亏.sum():,.0f} 元｜平均 {mb.止损后收益率.mean()*100:.2f}%"
      f"｜胜率 {(mb.止损后净盈亏>0).mean()*100:.2f}%")
    non = df[~df.主板]
    p(f"  非主板 {len(non):,} 笔｜平均 {non.止损后收益率.mean()*100:.2f}%"
      f"｜总净 {non.止损后净盈亏.sum():,.0f} 元｜胜率 {(non.止损后净盈亏>0).mean()*100:.2f}%")

    # 分年主板 vs 全A
    yr = df.groupby("年份").apply(lambda g: pd.Series({
        "全A笔数": len(g),
        "全A平均": g.止损后收益率.mean() * 100,
        "全A胜率": (g.止损后净盈亏 > 0).mean() * 100,
        "主板笔数": g.主板.sum(),
        "主板平均": g.loc[g.主板, "止损后收益率"].mean() * 100,
        "主板胜率": (g.loc[g.主板, "止损后净盈亏"] > 0).mean() * 100,
    }), include_groups=False)
    yr.round(2).to_csv(f"{OUT}/F09b_全信号_分年_主板vs全A.csv", encoding="utf-8-sig")
    p("")
    p("分年（止损后）:")
    p(yr.round(2).to_string())

    # ---------- 2. 并发与资金占用 ----------
    p("")
    p("=" * 78)
    p("【2】并发持仓与资金占用（“不限并发”到底需要多少钱）")
    p("=" * 78)
    ev = df[["买入日", "止损后卖出日"]].copy()
    ev["买入日"] = pd.to_datetime(ev["买入日"])
    ev["止损后卖出日"] = pd.to_datetime(ev["止损后卖出日"])
    starts = ev.groupby("买入日").size().rename("open")
    ends = ev.groupby("止损后卖出日").size().rename("close")
    cal = pd.date_range(ev.买入日.min(), ev.止损后卖出日.max(), freq="B")
    flow = pd.DataFrame(index=cal)
    flow["open"] = starts.reindex(cal).fillna(0)
    flow["close"] = ends.reindex(cal).fillna(0)
    flow["持仓数"] = (flow.open - flow.close).cumsum()
    flow["占用资金_万元"] = flow.持仓数 * 1.0
    p(f"峰值并发持仓 {int(flow.持仓数.max()):,} 笔（{flow.持仓数.idxmax().date()}）"
      f"｜对应名义资金 {flow.持仓数.max():,.0f} 万元（{flow.持仓数.max()/1e4:.2f} 亿元）")
    p(f"日均并发 {flow.持仓数.mean():.0f} 笔｜中位 {flow.持仓数.median():.0f}｜"
      f"P95 {flow.持仓数.quantile(.95):.0f}｜>2 仓的交易日占比 {(flow.持仓数>2).mean()*100:.1f}%")
    p(f"有买入信号的交易日 {len(starts):,} 天｜单日最多 {int(starts.max())} 笔｜单日买 ≥3 笔的天数 {(starts>=3).sum():,}")
    p(f"买入当日即持有 ≥2 笔的天数占比 {(flow.持仓数>=2).mean()*100:.1f}%")
    flow.reset_index().rename(columns={"index": "date"}).to_csv(
        f"{OUT}/F09b_并发与资金占用.csv", index=False, encoding="utf-8-sig")

    # 同一股票内的信号间隔
    df["首个信号日"] = pd.to_datetime(df["首个信号日"])
    gap = df.sort_values(["代码", "首个信号日"]).groupby("代码")["首个信号日"].diff().dt.days.dropna()
    p(f"同一股票相邻信号段间隔中位 {gap.median():.0f} 天｜≤20 天的占比 {(gap<=20).mean()*100:.1f}%"
      f"（=同一笔20日持仓期内再次发信号）")
    seg_per_stock = df.groupby("代码").size()
    p(f"每股信号段数：中位 {seg_per_stock.median():.0f}｜P90 {seg_per_stock.quantile(.9):.0f}｜最大 {seg_per_stock.max()}")

    # ---------- 3. 每股首笔 ----------
    p("")
    p("=" * 78)
    p("【3】每股首笔")
    p("=" * 78)
    fb = pd.read_csv(f"{SIG}/全信号_一万_每股首笔.csv", encoding="utf-8-sig")
    p(f"{len(fb):,} 只｜总净 {fb.止损后净盈亏.sum():,.0f} 元｜平均 {fb.止损后净盈亏.mean():.2f} 元"
      f"｜平均收益率 {fb.止损后收益率.mean()*100:.2f}%｜胜率 {(fb.止损后净盈亏>0).mean()*100:.2f}%")
    fb["主板"] = fb.代码.map(is_main)
    p(f"主板 {fb.主板.sum():,} 只｜平均 {fb.loc[fb.主板,'止损后收益率'].mean()*100:.2f}%"
      f"｜胜率 {(fb.loc[fb.主板,'止损后净盈亏']>0).mean()*100:.2f}%")

    # ---------- 4. 70/15 子池 ----------
    p("")
    p("=" * 78)
    p("【4】70/15 子池（成交额≤7000万 & 股价≤15）逐笔复核")
    p("=" * 78)
    sub = df[(df.成交额 <= 7e7) & (df.股价 <= 15)]
    p(f"{len(sub):,} 笔｜股票 {sub.代码.nunique():,}｜平均 {sub.止损后收益率.mean()*100:.2f}%"
      f"｜胜率 {(sub.止损后净盈亏>0).mean()*100:.2f}%｜总净 {sub.止损后净盈亏.sum():,.0f} 元")
    s2 = sub.止损后净盈亏.sort_values(ascending=False)
    p(f"前5笔占比 {s2.head(5).sum()/s2.sum()*100:.1f}%｜去掉前5笔平均 {s2.iloc[5:].mean():.2f} 元"
      f"｜去掉前5笔胜率 {(s2.iloc[5:]>0).mean()*100:.2f}%")
    p(f"子池内非主板笔数 {(~sub.主板).sum()}｜非主板净盈亏 {sub.loc[~sub.主板,'止损后净盈亏'].sum():,.0f} 元"
      f"｜占子池总净 {sub.loc[~sub.主板,'止损后净盈亏'].sum()/s2.sum()*100:.1f}%")
    p("子池前 8 笔（按净盈亏）:")
    p(sub.nlargest(8, "止损后净盈亏")[["代码", "买入日", "止损后收益率", "止损后净盈亏"]]
      .assign(止损后收益率=lambda x: (x.止损后收益率 * 100).round(1)).to_string(index=False))
    sub_mb = sub[sub.主板]
    p(f"仅主板子池：{len(sub_mb):,} 笔｜平均 {sub_mb.止损后收益率.mean()*100:.2f}%"
      f"｜胜率 {(sub_mb.止损后净盈亏>0).mean()*100:.2f}%｜总净 {sub_mb.止损后净盈亏.sum():,.0f} 元")
    p(f"仅主板子池前5笔占比 {sub_mb.止损后净盈亏.nlargest(5).sum()/sub_mb.止损后净盈亏.sum()*100:.1f}%")

    # ---------- 5. 持仓天数 ----------
    p("")
    p("=" * 78)
    p("【5】持仓天数与路径（复核报告数字 vs 交付 CSV）")
    p("=" * 78)
    path = pd.read_csv(f"{DAY}/路径统计.csv", encoding="utf-8-sig")
    fix = pd.read_csv(f"{DAY}/固定天数退出.csv", encoding="utf-8-sig")
    wl = pd.read_csv(f"{DAY}/赢家输家.csv", encoding="utf-8-sig")
    for smp in ["全信号", "主系统150", "70/15子池"]:
        r1 = path[(path.样本 == smp) & (path.持有天数 == 1)].iloc[0]
        r20 = path[(path.样本 == smp) & (path.持有天数 == 20)].iloc[0]
        f1 = fix[(fix.样本 == smp) & (fix.持有天数 == 1)].iloc[0]
        f3 = fix[(fix.样本 == smp) & (fix.持有天数 == 3)].iloc[0]
        f20 = fix[(fix.样本 == smp) & (fix.持有天数 == 20)].iloc[0]
        p(f"{smp}: 毛胜率 d1 {r1.上涨概率*100:.1f}% → d20 {r20.上涨概率*100:.1f}%"
          f"｜净 d1 {f1.净平均*100:+.1f}% / d3 {f3.净平均*100:+.1f}% / d20 {f20.净平均*100:+.1f}%"
          f"｜净胜率 d20 {f20.净胜率*100:.1f}%｜毛均值 d20 {r20.平均收益*100:.2f}%")
    idx = wl[(wl.样本 == "全信号") & (wl.组别 == "最终上涨")].iloc[0]
    idn = wl[(wl.样本 == "全信号") & (wl.组别 == "最终下跌")].iloc[0]
    p(f"全信号 赢家 n={int(idx.笔数):,}（{idx.占比*100:.1f}%）曾浮亏 {idx.曾浮亏比例*100:.1f}%"
      f"｜输家 n={int(idn.笔数):,} 曾浮盈 {idn.曾浮盈比例*100:.1f}%")
    p(f"全信号 d20：赢家均 {idx.期末平均*100:.1f}% / 输家均 {idn.期末平均*100:.1f}%")

    # 独立复算：逐笔路径摘要
    ps = pd.read_csv(f"{DAY}/逐笔路径摘要.csv", encoding="utf-8-sig",
                     usecols=["样本", "代码", "买入日", "第1天收益", "第3天收益", "第20天收益",
                              "收盘最大浮亏", "收盘最大浮盈", "第20天净收益"])
    p("")
    p("[独立复算] 逐笔路径摘要（不用报告汇总，直接对逐笔）:")
    for smp, g in ps.groupby("样本"):
        g20 = g.dropna(subset=["第20天收益"])
        p(f"  {smp}: n={len(g):,}｜d1 毛胜率 {(g.第1天收益>0).mean()*100:.1f}%（有效 {g.第1天收益.notna().sum():,}）"
          f"｜d20 毛胜率 {(g20.第20天收益>0).mean()*100:.1f}%｜d20 毛均 {g20.第20天收益.mean()*100:.2f}%"
          f"｜d20 净均 {g20.第20天净收益.mean()*100:.2f}%｜d20 净胜率 {(g20.第20天净收益>0).mean()*100:.1f}%")
        win = g20[g20.第20天收益 > 0]
        lose = g20[g20.第20天收益 < 0]
        p(f"      曾浮亏(赢家) {(win.收盘最大浮亏<0).mean()*100:.1f}%｜曾浮盈(输家) {(lose.收盘最大浮盈>0).mean()*100:.1f}%")
    ps["主板"] = ps.代码.map(is_main)
    g20 = ps[(ps.样本 == "全信号") & ps.第20天收益.notna()]
    p(f"  全信号仅主板: n={g20.主板.sum():,}｜d20 毛均 {g20.loc[g20.主板,'第20天收益'].mean()*100:.2f}%"
      f"｜净均 {g20.loc[g20.主板,'第20天净收益'].mean()*100:.2f}%"
      f"｜净胜率 {(g20.loc[g20.主板,'第20天净收益']>0).mean()*100:.1f}%")

    # ---------- 6. 报表一致性 ----------
    p("")
    p("=" * 78)
    p("【6】报告间一致性抽查")
    p("=" * 78)
    p("全信号 20日：不止损净均 +2.0%（全信号表 +788万/40,279=195.7元/1万=1.96%）"
      "；有报告写 +2.3%/笔（毛均）。同一“20日均值”存在毛/净两套数字，需按口径读。")
    p("2仓系统 2022 年 +59.9%（45仓位/年度_收益与大盘.csv 0.5986）；全信号 2022 平均 -2.03%。")
    p("2仓系统“前五笔贡献”：系统统计 83%（账户口径）vs 信号优化_候选 71.6%（净盈亏口径），同一系统两个数。")

    with open(f"{OUT}/F09b_运行日志.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
