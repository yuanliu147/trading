#!/usr/bin/env python3
"""独立校验：从路径原始数据重算关键数字，与 CSV/Excel/HTML 对账。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "03_结果"
PATH = RES / "买入后逐日路径.parquet"
TRADES = BASE.parent / "03_结果" / "指标优化" / "2005-2026" / "C_排除ST_逐笔.csv"


def pct(x):
    return f"{x*100:.1f}%"


def main():
    df = pd.read_parquet(PATH)
    df["period"] = np.where(df.signal_date < pd.Timestamp("2017-01-01"), "2005-2016", "2017-2026")
    df["score"] = df[["below_entry", "no_new_high_since_entry", "break_entry_low_close",
                      "below_ma5", "two_lower"]].sum(axis=1)
    df["comp"] = df.below_entry & df.no_new_high_since_entry & df.below_ma5

    checks = []

    def py(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    def add(name, got, want, tol=1e-9, note=""):
        ok = (pd.isna(got) and pd.isna(want)) or (abs(got - want) <= tol)
        checks.append({"检查": name, "实测": py(got), "期望": py(want), "通过": bool(ok), "备注": note})

    # 1. 规模
    add("路径行数", len(df), 18789, 0)
    add("全部交易数", df.groupby(["signal_date", "code"]).ngroups, 1256, 0)
    add("实际成交数", df[df.is_actual].groupby(["signal_date", "code"]).ngroups, 219, 0)
    add("信号日数", df.signal_date.nunique(), 689, 0)
    add("信号段数", pd.read_csv(RES / "逐笔摘要.csv", encoding="utf-8-sig")["信号段"].nunique(), 925, 0)

    # 2. 基准
    t = df.groupby(["signal_date", "code"]).agg(
        outcome=("final_ret_sim", "last"), tp=("tp_any", "max"),
        hold15=("hold15_ret", "first"), maxdd=("runmin_close", "last"),
        entry=("entry_open", "first"),
    ).reset_index()
    t = t[t.outcome.notna()]
    add("全部基准笔数", len(t), 1251, 0)
    add("全部基准最终亏损率", (t.outcome < 0).mean(), 0.350, 0.0005)
    add("全部基准胜率", (t.outcome > 0).mean(), 0.648, 0.0005)
    add("全部基准止盈率", t.tp.mean(), 0.289, 0.0005)

    # 3. 关键规则（全期、全部信号前二、风险集）
    def risk(d, scope="all", period="全期"):
        x = df[(df.d == d) & (~df.tp_hit_so_far) & df.final_ret_sim.notna()].copy()
        if scope == "actual":
            x = x[x.is_actual]
        if period != "全期":
            x = x[x.period == period]
        return x

    co = pd.read_csv(RES / "组合规则.csv", encoding="utf-8-sig")

    def want(d, rule, col="最终亏损率"):
        r = co[(co["样本"] == "全部信号前二") & (co["期间"] == "全期") &
               (co["天数"] == d) & (co["规则"] == rule)]
        return r[col].iloc[0]

    for d in [1, 2, 3, 5, 10]:
        x = risk(d)
        add(f"D+{d} 收益<-2% N", ((x.ret < -0.02).sum()), want(d, "收益<-2%", "N"), 0)
        add(f"D+{d} 收益<-2% 最终亏损", x[x.ret < -0.02].final_ret_sim.lt(0).mean(),
            want(d, "收益<-2%"), 1e-9)
        add(f"D+{d} 三项复合 最终亏损", x[x.comp].final_ret_sim.lt(0).mean(),
            want(d, "跌破买入价+未创新高+破5日线"), 1e-9)
        add(f"D+{d} 三项复合 深度亏损", x[x.comp].final_ret_sim.le(-0.10).mean(),
            want(d, "跌破买入价+未创新高+破5日线", "深度亏损率"), 1e-9)
        add(f"D+{d} 预警分>=4 最终亏损", x[x.score >= 4].final_ret_sim.lt(0).mean(),
            want(d, "预警分>=4"), 1e-9)

    x3 = risk(3); e3 = x3[x3.ret < -0.02]
    x5 = risk(5); e5 = x5[x5.ret < -0.02]
    add("D+3 收益<-2% 之后止盈率", e3.tp_any.mean(), 0.094, 0.001)
    add("D+5 收益<-2% 之后止盈率", e5.tp_any.mean(), 0.085, 0.001)

    # 4. 早退
    ea = pd.read_csv(RES / "早退反事实.csv", encoding="utf-8-sig")
    x = risk(5)
    e = x[x.ret < -0.02]
    add("D+5 收益<-2% 早退-持有pp", (e.exit_next_open - e.final_ret_sim).mean() * 100,
        ea[(ea["样本"] == "全部信号前二") & (ea["期间"] == "全期") & (ea["天数"] == 5) &
           (ea["规则"] == "收益<-2%")]["早退-持有pp"].iloc[0], 1e-8)
    add("D+5 收益<-2% 持有<=-15%", (e.final_ret_sim <= -0.15).mean(),
        ea[(ea["样本"] == "全部信号前二") & (ea["期间"] == "全期") & (ea["天数"] == 5) &
           (ea["规则"] == "收益<-2%")]["持有<=-15%"].iloc[0], 1e-9)

    # 5. 实际成交对账
    actual = pd.read_csv(TRADES, parse_dates=["signal_date", "entry_date"])
    path_actual = df[df.is_actual].groupby(["signal_date", "code"]).agg(
        entry=("date", "first"), final_sim=("final_ret_sim", "last"),
        logged=("logged_ret", "last"), tp_any=("tp_any", "max"),
        path_exit_reason=("logged_exit_reason", "last"),
    ).reset_index()
    m = path_actual.merge(actual, on=["signal_date", "code"], how="inner")
    add("实际成交对账笔数", len(m), 219, 0)
    add("实际买入日最大差（天）", (m.entry - m.entry_date).dt.days.abs().max(), 0, 0)
    add("实际模拟 vs 逐笔 中位绝对差", (m.final_sim - m.logged).abs().median(), 0.002557, 0.002)
    tp_actual = m[m.exit_reason == "take_profit_intraday"]
    add("逐笔止盈但路径未识别数", (~tp_actual.tp_any).sum(), 0, 4,
        "允许 0-4：跳空/开盘价高于目标价时路径只按盘中高点识别")
    hold_actual = m[m.exit_reason == "max_holding"]
    add("逐笔到期但路径识别为止盈数", hold_actual.tp_any.sum(), 0, 3,
        "允许 0-3：盘中曾触及 +20% 但未成交/跌停顺延")
    add("实际成交 2017-2026 笔数", len(m[(m.signal_date >= "2017-01-01") & (m.exit_reason != "eod")]), 107, 0)

    # 6. 未来函数抽查：随机 3 笔，检查 d=0/1/5 的 ret / runmax
    rng = np.random.default_rng(42)
    keys = list(df.groupby(["signal_date", "code"]).groups.keys())
    for key in rng.choice(len(keys), 3, replace=False):
        sig, code = keys[key]
        w = df[(df.signal_date == sig) & (df.code == code)].sort_values("d")
        entry = w.entry_open.iloc[0]
        add(f"抽查 {sig.date()} {code} d=0 ret", w.iloc[0].ret, w.iloc[0].hfq_close / entry - 1, 1e-12)
        add(f"抽查 {sig.date()} {code} d=5 ret", w.iloc[5].ret, w.iloc[5].hfq_close / entry - 1, 1e-12)
        add(f"抽查 {sig.date()} {code} d=5 runmax", w.iloc[5].runmax_high, w.iloc[:6].hfq_high.max(), 1e-12)

    # 7. 候选数量分层
    cb = pd.read_csv(RES / "候选数量_基准.csv", encoding="utf-8-sig")
    cw = pd.read_csv(RES / "候选数量_预警.csv", encoding="utf-8-sig")
    add("候选数量基准行数", len(cb), 18, 0)
    add("候选数量预警行数", len(cw), 69, 0)
    add("候选基准交易合计", cb[cb["期间"] == "全期"]["交易数"].sum(), 1251, 0)
    def cbv(b, col):
        return cb[(cb["期间"] == "全期") & (cb["候选级别"] == b)][col].iloc[0]
    def cwv(b, col):
        r = cw[(cw["期间"] == "全期") & (cw["天数"] == 5) & (cw["规则"] == "收益<-2%") & (cw["候选级别"] == b)]
        return r[col].iloc[0]
    add("100+ 基准亏损率", cbv("100+", "最终亏损率"), 0.293, 0.001)
    add("100+ D+5触发亏损率", cwv("100+", "最终亏损率"), 0.529, 0.001)
    add("100+ D+5早退差pp", cwv("100+", "早退相对持有") * 100, -10.2, 0.1)
    add("1-2 D+5触发亏损率", cwv("1-2", "最终亏损率"), 0.670, 0.001)
    add("3-9 D+5触发亏损率", cwv("3-9", "最终亏损率"), 0.703, 0.001)
    add("50-99 D+5触发亏损率", cwv("50-99", "最终亏损率"), 0.781, 0.001)
    html = (BASE / "买入后证伪.html").read_text(encoding="utf-8")
    add("HTML 含候选数量章节", int("6.5 信号日候选数量" in html), 1, 0)
    add("HTML 含 52.9%", int("52.9%" in html), 1, 0)

    # 8. D+3 三态
    st_t = df.groupby(["signal_date", "code"]).agg(
        outcome=("final_ret_sim", "last"), logged=("logged_ret", "last"),
        actual=("is_actual", "max"), exit=("logged_exit_reason", "last"),
        period=("period", "first"),
    ).reset_index()
    st_d3 = df[df.d == 3][["signal_date", "code", "ret", "tp_hit_so_far"]].rename(
        columns={"ret": "ret3", "tp_hit_so_far": "tp3"})
    st_all = st_t.merge(st_d3, on=["signal_date", "code"])
    st_x = st_all.query("~tp3").copy()
    st_x["state"] = np.select([st_x.ret3 >= 0.03, st_x.ret3 <= -0.02],
                              ["上涨", "下跌"], default="中震荡")
    st_base = st_x[st_x.outcome.notna()]
    add("三态可评估笔数", len(st_base), 1167, 0)
    add("三态上涨笔数", (st_base.state == "上涨").sum(), 437, 0)
    add("三态中震荡笔数", (st_base.state == "中震荡").sum(), 348, 0)
    add("三态下跌笔数", (st_base.state == "下跌").sum(), 382, 0)
    add("三态+D3前止盈+未满15日=1256",
        len(st_base) + int(st_all.tp3.sum()) + int(st_x.outcome.isna().sum()), 1256, 0)
    for st, loss, mean in [("上涨", 0.162, 0.104), ("中震荡", 0.374, 0.039), ("下跌", 0.620, -0.038)]:
        g = st_base[st_base.state == st]
        add(f"三态 {st} 最终亏损", (g.outcome < 0).mean(), loss, 0.001)
        add(f"三态 {st} 均值", g.outcome.mean(), mean, 0.001)
    sb = pd.read_csv(RES / "路径三态_基准.csv", encoding="utf-8-sig")
    add("三态基准行数", len(sb), 18, 0)
    add("三态基准交易合计",
        sb[(sb["样本"] == "全部信号前二") & (sb["期间"] == "全期")]["笔数"].sum(), 1167, 0)

    st_d5 = df[df.d == 5][["signal_date", "code", "ret", "hfq_high", "hfq_low",
                           "hfq_close", "break_entry_low_close", "score", "comp",
                           "tp_hit_so_far", "exit_next_open"]].rename(columns={
        "ret": "ret5", "hfq_high": "high5", "hfq_low": "low5", "hfq_close": "close5",
        "break_entry_low_close": "break5", "score": "score5", "comp": "comp5",
        "tp_hit_so_far": "tp5", "exit_next_open": "exit5",
    })
    st_mh = df[df.d <= 5].groupby(["signal_date", "code"]).hfq_high.max().rename("maxhigh5").reset_index()
    st5 = st_base.merge(st_d5, on=["signal_date", "code"], how="left").merge(
        st_mh, on=["signal_date", "code"], how="left")
    st5 = st5[(~st5.tp5.fillna(False).astype(bool)) & st5.ret5.notna() & st5.outcome.notna()]
    st5["early5"] = st5.exit5 - st5.outcome
    add("三态D+5风险集笔数", len(st5), 1104, 0)
    for st, rule, n, loss in [
        ("上涨", "D+5 回吐到成本下", 40, 0.475),
        ("上涨", "D+5 距 D+0–5 最高回撤≥10%", 43, 0.442),
        ("中震荡", "D+5 < -2%", 80, 0.588),
        ("中震荡", "D+5 < -5%", 35, 0.771),
        ("下跌", "D+5 预警分≥4", 172, 0.756),
    ]:
        g = st5[st5.state == st]
        if rule == "D+5 回吐到成本下":
            e = g[g.ret5 < 0]
        elif rule == "D+5 距 D+0–5 最高回撤≥10%":
            e = g[g.close5 / g.maxhigh5 - 1 <= -0.10]
        elif rule == "D+5 < -2%":
            e = g[g.ret5 < -0.02]
        elif rule == "D+5 < -5%":
            e = g[g.ret5 < -0.05]
        else:
            e = g[g.score5 >= 4]
        add(f"三态规则 {st} {rule} N", len(e), n, 0)
        add(f"三态规则 {st} {rule} 亏损", (e.outcome < 0).mean(), loss, 0.001)
    up0 = st5[(st5.state == "上涨") & (st5.ret5 < 0)]
    add("上涨回吐早退差pp", up0.early5.mean() * 100, -5.2, 0.2)
    mid5 = st5[(st5.state == "中震荡") & (st5.ret5 < -0.05)]
    add("中震荡<-5%持有<=-15%", (mid5.outcome <= -0.15).mean(), 0.114, 0.001)
    add("中震荡<-5%早退<=-15%", (mid5.exit5 <= -0.15).mean(), 0.057, 0.001)

    # 市场环境 × 三态
    mk = pd.read_parquet(BASE.parent / "01_数据" / "market_daily.parquet",
                         columns=["date", "market_return"])
    mk["date"] = pd.to_datetime(mk.date)
    mk = mk.sort_values("date").reset_index(drop=True)
    mk["idx"] = (1 + mk.market_return.fillna(0)).cumprod()
    mk["ma250"] = mk.idx.rolling(250, min_periods=250).mean()
    mk["ret120"] = mk.idx / mk.idx.shift(120) - 1
    mk["slope250"] = mk.ma250 / mk.ma250.shift(20) - 1
    mk["above250"] = mk.idx / mk.ma250 - 1
    mk["市场环境"] = np.select(
        [(mk.above250 > .02) & (mk.ret120 > .03) & (mk.slope250 > 0),
         (mk.above250 < -.02) & (mk.ret120 < -.03) & (mk.slope250 < 0)],
        ["上涨市", "下跌市"], default="中震荡市")
    st_m = st_base.merge(mk[["date", "市场环境"]], left_on="signal_date", right_on="date", how="left")
    st_m["_loss"] = st_m.outcome < 0
    for rg, st, loss in [("上涨市", "上涨", .122), ("上涨市", "中震荡", .302),
                         ("上涨市", "下跌", .522), ("下跌市", "上涨", .159),
                         ("下跌市", "中震荡", .429), ("下跌市", "下跌", .724)]:
        g = st_m[(st_m["市场环境"] == rg) & (st_m.state == st)]
        add(f"市场环境 {rg} {st} 亏损", (g.outcome < 0).mean(), loss, 0.001)
    add("市场环境全期笔数", st_m["市场环境"].notna().sum(), 1167, 0)

    html = (BASE / "买入后证伪.html").read_text(encoding="utf-8")
    add("HTML 含三态章节", int("6.8 上涨 / 中震荡 / 下跌" in html), 1, 0)
    add("HTML 含 47.5%", int("47.5%" in html), 1, 0)
    add("HTML 含 77.1%", int("77.1%" in html), 1, 0)
    add("HTML 含 33.7%", int("33.7%" in html), 1, 0)

    # 9. 文件存在与 HTML 包含关键数字
    for f in ["买入后证伪.html", "00_一页纸_买入后证伪.md", "01_完整报告_买入后证伪.md",
              "买入后证伪_数据表.xlsx", "02_代码/analyze_path_states.py",
              "03_结果/路径三态_基准.csv", "03_结果/路径三态_转弱.csv",
              "04_图表/10_路径三态.png", "04_图表/11_三态转弱.png",
              "04_图表/12_市场环境稳健.png"]:
        add(f"文件存在 {f}", int((BASE / f).exists()), 1, 0)
    html = (BASE / "买入后证伪.html").read_text(encoding="utf-8")
    for s in ["62.0%", "65.1%", "71.4%", "42.2%", "42.4%", "4.3pp"]:
        add(f"HTML 含 {s}", int(s in html), 1, 0)
    add("HTML 内嵌图片数", len(re.findall(r"data:image/png;base64,", html)), 12, 0)
    add("HTML 大小KB>500", int((BASE / "买入后证伪.html").stat().st_size > 500_000), 1, 0)
    from openpyxl import load_workbook
    wb = load_workbook(BASE / "买入后证伪_数据表.xlsx", read_only=True)
    add("Excel工作表数", len(wb.sheetnames), 18, 0)
    for s in ["12_路径三态基准", "14_路径三态转弱", "16_市场环境", "17_结论摘要"]:
        add(f"Excel含 {s}", int(s in wb.sheetnames), 1, 0)
    wb.close()
    one = (BASE / "00_一页纸_买入后证伪.md").read_text(encoding="utf-8")
    full = (BASE / "01_完整报告_买入后证伪.md").read_text(encoding="utf-8")
    add("一页纸含三态附二", int("附二：D+3 三态" in one), 1, 0)
    add("完整报告含三态章节", int("## 10. 买入后 D+3 三态" in full), 1, 0)
    add("完整报告含18个中文表", int("18 个中文表" in full), 1, 0)

    # 输出
    out = pd.DataFrame(checks)
    out.to_csv(RES / "校验报告.csv", index=False, encoding="utf-8-sig")
    n_fail = (~out["通过"]).sum()
    print(out.to_string(index=False))
    print(f"\n通过 {len(out)-n_fail}/{len(out)}；失败 {n_fail}")
    (RES / "校验报告.json").write_text(
        json.dumps({"checks": checks, "passed": int(len(out) - n_fail), "failed": int(n_fail)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
