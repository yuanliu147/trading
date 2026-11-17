#!/usr/bin/env python3
"""买入次日跌 4-5 个点：历史样本与最终收益。

数据源：12_买入后证伪_哪天没达标/03_结果/买入后逐日路径.parquet
样本：C 池全部信号前二 1,256 笔（含退市、信号日排除风险警示）；实际成交 219 笔。
口径：
  - D+0=买入日（T+3 开盘买）；D+1=买入后第 1 个交易日。
  - 主事件：D+1 收盘 / D+0 收盘 - 1 ∈ [-5%, -4%)，即“次日跌 4-5 个点”。
  - 最终收益：+20% 盘中止盈（D+1 起算）或 15 日到期收盘；未扣交易成本。
  - 实际成交的净收益用逐笔记录覆盖，仅作小样本对照。
"""
from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "03_结果"
OUT.mkdir(parents=True, exist_ok=True)
PATH = ROOT / "12_买入后证伪_哪天没达标" / "03_结果" / "买入后逐日路径.parquet"
STOCKS = ROOT / "05_证据" / "主板股票清单.csv"

PERIOD_OLD = "2005-2016"
PERIOD_NEW = "2017-2026"
COST = 0.0035  # 双边约 0.35%，仅用于净收益注释

REASON_CN = {
    "max_holding": "15日到期",
    "take_profit_intraday": "盘中止盈",
    "take_profit_open": "开盘止盈",
    "pending_max_holding_limit_down": "到期跌停顺延",
    "eod": "期末未满",
}


def half_up(x: float, digits: int = 2) -> float:
    q = Decimal(1).scaleb(-digits)
    return float(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return half_up(float(x) * 100, 2)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def load_trades() -> pd.DataFrame:
    df = pd.read_parquet(PATH)
    d0 = df[df.d == 0].set_index(["signal_date", "code"])[
        ["date", "entry_open", "entry_close", "final_ret_sim", "final_loss_sim",
         "hold15_ret", "is_actual", "rank", "logged_ret", "logged_exit_reason",
         "mkt_cum", "drawdown250_signal"]
    ].add_prefix("d0_")
    d1 = df[df.d == 1].set_index(["signal_date", "code"])[
        ["date", "ret", "ret_prev", "hfq_low", "hfq_high", "mkt_cum"]
    ].add_prefix("d1_")
    d2 = df[df.d == 2].set_index(["signal_date", "code"])[
        ["date", "ret", "ret_prev", "mkt_cum"]
    ].add_prefix("d2_")  # d2_date 也保留
    d3 = df[df.d == 3].set_index(["signal_date", "code"])[["ret", "ret_prev"]].add_prefix("d3_")
    d5 = df[df.d == 5].set_index(["signal_date", "code"])[["ret", "ret_prev"]].add_prefix("d5_")
    path = d0.join([d1, d2, d3, d5])

    # 全路径最大浮亏（收盘口径）与止盈日
    agg = df.groupby(["signal_date", "code"]).agg(
        runmin_close=("runmin_close", "last"),
        tp_any=("tp_any", "last"),
        tp_day=("tp_day", "last"),
        n_days=("d", "size"),
    )
    path = path.join(agg)
    path["mdd"] = path["runmin_close"] / path["d0_entry_open"] - 1
    path["d1_low_vs_entry"] = path["d1_hfq_low"] / path["d0_entry_open"] - 1
    path["d1_high_vs_entry"] = path["d1_hfq_high"] / path["d0_entry_open"] - 1
    path["mkt_d1"] = (1 + path["d1_mkt_cum"]) / (1 + path["d0_mkt_cum"]) - 1
    path["year"] = path["d0_date"].dt.year
    path["period"] = np.where(path["year"] <= 2016, PERIOD_OLD, PERIOD_NEW)

    stocks = pd.read_csv(STOCKS, usecols=["code", "code_name"]).rename(columns={"code_name": "name"})
    path = path.reset_index().merge(stocks, on="code", how="left")
    return path


def final_stats(s: pd.DataFrame) -> dict:
    r = s["d0_final_ret_sim"].dropna()
    n = len(r)
    if n == 0:
        return dict(n_event=len(s), n_final=0)
    mean = r.mean()
    sd = r.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = t_dist.ppf(0.975, n - 1) if n > 1 else 1.96
    loss_n = int((r < 0).sum())
    lo, hi = wilson(loss_n, n)
    actual = s.loc[s["d0_is_actual"], "d0_logged_ret"].dropna()
    tp_n = int((r >= 0.1999).sum())
    non_tp = r[r < 0.1999]
    return {
        "n_event": len(s),
        "n_final": n,
        "n_actual": int(s["d0_is_actual"].sum()),
        "tp_n": tp_n,
        "non_tp_n": len(non_tp),
        "non_tp_mean": pct(non_tp.mean()) if len(non_tp) else None,
        "mean": pct(mean),
        "median": pct(r.median()),
        "std": pct(sd),
        "win": pct((r > 0).mean()),
        "loss": pct((r < 0).mean()),
        "tp20": pct((r >= 0.1999).mean()),
        "le10": pct((r <= -0.10).mean()),
        "le15": pct((r <= -0.15).mean()),
        "ge10": pct((r >= 0.10).mean()),
        "p10": pct(r.quantile(0.10)),
        "p25": pct(r.quantile(0.25)),
        "p75": pct(r.quantile(0.75)),
        "p90": pct(r.quantile(0.90)),
        "min": pct(r.min()),
        "max": pct(r.max()),
        "mean_ci_lo": pct(mean - tcrit * se),
        "mean_ci_hi": pct(mean + tcrit * se),
        "loss_ci_lo": pct(lo),
        "loss_ci_hi": pct(hi),
        "net_mean": pct((r - COST).mean()),
        "net_median": pct((r - COST).median()),
        "actual_mean": pct(actual.mean()) if len(actual) else None,
        "actual_median": pct(actual.median()) if len(actual) else None,
        "actual_loss": pct((actual < 0).mean()) if len(actual) else None,
        "mdd_mean": pct(s["mdd"].mean()),
        "mdd_median": pct(s["mdd"].median()),
        "mdd_p10": pct(s["mdd"].quantile(0.10)),
        "d2_mean": pct(s["d2_ret"].mean()),
        "d2_up": pct((s["d2_ret"] > 0).mean()),
        "d2_c2c_mean": pct(s["d2_ret_prev"].mean()),
        "d2_c2c_up": pct((s["d2_ret_prev"] > 0).mean()),
        "d3_mean": pct(s["d3_ret"].mean()),
        "d3_up": pct((s["d3_ret"] > 0).mean()),
        "d5_mean": pct(s["d5_ret"].mean()),
        "d5_up": pct((s["d5_ret"] > 0).mean()),
    }


def row(label: str, definition: str, s: pd.DataFrame, date_col: str = "d1_date") -> dict:
    d = final_stats(s)
    dates = s[date_col].dropna().dt.date.nunique() if date_col in s else None
    return {
        "口径": label,
        "事件定义": definition,
        "笔数": d.get("n_event"),
        "独立日期数": dates,
        "实际成交数": d.get("n_actual"),
        "最终样本": d.get("n_final"),
        "均值(%)": d.get("mean"),
        "中位数(%)": d.get("median"),
        "胜率(%)": d.get("win"),
        "亏损率(%)": d.get("loss"),
        "止盈率(%)": d.get("tp20"),
        "止盈笔数": d.get("tp_n"),
        "非止盈笔数": d.get("non_tp_n"),
        "非止盈均值(%)": d.get("non_tp_mean"),
        "亏损10(%)": d.get("le10"),
        "亏损15(%)": d.get("le15"),
        "盈利10(%)": d.get("ge10"),
        "均值95CI下限(%)": d.get("mean_ci_lo"),
        "均值95CI上限(%)": d.get("mean_ci_hi"),
        "亏损率95CI下限(%)": d.get("loss_ci_lo"),
        "亏损率95CI上限(%)": d.get("loss_ci_hi"),
        "净均值(%)": d.get("net_mean"),
        "净中位数(%)": d.get("net_median"),
        "实际成交均值(%)": d.get("actual_mean"),
        "最大浮亏均值(%)": d.get("mdd_mean"),
        "最大浮亏中位数(%)": d.get("mdd_median"),
    }


def main() -> None:
    t = load_trades()
    print("交易数", len(t), "最终样本", t["d0_final_ret_sim"].notna().sum())

    events = {
        "个股次日收盘跌4-5%（主口径）": (
            "D+1 收盘 / D+0 收盘 - 1 ∈ [-5%, -4%)",
            (t["d1_ret_prev"] >= -0.05) & (t["d1_ret_prev"] < -0.04),
            "d1_date",
        ),
        "个股次日收盘较买入价跌4-5%": (
            "D+1 收盘 / 买入价 - 1 ∈ [-5%, -4%)",
            (t["d1_ret"] >= -0.05) & (t["d1_ret"] < -0.04),
            "d1_date",
        ),
        "个股次日跌4-5%（宽口径）": (
            "D+1 收盘 / D+0 收盘 - 1 ∈ [-5.5%, -4%)",
            (t["d1_ret_prev"] >= -0.055) & (t["d1_ret_prev"] < -0.04),
            "d1_date",
        ),
        "个股D+2跌4-5%": (
            "D+2 收盘 / D+1 收盘 - 1 ∈ [-5%, -4%)",
            (t["d2_ret_prev"] >= -0.05) & (t["d2_ret_prev"] < -0.04),
            "d2_date",
        ),
        "大盘次日跌4-5%": (
            "主板等权 D+1 涨跌幅 ∈ [-5%, -4%)",
            (t["mkt_d1"] >= -0.05) & (t["mkt_d1"] < -0.04),
            "d1_date",
        ),
    }

    summary_rows = [row("基准（全部信号前二）", "全部已买入交易", t, "d0_date")]
    for label, (definition, mask, date_col) in events.items():
        summary_rows.append(row(label, definition, t[mask], date_col))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "汇总_口径对照.csv", index=False, encoding="utf-8-sig")

    # 大盘口径分状态
    mkt_mask = events["大盘次日跌4-5%"][1]
    mkt = t[mkt_mask].copy()
    mkt_rows = [
        row("大盘事件-全部", "主板等权 D+1 跌4-5%", mkt, "d1_date"),
        row("大盘事件-2008", "2008 年", mkt[mkt["year"] == 2008], "d1_date"),
        row("大盘事件-非2008", "2007/2012/2013/2015/2016", mkt[mkt["year"] != 2008], "d1_date"),
        row("大盘事件-2015至2016", "2015-2016 年", mkt[mkt["year"].isin([2015, 2016])], "d1_date"),
        row("大盘事件-其他", "2007/2012/2013", mkt[mkt["year"].isin([2007, 2012, 2013])], "d1_date"),
    ]
    pd.DataFrame(mkt_rows).to_csv(OUT / "大盘分状态.csv", index=False, encoding="utf-8-sig")

    # 主口径
    main_mask = events["个股次日收盘跌4-5%（主口径）"][1]
    ev = t[main_mask].copy()

    # 分榜
    rank_rows = [
        row("榜一-事件", "主口径", ev[ev["d0_rank"] == 1]),
        row("榜一-基准", "全部已买入交易", t[t["d0_rank"] == 1]),
        row("榜二-事件", "主口径", ev[ev["d0_rank"] == 2]),
        row("榜二-基准", "全部已买入交易", t[t["d0_rank"] == 2]),
    ]
    pd.DataFrame(rank_rows).to_csv(OUT / "分榜_主口径.csv", index=False, encoding="utf-8-sig")

    # 分阶段
    period_rows = []
    for p in [PERIOD_OLD, PERIOD_NEW]:
        period_rows.append(row(f"事件-{p}", "主口径", ev[ev["period"] == p]))
        period_rows.append(row(f"基准-{p}", "全部已买入交易", t[t["period"] == p]))
    pd.DataFrame(period_rows).to_csv(OUT / "分阶段_主口径.csv", index=False, encoding="utf-8-sig")

    # 相邻区间（D+1 收盘涨跌幅）
    bins = [-1, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 1]
    labels = ["<-7%", "-7~-6%", "-6~-5%", "-5~-4%", "-4~-3%", "-3~-2%",
              "-2~-1%", "-1~0%", "0~+1%", "+1~+2%", "+2~+3%", "+3~+4%",
              "+4~+5%", "+5~+7%", ">+7%"]
    t["bucket"] = pd.cut(t["d1_ret_prev"], bins=bins, labels=labels, right=False)
    grid = []
    for lab in labels:
        s = t[t["bucket"] == lab]
        r = s["d0_final_ret_sim"].dropna()
        grid.append({
            "次日涨跌幅区间": lab,
            "笔数": len(s),
            "均值(%)": pct(r.mean()),
            "中位数(%)": pct(r.median()),
            "亏损率(%)": pct((r < 0).mean()),
            "止盈率(%)": pct((r >= 0.1999).mean()),
            "亏损15(%)": pct((r <= -0.15).mean()),
        })
    pd.DataFrame(grid).to_csv(OUT / "相邻区间.csv", index=False, encoding="utf-8-sig")

    # 收益分布（事件 vs 基准）
    dist_bins = [-1, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.2001]
    dist_labels = ["≤-15%", "-15~-10%", "-10~-5%", "-5~0%",
                   "0~+5%", "+5~+10%", "+10~+20%", "+20%止盈"]
    dist_rows = []
    for lab in dist_labels:
        lo = dist_bins[dist_labels.index(lab)]
        hi = dist_bins[dist_labels.index(lab) + 1]
        e = ev["d0_final_ret_sim"].dropna()
        b = t["d0_final_ret_sim"].dropna()
        dist_rows.append({
            "最终收益区间": lab,
            "事件笔数": int(((e >= lo) & (e < hi)).sum()),
            "事件占比(%)": pct(((e >= lo) & (e < hi)).mean()),
            "基准占比(%)": pct(((b >= lo) & (b < hi)).mean()),
        })
    pd.DataFrame(dist_rows).to_csv(OUT / "收益分布_主口径.csv", index=False, encoding="utf-8-sig")

    # 事件明细
    detail = ev.copy()
    detail["退出原因"] = detail["d0_logged_exit_reason"].map(REASON_CN).fillna("")
    detail["是否实际成交"] = np.where(detail["d0_is_actual"], "是", "否")
    detail["是否止盈"] = np.where(detail["tp_any"], "是", "否")
    detail = detail.sort_values(["d0_date", "d0_rank"])
    out = pd.DataFrame({
        "信号日": detail["signal_date"].dt.strftime("%Y-%m-%d"),
        "买入日": detail["d0_date"].dt.strftime("%Y-%m-%d"),
        "代码": detail["code"],
        "当前简称": detail["name"].fillna(""),
        "排名": detail["d0_rank"],
        "次日涨跌幅(%)": detail["d1_ret_prev"].map(pct),
        "次日收盘相对买入价(%)": detail["d1_ret"].map(pct),
        "次日最低相对买入价(%)": detail["d1_low_vs_entry"].map(pct),
        "次日大盘涨跌幅(%)": detail["mkt_d1"].map(pct),
        "D+2涨跌幅(%)": detail["d2_ret_prev"].map(pct),
        "D+2累计(%)": detail["d2_ret"].map(pct),
        "D+3累计(%)": detail["d3_ret"].map(pct),
        "D+5累计(%)": detail["d5_ret"].map(pct),
        "最终收益(%)": detail["d0_final_ret_sim"].map(pct),
        "是否止盈": detail["是否止盈"],
        "止盈日": detail["tp_day"].map(lambda x: f"D+{int(x)}" if pd.notna(x) else ""),
        "最大浮亏(%)": detail["mdd"].map(pct),
        "实际成交": detail["是否实际成交"],
        "实际净收益(%)": detail["d0_logged_ret"].map(pct),
        "退出原因": detail["退出原因"],
        "信号日250日回撤(%)": detail["d0_drawdown250_signal"].map(pct),
        "期间": detail["period"],
    })
    out.to_csv(OUT / "事件交易明细_主口径.csv", index=False, encoding="utf-8-sig")

    # 后续路径（主口径）
    path_rows = []
    for label, col, cumulative in [
        ("D+2 当日", "d2_ret_prev", False),
        ("D+2 累计（相对买入价）", "d2_ret", True),
        ("D+3 当日", "d3_ret_prev", False),
        ("D+3 累计（相对买入价）", "d3_ret", True),
        ("D+5 累计（相对买入价）", "d5_ret", True),
    ]:
        x = ev[col].dropna()
        path_rows.append({
            "观察点": label,
            "样本": len(x),
            "均值(%)": pct(x.mean()),
            "中位数(%)": pct(x.median()),
            "上涨概率(%)": pct((x > 0).mean()),
        })
    pd.DataFrame(path_rows).to_csv(OUT / "后续路径_主口径.csv", index=False, encoding="utf-8-sig")

    # 口径 JSON
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(PATH.relative_to(ROOT)),
        "trades_all": int(len(t)),
        "trades_actual": int(t["d0_is_actual"].sum()),
        "final_sample": int(t["d0_final_ret_sim"].notna().sum()),
        "date_min": t["d0_date"].min().strftime("%Y-%m-%d"),
        "date_max": t["d1_date"].max().strftime("%Y-%m-%d"),
        "event_main": "D+1 close / D+0 close - 1 in [-5%, -4%)",
        "outcome": "+20% intraday take-profit from D+1, else 15-day close; gross of costs",
        "cost_note": "双边约 0.35%；净均值/净中位数已扣",
        "partial": "2026-10-16 信号 000426 未满 15 日且未止盈，最终样本 34/35",
    }
    (OUT / "口径.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary[["口径", "笔数", "最终样本", "均值(%)", "中位数(%)", "亏损率(%)", "止盈率(%)"]].to_string(index=False))
    print("\n主口径事件", len(ev), "笔")
    print(pd.DataFrame(path_rows).to_string(index=False))


if __name__ == "__main__":
    main()
