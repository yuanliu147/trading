#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补测：只放宽硬止损、仍然死叉退出。

这是上一轮 17_股性自适应验证.py 的缺口：
- 17 里的 ATR/类型自适应是“硬止损 + ATR 追踪”，追踪把赢家提前砍掉；
- 这里不追踪，只在收盘跌破硬止损时卖出，其余仍等死叉。

对照：
  死叉
  死叉 + 硬止损 8% / 10% / 12%
  死叉 + 2.0 / 2.5 / 3.0 × ATR 硬止损（6%-15%）
  死叉 + 类型自适应 ATR 硬止损（低波低量 2.2、低波活跃 2.5、高波低量 2.8、高波活跃 3.0）

买点固定 F2 + 指数 MA20；每笔 1 万、本金无限；T+1 开盘；费用同前。
输出：06_股性自适应验证/硬止损对照_*.csv
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "06_股性自适应验证"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "adaptive17", HERE / "17_股性自适应验证.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["adaptive17"] = mod
spec.loader.exec_module(mod)
eng = mod.eng

START = pd.Timestamp("2010-07-01")
END = pd.Timestamp("2026-10-13")
ALLOC = 10_000.0

EXITS = {
    "死叉": dict(kind="death"),
    "死叉+硬止损8": dict(kind="death_hard", hard=0.08),
    "死叉+硬止损10": dict(kind="death_hard", hard=0.10),
    "死叉+硬止损12": dict(kind="death_hard", hard=0.12),
    "死叉+2.0ATR止损": dict(kind="death_atr", stop_mult=2.0),
    "死叉+2.5ATR止损": dict(kind="death_atr", stop_mult=2.5),
    "死叉+3.0ATR止损": dict(kind="death_atr", stop_mult=3.0),
    "死叉+类型ATR止损": dict(kind="death_type"),
}


def resolve(spec: dict, atr: float, typ: str) -> dict:
    atr = max(float(atr) if np.isfinite(atr) else 0.03, 0.005)
    if spec["kind"] == "death":
        hard = None
    elif spec["kind"] == "death_hard":
        hard = spec["hard"]
    elif spec["kind"] == "death_atr":
        hard = float(np.clip(spec["stop_mult"] * atr, 0.06, 0.15))
    elif spec["kind"] == "death_type":
        sm, _ = mod.TYPE_MULT.get(typ, mod.TYPE_MULT["未分类"])
        hard = float(np.clip(sm * atr, 0.06, 0.15))
    else:
        raise ValueError(spec["kind"])
    return dict(kind="death", hard=hard, trail=None, trigger=None, time=None)


def main() -> None:
    print("加载数据与信号…", flush=True)
    data = eng.load_pit_data()
    eng.START, eng.END = START, END
    bench = eng.load_benchmark()
    dates = pd.to_datetime(bench["date"]).tolist()
    dates = [d for d in dates if START <= d <= END]
    snaps = eng.load_snapshots()
    uni = eng.Universe(snaps)

    sig = pd.read_csv(HERE / "01_数据" / "信号特征_时点500_内部.csv",
                      parse_dates=["signal_date"])
    sig = sig[sig["in_universe"] == 1].copy()
    sig = mod.add_entry_masks(sig)
    sig = sig[sig["m_f2idx"]].copy()
    print(f"  F2+指数 信号 {len(sig):,}", flush=True)

    idx_maps = {c: {d: i for i, d in enumerate(df["date"].tolist())}
                for c, df in data.items()}
    next_date = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
    arr_cache = {}
    stats = Counter()
    rows = []

    for k, r in enumerate(sig.itertuples(index=False), 1):
        code = r.code
        sd = pd.Timestamp(r.signal_date)
        df = data.get(code)
        if df is None or sd not in idx_maps.get(code, {}):
            stats["信号日无行情"] += 1
            continue
        ed = next_date.get(sd)
        if ed is None:
            stats["无下一交易日"] += 1
            continue
        j0 = idx_maps[code].get(ed)
        if j0 is None:
            stats["买入日无行情"] += 1
            continue
        if code not in uni.snapshot_for(ed):
            stats["买入日已非成分"] += 1
            continue
        prev_c = float(df["close"].iat[j0 - 1]) if j0 > 0 else float(df["open"].iat[j0])
        entry = float(df["open"].iat[j0])
        lim = eng.limit_pct(code, ed)
        if entry >= prev_c * (1.0 + lim) * 0.999:
            stats["买入涨停跳过"] += 1
            continue
        if entry <= 0:
            stats["买入价异常"] += 1
            continue

        buy_fee = ALLOC * eng.COST_BUY
        shares = (ALLOC - buy_fee) / entry
        stats["买入成交"] += 1

        if code not in arr_cache:
            arr_cache[code] = (
                df["open"].to_numpy(float),
                df["high"].to_numpy(float),
                df["low"].to_numpy(float),
                df["close"].to_numpy(float),
                df["death_cross"].to_numpy(int),
            )
        o, h, l, c, dc = arr_cache[code]
        dts = df["date"]
        atr = float(r.atr20) if np.isfinite(r.atr20) else 0.03
        typ = getattr(r, "type", "未分类")
        risk_unit = float(np.clip(2.5 * max(atr, 0.005), 0.06, 0.15))

        for exit_name, ex_spec in EXITS.items():
            rr = resolve(ex_spec, atr, typ)
            ex = mod.simulate_exit(o, h, l, c, dc, dts, code, j0, rr)
            proceeds = shares * ex["exit_px"]
            sell_fee = proceeds * eng.COST_SELL
            net = proceeds - sell_fee
            pnl = net - ALLOC
            ret = pnl / ALLOC
            rows.append({
                "卖点": exit_name, "代码": code,
                "名称": getattr(r, "name", ""),
                "信号日": sd, "买入日": ed,
                "卖出日": df["date"].iat[ex["exit_idx"]],
                "买入价": entry, "卖出价": ex["exit_px"],
                "股数": shares, "买入费": buy_fee, "卖出费": sell_fee,
                "投入": ALLOC, "净回款": net, "净盈亏": pnl,
                "净收益率": ret, "R倍数": ret / risk_unit,
                "风险单位": risk_unit,
                "持有交易日": ex["hold_days"],
                "离场原因": ex["reason"],
                "峰值浮盈": ex["peak_profit"],
                "期间最高": ex["mfe"], "期间最低": ex["mae"],
                "信号日收盘": float(r.close),
                "入场跳空": entry / float(r.close) - 1.0,
                "股性类型": typ,
                "股性子类": getattr(r, "subtype", ""),
                "波动分": getattr(r, "s_vol", np.nan),
                "活跃分": getattr(r, "s_heat", np.nan),
            })
        if k % 5000 == 0:
            print(f"  {k:,}/{len(sig):,}", flush=True)

    tr = pd.DataFrame(rows)
    tr["年份"] = pd.to_datetime(tr["信号日"]).dt.year
    tr["阶段"] = np.where(tr["年份"] <= 2017,
                       "设计2010-2017", "样本外2018-2026")
    tr["月份"] = pd.to_datetime(tr["信号日"]).dt.to_period("M").astype(str)

    def summary_rows() -> pd.DataFrame:
        out = []
        for exit_name, g0 in tr.groupby("卖点"):
            for stage, g in g0.groupby("阶段"):
                s = mod.summarize(g)
                s["卖点"] = exit_name
                s["阶段"] = stage
                s["硬止损触发率%"] = (g["离场原因"] == "硬止损").mean() * 100
                out.append(s)
        x = pd.DataFrame(out)
        return x[["卖点", "阶段", "笔数", "胜率%", "平均收益率%",
                  "收益率中位数%", "平均R", "R中位数", "平均盈利%",
                  "平均亏损%", "盈亏比", "PF", "p5%", "p95%",
                  "最差单笔%", "平均持有", "平均峰值浮盈%",
                  "平均最大不利%", "top5%盈利占比%", "硬止损触发率%"]]

    summ = summary_rows()
    summ.to_csv(OUT / "硬止损对照_汇总.csv", index=False, encoding="utf-8-sig")

    type_rows = []
    for (typ, exit_name, stage), g in tr.groupby(["股性类型", "卖点", "阶段"]):
        s = mod.summarize(g)
        s["股性类型"] = typ
        s["卖点"] = exit_name
        s["阶段"] = stage
        s["硬止损触发率%"] = (g["离场原因"] == "硬止损").mean() * 100
        type_rows.append(s)
    type_sum = pd.DataFrame(type_rows)
    type_sum = type_sum[["股性类型", "卖点", "阶段", "笔数", "胜率%",
                         "平均收益率%", "平均R", "PF", "p5%",
                         "最差单笔%", "硬止损触发率%"]]
    type_sum.to_csv(OUT / "硬止损对照_分类型.csv", index=False,
                    encoding="utf-8-sig")

    year_rows = []
    for (exit_name, year), g in tr.groupby(["卖点", "年份"]):
        s = mod.summarize(g)
        s["卖点"] = exit_name
        s["年份"] = int(year)
        year_rows.append(s)
    pd.DataFrame(year_rows).to_csv(OUT / "硬止损对照_分年.csv", index=False,
                                   encoding="utf-8-sig")

    # bootstrap + 配对差异（相对死叉）
    oos = tr[tr["阶段"] == "样本外2018-2026"]
    boot = []
    for exit_name, g in oos.groupby("卖点"):
        m, lo, hi = mod.cluster_bootstrap_mean(g["净收益率"], g["月份"])
        mr, lor, hir = mod.cluster_bootstrap_mean(g["R倍数"], g["月份"])
        boot.append({"卖点": exit_name, "笔数": len(g),
                     "平均收益率%": m * 100, "收益CI下%": lo * 100,
                     "收益CI上%": hi * 100, "平均R": mr,
                     "R CI下": lor, "R CI上": hir})
    pd.DataFrame(boot).to_csv(OUT / "硬止损对照_bootstrap.csv", index=False,
                              encoding="utf-8-sig")

    pair = []
    for stage in ["设计2010-2017", "样本外2018-2026"]:
        g = tr[tr["阶段"] == stage]
        pivot = g.pivot_table(index=["代码", "信号日"], columns="卖点",
                              values=["净收益率", "R倍数"], aggfunc="first")
        for a in ["死叉+硬止损8", "死叉+硬止损10", "死叉+硬止损12",
                  "死叉+2.0ATR止损", "死叉+2.5ATR止损",
                  "死叉+3.0ATR止损", "死叉+类型ATR止损"]:
            d = (pivot[("净收益率", a)] - pivot[("净收益率", "死叉")]).dropna()
            dr = (pivot[("R倍数", a)] - pivot[("R倍数", "死叉")]).dropna()
            months = pd.Series(d.index.get_level_values("信号日")).dt.to_period("M").astype(str).values
            m, lo, hi = mod.cluster_bootstrap_mean(d.values, months)
            mr, lor, hir = mod.cluster_bootstrap_mean(dr.values, months)
            pair.append({"阶段": stage, "卖点A": a, "卖点B": "死叉",
                         "配对笔数": len(d), "平均收益差%": m * 100,
                         "收益差CI下%": lo * 100, "收益差CI上%": hi * 100,
                         "平均R差": mr, "R差CI下": lor, "R差CI上": hir})
    pd.DataFrame(pair).to_csv(OUT / "硬止损对照_配对差异.csv", index=False,
                              encoding="utf-8-sig")

    # 累计盈亏/回撤（同 17 的口径，非组合净值）
    eq_rows = []
    for exit_name in ["死叉", "死叉+硬止损8", "死叉+硬止损10",
                      "死叉+硬止损12", "死叉+2.5ATR止损",
                      "死叉+类型ATR止损"]:
        g = tr[tr["卖点"] == exit_name]
        pnl = mod.build_daily_pnl(g, data, idx_maps)
        pnl["卖点"] = exit_name
        pnl["累计盈亏"] = pnl["当日盈亏"].cumsum()
        pnl["峰值"] = pnl["累计盈亏"].cummax()
        pnl["回撤"] = pnl["累计盈亏"] - pnl["峰值"]
        eq_rows.append(pnl)
    eq = pd.concat(eq_rows, ignore_index=True)
    eq.to_csv(OUT / "硬止损对照_权益.csv", index=False, encoding="utf-8-sig")

    for exit_name in ["死叉+硬止损10", "死叉+2.5ATR止损",
                      "死叉+类型ATR止损"]:
        g = tr[tr["卖点"] == exit_name].sort_values(["信号日", "代码"])
        g.to_csv(OUT / f"交易_F2+指数_{exit_name}.csv", index=False,
                 encoding="utf-8-sig")

    with open(OUT / "硬止损对照_日志.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nF2+指数 信号: {len(sig)}\n交易行数: {len(tr)}\n")
        f.write("\n汇总：\n")
        f.write(summ.round(4).to_string(index=False))
        f.write("\n\n配对差异：\n")
        f.write(pd.DataFrame(pair).round(4).to_string(index=False))
        f.write("\n")

    print("\n汇总：")
    print(summ.round(3).to_string(index=False))
    print("\n配对差异：")
    print(pd.DataFrame(pair).round(3).to_string(index=False))
    print(f"\n完成：{OUT}")


if __name__ == "__main__":
    main()
