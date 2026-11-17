#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1 槽（100% 单仓）精选与提前离场实验。

- 时点中证500；T 收盘金叉，T+1 开盘买；涨停不买、跌停顺延；成本同主引擎；
- 只使用信号日收盘可知的数值过滤与打分；
- 打分模型为 11_滚动预测.py 的逐年滚动预测（不用股票名称）；
- 提前离场全部为收盘确认、次日开盘执行。
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = HERE / "01_数据"
OUT = HERE / "04_单槽精选"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("strict_engine", BASE / "03_复现" / "严格重跑.py")
eng = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = eng
spec.loader.exec_module(eng)

START = pd.Timestamp("2013-01-01")  # 滚动模型从 2013 起有预测
END = pd.Timestamp("2026-10-13")
SUB18 = pd.Timestamp("2018-01-01")
SUB23 = pd.Timestamp("2023-01-01")


def load_all():
    data = eng.load_pit_data()
    for code, df in data.items():
        c = df["close"].astype(float)
        for n in (5, 10, 20, 60):
            df[f"ma{n}"] = c.rolling(n).mean()
    eng.START, eng.END = START, END
    bench = eng.load_benchmark()
    dates = [d for d in pd.to_datetime(bench["date"]) if START <= d <= END]
    uni = eng.Universe(eng.load_snapshots())
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", parse_dates=["signal_date"])
    sig = sig[sig["in_universe"] == 1].copy()
    pred = pd.read_csv(OUT / "信号_滚动预测.csv", parse_dates=["signal_date"])
    sig = sig.merge(pred, on=["code", "signal_date"], how="left")
    sig["hist_raw"] = sig["hist_norm"] * sig["close"]
    return data, bench, dates, uni, sig


def masks(s: pd.DataFrame) -> dict[str, pd.Series]:
    A = (s.ret20 <= 0.15) & (s.dist_ma20 <= 0.08) & (s.rsi14 <= 70)
    G = s.n_golden_60 >= 3
    V = s.s_vol <= 0.72
    Q = s.amt5_60 <= 0.26
    T = s.type != "高波活跃型"
    K = s.dist_ma60 >= -0.08
    out = {
        "F1": A & G & V & Q & T,
        "F2": A & G & V & Q & K,
        "F2_去追高约束": G & V & Q & K,
        "F2_去多次金叉": A & V & Q & K,
        "F2_去低波": A & G & Q & K,
        "F2_去缩量": A & G & V & K,
        "F2_去MA60": A & G & V & Q,
        "F2_去高波活跃": A & G & V & Q & K & T,
        "F2_MA20": A & G & V & Q & K & (s.idx_above_ma20 == 1),
        "F2_MA60": A & G & V & Q & K & (s.idx_above_ma60 == 1),
        "F1_MA20": A & G & V & Q & T & (s.idx_above_ma20 == 1),
    }
    out["全部"] = pd.Series(True, index=s.index)
    return out


def make_exit(kind: str, atr_map=None):
    """返回 exit_filter(pos, df, idx) -> reason or None。只用当日收盘及以前。"""
    def fail(pos, df, idx, days=10, mfe_max=0.02):
        if pos.hold_days >= days and pos.highest / pos.entry_price - 1.0 < mfe_max:
            if float(df["close"].iat[idx]) < pos.entry_price:
                return "走势不及预期"
        return None

    def ma_break(pos, df, idx, days=3, mfe_max=0.05):
        if pos.hold_days >= days and pos.highest / pos.entry_price - 1.0 < mfe_max:
            ma = float(df["ma20"].iat[idx])
            if np.isfinite(ma) and float(df["close"].iat[idx]) < ma:
                return "跌破MA20"
        return None

    def breakeven(pos, df, idx, trigger=0.05):
        if pos.highest / pos.entry_price - 1.0 >= trigger:
            if float(df["close"].iat[idx]) <= pos.entry_price * 1.002:
                return "保本离场"
        return None

    if kind == "无":
        return None
    if kind == "ATR止损":
        def atr_stop(p, d, i):
            a = (atr_map or {}).get((p.code, p.signal_date), 0.06)
            stop = min(0.12, max(0.06, 2.5 * float(a)))
            if float(d["close"].iat[i]) <= p.entry_price * (1.0 - stop):
                return "ATR止损"
            return None
        return atr_stop
    if kind == "不及预期10":
        return lambda p, d, i: fail(p, d, i, 10, 0.02)
    if kind == "不及预期15":
        return lambda p, d, i: fail(p, d, i, 15, 0.03)
    if kind == "不及预期5":
        return lambda p, d, i: fail(p, d, i, 5, 0.02)
    if kind == "跌破MA20":
        return lambda p, d, i: ma_break(p, d, i, 3, 0.05)
    if kind == "保本5":
        return lambda p, d, i: breakeven(p, d, i, 0.05)
    if kind == "保本8":
        return lambda p, d, i: breakeven(p, d, i, 0.08)
    if kind == "不及预期10+保本5":
        def both(p, d, i):
            return fail(p, d, i, 10, 0.02) or breakeven(p, d, i, 0.05)
        return both
    raise ValueError(kind)


def concentration(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {}
    x = tr.sort_values("total_pnl", ascending=False)
    total = float(x.total_pnl.sum())
    top5 = float(x.total_pnl.head(5).sum())
    return {
        "前5笔盈亏占比": top5 / total * 100 if total > 0 else np.nan,
        "剔除前5笔后盈亏": total - top5,
    }


def evaluate(r: dict) -> dict:
    rows = {}
    for label, st in [("全期", START), ("2018起", SUB18), ("2023起", SUB23)]:
        pm = bt.period_metrics(r["equity_curve"], st, END)
        tr = r["trades"].copy()
        if len(tr):
            tr["entry_date"] = pd.to_datetime(tr["entry_date"])
            tr = tr[tr.entry_date >= st]
        ts = bt.trade_stats(tr)
        conc = concentration(tr) if len(tr) else {}
        rows[label] = {**pm, **ts, **conc}
    return rows


def run(name, sig, mask_name, score_col, strategy, exit_kind,
        hard_stop=None, time_stop=None, threshold=None,
        data=None, bench=None, dates=None, uni=None, save=False,
        random_seed=None, negate_score=False, weight=1.0, atr_map=None):
    m = MS[mask_name]
    s = sig.loc[m].copy()
    allowed = set(zip(s["code"], s["signal_date"]))
    score_map = dict(zip(zip(s["code"], s["signal_date"]),
                         s[score_col].astype(float))) if score_col else {}
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        score_map = {k: float(rng.random()) for k in score_map}
    if negate_score:
        score_map = {k: -v for k, v in score_map.items()}
    def entry_filter(code, df, idx, date):
        return (code, date) in allowed
    def score_fn(code, df, idx, date):
        v = score_map.get((code, date), np.nan)
        return v
    # 阈值：低于阈值的信号直接不挂单（在 entry_filter 中做，保证不浪费槽位）
    if threshold is not None:
        def entry_filter(code, df, idx, date):
            v = score_map.get((code, date), np.nan)
            return (code, date) in allowed and np.isfinite(v) and v >= threshold
    print(f"\n运行 {name}: 信号 {len(allowed):,} 分数 {score_col} 阈值 {threshold}", flush=True)
    r = eng.run_backtest(
        data, dates, strategy, universe=uni,
        max_positions=1, flat_weight=weight,
        entry_filter=entry_filter, hard_stop=hard_stop, time_stop=time_stop,
        score_fn=score_fn if score_col else None,
        exit_filter=make_exit(exit_kind, atr_map), label=name)
    rows = evaluate(r)
    rec = {"配置": name, "过滤": mask_name, "打分": score_col or "hist",
           "离场": exit_kind, "硬止损": hard_stop, "时间止损": time_stop,
           "阈值": threshold, "信号数": len(allowed)}
    for label, d in rows.items():
        rec[f"{label}_年化"] = d.get("年化")
        rec[f"{label}_回撤"] = d.get("最大回撤")
        rec[f"{label}_Calmar"] = d.get("Calmar")
        rec[f"{label}_Sharpe"] = d.get("Sharpe")
        rec[f"{label}_交易"] = d.get("交易")
        rec[f"{label}_胜率"] = d.get("胜率")
        rec[f"{label}_平均每笔"] = d.get("平均每笔")
        rec[f"{label}_PF"] = d.get("PF")
        rec[f"{label}_前5占比"] = d.get("前5笔盈亏占比")
        rec[f"{label}_剔除前5盈亏"] = d.get("剔除前5笔后盈亏")
    if save:
        eq = r["equity_curve"].copy(); eq.columns = ["日期", "权益", "持仓数", "持仓市值", "仓位比例"]
        eq.to_csv(OUT / f"权益_{name}.csv", index=False, encoding="utf-8-sig")
        tr = r["trades"].copy()
        if len(tr):
            tr.to_csv(OUT / f"交易_{name}.csv", index=False, encoding="utf-8-sig")
    return rec, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="score")
    args = ap.parse_args()
    print("加载数据…", flush=True)
    data, bench, dates, uni, sig = load_all()
    global bt
    spec2 = importlib.util.spec_from_file_location("bt", HERE / "04_真实回测.py")
    bt = importlib.util.module_from_spec(spec2); sys.modules["bt"] = bt; spec2.loader.exec_module(bt)
    global MS
    MS = masks(sig)
    rows = []
    if args.mode == "score":
        for score_col in [None, "pred_死亡", "pred_fwd20", "pred_fwd40"]:
            name = f"F2_b3_{score_col or 'hist'}"
            rec, _ = run(name, sig, "F2", score_col, "b3_prior_peak", "无",
                         data=data, bench=bench, dates=dates, uni=uni, save=False)
            rows.append(rec)
    elif args.mode == "exit":
        for score_col in [None, "pred_死亡", "pred_fwd20"]:
            for exit_kind in ["无", "不及预期5", "不及预期10", "不及预期15",
                              "跌破MA20", "保本5", "保本8", "不及预期10+保本5"]:
                for hs in [None, 0.08]:
                    if hs is not None and exit_kind not in ("无", "不及预期10"):
                        continue
                    name = f"F2_b3_{score_col or 'hist'}_{exit_kind}_hs{hs}"
                    rec, _ = run(name, sig, "F2", score_col, "b3_prior_peak", exit_kind,
                                 hard_stop=hs, data=data, bench=bench, dates=dates, uni=uni)
                    rows.append(rec)
    elif args.mode == "atr":
        atr_map = dict(zip(zip(sig["code"], sig["signal_date"]), sig["atr20"].astype(float)))
        for w in [1.0, 0.3]:
            for sc in [None, "hist_norm"]:
                name = f"F2_b3_{sc or 'hist'}_ATR_w{w}"
                rec, _ = run(name, sig, "F2", sc, "b3_prior_peak", "ATR止损",
                             data=data, bench=bench, dates=dates, uni=uni,
                             weight=w, atr_map=atr_map)
                rec["单槽权重"] = w
                rows.append(rec)
    elif args.mode == "audit":
        for w in [1.0, 0.3]:
            for sc in [None, "hist_raw", "hist_norm"]:
                name = f"F2_b3_{sc or 'hist'}_hs0.08_w{w}"
                rec, r = run(name, sig, "F2", sc, "b3_prior_peak", "无",
                             hard_stop=0.08, data=data, bench=bench, dates=dates, uni=uni,
                             weight=w, save=True)
                rec["单槽权重"] = w
                tr = r["trades"]
                rec["入场价中位数"] = float(tr.entry_price.median()) if len(tr) else np.nan
                rec["入场价均值"] = float(tr.entry_price.mean()) if len(tr) else np.nan
                rec["高价股占比>30元"] = float((tr.entry_price > 30).mean() * 100) if len(tr) else np.nan
                rows.append(rec)
    elif args.mode == "threshold":
        for w in [1.0, 0.3]:
            for th in [None, 0.001, 0.0015, 0.002, 0.003, 0.004]:
                name = f"F2_b3_histnorm_th{th}_w{w}"
                rec, _ = run(name, sig, "F2", "hist_norm", "b3_prior_peak", "无",
                             hard_stop=0.08, threshold=th,
                             data=data, bench=bench, dates=dates, uni=uni, weight=w)
                rec["单槽权重"] = w
                rows.append(rec)
    elif args.mode == "ablate":
        for mask_name in ["全部", "F2", "F2_去追高约束", "F2_去多次金叉",
                          "F2_去低波", "F2_去缩量", "F2_去MA60", "F2_去高波活跃"]:
            for w in [1.0, 0.3]:
                name = f"{mask_name}_b3_hs0.08_w{w}"
                rec, _ = run(name, sig, mask_name, None, "b3_prior_peak", "无",
                             hard_stop=0.08, data=data, bench=bench, dates=dates, uni=uni,
                             weight=w)
                rec["单槽权重"] = w
                rows.append(rec)
    elif args.mode == "hard":
        for w in [1.0, 0.3]:
            for hs in [None, 0.06, 0.08, 0.10, 0.12]:
                name = f"F2_b3_hist_hs{hs}_w{w}"
                rec, _ = run(name, sig, "F2", None, "b3_prior_peak", "无",
                             hard_stop=hs, data=data, bench=bench, dates=dates, uni=uni,
                             weight=w)
                rec["单槽权重"] = w
                rows.append(rec)
            for ts in [40, 60, 80]:
                name = f"F2_b3_hist_time{ts}_w{w}"
                rec, _ = run(name, sig, "F2", None, "b3_prior_peak", "无",
                             time_stop=ts, data=data, bench=bench, dates=dates, uni=uni,
                             weight=w)
                rec["单槽权重"] = w
                rows.append(rec)
    elif args.mode == "weight":
        for mask_name in ["F2", "F1"]:
            for w in [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]:
                name = f"{mask_name}_b3_hist_hs0.08_w{w}"
                rec, _ = run(name, sig, mask_name, None, "b3_prior_peak", "无",
                             hard_stop=0.08, data=data, bench=bench, dates=dates, uni=uni,
                             weight=w, save=(w in (0.3, 0.5, 1.0)))
                rec["单槽权重"] = w
                rows.append(rec)
    elif args.mode == "random":
        base_name = "F2_b3_hist_hs0.08"
        rec, _ = run(base_name, sig, "F2", None, "b3_prior_peak", "无",
                     hard_stop=0.08, data=data, bench=bench, dates=dates, uni=uni)
        rows.append({**rec, "随机种子": "实际hist"})
        for seed in range(20):
            rec, _ = run(f"F2_b3_rand{seed}_hs0.08", sig, "F2", "hist_norm",
                         "b3_prior_peak", "无", hard_stop=0.08,
                         data=data, bench=bench, dates=dates, uni=uni,
                         random_seed=seed)
            rows.append({**rec, "随机种子": seed})
        rec, _ = run("F2_b3_worsthist_hs0.08", sig, "F2", "hist_norm",
                     "b3_prior_peak", "无", hard_stop=0.08,
                     data=data, bench=bench, dates=dates, uni=uni,
                     negate_score=True)
        rows.append({**rec, "随机种子": "反向hist"})
    elif args.mode == "start":
        for st in ["2013-01-01", "2015-01-01", "2017-01-01",
                   "2019-01-01", "2021-01-01", "2023-01-01"]:
            global START
            START = pd.Timestamp(st)
            # 重新裁日期；权益仍从 100 万开始
            ds = [d for d in dates if d >= START]
            name = f"F2_b3_hist_hs0.08_start{st[:4]}"
            rec, _ = run(name, sig, "F2", None, "b3_prior_peak", "无",
                         hard_stop=0.08, data=data, bench=bench, dates=ds, uni=uni)
            rows.append({"起始": st, **rec})
    elif args.mode == "market":
        for mask_name in ["F2", "F2_MA20", "F2_MA60", "F1_MA20"]:
            for score_col in [None, "pred_死亡"]:
                for hs in [None, 0.08]:
                    name = f"{mask_name}_b3_{score_col or 'hist'}_hs{hs}"
                    rec, _ = run(name, sig, mask_name, score_col, "b3_prior_peak", "无",
                                 hard_stop=hs, data=data, bench=bench, dates=dates, uni=uni,
                                 save=(mask_name in ("F2", "F2_MA20") and hs is not None))
                    rows.append(rec)
    df = pd.DataFrame(rows)
    path = OUT / f"单槽_{args.mode}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("\n", df.round(2).to_string(index=False))
    print("\n保存", path)


if __name__ == "__main__":
    main()
