#!/usr/bin/env python3
"""从 250 日高点到信号日/买入日的对数下跌斜率：越陡越好，还是越缓越好？

口径：
- C 池：全主板含退市股；信号日 is_st=1 不买（PIT），不按当前名称排除后来变 ST 的股票。
- 高点：信号日前 250 个“个股成交日”内的最高后复权价，与 drawdown250 同一窗口。
- 斜率：ln(信号日收盘 / 高点) / 全市场交易日数 × 100，单位 %/日；越负越陡。
  用全市场交易日而不是个股成交日，避免长期停牌把斜率人为拉陡。
- 买入：T+3 开盘，开盘涨停/停牌不买；1w、100 股整手、滑点/佣金/过户费/印花税。
- 15日收益：买入后第 15 个交易日收盘 / 买入开盘 - 1（后复权收益，不含成本）；
  第 15 日停牌则顺延到下一个可成交收盘。策略盈亏另按 +20% 盘中止盈 / 15 日到期。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "11_下跌斜率_对数坐标"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "02_代码"))

from strategy_lib import (Config, _up_limit, build_signals, load_panel,  # noqa: E402
                          load_stocks)
from analyze_exit_variants import simulate  # noqa: E402

PERIODS = [
    ("2005-2026", "2005-01-01", "2026-10-23"),
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2017-2026", "2017-10-19", "2026-10-23"),
]
BUDGET, SLIP, COMM, MIN_COMM, TRANSFER = 10_000.0, 0.001, 0.00025, 5.0, 0.00002


def period_of(d: pd.Timestamp) -> str:
    return "2005-2016" if d <= pd.Timestamp("2016-12-31") else "2017-2026"


def window_stats(g: pd.DataFrame, i: int, n: int, di: dict) -> dict:
    lo = max(0, i - n + 1)
    w = g.iloc[lo:i+1]
    k = int(w.hfq_high.to_numpy().argmax())
    high = float(w.hfq_high.iloc[k])
    hd = w.date.iloc[k]
    close = float(g.hfq_close.iloc[i])
    stock_dur = i - (lo + k)
    mkt_dur = di[g.date.iloc[i]] - di[hd]
    depth = close / high - 1 if high > 0 else np.nan
    slope = (np.log(close / high) / mkt_dur * 100
             if mkt_dur > 0 and high > 0 and close > 0 else np.nan)
    return dict(high=high, high_date=hd, stock_dur=stock_dur, mkt_dur=mkt_dur,
                depth=depth, slope=slope)


def forward_close(b: pd.DataFrame, ei: int, h: int) -> tuple[float, pd.Timestamp] | tuple[float, pd.NaT]:
    """买入后第 h 个交易日收盘；停牌则顺延到下一个有成交的收盘。"""
    k = ei + h - 1
    while k < len(b):
        r = b.iloc[k]
        if np.isfinite(r.hfq_close) and np.isfinite(r.volume) and r.volume > 0:
            return float(r.hfq_close), r.date
        k += 1
    return np.nan, pd.NaT


def main():
    panel = load_panel(columns=[
        "code", "date", "open", "high", "low", "close", "preclose", "volume", "amount",
        "hfq_open", "hfq_high", "hfq_low", "hfq_close", "is_st", "amount20",
        "listing_session", "return_250d", "drawdown250", "rsi14", "close_location",
        "market_return20_all", "market_return20_listed", "market_return"])
    panel["date"] = pd.to_datetime(panel.date)
    stocks = load_stocks()
    names = stocks.set_index("code")["code_name"].to_dict()
    cfg = Config(start="2005-01-01", end="2026-10-23", universe="all", gate="all",
                 extra_filter="is_st == 0", max_rank=2, label="C_top2")
    sig = build_signals(panel, stocks, cfg)
    sig["名称"] = sig.code.map(names).fillna("")

    cal = pd.DatetimeIndex(sorted(panel.date.unique()))
    di = {d: i for i, d in enumerate(cal)}
    mkt = pd.read_parquet(ROOT / "01_数据/market_daily.parquet")
    mkt["date"] = pd.to_datetime(mkt.date)
    mkt = mkt.set_index("date").sort_index()
    mkt_idx = (1 + mkt.market_return.fillna(0)).cumprod()

    codes = set(sig.code)
    p2 = panel[panel.code.isin(codes)][
        ["code", "date", "hfq_high", "hfq_close"]].sort_values(["code", "date"])
    by_code = {c: g.reset_index(drop=True) for c, g in p2.groupby("code")}
    bars = pd.read_parquet(
        ROOT / "01_数据/bars.parquet",
        columns=["code", "date", "open", "high", "low", "close", "preclose", "volume",
                 "is_st", "hfq_open", "hfq_high", "hfq_low", "hfq_close"])
    bars["date"] = pd.to_datetime(bars.date)
    bcode = {c: g.sort_values("date").reset_index(drop=True) for c, g in bars.groupby("code")}

    rows = []
    for r in sig.itertuples():
        g = by_code.get(r.code)
        if g is None:
            continue
        pos = g.index[g.date == r.date]
        if len(pos) == 0:
            continue
        i = int(pos[0])
        w250 = window_stats(g, i, 250, di)
        w60 = window_stats(g, i, 60, di)
        w20 = window_stats(g, i, 20, di)
        row = dict(
            code=r.code, name=r.名称, signal_date=r.date, rank=int(r.rank),
            signal_count=int(r.signal_count) if hasattr(r, "signal_count") else np.nan,
            drawdown250=float(r.drawdown250), rsi14=float(r.rsi14),
            amount20=float(r.amount20) if np.isfinite(r.amount20) else np.nan,
            high250=w250["high"], high250_date=w250["high_date"],
            stock_dur250=w250["stock_dur"], mkt_dur250=w250["mkt_dur"],
            gap250=w250["mkt_dur"] - w250["stock_dur"],
            depth250=w250["depth"], slope250=w250["slope"],
            high60=w60["high"], high60_date=w60["high_date"],
            mkt_dur60=w60["mkt_dur"], depth60=w60["depth"], slope60=w60["slope"],
            high20=w20["high"], high20_date=w20["high_date"],
            mkt_dur20=w20["mkt_dur"], depth20=w20["depth"], slope20=w20["slope"],
            status="no_entry")
        gb = bcode.get(r.code)
        if gb is not None:
            bp = gb.index[gb.date == r.date]
            if len(bp):
                j = int(bp[0])
                if j + 3 + 14 < len(gb):
                    e = gb.iloc[j+3]
                    lim = _up_limit(e.preclose, int(e.is_st))
                    if (np.isfinite(e.open) and e.open > 0 and np.isfinite(e.volume)
                            and e.volume > 0
                            and not (np.isfinite(lim) and e.open >= lim - 1e-9)):
                        ei = j + 3
                        eh = float(e.hfq_open)
                        r5, _ = forward_close(gb, ei, 5)
                        r10, _ = forward_close(gb, ei, 10)
                        r15, d15 = forward_close(gb, ei, 15)
                        if np.isfinite(r15) and eh > 0:
                            seg = gb.iloc[ei:ei+15]
                            # 策略止盈只从买入次日开始检查；浮盈浮亏含买入日
                            tp_high = seg.hfq_high.iloc[1:].max() if len(seg) > 1 else np.nan
                            row.update(
                                status="filled", entry_date=e.date, entry_px=float(e.open),
                                entry_hfq=eh, exit15_date=d15,
                                r5=r5 / eh - 1 if np.isfinite(r5) else np.nan,
                                r10=r10 / eh - 1 if np.isfinite(r10) else np.nan,
                                r15=r15 / eh - 1,
                                mae=float(seg.hfq_low.min()) / eh - 1,
                                mfe=float(seg.hfq_high.max()) / eh - 1,
                                tp20=bool(np.isfinite(tp_high) and tp_high / eh - 1 >= 0.20))
                        # 买入日斜率：高点到 T+3 开盘
                        md = di[e.date] - di[w250["high_date"]]
                        row["mkt_dur_entry"] = md
                        row["slope250_entry"] = (np.log(eh / w250["high"]) / md * 100
                                                 if md > 0 and eh > 0 else np.nan)
                        md60 = di[e.date] - di[w60["high_date"]]
                        row["slope60_entry"] = (np.log(eh / w60["high"]) / md60 * 100
                                                if md60 > 0 and eh > 0 else np.nan)
                        md20 = di[e.date] - di[w20["high_date"]]
                        row["slope20_entry"] = (np.log(eh / w20["high"]) / md20 * 100
                                                if md20 > 0 and eh > 0 else np.nan)
                        # 市场超额：买入日到第15日
                        if (e.date in mkt_idx.index and d15 in mkt_idx.index
                                and np.isfinite(row.get("r15", np.nan))):
                            mret = float(mkt_idx.loc[d15] / mkt_idx.loc[e.date] - 1)
                            row["mkt15"] = mret
                            row["excess15"] = row["r15"] - mret
                        # 策略盈亏（仅前二；与 09/10 同一 simulate）
                        if int(r.rank) <= 2:
                            sim = simulate(gb, j, "tp20")
                            row["sim_status"] = sim.get("status")
                            if sim.get("status") == "filled":
                                row["pnl"] = sim["pnl"]
                                row["ret"] = sim["ret"]
                                row["exit_reason"] = sim["reason"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["期间"] = df.signal_date.map(period_of)
    df["signal_count"] = df.groupby("signal_date")["code"].transform("size")
    df.to_csv(OUT / "下跌斜率_全信号.csv", index=False, encoding="utf-8-sig")

    filled = df[df.status == "filled"].copy()
    top2 = filled[filled["rank"] <= 2].copy()
    top2 = top2[np.isfinite(top2.slope250)].copy()
    print("all signals", len(df), "filled", len(filled), "top2", len(top2))

    # ---------- 1. 前二：五分位 / 十分位 ----------
    def bucket_table(x: pd.DataFrame, by: str, labels: list[str], sample: str,
                     has_pnl: bool = True) -> pd.DataFrame:
        x = x.copy()
        x["组"] = pd.qcut(x[by], len(labels), labels=labels)
        out = x.groupby("组", observed=True).agg(
            笔数=("r15", "size"),
            日均跌幅=("slope250", lambda s: -s.mean()),
            回撤均值=("depth250", "mean"),
            下跌天数=("mkt_dur250", "mean"),
            十五日收益=("r15", "mean"),
            十五日中位=("r15", "median"),
            十五日胜率=("r15", lambda s: (s > 0).mean()),
            止盈20率=("tp20", "mean"),
            最大浮亏=("mae", "mean"),
            最大浮盈=("mfe", "mean"),
            超额十五日=("excess15", "mean"),
        ).reset_index()
        if has_pnl:
            p = x.groupby("组", observed=True)["pnl"]
            out["盈亏笔数"] = p.count().values
            out["平均盈亏"] = p.mean().values
            out["策略收益"] = x.groupby("组", observed=True)["ret"].mean().values
            out["盈亏胜率"] = p.apply(lambda s: (s.dropna() > 0).mean()).values
            out["最差"] = p.min().values
            out["最好"] = p.max().values
        else:
            for c in ["盈亏笔数", "平均盈亏", "策略收益", "盈亏胜率", "最差", "最好"]:
                out[c] = np.nan
        out.insert(0, "样本", sample)
        return out

    q_labels = ["Q1最陡", "Q2", "Q3", "Q4", "Q5最缓"]
    d_labels = [f"D{i}" for i in range(1, 11)]
    top2_q = bucket_table(top2, "slope250", q_labels, "前二_五分位")
    top2_d = bucket_table(top2, "slope250", d_labels, "前二_十分位")

    # 100+ 日全部候选
    cnt = filled.groupby("signal_date").size()
    high_days = set(cnt[cnt >= 100].index)
    all100 = filled[filled.signal_date.isin(high_days)].copy()
    all100_q = bucket_table(all100, "slope250", q_labels, "100+全部候选_五分位",
                            has_pnl=False)

    # 100+ 实际前二（09 不重复 37 笔）
    t09 = pd.read_csv(ROOT / "09_百信号日_榜一二_1w/榜一二_1w_逐笔.csv",
                      parse_dates=["信号日", "entry_date", "exit_date"])
    t09 = t09[(t09["期间"] == "2005-2026") & (t09["版本"] == "不重复")].copy()
    sl = df[["code", "signal_date", "slope250", "depth250", "mkt_dur250",
             "r15", "excess15", "tp20", "mae", "mfe"]].drop_duplicates(
        ["code", "signal_date"])
    t09 = t09.merge(sl, left_on=["代码", "信号日"], right_on=["code", "signal_date"], how="left")
    t09["组"] = pd.qcut(t09.slope250, 3, labels=["陡", "中", "缓"])
    actual = t09.groupby("组", observed=True).agg(
        笔数=("pnl", "size"),
        日均跌幅=("slope250", lambda s: -s.mean()),
        回撤均值=("depth250", "mean"),
        下跌天数=("mkt_dur250", "mean"),
        十五日收益=("r15", "mean"),
        十五日中位=("r15", "median"),
        十五日胜率=("r15", lambda s: (s > 0).mean()),
        止盈20率=("tp20", "mean"),
        最大浮亏=("mae", "mean"),
        最大浮盈=("mfe", "mean"),
        超额十五日=("excess15", "mean"),
        盈亏笔数=("pnl", "size"),
        平均盈亏=("pnl", "mean"),
        策略收益=("ret", "mean"),
        盈亏胜率=("pnl", lambda s: (s > 0).mean()),
        最差=("pnl", "min"),
        最好=("pnl", "max"),
    ).reset_index()
    actual.insert(0, "样本", "100+实际买入_三分位")

    buckets = pd.concat([top2_q, top2_d, all100_q, actual], ignore_index=True)
    buckets.to_csv(OUT / "下跌斜率_分桶.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. 回归 / 同日检验 ----------
    def ols_cluster(y, X, groups):
        X = np.asarray(X, dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for gi in pd.unique(groups):
            m = groups == gi
            s = X[m].T @ e[m]
            meat += np.outer(s, s)
        G, N, K = len(pd.unique(groups)), len(y), X.shape[1]
        V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (N - 1) / (N - K)
        return beta, np.sqrt(np.diag(V)), N, G

    reg_rows = []
    t2 = top2.dropna(subset=["r15", "steep", "depth250"]).copy() if "steep" in top2 else top2.copy()
    t2["steep"] = -t2.slope250
    t2["year"] = t2.signal_date.dt.year
    t2["ym"] = t2.signal_date.dt.to_period("M").astype(str)
    for ycol in ["r15", "pnl"]:
        x = t2.dropna(subset=[ycol]).copy()
        for fe in [None, "year", "ym"]:
            X = x[["steep", "depth250"]].to_numpy()
            if fe == "year":
                X = np.column_stack([X, pd.get_dummies(x.year, drop_first=True, dtype=float).to_numpy()])
            elif fe == "ym":
                X = np.column_stack([X, pd.get_dummies(x.ym, drop_first=True, dtype=float).to_numpy()])
            beta, se, N, G = ols_cluster(x[ycol].to_numpy(), X, x.signal_date.to_numpy())
            reg_rows.append(dict(检验=f"前二_{ycol}_固定效应", 模型=fe or "无",
                                 变量="斜率(陡为正)", 系数=beta[1], 标准误=se[1],
                                 t值=beta[1] / se[1], 样本=N, 天数=G))
            reg_rows.append(dict(检验=f"前二_{ycol}_固定效应", 模型=fe or "无",
                                 变量="回撤深度", 系数=beta[2], 标准误=se[2],
                                 t值=beta[2] / se[2], 样本=N, 天数=G))

    # 同日：两只前二，陡 vs 缓
    pair_rows = []
    for day, g in t2.groupby("signal_date"):
        if len(g) != 2:
            continue
        s = g.sort_values("slope250").iloc[0]
        f = g.sort_values("slope250").iloc[-1]
        pair_rows.append(dict(date=day, s_r15=s.r15, f_r15=f.r15,
                              s_pnl=s.pnl, f_pnl=f.pnl,
                              s_slope=s.slope250, f_slope=f.slope250,
                              s_depth=s.depth250, f_depth=f.depth250,
                              s_rank=s["rank"], f_rank=f["rank"]))
    pair = pd.DataFrame(pair_rows)
    for metric in ["r15", "pnl"]:
        sub = pair[[f"s_{metric}", f"f_{metric}"]].dropna()
        d = sub[f"s_{metric}"] - sub[f"f_{metric}"]
        reg_rows.append(dict(检验="同日两只前二_陡-缓", 模型="配对", 变量=metric,
                             系数=d.mean(), 标准误=d.std(ddof=1) / np.sqrt(len(d)),
                             t值=ttest_rel(sub[f"s_{metric}"], sub[f"f_{metric}"]).statistic,
                             样本=len(sub), 天数=len(sub)))

    # 同日全部候选：Q1/Q2/Q5
    within_rows = []
    for name, x in [("全部信号_同日", filled[filled.signal_date.map(cnt) >= 10]),
                    ("100+_同日", all100)]:
        recs = []
        for day, g in x.groupby("signal_date"):
            if len(g) < 10:
                continue
            g = g.sort_values("slope250")
            k = max(1, int(round(len(g) * 0.2)))
            recs.append(dict(date=day, q1=g.iloc[:k].r15.mean(),
                             q2=g.iloc[k:2*k].r15.mean(), q5=g.iloc[-k:].r15.mean(),
                             n=len(g)))
        p = pd.DataFrame(recs)
        for a, b in [("q1", "q5"), ("q1", "q2"), ("q2", "q5")]:
            diff = p[a] - p[b]
            within_rows.append(dict(样本=name, 对比=f"{a.upper()}-{b.upper()}",
                                    平均差=diff.mean(), t值=ttest_rel(p[a], p[b]).statistic,
                                    胜出天数占比=(diff > 0).mean(), 天数=len(p)))
        # dummy regression
        x = x.copy()
        x["steep_top"] = x.groupby("signal_date")["slope250"].transform(
            lambda s: (s <= s.quantile(1/3)).astype(float))
        x["flat_top"] = x.groupby("signal_date")["slope250"].transform(
            lambda s: (s >= s.quantile(2/3)).astype(float))
        for c in ["r15", "depth250"]:
            x[c + "_dm"] = x[c] - x.groupby("signal_date")[c].transform("mean")
        for cols in [["steep_top", "flat_top"], ["steep_top", "flat_top", "depth250_dm"]]:
            beta, se, N, G = ols_cluster(x.r15_dm.to_numpy(), x[cols].to_numpy(),
                                         x.signal_date.to_numpy())
            for j, c in enumerate(cols, 1):
                within_rows.append(dict(样本=name, 对比="同日虚拟变量回归", 平均差=beta[j],
                                        t值=beta[j] / se[j], 胜出天数占比=np.nan, 天数=G,
                                        变量=c, 样本量=N))
    within = pd.DataFrame(within_rows)

    pd.DataFrame(reg_rows).to_csv(OUT / "下跌斜率_回归.csv", index=False, encoding="utf-8-sig")
    within.to_csv(OUT / "下跌斜率_同日对比.csv", index=False, encoding="utf-8-sig")

    # ---------- 3. 相关性：250/60/20 日、信号/买入 ----------
    corr_rows = []
    xname = {"slope250": "250日高点_信号日", "slope250_entry": "250日高点_买入日",
             "slope60": "60日高点_信号日", "slope60_entry": "60日高点_买入日",
             "slope20": "20日高点_信号日", "slope20_entry": "20日高点_买入日"}
    yname = {"r5": "5日收益", "r10": "10日收益", "r15": "15日收益", "ret": "策略收益"}
    for xcol in ["slope250", "slope250_entry", "slope60", "slope60_entry",
                 "slope20", "slope20_entry"]:
        for ycol in ["r5", "r10", "r15", "ret"]:
            d = top2[[xcol, ycol]].dropna()
            if len(d) > 10:
                c, p = spearmanr(-d[xcol], d[ycol])
                corr_rows.append(dict(斜率口径=xname[xcol], 结果=yname[ycol],
                                      相关系数=c, 样本=len(d), p值=p,
                                      说明="陡峭度= -对数斜率；正=越陡越好"))
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "下跌斜率_相关性.csv", index=False, encoding="utf-8-sig")

    # ---------- 4. 前二逐笔（中文表头） ----------
    out = top2.copy()
    out["信号数"] = out.groupby("signal_date")["code"].transform("size")
    trade = pd.DataFrame({
        "期间": out["期间"],
        "代码": out["code"],
        "代码6位": out["code"].str.split(".").str[-1].str.zfill(6),
        "名称": out["name"],
        "信号日": out["signal_date"].dt.date,
        "排名": out["rank"].astype(int),
        "信号数": out["signal_count"].astype(int),
        "250日高点": out["high250"].round(3),
        "高点日": out["high250_date"].dt.date,
        "下跌交易日": out["mkt_dur250"].astype(int),
        "对数斜率%每日": out["slope250"].round(3),
        "日均跌幅%每日": (-out["slope250"]).round(3),
        "回撤深度": out["depth250"].round(4),
        "买入日": out["entry_date"].dt.date,
        "买入价": out["entry_px"].round(2),
        "十五日收益": out["r15"].round(4),
        "超额十五日": out["excess15"].round(4),
        "策略盈亏": out["pnl"].round(2),
        "策略收益": out["ret"].round(4),
        "止盈20": out["tp20"],
        "最大浮亏": out["mae"].round(4),
        "最大浮盈": out["mfe"].round(4),
    })
    trade.to_csv(OUT / "下跌斜率_逐笔.csv", index=False, encoding="utf-8-sig")

    print("\n前二五分位：")
    print(top2_q.round(3).to_string(index=False))
    print("\n100+全部候选五分位：")
    print(all100_q.round(3).to_string(index=False))
    print("\n100+实际买入三分位：")
    print(actual.round(3).to_string(index=False))
    print("\n回归：")
    print(pd.DataFrame(reg_rows).round(4).to_string(index=False))
    print("\n同日：")
    print(within.round(4).to_string(index=False))
    print("\n相关性：")
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
