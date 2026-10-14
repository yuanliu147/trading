#!/usr/bin/env python3
"""MACD 环境与量价研究；仅作历史统计，不作投资建议。"""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from MACD金叉死叉研究 import CAPITAL, COMM, DB, WARM, stamp

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
END = pd.Timestamp("2026-10-12")
SECTOR_CODES = {
    "化学制药": "881140.TI", "医疗器械": "881144.TI", "房地产": "881153.TI",
    "文化传媒": "881164.TI", "机场航运": "881151.TI", "油气开采及服务": "881107.TI",
    "消费电子": "881124.TI", "港口航运": "881148.TI", "煤炭开采加工": "881105.TI",
    "燃气": "881146.TI", "物流": "881152.TI", "石油加工贸易": "881180.TI",
    "纺织制造": "881135.TI", "综合": "881165.TI", "美容护理": "881182.TI",
    "计算机设备": "881130.TI", "证券": "881157.TI", "通信设备": "881129.TI",
    "造纸": "881137.TI", "非金属材料": "881167.TI",
}
WINDOWS = {
    "前期2018-2021": (pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")),
    "共同期2022H2-2026": (pd.Timestamp("2022-09-01"), END),
    "前段2022H2-2023": (pd.Timestamp("2022-09-01"), pd.Timestamp("2023-12-31")),
    "后段2024-2026": (pd.Timestamp("2024-01-01"), END),
}
STOCK_SQL = """SELECT thscode, date, open, high, low, close, volume, amount
FROM v_daily_qfq WHERE thscode IN (SELECT UNNEST(?::VARCHAR[]))
AND date <= '2026-10-13' ORDER BY thscode, date"""


def read_panel(path):
    d = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    assert not d.date.duplicated().any(), str(path)
    assert d.date.notna().all()
    return d


def technical(d):
    d = d.copy()
    c = d.close
    d["ret1"] = c.pct_change(fill_method=None)
    for n in (5, 20, 60):
        d[f"ret{n}"] = c.pct_change(n, fill_method=None)
    for n in (10, 20, 60, 250):
        d[f"ma{n}"] = c.rolling(n).mean()
    d["ma60_slope"] = d.ma60 / d.ma60.shift(10) - 1
    d["dif"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["dea"] = d.dif.ewm(span=9, adjust=False).mean()
    d["hist"] = 2 * (d.dif - d.dea)
    d["golden"] = (d.dif > d.dea) & (d.dif.shift() <= d.dea.shift())
    d["death"] = (d.dif < d.dea) & (d.dif.shift() >= d.dea.shift())
    d["hist_change"] = d["hist"].diff()
    d["hist_shrink2"] = ((d["hist"] > 0) & (d.hist_change < 0) & (d.hist_change.shift() < 0)).astype(float)
    d["dif_change"] = d.dif.diff()
    d["vol20"] = d.ret1.rolling(20).std()
    d["vol_ratio"] = d.vol20 / d.vol20.shift().rolling(250).median()
    return d


def index_panel(path, prefix):
    d = technical(read_panel(path))
    columns = ["close", "ma10", "ma20", "ma60", "ma60_slope", "ret1", "ret5", "ret20",
               "hist", "hist_change", "hist_shrink2", "golden", "vol20", "vol_ratio"]
    d["golden"] = d.golden.astype(float)
    d.loc[d.index < WARM, ["hist", "hist_change", "hist_shrink2", "golden"]] = np.nan
    return d[["date", *columns]].rename(columns={c: prefix + c for c in columns})


def stock_features(d):
    d = technical(d)
    c = d.close
    d["rvol"] = d.volume / d.volume.shift().rolling(20).mean()
    d["ramount"] = d.amount / d.amount.shift().rolling(20).mean()
    d["volume_trend"] = d.volume.rolling(5).mean() / d.volume.rolling(20).mean()
    up = d.volume.where(d.ret1 > 0, 0).rolling(10).sum()
    down = d.volume.where(d.ret1 < 0, 0).rolling(10).sum()
    d["up_volume_share"] = up / (up + down).replace(0, np.nan)
    d["amount_rank"] = d.amount.rolling(250).rank(pct=True)
    d["amount_mean250"] = d.amount.rolling(250).mean()
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / c
    d["efficiency20"] = c.diff(20).abs() / c.diff().abs().rolling(20).sum().replace(0, np.nan)
    d["range_position"] = (c - c.rolling(250).min()) / (c.rolling(250).max() - c.rolling(250).min()).replace(0, np.nan)
    d["upper_shadow"] = (d.high - d[["open", "close"]].max(axis=1)) / (d.high - d.low).replace(0, np.nan)
    return d


def build():
    leaders = pd.read_csv(HERE / "MACD研究股票名单.csv")
    with duckdb.connect(DB, read_only=True) as con:
        raw = con.execute(STOCK_SQL, [leaders.thscode.tolist()]).fetchdf()
    raw["date"] = pd.to_datetime(raw.date)
    hs = index_panel(BASE / "hs300_daily.csv", "hs_")
    breadth = read_panel(BASE / "市场宽度历史面板.csv")[["date", "n", "pct20", "pct60", "pct250"]]
    breadth["breadth_change5"] = breadth.pct20.diff(5)
    assert breadth[["pct20", "pct60", "pct250"]].ge(0).all().all()
    assert breadth[["pct20", "pct60", "pct250"]].le(1).all().all()
    sectors = {ind: index_panel(BASE / "data_cache/sector_hist" / f"{code}.csv", "sec_")
               for ind, code in SECTOR_CODES.items()}
    panels = {}
    for row in leaders.itertuples():
        d = raw.loc[raw.thscode == row.thscode].reset_index(drop=True)
        assert not d.date.duplicated().any()
        assert d[["open", "high", "low", "close"]].gt(0).all().all()
        assert d.high.ge(d[["open", "close", "low"]].max(axis=1)).all()
        assert d.low.le(d[["open", "close"]].min(axis=1)).all()
        d = stock_features(d)
        for context in (hs, breadth, sectors[row.industry]):
            d = d.merge(context, on="date", how="left", validate="one_to_one")
        d["sector_relative20"] = d.sec_ret20 - d.hs_ret20
        calendar_close = d.set_index("date").close.reindex(hs.date).ffill()
        # 停牌日沿用最近收盘价，仅用于对齐市场交易日的相对收益。
        for n in (5, 20):
            aligned_return = calendar_close.pct_change(n, fill_method=None).reindex(d.date).to_numpy()
            for prefix in ("hs", "sec"):
                d[f"relative_{prefix}{n}"] = aligned_return - d[f"{prefix}_ret{n}"]
        d["name"], d["industry"] = row.name, row.industry
        panels[row.thscode] = d
    return leaders, panels


def rules(d):
    entry, exits = {}, {}

    def add(target, name, columns, condition):
        known = d[columns].notna().all(axis=1)
        target[name] = condition.astype(float).where(known).to_numpy()

    E = lambda name, cols, cond: add(entry, name, cols, cond)
    X = lambda name, cols, cond: add(exits, name, cols, cond)
    E("大盘在MA60上方", ["hs_close", "hs_ma60"], d.hs_close > d.hs_ma60)
    E("大盘MA60上行", ["hs_ma60_slope"], d.hs_ma60_slope > 0)
    E("大盘MACD多头状态", ["hs_hist"], d.hs_hist > 0)
    E("大盘当日金叉", ["hs_golden"], d.hs_golden > 0)
    E("大盘低波动", ["hs_vol_ratio"], d.hs_vol_ratio < 1)
    E("大盘高波动", ["hs_vol_ratio"], d.hs_vol_ratio >= 1)
    E("市场宽度低于25%", ["pct20"], d.pct20 < .25)
    E("市场宽度高于50%", ["pct20"], d.pct20 > .5)
    E("市场宽度5日回升超5百分点", ["breadth_change5"], d.breadth_change5 > .05)
    E("行业在MA60上方", ["sec_close", "sec_ma60"], d.sec_close > d.sec_ma60)
    E("行业MA60上行", ["sec_ma60_slope"], d.sec_ma60_slope > 0)
    E("行业MACD多头状态", ["sec_hist"], d.sec_hist > 0)
    E("行业当日金叉", ["sec_golden"], d.sec_golden > 0)
    E("行业20日强于大盘", ["sector_relative20"], d.sector_relative20 > 0)
    E("行业20日弱于大盘", ["sector_relative20"], d.sector_relative20 < 0)
    E("个股20日强于行业", ["relative_sec20"], d.relative_sec20 > 0)
    E("个股20日强于大盘", ["relative_hs20"], d.relative_hs20 > 0)
    E("成交量大于前20日均量1.5倍", ["rvol"], d.rvol >= 1.5)
    E("成交量低于前20日均量0.8倍", ["rvol"], d.rvol < .8)
    E("成交额大于前20日均额1.5倍", ["ramount"], d.ramount >= 1.5)
    E("量能扩张5日均量大于20日", ["volume_trend"], d.volume_trend > 1)
    E("10日上涨日成交量占比大于60%", ["up_volume_share"], d.up_volume_share > .6)
    E("成交额处于250日最高四分位", ["amount_rank"], d.amount_rank > .75)
    E("近250日日均成交额大于2亿", ["amount_mean250"], d.amount_mean250 > 2e8)
    E("ATR波动幅度低于3%", ["atr_pct"], d.atr_pct < .03)
    E("价格趋势效率大于0.3", ["efficiency20"], d.efficiency20 > .3)
    E("价格位于年内区间下四分位", ["range_position"], d.range_position < .25)
    E("大盘行业均在MA60上方", ["hs_close", "hs_ma60", "sec_close", "sec_ma60"],
      (d.hs_close > d.hs_ma60) & (d.sec_close > d.sec_ma60))
    E("行业多头且个股强于行业", ["sec_hist", "relative_sec20"], (d.sec_hist > 0) & (d.relative_sec20 > 0))
    E("行业多头且放量", ["sec_hist", "rvol"], (d.sec_hist > 0) & (d.rvol >= 1.5))
    E("大盘低波动且宽度低于25%", ["hs_vol_ratio", "pct20"], (d.hs_vol_ratio < 1) & (d.pct20 < .25))
    E("市场宽度回升且上涨量占优", ["breadth_change5", "up_volume_share"],
      (d.breadth_change5 > .05) & (d.up_volume_share > .6))
    shrink = d.hist_shrink2 > 0
    X("仅个股红柱连缩2天", ["hist_shrink2"], shrink)
    X("个股行业红柱均连缩2天", ["hist_shrink2", "sec_hist_shrink2"], shrink & (d.sec_hist_shrink2 > 0))
    X("个股红柱缩短且大盘MACD空头", ["hist_shrink2", "hs_hist"], shrink & (d.hs_hist < 0))
    X("个股红柱缩短且市场宽度5日下降10百分点", ["hist_shrink2", "breadth_change5"],
      shrink & (d.breadth_change5 < -.1))
    X("放量下跌破MA10且DIF向下", ["ret1", "rvol", "ma10", "dif_change"],
      (d.ret1 < 0) & (d.rvol >= 1.5) & (d.close < d.ma10) & (d.dif_change < 0))
    X("个股破MA10且行业破MA20", ["ma10", "sec_close", "sec_ma20"],
      (d.close < d.ma10) & (d.sec_close < d.sec_ma20))
    X("个股红柱缩短且5日弱于行业", ["hist_shrink2", "relative_sec5"], shrink & (d.relative_sec5 < 0))
    X("放量长上影且个股红柱缩短", ["hist_shrink2", "upper_shadow", "rvol"],
      shrink & (d.upper_shadow > .5) & (d.rvol >= 1.5))
    return entry, exits


def simulate(d, start, end, gate=None, early_exit=None, slippage=0.0):
    dates = d.date.to_numpy()
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    golden, death = d.golden.to_numpy(), d.death.to_numpy()
    cash, shares, pos, pending = CAPITAL, 0.0, None, None
    equity, held, records = [], [], []
    indices = np.flatnonzero((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))
    if not len(indices):
        return {"final": CAPITAL, "mdd": 0.0, "exposure": 0.0}, pd.DataFrame()
    for t in indices:
        if pending == "sell" and not (h[t] == l[t] and o[t] < c[t - 1]):
            price = o[t] * (1 - slippage)
            proceeds = shares * price
            cash = proceeds - max(5.0, proceeds * COMM) - proceeds * stamp(pd.Timestamp(dates[t]))
            records.append({**pos, "x": t, "exit_date": dates[t], "exit_px": price,
                            "ret": cash / pos["capital"] - 1, "pnl": cash - pos["capital"],
                            "hold": t - pos["e"]})
            shares, pos, pending = 0.0, None, None
        elif pending == "buy":
            pending = None
            if not (h[t] == l[t] and o[t] > c[t - 1]):
                price = o[t] * (1 + slippage)
                pos = {"i": t - 1, "e": t, "signal_date": dates[t - 1], "entry_date": dates[t],
                       "entry_px": price, "capital": cash}
                shares = (cash - max(5.0, cash * COMM / (1 + COMM))) / price
                cash = 0.0
        equity.append(cash + shares * c[t])
        held.append(pos is not None)
        if t < WARM or t == indices[-1] or pending is not None:
            continue
        if pos is not None:
            if death[t] or (early_exit is not None and early_exit[t] == 1):
                pending = "sell"
                pos.update(d=t, reason="death" if death[t] else "early")
        elif golden[t] and (gate is None or gate[t] == 1):
            pending = "buy"
    curve = np.r_[CAPITAL, equity]
    trade_df = pd.DataFrame(records)
    profit = sum(r["pnl"] for r in records)
    if pos is not None:
        profit += shares * c[indices[-1]] - pos["capital"]
    assert np.isclose(CAPITAL + profit, curve[-1])
    if records:
        assert all(r["i"] < r["e"] <= r["d"] < r["x"] for r in records)
        assert all(records[k]["x"] < records[k + 1]["e"] for k in range(len(records) - 1))
    return {"final": curve[-1], "mdd": (curve / np.maximum.accumulate(curve) - 1).min(),
            "exposure": np.mean(held), "open": int(pos is not None)}, trade_df


def stats(r):
    r = pd.Series(r, dtype=float).dropna()
    w, loss = r[r > 0], r[r < 0]
    return {"笔数": len(r), "胜率": (r > 0).mean(), "盈亏比": w.mean() / -loss.mean() if len(loss) else np.nan,
            "PF": w.sum() / -loss.sum() if len(loss) else np.nan, "每笔均收益": r.mean()}


def compare(panels, masks, window, mode, name, slippage=0):
    start, end = window
    base_metrics, metrics, trades, baseline_trades, triggers = [], [], [], [], []
    for code, d in panels.items():
        v = masks[code] if masks is not None else None
        available = np.isfinite(v).astype(float) if v is not None else None
        if v is not None:
            inside = d.date.between(start, end).to_numpy() & (np.arange(len(d)) >= WARM)
            if not np.any(np.isfinite(v) & inside):
                continue
        base_gate = available if mode == "入场过滤" else None
        bm, bt = simulate(d, start, end, gate=base_gate, slippage=slippage)
        m, tt = simulate(d, start, end, gate=v if mode == "入场过滤" else None,
                         early_exit=v if mode == "逐日退出" else None, slippage=slippage)
        base_metrics.append({"thscode": code, **bm})
        metrics.append({"thscode": code, **m})
        if len(tt):
            trades.append(tt.assign(thscode=code))
        if len(bt):
            baseline_trades.append(bt.assign(thscode=code))
        if mode == "逐日退出" and len(bt):
            for tr in bt.itertuples():
                hits = np.flatnonzero(v[tr.e:tr.d] == 1)
                gain, hit = 0.0, len(hits) > 0
                if hit:
                    x = tr.e + hits[0] + 1
                    while x < tr.x and d.high.iloc[x] == d.low.iloc[x] and d.open.iloc[x] < d.close.iloc[x - 1]:
                        x += 1
                    shares = (tr.capital - max(5, tr.capital * COMM / (1 + COMM))) / tr.entry_px
                    proceeds = shares * d.open.iloc[x] * (1 - slippage)
                    net = proceeds - max(5, proceeds * COMM) - proceeds * stamp(d.date.iloc[x])
                    gain = net / tr.capital - 1 - tr.ret
                triggers.append({"win": tr.ret > 0, "trigger": hit, "improvement": gain})
    if not metrics:
        return None, None
    M, B = pd.DataFrame(metrics).set_index("thscode"), pd.DataFrame(base_metrics).set_index("thscode")
    T = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["ret"])
    BT = pd.concat(baseline_trades, ignore_index=True) if baseline_trades else pd.DataFrame(columns=["ret"])
    r = np.log(M.final / B.final)
    row = {"模式": mode, "条件": name, **stats(T.ret), "基准笔数": len(BT),
           "基准胜率": stats(BT.ret)["胜率"], "基准PF": stats(BT.ret)["PF"],
           "期末权益中位数": M.final.median(), "基准期末权益中位数": B.final.median(),
           "期末比基准几何均值": np.exp(r.mean()), "胜基准只数": int((M.final > B.final + 1e-7).sum()),
           "标的数": len(M), "MDD中位": M.mdd.median(), "基准MDD中位": B.mdd.median(),
           "持仓比例": M.exposure.mean(), "基准持仓比例": B.exposure.mean()}
    if triggers:
        F = pd.DataFrame(triggers)
        fired = F[F.trigger]
        row.update({"亏损单触发率": F.loc[~F.win, "trigger"].mean(),
                    "盈利单触发率": F.loc[F.win, "trigger"].mean(),
                    "触发后提前卖更好比例": (fired.improvement > 1e-12).mean(),
                    "全部原交易平均改善": F.improvement.mean()})
    return row, T


def feature_table(panels, masks, window, name):
    rows = []
    start, end = window
    for code, d in panels.items():
        _, trades = simulate(d, start, end)
        if len(trades):
            values = masks[code][trades.i.to_numpy()]
            rows.append(trades[["ret", "signal_date"]].assign(feature=values, thscode=code))
    if not rows:
        return None
    all_trades = pd.concat(rows, ignore_index=True)
    T = all_trades[all_trades.feature.notna()]
    if T.empty:
        return None
    yes, win = T.feature == 1, T.ret > 0
    r = {"条件": name, "可用样本": len(T), "缺失样本": len(all_trades) - len(T),
         "满足样本": int(yes.sum()), "覆盖率": yes.mean(),
         "亏损单符合率": yes[~win].mean(), "盈利单符合率": yes[win].mean(),
         "不满足时胜率": stats(T.loc[~yes, "ret"])["胜率"], "同口径基准胜率": win.mean()}
    r.update(stats(T.loc[yes, "ret"]))
    return r


def validate(panels):
    ref = pd.read_csv(HERE / "MACD金叉死叉个股汇总.csv").set_index("thscode")
    closed, max_error = 0, 0.0
    for code, d in panels.items():
        m, t = simulate(d, d.date.iloc[0], pd.Timestamp("2026-10-13"))
        max_error = max(max_error, abs(m["final"] - ref.at[code, "final_equity"]))
        closed += len(t)
        for k in (500, 1300, len(d) - 3):
            shorter = stock_features(d[["date", "open", "high", "low", "close", "volume", "amount"]].iloc[:k])
            for col in ("dif", "dea", "rvol", "atr_pct", "efficiency20", "amount_rank"):
                np.testing.assert_allclose(shorter[col], d[col].iloc[:k], equal_nan=True)
        assert d.loc[d.date < "2022-01-04", "sector_relative20"].isna().all()
    assert max_error < 1e-6 and closed == 1724, (max_error, closed)
    print(f"验证: 基准已平仓 {closed} 笔，逐只期末最大误差 {max_error:.10f} 元；技术指标前缀一致。")


def robustness(panels, entries, exits):
    candidates = [("入场过滤", "市场宽度5日回升超5百分点", entries["市场宽度5日回升超5百分点"]),
                  ("入场过滤", "行业20日弱于大盘", entries["行业20日弱于大盘"]),
                  ("逐日退出", "放量长上影且个股红柱缩短", exits["放量长上影且个股红柱缩短"])]
    rows = []
    for mode, name, masks in candidates:
        for slip in (0, .0005, .001):
            row, _ = compare(panels, masks, WINDOWS["共同期2022H2-2026"], mode, name, slippage=slip)
            rows.append({"检查": "单边滑点", "参数": slip, **row})
        ratios = []
        for code, d in panels.items():
            available = np.isfinite(masks[code]).astype(float) if mode == "入场过滤" else None
            bm, _ = simulate(d, *WINDOWS["共同期2022H2-2026"], gate=available)
            sm, _ = simulate(d, *WINDOWS["共同期2022H2-2026"],
                             gate=masks[code] if mode == "入场过滤" else None,
                             early_exit=masks[code] if mode == "逐日退出" else None)
            ratios.append(np.log(sm["final"] / bm["final"]))
        ratios = np.array(ratios)
        rng = np.random.default_rng(20261015)
        boot = np.exp(rng.choice(ratios, (10000, len(ratios)), replace=True).mean(axis=1))
        rows.append({"检查": "逐股重采样描述区间", "条件": name, "模式": mode,
                     "区间下界": np.quantile(boot, .025), "区间上界": np.quantile(boot, .975),
                     "删除最有利股票后的相对权益": np.exp((ratios.sum() - ratios.max()) / (len(ratios) - 1))})
    for threshold in (.025, .05, .075):
        masks = {c: (d.breadth_change5 > threshold).astype(float).where(d.breadth_change5.notna()).to_numpy()
                 for c, d in panels.items()}
        row, _ = compare(panels, masks, WINDOWS["共同期2022H2-2026"], "入场过滤", "市场宽度回升阈值敏感性")
        rows.append({"检查": "5日回升阈值", "参数": threshold, **row})
    return pd.DataFrame(rows)


def main():
    leaders, panels = build()
    validate(panels)
    prepared = {c: rules(d) for c, d in panels.items()}
    entries = {name: {c: p[0][name] for c, p in prepared.items()} for name in next(iter(prepared.values()))[0]}
    exits = {name: {c: p[1][name] for c, p in prepared.items()} for name in next(iter(prepared.values()))[1]}
    print("源: hithink-finance本地前复权日线/远端行业指数缓存；截止由沪深300缓存限制为2026-10-12。")
    print("换手率未测试：没有历史流通股本；成交额、成交量不替代换手率。")
    print("大盘使用沪深300；市场宽度使用至少250根本地日线且当日有报价的股票，不使用股价几何均值。")
    print("每只独立1万元全仓、允许碎股、佣金万2.5最低5元，历史印花税；一字板简化，非真实成交复原。")
    print("窗口分别以1万元空仓重启；未平仓计入权益但不计入胜率；不可用特征不算不满足。")
    print("所有阈值为固定探索性设定；历史窗口已被反复查看，时间切分不是未经查看的样本外验证。")
    print("SQL:", STOCK_SQL)
    print("股票日线行数:", sum(len(d) for d in panels.values()))
    print("行业缓存行数:", sum(len(read_panel(BASE / "data_cache/sector_hist" / f"{v}.csv")) for v in SECTOR_CODES.values()))
    print("数据库:", DB)
    summary, features, daily = [], [], []
    for window_name, window in WINDOWS.items():
        baseline, _ = compare(panels, None, window, "基准", "金叉买死叉卖")
        summary.append({"窗口": window_name, **baseline})
        for name, masks in entries.items():
            row = feature_table(panels, masks, window, name)
            if row is not None:
                features.append({"窗口": window_name, **row})
            row, _ = compare(panels, masks, window, "入场过滤", name)
            if row is not None:
                summary.append({"窗口": window_name, **row})
        for name, masks in exits.items():
            row, _ = compare(panels, masks, window, "逐日退出", name)
            if row is not None:
                daily.append({"窗口": window_name, **row})
    S, F, D = pd.DataFrame(summary), pd.DataFrame(features), pd.DataFrame(daily)
    S.to_csv(HERE / "MACD环境条件账户对比.csv", index=False)
    F.to_csv(HERE / "MACD环境条件符合率.csv", index=False)
    D.to_csv(HERE / "MACD环境联合逐日退出.csv", index=False)
    R = robustness(panels, entries, exits)
    R.to_csv(HERE / "MACD环境策略稳健性检查.csv", index=False)
    signals = []
    for code, d in panels.items():
        _, t = simulate(d, pd.Timestamp("2017-01-01"), END)
        inputs = d.iloc[t.i.to_numpy()].reset_index(drop=True)
        signals.append(pd.concat([t.reset_index(drop=True), inputs.drop(columns="thscode")], axis=1).assign(thscode=code))
    signal_df = pd.concat(signals, ignore_index=True)
    signal_df.to_csv(HERE / "MACD环境研究信号明细.csv", index=False)
    pd.DataFrame([{"status": "invalid_superseded", "reason": "此前五股组合存在缺失报价估值与跌停挂单问题，本轮只验证每只独立全仓账户，不再引用旧组合结果"}]).to_csv(HERE / "MACD旧组合结果已作废.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 30)
    columns = ["条件", "笔数", "胜率", "盈亏比", "PF", "每笔均收益", "期末权益中位数", "期末比基准几何均值", "胜基准只数", "MDD中位", "持仓比例"]
    for w in ("共同期2022H2-2026", "后段2024-2026"):
        print(f"\n{w} 入场条件:")
        print(S.loc[S.窗口 == w, columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n{w} 逐日监控:")
        print(D.loc[D.窗口 == w, columns + ["亏损单触发率", "盈利单触发率", "触发后提前卖更好比例", "全部原交易平均改善"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    sources = [HERE / "MACD研究股票名单.csv", BASE / "hs300_daily.csv", BASE / "市场宽度历史面板.csv"]
    sources += [BASE / "data_cache/sector_hist" / f"{v}.csv" for v in SECTOR_CODES.values()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    print("\n输入指纹:", json.dumps(hashes, ensure_ascii=False))
    print("\n敏感性检查（逐股重采样区间未控制行业相关性和多重筛选，不是显著性证明）:")
    print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"输出行数: gates={len(S)}, features={len(F)}, daily_exits={len(D)}, signals={len(signal_df)}, robustness={len(R)}")


if __name__ == "__main__":
    main()
