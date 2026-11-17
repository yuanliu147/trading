#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两类特征点的大样本事件研究（预注册见 00_预注册_特征点检验.md）
输出：事件明细、汇总、年度、过滤对比、出场方案、止损后反弹。
"""
import os, pickle, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "..", "2026-10-18_成交量融合_2010-2025", "01_数据", "daily_cache_2016_2026.pkl")
COST = 0.003
START = pd.Timestamp("2016-01-01")
END = pd.Timestamp("2026-09-30")

print("loading cache ...", flush=True)
with open(CACHE, "rb") as f:
    D = pickle.load(f)
stocks = D["stocks"]
zz = D["zz500"].copy()
zz["date"] = pd.to_datetime(zz["date"])
zz = zz.sort_values("date").reset_index(drop=True)
for n in (20, 60):
    zz[f"ma{n}"] = zz["close"].rolling(n).mean()
zz["ret20"] = zz["close"] / zz["close"].shift(20) - 1
zz["vol20"] = zz["close"].pct_change().rolling(20).std() * math.sqrt(252)
zz["mkt_ma20"] = zz["close"] > zz["ma20"]
zz["mkt_ma60"] = zz["close"] > zz["ma60"]
zz_dates = zz["date"].values
zz_close = zz["close"].values
zz_open = zz["open"].values


def board_limit(code, date):
    if code[3:6] in ("300", "301", "688") and date >= pd.Timestamp("2020-08-24"):
        return 0.195
    return 0.095


def detect_events(code, g):
    """返回该股票的事件行（信号日、类型、结构止损、特征）。"""
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < 320:
        return []
    c, h, l, o, v, a = (g[x].astype(float) for x in ("close", "high", "low", "open", "volume", "amount"))
    ret = c.pct_change()
    ma5, ma10, ma20, ma60, ma120 = (c.rolling(n).mean() for n in (5, 10, 20, 60, 120))
    high20p = h.rolling(20).max().shift(1)
    high60p = h.rolling(60).max().shift(1)
    low5 = l.rolling(5).min()
    low10 = l.rolling(10).min()
    vol5, vol20 = v.rolling(5).mean(), v.rolling(20).mean()
    amt20, amt60 = a.rolling(20).mean(), a.rolling(60).mean()
    atr20 = (h - l).rolling(20).mean()
    vol120 = ret.rolling(120).std() * math.sqrt(252)
    range250 = h.rolling(250).max() / l.rolling(250).min() - 1
    act = amt20 / a.rolling(120).mean() - 1

    lim = np.array([board_limit(code, d) for d in g["date"]])
    absret = ret.abs().values
    limday = (absret >= lim * 0.95).astype(float)
    limfrac = pd.Series(limday).rolling(250).mean()
    gap = (o / c.shift(1) - 1).abs() > 0.02
    gapfrac = gap.rolling(250).mean()

    uptrend = (c > ma60) & (ma20 > ma60) & (ma60 > ma60.shift(20))
    dist20 = c / ma20 - 1
    vr = vol5 / vol20

    tb = (uptrend & (c > high60p) & (c.shift(1) <= high60p.shift(1))
          & (dist20 > 0) & (dist20 <= 0.12) & (vr <= 1.5) & (amt20 >= 3e7))
    pbh = high20p / c - 1
    touch = (l <= ma10 * 1.015) | (l <= ma20 * 1.015)
    resume = (c > o) & (c > h.shift(1)) & (c > ma10) & (c > ma5)
    pb = (uptrend & (pbh >= 0.03) & (pbh <= 0.12) & touch & resume
          & (vr <= 0.95) & (amt20 >= 3e7))
    ut = uptrend & ~tb & ~pb & (amt20 >= 3e7)

    def cooldown(mask, gap_n=10):
        idx = np.where(mask.values)[0]
        keep, last = [], -10**9
        for i in idx:
            if i - last > gap_n:
                keep.append(i); last = i
        return keep

    rows = []
    for setup, mask, stop_struct in (
        ("TB", tb, low5 * 0.995),
        ("PB", pb, np.minimum(low5, ma20 * 0.99) * 0.995),
        ("UT", ut, np.minimum(low5, ma20 * 0.99) * 0.995),
    ):
        for i in cooldown(mask):
            if not (START <= g.loc[i, "date"] <= END):
                continue
            rows.append(dict(
                code=code, date=g.loc[i, "date"], setup=setup,
                close=c.iloc[i], ma5=ma5.iloc[i], ma10=ma10.iloc[i],
                ma20=ma20.iloc[i], ma60=ma60.iloc[i],
                atr20=atr20.iloc[i], stop_struct=stop_struct.iloc[i],
                vol5=vol5.iloc[i], vol20=vol20.iloc[i], vr=vr.iloc[i],
                amount20=amt20.iloc[i], amount60=amt60.iloc[i],
                platform20=(h.rolling(20).max() / l.rolling(20).min() - 1).iloc[i],
                ret20=ret.rolling(20).sum().iloc[i] if False else c.iloc[i] / c.iloc[max(0, i - 20)] - 1,
                dist_ma20=dist20.iloc[i],
                pullback=pbh.iloc[i] if setup == "PB" else np.nan,
                vol120=vol120.iloc[i], range250=range250.iloc[i], act=act.iloc[i],
                limfrac=limfrac.iloc[i], gapfrac=gapfrac.iloc[i],
                sig_idx=i,
            ))
    return rows


print("detecting events ...", flush=True)
all_rows = []
for j, (code, g) in enumerate(stocks.items()):
    if len(g) < 320:
        continue
    all_rows.extend(detect_events(code, g))
    if (j + 1) % 200 == 0:
        print("  ", j + 1, "stocks,", len(all_rows), "events", flush=True)
ev = pd.DataFrame(all_rows)
print("events:", ev.shape, ev.setup.value_counts().to_dict(), flush=True)

# 时点成分：快照后下一个交易日生效；用月度快照
snaps = D["snapshots"]
trading_dates = np.sort(pd.to_datetime(zz["date"]).values)
snap_rows = []
for d, codes in snaps.items():
    later = trading_dates[trading_dates > np.datetime64(d)]
    eff = pd.Timestamp(later[0]) if len(later) else pd.Timestamp(d)
    for code in codes:
        snap_rows.append((eff, code))
snap_df = pd.DataFrame(snap_rows, columns=["eff", "code"]).sort_values("eff")
ev = pd.merge_asof(ev.sort_values("date"), snap_df, left_on="date", right_on="eff", by="code", direction="backward")
ev = ev[ev.eff.notna()].copy()
print("after membership:", ev.shape, ev.setup.value_counts().to_dict(), flush=True)

# 入场与路径结果
print("computing outcomes ...", flush=True)
zz_idx = {pd.Timestamp(d): i for i, d in enumerate(zz_dates)}
feat_rows = []
out_rows = []
for j, (code, g) in enumerate(stocks.items()):
    sub = ev[ev.code == code]
    if len(sub) == 0:
        continue
    g = g.sort_values("date").reset_index(drop=True)
    dates = g["date"].values
    pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    o, h, l, c = (g[x].values.astype(float) for x in ("open", "high", "low", "close"))
    v = g["volume"].values.astype(float)
    ma20 = g["close"].rolling(20).mean().values
    for _, r in sub.iterrows():
        si = pos.get(pd.Timestamp(r["date"]))
        if si is None or si + 1 >= len(g):
            continue
        ei = si + 1
        if (g.loc[ei, "date"] - g.loc[si, "date"]).days > 10:
            continue
        if v[ei] <= 0:
            continue
        lim = board_limit(code, g.loc[ei, "date"])
        if o[ei] / c[si] - 1 >= lim - 0.005:
            continue
        entry = o[ei] * 1.001
        atr = float(r["atr20"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        stop = min(max(float(r["stop_struct"]), entry - 3 * atr), entry - 1.0 * atr)
        if not (stop < entry):
            continue
        stop_pct = (entry - stop) / entry
        if stop_pct < 0.015 or stop_pct > 0.12:
            continue
        last = min(len(g) - 1, ei + 40)
        if last < ei + 5:
            continue
        # 基准
        zd = pd.Timestamp(g.loc[ei, "date"])
        zi = zz_idx.get(zd)
        if zi is None:
            continue
        out = dict(idx=len(out_rows), code=code, date=g.loc[si, "date"], setup=r["setup"],
                   entry_date=g.loc[ei, "date"], entry=entry, stop=stop, stop_pct=stop_pct,
                   atr_pct=atr / entry, type=np.nan, s_vol=np.nan, s_heat=np.nan,
                   mkt_ma20=bool(zz.loc[zi, "mkt_ma20"]), mkt_ma60=bool(zz.loc[zi, "mkt_ma60"]),
                   zz_ret20=zz.loc[zi, "ret20"], zz_vol20=zz.loc[zi, "vol20"],
                   vr=r["vr"], platform=r["platform20"], ret20=r["ret20"],
                   dist_ma20=r["dist_ma20"], amount20=r["amount20"])
        for hh in (5, 10, 20, 40):
            if ei + hh <= last:
                ret = c[ei + hh] / entry - 1 - COST
                bret = zz_close[zi + hh] / zz_open[zi] - 1 if zi + hh < len(zz_close) else np.nan
                out[f"ret{hh}"] = ret
                out[f"ex{hh}"] = ret - bret
                out[f"mfe{hh}"] = np.max(h[ei:ei + hh + 1]) / entry - 1
                out[f"mae{hh}"] = np.min(l[ei:ei + hh + 1]) / entry - 1
                # 止损触发（20日内）
                if hh == 20:
                    hit = np.where(l[ei:ei + hh + 1] <= stop)[0]
                    out["stop_hit20"] = len(hit) > 0
                    out["stop_day"] = int(hit[0]) if len(hit) else np.nan
                    out["stop_ret"] = (min(o[ei + hit[0]], stop) / entry - 1 - COST) if len(hit) else np.nan
            else:
                out[f"ret{hh}"] = np.nan
                out[f"ex{hh}"] = np.nan
                out[f"mfe{hh}"] = np.nan
                out[f"mae{hh}"] = np.nan
        # 路径模拟：止损 + 破 MA20，最多 40 日
        px = None; reason = None
        for k in range(ei, last + 1):
            if k == ei:
                if l[k] <= stop:
                    px = min(o[k], stop); reason = "止损"; break
            else:
                if o[k] <= stop:
                    px = o[k]; reason = "止损跳空"; break
                if l[k] <= stop:
                    px = stop; reason = "止损"; break
                if np.isfinite(ma20[k]) and c[k] < ma20[k] and k > ei + 2:
                    px = c[k]; reason = "破MA20"; break
        if px is None:
            px = c[last]; reason = "到期"
        out["exit_px"] = px; out["exit_reason"] = reason
        out["ret_trail"] = px / entry - 1 - COST
        out["R_trail"] = out["ret_trail"] / stop_pct
        out["held"] = (k - ei + 1) if px is not None else (last - ei + 1)

        # 固定止盈 2R / 3R
        for mult in (2.0, 3.0):
            target = entry + mult * (entry - stop)
            px2 = None; why2 = None
            for k in range(ei, last + 1):
                if k == ei:
                    if l[k] <= stop:
                        px2 = min(o[k], stop); why2 = "止损"; break
                else:
                    if o[k] <= stop:
                        px2 = o[k]; why2 = "止损跳空"; break
                    if l[k] <= stop:
                        px2 = stop; why2 = "止损"; break
                    if h[k] >= target:
                        px2 = target; why2 = "止盈"; break
                    if np.isfinite(ma20[k]) and c[k] < ma20[k] and k > ei + 2:
                        px2 = c[k]; why2 = "破MA20"; break
            if px2 is None:
                px2 = c[last]; why2 = "到期"
            out[f"ret_tp{int(mult)}R"] = px2 / entry - 1 - COST
            out[f"R_tp{int(mult)}R"] = out[f"ret_tp{int(mult)}R"] / stop_pct

        # 分批：1/3 或 1/2 在 2R 止盈，剩余按 trail
        for frac, tag in ((1/3, "third"), (0.5, "half")):
            target = entry + 2.0 * (entry - stop)
            part_px = None; stop_px = None; why3 = None
            for k in range(ei, last + 1):
                if k == ei:
                    if l[k] <= stop:
                        stop_px = min(o[k], stop); why3 = "止损"; break
                else:
                    if o[k] <= stop:
                        stop_px = o[k]; why3 = "止损跳空"; break
                    if l[k] <= stop:
                        stop_px = stop; why3 = "止损"; break
                    if part_px is None and h[k] >= target:
                        part_px = target
                    if np.isfinite(ma20[k]) and c[k] < ma20[k] and k > ei + 2:
                        stop_px = c[k]; why3 = "破MA20"; break
            if stop_px is None:
                stop_px = c[last]; why3 = "到期"
            if part_px is not None:
                out[f"ret_{tag}"] = frac * (part_px / entry - 1) + (1 - frac) * (stop_px / entry - 1) - COST
            else:
                out[f"ret_{tag}"] = stop_px / entry - 1 - COST
            out[f"R_{tag}"] = out[f"ret_{tag}"] / stop_pct

        # 保本止损：达到 +1R 后止损上移到 entry
        be_stop = stop; px4 = None; why4 = None
        for k in range(ei, last + 1):
            if k == ei:
                if l[k] <= be_stop:
                    px4 = min(o[k], be_stop); why4 = "止损"; break
            else:
                if o[k] <= be_stop:
                    px4 = o[k]; why4 = "止损跳空"; break
                if l[k] <= be_stop:
                    px4 = be_stop; why4 = "止损"; break
                if h[k] >= entry + (entry - stop):
                    be_stop = max(be_stop, entry)
                if np.isfinite(ma20[k]) and c[k] < ma20[k] and k > ei + 2:
                    px4 = c[k]; why4 = "破MA20"; break
        if px4 is None:
            px4 = c[last]; why4 = "到期"
        out["ret_be"] = px4 / entry - 1 - COST
        out["R_be"] = out["ret_be"] / stop_pct

        out_rows.append(out)
    if (j + 1) % 200 == 0:
        print("  outcomes", j + 1, len(out_rows), flush=True)

res = pd.DataFrame(out_rows)
res.to_pickle(os.path.join(BASE, "特征点_事件结果_未分类.pkl"))
print("valid outcomes:", res.shape, flush=True)

# 股性标签：在事件日横截面上重算
print("computing stock type ...", flush=True)
feat_rows = []
for code, g in stocks.items():
    if len(g) < 320:
        continue
    g = g.sort_values("date").reset_index(drop=True)
    m = (g["date"] >= pd.Timestamp("2015-06-01")) & (g["date"] <= END)
    if not m.any():
        continue
    gg = g.loc[m].copy()
    c = g["close"]; h = g["high"]; l = g["low"]; o = g["open"]; a = g["amount"]
    ret = c.pct_change()
    vol120 = (ret.rolling(120).std() * math.sqrt(252)).loc[m]
    range250 = (h.rolling(250).max() / l.rolling(250).min() - 1).loc[m]
    act = (a.rolling(20).mean() / a.rolling(120).mean() - 1).loc[m]
    lim_all = np.array([board_limit(code, d) for d in g["date"]])
    limfrac = pd.Series((ret.abs().values >= lim_all * 0.95).astype(float), index=g.index).rolling(250).mean().loc[m]
    gapfrac = ((o / c.shift(1) - 1).abs() > 0.02).rolling(250).mean().loc[m]
    amt60 = a.rolling(60).mean().loc[m]
    feat_rows.append(pd.DataFrame({
        "date": gg["date"].values, "code": code,
        "vol120": vol120.values, "range250": range250.values, "act": act.values,
        "limfrac": limfrac.values, "gapfrac": gapfrac.values, "amount60": amt60.values,
    }))
feat = pd.concat(feat_rows, ignore_index=True)
feat = feat[(feat.amount60 >= 2e7) & feat.vol120.notna() & feat.range250.notna() & feat.act.notna()]
edates = set(res.date.unique())
feat = feat[feat.date.isin(edates)].copy()
print("feature rows on event dates:", len(feat), flush=True)
for col in ("vol120", "range250", "act", "limfrac", "gapfrac"):
    feat["r_" + col] = feat.groupby("date")[col].rank(pct=True)
feat["s_vol"] = feat[["r_vol120", "r_range250"]].mean(axis=1)
feat["s_heat"] = feat[["r_act", "r_limfrac", "r_gapfrac"]].mean(axis=1)
feat["type"] = np.select(
    [(feat.s_vol >= .5) & (feat.s_heat >= .5), (feat.s_vol >= .5) & (feat.s_heat < .5),
     (feat.s_vol < .5) & (feat.s_heat >= .5)],
    ["高波活跃型", "高波低量型", "低波活跃型"], default="低波低量型")
res = res.drop(columns=[c for c in ("type", "s_vol", "s_heat") if c in res.columns])
res = res.merge(feat[["date", "code", "s_vol", "s_heat", "type"]], on=["date", "code"], how="left")
print("type missing:", res.type.isna().sum(), flush=True)

# 时期与分组
res["period"] = np.where(res.date <= pd.Timestamp("2021-12-31"), "2016-2021", "2022-2026")
res["quiet"] = ((res.vr <= 0.8) & (res.platform <= 0.18) & (res.ret20 <= 0.20))
res["mkt"] = np.select([res.mkt_ma20 & res.mkt_ma60, res.mkt_ma20 & ~res.mkt_ma60],
                       ["MA20上/MA60上", "MA20上/MA60下"], default="MA20下")
res.to_csv(os.path.join(BASE, "特征点_事件明细.csv"), index=False, encoding="utf-8-sig")
print("saved detail", flush=True)

# 汇总函数
def agg(df):
    return pd.Series({
        "n": len(df),
        "胜率20": (df.ret20 > 0).mean(),
        "20日均值": df.ret20.mean(),
        "20日中位": df.ret20.median(),
        "20日超额": df.ex20.mean(),
        "20日超额中位": df.ex20.median(),
        "MAE20": df.mae20.mean(),
        "MFE20": df.mfe20.mean(),
        "止损率20": df.stop_hit20.mean(),
        "trail均值": df.ret_trail.mean(),
        "trail中位": df.ret_trail.median(),
        "trail胜率": (df.ret_trail > 0).mean(),
        "平均R": df.R_trail.mean(),
        "平均持有": df.held.mean(),
    })

rows = []
for (setup, period), g in res.groupby(["setup", "period"]):
    rows.append({"setup": setup, "period": period, "group": "全部", **agg(g)})
for (setup, period, typ), g in res.groupby(["setup", "period", "type"]):
    rows.append({"setup": setup, "period": period, "group": typ, **agg(g)})
for (setup, period, q), g in res.groupby(["setup", "period", "quiet"]):
    rows.append({"setup": setup, "period": period, "group": "安静过滤" if q else "未过滤", **agg(g)})
for (setup, period, m), g in res.groupby(["setup", "period", "mkt"]):
    rows.append({"setup": setup, "period": period, "group": m, **agg(g)})
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(BASE, "特征点_汇总.csv"), index=False, encoding="utf-8-sig")

# 年度
yearly = []
for (setup, y), g in res.groupby(["setup", res.date.dt.year]):
    yearly.append({"setup": setup, "year": y, "n": len(g), "win20": (g.ret20 > 0).mean(),
                   "ret20": g.ret20.mean(), "ex20": g.ex20.mean(), "mae20": g.mae20.mean(),
                   "stop_rate": g.stop_hit20.mean()})
pd.DataFrame(yearly).to_csv(os.path.join(BASE, "特征点_年度.csv"), index=False, encoding="utf-8-sig")

# 过滤对比
frows = []
for (setup, period, q), g in res.groupby(["setup", "period", "quiet"]):
    frows.append({"setup": setup, "period": period, "quiet": bool(q), "n": len(g),
                  "win20": (g.ret20 > 0).mean(), "ret20": g.ret20.mean(), "ex20": g.ex20.mean(),
                  "mae20": g.mae20.mean(), "stop_rate": g.stop_hit20.mean(),
                  "trail": g.ret_trail.mean(), "avgR": g.R_trail.mean()})
pd.DataFrame(frows).to_csv(os.path.join(BASE, "特征点_过滤对比.csv"), index=False, encoding="utf-8-sig")

# 出场方案
erows = []
for (setup, period), g in res.groupby(["setup", "period"]):
    row = {"setup": setup, "period": period, "n": len(g)}
    for col, name in (("ret_trail", "止损+破MA20"), ("ret_tp2R", "固定2R"),
                      ("ret_tp3R", "固定3R"), ("ret_third", "1/3在2R+余仓trail"),
                      ("ret_half", "1/2在2R+余仓trail"), ("ret_be", "1R后保本+trail")):
        row[name + "_均值"] = g[col].mean()
        row[name + "_胜率"] = (g[col] > 0).mean()
        row[name + "_中位"] = g[col].median()
    row["trail平均R"] = g.R_trail.mean()
    erows.append(row)
pd.DataFrame(erows).to_csv(os.path.join(BASE, "出场方案对比.csv"), index=False, encoding="utf-8-sig")
print("done", flush=True)
