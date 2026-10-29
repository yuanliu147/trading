#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""候选·风险预算：把市场状态从"硬门/择时"改造成仓位/单票风险/持仓数预算。

预注册：03_结果/候选_风险预算/00_预注册_风险预算.md（在任何叠加收益之前写定）

口径要点
--------
* 收益一律用冻结层 `factor`（= Π(1+pct_chg)，总收益指数），**不用** `hfq_*` 列：
  实测 `hfq_close` 日比值 ≈ 2×官方涨跌幅（`hfq = raw × factor`，factor 已含当日收益）。
  `--stage audit` 会把这一缺陷量化落盘。
* 隔夜/日内：ovn = open/preclose − 1，idt = close/open − 1，乘积恰为 pct_chg。
* 状态 t 日收盘定、t+1 开盘执行 → 日内收益用 e[t−1]，隔夜用 e[t−2]，换手成本
  |e[t−1]−e[t−2]| 在 t 日开盘计。比常见 `pos.shift(1)` 更保守一天。
* 窗口：开发 2008-2015、验证 2016-2020；2021+ 一律不读。

用法
----
  python3 候选_风险预算.py --stage audit     # 冻结层口径审计 + 状态可复现性
  python3 候选_风险预算.py --stage base      # 构建三个载体（C1/C2/C3）
  python3 候选_风险预算.py --stage stateinfo # H1–H4：状态到底预测什么
  python3 候选_风险预算.py --stage overlay   # 59 配置：阶梯/延迟/成本/对照/归因/空分布
  python3 候选_风险预算.py --stage check     # 7 项自检（恒等/截断/对齐/映射/冻结日）
  python3 候选_风险预算.py --stage account   # 账户级机制验证（2016-2020，公共 Account）
  python3 候选_风险预算.py --stage report    # 判定表/关键对照/年度拆解/净值
  python3 候选_风险预算.py --stage all       # 全流程，约 110 秒
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "02_代码"
sys.path.insert(0, str(CODE))
DB = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
OUT = ROOT / "03_结果" / "候选_风险预算"
CACHE = OUT / "cache"
STATE_CSV = ROOT / "03_结果" / "市场状态" / "市场状态_日频许可_2006_2020.csv"

FREEZE = "2026-10-26"
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
FULL = ("2008-01-01", "2020-12-31")
TD = 243.0
FEE_BP = 10.0
LADDERS = (0.25, 0.50, 0.75, 1.00)
PRIMARY_M = 0.50


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(df: pd.DataFrame, name: str) -> Path:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"写出 {p.relative_to(ROOT)}  rows={len(df)}")
    return p


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB), read_only=True)


# =====================================================================
# 0. 口径审计 + 状态复现
# =====================================================================
def load_state() -> pd.DataFrame:
    st = pd.read_csv(STATE_CSV, usecols=["date", "state", "B1", "B2", "B3", "B4", "B5", "P",
                                         "q70", "q55", "q45", "q30"])
    st["date"] = pd.to_datetime(st["date"])
    return st.sort_values("date").reset_index(drop=True)


def _rz(s: pd.Series, win: int = 252, minp: int = 120) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean().shift(1)
    sd = s.rolling(win, min_periods=minp).std().shift(1)
    return ((s - m) / sd).clip(-3, 3)


def rebuild_state() -> pd.DataFrame:
    """用冻结层独立重建第一阶段状态（公式照 市场状态_研究规范.md §3–4.1）。"""
    con = connect()
    q = """
    WITH f AS (
      SELECT symbol, date, pct_chg/100.0 AS ret, factor AS px, amount, turn, is_st,
             CASE WHEN symbol LIKE '300%' OR symbol LIKE '301%' THEN 'cyb'
                  WHEN symbol LIKE '688%' OR symbol LIKE '689%' THEN 'star' ELSE 'main' END AS brd,
             row_number() OVER (PARTITION BY symbol ORDER BY date) AS bar_no
      FROM daily WHERE date BETWEEN DATE '2003-10-01' AND DATE '2020-12-31'
    ), w AS (
      SELECT *,
        avg(px) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
        avg(px) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
        avg(px) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW) AS ma120,
        max(px) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS hi250,
        min(px) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) AS lo250,
        avg(amount) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amt20,
        count(*) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS bars60,
        lag(px,20) OVER (PARTITION BY symbol ORDER BY date) AS c20,
        lag(px,60) OVER (PARTITION BY symbol ORDER BY date) AS c60
      FROM f
    ), e AS (
      SELECT * FROM w
      WHERE is_st = 0 AND bar_no >= 60 AND bars60 >= 50 AND amt20 > 0 AND px > 0 AND ret IS NOT NULL
        AND date >= DATE '2003-12-30'
    )
    SELECT date,
      count(*) AS n_univ,
      avg(CASE WHEN ret > 0 THEN 1.0 ELSE 0.0 END) AS pct_up,
      avg(CASE WHEN px > ma20 THEN 1.0 ELSE 0.0 END) AS pct_ma20,
      avg(CASE WHEN px > ma60 THEN 1.0 ELSE 0.0 END) AS pct_ma60,
      avg(CASE WHEN px > ma120 THEN 1.0 ELSE 0.0 END) AS pct_ma120,
      avg(CASE WHEN px >= hi250*0.999 THEN 1.0 ELSE 0.0 END) AS pct_nh250,
      avg(CASE WHEN px <= lo250*1.001 THEN 1.0 ELSE 0.0 END) AS pct_nl250,
      median(px/c20 - 1) AS med_r20,
      median(px/c60 - 1) AS med_r60,
      stddev_samp(px/c20 - 1) AS disp_r20,
      sum(amount) AS amt_total,
      median(turn) AS med_turn,
      avg(CASE WHEN ret <= -CASE WHEN brd='cyb' AND date>=DATE '2020-08-24' THEN 0.20
                                WHEN brd='star' THEN 0.20 ELSE 0.10 END * 0.98
               THEN 1.0 ELSE 0.0 END) AS pct_limit_dn
    FROM e GROUP BY date ORDER BY date
    """
    mkt = con.execute(q).df()
    idx = con.execute("""
      SELECT symbol, date, close, pct_chg/100.0 AS ret1
      FROM index_daily WHERE symbol IN ('000300.SH','000905.SH')
        AND date BETWEEN DATE '2003-10-01' AND DATE '2020-12-31' ORDER BY symbol, date
    """).df()
    con.close()

    mkt["date"] = pd.to_datetime(mkt["date"])
    idx["date"] = pd.to_datetime(idx["date"])
    feats = {}
    for sym, tag in (("000300.SH", "hs300"), ("000905.SH", "csi500")):
        g = idx[idx.symbol == sym].sort_values("date").set_index("date")
        c, r = g["close"].astype(float), g["ret1"].astype(float)
        feats[f"{tag}_close"] = c
        for w in (20, 60, 120, 250):
            feats[f"{tag}_ma{w}"] = c.rolling(w, min_periods=max(5, w // 2)).mean()
        feats[f"{tag}_vol20"] = r.rolling(20, min_periods=10).std() * np.sqrt(TD)
        feats[f"{tag}_r60"] = c / c.shift(60) - 1
    iw = pd.DataFrame(feats).reindex(mkt["date"]).ffill(limit=5)
    d = pd.concat([mkt.set_index("date"), iw], axis=1).reset_index()

    d["amt_ratio"] = d["amt_total"] / d["amt_total"].rolling(60, min_periods=30).mean()
    d["pct_up_ema20"] = d["pct_up"].ewm(span=20, adjust=False).mean()
    d["nh_nl"] = d["pct_nh250"] - d["pct_nl250"]

    d["B1"] = pd.concat([_rz(d["pct_up_ema20"]), _rz(d["pct_ma60"]), _rz(d["nh_nl"])],
                        axis=1).mean(axis=1, skipna=True)

    def trend(close, ma20, ma60, ma120):
        s = 0.4 * np.sign(close - ma120) + 0.3 * np.sign(ma20 - ma60) + 0.3 * np.sign(ma60 - ma120)
        return s.where(ma120.notna())

    t300 = trend(d["hs300_close"], d["hs300_ma20"], d["hs300_ma60"], d["hs300_ma120"])
    t500 = trend(d["csi500_close"], d["csi500_ma20"], d["csi500_ma60"], d["csi500_ma120"])
    d["B2"] = pd.concat([t300, t500], axis=1).mean(axis=1, skipna=True)
    d["B3"] = pd.concat([_rz(d["hs300_vol20"]), _rz(d["disp_r20"]), _rz(d["pct_limit_dn"])],
                        axis=1).mean(axis=1, skipna=True)
    d["B4"] = pd.concat([_rz(d["amt_ratio"]), _rz(d["med_turn"])], axis=1).mean(axis=1, skipna=True)
    d["B5"] = pd.concat([_rz(d["med_r60"]), _rz(d["med_r20"])], axis=1).mean(axis=1, skipna=True)
    d["P"] = 0.30 * d["B1"] + 0.30 * d["B2"] + 0.20 * d["B5"] - 0.20 * d["B3"]

    for tag, qq in (("q70", .70), ("q55", .55), ("q45", .45), ("q30", .30)):
        d[tag] = d["P"].expanding(min_periods=250).quantile(qq)
    p = d["P"].to_numpy()
    q70, q55, q45, q30 = (d[k].to_numpy() for k in ("q70", "q55", "q45", "q30"))
    st = np.empty(len(d), dtype=object)
    cur = "na"
    for i in range(len(d)):
        if np.isnan(p[i]) or np.isnan(q70[i]):
            st[i] = "na"
            cur = "na"
            continue
        if cur == "na":
            cur = "risk_on" if p[i] >= q70[i] else ("risk_off" if p[i] <= q30[i] else "neutral")
        elif cur == "risk_on":
            cur = "risk_on" if p[i] >= q55[i] else ("neutral" if p[i] > q30[i] else "risk_off")
        elif cur == "risk_off":
            cur = "risk_off" if p[i] <= q45[i] else ("neutral" if p[i] < q70[i] else "risk_on")
        else:
            cur = "risk_on" if p[i] >= q70[i] else ("risk_off" if p[i] <= q30[i] else "neutral")
        st[i] = cur
    d["state"] = pd.Series(st, dtype=object).astype(str)
    return d[["date", "state", "B1", "B2", "B3", "B4", "B5", "P",
              "hs300_vol20", "med_r20", "med_r60", "disp_r20"]]


def stage_audit() -> None:
    con = connect()
    q = """
    select symbol, date, pct_chg/100.0 as pct, factor,
           factor/lag(factor) over (partition by symbol order by date) - 1 as fac_ret,
           hfq_close/lag(hfq_close) over (partition by symbol order by date) - 1 as hfq_ret
    from daily where date between date '2008-01-01' and date '2020-12-31'
    """
    d = con.execute(q).df().dropna(subset=["pct", "fac_ret", "hfq_ret"])
    con.close()

    m = d[d.pct.abs() > 0.01]
    d["y"] = pd.to_datetime(d["date"]).dt.year
    g = (d.groupby(["symbol", "y"])
          .agg(fac=("fac_ret", lambda x: float(np.prod(1 + x)) - 1),
               hfq=("hfq_ret", lambda x: float(np.prod(1 + x)) - 1))
          .reset_index())
    sel = g[g.fac.abs() > 0.05]
    users = []
    for p in sorted(CODE.glob("*.py")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "hfq_close" in txt or "hfq_open" in txt or 'adjust="hfq"' in txt or "adjust='hfq'" in txt:
            if p.name != Path(__file__).name:
                users.append(p.name)

    aud = pd.DataFrame([
        dict(check="rows_2008_2020", value=len(d), note="相邻行"),
        dict(check="factor收益-官方pct_最大绝对差", value=float((d.fac_ret - d.pct).abs().max()), note="应≈0"),
        dict(check="factor收益偏差>0.5bp占比", value=float(((d.fac_ret - d.pct).abs() > 5e-5).mean()), note="应=0"),
        dict(check="hfq收益-官方pct_最大绝对差", value=float((d.hfq_ret - d.pct).abs().max()), note="应≈0"),
        dict(check="hfq收益偏差>0.5bp占比", value=float(((d.hfq_ret - d.pct).abs() > 5e-5).mean()), note="应=0"),
        dict(check="hfq收益/pct_中位比", value=float((m.hfq_ret / m.pct).median()), note="1.0正常；≈2=收益记两遍"),
        dict(check="证券年总收益_hfq/factor_中位比", value=float((sel.hfq / sel.fac).median()), note="1.0正常"),
        dict(check="引用hfq列的脚本数", value=len(users), note=", ".join(users)),
    ])
    save(aud, "00_口径审计.csv")
    log(aud.to_string(index=False))

    st_reb = rebuild_state()
    st_reb.to_parquet(CACHE / "state_rebuilt.parquet", index=False)
    pub = load_state()
    cmp = pub[["date", "state", "P", "B1", "B2", "B3", "B5"]].merge(
        st_reb[["date", "state", "P", "B1", "B2", "B3", "B5"]], on="date", suffixes=("_pub", "_reb"))
    cmp = cmp[cmp.date >= "2008-01-01"].dropna(subset=["P_pub", "P_reb"])
    agree = float((cmp.state_pub == cmp.state_reb).mean())
    corr = float(np.corrcoef(cmp.P_pub, cmp.P_reb)[0, 1])
    trans_pub = int((cmp.state_pub != cmp.state_pub.shift(1)).sum() - 1)
    trans_reb = int((cmp.state_reb != cmp.state_reb.shift(1)).sum() - 1)
    dist = st_reb[st_reb.date >= "2008-01-01"].state.value_counts().to_dict()

    lines = [
        "# 状态与数据口径复核（冻结层独立重建）",
        "",
        f"窗口 2008-2020，可与第一阶段状态对齐的交易日 **{len(cmp)}**。",
        "",
        "## 1. 状态可复现性",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 状态逐日一致率 | **{agree:.4f}** |",
        f"| P 的相关系数 | **{corr:.4f}** |",
        f"| 状态切换次数（公开 / 重建） | {trans_pub} / {trans_reb} |",
        f"| 状态分布（重建） | {dist} |",
        "",
        "## 2. 冻结层 `hfq_*` 列存在收益倍数缺陷（重大）",
        "",
        "`02_代码/构建冻结数据.py` 中 `factor = exp(Σ ln(1+ret))`（ret=pct_chg/100），",
        "即 factor 本身已经是**累计总收益指数**；随后又定义 `hfq = raw × factor`。",
        "由于 factor 已含当日收益，`hfq_close` 的相邻比值 ≈ (1+收益)²，把收益记了两遍。",
        "",
        f"- `hfq_close` 比值与官方 `pct_chg` 的偏差，在 **{((d.hfq_ret - d.pct).abs() > 5e-5).mean():.2%}** 的行上 >0.5bp；",
        f"- |涨跌|>1% 的行上，hfq 收益/官方收益的中位比 = **{(m.hfq_ret / m.pct).median():.3f}**（应为 1.0）；",
        f"- 证券-年总收益的中位比 = **{(sel.hfq / sel.fac).median():.3f}**。",
        "",
        "正确口径：`factor` 的相邻比值与官方涨跌幅的最大绝对差 = "
        f"**{(d.fac_ret - d.pct).abs().max():.2e}**，0 行超过 0.5bp。",
        "",
        "**本模块所有收益只用 `factor`/`pct_chg`；`hfq_*` 一律不用。**",
        "仍在引用 `hfq_*` 或 `DataLayer.daily(adjust='hfq')` 的脚本：",
        f"{', '.join(users) if users else '（无）'}。这些脚本的收益类结果需要重跑核对。",
    ]
    (OUT / "01_状态复核.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"状态一致率 {agree:.4f}  P相关 {corr:.4f}")


# =====================================================================
# 1. 主板面板
# =====================================================================
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    symbols: np.ndarray
    ret: np.ndarray
    ovn: np.ndarray
    idt: np.ndarray
    close: np.ndarray
    susp: np.ndarray
    st: np.ndarray
    tds: np.ndarray
    limup_open: np.ndarray
    limdn_open: np.ndarray
    avg_amt20: np.ndarray
    delist_soon: np.ndarray

    def idx(self, d) -> int:
        return int(self.dates.get_loc(pd.Timestamp(d)))


def load_panel(start: str = "2007-01-01", end: str = "2020-12-31") -> Panel:
    con = connect()
    df = con.execute(f"""
        SELECT symbol, date, close, open, preclose, amount, factor, is_suspended, is_st,
               td_since_list, is_limit_up_open, is_limit_down_open, delist_date
        FROM daily
        WHERE date BETWEEN DATE '{start}' AND DATE '{end}' AND board = '主板'
        ORDER BY date, symbol
    """).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df["delist_date"] = pd.to_datetime(df["delist_date"])

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    symbols = np.array(sorted(df["symbol"].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    si = pd.Series(np.arange(len(symbols)), index=symbols)
    r = di.reindex(df["date"]).to_numpy()
    c = si.reindex(df["symbol"]).to_numpy()
    shape = (len(dates), len(symbols))

    def mat(col, dtype="float32", fill=np.nan):
        m = np.full(shape, fill, dtype=dtype)
        vals = df[col]
        if str(dtype) == "bool":
            vals = vals.fillna(False).astype(bool)
        m[r, c] = vals.to_numpy()
        return m

    tr = mat("factor", "float64")
    ret = np.full(shape, np.nan, dtype="float64")
    ret[1:] = tr[1:] / tr[:-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan

    ovn = mat("open", "float64") / mat("preclose", "float64") - 1.0
    idt = mat("close", "float64") / mat("open", "float64") - 1.0
    ovn[~np.isfinite(ovn)] = np.nan
    idt[~np.isfinite(idt)] = np.nan

    amount = mat("amount", "float64")
    avg_amt20 = pd.DataFrame(amount, index=dates, columns=symbols) \
        .rolling(20, min_periods=15).mean().to_numpy()

    dd = df.groupby("symbol")["delist_date"].max()
    delist_soon = np.zeros(shape, dtype=bool)
    for j, sym in enumerate(symbols):
        dv = dd.get(sym)
        if pd.notna(dv):
            delist_soon[:, j] = dates >= (dv - pd.Timedelta(days=45))

    log(f"面板 {len(dates)} 日 × {len(symbols)} 只（主板）")
    return Panel(dates=dates, symbols=symbols,
                 ret=ret.astype(np.float32), ovn=ovn.astype(np.float32), idt=idt.astype(np.float32),
                 close=mat("close", "float32"), susp=mat("is_suspended", "bool", False),
                 st=mat("is_st", "bool", False), tds=mat("td_since_list", "float32", -1.0),
                 limup_open=mat("is_limit_up_open", "bool", False),
                 limdn_open=mat("is_limit_down_open", "bool", False),
                 avg_amt20=avg_amt20, delist_soon=delist_soon)


def load_index(symbol: str = "000905.SH") -> pd.DataFrame:
    con = connect()
    d = con.execute(f"""
        SELECT date, open, close, preclose FROM index_daily
        WHERE symbol='{symbol}' AND date BETWEEN DATE '2007-01-01' AND DATE '2020-12-31'
        ORDER BY date""").df()
    con.close()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date")
    d["ovn"] = d["open"] / d["preclose"] - 1.0
    d["idt"] = d["close"] / d["open"] - 1.0
    d["ret"] = d["close"] / d["preclose"] - 1.0
    return d


# =====================================================================
# 2. 载体
# =====================================================================
_R20: dict = {}


def prepare_lookbacks(P: Panel) -> None:
    r = np.nan_to_num(P.ret, nan=0.0).astype(np.float64)
    tr = np.cumprod(1.0 + r, axis=0)
    r20 = np.full_like(tr, np.nan)
    r20[20:] = tr[20:] / tr[:-20] - 1.0
    vol60 = pd.DataFrame(r).rolling(60, min_periods=40).std().to_numpy() * np.sqrt(TD)
    _R20["r20"] = r20
    _R20["vol60"] = vol60


def eligible(P: Panel, i: int) -> np.ndarray:
    ok = ((P.tds[i] >= 120) & (~P.susp[i]) & (~P.st[i]) & (P.close[i] >= 2.0)
          & (P.avg_amt20[i] >= 3e7) & (~P.delist_soon[i]))
    ok &= np.isfinite(P.ret[i])
    if i > 0:  # 预注册：复牌首日不新开仓
        ok &= ~((~P.susp[i]) & P.susp[i - 1])
    return ok


def _target_weights(P: Panel, i: int, plan: dict, st_arr, npos_state, risk_w,
                    targets: list) -> np.ndarray:
    """t 日收盘（只用 ≤t 数据）算目标权重。"""
    n_s = P.ret.shape[1]
    ok = eligible(P, i)
    w_tgt = np.zeros(n_s)
    if plan["kind"] == "equal":
        sel = np.flatnonzero(ok)
        if len(sel) > 0:
            w_tgt[sel] = 1.0 / len(sel)
        return w_tgt
    n_hold = int(plan.get("n", 10))
    if npos_state is not None and st_arr is not None:
        n_hold = int(npos_state.get(st_arr[i], n_hold))
    r20 = _R20["r20"][i]
    cand = ok & np.isfinite(r20)
    idx = np.flatnonzero(cand)
    if len(idx) == 0:
        return w_tgt
    order = idx[np.argsort(r20[idx])]
    pick = order[:n_hold]
    if risk_w and st_arr is not None:
        vol = _R20["vol60"][i]
        inv = 1.0 / np.clip(vol[pick], 0.005, 5.0)
        w = inv / inv.sum()
        w = np.clip(w, 0.5 / len(pick), 1.5 / len(pick))
        w_tgt[pick] = w / w.sum()
    else:
        w_tgt[pick] = 1.0 / len(pick)
    rec = dict(date=P.dates[i], n=len(pick), state=st_arr[i] if st_arr is not None else "na")
    for k, j in enumerate(pick):
        rec[f"s{k}"] = P.symbols[j]
    targets.append(rec)
    return w_tgt


def run_carrier(P: Panel, plan: dict, fee_bp: float = FEE_BP) -> dict:
    """载体模拟（e=1）。

    严格时序：t 日收盘用 ≤t 数据算目标权重 → t+1 **开盘**成交 → t+1 日内收益属于新权重。
    当日隔夜收益属于前一日持有的权重。t+1 停牌 / 开盘涨停不买，开盘跌停不卖。
    """
    n_d, n_s = P.ret.shape
    dates = P.dates
    if plan["kind"] == "equal":
        ym = (dates.year * 100 + dates.month)
        reb = [i - 1 for i in range(1, n_d) if ym[i] != ym[i - 1] and i - 1 > 0]
    else:
        cyc = int(plan.get("cycle", 20))
        reb = list(range(int(plan.get("start_i", 0)), n_d - 1, cyc))
    reb_set = set(reb)
    st_s = plan.get("state_series")
    st_arr = None if st_s is None else np.asarray(st_s.to_numpy(), dtype=object)
    npos_state = plan.get("npos_state")
    risk_w = plan.get("risk_weight", False)

    h = np.zeros(n_s)
    cash = 1.0
    nav = 1.0
    rec = {k: np.zeros(n_d) for k in ("ovn", "idt", "cost", "invested", "npos", "turnover")}
    rec["nav"] = np.zeros(n_d)
    rec["nav"][0] = 1.0
    targets = []
    pending = None

    for i in range(n_d):
        # 1) 隔夜（旧权重）
        if i > 0:
            h = h * (1.0 + np.nan_to_num(P.ovn[i], nan=0.0))
        v_open = cash + h.sum()
        ovn_p = (v_open / nav - 1.0) if (i > 0 and nav > 0) else 0.0

        # 2) 开盘成交（昨日收盘决定的权重）
        cost_frac = 0.0
        if pending is not None and v_open > 0:
            resume_i = P.susp[i - 1] if i > 0 else np.zeros(n_s, dtype=bool)
            can_buy = (~P.susp[i]) & (~P.limup_open[i]) & (~resume_i)
            can_sell = (~P.susp[i]) & (~P.limdn_open[i])
            tgt_val = pending * v_open
            sell_mask = (tgt_val < h) & can_sell
            sold = np.where(sell_mask, h - tgt_val, 0.0)
            h = np.where(sell_mask, tgt_val, h)
            cash += sold.sum()
            buy_mask = (tgt_val > h) & can_buy
            want = np.where(buy_mask, tgt_val - h, 0.0)
            scale = 1.0 if want.sum() <= cash else (cash / want.sum() if want.sum() > 0 else 0.0)
            bought = want * scale
            h = h + bought
            cash -= bought.sum()
            notional = sold.sum() + bought.sum()
            cost_frac = fee_bp / 1e4 * notional / v_open
            cash -= cost_frac * v_open
            rec["turnover"][i] = notional / v_open
            pending = None

        # 3) 日内（新权重）
        h = h * (1.0 + np.nan_to_num(P.idt[i], nan=0.0))
        v_close = cash + h.sum()
        idt_p = (v_close / (v_open * (1 - cost_frac)) - 1.0) if v_open > 0 else 0.0
        nav = v_close
        rec["ovn"][i] = ovn_p
        rec["idt"][i] = idt_p
        rec["cost"][i] = cost_frac
        rec["nav"][i] = nav
        rec["invested"][i] = (h.sum() / nav) if nav > 0 else 0.0
        rec["npos"][i] = int((h > 1e-9).sum())

        # 4) 收盘决策，次日开盘执行
        if i in reb_set:
            pending = _target_weights(P, i, plan, st_arr, npos_state, risk_w, targets)

    out = {k: pd.Series(v, index=dates) for k, v in rec.items()}
    out["dates"] = dates
    out["targets"] = targets
    return out


def run_index(d: pd.DataFrame) -> dict:
    rec = {k: pd.Series(d[k].to_numpy(), index=d.index) for k in ("ovn", "idt", "ret")}
    rec["nav"] = pd.Series(np.cumprod(1.0 + d["ret"].to_numpy()), index=d.index)
    rec["cost"] = pd.Series(np.zeros(len(d)), index=d.index)
    rec["invested"] = pd.Series(np.ones(len(d)), index=d.index)
    rec["npos"] = pd.Series(np.ones(len(d)), index=d.index)
    rec["turnover"] = pd.Series(np.zeros(len(d)), index=d.index)
    rec["dates"] = pd.DatetimeIndex(d.index)
    rec["targets"] = []
    return rec


def stage_base() -> None:
    P = load_panel()
    prepare_lookbacks(P)
    log("C1 主宇宙等权月度再平衡 …")
    c1 = run_carrier(P, dict(kind="equal"))
    log("C2 中证500 …")
    c2 = run_index(load_index("000905.SH"))
    log("C3 20日反转 10 只 …")
    start_i = int(np.argmax(P.dates >= pd.Timestamp("2007-12-20")))
    c3 = run_carrier(P, dict(kind="reversal", n=10, cycle=20, start_i=start_i))

    for tag, b in (("C1", c1), ("C2", c2), ("C3", c3)):
        df = pd.DataFrame({k: v for k, v in b.items() if k not in ("dates", "targets")},
                          index=b["dates"])
        df.to_parquet(CACHE / f"base_{tag}.parquet")
        log(f"{tag}: 末值={b['nav'].iloc[-1]:.3f} 平均持仓={b['npos'].mean():.1f} "
            f"平均投入={b['invested'].mean():.3f} 年换手={b['turnover'].mean()*TD:.2f}")
    pd.DataFrame(c3["targets"]).to_parquet(CACHE / "targets_C3.parquet", index=False)
    st = load_state().set_index("date")["state"]
    st.to_frame().to_parquet(CACHE / "state_pub.parquet")
    _panel_cache(P)


def _panel_cache(P: Panel) -> None:
    np.savez_compressed(CACHE / "panel_meta.npz",
                        dates=P.dates.values.astype("datetime64[D]").astype(str),
                        symbols=P.symbols)


# =====================================================================
# 3. 状态的信息内容（H1–H4）
# =====================================================================
def _spearman(a, b) -> float:
    x = pd.Series(a).rank()
    y = pd.Series(b).rank()
    m = x.notna() & y.notna()
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _partial_corr(x, y, z) -> float:
    """x 与 y 在控制 z 后的偏相关（rank 残差法）。"""
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 100:
        return np.nan
    for c in ("x", "y", "z"):
        d[c] = d[c].rank()
    Z = np.column_stack([np.ones(len(d)), d["z"].to_numpy()])
    rx = d["x"].to_numpy() - Z @ np.linalg.lstsq(Z, d["x"].to_numpy(), rcond=None)[0]
    ry = d["y"].to_numpy() - Z @ np.linalg.lstsq(Z, d["y"].to_numpy(), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def stage_stateinfo() -> pd.DataFrame:
    """状态到底预测什么：波动 / 方向 / 上行 / 下行，以及控制当期波动后的增量。"""
    b = pd.read_parquet(CACHE / "base_C1.parquet")
    b.index = pd.to_datetime(b.index)
    r = ((1 + b["ovn"]) * (1 - b["cost"]) * (1 + b["idt"]) - 1).rename("r")
    st = load_state().set_index("date")

    df = pd.DataFrame({"r": r})
    df["fwd20_ret"] = df["r"].rolling(20).sum().shift(-20)
    df["fwd20_vol"] = df["r"].rolling(20).std().shift(-20) * np.sqrt(TD)
    df["fwd20_up"] = df["r"].clip(lower=0).rolling(20).std().shift(-20) * np.sqrt(TD)
    df["fwd20_dn"] = df["r"].clip(upper=0).rolling(20).std().shift(-20) * np.sqrt(TD)
    df["fwd20_mdd"] = (df["r"] + 1).rolling(20).apply(
        lambda x: float((np.cumprod(x) / np.maximum.accumulate(np.cumprod(x)) - 1).min())) .shift(-20)
    df["vol20"] = df["r"].rolling(20).std() * np.sqrt(TD)
    df["vol60"] = df["r"].rolling(60).std() * np.sqrt(TD)
    df = df.join(st[["state", "B3", "P"]], how="left")

    rows = []
    for wname, (a, z) in (("dev", DEV), ("val", VAL), ("full", FULL)):
        d = df.loc[a:z].dropna(subset=["state", "fwd20_vol", "fwd20_ret"])
        off = (d["state"] == "risk_off").astype(float)
        rows.append(dict(window=wname, n=len(d),
                         rho_state_fwdvol=_spearman(off, d["fwd20_vol"]),
                         rho_state_fwdret=_spearman(off, d["fwd20_ret"]),
                         rho_B3_fwdvol=_spearman(d["B3"], d["fwd20_vol"]),
                         rho_P_fwdret=_spearman(d["P"], d["fwd20_ret"]),
                         rho_vol20_fwdvol=_spearman(d["vol20"], d["fwd20_vol"]),
                         rho_state_fwdvol_ctrl_vol20=_partial_corr(off, d["fwd20_vol"], d["vol20"]),
                         rho_B3_fwdvol_ctrl_vol20=_partial_corr(d["B3"], d["fwd20_vol"], d["vol20"]),
                         rho_state_fwdvol_ctrl_vol60=_partial_corr(off, d["fwd20_vol"], d["vol60"]),
                         rho_state_fwdret_ctrl_vol20=_partial_corr(off, d["fwd20_ret"], d["vol20"]),
                         rho_state_fwdup_ctrl=_partial_corr(off, d["fwd20_up"], d["vol20"]),
                         rho_state_fwddn_ctrl=_partial_corr(off, d["fwd20_dn"], d["vol20"]),
                         )) 
    tab = pd.DataFrame(rows)

    # 状态 × 未来 20 日（水平值）
    lev = df.loc[FULL[0]:FULL[1]].dropna(subset=["state"]).groupby("state").agg(
        n=("r", "size"), fwd20_ret=("fwd20_ret", "mean"), fwd20_vol=("fwd20_vol", "mean"),
        fwd20_mdd=("fwd20_mdd", "mean"), vol20=("vol20", "mean")).reset_index()
    lev["window"] = "full"

    # 状态新鲜度（驻留天数）分档 —— 独立复核第一阶段 5.7
    s = df["state"].copy()
    run = np.zeros(len(s), dtype=int)
    prev = None
    cnt = 0
    for i, v in enumerate(s.to_numpy()):
        cnt = cnt + 1 if v == prev else 1
        run[i] = cnt
        prev = v
    df["run"] = run
    df["bucket"] = pd.cut(df["run"], [0, 5, 20, 60, 10 ** 6],
                          labels=["1-5", "6-20", "21-60", "60+"])
    fresh = (df.loc[FULL[0]:FULL[1]].dropna(subset=["state", "fwd20_ret"])
             .groupby(["state", "bucket"], observed=True)
             .agg(n=("fwd20_ret", "size"), fwd20_ret=("fwd20_ret", "mean"),
                  pos_rate=("fwd20_ret", lambda x: float((x > 0).mean()))).reset_index())
    fresh["window"] = "2008-2020"

    save(tab.round(4), "02_状态检验.csv")
    save(lev.round(4), "02b_状态水平.csv")
    save(fresh.round(4), "02c_状态新鲜度.csv")
    df.reset_index().rename(columns={"index": "date", "date": "date"}).to_parquet(
        CACHE / "stateinfo.parquet", index=False)
    log(tab.round(3).to_string(index=False))
    return tab


# =====================================================================
# 4. 叠加
# =====================================================================
def load_bases() -> dict:
    bases = {}
    for tag in ("C1", "C2", "C3"):
        df = pd.read_parquet(CACHE / f"base_{tag}.parquet")
        df.index = pd.to_datetime(df.index)
        bases[tag] = {c: df[c] for c in df.columns}
        bases[tag]["dates"] = pd.DatetimeIndex(df.index)
    return bases


def base_net(b: dict) -> pd.Series:
    return ((1 + b["ovn"]) * (1 - b["cost"]) * (1 + b["idt"]) - 1).rename("r")


def ladder(state: pd.Series, m: float, reverse: bool = False,
           duration_gate: int | None = None) -> pd.Series:
    mid = (1.0 + m) / 2.0
    mp = {"risk_on": 1.0, "neutral": mid, "risk_off": m, "na": 1.0}
    if reverse:
        mp = {"risk_on": m, "neutral": mid, "risk_off": 1.0, "na": 1.0}
    if duration_gate is None:
        return state.map(mp).fillna(1.0).clip(0, 1)
    run, out = 0, []
    for s in state.to_numpy():
        run = run + 1 if s == "risk_off" else 0
        if s == "risk_off" and run < duration_gate:
            out.append(mid)
        else:
            out.append(mp.get(s, 1.0))
    return pd.Series(out, index=state.index).clip(0, 1)


def tercile_ladder(s: pd.Series, m: float, high_is_good: bool) -> pd.Series:
    q1 = s.expanding(min_periods=250).quantile(1 / 3)
    q2 = s.expanding(min_periods=250).quantile(2 / 3)
    mid = (1.0 + m) / 2.0
    t = np.where(s.isna() | q1.isna(), np.nan,
                 np.where(s <= q1, 0, np.where(s <= q2, 1, 2)))
    mp0, mp2 = (m, 1.0) if high_is_good else (1.0, m)
    e = pd.Series(t, index=s.index).map({0.0: mp0, 1.0: mid, 2.0: mp2})
    return e.fillna(1.0).clip(0, 1)


def apply_overlay(b: dict, e: pd.Series, fee_bp: float = FEE_BP) -> pd.Series:
    idx = b["dates"]
    e = e.reindex(idx).ffill().bfill()
    e_id = e.shift(1).bfill()
    e_on = e.shift(2).bfill()
    ovn, idt, cost = b["ovn"].to_numpy(), b["idt"].to_numpy(), b["cost"].to_numpy()
    d_e = (e_id - e_on).abs().to_numpy()
    r = ((1 + e_on.to_numpy() * ovn)
         * (1 - e_id.to_numpy() * cost - fee_bp / 1e4 * d_e)
         * (1 + e_id.to_numpy() * idt) - 1)
    return pd.Series(r, index=idx)


def perf(r: pd.Series, a: str, z: str) -> dict:
    s = r.loc[a:z].dropna()
    if len(s) < 60:
        return {}
    nav = (1 + s).cumprod()
    yrs = max((s.index[-1] - s.index[0]).days, 1) / 365.25
    cagr = float(nav.iloc[-1] ** (1 / yrs) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(TD))
    dd = nav / nav.cummax() - 1
    mdd = float(dd.min())
    return dict(n=int(len(s)), cagr=cagr, vol=vol,
                sharpe=float(s.mean() * TD / vol) if vol > 0 else np.nan,
                mdd=mdd, calmar=cagr / abs(mdd) if mdd < -1e-12 else np.nan,
                ulcer=float(np.sqrt((dd ** 2).mean())),
                worst20=float((nav / nav.shift(20) - 1).min()))


def nw_t(x: np.ndarray, lags: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 30:
        return np.nan
    mu = x.mean()
    e = x - mu
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(mu / np.sqrt(max(s, 1e-18) / n))


def alpha_beta(rs: pd.Series, rb: pd.Series) -> dict:
    d = pd.DataFrame({"s": rs, "b": rb}).dropna()
    x, y = d["b"].to_numpy(), d["s"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    XtX = np.linalg.inv(X.T @ X)
    S = np.zeros((2, 2))
    for l in range(6):
        w = 1 - l / 6
        for t in range(l, len(x)):
            S += w * np.outer(X[t] * resid[t], X[t - l] * resid[t - l])
    V = XtX @ S @ XtX * (len(x) / (len(x) - 2))
    se = np.sqrt(np.diag(V))
    return dict(beta=float(coef[1]), alpha_ann=float(coef[0]) * TD,
                alpha_t=float(coef[0] / se[0]) if se[0] > 0 else np.nan)


def decompose(rs: pd.Series, rb: pd.Series, e: pd.Series) -> dict:
    """mean(rs) = E[e]·mean(rb) + Cov(e_{t-1}, rb_t) − 成本项。"""
    idx = rs.index
    e_al = e.reindex(idx).ffill().bfill().shift(1)
    d = pd.DataFrame({"rs": rs, "rb": rb, "e": e_al}).dropna()
    eb, mb = float(d.e.mean()), float(d.rb.mean())
    cov = float(np.cov(d.e, d.rb, ddof=1)[0, 1]) * TD
    total = float(d.rs.mean()) * TD
    return dict(mean_ann=total, beta_term=eb * mb * TD, timing_term=cov,
                cost_term=total - eb * mb * TD - cov, mean_e=eb)


def vol_target_e(r: pd.Series, target: float, lam: float = 0.94) -> pd.Series:
    var = r.fillna(0.0).pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    sig = np.sqrt(var) * np.sqrt(TD)
    return (target / sig.replace(0, np.nan)).clip(0, 1).shift(1).fillna(1.0)


def _state_series(src: str, idx: pd.DatetimeIndex) -> pd.Series:
    if src == "pub":
        s = load_state().set_index("date")["state"]
    else:
        s = pd.read_parquet(CACHE / "state_rebuilt.parquet").set_index("date")["state"]
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def _feature_series(col: str, idx: pd.DatetimeIndex) -> pd.Series:
    s = load_state().set_index("date")[col]
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def build_channel_carriers() -> dict:
    """K2/K3 通道需要改变载体本身（权重/持仓数），重跑 C3。"""
    P = load_panel()
    prepare_lookbacks(P)
    st_pub = _state_series("pub", P.dates)
    start_i = int(np.argmax(P.dates >= pd.Timestamp("2007-12-20")))
    out = {}
    out["C3_K2_state"] = run_carrier(P, dict(kind="reversal", n=10, cycle=20, start_i=start_i,
                                             state_series=st_pub, risk_weight=True))
    out["C3_K3_state"] = run_carrier(P, dict(kind="reversal", n=10, cycle=20, start_i=start_i,
                                             state_series=st_pub,
                                             npos_state={"risk_on": 10, "neutral": 8, "risk_off": 6}))
    for k, v in out.items():
        pd.DataFrame({c: s for c, s in v.items() if c not in ("dates", "targets")},
                     index=v["dates"]).to_parquet(CACHE / f"base_{k}.parquet")
    return out


def stage_overlay() -> None:
    bases = load_bases()
    idx = bases["C1"]["dates"]
    st_pub = _state_series("pub", idx)
    st_reb = _state_series("reb", idx)
    rt_base = {k: base_net(v) for k, v in bases.items()}

    ch = build_channel_carriers()
    bases["C3_K2_state"] = {c: s for c, s in ch["C3_K2_state"].items() if c not in ("dates", "targets")}
    bases["C3_K2_state"]["dates"] = ch["C3_K2_state"]["dates"]
    bases["C3_K3_state"] = {c: s for c, s in ch["C3_K3_state"].items() if c not in ("dates", "targets")}
    bases["C3_K3_state"]["dates"] = ch["C3_K3_state"]["dates"]
    rt_base["C3_K2_state"] = base_net(bases["C3_K2_state"])
    rt_base["C3_K3_state"] = base_net(bases["C3_K3_state"])
    bases["C3_K2_only"] = bases["C3_K2_state"]
    rt_base["C3_K2_only"] = rt_base["C3_K2_state"]

    sig_star = {k: float(rt_base[k].loc[DEV[0]:DEV[1]].std() * np.sqrt(TD))
                for k in ("C1", "C2", "C3")}
    prim_net = {k: apply_overlay(bases[k], ladder(st_pub, PRIMARY_M), FEE_BP)
                for k in ("C1", "C2", "C3")}

    cfgs = []
    for c in ("C1", "C2", "C3"):
        for m in LADDERS:
            cfgs.append(dict(variant="K1", carrier=c, m=m, delay=0, cost=FEE_BP,
                             note=f"阶梯 m={m}", e_src="pub"))
    for dl in (0, 1, 5, 10, 20):
        for cst in (0.0, 10.0, 30.0):
            cfgs.append(dict(variant="K1", carrier="C1", m=PRIMARY_M, delay=dl, cost=cst,
                             note=f"延迟{dl}/成本{cst:g}bp", e_src="pub"))
    cfgs += [
        dict(variant="K2_state", carrier="C3_K2_state", m=PRIMARY_M, delay=0, cost=FEE_BP,
             note="单票风险+状态降险", e_src="pub"),
        dict(variant="K2_only", carrier="C3_K2_only", m=1.0, delay=0, cost=FEE_BP,
             note="单票风险，不降总暴露", e_src="pub"),
        dict(variant="K3_state", carrier="C3_K3_state", m=1.0, delay=0, cost=FEE_BP,
             note="持仓数10/8/6，不降总暴露", e_src="pub"),
        dict(variant="K3_only", carrier="C3", m=1.0, delay=0, cost=FEE_BP,
             note="持仓数基准10", e_src="pub"),
    ]
    for c in ("C1", "C2", "C3"):
        for v in ("base", "static", "volmatch", "levneutral", "voltarget"):
            cfgs.append(dict(variant=v, carrier=c, m=PRIMARY_M, delay=0, cost=FEE_BP,
                             note=v, e_src="pub"))
    for c in ("C1", "C2", "C3"):
        for v in ("state", "B3", "P"):
            cfgs.append(dict(variant=f"tercile_{v}", carrier=c, m=PRIMARY_M, delay=0,
                             cost=FEE_BP, note=f"{v}三分位同阶梯", e_src="pub"))
    cfgs.append(dict(variant="reverse", carrier="C1", m=PRIMARY_M, delay=0, cost=FEE_BP,
                     note="反向阶梯", e_src="pub"))
    cfgs.append(dict(variant="duration20", carrier="C1", m=PRIMARY_M, delay=0, cost=FEE_BP,
                     note="防守≥20日才降", e_src="pub"))
    for c in ("C1", "C2", "C3"):
        cfgs.append(dict(variant="K1", carrier=c, m=PRIMARY_M, delay=0, cost=FEE_BP,
                         note="重建状态", e_src="reb"))

    seen, uniq = set(), []
    for c in cfgs:
        key = (c["variant"], c["carrier"], c["m"], c["delay"], c["cost"], c["e_src"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    cfgs = uniq
    log(f"配置数（去重后）={len(cfgs)}")

    rows, attrib, trials, navs = [], [], [], {}
    for k, c in enumerate(cfgs, 1):
        cid = f"RB{k:03d}"
        b = bases[c["carrier"]]
        rb = rt_base[c["carrier"]]
        st = st_pub if c["e_src"] == "pub" else st_reb
        v = c["variant"]

        if v == "base":
            e = pd.Series(1.0, index=idx)
        elif v in ("static", "volmatch", "levneutral"):
            e = ladder(st, PRIMARY_M)
        elif v == "voltarget":
            e = vol_target_e(rb, sig_star.get(c["carrier"], 0.20))
        elif v == "tercile_B3":
            e = tercile_ladder(_feature_series("B3", idx), c["m"], high_is_good=False)
        elif v == "tercile_P":
            e = tercile_ladder(_feature_series("P", idx), c["m"], high_is_good=True)
        elif v == "tercile_state":
            e = ladder(st, c["m"])
        elif v == "reverse":
            e = ladder(st, c["m"], reverse=True)
        elif v == "duration20":
            e = ladder(st, c["m"], duration_gate=20)
        elif v in ("K2_only", "K3_state", "K3_only"):
            e = pd.Series(1.0, index=idx)
        else:
            e = ladder(st, c["m"])
        if c["delay"]:
            e = e.shift(c["delay"]).bfill()

        if v not in ("static", "volmatch", "levneutral"):
            r_net = apply_overlay(b, e, c["cost"])
            navs[cid] = r_net
        else:
            r_net = None

        for wname, (a, z) in (("dev", DEV), ("val", VAL), ("full", FULL)):
            if v == "static":
                cst = float(prim_e_c(carrier=c["carrier"], st=st_pub).loc[a:z].mean())
                rr = apply_overlay(b, pd.Series(cst, index=idx), c["cost"])
                extra = cst
            elif v == "volmatch":
                pv = float(prim_net[c["carrier"]].loc[a:z].std())
                bv = float(rb.loc[a:z].std())
                cst = min(pv / bv, 1.0) if bv > 0 else 1.0
                rr = apply_overlay(b, pd.Series(cst, index=idx), c["cost"])
                extra = cst
            elif v == "levneutral":
                cst = float(e.loc[a:z].mean())
                rr = apply_overlay(b, (e / cst).clip(0, 3), c["cost"])
                extra = cst
            else:
                rr = r_net
                extra = float(e.loc[a:z].mean())
            m = perf(rr, a, z)
            if not m:
                continue
            rows.append(dict(config_id=cid, variant=v, carrier=c["carrier"], m=c["m"],
                             delay=c["delay"], cost_bp=c["cost"], state_src=c["e_src"],
                             window=wname, mean_e=extra, **m))
            if v not in ("static", "volmatch", "levneutral"):
                ab = alpha_beta(rr.loc[a:z], rb.loc[a:z])
                dc = decompose(rr.loc[a:z], rb.loc[a:z], e)
                attrib.append(dict(config_id=cid, variant=v, carrier=c["carrier"], m=c["m"],
                                   delay=c["delay"], cost_bp=c["cost"], state_src=c["e_src"],
                                   window=wname, **ab, **dc))
        trials.append(dict(config_id=cid, **{k2: c[k2] for k2 in
                                             ("variant", "carrier", "m", "delay", "cost", "note", "e_src")}))

    res = pd.DataFrame(rows)
    att = pd.DataFrame(attrib)
    save(res.round(5), "10_叠加结果.csv")
    save(att.round(5), "11_归因分解.csv")
    save(pd.DataFrame(trials), "12_试验清单.csv")
    pd.DataFrame({k: (1 + v).cumprod() for k, v in navs.items()}).loc[FULL[0]:FULL[1]] \
        .to_parquet(CACHE / "navs.parquet")

    _null_tests(bases, rt_base, st_pub, idx, res, out_csv="14_空分布.csv")


def prim_e_c(carrier: str, st: pd.Series) -> pd.Series:
    return ladder(st, PRIMARY_M)


# =====================================================================
# 5. 账户级机制验证（2016-2020，公共 Account 引擎）
# =====================================================================
class FrozenMarketData:
    """把冻结层的日线/公司行动包装成 账户撮合 需要的适配器（只读）。"""

    def __init__(self, start: str, end: str, symbols: list[str]):
        import importlib
        self.acct = importlib.import_module("账户撮合")
        con = connect()
        syms = ",".join("'" + s + "'" for s in symbols)
        df = con.execute(f"""
            SELECT symbol, date, open, high, low, close, preclose, volume, amount, is_suspended, is_st
            FROM daily WHERE date BETWEEN DATE '{start}' AND DATE '{end}' AND symbol IN ({syms})
        """).df()
        ev = con.execute(f"""
            SELECT symbol, ex_date, dividend_per_share, per_share_bonus, rights_ratio, rights_price
            FROM corp_actions WHERE ex_date BETWEEN DATE '{start}' AND DATE '{end}' AND symbol IN ({syms})
        """).df()
        inf = con.execute("SELECT symbol, name, ipo_date, delist_date FROM symbols").df()
        cal = con.execute(f"""
            SELECT date FROM calendar WHERE is_trading_day=1
              AND date BETWEEN DATE '{start}' AND DATE '{end}' ORDER BY date""").df()
        con.close()

        df["date"] = pd.to_datetime(df["date"])
        self._bars: dict = {}
        for d, g in df.groupby("date"):
            dd = {}
            for r in g.itertuples():
                try:
                    dd[r.symbol] = self.acct.Bar(
                        code=r.symbol, date=d.date(), open=float(r.open), high=float(r.high),
                        low=float(r.low), close=float(r.close), volume=float(r.volume or 0.0),
                        amount=float(r.amount) if r.amount is not None else None,
                        prev_close=float(r.preclose), prev_close_is_ref=True,
                        is_st=bool(r.is_st), trade_status=0 if bool(r.is_suspended) else 1)
                except Exception:
                    continue
            self._bars[d.date()] = dd
        self._events: dict = {}
        ev["ex_date"] = pd.to_datetime(ev["ex_date"])
        for d, g in ev.groupby("ex_date"):
            self._events[d.date()] = [
                self.acct.CorporateAction(code=r.symbol, ex_date=d.date(),
                                          dividend_per_share=float(r.dividend_per_share or 0),
                                          per_share_bonus=float(r.per_share_bonus or 0),
                                          rights_ratio=float(r.rights_ratio or 0),
                                          rights_price=float(r.rights_price or 0),
                                          source="frozen.corp_actions")
                for r in g.itertuples()]
        self.symbols = {
            r.symbol: self.acct.SymbolInfo(
                code=r.symbol, name=r.name or "",
                ipo_date=pd.to_datetime(r.ipo_date).date() if pd.notna(r.ipo_date) else None,
                out_date=pd.to_datetime(r.delist_date).date() if pd.notna(r.delist_date) else None)
            for r in inf.itertuples()}
        self.calendar = [d.date() for d in pd.to_datetime(cal["date"])]
        self._all_dates = sorted(self._bars)

    def bars_on(self, d):
        return self._bars.get(d, {})

    def events_on(self, d):
        return self._events.get(d, [])


def stage_account() -> None:
    import importlib
    acct = importlib.import_module("账户撮合")
    tg = pd.read_parquet(CACHE / "targets_C3.parquet")
    tg["date"] = pd.to_datetime(tg["date"])
    win = (tg.date >= VAL[0]) & (tg.date <= VAL[1])
    tg = tg[win].reset_index(drop=True)
    cols = [c for c in tg.columns if c.startswith("s") and c != "state"]
    tgt: dict = {}
    for r in tg.itertuples():
        syms = [getattr(r, c) for c in cols if isinstance(getattr(r, c), str)]
        tgt[r.date.date()] = syms
    all_syms = sorted({s for v in tgt.values() for s in v})
    log(f"账户验证：{len(tgt)} 个再平衡日，{len(all_syms)} 只标的")

    md = FrozenMarketData(VAL[0], VAL[1], all_syms)
    st = load_state().set_index("date")["state"]
    e_s = ladder(st, PRIMARY_M)

    # 周期内恒定的暴露：在每个再平衡决策日取状态乘数，持有到下一次
    dates_cycle = sorted(tgt)
    e_cycle = pd.Series(np.nan, index=pd.DatetimeIndex(md.calendar))
    cur = 1.0
    for d in md.calendar:
        ts = pd.Timestamp(d)
        if d in tgt:
            cur = float(e_s.get(ts, 1.0))
        e_cycle.loc[ts] = cur

    res = []
    for tag, use_e in (("控制_e=1", False), ("状态_e_(周期恒定)", True)):
        cfg = acct.AccountConfig(
            initial_cash=1_000_000.0,
            fee=acct.fee.FeeConfig(slippage_bps=10.0, max_participation=0.02),
            cash_settle_days=0, enforce_t1=True, delisting_haircut=0.10)
        a = acct.Account(cfg, md.calendar, freeze_date=md.calendar[-1], symbol_info=md.symbols)
        navs = []
        for d in md.calendar:
            bars = md.bars_on(d)
            st_codes = {c for c, b in bars.items() if b.is_st}
            rec = a.on_day(d, bars=bars, events=md.events_on(d), st_codes=st_codes)
            navs.append((d, rec["nav"]))
            if d not in tgt:
                continue
            nav = rec["nav"]
            e = float(e_cycle.loc[pd.Timestamp(d)]) if use_e else 1.0
            want = set(tgt[d])
            held = {c for c, p in a.positions.items() if p.shares > 0}
            for c in sorted(held - want):
                q = a.positions[c].available
                if q > 0:
                    a.sell_open(c, q, d)
            n = max(len(want), 1)
            for c in sorted(want):
                b = bars.get(c)
                if b is None or b.close <= 0:
                    continue
                tval = nav * e / n
                cur_q = a.positions[c].shares if c in a.positions else 0
                tq = int(tval / b.close / 100) * 100
                dq = tq - cur_q
                if dq >= 100:
                    a.buy_open(c, dq, d)
                elif dq <= -100 and c in a.positions:
                    q = min(-dq, a.positions[c].available)
                    if q > 0:
                        a.sell_open(c, q, d)
        fin = a.finalize(liquidate=False)
        s = pd.Series([v for _, v in navs],
                      index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in navs]))
        r = s.pct_change().dropna()
        m = perf(r, VAL[0], VAL[1])
        res.append(dict(arm=tag, nav_end=float(fin["nav"]), n_trades=len(a.trades),
                        turnover_ann=float(sum(t.amount for t in a.trades) / 1e6
                                           / max(len(r) / TD, 1e-9)),
                        flags=dict(a.flag_counts), **m))
        log(f"{tag}: 末值 {fin['nav']:,.0f} 成交 {len(a.trades)} 笔 CAGR {m['cagr']:.4f} "
            f"MDD {m['mdd']:.4f} Calmar {m['calmar']:.4f}")

    # 组合层同口径对照（周期恒定暴露）
    b3 = pd.read_parquet(CACHE / "base_C3.parquet")
    b3.index = pd.to_datetime(b3.index)
    bases = {c: b3[c] for c in b3.columns}
    bases["dates"] = pd.DatetimeIndex(b3.index)
    for tag, e in (("组合层_e=1", pd.Series(1.0, index=b3.index)),
                   ("组合层_e_cycle", e_cycle.reindex(b3.index).ffill())):
        rr = apply_overlay(bases, e, FEE_BP)
        m = perf(rr, VAL[0], VAL[1])
        res.append(dict(arm=tag, nav_end=float((1 + rr.loc[VAL[0]:VAL[1]]).prod()),
                        n_trades=np.nan, turnover_ann=np.nan, flags={}, **m))
    out = pd.DataFrame(res)
    save(out.round(4), "13_账户级验证.csv")
    log(out[["arm", "cagr", "vol", "mdd", "calmar", "n_trades", "turnover_ann"]].round(4).to_string(index=False))


# =====================================================================
# 6. 自检
# =====================================================================
def stage_check() -> None:
    rows = []
    bases = load_bases()
    idx = bases["C1"]["dates"]
    st = _state_series("pub", idx)
    b = bases["C1"]
    rb = base_net(b)

    # 1) 恒等：e≡1 必须逐日等于基准
    r1 = apply_overlay(b, pd.Series(1.0, index=idx), FEE_BP)
    rows.append(dict(check="叠加恒等_e=1", value=float((r1 - rb).abs().max()), pass_=bool((r1 - rb).abs().max() < 1e-12)))

    # 2) e≡0：无暴露、无收益（除首日建仓成本）
    r0 = apply_overlay(b, pd.Series(0.0, index=idx), FEE_BP)
    rows.append(dict(check="叠加_e=0_最大绝对日收益", value=float(r0.abs().max()), pass_=bool(r0.abs().max() < 1e-12)))

    # 3) e≡c：等于手工缩放公式
    c = 0.5
    rc = apply_overlay(b, pd.Series(c, index=idx), FEE_BP)
    man = (1 + c * b["ovn"]) * (1 - c * b["cost"]) * (1 + c * b["idt"]) - 1
    rows.append(dict(check="叠加_e=0.5_对缩放公式", value=float((rc - man).abs().max()), pass_=bool((rc - man).abs().max() < 1e-12)))

    # 4) 对齐：状态滞后 1 日，序列必须变（否则说明没按 t+1 开盘执行）
    e0 = ladder(st, PRIMARY_M)
    e1 = e0.shift(1).bfill()
    rA, rB = apply_overlay(b, e0, FEE_BP), apply_overlay(b, e1, FEE_BP)
    rows.append(dict(check="延迟1日_与不延迟的差异", value=float((rA - rB).abs().max()), pass_=bool((rA - rB).abs().max() > 1e-6)))

    # 5) 截断不变：只用 T 之前的数据重建 C1 载体，T 之前净值必须逐位一致
    T = pd.Timestamp("2014-06-30")
    P = load_panel("2007-01-01", T.strftime("%Y-%m-%d"))
    prepare_lookbacks(P)
    c1t = run_carrier(P, dict(kind="equal"))
    nav_t = (1 + base_net(c1t)).cumprod()
    nav_f = (1 + rb).cumprod()
    common = nav_t.index.intersection(nav_f.index)
    common = common[common <= T]
    rel = float((nav_t.loc[common] / nav_f.loc[common] - 1).abs().max()) if len(common) else np.nan
    rows.append(dict(check="截断不变_C1_净值最大相对差", value=rel, pass_=bool(rel < 1e-9)))

    # 6) 状态映射
    e = ladder(st, 0.5)
    ok_map = bool(e[st == "risk_on"].max() == 1.0 and e[st == "risk_off"].max() == 0.5
                  and abs(e[st == "neutral"].iloc[0] - 0.75) < 1e-12)
    rows.append(dict(check="阶梯映射_risk_on=1/neutral=.75/risk_off=.5", value=float(e.mean()), pass_=ok_map))

    # 7) 冻结日与窗口
    mx = max(b["dates"].max(), bases["C2"]["dates"].max(), bases["C3"]["dates"].max())
    rows.append(dict(check="最大日期≤2020-12-31", value=float(pd.Timestamp(mx).year), pass_=bool(pd.Timestamp(mx) <= pd.Timestamp("2020-12-31"))))

    out = pd.DataFrame(rows)
    out["pass_"] = out["pass_"].map({True: "PASS", False: "FAIL"})
    save(out, "09_自检.csv")
    log(out.to_string(index=False))
    if (out["pass_"] == "FAIL").any():
        raise SystemExit("自检失败")


def stage_report() -> None:
    r = pd.read_csv(OUT / "10_叠加结果.csv")
    a = pd.read_csv(OUT / "11_归因分解.csv")
    nul = pd.read_csv(OUT / "14_空分布.csv")

    key = [("base", "C1", 0.5), ("K1", "C1", 0.25), ("K1", "C1", 0.5), ("K1", "C1", 0.75),
           ("K1", "C1", 1.0), ("static", "C1", 0.5), ("volmatch", "C1", 0.5),
           ("levneutral", "C1", 0.5), ("voltarget", "C1", 0.5), ("reverse", "C1", 0.5),
           ("duration20", "C1", 0.5), ("tercile_B3", "C1", 0.5), ("tercile_P", "C1", 0.5)]
    rows = []
    for v, c, m in key:
        s = r[(r.variant == v) & (r.carrier == c) & (np.isclose(r.m, m)) & (r.delay == 0)
              & (np.isclose(r.cost_bp, 10)) & (r.state_src == "pub")]
        for _, x in s.iterrows():
            rows.append(dict(arm=v, carrier=c, m=m, window=x.window, mean_e=x.mean_e, cagr=x.cagr,
                             vol=x.vol, sharpe=x.sharpe, mdd=x.mdd, calmar=x.calmar))
    kt = pd.DataFrame(rows)
    save(kt.round(4), "16_关键对照.csv")

    # 判定表
    def get(v, c, m, w, col):
        s = r[(r.variant == v) & (r.carrier == c) & (np.isclose(r.m, m)) & (r.delay == 0)
              & (np.isclose(r.cost_bp, 10)) & (r.state_src == "pub") & (r.window == w)]
        return float(s[col].iloc[0]) if len(s) else np.nan

    j = []
    d_dev = get("K1", "C1", .5, "dev", "calmar") - get("static", "C1", .5, "dev", "calmar")
    d_val = get("K1", "C1", .5, "val", "calmar") - get("static", "C1", .5, "val", "calmar")
    j.append(dict(id="J1", desc="对同平均暴露静态：dev与val的Calmar增益同号且都≥+0.03",
                  dev=d_dev, val=d_val, pass_=bool(d_dev >= .03 and d_val >= .03)))
    j.append(dict(id="J2", desc="对同波动静态：val CAGR ≥ 同波动静态-0.5pp",
                  dev=np.nan, val=get("K1", "C1", .5, "val", "cagr") - get("volmatch", "C1", .5, "val", "cagr"),
                  pass_=bool(get("K1", "C1", .5, "val", "cagr") >= get("volmatch", "C1", .5, "val", "cagr") - .005)))
    j.append(dict(id="J3", desc="对通用波动目标：val Calmar ≥ 波动目标",
                  dev=np.nan, val=get("K1", "C1", .5, "val", "calmar") - get("voltarget", "C1", .5, "val", "calmar"),
                  pass_=bool(get("K1", "C1", .5, "val", "calmar") >= get("voltarget", "C1", .5, "val", "calmar"))))
    d1 = r[(r.variant == "K1") & (r.carrier == "C1") & (np.isclose(r.m, .5)) & (r.delay == 1)
           & (np.isclose(r.cost_bp, 10)) & (r.window == "full")]["calmar"].iloc[0]
    d0 = r[(r.variant == "K1") & (r.carrier == "C1") & (np.isclose(r.m, .5)) & (r.delay == 0)
           & (np.isclose(r.cost_bp, 10)) & (r.window == "full")]["calmar"].iloc[0]
    st_full = r[(r.variant == "static") & (r.carrier == "C1") & (r.window == "full")]["calmar"].iloc[0]
    d30 = r[(r.variant == "K1") & (r.carrier == "C1") & (np.isclose(r.m, .5)) & (r.delay == 0)
            & (np.isclose(r.cost_bp, 30)) & (r.window == "full")]["calmar"].iloc[0]
    gain0, gain1 = d0 - st_full, d1 - st_full
    j.append(dict(id="J4", desc="延迟1日保留≥50%增益；30bp成本下仍为正",
                  dev=gain1 / gain0 if gain0 else np.nan, val=d30 - st_full,
                  pass_=bool(gain1 >= .5 * gain0 and d30 - st_full > 0)))
    j.append(dict(id="J5", desc="反向阶梯不得改善（ΔCalmar≤+0.02）",
                  dev=get("reverse", "C1", .5, "dev", "calmar") - get("static", "C1", .5, "dev", "calmar"),
                  val=get("reverse", "C1", .5, "val", "calmar") - get("static", "C1", .5, "val", "calmar"),
                  pass_=bool((get("reverse", "C1", .5, "dev", "calmar") - get("static", "C1", .5, "dev", "calmar")) <= .02
                             and (get("reverse", "C1", .5, "val", "calmar") - get("static", "C1", .5, "val", "calmar")) <= .02)))
    p_full = float(nul[nul.window == "full"].p_value.iloc[0])
    p_val = float(nul[nul.window == "val"].p_value.iloc[0])
    j.append(dict(id="J6", desc="环形平移零分布 p≤0.10（full 与 val）",
                  dev=p_full, val=p_val, pass_=bool(p_full <= .10 and p_val <= .10)))
    jt = pd.DataFrame(j)
    jt["pass_"] = jt["pass_"].map({True: "PASS", False: "FAIL"})
    save(jt.round(4), "17_判定.csv")

    # 稳健性长表
    rob = r[(r.variant == "K1") & (r.carrier == "C1") & (np.isclose(r.m, .5)) & (r.state_src == "pub")]
    save(rob[["window", "delay", "cost_bp", "mean_e", "cagr", "vol", "mdd", "calmar"]].round(4),
         "14_稳健性.csv")
    log(jt.to_string(index=False))

    # 关键净值落盘
    navs = pd.read_parquet(CACHE / "navs.parquet")
    tri = pd.read_csv(OUT / "12_试验清单.csv")
    keep = tri[tri.variant.isin(["K1", "base", "voltarget", "static", "duration20", "reverse"])
               & tri.carrier.eq("C1")]["config_id"].tolist()
    keep = [c for c in keep if c in navs.columns]
    nd = OUT / "16_净值"
    nd.mkdir(exist_ok=True)
    sub = navs[keep].loc[FULL[0]:FULL[1]]
    sub = sub / sub.iloc[0]          # 归一到窗口起点 = 1
    sub.to_csv(nd / "C1_关键净值.csv", encoding="utf-8-sig")
    log(f"写出 16_净值/C1_关键净值.csv 列={len(keep)} 行={len(sub)}")

    # 年度拆解：叠加 vs 同暴露静态（C1, m=0.5, 10bp）
    bases = load_bases()
    b = bases["C1"]
    rb = base_net(b)
    st = _state_series("pub", b["dates"])
    e = ladder(st, PRIMARY_M)
    ro = apply_overlay(b, e, FEE_BP)
    cst = float(e.loc[FULL[0]:FULL[1]].mean())
    rs = apply_overlay(b, pd.Series(cst, index=b["dates"]), FEE_BP)
    yrs = []
    for y in range(2008, 2021):
        a, z = f"{y}-01-01", f"{y}-12-31"
        x, y2 = ro.loc[a:z], rs.loc[a:z]
        yrs.append(dict(year=y, overlay_ret=float((1 + x).prod() - 1),
                        static_ret=float((1 + y2).prod() - 1),
                        diff=float((1 + x).prod() - (1 + y2).prod()),
                        overlay_mdd=float(((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()),
                        static_mdd=float(((1 + y2).cumprod() / (1 + y2).cumprod().cummax() - 1).min())))
    yt = pd.DataFrame(yrs)
    save(yt.round(4), "18_年度拆解.csv")

    # 暴露换手
    dd = pd.DataFrame({
        "item": ["暴露年换手(单边)", "平均暴露", "同窗口静态暴露", "状态切换次数/年"],
        "value": [float(e.diff().abs().loc[FULL[0]:FULL[1]].sum() / 13.0),
                  float(e.loc[FULL[0]:FULL[1]].mean()), cst,
                  float((st.loc[FULL[0]:FULL[1]] != st.loc[FULL[0]:FULL[1]].shift(1)).sum() / 13.0)]})
    save(dd.round(4), "19_暴露统计.csv")


def _null_tests(bases, rt_base, st, idx, res, out_csv: str, n_iter: int = 200) -> None:
    rng = np.random.default_rng(20261026)
    b = bases["C1"]
    rb = rt_base["C1"]
    e0 = ladder(st, PRIMARY_M)
    r0 = apply_overlay(b, e0, FEE_BP)
    out = []
    for wname, (a, z) in (("full", FULL), ("val", VAL)):
        cst = float(e0.loc[a:z].mean())
        static = perf(apply_overlay(b, pd.Series(cst, index=idx), FEE_BP), a, z)
        actual = perf(r0, a, z)
        d_act = actual["calmar"] - static["calmar"]
        draws = []
        arr = e0.to_numpy()
        for k in range(n_iter):
            sh = int(rng.integers(20, len(arr) - 20))
            es = pd.Series(np.roll(arr, sh), index=idx)
            c2 = float(es.loc[a:z].mean())
            p = perf(apply_overlay(b, es, FEE_BP), a, z)
            ps = perf(apply_overlay(b, pd.Series(c2, index=idx), FEE_BP), a, z)
            draws.append(p["calmar"] - ps["calmar"])
        draws = np.array(draws)
        out.append(dict(window=wname, actual_dcalmar=d_act, static_calmar=static["calmar"],
                        strat_calmar=actual["calmar"],
                        null_mean=float(draws.mean()), null_p95=float(np.percentile(draws, 95)),
                        p_value=float((draws >= d_act).mean()), n_iter=n_iter))
    save(pd.DataFrame(out).round(5), out_csv)
    log(pd.DataFrame(out).round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["audit", "base", "stateinfo", "overlay", "account", "check", "report", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    fns = {"audit": stage_audit, "base": stage_base, "stateinfo": stage_stateinfo,
           "overlay": stage_overlay, "account": stage_account, "check": stage_check,
           "report": stage_report}
    for st in (["audit", "base", "stateinfo", "overlay", "check", "account", "report"] if a.stage == "all" else [a.stage]):
        fns[st]()


if __name__ == "__main__":
    main()
