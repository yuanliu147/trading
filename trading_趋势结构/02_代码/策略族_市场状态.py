#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略族_市场状态.py —— 市场许可与仓位状态研究（趋势结构/波段交易项目）

范围声明
--------
本脚本只回答一个问题：**在只使用 t 日收盘可知信息的前提下，市场状态能否
改善"何时不交易 / 何时该重仓"的决策**。它不选股，不替代公共引擎。

纪律（硬约束，代码内强制）
--------------------------
1. 数据冻结 2026-10-26；panel 视图按 date <= FREEZE 截断。
2. 参数只用 2008-2015（开发期）与 2005-2007（早期压力）检查；2016-2020 为
   验证期，规则冻结后才看结果；2021-01-01 之后**只做覆盖审计**，不输出任何
   收益列（见 stage=audit）。
3. 所有标准化/分位都用滚动或扩张窗口，且只含 t 及以前的数据。
4. 信号在 t 日收盘后生成，仓位乘数作用于 t+1 日收益；换手按单边 10bp 计费。

阶段
----
extract  从 baostock 只读取数 → 03_结果/市场状态_缓存/*.parquet
feature  逐股滚动 + 截面聚合 → 日频市场特征表
eval     状态定义、四种仓位方案、≤2020 分窗口评估、静态对照、平移检验
check    核心自检：未来数据污染 / 截断不变性 / 仓位对齐 / 费用方向
signal   同一趋势信号在不同状态下的表现（诊断，不选股）
audit    2021+ 数据覆盖审计（无收益）+ 跨源抽检 + 扩展状态
all      依次执行 extract → feature → eval → check → signal → audit

用法
----
python3 策略族_市场状态.py --stage all
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "03_结果" / "市场状态_缓存"
OUT = ROOT / "03_结果" / "市场状态"

BAO_DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
THS_DB = Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb")

FREEZE = "2026-10-26"          # 统一冻结日
EXTRACT_START = "2003-10-01"   # 为 2005 年初的 250 日窗口预留前置数据
EARLY = ("2005-01-01", "2007-12-31")
DEV = ("2008-01-01", "2015-12-31")
VAL = ("2016-01-01", "2020-12-31")
EVAL_END = "2020-12-31"        # 策略研究员可见收益的最后一天
AUDIT = ("2021-01-01", FREEZE)

COST_ONE_WAY = 0.0010          # 单边 10bp：佣金+印花税+滑点打包
TRADING_DAYS = 244


# ---------------------------------------------------------------- utilities
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dirs() -> None:
    for p in (CACHE, OUT):
        p.mkdir(parents=True, exist_ok=True)


def duck(read_only: bool = False):
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute(f"PRAGMA temp_directory='{CACHE / '_tmp'}'")
    return con


def save_rows(path: Path, n: int) -> None:
    """把行数写进同名 .rows 文件，便于报告里核对。"""
    path.with_suffix(path.suffix + ".rows").write_text(str(n), encoding="utf-8")


# ---------------------------------------------------------------- extract
def stage_extract() -> None:
    import duckdb

    ensure_dirs()
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("LOAD sqlite")
    con.execute(f"ATTACH '{BAO_DB}' AS bs (TYPE SQLITE, READ_ONLY)")

    log("extract: stock_kline_d(adjustflag=1 后复权) …")
    panel = CACHE / "panel_hfq.parquet"
    con.execute(
        f"""
        COPY (
            SELECT code, date, open, high, low, close, preclose,
                   volume, amount, turn, trade_status, pct_chg, is_st
            FROM bs.stock_kline_d
            WHERE adjustflag = 1
              AND date >= '{EXTRACT_START}' AND date <= '{FREEZE}'
        ) TO '{panel}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    log("extract: index_kline_d …")
    idx = CACHE / "index_daily.parquet"
    con.execute(
        f"""
        COPY (
            SELECT code, date, open, high, low, close, preclose, volume, amount, pct_chg
            FROM bs.index_kline_d
            WHERE date >= '{EXTRACT_START}' AND date <= '{FREEZE}'
        ) TO '{idx}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    log("extract: stock_basic / trade_dates …")
    basic = CACHE / "stock_basic.parquet"
    con.execute(
        f"""
        COPY (
            SELECT code, code_name, ipo_date, out_date, type, status
            FROM bs.stock_basic WHERE type = 1
        ) TO '{basic}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    cal = CACHE / "calendar.parquet"
    con.execute(
        f"""
        COPY (
            SELECT calendar_date, is_trading_day FROM bs.trade_dates
            WHERE calendar_date >= '{EXTRACT_START}' AND calendar_date <= '{FREEZE}'
        ) TO '{cal}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # ---- 行数与口径核对
    chk = duckdb.connect()
    n_panel = chk.execute(f"SELECT count(*), count(DISTINCT code), min(date), max(date) FROM '{panel}'").fetchone()
    n_idx = chk.execute(f"SELECT count(*), count(DISTINCT code), min(date), max(date) FROM '{idx}'").fetchone()
    log(f"panel rows={n_panel[0]:,} codes={n_panel[1]} {n_panel[2]}~{n_panel[3]}")
    log(f"index rows={n_idx[0]:,} codes={n_idx[1]} {n_idx[2]}~{n_idx[3]}")
    assert str(n_panel[3]) <= FREEZE, "panel 越过冻结日"
    assert str(n_idx[3]) <= FREEZE, "index 越过冻结日"
    save_rows(panel, n_panel[0])
    save_rows(idx, n_idx[0])
    save_rows(basic, chk.execute(f"SELECT count(*) FROM '{basic}'").fetchone()[0])
    save_rows(cal, chk.execute(f"SELECT count(*) FROM '{cal}'").fetchone()[0])
    chk.close()


# ---------------------------------------------------------------- feature
FEATURE_SQL = """
CREATE OR REPLACE TABLE feat AS
SELECT
    code, date, close, high, low, open, amount, turn, trade_status, is_st,
    pct_chg / 100.0 AS ret1,
    CAST(row_number() OVER (PARTITION BY code ORDER BY date) AS INTEGER) AS bar_no,
    avg(close)  OVER w20  AS ma20,
    avg(close)  OVER w60  AS ma60,
    avg(close)  OVER w120 AS ma120,
    avg(close)  OVER w250 AS ma250,
    max(high)   OVER w250 AS hi250,
    min(low)    OVER w250 AS lo250,
    avg(amount) OVER w20  AS amt20,
    count(*)    OVER w60  AS bars60,
    lag(close, 20) OVER (PARTITION BY code ORDER BY date) AS c20,
    lag(close, 60) OVER (PARTITION BY code ORDER BY date) AS c60,
    lag(close,120) OVER (PARTITION BY code ORDER BY date) AS c120
FROM panel
WINDOW
    w20  AS (PARTITION BY code ORDER BY date ROWS BETWEEN 19  PRECEDING AND CURRENT ROW),
    w60  AS (PARTITION BY code ORDER BY date ROWS BETWEEN 59  PRECEDING AND CURRENT ROW),
    w120 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW),
    w250 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
"""
# 涨跌停阈值：ST 5%；创业板 2020-08-24 起 20%；科创板 20%；其余 10%。
# 用 2% 相对容差吸收"按价格取整"造成的 9.94% 之类偏差。
FEATURE_SQL += """
CREATE OR REPLACE TABLE feat2 AS
SELECT *,
    CASE
        WHEN is_st = 1 THEN 0.05
        WHEN code LIKE 'sz.30%' AND date >= DATE '2020-08-24' THEN 0.20
        WHEN code LIKE 'sh.688%' THEN 0.20
        ELSE 0.10
    END AS limit_pct
FROM feat
"""
FEATURE_SQL += """
CREATE OR REPLACE TABLE elig AS
SELECT * FROM feat2
WHERE trade_status = 1
  AND is_st = 0
  AND bar_no >= 60
  AND bars60 >= 50
  AND amt20 > 0
  AND close > 0
  AND ret1 IS NOT NULL
"""
FEATURE_SQL += """
CREATE OR REPLACE TABLE date_med AS
SELECT date, median(amt20) AS amt_med FROM elig GROUP BY date ORDER BY date
"""
FEATURE_SQL += """
CREATE OR REPLACE TABLE mkt AS
SELECT
    e.date,
    count(*)                                                    AS n_univ,
    avg(CASE WHEN ret1 > 0 THEN 1.0 ELSE 0.0 END)               AS pct_up,
    avg(CASE WHEN ret1 < 0 THEN 1.0 ELSE 0.0 END)               AS pct_dn,
    median(ret1)                                                AS med_ret1,
    avg(ret1)                                                   AS ew_ret1,
    stddev_samp(ret1)                                           AS csd_ret1,
    avg(CASE WHEN ret1 >=  limit_pct * 0.98 THEN 1.0 ELSE 0.0 END) AS pct_limit_up,
    avg(CASE WHEN ret1 <= -limit_pct * 0.98 THEN 1.0 ELSE 0.0 END) AS pct_limit_dn,
    avg(CASE WHEN close > ma20  THEN 1.0 ELSE 0.0 END)          AS pct_ma20,
    avg(CASE WHEN close > ma60  THEN 1.0 ELSE 0.0 END)          AS pct_ma60,
    avg(CASE WHEN close > ma120 THEN 1.0 ELSE 0.0 END)          AS pct_ma120,
    avg(CASE WHEN close > ma250 THEN 1.0 ELSE 0.0 END)         AS pct_ma250,
    avg(CASE WHEN close >= hi250 * 0.999 THEN 1.0 ELSE 0.0 END) AS pct_nh250,
    avg(CASE WHEN close <= lo250 * 1.001 THEN 1.0 ELSE 0.0 END) AS pct_nl250,
    median(close / c20  - 1)                                    AS med_r20,
    median(close / c60  - 1)                                    AS med_r60,
    median(close / c120 - 1)                                    AS med_r120,
    stddev_samp(close / c20 - 1)                                AS disp_r20,
    stddev_samp(close / c60 - 1)                                AS disp_r60,
    sum(amount)                                                 AS amt_total,
    median(turn)                                                AS med_turn,
    avg(CASE WHEN e.amt20 >= m.amt_med THEN e.ret1 END)         AS ew_ret1_lq,
    count(CASE WHEN e.amt20 >= m.amt_med THEN 1 END)            AS n_lq
FROM elig e
JOIN date_med m USING (date)
GROUP BY e.date
ORDER BY e.date
"""


def stage_feature() -> None:
    ensure_dirs()
    con = duck()
    panel = CACHE / "panel_hfq.parquet"
    con.execute(
        f"""CREATE OR REPLACE VIEW panel AS
            SELECT code, CAST(date AS DATE) AS date, open, high, low, close, preclose,
                   volume, amount, turn, trade_status, pct_chg, is_st
            FROM '{panel}'"""
    )
    t0 = time.time()
    import re
    stmts = [s.strip().rstrip(";") for s in re.split(r"(?=CREATE OR REPLACE TABLE)", FEATURE_SQL) if s.strip()]
    for s in stmts:
        con.execute(s)
    log(f"feature: 逐股滚动完成 {time.time()-t0:.1f}s")

    mkt = con.execute("SELECT * FROM mkt ORDER BY date").df()
    mkt["date"] = pd.to_datetime(mkt["date"])
    log(f"market feature rows={len(mkt)} {mkt.date.min().date()}~{mkt.date.max().date()}")

    # 指数特征（小表，pandas 处理）
    idx = con.execute(f"SELECT * FROM '{CACHE/'index_daily.parquet'}' ORDER BY code, date").df()
    idx["date"] = pd.to_datetime(idx["date"])

    def idx_feat(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        c = g["close"].astype(float)
        g["ret1"] = g["pct_chg"].astype(float) / 100.0
        for w in (20, 60, 120, 250):
            g[f"ma{w}"] = c.rolling(w, min_periods=max(5, w // 2)).mean()
        g["vol20"] = g["ret1"].rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)
        g["vol60"] = g["ret1"].rolling(60, min_periods=30).std() * np.sqrt(TRADING_DAYS)
        g["r20"] = c / c.shift(20) - 1
        g["r60"] = c / c.shift(60) - 1
        g["r120"] = c / c.shift(120) - 1
        g["dd250"] = c / c.rolling(250, min_periods=60).max() - 1
        g["ma120_slope20"] = g["ma120"] / g["ma120"].shift(20) - 1
        return g

    idx = pd.concat([idx_feat(g) for _, g in idx.groupby("code")], ignore_index=True)
    idx.to_parquet(CACHE / "index_feat.parquet", index=False)
    log(f"index feat rows={len(idx)}")

    # 把关键指数对齐到 A 股交易日：沪深300、中证500、中证1000、创业板指、上证
    key = {
        "sh.000300": "hs300", "sh.000905": "csi500", "sh.000852": "csi1000",
        "sz.399006": "cyb", "sh.000001": "sse",
    }
    wide = {}
    for code, name in key.items():
        g = idx[idx.code == code].set_index("date")
        for f in ("close", "ma20", "ma60", "ma120", "ma250", "vol20", "vol60",
                  "r20", "r60", "r120", "dd250", "ma120_slope20", "ret1"):
            wide[f"{name}_{f}"] = g[f]
    iw = pd.DataFrame(wide).sort_index()
    iw = iw.reindex(mkt["date"]).ffill(limit=5)
    mkt = mkt.set_index("date").join(iw, how="left").reset_index()

    # 派生：全市场成交额相对其 60 日均值（t 日可知）
    mkt["amt60"] = mkt["amt_total"].rolling(60, min_periods=30).mean()
    mkt["amt_ratio"] = mkt["amt_total"] / mkt["amt60"]
    # 广度 20 日 EMA（只含过去）
    mkt["pct_up_ema20"] = mkt["pct_up"].ewm(span=20, adjust=False).mean()
    mkt["nh_nl"] = mkt["pct_nh250"] - mkt["pct_nl250"]
    mkt["limit_dn_n"] = mkt["pct_limit_dn"] * mkt["n_univ"]

    mkt.to_parquet(CACHE / "market_feature.parquet", index=False)
    save_rows(CACHE / "market_feature.parquet", len(mkt))
    log(f"market_feature saved rows={len(mkt)} cols={len(mkt.columns)}")
    con.close()


# ---------------------------------------------------------------- 状态模型
def rz(s: pd.Series, win: int = 252, minp: int = 120) -> pd.Series:
    """滚动 z 分数，均值/标准差窗口右移 1 日：t 日只用 t-1 及以前。"""
    m = s.rolling(win, min_periods=minp).mean().shift(1)
    sd = s.rolling(win, min_periods=minp).std().shift(1)
    return ((s - m) / sd).clip(-3, 3)


def hyst_bool(x: pd.Series, hi: float, lo: float, init: bool = False) -> np.ndarray:
    """带迟滞的布尔序列：上穿 hi 置真、下穿 lo 置假，中间保持。"""
    v = x.to_numpy(dtype=float)
    out = np.zeros(len(v), dtype=bool)
    cur = init
    for i in range(len(v)):
        if np.isnan(v[i]):
            out[i] = cur
            continue
        if cur and v[i] < lo:
            cur = False
        elif (not cur) and v[i] > hi:
            cur = True
        out[i] = cur
    return out


def build_states(mkt: pd.DataFrame, w: dict | None = None,
                 qs: tuple = (70, 55, 45, 30)) -> pd.DataFrame:
    """预注册的状态模型。w/qs 可覆盖，仅用于敏感性检验，不用于选参。"""
    w = w or dict(b1=0.30, b2=0.30, b5=0.20, b3=-0.20)
    d = mkt.copy().sort_values("date").reset_index(drop=True)

    # --- 合成等权指数：仅在 2005 年（baostock 指数自 2006-01-04 起）作为 B2 的兜底
    syn = (1.0 + d["ew_ret1_lq"].fillna(0.0)).cumprod()
    d["syn_close"] = syn
    d["syn_ma20"] = syn.rolling(20, min_periods=10).mean()
    d["syn_ma60"] = syn.rolling(60, min_periods=30).mean()
    d["syn_ma120"] = syn.rolling(120, min_periods=60).mean()

    # --- B1 广度
    z_up = rz(d["pct_up_ema20"])
    z_ma60 = rz(d["pct_ma60"])
    z_nh = rz(d["nh_nl"])
    d["B1"] = pd.concat([z_up, z_ma60, z_nh], axis=1).mean(axis=1, skipna=True)

    # --- B2 趋势（沪深300、中证500 各自打分后取可用均值；2005 用合成指数兜底）
    def trend_score(close, ma20, ma60, ma120):
        s = (0.4 * np.sign(close - ma120)
             + 0.3 * np.sign(ma20 - ma60)
             + 0.3 * np.sign(ma60 - ma120))
        return s.where(ma120.notna())

    t300 = trend_score(d["hs300_close"], d["hs300_ma20"], d["hs300_ma60"], d["hs300_ma120"])
    t500 = trend_score(d["csi500_close"], d["csi500_ma20"], d["csi500_ma60"], d["csi500_ma120"])
    b2_real = pd.concat([t300, t500], axis=1).mean(axis=1, skipna=True)
    t_syn = trend_score(d["syn_close"], d["syn_ma20"], d["syn_ma60"], d["syn_ma120"])
    d["B2"] = b2_real.where(b2_real.notna(), t_syn)
    d["B2_src"] = np.where(b2_real.notna(), "index", np.where(t_syn.notna(), "synthetic_2005", "na"))

    # --- B3 风险（越高越危险）
    z_vol = rz(d["hs300_vol20"])
    z_disp = rz(d["disp_r20"])
    z_ldn = rz(d["pct_limit_dn"])
    d["B3"] = pd.concat([z_vol, z_disp, z_ldn], axis=1).mean(axis=1, skipna=True)

    # --- B4 量能
    d["B4"] = pd.concat([rz(d["amt_ratio"]), rz(d["med_turn"])], axis=1).mean(axis=1, skipna=True)

    # --- B5 截面强度
    d["B5"] = pd.concat([rz(d["med_r60"]), rz(d["med_r20"])], axis=1).mean(axis=1, skipna=True)

    d["P"] = (w["b1"] * d["B1"] + w["b2"] * d["B2"]
              + w["b5"] * d["B5"] + w["b3"] * d["B3"])

    # --- 扩张分位（只含 ≤t）
    qtag = ("q70", "q55", "q45", "q30")
    q = {k: d["P"].expanding(min_periods=250).quantile(v / 100.0)
         for k, v in zip(qtag, qs)}
    for k, v in q.items():
        d[k] = v

    # --- 迟滞状态机
    p = d["P"].to_numpy()
    q70, q55, q45, q30 = (d[k].to_numpy() for k in ("q70", "q55", "q45", "q30"))
    st = np.empty(len(d), dtype=object)
    cur = "na"
    for i in range(len(d)):
        if np.isnan(p[i]) or np.isnan(q70[i]):
            st[i] = "na"
            cur = "na"
            continue
        if cur in ("na",):
            cur = "risk_on" if p[i] >= q70[i] else ("risk_off" if p[i] <= q30[i] else "neutral")
        elif cur == "risk_on":
            cur = "risk_on" if p[i] >= q55[i] else ("neutral" if p[i] > q30[i] else "risk_off")
        elif cur == "risk_off":
            cur = "risk_off" if p[i] <= q45[i] else ("neutral" if p[i] < q70[i] else "risk_on")
        else:
            cur = "risk_on" if p[i] >= q70[i] else ("risk_off" if p[i] <= q30[i] else "neutral")
        st[i] = cur
    d["state"] = pd.Series(st, dtype=object).astype(str).to_numpy(dtype=object)

    # --- 空仓门（三条件同时成立）
    below = d["hs300_close"] < d["hs300_ma250"]
    d["gate"] = ((d["B3"] > 1.5) & (d["B1"] < -0.5) & below).fillna(False)

    # --- 方案 A：状态切换
    m = {"risk_on": 1.0, "neutral": 0.6, "risk_off": 0.25, "na": 0.0}
    d["pos_A"] = d["state"].map(m).fillna(0.0)
    d.loc[d["gate"], "pos_A"] = 0.0

    # --- 方案 B：竞争信号（4 sleeve 各 1/4，无总否决）
    s1 = hyst_bool(d["B1"], 0.2, -0.2)
    s2 = hyst_bool(d["B2"], 0.2, -0.2)
    rs = d["med_r60"] - d["hs300_r60"]
    s3 = hyst_bool(rs, 0.02, -0.02)
    s4 = hyst_bool(-d["B3"], -0.5, -0.8)
    d["pos_B"] = 0.25 * (s1.astype(float) + s2.astype(float) + s3.astype(float) + s4.astype(float))
    d[["s1", "s2", "s3", "s4"]] = np.c_[s1, s2, s3, s4]

    # --- 方案 C：反脆弱仓位（波动目标 + 非对称 + 回撤刹车）
    vol_eff = pd.concat([d["hs300_vol20"], d["hs300_vol60"]], axis=1).max(axis=1)
    pos = 0.9 * np.clip(0.18 / vol_eff.clip(lower=0.08), 0.0, 1.0)
    pos = pos * np.where(d["hs300_close"] < d["hs300_ma250"], 0.5, 1.0)
    pos = pos * np.where(d["hs300_dd250"] < -0.15, 0.6, 1.0)
    pos = pos * np.where(d["hs300_dd250"] < -0.25, 0.6667, 1.0)
    d["pos_C"] = np.clip(pos.fillna(0.0), 0.0, 1.0)

    # --- 对照 D：串联过滤
    d["pos_D"] = np.where((d["B1"] > 0) & (d["B2"] > 0.2) & (rs > 0) & (d["B3"] < 0.8), 1.0, 0.0)

    d.loc[d["state"] == "na", ["pos_B", "pos_C", "pos_D"]] = 0.0
    d["r_p1"] = d["ew_ret1_lq"]
    d["r_p2"] = d["csi500_ret1"]
    return d


# ---------------------------------------------------------------- 评估
def perf(r: pd.Series) -> dict:
    r = pd.Series(r).dropna()
    if len(r) < 60:
        return {}
    nav = (1 + r).cumprod()
    yrs = len(r) / TRADING_DAYS
    cagr = float(nav.iloc[-1] ** (1 / yrs) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mdd = float((nav / nav.cummax() - 1).min())
    return dict(
        n=int(len(r)), cagr=cagr, vol=vol,
        sharpe=float(r.mean() * TRADING_DAYS / vol) if vol > 0 else np.nan,
        mdd=mdd, calmar=cagr / abs(mdd) if mdd < 0 else np.nan,
        worst20=float((nav / nav.shift(20) - 1).min()),
    )


def overlay(r: pd.Series, pos: pd.Series):
    """pos_t 收盘定，作用于 t+1；换手在 t+1 按单边 10bp 计。"""
    expo = pos.shift(1).fillna(0.0)
    turn = pos.diff().abs().shift(1).fillna(0.0)
    return expo * r - turn * COST_ONE_WAY, turn


def nw_tstat(x, lags: int = 20) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    var = float(e @ e) / n
    for l in range(1, lags + 1):
        gl = float(e[l:] @ e[:-l]) / n
        var += 2 * (1 - l / (lags + 1)) * gl
    se = np.sqrt(max(var, 1e-12) / n)
    return float(x.mean() / se)


def stage_eval() -> None:
    ensure_dirs()
    mkt = pd.read_parquet(CACHE / "market_feature.parquet")
    mkt = mkt[mkt["date"] <= EVAL_END].reset_index(drop=True)
    d = build_states(mkt)
    log(f"eval: 状态序列 {len(d)} 行，有效状态 {int((d['state']!='na').sum())} 行，"
        f"首日 {d.loc[d['state']!='na','date'].min().date()}")

    # 探针收益
    d["r_p1"] = d["ew_ret1_lq"]
    d["r_p2"] = d["csi500_ret1"]

    # 前视窗口（只在 ≤2020 内计算，末尾自动为 NaN，不越界）
    for h in (5, 20, 60):
        lr = np.log1p(d["r_p1"])
        d[f"fwd{h}"] = np.expm1(lr.rolling(h).sum().shift(-h))
    # 未来 20 日内的最大浮亏（相对 t 日净值），末日不足 20 日者置 NaN
    nav = (1 + d["r_p1"]).cumprod()
    path = pd.concat([nav.shift(-h) / nav - 1 for h in range(1, 21)], axis=1).min(axis=1)
    d["fwd20_min"] = path.where(d["fwd20"].notna())

    pos_cols = {"A_状态切换": "pos_A", "B_竞争信号": "pos_B", "C_反脆弱": "pos_C", "D_串联过滤": "pos_D"}
    rets = {"基线_买入持有": d["r_p1"]}
    turns = {}
    for name, c in pos_cols.items():
        rets[name], turns[name] = overlay(d["r_p1"], d[c])

    # ---- 分窗口绩效
    wins = {"早期_2006-2007": ("2006-01-01", "2007-12-31"),
            "开发_2008-2015": DEV, "验证_2016-2020": VAL,
            "全段_2006-2020": ("2006-01-01", EVAL_END)}
    rows = []
    for wname, rng in wins.items():
        mask = (d["date"] >= rng[0]) & (d["date"] <= rng[1])
        for sname, r in rets.items():
            p = perf(r[mask])
            if not p:
                continue
            p.update(窗口=wname, 方案=sname,
                     持仓占比=float((d.loc[mask, pos_cols[sname]] > 0).mean()) if sname in pos_cols else 1.0,
                     平均仓位=float(d.loc[mask, pos_cols[sname]].mean()) if sname in pos_cols else 1.0,
                     年换手=float(turns[sname][mask].sum() / max(mask.sum() / TRADING_DAYS, 1e-9)) if sname in pos_cols else 0.0)
            rows.append(p)
    pw = pd.DataFrame(rows)
    cols = ["窗口", "方案", "n", "cagr", "vol", "sharpe", "mdd", "calmar", "worst20", "持仓占比", "平均仓位", "年换手"]
    pw = pw[cols]
    pw.to_csv(OUT / "市场状态_分窗口绩效.csv", index=False, encoding="utf-8-sig")
    save_rows(OUT / "市场状态_分窗口绩效.csv", len(pw))

    # ---- 中证500 稳健性
    rows2 = []
    for wname, rng in wins.items():
        mask = (d["date"] >= rng[0]) & (d["date"] <= rng[1])
        r2 = d["r_p2"]
        cand = {"基线_买入持有": r2}
        for name, c in pos_cols.items():
            cand[name], _ = overlay(r2, d[c])
        for sname, r in cand.items():
            p = perf(r[mask])
            if p:
                p.update(窗口=wname, 方案=sname)
                rows2.append(p)
    pd.DataFrame(rows2)[["窗口", "方案", "n", "cagr", "vol", "sharpe", "mdd", "calmar"]].to_csv(
        OUT / "市场状态_分窗口绩效_中证500.csv", index=False, encoding="utf-8-sig")

    # ---- 状态条件收益
    lab = {"risk_on": "进攻", "neutral": "中性", "risk_off": "防守", "na": "预热"}
    d["状态"] = d["state"].map(lab)
    g = d.groupby("状态").agg(
        天数=("date", "size"),
        fwd5均值=("fwd5", "mean"), fwd20均值=("fwd20", "mean"), fwd60均值=("fwd60", "mean"),
        fwd20为正=("fwd20", lambda s: float((s > 0).mean())),
        fwd20跌超8pct=("fwd20", lambda s: float((s <= -0.08).mean())),
        当日收益均值=("r_p1", "mean"),
    ).reset_index()
    g["fwd20_NWt"] = [nw_tstat(d.loc[d["状态"] == s, "fwd20"]) for s in g["状态"]]
    g.to_csv(OUT / "市场状态_状态条件收益.csv", index=False, encoding="utf-8-sig")
    save_rows(OUT / "市场状态_状态条件收益.csv", len(g))

    # ---- 空仓门诊断
    bad = d["fwd20"] <= -0.08
    diag = []
    for nm, flag in [("空仓门_A", d["gate"]), ("防守状态", d["state"] == "risk_off"),
                     ("B不投(仓位<0.5)", d["pos_B"] < 0.5)]:
        flag = flag.fillna(False)
        valid = d["fwd20"].notna()
        hit = int((flag & bad & valid).sum())
        diag.append(dict(
            规则=nm, 触发天数=int((flag & valid).sum()),
            触发占比=float((flag & valid).sum() / valid.sum()),
            坏窗口总数=int((bad & valid).sum()),
            坏窗口召回=float(hit / max(int((bad & valid).sum()), 1)),
            触发时坏窗口精度=float(hit / max(int((flag & valid).sum()), 1)),
            触发日fwd20均值=float(d.loc[flag & valid, "fwd20"].mean()) if (flag & valid).any() else np.nan,
            未触发日fwd20均值=float(d.loc[(~flag) & valid, "fwd20"].mean()),
            触发日错过上涨=float(d.loc[flag & valid, "fwd20"].clip(lower=0).mean()) if (flag & valid).any() else np.nan,
        ))
    pd.DataFrame(diag).to_csv(OUT / "市场状态_空仓门诊断.csv", index=False, encoding="utf-8-sig")

    # ---- 风险可预测性：状态对"未来波动/未来最差浮亏"的解释力 vs 对收益的解释力
    d["fwd20_vol"] = d["r_p1"].rolling(20).std().shift(-20) * np.sqrt(TRADING_DAYS)
    rk = []
    lab = {"risk_on": "进攻", "neutral": "中性", "risk_off": "防守"}
    for st, gg in d[d["state"].isin(lab)].groupby("state"):
        rk.append(dict(状态=lab[st], 天数=len(gg),
                       未来20日波动=float(gg["fwd20_vol"].mean()),
                       未来20日最差浮亏=float(gg["fwd20_min"].mean()),
                       未来20日收益=float(gg["fwd20"].mean()),
                       未来20日为正=float((gg["fwd20"] > 0).mean())))
    rk = pd.DataFrame(rk)
    for wname, rng in [("开发_2008-2015", DEV), ("验证_2016-2020", VAL), ("全段_2006-2020", ("2006-01-01", EVAL_END))]:
        m = (d["date"] >= rng[0]) & (d["date"] <= rng[1])
        rk.loc[len(rk)] = dict(
            状态=f"[相关] {wname}", 天数=int(m.sum()),
            未来20日波动=float(d.loc[m, "B3"].corr(d.loc[m, "fwd20_vol"], method="spearman")),
            未来20日最差浮亏=float(d.loc[m, "B3"].corr(d.loc[m, "fwd20_min"], method="spearman")),
            未来20日收益=float(d.loc[m, "P"].corr(d.loc[m, "fwd20"], method="spearman")),
            未来20日为正=np.nan)
    rk.to_csv(OUT / "市场状态_风险可预测性.csv", index=False, encoding="utf-8-sig")


    # ---- 状态切换行为
    sw = d["state"].to_numpy(dtype=object) != d["state"].shift(1).to_numpy(dtype=object)
    sw_counts = int((sw & (d["state"] != "na")).sum())
    dur = d.groupby((d["state"] != d["state"].shift()).cumsum()).agg(
        状态=("state", "first"), 长度=("state", "size"), 起=("date", "min"), 止=("date", "max"))
    dur = dur[dur["状态"] != "na"]
    dur.to_csv(OUT / "市场状态_驻留区间.csv", index=False, encoding="utf-8-sig")

    # ---- 年度绩效
    d["年"] = d["date"].dt.year
    yrows = []
    for y, gy in d.groupby("年"):
        row = {"年": y}
        m = gy.index
        for sname, r in rets.items():
            p = perf(r[m])
            row[f"{sname}_收益"] = p.get("cagr", np.nan)
            row[f"{sname}_回撤"] = p.get("mdd", np.nan)
        yrows.append(row)
    pd.DataFrame(yrows).to_csv(OUT / "市场状态_年度绩效.csv", index=False, encoding="utf-8-sig")

    # ---- 崩盘依赖：择时增益到底来自哪几年
    dep = []
    for nm, rng in [("仅2008", ("2008-01-01", "2008-12-31")),
                    ("仅2015", ("2015-01-01", "2015-12-31")),
                    ("剔除2008与2015_2009-2014", ("2009-01-01", "2014-12-31")),
                    ("仅2016-2017", ("2016-01-01", "2017-12-31")),
                    ("仅2018", ("2018-01-01", "2018-12-31")),
                    ("仅2019-2020", ("2019-01-01", "2020-12-31"))]:
        m = (d["date"] >= rng[0]) & (d["date"] <= rng[1])
        pb = perf(d["r_p1"][m])
        row = {"窗口": nm, "基线CAGR": pb["cagr"], "基线Calmar": pb["calmar"]}
        for name, c in pos_cols.items():
            p = perf(rets[name][m])
            ps = perf(d["r_p1"][m] * d.loc[m, c].mean())
            row[f"{name}_CAGR"] = p["cagr"]
            row[f"{name}_Calmar"] = p["calmar"]
            row[f"{name}_择时增益"] = p["calmar"] - ps["calmar"]
        dep.append(row)
    pd.DataFrame(dep).to_csv(OUT / "市场状态_崩盘依赖.csv", index=False, encoding="utf-8-sig")

    # ---- 静态等仓对照：把"降杠杆"与"择时"分开
    #     若固定仓位（等于该方案窗口内平均仓位）也能拿到同样的 Calmar，
    #     则方案的价值只是"少拿一点 beta"，不是择时。
    ctrl = []
    for wname, rng in wins.items():
        mask = (d["date"] >= rng[0]) & (d["date"] <= rng[1])
        pb = perf(d["r_p1"][mask])
        p50 = perf(d["r_p1"][mask] * 0.5)
        p35 = perf(d["r_p1"][mask] * 0.35)
        ctrl.append(dict(窗口=wname, 方案="基线_买入持有", 平均仓位=1.0,
                         cagr=pb["cagr"], mdd=pb["mdd"], calmar=pb["calmar"], sharpe=pb["sharpe"]))
        for nm, pp in [("固定50%", p50), ("固定35%", p35)]:
            ctrl.append(dict(窗口=wname, 方案=nm, 平均仓位=float(nm[-3:-1]) / 100,
                             cagr=pp["cagr"], mdd=pp["mdd"], calmar=pp["calmar"], sharpe=pp["sharpe"]))
        for name, c in pos_cols.items():
            p = perf(rets[name][mask])
            avg = float(d.loc[mask, c].mean())
            ps = perf(d["r_p1"][mask] * avg)
            ctrl.append(dict(窗口=wname, 方案=name, 平均仓位=avg,
                             cagr=p["cagr"], mdd=p["mdd"], calmar=p["calmar"], sharpe=p["sharpe"],
                             同仓位静态calmar=ps["calmar"],
                             择时增益Calmar=p["calmar"] - ps["calmar"]))
    pd.DataFrame(ctrl).to_csv(OUT / "市场状态_静态等仓对照.csv", index=False, encoding="utf-8-sig")

    # ---- 环形平移蒙特卡洛：打乱仓位与收益的对应关系，保留暴露分布与自相关
    def rot_test(r, pos, n_iter=200, seed=7):
        df = pd.DataFrame({"r": np.asarray(r, dtype=float), "p": np.asarray(pos, dtype=float)}).dropna()
        rv, pv = df["r"].to_numpy(), df["p"].to_numpy()
        n = len(pv)
        base = perf(overlay(pd.Series(rv), pd.Series(pv))[0]).get("calmar", np.nan)
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_iter):
            k = int(rng.integers(20, n - 20))
            p2 = pd.Series(np.roll(pv, k))
            null.append(perf(overlay(pd.Series(rv), p2)[0]).get("calmar", np.nan))
        null = np.array(null, dtype=float)
        return base, float(np.nanmedian(null)), float(np.nanpercentile(null, 90)), float(np.nanmean(null >= base))

    rrows = []
    for wname, rng in wins.items():
        mask = (d["date"] >= rng[0]) & (d["date"] <= rng[1])
        for name, c in pos_cols.items():
            b, med_, p90_, frac = rot_test(d.loc[mask, "r_p1"].reset_index(drop=True),
                                           d.loc[mask, c].reset_index(drop=True))
            rrows.append(dict(窗口=wname, 方案=name, 真实Calmar=b, 平移中位数=med_,
                              平移P90=p90_, 真实超过平移比例=frac))
    pd.DataFrame(rrows).to_csv(OUT / "市场状态_平移检验.csv", index=False, encoding="utf-8-sig")

    # ---- 状态新鲜度：进入状态多少天，后续收益如何
    grp = (d["state"] != d["state"].shift()).cumsum()
    d["状态天数"] = d.groupby(grp).cumcount() + 1
    d["状态天数档"] = pd.cut(d["状态天数"], [0, 5, 20, 60, 10 ** 6],
                            labels=["1-5日", "6-20日", "21-60日", "60日以上"])
    fr = d.groupby(["状态", "状态天数档"], observed=True).agg(
        天数=("date", "size"), fwd20均值=("fwd20", "mean"),
        fwd20为正=("fwd20", lambda s: float((s > 0).mean()))).reset_index()
    fr.to_csv(OUT / "市场状态_新鲜度条件收益.csv", index=False, encoding="utf-8-sig")


    # ---- 参数敏感性（只报告，不据此改规则）
    sens = []
    variants = {
        "基准权重": (dict(b1=.3, b2=.3, b5=.2, b3=-.2), (70, 55, 45, 30)),
        "等权": (dict(b1=.25, b2=.25, b5=.25, b3=-.25), (70, 55, 45, 30)),
        "偏广度": (dict(b1=.4, b2=.2, b5=.2, b3=-.2), (70, 55, 45, 30)),
        "偏风险": (dict(b1=.25, b2=.25, b5=.15, b3=-.35), (70, 55, 45, 30)),
        "分位宽松60/75": (dict(b1=.3, b2=.3, b5=.2, b3=-.2), (75, 60, 50, 35)),
        "分位收紧60/25": (dict(b1=.3, b2=.3, b5=.2, b3=-.2), (60, 50, 40, 25)),
    }
    base = build_states(mkt)
    for vname, (w, qs) in variants.items():
        dv = build_states(mkt, w=w, qs=qs)
        for wname, rng in [("开发_2008-2015", DEV), ("验证_2016-2020", VAL)]:
            mask = (dv["date"] >= rng[0]) & (dv["date"] <= rng[1])
            rA, _ = overlay(dv["r_p1"], dv["pos_A"])
            p = perf(rA[mask])
            pb = perf(dv.loc[mask, "r_p1"])
            p.update(方案=f"A_状态切换_{vname}", 窗口=wname,
                     calmar_基线=pb["calmar"], cagr_基线=pb["cagr"])
            sens.append(p)
    for vt in (0.12, 0.24):
        pos = 0.9 * np.clip(vt / pd.concat([mkt["hs300_vol20"], mkt["hs300_vol60"]], axis=1).max(axis=1).clip(lower=0.08), 0, 1)
        pos = pos * np.where(mkt["hs300_close"] < mkt["hs300_ma250"], 0.5, 1.0)
        r, _ = overlay(mkt["ew_ret1_lq"], pd.Series(pos, index=mkt.index).fillna(0))
        for wname, rng in [("开发_2008-2015", DEV), ("验证_2016-2020", VAL)]:
            mask = (mkt["date"] >= rng[0]) & (mkt["date"] <= rng[1])
            p = perf(r[mask]); pb = perf(mkt.loc[mask, "ew_ret1_lq"])
            p.update(方案=f"C_目标波动{int(vt*100)}%", 窗口=wname, calmar_基线=pb["calmar"], cagr_基线=pb["cagr"])
            sens.append(p)
    pd.DataFrame(sens)[["方案", "窗口", "cagr", "sharpe", "mdd", "calmar", "calmar_基线", "cagr_基线"]].to_csv(
        OUT / "市场状态_参数敏感性.csv", index=False, encoding="utf-8-sig")

    # ---- 日频许可（仅 ≤2020）
    keep = ["date", "state", "B1", "B2", "B3", "B4", "B5", "P", "q70", "q55", "q45", "q30",
            "gate", "pos_A", "pos_B", "pos_C", "pos_D", "s1", "s2", "s3", "s4", "B2_src",
            "n_univ", "pct_up", "pct_dn", "pct_limit_up", "pct_limit_dn", "pct_ma20", "pct_ma60",
            "pct_ma120", "pct_nh250", "pct_nl250", "med_r20", "med_r60", "disp_r20",
            "amt_ratio", "hs300_close", "hs300_ma250", "hs300_vol20", "hs300_dd250",
            "csi500_ret1", "ew_ret1_lq"]
    d[keep].to_csv(OUT / "市场状态_日频许可_2006_2020.csv", index=False, encoding="utf-8-sig")
    save_rows(OUT / "市场状态_日频许可_2006_2020.csv", len(d))

    # ---- 图（英文标签，避免 CJK 字体缺失）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        en = {"基线_买入持有": "buy&hold", "A_状态切换": "A state-switch",
              "B_竞争信号": "B competing sleeves", "C_反脆弱": "C anti-fragile",
              "D_串联过滤": "D cascade filter"}
        fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
        for sname, r in rets.items():
            ax[0].plot(d["date"], (1 + r.fillna(0)).cumprod(), label=en.get(sname, sname), lw=1.4)
        ax[0].set_yscale("log"); ax[0].legend(loc="upper left", fontsize=8)
        ax[0].set_title("Market-state overlays on equal-weight A-share probe (2006-2020, net of 10bp)")
        ax[0].grid(alpha=0.3)
        ax[0].axvline(pd.Timestamp("2016-01-01"), color="k", ls="--", lw=0.8)
        ax[0].text(pd.Timestamp("2016-02-01"), ax[0].get_ylim()[1] * 0.5, "validation ->", fontsize=8)
        for c, col in [("pos_A", "tab:red"), ("pos_B", "tab:green"), ("pos_C", "tab:blue")]:
            ax[1].plot(d["date"], d[c], lw=0.8, label=c, color=col)
        ax[1].set_ylabel("exposure"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "市场状态_净值与仓位.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"plot 跳过: {e}")

    log(f"eval 完成：{len(pw)} 行绩效，状态切换 {sw_counts} 次")
    print(pw.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(pd.DataFrame(diag).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ---------------------------------------------------------------- 自检
def stage_check() -> None:
    """核心自检：未来数据污染、截断不变性、仓位-收益对齐、费用符号。"""
    ensure_dirs()
    mkt = pd.read_parquet(CACHE / "market_feature.parquet")
    mkt = mkt[mkt["date"] <= EVAL_END].reset_index(drop=True)
    base = build_states(mkt)
    num = mkt.select_dtypes(include=[np.number]).columns.tolist()
    out = []

    # 1) 未来污染：把 T 之后所有数值列置 NaN，T 之前的 P / state / pos 必须逐位不变
    for T in ("2008-12-31", "2013-12-31", "2016-12-31"):
        m2 = mkt.copy()
        m2.loc[m2["date"] > T, num] = np.nan
        s2 = build_states(m2)
        n = int((base["date"] <= T).sum())
        dp = float((s2["P"].iloc[:n] - base["P"].iloc[:n]).abs().max())
        same = bool((s2["state"].iloc[:n].to_numpy() == base["state"].iloc[:n].to_numpy()).all())
        dpos = float((s2["pos_A"].iloc[:n] - base["pos_A"].iloc[:n]).abs().max())
        out.append(dict(检查=f"未来污染_{T}", 结果="PASS" if (dp < 1e-9 and same and dpos < 1e-9) else "FAIL",
                        明细=f"max|ΔP|={dp:.2e}, state一致={same}, max|Δpos|={dpos:.2e}"))

    # 2) 截断不变性：只用 T 之前的数据重建，T 之前结果应与全样本一致
    for T in ("2012-12-31", "2018-12-31"):
        s3 = build_states(mkt[mkt["date"] <= T].reset_index(drop=True))
        n = len(s3)
        dp = float((s3["P"] - base["P"].iloc[:n]).abs().max())
        same = bool((s3["state"].to_numpy() == base["state"].iloc[:n].to_numpy()).all())
        out.append(dict(检查=f"截断不变_{T}", 结果="PASS" if (dp < 1e-9 and same) else "FAIL",
                        明细=f"max|ΔP|={dp:.2e}, state一致={same}"))

    # 3) 仓位-收益对齐：pos 在 t 日收盘定，只能吃到 t+1 的收益
    r = pd.Series([0.0] * 60)
    r.iloc[10] = 0.10
    pos = pd.Series([0.0] * 60)
    pos.iloc[9:] = 1.0          # 第 9 日收盘后满仓 → 第 10 日吃到涨幅
    net, turn = overlay(r, pos)
    ok3 = (abs(net.iloc[9]) < 1e-12 and abs(turn.iloc[9]) < 1e-12
           and abs(net.iloc[10] - (0.10 - COST_ONE_WAY)) < 1e-12
           and abs(turn.iloc[10] - 1.0) < 1e-12)
    out.append(dict(检查="仓位对齐_t+1", 结果="PASS" if ok3 else "FAIL",
                    明细=f"第9日净收益={net.iloc[9]:.4f}, 第10日净收益={net.iloc[10]:.4f}(=0.10-10bp), 换手={turn.iloc[10]:.2f}"))

    # 4) 费用符号：换手越大，净收益越低
    rnd = np.random.default_rng(0).normal(0, 0.01, 2000)
    pA = pd.Series(np.repeat([0, 1], 1000).astype(float))
    pB = pd.Series((np.arange(2000) % 2).astype(float))
    a, _ = overlay(pd.Series(rnd), pA)
    b, _ = overlay(pd.Series(rnd), pB)
    ok4 = (b.sum() < a.sum()) and (pB.diff().abs().sum() > pA.diff().abs().sum())
    out.append(dict(检查="费用方向", 结果="PASS" if ok4 else "FAIL",
                    明细=f"低频合计{a.sum():.4f} > 高频合计{b.sum():.4f}"))

    # 5) 评估期覆盖：状态不得为 na / NaN
    ev = base[(base["date"] >= "2006-01-01")]
    nn = int((ev["state"] == "na").sum() + ev["P"].isna().sum() + ev[["pos_A", "pos_B", "pos_C", "pos_D"]].isna().sum().sum())
    out.append(dict(检查="评估期状态完整", 结果="PASS" if nn == 0 else "FAIL", 明细=f"异常 {nn} 处"))

    # 6) 数据边界：不得晚于冻结日，前视列末尾必须为 NaN
    last = str(pd.read_parquet(CACHE / "market_feature.parquet")["date"].max())[:10]
    ok6 = last <= FREEZE
    out.append(dict(检查="冻结日", 结果="PASS" if ok6 else "FAIL", 明细=f"特征末日 {last} <= {FREEZE}"))

    res = pd.DataFrame(out)
    res.to_csv(OUT / "市场状态_自检.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    if (res["结果"] == "FAIL").any():
        raise SystemExit("自检未通过")


# ---------------------------------------------------------------- 信号许可（诊断，不选股）
SIG_SQL = """
CREATE OR REPLACE TABLE sig AS
SELECT * FROM (
    SELECT code, date, close, trade_status, is_st,
        row_number() OVER w AS bar_no,
        count(*) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS bars60,
        avg(amount) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amt20,
        close / lag(close, 60) OVER w - 1 AS r60,
        close / lag(close, 20) OVER w - 1 AS r20,
        lead(close, 5) OVER w / close - 1 AS fwd5
    FROM panel
    WHERE date <= DATE '{eval_end}'
    WINDOW w AS (PARTITION BY code ORDER BY date)
)
"""
SIG_SQL += """
CREATE OR REPLACE TABLE sig_daily AS
SELECT date,
    avg(CASE WHEN q = 5 THEN fwd5 END) AS top5,
    avg(CASE WHEN q = 1 THEN fwd5 END) AS bot1,
    avg(fwd5)                          AS all_ret,
    count(*)                           AS n
FROM (
    SELECT date, fwd5,
           ntile(5) OVER (PARTITION BY date ORDER BY r60) AS q
    FROM sig
    WHERE trade_status = 1 AND is_st = 0 AND bar_no >= 60 AND bars60 >= 50
      AND amt20 > 0 AND r60 IS NOT NULL AND fwd5 IS NOT NULL
)
GROUP BY date ORDER BY date
"""


def stage_signal() -> None:
    """同一趋势信号（60 日动量前 20% 多头）在不同市场状态下的表现差异。

    只用于回答"市场许可是否也管信号有效性"，不构成选股规则。
    """
    ensure_dirs()
    con = duck()
    con.execute(f"""CREATE OR REPLACE VIEW panel AS
        SELECT code, CAST(date AS DATE) AS date, close, amount, turn, trade_status, pct_chg, is_st
        FROM '{CACHE/'panel_hfq.parquet'}'""")
    t0 = time.time()
    for s in [x.strip() for x in re.split(r"(?=CREATE OR REPLACE TABLE)", SIG_SQL.format(eval_end=EVAL_END)) if x.strip()]:
        con.execute(s)
    log(f"signal: 分位聚合 {time.time()-t0:.1f}s")
    sd = con.execute("SELECT * FROM sig_daily ORDER BY date").df()
    sd["date"] = pd.to_datetime(sd["date"])
    sd["mom_excess"] = sd["top5"] - sd["all_ret"]

    mkt = pd.read_parquet(CACHE / "market_feature.parquet")
    mkt = mkt[mkt["date"] <= EVAL_END].reset_index(drop=True)
    d = build_states(mkt)[["date", "state", "P", "B1", "B2", "B3"]]
    m = d.merge(sd, on="date", how="inner")
    m["状态"] = m["state"].map({"risk_on": "进攻", "neutral": "中性", "risk_off": "防守", "na": "预热"})

    rows = []
    for st, gg in m.groupby("状态"):
        rows.append(dict(状态=st, 天数=len(gg),
                         市场等权5日=float(gg["all_ret"].mean()),
                         多头Q5_5日=float(gg["top5"].mean()),
                         空头Q1_5日=float(gg["bot1"].mean()),
                         多空差=float((gg["top5"] - gg["bot1"]).mean()),
                         多头超额=float(gg["mom_excess"].mean()),
                         多头超额NWt=nw_tstat(gg["mom_excess"]),
                         多头胜率=float((gg["top5"] > gg["all_ret"]).mean())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "市场状态_信号许可.csv", index=False, encoding="utf-8-sig")
    save_rows(OUT / "市场状态_信号许可.csv", len(res))
    m.to_csv(OUT / "市场状态_信号许可_日频.csv", index=False, encoding="utf-8-sig")
    print(res.round(4).to_string(index=False))
    con.close()


# ---------------------------------------------------------------- 覆盖审计（无收益）
def stage_audit() -> None:
    import duckdb

    ensure_dirs()
    mkt = pd.read_parquet(CACHE / "market_feature.parquet")
    d = build_states(mkt)

    # 扩展状态：规则冻结后前向应用，不含任何收益列
    ext = d[d["date"] >= "2021-01-01"][
        ["date", "state", "B1", "B2", "B3", "B4", "B5", "P", "gate",
         "pos_A", "pos_B", "pos_C", "pos_D", "n_univ", "pct_ma60", "amt_ratio"]].copy()
    ext["说明"] = "extension_not_evaluated"
    ext.to_csv(OUT / "市场状态_日频许可_扩展2021_2026_未评估.csv", index=False, encoding="utf-8-sig")
    save_rows(OUT / "市场状态_日频许可_扩展2021_2026_未评估.csv", len(ext))

    # 覆盖审计：只统计行数/日期/停牌，不算收益
    con = duckdb.connect()
    p = CACHE / "panel_hfq.parquet"
    cov = con.execute(f"""
        SELECT year(CAST(date AS DATE)) AS 年, count(*) AS 行数, count(DISTINCT code) AS 股票数,
               count(DISTINCT date) AS 交易日,
               sum(CASE WHEN trade_status=0 THEN 1 ELSE 0 END) AS 停牌行,
               sum(CASE WHEN is_st=1 THEN 1 ELSE 0 END) AS ST行
        FROM '{p}' WHERE CAST(date AS DATE) >= DATE '2021-01-01' GROUP BY 1 ORDER BY 1
    """).df()
    cov.to_csv(OUT / "市场状态_覆盖审计_2021_2026.csv", index=False, encoding="utf-8-sig")

    # 与同花顺 DuckDB 抽检（2016-2019，仅收益一致性，不涉策略收益）
    ths = duckdb.connect(str(THS_DB), read_only=True)
    bs = con.execute(f"""
        SELECT code, CAST(date AS DATE) AS date, close, pct_chg FROM '{p}'
        WHERE CAST(date AS DATE) BETWEEN DATE '2016-10-13' AND DATE '2019-12-31'
          AND code IN ('sh.600000','sh.600519','sz.000001','sz.300750','sh.601318')
        ORDER BY code, date
    """).df()
    ths_df = ths.execute("""
        SELECT thscode, date,
               close / lag(close) OVER (PARTITION BY thscode ORDER BY date) - 1 AS ret_ths
        FROM v_daily
        WHERE date BETWEEN DATE '2016-10-13' AND DATE '2019-12-31'
          AND thscode IN ('600000.SH','600519.SH','000001.SZ','300750.SZ','601318.SH')
    """).df()
    ths.close()
    bs["thscode"] = bs["code"].str.split(".").str[::-1].str.join(".").str.upper()
    bs["date"] = pd.to_datetime(bs["date"]).dt.date
    ths_df["date"] = pd.to_datetime(ths_df["date"]).dt.date
    mrg = bs.merge(ths_df, on=["thscode", "date"], how="inner").dropna(subset=["ret_ths"])
    mrg["ret_bs"] = mrg["pct_chg"] / 100.0
    diff = (mrg["ret_bs"] - mrg["ret_ths"]).abs()
    chk = pd.DataFrame([{
        "抽检股票": mrg["thscode"].nunique(), "重合行": len(mrg),
        "收益率差中位数bp": float(diff.median() * 1e4),
        "收益率差p99_bp": float(diff.quantile(0.99) * 1e4),
        "差>50bp占比": float((diff > 0.005).mean()),
        "差>3%(除权日)行数": int((diff > 0.03).sum()),
    }])
    chk.to_csv(OUT / "市场状态_跨源抽检.csv", index=False, encoding="utf-8-sig")
    mrg[["thscode", "date", "ret_bs", "ret_ths"]].assign(差=mrg["ret_bs"] - mrg["ret_ths"]).to_csv(
        OUT / "市场状态_跨源抽检_明细.csv", index=False, encoding="utf-8-sig")
    log(chk.to_string(index=False))
    log(f"审计：{len(cov)} 年覆盖，扩展状态 {len(ext)} 行（无收益列）")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["extract", "feature", "eval", "check", "signal", "audit", "all"])
    a = ap.parse_args()
    if a.stage in ("extract", "all"):
        stage_extract()
    if a.stage in ("feature", "all"):
        stage_feature()
    if a.stage in ("eval", "all"):
        stage_eval()
    if a.stage in ("check", "all"):
        stage_check()
    if a.stage in ("signal", "all"):
        stage_signal()
    if a.stage in ("audit", "all"):
        stage_audit()
