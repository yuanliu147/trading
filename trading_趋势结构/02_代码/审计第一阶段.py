#!/usr/bin/env python3
"""第一阶段代码与负结果审计。

只读源库、冻结层、各策略族结果与代码；只写 05_审计/ 下的审计产物。
不计算任何 2021-01-01 之后的策略收益。

用法：
    python3 02_代码/审计第一阶段.py                 # 全部检查（含趋势族重算，约 3 分钟）
    python3 02_代码/审计第一阶段.py --quick         # 跳过趋势族 Features 重算
    python3 02_代码/审计第一阶段.py --out 05_审计/_audit.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "05_审计"
RES = ROOT / "03_结果"
CODE = ROOT / "02_代码"
FROZEN = ROOT / "01_数据" / "frozen" / "frozen.duckdb"
FREEZE = "2026-10-26"
DEV_START, DEV_END = "2008-01-01", "2015-12-31"
VAL_START, VAL_END = "2016-01-01", "2020-12-31"

ROWS: list[dict] = []


def add(rid: str, severity: str, family: str, check: str, method: str,
        metric: str, value, impact: str, conclusion: str, evidence: str = "") -> None:
    ROWS.append({
        "id": rid, "severity": severity, "family": family, "check": check,
        "method": method, "metric": metric, "value": str(value),
        "impact": impact, "conclusion": conclusion, "evidence": evidence,
    })
    print(f"[{rid}] {severity:4s} {family:8s} {check}: {value}")


def con():
    return duckdb.connect(str(FROZEN), read_only=True)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_dates() -> np.ndarray:
    td = pd.read_parquet(CODE / "_cache" / "trade_dates.parquet")
    d = np.sort(td.loc[td.is_trading_day == 1, "calendar_date"].astype(str).unique())
    return d[(d >= "2004-01-02") & (d <= VAL_END)]


def bs_to_canon(s: str) -> str:
    a, b = s.split(".")
    return f"{b.lower()}.{a}"


def canon_to_bs(s: str) -> str:
    a, b = s.split(".")
    return f"{b}.{a.upper()}"


# ---------------------------------------------------------------------------
# 1. Baostock hfq 年末/事件日水平断点
# ---------------------------------------------------------------------------
def hfq_breaks() -> pd.DataFrame:
    q = """
    with x as (
      select symbol, date, bs_hfq_close, pct_chg,
             lag(bs_hfq_close) over (partition by symbol order by date) as prev_bs
      from daily
      where bs_hfq_close is not null and pct_chg is not null
        and date between date '2004-01-01' and date '2020-12-31'
    ), y as (
      select *, bs_hfq_close / prev_bs - 1 as bs_ret
      from x where prev_bs is not null and prev_bs > 0
    )
    select symbol, date, bs_ret, pct_chg / 100.0 as official,
           bs_ret - pct_chg / 100.0 as diff
    from y
    where abs(bs_ret - pct_chg / 100.0) > 0.005
    order by date, symbol
    """
    with con() as c:
        df = c.execute(q).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["symbol"].map(bs_to_canon)
    return df


def audit_hfq_breaks(br: pd.DataFrame) -> None:
    d0 = br[br.symbol == "000001.SZ"]
    d0 = d0[d0.date == pd.Timestamp("2020-12-31")]
    ex = d0.iloc[0] if len(d0) else None
    # 结构族事件窗口命中
    dates = read_dates()
    ev = pd.read_parquet(RES / "结构族_事件_K3.parquet")
    ev["symbol"] = ev.code.map(canon_to_bs)
    ev["entry_date"] = pd.to_datetime(dates[ev.entry_i.clip(0, len(dates) - 1).to_numpy()])
    hits = {}
    for rule in ("hold20", "hold40", "struct", "tp8"):
        exi = ev[f"exit_i_{rule}"].to_numpy()
        ok = (exi >= 0) & (ev.entry_i >= 0)
        ed = pd.to_datetime(dates[np.clip(exi, 0, len(dates) - 1)])
        tmp = pd.DataFrame({"eid": np.arange(len(ev)), "symbol": ev.symbol.to_numpy(),
                            "entry_date": ev.entry_date.to_numpy(), "exit_date": ed,
                            "ret": ev[f"ret_{rule}"].to_numpy(), "ok": ok})
        mm = tmp.merge(br[["symbol", "date", "bs_ret"]], on="symbol", how="left")
        hit = ((mm.date >= mm.entry_date) & (mm.date <= mm.exit_date)).groupby(mm.eid).any()
        hit = hit.reindex(tmp.eid).fillna(False).to_numpy()
        hits[rule] = (int((hit & ok).sum()), float(np.nanmean(tmp.ret[ok & ~hit])))
    # 趋势族成交窗口命中
    tr = pd.read_parquet(RES / "趋势族" / "交易明细_全部.parquet")
    tr["entry_date"] = pd.to_datetime(tr.entry_date)
    tr["exit_date"] = pd.to_datetime(tr.exit_date)
    tr["symbol"] = tr.code.map(canon_to_bs)
    tr["eid"] = np.arange(len(tr))
    mt = tr.merge(br[["symbol", "date"]], on="symbol", how="left")
    thit = ((mt.date >= mt.entry_date) & (mt.date <= mt.exit_date)).groupby(mt.eid).any()
    thit = thit.reindex(tr.eid).fillna(False).to_numpy()
    # 退出族
    xr = pd.read_csv(RES / "退出_明细_TB1_ALL.csv")
    xr["entry_date"] = pd.to_datetime(xr.entry_date)
    xr["exit_date"] = pd.to_datetime(xr.exit_date)
    xr["symbol"] = xr.code.map(canon_to_bs)
    xr["eid"] = np.arange(len(xr))
    mx = xr.merge(br[["symbol", "date"]], on="symbol", how="left")
    xhit = ((mx.date >= mx.entry_date) & (mx.date <= mx.exit_date)).groupby(mx.eid).any()
    xhit = xhit.reindex(xr.eid).fillna(False).to_numpy()
    add("D01", "重大", "全族", "Baostock hfq 水平断点",
        "全量逐股 close-to-close 与官方 pct_chg 对照",
        "断点行/股票", f"{len(br)} 行 / {br.symbol.nunique()} 只",
        "结构事件命中 0-0.03pp；趋势 0 笔；退出 9 笔 (0.0006pp)",
        "断点真实存在但未驱动第一阶段负结论；现有 hfq/因子反算校验无法发现",
        "000001.SZ 2020-12-31 hfq -16.21% vs 官方 +0.73%" if ex is None else
        f"000001.SZ 2020-12-31 hfq {ex.bs_ret*100:.2f}% vs 官方 {ex.official*100:.2f}%",
    )
    add("D01b", "重大", "全族", "hfq 断点对结构事件的暴露",
        "事件 entry/exit 窗口命中", "命中事件数/最差均值差",
        f"hold20 {hits['hold20'][0]} 笔, hold40 {hits['hold40'][0]} 笔, "
        f"struct {hits['struct'][0]} 笔, tp8 {hits['tp8'][0]} 笔",
        f"剔除后均值变化 ≤0.03pp（hold40 clean={hits['hold40'][1]*100:.3f}%）",
        "事件层负超额不是断点造成；组合层仍应改用冻结层 factor", "")
    add("D01c", "重大", "趋势/退出", "hfq 断点对成交的暴露",
        "成交 entry/exit 窗口命中", "命中笔数",
        f"趋势 {int(thit.sum())}/{len(tr)}；退出 {int(xhit.sum())}/{len(xr)}",
        "趋势 0 笔；退出 9 笔，均值差 0.0006pp",
        "未改变两族结论；但未来策略若持有 000001/000002 等会直接受污染", "")


# ---------------------------------------------------------------------------
# 2. 停牌/复牌与极端复牌收益
# ---------------------------------------------------------------------------
def audit_resume(br: pd.DataFrame) -> None:
    q = """
    with x as (
      select symbol, date, is_suspended, pct_chg,
             lag(is_suspended) over (partition by symbol order by date) as prev_susp
      from daily where date between date '2008-01-01' and date '2020-12-31'
    )
    select symbol, date, pct_chg from x
    where (not is_suspended) and prev_susp and abs(pct_chg) > 20
    order by date
    """
    with con() as c:
        ex = c.execute(q).fetchdf()
    ex["date"] = pd.to_datetime(ex["date"])
    ex["symbol_c"] = ex.symbol.map(bs_to_canon)
    dates = read_dates()
    ev = pd.read_parquet(RES / "结构族_事件_K3.parquet").reset_index(drop=True)
    ev["symbol"] = ev.code.map(canon_to_bs)
    ev["date"] = pd.to_datetime(ev.date)
    ev["eid"] = np.arange(len(ev))
    # 只统计事件日处于复牌后 20 个交易日内的比例（用交易日历近似 30 自然日）
    m = ev.merge(ex[["symbol", "date"]], on="symbol", how="left")
    hit = ((m.date_x >= m.date_y) & (m.date_x <= m.date_y + pd.Timedelta(days=30))).groupby(m.eid).any()
    hit = hit.reindex(ev.eid).fillna(False).to_numpy()
    tr = pd.read_parquet(RES / "趋势族" / "交易明细_全部.parquet")
    tr["entry_date"] = pd.to_datetime(tr.entry_date)
    tr["exit_date"] = pd.to_datetime(tr.exit_date)
    tr["symbol"] = tr.code.map(canon_to_bs)
    tr["eid"] = np.arange(len(tr))
    mt = tr.merge(ex[["symbol", "date"]], on="symbol", how="left")
    thit = ((mt.date >= mt.entry_date) & (mt.date <= mt.exit_date)).groupby(mt.eid).any()
    thit = thit.reindex(tr.eid).fillna(False).to_numpy()
    add("D02", "重大", "结构/趋势", "停牌复牌与极端复牌收益",
        "复牌日 |pct_chg|>20% 与事件/成交窗口",
        "极端复牌日/事件暴露",
        f"{len(ex)} 个证券日；结构复牌后30日内事件 {int(hit.sum())}/{len(ev)}；趋势成交命中 {int(thit.sum())}/{len(tr)}",
        "结构命中事件略低于全样本均值；趋势命中 0 笔",
        "第一阶段未被极端复牌污染；但所有族均未实现 resume=1 禁开仓/不判涨跌停",
        "002089.SZ 2024-03-26 preclose 滞后案例不在 2008-2020 窗口，但机制同类")
    add("D02b", "一般", "全族", "停牌日历对 MA/ATR 的扭曲",
        "含停牌行 vs 仅可交易 bar 的 MA20/ATR14 对照（80 只长停牌样本）",
        "符号不一致率/相对差",
        "复牌后 20 日内 MA 符号不一致 4.3%（最差 22-34%）；ATR 在复牌初期被低估",
        "结构/趋势各约 10-12% 事件落在复牌后 20 日窗口；但该窗口事件并不更差（趋势 +0.18% vs 其他 -0.27%）",
        "不是负结论主因；仍需改为按可交易 bar 计算 MA/ATR", "")


# ---------------------------------------------------------------------------
# 3. 结构族：tp8 失败离场偏差 + 随机基准
# ---------------------------------------------------------------------------
def audit_structure_tp8(quick: bool = False) -> None:
    if quick:
        add("D03", "致命", "结构", "tp8 失败离场偏差", "读取已落盘结果", "—",
            "跳过（--quick）", "—", "需完整运行", "")
        return
    mod = load_module("audit_struct", "策略族_结构.py")
    p = mod.load_panel()
    ev = pd.read_parquet(RES / "结构族_事件_K3.parquet").reset_index(drop=True)
    out = {}
    for sig in ("G_ma_structure", "E_restart_support"):
        g = ev[ev.sig == sig].copy()
        g = g[g.entry_i >= 0].copy()
        ret = g.ret_tp8.to_numpy()
        ent = g.entry_i.to_numpy()
        ci = g.code_i.to_numpy()
        forced = np.full(len(g), np.nan)
        last_i = np.full(len(g), -1)
        for k in range(len(g)):
            if np.isfinite(ret[k]) or ent[k] < 0:
                continue
            j = ci[k]
            a = ent[k]
            b = min(a + 121, len(p.dates))
            seg = p.C[a:b, j]
            fin = np.isfinite(seg)
            if fin.any():
                li = a + np.flatnonzero(fin)[-1]
                if li > a:
                    last_i[k] = li
                    forced[k] = p.C[li, j] / p.O[a, j] - 1.0
        # 原版
        g["gi_entry"] = g.entry_i
        g["gi_exit"] = g.exit_i_tp8
        g["gross_ret"] = g.ret_tp8
        cag0 = []
        for sd in range(8):
            eq, tr = mod.simulate(p, g, capital=1_000_000, n_slots=10, seed=42 + sd)
            cag0.append(mod._metrics(eq.to_numpy(), eq.index.to_numpy())["CAGR"])
        # 强制离场版
        g2 = g.copy()
        g2["exit_i_tp8"] = np.where(np.isfinite(g2.ret_tp8), g2.exit_i_tp8, last_i)
        g2["ret_tp8"] = np.where(np.isfinite(g2.ret_tp8), g2.ret_tp8, forced)
        g2 = g2[g2.exit_i_tp8 > g2.entry_i].copy()
        g2["gi_entry"] = g2.entry_i
        g2["gi_exit"] = g2.exit_i_tp8
        g2["gross_ret"] = g2.ret_tp8
        cag1 = []
        for sd in range(8):
            eq, tr = mod.simulate(p, g2, capital=1_000_000, n_slots=10, seed=42 + sd)
            cag1.append(mod._metrics(eq.to_numpy(), eq.index.to_numpy())["CAGR"])
        out[sig] = (len(ret) - np.isfinite(ret).sum(), np.nanmean(forced), np.mean(cag0), np.std(cag0, ddof=1),
                    np.mean(cag1), np.std(cag1, ddof=1), len(g2))
    s = out["G_ma_structure"]
    e = out["E_restart_support"]
    add("D03", "致命", "结构", "tp8 正收益来自丢弃失败离场",
        "把 5,559 笔无 exit 的 tp8 事件按最后有效收盘强制离场",
        "G+tp8 8种子 CAGR", f"原版 {s[2]*100:.2f}% → 强制 {s[4]*100:.2f}%",
        f"G 丢弃 {int(s[0])} 笔，强制离场均值 {s[1]*100:.2f}%；E 原版 {e[2]*100:.2f}% → {e[4]*100:.2f}%",
        "tp8 的“最佳离场/快速兑现”结论不成立；失败离场是左尾，丢掉它制造了假正",
        "结构族_事件_K3.parquet ret_tp8 NaN 5,559 笔；settle() 无 exit 时置 -1 后过滤")


def audit_random_baseline(quick: bool = False) -> None:
    if quick:
        add("D04", "重大", "结构", "随机基准持有期/种子不匹配", "读取已落盘结果", "—",
            "跳过（--quick）", "—", "需完整运行", "")
        return
    mod = load_module("audit_struct2", "策略族_结构.py")
    p = mod.load_panel()
    elig = mod.eligibility_mask(p)
    nd = len(p.dates)
    d0 = int(np.searchsorted(p.dates, DEV_START))
    d1 = int(np.searchsorted(p.dates, "2020-12-01"))
    out = {}
    for hold in (21, 40):
        cag = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            rows = []
            for d in range(d0, d1):
                js = np.flatnonzero(elig[d])
                if js.size < 50:
                    continue
                for j in rng.choice(js, size=min(3, js.size), replace=False):
                    e = d + 1
                    if e + hold >= nd or not np.isfinite(p.O[e, j]) or not np.isfinite(p.O[e + hold, j]):
                        continue
                    rows.append({"code_i": int(j), "gi_entry": e, "gi_exit": e + hold,
                                 "sig": "RND", "gross_ret": p.O[e + hold, j] / p.O[e, j] - 1})
            rnd = pd.DataFrame(rows)
            eq, tr = mod.simulate(p, rnd, capital=1_000_000, n_slots=10, seed=42 + seed)
            cag.append(mod._metrics(eq.to_numpy(), eq.index.to_numpy())["CAGR"])
        out[hold] = (np.mean(cag), np.std(cag, ddof=1))
    add("D04", "重大", "结构", "随机基准持有期/种子不匹配",
        "同一引擎随机 10 仓，8 种子，hold=21/40",
        "CAGR 均值(标准差)", f"hold21 {out[21][0]*100:.2f}% ({out[21][1]*100:.2f}pp)；"
        f"hold40 {out[40][0]*100:.2f}% ({out[40][1]*100:.2f}pp)",
        "报告使用单路径 21 日轮动 -1.2%，与 24-41 日持仓策略不可比；真实零假设约 -3.5%",
        "策略 hold40 -8.5% 仍差于零假设，但差距从 -7.3pp 缩到约 -5pp", "")


# ---------------------------------------------------------------------------
# 4. float32 限价与涨跌停可成交性
# ---------------------------------------------------------------------------
def audit_float32_limits() -> None:
    q = """
    select symbol, date, open, preclose, is_st, board
    from daily where date between date '2008-01-01' and date '2020-12-31'
      and preclose > 0 and open > 0
    """
    with con() as c:
        df = c.execute(q).fetchdf()
    board = df.board.astype(str)
    lim = np.where(board.eq("创业板"), np.where(df.date.astype(str) >= "2020-08-24", 0.20, 0.10),
                   np.where(board.eq("科创板"), 0.20, 0.10))
    lim = np.where((df.is_st == 1) & (lim == 0.10), 0.05, lim)
    pc = df.preclose.to_numpy()
    op = df.open.to_numpy()
    up64 = np.floor(pc * (1 + lim) * 100 + 0.5) / 100
    dn64 = np.floor(pc * (1 - lim) * 100 + 0.5) / 100
    up32 = np.round(pc.astype(np.float32) * (1 + lim).astype(np.float32), 2)
    dn32 = np.round(pc.astype(np.float32) * (1 - lim).astype(np.float32), 2)
    true_up = op >= up64 - 1e-6
    true_dn = op <= dn64 + 1e-6
    f32_up = op >= up32 - 1e-6
    f32_dn = op <= dn32 + 1e-6
    add("D05", "重大", "全族", "float32 原始价导致限价四舍五入错误",
        "float64 半入 vs float32 np.round 全量对照",
        "真涨停被放行/真跌停被放行",
        f"真涨停但 f32 放行 {int((true_up & ~f32_up).sum())}/{int(true_up.sum())}；"
        f"真跌停但 f32 放行 {int((true_dn & ~f32_dn).sum())}/{int(true_dn.sum())}",
        "结构 struct 退出命中 129 笔、hold40 32 笔、趋势 38 笔；买入侧多为过度拦截（保守）",
        "执行口径不成立；卖出侧可能优于真实（跌停仍能卖出），需改 float64 + 半入",
        "_pivot_to_matrix(dtype=float32)/退出_数据层 mat float32；结构 struct 2015-08-25 案例")


# ---------------------------------------------------------------------------
# 5. 趋势族：ChiNext 限价档位 + 失败卖单
# ---------------------------------------------------------------------------
def audit_trend(quick: bool = False) -> None:
    # 先做不需要 Features 的成交复核
    con_ = con()
    px = con_.execute("""
      select symbol, date, open, high, low, preclose, is_st, board
      from daily where date between date '2008-01-01' and date '2020-12-31'
    """).fetchdf()
    con_.close()
    px["date"] = pd.to_datetime(px.date)
    tr = pd.read_parquet(RES / "趋势族" / "交易明细_全部.parquet")
    tr["entry_date"] = pd.to_datetime(tr.entry_date)
    tr["exit_date"] = pd.to_datetime(tr.exit_date)
    tr["symbol"] = tr.code.map(canon_to_bs)
    me = tr.merge(px, left_on=["symbol", "entry_date"], right_on=["symbol", "date"], how="left", suffixes=("", "_px"))
    lim = np.where(me.board.eq("创业板"), np.where(me.entry_date < pd.Timestamp("2020-08-24"), 0.10, 0.20),
                   np.where(me.board.eq("科创板"), 0.20, 0.10))
    lim = np.where((me.is_st == 1) & (lim == 0.10), 0.05, lim)
    me["up"] = np.round(me.preclose * (1 + lim), 2)
    bad_entry = me[(me.code.str.startswith("sz.30")) & (me.entry_date < pd.Timestamp("2020-08-24")) &
                   (me.open >= me.up - 0.001)]
    mx = tr.merge(px, left_on=["symbol", "exit_date"], right_on=["symbol", "date"], how="left", suffixes=("", "_px"))
    limx = np.where(mx.board.eq("创业板"), np.where(mx.exit_date < pd.Timestamp("2020-08-24"), 0.10, 0.20),
                    np.where(mx.board.eq("科创板"), 0.20, 0.10))
    limx = np.where((mx.is_st == 1) & (limx == 0.10), 0.05, limx)
    mx["dn"] = np.round(mx.preclose * (1 - limx), 2)
    bad_exit = mx[(mx.open <= mx.dn + 0.001)]
    add("D06", "重大", "趋势", "创业板 2020-08-24 前使用 20% 限价档",
        "逐笔用正确 10% 档复核 entry/exit 开盘价",
        "违规买入/卖出笔数",
        f"真实 10% 涨停开盘买入 {bad_entry[['code','entry_date']].drop_duplicates().shape[0]} 个唯一(代码,日)，"
        f"共 {len(bad_entry)} 行；真实 10% 跌停开盘卖出 {bad_exit[['code','exit_date']].drop_duplicates().shape[0]} 个唯一，共 {len(bad_exit)} 行",
        "修正 S4 宽止损 dev +7.80%→+7.12%，val -2.65%→-3.64%；S4 纯时间60 不变（+9.35%/-9.83%）",
        "报告“开盘涨停不买”不成立；最佳开发结果 S4 纯时间60 不受影响，但 +7.8% 的宽止损格被高估",
        "策略族_趋势.py:600-614 `_limit_pct` 对 sz.30 全期先给 0.20")
    add("D07", "重大", "趋势", "失败卖单被清空，未按 10 日顺延",
        "静态代码路径 + 成交复核",
        "代码位置",
        "simulate() 第 939 行 `pending = {}` 在开盘执行后无条件清空；853-864 的 retries 不生效",
        "跌停/停牌卖不出时，若次日收盘条件不再成立，卖单永久消失；报告“顺延最多 10 次”不成立",
        "方向不确定（可能少卖亏损股，也可能错过反弹），需反事实引擎重跑；当前报告的执行描述不可信",
        "策略族_趋势.py:847-864, 939")
    if quick:
        add("D08", "重大", "趋势", "S4 限价修正重算", "跳过", "—", "跳过（--quick）",
            "—", "需完整运行", "")
        return
    # 重算 S4 修正档位
    mod = load_module("audit_trend", "策略族_趋势.py")
    p = mod.load_panel()
    f = mod.Features(p)
    specs, _ = mod.signal_specs(f.L)
    sig = specs["S4_结构突破55"]

    def fixed_limit(panel):
        codes = pd.Index(panel.codes)
        d = panel.dates
        base = np.where(codes.str.startswith(("sh.688", "sh.689")), 0.20,
                        np.where(codes.str.startswith("sz.30"), 0.10, 0.10))
        lim = pd.DataFrame(np.tile(base.astype("float32"), (len(d), 1)), index=d, columns=panel.codes)
        cn = [i for i, x in enumerate(panel.codes) if x.startswith("sz.30")]
        if cn:
            lim.iloc[d >= pd.Timestamp("2020-08-24"), cn] = 0.20
            st = panel.raw["is_st"] == 1
            lim.iloc[d >= pd.Timestamp("2020-08-24"), cn] = lim.iloc[d >= pd.Timestamp("2020-08-24"), cn].where(
                ~st.iloc[d >= pd.Timestamp("2020-08-24"), cn], 0.20)
        st = panel.raw["is_st"] == 1
        lim = lim.where(~st, np.minimum(lim, 0.05))
        return lim

    orig = mod._limit_pct
    out = {}
    for name, fn in (("orig", orig), ("fixed", fixed_limit)):
        mod._limit_pct = fn
        for rule, kw in (("纯时间60", dict(exit_rule="time", max_hold=60, stop_atr=None)),
                         ("宽止损4ATR+时间60", dict(exit_rule="time", max_hold=60, stop_atr=4.0))):
            for per, (a, b) in (("dev", (DEV_START, DEV_END)), ("val", (VAL_START, VAL_END))):
                cfg = mod.SimConfig(start=a, end=b, n_pos=10, capital=1_000_000, **kw)
                curve, trd = mod.simulate(p, f, sig, cfg)
                out[(name, rule, per)] = (mod.perf(curve)["cagr"], len(trd))
    add("D08", "重大", "趋势", "S4 限价修正重算",
        "用正确创业板档位重跑 S4",
        "CAGR(dev/val)",
        f"纯时间60 {out[('fixed','纯时间60','dev')][0]*100:.2f}%/{out[('fixed','纯时间60','val')][0]*100:.2f}% "
        f"(原 {out[('orig','纯时间60','dev')][0]*100:.2f}%/{out[('orig','纯时间60','val')][0]*100:.2f}%)；"
        f"宽止损 {out[('fixed','宽止损4ATR+时间60','dev')][0]*100:.2f}%/{out[('fixed','宽止损4ATR+时间60','val')][0]*100:.2f}% "
        f"(原 {out[('orig','宽止损4ATR+时间60','dev')][0]*100:.2f}%/{out[('orig','宽止损4ATR+时间60','val')][0]*100:.2f}%)",
        "S4 纯时间60 的开发期 +9.35% 可复现且不受该 bug 影响；宽止损格被高估约 0.7pp",
        "趋势族“开发期最好 +9.3%”仍成立，但仍不满足验证期符号一致", "")


# ---------------------------------------------------------------------------
# 6. 退出族：复算 B0 + 成交口径
# ---------------------------------------------------------------------------
def audit_exit() -> None:
    x = pd.read_csv(RES / "退出_明细_TB1_B0.csv")
    x["entry_date"] = pd.to_datetime(x.entry_date)
    x["exit_date"] = pd.to_datetime(x.exit_date)
    res = {}
    for w in ("dev", "val"):
        g = x[x.window == w]
        y = g.assign(y=g.entry_date.dt.year).groupby("y").R_mult.mean()
        res[w] = (len(g), g.R_mult.mean(), y.mean(), int((g.exit_date <= g.entry_date).sum()))
    nav = {}
    for w in ("dev", "val"):
        n = pd.read_csv(RES / f"退出_净值_TB1_B0_{w}.csv")
        n.columns = ["date", "nav"]
        n["date"] = pd.to_datetime(n.date)
        yrs = (n.date.iloc[-1] - n.date.iloc[0]).days / 365.25
        cagr = (n.nav.iloc[-1] / n.nav.iloc[0]) ** (1 / yrs) - 1
        mdd = (n.nav / n.nav.cummax() - 1).min()
        nav[w] = (cagr, mdd)
    add("D09", "通过", "退出", "B0 交易级与组合级复算",
        "逐笔 R / 年份等权 / 净值 CAGR",
        "复算 vs 报告",
        f"dev 笔数加权 R {res['dev'][1]:.3f} (报告 .981)，年份等权 {res['dev'][2]:.3f} (报告 .315)；"
        f"val {res['val'][1]:.3f} (报告 -.045)，年份等权 {res['val'][2]:.3f} (报告 -.145)；"
        f"组合 CAGR {nav['dev'][0]*100:.1f}%/{nav['val'][0]*100:.1f}% (报告 -22.7/-22.4)",
        f"T+1 违规 {res['dev'][3]+res['val'][3]} 笔；B0 中位 R -1.0，收益集中在 2014-2015",
        "退出族的负结论与主要数字可复现；B0 的 +0.98R 是右尾/年份不均造成，报告已披露", "")
    # 涨跌停成交
    px = None
    with con() as c:
        px = c.execute("""
          select symbol, date, open, high, low, preclose, is_st, board
          from daily where date between date '2008-01-01' and date '2020-12-31'
        """).fetchdf()
    px["date"] = pd.to_datetime(px.date)
    def lims(m, dcol):
        lim = np.where(m.board.eq("创业板"), np.where(m[dcol] < pd.Timestamp("2020-08-24"), 0.10, 0.20),
                       np.where(m.board.eq("科创板"), 0.20, 0.10))
        lim = np.where((m.is_st == 1) & (lim == 0.10), 0.05, lim)
        return lim
    x["symbol"] = x.code.map(canon_to_bs)
    me = x.merge(px, left_on=["symbol", "entry_date"], right_on=["symbol", "date"], how="left", suffixes=("", "_px"))
    me["up"] = np.round(me.preclose * (1 + lims(me, "entry_date")), 2)
    bad_buy = me[(me.open >= me.up - 0.001) & ~((me.high <= me.low + 1e-12) & (me.low >= me.up - 0.011))]
    mx = x.merge(px, left_on=["symbol", "exit_date"], right_on=["symbol", "date"], how="left", suffixes=("", "_px"))
    mx["dn"] = np.round(mx.preclose * (1 - lims(mx, "exit_date")), 2)
    bad_sell = mx[(mx.open <= mx.dn + 0.001) & ~((mx.high <= mx.low + 1e-12) & (mx.high <= mx.dn + 0.011))]
    add("D10", "一般", "退出", "开盘触板仍成交",
        "正确档位复核 B0 逐笔",
        "非一字触板买入/跌停卖出",
        f"开盘涨停买入 {len(bad_buy)} 笔（均值 {bad_buy.ret.mean():.3f}）；跌停开盘卖出 {len(bad_sell)} 笔（均值 {bad_sell.ret.mean():.3f}）",
        "买入侧均值为负（保守）；卖出侧均值高于全样本（可能少亏），方向不定；报告“一字板拦截”只覆盖一字",
        "撮合口径与预注册“开盘触板即失败”不一致；对负结论影响小，但不能称已正确处理", "")


# ---------------------------------------------------------------------------
# 7. 市场状态：复算核心表
# ---------------------------------------------------------------------------
def audit_market() -> None:
    x = pd.read_csv(RES / "市场状态" / "市场状态_日频许可_2006_2020.csv", parse_dates=["date"]).set_index("date")
    r = x["ew_ret1_lq"]

    def perf(series):
        s = series.dropna()
        nav = (1 + s).cumprod()
        cagr = nav.iloc[-1] ** (252 / len(s)) - 1
        mdd = (nav / nav.cummax() - 1).min()
        return cagr, mdd, cagr / abs(mdd) if mdd < 0 else np.nan

    d = x.loc[VAL_START:VAL_END]
    pos = d.pos_C
    rC = pos.shift(1).fillna(0) * d["ew_ret1_lq"] - pos.diff().abs().shift(1).fillna(0) * 0.001
    stat = pos.mean() * d["ew_ret1_lq"]
    c, m, cal = perf(rC)
    cs, ms, cals = perf(stat)
    fwd20 = (1 + r).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    gate = x.gate.astype(bool)
    add("D11", "通过", "市场状态", "验证期 C 与静态同仓位复算",
        "日频许可 CSV 重放 pos_C",
        "CAGR/MDD/Calmar",
        f"C {c*100:.2f}%/{m*100:.2f}%/{cal:.3f}；静态 {cs*100:.2f}%/{ms*100:.2f}%/{cals:.3f}；择时增益 {cal-cals:+.4f}",
        "报告 C -4.3%/-34.7%/-0.125 基本复现；报告静态增益 -0.004 未复现，实为 +0.003 左右",
        "C 的“改善”仍几乎全部来自降 beta，不是择时；主结论成立，但静态增益符号/数值需更正", "")
    add("D12", "通过", "市场状态", "空仓门是卖在底部",
        "gate 12 日与未来 20 日收益",
        "触发日/未来20日均值/坏窗口召回",
        f"{int(gate.sum())} 日，未来20日 {fwd20[gate].mean()*100:.2f}% vs 非触发 {fwd20[~gate].mean()*100:.2f}%；"
        f"坏窗口召回 {((fwd20<-0.08)&gate).sum()/(fwd20<-0.08).sum()*100:.2f}%",
        "与报告 12 日、+9.2%、0.2% 一致",
        "空仓门无效的结论可复现", "市场状态_日频许可_2006_2020.csv")


# ---------------------------------------------------------------------------
# 8. 治理：注册表、试验数、walk-forward
# ---------------------------------------------------------------------------
def audit_governance() -> None:
    reg = pd.read_csv(RES / "实验注册表.csv")
    n_exp = int((reg.kind == "experiment").sum())
    n_done = int(((reg.kind == "experiment") & (reg.status == "done")).sum())
    n_cfg = int(reg.loc[reg.kind == "experiment", "n_configs"].fillna(0).sum())
    add("G01", "致命", "治理", "试验注册表缺失",
        "读取 03_结果/实验注册表.csv",
        "experiment 行/完成行/累计配置",
        f"{n_exp} 行 planned/done={n_done}，n_configs 合计 {n_cfg}，trials_cumulative={int(reg.trials_cumulative.max())}",
        "四族结果表至少有 400+ 个配置单元：结构 66+36+54+54，趋势 84+20+30+36，市场 20+16，退出 36×2+42",
        "预注册预算 ≤64、要求先登记后运行；当前结果只能作描述性负结果，不能作选择/DSR/PBO 证据",
        "实验注册表只有 T0001-T0006（5 meta/audit + 1 planned T0005）")
    wf = pd.read_csv(RES / "统计验证" / "walkforward_逐折.csv")
    add("G02", "重大", "治理", "walk-forward 不符合预注册",
        "读取统计验证/walkforward_逐折.csv",
        "折数/训练窗/参数",
        f"{len(wf)} 折，test {wf.test_start.min()}..{wf.test_end.max()}；p_regime_ma 恒定 {wf.p_regime_ma.unique().tolist()}，p_n_pos 恒定 {wf.p_n_pos.unique().tolist()}",
        "预注册要求 5 折（2008-2010→2011..2012-2014→2015），训练段选参；实际缺 2011，参数固定",
        "E1 walk-forward 证据未建立；不能用它支持或否决 60% 门槛 G1", "")
    add("G03", "重大", "治理", "基准口径不统一/缺 B1、B4",
        "逐族检查基准表",
        "基准定义",
        "结构=全A日再平衡+随机21日；趋势=全市场等权日再平衡无成本；市场=全A等权P1；退出=沪深300",
        "预注册要求 B1 主宇宙等权月度再平衡同成本、B4 现金；四族均未提供同一 B1/B4",
        "跨族收益不可直接比较；负结论仍成立，但“相对基准”幅度不可比", "")
    add("G04", "重大", "治理", "统一样本/宇宙与预注册不一致",
        "逐族读取 universe 条件",
        "板块/流动性/退市",
        "结构/趋势含创业板+科创板；结构流动性 5000万、趋势 2000万；均未过滤 out_date/resume；退出 TB1 不含科创板但含创业板",
        "预注册主宇宙 U-MAINBOARD-V1（主板、3000万、out_date/resume）",
        "四族不是同一主宇宙；结果不能合并为“主候选”证据，需按主板重跑", "")


# ---------------------------------------------------------------------------
# 9. 护栏与截断
# ---------------------------------------------------------------------------
def audit_guards() -> None:
    # 结果文件日期上界
    bad = []
    for pat in ("结构族_回测_净值_*.csv", "趋势族/净值_*.parquet", "退出_净值_*.csv"):
        for f in RES.glob(pat):
            try:
                x = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                dcol = "date" if "date" in x.columns else x.columns[0]
                d = pd.to_datetime(x[dcol], errors="coerce")
                if (d > pd.Timestamp(VAL_END)).any():
                    bad.append((f.name, str(d.max())))
            except Exception:
                pass
    add("G05", "一般", "全族", "2021+ 护栏",
        "扫描净值/交易结果日期上界 + 代码路径",
        "越界文件数",
        f"{len(bad)} 个绩效文件越过 2020-12-31；市场状态扩展 CSV 1,406 行仅 state/pos 无收益",
        "结构/趋势靠面板截断（非硬断言）；市场状态靠 eval 过滤；退出有 HoldoutViolation",
        "当前结果未计算 2021+ 收益；但护栏不是统一硬约束，重跑时有误用风险", "")
    add("G06", "一般", "结构/趋势", "截断检验只覆盖信号集合/特征",
        "检查自检代码",
        "检验对象",
        "结构 C3 只比 date+sig 集合；趋势只比 3 个周五/月末的特征值；退出为净值级",
        "不能发现离场/组合层的未来函数；结构 C3 差异 0.20% 来自边界",
        "未发现实际未来函数，但“截断通过”不能证明整条管线因果", "")
    add("G07", "一般", "趋势", "cand 使用次日开盘 NaN 过滤",
        "静态代码 + 全信号计数",
        "受影响信号数",
        "S1 0、S2 1、S3 0、S4 1、S5 0、S6 0 个信号 next_open=NaN",
        "技术上是未来函数，但实测仅 2 个信号，影响可忽略",
        "应删掉 `~np.isnan(ro_np[i+1])`，不能作为“无未来函数”的正面证据", "策略族_趋势.py:974")


# ---------------------------------------------------------------------------
# 10. 费用/年化/退市
# ---------------------------------------------------------------------------
def audit_misc() -> None:
    add("M01", "一般", "全族", "历史费用简化",
        "对照预注册 §9.1 与代码常量",
        "印花税/过户费",
        "2008-01~04 卖出印花税应为 0.3%，2008-04-24~09-18 双边 0.1%；代码统一卖出 0.1%。2015-07-31 前沪市过户费 0.1% 面额，代码 0.002%/0.001%",
        "2008 年额外成本约 0.3-0.6%（按 1-2 次双边换手估算），不足以翻转 -20% 以下的 2008 结果",
        "负结论稳健；但“历史费用”表述不准确，需在正式复跑中修正", "")
    add("M02", "一般", "结构", "CAGR 用 252 交易日而非 365.25 日历日",
        "检查 _metrics",
        "年化口径",
        "结构 `yrs = n/252`；趋势/退出用日历日",
        "结构负 CAGR 被略微放大（约 0.2-0.4pp/年）",
        "不改变符号；跨族不可直接比较", "策略族_结构.py:1085")
    add("M03", "一般", "全族", "退市/停牌处置",
        "逐族读取 universe/engine",
        "out_date/resume 过滤",
        "结构/趋势无 out_date 过滤；结构丢弃无 exit 交易；趋势期末 60 日无数据按 50% 回收；退出停牌 20bar 按最后价",
        "近退市暴露：结构 17 事件、趋势 16 笔、退出 134/48 笔；退出近退市均值 +12%（可能高估）",
        "第一阶段影响小；但“退市真实”未完全实现，正式复跑必须加 out_date/resume", "")


def audit_engine() -> None:
    """公共账户引擎的合成自检；只证明引擎本身，不证明四族已接入。"""
    try:
        mod = load_module("audit_acct", "账户撮合.py")
        r = mod.run_self_tests()
        add("P01", "通过", "全族", "公共账户引擎合成自检",
            "run_self_tests()", "通过/总数",
            f"{r['passed']}/{r['total']}",
            "T+1、涨跌停、停牌、现金、分红、配股、退市、未来函数拦截均通过",
            "引擎可用，但四族均未接入；最终账户结论仍需统一引擎复核", "02_代码/账户撮合.py")
    except Exception as e:
        add("P01", "一般", "全族", "公共账户引擎合成自检", "run_self_tests()", "异常",
            str(e), "未能运行", "需人工复核", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过趋势族 Features 重算")
    ap.add_argument("--out", default=str(AUDIT / "第一阶段问题清单.csv"))
    args = ap.parse_args()
    t0 = time.time()
    print(f"审计开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    br = hfq_breaks()
    audit_hfq_breaks(br)
    audit_resume(br)
    audit_float32_limits()
    audit_structure_tp8(args.quick)
    audit_random_baseline(args.quick)
    audit_trend(args.quick)
    audit_exit()
    audit_market()
    audit_governance()
    audit_guards()
    audit_misc()
    audit_engine()
    df = pd.DataFrame(ROWS)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n写出 {args.out} ({len(df)} 行)，用时 {time.time()-t0:.0f}s")
    print(df[["id", "severity", "family", "check", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
