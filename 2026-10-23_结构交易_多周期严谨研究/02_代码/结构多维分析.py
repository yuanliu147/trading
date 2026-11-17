"""固定C队列的描述性分层；不产生/修改策略，不联网，不读取密钥。
运行：python3 -B 本文件绝对路径。只新增结构多维_*.csv及结构多维审计.json。
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import heapq
import json
import math
import re
import time

import numpy as np
import pandas as pd

from 交易数据 import TradingData, mainboard, OLD
from 结构特征 import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, EVIDENCE = ROOT / "01_数据", ROOT / "03_结果", ROOT / "04_证据"
SOURCE = OUT / "命题实证_C_第二驱动事件明细.csv"
SEED, REPEATS, BLOCK = 20261023, 500, 20
STATES = {"success": "成功", "failure": "失败", "unfinished": "未完成",
          "right_censored": "右删失", "path_gap_censored": "路径缺口删失",
          "initial_low_missing": "初始低点缺失", "cross_segment": "跨段"}
KNOWN = ("success", "failure", "unfinished")
SCOPES = [("全期", "2018-01-01", "2026-12-31"),
          ("开发2018-2020", "2018-01-01", "2020-12-31"),
          ("验证2021-2022", "2021-01-01", "2022-12-31"),
          ("留出2023-2025", "2023-01-01", "2025-12-31"),
          ("2026", "2026-01-01", "2026-12-31")]
SCOPES += [(f"年度{y}", f"{y}-01-01", f"{y}-12-31") for y in range(2018, 2027)]
UNIVERSES = ("全可得沪深", "主板代码")
HISTORY_GROUPS = ["历史不足", "0-40%", "40-60%", "60-100%"]
BREADTH_GROUPS = ["下行宽度（小于40%）", "中间宽度（40%-60%）", "上行宽度（大于60%）", "未知"]
RISK_GROUPS = ["高于历史中位数", "不高于历史中位数", "未知"]
VOL_GROUPS = ["小于25%", "25%-50%", "大于50%", "未知"]
SOURCE_NAMES = {
    "start_i": "起点交易日序号", "horizon": "观察期限交易日", "initial_low": "初始低点",
    "first_peak_i": "第一峰交易日序号", "first_peak_confirm_i": "第一峰确认交易日序号",
    "first_peak": "第一峰", "pullback_i": "回调低点交易日序号",
    "pullback_confirm_i": "回调确认交易日序号", "pullback_low": "回调低点",
    "second_break_i": "再次突破交易日序号", "invalidation_i": "失效交易日序号",
    "expiration_i": "预定到期交易日序号", "observed_through_i": "观察截至交易日序号",
    "terminal_i": "原始终局交易日序号", "censor_i": "删失交易日序号",
    "status": "原始结果", "censor_reason": "删失原因", "failure_priority_same_day": "同日失败优先",
    "code": "来源代码", "mainboard": "是否主板代码", "weekly_up_at_start": "起点周线上升",
    "start_adjusted_close": "起点后复权收盘", "initial_low_known_at_start": "起点初始低点已知",
    "delist_date": "退市日期", "start_date": "起点日期", "first_peak_date": "第一峰日期",
    "first_peak_confirm_date": "第一峰确认日期", "pullback_date": "回调低点日期",
    "pullback_confirm_date": "回调确认日期", "second_break_date": "再次突破日期",
    "invalidation_date": "失效日期", "expiration_date": "预定到期日期",
    "observed_through_date": "观察截至日期", "terminal_date": "原始终局日期",
    "censor_date": "删失日期", "expiration_beyond_available_calendar": "到期超出已知日历",
    "definition": "固定定义",
}


def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_counts(starts, terminals, statuses):
    """只把起点之前已终结的先前事件移出pending；绝不按起点shift标签。"""
    result = np.zeros((len(starts), 3), dtype=int)
    pending, included = [], []
    counts = np.zeros(3, dtype=int)
    links = []
    for j, start in enumerate(starts):
        while pending and pending[0][0] < start:
            terminal, k = heapq.heappop(pending)
            assert starts[k] < start and terminal < start and statuses[k] in KNOWN
            included.append(k)
            counts[KNOWN.index(statuses[k])] += 1
        # 每个事件独立过滤复核，含同日终局、重叠未终局和删失排除。
        expected = np.flatnonzero((starts < start) & (terminals >= 0) & (terminals < start)
                                  & np.isin(statuses, KNOWN))
        assert sorted(included) == expected.tolist()
        result[j] = counts
        links.append(expected)
        if statuses[j] in KNOWN:
            assert terminals[j] > start
            heapq.heappush(pending, (int(terminals[j]), j))
    return result, links


def self_tests():
    starts = np.array([0, 2, 4, 6, 8, 10, 12])
    terminals = np.array([8, 3, 6, -1, 9, 12, 14])
    statuses = np.array(["success", "failure", "unfinished", "right_censored",
                         "success", "failure", "success"])
    counts, _ = history_counts(starts, terminals, statuses)
    assert counts[1].sum() == 0  # 较早起点不代表较早可用。
    assert counts[3].tolist() == [0, 1, 0]  # 终局同日不得进入。
    assert counts[4].tolist() == [0, 1, 1]  # 未完成仅在到期之后可用。
    assert counts[5].tolist() == [2, 1, 1]
    assert counts[6].tolist() == [2, 1, 1]
    return ["重叠标签未泄漏", "终局同日严格排除", "未完成到期后可用", "删失永不充当历史结果",
            "逐事件直接过滤与待完成队列一致"]


def save(frame, suffix, outputs):
    assert all(not re.search(r"[A-Za-z]", str(c)) for c in frame.columns)
    path = OUT / f"结构多维_{suffix}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
    outputs.append({"路径": str(path), "行数": len(frame), "列数": len(frame.columns), "校验": digest(path)})


def summary(sub, scope, universe, dimension, group):
    vc = sub["分段结果"].value_counts()
    n = len(sub)
    row = {"分期": scope, "股票范围": universe, "维度": dimension, "分组": group,
           "观察期限交易日": 60, "样本数": n, "股票数": sub.code.nunique(),
           "起点日期数": sub.start_date.nunique()}
    row.update({f"{name}数": int(vc.get(key, 0)) for key, name in STATES.items()})
    known = sum(row[f"{STATES[k]}数"] for k in KNOWN)
    censored = sum(row[f"{STATES[k]}数"] for k in ("right_censored", "path_gap_censored", "initial_low_missing"))
    row.update({"已分类分母": known, "删失数": censored,
                "成功率占全部": float(ratio(row["成功数"], n)),
                "失败率占全部": float(ratio(row["失败数"], n)),
                "未完成率占全部": float(ratio(row["未完成数"], n)),
                "删失率占全部": float(ratio(censored, n)),
                "右删失率占全部": float(ratio(row["右删失数"], n)),
                "路径缺口删失率占全部": float(ratio(row["路径缺口删失数"], n)),
                "跨段率占全部": float(ratio(row["跨段数"], n)),
                "成功率占已分类": float(ratio(row["成功数"], known)),
                "失败率占已分类": float(ratio(row["失败数"], known)),
                "未完成率占已分类": float(ratio(row["未完成数"], known))})
    assert sum(row[f"{name}数"] for name in STATES.values()) == n
    return row


def bootstrap(sub, timeline, scope, universe, comparison, column, high, low, seed):
    # 同日全部股票作为一簇；各组分子分母同步抽样。零事件日仍在日历轴上。
    daily = pd.DataFrame({"日期": timeline})
    arrays, sizes = [], []
    for name, group in (("高组", high), ("低组", low)):
        chosen = sub.loc[sub[column].eq(group) & sub["分段结果"].isin(KNOWN)]
        count = chosen.groupby("start_date").size().reindex(timeline, fill_value=0).to_numpy()
        success = chosen.loc[chosen["分段结果"].eq("success")].groupby("start_date").size().reindex(timeline, fill_value=0).to_numpy()
        daily[f"{name}成功数"], daily[f"{name}已分类数"] = success, count
        daily[f"{name}成功率"] = ratio(success, count)
        arrays.extend([success, count])
        active_blocks = len(np.unique(np.flatnonzero(count) // BLOCK))
        sizes.append((int(count.sum()), int((count > 0).sum()), active_blocks))
    a = np.stack(arrays, axis=1)
    n = len(timeline)
    assert n >= BLOCK and timeline.is_monotonic_increasing and not timeline.has_duplicates
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - BLOCK + 1, (REPEATS, math.ceil(n / BLOCK)))
    indices = (starts[:, :, None] + np.arange(BLOCK)).reshape(REPEATS, -1)[:, :n]
    totals = a[indices].sum(axis=1)
    differences = ratio(totals[:, 0], totals[:, 1]) - ratio(totals[:, 2], totals[:, 3])
    valid = differences[np.isfinite(differences)]
    enough = n >= 100 and all(x >= 30 and d >= 20 and b >= 5 for x, d, b in sizes) and len(valid) >= 475
    lo, hi = np.quantile(valid, [.025, .975]) if enough else (np.nan, np.nan)
    point = float(ratio(a[:, 0].sum(), a[:, 1].sum()) - ratio(a[:, 2].sum(), a[:, 3].sum()))
    row = {"分期": scope, "股票范围": universe, "比较": comparison, "高组": high, "低组": low,
           "估计量": "事件加权成功率占已分类之差（高减低）", "完整交易日数": n,
           "双组均有事件日数": int(((a[:, 1] > 0) & (a[:, 3] > 0)).sum()),
           "双组均无事件日数": int(((a[:, 1] == 0) & (a[:, 3] == 0)).sum()),
           "高组已分类数": sizes[0][0], "低组已分类数": sizes[1][0],
           "高组事件日数": sizes[0][1], "低组事件日数": sizes[1][1],
           "高组占用二十日格数": sizes[0][2], "低组占用二十日格数": sizes[1][2],
           "高组成功率": float(ratio(a[:, 0].sum(), a[:, 1].sum())),
           "低组成功率": float(ratio(a[:, 2].sum(), a[:, 3].sum())),
           "成功率差": point, "百分之九十五区间下限": lo, "百分之九十五区间上限": hi,
           "抽样次数": REPEATS, "有效抽样次数": len(valid), "块长交易日": BLOCK, "随机种子": seed,
           "样本判断": "达到最低报告门槛" if enough else "样本不足，区间不报告"}
    daily.insert(0, "比较", comparison)
    daily.insert(0, "股票范围", universe)
    daily.insert(0, "分期", scope)
    assert np.array_equal(daily["日期"].to_numpy(), timeline.to_numpy())
    return row, daily


def run():
    started = time.time()
    assert all(p.is_dir() for p in (DATA, OUT, EVIDENCE))
    tests = self_tests()
    # 只监测本分析输入；并行账户任务的输出不属于本脚本，也不读其内容。
    inputs = [SOURCE, OUT / "命题实证_C_第二驱动分期年度汇总.csv", OUT / "命题实证_数据覆盖.csv",
              EVIDENCE / "历史交易状态覆盖.csv", DATA / "特征构建审计.csv",
              DATA / "市场上下文.csv", DATA / "市场行业上下文.npz", DATA / "交易日历.csv",
              OLD / "申万行业分类_时点.csv"]
    inputs += list((DATA / "结构特征").glob("*.parquet"))
    inputs += list((DATA / "原始分股").glob("*.parquet"))
    original = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    protected = {str(ROOT / "02_代码" / name): digest(ROOT / "02_代码" / name)
                 for name in ("交易数据.py", "结构特征.py", "构建研究数据.py")}
    data = TradingData(feature_cache=1)
    dates, cal = data.dates, data.full_calendar
    source = pd.read_csv(SOURCE)
    assert set(source.columns) == set(SOURCE_NAMES)
    e = source.loc[source.horizon.eq(60)].copy().sort_values(["code", "start_i"]).reset_index(drop=True)
    assert not e.duplicated(["code", "start_i"]).any()
    assert set(e.status) <= set(STATES) - {"cross_segment"}
    assert (e.expiration_i == e.start_i + 60).all()
    for col in [c for c in e if c.endswith("_date")]:
        e[col] = pd.to_datetime(e[col], errors="raise")
    assert np.array_equal(e.start_date.to_numpy(), dates[e.start_i].to_numpy())
    assert np.array_equal(e.mainboard.to_numpy(), e.code.map(mainboard).to_numpy())
    e["标签可得交易日序号"] = np.where(e.status.eq("unfinished"), e.expiration_i, e.terminal_i)
    e.loc[~e.status.isin(KNOWN), "标签可得交易日序号"] = -1
    unfinished = e.status.eq("unfinished")
    assert (e.loc[unfinished, "observed_through_i"] == e.loc[unfinished, "expiration_i"]).all()
    assert (e.loc[unfinished, "expiration_i"] < len(dates)).all()
    assert e.loc[unfinished, "terminal_date"].isna().all()
    terminal = e.status.isin(("success", "failure"))
    assert np.array_equal(e.loc[terminal, "terminal_date"].to_numpy(), dates[e.loc[terminal, "terminal_i"]].to_numpy())
    e["标签可得日期"] = pd.to_datetime([dates[i] if i >= 0 else pd.NaT for i in e["标签可得交易日序号"]])

    market = pd.read_csv(DATA / "市场上下文.csv", parse_dates=["日期"]).set_index("日期")
    assert market.index.equals(dates)
    np.testing.assert_allclose(market["宽度六十"], data.market["breadth60"], equal_nan=True, rtol=1e-12)
    ld = market["估计跌停占比"]
    # 起点收盘已知，窗口含当日；要求完整20/250个日历交易日有效值。
    mean20 = ld.rolling(20, min_periods=20).mean()
    median250 = ld.rolling(250, min_periods=250).median()
    idx = e.start_i.to_numpy()
    e["市场宽度六十"] = market["宽度六十"].to_numpy()[idx]
    e["估计跌停占比二十日均值"] = mean20.to_numpy()[idx]
    e["估计跌停占比二百五十日中位数"] = median250.to_numpy()[idx]
    e["市场状态可得日期"] = market.index[idx]
    b = e["市场宽度六十"]
    e["市场宽度分组"] = np.select([b.lt(.4), b.between(.4, .6), b.gt(.6)], BREADTH_GROUPS[:3], "未知")
    known_risk = e["估计跌停占比二十日均值"].notna() & e["估计跌停占比二百五十日中位数"].notna()
    higher = e["估计跌停占比二十日均值"] > e["估计跌停占比二百五十日中位数"]
    e["情绪风险代理分组"] = np.select([known_risk & higher, known_risk & ~higher], RISK_GROUPS[:2], "未知")

    industry_path = OLD / "申万行业分类_时点.csv"
    industry = pd.read_csv(industry_path, parse_dates=["start_date"])
    industry["code"] = industry.code.str.split(".").str[1] + "." + industry.code.str.split(".").str[0].str.upper()
    industry_groups = {code: g.sort_values("start_date") for code, g in industry.groupby("code")}
    e["起点历史行业"] = "未知"
    e["行业生效日期"] = pd.NaT
    e["个股二十日波动率"] = np.nan
    e["个股特征可得日期"] = pd.NaT
    counts_all = np.zeros((len(e), 3), dtype=int)
    latest = np.full(len(e), -1, dtype=int)
    examples, example_kinds, prefix_candidates = [], set(), []
    link_count, exclusions, same_day_exclusions = 0, 0, 0
    for number, (code, g) in enumerate(e.groupby("code", sort=True), 1):
        rows, starts = g.index.to_numpy(), g.start_i.to_numpy()
        terminals, statuses = g["标签可得交易日序号"].to_numpy(), g.status.to_numpy()
        counts, links = history_counts(starts, terminals, statuses)
        counts_all[rows] = counts
        for j, selected in enumerate(links):
            current = g.iloc[j]
            if len(selected):
                previous = g.iloc[selected]
                assert previous.horizon.eq(60).all()
                assert previous.status.isin(KNOWN).all()
                assert (previous["标签可得日期"] < current.start_date).all()
                latest[rows[j]] = terminals[selected].max()
                link_count += len(selected)
            excluded = np.flatnonzero((starts < starts[j]) & (terminals >= starts[j]) & np.isin(statuses, KNOWN))
            exclusions += len(excluded)
            same_day_exclusions += int(np.sum(terminals[excluded] == starts[j]))
            kinds = []
            if len(selected) >= 5:
                kinds.append("至少五次已知历史")
            if len(selected) and (statuses[selected] == "unfinished").any():
                kinds.append("未完成到期后纳入")
            if len(excluded):
                kinds.append("未终局历史排除")
            if len(excluded) and (terminals[excluded] == starts[j]).any():
                kinds.append("同日终局排除")
            for kind in kinds:
                if kind in example_kinds:
                    continue
                example_kinds.add(kind)
                def show(k):
                    p = g.iloc[k]
                    return {"来源代码": code, "历史起点日期": str(p.start_date.date()),
                            "期限": 60, "结果": STATES[p.status],
                            "原始终局日期": None if pd.isna(p.terminal_date) else str(p.terminal_date.date()),
                            "标签终止可得日期": str(p["标签可得日期"].date()),
                            "严格早于本次起点": bool(p["标签可得日期"] < current.start_date)}
                examples.append({"案例": kind, "来源代码": code, "本次起点日期": str(current.start_date.date()),
                                 "已知历史数": int(counts[j].sum()), "纳入历史": [show(k) for k in selected],
                                 "尚不可得历史": [show(k) for k in excluded]})
        f = pd.read_parquet(DATA / "结构特征" / f"{code}.parquet",
                            columns=["d_volatility20", "d_close", "d_breakout", "w_up10", "d_pivot_low"])
        assert f.index.equals(dates)
        assert f.d_breakout.iloc[starts].all() and f.w_up10.iloc[starts].all()
        rebuilt_vol = f.d_close.pct_change(fill_method=None).rolling(20, min_periods=20).std(ddof=1)
        np.testing.assert_allclose(f.d_volatility20, rebuilt_vol, equal_nan=True, rtol=1e-12, atol=1e-14)
        e.loc[rows, "个股二十日波动率"] = f.d_volatility20.iloc[starts].to_numpy()
        e.loc[rows, "个股特征可得日期"] = f.index[starts]
        ids = data.industry_ids.get(code, np.zeros(len(dates), dtype="int16"))[starts]
        e.loc[rows, "起点历史行业"] = np.array(data.industry_names)[ids]
        if code in industry_groups:
            ig = industry_groups[code]
            effective = pd.DatetimeIndex(ig.start_date).as_unit("ns")
            loc = np.searchsorted(effective.to_numpy(), dates[starts].as_unit("ns").to_numpy(), side="right") - 1
            found = loc >= 0
            names = np.where(found, ig.industry.to_numpy()[np.maximum(loc, 0)], "未知")
            assert np.array_equal(names, e.loc[rows, "起点历史行业"].to_numpy())
            e.loc[rows[found], "行业生效日期"] = effective[loc[found]]
        if code in ("000001.SZ", "600000.SH", "300001.SZ"):
            for begin, end in (("2018-01-01", "2020-12-31"), ("2023-01-01", "2025-12-31")):
                available = g[g.start_date.between(begin, end)]
                if len(available):
                    prefix_candidates.append((code, available.iloc[0].start_date))
        if number % 1000 == 0:
            print(f"只读核验 {number}/{e.code.nunique()} 股票", flush=True)
    e[["历史成功数", "历史失败数", "历史未完成数"]] = counts_all
    e["历史已分类数"] = counts_all.sum(axis=1)
    e["历史成功率"] = np.where(e["历史已分类数"] >= 5, ratio(counts_all[:, 0], counts_all.sum(axis=1)), np.nan)
    h = e["历史成功率"]
    e["历史成功率分组"] = np.select([h.isna(), h.le(.4), h.gt(.4) & h.lt(.6), h.ge(.6)], HISTORY_GROUPS, "异常")
    assert not e["历史成功率分组"].eq("异常").any()
    e["历史最近标签可得日期"] = pd.to_datetime([dates[i] if i >= 0 else pd.NaT for i in latest])
    e["个股年化波动率"] = e["个股二十日波动率"] * np.sqrt(252)
    v = e["个股年化波动率"]
    e["波动率分组"] = np.select([v.lt(.25), v.between(.25, .5), v.gt(.5)], VOL_GROUPS[:3], "未知")
    assert (e["历史最近标签可得日期"].dropna() < e.loc[e["历史最近标签可得日期"].notna(), "start_date"]).all()
    for col in ("个股特征可得日期", "市场状态可得日期", "行业生效日期"):
        assert (e.loc[e[col].notna(), col] <= e.loc[e[col].notna(), "start_date"]).all()
    assert e["个股特征可得日期"].notna().all() and e["市场状态可得日期"].notna().all()

    prefix_checks = []
    for code, date in prefix_candidates:
        raw = pd.read_parquet(DATA / "原始分股" / f"{code}.parquet")
        bars = raw.loc[raw.index <= date, ["open", "high", "low", "close", "volume", "amount"]].copy()
        bars[["open", "high", "low", "close"]] = bars[["open", "high", "low", "close"]].mul(raw.adj_factor, axis=0)
        rebuilt = build_features(bars, cal, date)
        saved = pd.read_parquet(DATA / "结构特征" / f"{code}.parquet")
        pd.testing.assert_frame_equal(rebuilt, saved.loc[saved.index <= date], check_freq=False, rtol=1e-12, atol=1e-12)
        assert mean20.loc[date] == ld.loc[:date].rolling(20, min_periods=20).mean().iloc[-1] or pd.isna(mean20.loc[date])
        assert median250.loc[date] == ld.loc[:date].rolling(250, min_periods=250).median().iloc[-1] or pd.isna(median250.loc[date])
        prefix_checks.append({"来源代码": code, "起点日期": str(date.date()), "比较列数": 108, "截断重算一致": True})

    outputs, summaries, industry_summaries, boot_rows, daily_rows, reconciliations = [], [], [], [], [], []
    original_summary = pd.read_csv(OUT / "命题实证_C_第二驱动分期年度汇总.csv")
    dimensions = [("总体", None, ["全部"]), ("股票自身历史", "历史成功率分组", HISTORY_GROUPS),
                  ("市场宽度", "市场宽度分组", BREADTH_GROUPS), ("情绪风险代理", "情绪风险代理分组", RISK_GROUPS),
                  ("个股波动", "波动率分组", VOL_GROUPS)]
    contrasts = [("历史成功率高减低", "历史成功率分组", "60-100%", "0-40%"),
                 ("上行宽度减下行宽度", "市场宽度分组", BREADTH_GROUPS[2], BREADTH_GROUPS[0])]
    for s, (scope, begin, end) in enumerate(SCOPES):
        timeline = dates[(dates >= begin) & (dates <= end)]
        segment_end = int(cal.searchsorted(pd.Timestamp(end), side="right") - 1)
        for u, universe in enumerate(UNIVERSES):
            sub = e.loc[e.start_date.between(begin, end) & (e.mainboard if u else True)].copy()
            sub["分段结果"] = sub.status
            if scope != "全期":
                sub.loc[sub.expiration_i > segment_end, "分段结果"] = "cross_segment"
            baseline = summary(sub, scope, universe, "总体", "全部")
            old = original_summary.loc[original_summary.scope.eq(scope) & original_summary.universe.eq(universe)
                                       & original_summary.horizon.eq(60)].iloc[0]
            assert baseline["样本数"] == old.cohort_denominator
            assert baseline["已分类分母"] == old.fully_classified_denominator
            for key, name in STATES.items():
                assert baseline[f"{name}数"] == old[f"n_{key}"]
            reconciliations.append({"分期": scope, "股票范围": universe, "样本数": len(sub), "全部状态及分母一致": True})
            for dimension, col, groups in dimensions:
                partition = []
                for group in groups:
                    chosen = sub if col is None else sub[sub[col].eq(group)]
                    partition.append(summary(chosen, scope, universe, dimension, group))
                assert sum(x["样本数"] for x in partition) == len(sub)
                summaries.extend(partition)
            partition = [summary(sub[sub["起点历史行业"].eq(group)], scope, universe, "起点历史行业", group)
                         for group in data.industry_names]
            assert sum(x["样本数"] for x in partition) == len(sub)
            industry_summaries.extend(partition)
            for c, (comparison, col, high, low) in enumerate(contrasts):
                row, daily = bootstrap(sub, timeline, scope, universe, comparison, col, high, low, SEED + 100 * s + 10 * u + c)
                boot_rows.append(row)
                daily_rows.append(daily)

    coverage = pd.read_csv(OUT / "命题实证_数据覆盖.csv")
    states = pd.read_csv(EVIDENCE / "历史交易状态覆盖.csv")
    build = pd.read_csv(DATA / "特征构建审计.csv")
    gaps = []
    for universe in UNIVERSES:
        is_main = universe == "主板代码"
        cov = coverage.loc[coverage.code.map(mainboard)] if is_main else coverage
        st = states.loc[states["代码"].map(mainboard)] if is_main else states
        universe_build = build.loc[build["代码"].map(mainboard)] if is_main else build
        event_sub = e.loc[e.mainboard] if is_main else e
        gaps.append({"股票范围": universe, "可得股票数": int(universe_build["状态"].eq("成功").sum()),
                     "缺真实行情股票数": int(universe_build["状态"].eq("缺真实行情").sum()),
                     "有事件股票数": event_sub.code.nunique(), "事件数": len(event_sub),
                     "适配层有成交日数": int(st["有成交日"].sum()), "适配层历史状态已知日数": int(st["已知状态日"].sum()),
                     "适配层历史状态已知比例": float(ratio(st["已知状态日"].sum(), st["有成交日"].sum())),
                     "原命题合格观测日数": int(cov.eligible_days.sum()),
                     "原命题历史状态已知日数": int(cov.historical_st_known_eligible.sum()),
                     "原命题历史状态已知比例": float(ratio(cov.historical_st_known_eligible.sum(), cov.eligible_days.sum())),
                     "事件行业未知数": int(event_sub["起点历史行业"].eq("未知").sum()),
                     "事件情绪代理未知数": int(event_sub["情绪风险代理分组"].eq("未知").sum()),
                     "事件波动未知数": int(event_sub["波动率分组"].eq("未知").sum()),
                     "事件历史不足数": int(event_sub["历史成功率分组"].eq("历史不足").sum())})
    detail = e.rename(columns=SOURCE_NAMES).copy()
    detail["原始结果"] = e.status.map(STATES)
    detail["来源文件"] = SOURCE.name
    detail = detail.sort_values(["起点交易日序号", "来源代码"])
    save(detail, "事件明细", outputs)
    save(pd.DataFrame(summaries), "分期年度汇总", outputs)
    save(pd.DataFrame(industry_summaries), "行业分期年度汇总", outputs)
    save(pd.DataFrame(boot_rows), "日期块区间", outputs)
    save(pd.concat(daily_rows, ignore_index=True), "日期聚类计数", outputs)
    save(pd.DataFrame(gaps), "数据缺口", outputs)
    case_rows = []
    for example in examples:
        for kind, key in (("纳入", "纳入历史"), ("排除", "尚不可得历史")):
            for prior in example[key]:
                case_rows.append({"案例": example["案例"], "本次起点日期": example["本次起点日期"],
                                  "历史处置": kind, **prior})
    save(pd.DataFrame(case_rows), "历史标签审计案例", outputs)
    assert all((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == stat for p, stat in original.items())
    assert all(digest(Path(p)) == h for p, h in protected.items())
    assert all(Path(x["路径"]).parent == OUT and Path(x["路径"]).name.startswith("结构多维_") for x in outputs)
    audit = {
        "完成": True, "运行秒数": round(time.time() - started, 3), "脚本": str(Path(__file__).resolve()),
        "脚本校验": digest(Path(__file__)), "输入事件校验": digest(SOURCE), "历史行业表校验": digest(industry_path),
        "关键输入校验": {str(p): digest(p) for p in (DATA / "市场上下文.csv", DATA / "市场行业上下文.npz", DATA / "交易日历.csv")},
        "既有输入文件未改动检查数": len(original), "复用代码内容未改动": protected,
        "写入范围": "仅七份结构多维_中文CSV与结构多维审计.json；并行任务的无关文件不纳入输入完整性断言。",
        "缺真实行情代码": build.loc[build["状态"].eq("缺真实行情"), "代码"].tolist(),
        "版本": {"数值库": np.__version__, "表格库": pd.__version__},
        "原始字段中文映射": SOURCE_NAMES, "股票与数据缺口": gaps,
        "规则": {
            "边界": "只读既有数据；不联网、不读取密钥；仅指定新增文件；未导入交易信号、订单或账户模块。",
            "固定队列": "原C每股周EMA上升连续段第一个日突破，先峰确认再低点确认后再突破；初始低点先破失败，同日失败优先；原始状态完全保留，不重选事件。",
            "历史": "仅原C的60日事件；同股先前起点且标签终止可得日期严格早于当前起点；只纳入success/failure/unfinished；pending最小堆并逐事件直接过滤复核；不含120日重复。",
            "未完成标签可得性": "原C的unfinished终局日期为空/序号-1；验证观察截至=预定到期后，以预定第60交易日作标签终止可得日期；原始终局字段不改，删失不纳入。",
            "历史分母": "已分类=成功+失败+期限内未完成，至少5个才计算成功/已分类；这里已完成指60日结果已可知而非仅成功失败。2018以前事件不在输入中，不另造历史。",
            "历史边界": "历史不足；0-40%含40%；40-60%两端不含；60-100%含60%。",
            "市场宽度": "起点收盘已知的全可得沪深宽度六十；<40%、[40%,60%]、>60%，未知单列；上行/下行指宽度水平而不是斜率。主板同样使用该市场上下文。",
            "情绪代理": "起点收盘已知，估计跌停占比在[t-19,t]均值比较其原序列[t-249,t]中位数；严格20/250有效交易日，少于则未知；均值>中位数为较高，其余不高于；不是新闻情绪模型。",
            "波动": "个股d_volatility20乘sqrt(252)，固定<25%、[25%,50%]、>50%和未知，不用全样本分位。仅波动年化，不是收益年化。",
            "行业": "直接使用TradingData.industry_ids按事件起点索引；对原历史表start_date后向匹配逐事件核对，生效日期<=起点；未知单列；不按行业收益排序、选组或回填信号。",
            "分段": "全期维持原C已知终局；开发/验证/留出/2026及逐年按起点入段，预定start_i+60超过段末交易日则跨段，哪怕早已成功失败；超过2026已知日历亦跨段；跨段优先于其他删失。",
            "结果分母": "每层总样本包含成功失败未完成、右删失、路径缺口、初始低点缺失和跨段；删失率不包含单列跨段；已分类成功率=成功/(成功+失败+未完成)。",
            "区间": "两预定核心对照；在段内完整交易日轴同步抽20日移动块500次，固定种子；整日全部股票一起抽，保留零事件日，每次分别汇总分子分母求事件加权率差；非同日配对均值，非独立股票t检验。95%百分位区间。",
            "样本门槛": "预先固定的最低报告门槛：段长>=100日、每组已分类>=30且有事件>=20日且占用>=5个固定20日历交易日格；有效抽样>=475/500；未达标仅报点估计及样本不足，绝非功效保证。",
        },
        "验证": {"合成测试": tests, "历史逐事件检查数": len(e), "历史关联严格早于起点检查数": link_count,
                 "未终局历史排除关联数": exclusions, "同日终局排除关联数": same_day_exclusions,
                 "起点特征时序断言事件数": len(e), "行业后向匹配及波动重算股票数": e.code.nunique(),
                 "原始行情截断重算": prefix_checks, "与原C分期年度逐状态分母核对": reconciliations,
                 "全部分组分母守恒": True, "全部输出中文表头": True, "日历缺日未压缩": True},
        "历史标签实际案例": examples,
        "局限与未覆盖": [
            "可得5349股中主板3317；125只缺真实行情（主板83），不能称完整无幸存偏差全A。",
            "适配层历史ST状态约6%已知；原C原始字段的合格日状态覆盖更低，两种分母单列，不把未知当非ST。已知ST沿用原C也未排除。",
            "行业为历史生效映射，但缺逐次公告可得时间及数据库版本快照；不能保证供应商回修前的真实时点可得性。",
            "估计跌停采用普通涨跌幅代理且ST/新股规则与行情缺口有误差，不是真实精确跌停或新闻情绪。",
            "完整标签条件于可观察结局，删失可能非随机；20日块可能不足覆盖60日重叠依赖；多组多期未校正多重比较，不作因果结论。",
            "历史仅始于2018输入队列，早年或新股历史不足；历史留出已被本次描述性查看，不再是未来策略选择的未触碰测试集。",
            "不估算2万元账户收益，不校验100股整数、成交/停牌/涨跌停约束、费税滑点及组合资金；主板代码不是可执行股票池，不选择任何策略。",
            "仅预定单维分组及两核心对照；未分析交互、收益最强行业、新闻情绪模型、资金流、因果或阈值寻优。"],
        "输出": outputs,
    }
    audit_path = EVIDENCE / "结构多维审计.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(gaps).to_string(index=False), flush=True)
    print(pd.DataFrame(boot_rows).query('分期 == "全期"').to_string(index=False), flush=True)
    print(f"完成：{len(e)}个60日事件，{len(outputs)}份中文CSV；{audit_path}", flush=True)


if __name__ == "__main__":
    run()
