"""限定的描述性实证 A/B/C；不导入账户/订单模块，不选择策略，不年化。

运行：python3 -B /绝对路径/02_代码/结构命题实证.py
只读取分股 parquet、日历和元数据；只写命题实证_*.csv及命题实证审计.json。
收益为 t+h 后复权收盘 / t+1 后复权开盘 - 1，不是可执行账户收益。
第二驱动是本脚本固定的研究者定义，不是原文唯一公式。
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, AUDIT = ROOT / "01_数据", ROOT / "03_结果", ROOT / "04_证据"
START, END = "2018-01-01", "2026-10-22"
HORIZONS, C_HORIZONS = (20, 60), (60, 120)
SEED, BOOTSTRAPS, BLOCK, COOLDOWN = 20261023, 500, 20, 20
UNIVERSES = ("全可得沪深", "主板代码")
MAIN, PULL = "日回调_周up10_月up6_主", "日回调"
STATUS = ("complete", "data_end", "entry_missing", "exit_missing",
          "stock_coverage_end", "after_delisting")
C_STATUS = ("success", "failure", "unfinished", "right_censored",
            "path_gap_censored", "initial_low_missing", "cross_segment")
BOOL_COLS = ("d_breakout", "d_pullback_trigger", "w_up10", "m_up6", "q_up6")
FEATURE_COLS = list(BOOL_COLS) + [
    "d_open", "d_high", "d_low", "d_close", "d_age", "d_amount20",
    "d_volume20_prior", "d_volume5_prior", "d_pivot_high", "d_pivot_low",
    "w_ema10", "w_pivot_low", "m_ema6", "w_close", "m_close",
]
BINS = ("负值", "0至5%", "5至10%", "10至20%", "大于20%", "缺失")
GROUPS = ["同日合格基准", "日突破", PULL, "日回调_周up10", MAIN,
          "主_季up6", "主_当日量比1至2", "主_过去5均量小于20均量"]
GROUPS += [f"主_周EMA距离_{b}" for b in BINS]
GROUPS += [f"主_周低点止损距离_{b}" for b in BINS]
GROUPS += ["回调_月up周up", "回调_月up周非up", "回调_月非up周up",
           "回调_月非up周非up", "回调_月周状态未知"]
SCOPES = [("全期", START, "2026-12-31"),
          ("开发2018-2020", START, "2020-12-31"),
          ("验证2021-2022", "2021-01-01", "2022-12-31"),
          ("留出2023-2025", "2023-01-01", "2025-12-31"),
          ("2026", "2026-01-01", "2026-12-31")]
SCOPES += [(f"年度{y}", f"{y}-01-01", f"{y}-12-31") for y in range(2018, 2027)]
A_COLS = ("eligible", "previous_close_known", "bullish", "bearish", "flat",
          "bullish_but_down", "bearish_but_up")


def safe_div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def board(code):
    return ((code.endswith(".SH") and code.startswith(("600", "601", "603", "605")))
            or (code.endswith(".SZ") and code.startswith(("000", "001", "002", "003"))))


def cooldown(mask):
    # 对股票/组在整个时间轴首触发优先，不按标签、年份、阶段重置。
    kept, last = [], -COOLDOWN
    for i in np.flatnonzero(mask):
        if i - last >= COOLDOWN:
            kept.append(i)
            last = i
    return np.asarray(kept, dtype=int)


def distance_masks(x):
    # 边界：负值；[0,5%]；(5%,10%]；(10%,20%]；(20%,inf)。
    return [x < 0, (x >= 0) & (x <= .05), (x > .05) & (x <= .10),
            (x > .10) & (x <= .20), x > .20, ~np.isfinite(x)]


def event_masks(f, raw, eligible):
    # 必须 copy；绝不对 f 的布尔视图使用 &=、|= 等就地运算。
    b, p, w, m, q = [f[c].to_numpy(dtype=bool, copy=True) for c in BOOL_COLS]
    main = p & w & m
    v20, v5 = f.d_volume20_prior.to_numpy(), f.d_volume5_prior.to_numpy()
    ratio = safe_div(raw.volume.to_numpy(), v20)
    line = safe_div(f.d_close.to_numpy(), f.w_ema10.to_numpy()) - 1
    stop = 1 - safe_div(f.w_pivot_low.to_numpy(), f.d_close.to_numpy())
    known = np.isfinite(f.w_ema10) & np.isfinite(f.m_ema6)
    known = known.to_numpy() & np.isfinite(f.w_close) & np.isfinite(f.m_close)
    known = np.asarray(known)
    masks = [eligible.copy(), b, p, p & w, main, main & q,
             main & (ratio >= 1) & (ratio <= 2), main & (v5 < v20)]
    masks += [main & x for x in distance_masks(line)]
    masks += [main & x for x in distance_masks(stop)]
    masks += [p & known & m & w, p & known & m & ~w,
              p & known & ~m & w, p & known & ~m & ~w, p & ~known]
    assert len(masks) == len(GROUPS)
    return [np.asarray(x, dtype=bool) & eligible for x in masks]


def labels(op, cl, out_date, dates):
    n = len(dates)
    idx = np.arange(n)
    valid = np.isfinite(cl)
    last = np.flatnonzero(valid)[-1] if valid.any() else -1
    results = []
    for h in HORIZONS:
        entry, exit_ = idx + 1, idx + h
        s = np.zeros(n, dtype=np.int8)
        end = exit_ >= n
        missing_entry = ~np.isfinite(op[np.minimum(entry, n - 1)])
        missing_exit = ~np.isfinite(cl[np.minimum(exit_, n - 1)])
        missing = missing_entry | missing_exit
        s[missing_exit] = 3
        s[missing_entry] = 2
        s[missing & ((entry > last) | (exit_ > last))] = 4
        if pd.notna(out_date):
            after = dates[np.minimum(exit_, n - 1)] >= out_date
            s[missing & after] = 5
        s[end] = 1
        r = np.full(n, np.nan)
        k = idx[s == 0]
        r[k] = cl[k + h] / op[k + 1] - 1
        assert np.isfinite(r[s == 0]).all() and np.isnan(r[s != 0]).all()
        results.append((s, r))
    return results


def pivots(high, low):
    """重建两根有效K线确认事件，包含峰/谷发生日，而不是用ffill值变化猜事件。"""
    n = len(high)
    ix = np.flatnonzero(np.isfinite(high) & np.isfinite(low))
    values, origins = [], []
    for price, is_high in ((high, True), (low, False)):
        event = np.full(n, np.nan)
        origin = np.full(n, -1, dtype=int)
        if len(ix) >= 5:
            v = price[ix]
            center = v[2:-2]
            neighbors = np.stack([v[:-4], v[1:-3], v[3:-1], v[4:]])
            yes = np.all(center > neighbors, axis=0) if is_high else np.all(center < neighbors, axis=0)
            confirm, peak = ix[4:][yes], ix[2:-2][yes]
            event[confirm], origin[confirm] = price[peak], peak
        values.append(event)
        origins.append(origin)
    return values[0], origins[0], values[1], origins[1]


def first_in_week_segment(up, breakout):
    first, used = [], False
    for i in range(len(up)):
        if not up[i]:
            used = False
        elif breakout[i] and not used:
            first.append(i)
            used = True
    return np.asarray(first, dtype=int)


def trace_second(start, horizon, initial_low, high, low, ph, phi, pl, pli,
                 no_trade, last):
    """固定初始边界；失败优先；先高点确认、后低点确认、之后才允许突破。

    连续周up10只决定起点，未来翻转不取消/改写队列。已知停牌跳过；不明缺口
    在首次缺口删失，避免把缺失期间可能的失效当成未发生。终局发生后不再扫描。
    """
    r = dict(start_i=int(start), horizon=horizon, initial_low=float(initial_low),
             first_peak_i=-1, first_peak_confirm_i=-1, first_peak=np.nan,
             pullback_i=-1, pullback_confirm_i=-1, pullback_low=np.nan,
             second_break_i=-1, invalidation_i=-1, expiration_i=int(start + horizon),
             observed_through_i=int(start), terminal_i=-1, censor_i=-1,
             status="unfinished", censor_reason="", failure_priority_same_day=False)
    if not np.isfinite(initial_low) or initial_low <= 0:
        r.update(status="initial_low_missing", censor_reason="起点无已确认日低点", censor_i=int(start))
        return r
    for j in range(start + 1, start + horizon + 1):
        # 最后一根有效K之后若仍有明确停牌记录，仍属可观察的未发生，不能提前删失。
        if j >= len(high) or (j > last and not no_trade[j]):
            r.update(status="right_censored", censor_i=int(j),
                     censor_reason="研究数据末端" if j >= len(high) else "股票行情尾端")
            return r
        if not np.isfinite(high[j]) or not np.isfinite(low[j]):
            if no_trade[j]:
                r["observed_through_i"] = int(j)
                continue
            r.update(status="path_gap_censored", censor_i=int(j), censor_reason="无法确认停牌的行情缺口")
            return r
        r["observed_through_i"] = int(j)
        can_break = r["pullback_confirm_i"] >= 0 and j > r["pullback_confirm_i"]
        touched_peak = can_break and high[j] > r["first_peak"]
        if low[j] < initial_low:
            r.update(status="failure", invalidation_i=int(j), terminal_i=int(j),
                     failure_priority_same_day=bool(touched_peak))
            return r
        if touched_peak:
            r.update(status="success", second_break_i=int(j), terminal_i=int(j))
            return r
        if r["first_peak_confirm_i"] < 0:
            if np.isfinite(ph[j]) and phi[j] > start:
                r.update(first_peak_i=int(phi[j]), first_peak_confirm_i=int(j), first_peak=float(ph[j]))
        elif r["pullback_confirm_i"] < 0:
            if (np.isfinite(pl[j]) and j > r["first_peak_confirm_i"]
                    and pli[j] > r["first_peak_i"]):
                r.update(pullback_i=int(pli[j]), pullback_confirm_i=int(j), pullback_low=float(pl[j]))
    return r


def paired_bootstrap(x, seed):
    """先按日期均值；在完整市场日历上抽20日移动块，不压缩无事件日。"""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    n = len(x)
    length = min(BLOCK, n)
    starts = rng.integers(0, n - length + 1, size=(BOOTSTRAPS, math.ceil(n / length)))
    indices = (starts[:, :, None] + np.arange(length)).reshape(BOOTSTRAPS, -1)[:, :n]
    sampled = x[indices]
    count = np.isfinite(sampled).sum(axis=1)
    means = safe_div(np.nansum(sampled, axis=1), count)
    finite = means[np.isfinite(means)]
    if not len(finite):
        return np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [.025, .975])
    return float(lo), float(hi), len(finite)


def self_tests():
    tests = []
    def ok(name, condition):
        assert condition, name
        tests.append(name)
    ok("冷却固定20日且首触发保留", cooldown(np.isin(np.arange(50), [0, 1, 19, 20, 21, 40])).tolist() == [0, 20, 40])
    ok("周状态false才重开段", first_in_week_segment([1, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1]).tolist() == [1, 4])
    n = 18
    h, l = np.full(n, 12.), np.full(n, 9.)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    phi, pli = np.full(n, -1), np.full(n, -1)
    ph[2], phi[2] = 99, 0  # 起点之前的峰必须忽略。
    ph[5], phi[5], pl[8], pli[8] = 14, 3, 9, 6
    h[10] = 15
    nt = np.zeros(n, dtype=bool)
    def trace(hi=h, lo=l, horizon=12, end=17, hh=ph, ll=pl, ip=phi, il=pli, no=nt):
        return trace_second(1, horizon, 8, hi, lo, hh, ip, ll, il, no, end)
    r = trace()
    ok("新峰确认回调确认再突破", r["status"] == "success" and r["first_peak_i"] == 3 and r["second_break_i"] == 10)
    lo = l.copy(); lo[10] = 7
    r = trace(lo=lo)
    ok("同日触峰破底失败优先", r["status"] == "failure" and r["failure_priority_same_day"])
    lo = l.copy(); lo[4] = 7
    ok("先破初始边界立即失败", trace(lo=lo)["invalidation_i"] == 4)
    hi = h.copy(); hi[10] = 12
    ok("普通新高不能代替完整二驱动", trace(hi=hi, ll=np.full(n, np.nan))["status"] == "unfinished")
    hi = h.copy(); hi[8] = 16; hi[10] = 12
    ok("低点确认日不追认盘中突破", trace(hi=hi)["status"] == "unfinished")
    ok("期限内未完成单列", trace(horizon=7)["status"] == "unfinished")
    ok("未来不足右删失", trace(end=7)["status"] == "right_censored")
    hi, lo = h.copy(), l.copy(); hi[7] = lo[7] = np.nan
    ok("中途不明缺口删失", trace(hi=hi, lo=lo)["status"] == "path_gap_censored")
    no = nt.copy(); no[7] = True
    ok("已知停牌不造K线且可继续", trace(hi=hi, lo=lo, no=no)["status"] == "success")
    hi, lo, no = h.copy(), l.copy(), nt.copy()
    hi[8:] = lo[8:] = np.nan; no[8:] = True
    ok("行情尾端明确停牌至到期算未完成", trace(hi=hi, lo=lo, end=7, no=no)["status"] == "unfinished")
    ph2, phi2, pl2, pli2 = pivots(np.array([1., 2., 5., 2., 1.]), np.array([1., 2., .5, 2., 1.]))
    ok("五根严格分型两根确认", ph2[4] == 5 and phi2[4] == 2 and pl2[4] == .5 and pli2[4] == 2)
    dd = pd.date_range("2020-01-01", periods=70)
    op, cl = np.full(70, 10.), np.full(70, 12.)
    op[2] = np.nan; cl[22] = np.nan
    ss, rr = labels(op, cl, pd.NaT, dd)[0]
    ok("标签使用次日开盘和t加h收盘", np.isclose(rr[0], .2))
    ok("缺端点不填零且末端保留", ss[1] == 2 and ss[2] == 3 and ss[-1] == 1 and np.isnan(rr[1]))
    a = paired_bootstrap(np.arange(45, dtype=float), SEED)
    ok("块自助固定种子500次", a == paired_bootstrap(np.arange(45, dtype=float), SEED) and a[2] == 500)
    return tests


def run():
    started = time.time()
    tests = self_tests()
    for directory in (OUT, AUDIT):
        assert directory.is_dir()
    c = pd.read_csv(DATA / "交易日历.csv")
    calendar = pd.DatetimeIndex(pd.to_datetime(c.loc[c["交易日标记"].eq(1), "日期"]))
    calendar = calendar[calendar >= "2016-10-13"]
    assert calendar.is_monotonic_increasing and not calendar.has_duplicates
    dates = calendar[calendar <= END]
    n = len(dates)
    dates_str = np.array(dates.strftime("%Y-%m-%d"))
    calendar_str = np.array(calendar.strftime("%Y-%m-%d"))
    positions = np.arange(n)
    scopes = [(name, (dates >= begin) & (dates <= end),
               int(calendar.searchsorted(pd.Timestamp(end), side="right") - 1))
              for name, begin, end in SCOPES]
    names = pd.read_parquet(DATA / "需补状态名单.parquet")
    metadata = names.set_index("thscode")
    build_log = pd.read_csv(DATA / "特征构建审计.csv")
    expected = set(build_log.loc[build_log["状态"].eq("成功"), "代码"])
    files = sorted((DATA / "结构特征").glob("*.parquet"))
    assert {p.stem for p in files} == expected
    feature_source = ROOT / "02_代码" / "结构特征.py"
    feature_hash = hashlib.sha256(feature_source.read_bytes()).hexdigest()
    version = json.loads((DATA / "结构特征版本.json").read_text())
    assert version["代码校验"] == feature_hash

    ng = len(GROUPS)
    candidate = np.zeros((2, ng, n), dtype=np.int64)
    kept_count = np.zeros_like(candidate)
    status_count = np.zeros((2, ng, n, 2, len(STATUS)), dtype=np.int64)
    return_sum = np.zeros((2, ng, n, 2), dtype=float)
    positive = np.zeros((2, ng, n, 2), dtype=np.int64)
    acount = np.zeros((2, n, len(A_COLS)), dtype=np.int64)
    examples = {(u, k): [] for u in range(2) for k in ("阳线但前收下跌", "阴线但前收上涨")}
    coverage, cohorts, manifest = [], [], []
    totals = dict(stocks=0, raw_rows=0, feature_rows=0, feature_columns=108,
                  valid_stock_days=0, eligible_stock_days=0, historical_st_known_eligible=0,
                  historical_st_true_eligible=0, trade_status_known_eligible=0,
                  first_breakouts_in_week_segments=0, first_breakouts_since_2018=0,
                  first_breakouts_eligible=0, missing_path_days=0, boolean_unchanged_stocks=0,
                  pivot_reconstruction_matched_stocks=0, raw_feature_price_matched_stocks=0,
                  positive_trade_but_status_zero_days=0)
    bool_digest = hashlib.sha256()
    actual_checks, status_conflicts = [], []
    check_quota = {s: 0 for s in ("success", "failure")}

    for stock_no, path in enumerate(files, 1):
        code = path.stem
        assert code.endswith((".SH", ".SZ"))
        raw_path = DATA / "原始分股" / path.name
        raw0 = pd.read_parquet(raw_path)
        assert len(pq.ParquetFile(path).schema_arrow.names) == 109  # 108列加date索引
        f = pd.read_parquet(path, columns=FEATURE_COLS)
        assert f.index.equals(dates) and not raw0.index.has_duplicates
        assert raw0.index.is_monotonic_increasing and raw0.index.isin(dates).all()
        raw = raw0.reindex(dates)
        before = f.loc[:, BOOL_COLS].to_numpy(copy=True)
        op, hi, lo, cl = [f[f"d_{s}"].to_numpy(copy=True) for s in ("open", "high", "low", "close")]
        valid = np.isfinite(cl)
        raw_valid = (np.isfinite(raw[["open", "high", "low", "close", "volume", "amount", "adj_factor"]]).all(axis=1)
                     & raw[["open", "high", "low", "close", "volume", "amount", "adj_factor"]].gt(0).all(axis=1))
        raw_valid &= raw.high.ge(raw[["open", "close", "low"]].max(axis=1)) & raw.low.le(raw[["open", "close"]].min(axis=1))
        assert np.array_equal(valid, raw_valid.to_numpy())
        for a, s in zip((op, hi, lo, cl), ("open", "high", "low", "close")):
            np.testing.assert_allclose(a[valid], (raw[s] * raw.adj_factor).to_numpy()[valid], rtol=1e-12, atol=0)
        totals["raw_feature_price_matched_stocks"] += 1
        st = pd.to_numeric(raw["isST"], errors="coerce") if "isST" in raw else pd.Series(np.nan, index=dates)
        ts = pd.to_numeric(raw["tradestatus"], errors="coerce") if "tradestatus" in raw else pd.Series(np.nan, index=dates)
        conflict = valid & ts.eq(0).to_numpy()
        totals["positive_trade_but_status_zero_days"] += int(conflict.sum())
        for i in np.flatnonzero(conflict):
            status_conflicts.append(dict(code=code, date=dates_str[i], volume=float(raw.volume.iloc[i]),
                                         amount=float(raw.amount.iloc[i]), raw_close=float(raw.close.iloc[i])))
        eligible = ((dates >= START) & valid & f.d_age.ge(120).to_numpy()
                    & f.d_amount20.ge(3e7).to_numpy() & raw.close.ge(2).to_numpy()
                    & raw.volume.gt(0).to_numpy() & raw.amount.gt(0).to_numpy())
        mainboard = board(code)
        universes = (0, 1) if mainboard else (0,)
        out_date = pd.to_datetime(metadata.loc[code, "outDate"], errors="coerce")
        no_trade = ~valid & (ts.eq(0) | raw.volume.eq(0) | raw.amount.eq(0)).to_numpy()
        ix = np.flatnonzero(valid)
        last = int(ix[-1]) if len(ix) else -1
        first = int(ix[0]) if len(ix) else n
        path_gap = (~valid) & (~no_trade) & (positions >= first) & (positions <= last)
        gap_n = int(path_gap.sum())
        totals["missing_path_days"] += gap_n
        totals["stocks"] += 1
        totals["raw_rows"] += len(raw0)
        totals["feature_rows"] += len(f)
        totals["valid_stock_days"] += int(valid.sum())
        totals["eligible_stock_days"] += int(eligible.sum())
        totals["historical_st_known_eligible"] += int((eligible & st.isin([0, 1])).sum())
        totals["historical_st_true_eligible"] += int((eligible & st.eq(1)).sum())
        totals["trade_status_known_eligible"] += int((eligible & ts.isin([0, 1])).sum())
        coverage.append(dict(code=code, mainboard=mainboard, status="loaded", raw_rows=len(raw0),
                             feature_rows=len(f), raw_first=str(raw0.index.min().date()), raw_last=str(raw0.index.max().date()),
                             valid_first=dates_str[first] if first < n else "", valid_last=dates_str[last] if last >= 0 else "",
                             valid_days=int(valid.sum()), eligible_days=int(eligible.sum()), path_gap_days=gap_n,
                             historical_st_known_eligible=int((eligible & st.isin([0, 1])).sum()),
                             historical_st_true_eligible=int((eligible & st.eq(1)).sum()),
                             trade_status_known_eligible=int((eligible & ts.isin([0, 1])).sum()),
                             delist_date=str(out_date.date()) if pd.notna(out_date) else ""))
        manifest.append((code, raw_path.stat().st_size, raw_path.stat().st_mtime_ns,
                         path.stat().st_size, path.stat().st_mtime_ns, len(raw0), len(f)))

        # A：上一有效收盘；日内阴阳用原价开收，不从当前名称推历史状态。
        prev_i = np.maximum.accumulate(np.where(valid, positions, -1))
        prev_i = np.concatenate(([-1], prev_i[:-1]))
        prev = np.where(prev_i >= 0, cl[np.maximum(prev_i, 0)], np.nan)
        known_prev = np.isfinite(prev)
        bull = raw.close.gt(raw.open).to_numpy()
        bear = raw.close.lt(raw.open).to_numpy()
        bull_down = bull & (cl < prev)
        bear_up = bear & (cl > prev)
        columns = [eligible, eligible & known_prev, eligible & known_prev & bull,
                   eligible & known_prev & bear, eligible & known_prev & ~(bull | bear),
                   eligible & bull_down, eligible & bear_up]
        for u in universes:
            acount[u] += np.stack(columns, axis=1)
            for kind, mask in (("阳线但前收下跌", bull_down), ("阴线但前收上涨", bear_up)):
                for i in np.flatnonzero(eligible & mask)[:20]:
                    j = prev_i[i]
                    examples[u, kind].append(dict(universe=UNIVERSES[u], kind=kind, date=dates_str[i], code=code,
                        previous_valid_date=dates_str[j], raw_open=float(raw.open.iloc[i]), raw_close=float(raw.close.iloc[i]),
                        adjusted_open=float(op[i]), adjusted_close=float(cl[i]), previous_adjusted_close=float(prev[i]),
                        close_to_previous_return=float(cl[i] / prev[i] - 1), volume=float(raw.volume.iloc[i]),
                        amount=float(raw.amount.iloc[i]), d_amount20=float(f.d_amount20.iloc[i]), age=int(f.d_age.iloc[i]),
                        historical_isST=float(st.iloc[i]) if pd.notna(st.iloc[i]) else None))
                examples[u, kind] = sorted(examples[u, kind], key=lambda r: (r["date"], r["code"]))[:20]

        # B：标签不参与去重；基准是该日期全部合格股票，不做冷却。
        lr = labels(op, cl, out_date, dates)
        masks = event_masks(f, raw, eligible)
        for g, mask in enumerate(masks):
            chosen = np.flatnonzero(mask) if g == 0 else cooldown(mask)
            assert g == 0 or len(chosen) < 2 or np.diff(chosen).min() >= COOLDOWN
            for u in universes:
                candidate[u, g] += mask
                kept_count[u, g, chosen] += 1
                for hidx, (status, ret) in enumerate(lr):
                    status_count[u, g, chosen, hidx, status[chosen]] += 1
                    complete = chosen[status[chosen] == 0]
                    return_sum[u, g, complete, hidx] += ret[complete]
                    positive[u, g, complete, hidx] += ret[complete] > 0

        # C：首触发选择先于资格过滤，不能在同一周段等到后一个合格突破。
        ph, phi, pl, pli = pivots(hi, lo)
        for event, col in ((ph, "d_pivot_high"), (pl, "d_pivot_low")):
            rebuilt = pd.Series(event).ffill().to_numpy()
            np.testing.assert_allclose(rebuilt, f[col].to_numpy(), rtol=1e-12, atol=0, equal_nan=True)
        totals["pivot_reconstruction_matched_stocks"] += 1
        firsts = first_in_week_segment(before[:, 2], before[:, 0])
        totals["first_breakouts_in_week_segments"] += len(firsts)
        totals["first_breakouts_since_2018"] += int((dates[firsts] >= START).sum())
        selected = firsts[eligible[firsts]]
        totals["first_breakouts_eligible"] += len(selected)
        for i in selected:
            for horizon in C_HORIZONS:
                r = trace_second(int(i), horizon, f.d_pivot_low.iloc[i], hi, lo, ph, phi, pl, pli, no_trade, last)
                r.update(code=code, mainboard=mainboard, weekly_up_at_start=True,
                         start_adjusted_close=float(cl[i]), initial_low_known_at_start=True,
                         delist_date=str(out_date.date()) if pd.notna(out_date) else "")
                r["initial_low_known_at_start"] = bool(np.isfinite(r["initial_low"]))
                peak_c, pull_c, success, failure = (r[k] for k in ("first_peak_confirm_i", "pullback_confirm_i", "second_break_i", "invalidation_i"))
                if peak_c >= 0:
                    assert r["first_peak_i"] > i and peak_c > r["first_peak_i"]
                    assert phi[peak_c] == r["first_peak_i"]
                if pull_c >= 0:
                    assert pull_c > peak_c and r["pullback_i"] > r["first_peak_i"]
                if success >= 0:
                    assert i < peak_c < pull_c < success <= i + horizon and failure < 0
                    assert hi[success] > r["first_peak"]
                    assert not np.any(lo[i + 1:success + 1] < r["initial_low"])
                if failure >= 0:
                    assert i < failure <= i + horizon and success < 0 and lo[failure] < r["initial_low"]
                    assert not np.any(lo[i + 1:failure] < r["initial_low"])
                if r["status"] in check_quota and check_quota[r["status"]] < 4 and horizon == 60:
                    k = r["status"]
                    actual_checks.append(dict(code=code, status=k, start=dates_str[i],
                        initial_low=r["initial_low"], first_peak_confirm=dates_str[peak_c] if peak_c >= 0 else None,
                        pullback_confirm=dates_str[pull_c] if pull_c >= 0 else None,
                        terminal=dates_str[r["terminal_i"]], first_peak=r["first_peak"],
                        terminal_high=float(hi[r["terminal_i"]]), terminal_low=float(lo[r["terminal_i"]]),
                        assertions_passed=True))
                    check_quota[k] += 1
                cohorts.append(r)
        after = f.loc[:, BOOL_COLS].to_numpy(copy=True)
        assert np.array_equal(before, after)
        bool_digest.update(code.encode()); bool_digest.update(before.tobytes())
        totals["boolean_unchanged_stocks"] += 1
        if stock_no % 250 == 0 or stock_no == len(files):
            print(json.dumps({"stocks": stock_no, "total": len(files), "elapsed_seconds": round(time.time() - started, 1),
                              "C_horizon_rows": len(cohorts)}, ensure_ascii=False), flush=True)

    missing = build_log.loc[build_log["状态"].ne("成功"), "代码"].sort_values().tolist()
    for code in missing:
        coverage.append(dict(code=code, mainboard=board(code), status="missing_real_prices", raw_rows=0, feature_rows=0))
    outputs = []
    def save(frame, suffix):
        path = OUT / f"命题实证_{suffix}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")
        outputs.append(dict(path=str(path), rows=len(frame), columns=len(frame.columns), bytes=path.stat().st_size))

    save(pd.DataFrame(coverage).sort_values("code"), "数据覆盖")
    ar = []
    for scope, mask, _ in scopes:
        for u, universe in enumerate(UNIVERSES):
            row = dict(scope=scope, universe=universe)
            row.update(zip(A_COLS, acount[u, mask].sum(axis=0).tolist()))
            row.update(bullish_down_rate_of_bullish=float(safe_div(row["bullish_but_down"], row["bullish"])),
                       bearish_up_rate_of_bearish=float(safe_div(row["bearish_but_up"], row["bearish"])),
                       bullish_down_rate_of_eligible=float(safe_div(row["bullish_but_down"], row["eligible"])),
                       bearish_up_rate_of_eligible=float(safe_div(row["bearish_but_up"], row["eligible"])))
            ar.append(row)
    save(pd.DataFrame(ar), "A_阴阳与前收基准率")
    save(pd.DataFrame([r for rows in examples.values() for r in rows]).sort_values(["date", "code", "universe", "kind"]), "A_固定排序真实反例")

    complete_count = status_count[..., 0]
    assert np.array_equal(status_count.sum(axis=-1), np.repeat(kept_count[..., None], 2, axis=-1))
    assert np.all(kept_count <= candidate)
    mean_return = safe_div(return_sum, complete_count)
    baseline = mean_return[:, 0, :, :]
    excess = mean_return - baseline[:, None, :, :]
    # 同日事件标签存在时，基准必然至少包含此股票，不能另行选择基准样本。
    assert np.isfinite(excess[complete_count > 0]).all()
    summary, daily = [], []
    for scope, mask, scope_end in scopes:
        for u, universe in enumerate(UNIVERSES):
            for g, group in enumerate(GROUPS):
                for hi_idx, h in enumerate(HORIZONS):
                    cross = np.zeros(n, dtype=bool) if scope == "全期" else positions + h > scope_end
                    allowed = mask & ~cross
                    counts = status_count[u, g, allowed, hi_idx].sum(axis=0)
                    ncross = int(kept_count[u, g, mask & cross].sum())
                    nsignal = int(kept_count[u, g, mask].sum())
                    assert counts.sum() + ncross == nsignal
                    ncomplete = int(counts[0])
                    days = allowed & (complete_count[u, g, :, hi_idx] > 0)
                    day_n = int(days.sum())
                    row = dict(scope=scope, universe=universe, group=group, horizon=h,
                        candidate_signals=int(candidate[u, g, mask].sum()), signal_denominator=nsignal,
                        cooldown_removed=int(candidate[u, g, mask].sum()) - nsignal,
                        cross_segment_censored=ncross, completed_dates=day_n,
                        event_mean_label=float(safe_div(return_sum[u, g, allowed, hi_idx].sum(), ncomplete)),
                        event_positive_fraction=float(safe_div(positive[u, g, allowed, hi_idx].sum(), ncomplete)),
                        date_mean_label=float(np.mean(mean_return[u, g, days, hi_idx])) if day_n else np.nan,
                        date_mean_baseline=float(np.mean(baseline[u, days, hi_idx])) if day_n else np.nan,
                        date_mean_excess=float(np.mean(excess[u, g, days, hi_idx])) if day_n else np.nan,
                        event_mean_excess=float(safe_div(np.sum(excess[u, g, days, hi_idx] * complete_count[u, g, days, hi_idx]), ncomplete)))
                    row.update({f"n_{s}": int(counts[k]) for k, s in enumerate(STATUS)})
                    summary.append(row)
                    if scope == "全期":
                        for i in np.flatnonzero(mask & (candidate[u, g] > 0)):
                            item = dict(universe=universe, group=group, date=dates_str[i], horizon=h,
                                candidate_signals=int(candidate[u, g, i]), signal_denominator=int(kept_count[u, g, i]),
                                date_mean_label=mean_return[u, g, i, hi_idx], same_date_baseline=baseline[u, i, hi_idx],
                                date_mean_excess=excess[u, g, i, hi_idx])
                            item.update({f"n_{s}": int(status_count[u, g, i, hi_idx, k]) for k, s in enumerate(STATUS)})
                            daily.append(item)
    summary_df = pd.DataFrame(summary)
    save(summary_df, "B_分期年度分组汇总")
    save(pd.DataFrame(daily), "B_逐日均值与删失")

    bs = []
    mg, pg = GROUPS.index(MAIN), GROUPS.index(PULL)
    for si, (scope, mask, scope_end) in enumerate(scopes):
        for u, universe in enumerate(UNIVERSES):
            for hi_idx, h in enumerate(HORIZONS):
                timeline = np.flatnonzero(mask)
                allowed = np.ones(n, dtype=bool) if scope == "全期" else positions + h <= scope_end
                pair = allowed & (complete_count[u, mg, :, hi_idx] > 0) & (complete_count[u, pg, :, hi_idx] > 0)
                difference = np.full(n, np.nan)
                difference[pair] = excess[u, mg, pair, hi_idx] - excess[u, pg, pair, hi_idx]
                x = difference[timeline]
                seed = SEED + si * 100 + u * 10 + hi_idx
                lower, upper, valid_replicates = paired_bootstrap(x, seed)
                npair = int(np.isfinite(x).sum())
                bs.append(dict(scope=scope, universe=universe, horizon=h, comparison=f"{MAIN}减{PULL}",
                    paired_dates=npair, calendar_days=len(timeline),
                    date_mean_excess_difference=float(np.nanmean(x)) if npair else np.nan,
                    ci95_low=lower, ci95_high=upper, bootstrap_replicates=BOOTSTRAPS,
                    valid_replicates=valid_replicates, block_trading_days=BLOCK, seed=seed))
    bs_df = pd.DataFrame(bs)
    save(bs_df, "B_主组减日回调_日期块bootstrap")

    cd = pd.DataFrame(cohorts).sort_values(["start_i", "code", "horizon"]).reset_index(drop=True)
    index_cols = [col for col in cd if col.endswith("_i")]
    for col in index_cols:
        vals = cd[col].to_numpy()
        cd[col.removesuffix("_i") + "_date"] = [calendar_str[i] if 0 <= i < len(calendar_str) else "" for i in vals]
    cd["expiration_beyond_available_calendar"] = cd.expiration_i >= len(calendar)
    cd["definition"] = "研究者固定二驱动定义_非原文唯一公式"
    assert len(cd) == totals["first_breakouts_eligible"] * 2
    # 较长期不得推翻较短期已观察到的终局；短期未完成可以在后续终结。
    c60 = cd[cd.horizon.eq(60)].set_index(["code", "start_i"])
    c120 = cd[cd.horizon.eq(120)].set_index(["code", "start_i"])
    for terminal_status in ("success", "failure", "initial_low_missing", "path_gap_censored", "right_censored"):
        keys = c60.index[c60.status.eq(terminal_status)]
        assert c120.loc[keys, "status"].eq(terminal_status).all()
    save(cd, "C_第二驱动事件明细")
    cs = []
    for scope, mask, scope_end in scopes:
        for u, universe in enumerate(UNIVERSES):
            for h in C_HORIZONS:
                select = mask[cd.start_i.to_numpy()] & cd.horizon.eq(h).to_numpy()
                if u == 1:
                    select &= cd.mainboard.to_numpy()
                sub = cd.loc[select]
                statuses = sub.status.copy()
                if scope != "全期":
                    statuses.loc[sub.expiration_i > scope_end] = "cross_segment"
                counts = statuses.value_counts()
                row = dict(scope=scope, universe=universe, horizon=h, cohort_denominator=len(sub))
                row.update({f"n_{s}": int(counts.get(s, 0)) for s in C_STATUS})
                observed = row["n_success"] + row["n_failure"] + row["n_unfinished"]
                row.update(fully_classified_denominator=observed,
                    success_fraction_of_all=float(safe_div(row["n_success"], len(sub))),
                    success_fraction_of_classified=float(safe_div(row["n_success"], observed)),
                    failure_fraction_of_classified=float(safe_div(row["n_failure"], observed)))
                assert sum(row[f"n_{s}"] for s in C_STATUS) == len(sub)
                cs.append(row)
    cs_df = pd.DataFrame(cs)
    save(cs_df, "C_第二驱动分期年度汇总")
    for obj in actual_checks:
        for k, v in obj.items():
            if isinstance(v, float) and not math.isfinite(v):
                obj[k] = None
    audit = {
        "finished": True, "runtime_seconds": round(time.time() - started, 3),
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_module_sha256": feature_hash,
        "feature_boolean_content_sha256": bool_digest.hexdigest(),
        "input_manifest_stat_sha256": hashlib.sha256(json.dumps(manifest, ensure_ascii=False).encode()).hexdigest(),
        "input_manifest_note": "文件名/大小/mtime/行数摘要，非全量二进制哈希；全部股票逐值核对后复权OHLC和两根确认分型。",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "date_ranges": {"feature_start": dates_str[0], "feature_end": dates_str[-1],
                        "feature_calendar_days": n, "analysis_start_lower_bound": START,
                        "eligible_first": dates_str[np.flatnonzero(acount[0, :, 0])[0]],
                        "eligible_last": dates_str[np.flatnonzero(acount[0, :, 0])[-1]],
                        "analysis_calendar_days": int((dates >= START).sum()),
                        "known_calendar_end": calendar_str[-1]},
        "actual_rows": totals, "missing_real_price_stocks": len(missing), "missing_codes": missing,
        "source_build_status_counts": build_log["状态"].value_counts().to_dict(),
        "positive_trade_status_conflicts": status_conflicts,
        "rules": {
            "trade_status_conflict": "若tradestatus=0但原价/因子/量额均为正且OHLC合法，按精确价量资格保留有效K，逐条记录冲突；不将有效成交当成无成交。",
            "eligibility": "2018起，有效原始OHLC/成交量/成交额/复权因子正有限，d_age>=120，d_amount20>=3000万，原价close>=2。A/B/C同一资格。",
            "ST": "不从证券当前名称回填历史ST；已知ST也未排除，原任务精确资格未要求排除。未知历史状态不当非ST。主板代码宇宙不是普通非ST可交易清单。",
            "universe": "全可得沪深.SH/.SZ；主板为SH600/601/603/605，SZ000/001/002/003。缺125股使可得样本仍有幸存/覆盖偏差。",
            "A": "阴阳按当日原价close-open；涨跌按当日后复权close对上一有效后复权close。每宇宙每反例类型日期+代码升序前20。",
            "B_label": "t+h后复权close/(t+1后复权open)-1；h=20/60个市场交易日。缺端点不补、不当0、不延迟入场。非账户收益，无撮合、费用、滑点、涨跌停、资金和整股约束，不年化。",
            "B_cooldown": "股票/组独立，首次合格信号优先，相邻保留起点索引差>=20；跨年跨阶段不重置；不查看未来标签；基准全部合格股票不去重。子组独立冷却后不必是主组已保留事件的子集。",
            "B_baseline": "同日期同宇宙全部合格股票可观测20/60日标签均值。基准也保留全部资格分母及删失；缺失结局未知不能补零。超额=股票标签减同日基准。",
            "B_summary": "同时输出事件加权均值及先股票后日期均值，bootstrap使用同日两个组均有完整标签的配对日均超额差。主组与未加框架回调有重叠，不是独立样本。",
            "censor_priority": "分段汇总先跨段；其余先数据末端，再已知退市日期之后缺端点，再股票最后有效行情之后缺端点，再次日开盘缺失，再出场收盘缺失。各原因互斥、加回完整分母。",
            "scope": "全期、开发2018-2020、验证2021-2022、留出2023-2025、2026及逐年。所有分段按预定t+h在本段内保留完整标签；跨段单列，2026超日历末日必跨段。全期无跨段剔除。",
            "volume": "当日原始量/过去20市场交易日均量在[1,2]；过去5均量<过去20均量，两均量均不含当日。",
            "distance": "EMA距离=d_close/w_ema10-1；周低点止损距离=(d_close-w_pivot_low)/d_close。分箱<0，[0,.05]，(.05,.10]，(.10,.20]，>.20及缺失。",
            "framework": "回调按月up6/周up10四组合及均线未形成的未知组。非up不等于明确下降；不偷换成空头。",
            "bootstrap": "在各段完整市场交易日轴抽20交易日移动块500次，固定seed=20261023加确定性范围/宇宙/h编号；无配对日保留NaN而非压缩日历；95%百分位区间；不做独立股票t检验。",
            "C_start": "全部已知历史内每股w_up10连续true段首次d_breakout，先找首突破再过滤2018和资格，不把后续合格突破替代首个；false才新段，未来不因框架变false取消或改变原队列。",
            "C_second_drive": "研究者固定定义而非原文唯一公式：起点已确认d_pivot_low固定失效边界。从t+1监测，先取峰发生日>t的新日高点（严格五根分型，右两根有效K确认），随后确认发生日在该峰之后的日低点；低点确认后下一个市场交易日起，日high>固定第一峰才成功。新高本身不叫第二波。",
            "C_failure": "日low<固定初始低点先判失败；同日low触发失败且high触发二突破时失败优先。终局前每一天检查，不等收盘。平于边界不算严格突破/跌破。",
            "C_missing": "60/120市场日内未达终局=unfinished；数据/个股有效行情尾端前未达终局=right_censored。中途无行情且非明确0量/额或tradestatus=0，首次缺口=path_gap_censored；初始低点缺失独立删失。已知无成交跳过，不造K线。已观察到终局不要求未来凑满期限。分段表仍以预定horizon跨段规则删失。",
            "C_expiration": "保留市场日索引到期值。真实日历只到2026-12-31，超出日历的到期日期留空并标记，绝不伪造2027交易日。",
        },
        "synthetic_tests_passed": tests,
        "real_validation": {"per_stock_adjusted_ohlc_matches": totals["raw_feature_price_matched_stocks"],
                            "per_stock_confirmed_pivots_match": totals["pivot_reconstruction_matched_stocks"],
                            "per_stock_boolean_unchanged": totals["boolean_unchanged_stocks"],
                            "all_C_rows_temporal_assertions_passed": len(cd),
                            "horizon_terminal_consistency_passed": True,
                            "B_and_C_denominator_conservation_passed": True,
                            "chronology_examples": actual_checks},
        "limitations": ["125只缺真实行情；可得宇宙不能消除退市/幸存偏差。",
                        "历史ST/交易状态覆盖不足，主板代码不等于2万元普通非ST可执行宇宙。",
                        "端点收益是描述性标记，不是策略或账户净收益；未模拟成本/撮合/限价/资金。",
                        "完整标签及配对日期区间条件于可观察结局，缺失可能非随机。",
                        "多组多期描述性比较未作多重检验校正，不能据此筛选最优方案或声称因果。",
                        "固定20日块可能不足以覆盖60日重叠标签的全部依赖，区间仅为指定块长的敏感性有限估计。",
                        "没有加载行业分组；未训练、优化或选择账户策略。"],
        "outputs": outputs,
    }
    audit_path = AUDIT / "命题实证审计.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("A full period:", flush=True)
    print(pd.DataFrame(ar).query('scope == "全期"').to_string(index=False), flush=True)
    print("B main minus pullback bootstrap:", flush=True)
    print(bs_df[~bs_df.scope.str.startswith("年度")].to_string(index=False), flush=True)
    print("C full period:", flush=True)
    print(cs_df.query('scope == "全期"').to_string(index=False), flush=True)
    print(json.dumps({"audit": str(audit_path), "rows": totals, "outputs": outputs,
                      "seconds": audit["runtime_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
