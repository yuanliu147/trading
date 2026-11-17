"""只读现有真实行情，分股票生成原始行情与因果特征；不产生策略收益。"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import duckdb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from 结构特征 import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_数据"
DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
START, END = "2016-10-13", "2026-10-22"


def calendar():
    c = pd.read_csv(DATA / "交易日历.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(c.loc[c["交易日标记"].eq(1), "日期"]))
    # 开始日反映原始数据覆盖边界，不能把2016年末三个月当完整2016年。
    return dates[dates >= START].rename("date")


def load_raw(code, connection, missing=False):
    if missing:
        a = DATA / "缺失股原始行情" / f"{code}.parquet"
        b = DATA / "缺失股后复权行情" / f"{code}.parquet"
        if not a.exists() or not b.exists():
            return None
        x = pd.read_parquet(a)
        h = pd.read_parquet(b)
        x = x.merge(h[["date", "close"]].rename(columns={"close": "hfq_close"}), on="date", validate="one_to_one")
        x["adj_factor"] = x.hfq_close / x.close
        x["source"] = "历史补充"
    else:
        x = connection.execute("""
            select r.date,r.open,r.high,r.low,r.close,r.volume,r.amount,
                   f.backward_factor as adj_factor
            from v_daily r join calc_adjust_factor_daily f using(thscode,date)
            where r.thscode=? and r.date between ? and ? order by r.date
        """, [code, START, END]).df()
        x["source"] = "本地同花顺"
    if x.empty:
        return None
    x["date"] = pd.to_datetime(x.date)
    x = x.sort_values("date").drop_duplicates("date").set_index("date")
    for col in ["open", "high", "low", "close", "volume", "amount", "adj_factor"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    has_trade = x.get("tradestatus", pd.Series(1, index=x.index)).eq(1) & (x.volume > 0) & (x.amount > 0)
    if not np.isfinite(x.loc[has_trade, ["open", "high", "low", "close", "adj_factor"]]).all().all():
        raise ValueError(f"{code}有效成交行原价或因子缺失，不静默清洗")
    # 已核验的停牌行允许全空；保留行和状态，特征模块不把它算作有效K线。
    return x


def adjusted(raw):
    f = raw[["open", "high", "low", "close", "volume", "amount"]].copy()
    f.loc[:, ["open", "high", "low", "close"]] = f[["open", "high", "low", "close"]].mul(raw.adj_factor, axis=0)
    return f


def digest_frame(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def build(limit=None, only_codes=None):
    for name in ["原始分股", "结构特征"]:
        (DATA / name).mkdir(exist_ok=True)
    universe = pd.read_parquet(DATA / "需补状态名单.parquet").sort_values("thscode")
    if only_codes:
        universe = universe[universe.thscode.isin(only_codes)]
    if limit:
        universe = universe.iloc[:limit]
    cal = calendar()
    con = duckdb.connect(DB, read_only=True)
    con.execute("set threads=2")
    con.execute("set memory_limit='4GB'")
    feature_hash = hashlib.sha256((Path(__file__).parent / "结构特征.py").read_bytes()).hexdigest()
    version_path = DATA / "结构特征版本.json"
    version = json.loads(version_path.read_text()) if version_path.exists() else {}
    if version and version.get("代码校验") != feature_hash:
        raise RuntimeError("特征代码版本改变，请明确重建，不能混用新旧结果")
    if not version:
        version_path.write_text(json.dumps({"代码校验": feature_hash, "开始": START, "结束": END,
                                          "含未来已知日历至": str(cal.max().date())}, ensure_ascii=False, indent=2))
    log_path = DATA / "特征构建审计.csv"
    previous = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    recorded = {x["代码"] for x in previous if x["状态"] == "成功"}
    records, started = [], time.time()
    for i, row in enumerate(universe.itertuples(), 1):
        code = row.thscode
        raw_path = DATA / "原始分股" / f"{code}.parquet"
        feature_path = DATA / "结构特征" / f"{code}.parquet"
        if raw_path.exists() and feature_path.exists() and code in recorded:
            continue
        raw = load_raw(code, con, bool(row.missing))
        if raw is None:
            records.append({"代码": code, "状态": "缺真实行情", "来源": "历史缺失"})
            continue
        bars = adjusted(raw)
        features = build_features(bars, cal, END)
        if len(features) != int((cal <= END).sum()):
            raise AssertionError("日历对齐失败")
        raw.to_parquet(raw_path, compression="zstd")
        features.to_parquet(feature_path, compression="zstd")
        records.append({"代码": code, "状态": "成功", "来源": raw.source.iloc[0], "原始行数": len(raw),
                        "首日": str(raw.index.min().date()), "末日": str(raw.index.max().date()),
                        "特征行数": len(features), "特征列数": len(features.columns),
                        "日突破次数": int(features.d_breakout.sum()), "日回调次数": int(features.d_pullback_trigger.sum()),
                        "原价内容校验": digest_frame(raw.drop(columns="source"))})
        if i % 100 == 0:
            print(f"{i}/{len(universe)}，{time.time()-started:.1f}秒", flush=True)
            pd.DataFrame(previous + records).drop_duplicates("代码", keep="last").to_csv(log_path, index=False, encoding="utf-8-sig")
    con.close()
    pd.DataFrame(previous + records).drop_duplicates("代码", keep="last").to_csv(log_path, index=False, encoding="utf-8-sig")
    log = pd.read_csv(log_path)
    print(json.dumps({"状态统计": log["状态"].value_counts().to_dict(), "本批秒数": time.time()-started}, ensure_ascii=False), flush=True)


def verify_real():
    cal = calendar()
    # 开发、验证、留出边界及月季末；仅比较特征，不计算/查看任何收益。
    checks = ["2018-06-15", "2020-12-31", "2022-06-30", "2023-09-28", "2025-12-31", "2026-10-16"]
    codes = ["000001.SZ", "600000.SH", "300001.SZ", "688001.SH", "600069.SH"]
    results = []
    for code in codes:
        path = DATA / "原始分股" / f"{code}.parquet"
        if not path.exists():
            results.append({"代码": code, "截断日期": "", "一致": False, "说明": "未构建，不能计通过"})
            continue
        bars = adjusted(pd.read_parquet(path))
        full = build_features(bars, cal, END)
        for cutoff in checks:
            partial = build_features(bars.loc[:cutoff], cal, cutoff)
            assert_frame_equal(full.loc[:cutoff], partial, check_exact=True)
            results.append({"代码": code, "截断日期": cutoff, "一致": True, "说明": "108列严格逐值一致"})
    pd.DataFrame(results).to_csv(ROOT / "04_证据" / "真实行情截断因果核验.csv", index=False, encoding="utf-8-sig")
    print(f"真实截断核验：{sum(x['一致'] for x in results)}/{len(results)}", flush=True)
    if not all(x["一致"] for x in results):
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "verify"])
    p.add_argument("--limit", type=int)
    p.add_argument("--codes", nargs="+")
    a = p.parse_args()
    if a.mode == "build":
        build(a.limit, a.codes)
    else:
        verify_real()
