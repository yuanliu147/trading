"""为账户分段构建同日现金与主要指数参考曲线；不含交易账户假设。"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from 统计检验 import account_metrics
from 交易数据 import ROOT

OLD = ROOT.parent / "2026-10-18_成交量融合_2010-2025" / "01_数据" / "daily_cache_2016_2026.pkl"
INDEXES = {"sh000001": "上证指数", "sh000300": "沪深300", "sh000905": "中证500"}


def fetch_recent(symbol, end):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
    rows, cursor = [], end
    for _ in range(5):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,{cursor - pd.Timedelta(days=20):%Y-%m-%d},{end:%Y-%m-%d},31,"
        result = requests.get(url, headers=headers, timeout=20).json()
        rows += result.get("data", {}).get(symbol, {}).get("day") or []
        first = rows[0][0] if rows else None
        if not rows or first >= "2026-10-13":
            break
        cursor = pd.Timestamp(first) - pd.Timedelta(days=1)
    if not rows:
        raise RuntimeError(f"未能补充指数{symbol}末端行情")
    frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    frame["date"] = pd.to_datetime(frame.date)
    for column in ["open", "close", "high", "low"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("date").sort_values("date")


def load_benchmarks():
    with OLD.open("rb") as handle:
        cache = pickle.load(handle)
    out = {}
    for symbol, name in INDEXES.items():
        frame = cache["indexes"][symbol].copy()
        if frame.date.max() < pd.Timestamp("2026-10-22"):
            recent = fetch_recent(symbol, pd.Timestamp("2026-10-22"))
            frame = pd.concat([frame, recent]).drop_duplicates("date", keep="last")
        frame = frame.sort_values("date")
        out[name] = frame.set_index("date")[["open", "close", "high", "low", "volume"]]
    return out


def main():
    benchmarks = load_benchmarks()
    dates = pd.read_csv(ROOT / "01_数据" / "交易日历.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(dates.loc[dates["交易日标记"].eq(1), "日期"]))
    dates = dates[dates <= "2026-10-22"]
    periods = {"开发": ("2018-01-01", "2020-12-31"), "验证": ("2021-01-01", "2022-12-31"),
               "历史留出": ("2023-01-01", "2025-12-31"), "二零二六": ("2026-01-01", "2026-10-22"),
               "全期": ("2018-01-01", "2026-10-22")}
    curves, metrics = [], []
    for period, (start, end) in periods.items():
        ds = dates[(dates >= start) & (dates <= end)]
        for name in ["现金", *INDEXES.values()]:
            if name == "现金":
                equity = pd.Series(20000.0, index=ds)
            else:
                series = benchmarks[name].close.reindex(ds).ffill()
                if series.isna().any():
                    raise RuntimeError(f"{name}缺少{start}—{end}行情")
                equity = series / series.iloc[0] * 20000
            metrics.append({"区段": period, "基准": name, **account_metrics(equity, 20000, start)})
            curves.append(pd.DataFrame({"日期": ds, "区段": period, "基准": name, "权益": equity.to_numpy()}))
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(ROOT / "03_结果" / "现金与指数基准指标.csv", index=False, encoding="utf-8-sig")
    pd.concat(curves).to_csv(ROOT / "03_结果" / "现金与指数基准曲线.csv", index=False, encoding="utf-8-sig")
    print("基准完成：现金与沪深300等按同日价格标记，不含申购费用；指数不是2万元真实账户成交。")


if __name__ == "__main__":
    main()
