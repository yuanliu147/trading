"""只向本次研究目录补证券生命周期、官方口径字段及日历，不写行情数据库。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import socket
import time
import traceback

import baostock as bs
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_数据"
DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
START, END = "2016-10-13", "2026-10-22"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,tradestatus,pctChg,isST"
STATE_FIELDS = "date,preclose,tradestatus,isST"


def rows(rs):
    if rs.error_code != "0":
        raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
    result = []
    while rs.error_code == "0" and rs.next():
        result.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
    return pd.DataFrame(result, columns=rs.fields)


def login():
    if (ROOT / "04_证据" / "数据源访问限制.json").exists():
        raise RuntimeError("数据源已返回访问限制，停止继续请求；须由数据源管理员恢复访问，不自动重试")
    socket.setdefaulttimeout(20)
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"登录失败 {result.error_code}: {result.error_msg}")


def metadata():
    login()
    try:
        basic = rows(bs.query_stock_basic())
        cal = rows(bs.query_trade_dates(start_date="2016-01-01", end_date="2026-12-31"))
    finally:
        bs.logout()
    basic.to_parquet(DATA / "证券生命周期.parquet", index=False)
    basic.rename(columns={"code": "代码", "code_name": "名称", "ipoDate": "上市日", "outDate": "退市日",
                          "type": "类型", "status": "当前状态"}).to_csv(DATA / "证券生命周期.csv", index=False, encoding="utf-8-sig")
    cal.rename(columns={"calendar_date": "日期", "is_trading_day": "交易日标记"}).to_csv(
        DATA / "交易日历.csv", index=False, encoding="utf-8-sig")
    con = duckdb.connect(DB, read_only=True)
    have = set(con.execute("select distinct thscode from v_daily").df().thscode)
    actual = set(con.execute("select distinct date from v_daily order by date").df().date.dt.strftime("%Y-%m-%d"))
    con.close()
    b = basic[(basic.type == "1") & basic.code.str.startswith(("sh.", "sz."))].copy()
    b = b[(b.ipoDate <= END) & ((b.outDate == "") | (b.outDate >= START))]
    b["thscode"] = b.code.str.split(".").str[1] + "." + b.code.str.split(".").str[0].str.upper()
    b["missing"] = ~b.thscode.isin(have)
    b.to_parquet(DATA / "需补状态名单.parquet", index=False)
    b[b.missing].rename(columns={"code": "数据源代码", "code_name": "名称", "ipoDate": "上市日", "outDate": "退市日",
                                "type": "类型", "status": "当前状态", "thscode": "代码", "missing": "本地缺失"}).to_csv(
        DATA / "历史缺失股票名单.csv", index=False, encoding="utf-8-sig")
    expected = set(cal.loc[(cal.is_trading_day == "1") & (cal.calendar_date >= START) & (cal.calendar_date <= END), "calendar_date"])
    report = {"数据源": "baostock只读查询", "查询时间": pd.Timestamp.now().isoformat(),
              "证券资料全部行数": len(basic), "覆盖窗口沪深股票": len(b), "本地缺失股票": int(b.missing.sum()),
              "当前已退市且窗口有交集": int((b.outDate != "").sum()),
              "数据库有但基础表窗口未覆盖": sorted(have - set(b.thscode)),
              "交易日历缺日": sorted(expected - actual), "行情非日历交易日": sorted(actual - expected),
              "说明": "当前证券名称仅作展示，不回填历史ST或行业；生命周期日期用于覆盖审计，不把当前活跃状态作历史筛选。"}
    (ROOT / "04_证据" / "补充元数据审计.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, ensure_ascii=False), flush=True)


def convert_daily(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df.date)
    for c in df.columns:
        if c not in ("date", "code"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def fetch_one(job):
    code, thscode, missing = job
    state_file = DATA / "历史状态" / f"{thscode}.parquet"
    raw_file = DATA / "缺失股原始行情" / f"{thscode}.parquet"
    hfq_file = DATA / "缺失股后复权行情" / f"{thscode}.parquet"
    if state_file.exists() and (not missing or (raw_file.exists() and hfq_file.exists())):
        return {"代码": thscode, "状态": "已有缓存"}
    for attempt in range(2):
        try:
            raw = convert_daily(rows(bs.query_history_k_data_plus(
                code, FIELDS if missing else STATE_FIELDS, start_date=START, end_date=END, frequency="d", adjustflag="3")))
            if raw.empty:
                return {"代码": thscode, "状态": "空数据", "行数": 0}
            if missing:
                hfq = convert_daily(rows(bs.query_history_k_data_plus(
                    code, "date,open,high,low,close", start_date=START, end_date=END, frequency="d", adjustflag="1")))
                if hfq.empty or len(hfq) != len(raw):
                    raise RuntimeError("缺失股原始与后复权数据行数不一致")
                raw.to_parquet(raw_file, index=False)
                hfq.to_parquet(hfq_file, index=False)
            raw[["date", "preclose", "tradestatus", "isST"]].to_parquet(state_file, index=False)
            return {"代码": thscode, "状态": "成功", "行数": len(raw),
                    "起日": str(raw.date.min().date()), "末日": str(raw.date.max().date()), "新增行情": bool(missing)}
        except Exception as exc:
            if attempt == 0:
                try:
                    bs.logout()
                    login()
                except Exception:
                    pass
            else:
                return {"代码": thscode, "状态": "失败", "原因": str(exc)}


def download(workers=2, limit=None, only_missing=False):
    for name in ["历史状态", "缺失股原始行情", "缺失股后复权行情"]:
        (DATA / name).mkdir(exist_ok=True)
    b = pd.read_parquet(DATA / "需补状态名单.parquet").sort_values(["missing", "code"], ascending=[False, True])
    if only_missing:
        b = b[b.missing]
    jobs = [(r.code, r.thscode, bool(r.missing)) for r in b.itertuples()]
    jobs = [j for j in jobs if not ((DATA / "历史状态" / f"{j[1]}.parquet").exists()
            and (not j[2] or ((DATA / "缺失股原始行情" / f"{j[1]}.parquet").exists()
                              and (DATA / "缺失股后复权行情" / f"{j[1]}.parquet").exists())))]
    if limit is not None:
        jobs = jobs[:limit]
    log_path = DATA / "补充查询日志.csv"
    previous = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    start, done, logs = time.time(), 0, []
    print(f"开始补充 {len(jobs)} 只，工作进程 {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=login) as pool:
        tasks = {pool.submit(fetch_one, j): j for j in jobs}
        for f in as_completed(tasks):
            try:
                logs.append(f.result())
            except Exception:
                logs.append({"代码": tasks[f][1], "状态": "进程失败", "原因": traceback.format_exc()})
            done += 1
            if done % 25 == 0 or done == len(jobs):
                log = pd.DataFrame(logs)
                pd.DataFrame(previous + logs).drop_duplicates("代码", keep="last").to_csv(
                    log_path, index=False, encoding="utf-8-sig")
                print(f"{done}/{len(jobs)}，耗时{time.time()-start:.1f}秒，状态{log['状态'].value_counts().to_dict()}", flush=True)
    failures = [x for x in logs if x["状态"] in ("失败", "进程失败", "空数据")]
    print(f"结束，未完成{len(failures)}只；完整日志 {DATA / '补充查询日志.csv'}", flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["metadata", "download"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--only-missing", action="store_true")
    a = p.parse_args()
    if a.mode == "metadata":
        metadata()
    else:
        download(a.workers, a.limit, a.only_missing)
