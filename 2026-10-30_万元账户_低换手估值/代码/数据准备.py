#!/usr/bin/env python3
"""只读本地 2010—2020 数据；不含策略、未来收益、同步或母项目导入。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "数据"
CHECK = ROOT / "证据" / "数据准备检查.json"
FREEZE = ROOT / "00_研究冻结.json"
SOURCE = Path("/Users/admin/WorkSpace/trading/trading_趋势结构/01_数据/frozen")
DB = Path("/Users/admin/WorkSpace/baostock_data/data/baostock.db")
YEARS = tuple(range(2010, 2021))
START, END = "2010-01-01", "2020-12-31"
COLUMNS = [
    "symbol", "date", "board", "ipo_date", "td_since_list", "open", "high",
    "low", "close", "preclose", "volume", "amount", "is_suspended", "is_st",
    "pct_chg", "pe_ttm", "pb_mrq", "limit_up_px", "limit_down_px",
]
RAW = {
    "O": "open", "C": "close", "PC": "preclose", "LUP": "limit_up_px",
    "LDP": "limit_down_px", "H": "high", "L": "low", "V": "volume",
    "AMT": "amount", "PE": "pe_ttm", "PB": "pb_mrq", "PCT": "pct_chg",
}
DTYPES = {name: "float64" for name in ("O", "C", "PC", "LUP", "LDP")}
DTYPES.update({name: "float32" for name in (
    "H", "L", "V", "AMT", "PE", "PB", "PCT", "AC", "AO", "ATR20",
    "ADV20", "VAVG20",
)})
DTYPES.update({name: "bool" for name in (
    "ST", "SUSP", "RESUME", "U", "CONFLICT", "PRESENT",
)})
ACTION_MAP = {
    "symbol": "证券", "ex_date": "除权日", "dividend_per_share": "每股分红",
    "per_share_bonus": "送转比例", "rights_ratio": "配股比例", "rights_price": "配股价",
}
CAL_SQL = (
    "SELECT calendar_date,is_trading_day FROM trade_dates "
    "WHERE calendar_date BETWEEN ? AND ? ORDER BY calendar_date"
)
INDEX_SQL = (
    "SELECT date,close FROM index_kline_d "
    "WHERE code=? AND date BETWEEN ? AND ? ORDER BY date"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_hash(value):
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_stat(path):
    stat = path.stat()
    return {"字节数": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_database():
    # 不哈希整个 DB：只读取允许期两个 SELECT 的逻辑内容。
    connection = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=1")
        calendar = connection.execute(CAL_SQL, (START, END)).fetchall()
        index = connection.execute(INDEX_SQL, ("sh.000300", START, END)).fetchall()
    finally:
        connection.close()
    payload = {"trade_dates": calendar, "index_kline_d": index}
    return calendar, index, result_hash(payload)


def enumerate_daily():
    files = []
    for year in YEARS:
        partition = SOURCE / "daily" / f"year={year}"
        found = sorted(partition.glob("data_*.parquet"))
        require(bool(found), f"缺少允许年份分区文件：{partition}")
        for path in found:
            require(path.resolve().parent == partition.resolve(), f"文件越出分区：{path}")
            schema = pq.ParquetFile(path).schema_arrow
            require(set(COLUMNS).issubset(schema.names), f"日线字段不符：{path}")
            files.append((year, path))
    return files


def daily_read(path, columns, audit, year):
    require(year in YEARS and path.parent.name == f"year={year}", "禁止读取边界外分区")
    require(set(columns).issubset(COLUMNS), "禁止读取白名单外日线字段")
    audit["日线读取"].append({"路径": str(path), "年份": year, "列": list(columns)})
    return pd.read_parquet(path, columns=columns)


def rolling_positive(values):
    values = np.asarray(values, dtype=np.float64)
    positive = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    return pd.Series(positive).rolling(20, min_periods=15).mean().to_numpy()


def continuous_features(close, opened, high, low, pct, present, suspended):
    """逐证券沿完整日历走一次；未知缺口不填零、不重新归一化。"""
    count = len(close)
    ac = np.full(count, np.nan, dtype=np.float64)
    started, broken, level = False, False, np.nan
    first_break = None
    for t in range(count):
        known_return = np.isfinite(pct[t]) and pct[t] > -1
        valid_close = np.isfinite(close[t]) and close[t] > 0
        if not started:
            if present[t] and valid_close and known_return:
                level, started = 100.0, True
                ac[t] = level
            continue
        if not present[t] or not known_return:
            if not broken:
                first_break = t
            broken = True
        if not broken:
            level *= 1.0 + float(pct[t])
            ac[t] = level
    scale = np.divide(ac, close, out=np.full(count, np.nan), where=np.isfinite(close) & (close > 0))
    ao = scale * opened
    ah, al = scale * high, scale * low
    # 只有有源行且明确停牌，调整空间缺失价才允许保持；原始数组不补。
    known_susp = present & suspended & np.isfinite(ac)
    ah = np.where(known_susp & (~np.isfinite(ah) | (ah <= 0)), ac, ah)
    al = np.where(known_susp & (~np.isfinite(al) | (al <= 0)), ac, al)
    previous = np.r_[np.nan, ac[:-1]]
    tr = np.maximum.reduce([ah - al, np.abs(ah - previous), np.abs(al - previous)])
    valid = np.isfinite(previous) & (previous > 0) & (ah >= al) & (al > 0)
    tr[~valid] = np.nan
    atr = pd.Series(tr).rolling(20, min_periods=20).mean().to_numpy()
    return ac, ao, atr, first_break


def self_test():
    """内存内合成用例，不创建测试文件或越界真实数据。"""
    n = 45
    pct = np.full(n, 0.01)
    close = 10 * (1.01 ** np.arange(n))
    close[10:] /= 2  # 原始价除权，但官方收益不变；不能再乘一次原始价收益。
    present = np.ones(n, dtype=bool)
    susp = np.zeros(n, dtype=bool)
    ac, ao, atr, broken = continuous_features(close, close * .99, close * 1.02, close * .98, pct, present, susp)
    np.testing.assert_allclose(ac, 100 * (1.01 ** np.arange(n)), rtol=1e-13)
    np.testing.assert_allclose(ao, ac * .99)
    require(broken is None and np.isnan(atr[:20]).all() and np.isfinite(atr[20:]).all(), "ATR 预热测试失败")
    manual_tr = np.maximum.reduce([ac[1:] * .04, np.abs(ac[1:] * 1.02 - ac[:-1]), np.abs(ac[1:] * .98 - ac[:-1])])
    np.testing.assert_allclose(atr[20], manual_tr[:20].mean())
    present[22] = False
    ac_gap, _, atr_gap, broken = continuous_features(close, close, close, close, pct, present, susp)
    require(broken == 22 and np.isnan(ac_gap[22:]).all() and np.isnan(atr_gap[22:]).all(), "未知缺行不得压缩或接续")
    present[22] = True
    susp[22] = True
    pct[22] = 0
    close[22] = np.nan
    ac_susp, _, _, _ = continuous_features(close, close, close, close, pct, present, susp)
    require(ac_susp[22] == ac_susp[21], "明确停牌连续价保持失败")
    values = np.r_[np.ones(15), np.zeros(5), np.ones(5)]
    means = rolling_positive(values)
    require(np.isnan(means[:14]).all() and means[19] == 1 and means[20] == 1, "有效正成交窗口测试失败")
    values[0] = np.nan
    require(np.isnan(rolling_positive(values)[19]), "不足15个有效日必须缺失")
    return {"通过": True, "用例": ["收益仅累乘一次", "AO同尺度", "ATR前收盘及20日预热", "缺行不压缩不接续", "明确停牌保持价格", "成交窗口正值及至少15日"]}


def prepare_actions(audit):
    path = SOURCE / "corp_actions.parquet"
    require(set(ACTION_MAP).issubset(pq.ParquetFile(path).schema_arrow.names), "公司行动字段不符")
    frame = pd.read_parquet(path, columns=list(ACTION_MAP), filters=[
        ("ex_date", ">=", date(2010, 1, 1)), ("ex_date", "<=", date(2020, 12, 31)),
    ])
    require(not frame[list(ACTION_MAP)].isna().any().any(), "公司行动字段缺失，不能猜测为0")
    before = len(frame)
    frame = frame.drop_duplicates(list(ACTION_MAP))
    conflict = frame.duplicated(["symbol", "ex_date"], keep=False)
    records = frame.loc[conflict].copy()
    records["ex_date"] = records["ex_date"].astype(str)
    audit["公司行动"] = {
        "允许期源行数": before, "相同字段重复去重数": before - len(frame),
        "冲突行数": int(conflict.sum()), "冲突明细": records.to_dict("records"),
        "读取列": list(ACTION_MAP), "过滤起止": [START, END],
        "范围": "保留允许期全部证券行动，不依未来退出过滤；使用时按证券关联",
    }
    require(not conflict.any(), "公司行动同证券同日字段冲突，停止而非静默覆盖")
    frame = frame.sort_values(["symbol", "ex_date"]).reset_index(drop=True)
    frame["ex_date"] = pd.to_datetime(frame["ex_date"]).dt.strftime("%Y-%m-%d")
    require(frame["ex_date"].between(START, END).all(), "公司行动日期越界")
    return frame.rename(columns=ACTION_MAP)


def add_examples(target, rows, limit=30):
    target.extend(rows[:max(0, limit - len(target))])


def main():
    audit = {"状态": "运行中", "开始UTC": datetime.now(timezone.utc).isoformat(), "日线读取": []}
    try:
        require(all(path.is_dir() for path in (DATA, ROOT / "代码", ROOT / "证据")), "授权目录必须已存在")
        config = json.loads(FREEZE.read_text(encoding="utf-8"))
        require(config["数据"]["日线允许列"] == COLUMNS, "冻结设计白名单与实现不符，停止")
        require(config["边界"]["允许数据起"] == START and config["边界"]["允许数据止"] == END, "冻结日期与实现不符")
        require(config["用户选择"]["初始资金"] == 10000, "冻结账户金额与任务不符")
        audit["合成测试"] = self_test()
        files = enumerate_daily()
        provenance = []
        for path in [FREEZE, *(path for _, path in files), SOURCE / "corp_actions.parquet"]:
            provenance.append({"路径": str(path), "SHA256": sha256_file(path), "哈希范围": "完整文件字节", **source_stat(path)})
        db_stat = source_stat(DB)
        calendar_rows, index_rows, db_hash = read_database()
        all_dates = [r[0] for r in calendar_rows]
        require(all_dates == pd.date_range(START, END).strftime("%Y-%m-%d").tolist(), "研究期自然日日历不完整或重复")
        require(all(r[1] in (0, 1) for r in calendar_rows), "日历交易日标记不符")
        dates = [r[0] for r in calendar_rows if r[1] == 1]
        require([r[0] for r in index_rows] == dates, "沪深300日期未与完整交易日历严格对齐")
        index_values = np.array([r[1] for r in index_rows], dtype=np.float64)
        require((np.isfinite(index_values) & (index_values > 0)).all(), "指数收盘缺失或非正")
        provenance.append({
            "路径": str(DB), "SHA256": db_hash, "哈希范围": "仅允许期有序SELECT结果的UTF-8紧凑JSON，不是整库文件哈希",
            "序列化": {"顶层键顺序": ["trade_dates", "index_kline_d"], "ensure_ascii": False, "separators": [",", ":"]},
            "SQL": [CAL_SQL, INDEX_SQL], "参数": [[START, END], ["sh.000300", START, END]], **db_stat,
        })
        actions = prepare_actions(audit)
        symbols = set()
        for year, path in files:
            scan = daily_read(path, ["symbol", "board"], audit, year)
            selected = scan.loc[scan["board"].eq("主板"), "symbol"]
            require(selected.notna().all(), f"证券代码缺失：{path}")
            symbols.update(selected.tolist())
            del scan, selected
        symbols = sorted(symbols)
        require(bool(symbols), "主板证券为空")
        symbol_index = pd.Index(symbols)
        date_index = pd.DatetimeIndex(dates)
        shape = (len(symbols), len(dates))
        print(f"数组 shape={shape}，逐年读取并写入内存映射", flush=True)
        arrays = {}
        for name, dtype in DTYPES.items():
            array = np.lib.format.open_memmap(DATA / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            array[:] = (name in ("ST", "SUSP")) if dtype == "bool" else np.nan
            array.flush()
            arrays[name] = array
        audit["源年度"] = []
        audit["收益缺失"] = {"明确停牌填零": 0, "无法确定": 0, "无效小于等于负100百分比": 0, "无法确定示例": []}
        audit["源标记缺失"] = {"ST": 0, "SUSP": 0}
        audit["价格收益源冲突"] = {"阈值小数": 0.00005, "总数": 0, "明细": []}
        sampled = []
        rng = np.random.default_rng(20261030)
        for year, path in files:
            frame = daily_read(path, COLUMNS, audit, year)
            source_dates = pd.to_datetime(frame["date"])
            require(source_dates.notna().all() and source_dates.dt.year.eq(year).all(), f"分区含越界或缺失日期：{path}")
            require(source_dates.between(START, END).all(), "日线日期越界")
            frame = frame.loc[frame["board"].eq("主板")].copy()
            frame["date"] = pd.to_datetime(frame["date"])
            require(not frame.duplicated(["symbol", "date"]).any(), f"日线同证券日重复：{path}")
            si = symbol_index.get_indexer(frame["symbol"])
            ti = date_index.get_indexer(frame["date"])
            require((si >= 0).all() and (ti >= 0).all(), "证券或日线日期不在索引内")
            require(not arrays["PRESENT"][si, ti].any(), f"文件间同证券日重复：{path}")
            for col in ("is_st", "is_suspended"):
                require(frame[col].dropna().isin([0, 1, False, True]).all(), f"非法源标记：{col}")
            st_known = frame["is_st"].notna().to_numpy()
            susp_known = frame["is_suspended"].notna().to_numpy()
            st = frame["is_st"].fillna(1).astype(bool).to_numpy()
            susp = frame["is_suspended"].fillna(True).astype(bool).to_numpy()
            audit["源标记缺失"]["ST"] += int((~st_known).sum())
            audit["源标记缺失"]["SUSP"] += int((~susp_known).sum())
            for name, col in RAW.items():
                values = frame[col].to_numpy(dtype=np.float64, na_value=np.nan)
                arrays[name][si, ti] = values / 100.0 if name == "PCT" else values
            arrays["PRESENT"][si, ti] = True
            arrays["ST"][si, ti] = st
            arrays["SUSP"][si, ti] = susp
            official = frame["pct_chg"].to_numpy(dtype=np.float64) / 100.0
            missing = ~np.isfinite(official)
            zero = missing & susp_known & susp
            unknown = missing & ~zero
            invalid = np.isfinite(official) & (official <= -1)
            arrays["PCT"][si[zero], ti[zero]] = 0.0
            arrays["PCT"][si[invalid | unknown], ti[invalid | unknown]] = np.nan
            audit["收益缺失"]["明确停牌填零"] += int(zero.sum())
            audit["收益缺失"]["无法确定"] += int(unknown.sum())
            audit["收益缺失"]["无效小于等于负100百分比"] += int(invalid.sum())
            missing_records = frame.loc[unknown | invalid, ["symbol", "date"]].copy()
            missing_records["date"] = missing_records["date"].dt.strftime("%Y-%m-%d")
            add_examples(audit["收益缺失"]["无法确定示例"], missing_records.to_dict("records"))
            c = frame["close"].to_numpy(dtype=np.float64)
            pc = frame["preclose"].to_numpy(dtype=np.float64)
            valid = np.isfinite(c) & (c > 0) & np.isfinite(pc) & (pc > 0) & np.isfinite(official)
            raw_return = np.divide(c, pc, out=np.full(len(c), np.nan), where=np.isfinite(pc) & (pc > 0)) - 1
            conflict = valid & (np.abs(raw_return - official) > 0.00005)
            arrays["CONFLICT"][si, ti] = conflict
            for row in np.flatnonzero(conflict):
                audit["价格收益源冲突"]["明细"].append({
                    "证券": symbols[si[row]], "日期": dates[ti[row]], "close": float(c[row]),
                    "preclose": float(pc[row]), "官方小数收益": float(official[row]),
                    "差异bp": float(abs(raw_return[row] - official[row]) * 10000),
                })
            audit["价格收益源冲突"]["总数"] += int(conflict.sum())
            ipo = pd.to_datetime(frame["ipo_date"])
            age = (frame["date"] - ipo).dt.days.to_numpy(dtype=np.float64)
            td = frame["td_since_list"].to_numpy(dtype=np.float64)
            pe = arrays["PE"][si, ti]
            pb = arrays["PB"][si, ti]
            # 这里只使用当行上市年龄、标记及估值；随后补上窗口和复牌条件。
            base_u = (
                st_known & susp_known & ~st & ~susp & ~conflict & np.isfinite(c) & (c >= 2)
                & np.isfinite(age) & (age >= 250) & np.isfinite(td) & (td >= 120)
                & np.isfinite(pe) & (pe > 0) & np.isfinite(pb) & (pb > 0)
            )
            arrays["U"][si, ti] = base_u
            positions = rng.choice(len(frame), size=min(500, len(frame)), replace=False)
            for row in positions:
                if np.isfinite(official[row]):
                    sampled.append((int(si[row]), int(ti[row]), float(official[row])))
            audit["源年度"].append({
                "年份": year, "路径": str(path), "主板行数": len(frame), "证券数": int(frame["symbol"].nunique()),
                "起": frame["date"].min().strftime("%Y-%m-%d"), "止": frame["date"].max().strftime("%Y-%m-%d"),
                "原始缺失": {col: int(value) for col, value in frame.isna().sum().items()},
                "缺失收益停牌填零": int(zero.sum()), "价格收益冲突": int(conflict.sum()),
            })
            for array in arrays.values():
                array.flush()
            print(f"{year}：主板 {len(frame):,} 行完成", flush=True)
            del frame, source_dates, si, ti
        audit["缺行与恢复"] = {
            "首个源行之前格数": 0, "首个源行之后缺行格数": 0,
            "缺行后重新出现源行次数": 0, "缺行后恢复交易次数": 0,
            "缺行恢复交易示例": [], "已知昨日停牌今日交易次数": 0,
        }
        audit["连续价格"] = {
            "初始尺度": 100.0, "未知缺口后不接续证券数": 0,
            "断链后仍有源行证券数": 0, "断链后有源行格数": 0, "断链示例": [],
            "说明": "首个有效收盘且收益已知的观测归一为100，首日收益不重复应用；缺行不能证明停牌，遇未知收益缺口后不接续或重新归一化。",
        }
        for s, symbol in enumerate(symbols):
            present = np.asarray(arrays["PRESENT"][s])
            susp = np.asarray(arrays["SUSP"][s])
            close = np.asarray(arrays["C"][s])
            volume = np.asarray(arrays["V"][s])
            traded = present & ~susp & np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0)
            resume = np.zeros(len(dates), dtype=bool)
            resume[1:] = present[:-1] & susp[:-1] & traded[1:]
            arrays["RESUME"][s] = resume
            first = int(np.flatnonzero(present)[0])
            seen_before = np.r_[False, np.logical_or.accumulate(present)[:-1]]
            missing_previous = np.r_[False, ~present[:-1]] & seen_before
            recovery = missing_previous & traded
            audit["缺行与恢复"]["首个源行之前格数"] += first
            audit["缺行与恢复"]["首个源行之后缺行格数"] += int((~present[first:]).sum())
            audit["缺行与恢复"]["缺行后重新出现源行次数"] += int((missing_previous & present).sum())
            audit["缺行与恢复"]["缺行后恢复交易次数"] += int(recovery.sum())
            audit["缺行与恢复"]["已知昨日停牌今日交易次数"] += int(resume.sum())
            add_examples(audit["缺行与恢复"]["缺行恢复交易示例"], [{"证券": symbol, "日期": dates[t]} for t in np.flatnonzero(recovery)])
            ac, ao, atr, broken = continuous_features(
                close, arrays["O"][s], arrays["H"][s], arrays["L"][s], arrays["PCT"][s], present, susp,
            )
            arrays["AC"][s], arrays["AO"][s], arrays["ATR20"][s] = ac, ao, atr
            if broken is not None:
                after = int(present[broken:].sum())
                audit["连续价格"]["未知缺口后不接续证券数"] += 1
                audit["连续价格"]["断链后仍有源行证券数"] += int(after > 0)
                audit["连续价格"]["断链后有源行格数"] += after
                add_examples(audit["连续价格"]["断链示例"], [{"证券": symbol, "日期": dates[broken], "其后有源行数": after}])
            arrays["ADV20"][s] = rolling_positive(arrays["AMT"][s])
            arrays["VAVG20"][s] = rolling_positive(volume)
            arrays["U"][s] &= ~resume & np.isfinite(arrays["ADV20"][s]) & (arrays["ADV20"][s] >= 30000000)
        for array in arrays.values():
            array.flush()
        audit["独立源复核"] = source_recheck(arrays, files, audit, symbol_index, date_index)
        audit["滚动窗口抽查"] = window_checks(arrays, sampled)
        audit["分年估值与宇宙"] = yearly_checks(arrays, dates)
        audit["数组检查"] = array_checks(arrays, shape)
        audit["AC收益抽查"] = sample_checks(arrays, sampled, symbols, dates)
        audit["日期检查"] = {
            "研究允许起止": [START, END], "交易日实际起止": [dates[0], dates[-1]],
            "交易日数": len(dates), "自然日日历完整": True, "指数日期完全对齐": True,
            "日线文件实际年份": sorted({row["年份"] for row in audit["源年度"]}),
            "仅允许年份": all(row["年份"] in YEARS for row in audit["日线读取"]),
        }
        audit["源未改写"] = {
            "完整文件哈希复核": all(sha256_file(Path(item["路径"])) == item["SHA256"] for item in provenance[:-1]),
            "DB文件属性不变": source_stat(DB) == db_stat, "DB允许期查询指纹不变": read_database()[2] == db_hash,
        }
        require(all(audit["源未改写"].values()), "运行期间源内容或数据库属性发生变化")
        pd.DataFrame({"日期": dates, "沪深300收盘": index_values}).to_csv(DATA / "日历指数.csv", index=False)
        actions.to_csv(DATA / "公司行动.csv", index=False)
        gaps = [
            "历史估值首次公布及重述时点不可识别，行情日PE/PB不是严格财务PIT。",
            "公司行动缺实际派息到账日、股份可售上市日和送股/转增税基区别；不补造这些字段。",
            "不引入2009预热；2010初期或新股窗口不足时保留NaN；20/60交易日滞后估值留给后续策略。",
            "缺行不能证明停牌；ST/SUSP对缺行保守置True且PRESENT=False，原始数值仍为NaN；缺行恢复仅审计，不冒称已知复牌。",
            "DB SHA256是允许期查询结果逻辑指纹，不是整库物理哈希；未读取边界外行情。",
            "不按未来退市日或名称过滤；证券轴允许包含期内后上市证券，但准入完全按当日行。",
            f"实际发现{audit['价格收益源冲突']['总数']}条价格/收益冲突，已保留原值并排除当日U。",
            f"实际发现{audit['连续价格']['断链后仍有源行证券数']}只证券未知缺行后又出现，共{audit['连续价格']['断链后有源行格数']}个有源行的AC/AO/ATR保持未知，不借此额外过滤U。",
        ]
        meta = {
            "证券": symbols, "日期": dates, "证券数": shape[0], "交易日数": shape[1],
            "源文件": provenance, "源年份": list(YEARS), "研究允许起止": [START, END],
            "数组": {name: {"dtype": dtype, "shape": list(shape)} for name, dtype in DTYPES.items()},
            "模块": {"路径": str(Path(__file__).resolve()), "SHA256": sha256_file(Path(__file__).resolve())},
            "口径": {
                "PCT": "官方pct_chg除以100的小数收益；仅明确停牌的缺收益置0，原始缺失数另审计。",
                "AC": audit["连续价格"]["说明"],
                "AO": "AC * 原始O / 原始C；调整H/L仅在内存用于ATR，不输出额外数组。",
                "ATR20": "完整交易日网格上调整空间TR的20日简单均值，min_periods=20；首观测无前日AC故TR为NaN。",
                "ADV20/VAVG20": "截至当日20个完整交易日内有限且正的金额/量均值，至少15个有效日。",
                "RESUME": "有源行的昨日停牌且今日明确非停牌、有有效正收盘及正成交量；源缺行恢复另审计。",
                "U": "PRESENT且已知非ST/停牌/复牌/冲突，C>=2，ADV20>=3000万，IPO自然日>=250，td_since_list>=120，PE/PB有限正。",
                "缺行原值": "数值原值NaN，PRESENT=False；ST/SUSP=True仅为保守哨兵，不代表已知风险警示或停牌。",
                "精度": "价/收益冲突使用源float64和0.5bp阈值；输出PCT/连续特征等按契约float32。",
            },
            "非严格财务PIT": True, "未闭合数据缺口": gaps,
        }
        write_json(DATA / "数据元信息.json", meta)
        audit["未闭合数据缺口"] = gaps
        audit["输出文件"] = [str(DATA / f"{name}.npy") for name in DTYPES] + [
            str(DATA / "日历指数.csv"), str(DATA / "公司行动.csv"), str(DATA / "数据元信息.json"), str(CHECK), str(Path(__file__).resolve()),
        ]
        audit["输出文件SHA256"] = {str(DATA / name): sha256_file(DATA / name) for name in [
            *(f"{key}.npy" for key in DTYPES), "日历指数.csv", "公司行动.csv", "数据元信息.json",
        ]}
        loaded = load()
        require(all(isinstance(loaded[name], np.memmap) and not loaded[name].flags.writeable for name in DTYPES), "load必须返回只读mmap")
        require(np.array_equal(loaded["index"], index_values), "load指数不一致")
        require(len(loaded["actions"]) == len(actions), "load公司行动不一致")
        audit["load检查"] = {"通过": True, "只读mmap数组数": len(DTYPES), "公司行动行数": len(actions)}
        audit["状态"] = "通过"
        audit["结束UTC"] = datetime.now(timezone.utc).isoformat()
        write_json(CHECK, audit)
        print(json.dumps({"状态": audit["状态"], "shape": shape, "数组数": len(DTYPES), "检查": str(CHECK)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        audit["状态"] = "失败"
        audit["错误"] = f"{type(exc).__name__}: {exc}"
        write_json(CHECK, audit)
        raise


def source_recheck(arrays, files, audit, symbols, dates):
    """重新逐文件核对全部原值和U，含未作为数组落盘的当日上市年龄。"""
    counts = {"源行数": 0, "原值不一致": 0, "PRESENT不一致": 0, "标记不一致": 0, "CONFLICT不一致": 0, "U不一致": 0, "U上市自然日不足": 0, "U上市交易日不足": 0}
    for year, path in files:
        frame = daily_read(path, COLUMNS, audit, year)
        frame = frame.loc[frame["board"].eq("主板")]
        s = symbols.get_indexer(frame["symbol"])
        t = dates.get_indexer(pd.to_datetime(frame["date"]))
        known = frame["is_st"].notna().to_numpy() & frame["is_suspended"].notna().to_numpy()
        st = frame["is_st"].fillna(1).astype(bool).to_numpy()
        susp = frame["is_suspended"].fillna(True).astype(bool).to_numpy()
        for name, col in RAW.items():
            expected = frame[col].to_numpy(dtype=np.float64, na_value=np.nan).copy()
            if name == "PCT":
                expected /= 100
                expected[~np.isfinite(expected) & frame["is_suspended"].eq(True).fillna(False).to_numpy(dtype=bool)] = 0
                expected[~np.isfinite(expected) | (expected <= -1)] = np.nan
            expected = expected.astype(DTYPES[name])
            actual = arrays[name][s, t]
            counts["原值不一致"] += int((~((expected == actual) | (np.isnan(expected) & np.isnan(actual)))).sum())
        counts["源行数"] += len(frame)
        counts["PRESENT不一致"] += int((~arrays["PRESENT"][s, t]).sum())
        counts["标记不一致"] += int((arrays["ST"][s, t] != st).sum() + (arrays["SUSP"][s, t] != susp).sum())
        close = frame["close"].to_numpy(dtype=np.float64)
        pc = frame["preclose"].to_numpy(dtype=np.float64)
        official = frame["pct_chg"].to_numpy(dtype=np.float64) / 100
        with np.errstate(divide="ignore", invalid="ignore"):
            conflict = np.isfinite(close) & (close > 0) & np.isfinite(pc) & (pc > 0) & np.isfinite(official) & (np.abs(close / pc - 1 - official) > .00005)
        counts["CONFLICT不一致"] += int((conflict != arrays["CONFLICT"][s, t]).sum())
        age = (pd.to_datetime(frame["date"]) - pd.to_datetime(frame["ipo_date"])).dt.days.to_numpy(dtype=np.float64)
        td = frame["td_since_list"].to_numpy(dtype=np.float64)
        pe, pb = frame["pe_ttm"].to_numpy(dtype=np.float64), frame["pb_mrq"].to_numpy(dtype=np.float64)
        adv = arrays["ADV20"][s, t]
        expected_u = known & ~st & ~susp & ~arrays["RESUME"][s, t] & ~conflict & np.isfinite(close) & (close >= 2) & np.isfinite(adv) & (adv >= 30000000) & (age >= 250) & (td >= 120) & np.isfinite(pe) & (pe > 0) & np.isfinite(pb) & (pb > 0)
        u = arrays["U"][s, t]
        counts["U不一致"] += int((expected_u != u).sum())
        counts["U上市自然日不足"] += int((u & ~(age >= 250)).sum())
        counts["U上市交易日不足"] += int((u & ~(td >= 120)).sum())
    require(counts["源行数"] == int(arrays["PRESENT"].sum()), "PRESENT数量与源行不符")
    require(not any(value for key, value in counts.items() if key != "源行数"), f"独立源复核不通过：{counts}")
    return {**counts, "通过": True}


def window_checks(arrays, sampled):
    """不用rolling重算随机20日窗口，并核对全部复牌标记。"""
    counts = {"成交窗口核对数": 0, "ATR窗口核对数": 0, "RESUME不一致": 0}
    for s, t, _ in sampled:
        start = max(0, t - 19)
        for raw, feature in (("AMT", "ADV20"), ("V", "VAVG20")):
            values = np.asarray(arrays[raw][s, start:t + 1], dtype=np.float64)
            values = values[np.isfinite(values) & (values > 0)]
            expected = np.float32(values.mean()) if len(values) >= 15 else np.nan
            actual = arrays[feature][s, t]
            require((np.isnan(expected) and np.isnan(actual)) or np.isclose(expected, actual, rtol=2e-7, atol=0), f"成交窗口错误：{raw}/{s}/{t}")
            counts["成交窗口核对数"] += 1
        if start == 0:
            require(np.isnan(arrays["ATR20"][s, t]), "ATR缺少前日AC却提前有效")
            continue
        ac = np.asarray(arrays["AC"][s, start:t + 1], dtype=np.float64)
        previous = np.asarray(arrays["AC"][s, start - 1:t], dtype=np.float64)
        close = arrays["C"][s, start:t + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ah = ac * arrays["H"][s, start:t + 1] / close
            al = ac * arrays["L"][s, start:t + 1] / close
        susp = arrays["PRESENT"][s, start:t + 1] & arrays["SUSP"][s, start:t + 1] & np.isfinite(ac)
        ah[susp & (~np.isfinite(ah) | (ah <= 0))] = ac[susp & (~np.isfinite(ah) | (ah <= 0))]
        al[susp & (~np.isfinite(al) | (al <= 0))] = ac[susp & (~np.isfinite(al) | (al <= 0))]
        tr = np.maximum.reduce([ah - al, abs(ah - previous), abs(al - previous)])
        tr[~(np.isfinite(previous) & (previous > 0) & (ah >= al) & (al > 0))] = np.nan
        expected = tr.mean()
        actual = arrays["ATR20"][s, t]
        require((np.isnan(expected) and np.isnan(actual)) or np.isclose(expected, actual, rtol=5e-5, atol=1e-6), f"ATR窗口错误：{s}/{t}")
        counts["ATR窗口核对数"] += 1
    for s in range(arrays["C"].shape[0]):
        p, susp, c, v = (arrays[key][s] for key in ("PRESENT", "SUSP", "C", "V"))
        expected = np.r_[False, p[:-1] & susp[:-1]] & p & ~susp & np.isfinite(c) & (c > 0) & np.isfinite(v) & (v > 0)
        counts["RESUME不一致"] += int((expected != arrays["RESUME"][s]).sum())
    require(counts["RESUME不一致"] == 0, "RESUME复核失败")
    return {**counts, "ATR容差": "使用落盘float32 AC复核，rtol=5e-5,atol=1e-6", "通过": True}


def yearly_checks(arrays, dates):
    result = []
    years = np.array([int(value[:4]) for value in dates])
    for year in YEARS:
        positions = np.flatnonzero(years == year)
        sl = slice(int(positions[0]), int(positions[-1]) + 1)
        p = arrays["PRESENT"][:, sl]
        u = arrays["U"][:, sl]
        pe, pb = arrays["PE"][:, sl], arrays["PB"][:, sl]
        daily = np.asarray(u.sum(axis=0), dtype=int)
        result.append({
            "年份": year, "交易日数": len(positions), "PRESENT格数": int(p.sum()),
            "PE缺失或非有限": int((p & ~np.isfinite(pe)).sum()), "PB缺失或非有限": int((p & ~np.isfinite(pb)).sum()),
            "PE有限非正": int((p & np.isfinite(pe) & (pe <= 0)).sum()), "PB有限非正": int((p & np.isfinite(pb) & (pb <= 0)).sum()),
            "任一估值无效": int((p & (~np.isfinite(pe) | (pe <= 0) | ~np.isfinite(pb) | (pb <= 0))).sum()),
            "U内AC缺失": int((u & ~np.isfinite(arrays["AC"][:, sl])).sum()),
            "U内ATR20缺失": int((u & ~np.isfinite(arrays["ATR20"][:, sl])).sum()),
            "宇宙日均": float(daily.mean()), "宇宙中位数": float(np.median(daily)),
            "宇宙最小": int(daily.min()), "宇宙最大": int(daily.max()), "宇宙期末": int(daily[-1]),
            "宇宙每日": daily.tolist(), "CONFLICT": int(arrays["CONFLICT"][:, sl].sum()),
        })
    return result


def array_checks(arrays, shape):
    stats = {}
    violations = {key: 0 for key in ("无源行", "ST", "停牌", "复牌", "冲突", "无效估值", "无效价格", "ADV不足", "缺行原始数值非NaN")}
    for name, array in arrays.items():
        require(array.shape == shape and str(array.dtype) == DTYPES[name], f"数组shape/dtype错误：{name}")
        stats[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.dtype.kind == "b":
            stats[name]["True数"] = int(array.sum())
        else:
            finite = 0
            inf = 0
            for row in array:
                finite += int(np.isfinite(row).sum())
                inf += int(np.isinf(row).sum())
            stats[name].update({"有限数": finite, "NaN数": int(array.size - finite - inf), "Inf数": inf})
            require(inf == 0, f"数组存在Inf：{name}")
    for s in range(shape[0]):
        u, present = arrays["U"][s], arrays["PRESENT"][s]
        violations["无源行"] += int((u & ~present).sum())
        for title, name in (("ST", "ST"), ("停牌", "SUSP"), ("复牌", "RESUME"), ("冲突", "CONFLICT")):
            violations[title] += int((u & arrays[name][s]).sum())
        pe, pb, c, adv = (arrays[key][s] for key in ("PE", "PB", "C", "ADV20"))
        violations["无效估值"] += int((u & (~np.isfinite(pe) | (pe <= 0) | ~np.isfinite(pb) | (pb <= 0))).sum())
        violations["无效价格"] += int((u & (~np.isfinite(c) | (c < 2))).sum())
        violations["ADV不足"] += int((u & (~np.isfinite(adv) | (adv < 30000000))).sum())
        for name in RAW:
            violations["缺行原始数值非NaN"] += int((~present & ~np.isnan(arrays[name][s])).sum())
    require(not any(violations.values()), f"数组准入检查不通过：{violations}")
    return {"数组": stats, "U及缺行违规计数": violations, "通过": True}


def sample_checks(arrays, sampled, symbols, dates):
    errors, examples = [], []
    skipped = 0
    for s, t, official in sampled:
        if t == 0 or not np.isfinite(arrays["AC"][s, t - 1:t + 1]).all():
            skipped += 1
            continue
        observed = float(arrays["AC"][s, t]) / float(arrays["AC"][s, t - 1]) - 1
        error = abs(observed - official)
        errors.append(error)
        add_examples(examples, [{"证券": symbols[s], "日期": dates[t], "官方小数收益": official, "AC相邻收益": observed, "绝对误差": error}], 20)
    require(bool(errors) and max(errors) <= 5e-7, "AC相邻收益与官方源抽查不一致")
    return {"随机种子": 20261030, "源随机观测数": len(sampled), "可核对数": len(errors), "无有效相邻AC跳过数": skipped, "float32容差": 5e-7, "最大绝对误差": max(errors), "示例": examples, "通过": True}


def load():
    """仅加载本模块产物；不访问任何原始源，不隐式准备数据。"""
    meta = json.loads((DATA / "数据元信息.json").read_text(encoding="utf-8"))
    expected = (meta["证券数"], meta["交易日数"])
    result = {name: np.load(DATA / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in DTYPES}
    for name in DTYPES:
        require(result[name].shape == expected and str(result[name].dtype) == DTYPES[name], f"落盘数组不符合元信息：{name}")
    index = pd.read_csv(DATA / "日历指数.csv", dtype={"日期": str})
    require(index["日期"].tolist() == meta["日期"], "落盘指数日期与元信息不符")
    result["meta"] = meta
    result["index"] = index["沪深300收盘"].to_numpy(dtype=np.float64)
    result["actions"] = pd.read_csv(DATA / "公司行动.csv", dtype={"证券": str, "除权日": str})
    return result


if __name__ == "__main__":
    main()
