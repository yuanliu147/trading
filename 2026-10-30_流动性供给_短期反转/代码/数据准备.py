# -*- coding: utf-8 -*-
"""数据准备：只读母项目 frozen 分区，构造 2009-10..2020-12 沪深主板 PIT 面板。

硬约束（00A A1）
----------------
* 只打开 year=2009..2020 的 Parquet 分区，2021 及以后的文件从不进入文件列表；
* 列白名单：不读取 name / delist_date / 任何全样本行数统计；
* 不修改母项目任何文件；输出只写本项目 结果/中间数据/；
* 所有滚动特征窗口右端 = 当日（含当日收盘已知信息），不使用未来数据。

滚动特征用 (证券 × 交易日) 矩阵 + 前缀和实现；停牌日源库保留行，
因此矩阵在证券存续期内是完整网格，缺失日不会被压缩成跨日收益。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 配置 as C

# 白名单列（A1：不得出现 delist_date / name / 全样本统计）
读取列 = [
    "symbol", "date", "board", "ipo_date", "td_since_list",
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "is_suspended", "is_st", "pct_chg", "factor",
    "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    "limit_up_px", "limit_down_px",
    "is_limit_up_open", "is_limit_down_open",
    "px_breaks_limit_up", "px_breaks_limit_down", "float_mktcap",
]

禁用列 = {"name", "delist_date", "bs_hfq_close", "close_ret_check", "turn", "trade_status"}


# --------------------------------------------------------------------------- #
# 矩阵工具
# --------------------------------------------------------------------------- #
def 滚动和(M: np.ndarray, w: int) -> np.ndarray:
    """按列滚动求和，允许部分窗口（前 w-1 列为已有部分）。NaN 记 0。"""
    Z = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    cs = np.cumsum(Z, axis=1)
    out = cs.copy()
    out[:, w:] = cs[:, w:] - cs[:, :-w]
    return out


def 滚动计数(M: np.ndarray, w: int) -> np.ndarray:
    return 滚动和(np.isfinite(M).astype(np.float64), w)


def 滚动标准差(M: np.ndarray, w: int, 最少: int) -> np.ndarray:
    V = np.isfinite(M).astype(np.float64)
    X = np.where(V > 0, M, 0.0)
    n = 滚动和(V, w)
    s1 = 滚动和(X, w)
    s2 = 滚动和(X * X, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
    out = np.sqrt(np.maximum(var, 0.0))
    out[n < 最少] = np.nan
    return out


def 滚动协方差比(X: np.ndarray, Y: np.ndarray, w: int, 最少: int) -> np.ndarray:
    """cov(X,Y)/var(Y) 的滚动值，X/Y 同形（Y 为市场收益，逐行相同）。"""
    V = (np.isfinite(X) & np.isfinite(Y)).astype(np.float64)
    A = np.where(V > 0, X, 0.0)
    B = np.where(V > 0, Y, 0.0)
    n = 滚动和(V, w)
    sa, sb = 滚动和(A, w), 滚动和(B, w)
    sab, sbb = 滚动和(A * B, w), 滚动和(B * B, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (sab - sa * sb / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
        var = (sbb - sb * sb / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
        beta = cov / var
    beta[(n < 最少) | ~(var > 0)] = np.nan
    return beta


# --------------------------------------------------------------------------- #
def 文件清单() -> list[Path]:
    全部 = sorted(C.母项目数据.glob("daily/year=*/data_*.parquet"))
    选中 = [f for f in 全部 if int(f.parent.name.split("=")[1]) in C.数据年]
    年集 = {int(f.parent.name.split("=")[1]) for f in 选中}
    assert max(年集) <= 2020, f"读到了封存年：{max(年集)}"
    assert min(年集) >= 2009
    return 选中


def 主流程(强制: bool = False) -> dict:
    C.中间目录.mkdir(parents=True, exist_ok=True)
    面板路径 = C.中间目录 / "面板_2009_2020.parquet"
    清单路径 = C.中间目录 / "数据准备清单.json"
    if 面板路径.exists() and not 强制:
        print(f"[跳过] 已存在 {面板路径}（--force 重建）")
        return json.loads(清单路径.read_text(encoding="utf-8"))

    files = 文件清单()
    assert all("2021" not in str(f) for f in files)
    print(f"读取 {len(files)} 个分区：{files[0].parent.name} .. {files[-1].parent.name}", flush=True)
    frames = []
    for f in files:
        t = pd.read_parquet(f, columns=读取列)
        多出 = 禁用列.intersection(t.columns)
        assert not 多出, f"白名单外字段进入：{多出}"
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    del frames
    n_all = len(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["board"] == C.板块].copy()
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    print(f"主板行数 {len(df)}，证券 {df['symbol'].nunique()}", flush=True)

    # ---- 交易日索引 ----
    cal = pd.read_parquet(C.母项目数据 / "calendar.parquet")
    cal = cal[cal["is_trading_day"] == 1].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.sort_values("date").reset_index(drop=True)
    cal["td"] = np.arange(len(cal), dtype=np.int32)
    td_map = dict(zip(cal["date"], cal["td"]))
    df["td"] = df["date"].map(td_map).astype(np.int32)
    assert df["td"].notna().all()

    symbols = np.array(sorted(df["symbol"].unique()))
    sym_id = pd.Categorical(df["symbol"], categories=symbols).codes.astype(np.int32)
    td_min, td_max = int(df["td"].min()), int(df["td"].max())
    n_sym, n_td = len(symbols), td_max - td_min + 1
    rid = np.full((n_sym, n_td), -1, dtype=np.int32)
    rid[sym_id, df["td"].to_numpy() - td_min] = np.arange(len(df), dtype=np.int32)
    print(f"矩阵 {n_sym} × {n_td}", flush=True)

    def 矩阵(col, dtype=np.float64, 无效=np.nan):
        M = np.full((n_sym, n_td), 无效, dtype=dtype)
        M[sym_id, df["td"].to_numpy() - td_min] = df[col].to_numpy(dtype=dtype)
        return M

    O = 矩阵("open")
    HH = 矩阵("high")
    LL = 矩阵("low")
    CL = 矩阵("close")
    PC = 矩阵("preclose")
    VOL = 矩阵("volume")
    AMT = 矩阵("amount")
    HFQO = 矩阵("hfq_open")
    HFQH = 矩阵("hfq_high")
    HFQL = 矩阵("hfq_low")
    HFQC = 矩阵("hfq_close")
    LUP = 矩阵("limit_up_px")
    LDP = 矩阵("limit_down_px")
    MCAP = 矩阵("float_mktcap")
    PCT = 矩阵("pct_chg")
    SUSP = 矩阵("is_suspended", bool, False)
    ST = 矩阵("is_st", np.int8, 0)
    LUO = 矩阵("is_limit_up_open", bool, False)
    LDO = 矩阵("is_limit_down_open", bool, False)
    BRKU = 矩阵("px_breaks_limit_up", bool, False)
    BRKD = 矩阵("px_breaks_limit_down", bool, False)

    # pct_chg 缺失（停牌行）用 close/preclose-1 补，再缺记 0
    with np.errstate(invalid="ignore", divide="ignore"):
        R_RAW = PCT / 100.0
        补 = CL / PC - 1.0
    R_RAW = np.where(np.isfinite(R_RAW), R_RAW, 补)
    R_RAW = np.where(np.isfinite(R_RAW), R_RAW, 0.0)

    # ---- 停牌 / 复牌 ----
    前日停牌 = np.zeros_like(SUSP)
    前日停牌[:, 1:] = SUSP[:, :-1]
    复牌首日 = (~SUSP) & 前日停牌

    # ---- 数据冲突（数据字典 §5.1 的 pct_chg 源冲突；主板上仅 001965.SZ 2017-12-25） ----
    with np.errstate(invalid="ignore", divide="ignore"):
        冲突 = np.abs(CL / PC - 1.0 - PCT / 100.0) > 0.006
    冲突 &= np.isfinite(CL) & np.isfinite(PC) & np.isfinite(PCT)

    价格有效 = np.isfinite(O) & np.isfinite(CL) & (CL > 0) & (O > 0)
    有效观测 = (~SUSP) & (~复牌首日) & (~冲突)

    # ---- ADV20 ----
    AMT有效 = np.where(np.isfinite(AMT) & (AMT > 0), AMT, np.nan)
    adv_sum = 滚动和(AMT有效, C.ADV窗口)
    adv_cnt = 滚动计数(AMT有效, C.ADV窗口)
    with np.errstate(invalid="ignore", divide="ignore"):
        ADV20 = adv_sum / adv_cnt
    ADV20[adv_cnt < C.ADV最少有效日] = np.nan

    # ---- 形成期收益 ----
    R = {}
    for k in C.k列表:
        M = np.full((n_sym, n_td), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            M[:, k:] = HFQC[:, k:] / HFQC[:, :-k] - 1.0
        R[k] = M

    # ---- 窗口完整性：k+1 个交易日全部为有效观测（含无停牌/无复牌/无冲突） ----
    完整 = {}
    for k in C.k列表:
        有效数 = 滚动和(有效观测.astype(np.float64), k + 1)
        完整[k] = 有效数 >= (k + 1) - 1e-9

    # ---- 波动与 beta ----
    波动 = 滚动标准差(R_RAW, C.波动窗口, C.波动最少有效日)
    指数文件 = []
    for y in C.数据年:
        p = C.母项目数据 / f"index_daily/year={y}/data_0.parquet"
        if p.exists():
            指数文件.append(p)
    ixdf = pd.concat([pd.read_parquet(p, columns=["symbol", "date", "pct_chg"]) for p in 指数文件],
                     ignore_index=True)
    ixdf = ixdf[ixdf["symbol"] == C.基准指数].copy()
    ixdf["date"] = pd.to_datetime(ixdf["date"])
    ixdf = ixdf[(ixdf["date"] >= df["date"].min()) & (ixdf["date"] <= df["date"].max())]
    ix_ret = np.full(n_td, np.nan)
    ix_ret[(ixdf["date"].map(td_map).astype(int) - td_min).to_numpy()] = ixdf["pct_chg"].to_numpy() / 100.0
    ix_ret = np.where(np.isfinite(ix_ret), ix_ret, 0.0)
    BETA = 滚动协方差比(R_RAW, np.broadcast_to(ix_ret, R_RAW.shape), C.BETA窗口, C.BETA最少有效日)

    # ---- 上市时长 ----
    ipo = np.full((n_sym, n_td), np.datetime64("NaT"))
    ipo[sym_id, df["td"].to_numpy() - td_min] = df["ipo_date"].to_numpy()
    tdsl = 矩阵("td_since_list", np.float64)

    # ---- 铺回面板 ----
    df["R3"], df["R5"], df["R10"] = R[3][sym_id, df["td"].to_numpy() - td_min], R[5][sym_id, df["td"].to_numpy() - td_min], R[10][sym_id, df["td"].to_numpy() - td_min]
    df["窗口完整3"] = 完整[3][sym_id, df["td"].to_numpy() - td_min]
    df["窗口完整5"] = 完整[5][sym_id, df["td"].to_numpy() - td_min]
    df["窗口完整10"] = 完整[10][sym_id, df["td"].to_numpy() - td_min]
    df["ADV20"] = ADV20[sym_id, df["td"].to_numpy() - td_min]
    df["有效日20"] = adv_cnt[sym_id, df["td"].to_numpy() - td_min]
    df["波动20"] = 波动[sym_id, df["td"].to_numpy() - td_min]
    df["BETA60"] = BETA[sym_id, df["td"].to_numpy() - td_min]
    df["复牌首日"] = 复牌首日[sym_id, df["td"].to_numpy() - td_min]
    df["数据冲突"] = 冲突[sym_id, df["td"].to_numpy() - td_min]
    df["前日停牌"] = 前日停牌[sym_id, df["td"].to_numpy() - td_min]

    df["ipo_date"] = pd.to_datetime(df["ipo_date"])
    df["ipo自然日"] = (df["date"] - df["ipo_date"]).dt.days
    df["价格达标"] = (df["close"] >= C.最低原始收盘) & 价格有效[sym_id, df["td"].to_numpy() - td_min]
    df["流动性达标"] = df["ADV20"] >= C.最低ADV20
    df["上市达标"] = (df["ipo自然日"] >= C.最短自然日) & (df["td_since_list"] >= C.最短交易日)
    df["非ST"] = df["is_st"].fillna(0).astype(int).eq(0)
    df["当日可交易"] = ~df["is_suspended"].astype(bool)
    df["非复牌首日"] = ~df["复牌首日"]
    df["宇宙"] = (df["价格达标"] & df["流动性达标"] & df["上市达标"] & df["非ST"]
                  & df["当日可交易"] & df["非复牌首日"] & ~df["数据冲突"])
    df["开盘涨停"] = df["is_limit_up_open"].fillna(False).astype(bool)
    df["开盘跌停"] = df["is_limit_down_open"].fillna(False).astype(bool)
    df["规则例外"] = df["px_breaks_limit_up"].fillna(False).astype(bool) | \
                    df["px_breaks_limit_down"].fillna(False).astype(bool)

    df = df[df["date"] >= pd.Timestamp(C.面板起)].copy()
    assert df["date"].max() <= pd.Timestamp(C.研究期止), "面板越过 2020-12-31"
    keep = ["symbol", "date", "td", "ipo_date", "td_since_list", "ipo自然日",
            "open", "high", "low", "close", "preclose", "volume", "amount", "pct_chg", "factor",
            "hfq_open", "hfq_high", "hfq_low", "hfq_close",
            "limit_up_px", "limit_down_px", "is_limit_up_open", "is_limit_down_open",
            "px_breaks_limit_up", "px_breaks_limit_down", "float_mktcap",
            "is_suspended", "is_st", "复牌首日", "前日停牌", "数据冲突", "规则例外",
            "ADV20", "有效日20", "波动20", "BETA60",
            "R3", "R5", "R10", "窗口完整3", "窗口完整5", "窗口完整10",
            "价格达标", "流动性达标", "上市达标", "非ST", "当日可交易", "非复牌首日",
            "宇宙", "开盘涨停", "开盘跌停"]
    df = df[keep].reset_index(drop=True)
    df.to_parquet(面板路径, index=False)

    # ---- 自检 ----
    检查 = 自检(df)

    清单 = {
        "分区文件": [str(f) for f in files],
        "分区文件_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "读取分区年份": sorted({int(f.parent.name.split("=")[1]) for f in files}),
        "最大日期": str(df["date"].max().date()),
        "最小日期": str(df["date"].min().date()),
        "原始行数_全部板块": int(n_all),
        "主板行数": int(len(df)),
        "证券数": int(df["symbol"].nunique()),
        "封存年读取": False,
        "列": list(df.columns),
        "禁用列未读": sorted(禁用列),
        "宇宙口径": {
            "名称": "U-PIT-MAIN-LIQ1",
            "板块": C.板块, "最短自然日": C.最短自然日, "最短交易日": C.最短交易日,
            "最低原始收盘": C.最低原始收盘, "最低ADV20": C.最低ADV20,
            "ADV窗口": C.ADV窗口, "ADV最少有效日": C.ADV最少有效日,
            "ADV定义": "过去20个交易日行内 amount>0 的成交额均值，至少15个有效日（含当日收盘）",
            "复牌首日定义": "前一日停牌且当日未停牌",
            "退市整理期过滤": "无（不继承旧宇宙的未来退市日/最后行情日过滤）",
            "ST": "当日 is_st==0",
        },
        "自检": 检查,
    }
    清单路径.write_text(json.dumps(清单, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(检查, ensure_ascii=False, indent=2), flush=True)
    return 清单


def 自检(df: pd.DataFrame) -> dict:
    出 = {}
    出["最大日期"] = str(df["date"].max().date())
    assert df["date"].max() <= pd.Timestamp(C.研究期止)
    # A 空间日收益 = 官方 pct_chg
    s = df.sort_values(["symbol", "date"])
    r_hfq = s.groupby("symbol")["hfq_close"].pct_change()
    r_off = s["pct_chg"] / 100.0
    m = r_hfq.notna() & r_off.notna() & (s["is_suspended"] == False)
    diff = (r_hfq[m] - r_off[m]).abs()
    出["A空间日收益_vs_pct_chg_最大偏差"] = float(diff.max())
    出["A空间日收益_vs_pct_chg_超0.5bp行数"] = int((diff > 5e-5).sum())
    出["A空间校验行数"] = int(m.sum())
    assert diff.max() < 1e-9
    # OHLC 比例保持
    ratio = (s["hfq_high"] / s["hfq_low"]) / (s["high"] / s["low"])
    出["OHLC比例最大误差"] = float((ratio - 1).abs().max())
    assert (ratio - 1).abs().max() < 1e-8
    出["宇宙股票日"] = int(df["宇宙"].sum())
    出["宇宙证券数"] = int(df.loc[df["宇宙"], "symbol"].nunique())
    出["数据冲突行"] = int(df["数据冲突"].sum())
    出["规则例外行"] = int(df["规则例外"].sum())
    出["规则例外且在宇宙"] = int((df["规则例外"] & df["宇宙"]).sum())
    出["ST股票日占比"] = float(df["is_st"].mean())
    return 出


if __name__ == "__main__":
    主流程(强制="--force" in sys.argv)
