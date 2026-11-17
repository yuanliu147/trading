# -*- coding: utf-8 -*-
"""事件层：R_k × H_h 12 格短期反转，PIT 宇宙、as-executed 收益、对照与统计。

时点定义
--------
t          信号日（收盘后可用 <=t 的信息计算 R_k 并分档）
t+1 开盘   买入；停牌 / 开盘涨停 / 无行情 / 不足 100 股 → 订单失败，不补成交
t+1+h 开盘 卖出（h=1 即 t+2，满足 T+1）；停牌 / 开盘跌停 → 逐日顺延至可卖或样本末

收益口径
--------
毛收益 = A 空间（后复权）开盘价之比 - 1（含分红再投资）
净收益 = 以原始价计整股、最低佣金、印花税、过户费、10bp 滑点 + 冲击后的现金流收益，
         再按 A 空间总收益/原始价收益的比值补回分红部分
事件名义金额 = 主资金/(每日只数×h)，与主账户每笔金额一致（00A A5）

对照
----
C1 同日同宇宙 SHA256 中性随机（排除当日输家档）
C2 同日同 ADV20 五分位 SHA256 中性随机（排除当日输家档），1:1
NN 同日 1:1 最近邻（log ADV / log 流通市值 / 20日波动，标准化欧氏，排除自身，有放回）
C3 可执行宇宙等权基准
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 配置 as C
import 费用与交易约束 as F

研究期起 = pd.Timestamp(C.研究期起)
研究期止 = pd.Timestamp(C.研究期止)


# --------------------------------------------------------------------------- #
# SHA256 中性哈希（A6：禁止 Python 内置 hash）
# --------------------------------------------------------------------------- #
def 哈希键(符号: np.ndarray, td: np.ndarray, 种子: int, 用途: bytes) -> np.ndarray:
    """SHA256( symbol|td|seed|用途 ) 前 8 字节 → uint64。仅依赖代码与日期，与收益无关。"""
    out = np.empty(len(符号), dtype=np.uint64)
    seed_b = int(种子).to_bytes(8, "little")
    for i, (s, t) in enumerate(zip(符号, td)):
        b = str(s).encode("ascii") + b"|" + int(t).to_bytes(4, "little") + b"|" + seed_b + b"|" + 用途
        out[i] = int.from_bytes(hashlib.sha256(b).digest()[:8], "little")
    return out


# --------------------------------------------------------------------------- #
class 面板:
    def __init__(self, 路径: Path | None = None):
        路径 = 路径 or (C.中间目录 / "面板_2009_2020.parquet")
        df = pd.read_parquet(路径)
        df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
        assert df["date"].max() <= 研究期止
        self.df = df
        df["sym"] = pd.Categorical(df["symbol"], categories=np.array(sorted(df["symbol"].unique()))).codes.astype(np.int32)
        self.symbols = np.array(sorted(df["symbol"].unique()))
        self.sym_id = pd.Categorical(df["symbol"], categories=self.symbols).codes.astype(np.int32)
        self.td_min = int(df["td"].min())
        self.td_max = int(df["td"].max())
        self.n_sym = len(self.symbols)
        self.n_td = self.td_max - self.td_min + 1
        self.rid = np.full((self.n_sym, self.n_td), -1, dtype=np.int32)
        self.rid[self.sym_id, df["td"].to_numpy() - self.td_min] = np.arange(len(df), dtype=np.int32)

        def col(n, dtype=np.float64):
            return df[n].to_numpy(dtype=dtype)

        self.O, self.H, self.L, self.C = col("open"), col("high"), col("low"), col("close")
        self.HO, self.HH, self.HL, self.HC = col("hfq_open"), col("hfq_high"), col("hfq_low"), col("hfq_close")
        self.VOL, self.AMT = col("volume"), col("amount")
        self.LUP, self.LDP = col("limit_up_px"), col("limit_down_px")
        self.SUSP = col("is_suspended", bool)
        self.LUO = col("is_limit_up_open", bool)
        self.LDO = col("is_limit_down_open", bool)
        self.UNI = col("宇宙", bool)
        self.ADV = col("ADV20")
        self.MCAP = col("float_mktcap")
        self.VOL20 = col("波动20")
        self.BETA = col("BETA60")
        self.ST = col("is_st")
        self.TD = col("td", np.int32)
        self.SYM = self.sym_id
        self.DATE = df["date"].to_numpy()

    def row(self, sym_idx, td):
        t = np.asarray(td)
        out = np.full(t.shape, -1, dtype=np.int64)
        ok = (t >= self.td_min) & (t <= self.td_max)
        if ok.any():
            out[ok] = self.rid[np.asarray(sym_idx)[ok], (t[ok] - self.td_min)]
        return out

    def 交易日日期(self, td):
        return self.df["date"].to_numpy()[np.clip(self.row(0, td), 0, None)]


# --------------------------------------------------------------------------- #
# 交易模拟
# --------------------------------------------------------------------------- #
def _买滑价(开: np.ndarray, bps: np.ndarray, 取整: bool = True) -> np.ndarray:
    raw = 开 * (1.0 + bps / 10000.0)
    if not 取整:
        return np.maximum(0.01, raw)
    return np.maximum(0.01, np.ceil(raw * 100.0 - 1e-9) / 100.0)


def _卖滑价(开: np.ndarray, bps: np.ndarray, 取整: bool = True) -> np.ndarray:
    raw = 开 * (1.0 - bps / 10000.0)
    if not 取整:
        return np.maximum(0.01, raw)
    return np.maximum(0.01, np.floor(raw * 100.0 + 1e-9) / 100.0)


def _费用(日期: np.ndarray, 代码: np.ndarray, 方向: str, 价: np.ndarray, 量: np.ndarray,
          倍数: float) -> dict:
    金额 = 价 * 量
    佣金 = np.maximum(金额 * F.FeeConfig().commission_rate * 倍数, F.FeeConfig().min_commission * 倍数)
    印花 = np.where(方向 == "sell", 金额 * 0.0010 * 倍数, 0.0)   # 2008-09-19 起卖出单边 0.1%
    沪 = np.char.startswith(代码.astype(str), "6")
    早 = 日期 < np.datetime64("2015-08-01")
    过户 = np.where(早, np.where(沪, 量 * 0.0006, 0.0), 金额 * 0.00002)
    def 分(x):
        return np.floor(x * 100.0 + 0.5) / 100.0
    佣金, 印花, 过户 = 分(佣金), 分(印花), 分(过户)
    return {"佣金": 佣金, "印花税": 印花, "过户费": 过户, "合计": 分(佣金 + 印花 + 过户)}


def 执行(P: 面板, sym: np.ndarray, sd: np.ndarray, h: int, *,
         成本倍数: float = 1.0, 延迟: int = 0, 名义金额: float | None = None,
         记录路径: bool = False, 滑点bp: float | None = None,
         冲击系数: float | None = None, 取整: bool = True) -> dict:
    """sd = 信号日 td。返回逐事件结果（含失败原因）。"""
    n = len(sd)
    名义 = C.事件名义金额(h) if 名义金额 is None else float(名义金额)
    结果 = {
        "可买": np.zeros(n, bool), "完结": np.zeros(n, bool),
        "未成交原因": np.full(n, "", dtype=object),
        "毛收益": np.full(n, np.nan), "净收益": np.full(n, np.nan),
        "顺延天数": np.full(n, -1, np.int16), "持有日": np.full(n, -1, np.int32),
        "买入价": np.full(n, np.nan), "卖出价": np.full(n, np.nan),
        "数量": np.zeros(n, np.float64),
        "佣金": np.full(n, np.nan), "印花税": np.full(n, np.nan), "过户费": np.full(n, np.nan),
        "费用合计": np.full(n, np.nan), "执行价差": np.full(n, np.nan),
        "买入td": np.full(n, -1, np.int32), "卖出td": np.full(n, -1, np.int32),
        "部分成交": np.zeros(n, bool),
        "MFE": np.full(n, np.nan), "MAE": np.full(n, np.nan),
    }
    买td = np.asarray(sd, dtype=np.int64) + 1 + 延迟
    br = P.row(sym, 买td)
    有行 = br >= 0
    b0 = np.maximum(br, 0)
    可 = (有行 & ~P.SUSP[b0] & ~P.LUO[b0] & (P.VOL[b0] > 0)
          & np.isfinite(P.O[b0]) & (P.O[b0] > 0))
    原因 = np.where(~有行, "次日无行情",
                    np.where(P.SUSP[b0], "次日停牌",
                             np.where(~(P.VOL[b0] > 0), "次日无成交",
                                      np.where(P.LUO[b0], "次日开盘涨停",
                                               np.where(~np.isfinite(P.O[b0]) | ~(P.O[b0] > 0), "次日无有效开盘价", ""))))).astype(object)
    结果["未成交原因"] = 原因

    idx = np.where(可)[0]
    if len(idx) == 0:
        return 结果

    b = br[idx]
    开 = P.O[b]
    上限 = np.where(np.isfinite(P.LUP[b]) & (P.LUP[b] > 0), P.LUP[b], np.inf)
    基础bp = (C.基础滑点bp if 滑点bp is None else 滑点bp) * 成本倍数
    冲击系数 = C.冲击系数bp if 冲击系数 is None else 冲击系数
    # 两轮：滑点/冲击依赖下单数量，数量依赖价格
    价 = np.minimum(_买滑价(开, np.full(len(idx), 基础bp), 取整), 上限)
    量 = np.floor(名义 / 价 / 100.0) * 100.0
    量上限 = np.floor(P.VOL[b] * C.最大参与率 / 100.0) * 100.0
    部分 = 量 > 量上限
    量 = np.minimum(量, 量上限)
    参与 = np.where(P.VOL[b] > 0, 量 / np.maximum(P.VOL[b], 1), 1.0)
    冲击bp = 冲击系数 * 成本倍数 * np.sqrt(np.maximum(参与, 0.0))
    价 = np.minimum(_买滑价(开, np.full(len(idx), 基础bp) + 冲击bp, 取整), 上限)
    量2 = np.minimum(np.floor(名义 / 价 / 100.0) * 100.0, 量上限)
    量, 部分 = 量2, (量2 < np.floor(名义 / 价 / 100.0) * 100.0)
    够 = 量 >= 100
    失败 = ~够
    if 失败.any():
        结果["未成交原因"][idx[失败]] = np.where(量上限[失败] < 100, "参与率上限导致0股", "整股不足")
    idx, b, 开, 价, 量, 部分 = idx[够], b[够], 开[够], 价[够], 量[够], 部分[够]
    if len(idx) == 0:
        return 结果
    结果["可买"][idx] = True
    结果["买入价"][idx] = 价
    结果["数量"][idx] = 量
    结果["部分成交"][idx] = 部分
    结果["买入td"][idx] = 买td[idx]

    # ---- 卖出：逐日顺延 ----
    sym_ok = sym[idx]
    卖td = 买td[idx] + h
    剩 = np.ones(len(idx), bool)
    卖行 = np.full(len(idx), -1, np.int64)
    顺延 = np.zeros(len(idx), np.int32)
    for _ in range(4000):
        if not 剩.any():
            break
        t = 卖td[剩]
        rr = P.row(sym_ok[剩], t)
        ok = (t <= P.td_max) & (rr >= 0)
        rr0 = np.maximum(rr, 0)
        ok &= ~P.SUSP[rr0] & ~P.LDO[rr0] & (P.VOL[rr0] > 0) & np.isfinite(P.O[rr0])
        pos = np.where(剩)[0]
        命中 = pos[ok]
        卖行[命中] = rr[ok]
        剩[命中] = False
        卖td[pos[~ok]] += 1
        顺延[pos[~ok]] += 1
        if (卖td[剩] > P.td_max).all() and 剩.any():
            break
    完成 = ~剩

    s = 卖行
    o_s = P.O[np.maximum(s, 0)]
    ld = np.where(np.isfinite(P.LDP[np.maximum(s, 0)]) & (P.LDP[np.maximum(s, 0)] > 0),
                  P.LDP[np.maximum(s, 0)], 0.0)
    参与s = np.where(P.VOL[np.maximum(s, 0)] > 0, 量 / np.maximum(P.VOL[np.maximum(s, 0)], 1), 1.0)
    冲击s = 冲击系数 * 成本倍数 * np.sqrt(np.maximum(参与s, 0.0))
    卖价 = np.maximum(_卖滑价(o_s, np.full(len(idx), 基础bp) + 冲击s, 取整), ld)
    卖价 = np.where(np.isfinite(o_s), 卖价, np.nan)

    毛 = P.HO[np.maximum(s, 0)] / P.HO[b] - 1.0
    买费 = _费用(P.DATE[b], P.symbols[P.SYM[b]], "buy", 价, 量, 成本倍数)
    卖费 = _费用(P.DATE[np.maximum(s, 0)], P.symbols[P.SYM[np.maximum(s, 0)]], "sell", 卖价, 量, 成本倍数)
    现金流比 = (卖价 * 量 - 卖费["合计"]) / (价 * 量 + 买费["合计"])
    价格比 = o_s / 开
    净 = (1.0 + 毛) * 现金流比 / 价格比 - 1.0
    价差 = (价 / 开) * (o_s / 卖价) - 1.0

    for key, val in [("毛收益", 毛), ("净收益", 净), ("卖出价", 卖价), ("执行价差", 价差),
                     ("佣金", 买费["佣金"] + 卖费["佣金"]), ("印花税", 卖费["印花税"]),
                     ("过户费", 买费["过户费"] + 卖费["过户费"]),
                     ("费用合计", 买费["合计"] + 卖费["合计"])]:
        结果[key][idx] = val
    结果["顺延天数"][idx] = np.where(完成, 顺延, -1)
    结果["持有日"][idx] = np.where(完成, s - b, -1)
    结果["卖出td"][idx] = np.where(完成, 卖td, -1)
    结果["完结"][idx] = 完成
    结果["未成交原因"][idx[~完成]] = "顺延至样本末仍不可卖"
    # 未实现的按最后可估值日期记 MFE/MAE 不可判，净收益留空
    结果["毛收益"][idx[~完成]] = np.nan
    结果["净收益"][idx[~完成]] = np.nan

    if 记录路径:
        ok_i = idx[完成]
        btd_ok = 买td[idx][完成]
        std_ok = 卖td[完成]
        最长 = int(min((std_ok - btd_ok).max(), 40)) if len(std_ok) else 0
        mfe = np.full(len(ok_i), -np.inf)
        mae = np.full(len(ok_i), np.inf)
        for j in range(最长 + 1):
            act = (std_ok - btd_ok) >= j
            if not act.any():
                continue
            rr = P.row(sym_ok[完成][act], btd_ok[act] + j)
            rr0 = np.maximum(rr, 0)
            hi = np.where(rr >= 0, P.HH[rr0] / P.HO[b[完成][act]] - 1, -np.inf)
            lo = np.where(rr >= 0, P.HL[rr0] / P.HO[b[完成][act]] - 1, np.inf)
            ai = np.where(act)[0]
            mfe[ai] = np.maximum(mfe[ai], hi)
            mae[ai] = np.minimum(mae[ai], lo)
        结果["MFE"][ok_i] = mfe
        结果["MAE"][ok_i] = mae
    return 结果


# --------------------------------------------------------------------------- #
# 宇宙快照 / 信号 / 对照
# --------------------------------------------------------------------------- #
def 宇宙快照(P: 面板) -> pd.DataFrame:
    """研究期内全部可执行宇宙股票日，附 ADV20 五分位与两组 SHA256 排序键。"""
    df = P.df
    m = (df["date"] >= 研究期起) & (df["date"] <= 研究期止) & df["宇宙"].to_numpy()
    u = df.loc[m, ["symbol", "date", "td", "ADV20", "float_mktcap", "波动20"]].copy()
    u["sym"] = pd.Categorical(u["symbol"], categories=P.symbols).codes.astype(np.int32)
    g = u.groupby("td", sort=False)["ADV20"]
    r = g.rank(method="first").to_numpy()
    n = g.transform("size").to_numpy()
    u["aq"] = np.minimum(((r - 1) * 5 // n), 4).astype(np.int8)
    u["h1"] = 哈希键(u["symbol"].to_numpy(), u["td"].to_numpy(), C.主种子, b"C1")
    u["h2"] = 哈希键(u["symbol"].to_numpy(), u["td"].to_numpy(), C.主种子, b"C2")
    return u.reset_index(drop=True)


def 生成信号(P: 面板, k: int, h: int) -> pd.DataFrame:
    df = P.df
    td = df["td"].to_numpy()
    m = ((df["date"].to_numpy() >= np.datetime64(C.研究期起)) &
         (df["date"].to_numpy() <= np.datetime64(C.研究期止)) &
         df["宇宙"].to_numpy() &
         df[f"窗口完整{k}"].to_numpy(dtype=bool) &
         df[f"R{k}"].notna().to_numpy() &
         (df[f"R{k}"].to_numpy() < 0) &
         (td + 1 + h <= P.td_max))
    s = df.loc[m, ["symbol", "date", "td", f"R{k}"]].copy()
    s["sym"] = pd.Categorical(s["symbol"], categories=P.symbols).codes.astype(np.int32)
    g = s.groupby("td", sort=False)[f"R{k}"]
    r = g.rank(method="first").to_numpy()
    n = g.transform("size").to_numpy()
    s["档"] = np.minimum(((r - 1) * C.分位档数 // n + 1), C.分位档数).astype(int)
    s["池子数"] = n
    s = s[s["档"] == 1].copy()
    s["组内名次"] = s.groupby("td").cumcount()
    return s.reset_index(drop=True)


def 生成对照(信号: pd.DataFrame, u: pd.DataFrame) -> dict:
    输 = 信号[["td", "sym", "组内名次"]].copy()
    候 = u.merge(信号[["td", "sym"]].assign(是输家=True), on=["td", "sym"], how="left")
    候 = 候[候["是输家"].isna()].drop(columns=["是输家"]).copy()

    # C1：同日 h1 升序取前 N，N = 当日输家数，按名次 1:1
    c1 = 候.sort_values(["td", "h1"], kind="mergesort").copy()
    c1["组内名次"] = c1.groupby("td").cumcount()
    n_need = 输.groupby("td").size().rename("n_need").reset_index()
    c1 = c1.merge(n_need, on="td", how="inner")
    c1 = c1[c1["组内名次"] < c1["n_need"]]
    对1 = 输.merge(c1[["td", "组内名次", "sym"]], on=["td", "组内名次"], suffixes=("", "_c"))

    # C2：同 (td, aq) 内 h2 升序，按取模轮转配对
    输q = 信号[["td", "sym", "组内名次", "池子数"]].merge(u[["td", "sym", "aq"]], on=["td", "sym"], how="left")
    输q["j"] = 输q.groupby(["td", "aq"]).cumcount()
    c2 = 候.sort_values(["td", "aq", "h2"], kind="mergesort").copy()
    c2["pos"] = c2.groupby(["td", "aq"]).cumcount()
    L = c2.groupby(["td", "aq"]).size().rename("L").reset_index()
    输q = 输q.merge(L, on=["td", "aq"], how="left")
    输q["pos"] = 输q["j"] % 输q["L"]
    对2 = 输q[["td", "sym", "aq", "pos"]].merge(
        c2[["td", "aq", "pos", "sym"]], on=["td", "aq", "pos"], suffixes=("", "_c"))
    return {"C1": 对1[["td", "sym", "sym_c"]], "C2": 对2[["td", "sym", "sym_c", "aq"]]}


def 最近邻匹配(u: pd.DataFrame, 信号: pd.DataFrame) -> pd.DataFrame:
    """同日 1:1 最近邻：标准化欧氏距离（log ADV20, log 流通市值, 20日波动），排除自身，有放回。"""
    特征列 = ["ADV20", "float_mktcap", "波动20"]
    idx = {td: g for td, g in u.groupby("td", sort=False)}
    行 = []
    for td, gg in 信号.groupby("td", sort=True):
        pool = idx.get(td)
        if pool is None or len(pool) == 0:
            continue
        Xp = np.column_stack([np.log(pool["ADV20"].to_numpy()), np.log(pool["float_mktcap"].to_numpy()),
                              pool["波动20"].to_numpy()])
        mu, sd = Xp.mean(0), Xp.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Zp = (Xp - mu) / sd
        Xl = np.column_stack([np.log(gg["ADV20"].to_numpy()), np.log(gg["float_mktcap"].to_numpy()),
                              gg["波动20"].to_numpy()])
        Zl = (Xl - mu) / sd
        pool_sym = pool["sym"].to_numpy()
        d2 = (Zl ** 2).sum(1)[:, None] + (Zp ** 2).sum(1)[None, :] - 2.0 * (Zl @ Zp.T)
        same = pool_sym[None, :] == gg["sym"].to_numpy()[:, None]
        d2[same] = np.inf
        j = np.argmin(d2, axis=1)
        ok = np.isfinite(d2[np.arange(len(j)), j])
        行.append(pd.DataFrame({"td": td, "sym": gg["sym"].to_numpy()[ok],
                                "sym_c": pool_sym[j][ok],
                                "距离": np.sqrt(d2[np.arange(len(j)), j][ok])}))
    return pd.concat(行, ignore_index=True) if 行 else pd.DataFrame(columns=["td", "sym", "sym_c", "距离"])


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def 日均(td: np.ndarray, v: np.ndarray, 基准td: np.ndarray) -> np.ndarray:
    s = pd.Series(v).groupby(td).mean().reindex(基准td)
    return s.to_numpy()


def 块自助(序列: np.ndarray, B: int, 块长: int, 种子: int) -> np.ndarray:
    """循环块自助（按日重采样，块长 >= h）。序列中 NaN 视为缺失日不参与。"""
    x = 序列[np.isfinite(序列)]
    n = len(x)
    if n < 5:
        return np.array([])
    块长 = int(min(max(块长, 1), n))
    k = int(np.ceil(n / 块长))
    rng = np.random.default_rng(种子)
    starts = rng.integers(0, n, size=(B, k))
    idx = (starts[:, :, None] + np.arange(块长)[None, None, :]) % n
    idx = idx.reshape(B, -1)[:, :n]
    return x[idx].mean(axis=1)


def 聚类自助(日值: np.ndarray, 日数: np.ndarray, B: int, 种子: int) -> np.ndarray:
    """按信号日聚类的加权（事件加权）自助。"""
    n = len(日值)
    if n == 0:
        return np.array([])
    rng = np.random.default_rng(种子)
    M = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M @ 日值) / (M @ 日数)


def 两侧p(分布: np.ndarray) -> float:
    if len(分布) == 0:
        return np.nan
    return float(min(1.0, 2 * min(np.mean(分布 <= 0), np.mean(分布 >= 0))))


def BH_FDR(p: np.ndarray, q: float = C.FDR阈值) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    qv = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        qv[order[i]] = prev
    return qv


def 分布摘要(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    正 = np.clip(v, 0, None).sum()
    srt = np.sort(v)[::-1]
    return {
        "笔数": int(len(v)), "均值": float(v.mean()), "中位数": float(np.median(v)),
        "胜率": float((v > 0).mean()), "标准差": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
        "最差": float(v.min()), "p1": float(np.quantile(v, 0.01)), "p5": float(np.quantile(v, 0.05)),
        "p95": float(np.quantile(v, 0.95)), "p99": float(np.quantile(v, 0.99)), "最好": float(v.max()),
        "右尾1%贡献": float(srt[:max(1, int(np.ceil(len(v) * 0.01)))].sum() / 正) if 正 > 0 else np.nan,
        "右尾5%贡献": float(srt[:max(1, int(np.ceil(len(v) * 0.05)))].sum() / 正) if 正 > 0 else np.nan,
    }


def 时代标签(年: np.ndarray) -> np.ndarray:
    out = np.full(len(年), "", dtype=object)
    for lab, a, b in C.两段:
        out[(年 >= a) & (年 <= b)] = lab
    return out


def 四期标签(年: np.ndarray) -> np.ndarray:
    out = np.full(len(年), "", dtype=object)
    for lab, a, b in C.四段:
        out[(年 >= a) & (年 <= b)] = lab
    return out
