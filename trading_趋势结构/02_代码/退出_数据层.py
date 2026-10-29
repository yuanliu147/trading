"""退出研究的数据层：从本地缓存构建对齐到统一交易日的面板矩阵。

只构建 2006-01-04 ~ 2020-12-31。留出窗口（2021+）在结构上就不进入面板，
这是时间纪律的第一道硬约束；第二道在 持仓状态机.assert_in_window。

矩阵布局：shape = (n_codes, n_days)，float32，缺失为 NaN。
已复权口径：所有指标与止损都在 hfq（后复权）空间；成交在原始价空间，
两者通过每日因子 f = close_hfq / close_raw 换算。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "01_数据", "exit_cache.duckdb")
PANEL = os.path.join(ROOT, "01_数据", "exit_panel.npz")

PANEL_START = "2006-01-04"
PANEL_END = "2020-12-31"

ATR_N = 20
FIELDS = ["o", "h", "l", "c", "pc", "vol", "amt", "f", "atr", "ma20", "ma60", "vma20",
          "oh", "hh", "lh", "ch", "pch"]


def _wilder_atr(h, l, c, n=ATR_N):
    """在压缩后的连续有效序列上计算，避免停牌/晚上市的 NaN 传染。"""
    m = len(c)
    out = np.full(m, np.nan)
    if m <= n:
        return out
    prev = np.empty_like(c)
    prev[0] = c[0]
    prev[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out[n] = tr[1 : n + 1].mean()
    a = 1.0 / n
    for i in range(n + 1, m):
        out[i] = out[i - 1] + a * (tr[i] - out[i - 1])
    return out


def _pivot_low_conf(l, left=2, right=2):
    """5 根分型的中间低点，只在右侧 right 根确认后才可用（禁止回填）。"""
    out = np.full_like(l, np.nan)
    n = len(l)
    for j in range(left, n - right):
        v = l[j]
        if np.isnan(v):
            continue
        ok = True
        for k in range(1, left + 1):
            if not (v < l[j - k]):
                ok = False
                break
        if ok:
            for k in range(1, right + 1):
                if not (v < l[j + k]):
                    ok = False
                    break
        if ok:
            out[j + right] = v
    return out


def _rolling_mean(x, n):
    """NaN 感知的简单移动平均，窗口内有效值不足 n 则 NaN。"""
    s = pd.Series(x, dtype="float64")
    return s.rolling(n, min_periods=n).mean().to_numpy()


def build(force: bool = False) -> None:
    if os.path.exists(PANEL) and not force:
        print(f"panel 已存在，跳过构建：{PANEL}")
        return
    import duckdb

    con = duckdb.connect(CACHE, read_only=True)
    cal = con.execute(
        f"select distinct date from px_raw where date between '{PANEL_START}' and '{PANEL_END}' order by 1"
    ).df()["date"].astype(str).to_numpy()
    codes = con.execute(
        f"""select distinct r.code from px_raw r join px_hfq h using(code, date)
            where r.date between '{PANEL_START}' and '{PANEL_END}'
            group by 1 having count(*) > 260 order by 1"""
    ).df()["code"].astype(str).to_numpy()

    d_idx = {d: i for i, d in enumerate(cal)}
    c_idx = {c: i for i, c in enumerate(codes)}
    nc, nd = len(codes), len(cal)
    print(f"面板 {nc} 只 × {nd} 个交易日")

    mat = {k: np.full((nc, nd), np.nan, dtype=np.float32) for k in FIELDS}
    tradable = np.zeros((nc, nd), dtype=bool)
    is_st = np.zeros((nc, nd), dtype=bool)
    listed = np.zeros((nc, nd), dtype=np.int32)

    q = f"""
        select r.code, r.date, r.open, r.high, r.low, r.close, r.preclose,
               r.volume, r.amount, r.trade_status, r.is_st, h.close_hfq
        from px_raw r join px_hfq h on r.code=h.code and r.date=h.date
        where r.date between '{PANEL_START}' and '{PANEL_END}'
    """
    cur = con.execute(q)
    done = 0
    while True:
        chunk = cur.fetch_df_chunk(8)
        if len(chunk) == 0:
            break
        ci = chunk["code"].map(c_idx).to_numpy()
        m = chunk["code"].notna().to_numpy()
        ci = pd.Series(chunk["code"]).map(c_idx).to_numpy()
        di = pd.Series(chunk["date"].astype(str)).map(d_idx).to_numpy()
        ok = pd.notna(ci) & pd.notna(di)
        ci = ci[ok].astype(np.int64)
        di = di[ok].astype(np.int64)
        df = chunk[ok]
        o = df["open"].to_numpy(np.float32)
        h = df["high"].to_numpy(np.float32)
        lo = df["low"].to_numpy(np.float32)
        cl = df["close"].to_numpy(np.float32)
        pc = df["preclose"].to_numpy(np.float32)
        vo = df["volume"].to_numpy(np.float32)
        am = df["amount"].to_numpy(np.float32)
        hf = df["close_hfq"].to_numpy(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where((cl > 0) & (hf > 0), hf / cl, np.nan)
        for name, arr in zip(["o", "h", "l", "c", "pc", "vol", "amt", "f"], [o, h, lo, cl, pc, vo, am, f]):
            mat[name][ci, di] = arr
        tradable[ci, di] = (df["trade_status"].to_numpy() == 1) & (vo > 0)
        is_st[ci, di] = df["is_st"].to_numpy() == 1
        done += len(df)
    con.close()
    print(f"载入 {done} 行")

    # 逐只股票：hfq OHLC、ATR、均线、分型（全部在压缩后的有效序列上算）
    mat["piv"] = np.full((nc, nd), np.nan, dtype=np.float32)
    for i in range(nc):
        cl = mat["c"][i]
        valid = ~np.isnan(cl)
        if valid.sum() < 260:
            continue
        f = mat["f"][i].copy()
        # 因子只在有效日有值，向前填充用于复权换算
        f = pd.Series(f).ffill().bfill().to_numpy()
        mat["f"][i] = f
        pos = np.where(valid)[0]
        oh = mat["o"][i, pos] * f[pos]
        hh = mat["h"][i, pos] * f[pos]
        ll = mat["l"][i, pos] * f[pos]
        cc = mat["c"][i, pos] * f[pos]
        mat["oh"][i, pos] = oh.astype(np.float32)
        mat["hh"][i, pos] = hh.astype(np.float32)
        mat["lh"][i, pos] = ll.astype(np.float32)
        mat["ch"][i, pos] = cc.astype(np.float32)
        mat["pch"][i, pos] = (mat["pc"][i, pos] * f[pos]).astype(np.float32)
        mat["atr"][i, pos] = _wilder_atr(hh, ll, cc).astype(np.float32)
        mat["ma20"][i, pos] = _rolling_mean(cc, 20).astype(np.float32)
        mat["ma60"][i, pos] = _rolling_mean(cc, 60).astype(np.float32)
        mat["vma20"][i, pos] = _rolling_mean(mat["vol"][i, pos], 20).astype(np.float32)
        listed[i] = np.cumsum(valid).astype(np.int32)
        # 分型低点（hfq）另存，避免用原始价误判
        mat["piv"][i, pos] = _pivot_low_conf(ll).astype(np.float32)
        if (i + 1) % 500 == 0:
            print(f"  指标 {i+1}/{nc}", flush=True)

    np.savez_compressed(
        PANEL,
        dates=cal.astype("U10"),
        codes=codes.astype("U12"),
        tradable=tradable,
        is_st=is_st,
        listed=listed,
        **{k: v for k, v in mat.items()},
    )
    print(f"已写入 {PANEL}")


class Panel:
    def __init__(self, path: str = PANEL):
        z = np.load(path, allow_pickle=False)
        self.dates = z["dates"]
        self.codes = z["codes"]
        self.tradable = z["tradable"]
        self.is_st = z["is_st"]
        self.listed = z["listed"]
        self.m = {k: z[k] for k in z.files if k not in ("dates", "codes", "tradable", "is_st", "listed")}
        self.d_idx = {d: i for i, d in enumerate(self.dates)}
        self.c_idx = {c: i for i, c in enumerate(self.codes)}
        self.n_days = len(self.dates)
        self.n_codes = len(self.codes)
        # 涨跌幅规则的快速查表：创业板/科创板板块 + 2020-08-24 之后的 20% 制度
        pfx = np.array([str(c)[3:6] for c in self.codes])
        self.board_id = ((pfx == "300") | (pfx == "301")).astype(np.int8)
        self.day20 = self.dates >= "2020-08-24"
        self.date_str = [str(d) for d in self.dates]
        self.code_str = [str(c) for c in self.codes]

    def __getitem__(self, k):
        return self.m[k]

    def limit_pct(self, ci: int, di: int) -> float:
        if self.board_id[ci] and self.day20[di]:
            return 0.20
        return 0.05 if self.is_st[ci, di] else 0.10

    def limits(self, ci: int, di: int):
        pc = float(self.m["pc"][ci, di])
        if not np.isfinite(pc) or pc <= 0:
            return np.nan, np.nan
        L = self.limit_pct(ci, di)
        return round(pc * (1 + L), 2), round(pc * (1 - L), 2)

    def truncate(self, end: str) -> "Panel":
        """按 end 截断日历，用于"后续数据不可见"的因果性检验。"""
        i1 = int(np.searchsorted(self.dates, end, side="right"))
        new = object.__new__(Panel)
        new.dates = self.dates[:i1]
        new.codes = self.codes
        new.tradable = self.tradable[:, :i1]
        new.is_st = self.is_st[:, :i1]
        new.listed = self.listed[:, :i1]
        new.m = {k: v[:, :i1] for k, v in self.m.items()}
        new.d_idx = {str(d): i for i, d in enumerate(new.dates)}
        new.c_idx = self.c_idx
        new.n_days = i1
        new.n_codes = self.n_codes
        new.board_id = self.board_id
        new.day20 = self.day20[:i1]
        new.date_str = self.date_str[:i1]
        new.code_str = self.code_str
        return new

    def bar(self, ci: int, di: int):
        """返回该日 bar 的原始价与 hfq 价，供撮合与状态机使用。热路径，避免生成器与逐字段 float()。"""
        g = self.m
        f = g["f"][ci, di]
        o = g["o"][ci, di]
        h = g["h"][ci, di]
        lo = g["l"][ci, di]
        c = g["c"][ci, di]
        if not (f > 0 and o > 0 and h > 0 and lo > 0 and c > 0):
            return None
        pc = g["pc"][ci, di]
        if self.board_id[ci] and self.day20[di]:
            L = 0.20
        elif self.is_st[ci, di]:
            L = 0.05
        else:
            L = 0.10
        if pc > 0:
            up = round(pc * (1 + L), 2)
            dn = round(pc * (1 - L), 2)
        else:
            up = dn = float("nan")
        return dict(
            date=self.date_str[di],
            o=o, h=h, l=lo, c=c,
            oh=o * f, hh=h * f, lh=lo * f, ch=c * f, f=f,
            pc=pc, pch=g["pch"][ci, di],
            vol=g["vol"][ci, di], amt=g["amt"][ci, di],
            tradable=self.tradable[ci, di], st=self.is_st[ci, di],
            up=up, dn=dn,
            atr=g["atr"][ci, di], ma20=g["ma20"][ci, di], ma60=g["ma60"][ci, di],
            vma20=g["vma20"][ci, di], piv=g["piv"][ci, di],
        )


def audit() -> pd.DataFrame:
    """数据覆盖与机制审计，只读，不产生任何收益数字。"""
    p = Panel()
    rows = []
    c = p["c"]
    valid = np.isfinite(c)
    rows.append(("交易日数", p.n_days))
    rows.append(("股票数", p.n_codes))
    rows.append(("有效bar数", int(valid.sum())))
    rows.append(("可交易bar数", int(p.tradable.sum())))
    rows.append(("停牌bar数", int((valid & ~p.tradable).sum())))
    rows.append(("ST bar数", int((valid & p.is_st).sum())))
    rows.append(("ATR 可用bar数", int(np.isfinite(p["atr"]).sum())))
    rows.append(("MA60 可用bar数", int(np.isfinite(p["ma60"]).sum())))
    rows.append(("分型确认数", int(np.isfinite(p["piv"]).sum())))
    # 一字板
    hh, ll, up = p["h"], p["l"], None
    oneprice = np.isfinite(c) & (np.abs(hh - ll) < 1e-9)
    rows.append(("一字板bar数", int(oneprice.sum())))
    # 极端跳空（复牌/缩股等异常）
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c / p["pc"] - 1
    rows.append(("|涨跌幅|>45% 异常bar数", int(np.nansum(np.abs(r) > 0.45))))
    return pd.DataFrame(rows, columns=["指标", "数值"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        print(audit().to_string(index=False))
    else:
        build(force="--force" in sys.argv)
        print(audit().to_string(index=False))
