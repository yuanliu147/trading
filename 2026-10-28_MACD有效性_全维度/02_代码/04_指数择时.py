#!/usr/bin/env python3
"""04 指数择时：MACD 作为时间序列趋势过滤器是否有效

标的：沪深300、中证500、创业板指、上证指数（2006-2026，指数日线）。
规则：T 日收盘信号，T+1 开盘切换（无未来）；多头/空仓，单边成本 0.05%。
对照：买入持有；DIF>DEA；DIF>0；DIF>DEA 且 DIF>0；周线 DIF>DEA。
参数网格：fast/slow/signal，输出 CAGR/Sharpe/MDD/Calmar/换手/胜率。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
FEE = 0.0005


def load_idx():
    df = pd.read_parquet(os.path.join(RES, "指数_日线.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for name, g in df.groupby("idx_name"):
        g = g.sort_values("date").set_index("date")
        out[name] = g[["open", "high", "low", "close"]].astype(float)
    return out


def perf(ret, pos, fee=FEE):
    """ret: 日收益；pos: 目标仓位（T+1 开盘后生效）。返回年化/波动/Sharpe/MDD/Calmar/换手。"""
    pos = pd.Series(pos).fillna(0).clip(0, 1)
    turn = pos.diff().abs().fillna(pos.abs())
    r = pos.shift(1).fillna(0) * ret - turn * fee
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar,
                switches=int((turn > 0).sum()), exposure=pos.mean(),
                final=eq.iloc[-1])


def signal_pos(df, kind="cross", fast=12, slow=26, sig=9):
    c = df.close
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    # 信号在 T 收盘；pos 在 T+1 生效（perf 内 shift 1）
    if kind == "cross":
        s = (dif > dea).astype(float)
    elif kind == "zero":
        s = (dif > 0).astype(float)
    elif kind == "both":
        s = ((dif > dea) & (dif > 0)).astype(float)
    elif kind == "hist":
        s = ((dif - dea) > 0).astype(float)
    else:
        raise ValueError(kind)
    return s


def weekly_pos(df, fast=12, slow=26, sig=9, kind="cross"):
    c = df.close
    wk = c.resample("W-FRI").last()
    dif = wk.ewm(span=fast, adjust=False).mean() - wk.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    if kind == "cross":
        s = (dif > dea).astype(float)
    elif kind == "both":
        s = ((dif > dea) & (dif > 0)).astype(float)
    else:
        s = (dif > 0).astype(float)
    # 用已完成周线，下一周生效；映射到日
    s = s.shift(1).reindex(c.index, method="ffill").fillna(0)
    return s


def run():
    idx = load_idx()
    rows = []
    for name, df in idx.items():
        if len(df) < 500:
            continue
        # T+1 开盘执行：open-to-open 收益；perf 内 pos.shift(1) 使信号在 T+1 开盘生效
        ret = (df.open.shift(-1) / df.open - 1).fillna(0)
        # 买入持有
        bh = perf(ret, pd.Series(1.0, index=df.index))
        rec = dict(指数=name, 规则="买入持有", **bh)
        rows.append(rec)
        for kind, label in [("cross", "DIF>DEA"), ("zero", "DIF>0"),
                            ("both", "DIF>DEA且DIF>0")]:
            pos = signal_pos(df, kind)
            rec = dict(指数=name, 规则=label, **perf(ret, pos))
            rows.append(rec)
        pos = weekly_pos(df, kind="cross")
        rec = dict(指数=name, 规则="周线DIF>DEA", **perf(ret, pos))
        rows.append(rec)
        pos = weekly_pos(df, kind="both")
        rec = dict(指数=name, 规则="周线DIF>DEA且DIF>0", **perf(ret, pos))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "指数择时_规则对比.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))

    # 参数网格：沪深300 / 中证500 / 创业板
    grid = []
    for name in ["CSI300", "CSI500", "CYB"]:
        if name not in idx:
            continue
        df = idx[name]
        # T+1 开盘执行：open-to-open 收益；perf 内 pos.shift(1) 使信号在 T+1 开盘生效
        ret = (df.open.shift(-1) / df.open - 1).fillna(0)
        for fast in [5, 8, 10, 12, 15, 20, 25, 30]:
            for slow in [20, 26, 30, 35, 40, 50, 60, 80]:
                if fast >= slow:
                    continue
                for sig in [5, 9, 15, 20]:
                    s = signal_pos(df, "cross", fast, slow, sig)
                    p = perf(ret, s)
                    grid.append(dict(指数=name, fast=fast, slow=slow, signal=sig, **p))
    g = pd.DataFrame(grid)
    g.to_csv(os.path.join(RES, "指数择时_参数网格.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 参数网格 best CAGR ===")
    print(g.sort_values("cagr", ascending=False).head(15).to_string(index=False))
    print("\n=== 参数网格 best Sharpe ===")
    print(g.sort_values("sharpe", ascending=False).head(10).to_string(index=False))
    print("\n=== 12/26/9 ===")
    print(g[(g.fast == 12) & (g.slow == 26) & (g.signal == 9)].to_string(index=False))


if __name__ == "__main__":
    run()
