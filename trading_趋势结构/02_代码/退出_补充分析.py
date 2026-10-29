"""补充分析：年份等权口径 + 假设失效状态的区分力。

年份等权是事后加的稳健性口径：预注册的"平均R"按交易笔数加权，而候选信号数
在 2014-2015 年占了一半，会让少数年份主导结论。两个口径都报，不择优。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "03_结果")

from 持仓状态机 import ARMS  # noqa: E402
from 退出_数据层 import Panel  # noqa: E402
from 退出_回测引擎 import BTConfig, build_signals, run_trades_multi  # noqa: E402

ARM_ORDER = ["B0", "S_fix", "S_struct", "T2", "T3", "T45", "T6", "T_struct",
             "BE", "TS10", "TS20", "TS40", "SC", "TG2", "SE20", "MX60", "STK", "ALL"]


def year_equal():
    rows = []
    for tb in ("TB1", "TB2"):
        d = pd.read_csv(os.path.join(OUT, f"退出_分年_{tb}.csv"))
        for w, yrs in (("dev", range(2008, 2016)), ("val", range(2016, 2021))):
            x = d[d.year.isin(yrs)]
            piv = x.pivot_table(index="arm", columns="year", values="avg_R")
            for arm in ARM_ORDER:
                if arm not in piv.index:
                    continue
                rows.append(dict(testbed=tb, window=w, arm=arm,
                                 years_equal_avg_R=float(piv.loc[arm].mean()),
                                 years_used=int(piv.loc[arm].notna().sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "退出_年份等权.csv"), index=False)
    for tb in ("TB1", "TB2"):
        for w in ("dev", "val"):
            s = df[(df.testbed == tb) & (df.window == w)].sort_values(
                "years_equal_avg_R", ascending=False)
            print(f"{tb} {w}: " + ", ".join(f"{r.arm}={r.years_equal_avg_R:+.3f}"
                                            for r in s.itertuples()))


def state_mx60():
    """用 MX60（会触发"趋势失效"）做状态日志，补齐"假设失效"状态的样本。"""
    p = Panel()
    sig = build_signals(p, "TB1", cache_dir=os.path.join(ROOT, "01_数据", "cache"))
    cfg = BTConfig(start="2008-01-01", end="2020-12-31", testbed="TB1")
    log: list = []
    run_trades_multi(p, sig, cfg, {"MX60": ARMS["MX60"]},
                     state_log_arm="MX60", state_log=log)
    d = pd.DataFrame([dict(state=a, fwd5=b, window="dev" if c else "val",
                           board="创业板" if e else "主板") for a, b, c, e in log])
    d.to_csv(os.path.join(OUT, "退出_状态标签_TB1_MX60.csv.gz"), index=False, compression="gzip")
    g = d.groupby(["window", "state"]).fwd5.agg(["size", "mean", "median"])
    g.to_csv(os.path.join(OUT, "退出_状态区分力_MX60.csv"))
    print(g.round(4).to_string())


if __name__ == "__main__":
    year_equal()
    state_mx60()
