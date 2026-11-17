# -*- coding: utf-8 -*-
"""
F05 补充：
1) 信号层主板子集（20日前瞻超额，按代码前缀过滤）；
2) 冻结主板面板对八系统宇宙的覆盖评估（是否值得补跑，不实际补跑）；
3) 双源价格交叉验证：八系统缓存(qfq收盘) vs 冻结面板 signal_close(后复权) 的日收益一致性。
只读源数据；输出写本目录。
"""
import os
import pickle

import numpy as np
import pandas as pd

SRC = "/Users/admin/WorkSpace/trading/八系统回测"
HIST = os.path.join(SRC, "结果_历史成分")
PANEL = "/Users/admin/WorkSpace/trading/trading_finally/data/main_panel.parquet"
OUT = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = ["MACD入门", "MACD波段", "波段为王1", "波段为王2", "看盘", "短线30技", "裸K", "KDJ"]

log_lines = []


def log(s=""):
    print(s, flush=True)
    log_lines.append(str(s))


def is_main(code):
    return code.startswith("sh.60") or code.startswith("sz.00")


def to_panel_code(code):
    ex, num = code.split(".")
    return f"{num}.{ex.upper()}"


def part_signal_main():
    log("=" * 70)
    log("A. 信号层主板子集：20日市场调整后超额（扣0.30%成本）")
    sig = pd.read_csv(os.path.join(HIST, "组合", "信号_前瞻收益.csv"))
    sig["main"] = sig.code.map(is_main)
    rows = []
    for s in SYSTEMS:
        for flag, sub in [("全部", sig[sig.system == s]), ("主板", sig[(sig.system == s) & sig.main])]:
            x = sub.ex20.dropna()
            rows.append(dict(system=s, subset=flag, n=len(x), mean=x.mean(),
                             median=x.median(), win=(x > 0).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "PA_信号层_主板子集.csv"), index=False, encoding="utf-8-sig")
    for s in SYSTEMS:
        a = df[(df.system == s) & (df.subset == "全部")].iloc[0]
        m = df[(df.system == s) & (df.subset == "主板")].iloc[0]
        log(f"{s:8s} 全部 n={int(a.n)} 均值={a['mean']:+.3%} | 主板 n={int(m.n)} 均值={m['mean']:+.3%} 中位={m['median']:+.3%} 胜率={m['win']:.1%}")
    x = sig[sig.main].ex20.dropna()
    log(f"八系统主板信号合计: n={len(x)} 均值={x.mean():+.3%} 中位={x.median():+.3%}")


def part_panel():
    log("=" * 70)
    log("B. 冻结主板面板覆盖评估")
    snaps = {}
    with open(os.path.join(SRC, "hist_constituents.txt")) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) > 1:
                snaps[p[0]] = set(p[1:])
    union = set().union(*snaps.values())
    umb = sorted(c for c in union if is_main(c))
    panel_codes = {to_panel_code(c) for c in umb}
    panel = pd.read_parquet(PANEL, columns=["code", "date", "signal_close", "is_st", "data_valid"])
    panel = panel[panel.code.isin(panel_codes)]
    panel["date"] = pd.to_datetime(panel["date"])
    win = panel[(panel.date >= "2021-01-01") & (panel.date <= "2025-12-31")]
    covered = set(win.code.unique())
    missing = panel_codes - covered
    log(f"主板并集 {len(umb)} 只; 冻结面板覆盖 {len(covered)} 只; 缺失 {len(missing)} 只: {sorted(missing)[:10]}")
    log(f"2021-2025 窗口行数 {len(win):,}; data_valid=False {int((~win.data_valid).sum()):,}; ST 行 {int(win.is_st.sum()):,}")
    per_year = win.groupby(win.date.dt.year).code.nunique()
    log("逐年覆盖股票数: " + " ".join(f"{y}:{n}" for y, n in per_year.items()))
    pd.DataFrame([dict(item="主板并集", n=len(umb)), dict(item="面板覆盖", n=len(covered)),
                  dict(item="缺失", n=len(missing))]).to_csv(
        os.path.join(OUT, "PB_面板覆盖.csv"), index=False, encoding="utf-8-sig")

    log("=" * 70)
    log("C. 双源价格交叉验证：缓存qfq日收益 vs 面板后复权日收益（2022-2025）")
    with open(os.path.join(SRC, "daily_cache_hist.pkl"), "rb") as f:
        cache = pickle.load(f)
    # 抽样：主板股票中取交易最频繁的前20只
    allt = pd.read_csv(os.path.join(HIST, "全部交易_历史成分.csv"))
    mb_tr = allt[allt.code.map(is_main)]
    sample = mb_tr.code.value_counts().head(20).index.tolist()
    panel_idx = panel.set_index(["code", "date"]).signal_close.sort_index()
    rows = []
    for code in sample:
        raw = cache["stocks"].get(code)
        if raw is None:
            continue
        c = raw.copy()
        c["date"] = pd.to_datetime(c["date"])
        c = c[(c.date >= "2022-01-01") & (c.date <= "2025-12-31")].set_index("date").close.astype(float)
        r_cache = c.pct_change().dropna()
        pc = to_panel_code(code)
        p = panel_idx.loc[pc] if pc in panel_idx.index.get_level_values(0) else None
        if p is None:
            continue
        p = p[(p.index >= "2022-01-01") & (p.index <= "2025-12-31")]
        r_panel = p.pct_change().dropna()
        common = r_cache.index.intersection(r_panel.index)
        d = (r_cache.loc[common] - r_panel.loc[common]).abs()
        rows.append(dict(code=code, days=len(common), match_1e4=(d < 1e-4).mean(),
                         match_1e3=(d < 1e-3).mean(), max_abs=d.max(), median_abs=d.median()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "PC_双源日收益对比.csv"), index=False, encoding="utf-8-sig")
    for r in df.itertuples():
        log(f"{r.code}: 共同日 {r.days}, |Δ|<1e-4 占比 {r.match_1e4:.2%}, <1e-3 占比 {r.match_1e3:.2%}, 最大差 {r.max_abs:.4f}")
    log(f"合计: {len(df)} 只, 平均精确匹配率 {df.match_1e4.mean():.2%}, 平均1e-3匹配率 {df.match_1e3.mean():.2%}")


def main():
    part_signal_main()
    part_panel()
    with open(os.path.join(OUT, "f05_panel_check_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()