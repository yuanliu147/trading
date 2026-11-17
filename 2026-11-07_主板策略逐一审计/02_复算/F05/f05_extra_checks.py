# -*- coding: utf-8 -*-
"""F05 附加核验：低波活跃过滤权益、600056关键单笔、缓存池规模、中证500基准双源核对。只读。"""
import os
import pickle

import pandas as pd

SRC = "/Users/admin/WorkSpace/trading/八系统回测"
OUT = os.path.dirname(os.path.abspath(__file__))
lines = []


def log(s):
    print(s, flush=True)
    lines.append(str(s))


eq = pd.read_csv(os.path.join(SRC, "结果_历史成分/扩展2020/股性过滤/权益_原版三法取二_只做低波活跃型.csv"),
                 parse_dates=["date"]).set_index("date").equity
log(f"低波活跃过滤2020-2025: total={eq.iloc[-1]/eq.iloc[0]-1:+.4%} MDD={(eq/eq.cummax()-1).min():.4%} (交付 +84.18% / -37.19%)")

tr = pd.read_csv(os.path.join(SRC, "结果_历史成分/交易_历史成分_波段为王2.csv"))
t = tr.loc[tr.pnl.idxmax()]
log(f"波段为王2最大单笔: {t.code} {t.buy}→{t.sell} pnl={t.pnl:,.0f} ret={t.ret:+.1%} reason={t.reason} (报告 600056 +27.21万 +135.9%)")

with open(os.path.join(SRC, "daily_cache_hist.pkl"), "rb") as f:
    cache = pickle.load(f)
n = len(cache["stocks"])
ok = sum(len(v) >= 300 for v in cache["stocks"].values())
log(f"缓存股票 {n} 只, ≥300根日线 {ok} 只 (报告: 并集893, 891有效; 缓存仅存有效891)")

ip = pd.read_parquet("/Users/admin/WorkSpace/trading/trading_finally/data/index_panel.parquet")
z = ip[ip.code == "000905.SH"].copy()
z["date"] = pd.to_datetime(z.date)
z = z[(z.date >= "2022-01-01") & (z.date <= "2025-12-31")].sort_values("date").set_index("date").close
zz = pd.read_pickle(os.path.join(SRC, "zz500_index.pkl"))
zz["date"] = pd.to_datetime(zz["date"])
zz = zz[(zz.date >= "2022-01-01") & (zz.date <= "2025-12-31")].set_index("date").close
common = z.index.intersection(zz.index)
d = (z.loc[common] - zz.loc[common]).abs()
y = z.groupby(z.index.year).last()
yr = y / y.shift(1) - 1
yr.iloc[0] = y.iloc[0] / z.iloc[0] - 1
log(f"中证500基准(冻结指数面板): total={z.iloc[-1]/z.iloc[0]-1:+.4%} MDD={(z/z.cummax()-1).min():.2%} "
    f"年度=" + " ".join(f"{k}:{v:+.2%}" for k, v in yr.items()))
log(f"交付缓存基准 total={zz.iloc[-1]/zz.iloc[0]-1:+.4%}; 共同日 {len(common)}, 收盘最大绝对差 {d.max()}")

with open(os.path.join(OUT, "f05_extra_checks_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")