"""Read-only comparison of recovered B signals against the original mother panel."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/admin/WorkSpace/trading")
PICKLE = BASE / "2026-10-20_涨幅趋势_全维度定制/01_数据/全A面板_2016_2026_含退市.pkl"
OURS = BASE / "trading_finally/results/legacy_b_signals.parquet"
MAIN = BASE / "trading_finally/data/main_panel.parquet"
OUT = BASE / "trading_finally/.sdd-legacy/b_signal_comparison.json"

P = pickle.load(open(PICKLE, "rb"))
dates = pd.DatetimeIndex(P["meta"]["dates"])
codes = list(P["meta"]["codes"])
U = P["up_struct"].values
AMT = P["amt20"].values
C = P["close"].values
AGE = P["trend_age"].values
ATR = P["atr_pct"].values
DD60 = P["dd60"].values
LU = P["lu_count20"].values
SBR = P["sector_breadth20"].values
SR20 = P["sector_ret20"].values
MB = P["market_breadth20"].values.reshape(-1, 1)
date_ok = (dates >= pd.Timestamp("2017-10-13"))[:, None]
mother = (
    U & (AMT >= 3e7) & (C >= 2) & (AGE >= 5) & (AGE <= 30)
    & (ATR <= 0.044) & (SBR >= 0.60) & (SR20 > 0.05) & (DD60 > -0.08)
    & (LU == 0) & (MB > 0.50) & date_ok
)
mother_keys = set()
ri, ci = np.argwhere(mother).T
for r, c in zip(ri, ci):
    mother_keys.add((dates[r].strftime("%Y-%m-%d"), codes[c]))

ours = pd.read_parquet(OURS, columns=["date", "code", "entry_signal"])
our_keys = set(zip(ours.loc[ours["entry_signal"], "date"], ours.loc[ours["entry_signal"], "code"]))
main_codes = set(pd.read_parquet(MAIN, columns=["code"])["code"].unique())
mother_mainboard = {key for key in mother_keys if key[1] in main_codes}
mother_non_mainboard = mother_keys - mother_mainboard
mainboard_common = mother_mainboard & our_keys
common = mother_keys & our_keys
report = {
    "mother_keys": len(mother_keys),
    "mother_mainboard": len(mother_mainboard),
    "mother_non_mainboard": len(mother_non_mainboard),
    "our_keys": len(our_keys),
    "common": len(common),
    "mainboard_common": len(mainboard_common),
    "mainboard_reproduce_share": len(mainboard_common) / max(len(mother_mainboard), 1),
    "mother_only": len(mother_keys - our_keys),
    "our_only": len(our_keys - mother_keys),
    "jaccard": len(common) / max(len(mother_keys | our_keys), 1),
    "mother_window": [str(dates.min().date()), str(dates.max().date())],
}
OUT.write_text(pd.Series(report).to_json(force_ascii=False, indent=2) + "\n", encoding="utf-8")
print(report)
