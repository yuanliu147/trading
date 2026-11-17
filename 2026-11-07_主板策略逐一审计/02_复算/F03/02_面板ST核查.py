# -*- coding: utf-8 -*-
"""F03 补充核查：用冻结主板面板(只读parquet)对交付逐笔做点时ST/停牌/价格抽查。"""
import os
import pandas as pd
import duckdb

ROOT = "/Users/admin/WorkSpace/trading"
OUT = os.path.join(ROOT, "2026-11-07_主板策略逐一审计/02_复算/F03")
PANEL = os.path.join(ROOT, "trading_finally/data/main_panel.parquet")
con = duckdb.connect()

TR = [
    ("F2", "2026-10-23_主板专用_策略优化/13_F2_逐笔.csv"),
    ("40核心_实际价成交", "2026-10-24_结构交易_主板40系统/05_证据/核心_实际价成交_逐笔.csv"),
    ("40进攻_实际价成交", "2026-10-24_结构交易_主板40系统/05_证据/进攻_实际价成交_逐笔.csv"),
    ("进攻版R1", "2026-10-24_进攻版_2016_2026_回测与优化/03_结果/R1_进攻版_逐笔.csv"),
]
rows, detail = [], []
for name, rel in TR:
    df = pd.read_csv(os.path.join(ROOT, rel), encoding="utf-8-sig")
    pairs = df[["code", "entry_date"]].drop_duplicates()
    vals = ",".join(f"('{r.code}','{r.entry_date}')" for r in pairs.itertuples())
    pan = con.sql(f"""SELECT code,date,open,close,prev_close,trade_status,is_st,data_valid
                      FROM '{PANEL}' WHERE (code,date) IN ({vals})""").df()
    m = df.merge(pan, left_on=["code", "entry_date"], right_on=["code", "date"], how="left")
    st = m[m["is_st"] == 1]
    susp = m[m["trade_status"] == 0]
    missing = m[m["date"].isna()]
    rows.append(dict(name=name, n=len(m), st_entries=len(st), st_pnl=st["pnl"].sum(),
                     suspended_entries=len(susp), panel_missing=len(missing),
                     invalid_rows=int((m["data_valid"] == False).sum())))
    for _, t in st.iterrows():
        detail.append(dict(name=name, code=t["code"], entry_date=t["entry_date"],
                           pnl=t["pnl"], is_st=1))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "F03_面板ST停牌核查.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(detail).to_csv(os.path.join(OUT, "F03_面板ST交易明细.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).to_string(index=False))
print("\nST入场明细:")
print(pd.DataFrame(detail).to_string(index=False))