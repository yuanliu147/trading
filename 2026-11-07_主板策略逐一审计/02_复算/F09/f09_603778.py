# -*- coding: utf-8 -*-
"""F09 / 603778 个案抽查：baostock 单股只读查询，复算换手率区间统计。
不复跑原脚本、不写原目录；输出仅写本目录。运行: python3 f09_603778.py
"""
import os
import sqlite3
import numpy as np
import pandas as pd

ROOT = "/Users/admin/WorkSpace/trading"
OUT = os.path.dirname(os.path.abspath(__file__))
BAO = "/Users/admin/WorkSpace/baostock_data/data/baostock.db"
REF = os.path.join(ROOT, "2026-11-03_603778_换手率区间/03_结果")
CUTOFF = "2026-10-29"  # 与原研究 baostock 窗口对齐（原研究另补同花顺两根，不入历史统计）

con = sqlite3.connect(f"file:{BAO}?mode=ro", uri=True)
raw = pd.read_sql(
    """select date,open,high,low,close,volume,amount,turn,pct_chg
       from stock_kline_d
       where code='sh.603778' and adjustflag='3' and date<=? order by date""",
    con, params=[CUTOFF], parse_dates=["date"])
hfq = pd.read_sql(
    """select date,open,close from stock_kline_d
       where code='sh.603778' and adjustflag='1' and date<=? order by date""",
    con, params=[CUTOFF], parse_dates=["date"])
con.close()
df = raw.merge(hfq.rename(columns={"open": "open_h", "close": "close_h"}), on="date", how="left")
df = df.dropna(subset=["turn"]).reset_index(drop=True)
print(f"603778 有效交易日(turn非空, <= {CUTOFF}): {len(df)}  原称 2591")

# 与原研究同口径：T+1 开盘买，T+n 收盘卖（后复权），未扣费
o1 = df.open_h.shift(-1)
for n in (20, 60):
    df[f"f{n}"] = (df.close_h.shift(-n) / o1 - 1) * 100
df["year"] = df.date.dt.year
df["hi250"] = df.high.rolling(250).max()  # 仅用于 dd250 近似（原用后复权high，此处补查）
# 补后复权 high/low 以复现 dd250/pos250
con = sqlite3.connect(f"file:{BAO}?mode=ro", uri=True)
hl = pd.read_sql("""select date,high,low from stock_kline_d
                    where code='sh.603778' and adjustflag='1' and date<=? order by date""",
                 con, params=[CUTOFF], parse_dates=["date"])
con.close()
hlm = hl.set_index("date")
hh = df.date.map(hlm.high); ll = df.date.map(hlm.low)
df["hi250"] = hh.rolling(250).max()
df["dd250"] = (df.close_h / df.hi250 - 1) * 100

BINS = [0, 0.5, 1, 2, 3.5, 5, 8, 12, 20, 1e9]
LABELS = ["<0.5", "0.5-1", "1-2", "2-3.5", "3.5-5", "5-8", "8-12", "12-20", ">20"]
df["band"] = pd.cut(df.turn, bins=BINS, labels=LABELS, right=False)

g = df.groupby("band", observed=True)
tab = pd.DataFrame({
    "天数": g.size(),
    "换手中位": g.turn.median().round(2),
    "T20中位": g.f20.median().round(2),
    "T20均值": g.f20.mean().round(2),
    # 原脚本口径：(s>0).mean()，NaN(末尾无未来收益)计入分母且视为非胜；此处照抄以便逐格对账
    "T20胜率": (g.f20.apply(lambda s: (s > 0).mean()) * 100).round(0),
})
ref = pd.read_csv(os.path.join(REF, "换手区间统计.csv")).set_index("区间(%)")
tab["原天数"] = ref["天数"]
tab["原T20中位"] = ref["T+20中位(%)"]
tab["原T20胜率"] = ref["T+20胜率(%)"]
print("\n== 九区间复算 vs 原结果 ==")
print(tab.to_string())
tab.to_csv(os.path.join(OUT, "603778_区间复算.csv"))

rows = []


def rec(item, claimed, actual, ok, note=""):
    rows.append(dict(项=item, 原称=claimed, 复核=actual, 判定=ok, 备注=note))
    print(f"[603778] {item}: 原称={claimed} 复核={actual} {ok} {note}")


rec("有效交易日", "2591", len(df), "一致" if len(df) == 2591 else "偏差")
match = (tab["天数"] == tab["原天数"]).all() and np.allclose(tab["T20中位"], tab["原T20中位"], atol=0.05)
rec("九区间天数/T+20中位", "见一页纸表", "逐格对比见CSV", "一致" if match else "偏差")

# 5-8% 剔除2026
b = df[(df.band == "5-8") & (df.year != 2026)]
b20 = b.f20.dropna(); b60 = b.f60.dropna()
rec("5-8%带剔除2026 样本天数", "239", len(b), "一致" if len(b) == 239 else "偏差")
rec("5-8%带剔除2026 T+20中位", "-2.4%", f"{b20.median():+.1f}%",
    "一致" if abs(b20.median() + 2.4) < 0.15 else "偏差")
rec("5-8%带剔除2026 T+60中位/胜率", "-12.5% / 29%",
    f"{b60.median():+.1f}% / {(b60>0).mean():.0%}",
    "一致" if abs(b60.median() + 12.5) < 0.3 and abs((b60 > 0).mean() - 0.29) < 0.02 else "偏差")

# 6.4% 换手的全历史分位
pct = (df.turn <= 6.4).mean() * 100
rec("换手6.4%全历史分位", "73.5%", f"{pct:.1f}%",
    "一致" if abs(pct - 73.5) < 1.0 else "偏差", "以 baostock<=10-29 样本计")

# 深度回撤>60% + 换手5-8%：2016-2025 仅2天
deep = df[(df.dd250 < -60) & (df.turn >= 5) & (df.turn < 8)]
dold = deep[deep.year <= 2025]
rec("深回撤>60%+换手5-8% 2016-2025天数", "2", len(dold),
    "一致" if len(dold) == 2 else "偏差", f"日期={list(dold.date.dt.strftime('%Y-%m-%d'))}; 2026年={len(deep)-len(dold)}天")

# <1% 唯一正期望带 / 2-8% 负期望带
lt1 = df[df.turn < 1].f20.dropna()
b28 = df[(df.turn >= 2) & (df.turn < 8)].f20.dropna()
rec("<1%带 T+20中位", "正(+1.6~+2.9%)", f"{lt1.median():+.2f}%",
    "一致" if lt1.median() > 0 else "偏差")
rec("2-8%带 T+20中位", "负", f"{b28.median():+.2f}%",
    "一致" if b28.median() < 0 else "偏差")

# 冻结主板面板交叉：603778 实际收盘价（面板无 turn 列，换手只能来自 baostock）
panel = pd.read_parquet(os.path.join(ROOT, "trading_finally/data/main_panel.parquet"),
                        columns=["date", "code", "close", "is_st"],
                        filters=[("code", "=", "603778.SH")])
panel["date"] = pd.to_datetime(panel["date"])
j = panel.merge(df[["date", "close"]], on="date", suffixes=("_panel", "_bao"))
diff = (j["close_panel"] - j["close_bao"]).abs()
rec("主板面板 vs baostock 实际收盘价", "应一致",
    f"重叠{len(j)}天, 最大差{diff.max():.4f}, 不一致{(diff>0.005).sum()}天",
    "一致" if (diff <= 0.005).all() else "偏差", "面板确无turn列，换手率字段以baostock为准")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "603778_复核结论.csv"), index=False)
print(f"\n共 {len(rows)} 项，已写 603778_复核结论.csv / 603778_区间复算.csv")