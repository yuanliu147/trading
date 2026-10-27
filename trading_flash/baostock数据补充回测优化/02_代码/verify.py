#!/usr/bin/env python3
"""独立校验：从权益、逐笔、面板和原始库重算关键结论。"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "03_结果"
E = ROOT / "05_证据"
PANEL = ROOT / "01_数据" / "panel.parquet"
BARS = ROOT / "01_数据" / "bars.parquet"
OLD_PANEL = Path("/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/01_数据/research_panel.parquet")
OLD_TRADES = Path("/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/03_结果/实际价_修正版_逐笔.csv")
HITHINK = Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb")

checks = []


def check(name: str, ok: bool, detail: str):
    checks.append(dict(name=name, ok=bool(ok), detail=detail))
    print(("PASS" if ok else "FAIL"), name, "-", detail)


def metrics_from(eq: pd.DataFrame, tr: pd.DataFrame) -> dict:
    e = eq.set_index("date").equity.astype(float)
    e0 = e.iloc[0]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e0) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    return dict(cagr=cagr, dd=dd, final=e.iloc[-1], n=len(tr),
                start=e.index[0], end=e.index[-1])


def verify_result(name: str, folder: Path, prefix: str):
    eq = pd.read_csv(folder / f"{prefix}_每日权益.csv", parse_dates=["date"])
    tr = pd.read_csv(folder / f"{prefix}_逐笔.csv", parse_dates=["signal_date", "entry_date", "exit_date"])
    met = json.loads((folder / f"{prefix}_指标.json").read_text())
    m = metrics_from(eq, tr)
    check(f"{name}_权益起点10万", abs(eq.equity.iloc[0] - 100000) < 1e-6, f"{eq.equity.iloc[0]:.2f}")
    check(f"{name}_无负现金", eq.cash.min() > -1e-6, f"min cash {eq.cash.min():.6f}")
    check(f"{name}_年化一致", abs(m["cagr"] - met["cagr"]) < 2e-6,
          f"{m['cagr']:.6%} vs {met['cagr']:.6%}")
    check(f"{name}_回撤一致", abs(m["dd"] - met["max_dd"]) < 2e-6,
          f"{m['dd']:.6%} vs {met['max_dd']:.6%}")
    check(f"{name}_交易数一致", m["n"] == met["n_trades"], f"{m['n']} vs {met['n_trades']}")
    pnl = (tr.proceeds - tr.invested).sum()
    check(f"{name}_逐笔盈亏与权益一致", abs(pnl + 100000 - m["final"]) < 0.05,
          f"{pnl+100000:.2f} vs {m['final']:.2f}")
    check(f"{name}_无异常收益率", bool(((tr.ret > -1) & (tr.ret < 1.01)).all()),
          f"min {tr.ret.min():.3f} max {tr.ret.max():.3f}")
    # T+3：信号日到买入日正好 3 个交易日
    con = duckdb.connect()
    dates = [pd.Timestamp(x) for x in con.execute(
        f"SELECT DISTINCT date FROM read_parquet('{BARS}') ORDER BY date").fetchdf().date]
    con.close()
    di = {d: i for i, d in enumerate(dates)}
    lag = [(di.get(pd.Timestamp(r.entry_date), np.nan) - di.get(pd.Timestamp(r.signal_date), np.nan))
           for r in tr.itertuples()]
    check(f"{name}_全部T+3买入", bool((np.array(lag) == 3).all()),
          f"lag set {sorted(set(lag))}")
    check(f"{name}_信号日早于买入日", bool((tr.entry_date > tr.signal_date).all()), "ok")
    return eq, tr, met


def main():
    pairs = [
        ("A_2017_2026", R / "2017_2026", "A_仅当前上市"),
        ("B_2017_2026", R / "2017_2026", "B_含退市股"),
        ("C_2017_2026", R / "指标组合_2017_2026", "C_排除ST"),
        ("A_2005_2026", R / "2005_2026", "A_仅当前上市"),
        ("B_2005_2026", R / "2005_2026", "B_含退市股"),
        ("C_2005_2026", R / "指标优化/2005-2026", "C_排除ST"),
        ("C_止损15_2017_2026", R / "风险叠加/2017-2026", "C_止损15"),
        ("C_止损25_2005_2026", R / "风险叠加/2005-2026", "C_止损25"),
        ("C_止损25仓位35_2005_2026", R / "风险叠加/2005-2026", "C_止损25_仓位35"),
    ]
    data = {}
    for name, folder, prefix in pairs:
        data[name] = verify_result(name, folder, prefix)

    # 2026-10-23 信号数与前二
    con = duckdb.connect()
    q = f"""
    SELECT count(*) n FROM read_parquet('{PANEL}')
    WHERE date='2026-10-23' AND listing_session>=250 AND amount20>=1e7
     AND drawdown250<-0.45 AND return_250d>0 AND rsi14<35 AND close_location>0.45
     AND market_return20_listed<-0.05
    """
    n = con.execute(q).fetchone()[0]
    check("2026-10-23信号数=123", n == 123, str(n))
    top = con.execute(f"""
    SELECT code, drawdown250 FROM read_parquet('{PANEL}')
    WHERE date='2026-10-23' AND listing_session>=250 AND amount20>=1e7
     AND drawdown250<-0.45 AND return_250d>0 AND rsi14<35 AND close_location>0.45
     AND market_return20_listed<-0.05 ORDER BY drawdown250, code LIMIT 2
    """).fetchdf()
    check("2026-10-23前二", list(top.code) == ["sh.603778", "sh.605376"], str(list(top.code)))

    # 同花顺校验：不复权价逐日一致
    h = duckdb.connect(str(HITHINK), read_only=True)
    old = h.execute("""SELECT thscode,date,close,open FROM v_daily WHERE date='2026-10-23'""").fetchdf()
    h.close()
    new = con.execute(f"""SELECT code,date,\"close\",\"open\" FROM read_parquet('{PANEL}') WHERE date='2026-10-23'""").fetchdf()
    new["thscode"] = new.code.str[3:] + "." + new.code.str[:2].str.upper()
    mm = new.merge(old, on="thscode", suffixes=("_bs", "_th"))
    check("同花顺不复权收盘逐只一致", len(mm) > 3100 and (mm.close_bs - mm.close_th).abs().max() == 0,
          f"n={len(mm)} maxdiff={(mm.close_bs-mm.close_th).abs().max()}")
    check("同花顺不复权开盘逐只一致", (mm.open_bs - mm.open_th).abs().max() == 0,
          f"maxdiff={(mm.open_bs-mm.open_th).abs().max()}")

    # 同花顺后复权收益一致（603778）
    b = con.execute(f"""SELECT date,hfq_close FROM read_parquet('{BARS}')
                       WHERE code='sh.603778' AND date>='2025-06-03' ORDER BY date""").fetchdf()
    h = duckdb.connect(str(HITHINK), read_only=True)
    o = h.execute("""SELECT date,close FROM v_daily_hfq WHERE thscode='603778.SH' AND date>='2025-06-03' ORDER BY date""").fetchdf()
    h.close()
    b["date"] = pd.to_datetime(b.date); o["date"] = pd.to_datetime(o.date)
    z = b.merge(o, on="date", suffixes=("_bs", "_th"))
    d = (z.hfq_close.pct_change() - z.close.pct_change()).abs().max()
    check("603778后复权日收益与同花顺一致", d < 1e-12, f"maxdiff={d:.3e}")

    # 特征差异
    oldp = duckdb.connect()
    op = oldp.execute(f"""SELECT thscode,drawdown250,return_250d,rsi14,amount20
        FROM read_parquet('{OLD_PANEL}') WHERE date='2026-10-23'""").fetchdf()
    oldp.close()
    npn = con.execute(f"""SELECT code,drawdown250,return_250d,rsi14,amount20
        FROM read_parquet('{PANEL}') WHERE date='2026-10-23'""").fetchdf()
    npn["thscode"] = npn.code.str[3:] + "." + npn.code.str[:2].str.upper()
    fm = npn.merge(op, on="thscode", suffixes=("_bs", "_th"))
    for col, tol in [("drawdown250", 0.01), ("return_250d", 0.01),
                     ("rsi14", 0.10), ("amount20", 0.01)]:
        diff = (fm[col + "_bs"] - fm[col + "_th"]).abs()
        check(f"特征_{col}_p99<{tol}", diff.quantile(0.99) < tol,
              f"median={diff.median():.2e} p99={diff.quantile(0.99):.2e} max={diff.max():.2e}")
    # 信号名单逐只一致
    oldsig = oldp_signals = None
    olds = duckdb.connect()
    so = olds.execute(f"""SELECT thscode FROM read_parquet('{OLD_PANEL}')
        WHERE date='2026-10-23' AND amount20>=1e7 AND drawdown250<-0.45 AND return_250d>0
          AND rsi14<35 AND close_location>0.45 AND market_return20<-0.05""").fetchdf()
    olds.close()
    sn = con.execute(f"""SELECT code FROM read_parquet('{PANEL}')
        WHERE date='2026-10-23' AND listing_session>=250 AND amount20>=1e7 AND drawdown250<-0.45
          AND return_250d>0 AND rsi14<35 AND close_location>0.45 AND market_return20_listed<-0.05""").fetchdf()
    sn["thscode"] = sn.code.str[3:] + "." + sn.code.str[:2].str.upper()
    check("2026-10-23信号名单逐只一致", set(so.thscode) == set(sn.thscode),
          f"old={len(set(so.thscode))} new={len(set(sn.thscode))} overlap={len(set(so.thscode)&set(sn.thscode))}")

    # 指数点位
    con2 = duckdb.connect()
    con2.execute("ATTACH '/Users/admin/WorkSpace/baostock_data/data/baostock.db' AS b (TYPE SQLITE, READ_ONLY);")
    idx = con2.execute("""SELECT date,close FROM b.index_kline_d WHERE code='sh.000001'
        AND date IN ('2007-10-16','2008-10-28','2015-06-12','2016-01-27') ORDER BY date""").fetchdf()
    con2.close()
    known = {"2007-10-16": 6092.057, "2008-10-28": 1771.821,
             "2015-06-12": 5166.35, "2016-01-27": 2735.558}
    ok = all(abs(float(r.close) - known[r.date]) < 0.01 for r in idx.itertuples())
    check("上证指数关键点位", ok, idx.to_dict("records"))

    # 退市覆盖
    st = pd.read_csv(E / "主板股票清单.csv")
    bars = con.execute(f"SELECT code,min(date) first_date,max(date) last_date FROM read_parquet('{BARS}') GROUP BY code").fetchdf()
    mm2 = st.merge(bars, on="code", how="left")
    miss = mm2[mm2.first_date.isna()]
    check("主板股票数3489", len(st) == 3489, str(len(st)))
    check("退市股295", int(st.is_delisted.sum()) == 295, str(int(st.is_delisted.sum())))
    check("有数据退市股280", len(mm2[(mm2.is_delisted == 1) & mm2.first_date.notna()]) == 280,
          str(len(mm2[(mm2.is_delisted == 1) & mm2.first_date.notna()])))
    check("缺失15只均在2004年前退市", len(miss) == 15 and (pd.to_datetime(miss.out_date) < "2004-01-01").all(),
          ",".join(miss.code))

    # 复权修复
    rep = pd.read_csv(E / "复权修复事件.csv")
    check("复权修复16行/14只", len(rep) == 16 and rep.code.nunique() == 14,
          f"{len(rep)}/{rep.code.nunique()}")

    # 002684
    btr = pd.read_csv(R / "2017_2026" / "B_含退市股_逐笔.csv", parse_dates=["entry_date", "exit_date"])
    x = btr[btr.code == "sz.002684"]
    check("002684在B中存在", len(x) == 1, str(len(x)))
    if len(x):
        r = x.iloc[0]
        check("002684退市整理首日开盘卖出", str(r.exit_date.date()) == "2022-06-06", str(r.exit_date.date()))
        check("002684亏损约-83%", abs(r.ret + 0.82943) < 0.001, f"{r.ret:.4f}")
    ctr = pd.read_csv(R / "指标组合_2017_2026" / "C_排除ST_逐笔.csv")
    check("002684不在C中", not (ctr.code == "sz.002684").any(), "ok")

    # Top2 重叠
    ov = pd.read_csv(R / "Top2重叠.csv")
    row = ov[(ov["期间"] == "2017-2026") & (ov["对比"] == "A vs B")].iloc[0]
    check("A/B Top2 Jaccard=0.8625", abs(row.Jaccard - 0.862475) < 1e-5, f"{row.Jaccard:.6f}")
    row = ov[(ov["期间"] == "2017-2026") & (ov["对比"] == "A vs C")].iloc[0]
    check("A/C Top2 Jaccard=0.8050", abs(row.Jaccard - 0.805019) < 1e-5, f"{row.Jaccard:.6f}")

    # 止损
    s15 = pd.read_csv(R / "风险叠加/2017-2026" / "C_止损15_逐笔.csv")
    stop15 = s15[s15.exit_reason.str.startswith("stop")]
    check("止损15有成交且均值约-15%", len(stop15) > 0 and abs(stop15.ret.mean() + 0.152) < 0.01,
          f"n={len(stop15)} avg={stop15.ret.mean():.4f}")
    s25 = pd.read_csv(R / "风险叠加/2005-2026" / "C_止损25_逐笔.csv")
    stop25 = s25[s25.exit_reason.str.startswith("stop")]
    check("止损25有成交且均值约-25%", len(stop25) > 0 and abs(stop25.ret.mean() + 0.252) < 0.01,
          f"n={len(stop25)} avg={stop25.ret.mean():.4f}")

    # 旧官方逐笔对照
    old_tr = pd.read_csv(OLD_TRADES)
    old_tr["code"] = old_tr.thscode.str[7:].str.lower() + "." + old_tr.thscode.str[:6]
    old_tr["entry_date"] = pd.to_datetime(old_tr.entry_date).dt.date
    atr = pd.read_csv(R / "2017_2026" / "A_仅当前上市_逐笔.csv")
    atr["entry_date"] = pd.to_datetime(atr.entry_date).dt.date
    both = old_tr.merge(atr, on=["entry_date", "code"], suffixes=("_old", "_new"))
    check("旧官方99笔中94笔同日同码复现", len(both) == 94, str(len(both)))
    diff = (both.ret_new - both.ret_old).abs()
    check("94笔中89笔收益差<0.15pp", int((diff < 0.0015).sum()) == 89,
          f"n<0.15pp={int((diff<0.0015).sum())}, median={diff.median():.6f}, max={diff.max():.4f}")

    # 保存
    out = pd.DataFrame(checks)
    out.to_csv(E / "校验报告.csv", index=False, encoding="utf-8-sig")
    summary = dict(total=len(checks), passed=int(out.ok.sum()), failed=int((~out.ok).sum()),
                   checks=checks)
    (E / "校验报告.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# 独立校验报告\n", f"通过 {summary['passed']}/{summary['total']}\n",
             "| 检查 | 结果 | 细节 |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['name']} | {'通过' if c['ok'] else '失败'} | {c['detail']} |")
    (E / "校验报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{summary['passed']}/{summary['total']} passed")


if __name__ == "__main__":
    main()
