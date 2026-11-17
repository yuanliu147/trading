#!/usr/bin/env python3
"""Portfolio backtest for the pre-registered entry variants (P1-P12).

Same 10-slot equal-weight engine, execution rules and costs as the V2 study.
Only the buy trigger and the exit rule change.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
WORK = f"{BASE}/02_代码/work.duckdb"
HERE = f"{BASE}/06_入场变体"
SRC = os.environ.get("SRC", f"{HERE}/02_代码/entries.parquet")
OUT = f"{HERE}/03_结果"
COST = float(os.environ.get("COST", "0.0020"))
SUF = os.environ.get("SUF", "")
ONLY = os.environ.get("ONLY", "")
HALF = COST / 2


@dataclass
class Pos:
    code: str
    j: int
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_i: int
    sig_date: pd.Timestamp


def limit_pct(code: str, date: pd.Timestamp, is_st: int) -> float:
    if is_st == 1:
        return 5.0
    if code.startswith("sh.68"):
        return 20.0
    if code.startswith("sz.30"):
        return 20.0 if date >= pd.Timestamp("2020-08-24") else 10.0
    return 10.0


def dedup_20(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    keep = np.zeros(len(df), dtype=bool)
    last: dict[str, int] = {}
    for i, code, ci in zip(df.index.values, df["code"].values, df["cal_idx"].values):
        if code not in last or ci - last[code] >= 20:
            keep[i] = True
            last[code] = ci
    return df.loc[keep].copy()


def metrics(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {"CAGR": np.nan, "MDD": np.nan, "Sharpe": np.nan, "Calmar": np.nan}
    ret = nav.pct_change().dropna()
    years = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    sharpe = ret.mean() / ret.std(ddof=1) * math.sqrt(252) if ret.std(ddof=1) > 0 else np.nan
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    sig = pd.read_parquet(SRC)
    sig["date"] = pd.to_datetime(sig["date"])
    sig["next_date"] = pd.to_datetime(sig["next_date"])
    sig = sig[sig["base_ok"] & sig["trend"] & sig["tradable"] & sig["wok"] & sig["mok"]].copy()
    print("entry universe", len(sig), flush=True)

    configs = [
        dict(name="P1_E0回踩MA20", flag="pullback", exit="fixed20", note="对照"),
        dict(name="P2_E2浅回踩MA10", flag="e2", exit="fixed10", note="主"),
        dict(name="P3_E2浅回踩MA10", flag="e2", exit="fixed20", note="主"),
        dict(name="P4_E2浅回踩MA10", flag="e2", exit="rule", note="主"),
        dict(name="P5_E2浅回踩MA10", flag="e2", exit="ma20only", note="主"),
        dict(name="P6_E1_20日新高", flag="e1", exit="fixed20", note="主"),
        dict(name="P7_E1_20日新高", flag="e1", exit="rule", note="主"),
        dict(name="P8_E1_20日新高", flag="e1", exit="ma20only", note="主"),
        dict(name="P9_E2浅回踩MA10", flag="e2", exit="fixed5", note="探索"),
        dict(name="P10_E2浅回踩MA10", flag="e2", exit="ma5", note="探索"),
        dict(name="P11_E1V放量突破", flag="e1v", exit="fixed20", note="探索"),
        dict(name="P12_E3平台突破", flag="e3", exit="fixed20", note="探索"),
    ]
    if ONLY:
        keys = [x.strip() for x in ONLY.split(",") if x.strip()]
        configs = [c for c in configs if c["name"].split("_")[0] in keys]

    # union of signal codes -> price panel
    need = set()
    for cfg in configs:
        f = sig[sig[cfg["flag"]]]
        if cfg["flag"] == "pullback":
            f = f[f["reclaim"]]
        need |= set(f["code"].unique())
    print("codes needed", len(need), flush=True)

    con = duckdb.connect(WORK, read_only=True)
    codes_sql = ",".join("'" + c + "'" for c in sorted(need))
    px = con.execute(f"""
        SELECT code, date, hfq_open, hfq_close, raw_preclose, trade_status, ma20, ma5, is_st
        FROM daily_feat
        WHERE date >= DATE '2011-12-01' AND code IN ({codes_sql})
    """).fetchdf()
    cal = con.execute("SELECT CAST(date AS DATE) AS date FROM calendar_idx WHERE CAST(date AS DATE) >= DATE '2012-01-03' ORDER BY 1").fetchdf()
    ew = con.execute("SELECT CAST(date AS DATE) AS date, nav FROM ew_nav WHERE CAST(date AS DATE) >= DATE '2012-01-03' ORDER BY 1").fetchdf()
    idx = con.execute("""
        SELECT i.code, CAST(i.date AS DATE) AS date, i.close
        FROM indices i WHERE CAST(i.date AS DATE) >= DATE '2012-01-03'
        ORDER BY i.code, i.date
    """).fetchdf()
    master = con.execute("SELECT code, out_date FROM stocks").fetchdf()
    con.close()

    px["date"] = pd.to_datetime(px["date"])
    cal["date"] = pd.to_datetime(cal["date"])
    dates = pd.DatetimeIndex(cal["date"])
    date_to_i = {d: i for i, d in enumerate(dates)}

    def mat(col: str) -> pd.DataFrame:
        return px.pivot(index="date", columns="code", values=col).reindex(dates)

    open_df = mat("hfq_open")
    close_df = mat("hfq_close")
    preclose_df = mat("raw_preclose")
    status_df = mat("trade_status")
    ma20_df = mat("ma20")
    ma5_df = mat("ma5")
    st_df = mat("is_st")
    close_ff_df = close_df.ffill()
    codes = list(open_df.columns)
    j_of = {c: j for j, c in enumerate(codes)}

    open_a = open_df.values.astype(float)
    close_a = close_df.values.astype(float)
    preclose_a = preclose_df.values.astype(float)
    status_a = status_df.values.astype(float)
    ma20_a = ma20_df.values.astype(float)
    ma5_a = ma5_df.values.astype(float)
    st_a = st_df.values.astype(float)
    close_ff_a = close_ff_df.values.astype(float)
    last_i = {}
    for c in codes:
        ld = close_df[c].last_valid_index()
        last_i[c] = date_to_i.get(ld, -1) if ld is not None else -1
    out_map = dict(zip(master["code"], master["out_date"]))
    is_delisted = {c: bool(out_map.get(c)) for c in codes}

    ew["date"] = pd.to_datetime(ew["date"])
    ew = ew.set_index("date").reindex(dates).ffill()
    ew_nav = (ew["nav"] / ew["nav"].iloc[0]).values
    idx_p_all = idx.pivot(index="date", columns="code", values="close").reindex(dates).ffill()
    idx_nav = {}
    for c in idx_p_all.columns:
        v = idx_p_all[c].values.astype(float)
        idx_nav[c] = v / v[0]

    oos_start = pd.Timestamp("2018-01-01")
    oos_i = int(np.searchsorted(dates.values, np.datetime64(oos_start)))

    summary = []
    for cfg in configs:
        v2 = sig[sig[cfg["flag"]]].copy()
        if cfg["flag"] == "pullback":
            v2 = v2[v2["reclaim"]]
        v2 = dedup_20(v2)
        v2["sig_i"] = v2["date"].map(date_to_i)
        v2["entry_i"] = v2["next_date"].map(date_to_i)
        v2 = v2.dropna(subset=["sig_i", "entry_i"])
        v2["sig_i"] = v2["sig_i"].astype(int)
        v2["entry_i"] = v2["entry_i"].astype(int)
        v2 = v2[v2["entry_i"] == v2["sig_i"] + 1]
        v2 = v2.sort_values(["next_date", "date", "code"])

        sig_by_entry: dict[int, list[tuple[int, str, pd.Timestamp]]] = {}
        for _, r in v2.iterrows():
            j = j_of.get(r["code"])
            if j is None:
                continue
            sig_by_entry.setdefault(int(r["entry_i"]), []).append((j, r["code"], r["date"]))

        cash = 1.0
        equity_prev = 1.0
        positions: dict[str, Pos] = {}
        pending_sells: dict[int, list[tuple[str, float]]] = {}
        curve = []
        trades = []
        delist_events = []

        for i in range(len(dates)):
            d = dates[i]
            for code, frac in pending_sells.pop(i, []):
                if code not in positions:
                    continue
                pos = positions[code]
                j = pos.j
                o = open_a[i, j]
                st = status_a[i, j]
                pc = preclose_a[i, j]
                is_st = int(st_a[i, j]) if np.isfinite(st_a[i, j]) else 0
                if not np.isfinite(o) or not np.isfinite(st) or st != 1:
                    pending_sells.setdefault(i + 1 if i + 1 < len(dates) else i, []).append((code, frac))
                    continue
                lp = limit_pct(code, d, is_st)
                if np.isfinite(pc) and o <= pc * (1 - lp / 100.0) + 0.001:
                    pending_sells.setdefault(i + 1 if i + 1 < len(dates) else i, []).append((code, frac))
                    continue
                price = o * (1 - HALF)
                cash += pos.shares * price
                trades.append({"code": code, "sig_date": pos.sig_date, "entry_date": pos.entry_date,
                               "exit_date": d, "entry_price": pos.entry_price, "exit_price": price,
                               "ret": price / pos.entry_price - 1, "hold_days": i - pos.entry_i,
                               "reason": cfg["exit"]})
                del positions[code]

            for j, code, sig_date in sig_by_entry.get(i, []):
                if len(positions) >= 10:
                    break
                if code in positions:
                    continue
                o = open_a[i, j]
                st = status_a[i, j]
                pc = preclose_a[i, j]
                is_st = int(st_a[i, j]) if np.isfinite(st_a[i, j]) else 0
                if not np.isfinite(o) or not np.isfinite(st) or st != 1:
                    continue
                lp = limit_pct(code, d, is_st)
                if np.isfinite(pc) and o >= pc * (1 + lp / 100.0) - 0.001:
                    continue
                target = equity_prev / 10
                alloc = min(target, cash)
                if alloc <= 1e-12:
                    break
                price = o * (1 + HALF)
                shares = alloc / price
                cash -= shares * price
                positions[code] = Pos(code, j, shares, price, d, i, sig_date)

            eq = cash
            for code, pos in positions.items():
                c = close_ff_a[i, pos.j]
                if not np.isfinite(c):
                    c = pos.entry_price
                eq += pos.shares * c

            for code, pos in list(positions.items()):
                c = close_a[i, pos.j]
                ma20 = ma20_a[i, pos.j]
                ma5 = ma5_a[i, pos.j]
                if not np.isfinite(c):
                    continue
                hold = i - pos.entry_i
                if cfg["exit"] == "fixed5":
                    hit = hold >= 5
                elif cfg["exit"] == "fixed10":
                    hit = hold >= 10
                elif cfg["exit"] == "fixed20":
                    hit = hold >= 20
                elif cfg["exit"] == "ma20only":
                    hit = (np.isfinite(ma20) and c < ma20) or hold >= 120
                elif cfg["exit"] == "ma5":
                    hit = (np.isfinite(ma5) and c < ma5) or (c < pos.entry_price * 0.92) or hold >= 20
                else:  # rule
                    hit = (np.isfinite(ma20) and c < ma20) or (c < pos.entry_price * 0.92) or hold >= 20
                if hit:
                    pending_sells.setdefault(i + 1 if i + 1 < len(dates) else i, []).append((code, 1.0))

            for code, pos in list(positions.items()):
                if last_i.get(code) == i and is_delisted.get(code, False):
                    c = close_a[i, pos.j]
                    if np.isfinite(c):
                        cash += pos.shares * c * (1 - HALF)
                        trades.append({"code": code, "sig_date": pos.sig_date, "entry_date": pos.entry_date,
                                       "exit_date": d, "entry_price": pos.entry_price,
                                       "exit_price": c * (1 - HALF),
                                       "ret": c * (1 - HALF) / pos.entry_price - 1,
                                       "hold_days": i - pos.entry_i, "reason": "delist"})
                        delist_events.append((code, d))
                        del positions[code]

            equity_prev = eq
            curve.append((d, eq, cash, len(positions)))

        eqdf = pd.DataFrame(curve, columns=["date", "equity", "cash", "n_pos"]).set_index("date")
        eqdf["ret"] = eqdf["equity"].pct_change().fillna(0)
        tr = pd.DataFrame(trades)
        if len(tr):
            tr["win"] = tr["ret"] > 0
            pf = tr.loc[tr["ret"] > 0, "ret"].sum() / abs(tr.loc[tr["ret"] < 0, "ret"].sum()) if (tr["ret"] < 0).any() else np.nan
            tstats = {"trades": len(tr), "win_rate": tr["win"].mean(), "avg_ret": tr["ret"].mean(),
                      "median_ret": tr["ret"].median(), "avg_hold": tr["hold_days"].mean(),
                      "profit_factor": pf, "delist_exits": len(delist_events),
                      "max_single_loss": tr["ret"].min(), "max_single_gain": tr["ret"].max()}
        else:
            tstats = {"trades": 0}

        m_full = metrics(eqdf["equity"])
        m_dev = metrics(eqdf["equity"].iloc[:oos_i])
        m_oos = metrics(eqdf["equity"].iloc[oos_i:] / eqdf["equity"].iloc[oos_i])
        row = {"name": cfg["name"], "flag": cfg["flag"], "exit": cfg["exit"], "note": cfg["note"],
               "n_signals": len(v2), **tstats,
               "avg_pos": eqdf["n_pos"].mean(),
               "cash_avg": (eqdf["cash"] / eqdf["equity"]).mean()}
        for k, v in [("full", m_full), ("dev", m_dev), ("oos", m_oos)]:
            row[f"{k}_CAGR"] = v["CAGR"]
            row[f"{k}_MDD"] = v["MDD"]
            row[f"{k}_Sharpe"] = v["Sharpe"]
        for k, v in [("EW", ew_nav), ("CSI500", idx_nav.get("sh.000905")), ("CSI1000", idx_nav.get("sh.000852")), ("CSI300", idx_nav.get("sh.000300"))]:
            if v is not None:
                mm_f = metrics(pd.Series(v, index=dates))
                mm_o = metrics(pd.Series(v[oos_i:] / v[oos_i], index=dates[oos_i:]))
                row[f"{k}_full_CAGR"] = mm_f["CAGR"]
                row[f"{k}_oos_CAGR"] = mm_o["CAGR"]
                row[f"{k}_full_MDD"] = mm_f["MDD"]
                row[f"{k}_oos_MDD"] = mm_o["MDD"]
        summary.append(row)

        tag = f"{cfg['name']}_{cfg['exit']}{SUF}"
        eqdf.to_csv(f"{OUT}/组合_{tag}_权益.csv", encoding="utf-8-sig")
        tr.to_csv(f"{OUT}/组合_{tag}_逐笔.csv", index=False, encoding="utf-8-sig")
        print(f"{tag}: sig={len(v2)}, full={m_full['CAGR']:.2%}/{m_full['MDD']:.2%}, "
              f"oos={m_oos['CAGR']:.2%}/{m_oos['MDD']:.2%}, trades={tstats.get('trades',0)}, "
              f"win={tstats.get('win_rate',np.nan):.1%}, hold={tstats.get('avg_hold',np.nan):.1f}", flush=True)

    cols = ["name", "exit", "note", "n_signals", "trades", "win_rate", "avg_ret", "median_ret", "avg_hold",
            "profit_factor", "full_CAGR", "full_MDD", "oos_CAGR", "oos_MDD", "dev_CAGR", "dev_MDD",
            "avg_pos", "cash_avg", "EW_full_CAGR", "EW_oos_CAGR", "CSI500_oos_CAGR", "CSI1000_oos_CAGR"]
    summ = pd.DataFrame(summary)
    summ.to_csv(f"{OUT}/组合_入场_汇总{SUF}.csv", index=False, encoding="utf-8-sig")
    print("\n", summ[cols].to_string(index=False))


if __name__ == "__main__":
    main()
