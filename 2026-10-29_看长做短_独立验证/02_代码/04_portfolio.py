#!/usr/bin/env python3
"""Executable portfolio backtest for the V2 看长做短 proxy.

10 equal-weight slots, T+1, next-open entries, next-open exits, limit-up
blocked, limit-down deferred, 0.20% round-trip cost, delisting forced out at
last close. No leverage, no shorting.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/02_代码"
OUT = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证/03_结果"
START = pd.Timestamp("2012-01-03")
COST = 0.0020
HALF = COST / 2
EXIT_MODE = os.environ.get("EXIT_MODE", "rule")
SIGNAL_FILE = os.environ.get("SIGNAL_FILE", "信号层_逐笔.csv")
SIGNAL_VARIANT = os.environ.get("SIGNAL_VARIANT", "V2_monthly")


@dataclass
class Pos:
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_i: int
    half_sold: bool = False


def limit_pct(code: str, date: pd.Timestamp, is_st: int) -> float:
    if is_st == 1:
        return 5.0
    if code.startswith("sh.68"):
        return 20.0
    if code.startswith("sz.30"):
        return 20.0 if date >= pd.Timestamp("2020-08-24") else 10.0
    return 10.0


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    events = pd.read_csv(f"{OUT}/{SIGNAL_FILE}", parse_dates=["date", "next_date"])
    v2 = events[events["variant"] == SIGNAL_VARIANT].copy()
    codes = sorted(v2["code"].unique())
    print("codes", len(codes), "signals", len(v2))

    con = duckdb.connect(f"{BASE}/work.duckdb", read_only=True)
    q = f"""
        SELECT code, date, hfq_open, hfq_close, raw_open, raw_preclose, trade_status, ma20, is_st
        FROM daily_feat
        WHERE date >= DATE '2011-12-01'
          AND code IN ({','.join("'" + c + "'" for c in codes)})
    """
    px = con.execute(q).fetchdf()
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

    # matrices
    def mat(col: str) -> pd.DataFrame:
        return px.pivot(index="date", columns="code", values=col).reindex(dates)

    open_m = mat("hfq_open")
    close_m = mat("hfq_close")
    raw_open_m = mat("raw_open")
    preclose_m = mat("raw_preclose")
    status_m = mat("trade_status")
    ma20_m = mat("ma20")
    st_m = mat("is_st")
    close_ff = close_m.ffill()
    last_date = {c: close_m[c].last_valid_index() for c in close_m.columns}
    out_map = dict(zip(master["code"], master["out_date"]))
    is_delisted = {c: bool(out_map.get(c)) for c in close_m.columns}

    # signals grouped by actual entry date; require entry is the next market day
    v2 = v2.merge(cal.assign(cal_idx=range(len(cal))), left_on="date", right_on="date", how="left")
    v2 = v2.dropna(subset=["cal_idx"])
    v2["cal_idx"] = v2["cal_idx"].astype(int)
    v2 = v2[v2["next_date"].isin(dates)]
    v2["entry_i"] = v2["next_date"].map(date_to_i)
    v2 = v2[v2["entry_i"] == v2["cal_idx"] + 1]
    print("signals with next-day entry", len(v2))

    sig_by_entry: dict[pd.Timestamp, list[pd.Series]] = {}
    for _, r in v2.sort_values(["next_date", "date", "code"]).iterrows():
        sig_by_entry.setdefault(r["next_date"], []).append(r)

    cash = 1.0
    equity_prev = 1.0
    positions: dict[str, Pos] = {}
    pending_sells: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    curve = []
    trades = []
    delist_events = []

    for i, d in enumerate(dates):
        # 1) sells at open
        for code, frac in pending_sells.pop(d, []):
            if code not in positions:
                continue
            pos = positions[code]
            o = open_m.at[d, code] if code in open_m.columns else np.nan
            st = status_m.at[d, code] if code in status_m.columns else np.nan
            pc = preclose_m.at[d, code] if code in preclose_m.columns else np.nan
            is_st = int(st_m.at[d, code]) if code in st_m.columns and not pd.isna(st_m.at[d, code]) else 0
            if pd.isna(o) or st != 1:
                pending_sells.setdefault(dates[i + 1] if i + 1 < len(dates) else d, []).append((code, frac))
                continue
            lp = limit_pct(code, d, is_st)
            if not pd.isna(pc) and o <= pc * (1 - lp / 100.0) + 0.001:
                pending_sells.setdefault(dates[i + 1] if i + 1 < len(dates) else d, []).append((code, frac))
                continue
            price = o * (1 - HALF)
            sell_shares = pos.shares if frac >= 1.0 else pos.shares * frac
            cash += sell_shares * price
            trades.append({
                "code": code, "entry_date": pos.entry_date, "exit_date": d,
                "entry_price": pos.entry_price, "exit_price": price,
                "ret": price / pos.entry_price - 1, "hold_days": i - pos.entry_i,
                "reason": "half" if frac < 1.0 else "rule",
            })
            if frac >= 1.0:
                del positions[code]
            else:
                pos.shares -= sell_shares

        # 2) buys at open
        for sig in sig_by_entry.get(d, []):
            if len(positions) >= 10:
                break
            code = sig["code"]
            if code in positions or code not in open_m.columns:
                continue
            o = open_m.at[d, code]
            st = status_m.at[d, code]
            pc = preclose_m.at[d, code]
            is_st = int(st_m.at[d, code]) if not pd.isna(st_m.at[d, code]) else 0
            if pd.isna(o) or st != 1:
                continue
            lp = limit_pct(code, d, is_st)
            if not pd.isna(pc) and o >= pc * (1 + lp / 100.0) - 0.001:
                continue
            target = equity_prev / 10.0
            alloc = min(target, cash)
            if alloc <= 1e-12:
                break
            price = o * (1 + HALF)
            shares = alloc / price
            cash -= shares * price
            positions[code] = Pos(shares, price, d, i)

        # 3) mark to market
        eq = cash
        for code, pos in positions.items():
            c = close_ff.at[d, code] if code in close_ff.columns else np.nan
            if pd.isna(c):
                c = pos.entry_price
            eq += pos.shares * c
        # 4) exit decisions at close
        for code, pos in list(positions.items()):
            c = close_m.at[d, code] if code in close_m.columns else np.nan
            ma = ma20_m.at[d, code] if code in ma20_m.columns else np.nan
            if pd.isna(c):
                continue
            hold = i - pos.entry_i
            if EXIT_MODE == "fixed20":
                hit = hold >= 20
            elif EXIT_MODE == "fixed40":
                hit = hold >= 40
            elif EXIT_MODE == "ma20_only":
                hit = (not pd.isna(ma) and c < ma)
            elif EXIT_MODE == "stop_only":
                hit = c < pos.entry_price * 0.92
            elif EXIT_MODE == "wide":
                hit = (not pd.isna(ma) and c < ma) or (c < pos.entry_price * 0.90) or (hold >= 40)
            elif EXIT_MODE == "half25":
                if (c / pos.entry_price - 1) >= 0.25 and not pos.half_sold:
                    pos.half_sold = True
                    pending_sells.setdefault(dates[i + 1] if i + 1 < len(dates) else d, []).append((code, 0.5))
                hit = (not pd.isna(ma) and c < ma) or (hold >= 20)
            else:
                hit = (not pd.isna(ma) and c < ma) or (c < pos.entry_price * 0.92) or (hold >= 20)
            if hit:
                pending_sells.setdefault(dates[i + 1] if i + 1 < len(dates) else d, []).append((code, 1.0))
        # 5) delisting: last bar of a stock with a real out_date
        for code, pos in list(positions.items()):
            if last_date.get(code) == d and is_delisted.get(code, False):
                c = close_m.at[d, code]
                if not pd.isna(c):
                    cash += pos.shares * c * (1 - HALF)
                    trades.append({
                        "code": code, "entry_date": pos.entry_date, "exit_date": d,
                        "entry_price": pos.entry_price, "exit_price": c * (1 - HALF),
                        "ret": c * (1 - HALF) / pos.entry_price - 1, "hold_days": i - pos.entry_i,
                        "reason": "delist",
                    })
                    delist_events.append((code, d))
                    del positions[code]

        equity_prev = eq
        curve.append({"date": d, "equity": eq, "cash": cash, "n_pos": len(positions)})

    eqdf = pd.DataFrame(curve).set_index("date")
    eqdf["ret"] = eqdf["equity"].pct_change().fillna(0)

    # benchmarks
    ew["date"] = pd.to_datetime(ew["date"])
    ew = ew.set_index("date").reindex(eqdf.index).ffill()
    ew["nav"] = ew["nav"] / ew["nav"].iloc[0]
    idx["date"] = pd.to_datetime(idx["date"])
    idx_p = idx.pivot(index="date", columns="code", values="close").reindex(eqdf.index).ffill()
    for c in idx_p.columns:
        idx_p[c] = idx_p[c] / idx_p[c].iloc[0]

    def metrics(nav: pd.Series) -> dict:
        nav = nav.dropna()
        ret = nav.pct_change().dropna()
        years = len(nav) / 252.0
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
        mdd = (nav / nav.cummax() - 1).min()
        sharpe = ret.mean() / ret.std(ddof=1) * math.sqrt(252) if ret.std(ddof=1) > 0 else np.nan
        return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
                "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
                "final": nav.iloc[-1], "days": len(nav)}

    rows = [{"name": "V2_portfolio", **metrics(eqdf["equity"])},
            {"name": "EW_allA", **metrics(ew["nav"])}]
    for c in idx_p.columns:
        rows.append({"name": c, **metrics(idx_p[c])})
    mdf = pd.DataFrame(rows)

    tr = pd.DataFrame(trades)
    if len(tr):
        tr["win"] = tr["ret"] > 0
        pf = tr.loc[tr["ret"] > 0, "ret"].sum() / abs(tr.loc[tr["ret"] < 0, "ret"].sum()) if (tr["ret"] < 0).any() else np.nan
        tstats = {
            "trades": len(tr), "win_rate": tr["win"].mean(), "avg_ret": tr["ret"].mean(),
            "median_ret": tr["ret"].median(), "avg_hold": tr["hold_days"].mean(),
            "profit_factor": pf, "delist_exits": len(delist_events),
            "max_single_loss": tr["ret"].min(), "max_single_gain": tr["ret"].max(),
        }
    else:
        tstats = {}

    eqdf.to_csv(f"{OUT}/组合_{SIGNAL_VARIANT}_{EXIT_MODE}_权益.csv", encoding="utf-8-sig")
    tr.to_csv(f"{OUT}/组合_{SIGNAL_VARIANT}_{EXIT_MODE}_逐笔.csv", index=False, encoding="utf-8-sig")
    mdf.to_csv(f"{OUT}/组合_{SIGNAL_VARIANT}_{EXIT_MODE}_指标.csv", index=False, encoding="utf-8-sig")
    pd.Series(tstats).to_csv(f"{OUT}/组合_{SIGNAL_VARIANT}_{EXIT_MODE}_交易统计.csv", encoding="utf-8-sig")

    print(mdf.to_string(index=False))
    print(tstats)
    print("avg positions", eqdf["n_pos"].mean(), "max", eqdf["n_pos"].max())
    print("cash avg", eqdf["cash"].mean())


if __name__ == "__main__":
    main()
