#!/usr/bin/env python3
"""Portfolio backtest with score-based stock selection.

Frozen V2 entry, execution rules and 10-slot portfolio. Selection is applied
before the 20-day per-stock de-duplication. Uses integer arrays in the daily
loop for speed.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
SEL = f"{BASE}/05_选股特征"
SIG = f"{SEL}/02_代码/信号_带评分.parquet"
CUT = f"{SEL}/03_结果/评分_切点.csv"
OUT = f"{SEL}/03_结果"
COST = 0.0020
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
    score: float


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
    ret = nav.pct_change().dropna()
    years = len(nav) / 252.0
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    sharpe = ret.mean() / ret.std(ddof=1) * math.sqrt(252) if ret.std(ddof=1) > 0 else np.nan
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
            "final": nav.iloc[-1], "days": len(nav)}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    sig = pd.read_parquet(SIG)
    sig["date"] = pd.to_datetime(sig["date"])
    sig["next_date"] = pd.to_datetime(sig["next_date"])
    cuts = pd.read_csv(CUT).set_index("score")
    print("signals", len(sig), "codes", sig.code.nunique(), flush=True)

    con = duckdb.connect(f"{BASE}/02_代码/work.duckdb", read_only=True)
    q = f"""
        SELECT code, date, hfq_open, hfq_close, raw_preclose, trade_status, ma20, is_st
        FROM daily_feat
        WHERE date >= DATE '2011-12-01'
          AND code IN ({','.join("'" + c + "'" for c in sorted(sig['code'].unique()))})
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

    def mat(col: str) -> pd.DataFrame:
        return px.pivot(index="date", columns="code", values=col).reindex(dates)

    open_df = mat("hfq_open")
    close_df = mat("hfq_close")
    preclose_df = mat("raw_preclose")
    status_df = mat("trade_status")
    ma20_df = mat("ma20")
    st_df = mat("is_st")
    close_ff_df = close_df.ffill()
    codes = list(open_df.columns)
    j_of = {c: j for j, c in enumerate(codes)}

    open_a = open_df.values.astype(float)
    close_a = close_df.values.astype(float)
    preclose_a = preclose_df.values.astype(float)
    status_a = status_df.values.astype(float)
    ma20_a = ma20_df.values.astype(float)
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

    configs = [
        dict(name="全信号", score=None, cut=None, gate=None, exit="rule", max_pos=10, sort="code"),
        dict(name="全信号", score=None, cut=None, gate=None, exit="fixed20", max_pos=10, sort="code"),
        dict(name="A前50", score="score_A", cut="q50", gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="A前30", score="score_A", cut="q70", gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="A前10", score="score_A", cut="q90", gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="A前30", score="score_A", cut="q70", gate=None, exit="fixed20", max_pos=10, sort="score"),
        dict(name="A前30_闸门", score="score_A", cut="q70", gate="mkt_gate", exit="rule", max_pos=10, sort="score"),
        dict(name="A前30_闸门", score="score_A", cut="q70", gate="mkt_gate", exit="fixed20", max_pos=10, sort="score"),
        dict(name="A前30_5槽", score="score_A", cut="q70", gate=None, exit="fixed20", max_pos=5, sort="score"),
        dict(name="Ag前30", score="score_A_greedy", cut="q70", gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="Ag前30", score="score_A_greedy", cut="q70", gate=None, exit="fixed20", max_pos=10, sort="score"),
        dict(name="B前30", score="score_B", cut="q70", gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="B前30", score="score_B", cut="q70", gate=None, exit="fixed20", max_pos=10, sort="score"),
        dict(name="A排序", score="score_A", cut=None, gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="A排序", score="score_A", cut=None, gate=None, exit="fixed20", max_pos=10, sort="score"),
        dict(name="A中段70_90", score="score_A", cut="q70", cut_hi="q90", gate=None, exit="fixed20", max_pos=10, sort="score"),
        dict(name="A中段70_90", score="score_A", cut="q70", cut_hi="q90", gate=None, exit="rule", max_pos=10, sort="score"),
        dict(name="A中段70_90_闸门", score="score_A", cut="q70", cut_hi="q90", gate="mkt_gate", exit="fixed20", max_pos=10, sort="score"),
    ]

    summary = []
    for cfg in configs:
        v2 = sig.copy()
        if cfg["gate"]:
            v2 = v2[v2[cfg["gate"]] == 1]
        if cfg["cut"]:
            c = float(cuts.loc[cfg["score"].replace("score_", ""), cfg["cut"]])
            v2 = v2[v2[cfg["score"]] >= c]
        if cfg.get("cut_hi"):
            ch = float(cuts.loc[cfg["score"].replace("score_", ""), cfg["cut_hi"]])
            v2 = v2[v2[cfg["score"]] < ch]
        if cfg["sort"] == "score":
            v2 = v2.dropna(subset=[cfg["score"]])
        v2 = dedup_20(v2)
        v2["sig_i"] = v2["date"].map(date_to_i)
        v2["entry_i"] = v2["next_date"].map(date_to_i)
        v2 = v2.dropna(subset=["sig_i", "entry_i"])
        v2["sig_i"] = v2["sig_i"].astype(int)
        v2["entry_i"] = v2["entry_i"].astype(int)
        v2 = v2[v2["entry_i"] == v2["sig_i"] + 1]
        if cfg["sort"] == "score":
            v2 = v2.sort_values(["next_date", cfg["score"], "code"], ascending=[True, False, True])
        else:
            v2 = v2.sort_values(["next_date", "date", "code"])

        sig_by_entry: dict[int, list[tuple[int, str, pd.Timestamp, float]]] = {}
        for _, r in v2.iterrows():
            j = j_of.get(r["code"])
            if j is None:
                continue
            sc = float(r[cfg["score"]]) if cfg["score"] else np.nan
            sig_by_entry.setdefault(int(r["entry_i"]), []).append((j, r["code"], r["date"], sc))

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
                sell_shares = pos.shares if frac >= 1.0 else pos.shares * frac
                cash += sell_shares * price
                trades.append({"code": code, "sig_date": pos.sig_date, "entry_date": pos.entry_date, "exit_date": d,
                               "entry_price": pos.entry_price, "exit_price": price,
                               "ret": price / pos.entry_price - 1, "hold_days": i - pos.entry_i,
                               "score": pos.score, "reason": "half" if frac < 1.0 else cfg["exit"]})
                if frac >= 1.0:
                    del positions[code]
                else:
                    pos.shares -= sell_shares

            for j, code, sig_date, score in sig_by_entry.get(i, []):
                if len(positions) >= cfg["max_pos"]:
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
                target = equity_prev / cfg["max_pos"]
                alloc = min(target, cash)
                if alloc <= 1e-12:
                    break
                price = o * (1 + HALF)
                shares = alloc / price
                cash -= shares * price
                positions[code] = Pos(code, j, shares, price, d, i, sig_date, score)

            eq = cash
            for code, pos in positions.items():
                c = close_ff_a[i, pos.j]
                if not np.isfinite(c):
                    c = pos.entry_price
                eq += pos.shares * c

            for code, pos in list(positions.items()):
                c = close_a[i, pos.j]
                ma = ma20_a[i, pos.j]
                if not np.isfinite(c):
                    continue
                hold = i - pos.entry_i
                if cfg["exit"] == "fixed20":
                    hit = hold >= 20
                elif cfg["exit"] == "fixed40":
                    hit = hold >= 40
                elif cfg["exit"] == "ma20":
                    hit = np.isfinite(ma) and c < ma
                else:
                    hit = (np.isfinite(ma) and c < ma) or (c < pos.entry_price * 0.92) or (hold >= 20)
                if hit:
                    pending_sells.setdefault(i + 1 if i + 1 < len(dates) else i, []).append((code, 1.0))

            for code, pos in list(positions.items()):
                if last_i.get(code) == i and is_delisted.get(code, False):
                    c = close_a[i, pos.j]
                    if np.isfinite(c):
                        cash += pos.shares * c * (1 - HALF)
                        trades.append({"code": code, "sig_date": pos.sig_date, "entry_date": pos.entry_date, "exit_date": d,
                                       "entry_price": pos.entry_price, "exit_price": c * (1 - HALF),
                                       "ret": c * (1 - HALF) / pos.entry_price - 1, "hold_days": i - pos.entry_i,
                                       "score": pos.score, "reason": "delist"})
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

        m = metrics(eqdf["equity"])
        row = {"name": cfg["name"], "exit": cfg["exit"], "max_pos": cfg["max_pos"],
               "score": cfg["score"] or "", "cut": cfg["cut"] or "", "gate": cfg["gate"] or "",
               "n_signals": len(v2), **m, **tstats,
               "avg_pos": eqdf["n_pos"].mean(), "cash_avg": (eqdf["cash"] / eqdf["equity"]).mean()}
        for k, v in [("EW", ew_nav), ("CSI500", idx_nav.get("sh.000905")), ("CSI1000", idx_nav.get("sh.000852")), ("CSI300", idx_nav.get("sh.000300"))]:
            if v is not None:
                mm = metrics(pd.Series(v, index=dates))
                row[f"{k}_CAGR"] = mm["CAGR"]
                row[f"{k}_MDD"] = mm["MDD"]
        summary.append(row)

        tag = f"{cfg['name']}_{cfg['exit']}_{cfg['max_pos']}"
        eqdf.to_csv(f"{OUT}/组合_选股_{tag}_权益.csv", encoding="utf-8-sig")
        tr.to_csv(f"{OUT}/组合_选股_{tag}_逐笔.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([row]).to_csv(f"{OUT}/组合_选股_{tag}_指标.csv", index=False, encoding="utf-8-sig")
        print(f"{tag}: signals={len(v2)}, CAGR={m['CAGR']:.2%}, MDD={m['MDD']:.2%}, "
              f"trades={tstats.get('trades',0)}, win={tstats.get('win_rate',np.nan):.1%}, "
              f"avg_pos={eqdf['n_pos'].mean():.2f}", flush=True)

    pd.DataFrame(summary).to_csv(f"{OUT}/组合_选股_汇总.csv", index=False, encoding="utf-8-sig")
    print("\n", pd.DataFrame(summary)[["name", "exit", "max_pos", "n_signals", "CAGR", "MDD", "Sharpe",
                                        "trades", "win_rate", "avg_ret", "avg_pos", "cash_avg",
                                        "EW_CAGR", "CSI500_CAGR"]].to_string(index=False))


if __name__ == "__main__":
    main()
