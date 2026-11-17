#!/usr/bin/env python3
"""63 波段审计：择时组合、超额 t、成本敏感（补 60 的引用数字）。"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import macd_core as M
import feature_core as F
import kdj_core as K
import stock_core as S
import importlib.util
from stats_utils import nw_t

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "03_结果")
spec = importlib.util.spec_from_file_location("cap", os.path.join(os.path.dirname(os.path.abspath(__file__)), "60_波段捕获.py"))
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)


def daily_from(P, trades, cost_side=0.00075):
    n, nc = P["shape"]
    pos = np.zeros((n, nc), dtype=np.int8)
    ent = np.zeros(n); ext = np.zeros(n)
    for r in trades.itertuples():
        pos[r.entry_i:r.exit_i, r.ci] = 1
        ent[r.entry_i] += 1; ext[r.exit_i] += 1
    active = pos.astype(bool); cnt = active.sum(axis=1).astype(float)
    s = np.nansum(np.where(active, P["r_oo"], 0), axis=1)
    x = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    x -= np.where(cnt > 0, cost_side * ent / np.maximum(cnt, 1), 0.0)
    prev = np.concatenate([[0], cnt[:-1]])
    x -= np.where(prev > 0, cost_side * ext / np.maximum(prev, 1), 0.0)
    return pd.Series(x, index=P["dates"]).loc["2010-01-01":].fillna(0.0)


def stats(daily, bench, label):
    nav = (1 + daily).cumprod()
    def sub(a, b):
        x = daily.loc[a:b]; nv = (1 + x).cumprod()
        return nv.iloc[-1] ** (252 / len(x)) - 1 if len(x) else np.nan
    diff = daily - bench
    return dict(方案=label, 年化=nav.iloc[-1] ** (252 / len(daily)) - 1,
                最大回撤=(nav / nav.cummax() - 1).min(),
                夏普=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
                早期=sub("2010-01-01", "2017-12-31"),
                后期=sub("2018-01-01", "2026-12-31"),
                日均超额=diff.mean(), 年化超额近似=diff.mean() * 252,
                超额t=nw_t(diff, 20))


def main():
    t0 = time.time()
    P = S.load_base(); base = P["universe_inv"]
    pv = pd.read_parquet(os.path.join(RES, "59_波段_拐点.parquet"))
    mf = F.build_macd_features(P); kd = K.kdj_features(P)
    c = P["close_ff"]; ma20 = M.rolling_mean(c, 20); ma200 = M.rolling_mean(c, 200)
    sig = {
        "MACD多头": (mf["dif"] > mf["dea"]) & P["valid"],
        "MACD空头": (mf["dif"] <= mf["dea"]) & P["valid"],
        "KDJ多头": (kd["k"] > kd["d"]) & P["valid"],
        "KDJ空头": (kd["k"] <= kd["d"]) & P["valid"],
        "MACD金叉": M.crosses(mf["dif"], mf["dea"], P)[0],
        "MACD死叉": M.crosses(mf["dif"], mf["dea"], P)[1],
        "KDJ金叉": kd["gold"], "KDJ死叉": kd["death"],
    }
    bench = pd.Series(np.nanmean(np.where(base, P["r_oo"], np.nan), axis=1),
                      index=P["dates"]).loc["2010-01-01":].fillna(0.0)
    rows = [stats(bench, bench, "全A等权")]
    cost_rows = []
    for th in (0.05, 0.10):
        d = cap.add_fills(P, cap.build_pairs(P, pv, th))
        d = cap.add_states(P, d, sig, kd, ma20, ma200)
        variants = [("价格", pd.DataFrame(dict(ci=d.ci, entry_i=d.entry_i, exit_i=d.exit_i))),
                    ("MACD确认多", pd.DataFrame(dict(ci=d[d.MACD确认多].ci, entry_i=d[d.MACD确认多].entry_i, exit_i=d[d.MACD确认多].exit_i))),
                    ("KDJ确认多", pd.DataFrame(dict(ci=d[d.KDJ确认多].ci, entry_i=d[d.KDJ确认多].entry_i, exit_i=d[d.KDJ确认多].exit_i))),
                    ("双确认多", pd.DataFrame(dict(ci=d[d.MACD确认多 & d.KDJ确认多].ci, entry_i=d[d.MACD确认多 & d.KDJ确认多].entry_i, exit_i=d[d.MACD确认多 & d.KDJ确认多].exit_i))),
                    ("MACD择时", cap.timing_trades(P, d, sig, sig["MACD多头"], sig["MACD空头"], "m")),
                    ("KDJ择时", cap.timing_trades(P, d, sig, sig["KDJ多头"], sig["KDJ空头"], "k"))]
        for lab, tr in variants:
            if len(tr) == 0:
                continue
            daily = daily_from(P, tr)
            rows.append(stats(daily, bench, f"{lab}_{int(th * 100)}%"))
            cr = {"方案": f"{lab}_{int(th * 100)}%"}
            for cs in (0.00075, 0.0015, 0.0025):
                x = daily_from(P, tr, cs)
                cr[f"单边{cs * 100:.3f}%"] = (1 + x).prod() ** (252 / len(x)) - 1
            cost_rows.append(cr)
        print("th", th, "%.1fs" % (time.time() - t0))
    for name in ("MACD交叉", "KDJ交叉"):
        tr = pd.read_parquet(os.path.join(RES, f"60_{name}_交易.parquet"))
        daily = daily_from(P, tr)
        rows.append(stats(daily, bench, name))
        cr = {"方案": name}
        for cs in (0.00075, 0.0015, 0.0025):
            x = daily_from(P, tr, cs)
            cr[f"单边{cs * 100:.3f}%"] = (1 + x).prod() ** (252 / len(x)) - 1
        cost_rows.append(cr)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "63_波段_择时与超额.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cost_rows).to_csv(os.path.join(RES, "63_波段_成本.csv"),
                                   index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(pd.DataFrame(cost_rows).to_string(index=False))
    print("done %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
