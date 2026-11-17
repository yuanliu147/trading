#!/usr/bin/env python3
"""67 零轴回踩+柱体放大：当“减仓/止盈”信号，是否比拿到波段高点确认更好？"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import stock_core as S
import importlib.util
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
spec = importlib.util.spec_from_file_location("cap", os.path.join(HERE, "60_波段捕获.py"))
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
RATIOS = (0.0, 0.2, 0.3, 0.5)


def daily_from(P, trades, cost_side=0.00075):
    n, nc = P["shape"]
    pos = np.zeros((n, nc), dtype=np.int8); ent = np.zeros(n); ext = np.zeros(n)
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


def stats(daily, label):
    nav = (1 + daily).cumprod()
    def sub(a, b):
        x = daily.loc[a:b]; nv = (1 + x).cumprod()
        return nv.iloc[-1] ** (252 / len(x)) - 1 if len(x) else np.nan
    return dict(方案=label, 年化=nav.iloc[-1] ** (252 / len(daily)) - 1,
                最大回撤=(nav / nav.cummax() - 1).min(),
                夏普=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
                早期=sub("2010-01-01", "2017-12-31"),
                后期=sub("2018-01-01", "2026-12-31"))


def main():
    t0 = time.time()
    P = S.load_base(); base = P["universe_inv"]; dates = P["dates"]
    mf = F.build_macd_features(P)
    hist = mf["hist"]; c = P["close_ff"]
    h = (hist / c * 100).astype(np.float32)
    hp = M.shift(h, 1)
    peak20 = M.rolling_max(h, 20, minp=10)
    trough5 = M.rolling_min(h, 5, minp=3)
    red = h > 0; expand = red & (h > hp)
    mask = base & P["valid"] & (P["bars"] >= 120)
    pv = pd.read_parquet(os.path.join(RES, "59_波段_拐点.parquet"))
    d = cap.add_fills(P, cap.build_pairs(P, pv, 0.10))
    print("pairs", len(d), "%.1fs" % (time.time() - t0))
    rows = []; port_rows = []
    base_tr = pd.DataFrame(dict(ci=d.ci, entry_i=d.entry_i, exit_i=d.exit_i))
    port_rows.append(stats(daily_from(P, base_tr), "基线：拿到高点确认"))
    for ratio in RATIOS:
        if ratio == 0.0:
            pull = trough5 <= 0; name = "回踩破0再翻红"
        else:
            pull = (trough5 > 0) & (trough5 <= ratio * peak20)
            name = f"回踩≤{ratio:.0%}峰值"
        pat = mask & pull & expand
        exits = d["exit_i"].to_numpy().copy()
        used = np.zeros(len(d), dtype=bool)
        for q, r in enumerate(d.itertuples()):
            a = int(r.entry_i)
            b = int(r.last_valid) if bool(r.forced) else int(r.hpivot_i)
            if b <= a:
                continue
            w = np.where(pat[a:b + 1, r.ci])[0]
            if len(w) == 0:
                continue
            t = a + int(w[0])
            x = cap.fill_open(P, t + 1, r.ci, "sell")
            if x > a:
                exits[q] = x; used[q] = True
        d2 = d.copy()
        d2["exit_pat"] = exits
        d2["ret_pat"] = (1 + P["open"][d2.exit_pat.to_numpy(), d2.ci.to_numpy()] /
                         P["open"][d2.entry_i.to_numpy(), d2.ci.to_numpy()] - 1) * (1 - cap.COST) - 1
        diff = d2.loc[used, "ret_pat"] - d2.loc[used, "ret_net"]
        # 月度配对差
        x = d2[used].copy()
        x["month"] = pd.to_datetime(dates[x["entry_i"].to_numpy()]).to_period("M")
        md = x.groupby("month").apply(lambda g: (g["ret_pat"] - g["ret_net"]).mean(), include_groups=False)
        rows.append(dict(
            形态=name, 有形态占比=float(used.mean()), 有形态笔数=int(used.sum()),
            基线中位=float(d2["ret_net"].median()), 形态中位=float(d2.loc[used, "ret_pat"].median()),
            差均值=float(diff.mean()), 差中位=float(diff.median()),
            差为正=float((diff > 0).mean()),
            月度差t=float(nw_t(md, 6)) if len(md) else np.nan,
            节省天数中位=float(np.median(d2.loc[used, "exit_i"] - d2.loc[used, "exit_pat"])),
        ))
        tr = pd.DataFrame(dict(ci=d2.ci, entry_i=d2.entry_i, exit_i=d2.exit_pat))
        port_rows.append(stats(daily_from(P, tr), f"{name} 离场"))
        print(name, "used", int(used.sum()), "%.1fs" % (time.time() - t0))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "67_零轴回踩_离场.csv"), index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(pd.DataFrame(port_rows).to_string(index=False))


if __name__ == "__main__":
    main()
