#!/usr/bin/env python3
"""50 MACD×KDJ 事件层：MACD 金叉按 KDJ 过滤；KDJ 金叉按 MACD 过滤；双金叉与时序。"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import macd_core as M
import feature_core as F
import kdj_core as K
from stats_utils import nw_t

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "03_结果")
COST = 0.0015
EARLY_END = pd.Timestamp("2017-12-31")
LATE_START = pd.Timestamp("2018-01-01")


def daily_mean(sub, col="ex20"):
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub.groupby("date")[col].mean()


def ev_stat(sub, col="ex20"):
    sub = sub.dropna(subset=[col])
    if len(sub) < 100:
        return dict(信号数=len(sub), 交易日=0, 毛超额=np.nan, 净超额=np.nan,
                    中位=np.nan, t值=np.nan, 正日占比=np.nan, 前期=np.nan, 后期=np.nan,
                    年数=0, 正年数=0)
    dm = daily_mean(sub, col)
    early = dm[dm.index <= EARLY_END]
    late = dm[dm.index >= LATE_START]
    yr = dm.groupby(dm.index.year).mean()
    return dict(信号数=len(sub), 交易日=len(dm),
                毛超额=float(dm.mean()), 净超额=float(dm.mean() - COST),
                中位=float(sub[col].median()),
                t值=float(nw_t(dm, lag=20)), 正日占比=float((dm > 0).mean()),
                前期=float(early.mean()) if len(early) else np.nan,
                后期=float(late.mean()) if len(late) else np.nan,
                年数=int(yr.notna().sum()), 正年数=int((yr > 0).sum()))


def contrast(a, b, col="ex20"):
    """同一批交易日的日度均值差；a/b 为事件子集。"""
    da, db = daily_mean(a, col), daily_mean(b, col)
    common = da.index.intersection(db.index)
    if len(common) < 50:
        return dict(差值=np.nan, t值=np.nan, 共同日=len(common),
                    A毛=np.nan, B毛=np.nan)
    diff = (da.reindex(common) - db.reindex(common)).dropna()
    return dict(差值=float(diff.mean()), t值=float(nw_t(diff, 20)),
                共同日=len(diff), A毛=float(da.reindex(common).mean()),
                B毛=float(db.reindex(common).mean()))


def main():
    t0 = time.time()
    P = M.load_panel()
    dates, codes = P["dates"], P["codes"]
    mf = M.market_features(P)
    macd = F.build_macd_features(P)
    st = F.build_structure(P, mf)
    base = F.base_mask(P)
    kd = K.kdj_features(P)
    wk = K.period_kdj_state(P, "W")
    mk = K.period_kdj_state(P, "M")
    print("features done", time.time() - t0)
    fwd, ok, mkt, loo = M.forward_returns(P, (5, 20, 60))
    exs = {h: (fwd[h] - loo[h]).astype(np.float32) for h in (5, 20, 60)}
    masks = {h: base & ok[h] & np.isfinite(exs[h]) for h in (5, 20, 60)}
    gold, death = M.crosses(macd["dif"], macd["dea"], P)
    k, d, j = kd["k"], kd["d"], kd["j"]
    qk = F.row_quantile(k, base, 5)
    qgap = F.row_quantile(k - d, base, 5)
    ma60 = st["ma60"]
    above60 = P["close_ff"] > ma60
    mkt_state = st["market"][:, 0]
    kdj_gold = kd["gold"]
    kdj_death = kd["death"]
    # 过去 N 日事件
    def recent(ev, n):
        cs = np.cumsum(ev, axis=0)
        return (cs - M.shift(cs, n, fill=0)) > 0
    kg5, kg20 = recent(kdj_gold, 5), recent(kdj_gold, 20)
    kd5, kd20 = recent(kdj_death, 5), recent(kdj_death, 20)
    mg5, mg20 = recent(gold, 5), recent(gold, 20)
    md5, md20 = recent(death, 5), recent(death, 20)
    top_div = (P["close_ff"] >= M.rolling_max(P["close_ff"], 20) - 1e-6) & \
              (k < M.rolling_max(k, 20) - 1e-9)
    bot_div = (P["close_ff"] <= M.rolling_min(P["close_ff"], 20) + 1e-6) & \
              (k > M.rolling_min(k, 20) + 1e-9)
    zero = np.full(P["shape"], -1, dtype=np.int8)
    zero[(macd["dif"] > 0) & (macd["dea"] > 0)] = 0
    zero[(macd["dif"] > 0) & (macd["dea"] <= 0)] = 1
    zero[(macd["dif"] <= 0) & (macd["dea"] > 0)] = 2
    zero[(macd["dif"] <= 0) & (macd["dea"] <= 0)] = 3

    # ---------- MACD 金叉事件 ----------
    e = pd.read_parquet(os.path.join(RES, "事件_金叉_12269.parquet"))
    e = e[np.isfinite(e.ex20)].copy()
    di = pd.Index(dates).get_indexer(e.date)
    ci = pd.Index(codes).get_indexer(e.code)
    good = (di >= 0) & (ci >= 0)
    e = e[good].copy()
    di, ci = di[good], ci[good]
    e["di"], e["ci"] = di, ci
    e["K"] = k[di, ci]; e["D"] = d[di, ci]; e["J"] = j[di, ci]
    e["Kq"] = qk[di, ci]; e["gapq"] = qgap[di, ci]
    e["K>D"] = kd["states"]["K>D"][di, ci]
    e["K>70"] = (k > 70)[di, ci]; e["K>80"] = (k > 80)[di, ci]
    e["K<20"] = (k < 20)[di, ci]; e["K<30"] = (k < 30)[di, ci]
    e["J<0"] = (j < 0)[di, ci]; e["J>100"] = (j > 100)[di, ci]
    e["kdj_gold_today"] = kdj_gold[di, ci]
    e["kdj_gold_5d"] = kg5[di, ci]
    e["kdj_gold_20d"] = kg20[di, ci]
    e["kdj_death_5d"] = kd5[di, ci]
    e["top_div"] = top_div[di, ci]; e["bot_div"] = bot_div[di, ci]
    e["wkdj"] = wk["state"][di, ci]; e["mkdj"] = mk["state"][di, ci]
    e["wgold"] = wk["gold_recent"][di, ci]; e["mgold"] = mk["gold_recent"][di, ci]
    e["above60"] = above60[di, ci]
    e["mkt"] = mkt_state[di]
    e["zero"] = zero[di, ci]
    e["净超额"] = e["ex20"] - COST

    # 过滤表
    rows = []
    def add_macd(lab, sel):
        rows.append(dict(过滤=lab, **ev_stat(e[sel])))
    add_macd("全部MACD金叉", pd.Series(True, index=e.index))
    for qi, ql in [(0, "KQ1(最低)"), (1, "KQ2"), (2, "KQ3"), (3, "KQ4"), (4, "KQ5(最高)")]:
        add_macd(ql, e.Kq == qi)
    add_macd("KDJ多(K>D)", e["K>D"].astype(bool))
    add_macd("KDJ空(K<D)", ~e["K>D"].astype(bool))
    add_macd("K>70", e["K>70"].astype(bool))
    add_macd("K>80", e["K>80"].astype(bool))
    add_macd("K<20", e["K<20"].astype(bool))
    add_macd("K<30", e["K<30"].astype(bool))
    add_macd("J<0", e["J<0"].astype(bool))
    add_macd("J>100", e["J>100"].astype(bool))
    add_macd("同日KDJ金叉", e.kdj_gold_today.astype(bool))
    add_macd("前5日内KDJ金叉(含今日)", e.kdj_gold_5d.astype(bool))
    add_macd("前5日内KDJ金叉(不含今日)", e.kdj_gold_5d.astype(bool) & ~e.kdj_gold_today.astype(bool))
    add_macd("前5日内无KDJ金叉", ~e.kdj_gold_5d.astype(bool))
    add_macd("前20日内无KDJ金叉", ~e.kdj_gold_20d.astype(bool))
    add_macd("前5日内KDJ死叉", e.kdj_death_5d.astype(bool))
    add_macd("KDJ顶背离", e.top_div.astype(bool))
    add_macd("KDJ底背离", e.bot_div.astype(bool))
    add_macd("周线KDJ多", e.wkdj.astype(bool))
    add_macd("周线KDJ空", ~e.wkdj.astype(bool))
    add_macd("月线KDJ多", e.mkdj.astype(bool))
    add_macd("月线KDJ空", ~e.mkdj.astype(bool))
    add_macd("周线KDJ近4周金叉", e.wgold.astype(bool))
    add_macd("月线KDJ近3月金叉", e.mgold.astype(bool))
    add_macd("价格>MA60", e.above60.astype(bool))
    add_macd("KDJ多+价格>MA60", e["K>D"].astype(bool) & e.above60.astype(bool))
    add_macd("KDJ空+价格>MA60", ~e["K>D"].astype(bool) & e.above60.astype(bool))
    em = pd.DataFrame(rows)
    em["净超额"] = em["毛超额"] - COST
    em.to_csv(os.path.join(RES, "KDJ_事件_MACD金叉.csv"), index=False, encoding="utf-8-sig")
    print("\nMACD金叉过滤")
    print(em[["过滤", "信号数", "毛超额", "净超额", "t值", "前期", "后期", "年数", "正年数"]].to_string(index=False))

    # 零轴 x KDJ
    rows = []
    for z, zl in [(0, "双上"), (1, "DIF上DEA下"), (2, "DIF下DEA上"), (3, "双下")]:
        for kk, kl in [(True, "KDJ多"), (False, "KDJ空")]:
            sub = e[(e.zero == z) & (e["K>D"].astype(bool) == kk)]
            rows.append(dict(零轴=zl, KDJ=kl, **ev_stat(sub)))
    ez = pd.DataFrame(rows)
    ez.to_csv(os.path.join(RES, "KDJ_事件_零轴xKDJ.csv"), index=False, encoding="utf-8-sig")
    print("\n零轴xKDJ")
    print(ez[["零轴", "KDJ", "信号数", "毛超额", "t值", "前期", "后期"]].to_string(index=False))

    # MACD 金叉内部：KDJ 过滤增量
    rows = []
    def add_contrast(lab, a, b):
        r = contrast(e[a], e[b])
        rows.append(dict(对比=lab, **r))
    add_contrast("KDJ多 vs KDJ空", e["K>D"].astype(bool), ~e["K>D"].astype(bool))
    add_contrast("K>70 vs K<=70", e["K>70"].astype(bool), ~e["K>70"].astype(bool))
    add_contrast("K<20 vs K>=20", e["K<20"].astype(bool), ~e["K<20"].astype(bool))
    add_contrast("同日KDJ金叉 vs 非同日", e.kdj_gold_today.astype(bool), ~e.kdj_gold_today.astype(bool))
    add_contrast("前5日内KDJ金叉 vs 无", e.kdj_gold_5d.astype(bool), ~e.kdj_gold_5d.astype(bool))
    add_contrast("KDJ顶背离 vs 无", e.top_div.astype(bool), ~e.top_div.astype(bool))
    add_contrast("KDJ底背离 vs 无", e.bot_div.astype(bool), ~e.bot_div.astype(bool))
    add_contrast("KQ1 vs KQ2-4", e.Kq == 0, e.Kq.isin([1, 2, 3]))
    add_contrast("KQ5 vs KQ2-4", e.Kq == 4, e.Kq.isin([1, 2, 3]))
    emc = pd.DataFrame(rows)
    emc.to_csv(os.path.join(RES, "KDJ_事件_MACD金叉增量.csv"), index=False, encoding="utf-8-sig")
    print("\nMACD金叉内部增量")
    print(emc.to_string(index=False))

    # ---------- KDJ 金叉事件 ----------
    ii, jj = np.where(kdj_gold & base & (P["bars"] >= 120))
    keep = dates[ii] >= pd.Timestamp("2010-01-01")
    ii, jj = ii[keep], jj[keep]
    # 只保留 ex20 可评估
    tr = masks[20][ii, jj]
    ii, jj = ii[tr], jj[tr]
    ret5 = np.full(P["shape"], np.nan, dtype=np.float32)
    ret5[5:] = P["close_ff"][5:] / P["close_ff"][:-5] - 1
    ek = pd.DataFrame({
        "date": dates[ii], "code": codes[jj],
        "K": k[ii, jj], "D": d[ii, jj], "J": j[ii, jj],
        "Kq": qk[ii, jj], "gapq": qgap[ii, jj],
        "dif": macd["dif"][ii, jj], "dea": macd["dea"][ii, jj],
        "hist": macd["hist"][ii, jj],
        "macd_bull": macd["states"]["DIF>DEA"][ii, jj],
        "zero": zero[ii, jj],
        "above60": above60[ii, jj],
        "ret5": ret5[ii, jj],
        "std20": st["std20"][ii, jj],
        "mkt": mkt_state[ii],
        "wkdj": wk["state"][ii, jj], "mkdj": mk["state"][ii, jj],
        "mg5": mg5[ii, jj], "mg20": mg20[ii, jj],
        "md5": md5[ii, jj],
        "top_div": top_div[ii, jj], "bot_div": bot_div[ii, jj],
    })
    for h in (5, 20, 60):
        ek[f"ex{h}"] = exs[h][ii, jj]
    ek["净超额"] = ek["ex20"] - COST
    ek["di"], ek["ci"] = ii, jj
    ek.to_parquet(os.path.join(RES, "KDJ_事件_KDJ金叉.parquet"), index=False)

    rows = []
    def add_kdj(lab, sel):
        rows.append(dict(过滤=lab, **ev_stat(ek[sel])))
    add_kdj("全部KDJ金叉", pd.Series(True, index=ek.index))
    add_kdj("MACD多", ek.macd_bull.astype(bool))
    add_kdj("MACD空", ~ek.macd_bull.astype(bool))
    add_kdj("MACD双上", ek.zero == 0)
    add_kdj("MACD双下", ek.zero == 3)
    add_kdj("MACD零轴上(DIF>0)", ek.dif > 0)
    add_kdj("MACD零轴下(DIF<=0)", ek.dif <= 0)
    add_kdj("价格>MA60", ek.above60.astype(bool))
    add_kdj("MACD多+价格>MA60", ek.macd_bull.astype(bool) & ek.above60.astype(bool))
    add_kdj("MACD空+价格>MA60", ~ek.macd_bull.astype(bool) & ek.above60.astype(bool))
    add_kdj("前5日内MACD金叉", ek.mg5.astype(bool))
    add_kdj("前20日内MACD金叉", ek.mg20.astype(bool))
    add_kdj("前5日内无MACD金叉", ~ek.mg5.astype(bool))
    add_kdj("前5日内MACD死叉", ek.md5.astype(bool))
    add_kdj("KQ1(最低)", ek.Kq == 0)
    add_kdj("KQ5(最高)", ek.Kq == 4)
    add_kdj("KDJ顶背离", ek.top_div.astype(bool))
    add_kdj("KDJ底背离", ek.bot_div.astype(bool))
    add_kdj("周线KDJ多", ek.wkdj.astype(bool))
    add_kdj("月线KDJ多", ek.mkdj.astype(bool))
    ekr = pd.DataFrame(rows)
    ekr["净超额"] = ekr["毛超额"] - COST
    ekr.to_csv(os.path.join(RES, "KDJ_事件_KDJ金叉.csv"), index=False, encoding="utf-8-sig")
    print("\nKDJ金叉过滤")
    print(ekr[["过滤", "信号数", "毛超额", "净超额", "t值", "前期", "后期", "年数", "正年数"]].to_string(index=False))

    # KDJ 金叉内部：MACD 过滤增量
    rows = []
    def add_contrast_k(lab, a, b):
        r = contrast(ek[a], ek[b])
        rows.append(dict(对比=lab, **r))
    add_contrast_k("MACD多 vs MACD空", ek.macd_bull.astype(bool), ~ek.macd_bull.astype(bool))
    add_contrast_k("MACD双上 vs 双下", ek.zero == 0, ek.zero == 3)
    add_contrast_k("价格>MA60 vs <=MA60", ek.above60.astype(bool), ~ek.above60.astype(bool))
    add_contrast_k("前5日内MACD金叉 vs 无", ek.mg5.astype(bool), ~ek.mg5.astype(bool))
    add_contrast_k("前20日内MACD金叉 vs 无", ek.mg20.astype(bool), ~ek.mg20.astype(bool))
    add_contrast_k("KQ1 vs KQ2-4", ek.Kq == 0, ek.Kq.isin([1, 2, 3]))
    add_contrast_k("KQ5 vs KQ2-4", ek.Kq == 4, ek.Kq.isin([1, 2, 3]))
    ekc = pd.DataFrame(rows)
    ekc.to_csv(os.path.join(RES, "KDJ_事件_KDJ金叉增量.csv"), index=False, encoding="utf-8-sig")
    print("\nKDJ金叉内部增量")
    print(ekc.to_string(index=False))

    # ---------- 双金叉/时序 ----------
    print("\n双金叉")
    double = e[e.kdj_gold_today.astype(bool)]
    double_prev5 = e[e.kdj_gold_5d.astype(bool) & ~e.kdj_gold_today.astype(bool)]
    double_none = e[~e.kdj_gold_5d.astype(bool)]
    kdj_after_macd5 = ek[ek.mg5.astype(bool)]
    kdj_after_macd20 = ek[ek.mg20.astype(bool)]
    kdj_after_macd_none = ek[~ek.mg20.astype(bool)]
    for lab, sub in [
        ("同日双金叉", double), ("MACD金叉+前5日KDJ金叉", double_prev5),
        ("MACD金叉+前5日无KDJ金叉", double_none),
        ("KDJ金叉+前5日MACD金叉", kdj_after_macd5),
        ("KDJ金叉+前20日MACD金叉", kdj_after_macd20),
        ("KDJ金叉+前20日无MACD金叉", kdj_after_macd_none),
    ]:
        print(lab, ev_stat(sub))
    # 逐年：双金叉/过滤
    rows = []
    for lab, sub in [
        ("MACD金叉全部", e), ("MACD金叉+KDJ多", e[e["K>D"].astype(bool)]),
        ("MACD金叉+KDJ空", e[~e["K>D"].astype(bool)]),
        ("同日双金叉", double), ("MACD金叉+KQ1", e[e.Kq == 0]),
        ("MACD金叉+KQ5", e[e.Kq == 4]),
        ("KDJ金叉全部", ek), ("KDJ金叉+MACD多", ek[ek.macd_bull.astype(bool)]),
        ("KDJ金叉+MACD空", ek[~ek.macd_bull.astype(bool)]),
    ]:
        dm = daily_mean(sub)
        for y, val in dm.groupby(dm.index.year).mean().items():
            rows.append(dict(类别=lab, 年份=int(y), 日均超额=float(val), 信号数=int((sub.date.dt.year == y).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "KDJ_事件_逐年.csv"), index=False, encoding="utf-8-sig")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
