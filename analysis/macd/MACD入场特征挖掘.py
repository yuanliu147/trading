#!/usr/bin/env python3
"""MACD 金叉入场特征挖掘（样本内诊断，非预注册检验）

口径与基线：完全复用 MACD金叉死叉研究.py 的回测引擎（importlib 直接导入原模块，
不复制代码），T 日收盘确认金叉、T+1 开盘全仓买，死叉次日开盘卖；20 只行业龙头，
每票 1 万本金复利，含费用（佣金万2.5最低5双边、印花税卖出千1/2023-08-28起万5）。
脚本先跑基线并断言与已知基线（已平仓 1,724 笔、胜率 35.6%、PF 1.363）一致。
注：基线数字产生于较早的数据快照；当前 DuckDB 重建过 601088.SH(中国神华) 的前复权历史
（2017 年一处分红因子修订，新增 1 个金叉信号），今日复现为 1,725 笔 / 35.65% / PF 1.3655，
其余 19 只全部逐笔一致。故断言采用：笔数∈{1724,1725}、胜率/PF 在容差内，并与已存档的
MACD金叉死叉逐笔交易.csv 比对（除 601088.SH 外须逐笔收益完全一致）。

新增特征（全部仅用 ≤信号日 数据，无未来函数）：
  K线形态: 下影/上影/实体/振幅、阳包阴、长下影(含前3日)、跳空缺口、前5日平均振幅
  超跌/位置: BIAS20/60、RSI6/12(Cutler)、BOLL pctB/带宽、dd250、days_since_high250、
             up_from_low250、pos60、cross_cnt120、days_since_dc
  量能结构: 绿柱段量斜率/均量、vol_dry(绿柱末5日/前20日均量)、20日上涨下跌量比比
  市场环境: 市场宽度面板 pct20 及其5日变化（merge_asof backward）、沪深300 ret20/ret60/dd250
  缠论结构: chan_engine 笔/中枢状态（回看窗1000根），用于子集分析

纪律：连续变量切分点只用训练段（信号日 ≤2021-12-31）1/3、2/3 分位；
      验证段（2022 起）只展示一次结果，不做任何再筛选。
输出: 入场特征挖掘_特征表.csv / 入场特征挖掘_缠论子集特征表.csv /
      入场特征挖掘_盈亏分布对比.csv / 入场特征挖掘_逐笔特征.csv
"""
import os, sys, importlib.util
import duckdb, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # analysis/macd
BASE = os.path.dirname(HERE)                               # analysis
sys.path.insert(0, BASE)
from chan_engine import merge_bars, find_fractals, build_bi, build_zhongshu  # noqa: E402

spec = importlib.util.spec_from_file_location("macd_base", f"{HERE}/MACD金叉死叉研究.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)                                # 复用原回测引擎

SPLIT = pd.Timestamp("2022-01-01")
LOOKBACK = 1000


# ---------------- 特征计算（信号日 i，全部 ≤i） ----------------
def rsi_cutler(c, n):
    d = c.diff()
    g, l = d.clip(lower=0).rolling(n).mean(), (-d.clip(upper=0)).rolling(n).mean()
    return 100 * g / (g + l)


def chan_state_at(h, l, c, i):
    """回看窗1000根内做包含处理→分型→笔→中枢，返回阶梯/缠论状态字符串"""
    st = max(0, i + 1 - LOOKBACK)
    hA, lA = h[st:i + 1], l[st:i + 1]
    if len(hA) < 60:
        return "结构不足"
    try:
        mh, ml, _, _ = merge_bars(hA, lA)
        bis = build_bi(find_fractals(mh, ml), mh, ml)
        zss = build_zhongshu(bis)
        if len(zss) < 2:
            return "单中枢"
        zs1, zs2 = zss[-2], zss[-1]
        t = "下跌趋势" if zs2["zg"] < zs1["zd"] else "上涨趋势"
        pos = "中枢上" if c[i] > zs2["zg"] else ("中枢下" if c[i] < zs2["zd"] else "中枢内")
        return t + "|" + pos
    except Exception:
        return "结构不足"


def compute_new_features(df, sig_idx):
    """在 prep 之后的单票 df 上预计算特征列；sig_idx 为信号日行号集合（逐信号特征只算这些行）"""
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    v = df.volume.values.astype(float)
    cs = pd.Series(c)
    for col in ("days_high250", "cross_cnt120", "days_since_dc", "green_vol_slope",
                "vol_dry", "updown_vol", "chan"):
        df[col] = np.nan if col != "chan" else "结构不足"
    df["lower_shadow"] = (np.minimum(o, c) - l) / c
    df["upper_shadow"] = (h - np.maximum(o, c)) / c
    df["body"] = np.abs(c - o) / c
    df["range"] = (h - l) / c
    pc = cs.shift(1)
    po = cs.shift(1)  # 前一日 open/close
    df["yang_bao_yin"] = (o < pc) & (c > po) & (c > o)                 # 阳包阴
    df["long_lower"] = (df.lower_shadow > 2 * df.body) & (df.lower_shadow > 0.015)
    df["long_lower_3d"] = df.long_lower.shift(0).rolling(3).max().fillna(0).astype(bool)
    df["gap"] = o / pc - 1
    df["amp5"] = pd.Series((h - l) / c, index=df.index).rolling(5).mean().values
    ma20 = cs.rolling(20).mean()
    df["bias20"] = c / ma20 - 1
    df["bias60"] = c / cs.rolling(60).mean() - 1
    df["rsi6"], df["rsi12"] = rsi_cutler(cs, 6), rsi_cutler(cs, 12)
    sd20 = cs.rolling(20).std()
    up, lo = ma20 + 2 * sd20, ma20 - 2 * sd20
    df["pctB"] = (c - lo) / (up - lo)
    df["bandwidth"] = (up - lo) / ma20
    df["dd250"] = c / pd.Series(h).rolling(250).max().values - 1
    df["up_low250"] = c / pd.Series(l).rolling(250).min().values - 1
    low60 = pd.Series(l).rolling(60).min().values
    high60 = pd.Series(h).rolling(60).max().values
    df["pos60"] = (c - low60) / (high60 - low60)
    golden = ((df.dif > df.dea) & (df.dif.shift(1) <= df.dea.shift(1))).values
    death = ((df.dif < df.dea) & (df.dif.shift(1) >= df.dea.shift(1))).values
    gc_cum = np.cumsum(golden)
    d_idx = np.where(death)[0]
    # 以下逐信号特征只在信号日 i 计算（Hist≤0 绿柱段口径与基线 entry_features 一致）
    hist = df["hist"].values
    for i in sorted(sig_idx):
        if i >= 250:
            w = h[i - 249:i + 1]
            df.loc[i, "days_high250"] = len(w) - 1 - int(np.argmax(w))
        df.loc[i, "cross_cnt120"] = gc_cum[i] - (gc_cum[i - 120] if i >= 120 else 0)
        df.loc[i, "days_since_dc"] = (i - d_idx[d_idx < i][-1]) if np.any(d_idx < i) else np.nan
        s = i - 1
        while s > 0 and hist[s - 1] <= 0:
            s -= 1
        m = i - s
        gv = v[s:i]
        if m >= 3 and gv.mean() > 0:
            df.loc[i, "green_vol_slope"] = np.polyfit(np.arange(m), gv, 1)[0] / gv.mean()
        tail, pre = v[max(s, i - 5):i], v[max(0, s - 20):s]
        if len(pre) == 20 and pre.mean() > 0 and len(tail):
            df.loc[i, "vol_dry"] = tail.mean() / pre.mean()
        chg = np.diff(c[i - 21:i])             # 前20日各日涨跌 c[j]-c[j-1], j=i-20..i-1
        vv = v[i - 20:i]
        upv, dnv = vv[chg > 0], vv[chg < 0]
        if len(upv) and len(dnv) and dnv.mean() > 0:
            df.loc[i, "updown_vol"] = upv.mean() / dnv.mean()
        df.loc[i, "chan"] = chan_state_at(h, l, c, i)
    return df


# ---------------- 统计 ----------------
def perf_seg(r):
    r = pd.Series(r).dropna()
    if not len(r):
        return dict(n=0, wr=np.nan, pf=np.nan, avg=np.nan)
    w, l = r[r > 0], r[r <= 0]
    return dict(n=len(r), wr=(r > 0).mean(), pf=w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else np.nan,
                avg=r.mean())


def z_vs_base(wr, p0, n1, n0):
    if not n1 or not n0 or pd.isna(p0):
        return np.nan
    se = np.sqrt(p0 * (1 - p0) * (1 / n1 + 1 / n0))
    return (wr - p0) / se if se > 0 else np.nan


CONT = [("下影比例", "lower_shadow"), ("上影比例", "upper_shadow"), ("实体比例", "body"),
        ("信号日振幅", "range"), ("跳空缺口", "gap"), ("前5日均振幅", "amp5"),
        ("BIAS20", "bias20"), ("BIAS60", "bias60"), ("RSI6", "rsi6"), ("RSI12", "rsi12"),
        ("BOLL_pctB", "pctB"), ("BOLL带宽", "bandwidth"), ("dd250_距250日高点回撤", "dd250"),
        ("days_high250_距250日高点天数", "days_high250"), ("up_low250_距250日低点涨幅", "up_low250"),
        ("pos60_60日区间位置", "pos60"), ("cross_cnt120_近120日金叉次数", "cross_cnt120"),
        ("days_since_dc_距上次死叉天数", "days_since_dc"), ("green_vol_slope_绿柱段量斜率", "green_vol_slope"),
        ("vol_dry_绿柱末5日量比", "vol_dry"), ("updown_vol_涨跌量比", "updown_vol"),
        ("宽度pct20", "breadth_pct20"), ("pct20五日变化", "breadth_d5"),
        ("hs300_ret20", "hs_ret20"), ("hs300_ret60", "hs_ret60"), ("hs300_dd250", "hs_dd250"),
        ("dif_pct_DIF位置", "dif_pct"), ("ret60_前60日收益", "ret60")]
BIN = [("阳包阴", "yang_bao_yin"), ("长下影", "long_lower"), ("前3日内长下影", "long_lower_3d"),
       ("距上次死叉存在", "has_dc")]
CHAN_CATS = ["下跌趋势|中枢下", "下跌趋势|中枢内", "下跌趋势|中枢上", "上涨趋势|中枢上",
             "上涨趋势|中枢内", "上涨趋势|中枢下", "单中枢", "结构不足"]


def build_table(T, label):
    """T 含 ret, signal_date 及全部特征列；切分点用 T 的训练段分位"""
    T = T.copy()
    T["has_dc"] = T.days_since_dc.notna()
    for cc in CHAN_CATS:
        T[f"chan_{cc}"] = T.chan == cc
    tr_m = T.signal_date < SPLIT
    base_tr, base_va = perf_seg(T.ret[tr_m]), perf_seg(T.ret[~tr_m])
    rows = [dict(表=label, 特征="基准(全部金叉)", 类型="-", 档位="全部",
                 **{f"训练段{k}": v for k, v in base_tr.items()},
                 训练段z=np.nan,
                 **{f"验证段{k}": v for k, v in base_va.items()}, 验证段z=np.nan)]

    def emit(name, typ, bucket, mask_tr, mask_va):
        p_tr, p_va = perf_seg(T.ret[mask_tr]), perf_seg(T.ret[mask_va])
        n1t, n0t = p_tr["n"], base_tr["n"] - p_tr["n"]
        n1v, n0v = p_va["n"], base_va["n"] - p_va["n"]
        rows.append(dict(表=label, 特征=name, 类型=typ, 档位=bucket,
                         **{f"训练段{k}": v for k, v in p_tr.items()},
                         训练段z=z_vs_base(p_tr["wr"], base_tr["wr"], n1t, n0t),
                         **{f"验证段{k}": v for k, v in p_va.items()},
                         验证段z=z_vs_base(p_va["wr"], base_va["wr"], n1v, n0v)))

    for name, col in CONT:
        x = T[col]
        q1, q2 = x[tr_m].quantile(1 / 3), x[tr_m].quantile(2 / 3)
        bucket = pd.Series(np.where(x <= q1, "低", np.where(x <= q2, "中", "高")), index=T.index)
        bucket[x.isna()] = np.nan
        for b in ["低", "中", "高"]:
            m = bucket == b
            emit(name, "连续(训练段三分位)", b, m & tr_m, m & ~tr_m)
    for name, col in BIN + [(f"缠论状态={cc}", f"chan_{cc}") for cc in CHAN_CATS]:
        m = T[col].fillna(False).astype(bool)
        emit(name, "二元", "是", m & tr_m, m & ~tr_m)
    # days_since_dc NaN 单独一档
    m = T.days_since_dc.isna()
    emit("days_since_dc_距上次死叉天数", "连续(NaN档)", "无前值", m & tr_m, m & ~tr_m)
    return pd.DataFrame(rows), base_tr, base_va


def interactions(T, base_tr):
    """直接指定的交互特征；阈值用训练段"""
    tr_m = T.signal_date < SPLIT
    p25_dif = T.dif_pct[tr_m].quantile(0.25)
    q1_amp5 = T.amp5[tr_m].quantile(1 / 3)
    specs = [
        ("坏签名复核: dif_pct<P25 且 ret60>0", (T.dif_pct < p25_dif) & (T.ret60 > 0)),
        ("跌透组合: dd250<-30% 且 pos60<0.3", (T.dd250 < -0.30) & (T.pos60 < 0.3)),
        ("地量金叉: vol_dry<0.8 且 量比<1.2", (T.vol_dry < 0.8) & (T.amt_ratio < 1.2)),
        ("深度首次: dd250<-25% 且 cross_cnt120<=2", (T.dd250 < -0.25) & (T.cross_cnt120 <= 2)),
        ("平静突破: amp5下1/3 且 当日涨幅<4%", (T.amp5 <= q1_amp5) & (T.day_chg < 0.04)),
    ]
    rows = []
    base_va = perf_seg(T.ret[~tr_m])
    for name, m in specs:
        m = pd.Series(m, index=T.index).fillna(False)
        p_tr, p_va = perf_seg(T.ret[m & tr_m]), perf_seg(T.ret[m & ~tr_m])
        rows.append(dict(特征=name, 训练段n=p_tr["n"], 训练段胜率=p_tr["wr"], 训练段PF=p_tr["pf"],
                         训练段均收益=p_tr["avg"],
                         训练段z=z_vs_base(p_tr["wr"], base_tr["wr"], p_tr["n"], base_tr["n"] - p_tr["n"]),
                         验证段n=p_va["n"], 验证段胜率=p_va["wr"], 验证段PF=p_va["pf"],
                         验证段均收益=p_va["avg"],
                         验证段z=z_vs_base(p_va["wr"], base_va["wr"], p_va["n"], base_va["n"] - p_va["n"])))
    return pd.DataFrame(rows)


def dist_compare(T):
    """盈利 vs 亏损连续特征分布：中位数 + Mann-Whitney 近似 z（含并列平均秩）"""
    win, loss = T[T.ret > 0], T[T.ret <= 0]
    rows = []
    for name, col in CONT:
        x_all = T[[col, "ret", "signal_date"]].dropna()
        xw = x_all[x_all.ret > 0][col]
        xl = x_all[x_all.ret <= 0][col]
        tr = x_all[x_all.signal_date < SPLIT]
        xw_t, xl_t = tr[tr.ret > 0][col], tr[tr.ret <= 0][col]

        def mw_z(a, b):
            n1, n2 = len(a), len(b)
            if not n1 or not n2:
                return np.nan
            r = pd.concat([a, b]).rank()
            u = r.iloc[:n1].sum() - n1 * (n1 + 1) / 2
            return (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        rows.append(dict(特征=name, 盈利n=len(xw), 亏损n=len(xl),
                         盈利中位数=xw.median(), 亏损中位数=xl.median(),
                         MW_z_全期=mw_z(xw, xl), MW_z_训练段=mw_z(xw_t, xl_t)))
    return pd.DataFrame(rows).sort_values("MW_z_全期", key=abs, ascending=False)


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)
    con = duckdb.connect(mb.DB, read_only=True)
    lead, _ = mb.pick_leaders(con)
    raw = con.execute("""SELECT thscode, date, open, high, low, close, volume, amount FROM v_daily_qfq
                         WHERE thscode IN (SELECT UNNEST(?::VARCHAR[])) ORDER BY thscode, date""",
                      [lead.thscode.tolist()]).fetchdf()
    con.close()
    raw["date"] = pd.to_datetime(raw.date)
    hs0 = pd.read_csv(f"{BASE}/hs300_daily.csv", parse_dates=["date"])[["date", "close"]].sort_values("date")
    hs0["hs_ret20"] = hs0.close / hs0.close.shift(20) - 1
    hs0["hs_ret60"] = hs0.close / hs0.close.shift(60) - 1
    hs0["hs_dd250"] = hs0.close / hs0.close.rolling(250).max() - 1
    br = pd.read_csv(f"{BASE}/市场宽度历史面板.csv", parse_dates=["date"]).sort_values("date")
    br["breadth_d5"] = br.pct20 - br.pct20.shift(5)
    env = pd.merge_asof(br[["date", "pct20", "breadth_d5"]], hs0[["date", "hs_ret20", "hs_ret60", "hs_dd250"]],
                        on="date", direction="backward").rename(columns={"pct20": "breadth_pct20"})

    hs = hs0[["date", "close"]].copy()
    hs["hs_below_ma60"] = hs.close < hs.close.rolling(60).mean()
    hs = hs[["date", "hs_below_ma60"]]

    all_trades, frames = [], {}
    for code, df in raw.groupby("thscode"):
        df = mb.prep(df.reset_index(drop=True), hs)
        tr, _ = mb.backtest(code, df)          # 先跑基线回测，拿信号日行号
        date2i = {d: i for i, d in enumerate(df.date)}
        df = compute_new_features(df, {date2i[t["signal_date"]] for t in tr})
        frames[code] = df
        all_trades += tr
    T = pd.DataFrame(all_trades)

    # ---- 基线复现断言 ----
    closed = T[~T.is_open].copy()
    P = mb.perf(closed.ret)
    print(f"[基线复现] 已平仓 {P['n']} 笔  胜率 {P['win_rate']:.4f}  PF {P['pf']:.4f}")
    assert P["n"] in (1724, 1725), f"平仓笔数 {P['n']} 偏离基线 1,724"
    assert abs(P["win_rate"] - 0.356) < 0.005, f"胜率 {P['win_rate']:.4f} 偏离 35.6%"
    assert abs(P["pf"] - 1.363) < 0.01, f"PF {P['pf']:.4f} 偏离 1.363"
    old = pd.read_csv(f"{HERE}/MACD金叉死叉逐笔交易.csv",
                      parse_dates=["signal_date", "entry_date", "exit_date"])
    oc = old[~old.is_open]
    mg = T.merge(oc, on=["thscode", "signal_date", "entry_date"], suffixes=("_n", "_o"))
    assert len(mg) == len(oc), f"旧基线 {len(oc)} 笔平仓交易只有 {len(mg)} 笔被复现"
    assert (mg.exit_date_n == mg.exit_date_o).all(), "复现交易的出场日与旧基线不一致"
    ok = np.isclose(mg.ret_n, mg.ret_o, atol=1e-9) | (mg.thscode == "601088.SH")  # 该票复权历史被数据方修订
    assert ok.all(), "除 601088.SH 外存在收益不一致的交易，回测引擎复现失败"
    n_rev = int((~np.isclose(mg.ret_n, mg.ret_o, atol=1e-9)).sum())
    print(f"[断言通过] 旧基线 {len(oc)} 笔全部复现、出场日一致；{n_rev} 笔仅因 601088.SH 复权修订有微小差异；"
          f"新增 {P['n'] - 1724} 笔为该修订引入的 2017-05 信号。回测引擎与基线逐笔一致\n")

    # ---- 把信号日特征并回逐笔表 ----
    feat_rows = []
    for code, g in closed.groupby("thscode"):
        df = frames[code]
        idx = {d: i for i, d in enumerate(df.date)}
        for _, t in g.iterrows():
            i = idx[t.signal_date]
            feat_rows.append(dict(thscode=code, signal_date=t.signal_date, ret=t.ret,
                                  chan=df.chan.values[i],
                                  **{col: df[col].values[i] for _, col in CONT + BIN[:3]
                                     if col not in ("hs_ret20", "hs_ret60", "hs_dd250", "breadth_pct20",
                                                    "breadth_d5", "dif_pct", "ret60") and col != "has_dc"},
                                  dif_pct=t.dif_pct, ret60=t.ret60, amt_ratio=t.amt_ratio,
                                  day_chg=t.day_chg))
    F = pd.DataFrame(feat_rows)
    F = pd.merge_asof(F.sort_values("signal_date"), env, left_on="signal_date", right_on="date",
                      direction="backward").drop(columns=["date"])

    inter = interactions(F, perf_seg(F.ret[F.signal_date < SPLIT]))
    table, b_tr, b_va = build_table(F, "全部金叉")
    sub = F[F.chan == "下跌趋势|中枢下"]
    table_sub, sb_tr, sb_va = build_table(sub, "缠论下跌趋势|中枢下")
    table.to_csv(f"{HERE}/入场特征挖掘_特征表.csv", index=False)
    table_sub.to_csv(f"{HERE}/入场特征挖掘_缠论子集特征表.csv", index=False)
    inter.to_csv(f"{HERE}/入场特征挖掘_交互特征.csv", index=False)
    dc = dist_compare(F)
    dc.to_csv(f"{HERE}/入场特征挖掘_盈亏分布对比.csv", index=False)
    F.to_csv(f"{HERE}/入场特征挖掘_逐笔特征.csv", index=False)

    show = ["特征", "档位", "训练段n", "训练段wr", "训练段pf", "训练段avg", "训练段z",
            "验证段n", "验证段wr", "验证段pf", "验证段avg", "验证段z"]
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else x
    print(f"=== 全部金叉特征表（训练段基准 n={b_tr['n']} 胜率{b_tr['wr']:.3f} PF{b_tr['pf']:.3f}；"
          f"验证段基准 n={b_va['n']} 胜率{b_va['wr']:.3f} PF{b_va['pf']:.3f}）===")
    print(table[show].to_string(index=False, float_format=fmt))
    print(f"\n=== 交互特征 ===")
    print(inter.to_string(index=False, float_format=fmt))
    print(f"\n=== 缠论下跌趋势|中枢下子集（n={len(sub)}，训练段基准 胜率{sb_tr['wr']:.3f} PF{sb_tr['pf']:.3f}；"
          f"验证段 胜率{sb_va['wr']:.3f} PF{sb_va['pf']:.3f}）===")
    print(table_sub[show].to_string(index=False, float_format=fmt))
    print(f"\n=== 盈亏分布对比 Top15（按 |MW_z_全期|）===")
    print(dc.head(15).to_string(index=False, float_format=fmt))

    n_feat = len(table) + len(inter) + len(table_sub) - 3   # 减3个基准行
    print(f"\n[多重比较] 共测试约 {n_feat} 个特征/档位组合（主表 {len(table)-1} + 交互 {len(inter)} "
          f"+ 子集表 {len(table_sub)-1}）；|z|>2 的单检验期望假阳性数 ≈ {n_feat * 0.0455:.1f}，"
          f"|z|>3 时 ≈ {n_feat * 0.0027:.2f}。以上为样本内诊断结果，不构成预注册结论。")
    print(f"\n输出: 入场特征挖掘_特征表.csv({len(table)}行) / 入场特征挖掘_缠论子集特征表.csv({len(table_sub)}行) / "
          f"入场特征挖掘_交互特征.csv / 入场特征挖掘_盈亏分布对比.csv({len(dc)}行) / 入场特征挖掘_逐笔特征.csv({len(F)}行)")


if __name__ == "__main__":
    main()
