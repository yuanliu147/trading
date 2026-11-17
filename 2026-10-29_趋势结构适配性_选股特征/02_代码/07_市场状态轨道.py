#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道二：市场状态条件化（用户 Q4）
项目：2026-10-29_趋势结构适配性_选股特征
预注册：§15（L1 全知上限 → L2 实时可实现 → L3 识别成本；5 套状态定义；§15.5 量化参与命题）

关键纪律：
  L1 使用未来信息计算"状态已知时能赚多少"的上限，**永远不得当策略绩效引用**，
  输出一律带 _ORACLE_上限_不可实现 标记。若 L1 上限本身不成立，命题死亡，不必研究识别。

动作代理映射（§15.3 的可执行定义用已算出的信号代理，如实披露）：
  A1_趋势跟随 <- S1_BREAK_N55     （突破进场 = 跟随趋势）
  A2_波段     <- S3_PULL_K10      （上升趋势内回撤至均线进场 = 区间低吸）
  A3_反弹     <- R1_NEWLOW_N55    （创 55 日新低 = 深度回撤后进场）
  A4_空仓     <- 超额恒为 0

产出：03_结果/市场状态_全知上限.csv / 市场状态_状态持续性.csv / 市场状态_汇总.json
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

母项目 = Path("/Users/admin/WorkSpace/trading/trading_趋势结构")
冻结目录 = 母项目 / "01_数据" / "frozen"
本项目 = Path("/Users/admin/WorkSpace/trading/2026-10-29_趋势结构适配性_选股特征")
数据目录 = 本项目 / "01_数据"
结果目录 = 本项目 / "03_结果"

精简事件 = 数据目录 / "事件_精简_h20_EW.parquet"
面板文件 = 数据目录 / "面板_2008预热_2020.parquet"

研究期起 = "2010-01-01"
研究期止 = "2020-12-31"
洗牌次数 = 1000
种子 = 20261029
预热年 = 2            # L2 期望选择用的最短历史

动作映射 = {
    "A1_趋势跟随": "S1_BREAK_N55",
    "A2_波段": "S3_PULL_K10",
    "A3_反弹": "R1_NEWLOW_N55",
    "A4_空仓": None,
}
时代 = {"E1": ("2010-01-01", "2012-12-31"), "E2": ("2013-01-01", "2015-12-31"),
        "E3": ("2016-01-01", "2017-12-31"), "E4": ("2018-01-01", "2020-12-31")}


def 建状态(con) -> pd.DataFrame:
    """5 套状态定义，全部只用 <= T-1 数据（§15.3）。返回 date × 各定义的状态标签。"""
    mkt = con.execute(f"""
        SELECT date, ew_ret, hs300_ret, pct_above_ma60, med_ret
        FROM read_parquet('{冻结目录}/market_daily.parquet')
        WHERE date >= DATE '2008-07-01' AND date <= DATE '{研究期止}' ORDER BY date
    """).fetch_df()
    idx = con.execute(f"""
        SELECT date, close FROM read_parquet('{冻结目录}/index_daily/*/*.parquet')
        WHERE symbol = '000300.SH' AND date >= DATE '2008-07-01'
          AND date <= DATE '{研究期止}' ORDER BY date
    """).fetch_df()
    # 横截面离散度（R4）：个股日收益的横截面标准差
    disp = con.execute(f"""
        SELECT date, stddev_samp(pct_chg) AS disp
        FROM read_parquet('{面板文件}')
        WHERE NOT is_suspended AND date >= DATE '2008-07-01'
        GROUP BY date ORDER BY date
    """).fetch_df()

    d = mkt.merge(idx, on="date", how="left").merge(disp, on="date", how="left")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    # R1 指数均线：价相对 MA200 + MA200 斜率
    c = d["close"].astype(float)
    ma200 = c.rolling(200).mean()
    斜率正 = ma200 > ma200.shift(20)
    上方 = c > ma200
    d["R1_指数均线"] = np.select(
        [上方 & 斜率正, (~上方) & (~斜率正)], ["牛", "熊"], default="震荡")

    # R2 市场宽度：pct_above_ma60 的三分位（滚动 250 日自身分位，避免用未来）
    br = d["pct_above_ma60"].astype(float)
    br_pct = br.rolling(250).rank(pct=True)
    d["R2_市场宽度"] = np.select(
        [br_pct >= 2 / 3, br_pct <= 1 / 3], ["牛", "熊"], default="震荡")

    # R3 波动率状态：指数 60 日已实现波动的滚动分位（高波动=熊，低波动=牛）
    r = c.pct_change()
    vol = r.rolling(60).std()
    vol_pct = vol.rolling(250).rank(pct=True)
    d["R3_波动率"] = np.select(
        [vol_pct <= 1 / 3, vol_pct >= 2 / 3], ["牛", "熊"], default="震荡")

    # R4 横截面离散度：高离散=趋势可做（记"牛"），低离散=震荡
    dp = d["disp"].astype(float).rolling(20).mean()
    dp_pct = dp.rolling(250).rank(pct=True)
    d["R4_离散度"] = np.select(
        [dp_pct >= 2 / 3, dp_pct <= 1 / 3], ["牛", "熊"], default="震荡")

    # R5 风格差：等权 − 沪深300（60 日累计），小盘占优=牛
    sty = (d["ew_ret"].fillna(0) - d["hs300_ret"].fillna(0)).rolling(60).sum()
    sty_pct = sty.rolling(250).rank(pct=True)
    d["R5_风格差"] = np.select(
        [sty_pct >= 2 / 3, sty_pct <= 1 / 3], ["牛", "熊"], default="震荡")

    # 全部状态列 shift(1)：T 日决策只能用 T-1 及更早（§15.3）
    状态列 = ["R1_指数均线", "R2_市场宽度", "R3_波动率", "R4_离散度", "R5_风格差"]
    for s in 状态列:
        d[s] = pd.Series(d[s]).shift(1)
    d = d[(d["date"] >= 研究期起) & (d["date"] <= 研究期止)]
    return d[["date"] + 状态列].reset_index(drop=True), 状态列


def 取动作日收益(con) -> pd.DataFrame:
    """每个动作在每个信号日的事件平均超额（日级）。A4 空仓恒 0。"""
    代码 = [v for v in 动作映射.values() if v]
    占位 = ",".join(f"'{c}'" for c in 代码)
    # 精简事件表的超额列名在不同生成版本下可能是 超额 或 超额EW_h20，自适应
    列 = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{精简事件}')").fetchall()]
    超额列 = "超额" if "超额" in 列 else "超额EW_h20"
    assert 超额列 in 列, f"精简事件表缺少超额列，实有列：{列}"
    df = con.execute(f"""
        SELECT 信号日 AS date, 信号代码, avg({超额列}) AS 超额, count(*) AS n
        FROM read_parquet('{精简事件}')
        WHERE 信号代码 IN ({占位})
        GROUP BY 信号日, 信号代码""").fetch_df()
    df["date"] = pd.to_datetime(df["date"])
    反查 = {v: k for k, v in 动作映射.items() if v}
    df["动作"] = df["信号代码"].map(反查)
    return df


def 状态动作矩阵(日收益: pd.DataFrame, 状态: pd.DataFrame, 定义: str) -> pd.DataFrame:
    """每个 (状态值, 动作) 的事件加权平均超额 + 事件数 + 日数。"""
    m = 日收益.merge(状态[["date", 定义]], on="date", how="inner").dropna(subset=[定义])
    g = m.groupby([定义, "动作"]).apply(
        lambda x: pd.Series({
            "加权超额": np.average(x["超额"], weights=x["n"]),
            "事件数": int(x["n"].sum()), "日数": int(len(x))}),
        include_groups=False).reset_index()
    # A4 空仓：每个状态值补 0
    for sv in m[定义].dropna().unique():
        g = pd.concat([g, pd.DataFrame([{
            定义: sv, "动作": "A4_空仓", "加权超额": 0.0,
            "事件数": int((状态[定义] == sv).sum()), "日数": int((状态[定义] == sv).sum())}])],
            ignore_index=True)
    g = g.rename(columns={定义: "状态值"})
    g["状态定义"] = 定义
    return g


def 全知上限(矩阵: pd.DataFrame, 状态: pd.DataFrame, 定义: str) -> dict:
    """L1：状态已知 + 每状态选事后最优动作。用未来信息，是上限，不可实现。"""
    权重 = 状态[定义].value_counts(normalize=True).to_dict()
    best, 组合 = {}, 0.0
    for sv, g in 矩阵.groupby("状态值"):
        r = g.loc[g["加权超额"].idxmax()]
        best[str(sv)] = {"最优动作": r["动作"], "超额": float(r["加权超额"])}
        组合 += 权重.get(sv, 0.0) * float(r["加权超额"])
    单一 = 矩阵.groupby("动作").apply(
        lambda x: np.average(x["加权超额"],
                             weights=[权重.get(s, 0) for s in x["状态值"]]),
        include_groups=False)
    最好单一 = float(单一.max())
    最好单一动作 = str(单一.idxmax())
    # 用户命题的对角组合：牛->A1，震荡->A2，熊->A3 / A4 取较优
    对角 = {"牛": "A1_趋势跟随", "震荡": "A2_波段", "熊": "A3_反弹"}
    对角值 = 0.0
    对角明细 = {}
    for sv, act in 对角.items():
        sub = 矩阵[(矩阵["状态值"] == sv) & (矩阵["动作"] == act)]
        v = float(sub["加权超额"].iloc[0]) if len(sub) else 0.0
        if sv == "熊":
            空 = 矩阵[(矩阵["状态值"] == sv) & (矩阵["动作"] == "A4_空仓")]
            v2 = float(空["加权超额"].iloc[0]) if len(空) else 0.0
            if v2 > v:
                act, v = "A4_空仓", v2
        对角明细[sv] = {"动作": act, "超额": v}
        对角值 += 权重.get(sv, 0.0) * v
    return {"状态定义": 定义,
            "L1_ORACLE_上限_不可实现": 组合,
            "L1_逐状态最优": best,
            "最好单一动作": 最好单一动作, "最好单一动作超额": 最好单一,
            "L1上限_减_最好单一": 组合 - 最好单一,
            "用户对角组合超额": 对角值, "用户对角明细": 对角明细,
            "对角_减_最好单一": 对角值 - 最好单一,
            "状态频率": {str(k): float(v) for k, v in 权重.items()}}


def 洗牌对照(日收益, 状态, 定义, rng) -> dict:
    """环形块洗牌状态标签（保留持续性与频率），重算 L1，看真实 L1 是否超过偶然对齐。"""
    lab = 状态[定义].to_numpy()
    n = len(lab)
    上限分布 = []
    for _ in range(洗牌次数):
        k = int(rng.integers(1, n))
        打乱 = pd.DataFrame({"date": 状态["date"].to_numpy(), 定义: np.roll(lab, k)})
        try:
            m = 状态动作矩阵(日收益, 打乱, 定义)
            上限分布.append(全知上限(m, 打乱, 定义)["L1_ORACLE_上限_不可实现"])
        except Exception:
            continue
    a = np.array(上限分布, dtype=float)
    return {"洗牌次数": int(len(a)), "洗牌L1均值": float(a.mean()),
            "洗牌L1_95分位": float(np.percentile(a, 95)),
            "洗牌L1_最大": float(a.max())}


def 实时可实现(日收益, 状态, 定义) -> dict:
    """L2：每状态的动作只用历史（扩张窗，前 预热年 年为预热）选，再向前评估。"""
    m = 日收益.merge(状态[["date", 定义]], on="date", how="inner").dropna(subset=[定义])
    m = m.sort_values("date")
    起 = pd.Timestamp(研究期起) + pd.DateOffset(years=预热年)
    评估 = m[m["date"] >= 起]
    if len(评估) == 0:
        return {"L2_实时可实现": None}
    累, 累n = 0.0, 0
    选择记录 = []
    for 日, g in 评估.groupby("date"):
        hist = m[m["date"] < 日]
        if len(hist) == 0:
            continue
        sv = g[定义].iloc[0]
        h = hist[hist[定义] == sv]
        if len(h) == 0:
            选动作 = "A4_空仓"
        else:
            表 = h.groupby("动作").apply(
                lambda x: np.average(x["超额"], weights=x["n"]), include_groups=False)
            选动作 = str(表.idxmax()) if float(表.max()) > 0 else "A4_空仓"
        if 选动作 == "A4_空仓":
            选择记录.append((日, sv, 选动作, 0.0, 0))
            continue
        行 = g[g["动作"] == 选动作]
        if len(行) == 0:
            选择记录.append((日, sv, 选动作, 0.0, 0))
            continue
        v = float(np.average(行["超额"], weights=行["n"]))
        nn = int(行["n"].sum())
        累 += v * nn; 累n += nn
        选择记录.append((日, sv, 选动作, v, nn))
    return {"L2_实时可实现": (累 / 累n) if 累n else None,
            "L2_评估事件数": 累n,
            "L2_评估起点": str(起.date()),
            "L2_空仓日占比": float(np.mean([r[2] == "A4_空仓" for r in 选择记录]))
            if 选择记录 else None}


def 状态持续性(状态: pd.DataFrame, 定义: str) -> list[dict]:
    """§15.5 用户"量化参与使状态难识别"命题的直接检验：分子时代的持续性与切换。"""
    out = []
    s = 状态[["date", 定义]].dropna().reset_index(drop=True)
    for 名, (d0, d1) in 时代.items():
        sub = s[(s["date"] >= d0) & (s["date"] <= d1)].reset_index(drop=True)
        if len(sub) < 30:
            continue
        lab = sub[定义].to_numpy()
        切换 = int((lab[1:] != lab[:-1]).sum())
        段长 = []
        cur = 1
        for i in range(1, len(lab)):
            if lab[i] == lab[i - 1]:
                cur += 1
            else:
                段长.append(cur); cur = 1
        段长.append(cur)
        年数 = len(sub) / 244.0
        # 打脸率：切换后 K 日内又切回
        打脸 = {}
        for K in (5, 10, 20):
            计, 回 = 0, 0
            for i in range(1, len(lab)):
                if lab[i] != lab[i - 1]:
                    计 += 1
                    if (lab[i + 1:i + 1 + K] == lab[i - 1]).any():
                        回 += 1
            打脸[f"打脸率_K{K}"] = 回 / 计 if 计 else None
        out.append({"状态定义": 定义, "时代": 名, "交易日数": len(sub),
                    "切换次数": 切换, "年化切换次数": 切换 / 年数,
                    "平均持续天数": float(np.mean(段长)),
                    "中位持续天数": float(np.median(段长)),
                    **打脸})
    return out


def main():
    结果目录.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    rng = np.random.default_rng(种子)

    print("[1/4] 建 5 套市场状态定义（全部 shift(1)，只用 T-1 及更早）")
    状态, 状态列 = 建状态(con)
    print(f"  交易日 {len(状态):,}  {状态['date'].min().date()}..{状态['date'].max().date()}")
    for s in 状态列:
        vc = 状态[s].value_counts(normalize=True)
        print(f"  {s:<12} " + "  ".join(f"{k}:{v:.1%}" for k, v in vc.items()))

    print("\n[2/4] 取动作日收益（动作代理映射见文件头）")
    日收益 = 取动作日收益(con)
    print(f"  日级行 {len(日收益):,}")
    for a, g in 日收益.groupby("动作"):
        print(f"  {a:<12} 日数{len(g):>6,}  事件{int(g['n'].sum()):>9,}  "
              f"加权超额 {np.average(g['超额'], weights=g['n']):+.4%}")

    print("\n[3/4] L1 全知上限 / 洗牌对照 / L2 实时可实现")
    矩阵全, 上限全 = [], []
    for 定义 in 状态列:
        m = 状态动作矩阵(日收益, 状态, 定义)
        矩阵全.append(m)
        u = 全知上限(m, 状态, 定义)
        u.update(洗牌对照(日收益, 状态, 定义, rng))
        u.update(实时可实现(日收益, 状态, 定义))
        上限全.append(u)
        print(f"  {定义:<12} L1上限 {u['L1_ORACLE_上限_不可实现']:+.4%}  "
              f"最好单一({u['最好单一动作']}) {u['最好单一动作超额']:+.4%}  "
              f"差 {u['L1上限_减_最好单一']:+.4%}  "
              f"洗牌95分位 {u['洗牌L1_95分位']:+.4%}  "
              f"用户对角 {u['用户对角组合超额']:+.4%}  "
              f"L2实时 {u['L2_实时可实现']:+.4%}" if u["L2_实时可实现"] is not None
              else f"  {定义} L2=None")

    print("\n[4/4] §15.5 状态持续性分时代（检验'量化参与使状态难识别'）")
    持续 = []
    for 定义 in 状态列:
        持续 += 状态持续性(状态, 定义)
    持续df = pd.DataFrame(持续)
    for 定义 in 状态列:
        sub = 持续df[持续df["状态定义"] == 定义]
        线 = "  ".join(f"{r['时代']}:{r['平均持续天数']:.1f}天/{r['年化切换次数']:.0f}次"
                       for _, r in sub.iterrows())
        print(f"  {定义:<12} {线}")

    pd.concat(矩阵全, ignore_index=True).to_csv(
        结果目录 / "市场状态_状态动作矩阵.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(上限全).to_csv(
        结果目录 / "市场状态_全知上限.csv", index=False, encoding="utf-8-sig")
    持续df.to_csv(结果目录 / "市场状态_状态持续性.csv", index=False, encoding="utf-8-sig")

    汇总 = {
        "生成时间": dt.datetime.now().isoformat(timespec="seconds"),
        "脚本": "02_代码/07_市场状态轨道.py",
        "预注册依据": ["§15.2 三段式", "§15.3 术语操作化", "§15.4 判定", "§15.5 量化参与命题"],
        "动作代理映射": 动作映射,
        "L1纪律": "L1 使用未来信息选每状态最优动作，是上限，永不得当策略绩效引用",
        "上限表": 上限全,
        "持续性": 持续,
    }
    (结果目录 / "市场状态_汇总.json").write_text(
        json.dumps(汇总, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n已写入 03_结果/市场状态_*.csv 与 市场状态_汇总.json")
    con.close()


if __name__ == "__main__":
    main()
