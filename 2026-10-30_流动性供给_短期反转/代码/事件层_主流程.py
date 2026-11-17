# -*- coding: utf-8 -*-
"""事件层主流程：12 格表、calendar-time 统计、对照、分时代、尾部、成本、匹配稳健性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 配置 as C
import 事件层 as E

结果 = C.结果目录
中间 = C.中间目录


def 日序列(td: np.ndarray, v: np.ndarray, 基准td: np.ndarray):
    s = pd.Series(v, index=np.arange(len(v))).groupby(td).sum().reindex(基准td, fill_value=0.0)
    c = pd.Series(v, index=np.arange(len(v))).groupby(td).size().reindex(基准td, fill_value=0)
    return s.to_numpy(), c.to_numpy(), (s / c.replace(0, np.nan)).to_numpy()


def 市场累计(P: E.面板) -> np.ndarray:
    """可执行宇宙等权日收益的累计（对数），用于形成期市场/个股分解。"""
    d = P.df[P.df["宇宙"].to_numpy()]
    m = d.groupby("td")["pct_chg"].mean() / 100.0
    m = m.reindex(np.arange(P.td_min, P.td_max + 1)).fillna(0.0)
    return np.log1p(m.to_numpy()).cumsum()


def 市场R(对数累计: np.ndarray, td: np.ndarray, k: int, 偏移: int) -> np.ndarray:
    td = np.asarray(td)
    idx = td - 偏移  # 列偏移：td_min → 0
    out = np.full(len(td), np.nan)
    ok = (idx - k >= 0) & (idx < len(对数累计))
    out[ok] = np.exp(对数累计[idx[ok]] - 对数累计[idx[ok] - k]) - 1.0
    return out


def 区间(序列: np.ndarray, 块长: int) -> tuple[float, float, float, np.ndarray]:
    bs = E.块自助(序列, C.自助抽样次数, 块长, C.统计种子)
    if len(bs) == 0:
        return np.nan, np.nan, np.nan, bs
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), E.两侧p(bs), bs


def 单格(P: E.面板, u: pd.DataFrame, 基准td: np.ndarray, 对数累计: np.ndarray, k: int, h: int) -> dict:
    主 = (k == C.主候选k and h == C.主候选h)
    s = E.生成信号(P, k, h)
    d = E.生成对照(s, u)
    s2 = s.merge(u[["td", "sym", "ADV20", "float_mktcap", "波动20"]], on=["td", "sym"], how="left")

    rl = E.执行(P, s["sym"].to_numpy(), s["td"].to_numpy(), h, 记录路径=True)
    # 对照
    c1a = E.执行(P, d["C1"]["sym"].to_numpy(), d["C1"]["td"].to_numpy(), h)
    c1b = E.执行(P, d["C1"]["sym_c"].to_numpy(), d["C1"]["td"].to_numpy(), h)
    c2a = E.执行(P, d["C2"]["sym"].to_numpy(), d["C2"]["td"].to_numpy(), h)
    c2b = E.执行(P, d["C2"]["sym_c"].to_numpy(), d["C2"]["td"].to_numpy(), h)
    nn = E.最近邻匹配(u, s2)
    nna = E.执行(P, nn["sym"].to_numpy(), nn["td"].to_numpy(), h)
    nnb = E.执行(P, nn["sym_c"].to_numpy(), nn["td"].to_numpy(), h)

    ok1 = c1a["完结"] & c1b["完结"]
    ok2 = c2a["完结"] & c2b["完结"]
    ok3 = nna["完结"] & nnb["完结"]
    差1 = c1a["净收益"][ok1] - c1b["净收益"][ok1]
    差2 = c2a["净收益"][ok2] - c2b["净收益"][ok2]
    差3 = nna["净收益"][ok3] - nnb["净收益"][ok3]
    td1, td2, td3 = d["C1"]["td"].to_numpy()[ok1], d["C2"]["td"].to_numpy()[ok2], nn["td"].to_numpy()[ok3]

    完 = rl["完结"]
    净, 毛 = rl["净收益"][完], rl["毛收益"][完]
    tdl = s["td"].to_numpy()[完]
    _, _, 日净 = 日序列(tdl, 净, 基准td)
    _, _, 日毛 = 日序列(tdl, 毛, 基准td)
    _, _, 日差2 = 日序列(td2, 差2, 基准td)
    _, _, 日差1 = 日序列(td1, 差1, 基准td)
    _, _, 日差3 = 日序列(td3, 差3, 基准td)

    块长 = max(h, 5)
    净CI = 区间(日净, 块长)
    _bs事件 = E.聚类自助(*日序列(tdl, 净, 基准td)[:2], C.自助抽样次数, C.统计种子)
    净CI_事件 = (float(np.quantile(_bs事件, 0.025)), float(np.quantile(_bs事件, 0.975)), E.两侧p(_bs事件))
    差2CI = 区间(日差2, 块长)
    差1CI = 区间(日差1, 块长)
    差3CI = 区间(日差3, 块长)
    差2CI_块h = 区间(日差2, max(h, 1))

    # 分时代
    y = pd.DatetimeIndex(df日期(P, tdl)).year.to_numpy()
    eras = []
    for lab, 掩l, 掩c in 时代掩码(P, tdl, td2, y):
        eras.append({
            "k": k, "h": h, "时期": lab,
            "输家笔数": int(掩l.sum()), "输家毛均值": float(np.nanmean(毛[掩l])) if 掩l.any() else np.nan,
            "输家净均值": float(np.nanmean(净[掩l])) if 掩l.any() else np.nan,
            "匹配笔数": int(掩c.sum()), "匹配超额": float(np.mean(差2[掩c])) if 掩c.any() else np.nan,
            "匹配超额CI下": 区间(日序列(td2[掩c], 差2[掩c], 基准td)[2], 块长)[0] if 掩c.sum() > 60 else np.nan,
            "输家净CI下": 区间(日序列(tdl[掩l], 净[掩l], 基准td)[2], 块长)[0] if 掩l.sum() > 60 else np.nan,
        })

    # 尾部：剔除最盈利 1% 信号日（按当日匹配超额均值）
    日表 = pd.DataFrame({"td": td2, "d": 差2})
    日m = 日表.groupby("td")["d"].mean()
    阈值 = 日m.quantile(0.99)
    剔除日 = set(日m[日m >= 阈值].index)
    保留 = ~np.isin(td2, list(剔除日))
    尾 = {"k": k, "h": h, "剔除信号日数": len(剔除日), "剩余笔数": int(保留.sum()),
          "剔除后匹配超额": float(np.mean(差2[保留])) if 保留.any() else np.nan,
          "剔除后CI下": 区间(日序列(td2[保留], 差2[保留], 基准td)[2], 块长)[0] if 保留.sum() > 60 else np.nan,
          "原匹配超额": float(np.mean(差2))}

    # 成本 ×2 / 延迟 1 日
    稳健 = []
    for 名, 倍数, 延 in [("基准", 1.0, 0), ("成本×2", 2.0, 0), ("延迟1日", 1.0, 1)]:
        a = E.执行(P, d["C2"]["sym"].to_numpy(), d["C2"]["td"].to_numpy(), h, 成本倍数=倍数, 延迟=延)
        b = E.执行(P, d["C2"]["sym_c"].to_numpy(), d["C2"]["td"].to_numpy(), h, 成本倍数=倍数, 延迟=延)
        ok = a["完结"] & b["完结"]
        dd = a["净收益"][ok] - b["净收益"][ok]
        t_ = d["C2"]["td"].to_numpy()[ok]
        稳健.append({"k": k, "h": h, "方案": 名, "输家净均值": float(np.nanmean(a["净收益"][ok])),
                     "对照净均值": float(np.nanmean(b["净收益"][ok])),
                     "匹配超额": float(np.mean(dd)) if len(dd) else np.nan,
                     "配对笔数": int(ok.sum()),
                     "CI下": 区间(日序列(t_, dd, 基准td)[2], 块长)[0] if ok.sum() > 60 else np.nan})

    # 统一 10 万元名义（诊断：剥离账户名义金额差异）
    r10 = E.执行(P, s["sym"].to_numpy(), s["td"].to_numpy(), h, 名义金额=100_000.0)
    # 成本瀑布：仅费用（滑点=0、无冲击、无取整）
    r0 = E.执行(P, s["sym"].to_numpy(), s["td"].to_numpy(), h, 滑点bp=0.0, 冲击系数=0.0, 取整=False)

    # 属性差异
    属性 = 属性差异(P, s2, d["C2"], h)

    # 结构诊断（A2）：入场前跳空、市场/个股分解、市场状态
    br = P.row(s["sym"].to_numpy(), s["td"].to_numpy() + 1)
    cr = P.row(s["sym"].to_numpy(), s["td"].to_numpy())
    跳空 = np.where((br >= 0) & (cr >= 0), P.O[np.maximum(br, 0)] / P.C[np.maximum(cr, 0)] - 1, np.nan)
    c2s = d["C2"]
    brc = P.row(c2s["sym_c"].to_numpy(), c2s["td"].to_numpy() + 1)
    crc = P.row(c2s["sym_c"].to_numpy(), c2s["td"].to_numpy())
    跳空c = np.where((brc >= 0) & (crc >= 0), P.O[np.maximum(brc, 0)] / P.C[np.maximum(crc, 0)] - 1, np.nan)
    mk = 市场R(对数累计, s["td"].to_numpy(), k, P.td_min)
    个股超额 = (1.0 + s[f"R{k}"].to_numpy()) / (1.0 + mk) - 1.0
    诊断 = {"跳空_输家均值": float(np.nanmean(跳空)), "跳空_输家中位": float(np.nanmedian(跳空)),
            "跳空_对照均值": float(np.nanmean(跳空c)),
            "跳空差": float(np.nanmean(跳空) - np.nanmean(跳空c)),
            "形成期市场均值": float(np.nanmean(mk)), "形成期个股超额均值": float(np.nanmean(个股超额)),
            "形成期R均值": float(np.nanmean(s[f"R{k}"].to_numpy()))}
    # 市场状态分组（描述性，不作为过滤器）
    状态行 = []
    for lab, 掩 in [("市场下跌", mk < 0), ("市场上行", mk >= 0)]:
        if 掩.sum() == 0:
            continue
        d2 = np.isin(td2, s["td"].to_numpy()[掩])
        掩完 = 掩[完]
        状态行.append({"k": k, "h": h, "市场状态": lab, "输家笔数": int(掩.sum()),
                       "输家毛均值": float(np.nanmean(毛[掩完])),
                       "输家净均值": float(np.nanmean(净[掩完])),
                       "匹配超额": float(np.mean(差2[d2])) if d2.any() else np.nan,
                       "市场R均值": float(np.nanmean(mk[掩]))})

    行 = {
        "k": k, "h": h, "主候选": 主,
        "事件名义金额": C.事件名义金额(h),
        "信号日数": int(s["td"].nunique()),
        "池子数_中位": float(s.groupby("td").size().median()),
        "池子数_均值": float(s.groupby("td").size().mean()),
        "信号数": int(len(s)), "可买数": int(rl["可买"].sum()),
        "可买率": float(rl["可买"].mean()),
        "完结数": int(完.sum()),
        "顺延退出数": int((rl["顺延天数"] > 0).sum()),
        "未实现数": int((rl["可买"] & ~完).sum()),
        "不可买数": int((~rl["可买"]).sum()),
        "不可买_次日停牌": int((rl["未成交原因"] == "次日停牌").sum()),
        "不可买_开盘涨停": int((rl["未成交原因"] == "次日开盘涨停").sum()),
        "不可买_整股不足": int((rl["未成交原因"] == "整股不足").sum()),
        "不可买_参与率0股": int((rl["未成交原因"] == "参与率上限导致0股").sum()),
        "部分成交数": int(rl["部分成交"].sum()),
    }
    行.update({f"毛{a}": b for a, b in E.分布摘要(毛).items()})
    行.update({f"净{a}": b for a, b in E.分布摘要(净).items()})
    行.update({
        "净收益_统一10万": float(np.nanmean(r10["净收益"])),
        "净收益_仅费用": float(np.nanmean(r0["净收益"])),
        "毛收益_仅费用口径": float(np.nanmean(r0["毛收益"])),
        "手续费率": float(np.nanmean((rl["费用合计"] / (rl["买入价"] * rl["数量"]))[完])),
        "执行摩擦": float(np.nanmean(rl["执行价差"][完])),
        "平均持有日": float(np.nanmean(rl["持有日"][完])),
        "平均顺延": float(np.nanmean(rl["顺延天数"][完])),
        "日净_均值": float(np.nanmean(日净)),
        "日净CI下": 净CI[0], "日净CI上": 净CI[1], "日净p": 净CI[2],
        "事件加权净CI下": 净CI_事件[0], "事件加权净CI上": 净CI_事件[1],
        "日毛_均值": float(np.nanmean(日毛)),
        "C1对照净": float(np.nanmean(c1b["净收益"][ok1])),
        "C1差": float(np.nanmean(差1)), "C1差CI下": 差1CI[0], "C1差CI上": 差1CI[1], "C1差p": 差1CI[2],
        "C2对照净": float(np.nanmean(c2b["净收益"][ok2])),
        "C2对照毛": float(np.nanmean(c2b["毛收益"][ok2])),
        "C2差": float(np.nanmean(差2)), "C2差CI下": 差2CI[0], "C2差CI上": 差2CI[1], "C2差p": 差2CI[2],
        "C2差CI下_块长h": 差2CI_块h[0],
        "C2配对笔数": int(ok2.sum()),
        "NN对照净": float(np.nanmean(nnb["净收益"][ok3])),
        "NN差": float(np.nanmean(差3)), "NN差CI下": 差3CI[0], "NN差CI上": 差3CI[1], "NN差p": 差3CI[2],
        "NN重复率": float(1 - nn["sym_c"].nunique() / len(nn)) if len(nn) else np.nan,
        "NN平均距离": float(nn["距离"].mean()) if len(nn) else np.nan,
        "MFE中位": float(np.nanmedian(rl["MFE"][完])),
        "MFE均值": float(np.nanmean(rl["MFE"][完])),
        "MAE中位": float(np.nanmedian(rl["MAE"][完])),
        "MAE均值": float(np.nanmean(rl["MAE"][完])),
    })
    行.update(属性)
    明细 = pd.DataFrame({
        "证券": s["symbol"].to_numpy()[完], "信号日": df日期(P, tdl),
        "毛收益": 毛, "净收益": 净, "执行摩擦": rl["执行价差"][完],
        "费用率": (rl["费用合计"] / (rl["买入价"] * rl["数量"]))[完],
        "MFE": rl["MFE"][完], "MAE": rl["MAE"][完],
        "持有日": rl["持有日"][完], "顺延天数": rl["顺延天数"][完],
        "R_k": s[f"R{k}"].to_numpy()[完],
    })
    日池 = s.groupby("td").size().reindex(基准td, fill_value=0)
    日宇宙 = u.groupby("td").size().reindex(基准td, fill_value=0)
    逐日 = pd.DataFrame({"date": df日期(P, 基准td), "td": 基准td, "k": k, "h": h,
                         "宇宙数": 日宇宙.to_numpy(), "池子数": 日池.to_numpy(),
                         "日净": 日净, "日毛": 日毛, "日差C2": 日差2, "日差C1": 日差1})
    行.update(诊断)
    return {"行": 行, "分时代": eras, "尾部": 尾, "稳健": 稳健, "明细": 明细, "逐日": 逐日, "状态": 状态行,
            "日差2": 日差2, "td2": td2, "差2": 差2, "日净": 日净, "净": 净, "tdl": tdl,
            "r10净": r10["净收益"]}


def df日期(P: E.面板, td: np.ndarray) -> np.ndarray:
    td = np.asarray(td)
    idx = np.clip(td - P.td_min, 0, P.n_td - 1)
    # 每个 td 的日期可由任一行取到；用日历映射表
    if not hasattr(P, "_td日期"):
        m = pd.Series(P.df["date"].to_numpy(), index=P.df["td"].to_numpy())
        P._td日期 = m.groupby(level=0).first().reindex(np.arange(P.td_min, P.td_max + 1)).to_numpy()
    return P._td日期[idx]


def 时代掩码(P, tdl, td2, y):
    全 = np.ones(len(tdl), bool)
    yield "全期", 全, np.ones(len(td2), bool)
    yc = pd.DatetimeIndex(df日期(P, td2)).year.to_numpy()
    for lab, a, b in C.两段:
        yield lab, (y >= a) & (y <= b), (yc >= a) & (yc <= b)
    for lab, a, b in C.四段:
        yield lab, (y >= a) & (y <= b), (yc >= a) & (yc <= b)


def 属性差异(P: E.面板, s2: pd.DataFrame, 对2: pd.DataFrame, h: int) -> dict:
    输 = s2.merge(P.df[["td", "sym", "BETA60", "close"]], on=["td", "sym"], how="left")
    控 = 对2.merge(P.df[["td", "sym", "ADV20", "float_mktcap", "波动20", "BETA60", "close"]],
                 left_on=["td", "sym_c"], right_on=["td", "sym"], how="left")
    out = {}
    for 名, a, b in [("logADV", np.log(输["ADV20"]), np.log(控["ADV20"])),
                     ("log流通市值", np.log(输["float_mktcap"]), np.log(控["float_mktcap"])),
                     ("波动20", 输["波动20"], 控["波动20"]),
                     ("BETA60", 输["BETA60"], 控["BETA60"]),
                     ("原始价", 输["close"], 控["close"])]:
        out[f"属性差_{名}"] = float(np.nanmean(a) - np.nanmean(b))
    return out


def 等权基准(P: E.面板, h: int) -> dict:
    """C3：可执行宇宙等权，同执行规则（逐日等权，剔除不可成交）。"""
    u = E.宇宙快照(P)
    r = E.执行(P, u["sym"].to_numpy(), u["td"].to_numpy(), h)
    ok = r["完结"]
    df = pd.DataFrame({"td": u["td"].to_numpy()[ok], "g": r["毛收益"][ok], "n": r["净收益"][ok]})
    g = df.groupby("td")["g"].mean()
    n = df.groupby("td")["n"].mean()
    bs_g = E.块自助(g.to_numpy(), C.自助抽样次数, max(h, 5), C.统计种子)
    bs_n = E.块自助(n.to_numpy(), C.自助抽样次数, max(h, 5), C.统计种子)
    return {"h": h, "基准毛均值_日均": float(g.mean()), "基准净均值_日均": float(n.mean()),
            "基准毛均值_事件": float(np.nanmean(r["毛收益"])), "基准净均值_事件": float(np.nanmean(r["净收益"])),
            "基准毛CI下": float(np.quantile(bs_g, 0.025)), "基准毛CI上": float(np.quantile(bs_g, 0.975)),
            "基准净CI下": float(np.quantile(bs_n, 0.025)), "基准净CI上": float(np.quantile(bs_n, 0.975)),
            "基准日数": int(len(g)), "基准笔数": int(ok.sum()),
            "基准可买率": float(r["可买"].mean())}


def 置换检验(P: E.面板, u: pd.DataFrame, k: int, h: int, 次数: int = C.置换次数) -> dict:
    """标签置换：日内打乱 R_k，保持同一选择规则（R_k<0 的最弱20%），看日差分布。"""
    # 全宇宙 (td,sym) → 净收益矩阵（clean exit；不可成交记 NaN）
    uu = u[(u["td"] + 1 + h <= P.td_max)].copy()
    r = E.执行(P, uu["sym"].to_numpy(), uu["td"].to_numpy(), h)
    uu["净"] = np.where(r["完结"], r["净收益"], np.nan)
    df = P.df[["td", "sym", "R3", "R5", "R10", "窗口完整3", "窗口完整5", "窗口完整10"]]
    uu = uu.merge(df, on=["td", "sym"], how="left")
    uu["Rk"] = uu[f"R{k}"]
    uu["有效"] = uu[f"窗口完整{k}"]
    日组 = {td: (g["Rk"].to_numpy(), g["净"].to_numpy(), g["有效"].to_numpy())
            for td, g in uu.groupby("td", sort=True)}
    日组 = {k2: v for k2, v in 日组.items() if v[0].shape[0] >= 20}

    def 日均差(映射):
        vals = []
        for td, (rk, ret, val) in 映射.items():
            m = val & np.isfinite(rk) & np.isfinite(ret)
            if m.sum() < 20:
                continue
            r_, x_ = rk[m], ret[m]
            neg = r_ < 0
            if neg.sum() < 5:
                continue
            q = np.quantile(r_[neg], 0.2)
            选 = m & (rk <= q) & (rk < 0)
            vals.append(ret[选].mean() if 选.any() else np.nan)
        return np.nanmean(vals)

    实际 = 日均差(日组)
    rng = np.random.default_rng(C.统计种子 + 99)
    零 = []
    for _ in range(次数):
        映射 = {td: (rng.permutation(rk), ret, val) for td, (rk, ret, val) in 日组.items()}
        零.append(日均差(映射))
    零 = np.array(零)
    return {"k": k, "h": h, "实际日均净": float(实际), "置换均值": float(np.nanmean(零)),
            "置换标准差": float(np.nanstd(零)),
            "置换p": float((np.sum(零 >= 实际) + 1) / (次数 + 1)),
            "置换分位": float((零 < 实际).mean()), "次数": 次数}


def 主流程():
    P = E.面板()
    u = E.宇宙快照(P)
    基准td = np.arange(P.td_min + 1, P.td_max + 1)
    print(f"面板 {P.n_sym}×{P.n_td}；宇宙快照 {len(u)} 行", flush=True)

    基准表 = {h: 等权基准(P, h) for h in C.h列表}
    print("等权基准:", {h: round(v["基准毛均值_日均"], 5) for h, v in 基准表.items()}, flush=True)
    对数累计 = 市场累计(P)

    行表, 代表, 尾表, 稳健表, 逐日表, 状态表 = [], [], [], [], [], []
    主缓存 = None
    for k in C.k列表:
        for h in C.h列表:
            r = 单格(P, u, 基准td, 对数累计, k, h)
            r["行"].update({f"基准_{a}": b for a, b in 基准表[h].items() if a != "h"})
            行表.append(r["行"])
            代表 += r["分时代"]
            尾表.append(r["尾部"])
            稳健表 += r["稳健"]
            状态表 += r["状态"]
            逐日表.append(r["逐日"])
            if k == C.主候选k and h == C.主候选h:
                主缓存 = r
                r["明细"].to_parquet(中间 / "事件层_主候选明细.parquet", index=False)
            print(f"k={k} h={h} 毛={r['行']['毛均值']:.5f} 净={r['行']['净均值']:.5f} "
                  f"C2差={r['行']['C2差']:.5f} q待定", flush=True)

    表 = pd.DataFrame(行表)
    表["C2差q值"] = E.BH_FDR(表["C2差p"].to_numpy())
    表["C1差q值"] = E.BH_FDR(表["C1差p"].to_numpy())
    表.to_csv(结果 / "事件层_12格.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(代表).to_csv(结果 / "事件层_分时代.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(尾表).to_csv(结果 / "事件层_尾部.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(稳健表).to_csv(结果 / "事件层_成本与延迟.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(状态表).to_csv(结果 / "事件层_市场状态.csv", index=False, encoding="utf-8-sig")
    逐日 = pd.concat(逐日表, ignore_index=True)
    逐日.to_parquet(中间 / "事件层_逐日.parquet", index=False)
    pd.DataFrame([基准表[h] for h in C.h列表]).to_csv(结果 / "事件层_等权基准.csv", index=False, encoding="utf-8-sig")

    置 = 置换检验(P, u, C.主候选k, C.主候选h)
    pd.DataFrame([置]).to_csv(结果 / "事件层_置换检验.csv", index=False, encoding="utf-8-sig")
    print("置换:", 置, flush=True)
    return P, u, 表, 主缓存, 逐日


if __name__ == "__main__":
    主流程()
