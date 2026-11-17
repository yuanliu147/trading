# -*- coding: utf-8 -*-
"""账户层：每日重叠持有队列（预注册 §7 / 00A A5、A6）。

规则
----
* 每个信号日 t 从输家池按 SHA256 中性哈希顺序取前 10 只（不看收益、不看强弱）；
* t+1 开盘买入，第 h 个交易日后开盘卖出；开盘涨停/停牌/无成交 → 订单失败，不替补；
* 目标金额 = 前日NAV/(每日只数×h)；100 股整数倍；现金不足按剩余槽位均分，绝不透支；
* 佣金双边 0.025% 且每笔最低 5 元，卖出印花税 0.1%，过户费按历史规则；
* 基础滑点 10bp + 冲击 10bp×sqrt(参与率)，按分不利取整；单票单日不超过成交量 10%；
* 卖出遇停牌/开盘跌停顺延至可卖；样本末仍不可卖按最后收盘价计入 NAV 并单列；
* 每日 NAV = 现金 + 持仓市值（原始收盘）；除权日先按旧股数计税后现金分红，再计送转与配股；
* 配股按旧股数计算应配股数，现金不足时全部放弃并记录，不得透支或免费增股。
"""
from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 配置 as C
import 事件层 as E
import 费用与交易约束 as F

红利税率 = 0.20   # 持有不足 1 个月


def 载入公司行动(P: E.面板) -> dict:
    列 = ["dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price"]
    ca = pd.read_parquet(
        C.母项目数据 / "corp_actions.parquet", columns=["symbol", "ex_date", *列],
        filters=[("ex_date", ">=", date(2009, 1, 1)),
                 ("ex_date", "<=", pd.Timestamp(C.研究期止).date())])
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    ca = ca[(ca["ex_date"] >= pd.Timestamp("2009-01-01")) & (ca["ex_date"] <= pd.Timestamp(C.研究期止))]
    ca = ca[(ca[列].fillna(0) > 0).any(axis=1)]
    return {(s, d.date()): dict(zip(列, map(float, v))) for s, d, v in
            zip(ca["symbol"], ca["ex_date"], ca[列].to_numpy())}


def 运行账户(P: E.面板, 池: dict, h: int, *, 资金: float = C.主资金, 每日只数: int = C.每日只数主,
             成本倍数: float = 1.0, 延迟: int = 0, 公司行动表: dict | None = None) -> dict:
    公司行动表 = 公司行动表 if 公司行动表 is not None else {}
    信号tds = sorted(池.keys())
    起td = 信号tds[0] + 1 + 延迟
    基础bp = C.基础滑点bp * 成本倍数
    日历 = pd.Series(P.df["date"].to_numpy(), index=P.df["td"].to_numpy()).groupby(level=0).first()

    def 行(sym: int, td: int) -> int:
        if td < P.td_min or td > P.td_max:
            return -1
        return int(P.rid[sym, td - P.td_min])

    现金 = float(资金)
    持仓: list[dict] = []
    净值行 = [{"日期": str(日历[起td - 1].date()), "td": int(起td - 1), "净值": float(资金),
              "现金": float(资金), "持仓市值": 0.0, "持仓数": 0, "仓位": 0.0,
              "现金流": 0.0, "费用": 0.0, "分红": 0.0}]
    交易行 = []
    计划笔数 = 成交笔数 = 未成交笔数 = 0
    计划金额 = 实际金额 = 0.0
    选中计数: dict[str, int] = {}
    前日NAV = float(资金)

    for td in range(起td, P.td_max + 1):
        日 = 日历[td].date()
        当日现金流 = 0.0
        当日费用 = 0.0
        当日分红 = 0.0

        # ---- 1) 公司行动（除权日买入不享受；三项均以行动前旧股数为基数） ----
        for p in 持仓:
            if p["买入日"] >= 日:
                continue
            ca = 公司行动表.get((p["symbol"], 日))
            if not ca:
                continue
            旧股数 = int(p["股数"])

            dps = ca["dividend_per_share"]
            if dps > 0:
                x = round(旧股数 * dps * (1 - 红利税率), 2)
                现金 += x
                当日现金流 += x
                当日分红 += x
                p["已分红"] += x
                交易行.append({"方向": "分红", "日期": str(日), "证券": p["symbol"], "价格": np.nan,
                             "股数": 旧股数, "旧股数": 旧股数, "金额": x, "佣金": 0.0,
                             "印花税": 0.0, "过户费": 0.0, "费用合计": 0.0, "分红": x,
                             "盈亏": 0.0, "持有交易日": 0, "顺延": 0, "信号日": "",
                             "买入价": np.nan, "买入费": 0.0, "买入日": str(p["买入日"])})

            送转比例 = ca["per_share_bonus"]
            if 送转比例 > 0:
                新增股数 = int(np.floor(旧股数 * 送转比例 + 1e-9))
                p["股数"] += 新增股数
                交易行.append({"方向": "送转", "日期": str(日), "证券": p["symbol"], "价格": np.nan,
                             "股数": 新增股数, "旧股数": 旧股数, "送转股数": 新增股数,
                             "金额": 0.0, "佣金": 0.0, "印花税": 0.0, "过户费": 0.0,
                             "费用合计": 0.0, "分红": 0.0, "盈亏": 0.0, "持有交易日": 0,
                             "顺延": 0, "信号日": "", "买入价": np.nan, "买入费": 0.0,
                             "买入日": str(p["买入日"])})

            配股比例 = ca["rights_ratio"]
            if 配股比例 > 0:
                配股价 = ca["rights_price"]
                if 配股价 <= 0:
                    raise ValueError(f"配股价格非正，禁止免费增股：{p['symbol']} {日} {配股价}")
                应配股数 = int(np.floor(旧股数 * 配股比例 + 1e-9))
                所需现金 = round(应配股数 * 配股价, 2)
                现金足够 = 所需现金 <= 现金 + 1e-9
                实配股数 = 应配股数 if 现金足够 else 0
                配股成本 = 所需现金 if 现金足够 else 0.0
                if 现金足够:
                    现金 -= 配股成本
                    当日现金流 -= 配股成本
                    p["股数"] += 实配股数
                    p["配股成本"] += 配股成本
                交易行.append({"方向": "配股", "日期": str(日), "证券": p["symbol"], "价格": 配股价,
                             "股数": 实配股数, "旧股数": 旧股数, "应配股数": 应配股数,
                             "配股股数": 实配股数, "配股成本": 配股成本, "金额": 配股成本,
                             "配股状态": "已认购" if 现金足够 else "现金不足_全部放弃",
                             "现金缺口": 0.0 if 现金足够 else round(所需现金 - 现金, 2),
                             "佣金": 0.0, "印花税": 0.0, "过户费": 0.0, "费用合计": 0.0,
                             "分红": 0.0, "盈亏": 0.0, "持有交易日": 0, "顺延": 0, "信号日": "",
                             "买入价": np.nan, "买入费": 0.0, "买入日": str(p["买入日"])})

        # ---- 2) 卖出 ----
        for p in list(持仓):
            if p["卖出td"] != td:
                continue
            r = 行(p["sym"], td)
            可卖 = (r >= 0 and not P.SUSP[r] and not P.LDO[r] and P.VOL[r] > 0
                    and np.isfinite(P.O[r]) and P.O[r] > 0)
            if not 可卖:
                p["卖出td"] += 1
                p["顺延"] += 1
                continue
            参与 = min(p["股数"] / max(P.VOL[r], 1.0), 1.0)
            bps = 基础bp + C.冲击系数bp * 成本倍数 * np.sqrt(max(参与, 0.0))
            价 = max(np.floor(P.O[r] * (1 - bps / 10000.0) * 100.0 + 1e-9) / 100.0, 0.01)
            if np.isfinite(P.LDP[r]) and P.LDP[r] > 0:
                价 = max(价, P.LDP[r])
            金额 = 价 * p["股数"]
            费 = F.trade_fees(日, p["symbol"], "sell", 价, int(p["股数"]))
            现金 += 金额 - 费.total
            当日现金流 += 金额 - 费.total
            当日费用 += 费.total
            交易行.append({"方向": "卖出", "日期": str(日), "证券": p["symbol"], "价格": 价,
                         "股数": int(p["股数"]), "金额": round(金额, 2),
                         "佣金": 费.commission, "印花税": 费.stamp_duty, "过户费": 费.transfer_fee,
                         "费用合计": 费.total, "分红": round(p["已分红"], 2),
                         "买入金额": round(p["买入金额"], 2), "配股成本": round(p["配股成本"], 2),
                         "盈亏": round(金额 - 费.total + p["已分红"] - p["买入金额"] - p["买入费用"]
                                      - p["配股成本"], 2),
                         "持有交易日": int(td - p["买入td"]), "顺延": int(p["顺延"]),
                         "信号日": p["信号日"], "买入价": p["买入价"], "买入费": p["买入费用"],
                         "买入日": p["买入日"]})
            持仓.remove(p)

        # ---- 3) 买入（信号来自 td-1-延迟） ----
        信号td = td - 1 - 延迟
        池子 = 池.get(信号td)
        if 池子 is not None and len(池子) > 0:
            目标 = 前日NAV / (每日只数 * h)
            待买 = list(池子[:每日只数])
            剩余槽 = len(待买)
            信号日 = str(日历[信号td].date())
            for sym in 待买:
                计划笔数 += 1
                计划金额 += 目标
                r = 行(sym, td)
                可买 = (r >= 0 and not P.SUSP[r] and not P.LUO[r] and P.VOL[r] > 0
                        and np.isfinite(P.O[r]) and P.O[r] > 0)
                if not 可买:
                    未成交笔数 += 1
                    剩余槽 -= 1
                    交易行.append({"方向": "买入失败", "日期": str(日), "证券": P.symbols[sym],
                                 "价格": np.nan, "股数": 0, "金额": 0.0, "佣金": 0.0, "印花税": 0.0,
                                 "过户费": 0.0, "费用合计": 0.0, "分红": 0.0, "盈亏": 0.0,
                                 "持有交易日": 0, "顺延": 0, "信号日": 信号日,
                                 "买入价": np.nan, "买入费": 0.0, "买入日": str(日)})
                    continue
                预算 = min(目标, max(现金, 0.0) / max(剩余槽, 1))
                剩余槽 -= 1
                if 预算 <= 0:
                    未成交笔数 += 1
                    continue
                开 = P.O[r]
                上限 = P.LUP[r] if (np.isfinite(P.LUP[r]) and P.LUP[r] > 0) else np.inf
                价 = min(np.ceil(开 * (1 + 基础bp / 10000.0) * 100.0 - 1e-9) / 100.0, 上限)
                量 = np.floor(预算 / 价 / 100.0) * 100.0
                量上限 = np.floor(P.VOL[r] * C.最大参与率 / 100.0) * 100.0
                量 = min(量, 量上限)
                if 量 < 100:
                    未成交笔数 += 1
                    continue
                参与 = 量 / max(P.VOL[r], 1.0)
                冲击 = C.冲击系数bp * 成本倍数 * np.sqrt(max(参与, 0.0))
                价 = min(np.ceil(开 * (1 + (基础bp + 冲击) / 10000.0) * 100.0 - 1e-9) / 100.0, 上限)
                量 = min(np.floor(预算 / 价 / 100.0) * 100.0, 量上限)
                金额 = 价 * 量
                费 = F.trade_fees(日, P.symbols[sym], "buy", 价, int(量))
                if 金额 + 费.total > 现金 + 1e-9:
                    量 = np.floor(max(现金 - 费.total, 0.0) / 价 / 100.0) * 100.0
                    if 量 < 100:
                        未成交笔数 += 1
                        continue
                    金额 = 价 * 量
                    费 = F.trade_fees(日, P.symbols[sym], "buy", 价, int(量))
                现金 -= 金额 + 费.total
                当日现金流 -= 金额 + 费.total
                当日费用 += 费.total
                成交笔数 += 1
                实际金额 += 金额
                选中计数[P.symbols[sym]] = 选中计数.get(P.symbols[sym], 0) + 1
                持仓.append({"sym": sym, "symbol": P.symbols[sym], "股数": 量, "买入价": 价,
                             "买入金额": 金额, "买入费用": 费.total, "买入td": td,
                             "卖出td": td + h, "买入日": 日, "顺延": 0, "已分红": 0.0,
                             "配股成本": 0.0, "信号日": 信号日})
                交易行.append({"方向": "买入", "日期": str(日), "证券": P.symbols[sym], "价格": 价,
                             "股数": int(量), "金额": round(金额, 2), "买入金额": round(金额, 2),
                             "佣金": 费.commission,
                             "印花税": 费.stamp_duty, "过户费": 费.transfer_fee,
                             "费用合计": 费.total, "分红": 0.0, "盈亏": 0.0, "持有交易日": 0,
                             "顺延": 0, "信号日": 信号日, "买入价": 价, "买入费": 费.total,
                             "买入日": str(日)})

        # ---- 4) MTM ----
        市值 = 0.0
        for p in 持仓:
            r = 行(p["sym"], td)
            px = P.C[r] if r >= 0 and np.isfinite(P.C[r]) else p["买入价"]
            市值 += p["股数"] * px
        nav = 现金 + 市值
        净值行.append({"日期": str(日), "td": int(td), "净值": nav, "现金": 现金, "持仓市值": 市值,
                     "持仓数": len(持仓), "仓位": 市值 / nav if nav > 0 else np.nan,
                     "现金流": 当日现金流, "费用": 当日费用, "分红": 当日分红})
        前日NAV = nav

    净值 = pd.DataFrame(净值行)
    日志 = pd.DataFrame(交易行)
    for col in ["旧股数", "送转股数", "应配股数", "配股股数", "配股成本", "现金缺口", "买入金额"]:
        if col not in 日志:
            日志[col] = 0.0
        else:
            日志[col] = pd.to_numeric(日志[col], errors="coerce").fillna(0.0)
    if "配股状态" not in 日志:
        日志["配股状态"] = ""
    else:
        日志["配股状态"] = 日志["配股状态"].fillna("")
    期末持仓 = pd.DataFrame([{"证券": p["symbol"], "股数": int(p["股数"])} for p in 持仓],
                           columns=["证券", "股数"])
    return {"净值": 净值, "日志": 日志, "期末持仓": 期末持仓, "资金": 资金,
            "计划笔数": 计划笔数, "成交笔数": 成交笔数, "未成交笔数": 未成交笔数,
            "计划金额": 计划金额, "实际金额": 实际金额, "选中计数": 选中计数}


def 指标(r: dict) -> dict:
    nv = r["净值"]
    if len(nv) == 0:
        return {}
    x = nv["净值"].to_numpy()
    日收益 = x[1:] / x[:-1] - 1
    天数 = len(x)
    cagr = (x[-1] / x[0]) ** (252.0 / max(天数 - 1, 1)) - 1 if x[0] > 0 and x[-1] > 0 else -1.0
    峰 = np.maximum.accumulate(x)
    买入金额 = r["日志"].loc[r["日志"]["方向"] == "买入", "金额"].sum() if len(r["日志"]) else 0.0
    费用 = r["日志"]["费用合计"].sum() if len(r["日志"]) else 0.0
    return {
        "期末净值": float(x[-1]), "总收益": float(x[-1] / x[0] - 1), "CAGR": float(cagr),
        "最大回撤": float(np.min(x / 峰 - 1)),
        "日波动": float(np.std(日收益, ddof=1)) if len(日收益) > 1 else np.nan,
        "夏普": float(np.mean(日收益) / np.std(日收益, ddof=1) * np.sqrt(252))
        if len(日收益) > 1 and np.std(日收益) > 0 else np.nan,
        "平均仓位": float(nv["仓位"].mean()), "最低仓位": float(nv["仓位"].min()),
        "平均现金比例": float((nv["现金"] / nv["净值"]).mean()),
        "交易日数": int(天数),
        "计划笔数": r["计划笔数"], "成交笔数": r["成交笔数"], "未成交笔数": r["未成交笔数"],
        "成交率": float(r["成交笔数"] / max(r["计划笔数"], 1)),
        "部署率": float(r["实际金额"] / r["计划金额"]) if r["计划金额"] > 0 else np.nan,
        "买入总额": float(买入金额), "费用总额": float(费用),
        "费用占买入": float(费用 / 买入金额) if 买入金额 > 0 else np.nan,
        "年换手": float(买入金额 / max(nv["净值"].mean(), 1) / max(天数, 1) * 252),
        "单票最大权重": float((nv["持仓市值"] / nv["净值"]).max()),
        "最大持仓数": int(nv["持仓数"].max()),
        "选中最多次数": int(max(r["选中计数"].values())) if r["选中计数"] else 0,
        "重复选中率": float(1 - len(r["选中计数"]) / max(sum(r["选中计数"].values()), 1)),
    }


def 分时代指标(nv: pd.DataFrame, 起: str, 止: str) -> dict:
    d = nv[(nv["日期"] >= 起) & (nv["日期"] <= 止)]
    if len(d) < 20:
        return {}
    x = d["净值"].to_numpy()
    日收益 = x[1:] / x[:-1] - 1
    峰 = np.maximum.accumulate(x)
    cagr = (x[-1] / x[0]) ** (252.0 / max(len(x) - 1, 1)) - 1 if x[0] > 0 and x[-1] > 0 else np.nan
    return {"CAGR": float(cagr), "最大回撤": float(np.min(x / 峰 - 1)),
            "日波动": float(np.std(日收益, ddof=1)) if len(日收益) > 1 else np.nan,
            "区间倍数": float(x[-1] / x[0])}


def 会计自检(r: dict) -> dict:
    nv, 日志, 期末 = r["净值"], r["日志"], r["期末持仓"]
    出 = {"现金为负天数": int((nv["现金"] < -1e-6).sum())}
    买 = 日志[日志["方向"] == "买入"]
    卖 = 日志[日志["方向"] == "卖出"]
    分 = 日志[日志["方向"] == "分红"]
    送 = 日志[日志["方向"] == "送转"]
    配 = 日志[日志["方向"] == "配股"]
    出["买入笔数"], 出["卖出笔数"] = int(len(买)), int(len(卖))
    出["送转笔数"], 出["配股笔数"] = int(len(送)), int(len(配))
    出["配股现金不足笔数"] = int((配["配股状态"] == "现金不足_全部放弃").sum())
    出["配股现金不足放弃股数"] = int((配["应配股数"] - 配["股数"]).sum())

    # 股数守恒：买入 + 送转 + 实配 = 卖出 + 期末持仓，按证券和总量同时核对。
    买股 = 买.groupby("证券")["股数"].sum().to_dict()
    送股 = 送.groupby("证券")["股数"].sum().to_dict()
    配股 = 配.groupby("证券")["股数"].sum().to_dict()
    卖股 = 卖.groupby("证券")["股数"].sum().to_dict()
    期末股 = 期末.groupby("证券")["股数"].sum().to_dict()
    证券集 = set().union(买股, 送股, 配股, 卖股, 期末股)
    股差 = {s: 买股.get(s, 0) + 送股.get(s, 0) + 配股.get(s, 0)
             - 卖股.get(s, 0) - 期末股.get(s, 0) for s in 证券集}
    出["买入股数"] = int(买["股数"].sum())
    出["送转股数"] = int(送["股数"].sum())
    出["配股股数"] = int(配["股数"].sum())
    出["卖出股数"] = int(卖["股数"].sum())
    出["期末持仓股数"] = int(期末["股数"].sum())
    出["股数守恒差"] = int(sum(股差.values()))
    出["股数守恒最大绝对差"] = float(max(map(abs, 股差.values()), default=0.0))
    出["股数不守恒证券数"] = int(sum(abs(x) > 1e-6 for x in 股差.values()))

    # 现金守恒用账户未舍入日现金流；日志另作逐日勾稽，配股付现属于现金流出。
    出["现金守恒差"] = float(nv["现金"].iloc[-1] - (r["资金"] + nv["现金流"].sum()))
    日志现金流 = ((卖["金额"] - 卖["费用合计"]).sum() - (买["金额"] + 买["费用合计"]).sum()
                + 分["金额"].sum() - 配["金额"].sum())
    出["日志现金守恒差"] = float(nv["现金"].iloc[-1] - (r["资金"] + 日志现金流))
    出["配股未付费增股数"] = int(配.loc[(配["股数"] > 0) & (配["金额"] <= 0), "股数"].sum())
    日净流 = {}
    for d, g in 日志.groupby("日期", sort=False):
        s = g[g["方向"] == "卖出"]
        b = g[g["方向"] == "买入"]
        日净流[d] = float((s["金额"] - s["费用合计"]).sum() - (b["金额"] + b["费用合计"]).sum()
                          + g.loc[g["方向"] == "分红", "金额"].sum()
                          - g.loc[g["方向"] == "配股", "金额"].sum())
    对齐 = nv.set_index("日期")["现金流"] - pd.Series(日净流).reindex(nv["日期"]).fillna(0.0)
    出["逐日现金流最大偏差"] = float(np.abs(对齐).max())
    出["NAV恒等式最大偏差"] = float(np.abs(nv["净值"] - nv["现金"] - nv["持仓市值"]).max())
    盈亏重算 = (卖["金额"] - 卖["费用合计"] + 卖["分红"] - 卖["买入金额"]
              - 卖["买入费"] - 卖["配股成本"])
    出["卖出盈亏重算最大偏差"] = float((卖["盈亏"] - 盈亏重算).abs().max()) if len(卖) else 0.0
    return 出
