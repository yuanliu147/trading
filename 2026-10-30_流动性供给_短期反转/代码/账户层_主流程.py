# -*- coding: utf-8 -*-
"""账户层主流程：主账户、8 种子、随机/ADV 匹配对照、容量、成本与延迟、会计自检。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 配置 as C
import 事件层 as E
import 账户层 as A

结果 = C.结果目录
中间 = C.中间目录


def 构建池(u: pd.DataFrame, s: pd.DataFrame, 种子: int) -> dict:
    uu = u[["td", "sym", "aq"]].copy()
    uu["h1"] = E.哈希键(u["symbol"].to_numpy(), u["td"].to_numpy(), 种子, b"A1")
    uu["h2"] = E.哈希键(u["symbol"].to_numpy(), u["td"].to_numpy(), 种子, b"A2")
    m = uu.merge(s[["td", "sym"]].assign(输=1), on=["td", "sym"], how="left")
    m["输"] = m["输"].fillna(0).astype(int)
    输m = m[m["输"] == 1].sort_values(["td", "h1"], kind="mergesort")
    输池 = {td: g["sym"].to_numpy() for td, g in 输m.groupby("td", sort=False)}

    c = m[m["输"] == 0].sort_values(["td", "aq", "h2"], kind="mergesort").copy()
    c["pos"] = c.groupby(["td", "aq"]).cumcount()
    L = c.groupby(["td", "aq"]).size().rename("L").reset_index()
    c1池 = {td: g["sym"].to_numpy() for td, g in c.sort_values(["td", "h1"], kind="mergesort").groupby("td", sort=False)}

    lg = s[["td", "sym"]].merge(uu[["td", "sym", "aq"]], on=["td", "sym"], how="left")
    lg["j"] = lg.groupby(["td", "aq"]).cumcount()
    lg = lg.merge(L, on=["td", "aq"], how="left")
    lg["pos"] = lg["j"] % lg["L"]
    lg = lg.merge(c[["td", "aq", "pos", "sym"]], on=["td", "aq", "pos"], how="left", suffixes=("", "_c"))
    lg = lg.sort_values(["td", "j"], kind="mergesort")
    c2池 = {td: g["sym_c"].to_numpy() for td, g in lg.groupby("td", sort=False)}
    return {"输家": 输池, "C1": c1池, "C2": c2池}


def 全指标(r: dict) -> dict:
    出 = A.指标(r)
    for lab, a, b in C.两段:
        出.update({f"{lab}_{k}": v for k, v in A.分时代指标(r["净值"], f"{a}-01-01", f"{b}-12-31").items()})
    return 出


def 主流程():
    P = E.面板()
    u = E.宇宙快照(P)
    s = E.生成信号(P, C.主候选k, C.主候选h)
    公司行动表 = A.载入公司行动(P)
    h = C.主候选h
    print(f"信号 {len(s)}，资金 {C.主资金:,.0f}，每日 {C.每日只数主} 只，h={h}", flush=True)

    汇总, 净值表, 自检表, 种子表 = [], [], [], []
    主池 = None
    for 种子 in C.种子列表:
        池 = 构建池(u, s, 种子)
        if 种子 == C.主种子:
            主池 = 池
        for 名, key in [("输家", "输家"), ("C1随机", "C1"), ("C2ADV匹配", "C2")]:
            r = A.运行账户(P, 池[key], h, 公司行动表=公司行动表)
            指标 = 全指标(r)
            指标.update({"种子": 种子, "组合": 名, "资金": C.主资金, "每日只数": C.每日只数主,
                       "成本倍数": 1.0, "延迟": 0})
            汇总.append(指标)
            if 种子 == C.主种子:
                净值表.append(r["净值"].assign(组合=名, 种子=种子))
                if 名 == "输家":
                    r["净值"].to_csv(结果 / "账户_主账户_逐日净值.csv", index=False, encoding="utf-8-sig")
                    r["日志"].to_csv(结果 / "账户_主账户_交易日志.csv", index=False, encoding="utf-8-sig")
                    自检 = A.会计自检(r)
            if 名 == "输家":
                种子表.append({"种子": 种子, **{k: 指标[k] for k in
                             ("CAGR", "最大回撤", "期末净值", "总收益", "2010-2015_CAGR", "2016-2020_CAGR",
                              "2010-2015_最大回撤", "2016-2020_最大回撤", "平均仓位", "成交率", "部署率")}})
        print(f"种子 {种子} 完成", flush=True)

    # 容量与小额
    for 资金 in C.资金档:
        if 资金 == C.主资金:
            continue
        r = A.运行账户(P, 主池["输家"], h, 资金=资金, 公司行动表=公司行动表)
        指 = 全指标(r)
        零手 = float((r["日志"]["方向"] == "买入失败").mean()) if len(r["日志"]) else np.nan
        指.update({"种子": C.主种子, "组合": f"资金{资金:,.0f}", "资金": 资金,
                  "每日只数": C.每日只数主, "成本倍数": 1.0, "延迟": 0, "零手或失败率": 零手})
        汇总.append(指)

    # 每日只数
    for n in C.每日只数档:
        if n == C.每日只数主:
            continue
        r = A.运行账户(P, 主池["输家"], h, 每日只数=n, 公司行动表=公司行动表)
        指 = 全指标(r)
        指.update({"种子": C.主种子, "组合": f"每日{n}只", "资金": C.主资金,
                  "每日只数": n, "成本倍数": 1.0, "延迟": 0})
        汇总.append(指)

    # 成本 ×2 与延迟 1 日
    for 名, 倍, 延 in [("成本×2", 2.0, 0), ("延迟1日", 1.0, 1)]:
        r = A.运行账户(P, 主池["输家"], h, 成本倍数=倍, 延迟=延, 公司行动表=公司行动表)
        指 = 全指标(r)
        指.update({"种子": C.主种子, "组合": 名, "资金": C.主资金, "每日只数": C.每日只数主,
                  "成本倍数": 倍, "延迟": 延})
        汇总.append(指)
        if 名 == "成本×2":
            r["净值"].to_csv(结果 / "账户_成本x2_逐日净值.csv", index=False, encoding="utf-8-sig")
        else:
            r["净值"].to_csv(结果 / "账户_延迟1日_逐日净值.csv", index=False, encoding="utf-8-sig")

    表 = pd.DataFrame(汇总)
    表.to_csv(结果 / "账户_汇总.csv", index=False, encoding="utf-8-sig")
    pd.concat(净值表, ignore_index=True).to_parquet(中间 / "账户_逐日净值_三组合.parquet", index=False)
    pd.DataFrame(种子表).to_csv(结果 / "账户_种子.csv", index=False, encoding="utf-8-sig")
    (结果 / "账户_会计自检.json").write_text(json.dumps(自检, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(种子表).round(4).to_string(), flush=True)
    return P, u, s, 表, 自检


if __name__ == "__main__":
    主流程()
