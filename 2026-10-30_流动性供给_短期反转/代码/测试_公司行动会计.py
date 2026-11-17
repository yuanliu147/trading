# -*- coding: utf-8 -*-
"""公司行动会计的合成单元测试与主账户真实事件核验。"""
from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import 配置 as C
import 账户层 as A

证券 = "000001.SZ"
除权日 = date(2020, 1, 3)
合成证据 = {}
真实证据 = {}


def 合成面板():
    df = pd.DataFrame({"symbol": [证券] * 4, "td": np.arange(4),
                     "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])})
    return SimpleNamespace(
        df=df, symbols=np.array([证券]), td_min=0, td_max=3,
        rid=np.arange(4, dtype=np.int32).reshape(1, 4),
        O=np.full(4, 10.0), C=np.full(4, 10.0), VOL=np.full(4, 1e9),
        SUSP=np.zeros(4, dtype=bool), LDO=np.zeros(4, dtype=bool),
        LUO=np.zeros(4, dtype=bool), LUP=np.full(4, np.inf), LDP=np.full(4, np.nan))


def 运行合成(公司行动):
    return A.运行账户(合成面板(), {0: np.array([0])}, 2, 资金=100_000.0, 每日只数=1,
                    公司行动表={(证券, 除权日): 公司行动})


class 合成公司行动测试(unittest.TestCase):
    def test_分红送转配股同日(self):
        r = 运行合成({"dividend_per_share": 0.5, "per_share_bonus": 0.2,
                    "rights_ratio": 0.3, "rights_price": 5.0})
        log = r["日志"]
        买 = log[log["方向"] == "买入"].iloc[0]
        卖 = log[log["方向"] == "卖出"].iloc[0]
        分 = log[log["方向"] == "分红"].iloc[0]
        送 = log[log["方向"] == "送转"].iloc[0]
        配 = log[log["方向"] == "配股"].iloc[0]
        旧股数 = int(买["股数"])
        分红 = round(旧股数 * 0.5 * 0.8, 2)
        送转 = int(np.floor(旧股数 * 0.2 + 1e-9))
        配股 = int(np.floor(旧股数 * 0.3 + 1e-9))
        配股成本 = round(配股 * 5.0, 2)

        self.assertEqual(int(分["旧股数"]), 旧股数)
        self.assertAlmostEqual(float(分["金额"]), 分红, places=2)
        self.assertEqual(int(送["股数"]), 送转)
        self.assertEqual(int(配["应配股数"]), 配股)  # 不以送转后的股数为基数
        self.assertEqual(int(配["股数"]), 配股)
        self.assertEqual(配["配股状态"], "已认购")
        self.assertAlmostEqual(float(配["金额"]), 配股成本, places=2)
        self.assertEqual(int(卖["股数"]), 旧股数 + 送转 + 配股)
        除权净值 = r["净值"].set_index("日期").loc["2020-01-03"]
        self.assertAlmostEqual(float(除权净值["现金流"]), 分红 - 配股成本, places=2)
        预期盈亏 = round(卖["金额"] - 卖["费用合计"] + 卖["分红"]
                       - 买["金额"] - 买["费用合计"] - 配股成本, 2)
        self.assertAlmostEqual(float(卖["盈亏"]), 预期盈亏, places=2)
        检 = A.会计自检(r)
        self.assertEqual(检["股数不守恒证券数"], 0)
        self.assertAlmostEqual(检["现金守恒差"], 0.0, places=6)
        self.assertAlmostEqual(检["逐日现金流最大偏差"], 0.0, places=2)
        self.assertAlmostEqual(检["卖出盈亏重算最大偏差"], 0.0, places=2)
        self.assertEqual(检["配股未付费增股数"], 0)
        合成证据["分红送转配股同日"] = {
            "旧股数": 旧股数, "税后分红": 分红, "送转股数": 送转,
            "应配及实配股数": 配股, "配股成本": 配股成本,
            "卖出股数": int(卖["股数"]), "卖出盈亏": float(卖["盈亏"]),
            "除权日现金流": float(除权净值["现金流"]), "会计自检": 检}

    def test_配股现金不足不得免费增股(self):
        r = 运行合成({"dividend_per_share": 0.0, "per_share_bonus": 0.0,
                    "rights_ratio": 0.3, "rights_price": 1e9})
        log = r["日志"]
        旧股数 = int(log.loc[log["方向"] == "买入", "股数"].iloc[0])
        配 = log[log["方向"] == "配股"].iloc[0]
        卖 = log[log["方向"] == "卖出"].iloc[0]
        self.assertEqual(配["配股状态"], "现金不足_全部放弃")
        self.assertGreater(int(配["应配股数"]), 0)
        self.assertEqual(int(配["股数"]), 0)
        self.assertEqual(float(配["金额"]), 0.0)
        self.assertGreater(float(配["现金缺口"]), 0.0)
        self.assertEqual(int(卖["股数"]), 旧股数)
        self.assertEqual(float(卖["配股成本"]), 0.0)
        检 = A.会计自检(r)
        self.assertEqual(检["现金为负天数"], 0)
        self.assertEqual(检["股数不守恒证券数"], 0)
        self.assertEqual(检["配股现金不足笔数"], 1)
        self.assertEqual(检["配股未付费增股数"], 0)
        self.assertAlmostEqual(检["现金守恒差"], 0.0, places=6)
        合成证据["配股现金不足"] = {
            "旧股数": 旧股数, "应配股数": int(配["应配股数"]), "实配股数": int(配["股数"]),
            "配股付现": float(配["金额"]), "现金缺口": float(配["现金缺口"]),
            "卖出股数": int(卖["股数"]), "会计自检": 检}


def 核验真实事件():
    log = pd.read_csv(C.结果目录 / "账户_主账户_交易日志.csv", low_memory=False)
    log["日期"] = pd.to_datetime(log["日期"])
    ca = pd.read_parquet(
        C.母项目数据 / "corp_actions.parquet",
        columns=["symbol", "ex_date", "dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price"],
        filters=[("ex_date", ">=", date(2010, 1, 1)), ("ex_date", "<=", date(2020, 12, 31))])
    ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    源 = {(r.symbol, r.ex_date): r for r in ca.itertuples(index=False)}
    log["买入日"] = pd.to_datetime(log["买入日"])

    # 独立按实际持仓区间重建应发生事件，防止只验证已记录事件而漏掉缺失事件。
    买 = log[log["方向"] == "买入"].copy()
    卖 = log[log["方向"] == "卖出"][["证券", "买入日", "日期"]].rename(columns={"日期": "卖出日"})
    assert not 买.duplicated(["证券", "买入日"]).any()
    assert not 卖.duplicated(["证券", "买入日"]).any()
    区间 = 买[["证券", "买入日"]].merge(卖, on=["证券", "买入日"], how="left", validate="one_to_one")
    源按证券 = {}
    for symbol, g in ca.groupby("symbol"):
        源按证券[symbol] = list(g.itertuples(index=False))
    应发生 = Counter()
    for p in 区间.itertuples(index=False):
        持有至 = p.卖出日 if pd.notna(p.卖出日) else log["日期"].max()
        for src in 源按证券.get(p.证券, []):
            if p.买入日 < src.ex_date <= 持有至:
                for direction, col in [("分红", "dividend_per_share"), ("送转", "per_share_bonus"),
                                       ("配股", "rights_ratio")]:
                    if getattr(src, col) > 0:
                        应发生[(p.证券, p.买入日, src.ex_date, direction)] += 1
    实际 = Counter((r.证券, r.买入日, r.日期, r.方向)
                 for r in log[log["方向"].isin(["分红", "送转", "配股"])].itertuples(index=False))
    assert 实际 == 应发生, f"公司行动漏记或多记：{实际 - 应发生} / {应发生 - 实际}"

    计数 = {d: int((log["方向"] == d).sum()) for d in ["分红", "送转", "配股"]}
    assert 计数["送转"] > 0 and 计数["配股"] > 0, f"真实事件覆盖不足：{计数}"
    样例 = {}
    不足 = 0
    for row in log[log["方向"].isin(["分红", "送转", "配股"])].itertuples(index=False):
        src = 源[(row.证券, row.日期)]
        if row.方向 == "分红":
            预期金额 = round(int(row.旧股数) * src.dividend_per_share * 0.8, 2)
            assert abs(row.金额 - 预期金额) < 0.005, (row, 预期金额)
        elif row.方向 == "送转":
            预期股数 = int(np.floor(int(row.旧股数) * src.per_share_bonus + 1e-9))
            assert int(row.股数) == 预期股数 and row.金额 == 0, (row, 预期股数)
            样例.setdefault("送转", {"证券": row.证券, "除权日": str(row.日期.date()),
                                  "旧股数": int(row.旧股数), "源比例": src.per_share_bonus,
                                  "实际增加股数": int(row.股数)})
        else:
            应配 = int(np.floor(int(row.旧股数) * src.rights_ratio + 1e-9))
            assert int(row.应配股数) == 应配, (row, 应配)
            if row.配股状态 == "已认购":
                assert int(row.股数) == 应配
                assert abs(row.金额 - round(应配 * src.rights_price, 2)) < 0.005
            else:
                assert row.配股状态 == "现金不足_全部放弃"
                assert int(row.股数) == 0 and row.金额 == 0 and row.现金缺口 > 0
                不足 += 1
            样例.setdefault("配股", {"证券": row.证券, "除权日": str(row.日期.date()),
                                  "旧股数": int(row.旧股数), "源比例": src.rights_ratio,
                                  "源配股价": src.rights_price, "应配股数": 应配,
                                  "实配股数": int(row.股数), "付现": float(row.金额),
                                  "状态": row.配股状态})
    return {"逐笔核验通过": True, "独立持仓区间无漏记或多记": True,
            "遗漏或多余事件数": int(sum((实际 - 应发生).values()) + sum((应发生 - 实际).values())),
            "事件笔数": 计数, "配股现金不足笔数": 不足,
            "现金不足全部放弃股数": int(log.loc[log["配股状态"] == "现金不足_全部放弃", "应配股数"].sum()),
            "样例": 样例}


class 真实公司行动测试(unittest.TestCase):
    def test_主账户真实受影响事件(self):
        global 真实证据
        真实证据 = 核验真实事件()
        self.assertTrue(真实证据["逐笔核验通过"])


if __name__ == "__main__":
    仅合成 = "--synthetic" in sys.argv
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(合成公司行动测试))
    if not 仅合成:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(真实公司行动测试))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful() and not 仅合成:
        证据 = {"生成时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
               "合成测试": 合成证据, "真实受影响事件核验": 真实证据,
               "主账户新会计自检": json.loads((C.结果目录 / "账户_会计自检.json").read_text(encoding="utf-8"))}
        (C.证据目录 / "公司行动会计测试.json").write_text(
            json.dumps(证据, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(证据, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
