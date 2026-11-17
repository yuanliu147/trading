#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2 槽小资金：不同硬止损档位的对照。

风险预算固定为 3% 本金（600 元），并按“止损 + 2% 跳空缓冲”倒推目标市值：
    target = 600 / (hard_stop + 0.02)
这样 5%/6%/8%/10%/12% 五档都在同一个计划风险下比较，不会因为止损越宽、
仓位越小而自动占便宜。
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("small", HERE / "15_两槽小资金.py")
small = importlib.util.module_from_spec(spec)
sys.modules["small"] = small
spec.loader.exec_module(small)

OUT = small.OUT
OUT.mkdir(exist_ok=True)


def main():
    data, bench, dates, uni, sig = small.load_inputs()
    masks = small.build_masks(sig)
    rows = []
    yearly = []
    for universe_name in ["F1_MA20", "F2_MA20"]:
        for hs in [0.05, 0.06, 0.08, 0.10, 0.12]:
            risk_stop = hs + 0.02
            name = f"{universe_name}_b3_HS{int(hs*100)}_缓冲2"
            print(f"\n运行 {name} …", flush=True)
            r = small.run_config(name, sig, masks[universe_name], "hist_norm",
                                 data, bench, dates, uni, max_slots=2,
                                 risk_stop=risk_stop, exit_mode="b3",
                                 hard_stop=hs)
            rows += small.add_metrics(name, r, bench)
            yearly.append(small.yearly(r))
            eq = r["equity"].copy(); eq.columns = ["日期", "权益", "持仓数", "持仓市值", "仓位比例"]
            eq.to_csv(OUT / f"权益_{name}.csv", index=False, encoding="utf-8-sig")
            r["trades"].to_csv(OUT / f"交易_{name}.csv", index=False, encoding="utf-8-sig")
            print(f"  期末 {r['final_equity']:,.0f}；交易 {len(r['trades'])}；{r['stats']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "两槽止损对照.csv", index=False, encoding="utf-8-sig")
    pd.concat(yearly, ignore_index=True).to_csv(OUT / "两槽止损对照_年度.csv",
                                                index=False, encoding="utf-8-sig")
    cols = ["配置", "区间", "年化", "最大回撤", "Calmar", "Sharpe", "交易", "胜率",
            "平均每笔", "PF", "平均仓位", "平均持仓数", "最大单笔亏损占本金%",
            "亏损>3%笔数", "前5笔占比%", "同暴露基准年化", "同暴露基准回撤"]
    print("\n", df[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
