"""一次读取每只股票，为所有已登记规则生成因果收盘信号；不计算任何账户收益。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from 策略注册 import registry
from 信号与订单 import make_orders, ORDER_COLUMNS
from 交易数据 import TradingData, DATA, ROOT, mainboard


def generate():
    outdir = DATA / "策略信号"
    outdir.mkdir(exist_ok=True)
    strategies = registry()
    arrays = {s.name: [] for s in strategies}
    data = TradingData(feature_cache=1)
    codes = [code for code in data.codes if mainboard(code)]
    started = time.time()
    for n, code in enumerate(codes, 1):
        raw, features = data.raw(code), data.features(code)
        context = data.context(code)
        for strategy in strategies:
            orders = make_orders(code, raw, features, strategy, context)
            if not orders.empty:
                arrays[strategy.name].append(orders)
        if n % 250 == 0:
            print(f"主板信号{n}/{len(codes)}，{time.time()-started:.1f}秒", flush=True)
    metrics = []
    for strategy in strategies:
        frames = arrays[strategy.name]
        orders = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ORDER_COLUMNS)
        orders = orders.sort_values(["signal_date", "code"]).reset_index(drop=True)
        if orders.duplicated(["signal_date", "code"]).any():
            raise AssertionError("同股票日重复信号")
        orders.to_parquet(outdir / f"{strategy.name}.parquet", index=False, compression="zstd")
        for period, start, end in [("开发", "2018-01-01", "2020-12-31"), ("验证", "2021-01-01", "2022-12-31"),
                                   ("历史留出", "2023-01-01", "2025-12-31"), ("二零二六", "2026-01-01", "2026-10-22")]:
            selected = orders[(orders.signal_date >= start) & (orders.signal_date <= end)]
            metrics.append({"策略": strategy.name, "区段": period, "信号数": len(selected),
                            "股票数": selected.code.nunique(), "未知历史状态信号数": int((~selected.st_status_known).sum()),
                            "可参与选择": strategy.select})
    pd.DataFrame(metrics).to_csv(ROOT / "03_结果" / "注册信号数量.csv", index=False, encoding="utf-8-sig")
    version = {"生成时间": pd.Timestamp.now().isoformat(), "主板可得股票数": len(codes),
               "本金": 20000, "仅普通主板": True, "策略": [asdict(s) for s in strategies],
               "信号模块校验": hashlib.sha256((Path(__file__).parent / "信号与订单.py").read_bytes()).hexdigest(),
               "说明": "只统计信号，没有读取或生成前瞻收益及账户收益，历史数据缺口见数据审计"}
    (ROOT / "04_证据" / "信号生成审计.json").write_text(json.dumps(version, ensure_ascii=False, indent=2))
    print(f"{len(strategies)}组已生成，总耗时{time.time()-started:.1f}秒", flush=True)


def refresh_episode_keys():
    """审计修复只改变阶段键；逐列断言信号日期、股票、止损、排序等全部不变。"""
    data = TradingData(feature_cache=1)
    changes = []
    for strategy in registry():
        if strategy.line != "pivot":
            continue
        path = DATA / "策略信号" / f"{strategy.name}.parquet"
        original = pd.read_parquet(path)
        updated = []
        for code, group in original.groupby("code"):
            new = make_orders(code, data.raw(code), data.features(code), strategy, data.context(code))
            old = group.reset_index(drop=True)
            pd.testing.assert_frame_equal(old.drop(columns="episode_key"), new.drop(columns="episode_key"), check_exact=True)
            updated.append(new)
        result = pd.concat(updated, ignore_index=True).sort_values(["signal_date", "code"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(original.drop(columns="episode_key"), result.drop(columns="episode_key"), check_exact=True)
        changes.append({"策略": strategy.name, "信号数": len(result),
                        "变更阶段键数量": int(original.episode_key.ne(result.episode_key).sum()), "其他字段完全不变": True})
        result.to_parquet(path, index=False, compression="zstd")
    audit = json.loads((ROOT / "04_证据" / "信号生成审计.json").read_text())
    audit["初版信号模块校验"] = audit["信号模块校验"]
    audit["信号模块校验"] = hashlib.sha256((Path(__file__).parent / "信号与订单.py").read_bytes()).hexdigest()
    audit["结构键审计修复"] = changes
    (ROOT / "04_证据" / "信号生成审计.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(changes, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if "--refresh-keys" in sys.argv:
        refresh_episode_keys()
    else:
        generate()
