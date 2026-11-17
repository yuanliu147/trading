#!/usr/bin/env python3
"""万元账户有限网格、匹配控制、种子分布与共同日历重采样。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from 数据准备 import load
from 账户引擎 import Assumptions, Engine, Spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "结果"
EVIDENCE = ROOT / "证据"
SEEDS = [20261030, 101, 202, 303, 404, 505, 606, 707]


def specs():
    core = [Spec(f"B{int(b * 100):02d}_M{m}", band=b, months=m)
            for b in (0.1, 0.2, 0.4) for m in (3, 6)]
    core += [Spec(f"M{w}", exit_kind="market", window=w) for w in (120, 180, 250)]
    core += [Spec(f"T{k}", exit_kind="trail", multiple=float(k)) for k in (2, 3, 4)]
    controls = [Spec(f"C_M{m}", months=m, score="control", group="control") for m in (3, 6)]
    controls += [Spec(f"C_MKT{w}", exit_kind="market", window=w,
                      score="control", group="control") for w in (120, 180, 250)]
    controls += [Spec(f"C_ATR{k}", exit_kind="trail", multiple=float(k),
                      score="control", group="control") for k in (2, 3, 4)]
    diagnostics = [Spec("D_EP", score="ep", group="diagnostic"),
                   Spec("D_BP", score="bp", group="diagnostic")]
    diagnostics += [Spec(f"D_LAG{lag}", lag=lag, group="diagnostic") for lag in (20, 60)]
    for n in (1, 4):
        diagnostics += [Spec(f"D_S{n}_BASE", slots=n, group="diagnostic"),
                        Spec(f"D_S{n}_M180", slots=n, exit_kind="market", window=180,
                             group="diagnostic"),
                        Spec(f"D_S{n}_T3", slots=n, exit_kind="trail", multiple=3.0,
                             group="diagnostic")]
    assert (len(core), len(controls), len(diagnostics)) == (12, 8, 10)
    return core, controls, diagnostics


def control_id(s):
    if s.exit_kind == "market":
        return f"C_MKT{s.window}"
    if s.exit_kind == "trail":
        return f"C_ATR{int(s.multiple)}"
    return f"C_M{s.months}"


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def freeze_check():
    p = EVIDENCE / "设计冻结哈希.json"
    expected = json.loads(p.read_text(encoding="utf-8"))
    for name, digest in expected["文件SHA256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"冻结设计已变更：{name}")
    return expected


def save_detail(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("净值", "日志", "回合", "期末持仓"):
        if key in result:
            result[key].to_csv(folder / f"{key}.csv", index=False, encoding="utf-8-sig")
    dump(folder / "会计自检.json", result["自检"])


def annual_rows(nv, ident, seed):
    dates = pd.to_datetime(nv["日期"])
    x = nv["净值"].to_numpy(dtype=float)
    previous = float(x[0])
    rows = []
    for year in range(2011, 2021):
        indices = np.flatnonzero(dates.dt.year.to_numpy() == year)
        if not len(indices):
            continue
        vals = np.r_[previous, x[indices]]
        rows.append({"配置": ident, "种子": seed, "年份": year,
                     "年度收益": float(vals[-1] / previous - 1),
                     "年度内最大回撤": float(np.min(vals / np.maximum.accumulate(vals) - 1)),
                     "年初权益": previous, "年末权益": float(vals[-1])})
        previous = float(vals[-1])
    return rows


def summarize(table, core_ids):
    rows = []
    for ident, g in table.groupby("配置", sort=False):
        r = {"配置": ident, "类别": g["类别"].iloc[0], "种子数": len(g),
             "正CAGR种子数": int((g.CAGR > 0).sum()),
             "两段均正种子数": int(((g["2011-2015_CAGR"] > 0)
                                          & (g["2016-2020_CAGR"] > 0)).sum()),
             "CAGR中位": float(g.CAGR.median()), "CAGR最小": float(g.CAGR.min()),
             "CAGR最大": float(g.CAGR.max()), "CAGR四分位下": float(g.CAGR.quantile(0.25)),
             "CAGR四分位上": float(g.CAGR.quantile(0.75)),
             "回撤中位": float(g["最大回撤"].median()), "最深回撤": float(g["最大回撤"].min())}
        for col in ("2011-2015_CAGR", "2016-2020_CAGR", "期末净值", "平均仓位",
                    "完整回合数", "胜率", "盈亏比", "费用合计", "最大单票权重"):
            r[f"{col}中位"] = float(g[col].median())
        rows.append(r)
    result = pd.DataFrame(rows)
    result["描述性前沿"] = False
    core = result[result["配置"].isin(core_ids)]
    for idx, r in core.iterrows():
        dominates = ((core["CAGR中位"] >= r["CAGR中位"])
                     & (core["回撤中位"] >= r["回撤中位"])
                     & ((core["CAGR中位"] > r["CAGR中位"])
                        | (core["回撤中位"] > r["回撤中位"])))
        result.loc[idx, "描述性前沿"] = not bool(dominates.any())
    return result


def bootstrap(core, all_specs, logs, dates, b=1000):
    ids = [s.id for s in all_specs]
    index = {name: i for i, name in enumerate(ids)}
    columns, labels = [], []
    for s in core:
        own = logs[index[s.id]].mean(axis=0)
        columns.append(own)
        labels.append({"配置": s.id, "比较": "账户对数收益对零", "控制": "现金零收益"})
        ctrl = logs[index[control_id(s)]].mean(axis=0)
        columns.append(own - ctrl)
        labels.append({"配置": s.id, "比较": "同种子同退出控制差", "控制": control_id(s)})
    y = np.column_stack(columns)
    assert y.shape[1] == 24
    n = len(y)
    block = 63
    rng = np.random.default_rng(20261030)
    means = y.mean(axis=0)
    boots = np.empty((b, y.shape[1]))
    for start in range(0, b, 25):
        count = min(25, b - start)
        beginnings = rng.integers(0, n, size=(count, int(np.ceil(n / block))))
        indices = ((beginnings[:, :, None] + np.arange(block)) % n).reshape(count, -1)[:, :n]
        boots[start:start + count] = y[indices].mean(axis=1)
    null_means = boots - means
    p = (1 + (null_means >= means).sum(axis=0)) / (b + 1)
    order = np.argsort(p, kind="stable")
    corrected = np.empty_like(p)
    corrected[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    factor = n / years
    limits = np.quantile(boots, [0.025, 0.975], axis=0)
    records = []
    for j, label in enumerate(labels):
        records.append({**label, "年化对数均值": means[j] * factor,
                        "95点区间下限": limits[0, j] * factor,
                        "95点区间上限": limits[1, j] * factor,
                        "单侧p": p[j], "Holm校正p": corrected[j],
                        "共同交易日数": n, "种子聚合": "逐日等权平均种子对数收益，非独立样本",
                        "块长交易日": block, "重采样次数": b,
                        "结论边界": "历史近似，点区间非选择后同时区间"})
    return pd.DataFrame(records)


def main():
    freeze = freeze_check()
    core, controls, diagnostics = specs()
    all_specs = core + controls + diagnostics
    dump(EVIDENCE / "全部配置登记.json", [asdict(s) for s in all_specs])
    data = load()
    engine = Engine(data)
    rows, audits, years = [], [], []
    matrices = []
    dates = None
    for spec in all_specs:
        seed_logs = []
        for seed in SEEDS:
            detail = seed == SEEDS[0] and spec.group in ("core", "control")
            r = engine.run(spec, seed, detail=detail)
            nv = r["净值"]
            run_dates = nv["日期"].astype(str).tolist()
            if dates is None:
                dates = run_dates
            assert run_dates == dates, f"日期不一致 {spec.id}/{seed}"
            assert dates[0] == "2010-12-31" and dates[-1] <= "2020-12-31"
            x = nv["净值"].to_numpy(dtype=float)
            if not np.all(np.isfinite(x)) or not np.all(x > 0):
                raise RuntimeError(f"无法进行正净值复利统计：{spec.id}/{seed}")
            assert abs(x[0] - 10000) < 1e-6
            seed_logs.append(np.log(x[1:] / x[:-1]))
            rows.append({"配置": spec.id, "类别": spec.group, "种子": seed, **r["汇总"]})
            audits.append({"配置": spec.id, "种子": seed, **r["自检"]})
            years.extend(annual_rows(nv, spec.id, seed))
            if detail:
                save_detail(OUT / "主种子账户" / spec.id, r)
        matrices.append(np.stack(seed_logs))
        print(f"完成 {spec.id}：8种子；不以单一种子择优", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "全部账户.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(OUT / "全部账户年度.csv", index=False, encoding="utf-8-sig")
    dump(EVIDENCE / "全部账户自检.json", audits)
    logs = np.stack(matrices)
    np.savez_compressed(OUT / "共同日历账户收益.npz", logs=logs,
                        ids=np.array([s.id for s in all_specs]), seeds=np.array(SEEDS),
                        dates=np.array(dates), return_dates=np.array(dates[1:]),
                        base_date=np.array(dates[0]))
    summary = summarize(table, [s.id for s in core])
    summary.to_csv(OUT / "配置种子分布.csv", index=False, encoding="utf-8-sig")
    stats = bootstrap(core, all_specs, logs, dates)
    stats.to_csv(OUT / "共同日期块统计.csv", index=False, encoding="utf-8-sig")

    # 压力不进入选择集；中心固定，前沿由预先声明的二目标定义产生。
    chosen = set(summary.loc[summary["描述性前沿"], "配置"]) | {"B20_M3", "M180", "T3"}
    chosen_specs = [s for s in core if s.id in chosen]
    cost_specs = chosen_specs + [s for s in controls if s.id in {control_id(x) for x in chosen_specs}]
    pressure, pressure_audits = [], []
    for spec in cost_specs:
        for seed in SEEDS:
            r = engine.run(spec, seed, Assumptions(fee_mult=2.0), detail=seed == SEEDS[0])
            pressure.append({"配置": spec.id, "情景": "执行费用与滑点双倍", "种子": seed, **r["汇总"]})
            pressure_audits.append({"配置": spec.id, "情景": "执行费用与滑点双倍", "种子": seed, **r["自检"]})
            if seed == SEEDS[0]:
                save_detail(OUT / "压力主种子" / f"{spec.id}_费用双倍", r)
    accounting = [
        ("红利即时可用假设", Assumptions(dividend_delay=0)),
        ("红利90日可用假设", Assumptions(dividend_delay=90)),
        ("送转即时可卖假设", Assumptions(bonus_lock=0)),
        ("送转60日可卖假设", Assumptions(bonus_lock=60)),
        ("送转不计税假设", Assumptions(bonus_tax=0.0)),
        ("即时低税组合假设", Assumptions(dividend_delay=0, bonus_lock=0, bonus_tax=0.0)),
        ("延迟组合假设", Assumptions(dividend_delay=90, bonus_lock=60)),
        ("连续缺行20日市值归零压力", Assumptions(stale_haircut=1.0)),
    ]
    for spec in chosen_specs:
        for label, assumption in accounting:
            r = engine.run(spec, SEEDS[0], assumption, detail=True)
            pressure.append({"配置": spec.id, "情景": label, "种子": SEEDS[0], **r["汇总"]})
            pressure_audits.append({"配置": spec.id, "情景": label, "种子": SEEDS[0], **r["自检"]})
            save_detail(OUT / "压力主种子" / f"{spec.id}_{label}", r)
        print(f"完成压力 {spec.id}", flush=True)
    pd.DataFrame(pressure).to_csv(OUT / "成本与会计情景.csv", index=False, encoding="utf-8-sig")
    dump(EVIDENCE / "压力账户自检.json", pressure_audits)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "代码").glob("*.py"))}
    dump(EVIDENCE / "运行记录.json", {"完成时间": str(pd.Timestamp.now()), "设计冻结": freeze,
          "代码SHA256": source_hashes, "主账户单元": len(rows), "压力单元": len(pressure),
          "核心配置": [s.id for s in core], "控制配置": [s.id for s in controls],
          "诊断配置": [s.id for s in diagnostics], "压力核心": sorted(chosen),
          "账户日期": [dates[0], dates[-1]], "共同日期数": len(dates),
          "复现命令": ["python3 -B 代码/数据准备.py", "python3 -B 代码/测试_账户引擎.py",
                        "python3 -B 代码/研究主流程.py"],
          "说明": "全部为已使用历史的近似账户，不是新样本外或实盘验证；未打开2021及以后年度行情分区"})
    print(summary[summary["类别"] == "core"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
