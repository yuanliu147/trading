#!/usr/bin/env python3
"""万元账户：只改持仓槽位与审视频率的冻结网格、匹配控制与共同日历重采样。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "2026-10-30_万元账户_低换手估值"
sys.path.insert(0, str(ROOT / "代码"))
sys.path.append(str(PRIOR / "代码"))

from 账户引擎 import Assumptions, Engine, Spec  # noqa: E402
from 数据准备 import load  # noqa: E402

OUT = ROOT / "结果"
EVIDENCE = ROOT / "证据"
SEEDS = [20261030, 101, 202, 303, 404, 505, 606, 707]
SLOTS = (3, 5, 8)
MONTHS = (1, 3, 6)
BLOCK = 63
BOOT = 1000
BOOT_SEED = 20261031


def specs():
    core = [Spec(f"S{s}_M{m}", slots=s, months=m) for s in SLOTS for m in MONTHS]
    controls = [Spec(f"C_S{s}_M{m}", slots=s, months=m, score="control", group="control")
                for s in SLOTS for m in MONTHS]
    assert len(core) == len(controls) == 9
    return core, controls


def control_id(spec):
    return f"C_{spec.id}"


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_check():
    record = json.loads((EVIDENCE / "设计冻结哈希.json").read_text(encoding="utf-8"))
    for name, digest in record["文件SHA256"].items():
        actual = sha256(ROOT / name)
        if actual != digest:
            raise RuntimeError(f"冻结文件已变更：{name}")
    return record


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


def index_benchmark(data, dates):
    full_dates = list(data["meta"]["日期"])
    index = np.asarray(data["index"], dtype=float)
    assert len(index) == len(full_dates)
    offset = full_dates.index(dates[0])
    index = index[offset:]
    assert len(index) == len(dates)
    start, end = index[0], index[-1]
    days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
    cagr = (end / start) ** (365.25 / days) - 1
    mdd = float((index / np.maximum.accumulate(index) - 1).min())
    return {"名称": "沪深300价格指数（不含分红，非ETF）",
            "起点日": dates[0], "终点日": dates[-1],
            "起点": float(start), "终点": float(end),
            "CAGR": float(cagr), "最大回撤": mdd,
            "总收益": float(end / start - 1)}


def bootstrap(core, all_specs, logs, dates, b=BOOT):
    ids = [s.id for s in all_specs]
    index = {name: i for i, name in enumerate(ids)}
    columns, labels = [], []
    for s in core:
        own = logs[index[s.id]].mean(axis=0)
        columns.append(own)
        labels.append({"配置": s.id, "比较": "账户对数收益对零", "控制": "现金零收益"})
        ctrl = logs[index[control_id(s)]].mean(axis=0)
        columns.append(own - ctrl)
        labels.append({"配置": s.id, "比较": "同槽位同频率随机对照差", "控制": control_id(s)})
    y = np.column_stack(columns)
    assert y.shape[1] == 18
    n = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    means = y.mean(axis=0)
    boots = np.empty((b, y.shape[1]))
    for start in range(0, b, 25):
        count = min(25, b - start)
        beginnings = rng.integers(0, n, size=(count, int(np.ceil(n / BLOCK))))
        indices = ((beginnings[:, :, None] + np.arange(BLOCK)) % n).reshape(count, -1)[:, :n]
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
                        "块长交易日": BLOCK, "重采样次数": b,
                        "结论边界": "历史近似，点区间非选择后同时区间"})
    return pd.DataFrame(records)


def summarize(table, core_ids, stats):
    rows = []
    for ident, g in table.groupby("配置", sort=False):
        r = {"配置": ident, "类别": g["类别"].iloc[0], "种子数": len(g),
             "槽位": int(g["槽位"].iloc[0]), "审视频率月": int(g["审视频率月"].iloc[0]),
             "正CAGR种子数": int((g.CAGR > 0).sum()),
             "两段均正种子数": int(((g["2011-2015_CAGR"] > 0)
                                          & (g["2016-2020_CAGR"] > 0)).sum()),
             "CAGR中位": float(g.CAGR.median()), "CAGR最小": float(g.CAGR.min()),
             "CAGR最大": float(g.CAGR.max()), "CAGR四分位下": float(g.CAGR.quantile(0.25)),
             "CAGR四分位上": float(g.CAGR.quantile(0.75)),
             "回撤中位": float(g["最大回撤"].median()), "最深回撤": float(g["最大回撤"].min()),
             "期末净值中位": float(g["期末净值"].median())}
        for col in ("2011-2015_CAGR", "2016-2020_CAGR", "平均仓位", "买入失败数",
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
    own = stats[stats["比较"] == "账户对数收益对零"].set_index("配置")
    diff = stats[stats["比较"] == "同槽位同频率随机对照差"].set_index("配置")
    for col, source, name in (("对零单侧p", own, "单侧p"), ("对零Holm", own, "Holm校正p"),
                              ("对对照单侧p", diff, "单侧p"), ("对对照Holm", diff, "Holm校正p"),
                              ("对对照年化对数差", diff, "年化对数均值")):
        result[col] = result["配置"].map(source[name])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="只跑S3_M1与对照、前2个种子，不跑压力")
    args = parser.parse_args()
    freeze = freeze_check()
    core, controls = specs()
    all_specs = core + controls
    data = load()
    engine = Engine(data)
    seeds = SEEDS[:2] if args.smoke else SEEDS
    chosen = [s for s in all_specs if (not args.smoke or s.id in ("S3_M1", "C_S3_M1"))]
    dump(EVIDENCE / "本轮配置登记.json", [asdict(s) for s in all_specs])
    rows, audits, years = [], [], []
    matrices, dates = [], None
    for spec in chosen:
        seed_logs = []
        for seed in seeds:
            detail = seed == seeds[0]
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
            for key, value in r["自检"].items():
                if abs(float(value)) > 1e-9:
                    raise RuntimeError(f"自检未通过：{spec.id}/{seed}/{key}={value}")
            seed_logs.append(np.log(x[1:] / x[:-1]))
            rows.append({"配置": spec.id, "类别": spec.group, "种子": seed,
                         "槽位": spec.slots, "审视频率月": spec.months, **r["汇总"]})
            audits.append({"配置": spec.id, "种子": seed, **r["自检"]})
            years.extend(annual_rows(nv, spec.id, seed))
            if detail:
                save_detail(OUT / "主种子账户" / spec.id, r)
        matrices.append(np.stack(seed_logs))
        print(f"完成 {spec.id}：{len(seeds)}种子", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "全部账户.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(OUT / "全部账户年度.csv", index=False, encoding="utf-8-sig")
    dump(EVIDENCE / "全部账户自检.json", audits)
    if args.smoke:
        print(table[["配置", "种子", "CAGR", "最大回撤", "期末净值", "买入失败数"]].to_string(index=False))
        return
    logs = np.stack(matrices)
    np.savez_compressed(OUT / "共同日历账户收益.npz", logs=logs,
                        ids=np.array([s.id for s in all_specs]),
                        seeds=np.array(SEEDS), dates=np.array(dates),
                        return_dates=np.array(dates[1:]), base_date=np.array(dates[0]))
    stats = bootstrap(core, all_specs, logs, dates)
    stats.to_csv(OUT / "共同日期块统计.csv", index=False, encoding="utf-8-sig")
    summary = summarize(table, [s.id for s in core], stats)
    benchmark = index_benchmark(data, dates)
    dump(OUT / "指数基准.json", benchmark)

    pressure = []
    pressure_audits = []
    for spec in core:
        for seed in SEEDS:
            r = engine.run(spec, seed, Assumptions(fee_mult=2.0))
            for key, value in r["自检"].items():
                if abs(float(value)) > 1e-9:
                    raise RuntimeError(f"压力自检未通过：{spec.id}/{seed}/{key}={value}")
            pressure.append({"配置": spec.id, "情景": "费用与滑点双倍", "种子": seed, **r["汇总"]})
            pressure_audits.append({"配置": spec.id, "情景": "费用与滑点双倍", "种子": seed, **r["自检"]})
        print(f"完成费用双倍 {spec.id}", flush=True)
    pressure_table = pd.DataFrame(pressure)
    fee2 = pressure_table.groupby("配置")["CAGR"].median()
    summary["费用双倍CAGR中位"] = summary["配置"].map(fee2)

    final = table.set_index(["配置", "种子"])["期末净值"]
    wins = []
    for spec in core:
        count = 0
        for seed in SEEDS:
            if float(final[(spec.id, seed)]) > float(final[(control_id(spec), seed)]) + 1e-9:
                count += 1
        wins.append({"配置": spec.id, "跑赢同种子对照种子数": count})
    summary = summary.merge(pd.DataFrame(wins), on="配置", how="left")
    summary.loc[summary["类别"] != "core", "跑赢同种子对照种子数"] = np.nan

    idx_cagr, idx_mdd = benchmark["CAGR"], benchmark["最大回撤"]
    summary["中位CAGR>0"] = summary["CAGR中位"] > 0
    summary["正种子>=6"] = summary["正CAGR种子数"] >= 6
    summary["两段中位>0"] = ((summary["2011-2015_CAGR中位"] > 0)
                            & (summary["2016-2020_CAGR中位"] > 0))
    summary["高于指数CAGR"] = summary["CAGR中位"] > idx_cagr
    summary["回撤不深于指数"] = summary["回撤中位"] >= idx_mdd
    summary["对照Holm<0.05"] = summary["对对照Holm"] < 0.05
    summary["费用双倍中位>0"] = summary["费用双倍CAGR中位"] > 0
    summary["候选线"] = (summary["中位CAGR>0"] & summary["正种子>=6"]
                         & summary["两段中位>0"] & (summary["类别"] == "core"))
    summary["交付线"] = (summary["候选线"] & summary["高于指数CAGR"]
                         & summary["回撤不深于指数"] & summary["对照Holm<0.05"]
                         & summary["费用双倍中位>0"])
    summary.to_csv(OUT / "配置种子分布.csv", index=False, encoding="utf-8-sig")

    candidates = summary.loc[summary["候选线"] & (summary["类别"] == "core"), "配置"].tolist()
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
    for ident in candidates:
        spec = next(s for s in core if s.id == ident)
        for label, assumption in accounting:
            r = engine.run(spec, SEEDS[0], assumption, detail=True)
            for key, value in r["自检"].items():
                if abs(float(value)) > 1e-9:
                    raise RuntimeError(f"会计情景自检未通过：{ident}/{label}/{key}={value}")
            pressure.append({"配置": ident, "情景": label, "种子": SEEDS[0], **r["汇总"]})
            pressure_audits.append({"配置": ident, "情景": label, "种子": SEEDS[0], **r["自检"]})
            save_detail(OUT / "压力主种子" / f"{ident}_{label}", r)
        print(f"完成会计情景 {ident}", flush=True)
    pd.DataFrame(pressure).to_csv(OUT / "成本与会计情景.csv", index=False, encoding="utf-8-sig")
    dump(EVIDENCE / "压力账户自检.json", pressure_audits)

    best_cagr = summary.loc[summary["类别"] == "core"].sort_values("CAGR中位", ascending=False).iloc[0]
    best_mdd = summary.loc[summary["类别"] == "core"].sort_values("回撤中位", ascending=False).iloc[0]
    best_p = summary.loc[summary["类别"] == "core"].sort_values("对对照Holm").iloc[0]
    deliverable = summary.loc[summary["交付线"], "配置"].tolist()
    conclusion = {
        "判定口径": "候选线=中位CAGR>0、正CAGR种子>=6/8、两段中位>0；交付线另需高于沪深300价格指数CAGR、回撤不深于指数、对对照Holm<0.05、费用双倍中位>0。",
        "指数基准": benchmark,
        "候选配置": candidates,
        "交付配置": deliverable,
        "最高中位CAGR": {"配置": best_cagr["配置"], "值": float(best_cagr["CAGR中位"])},
        "最浅中位回撤": {"配置": best_mdd["配置"], "值": float(best_mdd["回撤中位"])},
        "最小对对照Holm": {"配置": best_p["配置"], "值": float(best_p["对对照Holm"])},
        "结论": ("存在通过交付线的配置：" + ",".join(deliverable)) if deliverable else
                ("有配置通过候选线但未通过交付线：" + ",".join(candidates)) if candidates else
                "没有配置通过候选线。",
        "边界": "历史近似账户研究；8种子不是8段独立历史；2016-2020不是纯净样本外；未读取2021及以后。"
    }
    dump(OUT / "结论.json", conclusion)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p)
                     for p in sorted((ROOT / "代码").glob("*.py"))}
    dump(EVIDENCE / "运行记录.json", {
        "完成时间": str(pd.Timestamp.now()),
        "设计冻结": freeze,
        "代码SHA256": source_hashes,
        "主账户单元": len(rows), "压力单元": len(pressure),
        "核心配置": [s.id for s in core], "对照配置": [s.id for s in controls],
        "种子": SEEDS, "账户日期": [dates[0], dates[-1]], "共同日期数": len(dates),
        "复现命令": ["python3 -B 代码/测试_账户引擎.py", "python3 -B 代码/研究主流程.py"],
        "说明": "全部为已使用历史的近似账户，不是新样本外或实盘验证；未打开2021及以后年度行情分区"
    })
    print(summary[summary["类别"] == "core"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
