#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单槽 F2 决策卡：只接受数值特征，不接受股票代码/名称。

用途
----
把“今天这个金叉值不值得占用唯一仓位、如果已持仓该不该走”做成可复核的
确定性判断。所有阈值来自 07_全信号买卖点复盘/04_单槽精选 的复盘结果；
脚本本身不联网、不读行情、不调用大模型。

入场（--entry）
    必需字段（全部为数值）：
      ret20, dist_ma20, rsi14, n_golden_60, s_vol, amt5_60, dist_ma60,
      hist, close
    可选：
      hist_norm（若给出，直接用作排序分；否则用 hist/close）
      weight（单槽建议权重，默认 0.30）
    F2 条件：
      ret20 <= 0.15
      dist_ma20 <= 0.08
      rsi14 <= 70
      n_golden_60 >= 3
      s_vol <= 0.72
      amt5_60 <= 0.26
      dist_ma60 >= -0.08
    排序分 = hist / close（价格归一化，避免原始 hist 偏向高价股）。
    通过只代表“候选合格”，仍需与同日其它候选比较排序分。

离场（--exit）
    必需字段：
      entry_price, peak_price, close
    可选：
      low（当日最低价；不给则用 close 判断 b3）
      death_cross（0/1）
    规则（收盘确认，次日开盘执行）：
      硬止损：close <= entry_price * 0.92
      b3：peak/entry - 1 >= 0.10 且 low <= peak * 0.92
      死叉：death_cross == 1
    任一触发即卖；优先级仅用于记录原因。

示例
----
python3 13_决策卡.py --entry --input '{"ret20":0.05,"dist_ma20":0.02,"rsi14":55,"n_golden_60":4,"s_vol":0.4,"amt5_60":-0.1,"dist_ma60":0.03,"hist":0.12,"close":10.0}'
python3 13_决策卡.py --exit --input '{"entry_price":10,"peak_price":11.2,"close":10.1,"low":10.0,"death_cross":0}'
python3 13_决策卡.py --entry --batch candidates.csv --out decisions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ENTRY_FIELDS = ["ret20", "dist_ma20", "rsi14", "n_golden_60",
                "s_vol", "amt5_60", "dist_ma60", "hist", "close"]
FORBIDDEN_FIELDS = {"code", "name", "symbol", "ticker", "股票代码", "股票名称", "名称", "代码"}


class InputError(ValueError):
    pass


def _num(d: dict[str, Any], key: str, required: bool = True) -> float | None:
    if key not in d or d[key] in (None, ""):
        if required:
            raise InputError(f"缺少数值字段: {key}")
        return None
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise InputError(f"字段 {key} 不是数值: {d[key]!r}")
    if v != v or v in (float("inf"), float("-inf")):
        raise InputError(f"字段 {key} 不是有限数值")
    return v


def check_no_identity(d: dict[str, Any]) -> None:
    hit = FORBIDDEN_FIELDS.intersection({str(k).lower() for k in d})
    if hit:
        raise InputError(
            "本脚本只接受数值特征，不接受股票代码/名称；请删除字段: "
            + ", ".join(sorted(hit)))


def entry_card(d: dict[str, Any], weight: float = 0.30) -> dict[str, Any]:
    check_no_identity(d)
    vals = {k: _num(d, k) for k in ENTRY_FIELDS}
    if vals["close"] <= 0:
        raise InputError("close 必须大于 0")
    if not (0 <= weight <= 1):
        raise InputError("weight 必须在 0 到 1 之间")

    checks = {
        "前20日涨幅<=15%": vals["ret20"] <= 0.15,
        "距MA20<=8%": vals["dist_ma20"] <= 0.08,
        "RSI14<=70": vals["rsi14"] <= 70,
        "60日金叉数>=3": vals["n_golden_60"] >= 3,
        "股性波动分<=0.72": vals["s_vol"] <= 0.72,
        "5日/60日成交额<=0.26": vals["amt5_60"] <= 0.26,
        "距MA60>=-8%": vals["dist_ma60"] >= -0.08,
    }
    reasons = [k for k, ok in checks.items() if not ok]
    hn = _num(d, "hist_norm", required=False)
    score = hn if hn is not None else vals["hist"] / vals["close"]
    passed = len(reasons) == 0 and vals["hist"] > 0

    card = {
        "可买": bool(passed),
        "建议仓位": round(weight, 4) if passed else 0.0,
        "排序分": round(float(score), 8),
        "排序分口径": "hist_norm（外部给出）" if hn is not None else "hist/close",
        "拒绝原因": reasons if vals["hist"] > 0 else reasons + ["MACD柱<=0"],
        "条件明细": {k: bool(v) for k, v in checks.items()},
        "执行": {
            "买入时点": "信号日收盘确认，下一交易日开盘",
            "涨停": "开盘涨停则不买",
            "单槽": "只占一个槽位；建议不超过总权益的 30%",
        },
        "离场计划": {
            "硬止损": "收盘较买入价 -8%，次日开盘卖",
            "b3追踪": "浮盈>=10% 后，从前一日已知峰值回撤 8%，次日开盘卖",
            "死叉": "收盘死叉，次日开盘卖",
        },
        "备注": "通过仅代表候选合格；同一天有多个候选时，选排序分最高者。",
    }
    return card


def exit_card(d: dict[str, Any]) -> dict[str, Any]:
    check_no_identity(d)
    entry = _num(d, "entry_price")
    peak = _num(d, "peak_price")
    close = _num(d, "close")
    low = _num(d, "low", required=False)
    death = _num(d, "death_cross", required=False) or 0.0
    if entry <= 0 or peak <= 0 or close <= 0:
        raise InputError("entry_price / peak_price / close 必须大于 0")
    if low is None:
        low = close

    reasons = []
    if close <= entry * 0.92:
        reasons.append("硬止损")
    if peak / entry - 1.0 >= 0.10 and low <= peak * 0.92:
        reasons.append("b3追踪")
    if death == 1:
        reasons.append("死叉")

    return {
        "卖出": bool(reasons),
        "原因": reasons,
        "执行": "下一交易日开盘卖出" if reasons else "继续持有",
        "当前浮盈": round(peak / entry - 1.0, 6),
        "距峰值回撤": round(low / peak - 1.0, 6),
        "规则": {
            "硬止损线": round(entry * 0.92, 4),
            "b3触发线": round(peak * 0.92, 4) if peak / entry - 1.0 >= 0.10 else None,
        },
    }


def load_json(text: str, path: str | None) -> dict[str, Any]:
    if path:
        raw = Path(path).read_text(encoding="utf-8")
    elif text:
        raw = text
    else:
        raw = sys.stdin.read()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise InputError(f"JSON 解析失败: {e}")
    if not isinstance(obj, dict):
        raise InputError("输入必须是 JSON 对象")
    return obj


def run_batch(path: str, out: str | None, weight: float) -> None:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            try:
                card = entry_card(row, weight=weight)
                rows.append({"行号": i, **card})
            except InputError as e:
                rows.append({"行号": i, "可买": False, "错误": str(e)})
    if out:
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["行号", "可买", "建议仓位", "排序分",
                                                   "拒绝原因", "错误"])
            writer.writeheader()
            for r in rows:
                writer.writerow({k: json.dumps(r[k], ensure_ascii=False)
                                 if isinstance(r.get(k), (list, dict)) else r.get(k, "")
                                 for k in writer.fieldnames})
        print(f"已写 {out}（{len(rows)} 行）")
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


def demo() -> None:
    good = {"ret20": 0.05, "dist_ma20": 0.02, "rsi14": 55,
            "n_golden_60": 4, "s_vol": 0.40, "amt5_60": -0.10,
            "dist_ma60": 0.03, "hist": 0.12, "close": 10.0}
    bad = {"ret20": 0.28, "dist_ma20": 0.12, "rsi14": 78,
           "n_golden_60": 1, "s_vol": 0.85, "amt5_60": 0.45,
           "dist_ma60": -0.15, "hist": 0.05, "close": 10.0}
    print("合格示例：")
    print(json.dumps(entry_card(good), ensure_ascii=False, indent=2))
    print("\n不合格示例：")
    print(json.dumps(entry_card(bad), ensure_ascii=False, indent=2))
    print("\n离场示例：")
    print(json.dumps(exit_card({"entry_price": 10, "peak_price": 11.3,
                                "close": 10.2, "low": 10.1,
                                "death_cross": 0}), ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="单槽 F2 决策卡（只接受数值特征）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--entry", action="store_true", help="入场判断")
    g.add_argument("--exit", action="store_true", help="离场判断")
    g.add_argument("--demo", action="store_true", help="打印示例")
    ap.add_argument("--input", type=str, default="", help="JSON 字符串")
    ap.add_argument("--file", type=str, default="", help="JSON 文件")
    ap.add_argument("--batch", type=str, default="", help="批量 CSV（入场）")
    ap.add_argument("--out", type=str, default="", help="批量结果 CSV")
    ap.add_argument("--weight", type=float, default=0.30, help="单槽建议权重，默认 0.30")
    args = ap.parse_args()

    try:
        if args.demo:
            demo()
        elif args.batch:
            run_batch(args.batch, args.out or None, args.weight)
        else:
            d = load_json(args.input, args.file or None)
            card = exit_card(d) if args.exit else entry_card(d, weight=args.weight)
            print(json.dumps(card, ensure_ascii=False, indent=2))
    except InputError as e:
        print(json.dumps({"错误": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(2)


if __name__ == "__main__":
    main()
