"""交易制度参数与适配器自检。

运行：
    cd trading_趋势结构/02_代码
    python3 交易制度自检.py

检查三类：
A. 静态：JSON 结构、日期区间连续性、费率范围、板块前缀唯一性、规则单调性。
B. 单元：用真实历史案例验证涨跌停、44%首日帽、T+1、停牌、费用、红利税、整股。
C. 数据：源库覆盖/冻结截断、ST分板块证据、年度涨跌停分布证据、成本敏感性证据。

只读源库；不写源库。结果写到 05_审计/自检报告.json 与 05_审计/自检报告.md。
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = ROOT / "05_审计"
PARAMS = HERE / "交易制度参数.json"
sys.path.insert(0, str(HERE))

from 交易制度适配器 import RuleBook  # noqa: E402

FREEZE = date(2026, 10, 26)
CHECKS = []


def check(cid: str, name: str, ok: bool, detail: str = ""):
    CHECKS.append({"id": cid, "name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} {name}" + (f" :: {detail}" if detail else ""))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_csv(name: str):
    path = AUDIT / name
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def validate_schedule(segments, start_floor: date, label: str):
    """区间必须按 from 排序、无重叠、无空洞（从 max(from, start_floor) 到冻结日）。"""
    rows = [s for s in segments if isinstance(s, dict) and s.get("from")]
    rows.sort(key=lambda s: s["from"])
    problems = []
    prev_end = None
    for seg in rows:
        begin = date.fromisoformat(seg["from"])
        end = date.fromisoformat(seg["to"]) if seg.get("to") else FREEZE
        if end < begin:
            problems.append(f"{label}: end<start {seg}")
        if begin > FREEZE:
            continue
        if prev_end is not None and begin <= prev_end:
            problems.append(f"{label}: overlap {prev_end} -> {begin}")
        prev_end = end
    if rows:
        first = date.fromisoformat(rows[0]["from"])
        if first > start_floor:
            problems.append(f"{label}: starts {first}, later than {start_floor}")
        last = date.fromisoformat(rows[-1]["to"]) if rows[-1].get("to") else FREEZE
        if last < FREEZE:
            problems.append(f"{label}: ends {last}, before {FREEZE}")
    return problems


def main():
    rb = RuleBook(str(PARAMS))
    raw = rb.raw

    # ---------- A. 静态 ----------
    required = ["meta", "board_definitions", "price_limits", "ipo_rules", "delisting_rules",
                "execution_rules", "fees", "cost_scenarios", "research_defaults",
                "data_sources", "known_approximations", "evidence", "adapter_contract"]
    check("A01", "JSON 必填顶层键", all(k in raw for k in required),
          f"missing={[k for k in required if k not in raw]}")
    check("A02", "冻结日=2026-10-26", raw["meta"]["as_of_freeze"] == FREEZE.isoformat(), raw["meta"]["as_of_freeze"])

    prefixes = []
    for board, meta in raw["board_definitions"].items():
        if board == "EXCLUDED":
            continue
        prefixes.extend(meta.get("code_prefixes", []))
    check("A03", "板块前缀无重复", len(prefixes) == len(set(prefixes)),
          f"{len(prefixes)} prefixes, {len(set(prefixes))} unique")

    problems = []
    windows = {"SSE_main": date(2005, 1, 1), "SZSE_main": date(2005, 1, 1), "SME": date(2005, 1, 1),
               "ChiNext": date(2009, 10, 30), "STAR": date(2019, 7, 22), "BSE": date(2020, 7, 27)}
    delist_floors = {"SSE_main": date(2012, 1, 1), "SZSE_main": date(2012, 1, 1), "SME": date(2012, 1, 1),
                     "ChiNext": date(2012, 1, 1), "STAR": date(2019, 7, 22), "BSE": date(2020, 7, 27)}
    for board, floor in windows.items():
        problems += validate_schedule(raw["price_limits"][board], floor, f"price_limits.{board}")
        problems += validate_schedule(raw["ipo_rules"][board], floor, f"ipo_rules.{board}")
        problems += validate_schedule(raw["delisting_rules"][board], delist_floors[board], f"delisting_rules.{board}")
    if not raw["delisting_rules"].get("pre_2012", {}).get("no_delisting_period"):
        problems.append("delisting_rules.pre_2012 missing or not marked no_delisting_period")
    problems += validate_schedule(raw["fees"]["stamp_duty"], date(2001, 11, 16), "stamp_duty")
    problems += validate_schedule(raw["fees"]["transfer_fee"]["schedule"], date(2005, 1, 1), "transfer_fee")
    for exchange, segs in raw["fees"]["exchange_handling_fee"]["schedule_by_exchange"].items():
        floor = date(2021, 11, 15) if exchange == "BSE" else date(2005, 1, 1)
        problems += validate_schedule(segs, floor, f"handling_fee.{exchange}")
    problems += validate_schedule(raw["fees"]["dividend_tax"], date(2005, 1, 1), "dividend_tax")
    for name, segs in raw["fees"]["commission"]["scenarios"].items():
        floor = date(2020, 1, 1) if name == "modern_low" else date(2005, 1, 1)
        problems += validate_schedule(segs, floor, f"commission.{name}")
    check("A04", "全部制度/费率区间连续无重叠", not problems, "; ".join(problems[:6]))

    bad_rates = []
    for board, segs in raw["price_limits"].items():
        if board in ("rounding", "special_no_limit"):
            continue
        for seg in segs:
            for key in ("normal", "st"):
                v = seg.get(key)
                if v is not None and not (0 < v <= 0.5):
                    bad_rates.append((board, key, v))
    for seg in raw["fees"]["stamp_duty"]:
        for key in ("buy", "sell"):
            if not (0 <= seg.get(key, 0) <= 0.01):
                bad_rates.append(("stamp", key, seg.get(key)))
    check("A05", "费率/涨跌幅在合理范围", not bad_rates, str(bad_rates[:5]))

    check("A06", "佣金最低5元", close(raw["fees"]["commission"]["min_per_order_cny"], 5.0))
    check("A07", "2008-09-19后买入印花税为0",
          all(seg["buy"] == 0 for seg in raw["fees"]["stamp_duty"] if seg["from"] >= "2008-09-19"))
    check("A08", "最小买入单位均为正",
          all(int(v) > 0 for v in raw["execution_rules"]["min_buy_unit"].values()))
    check("A09", "科创板最小200股、北交所100股",
          raw["execution_rules"]["min_buy_unit"]["STAR"] == 200 and raw["execution_rules"]["min_buy_unit"]["BSE"] == 100)
    check("A10", "T+1 已声明", raw["execution_rules"]["t_plus"]["same_day_buy_sell_allowed"] is False)
    tw = raw["research_defaults"]["time_windows"]
    seq = [tw["warmup"], tw["development"], tw["validation"], tw["holdout"], tw["extension"]]
    tw_ok = (
        seq[0][0] == "2005-01-01"
        and seq[-1][1] == "2026-10-26"
        and all(date.fromisoformat(seq[i][1]).toordinal() + 1 == date.fromisoformat(seq[i + 1][0]).toordinal()
                for i in range(len(seq) - 1))
        and tw.get("not_clean_oos") is True
    )
    check("A11", "时间纪律窗口连续且声明非纯净OOS", bool(tw_ok), str(seq))

    # ---------- B. 单元 ----------
    board_cases = {
        "sh.600000": "SSE_main", "sz.000001": "SZSE_main", "sz.002594": "SME",
        "sz.300750": "ChiNext", "sz.301001": "ChiNext", "sh.688001": "STAR",
        "sh.689009": "STAR", "920242.BJ": "BSE", "bj.830799": "UNKNOWN",
    }
    check("B01", "代码->板块映射", all(rb.board_of(k) == v for k, v in board_cases.items()),
          str({k: rb.board_of(k) for k in board_cases}))

    check("B02", "主板ST 2026-07-03=5%", close(rb.price_limit_pct("sh.600000", "2026-07-03", is_st=True), 0.05))
    check("B03", "主板ST 2026-07-06=10%", close(rb.price_limit_pct("sh.600000", "2026-07-06", is_st=True), 0.10))
    check("B04", "创业板 2020-08-21=10%", close(rb.price_limit_pct("sz.300750", "2020-08-21"), 0.10))
    check("B05", "创业板 2020-08-24=20%", close(rb.price_limit_pct("sz.300750", "2020-08-24"), 0.20))
    check("B06", "科创板 20%", close(rb.price_limit_pct("sh.688001", "2024-01-02"), 0.20))
    check("B07", "北交所 30%", close(rb.price_limit_pct("920242.BJ", "2024-01-02"), 0.30))
    check("B08", "科创板IPO前5日无涨跌幅",
          rb.price_limit_pct("sh.688001", "2019-07-22", ipo_session=1) is None
          and rb.price_limit_pct("sh.688001", "2019-07-29", ipo_session=6) == 0.20)
    check("B09", "创业板IPO 2014-2020首日44%帽",
          rb.ipo_first_day_cap("sz.300750", "2018-06-11")["cap_up"] == 0.44)
    check("B10", "涨停价四舍五入 25.14*1.44=36.20", rb.limit_prices(25.14, 0.44)[1] == 36.20)
    check("B11", "跌停价四舍五入 44.00*0.90=39.60", rb.limit_prices(44.00, 0.10)[0] == 39.60)

    bar_300750 = {"open": 30.17, "high": 36.20, "low": 30.17, "close": 36.20}
    check("B12", "300750首日未触44%帽可买",
          rb.can_buy("sz.300750", "2018-06-11", bar_300750, 25.14, ipo_session=1, issue_price=25.14)[0] is True)
    bar_cap = {"open": 36.20, "high": 36.20, "low": 36.20, "close": 36.20}
    check("B13", "开盘触44%帽不可买",
          rb.can_buy("sz.300750", "2018-06-11", bar_cap, 25.14, ipo_session=1, issue_price=25.14)[1] == "ipo_first_day_cap_up")

    bar_300001 = {"open": 39.60, "high": 41.87, "low": 39.60, "close": 39.60}
    check("B14", "300001次日跌停可卖但不可按跌停买? 买入不受跌停限制",
          rb.can_buy("sz.300001", "2009-11-02", bar_300001, 44.00)[0] is True)
    check("B15", "跌停开盘不可卖",
          rb.can_sell("sz.300001", "2009-11-02", {"open": 39.60, "high": 39.60, "low": 39.60, "close": 39.60}, 44.00)[1] == "limit_down_open")
    check("B16", "T+1 当日买入不可卖",
          rb.can_sell("sh.600000", "2024-01-02", {"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0}, 10.0, held_days=0)[1] == "t_plus_one")
    check("B17", "停牌不可买",
          rb.can_buy("sh.600000", "2024-01-02", {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0}, 10.0, suspended=True)[1] == "suspended")
    check("B18", "一字涨停不可买",
          rb.can_buy("sh.600000", "2024-01-02", {"open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0}, 10.0)[1] == "limit_up_open")

    fee_cases = [
        ("buy", 100000, 10000, "2024-01-02", 31.00),
        ("sell", 100000, 10000, "2024-01-02", 81.00),
        ("buy", 2000, 200, "2024-01-02", 5.02),
        ("buy", 100000, 10000, "2007-06-01", 455.00),
        ("sell", 100000, 10000, "2023-08-28", 81.00),
    ]
    fee_ok = True
    fee_detail = []
    for side, amount, shares, day, expected in fee_cases:
        got = rb.total_fees(day, "SSE_main", side, amount, shares)["total"]
        fee_ok &= close(got, expected, 0.01)
        fee_detail.append(f"{day} {side} {amount}: {got} vs {expected}")
    check("B19", "费用案例", fee_ok, "; ".join(fee_detail))

    check("B20", "红利税 20/10/0",
          [rb.dividend_tax_rate("2020-01-02", d) for d in (20, 200, 400)] == [0.2, 0.1, 0.0])
    check("B21", "整股：主板150->100，科创250->250，科创99->0",
          rb.round_lot(150, "sh.600000") == 100 and rb.round_lot(250, "sh.688001") == 250 and rb.round_lot(99, "sh.688001") == 0)
    check("B22", "IPO首日36%跌幅帽",
          rb.can_sell("sz.300750", "2018-06-11", {"open": 16.00, "high": 16.00, "low": 16.00, "close": 16.00},
                      25.14, held_days=1, ipo_session=1, issue_price=25.14)[1] == "ipo_first_day_cap_down"
          and rb.can_sell("sz.300750", "2018-06-11", {"open": 16.10, "high": 16.10, "low": 16.10, "close": 16.10},
                          25.14, held_days=1, ipo_session=1, issue_price=25.14)[0] is True)
    check("B23", "2012前无退市整理期",
          rb.can_sell("sh.600000", "2011-01-04", {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                      10.0, delisting=True)[1] == "no_delisting_period"
          and rb.can_buy("sh.600000", "2011-01-04", {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0},
                         10.0, delisting=True)[1] == "delisting_period")

    # ---------- C. 数据 ----------
    profile = json.load(open(AUDIT / "证据_数据画像.json", encoding="utf-8"))
    duck = profile["duckdb"]
    bao = profile["baostock"]
    check("C01", "DuckDB 覆盖至冻结日后（需截断）", duck["max_date"] > FREEZE.isoformat(),
          f"{duck['min_date']}..{duck['max_date']}, rows={duck['rows']}, after_freeze={duck['rows_after_freeze']}")
    check("C02", "DuckDB 无退市股（最后交易日<2026-10-01仅5只）", duck["symbols_last_before_2026_10_01"] <= 5,
          str(duck["symbols_last_before_2026_10_01"]))
    check("C03", "DuckDB 北交所 prev_close 全空", duck["bse_prev_close_null"] == duck["bse_rows"],
          f"{duck['bse_prev_close_null']}/{duck['bse_rows']}")
    check("C04", "Baostock 无北交所", bao["bse_codes"] == 0, str(bao["bse_codes"]))
    check("C05", "Baostock 含退市股", bao["delisted_type1"] >= 300, str(bao["delisted_type1"]))
    check("C06", "Baostock 覆盖2005前", bao["stock_min_date"] <= "2005-01-01", bao["stock_min_date"])
    check("C07", "Baostock 冻结日后仍有数据（需截断）", bao["stock_max_date"] > FREEZE.isoformat(),
          f"{bao['stock_min_date']}..{bao['stock_max_date']}")

    # ST 分板块证据
    st_rows = load_csv("证据_ST分板块月度.csv")
    main_st = [r for r in st_rows if r["board"] == "MAIN"]
    pre = [abs(float(r["mx"])) for r in main_st if r["ym"] <= "2026-06"]
    post = [abs(float(r["mx"])) for r in main_st if r["ym"] >= "2026-07"]
    cn_st = [abs(float(r["mx"])) for r in st_rows if r["board"] == "CHINEXT"]
    star_st = [abs(float(r["mx"])) for r in st_rows if r["board"] == "STAR"]
    check("C08", "主板ST 2026-06前最大日收益<6%", max(pre) < 0.06, f"max={max(pre):.4f}")
    check("C09", "主板ST 2026-07后最大日收益>9%", max(post) > 0.09, f"max={max(post):.4f}")
    check("C10", "创业板ST最大日收益≈20%（<25%）", max(cn_st) < 0.25, f"max={max(cn_st):.4f}")
    check("C11", "科创板ST最大日收益≈20%（<25%）", max(star_st) < 0.25, f"max={max(star_st):.4f}")

    # 年度板块分布证据
    yr_rows = load_csv("证据_年度板块收益分布.csv")
    def get(yr, board):
        for r in yr_rows:
            if r["yr"] == yr and r["board"] == board:
                return r
        return None
    cn2013 = get("2013", "ChiNext")
    cn2024 = get("2024", "ChiNext")
    main2024 = get("2024", "SSE_main")
    cn2013_ok = cn2013 and abs(float(cn2013["mx"])) <= 0.11
    cn2024_ok = cn2024 and int(cn2024["r19"]) > 0
    main2024_ok = main2024 and int(main2024["r19"]) < 100
    check("C12", "创业板2013仍为10%（max<=11%）", bool(cn2013_ok), cn2013["mx"] if cn2013 else "missing")
    check("C13", "创业板2024出现20%档（r19>0）", bool(cn2024_ok), cn2024["r19"] if cn2024 else "missing")
    check("C14", "主板2024 19%+极少（<100）", bool(main2024_ok), main2024["r19"] if main2024 else "missing")

    # 成本敏感性证据
    cost_rows = load_csv("证据_成本敏感性.csv")
    def cost(account, positions, scenario):
        for r in cost_rows:
            if r["account"] == str(account) and r["positions"] == str(positions) and r["scenario"] == scenario:
                return float(r["roundtrip_cost"])
        return None
    c_2w10 = cost(20000, 10, "2023-08+ 保守(万3)")
    c_100w10 = cost(1000000, 10, "2023-08+ 保守(万3)")
    check("C15", "2万/10仓往返成本>0.7%", c_2w10 is not None and c_2w10 > 0.007, f"{c_2w10}")
    check("C16", "100万/10仓往返成本≈0.31%", c_100w10 is not None and 0.0025 < c_100w10 < 0.004, f"{c_100w10}")

    # 现场只读查询：确认源库仍可读、冻结条件有效
    try:
        import duckdb
        con = duckdb.connect(str(profile["duckdb"]["path"]), read_only=True)
        live_rows, live_max = con.execute("select count(*), max(date) from raw_kline_daily").fetchone()
        live_after = con.execute("select count(*) from raw_kline_daily where date > DATE '2026-10-26'").fetchone()[0]
        check("C17", "DuckDB 现场只读查询", live_rows >= duck["rows"] and str(live_max) >= duck["max_date"],
              f"rows={live_rows}, max={live_max}, after_freeze={live_after}")
        con.close()
    except Exception as exc:  # pragma: no cover
        check("C17", "DuckDB 现场只读查询", False, repr(exc))

    try:
        b = sqlite3.connect(f"file:{profile['baostock']['path']}?mode=ro", uri=True)
        live_max = b.execute("select max(date) from stock_kline_d").fetchone()[0]
        live_n = b.execute("select count(*) from stock_basic where type=1").fetchone()[0]
        check("C18", "Baostock 现场只读查询", live_n >= 5564 and live_max >= FREEZE.isoformat(),
              f"type1={live_n}, max={live_max}")
        b.close()
    except Exception as exc:  # pragma: no cover
        check("C18", "Baostock 现场只读查询", False, repr(exc))

    # 北交所 30% 档证据
    bse_rows = load_csv("证据_北交所年度收益分布.csv")
    bse_max = max(float(r["mx"]) for r in bse_rows)
    bse_2024 = [r for r in bse_rows if r["yr"] == "2024"]
    check("C19", "北交所最大日收益≈30%（<=31%）且2024有29%档",
          bse_max <= 0.31 and bse_2024 and float(bse_2024[0]["r29"]) > 0,
          f"max={bse_max}, 2024_r29={bse_2024[0]['r29'] if bse_2024 else 'missing'}")

    # 首日案例
    first_rows = load_csv("证据_首日案例.csv")
    first = {(r["code"], r["date"]): float(r["ret"]) for r in first_rows}
    first_ok = (
        abs(first.get(("sz.300750", "2018-06-11"), -1) - 0.4399) < 1e-4
        and abs(first.get(("sz.300001", "2009-10-30"), -1) - 0.8487) < 1e-4
        and abs(first.get(("sh.688001", "2019-07-22"), -1) - 1.2877) < 1e-4
        and abs(first.get(("sh.601398", "2006-10-27"), -1) - 0.0513) < 1e-4
    )
    check("C20", "首日案例收益与规则一致", first_ok, str(first))

    # ---------- 汇总 ----------
    failures = [c for c in CHECKS if c["status"] == "FAIL"]
    result = {
        "run_at": date.today().isoformat(),
        "params_file": str(PARAMS),
        "freeze": FREEZE.isoformat(),
        "total": len(CHECKS),
        "passed": len(CHECKS) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "checks": CHECKS,
    }
    (AUDIT / "自检报告.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 交易制度参数自检报告",
        "",
        f"- 运行日期：{result['run_at']}",
        f"- 参数文件：`{PARAMS}`",
        f"- 冻结日：{result['freeze']}",
        f"- 结果：**{result['passed']}/{result['total']} 通过，{result['failed']} 失败**",
        "",
        "| 编号 | 检查 | 结果 | 细节 |",
        "|---|---|---|---|",
    ]
    for c in CHECKS:
        detail = c["detail"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {c['id']} | {c['name']} | {c['status']} | {detail[:160]} |")
    (AUDIT / "自检报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{result['passed']}/{result['total']} passed, {result['failed']} failed")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
