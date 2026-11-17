"""二万元主板账户研究。开发/验证先行，未经冻结不准打开留出绩效。

公司行动模型初跑为待核验探索：本地已知分红送转、税前股利按20%预留税、
默认现金除息日到账/沪市红股下一交易日可卖/深市当日。逐笔真实日期可覆写，
未知历史ST和缺失83只主板必须保留失败门槛，不能称为完整严格验证。
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time

import duckdb
import numpy as np
import pandas as pd

from 策略注册 import registry
from 现金账本 import run_broker
from 信号与订单 import close_instruction
from 统计检验 import account_metrics, annual_monthly
from 交易数据 import TradingData, DATA, ROOT

DB = "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb"
PERIODS = {"开发": ("2018-01-01", "2020-12-31"), "验证": ("2021-01-01", "2022-12-31"),
           "历史留出": ("2023-01-01", "2025-12-31"), "二零二六": ("2026-01-01", "2026-10-22"),
           "全期": ("2018-01-01", "2026-10-22")}
FREEZE_SOURCES = ("现金账本.py", "交易数据.py", "信号与订单.py", "账户回测.py", "统计检验.py", "策略注册.py")
FREEZE_SCENARIOS = {"slippage_bps": [10, 20, 30], "unknown_limit": ["normal", "conservative"],
                    "action_lag": [0, 20]}


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_freeze(strategy, label, slippage=10, unknown_limit="normal", action_lag=0):
    """开发/验证返回None；其余读取ROOT/04_证据/留出前冻结.json，失败抛RuntimeError。

    schema_version=1；allowed_strategies为{name: asdict(strategy)}；source_sha256为
    六个FREEZE_SOURCES文件名到SHA-256的映射；input_sha256为{开发: {相对路径: SHA-256},
    验证: {相对路径: SHA-256}}，两组都须非空，逐项登记实际用于开发/验证的输入。
    输入路径相对ROOT且限于01_数据或03_结果，不允许越界。allowed_scenarios逐项登记
    slippage_bps、unknown_limit、action_lag的非空允许列表（限FREEZE_SCENARIOS的子集，
    三项作笛卡尔积）。本函数只读、不创建冻结，不以文件存在或策略名相同代替内容核验。
    """
    if label not in PERIODS:
        raise ValueError(f"未知区段: {label}")
    if label in ("开发", "验证"):
        return None
    path = ROOT / "04_证据" / "留出前冻结.json"
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("留出冻结缺失或不是有效JSON；拒绝读取订单") from exc
    required = {"schema_version", "allowed_strategies", "source_sha256", "input_sha256", "allowed_scenarios"}
    if not isinstance(frozen, dict) or not required.issubset(frozen):
        raise RuntimeError("留出冻结结构为空或缺项")
    if type(frozen["schema_version"]) is not int or frozen["schema_version"] != 1:
        raise RuntimeError("留出冻结schema_version必须为1")
    current = asdict(strategy)
    strategies = frozen["allowed_strategies"]
    if not isinstance(strategies, dict) or not strategies:
        raise RuntimeError("留出冻结允许策略不能为空")
    for name, spec in strategies.items():
        if not isinstance(spec, dict) or set(spec) != set(current) or spec.get("name") != name:
            raise RuntimeError("留出冻结策略asdict结构缺项")
    if strategies.get(strategy.name) != current:
        raise RuntimeError("策略未获留出授权或策略asdict与冻结不符")
    scenarios = frozen["allowed_scenarios"]
    if not isinstance(scenarios, dict) or set(scenarios) != set(FREEZE_SCENARIOS):
        raise RuntimeError("留出冻结压力情景缺项")
    for key, allowed in FREEZE_SCENARIOS.items():
        values = scenarios[key]
        if (not isinstance(values, list) or not values
                or any(type(v) not in (str, int, float) or v not in allowed for v in values)):
            raise RuntimeError(f"留出冻结压力情景无效: {key}")
    requested = {"slippage_bps": slippage, "unknown_limit": unknown_limit, "action_lag": action_lag}
    if any(type(v) not in (str, int, float) or v not in scenarios[k] for k, v in requested.items()):
        raise RuntimeError("当前压力情景未在留出冻结中显式允许")

    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    sources, inputs = frozen["source_sha256"], frozen["input_sha256"]
    if (not isinstance(sources, dict) or set(sources) != set(FREEZE_SOURCES)
            or not all(valid_hash(value) for value in sources.values())):
        raise RuntimeError("留出冻结源码SHA-256缺项或格式无效")
    if not isinstance(inputs, dict) or set(inputs) != {"开发", "验证"}:
        raise RuntimeError("留出冻结开发/验证输入清单缺项")
    input_paths = []
    for period, files in inputs.items():
        if not isinstance(files, dict) or not files:
            raise RuntimeError(f"留出冻结{period}输入清单不能为空")
        for relative, expected in files.items():
            relative_path = Path(relative)
            resolved = (ROOT / relative_path).resolve()
            if (not valid_hash(expected) or relative_path.is_absolute() or ".." in relative_path.parts
                    or not relative_path.parts or relative_path.parts[0] not in ("01_数据", "03_结果")
                    or not resolved.is_relative_to(ROOT.resolve())):
                raise RuntimeError(f"留出冻结{period}输入路径或SHA-256无效: {relative}")
            input_paths.append((resolved, expected))
    # 先完整检查结构、策略、压力情景及所有源码，再读取输入文件的字节；此处不解析订单。
    for file, expected in [(ROOT / "02_代码" / name, sources[name]) for name in FREEZE_SOURCES] + input_paths:
        try:
            actual = _file_sha256(file)
        except OSError as exc:
            raise RuntimeError(f"留出冻结文件不可读: {file}") from exc
        if actual != expected:
            raise RuntimeError(f"留出冻结SHA-256不符: {file}")
    return frozen


def actions_for(data, start, end, lag=0):
    con = duckdb.connect(DB, read_only=True)
    a = con.execute("""select thscode as code, ex_date, dividend_per_share, per_share_bonus, rights_ratio,rights_price
                       from raw_adjustment_events where ex_date between ? and ? order by ex_date,thscode""", [start, end]).df()
    con.close()
    a["ex_date"] = pd.to_datetime(a.ex_date)
    if not a.ex_date.isin(data.full_calendar).all():
        raise ValueError("存在非交易日公司行动，须先核验")
    a["share_multiplier"] = 1 + a.per_share_bonus
    a["cash_dividend_gross_per_old_share"] = a.dividend_per_share
    a["cash_dividend_per_old_share"] = a.dividend_per_share * .8
    full = data.full_calendar
    def after_day(d, n):
        i = full.searchsorted(d)
        if i + n < len(full):
            return full[i + n]
        return d + pd.Timedelta(days=n * 2 + 10)
    a["pay_date"] = [after_day(d, lag) for d in a.ex_date]
    a["stock_pay_date"] = [after_day(r.ex_date, max(lag, 1 if r.code.endswith(".SH") else 0)) for r in a.itertuples()]
    a["timing_verified"] = False
    override = DATA / "公司行动日期核验.csv"
    if override.exists():
        actual = pd.read_csv(override)
        actual["ex_date"] = pd.to_datetime(actual.ex_date)
        actual = actual.set_index(["code", "ex_date"])
        for i, r in a.iterrows():
            key = (r.code, r.ex_date)
            if key in actual.index:
                v = actual.loc[key]
                a.at[i, "pay_date"] = pd.Timestamp(v.pay_date)
                a.at[i, "stock_pay_date"] = pd.Timestamp(v.stock_pay_date)
                a.at[i, "timing_verified"] = True
    # 配股政策事前固定为不认购，不付配股资金也不增加配股股份；其损失由原价体现。
    return a


def roundtrips(result, orders, calendar, *, require_entry_audit=False):
    """净盈亏保留为已实现口径；总净盈亏含期末MTM，不预扣未发生的卖费。

    兼容旧fixture时缺失开仓审计值保持NaN，绝不用信号收盘反推实际成交风险。
    正式simulate要求成交表携带真实开仓审计字段。
    """
    columns = ["交易键", "代码", "买入日期", "卖出日期", "已闭合", "含费投入", "买入原价", "初始股数",
               "净盈亏", "已实现净盈亏", "未实现净盈亏", "总净盈亏", "净收益率", "含浮盈净收益率", "持有交易日",
               "信号日期", "初始止损原价", "初始风险金额", "开仓权益", "开仓权重", "开仓审计完整",
               "实际初始止损距离", "计划止损距离", "结构阶段键", "买入状态已知", "股利税假设"]
    fills = result["成交"]
    buys = fills[fills["方向"].eq("买入")].copy()
    rows = []
    calendar = pd.DatetimeIndex(calendar)
    open_positions = {(r["代码"], r["买入日期"]): r for r in result["期末持仓"].to_dict("records")}
    for code, g in buys.groupby("代码"):
        g = g.sort_values("日期")
        dates = g["日期"].tolist()
        sell = fills[fills["代码"].eq(code) & fills["方向"].eq("卖出")]
        divs = result["公司行动"]
        for k, buy in enumerate(g.to_dict("records")):
            begin = buy["日期"]
            finish = dates[k + 1] if k + 1 < len(dates) else pd.Timestamp.max
            # 同日旧仓开盘平仓再开新仓：卖单和除权日应收归旧仓；到账日不改变归属。
            sells = sell[(sell["日期"] > begin) & (sell["日期"] <= finish)]
            position = open_positions.get((code, begin))
            closed = position is None
            end = sells["日期"].max() if closed and len(sells) else pd.NaT
            invested = -float(buy["现金变动"])
            dividend = divs[(divs["代码"] == code) & (divs["日期"] > begin) & (divs["日期"] <= finish)]["新增应收"].sum() if len(divs) else 0.0
            pnl = float(sells["已实现盈亏"].sum() + dividend)
            unrealized = float(position["未实现盈亏"]) if position is not None else 0.0
            i = calendar.searchsorted(begin)
            previous = calendar[i - 1] if i > 0 else pd.NaT
            recorded_signal = buy.get("信号日期", pd.NaT)
            signal_date = recorded_signal if pd.notna(recorded_signal) else previous
            s = orders[orders.code.eq(code) & orders.signal_date.eq(signal_date)]
            signal = s.iloc[0] if len(s) == 1 else None
            entry = {key: buy.get(key, np.nan) for key in ("初始止损原价", "初始风险金额", "开仓权益", "开仓权重")}
            complete = (pd.notna(recorded_signal) and recorded_signal == previous and signal is not None
                        and all(pd.notna(v) and np.isfinite(v) and v > 0 for v in entry.values()))
            price = float(buy["成交原价"])
            actual_distance = (price - entry["初始止损原价"]) / price
            complete = bool(complete and 0 < actual_distance < 1
                            and np.isclose(entry["初始风险金额"], buy["股数"] * price * actual_distance, rtol=1e-10, atol=1e-7)
                            and np.isclose(entry["开仓权重"], buy["股数"] * price / entry["开仓权益"], rtol=1e-10, atol=1e-10))
            if require_entry_audit and not complete:
                raise AssertionError(f"买入成交缺少或不符合真实开仓审计: {code}/{begin}")
            rows.append({"交易键": code + "_" + begin.strftime("%Y%m%d"), "代码": code,
                         "买入日期": begin, "卖出日期": end, "已闭合": closed, "含费投入": invested,
                         "买入原价": price, "初始股数": buy["股数"], "净盈亏": pnl,
                         "已实现净盈亏": pnl, "未实现净盈亏": unrealized, "总净盈亏": pnl + unrealized,
                         "净收益率": pnl / invested if closed else np.nan,
                         "含浮盈净收益率": (pnl + unrealized) / invested,
                         "持有交易日": calendar.searchsorted(end) - i if pd.notna(end) else np.nan,
                         "信号日期": signal_date, **entry, "开仓审计完整": complete,
                         "实际初始止损距离": actual_distance,
                         "计划止损距离": signal.get("stop_distance", np.nan) if signal is not None else np.nan,
                         "结构阶段键": signal.get("episode_key", "未知") if signal is not None else "未知",
                         "买入状态已知": bool(signal.get("st_status_known", False)) if signal is not None else False,
                         "股利税假设": "全部预留百分之二十，未按真实持有时长退回"})
    return pd.DataFrame(rows, columns=columns)


def assert_accounting(result, net, initial_cash=20000.0):
    """独立重建每日现金/应收/股份及期末逐笔损益；公司行动不作为虚构成交。"""
    account = result["每日账户"].set_index("日期")
    fills, dividends = result["成交"], result["分红应收"]
    actions, holdings = result["公司行动"], result["每日持仓"]

    def equal(actual, expected, message):
        if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0, atol=1e-6):
            raise AssertionError(message)

    def daily(frame, date, column):
        return frame.groupby(date)[column].sum().reindex(account.index, fill_value=0).astype(float)

    cash_moves = daily(fills, "日期", "现金变动").cumsum()
    paid = daily(dividends, "实际入账日期", "应收金额").cumsum()
    declared = daily(actions, "日期", "新增应收").cumsum()
    equal(account["可用现金"], initial_cash + cash_moves + paid, "现金恒等式不成立")
    equal(account["分红应收"], declared - paid, "分红应收恒等式不成立")
    equal(account["持仓市值"], daily(holdings, "日期", "持仓市值"), "每日持仓市值不符")
    equal(account["未实现盈亏"], daily(holdings, "日期", "未实现盈亏"), "每日未实现盈亏不符")
    equal(account["累计已实现盈亏"], daily(fills, "日期", "已实现盈亏").cumsum() + declared, "已实现盈亏不符")
    equal(account["账户权益"], account["可用现金"] + account["分红应收"] + account["持仓市值"], "账户权益现金恒等式不成立")
    equal(account["账户权益"] - initial_cash, account["累计已实现盈亏"] + account["未实现盈亏"], "账户损益恒等式不成立")

    keys = ["日期", "代码"]
    share_moves = fills.assign(股份变动=fills["股数"] * fills["方向"].map({"买入": 1, "卖出": -1}))
    action_moves = actions.assign(股份变动=actions["新股数"] - actions["旧股数"])
    changes = pd.concat([share_moves[keys + ["股份变动"]], action_moves[keys + ["股份变动"]]])
    shares = pd.concat([changes.groupby(keys)["股份变动"].sum(), holdings.groupby(keys)["股数"].sum()], axis=1).astype(float).fillna(0).sort_index()
    equal(shares["股数"], shares["股份变动"].groupby(level="代码").cumsum(), "股份恒等式不成立")
    equal(holdings["股数"], holdings["可卖股数"] + holdings["锁定股数"], "股份解禁恒等式不成立")
    final = result["期末持仓"]
    last = holdings[holdings["日期"].eq(account.index[-1])]
    terminal = pd.concat([final.groupby(["代码", "买入日期"])["股数"].sum().rename("期末"),
                          last.groupby(["代码", "买入日期"])["股数"].sum().rename("每日")], axis=1).fillna(0)
    equal(terminal["期末"], terminal["每日"], "期末股份与每日持仓不符")
    equal(net["已实现净盈亏"].sum(), account["累计已实现盈亏"].iloc[-1], "逐笔已实现损益不符")
    equal(net["未实现净盈亏"].sum(), account["未实现盈亏"].iloc[-1], "逐笔未实现损益不符")
    equal(net["总净盈亏"].sum(), account["账户权益"].iloc[-1] - initial_cash, "逐笔总净盈亏与末权益减本金不符")


def simulate(strategy, label, data, output, slippage=10, action_lag=0):
    validate_freeze(strategy, label, slippage, data.unknown_limit, action_lag)
    start, end = PERIODS[label]
    index = data.dates
    si = max(index.searchsorted(start) - 1, 0)
    days = index[si:index.searchsorted(end, side="right")]
    order_file = DATA / "策略信号" / f"{strategy.name}.parquet"
    orders = pd.read_parquet(order_file)
    orders = orders[(orders.signal_date >= days[0]) & (orders.signal_date <= days[-1])]
    action = actions_for(data, days[0].strftime("%Y-%m-%d"), end, action_lag)
    partial = set()
    def after_close(date, positions):
        instructions = []
        for p in positions.to_dict("records"):
            code = p["代码"]
            row = data.features(code).loc[date]
            key = (code, p["买入日期"])
            held = data.day_index[date] - data.day_index[p["买入日期"]] + 1
            request = close_instruction(strategy, row, p, held, key in partial)
            if request is not None:
                if request.get("next_open_sell_fraction") == .5:
                    # 同一持仓只请求一次；卖不出由broker持久保留，不每日重复减半。
                    partial.add(key)
                instructions.append(request)
        return instructions
    result = run_broker(data, orders, days, strategy, corp_actions=action, after_close=after_close,
                        initial_cash=20000, slippage_bps=slippage, haircut=.5)
    account = result["每日账户"]
    account = account[account["日期"] >= start].copy()
    equity = account.set_index("日期")["账户权益"]
    met = account_metrics(equity, 20000, start)
    net = roundtrips(result, orders, data.dates, require_entry_audit=True)
    assert_accounting(result, net, 20000)
    closed = net[net["已闭合"]] if len(net) else net
    positives = closed.loc[closed["净盈亏"] > 0, "净盈亏"] if len(closed) else pd.Series(dtype=float)
    negatives = closed.loc[closed["净盈亏"] < 0, "净盈亏"] if len(closed) else pd.Series(dtype=float)
    touched = result["公司行动"]
    if len(touched):
        touched = touched.merge(action[["code", "ex_date", "timing_verified", "rights_ratio"]],
                                left_on=["代码", "日期"], right_on=["code", "ex_date"], how="left", validate="many_to_one")
    met.update({"策略": strategy.name, "区段": label, "本金": 20000,
                "买入笔数": int(result["成交"]["方向"].eq("买入").sum()), "已闭合交易": len(closed),
                "净胜率": float((closed["净盈亏"] > 0).mean()) if len(closed) else np.nan,
                "利润因子": float(positives.sum() / -negatives.sum()) if len(negatives) else np.nan,
                "平均持有交易日": float(closed["持有交易日"].mean()) if len(closed) else np.nan,
                "平均仓位": float((account["持仓市值"] / account["账户权益"]).mean()),
                "期末持仓数": len(result["期末持仓"]), "最少现金": float(account["可用现金"].min()),
                "状态未核实买入": int((~net["买入状态已知"]).sum()) if len(net) else 0,
                "行动日期未核实": int((~touched.timing_verified.fillna(False).astype(bool)).sum()) if len(touched) else 0,
                "涉及配股事件": int((touched.rights_ratio.fillna(0) > 0).sum()) if len(touched) else 0,
                "滑点基点": slippage, "未知限价": data.unknown_limit,
                "验证等级": "探索结果：覆盖与公司行动时点尚待核验，不是已通过的严谨全市场策略"})
    output.mkdir(exist_ok=True)
    result["每日账户"] = account
    for name, frame in result.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    net.to_csv(output / "逐笔闭环.csv", index=False, encoding="utf-8-sig")
    touched.to_csv(output / "持仓公司行动待核验.csv", index=False, encoding="utf-8-sig")
    yearly, monthly = annual_monthly(equity)
    yearly.to_csv(output / "年度收益.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "月度收益.csv", index=False, encoding="utf-8-sig")
    (output / "运行参数.json").write_text(json.dumps({"策略": asdict(strategy), "起日": start, "末日": end,
         "本金": 20000, "仅主板": True, "滑点": slippage, "行动延迟压力交易日": action_lag,
         "说明": "现金分红默认除息日到账，送转按市场通例默认可售日；股利预留20%税；60个缺价日后市值减半而不是虚构成交。历史状态不全。"}, ensure_ascii=False, indent=2))
    return met


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--period", choices=list(PERIODS), default="开发")
    p.add_argument("--names", nargs="+")
    p.add_argument("--slippage", type=float, default=10)
    p.add_argument("--unknown-limit", choices=["normal", "conservative"], default="normal")
    p.add_argument("--action-lag", type=int, default=0)
    a = p.parse_args()
    strategies = [s for s in registry() if a.names is None or s.name in a.names]
    if not strategies:
        raise SystemExit("未匹配策略名")
    for strategy in strategies:
        validate_freeze(strategy, a.period, a.slippage, a.unknown_limit, a.action_lag)
    data = TradingData(unknown_limit=a.unknown_limit, feature_cache=128)
    root = ROOT / "03_结果" / "账户探索"
    root.mkdir(exist_ok=True)
    stem = f"{a.period}_汇总_滑点{a.slippage:g}_{a.unknown_limit}_延迟{a.action_lag}"
    summary_path, stamp_path = root / f"{stem}.csv", root / f"{stem}_版本.json"
    stamp = {name: _file_sha256(ROOT / "02_代码" / name) for name in FREEZE_SOURCES}
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise RuntimeError("已有汇总来自不同代码版本，须先归档并完整重跑，不能混合版本")
    if summary_path.exists() and not stamp_path.exists():
        raise RuntimeError("已有汇总没有版本证据，须先归档")
    stamp_path.write_text(json.dumps(stamp, ensure_ascii=False, indent=2))
    previous = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    mets, started = [], time.time()
    for s in strategies:
        suffix = f"{a.period}_{s.name}_滑点{a.slippage:g}_{a.unknown_limit}_延迟{a.action_lag}"
        met = simulate(s, a.period, data, root / suffix, a.slippage, a.action_lag)
        mets.append(met)
        pd.DataFrame(previous + mets).drop_duplicates(["策略", "区段"], keep="last").to_csv(
            summary_path, index=False, encoding="utf-8-sig")
        print(json.dumps({k: met[k] for k in ["策略", "区段", "年化收益", "最大回撤", "买入笔数", "行动日期未核实"]}, ensure_ascii=False), flush=True)
    print(f"完成{len(mets)}组，{time.time()-started:.1f}秒", flush=True)


if __name__ == "__main__":
    main()
