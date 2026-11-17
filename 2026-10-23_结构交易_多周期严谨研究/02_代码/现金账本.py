"""独立、仅多头、T+1 的原价现金撮合；不计算信号或止损线。

run_broker(daily, orders, trading_days, strategy, *, close_instructions=None,
           corp_actions=None, after_close=None, initial_cash=20000,
           slippage_bps=10, missing_haircut_after=60, haircut=0)

输入为 DataFrame（orders/指令/公司行动也可传字典列表）。大市场daily还可传
提供get_day(date,codes)->{code:行字典}的只读对象，按当日候选/持仓懒加载，
避免把上千万行转成Python字典。适配层须校验完整日历、字段和唯一性。
* daily: date, code, open/high/low/close（未复权）, adj_factor（后复权收盘/原价），  prev_amount20（昨日已知均额）, tradable（适配层提供布尔值），minimum_buy_size。
  minimum_buy_size 只接受100或200，对应买入增量100或1。limit_up/limit_down可缺，
  缺上限拒买，缺下限拒卖/挂起；不猜ST/上市/除权规则，保守界限须由适配层明示填入。
  tradable=False或整行缺失不成交且不使用该行价格估值。不使用volume推断可交易性。
* orders: signal_date, code, stop_adj, rank, initial_risk_adj。
  信号日期必须在显式、严格递增的trading_days中，且只在紧接的交易日开盘尝试一次；
  rank越大越优先，同分代码升序。不加仓。initial_risk_adj为信号晚已知的后复权
  每股风险；结构止损用该风险与实际开盘风险较大者定仓。fixed8以实际开盘的8%
  定止损，atr2以实际开盘减信号日两倍波幅定止损，均按实际距离定仓。
  止损原价向下取整至0.01元，max_stop_distance约束实际开盘止损距离/open。权重约束股票市值，不含费用；
  现金约束含所有费用。买卖每单原价金额均不超过prev_amount20的0.1%。
* strategy: 使用策略注册.Strategy的slots/max_weight/risk_per_trade/heat_cap/
  max_stop_distance属性，不读取其余策略字段。
* close_instructions: date, code，以及可选stop_adj（只能抬升）、
  next_open_sell_fraction（0.5或1.0）。date表示当日收盘后，下一交易日才生效。
  after_close(date, positions)可替代/补充指令，返回上述字段的字典列表或DataFrame
  （无需date）。positions是仅截至当日的中文列名持仓副本，不含未来行情。
  这是数据可见性接口而非Python沙箱；调用者不能在回调闭包中偷用未来数据。
  同日同代码指令须预先合并。减半按卖出增量向下取整，零股仅最终全清时卖出。
* corp_actions: ex_date, pay_date, code, share_multiplier,
  cash_dividend_per_old_share；可选stock_pay_date、cash_dividend_gross_per_old_share。
  仅显式真实记录生效；除权前股数享有权益。新增整股自除权日计入持仓估值，
  至stock_pay_date才可卖；旧股仍遵守T+1。缺省可售日沿用ex_date并在公司行动表标注，
  不代表日期已核验完整；非交易日可售/到账在下一提供的交易日处理，无期末强平。
  分红除权日计应收/收益，pay_date才成为现金。实际应收用cash_dividend_per_old_share，
  可由适配层提供保守扣税后金额；红利税由适配层提供，broker不推断税率或后续补扣。
  gross仅用于缺价时last的理论除息转换，缺省兼容净金额，不改变实际应收。
  送转后总股数在1e-7容差内取近整数，否则向下取整；零碎份额单列为“未计价零碎权益”。
  这是保守分配假设：零碎权益不计现金/市值，不假定现金补偿，原成本保留，
  遗漏价值自然体现于未实现盈亏；成交不含小数股（不足一手的整数零股仍可最终全清）。
  无记录不表示数据完整；missing审计归适配层。不以复权因子虚构送股/分红，
  暂不扩展配股，配股仍需适配层显式数据处理，不能自动假定。

时序：公司行动及到账 -> 昨仓开盘预定退出/跳空止损 -> 昨日信号买入 ->
昨仓盘中已知止损 -> 收盘估值/明日指令。盘中回款不会回到开盘；无期末强平。
盘中只在low<=stop<=high且stop>limit_down时按止损原价模拟；否则触发后挂起。
开盘缺失不做盘中补成交。缺因子不模拟止损/新开仓，预定卖单仍可执行。
已有持仓因子缺失时也禁止新开仓，以免低估组合风险。
缺失有效收盘累计达到missing_haircut_after日时，将last估值折减haircut（非复利）；
当天有效开盘可作缺收盘回退价。开盘缺价只用昨日last和已发生的缺失计数。
公司行动日无价格时，last按(原last-每旧股税前现金或缺省净额)/股数倍数转换，不制造现金成交。
浮点金额不逐项舍入到分；滑点按成交原价金额另列成本，不改变成交价格。

返回dict[str, DataFrame]：成交、失败订单、每日账户、每日持仓、期末持仓、
分红应收、公司行动、收盘指令。全部列名中文，空表也保留列名，可直接to_csv。
持仓的止损字段/待卖股数为该日收盘指令处理后状态（下日生效）；未实现盈亏含买费，
不预扣尚未发生的卖费。每日账户累计已实现盈亏包含除权日确认的现金分红。
锁定股数含买入当日T+1及尚未到可售日的红股；可卖股数只指股份解禁，不保证市场可成交。
"""

import math

import pandas as pd


TRADE_COLUMNS = ["日期", "时点", "代码", "方向", "原因", "股数", "成交原价", "成交金额",
                 "佣金", "印花税", "过户费", "滑点成本", "总费用", "现金变动", "成交后现金", "已实现盈亏",
                 "信号日期", "初始止损原价", "初始风险金额", "开仓权益", "开仓权重"]
FAIL_COLUMNS = ["日期", "时点", "代码", "方向", "原因", "待卖股数"]
POSITION_COLUMNS = ["日期", "代码", "买入日期", "股数", "买入最小股数", "最新原价", "估值原价",
                    "后复权因子", "止损后复权", "止损原价", "持仓成本", "持仓市值", "未实现盈亏",
                    "连续缺价日数", "待卖股数", "待卖原因", "锁定股数", "可卖股数"]
ACCOUNT_COLUMNS = ["日期", "可用现金", "分红应收", "持仓市值", "账户权益", "累计已实现盈亏", "未实现盈亏", "持仓数"]
RECEIVABLE_COLUMNS = ["除权日期", "到账日期", "代码", "应收金额", "实际入账日期"]
ACTION_COLUMNS = ["日期", "代码", "旧股数", "股数倍数", "新股数", "每旧股现金", "新增应收",
                  "送转可售日期", "可售日期使用默认", "理论除息每旧股现金", "未计价零碎权益", "处理说明"]
INSTRUCTION_COLUMNS = ["日期", "生效日期", "代码", "止损后复权", "次日卖出比例"]


def _positive(value):
    return pd.notna(value) and math.isfinite(float(value)) and float(value) > 0


def _whole_shares(value):
    nearest = round(value)
    return nearest if abs(value - nearest) <= 1e-7 else math.floor(value)


def _floor_cent(value):
    return math.floor(value * 100 + 1e-7) / 100


def _stop_price(position):
    return _floor_cent(position["stop"] / position["factor"])


def _table(data, required, date_column, calendar=None):
    frame = pd.DataFrame(data).copy() if data is not None else pd.DataFrame(columns=required)
    if frame.empty:
        return pd.DataFrame(columns=list(dict.fromkeys([*required, *frame.columns])))
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"输入缺少字段: {sorted(missing)}")
    frame[date_column] = pd.to_datetime(frame[date_column])
    if frame[date_column].isna().any() or (frame[date_column] != frame[date_column].dt.normalize()).any():
        raise ValueError("日期须为无缺失的交易日期，不带日内时间")
    if calendar is not None and not frame[date_column].isin(calendar).all():
        raise ValueError(f"{date_column}不在显式交易日序列")
    if frame.code.isna().any():
        raise ValueError("代码不可缺失")
    frame["code"] = frame.code.astype(str)
    if frame.duplicated([date_column, "code"]).any():
        raise ValueError(f"{date_column}/code不可重复")
    return frame


def transaction_costs(date, amount, side, slippage_bps=10):
    """返回中文费用字典；日期变更当天开始使用新税率。"""
    if side not in ("买入", "卖出") or not _positive(amount):
        raise ValueError("方向必须为买入/卖出，金额必须为正")
    if not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("滑点基点必须非负")
    date = pd.Timestamp(date)
    commission = max(5.0, amount * 0.0003)
    stamp = amount * (0.001 if date < pd.Timestamp("2023-08-28") else 0.0005) if side == "卖出" else 0.0
    transfer = amount * (0.00002 if date < pd.Timestamp("2022-04-29") else 0.00001)
    slip = amount * slippage_bps / 10000
    return {"佣金": commission, "印花税": stamp, "过户费": transfer, "滑点成本": slip,
            "总费用": commission + stamp + transfer + slip}


def run_broker(daily, orders, trading_days, strategy, *, close_instructions=None,
               corp_actions=None, after_close=None, initial_cash=20000.0,
               slippage_bps=10, missing_haircut_after=60, haircut=0):
    calendar = pd.DatetimeIndex(trading_days)
    if (calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing
            or calendar.tz is not None or not calendar.equals(calendar.normalize())):
        raise ValueError("交易日必须无时区、无重复、严格递增且无日内时间")
    if not _positive(initial_cash) or not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("本金须为正，滑点须非负")
    if missing_haircut_after < 1 or int(missing_haircut_after) != missing_haircut_after or haircut not in (0, 0.5, 1):
        raise ValueError("缺价阈值须为正整数，折减只支持0/0.5/1")
    if strategy.slots < 1 or int(strategy.slots) != strategy.slots:
        raise ValueError("slots须为正整数")
    for field in ("max_weight", "risk_per_trade", "heat_cap", "max_stop_distance"):
        if not 0 < getattr(strategy, field) <= 1:
            raise ValueError(f"{field}须在(0,1]内")
    provider = daily if hasattr(daily, "get_day") else None
    bars = None if provider is not None else _table(
        daily, ["date", "code", "open", "high", "low", "close", "adj_factor",
                "prev_amount20", "tradable", "minimum_buy_size"], "date", calendar)
    candidates = _table(orders, ["signal_date", "code", "stop_adj", "rank", "initial_risk_adj"], "signal_date", calendar)
    instructions = _table(close_instructions, ["date", "code"], "date", calendar)
    actions = _table(corp_actions, ["ex_date", "pay_date", "code", "share_multiplier",
                                   "cash_dividend_per_old_share"], "ex_date", calendar)
    if not actions.empty:
        actions["pay_date"] = pd.to_datetime(actions.pay_date)
        if (actions.pay_date.isna().any() or (actions.pay_date < actions.ex_date).any()
                or (actions.pay_date != actions.pay_date.dt.normalize()).any()):
            raise ValueError("到账日期须有效且不早于除权日期")
        stock_dates = pd.to_datetime(actions.get("stock_pay_date", pd.Series(pd.NaT, index=actions.index)))
        actions["stock_pay_date_defaulted"] = stock_dates.isna()
        actions["stock_pay_date"] = stock_dates.fillna(actions.ex_date)
        if (actions.stock_pay_date.dt.tz is not None or (actions.stock_pay_date < actions.ex_date).any()
                or (actions.stock_pay_date != actions.stock_pay_date.dt.normalize()).any()):
            raise ValueError("送转可售日期须无时区、无日内时间且不早于除权日期")
        actions["cash_dividend_gross_per_old_share"] = actions.get(
            "cash_dividend_gross_per_old_share", actions.cash_dividend_per_old_share).fillna(actions.cash_dividend_per_old_share)
        for a in actions.to_dict("records"):
            if not _positive(a["share_multiplier"]):
                raise ValueError("股数倍数须为正")
            for field in ("cash_dividend_per_old_share", "cash_dividend_gross_per_old_share"):
                if not math.isfinite(a[field]) or a[field] < 0:
                    raise ValueError("现金分红及税前分红须非负且有限")

    def groups(frame, key):
        return {date: group.to_dict("records") for date, group in frame.groupby(key, sort=False)}

    bars_by_day = {} if provider is not None else groups(bars, "date")
    orders_by_day = groups(candidates, "signal_date")
    instructions_by_day = groups(instructions, "date")
    actions_by_day = groups(actions, "ex_date")
    cash, realized = float(initial_cash), 0.0
    positions, receivables = {}, []
    trades, failures, accounts, holdings, action_log, instruction_log = [], [], [], [], [], []

    def fail(date, phase, code, side, reason):
        failures.append(dict(zip(FAIL_COLUMNS, [date, phase, code, side, reason, positions.get(code, {}).get("pending", 0)])))

    def active(row):
        value = row.get("tradable", False)
        if pd.isna(value):
            return False
        if value not in (True, False):
            raise ValueError("tradable须由适配层提供布尔值")
        return bool(value)

    def mark(p, row=None):
        if row is not None and active(row) and _positive(row.get("open")):
            return float(row["open"])
        return p["last"] * (1 - haircut if p["missing"] >= missing_haircut_after else 1)

    def sellable(p, date):
        if p["buy_date"] >= date:
            return 0
        locked = sum(quantity for unlock, quantity in p["stock_locks"] if unlock > date)
        return _whole_shares(max(0, p["shares"] - locked))

    def snapshot(date):
        rows = []
        for code, p in sorted(positions.items()):
            value = p["shares"] * mark(p)
            available = sellable(p, date)
            rows.append([date, code, p["buy_date"], p["shares"], p["minimum"], p["last"], mark(p),
                         p["factor"], p["stop"], _stop_price(p), p["cost"], value,
                         value - p["cost"], p["missing"], p["pending"], p["reason"],
                         p["shares"] - available, available])
        return pd.DataFrame(rows, columns=POSITION_COLUMNS)

    def request(p, quantity, reason):
        if quantity >= p["pending"]:
            p["pending"], p["reason"] = quantity, reason

    def sell(date, code, row, price, phase):
        nonlocal cash, realized
        p = positions[code]
        if not active(row) or not _positive(price):
            fail(date, phase, code, "卖出", "不可交易或开盘缺价")
            return
        lower = row.get("limit_down")
        if not _positive(lower) or price <= lower:
            fail(date, phase, code, "卖出", "跌停或跌停价缺失，退出保留")
            return
        amount20 = row.get("prev_amount20")
        if not _positive(amount20):
            fail(date, phase, code, "卖出", "昨日均额缺失，退出保留")
            return
        capacity = amount20 * 0.001 / price
        step = 100 if p["minimum"] == 100 else 1
        available = sellable(p, date)
        if available <= 0:
            fail(date, phase, code, "卖出", "股份尚未解禁，退出保留")
            return
        quantity = min(p["pending"], available)
        # 只有整笔最终清仓且容量足够，才可带走不足一手的整数零股。
        if quantity < p["shares"] or capacity < quantity:
            quantity = math.floor(min(quantity, capacity) / step) * step
        if quantity <= 0 or (quantity < p["minimum"] and quantity < p["shares"]):
            fail(date, phase, code, "卖出", "参与率或减仓量不足最小卖出单位，退出保留")
            return
        amount = quantity * price
        costs = transaction_costs(date, amount, "卖出", slippage_bps)
        delta = amount - costs["总费用"]
        if cash + delta < 0:
            fail(date, phase, code, "卖出", "卖出回款不足支付费用，退出保留")
            return
        allocated_cost = p["cost"] * quantity / p["shares"]
        pnl = delta - allocated_cost
        cash += delta
        realized += pnl
        trades.append({"日期": date, "时点": phase, "代码": code, "方向": "卖出", "原因": p["reason"],
                       "股数": float(quantity), "成交原价": float(price), "成交金额": amount, **costs,
                       "现金变动": delta, "成交后现金": cash, "已实现盈亏": pnl})
        p["shares"] -= quantity
        p["cost"] -= allocated_cost
        p["pending"] = max(0.0, p["pending"] - quantity)
        if p["shares"] <= 1e-9:
            del positions[code]
        elif p["pending"] <= 1e-9:
            p["pending"], p["reason"] = 0.0, ""
        else:
            reason = "股份锁定或参与率限制部分成交，余量保留" if available < p["shares"] + quantity else "参与率限制部分成交，余量保留"
            fail(date, phase, code, "卖出", reason)

    for index, date in enumerate(calendar):
        signals = orders_by_day.get(calendar[index - 1], []) if index else []
        needed = set(positions) | {s["code"] for s in signals}
        today = (provider.get_day(date, needed) if provider is not None
                 else {r["code"]: r for r in bars_by_day.get(date, [])})
        # 公司行动只针对除权前持仓；回调/信号不会提前取得当日买入权益。
        for a in actions_by_day.get(date, []):
            p = positions.get(a["code"])
            if p is None:
                continue
            old = p["shares"]
            dividend = old * a["cash_dividend_per_old_share"]
            theoretical_shares = old * a["share_multiplier"]
            p["shares"] = _whole_shares(theoretical_shares)
            fractional = 0.0 if abs(theoretical_shares - p["shares"]) <= 1e-7 else theoretical_shares - p["shares"]
            p["pending"] = min(p["shares"], _whole_shares(p["pending"] * a["share_multiplier"]))
            p["stock_locks"] = [(unlock, quantity) for unlock, quantity in p["stock_locks"] if unlock > date]
            if a["share_multiplier"] < 1:
                # 缩股按倍数转换原锁定权益；可卖整股向下取整，避免零碎锁定权益提前出售。
                p["stock_locks"] = [(unlock, quantity * a["share_multiplier"]) for unlock, quantity in p["stock_locks"]]
            if p["shares"] > old and a["stock_pay_date"] > date:
                p["stock_locks"].append((a["stock_pay_date"], p["shares"] - old))
            p["last"] = max(0.0, (p["last"] - a["cash_dividend_gross_per_old_share"]) / a["share_multiplier"])
            if dividend:
                receivables.append(dict(zip(RECEIVABLE_COLUMNS, [date, a["pay_date"], a["code"], dividend, pd.NaT])))
                realized += dividend
            note = "可售日期由输入提供；不代表日期已核验完整。"
            if a["stock_pay_date_defaulted"]:
                note = "stock_pay_date缺省，默认沿用除权日；不代表日期已核验完整。"
            note += "保守分配假设：总股数在1e-7内取近整数，否则向下取整；未计价零碎权益不计现金/市值，不假定现金补偿；原成本保留，遗漏价值体现于未实现盈亏。"
            action_log.append(dict(zip(ACTION_COLUMNS, [date, a["code"], old, a["share_multiplier"],
                                                        p["shares"], a["cash_dividend_per_old_share"], dividend,
                                                        a["stock_pay_date"], a["stock_pay_date_defaulted"],
                                                        a["cash_dividend_gross_per_old_share"], fractional, note])))
        for dividend in receivables:
            if pd.isna(dividend["实际入账日期"]) and dividend["到账日期"] <= date:
                cash += dividend["应收金额"]
                dividend["实际入账日期"] = date

        # 开盘：全部旧仓先按已知开盘估值/当日因子处理，绝不读当日H/L/C。
        for code, p in list(positions.items()):
            row = today.get(code, {})
            factor = row.get("adj_factor")
            if _positive(factor):
                p["factor"] = float(factor)
            if (active(row) and _positive(row.get("open")) and _positive(factor)
                    and row["open"] <= _stop_price(p)):
                request(p, p["shares"], "开盘穿止损")
            if p["pending"]:
                sell(date, code, row, row.get("open"), "开盘")
            elif not _positive(factor):
                fail(date, "开盘", code, "卖出", "复权因子缺失，不模拟止损")

        # 每次开仓前重算权益/热度，包含此前开盘卖出及其费用，排除未来盘中回款。
        for order in sorted(signals, key=lambda o: (-float(o["rank"]), o["code"])):
            code = order["code"]
            row = today.get(code, {})
            price, factor = row.get("open"), row.get("adj_factor")
            reason = ""
            if code in positions:
                reason = "已有持仓，不加仓"
            elif len(positions) >= strategy.slots:
                reason = "仓位数量上限"
            elif not active(row) or not _positive(price):
                reason = "不可交易或开盘缺价"
            elif not _positive(row.get("limit_up")) or price >= row["limit_up"]:
                reason = "开盘涨停或涨停价缺失"
            elif not _positive(factor) or not _positive(order["stop_adj"]) or not _positive(order["initial_risk_adj"]):
                reason = "止损、初始风险或复权因子无效"
            elif any(not _positive(today.get(c, {}).get("adj_factor")) for c in positions):
                reason = "已有持仓因子缺失，无法审计热度"
            elif not _positive(row.get("prev_amount20")):
                reason = "昨日均额缺失"
            elif row.get("minimum_buy_size") not in (100, 200):
                reason = "买入最小股数缺失或无效"
            if reason:
                fail(date, "开盘", code, "买入", reason)
                continue
            stop_mode = getattr(strategy, "stop", "structure")
            if stop_mode == "fixed8":
                stop_raw = _floor_cent(price * .92)
            elif stop_mode == "atr2":
                stop_raw = _floor_cent(price - order["initial_risk_adj"] / factor)
            else:
                stop_raw = _floor_cent(order["stop_adj"] / factor)
            distance = price - stop_raw
            if stop_raw <= 0 or distance <= 0 or distance / price > strategy.max_stop_distance:
                fail(date, "开盘", code, "买入", "开盘止损距离非正或过大")
                continue
            marks = {c: mark(p, today.get(c, {})) for c, p in positions.items()}
            receivable = sum(d["应收金额"] for d in receivables if pd.isna(d["实际入账日期"]))
            equity = cash + receivable + sum(p["shares"] * marks[c] for c, p in positions.items())
            heat = sum(p["shares"] * max(0.0, marks[c] - _stop_price(p)) for c, p in positions.items())
            risk = distance if stop_mode in ("fixed8", "atr2") else max(distance, order["initial_risk_adj"] / factor)
            unit_costs = transaction_costs(date, price, "买入", slippage_bps)
            proportional = (unit_costs["过户费"] + unit_costs["滑点成本"]) / price
            affordable = max(0.0, min((cash - 5) / (1 + proportional), cash / (1.0003 + proportional)))
            maximum = min(affordable / price, equity * strategy.max_weight / price,
                          equity * strategy.risk_per_trade / risk,
                          max(0.0, equity * strategy.heat_cap - heat) / risk,
                          row["prev_amount20"] * 0.001 / price)
            minimum = int(row["minimum_buy_size"])
            step = 100 if minimum == 100 else 1
            quantity = minimum + math.floor((maximum - minimum) / step) * step if maximum >= minimum else 0
            if quantity < minimum:
                fail(date, "开盘", code, "买入", "现金/权重/风险/参与率不足最小买入单位")
                continue
            amount = price * quantity
            costs = transaction_costs(date, amount, "买入", slippage_bps)
            if amount + costs["总费用"] > cash:
                # 浮点边界只向下调整，不截断现金或允许透支。
                quantity -= step
                if quantity < minimum:
                    fail(date, "开盘", code, "买入", "含费现金不足")
                    continue
                amount = price * quantity
                costs = transaction_costs(date, amount, "买入", slippage_bps)
            delta = -(amount + costs["总费用"])
            cash += delta
            positions[code] = {"shares": float(quantity), "minimum": minimum, "buy_date": date,
                               "cost": -delta, "stop": float(stop_raw * factor), "factor": float(factor),
                               "last": float(price), "missing": 0, "pending": 0.0, "reason": "", "stock_locks": []}
            trades.append({"日期": date, "时点": "开盘", "代码": code, "方向": "买入", "原因": "昨日信号",
                           "股数": float(quantity), "成交原价": float(price), "成交金额": amount, **costs,
                           "现金变动": delta, "成交后现金": cash, "已实现盈亏": 0.0,
                           "信号日期": order["signal_date"], "初始止损原价": stop_raw,
                           "初始风险金额": quantity * distance, "开仓权益": equity, "开仓权重": amount / equity})

        # 盘中记录所有触损；新仓只挂起，直到T+1才可成交，不能因次日反弹取消。
        for code, p in list(positions.items()):
            row = today.get(code, {})
            if (not active(row) or not _positive(row.get("open"))
                    or not _positive(row.get("adj_factor")) or not _positive(row.get("low"))):
                continue
            stop = _stop_price(p)
            if row["low"] <= stop:
                had_pending = p["pending"] > 0
                request(p, p["shares"], "盘中触及止损")
                if p["buy_date"] == date:
                    request(p, p["shares"], "买入日触损待下一交易日")
                    fail(date, "盘中", code, "卖出", "当日买入不可卖，退出请求保留")
                    continue
                if had_pending:
                    continue
                if (not _positive(row.get("high")) or stop > row["high"]
                        or not _positive(row.get("limit_down")) or stop <= row["limit_down"]):
                    fail(date, "盘中", code, "卖出", "止损不可确认成交，退出保留")
                else:
                    sell(date, code, row, stop, "盘中")

        for code, p in positions.items():
            row = today.get(code, {})
            if active(row) and _positive(row.get("close")):
                p["last"], p["missing"] = float(row["close"]), 0
            else:
                if active(row) and _positive(row.get("open")):
                    p["last"] = float(row["open"])
                p["missing"] += 1

        todays_instructions = list(instructions_by_day.get(date, []))
        if after_close is not None:
            returned = after_close(date, snapshot(date).copy(deep=True))
            if returned is not None:
                callback = pd.DataFrame(returned).copy()
                if not callback.empty:
                    if "date" in callback and not pd.to_datetime(callback.date).eq(date).all():
                        raise ValueError("收盘回调只能发出当前日期的指令")
                    callback["date"] = date
                    todays_instructions += _table(callback, ["date", "code"], "date", calendar).to_dict("records")
        seen = set()
        for instruction in todays_instructions:
            code = instruction["code"]
            if code in seen:
                raise ValueError("同日同代码收盘指令须合并")
            seen.add(code)
            p = positions.get(code)
            if p is None:
                fail(date, "收盘", code, "指令", "无持仓，忽略指令")
                continue
            stop = instruction.get("stop_adj")
            fraction = instruction.get("next_open_sell_fraction")
            if pd.notna(stop):
                if not _positive(stop) or stop < p["stop"]:
                    raise ValueError("收盘止损只能抬升")
                p["stop"] = float(stop)
            if pd.notna(fraction):
                if fraction not in (0.5, 1.0):
                    raise ValueError("次日卖出比例只支持0.5或1.0")
                step = 100 if p["minimum"] == 100 else 1
                quantity = p["shares"] if fraction == 1 else math.floor(p["shares"] * fraction / step) * step
                if quantity:
                    request(p, quantity, "收盘预定退出")
                else:
                    fail(date, "收盘", code, "指令", "减半不足卖出单位，不卖零股")
            instruction_log.append(dict(zip(INSTRUCTION_COLUMNS, [date, calendar[index + 1] if index + 1 < len(calendar) else pd.NaT,
                                                                    code, p["stop"], fraction])))
        day_positions = snapshot(date)
        holdings.extend(day_positions.to_dict("records"))
        value = day_positions["持仓市值"].sum()
        unrealized = day_positions["未实现盈亏"].sum()
        receivable = sum(d["应收金额"] for d in receivables if pd.isna(d["实际入账日期"]))
        accounts.append(dict(zip(ACCOUNT_COLUMNS, [date, cash, receivable, value, cash + receivable + value,
                                                   realized, unrealized, len(positions)])))
        if cash < 0:
            raise AssertionError("现金不得透支")
    if len(calendar):
        for order in orders_by_day.get(calendar[-1], []):
            fail(calendar[-1], "收盘", order["code"], "买入", "信号后无提供的交易日，未执行")
    return {"成交": pd.DataFrame(trades, columns=TRADE_COLUMNS),
            "失败订单": pd.DataFrame(failures, columns=FAIL_COLUMNS),
            "每日账户": pd.DataFrame(accounts, columns=ACCOUNT_COLUMNS),
            "每日持仓": pd.DataFrame(holdings, columns=POSITION_COLUMNS),
            "期末持仓": snapshot(calendar[-1]) if len(calendar) else pd.DataFrame(columns=POSITION_COLUMNS),
            "分红应收": pd.DataFrame(receivables, columns=RECEIVABLE_COLUMNS),
            "公司行动": pd.DataFrame(action_log, columns=ACTION_COLUMNS),
            "收盘指令": pd.DataFrame(instruction_log, columns=INSTRUCTION_COLUMNS)}
