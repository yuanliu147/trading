#!/usr/bin/env python3
"""只读取本项目实际结果生成中文交付，不推断未运行的绩效。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "结果"
MAIN_SEED = 20261030


def label(ident):
    if ident.startswith("B"):
        band, months = ident[1:].split("_M")
        return f"估值前{int(band)}%／每{months}月审视"
    if ident.startswith("M"):
        return f"估值前20%／指数{ident[1:]}日线退出"
    if ident.startswith("T"):
        return f"估值前20%／{ident[1:]}倍真实波幅跟踪"
    return ident


def control_of(ident):
    if ident.startswith("M"):
        return f"C_MKT{ident[1:]}"
    if ident.startswith("T"):
        return f"C_ATR{ident[1:]}"
    return f"C_M{ident.split('_M')[1]}"


def percent(x):
    return "无有效值" if pd.isna(x) else f"{float(x):+.2%}"


def money(x):
    return f"{float(x):,.0f}元"


def benchmark():
    idx = pd.read_csv(ROOT / "数据" / "日历指数.csv", dtype={"日期": str})
    start = int(np.flatnonzero(idx["日期"].to_numpy() == "2010-12-31")[0])
    s = idx.iloc[start:].reset_index(drop=True)
    x = s["沪深300收盘"].to_numpy(dtype=float)
    year_end = s.groupby(s["日期"].str[:4].astype(int))["沪深300收盘"].last()
    annual, previous = {}, float(x[0])
    for year in range(2011, 2021):
        annual[year] = float(year_end[year] / previous - 1)
        previous = float(year_end[year])
    mid = s.loc[s["日期"] <= "2015-12-31"].index[-1]

    def period(a, b):
        span = (pd.Timestamp(s["日期"].iloc[b]) - pd.Timestamp(s["日期"].iloc[a])).days
        return float((x[b] / x[a]) ** (365.25 / span) - 1)

    return {"口径": "沪深300价格指数，不含分红，不是可交易ETF净值，未扣任何费用",
            "起止": [s["日期"].iloc[0], s["日期"].iloc[-1]],
            "CAGR": period(0, len(x) - 1), "2011-2015_CAGR": period(0, mid),
            "2016-2020_CAGR": period(mid, len(x) - 1),
            "最大回撤": float((x / np.maximum.accumulate(x) - 1).min()), "年度收益": annual}


def path_stats(core_ids):
    z = np.load(OUT / "共同日历账户收益.npz")
    ids = [str(v) for v in z["ids"]]
    dates = [str(z["base_date"])] + [str(v) for v in z["return_dates"]]
    rows = []
    for ident in core_ids:
        k = ids.index(ident)
        for j, seed in enumerate(z["seeds"]):
            nav = 10000 * np.exp(np.r_[0.0, np.cumsum(z["logs"][k, j])])
            peak = np.maximum.accumulate(nav)
            dd = nav / peak - 1
            trough = int(dd.argmin())
            top = int(nav[:trough + 1].argmax())
            rows.append({"配置": ident, "种子": int(seed), "最大回撤": float(dd[trough]),
                         "峰值日": dates[top], "谷底日": dates[trough],
                         "期末前恢复峰值": bool((nav[trough:] >= nav[top]).any()),
                         "期末低于历史峰值": float(nav[-1] / peak[-1] - 1)})
    return pd.DataFrame(rows)


def concentration(ident):
    folder = OUT / "主种子账户" / ident
    rounds = pd.read_csv(folder / "回合.csv")
    hold = pd.read_csv(folder / "期末持仓.csv")
    pnl = rounds["盈亏"].sort_values(ascending=False)
    return {"完整回合": int(len(rounds)),
            "胜率": float((rounds["盈亏"] > 0).mean()) if len(rounds) else None,
            "平均持有交易日": float(rounds["持有交易日"].mean()) if len(rounds) else None,
            "已平仓盈亏": float(pnl.sum()),
            "期末未平仓累计盈亏": float(hold["累计盈亏"].sum()) if len(hold) else 0.0,
            "前五笔盈利合计": float(pnl.head(5).sum()), "其余已平仓盈亏": float(pnl.iloc[5:].sum()),
            "说明": "仅主种子；集中度描述样本脆弱性，不作为删除赢家的硬门槛"}


def main():
    summary = pd.read_csv(OUT / "配置种子分布.csv")
    accounts = pd.read_csv(OUT / "全部账户.csv")
    years = pd.read_csv(OUT / "全部账户年度.csv")
    stats = pd.read_csv(OUT / "共同日期块统计.csv")
    pressure = pd.read_csv(OUT / "成本与会计情景.csv")
    core = summary[summary["类别"] == "core"].copy()
    assert len(core) == 12 and len(accounts) == 240
    core_ids = core["配置"].tolist()
    core["规则"] = core["配置"].map(label)
    core["描述性前沿"] = core["描述性前沿"].astype(str).str.lower().eq("true")
    median_cagr = summary.set_index("配置")["CAGR中位"]
    core["匹配控制"] = core["配置"].map(control_of)
    core["控制CAGR中位"] = core["匹配控制"].map(median_cagr)
    paired = stats[stats["比较"] == "同种子同退出控制差"].set_index("配置")
    absolute = stats[stats["比较"] == "账户对数收益对零"].set_index("配置")
    core["控制差年化对数均值"] = core["配置"].map(paired["年化对数均值"])
    core["控制差95下限"] = core["配置"].map(paired["95点区间下限"])
    core["控制差95上限"] = core["配置"].map(paired["95点区间上限"])
    core["控制差单侧p"] = core["配置"].map(paired["单侧p"])
    core["控制差Holm"] = core["配置"].map(paired["Holm校正p"])
    rank = {}
    for ident, g in accounts[accounts["类别"] == "core"].groupby("配置"):
        main_value = float(g.loc[g["种子"] == MAIN_SEED, "CAGR"].iloc[0])
        rank[ident] = int((g["CAGR"] > main_value).sum() + 1)
    core["主种子收益排名"] = core["配置"].map(rank)
    base_median = accounts.groupby("配置")["CAGR"].median()
    doubled = pressure[pressure["情景"] == "执行费用与滑点双倍"].groupby("配置")["CAGR"].median()
    core["费用双倍CAGR中位"] = core["配置"].map(doubled)
    main_base = accounts[accounts["种子"] == MAIN_SEED].set_index("配置")["CAGR"]
    accounting = pressure[pressure["情景"] != "执行费用与滑点双倍"].copy()
    accounting["变化"] = accounting["CAGR"] - accounting["配置"].map(main_base)
    accounting_max = accounting.groupby("配置")["变化"].apply(lambda v: float(v.abs().max()))
    paths = path_stats(core_ids)
    paths.to_csv(OUT / "各种子回撤区间.csv", index=False, encoding="utf-8-sig")
    path_summary = paths.groupby("配置").agg(
        峰值在2015年种子数=("峰值日", lambda v: int(v.str.startswith("2015").sum())),
        期末未恢复峰值种子数=("期末前恢复峰值", lambda v: int((~v.astype(bool)).sum())))
    core = core.join(path_summary, on="配置")
    bm = benchmark()
    annual = years[years["配置"].isin(core_ids)].groupby(["年份", "配置"])["年度收益"].median().unstack()
    annual = annual[core_ids]
    annual.columns = [label(c) for c in annual.columns]
    annual["沪深300价格指数"] = [bm["年度收益"][y] for y in annual.index]
    annual.to_csv(OUT / "逐年种子中位收益.csv", encoding="utf-8-sig")
    conc = {ident: concentration(ident) for ident in ["B10_M3", "B20_M3", "B20_M6", "M180", "T3"]}

    columns = {"配置": "配置编号", "规则": "规则", "CAGR中位": "净年化中位数",
               "CAGR最小": "净年化最低种子", "CAGR最大": "净年化最高种子",
               "回撤中位": "最大回撤中位数", "最深回撤": "最深种子回撤",
               "正CAGR种子数": "正收益种子数", "两段均正种子数": "两段均正种子数",
               "2011-2015_CAGR中位": "前段净年化中位数", "2016-2020_CAGR中位": "后段净年化中位数",
               "控制CAGR中位": "同规则随机选股净年化中位数",
               "控制差年化对数均值": "相对随机选股年化对数差", "控制差95下限": "相对差95区间下限",
               "控制差95上限": "相对差95区间上限", "控制差单侧p": "相对差单侧p", "控制差Holm": "相对差霍尔姆p",
               "费用双倍CAGR中位": "费用滑点双倍后净年化中位数", "主种子收益排名": "主种子在8种子中收益排名",
               "峰值在2015年种子数": "最大回撤起点在2015年的种子数", "期末未恢复峰值种子数": "期末仍未恢复峰值的种子数",
               "期末净值中位": "期末权益中位数", "平均仓位中位": "平均仓位中位数",
               "完整回合数中位": "完整交易回合数中位数", "胜率中位": "胜率中位数",
               "盈亏比中位": "实现盈亏比中位数", "费用合计中位": "费用合计中位数",
               "最大单票权重中位": "最大单票权重中位数", "描述性前沿": "处于描述性收益回撤前沿"}
    core[list(columns)].rename(columns=columns).to_csv(OUT / "核心系统中文比较.csv", index=False, encoding="utf-8-sig")

    positive = core[core["CAGR中位"] > 0]
    both_periods = core[(core["2011-2015_CAGR中位"] > 0) & (core["2016-2020_CAGR中位"] > 0)]
    positive_holm = absolute[(absolute["年化对数均值"] > 0) & (absolute["Holm校正p"] <= .05)]
    paired_holm = paired[(paired["年化对数均值"] > 0) & (paired["Holm校正p"] <= .05)]
    value_ids = [c for c in core_ids if c.startswith(("B10", "B20"))]
    value = core[core["配置"].isin(value_ids)]
    market = core[core["配置"].str.startswith("M")]
    trail = core[core["配置"].str.startswith("T")]
    b20 = conc["B20_M3"]
    unrecovered = int(core["期末未恢复峰值种子数"].sum())
    below_peak = int((paths["期末低于历史峰值"] < 0).sum())
    closest_gap = float(-paths["期末低于历史峰值"].max())
    by_id = core.set_index("配置")
    supplement = {"沪深300价格基准": bm, "主种子收益排名": rank, "费用双倍CAGR中位": doubled.to_dict(),
                  "会计情景主种子CAGR最大绝对变化": accounting_max.to_dict(), "主种子集中度": conc,
                  "回撤区间汇总": path_summary.reset_index().to_dict("records"),
                  "相对随机选股": core[["配置", "控制CAGR中位", "控制差年化对数均值", "控制差95下限",
                                     "控制差95上限", "控制差单侧p", "控制差Holm"]].to_dict("records")}
    (OUT / "补充核查.json").write_text(json.dumps(supplement, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    verdict = "按“扣费后稳定为正、回撤可承受、并能与同规则随机选股区分”的标准，本轮没有合格的可交付系统"
    result = {
        "生成时间": str(pd.Timestamp.now()), "用户本金": 10000, "判定": verdict,
        "允许数据期": "2010-2020；2010预热", "账户期": "2010-12-31初始净值，2011-2020交易",
        "证据等级": "历史近似账户研究，不是新样本外，不是已验证实盘系统",
        "核心配置数": len(core), "主账户单元数": len(accounts), "压力账户单元数": len(pressure),
        "中位净年化为正配置数": len(positive), "两段中位净年化均正配置数": len(both_periods),
        "对零为正且霍尔姆通过配置": positive_holm.index.tolist(),
        "对匹配控制为正且霍尔姆通过配置": paired_holm.index.tolist(),
        "相对控制最小未校正单侧p": float(paired["单侧p"].min()),
        "核心中位净年化范围": [float(core["CAGR中位"].min()), float(core["CAGR中位"].max())],
        "沪深300价格基准": {k: bm[k] for k in ("口径", "CAGR", "最大回撤")},
        "期末未恢复峰值的核心账户数": f"{unrecovered}/96",
        "不能据此认定": ["未来收益", "首次披露时点正确的财务信号", "真实派息与股份到账时点已还原",
                     "股票选择或择时有可复制超额", "本轮存在未触碰样本外"],
        "本轮停止": "全部冻结配置完成后，不自动追加过滤器或新策略",
    }
    (OUT / "结论.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "万元账户：低换手估值与风险管理研究", "",
        "身份：历史近似账户研究；不是已验证实盘系统，不提供当前买入名单。",
        "本金：10,000元。不预设年化目标，按历史近似模拟分布报告。",
        "时间：只读取2010—2020；2010用于预热，2011—2020实际模拟。未读取2021年以后行情。", "",
        "零、结论（先读）",
        f"{verdict}；不能把任何一组参数当作已研究出的交易系统使用。",
        f"收益最高的估值前10%/20%四组，8种子中位年化{percent(value['CAGR中位'].min())}至{percent(value['CAGR中位'].max())}，"
        f"中位最大回撤{percent(value['回撤中位'].min())}至{percent(value['回撤中位'].max())}。"
        f"同期沪深300价格指数年化{percent(bm['CAGR'])}、最大回撤{percent(bm['最大回撤'])}（不含分红，不是可交易ETF，只作量级参照）。",
        f"与同规则、同费用、同种子的随机选股相比，中位净年化更高；但以8种子等权日均对数收益差为检验对象，95%区间全部跨零；"
        f"相对差最小未校正单侧p为{paired['单侧p'].min():.3f}，24项霍尔姆校正后对零和对控制均为0项通过。",
        f"最大回撤多从2015年开始：96个核心账户中{int(core['峰值在2015年种子数'].sum())}个的回撤起点在2015年，"
        f"{unrecovered}个没有回到最大回撤起点前的峰值；{below_peak}/96个在2020年底低于各自历史最高净值（最接近者仍低{closest_gap:.1%}）。"
        f"前段（2011—2015）12组中{int((core['2011-2015_CAGR中位'] > 0).sum())}组中位为正，"
        f"同期沪深300价格指数年化{percent(bm['2011-2015_CAGR'])}；后段（2016—2020）只有{int((core['2016-2020_CAGR中位'] > 0).sum())}组中位为正，"
        f"同期指数年化{percent(bm['2016-2020_CAGR'])}。后段随机选股对照也普遍为负，估值排序在部分配置中位仍高于随机，但不足以普遍转正。",
        f"你关心的动态管理在这里没有改善收益：市场退出把平均仓位降到约{market['平均仓位中位'].min():.0%}—{market['平均仓位中位'].max():.0%}，"
        f"中位回撤仍为{percent(market['回撤中位'].min())}至{percent(market['回撤中位'].max())}，中位年化{percent(market['CAGR中位'].min())}至{percent(market['CAGR中位'].max())}。"
        f"跟踪止损相对估值前20%/季度基线（中位回撤{percent(by_id.loc['B20_M3', '回撤中位'])}、中位年化{percent(by_id.loc['B20_M3', 'CAGR中位'])}）："
        f"2倍降到{percent(by_id.loc['T2', '回撤中位'])}、4倍略降到{percent(by_id.loc['T4', '回撤中位'])}、3倍为{percent(by_id.loc['T3', '回撤中位'])}；"
        f"中位年化分别为{percent(by_id.loc['T2', 'CAGR中位'])}、{percent(by_id.loc['T4', 'CAGR中位'])}、{percent(by_id.loc['T3', 'CAGR中位'])}。"
        f"三组跟踪止损完整回合约{trail['完整回合数中位'].min():.0f}—{trail['完整回合数中位'].max():.0f}次，"
        f"10年显式费用（不含已计入成交价的滑点）约{money(trail['费用合计中位'].min())}—{money(trail['费用合计中位'].max())}，约为本金的一成。",
        f"利润集中：主种子估值前20%/季度配置{b20['完整回合']}个完整回合，前5笔盈利合计{money(b20['前五笔盈利合计'])}，"
        f"其余已平仓合计{money(b20['其余已平仓盈亏'])}。" + (
            "主种子在估值前20%/季度和估值前20%/半年两组中都是8个种子里收益最高的，交易明细偏乐观，应看种子分布。"
            if rank['B20_M3'] == 1 and rank['B20_M6'] == 1 else
            f"主种子在估值前20%/季度与估值前20%/半年中的收益排名分别为第{rank['B20_M3']}和第{rank['B20_M6']}（共8个），应看种子分布。"),
        f"费用滑点双倍（预先声明，覆盖描述性前沿和三个中心配置，各8种子）：估值前10%/季度、估值前20%/季度、估值前20%/半年三组中位年化下降"
        f"{-(value['费用双倍CAGR中位'] - value['CAGR中位']).max() * 100:.2f}至{-(value['费用双倍CAGR中位'] - value['CAGR中位']).min() * 100:.2f}个百分点；"
        f"指数180日线退出下降{-(by_id.loc['M180', '费用双倍CAGR中位'] - by_id.loc['M180', 'CAGR中位']) * 100:.2f}个百分点；"
        f"2倍、3倍跟踪止损分别下降{-(by_id.loc['T2', '费用双倍CAGR中位'] - by_id.loc['T2', 'CAGR中位']) * 100:.2f}和"
        f"{-(by_id.loc['T3', '费用双倍CAGR中位'] - by_id.loc['T3', 'CAGR中位']) * 100:.2f}个百分点。"
        f"公司行动到账、解锁、税基情景使主种子年化的最大绝对变化为{accounting_max.max() * 100:.2f}个百分点。结论不是由这些近似决定的。", "",
        "一、全部核心配置（8种子分布，不择优）",
        f"中位净年化为正：{len(positive)}/12；两段中位均正：{len(both_periods)}/12。",
        "霍尔姆检验对象是8种子等权日均对数收益及其精确控制差，不是表内的中位净年化；两种估计对象不得混读。",
        "这些种子只是持仓抽样，不是8段独立历史；前沿只是已见历史的描述，不能当作未来最优。", "",
        "配置｜净年化中位｜最大回撤中位｜正收益种子｜前段中位｜后段中位｜同规则随机选股中位",
    ]
    for _, r in core.iterrows():
        lines.append(f"{r['规则']}｜{percent(r['CAGR中位'])}｜{percent(r['回撤中位'])}｜{int(r['正CAGR种子数'])}/8｜"
                     f"{percent(r['2011-2015_CAGR中位'])}｜{percent(r['2016-2020_CAGR中位'])}｜{percent(r['控制CAGR中位'])}")
    lines += ["", "逐年种子中位收益见 逐年种子中位收益.csv；各种子回撤起止见 各种子回撤区间.csv。", "",
              "二、固定中心的完整规则（研究原型，不因结果改参数）",
              "股票池：沪深主板，当日非风险警示、非停牌及已知复牌首日；上市满250自然日与120交易日；原始收盘至少2元，过去20日平均成交额至少3000万元（至少15个有效日），当日市盈率、市净率均为有限正值。",
              "排序：分别计算盈利收益率（市盈率倒数）与账面市值比（市净率倒数）的百分位，等权平均。中心取最便宜20%；10%与40%只是事先登记的邻域。",
              "持仓：中心最多同时两只，不加杠杆。每季末收盘审视；仍在池中的持仓保留，不强行卖出再等权。新候选按固定SHA256顺序而非事后赢家排序。主种子20261030，另7个种子检查少量持仓的抽样敏感性。",
              "买入：审视后的下一交易日开盘才尝试。目标金额为前日权益除以槽位，实际预算受可用现金、税负预留、100股整手和费用约束。失败不替补；待卖但未卖完的股份仍占槽。",
              "基线退出：只在定期审视时卖出估值池外证券；当日行情中已观察到风险警示也发出次开退出（行情缺失不当作风险警示）。没有固定盈利目标。",
              "市场退出：独立版本使用沪深300收盘与满窗均线；中心180交易日，邻域120/250。指数不高于均线时退出且不新买，恢复后等下次定期审视才补位。它是价格指数状态，不是ETF收益。",
              "跟踪退出：另一独立版本用20日平均真实波幅，中心3倍、邻域2/4倍。入场连续价只用开盘前已知的昨日连续收盘与今日除权参考价换算；只在收盘后上移止损，不向下放宽；跌破后次开卖。没有把市场退出和跟踪退出再组合择优。",
              "实际成交：使用不复权原始价，不利取整到分；停牌、开盘涨停买入、开盘跌停卖出等不虚构成交。成交量限制来自前日可知数据，并对次日实际价量检查。",
              "重入：退出后不是每天补仓，仅在下一次定期审视重新参加。没有年末虚构清仓。", "",
              "三、万元账户费用与会计假设",
              "佣金双边0.025%，每笔最低5元；研究期卖出印花税0.1%。早期沪市过户费采用每股0.001元且最低1元的偏高统一假设，2015年8月后按成交额0.00002；不能称所有日期历史费率精确还原。基础单边滑点10个基点，加参与率冲击；另做双倍压力。",
              "现金分红统一计20%税负，净额先记应收，默认30自然日后可用；送转新增股默认20交易日后可卖，并按每股1元税基、20%计提税负；现金不足只记应付不透支。配股一律放弃，不免费增股，不抹掉原始价格稀释。",
              "上述到账、可售及税基只是情景，不是真实日期，也不保证上下界。主种子另跑多组到账/解锁/税负情景；成本压力跑全部8种子。缺行持仓不凭未来退市日删除，另做连续缺行20日后市值归零压力。",
              "376个账户（240个主账户及136个压力账户）的现金流、净值恒等式、应收应付、股数与可售锁定股数残差均为0，现金为负、超槽、T+1、预算和成交容量违规均为0。", "",
              "四、如何读赢亏与尾部",
              "胜率分母为完整开平仓回合；分批出售不是多个胜利。盈亏含买卖费用、净红利及送转计税；未平仓每天计市值，应收与应付都进入净值。",
              "少数大赢家可以构成真实右尾，因此不以删掉赢家后仍达到任意年化作必要门槛。但两只持仓、十年约10—20个回合时，结果几乎由少数几笔决定，统计上难以与运气区分。",
              "最大单票权重是实际逐日权重，初始分两槽不意味着永远不超过半仓。低回撤可能只是现金多，需与同费用、同退出、同种子的控制账户分开比较。", "",
              "五、证据限制与使用边界",
              "历史行情日PE/PB没有首次公告和重述版本链；滞后20/60交易日也不能修复这个缺口。不能称盈利超预期或严格财务PIT。",
              "已有历史曾被反复研究，2016—2020不是新的纯净样本外；本轮预先固定网格只减少追加选择，不能消除工作区历史试验的选择偏差。",
              "运行前修正了两处实现问题（缺行哨兵误作风险警示、跟踪入场使用当日收盘），见 证据/运行前实现修正.json；均在任何真实账户运行之前完成。",
              "公司行动时点、税基、开盘队列和证券代码迁移等仍有数据限制。完整会计恒等式成立只证明模型内部自洽，不证明市场交易完全可实现。",
              "本轮结束后不自动扩网格、不因负结果换筛选条件；没有验证过的策略，不给未来收益承诺。", "",
              "配套文件：核心系统中文比较.csv、逐年种子中位收益.csv、各种子回撤区间.csv、补充核查.json、全部账户.csv、全部账户年度.csv、共同日期块统计.csv、成本与会计情景.csv；主种子账户目录含每个核心和控制账户的净值、日志、回合与期末持仓。",
              "复现：先运行代码/数据准备.py，再运行代码/测试_账户引擎.py、代码/测试_统计.py，随后代码/研究主流程.py、代码/生成交付.py。",
              ]
    (OUT / "系统规则与研究结论.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"判定": verdict, "核心正中位": len(positive), "两段正中位": len(both_periods),
                      "对零霍尔姆通过": len(positive_holm), "相对控制霍尔姆通过": len(paired_holm),
                      "主种子排名": rank, "期末未恢复": unrecovered,
                      "交付": str(OUT / "系统规则与研究结论.txt")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
