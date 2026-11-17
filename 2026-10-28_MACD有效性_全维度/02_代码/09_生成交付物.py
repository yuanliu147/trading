#!/usr/bin/env python3
"""09 生成交付物：单文件 HTML + Excel 数据表"""
import os
import base64
import html
import numpy as np
import pandas as pd

ROOT = "/Users/admin/WorkSpace/trading/2026-10-28_MACD有效性_全维度"
RES = os.path.join(ROOT, "03_结果")
FIG = os.path.join(ROOT, "04_图表")


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def fmt(col, v):
    if pd.isna(v):
        return "—"
    if isinstance(v, str):
        return html.escape(v)
    is_t = (col.endswith("t") or "_t" in col or col in ("t", "t_nw", "t_cluster", "日度t", "聚类t", "毛t", "净t")) and col != "fast"
    if is_t:
        return f"{v:.2f}"
    if col in ("周期", "fast", "slow", "signal", "信号数", "信号日数", "交易数", "样本",
               "n", "n5", "n10", "n20", "n60", "日均持仓", "换手",
               "分位>50.5%的组合数", "中位超额>0的组合数", "胜率>50%的组合数", "q<0.10的组合数",
               "参数组合数", "分位大于50_5组合数", "中位大于0组合数",
               "交易日数", "股票日", "月数", "期数", "股票期", "样本数", "正系数年份", "年份数",
               "多空大于0年份", "平均持仓", "持有期"):
        return f"{int(v):,}"
    pct = any(k in col for k in ["分位", "超额", "胜率", "年化", "回撤", "仓位", "收益", "覆盖率", "share50",
                                 "大于0", "大于50", "占比", "占盈亏", "平均IC", "IC", "多空5_1",
                                 "系数", "原始差", "控制后差", "状态×水平", "状态1绝对", "状态0绝对",
                                 "市场绝对", "均值绝对", "中位绝对", "Q1", "Q2", "Q3", "Q4", "Q5",
                                 "P10", "P90", "绝对"])
    if pct:
        return f"{v*100:.2f}%"
    if col in ("夏普", "全样本夏普", "2016后夏普"):
        return f"{v:.2f}"
    if col in ("每年信号", "日均成交额万元"):
        return f"{v:.0f}"
    if "q" in col.lower() and ("FDR" in col or "q_bh" in col or "q_all" in col):
        return f"{v:.3f}"
    if abs(v) < 1:
        return f"{v:.4f}"
    return f"{v:.2f}"


def cell_class(col, v):
    if pd.isna(v) or isinstance(v, str):
        return ""
    if "分位" in col or col in ("share50", "分位大于50"):
        return "pos" if v >= 0.5 else "neg"
    if "超额" in col or "日度平均" in col:
        return "pos" if v > 0 else "neg"
    if (col.endswith("t") or "_t" in col or ("t" in col and col != "fast")):
        return "pos" if v >= 2 else ("neg" if v <= -2 else "")
    if any(k in col for k in ["IC", "系数", "原始差", "控制后差", "状态×水平", "多空5_1",
                              "状态1绝对", "状态0绝对", "市场绝对", "均值绝对", "中位绝对",
                              "Q1", "Q2", "Q3", "Q4", "Q5", "P10", "P90", "绝对"]):
        return "pos" if v > 0 else "neg"
    if "最大回撤" in col:
        return "neg" if v < -0.4 else ("warn" if v < -0.25 else "")
    return ""


def table(df, cols, max_rows=None, class_="data"):
    d = df.copy()
    if max_rows:
        d = d.head(max_rows)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for _, r in d.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            tds.append(f'<td class="{cell_class(c, v)}">{fmt(c, v)}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="table-wrap"><table class="{class_}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def img(path, caption):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


# ---------- 表 ----------
overall = read("最终_总体.csv")
yearly = read("最终_年度.csv")
cond = read("最终_条件.csv")
pdig = read("最终_参数摘要.csv")
grid = read("参数网格_含q.csv")
variants = read("信号变体.csv")
varport = read("最终_变体组合.csv")
exits = read("最终_离场.csv")
idx = read("最终_指数择时.csv")
ew = read("最终_指数择时_等权.csv")
dist = read("最终_分布.csv")
pockets = read("最终_交互格.csv")
pstock = read("最终_周期股票.csv")
pbest = read("最终_周期参数.csv")
cidx = read("最终_周期指数择时.csv")
multi = read("最终_多周期.csv")
tfeat = read("最终_特征全样本.csv")
tstruct = read("最终_特征结构.csv")
tstrict = read("最终_特征严格增量.csv")
tstate = read("最终_结构状态.csv")
tmonth = read("最终_月度下行.csv").rename(
    columns={"Q1超额": "Q1超额_月均", "Q5超额": "Q5超额_月均"})
tmonthfm = read("最终_月度下行FM.csv")
tmonthdist = read("月度下行_分布.csv")
ttrend = read("月度下行_趋势定义.csv")
tport = read("最终_特征组合.csv")
tperf = read("最终_周期特征.csv")
tperfs = read("最终_周期特征结构.csv")

# 关键条件
wanted = [
    ("is_st", "1.0", "ST 股"), ("成交额分组", "低额", "成交额低"), ("波动分组", "低波", "低波动"),
    ("量比", "<0.7", "量比 <0.7"), ("个股趋势", "年线下", "年线下方"), ("周线状态", "周线空头", "周线空头"),
    ("zero_state", "双下", "零轴双下"), ("信号密度", ">100", "信号密度 >100"), ("DIF位置", "<0", "DIF <0"),
    ("board", "主板", "主板"), ("市场趋势", "强趋势(年线上+20日涨)", "强趋势"),
    ("个股趋势", "年线上+年线升", "年线上+年线升"), ("zero_state", "双上", "零轴双上"),
    ("周线状态", "周线多头", "周线多头"), ("量比", ">2", "量比 >2"), ("成交额分组", "高额", "成交额高"),
    ("波动分组", "高波", "高波动"), ("股价年龄", "<1年", "上市 <1 年"), ("DIF位置", ">1.5", "DIF >1.5"),
]
rows = []
for dim, grp, label in wanted:
    x = cond[(cond["维度"] == dim) & (cond["分组"] == grp)]
    if len(x):
        r = x.iloc[0]
        rows.append(dict(条件=label, 信号数=int(r["信号数"]), 平均分位20=r["平均分位20"],
                         分位大于50=r["分位大于50"], 平均超额20=r["平均超额20"],
                         日度平均超额20=r["日度平均超额20"], 日度t=r["日度t"],
                         聚类t=r["聚类t"], 分位_2010_2017=r["分位_2010_2017"],
                         分位_2018_2026=r["分位_2018_2026"]))
keycond = pd.DataFrame(rows).sort_values("平均分位20", ascending=False)

# 参数 top
top = grid.sort_values("pct20", ascending=False).head(8)[
    ["fast", "slow", "signal", "n", "pct20", "share50", "ex20", "t20", "q_bh",
     "median20", "ex20_early", "ex20_late"]].copy()
top.columns = ["fast", "slow", "signal", "信号数", "平均分位20", "分位>50", "平均超额20",
               "t20", "FDR q", "中位超额20", "2010-17 超额20", "2018-26 超额20"]
default = grid[(grid.fast == 12) & (grid.slow == 26) & (grid.signal == 9)].copy()
default = default[["fast", "slow", "signal", "n", "pct20", "share50", "ex20", "t20", "q_bh",
                   "median20", "ex20_early", "ex20_late"]]
default.columns = top.columns

# 变体 + 组合
v = variants[["变体", "n", "mean_pct", "share50", "mean_ex", "t_cluster",
              "pct_early", "pct_late"]].copy()
v = v.merge(varport[["变体", "净年化", "净t"]], on="变体", how="left")
v.columns = ["变体", "信号数", "平均分位20", "分位>50", "平均超额20", "聚类t",
             "2010-17 分位", "2018-26 分位", "扣费组合年化", "扣费组合 t"]

# 指数表：宽表
idx["区间"] = idx["区间"].replace({"全样本": "2006-2026"})
idx_key = idx[idx["指数"].isin(["CSI300", "CSI500", "CSI1000"]) &
              idx["规则"].isin(["买入持有", "DIF>DEA", "DIF>0", "DIF>DEA且DIF>0"])].copy()
idx_key["指数"] = idx_key["指数"].map({"CSI300": "沪深300", "CSI500": "中证500", "CSI1000": "中证1000"})
idx_p = idx_key.pivot_table(index=["指数", "规则"], columns="区间",
                            values=["年化", "最大回撤", "夏普"], aggfunc="first").reset_index()
idx_p.columns = ["指数", "规则"] + [f"{a}_{b}" for a, b in idx_p.columns[2:]]
# 简化列名
idx_p = idx_p.rename(columns={
    "年化_2006-2026": "全样本年化", "年化_2016-2026": "2016后年化",
    "最大回撤_2006-2026": "全样本回撤", "最大回撤_2016-2026": "2016后回撤",
    "夏普_2006-2026": "全样本夏普", "夏普_2016-2026": "2016后夏普"})
idx_p = idx_p[["指数", "规则", "全样本年化", "全样本回撤", "全样本夏普",
               "2016后年化", "2016后回撤", "2016后夏普"]]

ew18 = ew[ew["区间"] == "2018-2026"].copy()

# 周期表
cidx16 = cidx[(cidx["区间"] == "2016-2026") & (cidx["规则"].isin(["DIF>DEA", "持有"]))].copy()
cidx16 = cidx16.rename(columns={"回撤": "最大回撤"})
keys = [
    ("日线+周/月", "日线金叉(基准)", "20日"), ("日线+周/月", "+周线多头", "20日"),
    ("日线+周/月", "+月线多头", "20日"), ("日线+周/月", "+周线多头+月线多头", "20日"),
    ("日线+周/月", "+周线空头+月线空头", "20日"), ("日线+周/月", "+周线空头+月线多头", "20日"),
    ("周线+月", "周线金叉(基准)", "4周"), ("周线+月", "+月线多头", "4周"),
    ("周线+月", "+月线多头", "12周"), ("月线+周", "月线金叉(基准)", "12月"),
    ("月线+周", "+周线多头", "12月"),
]
m2 = pd.concat([multi[(multi["层级"] == a) & (multi["组合"] == b) & (multi["持有期"] == c)]
                for a, b, c in keys], ignore_index=True)
m2 = m2.rename(columns={"pct": "平均分位", "ex": "平均超额", "t_cl": "聚类t",
                        "early": "2010-17 分位", "late": "2018-26 分位"})

# 决策表
decisions = [
    ("个股日线金叉（全样本）", "无效", "bad", "分位 49.79%；扣费组合 -5.7%/年，t=-7.96", "不能当买入信号"),
    ("金叉 + 零轴上 / 周线多头 / 放量 / 高额 / 高波 / 强趋势", "明显有害", "bad", "分位 46%–48%；平均超额 -0.3% ~ -0.9%/20日", "不要加这些确认，它们是最差条件"),
    ("金叉 + 零轴下 / 周线空头 / 低波 / 低额", "毛正、净不赚", "warn", "分位 50.8%–52.5%；毛超额 0.14%–0.56%；零轴/周线/低波日度 t 为负，低额 t≈0", "只能当多信号里的弱特征"),
    ("金叉 + ST", "日线金叉单维唯一稳健正 pocket", "good", "分位 53.19%；+0.97%/20日；组合 +1.3%/年，t=0.32", "容量/制度限制，不能当策略"),
    ("绿柱缩短（未金叉）", "原始略正，独立增量约 0", "warn", "分位 50.62%；+0.134%/20日；控制水平/动量/波动/换手/成交额后 -0.038%（t=-1.20）；扣费 -6.5%/年", "不要单独用"),
    ("日线连续特征（柱体/DIF/斜率/面积）", "全部负 IC，不是顺势指标", "bad", "12/12 特征 20日 IC<0；DIF -7.16%（t=-8.42），柱体 -2.80%；Q5-Q1 -1.30%", "高值应反向理解"),
    ("红柱放大 / 柱体上行", "唯一稳健的独立增量，方向为负", "bad", "严格 FM：-0.18%/-0.15%/20日（t=-5.9/-5.1）；高波/高换手/高位 -0.3% ~ -0.5%；剔除红柱放大 long-only 6.08% vs 6.09% 不增厚", "只能做卖出/回避/减仓参考"),
    ("DIF>DEA（金叉状态）", "严格增量为负", "bad", "-0.226%/20日（t=-2.98）；低 DIF 时交互为正、高 DIF 时为负", "不是买入信号"),
    ("月线 + 长期下跌趋势", "唯一强正结构，但只是相对反弹", "warn", "12m IC +11.8%（t=5.15），Q5-Q1 +10.1%；高流动 Q5 绝对 +10.0%（月均）仍跑输宽基；3m Q5 中位 -1.3%", "只能做相对/对冲/研究线索"),
    ("零轴穿越 / 周线金叉 / 底背离", "无效或负", "bad", "49.1% / 49.2% / 49.95%", "不要用"),
    ("死叉离场", "不如固定 20 日", "bad", "均值 +1.0% vs +1.32%；中位 -2.13% vs 0.00%；胜率 35% vs 50%", "不要单独用；用追踪/分批兑现"),
    ("个股金叉 + 弱市强股 + 低波/低额 + 洗盘日", "探索性正口袋", "warn", "20日 +1.66%；两段 +1.53/+1.83；12/15 年正；中位 +0.15%，组合 t=1.61，日额 2,666 万", "只能当研究线索"),
    ("个股周线/月线金叉", "周线更弱、月线 12 个月弱正", "warn", "周线 4 周 49.39%；月线 12 个月 50.73%，中位 -8.4%，t=1.27", "不能单独用"),
    ("日+周+月多头共振", "最差", "bad", "47.96%，t=-3.92；周线+月线多头 4 周 50.03%，后期 <50%", "不要用"),
    ("指数择时（宽基）", "日线/月线有效，周线最差", "good", "2016+ 月线 4.6/5.1/0.2%，回撤 -28/-20/-32 vs -47/-43/-57；1296 组 44% 正、中位 -0.2%；周线三指数全负", "日线用于 2016 前/高波动；2016 后月线；周线不要用"),
]
dec_html = "".join(
    f'<tr><td><b>{html.escape(a)}</b></td><td><span class="badge {c}">{html.escape(b)}</span></td>'
    f'<td>{html.escape(d)}</td><td>{html.escape(e)}</td></tr>'
    for a, b, c, d, e in decisions)

# ---------- HTML ----------
CSS = """
:root{--bg:#f5f6f8;--card:#fff;--ink:#15202b;--muted:#667085;--line:#e4e7ec;
--green:#0f766e;--green-bg:#e6f4f1;--red:#b42318;--red-bg:#fdecea;--warn:#b54708;
--warn-bg:#fff4e5;--blue:#1d4ed8;--navy:#101828;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;line-height:1.65}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 64px}
.hero{background:linear-gradient(135deg,#0b1220 0%,#16263d 55%,#1f3a52 100%);color:#fff;padding:56px 0 42px;margin-bottom:28px}
.hero .wrap{padding-bottom:0}
.kicker{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#9fb3c8;margin-bottom:12px}
h1{font-size:38px;line-height:1.2;margin:0 0 14px;letter-spacing:-.02em}
.sub{font-size:16px;color:#c9d6e3;max-width:900px;margin:0}
.hero-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:30px}
.kpi{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:700;letter-spacing:-.02em}
.kpi .l{font-size:12px;color:#a9bacb;margin-top:4px}
.kpi .v.bad{color:#ff9b8f}.kpi .v.good{color:#7fe0c3}
nav{position:sticky;top:0;z-index:10;background:rgba(245,246,248,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);margin-bottom:26px}
nav .wrap{display:flex;gap:18px;overflow:auto;padding-top:12px;padding-bottom:12px}
nav a{color:var(--muted);text-decoration:none;font-size:13px;white-space:nowrap}
nav a:hover{color:var(--navy)}
section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:28px 30px;margin:20px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
h2{font-size:25px;margin:0 0 8px;letter-spacing:-.01em}
h3{font-size:17px;margin:26px 0 8px}
p{margin:10px 0;color:#344054}
.lead{font-size:16px;color:#475467}
.tag{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px;margin-right:6px}
.tag.good{background:var(--green-bg);color:var(--green)}.tag.bad{background:var(--red-bg);color:var(--red)}.tag.warn{background:var(--warn-bg);color:var(--warn)}
.badge{display:inline-block;font-size:12px;font-weight:700;padding:3px 9px;border-radius:6px;white-space:nowrap}
.badge.good{background:var(--green-bg);color:var(--green)}.badge.bad{background:var(--red-bg);color:var(--red)}.badge.warn{background:var(--warn-bg);color:var(--warn)}
.callout{border-left:4px solid var(--blue);background:#f4f7ff;padding:14px 18px;border-radius:0 10px 10px 0;margin:18px 0;color:#20345c}
.callout.bad{border-left-color:var(--red);background:#fff6f5;color:#5b1d16}
.callout.good{border-left-color:var(--green);background:#f0faf7;color:#134e4a}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.table-wrap{overflow:auto;margin:16px 0;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
table.dec td,table.dec th{white-space:normal;vertical-align:top;line-height:1.45}
table.wrap td,table.wrap th{white-space:normal;vertical-align:top;font-size:12px;line-height:1.4}
table.wrap td:first-child,table.wrap th:first-child{min-width:64px}
table.dec td:nth-child(1){min-width:185px}
table.dec td:nth-child(3){min-width:260px}
table.dec td:nth-child(4){min-width:190px}
th{background:#f8fafc;color:#475467;font-weight:600;text-align:right;padding:10px 11px;border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0}
td{padding:9px 11px;border-bottom:1px solid #f0f2f5;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:120px}
tbody tr:hover{background:#fafbfc}
td.pos{color:var(--green);font-weight:600}td.neg{color:var(--red);font-weight:600}td.warn{color:var(--warn);font-weight:600}
figure{margin:20px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:12px;background:#fff}
figcaption{font-size:12px;color:var(--muted);margin-top:8px}
ul,ol{margin:10px 0 10px 22px;padding:0}
li{margin:6px 0;color:#344054}
.small{font-size:12px;color:var(--muted)}
.footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);margin-top:30px;padding-top:18px}
@media(max-width:900px){.hero-stats{grid-template-columns:repeat(2,1fr)}.grid2,.grid3{grid-template-columns:1fr}h1{font-size:30px}}
@media print{nav{display:none}section{break-inside:avoid;box-shadow:none}body{background:#fff}}
"""

head = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MACD 在 A 股的有效性（2010–2026）</title>
<style>{CSS}</style></head><body>
<header class="hero"><div class="wrap">
<div class="kicker">A-Share Signal Research · 2026-10-28</div>
<h1>MACD 在 A 股：什么时候有效，什么时候无效</h1>
<p class="sub">53.2 万个日线金叉、12 个连续特征、7 个结构维度、413 组参数、18 个条件、5 个宽基指数。结论不是“MACD 有用/没用”，而是：<b>个股日线越像买入信号越差，唯一稳健的增量是“红柱放大=卖/避”；月线只在长期下跌结构里做相对反弹；指数层面是危机保险。</b></p>
<div class="hero-stats">
<div class="kpi"><div class="v">53.2 万</div><div class="l">2010–2026 日线金叉</div></div>
<div class="kpi"><div class="v bad">49.79%</div><div class="l">20 日横截面平均分位</div></div>
<div class="kpi"><div class="v bad">-5.7%/年</div><div class="l">扣 0.15% 成本的 20 日组合</div></div>
<div class="kpi"><div class="v bad">0/12</div><div class="l">日线连续特征正 IC 数量</div></div>
<div class="kpi"><div class="v good">-38% vs -72%</div><div class="l">中证 500 回撤：MACD vs 持有</div></div>
</div></div></header>
<nav><div class="wrap">
<a href="#conclusion">结论</a><a href="#stock">个股全样本</a><a href="#conditions">条件</a><a href="#pockets">交互口袋</a>
<a href="#params">参数</a><a href="#variants">变体</a><a href="#features">特征与结构</a><a href="#exit">离场</a>
<a href="#index">指数择时</a><a href="#timeframe">周期</a><a href="#decision">判断表</a><a href="#method">方法</a>
</div></nav>
<main class="wrap">
"""

fs_show = tfeat[tfeat.特征.isin([
    "柱体水平", "柱体3日变化", "柱体加速度", "20日柱体均值", "DIF水平",
    "DIF5日斜率", "交叉后天数", "20日红柱占比", "基准_20日涨幅", "基准_距MA20"])].copy()
struct_show = tstruct.copy()
strict_show = tstrict.copy()
state_show = tstate[
    ((tstate.状态 == "红柱放大") & tstate.分组.isin(["高换手", "高波", "高位", "多头排列", "低波", "空头排列"]))
    | ((tstate.状态 == "DIF>DEA") & (tstate.分组 == "空头排列"))].copy()
month_show = tmonth.copy()
monthfm_show = tmonthfm.copy()
trenddef_show = ttrend[ttrend.周期 == 12][["定义", "平均IC", "IC_t12", "Q5_Q1月均"]].copy()
dist_show = tmonthdist[tmonthdist.特征 == "柱体水平"].copy()
port_show = tport[(tport.区间 == "2010-2026") & (tport.策略.isin([
    "全A等权", "剔除红柱放大", "剔除柱体上行", "剔除DIF>DEA", "只买DIF最低20%",
    "只买20日涨幅最低20%", "只买低波20%", "空头排列+绿柱缩短"]))].copy()
port_show["策略"] = pd.Categorical(port_show["策略"], categories=[
    "全A等权", "剔除红柱放大", "剔除柱体上行", "剔除DIF>DEA", "只买DIF最低20%",
    "只买20日涨幅最低20%", "只买低波20%", "空头排列+绿柱缩短"], ordered=True)
port_show = port_show.sort_values("策略")
perf_show = tperf.copy()

body = f"""
<section id="conclusion">
<h2>先看结论</h2>
<div class="callout bad"><b>个股日线金叉不是选股信号。</b> 20 日平均分位 49.79%（50% = 与市场同步），平均超额 -0.054%（聚类 t=-1.24），中位超额 -1.45%；扣 0.15% 成本后的 20 日持有组合年化 <b>-5.66%（t=-7.96）</b>。它是一枚右偏彩票：均值被少数大赢家拉起，典型交易在中位以下。</div>
<div class="grid3">
<div class="callout good"><b>日线金叉单维唯一稳健正 pocket：ST</b><br>分位 53.19%，+0.97%/20 日，扣费组合 +1.3%/年；但 t=0.32，容量/制度限制大。</div>
<div class="callout warn"><b>最好的参数也不够付手续费</b><br>413 组全部 49.77%–50.25%；最好毛超额 +0.115%/20 日 < 0.15% 成本；FDR q=0.53。</div>
<div class="callout good"><b>指数层面：危机保险</b><br>中证 500 回撤 -38% vs -72%；但 2016 后收益增益基本消失，只剩降回撤。</div>
</div>
<div class="callout bad"><b>把对象换成连续特征，结论更尖锐。</b> 12 个日线 MACD 特征的全样本 20 日 IC <b>全部为负</b>：DIF 水平 -7.16%（t=-8.42），柱体水平 -2.80%。控制 MACD 水平、1/5/20/60 日涨幅、波动、换手、成交额后，唯一稳健的独立增量是负向的：<b>柱体上行/红柱放大 -0.15% ~ -0.18%/20 日（t=-5 ~ -6）</b>；DIF>DEA -0.23%（t=-2.98）；绿柱缩短约 0。周线全负；<b>月线在长期下跌结构里出现唯一强正</b>（12 个月 IC +11.8%，Q5-Q1 +10.1%），但那是跌深反弹的相对信号，绝对收益仍跑输宽基。</div>
<h3>教科书条件正是最差条件</h3>
<p>零轴双上 47.96%、周线多头 48.30%、年线上+年线升 48.40%、高波动 46.94%、高成交额 46.61%、量比&gt;2 46.60%、上市&lt;1 年 46.29%、DIF&gt;1.5 46.12%。在 A 股，MACD 金叉是<b>反趋势</b>信号，不是趋势信号；用“零轴上、周线多头、放量”去确认，等于主动挑最差的条件。特征层同样：中期多头排列 IC -3.79%、高波 -4.62%、高位 -4.51%、高换手 -4.41%，是全表最差的四个结构。</p>
</section>

<section id="stock">
<h2>个股全样本：一个抛硬币的信号</h2>
<p class="lead">53.2 万个金叉，20 日平均分位 49.79%，中位 49.81%，胜率 49.81%。按信号加权平均超额 -0.054%（t=-1.24）；按日加权 -0.745%（t=-14.6）。差别的来源是信号聚集：金叉多的日子随后反弹更强，金叉少的日子更弱。两种口径都真实，但都不能给出正 edge。</p>
{table(overall, ["周期","信号数","平均超额","中位超额","胜率_超额大于0","日度平均超额","日度t","聚类t"])}
<p class="small">注：“胜率_超额大于0”是相对等权市场<b>均值</b>的胜率。市场收益右偏，均值高于中位，所以即使股票处在中位也会低于均值；公平的“抛硬币”胜率看分位&gt;50% = <b>49.81%</b>。</p>
<div class="callout"><b>为什么中位 -1.45%、均值却接近 0？</b> 横截面收益右偏：P10=-11.7%，P90=+12.5%，偏度 2.88。市场均值被少数大牛股拉高，处于中位的股票自然低于均值。用横截面分位复核，金叉股就在 49.8%——既没有选到更好的，也没有选到更差的。</div>
{img(os.path.join(FIG,"图6_收益分布.png"), "左：20 日超额分布，均值被右尾拉起；右：按超额分十组，最差组仍有约 7% 的 MFE。")}
<h3>彩票型：MFE 12%，中位 -1.45%</h3>
<p>MFE20 均值 <b>+12.0%</b>，MAE20 均值 <b>-8.6%</b>，中位超额 <b>-1.45%</b>。金叉之后大多数股票先给 12% 的浮盈机会，但收盘时中位只剩负超额。<b>方向不是问题，兑现才是问题。</b> 这解释了为什么“金叉买、死叉卖”直觉很顺，数据上却不如固定持有——死叉来得太晚，浮盈已经回吐。</p>
</section>

<section id="conditions">
<h2>条件：什么时候有效，什么时候无效</h2>
<p class="lead">把 18 个信号日可观测维度全部切开。正 pocket 几乎都是“信号聚集”的产物；只有 ST 同时通过“信号加权正 + 日期加权正 + 前后段都 &gt;50%”三道检验。</p>
{img(os.path.join(FIG,"图1_条件分位.png"), "20 日横截面平均分位：越靠右越好，50% 为与市场同步。")}
{table(keycond, ["条件","信号数","平均分位20","平均超额20","日度平均超额20","日度t","聚类t","分位_2010_2017","分位_2018_2026"])}
<h3>三条硬结论</h3>
<p><span class="tag good">1</span><b>正 pocket 大多是聚类幻觉。</b> 低波动分位 52.01%，但日度平均超额 -0.30%（t=-3.97）；零轴下 50.83%，日度 -0.72%（t=-10.41）；周线空头 50.90%，日度 -0.64%（t=-8.50）；年线下 50.97%，日度 -0.77%（t=-11.87）。它们“每个信号平均”略好，但“平均一天”是亏的。</p>
<p><span class="tag bad">2</span><b>教科书条件是最差条件。</b> 零轴双上、周线多头、年线上+年线升、高波动、高成交额、量比&gt;2、上市&lt;1 年、DIF&gt;1.5——全部在 46%–48% 分位。这不是“MACD 没用”，而是“用法用反了”：它是滞后突破信号，A 股 1–3 个月尺度反转占优。</p>
<p><span class="tag good">3</span><b>只有 ST 稳健。</b> 分位 53.19%，日度 +0.41%（t=2.58），2010-2017/2018-2026 都在 52.6% 以上。但 ST 有 5% 涨跌停、退市/重组博弈、流动性差、容量小；它不是 MACD 的功劳，不能当策略。</p>
</section>

<section id="pockets">
<h2>交互格：一个窄口袋（探索性）</h2>
<p class="lead">把 8 个维度交叉，得到 109 个 n≥1,000 的格子。绝大多数仍是负的；但有 4 个格子同时满足：平均超额 &gt; +1.3%、日度超额 &gt; 0（t&gt;2.5）、2010-2017 与 2018-2026 都 &gt; 0、BH q&lt;0.01。共同点是：<b>低成交额 + 洗盘日 + 低/中波 + 一组特定的趋势/零轴/周线状态</b>。这不是趋势跟随，而是“强/弱股在弱市洗盘日的反转”；MACD 金叉只是触发器。A 每年约 128 个信号，B/C/D 约 66–122 个。</p>
{table(pockets, ["格子","信号数","平均分位20","平均超额20","中位超额20","日度t","聚类t","超额_2010_2017","超额_2018_2026","日均成交额万元","前5笔占盈亏"])}
<div class="callout warn"><b>四盆冷水：</b>① 中位很薄——B/D 中位为负，A/C 只有 +0.15%/+0.24%，刚够 0.15% 成本；② 组合层不显著——A 在 0.15% 成本下年化 +13.7%（t=1.61），0.5% 成本 +5.2%（t=0.62），1.0% 成本转负；B/C/D 在 0.15% 成本下只有 +1.4%/+3.6%/+2.9%（t&lt;0.5）；③ 容量小——日均成交额 2,300–3,000 万，单笔 100 万就占 3%–4% 日成交；④ 109 个格子是看完单维结果后扫出来的，虽 BH q&lt;0.01，但不是预注册，也没有独立 OOS。</div>
<div class="callout"><b>结论：</b>A 是筛出的 4 个格子里唯一一个“两段都正、日度也正、组合毛 alpha 两位数”的个股口袋，值得作为预注册 OOS 的研究线索；但中位薄、容量小、成本敏感、无独立 OOS，不能直接实盘，也不改变“MACD 个股选股整体无效”的判断。</div>
</section>

<section id="params">
<h2>参数：413 组，没有一组能救</h2>
<p class="lead">fast 5–25、slow 15–60、signal 3–20，共 413 组（354 组可评估；signal=1 退化）。可评估组合的平均分位落在 49.77%–50.25%，没有一组超过 50.5%；没有一组中位超额为正或胜率超过 50%；FDR 最小 q=0.527。12/26/9 是全场最差约 1%（413 组中排第 5 差）。</p>
<div class="grid2">{img(os.path.join(FIG,"图2_参数热图.png"), "快慢线参数热图：signal=9 与 signal=20。")}{img(os.path.join(FIG,"图3_参数分布.png"), "413 组参数的分位分布：全部挤在 50% 附近。")}</div>
{table(pdig, ["项目","数值"])}
<h3>最好的 8 组（仍然是噪声）</h3>
{table(top, list(top.columns))}
<h3>默认参数 12/26/9</h3>
{table(default, list(default.columns))}
<div class="callout warn"><b>为什么参数救不了？</b> 改变 fast/slow/signal 只改变“滞后多少”，不改变“金叉发生在上涨之后”这一事实。A 股的 1–3 个月反转把“上涨之后”变成“均值回归之前”。慢参数只是更少交易、更接近买入持有，所以“最不坏”；快参数被成本和反转双重磨损，所以更差。参数空间里没有“趋势市开关”，只有“交易频率开关”。</div>
</section>

<section id="variants">
<h2>变体：零轴、绿柱、周线、背离</h2>
<p class="lead">14 个变体，统一口径。周线共振是负的；底背离没有 edge；绿柱缩短、零轴下、周线空头三者跨期都 >50，但扣费后都不可交易。</p>
{table(v, ["变体","信号数","平均分位20","分位>50","平均超额20","聚类t","2010-17 分位","2018-26 分位","扣费组合年化","扣费组合 t"])}
<div class="callout"><b>绿柱缩短：原始分位略正，独立增量约 0。</b> 分位 50.62%，早期 50.73%、晚期 50.56%，t=2.91；但控制柱体水平、1/5/20/60 日涨幅、波动、换手、成交额后，状态系数 <b>-0.038%（t=-1.20）</b>。它的正分位来自“绿柱缩短时水平通常更低/过去跌幅更大”的组成效应，不是独立信号；原始 0.134% 的毛 edge 也低于 0.15% 成本。</div>
</section>

<section id="features">
<h2>特征与结构：不是只有金叉</h2>
<p class="lead">把对象从“金叉事件”换成“每日状态”：12 个连续特征、8 个二元状态、7 个结构维度。结论不是“无效”，而是“方向反了”——日线 12/12 特征 IC 为负；唯一稳健的独立增量是红柱放大/柱体上行的负向；周线全负；月线只在长期下跌结构里强正，但只是相对反弹。</p>
<p class="small">背离是事件特征，已在“变体”节测试：金叉+底背离 49.95%，与非背离 49.75% 几乎一样；本节不重复。</p>
<div class="callout bad"><b>先给一句话：</b>日线 MACD 越像买入信号，未来 20 天越差；它不是趋势确认指标，而是极值反转指标。唯一可用的方向是“卖/避”，不是“买”。</div>
<h3>1. 全样本：12 个特征，IC 全部为负</h3>
<p>12 个连续特征（柱体、变化、加速度、面积、z、DIF/DEA、斜率、交叉年龄、红柱占比）在全 A、非 ST、上市≥120 日上，20 日超额 IC 全部为负。DIF 水平 <b>-7.16%（t=-8.42）</b>，柱体水平 -2.80%，20 日柱体均值 -3.93%；连柱体加速度也只有 -0.06%（t=-0.55）。对照组 20 日涨幅 IC -6.73%，距 MA20 -5.82%——MACD 没有比价格本身多出方向信息。</p>
{table(fs_show, ["特征","平均IC","IC_t","多空5_1","多空t","Q1","Q5","IC_2010_2017","IC_2018_2026","多空大于0年份","年份数"])}
{img(os.path.join(FIG,"图9_特征IC.png"), "12 个 MACD 连续特征的 20 日 IC：全部为负；DIF 水平最负，柱体加速度接近 0。")}
<div class="callout"><b>怎么读：</b>IC 为负不是“每只高 MACD 股票都跌”，而是“在同一交易日，MACD 更高的那一组，未来 20 天相对收益更低”。Q1（最低 MACD）平均超额 +0.37%，Q5（最高）+ -0.93%，Q5-Q1 = -1.30%/20 日。MACD 是平滑后的价格，仍然指向短期反转。</div>
<h3>2. 结构条件：只有中期空头排列弱正</h3>
<p>21 个结构桶里 20 个为负。唯一为正的是<b>中期空头排列 +0.82%（t=1.39，q=0.16）</b>，不显著；两段 +0.97%/+0.71%，方向一致但强度不够。最差的是多头排列 -3.79%（t=-8.09）、高波 -4.62%（t=-8.75）、高位 -4.51%（t=-7.17）、高换手 -4.41%（t=-8.30）。把 10 个特征 × 21 个桶全部算上，210 个格子里只有 6 个正 IC，全部在中期空头排列，且没有一个 t&gt;1.5。教科书里的“均线多头、放量、创新高、波动大”，正是 MACD 选股最差的结构。</p>
{table(struct_show, ["维度","分组","平均IC","IC_t","多空5_1","多空t","q_bh"], class_="data wrap")}
{img(os.path.join(FIG,"图10_结构热图.png"), "柱体水平 IC（20 日，%）：只有中期空头排列弱正；多头排列/高波/高位/高换手最差。")}
<h3>3. 严格增量：红柱放大 = 负</h3>
<p>每个交易日做横截面回归：<code>ex20 ~ 1 + MACD水平 + ret1 + ret5 + ret20 + ret60 + 波动 + 换手 + log成交额 + 状态 + 状态×水平</code>。状态系数是“在同等水平、动量、波动、换手、成交额下，状态 1 相对 0 的 20 日超额差”。</p>
{table(strict_show, ["检验","项","系数","t","系数_2010_2017","系数_2018_2026","正系数年份","年份数"], class_="data wrap")}
<div class="callout bad"><b>唯一稳健的独立增量是负向的。</b> 柱体上行 -0.152%/20 日（t=-5.08），红柱放大 -0.179%（t=-5.89），2/19、1/19 年为正；水平越高越差，高水平红柱放大约 -0.45%/20 日。<b>DIF>DEA 的严格增量也是 -0.226%（t=-2.98）</b>，不是买入信号（这是“当前状态”，不是“金叉当天”）；绿柱缩短 -0.038%（t=-1.20），约等于 0。柱体上穿 0 的平均正系数不可信：它随水平剧烈变化（交互 -4.65，t=-6.92），只有深度负值上穿时有一点反弹。</div>
<h3>4. 结构状态：红柱放大在哪里最差</h3>
<p>红柱放大在 20/21 个结构桶里都是负的；唯一翻正的是中期空头排列 +0.132%（t=1.42，q=0.16），不显著。高换手/高波/高位/多头排列最差；在这些结构里，红柱放大后的 20 日<b>绝对收益接近 0 或为负</b>（高波 -0.34%，高换手 -0.37%，市场 +0.08%），不是“少赚”，是“不赚”。</p>
{table(state_show, ["维度","分组","状态","控制后差","控制t","q_bh","状态1绝对","状态0绝对","市场绝对"], class_="data wrap")}
<div class="callout"><b>DIF>DEA 的结构表更弱：</b>只有空头排列 +0.225%（t=2.71，q=0.14），其余 0 或负；金叉状态没有结构上的稳定正效应。</div>
<h3>5. 周期：周线更负，月线唯一强正</h3>
<p>周线 4/12 周柱体水平 IC -5.99%/-5.65%，DIF -4.26%/-4.28%，长期上行/下行/过渡三个桶全负；<b>周线没有任何翻正的结构</b>。月线全样本也不显著：柱体水平 3 个月 -1.58%（t=-1.14）、12 个月 -1.32%（t=-0.79）；DIF 12 个月 -12.18%（t=-5.22）。但把月线按长期趋势拆开，出现强符号翻转：12 个月柱体水平，上行 -7.63%、下行 <b>+11.76%（t=5.15，lag=12）</b>、过渡 +5.43%；3 个月上行 -5.96%、下行 +7.73%、过渡 +4.60%。</p>
{table(perf_show, ["周期","持有期","特征","平均IC","IC_t","多空5_1","多空t","IC_2010_2017","IC_2018_2026"], class_="data wrap")}
<p class="small">月线结构表 t 用 lag=6；下面“月度下行稳健表”用 lag=12（更保守）。同一 IC 的 t 会略低。</p>
<table class="data wrap"><thead><tr><th>样本（月线）</th><th>3m IC (t)</th><th>3m Q5-Q1</th><th>12m IC (t)</th><th>12m Q5-Q1</th></tr></thead><tbody>
<tr><td>下行·全</td><td class="pos">+7.73% (4.93)</td><td class="pos">+2.70%</td><td class="pos">+11.76% (5.15)</td><td class="pos">+10.61%</td></tr>
<tr><td>下行·非ST·高价·高流动</td><td class="pos">+7.43% (4.18)</td><td class="pos">+2.38%</td><td class="pos">+12.68% (4.96)</td><td class="pos">+10.09%</td></tr>
<tr><td>下行·非ST·高价·低流动</td><td class="pos">+6.02% (4.79)</td><td class="pos">+2.05%</td><td class="pos">+7.85% (5.20)</td><td class="pos">+7.00%</td></tr>
<tr><td>上行·非ST·高价·高流动</td><td class="neg">-6.70% (-4.98)</td><td class="neg">-1.08%</td><td class="neg">-8.50% (-4.40)</td><td class="neg">-4.59%</td></tr>
</tbody></table>
<p class="small">注：本表 Q5/Q1 为各月等权平均（月均）；下面的分布表为股票-月加权。两者略有差异（如高流动 12m Q5 绝对：月均 +10.04%，股票-月 +10.46%），方向一致。Q5 相对宽基的超额见分布表。</p>
<p>换趋势定义也成立：三种“下行”定义（价<MA200、斜率<0、两者）的 12m IC 全为正，三种“上行”定义全为负；不是某一种 MA200 定义的偶然。</p>
{table(trenddef_show, ["定义","平均IC","IC_t12","Q5_Q1月均"], class_="data wrap")}
{img(os.path.join(FIG,"图11_月度下行.png"), "左：月线柱体水平 IC，下跌趋势正、上涨趋势负；右：高流动下跌样本的 Q1/Q5 均值与中位。")}
<p>这是全研究<b>唯一一个“某种趋势/结构下 MACD 有效”的强证据</b>：月线、长期下跌趋势、柱体水平/金叉作为跌深反弹的相对选择。它的含义与日线相反——在长期下跌里，高月线柱体不是“追高”，而是“跌深反弹的强度”；在长期上涨里，同样的高柱体是“过热”。</p>
{table(dist_show, ["周期","分位","样本数","均值绝对","中位绝对","胜率绝对","P10","P90","均值超额","中位超额"], class_="data wrap")}
<div class="callout warn"><b>但它是相对的，不是 long-only。</b> 高流动下跌样本里，Q5 的 3 个月中位 <b>-1.29%</b>、胜率 46.2%；12 个月中位 +1.30%、胜率 51.8%。Q5 的 12 个月绝对 +10.46%（股票-月加权），同期宽基约 +11.9%，<b>Q5 超额 -1.41%（中位 -7.89%）</b>；整个下跌池 12m 绝对 +4.87%，跑输市场。低流动子样本更强（3m/12m IC +6.0%/+7.8%，Q5 绝对 +3.48%/+17.29%），但那是小票/流动性效应。它是右偏的反弹彩票，不是稳定赚钱的 long-only。</div>
{table(month_show, ["周期","样本","月数","平均IC","IC_t12","IC_2010_2017","IC_2018_2026","Q1绝对","Q5绝对"], class_="data wrap")}
<p>Fama-MacBeth 控制 1/3/12 月动量、波动、换手、成交额后：柱体水平连续项 <b>+2.47%/3 个月（t=2.44）、+11.41%/12 个月（t=3.51）</b>；DIF>DEA 状态 +1.12%/3 个月（t=2.27）、+3.00%/12 个月（t=1.82）。所以相对信号不是动量/波动/流动性的复写；但绝对收益不够。（表中“柱体水平连续”的系数是水平从最低到最高的额外差，不是每 1% 的斜率。）</p>
{table(monthfm_show, ["周期","检验","项","系数","t","系数_2010_2017","系数_2018_2026","正系数年份"], class_="data wrap")}
<h3>6. 组合层：负信号不能变成 long-only 收益</h3>
<p>把特征做成 20 日非重叠等权组合：每 20 个交易日按 T 日信号选股，T+1 开盘买入，持有 20 日，扣 0.15% 往返成本；回撤为 20 日采样，用于横向比较，不是精确日频 MDD。</p>
{table(port_show, ["策略","年化","回撤","夏普","胜率","平均持仓"], class_="data wrap")}
{img(os.path.join(FIG,"图12_特征组合.png"), "20 日非重叠组合：剔除红柱放大与全 A 持平；剔除柱体上行/DIF>DEA 反而大幅跑输；低 DIF 与低 20 日涨幅几乎一样。")}
<div class="callout bad"><b>组合层是最终裁决：</b>剔除红柱放大 6.08% vs 全 A 6.09%，不增厚；剔除柱体上行 2.62%、剔除 DIF>DEA 2.90%，反而把上行暴露最高的强势股剔除了。只买 DIF 最低 20% 的 9.38% 看似好，但只买 20 日涨幅最低 20% 是 9.08%，60 日最低 8.36%，低波 7.33%——这是短期反转/低波，不是 MACD 的独立 alpha。负 IC 主要是相对/风格效应（很可能是 beta），long-only 不能用它做“卖出过滤器”来增厚收益。</div>
<div class="callout"><b>一张表：</b>日线 12/12 特征 IC&lt;0；唯一稳健增量是红柱放大/柱体上行的负向；DIF&gt;DEA 严格为负；绿柱缩短约 0；周线全负；月线只在长期下跌结构里强正，但只是相对反弹。<b>MACD 是“极值反转指标”，不是“趋势确认指标”。</b></div>
</section>

<section id="exit">
<h2>离场：死叉不是好出口</h2>
<p class="lead">53.2 万个金叉，比较“下一个死叉次日开盘卖”和固定持有。死叉平均持有 14.2 天，平均收益 +1.00%，中位 -2.13%，胜率 35.1%；固定 20 日平均 +1.32%，中位 0.00%，胜率 49.5%。死叉把浮亏变成实亏，不是风控。</p>
{table(exits, ["离场","交易数","平均收益","中位收益","胜率","平均持有天数","死叉覆盖率"])}
<div class="callout bad"><b>死叉是滞后确认。</b> 等它出现，价格已经跌了一段，等于把亏损锁在反弹之前。工作区此前的池 50/20 龙头研究也一致：死叉平均亏 -2.9% 到 -5.5%，b3 追踪兑现右尾才是正贡献。MACD 的出场不如入场，死叉尤其差。</div>
</section>

<section id="index">
<h2>指数择时：危机保险，不是收益发动机</h2>
<p class="lead">2006-2026，T+1 开盘执行，单边 0.05%。所有指数、所有规则的最大回撤都低于买入持有（10–50 个百分点）；但 2016 后上证指数单条件反而更深。收益分化，且 2016 年后大幅衰减。</p>
{img(os.path.join(FIG,"图4_指数择时.png"), "沪深300/中证500/中证1000：MACD 金叉择时 vs 买入持有；右列为回撤。")}
{table(idx_p, ["指数","规则","全样本年化","全样本回撤","全样本夏普","2016后年化","2016后回撤","2016后夏普"])}
<h3>等权指数：同样只降回撤</h3>
<p>等权指数用 T+1 收盘成交（信号滞后 2 日），比指数开盘口径更保守。2018-2026：DIF&gt;DEA 年化 4.34% vs 买入持有 5.96%，但回撤 -33.6% vs -37.9%；“双条件”回撤只有 -20.6%。<b>收益不增，回撤大降。</b></p>
{table(ew18, ["指数","规则","区间","年化","最大回撤","夏普","仓位","换手"])}
<div class="callout good"><b>准确说法：危机保险。</b> 平时付保费（牛市跑输、震荡磨损），危机时赔付（主要宽基回撤减半）。高波动、年线上行时相对有效；低波动、震荡市里被反复打脸。日度 t 都不到 0.3，收益来自少数几次大崩盘，不是稳定日度 alpha。</div>
</section>

<section id="timeframe">
<h2>其他周期：周线最差，月线在沪深300/500 有效</h2>
<p class="lead">上一版只做了日线，这是范围缺口，已补。本地库只有日线；周线/月线由日线按 W-FRI / 自然月合成，周五/月末收盘确认，下周一/下月首个交易日开盘。**没有 5/15/30/60 分钟数据，分钟级未做**。</p>
{img(os.path.join(FIG,"图7_周期对比.png"), "左：个股日/周/月平均分位；中：指数 2016+ 年化；右：回撤。")}
<h3>个股：周线更弱，月线 12 个月只是弱正</h3>
{table(pstock, ["周期","持有期","信号数","平均分位","平均超额","中位超额","聚类t","分位_2010_2017","分位_2018_2026","结论"], class_="data wrap")}
<p>413 组参数/周期：周线 4 周最高 50.31%，周线 26 周最高 50.47%（中位 -4.93%），月线 3 月最高 49.63%，月线 12 月最高 51.14%（228 组 &gt;50.5%，但 <b>中位&gt;0 的 0 组</b>）。<b>没有任何周期给出“中位为正”的参数。</b></p>
{table(pbest, ["周期","持有期","最优参数","最优平均分位","最优平均超额","最优聚类t","最优中位超额","分位大于50_5组合数","中位大于0组合数","结论"], class_="data wrap")}
<div class="callout bad"><b>个股结论：</b>日线/周线/月线 3 个月都无 edge；月线 12 个月 50.73%（+0.84%，t=1.27，中位 -8.37%）只是“长期彩票”，前段 50.77%/后段 50.72%，但中位大幅为负；不能单独用。</div>
<h3>指数：月线是 2016 后的最佳周期</h3>
{table(cidx16, ["指数","周期","规则","年化","最大回撤","夏普","换手"])}
{img(os.path.join(FIG,"图8_月线择时.png"), "月线 MACD vs 买入持有：净值与回撤。")}
<div class="callout good"><b>指数结论：</b>2016+ 月线 沪深300/中证500/中证1000 年化 4.59%/5.13%/0.16%，回撤 -28.4%/-19.9%/-31.6%，持有 -46.7%/-42.9%/-57.1%；周线 2016+ 三指数全负（-1.9%/-4.5%/-7.7%）。月线 1296 组参数中 570 组（44%）2016+ 年化为正，中位 -0.2%，中位回撤 -34.0%；12/26/9 在三个指数的年化分位 97.5%/95.8%/82.9%。<b>月线是慢速崩盘过滤器，不是 alpha</b>：2016 年 CSI1000 -30.6% vs 持有 -19.8%，2019 年 CSI500 +3.5% vs +27.1%，2024 年 -3.0% vs +5.3%——避开 2018/2022，错过牛市初期。</div>
<h3>多周期共振：三重多头是最差</h3>
{table(m2, ["层级","组合","持有期","信号数","平均分位","平均超额","聚类t","2010-17 分位","2018-26 分位","结论"], class_="data wrap")}
<div class="callout bad"><b>共振结论：</b>日线+周线多头+月线多头 47.96%，t=-3.92，是最差条件；周线+月线多头 4 周 50.03%（早期 50.54%、晚期 49.84%），12 周 49.83%（早期 50.84%、晚期 49.45%），后期跌破 50%；月线+周线多头 50.68%，不改善。唯一像样的是反趋势的“周线空头+月线多头”50.75%（t=2.79），但仍是弱信号。<b>多周期确认不改善，反而把最差条件叠加。</b></div>
</section>

<section id="decision">
<h2>最终判断：一张可执行的表</h2>
<div class="table-wrap"><table class="dec"><thead><tr><th>场景</th><th>判断</th><th>关键证据</th><th>建议</th></tr></thead><tbody>{dec_html}</tbody></table></div>
<div class="callout bad"><b>一句话：</b>在 A 股，MACD 不是“趋势确认指标”，而是“极值反转指标”：日线越像买入信号，未来 20 天越差；唯一稳健的独立增量是“红柱放大 = 卖/避”。月线在长期下跌结构里有强相对反弹信号（12m IC +11.8%，Q5-Q1 +10.1%），但下跌池本身跑输宽基，不能做 long-only；指数层面的月线择时仍是危机保险，不是收益发动机。</div>
</section>

<section id="method">
<h2>方法与局限</h2>
<div class="grid2">
<div>
<h3>数据与执行</h3>
<ul>
<li>2008-2026 全 A 日线，5,507 只，含 341 只退市；信号评估 2010 起。</li>
<li>baostock 不复权 + 后复权；DuckDB 2016 后重叠 1,011 万行，OHLC 仅 2 行差异。</li>
<li>T 日收盘确认，T+1 开盘买入；剔除一字涨停/停牌；个股往返 0.15%，指数单边 0.05%。</li>
<li>主指标：20 日收益的当日横截面分位；辅以信号加权/日期加权/聚类 t/组合净值。</li>
<li>413 组参数用 BH-FDR；条件分层是描述性/样本内，正 pocket 要求三道检验。</li>
</ul>
</div>
<div>
<h3>局限</h3>
<ul>
<li>日线、无盘中；T+1 开盘理想化；0.15% 成本对小盘/ST 可能偏低。</li>
<li>2016-2026 是样本内分段，不是独立 OOS；2027 后需盲测。</li>
<li>指数择时未计 ETF 跟踪误差/管理费；ST 容量/制度限制大。</li>
<li>行业/市值用成交额和价格代理；DuckDB 市值缺失 79%，未做精确市值分层。</li>
<li>周线用上周已完成周线，比当周部分周线保守；底背离是 60/60 日双底代理。</li>
</ul>
</div>
</div>
<p class="small">完整方法见 <b>01_方法与口径.md</b>；逐笔事件 532,219 行见 <b>03_结果/事件_金叉_12269.parquet</b>；数据审计见 <b>03_结果/00_数据审计.csv</b>。</p>
</section>

<div class="footer">
<p>研究目录：2026-10-28_MACD有效性_全维度 · 数据截至 2026-10-28 · 生成日期 2026-10-28</p>
<p>本报告是统计研究，不构成投资建议。所有历史结果都不保证未来；参数、样本、成本假设变化会改变结论。</p>
</div>
</main></body></html>
"""

with open(os.path.join(ROOT, "MACD有效性_可视化.html"), "w", encoding="utf-8") as f:
    f.write(head + body)

# ---------- Excel ----------
xls = os.path.join(ROOT, "MACD有效性_数据表.xlsx")
with pd.ExcelWriter(xls, engine="openpyxl") as w:
    # 一页纸
    summary = pd.DataFrame([
        ("研究问题", "MACD 在什么情况下有效、什么时候无效？参数变化能否带来意外收获？"),
        ("全样本金叉", "532,219 个（2010-2026）"),
        ("20 日平均分位", "49.79%"),
        ("20 日平均/中位超额", "-0.054% / -1.454%"),
        ("扣费组合年化", "-5.66%（t=-7.96）"),
        ("413 组中可评估 354 组的分位范围", "49.77%–50.25%，无一组 >50.5%"),
        ("413 组 FDR 最小 q", "0.527，无一组显著"),
        ("12/26/9 位置", "最差约 1%（413 组中排第 5 差）"),
        ("日线金叉单维唯一稳健正 pocket", "ST：53.19%，+0.97%/20 日，组合 +1.3%/年，t=0.32"),
        ("交互格窄口袋", "弱市强股+低波低额+洗盘日：+1.66%/20 日，两段 +1.53/+1.83；中位 +0.15%，组合 t=1.61，日额 2,666 万"),
        ("中证 500 指数择时", "日线 14.0% vs 10.7%，回撤 -38% vs -72%；月线 2016+ 5.1% vs -1.4%，回撤 -20% vs -43%"),
        ("周期边界", "个股周线 4/8/12 周 49.4%–49.5%，月线 12 月 50.73%（中位 -8.4%）；指数月线 2016+ 4.6/5.1/0.2%，回撤 -28/-20/-32 vs -47/-43/-57；周线 2016+ 全负"),
        ("沪深 300 2016 后", "年化 0.1% vs 0.9%，回撤 -39% vs -47%"),
        ("连续特征正 IC 数量", "0/12；DIF 水平 -7.16%（t=-8.42），柱体水平 -2.80%"),
        ("红柱放大严格增量", "-0.179%/20 日（t=-5.89）；高波/高换手/高位 -0.3% ~ -0.5%"),
        ("DIF>DEA 严格增量", "-0.226%/20 日（t=-2.98），不是买入信号"),
        ("月线 + 长期下跌", "12m IC +11.8%（t=5.15），Q5-Q1 +10.1%；高流动 Q5 仍跑输宽基（约 -1.4% ~ -2.5%）"),
        ("特征层组合", "剔除红柱放大 6.08% vs 全 A 6.09%；低 DIF 9.38% vs 低 20 日涨幅 9.08%（反转，不是 MACD）"),
        ("一句话", "日线越像买入信号越差，唯一稳健增量是红柱放大=卖/避；月线只在长期下跌结构里做相对反弹；指数层面是危机保险。"),
    ], columns=["项目", "内容"])
    summary.to_excel(w, sheet_name="00_一页纸", index=False)

    overall.to_excel(w, sheet_name="01_总体", index=False)
    yearly.to_excel(w, sheet_name="02_年度", index=False)
    cond.to_excel(w, sheet_name="03_条件", index=False)
    pdig.to_excel(w, sheet_name="04_参数摘要", index=False)
    grid.sort_values("pct20", ascending=False).head(80).to_excel(w, sheet_name="05_参数网格_前80", index=False)
    v.to_excel(w, sheet_name="06_变体", index=False)
    varport.to_excel(w, sheet_name="07_变体组合", index=False)
    exits.to_excel(w, sheet_name="08_离场", index=False)
    idx.to_excel(w, sheet_name="09_指数择时", index=False)
    ew.to_excel(w, sheet_name="10_指数择时_等权", index=False)
    pd.read_csv(os.path.join(RES, "00_数据审计.csv")).to_excel(w, sheet_name="11_数据审计", index=False)
    dist.to_excel(w, sheet_name="12_分布", index=False)
    pockets.to_excel(w, sheet_name="13_交互格", index=False)
    pstock.to_excel(w, sheet_name="14_周期股票", index=False)
    pbest.to_excel(w, sheet_name="15_周期参数", index=False)
    cidx.to_excel(w, sheet_name="16_周期指数", index=False)
    multi.to_excel(w, sheet_name="17_多周期", index=False)
    tfeat.to_excel(w, sheet_name="18_特征全样本", index=False)
    tstruct.to_excel(w, sheet_name="19_特征结构", index=False)
    tstrict.to_excel(w, sheet_name="20_特征严格增量", index=False)
    tstate.to_excel(w, sheet_name="21_结构状态", index=False)
    tmonth.to_excel(w, sheet_name="22_月度下行", index=False)
    tmonthfm.to_excel(w, sheet_name="23_月度下行FM", index=False)
    tmonthdist.to_excel(w, sheet_name="24_月度下行分布", index=False)
    tport.to_excel(w, sheet_name="25_特征组合", index=False)
    tperf.to_excel(w, sheet_name="26_周期特征", index=False)
    tperfs.to_excel(w, sheet_name="27_周期特征结构", index=False)
    ttrend.to_excel(w, sheet_name="28_月度下行趋势定义", index=False)

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    hdr_fill = PatternFill("solid", fgColor="1F3A52")
    hdr_font = Font(color="FFFFFF", bold=True, size=10)
    thin = Side(style="thin", color="E4E7EC")
    for sh in w.book.worksheets:
        sh.freeze_panes = "A2"
        for c in sh[1]:
            c.fill = hdr_fill
            c.font = hdr_font
            c.alignment = Alignment(horizontal="center", vertical="center")
        for col in range(1, sh.max_column + 1):
            width = 12
            for row in range(1, min(sh.max_row, 80) + 1):
                val = sh.cell(row=row, column=col).value
                if val is not None:
                    width = max(width, min(42, len(str(val)) * 1.1))
            sh.column_dimensions[get_column_letter(col)].width = width
        for row in sh.iter_rows(min_row=1, max_row=sh.max_row, max_col=sh.max_column):
            for c in row:
                c.border = Border(bottom=thin)
                if isinstance(c.value, float):
                    c.number_format = "0.0000"

print("html", os.path.getsize(os.path.join(ROOT, "MACD有效性_可视化.html")))
print("xlsx", os.path.getsize(xls))
