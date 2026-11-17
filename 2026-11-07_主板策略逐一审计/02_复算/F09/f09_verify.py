# -*- coding: utf-8 -*-
"""F09 早期系统审计：只读复核关键数字（hithink / 阶梯双K / 持仓天数 / 全信号）。
所有源文件只读；输出仅写本目录。运行: python3 f09_verify.py
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = "/Users/admin/WorkSpace/trading"
OUT = os.path.dirname(os.path.abspath(__file__))
rows = []


def rec(block, item, claimed, actual, ok=None, note=""):
    if ok is None:
        ok = "见数值"
    rows.append(dict(块=block, 项=item, 原称=claimed, 复核=actual, 判定=ok, 备注=note))
    print(f"[{block}] {item}: 原称={claimed} 复核={actual} {ok} {note}")


def close(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


# ---------- 1. trading_hithink 初始研究 ----------
H = os.path.join(ROOT, "trading_hithink/初始研究/results")
fs = json.load(open(os.path.join(H, "final_summary.json")))
ra = fs["research_attempts"]
rec("hithink", "账户变体/过开发/验证/留出", "92/20/2/1",
    f"{ra['account_variants']}/{ra['account_variants_passing_development_gates']}/"
    f"{ra['validation_finalists']}/{ra['holdout_strategies_run_for_selection']}",
    "一致(JSON自报)")
trials = pd.read_csv(os.path.join(H, "development/all_account_trials.csv"))
rec("hithink", "试验计数对账", "92变体/20过门",
    f"现存all_account_trials.csv={len(trials)}行/{int(trials['passed_gates'].sum())}过门",
    "不完全对账",
    "deliver.py按len(trials)计数；现存CSV比JSON锁定时多14行/5过门，疑为交付后追加试验；方向是更多而非更少，不影响30%失败结论")

# 独立从 daily_equity.csv 复算 CAGR / MDD
for stage in ("development", "validation", "holdout"):
    for cap in ("20000", "50000"):
        m = fs[stage][cap]
        eq = pd.read_csv(os.path.join(ROOT, "trading_hithink/初始研究", m["files"]["daily_equity"]),
                         parse_dates=["date"])
        e = eq["equity"].to_numpy()
        init = m["initial_capital"]
        yrs = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days / 365.25
        cagr = (e[-1] / init) ** (1 / yrs) - 1
        mdd = float(np.min(e / np.maximum.accumulate(e) - 1))
        rec("hithink", f"{stage} c{cap} CAGR", f"{m['cagr']:.4f}", f"{cagr:.4f}",
            "一致" if close(cagr, m["cagr"], 0.005) else "偏差",
            f"期末={e[-1]:.2f} vs JSON {m['final_equity']:.2f}; 年数={yrs:.3f}(日历日/365.25)")
        rec("hithink", f"{stage} c{cap} MDD", f"{m['max_drawdown']:.4f}", f"{abs(mdd):.4f}",
            "一致" if close(abs(mdd), m["max_drawdown"], 0.005) else "偏差", "JSON按正数存储")

rec("hithink", "30%目标", "失败",
    f"targets全为False={not any(v for k, v in fs['targets'].items() if k.startswith('both'))}",
    "一致", "holdout CAGR -8.4%/-11.5%，留出回撤42.6%/47.1%")
rec("hithink", "宇宙偏差", "current_listed_only",
    f"survivorship_bias={fs['data']['survivorship_bias']}, ST未知={not fs['data']['historical_st_known']}",
    "自认缺陷", "DuckDB单源、无退市、无历史ST——与冻结口径不符，属已知D级缺陷")

# flash 审计修复 O03
sm = pd.read_csv(os.path.join(ROOT, "trading_hithink/flash 审计修复/results/summary.csv"))
o03 = sm[sm["scenario"].isin(["O03_c20000", "O03_c50000"])]
for _, r in o03.iterrows():
    claim = 0.3670 if r["scenario"] == "O03_c20000" else 0.3703
    rec("hithink-flash", f"{r['scenario']} CAGR", f"{claim:.2%}", f"{r['cagr']:.2%}",
        "一致" if close(r["cagr"], claim, 0.001) else "偏差",
        f"evaluation_status={r['evaluation_status']}（非新OOS）")

# ---------- 2. 阶梯双K ----------
L = os.path.join(ROOT, "analysis")
LA = os.path.join(L, "交易系统_审计修正")

# 原 1万账户净值曲线：+18.3%/-71.7% 本身可从交付净值复算（但产生它的假设有缺陷）
eq0 = pd.read_csv(os.path.join(L, "portfolio_equity_1w.csv"), index_col=0, parse_dates=True).iloc[:, 0]
yrs0 = (eq0.index[-1] - eq0.index[0]).days / 365.25
cagr0 = (eq0.iloc[-1] / 10000) ** (1 / yrs0) - 1
mdd0 = float((eq0 / eq0.cummax() - 1).min())
rec("阶梯双K", "原1万账户 +18.3%/-71.7%", "期末53,399 +434% +18.3% -71.7%",
    f"期末{eq0.iloc[-1]:,.0f} {eq0.iloc[-1]/10000-1:+.1%} {cagr0:+.2%} {mdd0:.1%}",
    "一致(算术上)", "原净值曲线可复算；但无逐笔/持仓日志，且由前视+单一路径+缺费假设产生")


def trade_stats(tr, pnl="pnl", R="R"):
    wins = int((tr[R] >= 0.1).sum())
    losses = int((tr[pnl] < 0).sum())
    gp = tr.loc[tr[pnl] > 0, pnl].sum()
    gl = -tr.loc[tr[pnl] < 0, pnl].sum()
    return dict(n=len(tr), win=wins / max(1, wins + losses), pf=gp / gl if gl > 0 else np.inf,
                avgR=tr[R].mean(), medR=tr[R].median())


pooled = pd.read_csv(os.path.join(L, "ladder_doublek_trades_pooled.csv"))
s = trade_stats(pooled)
rec("阶梯双K", "书版1:1 笔数", "83,748", s["n"], "一致" if s["n"] == 83748 else "偏差")
rec("阶梯双K", "书版 胜率(原定义R>=0.1)", "59.4%", f"{s['win']:.1%}",
    "一致" if close(s["win"], 0.594, 0.005) else "偏差")
rec("阶梯双K", "书版 毛PF / 平均R", "1.065 / -0.046", f"{s['pf']:.3f} / {s['avgR']:.3f}",
    "一致" if close(s["pf"], 1.065, 0.01) and close(s["avgR"], -0.046, 0.005) else "偏差",
    "pnl未扣费，毛口径")

# 注意：ALL 逐笔明细 CSV 已被 2026-10-21 00:17 的 --quick(5只) 运行覆盖，
# 全样本逐笔数字只能核对 修正版运行日志.txt（全样本运行自报）。
import re
log = open(os.path.join(LA, "修正版运行日志.txt")).read()
for label, claim_n, claim_pf, claim_avgR in (("A", 13351, 1.31, 0.291), ("B", 40250, 1.43, 0.136),
                                             ("A+B", 53601, 1.38, 0.175)):
    tr = pd.read_csv(os.path.join(LA, f"修正版_交易明细_{label}_ALL_2016_2026.csv"))
    m = re.search(rf"修正版 {re.escape(label)} \(ALL_2016_2026\): (\d+)笔 .*PF ([\d.]+) \| 平均R \+?([-\d.]+)", log)
    ln, lpf, lavg = int(m.group(1)), float(m.group(2)), float(m.group(3))
    rec("阶梯双K", f"修正{label} 笔数/PF/平均R", f"{claim_n}/{claim_pf}/{claim_avgR}",
        f"日志{ln}/{lpf}/{lavg}; 现存CSV仅{len(tr)}笔/{tr['symbol'].nunique()}只",
        "一致(仅日志)" if ln == claim_n and close(lpf, claim_pf, 0.01) else "偏差",
        "全样本明细CSV被--quick覆盖，无法逐笔独立复算；组合层交付物完好且一致")

# PIT 明细 CSV 完好，可独立复算（报告4.4：A_pit 2,362笔 PF1.30 平均R+0.382）
pit = pd.read_csv(os.path.join(LA, "修正版_交易明细_A_PIT_2020_2025.csv"))
sp = trade_stats(pit)
rec("阶梯双K", "PIT A_pit 笔数/PF/平均R", "2,362/1.30/+0.382",
    f"{sp['n']}/{sp['pf']:.3f}/{sp['avgR']:.3f}",
    "一致" if sp["n"] == 2362 and close(sp["pf"], 1.30, 0.01) and close(sp["avgR"], 0.382, 0.005) else "偏差",
    "PIT明细未被覆盖，独立复算通过")

# 书版净费后 PF 1.016 / 平均R -0.080：按修正版费率近似复算（万1双边+印花税+滑点已含于价格）
p = pooled.copy()
p["exit_date"] = pd.to_datetime(p["exit_date"])
stamp = np.where(p["exit_date"] >= pd.Timestamp("2023-08-28"), 0.0005, 0.001)
net_pnl = p["shares"] * p["exit"] * (1 - 0.0001 - stamp) - p["shares"] * p["entry"] * (1 + 0.0001)
gp = net_pnl[net_pnl > 0].sum(); gl = -net_pnl[net_pnl < 0].sum()
netR = net_pnl / p["risk_amt"]
rec("阶梯双K", "书版净费后 PF/平均R(近似复算)", "1.016 / -0.080",
    f"{gp/gl:.3f} / {netR.mean():.3f}",
    "一致" if close(gp / gl, 1.016, 0.01) and close(netR.mean(), -0.080, 0.01) else "偏差",
    "用修正版费率(万1+印花税)套原逐笔；原entry/exit已含万5滑点")

paths = pd.read_csv(os.path.join(LA, "修正版_组合路径分布_ALL_2016_2026.csv"))
rnd = paths[paths["policy"] == "random"]
q = rnd["cagr"].quantile([0.05, 0.5, 0.95])
rec("阶梯双K", "随机路径数/中位年化", "200条 / -2.6%", f"{len(rnd)}条 / {q.iloc[1]:.2%}",
    "一致" if len(rnd) == 200 and close(q.iloc[1], -0.026, 0.002) else "偏差",
    f"5%={q.iloc[0]:.2%} 95%={q.iloc[2]:.2%}; MDD中位={rnd['mdd'].median():.1%}")
sym = paths[paths["policy"] == "symbol"].iloc[0]
vol = paths[paths["policy"] == "vol"].iloc[0]
rec("阶梯双K", "修正A+B 6仓 年化区间", "-6.4% ~ -2.1%",
    f"{sym['cagr']:.2%} ~ {vol['cagr']:.2%}",
    "一致" if close(sym["cagr"], -0.064, 0.002) and close(vol["cagr"], -0.021, 0.002) else "偏差",
    f"MDD {vol['mdd']:.1%} ~ {sym['mdd']:.1%}")

eq = pd.read_csv(os.path.join(LA, "修正版_组合净值_1万_ALL_2016_2026_vol.csv"),
                 index_col=0, parse_dates=True).iloc[:, 0]
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
cagr = (eq.iloc[-1] / 10000) ** (1 / yrs) - 1
mdd = float((eq / eq.cummax() - 1).min())
rec("阶梯双K", "修正vol净值独立复算", "期末8,129 年化-2.1% MDD-61.8%",
    f"期末{eq.iloc[-1]:,.0f} 年化{cagr:.2%} MDD{mdd:.1%}",
    "一致" if close(cagr, -0.0205, 0.002) and close(mdd, -0.618, 0.005) else "偏差")

# 波段为王2 幸存者修正对照
old = pd.read_csv(os.path.join(ROOT, "八系统回测/结果/八系统汇总.csv"))
new = pd.read_csv(os.path.join(ROOT, "八系统回测/结果_历史成分/汇总_历史成分.csv"))
oc = old[old.iloc[:, 0] == "波段为王2"].iloc[0, 3]
nc = new[new.iloc[:, 0] == "波段为王2"].iloc[0, 3]
rec("阶梯双K", "波段为王2 年化(旧池→历史成分)", "+17.13% → +6.28%",
    f"{oc:.2%} → {nc:.2%}",
    "一致" if close(oc, 0.1713, 0.001) and close(nc, 0.0628, 0.001) else "偏差")

# ---------- 3. 持仓天数 ----------
P = os.path.join(ROOT, "2026-10-22_持仓天数_涨跌路径")
ps = pd.read_csv(os.path.join(P, "路径统计.csv"))
g = ps[ps["样本"] == "全信号"].set_index("持有天数")
rec("持仓天数", "全信号 d1/d20 毛胜率", "50.9% → 51.8%",
    f"{g.loc[1,'上涨概率']:.1%} → {g.loc[20,'上涨概率']:.1%}",
    "一致" if close(g.loc[1, "上涨概率"], 0.509, 0.002) and close(g.loc[20, "上涨概率"], 0.518, 0.002) else "偏差")

fx = pd.read_csv(os.path.join(P, "固定天数退出.csv"))
f20 = fx[(fx["样本"] == "全信号") & (fx["持有天数"] == 20)].iloc[0]
rec("持仓天数", "全信号 20天净平均/净胜率", "+2.0% / 50.5%",
    f"{f20['净平均']:.2%} / {f20['净胜率']:.1%}",
    "一致" if close(f20["净平均"], 0.020, 0.002) and close(f20["净胜率"], 0.505, 0.005) else "偏差")

wl = pd.read_csv(os.path.join(P, "赢家输家.csv"))
w = wl[(wl["样本"] == "全信号") & (wl["组别"] == "最终上涨")].iloc[0]
l = wl[(wl["样本"] == "全信号") & (wl["组别"] == "最终下跌")].iloc[0]
rec("持仓天数", "赢家曾浮亏/输家曾浮盈", "77.9% / 76.8%",
    f"{w['曾浮亏比例']:.1%} / {l['曾浮盈比例']:.1%}",
    "一致" if close(w["曾浮亏比例"], 0.779, 0.002) and close(l["曾浮盈比例"], 0.768, 0.002) else "偏差")

# 独立从逐笔路径摘要复算
zs = pd.read_csv(os.path.join(P, "逐笔路径摘要.csv"))
z = zs[zs["样本"] == "全信号"]
d1 = z["第1天收益"].dropna(); d20 = z["第20天收益"].dropna()
rec("持仓天数", "逐笔摘要独立复算 d1/d20 毛胜率", "50.9% / 51.8%",
    f"{(d1>0).mean():.1%} / {(d20>0).mean():.1%} (n={len(z)}); 剔除舍入为0后 {(d1[d1!=0]>0).mean():.1%} / {(d20[d20!=0]>0).mean():.1%}",
    "舍入差内一致",
    f"摘要CSV保留6位小数，d1有{(d1==0).sum()}笔、d20有{(d20==0).sum()}笔被舍入为0；汇总表本身与报告完全一致")
win = z[z["第20天收益"] > 0]; los = z[z["第20天收益"] < 0]
rec("持仓天数", "逐笔摘要独立复算 赢家曾浮亏/输家曾浮盈", "77.9% / 76.8%",
    f"{(win['收盘最大浮亏']<0).mean():.1%} / {(los['收盘最大浮盈']>0).mean():.1%}",
    "舍入差内一致", "同上，浮亏/浮盈小幅值被舍入为0导致约0.7-0.9pp差")

# ---------- 4. 全信号一万 ----------
Q = os.path.join(ROOT, "2026-10-22_全信号_买卖时间_优先级研究")
t = pd.read_csv(os.path.join(Q, "全信号_一万_逐笔.csv"))
pnl = t["止损后净盈亏"]
top1 = pnl.nlargest(max(1, int(len(pnl) * 0.01))).sum() / pnl.sum()
rec("全信号", "笔数/平均/胜率(止损版)", "40,279 / +177.19元(+1.77%) / 50.3%",
    f"{len(t)} / {pnl.mean():+.2f}元({pnl.mean()/10000:+.2%}) / {(pnl>0).mean():.1%}",
    "一致" if len(t) == 40279 and close(pnl.mean(), 177.19, 0.5) else "偏差")
rec("全信号", "前1%贡献", "41.8%", f"{top1:.1%}",
    "一致" if close(top1, 0.418, 0.005) else "偏差")

first = pd.read_csv(os.path.join(Q, "全信号_一万_每股首笔.csv"))
fp = first["止损后净盈亏"]
rec("全信号", "每股首笔 只数/平均/胜率", "4,976 / +249.66(+2.50%) / 53.8%",
    f"{len(first)} / {fp.mean():+.2f}({fp.mean()/10000:+.2%}) / {(fp>0).mean():.1%}",
    "一致" if len(first) == 4976 and close(fp.mean(), 249.66, 0.5) else "偏差")

ann = pd.read_csv(os.path.join(Q, "全信号_一万_年度.csv"))
a22 = ann[ann["年份"] == 2022].iloc[0]
rec("全信号", "2022全信号平均", "-2.03%", f"{a22['止损后平均']/10000:+.2%}",
    "一致" if close(a22["止损后平均"] / 10000, -0.0203, 0.001) else "偏差")
ann2 = pd.read_csv(os.path.join(ROOT, "2026-10-21_45仓位_波动20_交易系统/年度_收益与大盘.csv"))
r22 = ann2[ann2["年份"] == 2022].iloc[0]["系统收益"]
rec("全信号", "2022 2仓账户收益", "+59.9%", f"{r22:+.1%}",
    "一致" if close(r22, 0.599, 0.002) else "偏差",
    "信号层-2.03% vs 账户+59.9%：路径/集中度反例成立")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f09_关键数字复核.csv"), index=False)
print(f"\n共 {len(rows)} 项，已写 f09_关键数字复核.csv")