#!/usr/bin/env python3
"""Build a standalone HTML summary for the independent verification."""
from __future__ import annotations

import base64
import io
import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = "/Users/admin/WorkSpace/trading/2026-10-29_看长做短_独立验证"
OUT = f"{BASE}/03_结果"
HTML = f"{BASE}/看长做短_独立验证.html"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def equity_chart() -> str:
    files = {
        "V2 规则退出": f"{OUT}/组合_V2_权益.csv",
        "V2 固定20日": f"{OUT}/组合_V2_权益_fixed20.csv",
        "V8 市场闸门+固定20": f"{OUT}/组合_V8_csi500_ma200_fixed20_权益.csv",
    }
    series = {}
    for name, f in files.items():
        eq = pd.read_csv(f, parse_dates=["date"]).set_index("date")["equity"]
        series[name] = eq
    con = duckdb.connect(f"{BASE}/02_代码/work.duckdb", read_only=True)
    ew = con.execute("select cast(date as date) d, nav from ew_nav where cast(date as date)>=date '2012-01-03' order by 1").fetchdf()
    idx = con.execute("select cast(date as date) d, close from indices where code='sh.000905' and cast(date as date)>=date '2012-01-03' order by 1").fetchdf()
    con.close()
    ew["d"] = pd.to_datetime(ew["d"]); ew = ew.set_index("d")["nav"]
    idx["d"] = pd.to_datetime(idx["d"]); idx = idx.set_index("d")["close"]
    series["等权全A"] = ew / ew.iloc[0]
    series["中证500"] = idx / idx.iloc[0]

    df = pd.DataFrame(series).sort_index().ffill()
    df = df / df.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = {"V2 规则退出": "#c0392b", "V2 固定20日": "#e67e22",
              "V8 市场闸门+固定20": "#2c7fb8", "等权全A": "#27ae60", "中证500": "#7f8c8d"}
    for c in df.columns:
        ax.plot(df.index, df[c], label=c, color=colors.get(c), lw=1.8 if c.startswith("V8") else 1.3)
    ax.set_yscale("log")
    ax.set_ylabel("净值（对数，2012-01-03=1）")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax.set_title("图上看长做短：机械代理 vs 基准（2012-2026）", fontsize=12)
    fig.tight_layout()
    return b64(fig)


def excess_bar() -> str:
    y = pd.read_csv(f"{OUT}/信号层_分年.csv")
    y = y[y.variant == "V2_monthly"].set_index("year")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(y.index, y["ex20_mean"] * 100, color=["#27ae60" if v > 0 else "#c0392b" for v in y["ex20_mean"]])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("20日平均超额（%）")
    ax.set_title("V2 信号：逐年 20 日平均超额（相对等权全A）", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return b64(fig)


def table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str]) -> str:
    head = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                try:
                    v = fmt[c].format(v)
                except Exception:
                    pass
            cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def main() -> None:
    eq_png = equity_chart()
    ex_png = excess_bar()

    sig = pd.read_csv(f"{OUT}/信号层_20日主表.csv")
    sig = sig[sig.period == "full"].copy()
    sig = sig[~sig["variant"].isin(["V2_monthly_ma"])].copy()
    sig = sig[["variant", "n", "mean", "median", "win", "mean_ex", "median_ex", "win_ex", "t_ex_cluster"]]
    sig["mean"] *= 100; sig["median"] *= 100; sig["win"] *= 100
    sig["mean_ex"] *= 100; sig["median_ex"] *= 100; sig["win_ex"] *= 100
    sig_html = table(sig, ["variant", "n", "mean", "median", "win", "mean_ex", "median_ex", "win_ex", "t_ex_cluster"],
                     {"mean": "{:.2f}%", "median": "{:.2f}%", "win": "{:.1f}%",
                      "mean_ex": "{:.2f}%", "median_ex": "{:.2f}%", "win_ex": "{:.1f}%",
                      "t_ex_cluster": "{:.2f}", "n": "{:,}"})

    port = pd.DataFrame([
        ["V2 破MA20/-8%/20日", -10.72, -92.58, 0.198, 2843, 23.5, -0.50, -2.64, 0.85],
        ["V2 破MA20/-10%/40日", -10.17, -91.03, 0.217, 2534, 23.0, -0.53, -2.65, 0.84],
        ["V2 固定20日", -1.92, -76.97, 0.758, 1285, 44.6, 0.25, -1.97, 1.04],
        ["V2 +25%卖一半", -11.62, -93.69, 0.172, 3046, 28.0, 1.33, -2.43, 1.42],
        ["V8 破MA20/-8%/20日", -6.01, -80.42, 0.413, 2136, 24.4, -0.28, -2.64, 0.92],
        ["V8 固定20日", 6.04, -67.30, 2.309, 900, 45.2, 1.31, -1.17, 1.25],
        ["等权全A", 13.27, -62.73, 5.917, None, None, None, None, None],
        ["中证500", 5.27, -65.20, 2.080, None, None, None, None, None],
    ], columns=["组合", "年化%", "回撤%", "期末", "交易", "胜率%", "均笔%", "中位%", "盈亏比"])
    port["年化%"] = port["年化%"].map(lambda x: f"{x:.2f}")
    port["回撤%"] = port["回撤%"].map(lambda x: f"{x:.2f}")
    port["期末"] = port["期末"].map(lambda x: f"{x:.3f}")
    port["胜率%"] = port["胜率%"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    port["均笔%"] = port["均笔%"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    port["中位%"] = port["中位%"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    port["盈亏比"] = port["盈亏比"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    port["交易"] = port["交易"].map(lambda x: "" if pd.isna(x) else f"{int(x)}")
    port_html = table(port, list(port.columns), {})

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>看长做短：独立数据验证</title>
<style>
  :root {{ --ink:#1f2933; --muted:#66727f; --line:#e3e8ee; --red:#c0392b; --green:#27ae60; --blue:#2c7fb8; --bg:#f7f9fb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:36px 24px 60px; }}
  h1 {{ font-size:30px; line-height:1.25; margin:0 0 6px; letter-spacing:-.5px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:24px; }}
  .verdict {{ background:#fff; border-left:5px solid var(--red); padding:18px 20px; border-radius:10px; box-shadow:0 1px 2px rgba(0,0,0,.04); margin:18px 0 26px; }}
  .verdict b {{ color:var(--red); }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0 28px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .card .k {{ color:var(--muted); font-size:12px; }}
  .card .v {{ font-size:23px; font-weight:700; margin-top:4px; letter-spacing:-.5px; }}
  .card .n {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .red {{ color:var(--red); }} .green {{ color:var(--green); }} .blue {{ color:var(--blue); }}
  h2 {{ font-size:19px; margin:36px 0 10px; }}
  p {{ margin:8px 0; }}
  .small {{ color:var(--muted); font-size:13px; }}
  img {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:10px; padding:8px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13px; margin:12px 0 6px; }}
  th,td {{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
  th {{ background:#eef3f7; color:#334; font-weight:600; }}
  td:first-child, th:first-child {{ text-align:left; }}
  tr:last-child td {{ border-bottom:none; }}
  .note {{ background:#fff8e6; border:1px solid #f0d9a8; border-radius:8px; padding:10px 14px; font-size:13px; color:#6b4d13; }}
  .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:760px) {{ .cards {{ grid-template-columns:1fr 1fr; }} .two {{ grid-template-columns:1fr; }} h1 {{ font-size:24px; }} }}
</style></head><body><div class="wrap">
  <h1>图上“看长做短”：独立数据验证</h1>
  <div class="sub">2026-10-29 · 2012-01-01 至 2026-10-28 · baostock 全历史 + 同花顺抽样核对 · 不引用项目已有回测结论</div>

  <div class="verdict">
    <b>结论：机械代理没有通过。</b>
    月/周/日三层全开（V2）的 20 日平均超额 <b>-0.384%</b>、中位 <b>-2.492%</b>、聚类 <b>t=-7.63</b>，
    开发期与验证期都为负。组合层“破 MA20”年化 <b>-10.72%</b>、回撤 <b>-92.58%</b>；
    改成固定 20 日仍为 <b>-1.92%</b>。加“中证500>MA200”市场闸门后 <b>+6.04%</b>，
    但低于等权全A <b>+13.27%</b>，且正收益高度依赖 2025 年。图上最值钱的是“等待”，它是风控，不是 alpha。
  </div>

  <div class="cards">
    <div class="card"><div class="k">V2 信号 20 日平均超额</div><div class="v red">-0.384%</div><div class="n">中位 -2.492% · t -7.63</div></div>
    <div class="card"><div class="k">V2 组合年化（破MA20）</div><div class="v red">-10.72%</div><div class="n">最大回撤 -92.58%</div></div>
    <div class="card"><div class="k">V8 组合年化（闸门+固定20）</div><div class="v blue">+6.04%</div><div class="n">回撤 -67.30% · 等权 +13.27%</div></div>
    <div class="card"><div class="k">2026 年 V2 信号 20 日中位</div><div class="v red">-4.69%</div><div class="n">胜率 36.7% · 单票案例不代表横截面</div></div>
  </div>

  <h2>1. 净值：机械代理 vs 基准</h2>
  <img src="data:image/png;base64,{eq_png}" alt="equity curves">
  <p class="small">V2 规则退出（红）在 2016/2018/2022-2024 连续深度回撤；固定 20 日（橙）好一些，但仍跑输等权；V8 市场闸门（蓝）避开了大部分熊市，却在 2025 年靠少数大赢家拉起，2026 年又回吐。</p>

  <h2>2. 信号层：逐年 20 日平均超额</h2>
  <img src="data:image/png;base64,{ex_png}" alt="yearly excess">
  <p class="small">V2 在 15 个年份里只有 2013、2017、2019、2020、2026 为正；2014/2015 大牛市和 2025 反弹都没有跑赢等权。超额为负不是某一年造成的，是长期结构。</p>

  <h2>3. 信号层主表（20 日，全期）</h2>
  {sig_html}
  <p class="small">V0→V1→V2：绝对收益从 +0.437% 升到 +0.934%，但超额从 -0.207% 降到 -0.384%——月/周过滤加的是 beta，不是 alpha。MACD 变体（V3/V6）更差。V8/V9 市场闸门把绝对收益拉高，但超额仍为负。</p>

  <h2>4. 组合层</h2>
  {port_html}
  <p class="small">“破 MA20”把正常回踩当破位：V2 平均只持有 8.3 天、胜率 23.5%、盈亏比 0.85。固定 20 日把年化从 -10.72% 拉回 -1.92%，说明问题主要在退出，不是入场。V8 固定 20 日最好，但仍低于等权，回撤更深。</p>
  <div class="note">V8 固定 20 日的 +6.04% 不是稳定收益：2012-2024 年化只有 +3.87%，2025 年 +76.3%，2026 年至 10/28 为 -19.8%。900 笔交易里只有 15 笔 >+50%；2025 年三笔 +122%、+118%、+100%。去掉右尾，结果就不成立。</div>

  <h2>5. 失败解剖</h2>
  <div class="two">
    <div>
      <p><b>路径太宽。</b> V2 的 MFE 均值 +13.6%、MAE 均值 -10.2%；20 日 P10 -15.8%、P90 +19.8%。用 MA20 止损会被正常回踩洗掉；用固定 20 日又要承受 -10% 浮亏。</p>
      <p><b>中位是亏的。</b> 平均 +0.934% 靠少数大赢家；中位 -0.962%、胜率 46.5%。这不是“高胜率回踩”，是“低胜率+右尾”。</p>
    </div>
    <div>
      <p><b>“收网早”与右尾矛盾。</b> +25% 卖一半的版本年化 -11.62%，比纯规则版 -10.72% 更差；早卖砍掉的正是唯一赚钱的来源。</p>
      <p><b>“等待”有用，但用途被说反。</b> 中证500>MA200 的闸门把 V2 固定 20 日从 -1.92% 拉到 +6.04%，主要贡献是 2018/2022/2023/2024 少交易；它是风险开关，不是收益引擎。</p>
    </div>
  </div>

  <h2>6. 单票案例 vs 横截面</h2>
  <p>图上的 9/15→10/27 是一笔单票、6 周复盘；作者自己写“太贪心账户红变绿”。2026 年 V2 信号 5,699 个，20 日平均 <b>-1.75%</b>、中位 <b>-4.69%</b>、胜率 <b>36.7%</b>；V8 5,279 个，平均 <b>-1.63%</b>、中位 <b>-4.68%</b>、胜率 <b>37.7%</b>。单笔成功与横截面负期望可以同时成立。</p>

  <h2>7. 数据与限制</h2>
  <p>5,564 只 type=1 股票、29,963,672 行日线、259 只窗口内退市股；`is_st` 逐日、ST 当日不买；后复权算信号/收益，不复权算 2 元过滤/涨跌停；同花顺抽样：不复权收盘 12/12 一致，20 日收益中位差异 1e-16。</p>
  <p class="small">最大限制：本库 2009-2026 只有 259 只退市股，比实际少；缺失的多是失败样本，因此本回测对策略偏乐观。如果补全，负结果只会更差。</p>

  <h2>8. 文件</h2>
  <p class="small">预注册：<a href="00_预注册.md">00_预注册.md</a> · 数据审计：<a href="01_数据审计.md">01_数据审计.md</a> · 完整报告：<a href="报告_看长做短_独立验证.md">报告_看长做短_独立验证.md</a><br>
  信号表：<a href="03_结果/信号层_20日主表.csv">03_结果/信号层_20日主表.csv</a> · 组合表：<a href="03_结果/组合_V8_csi500_ma200_fixed20_指标.csv">03_结果/组合_V8_csi500_ma200_fixed20_指标.csv</a></p>
</div></body></html>"""
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print("written", HTML, len(html))


if __name__ == "__main__":
    main()
