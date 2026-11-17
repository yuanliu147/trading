"""把研究关键结果合并成一个中文 Excel 数据簿；所有工作表与表头为中文。"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "03_结果"
OUT = ROOT / "结构交易研究_数据表.xlsx"


def pct(frame, cols):
    f = frame.copy()
    for c in cols:
        if c in f:
            f[c] = (f[c] * 100).round(2)
    return f


def main():
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        # 1. 结论
        pd.DataFrame({"项目": ["结论", "本金", "交易权限", "候选数", "双段通过数", "任一段年化50%数", "核心策略开发年化%", "核心策略验证年化%", "核心策略开发最大回撤%", "核心策略验证最大回撤%"],
                      "数值": ["无候选通过；不能宣称稳定盈利", 20000, "仅沪深主板", 28, 0, 0, -3.99, -5.59, -16.91, -13.39]}).to_excel(w, sheet_name="结论", index=False)
        # 2. 开发验证候选门槛
        m = pd.read_csv(R / "开发验证候选门槛汇总.csv")
        keep = ["策略", "总收益_开发", "年化收益_开发", "最大回撤_开发", "买入笔数_开发", "净胜率_开发",
                "总收益_验证", "年化收益_验证", "最大回撤_验证", "买入笔数_验证", "净胜率_验证", "候选通过", "年化50达标"]
        m = m[keep].rename(columns={"总收益_开发": "开发总收益", "年化收益_开发": "开发年化", "最大回撤_开发": "开发最大回撤",
                                    "买入笔数_开发": "开发买入笔数", "净胜率_开发": "开发净胜率",
                                    "总收益_验证": "验证总收益", "年化收益_验证": "验证年化", "最大回撤_验证": "验证最大回撤",
                                    "买入笔数_验证": "验证买入笔数", "净胜率_验证": "验证净胜率",
                                    "年化50达标": "任一段年化50达标"})
        pct(m, [c for c in m.columns if any(k in c for k in ("收益", "回撤", "胜率"))]).to_excel(w, sheet_name="候选门槛汇总", index=False)
        # 3. 阴阳线基准率
        a = pd.read_csv(R / "命题实证_A_阴阳与前收基准率.csv")
        a = a.rename(columns={"scope": "范围", "universe": "股票范围", "eligible": "合格日数", "bullish": "阳线日数",
                              "bearish": "阴线日数", "bullish_but_down": "阳线但下跌日数", "bearish_but_up": "阴线但上涨日数",
                              "bullish_down_rate_of_bullish": "阳线中下跌比例", "bearish_up_rate_of_bearish": "阴线中上涨比例"})
        pct(a, ["阳线中下跌比例", "阴线中上涨比例"]).to_excel(w, sheet_name="阴阳线与前收", index=False)
        # 4. 框架增量区间
        b = pd.read_csv(R / "命题实证_B_主组减日回调_日期块bootstrap.csv")
        b = b.rename(columns={"scope": "范围", "universe": "股票范围", "horizon": "观察交易日",
                              "date_mean_excess_difference": "日均超额差", "ci95_low": "区间下限", "ci95_high": "区间上限",
                              "bootstrap_replicates": "重抽样次数", "paired_dates": "配对日期数"})
        pct(b, ["日均超额差", "区间下限", "区间上限"]).to_excel(w, sheet_name="框架增量区间", index=False)
        # 5. 第二驱动汇总
        c = pd.read_csv(R / "命题实证_C_第二驱动分期年度汇总.csv").rename(columns={"分期": "范围", "universe": "股票范围", "horizon": "观察交易日",
            "cohort_denominator": "队列总数", "n_success": "成功数", "n_failure": "失败数", "n_unfinished": "未完成数",
            "n_right_censored": "右删失数", "n_path_gap_censored": "路径删失数",
            "success_fraction_of_classified": "已分类成功率", "failure_fraction_of_classified": "已分类失败率"})
        pct(c, ["已分类成功率", "已分类失败率"]).to_excel(w, sheet_name="第二驱动队列", index=False)
        # 6. 多维分层（全期主板）
        s = pd.read_csv(R / "结构多维_分期年度汇总.csv")
        s = s[(s["分期"] == "全期") & (s["股票范围"] == "主板代码")]
        pct(s, ["成功率占全部", "失败率占全部", "未完成率占全部", "成功率占已分类", "失败率占已分类"]).to_excel(w, sheet_name="多维分层_全期主板", index=False)
        # 7. 行业分层
        ind = pd.read_csv(R / "结构多维_行业分期年度汇总.csv")
        ind = ind[(ind["分期"] == "全期") & (ind["股票范围"] == "主板代码")]
        pct(ind, ["成功率占全部", "失败率占全部", "成功率占已分类", "失败率占已分类"]).to_excel(w, sheet_name="行业分层_全期主板", index=False)
        # 8. 基准
        bch = pd.read_csv(R / "现金与指数基准指标.csv")
        pct(bch, ["总收益", "年化收益", "最大回撤"]).to_excel(w, sheet_name="现金与指数基准", index=False)
        # 9. 数据缺口
        gap = pd.read_csv(ROOT / "04_证据" / "结构多维_数据缺口.csv") if (ROOT / "04_证据" / "结构多维_数据缺口.csv").exists() else None
        if gap is not None:
            gap.to_excel(w, sheet_name="数据缺口", index=False)
    print("Excel 已生成:", OUT)


if __name__ == "__main__":
    main()
