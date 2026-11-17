"""保留英文计算原表，输出全中文字段的可阅读副本；不改变任何数值。"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "adjusted_close":"后复权收盘", "adjusted_open":"后复权开盘", "age":"已观察成交天数", "amount":"成交额",
    "bearish":"阴线日数", "bearish_but_up":"阴线但上涨日数", "bearish_up_rate_of_bearish":"阴线中上涨比例",
    "bearish_up_rate_of_eligible":"合格日中阴线且上涨比例", "block_trading_days":"区块交易日数",
    "bootstrap_replicates":"重抽样次数", "bullish":"阳线日数", "bullish_but_down":"阳线但下跌日数",
    "bullish_down_rate_of_bullish":"阳线中下跌比例", "bullish_down_rate_of_eligible":"合格日中阳线且下跌比例",
    "calendar_days":"日历内交易日数", "candidate_signals":"去重前信号数", "censor_date":"删失日期",
    "censor_i":"删失日期序号", "censor_reason":"删失原因", "ci95_high":"百分之九十五区间上限",
    "ci95_low":"百分之九十五区间下限", "close_to_previous_return":"相对前有效收盘收益",
    "code":"证券代码", "cohort_denominator":"队列总数", "comparison":"比较项",
    "completed_dates":"完整标签日期数", "cooldown_removed":"冷却去重删除数",
    "cross_segment_censored":"跨段删失数", "d_amount20":"二十日平均成交额", "date":"日期",
    "date_mean_baseline":"按日平均基准收益", "date_mean_excess":"按日平均超额收益",
    "date_mean_excess_difference":"按日平均超额收益差", "date_mean_label":"按日平均标记收益",
    "definition":"定义", "delist_date":"退市日期", "eligible":"合格股票日数", "eligible_days":"合格交易日数",
    "event_mean_excess":"事件平均超额收益", "event_mean_label":"事件平均标记收益",
    "event_positive_fraction":"事件标记收益为正比例", "expiration_beyond_available_calendar":"到期超出可得日历",
    "expiration_date":"预定到期日期", "expiration_i":"预定到期序号",
    "failure_fraction_of_classified":"已分类队列失败比例", "failure_priority_same_day":"同日先按失败处理",
    "feature_rows":"特征行数", "first_peak":"第一峰值", "first_peak_confirm_date":"第一峰确认日期",
    "first_peak_confirm_i":"第一峰确认序号", "first_peak_date":"第一峰发生日期", "first_peak_i":"第一峰发生序号",
    "flat":"十字日数", "fully_classified_denominator":"完整分类分母", "group":"分组",
    "historical_isST":"历史特别处理标记", "historical_st_known_eligible":"合格日中状态已知数",
    "historical_st_true_eligible":"合格日中特别处理数", "horizon":"观察交易日数", "initial_low":"初始失效低点",
    "initial_low_known_at_start":"初始低点在起点已知", "invalidation_date":"失效日期", "invalidation_i":"失效序号",
    "kind":"类型", "mainboard":"是否主板", "n_after_delisting":"退市后价格缺失数", "n_complete":"完整标签数",
    "n_cross_segment":"跨段标签数", "n_data_end":"数据末尾标签不足数", "n_entry_missing":"入场价格缺失数",
    "n_exit_missing":"结束价格缺失数", "n_failure":"失败数", "n_initial_low_missing":"初始低点缺失数",
    "n_path_gap_censored":"路径缺口删失数", "n_right_censored":"右侧删失数", "n_stock_coverage_end":"个股行情终止数",
    "n_success":"成功数", "n_unfinished":"期限内未完成数", "observed_through_date":"已观察至日期",
    "observed_through_i":"已观察至序号", "paired_dates":"配对日期数", "path_gap_days":"路径缺口天数",
    "previous_adjusted_close":"前有效后复权收盘", "previous_close_known":"前收已知日数",
    "previous_valid_date":"前有效成交日期", "pullback_confirm_date":"回调低点确认日期",
    "pullback_confirm_i":"回调低点确认序号", "pullback_date":"回调低点发生日期", "pullback_i":"回调低点发生序号",
    "pullback_low":"回调低点", "raw_close":"未复权收盘", "raw_first":"原始行情首日", "raw_last":"原始行情末日",
    "raw_open":"未复权开盘", "raw_rows":"原始行数", "same_date_baseline":"同日期基准收益",
    "scope":"时间范围", "second_break_date":"第二次突破日期", "second_break_i":"第二次突破序号",
    "seed":"随机种子", "signal_denominator":"去重后信号分母", "start_adjusted_close":"起点后复权收盘",
    "start_date":"起点日期", "start_i":"起点序号", "status":"状态", "success_fraction_of_all":"全部队列成功比例",
    "success_fraction_of_classified":"已分类队列成功比例", "terminal_date":"终局日期", "terminal_i":"终局序号",
    "trade_status_known_eligible":"合格日中成交状态已知数", "universe":"股票范围", "valid_days":"有效成交日数",
    "valid_first":"有效成交首日", "valid_last":"有效成交末日", "valid_replicates":"有效重抽样次数",
    "volume":"成交股数", "weekly_up_at_start":"起点周趋势向上"
}
VALUES = {"complete":"标签完整", "data_end":"数据末尾不足", "entry_missing":"入场价格缺失",
          "exit_missing":"结束价格缺失", "stock_coverage_end":"个股行情终止", "after_delisting":"退市后价格不可得",
          "success":"成功", "failure":"失败", "unfinished":"期限内未完成", "right_censored":"右侧删失",
          "path_gap_censored":"路径缺口删失", "initial_low_missing":"初始低点缺失", "cross_segment":"跨段删失"}


def main():
    output = ROOT / "03_结果" / "中文数据表"
    output.mkdir(exist_ok=True)
    manifest = []
    for path in sorted((ROOT / "03_结果").glob("命题实证_*.csv")):
        frame = pd.read_csv(path)
        missing = set(frame.columns) - NAMES.keys()
        if missing:
            raise ValueError(f"中文字段未覆盖：{sorted(missing)}")
        before = frame.select_dtypes(include="number").copy()
        for col in frame.select_dtypes(include=["object", "str"]).columns:
            frame[col] = frame[col].replace(VALUES)
            if col in ("group", "comparison"):
                frame[col] = frame[col].str.replace("up10", "十周期均线上行", regex=False).str.replace("up6", "六周期均线上行", regex=False).str.replace("EMA", "指数均线", regex=False)
        pd.testing.assert_frame_equal(before, frame.select_dtypes(include="number"))
        translated = frame.rename(columns=NAMES)
        assert not any(re.search("[A-Za-z]", c) for c in translated.columns)
        translated.to_csv(output / path.name, index=False, encoding="utf-8-sig")
        manifest.append({"文件": path.name, "行数": len(frame), "列数": len(frame.columns), "数值不变": True})
    pd.DataFrame(manifest).to_csv(ROOT / "04_证据" / "中文副本核验.csv", index=False, encoding="utf-8-sig")
    print(f"中文数据副本{len(manifest)}份，全部数值一致")


if __name__ == "__main__":
    main()
