"""Final research report rendering for the frozen robustness run."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _number(value) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _actual(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, float):
        return _number(value)
    return str(value)


def render_final_report(summary: dict, output_path: str | Path | None = None) -> Path:
    """Render every gate and every real variant, including negative outcomes."""
    path = Path(output_path) if output_path is not None else PROJECT_ROOT / "90_研究结论.md"
    baseline = summary["baseline"]
    common = summary["common_window"]
    concentration = summary["profit_concentration"]
    bootstrap = summary["bootstrap"]
    stress = summary["stress"]
    lines = [
        "# 主板极简趋势系统：稳健性、历史压力与机械裁决",
        "",
        "本报告只交付预注册研究结果，不构成收益承诺或投资建议。历史区间与规则族已有研究覆盖，任何历史正收益都不构成新的确认性样本外证据。",
        "",
        f"## 最终裁决：{summary['verdict']}",
        "",
        f"- 冻结配置哈希：`{summary['config_sha256']}`",
        f"- 主窗口：{baseline['start_date']} 至 {baseline['end_date']}",
        f"- 主基线净收益：{_number(baseline['total_return'])}",
        f"- 主基线最大回撤：{_number(baseline['max_drawdown'])}",
        f"- 主基线期末权益：{_number(baseline['final_equity'])}",
        "- 裁决只使用冻结基线与机械门槛；没有从历史变体中选优替换基线。",
        "",
        "## 机械门槛",
        "",
        "| 门槛 | 实际值 | 冻结要求 | 是否通过 |",
        "| --- | --- | --- | --- |",
    ]
    for gate in summary["gates"]:
        lines.append(
            f"| {gate['id']} | {_actual(gate['actual'])} | {gate['requirement']} | "
            f"{'是' if gate['passed'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "裁决规则：任一非回撤门槛失败或最大回撤超过10%即为REJECTED；其他门槛均通过时，最大回撤不超过8%才可为UNREJECTED，8%—10%只能为BOUNDARY。",
            "",
            "## 完整真实变体矩阵",
            "",
            "每一行均复用完整账户、原价成交、整手、费用和公司行动规则；未用单笔交易均值代替账户回测。",
            "",
            "| 类别 | 变体 | 状态 | 净收益 | 最大回撤 | 输出 |",
            "| --- | --- | --- | ---: | ---: | --- |",
            f"| baseline | baseline | completed | {_number(baseline['total_return'])} | {_number(baseline['max_drawdown'])} | {summary['outputs']['robustness']}/baseline |",
        ]
    )
    for variant in summary["variants"]:
        lines.append(
            f"| {variant['family']} | {variant['name']} | {variant['status']} | "
            f"{_number(variant['total_return'])} | {_number(variant['max_drawdown'])} | {variant['output_dir']} |"
        )
    lines.extend(
        [
            "",
            "## 利润集中度：账户年度权益盈亏",
            "",
            "首年盈亏相对初始现金，后续年份盈亏相对上一年末权益；不使用仅闭合交易的已实现年度贡献。",
            "",
            "| 年度 | 权益盈亏（元） | 是否贡献最高年份 |",
            "| --- | ---: | --- |",
        ]
    )
    for row in summary["annual_equity"]:
        lines.append(
            f"| {row['year']} | {_number(row['equity_pnl'])} | {'是' if row['is_top_contributor'] else '否'} |"
        )
    lines.extend(
        [
            "",
            f"去掉贡献最高年份 {concentration['top_year']} 后，剩余年份账户权益盈亏合计为 {_number(concentration['remaining_equity_pnl'])} 元。",
            "",
            "## 双源复权共同窗口",
            "",
            f"- 同花顺面板起始日：{common['warmup_source_start']}；指标完整后的共同账户窗口：{common['start_date']} 至 {common['end_date']}。",
            "- 两源均从现金空仓起算，使用相同原价执行、组合、费用和公司行动规则。",
            "- 共同窗口不能视为完整主窗口，较短窗口结果不得冒充主基线。",
            f"- baostock共同窗口净收益/最大回撤：{_number(common['baostock_total_return'])} / {_number(common['baostock_max_drawdown'])}。",
            f"- 同花顺共同窗口净收益/最大回撤：{_number(common['hithink_total_return'])} / {_number(common['hithink_max_drawdown'])}。",
            f"- 入场信号差异：{common['entry_signal_differences']}；退出信号差异：{common['exit_signal_differences']}；完整差异表已落盘。",
            "",
            "## 成本、随机路径与自助区间",
            "",
            "双倍成本变体将全部佣金、印花税和过户费组成乘以2，并将单边滑点提高到20bp；仓位与现金路径仍由完整账户回测产生。",
            "20个随机排序路径只改变入场优先级；20个随机入场路径按原信号日数量从当日其他合格非信号股票中无放回替换，保留真实120日排序收益。",
            f"20交易日块、{bootstrap['iterations']}次自助只重采样基线日收益，属于探索性区间，不参与机械门槛。总收益分位数：2.5%={_number(bootstrap['total_return_quantiles']['p02_5'])}，50%={_number(bootstrap['total_return_quantiles']['p50'])}，97.5%={_number(bootstrap['total_return_quantiles']['p97_5'])}。",
            "",
            "## 年度、滚动与事后描述性分组",
            "",
            "主基线年度收益和月末滚动三年结果保存在各账户输出的annual_and_rolling.csv；年度权益盈亏见上表。",
            "",
        ]
    )
    for name, formula in summary["descriptive"]["formulas"].items():
        lines.append(f"- {name}：{formula}")
    lines.extend(
        [
            "",
            "市场状态与大小盘参数未预冻结，以下分组不参与门槛，也不变成交易闸门。",
            "",
            "| 分组类型 | 分组 | 天数 | 组内复合收益 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in summary["descriptive"]["groups"]:
        lines.append(
            f"| {row['group_type']} | {row['group']} | {row['days']} | {_number(row['compound_return'])} |"
        )
    lines.extend(
        [
            "",
            "## 2010—2015单源历史压力",
            "",
            "主结果和配置哈希冻结后才执行该压力试验；它不参与选参，也不能挽救或替换主基线。",
            "",
            f"- 状态：{stress['status']}；窗口：{stress['start_date']} 至 {stress['end_date']}。",
            f"- 净收益：{_number(stress['total_return'])}；最大回撤：{_number(stress['max_drawdown'])}。",
            f"- 数据模式：{stress['source_mode']}。{stress.get('limitation', '仅作单源历史压力，不用于回头调参或替换基线')}。",
            "",
            "## 输出与复现边界",
            "",
            f"- 主稳健性结果：`{summary['outputs']['robustness']}`；配置和主输出哈希：`main_freeze.json`。",
            f"- 历史压力结果：`{summary['outputs']['stress']}`。",
            "- 大表使用Parquet或CSV落盘；summary.json保存门槛、变体和运行状态。",
            "- 历史压力仅单源，共同窗口短于主窗口，描述性分组为事后口径；这些限制均不能被历史收益抵消。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
