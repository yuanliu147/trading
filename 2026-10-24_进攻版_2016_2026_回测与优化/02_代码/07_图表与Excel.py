# -*- coding: utf-8 -*-
"""最终图表 + Excel 汇总（中文表头）。"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '03_结果')
FIG = os.path.join(ROOT, '04_图表')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 读入
    E0 = pd.read_csv(os.path.join(OUT, '优化_V0_attack10_权益.csv'), index_col=0, parse_dates=True)
    E1 = pd.read_csv(os.path.join(OUT, '优化_V1_core20_权益.csv'), index_col=0, parse_dates=True)
    E2 = pd.read_csv(os.path.join(OUT, '优化_V2_blend1020_权益.csv'), index_col=0, parse_dates=True)
    hold = pd.read_csv(os.path.join(OUT, '稳健_持有邻域.csv'))
    jit = pd.read_csv(os.path.join(OUT, '稳健_持有抖动.csv'))
    opt = pd.read_csv(os.path.join(OUT, '优化_预注册对比.csv'))
    R1 = pd.read_csv(os.path.join(OUT, 'R1_进攻版_指标.csv'))
    yr = pd.read_csv(os.path.join(OUT, '深化_年度对照.csv'), index_col=0)

    # 图 1：三变体净值/回撤/年度/持有邻域
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0, 0]
    ax.plot(E0.index, E0.equity / 20000, label='V0 进攻版 10日（24.5%）', color='#c0504d', lw=1.3)
    ax.plot(E2.index, E2.equity / 20000, label='V2 混合版 10/20日（21.2%，推荐）', color='#e8a33d', lw=1.3)
    ax.plot(E1.index, E1.equity / 20000, label='V1 核心版 20日（16.8%）', color='#1f4e79', lw=1.3)
    ax.set_yscale('log')
    ax.set_title('净值（对数轴，2万本金，2017-03→2026-10-23，单边0.25%成本）')
    ax.legend(fontsize=9); ax.grid(alpha=.25)
    ax = axes[0, 1]
    for E, c, lb in [(E0, '#c0504d', 'V0'), (E2, '#e8a33d', 'V2'), (E1, '#1f4e79', 'V1')]:
        ax.fill_between(E.index, E.equity / E.equity.cummax() - 1, 0, color=c, alpha=.3, label=lb)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('回撤（V0 最大 -36.3%，V2 -26.8%，V1 -23.6%）')
    ax.legend(); ax.grid(alpha=.25)
    ax = axes[1, 0]
    x = np.arange(len(yr))
    ax.bar(x - 0.25, yr['V0_attack10'], 0.25, color='#c0504d', label='V0')
    ax.bar(x, yr['V2_blend1020'], 0.25, color='#e8a33d', label='V2')
    ax.bar(x + 0.25, yr['V1_core20'], 0.25, color='#1f4e79', label='V1')
    ax.set_xticks(x); ax.set_xticklabels(yr.index)
    ax.axhline(0, color='black', lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title('年度收益：2017 与 2018 是两个负年，2019 起转正')
    ax.legend(); ax.grid(alpha=.25, axis='y')
    ax = axes[1, 1]
    ax.plot(hold['持有天数'], hold['年化'] * 100, marker='o', color='#c0504d', label='固定持有 N 日')
    for _, r in jit.iterrows():
        ax.hlines(r['年化均值'] * 100, r['持有天数'] - 0.8, r['持有天数'] + 0.8,
                  color='#1f4e79', lw=3, alpha=.8)
    for _, row in hold.iterrows():
        ax.annotate(f"{row['年化']*100:.1f}", (row['持有天数'], row['年化'] * 100),
                    textcoords='offset points', xytext=(0, 7), fontsize=9)
    ax.set_xlabel('持有天数'); ax.set_ylabel('年化（%）')
    ax.set_title('持有邻域与±1天抖动均值（蓝线）：第10天是尖峰')
    ax.legend(fontsize=9); ax.grid(alpha=.25)
    fig.suptitle('主板“安静上涨”策略 2017-03→2026-10-23：数据修正后 + 预注册优化', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, '终图_三变体对比.png'), dpi=170)
    plt.close()

    # ---------- Excel ----------
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    head_font = Font(bold=True, color='FFFFFF')
    head_fill = PatternFill('solid', fgColor='1F4E79')

    def add_sheet(name, df, title=None, note=None):
        ws = wb.create_sheet(name)
        r0 = 1
        if title:
            ws.cell(1, 1, title).font = Font(bold=True, size=13)
            r0 = 3
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        for c in ws[r0]:
            if c.value is not None:
                c.font = head_font; c.fill = head_fill
                c.alignment = Alignment(horizontal='center')
        for col in ws.columns:
            width = max((len(str(c.value)) + sum('\u4e00' <= ch <= '\u9fff' for ch in str(c.value)) if c.value else 0)
                        for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
        if note:
            ws.cell(ws.max_row + 2, 1, note)
        return ws

    # 一页纸
    ws = wb.active
    ws.title = '一页纸'
    lines = [
        ('主板“安静上涨”策略：2016-10-13→2026-10-23 回测与优化（2026-10-24）', True),
        ('', False),
        ('数据：同花顺本地库（已同步至 2026-10-23）；4 只股票复权因子修订已用 baostock 分红对账验明；', False),
        ('外部核对：价格逐日一致、复权收益零差异、停牌日口径差已排除。回测信号起点 2017-03-01（80 交易日预热）。', False),
        ('', False),
        ('口径               年化     最大回撤   交易   最差年', False),
        ('原口径(2018起,旧数据) 27.69%   -24.59%    222   -21.99%(2018)', False),
        ('修正数据(2018起)     29.41%   -24.59%    223   -21.99%(2018)', False),
        ('全窗口(2017-03起)    24.53%   -36.32%    235   -22.46%(2018)，2017 年 -11.52%', False),
        ('±1 天抖动均值        15.61%   -40.10%(最差)', False),
        ('', False),
        ('预注册优化（9 变体同口径）胜者：V2 混合版 = attack10 + core20 各半资金', False),
        ('V2 混合版           21.15%   -26.75%    387   -16.27%(2018)；抖动均值 15.22%；0.35%成本 18.65%', False),
        ('V1 核心版(20日)     16.79%   -23.65%    152   -15.69%(2018)；抖动均值 15.19%', False),
        ('', False),
        ('今日（2026-10-23）：全A宽度 19.93%（需>50%），行业最高宽度 47%（需≥60%），0 信号，不买。', False),
        ('非投资建议。执行前先纸面跟踪。', False),
    ]
    for i, (txt, bold) in enumerate(lines, 1):
        c = ws.cell(i, 1, txt)
        if bold:
            c.font = Font(bold=True, size=13)
    ws.column_dimensions['A'].width = 110

    add_sheet('年度对照', yr, '年度收益（V0/V1/V2）')
    cols = ['变体', '全期年化', '全期回撤', '交易', '胜率', '盈亏比', '最差年', '负年份数',
            '训练年化', '检验年化', '抖动均值年化', '抖动最差年化', '成本035年化', '前五笔占比', '可行', 'maximin']
    add_sheet('优化对比', opt[cols], '预注册 9 变体（2017-03→2026-10-23，单边 0.25%）')
    add_sheet('持有邻域', hold, '持有天数邻域（进攻版）')
    add_sheet('持有抖动', jit, '持有 ±1 天抖动（10 种子）')
    add_sheet('稳健性_成本', pd.read_csv(os.path.join(OUT, '稳健_成本.csv')))
    add_sheet('稳健性_ST', pd.read_csv(os.path.join(OUT, '稳健_ST口径.csv')))
    add_sheet('稳健性_其他', pd.DataFrame([
        dict(项目='纯主板宽度闸门', 年化=0.1890, 最大回撤=-0.2564, 交易=205),
        dict(项目='随机排序均值(5种子)', 年化=0.1271, 最大回撤=np.nan, 交易=np.nan),
        dict(项目='起点2017-06', 年化=np.nan, 最大回撤=np.nan, 交易=np.nan),
    ]), '其他稳健性对照')
    add_sheet('深化审计', pd.read_csv(os.path.join(OUT, '深化_胜者审计.csv')), '月度/自举回撤/持仓')
    add_sheet('去前5笔', pd.read_csv(os.path.join(OUT, '深化_去前5笔.csv')), '剔除利润前 5 笔后精确重跑')
    add_sheet('R1进攻版指标', R1, '进攻版全窗口指标')
    add_sheet('信号与闸门按年', pd.read_csv(os.path.join(OUT, '信号与闸门_按年.csv')))
    # 拼接 2016 全年口径
    p1 = pd.read_csv(os.path.join(OUT, '拼接_P1_进攻版指标.csv'))
    p2 = pd.read_csv(os.path.join(OUT, '拼接_P2_V2混合版指标.csv'))
    p3 = pd.read_csv(os.path.join(OUT, '拼接_P3_核心版指标.csv'))
    p4 = pd.read_csv(os.path.join(OUT, '拼接_P4_抖动成本.csv'))
    add_sheet('拼接2016口径', pd.concat([p1, p2, p3], ignore_index=True),
              '真 2016 全年（2015-2026，baostock 前缀比率对齐拼接段，单独标注）')
    add_sheet('拼接2016抖动成本', p4)

    # 数据核对
    ver = pd.DataFrame([
        dict(检查项='OHLC 不复权逐日（baostock，206 只×全窗口，含全部成交股）', 结果='一致（0 条价格差异）'),
        dict(检查项='成交额', 结果='30 条 ≤8e-6 取整差异'),
        dict(检查项='复权日收益（baostock 后复权，60 只）', 结果='0 条 >0.3% 差异'),
        dict(检查项='交易日完备性（206 只，以 baostock 成交量>0 日为基准）', 结果='本地缺失 0 只 0 天；行数差全为停牌日口径'),
        dict(检查项='结构检查', 结果='OHLC 违规 0；重复行 0；年度交易日 242-244'),
        dict(检查项='复权因子修订（4 只）', 结果='补录 4 个遗漏分红，与 baostock 分红记录日期与比例全部吻合'),
        dict(检查项='门槛复现', 结果='原面板 27.685%/-24.594%/222 复现 PASS；v2 修正口径 29.539% PASS'),
        dict(检查项='新浪/东财第三方抽查', 结果='接口本环境不可达（ConnectionError），已如实记录；双源核对覆盖成交股全量'),
    ])
    add_sheet('数据核对', ver, '三层核对结论')

    wb.save(os.path.join(ROOT, '主板策略_2016_2026_回测优化.xlsx'))
    print('Excel 与图表完成')


if __name__ == '__main__':
    main()
