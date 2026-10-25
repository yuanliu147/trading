# -*- coding: utf-8 -*-
"""为每个信号日导出一份交易记录，并生成索引和说明。"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parents[1]
RES = OUT / '03_结果'
DAY = RES / '每信号日交易记录'
DAY.mkdir(parents=True, exist_ok=True)

C = pd.read_csv(RES / '起点汇总.csv',
                parse_dates=['signal_date', 'scheduled_entry',
                             'first_entry', 'last_exit'])
T = pd.read_csv(RES / '全部起点_逐笔.csv',
                parse_dates=['起点', 'signal_date', 'entry_date', 'exit_date'])
B = pd.read_csv(RES / '逐日首笔.csv', parse_dates=['signal_date'])

TR = T.copy()
TR['退出价(后复权)'] = TR['exit_fill'].fillna(TR['last_px'])
TR['退出原因'] = TR['exit_reason'].map(
    {'target': '+20%止盈', 'time': '15日到期', 'eod': '期末未平'}
).fillna(TR['exit_reason'])
TR = TR.rename(columns={
    '起点': '起点信号日', 'signal_date': '信号日', 'thscode': '代码',
    'entry_date': '入场日', 'exit_date': '退出日', 'days': '持有交易日',
    'entry_px': '入场价(后复权)', 'rk': '信号排名',
    'rsi14': '相对强弱14', 'close_location': '收盘位置',
})
TR['分配额(初始本金%)'] = TR['alloc'] * 100
TR['仓位占当时权益(%)'] = TR['alloc_frac'] * 100
TR['单笔收益(%)'] = TR['ret'] * 100
TR['盈亏贡献(初始本金%)'] = TR['alloc'] * TR['ret'] * 100
TR['250日回撤(%)'] = TR['drawdown250'] * 100
TR['市场20日(%)'] = TR['market_return20'] * 100
TR = TR[['起点信号日', '信号日', '代码', '入场日', '退出日', '退出原因',
         '持有交易日', '入场价(后复权)', '退出价(后复权)',
         '分配额(初始本金%)', '仓位占当时权益(%)', '单笔收益(%)',
         '盈亏贡献(初始本金%)', '信号排名', '250日回撤(%)',
         '相对强弱14', '收盘位置', '市场20日(%)']]
TR = TR.sort_values(['起点信号日', '入场日', '代码']).reset_index(drop=True)

idx_rows = []
for _, r in C.iterrows():
    d = pd.Timestamp(r.signal_date)
    name = d.strftime('%Y-%m-%d') + '.csv'
    q = TR[TR['起点信号日'] == d].copy()
    if len(q):
        q.to_csv(DAY / name, index=False, encoding='utf-8-sig')
    else:
        # 无成交/尚未入场：也留一个文件，写状态，保证 251 个信号日都有记录
        status = pd.DataFrame([{
            '状态': '尚未入场/无交易',
            '起点信号日': d.strftime('%Y-%m-%d'),
            '当日信号数': int(r.n_signals),
            '计划入场日': (pd.Timestamp(r.scheduled_entry).strftime('%Y-%m-%d')
                           if pd.notna(r.scheduled_entry) else ''),
            '说明': ('当日仅 1 只信号，尚未到入场日/数据不足'
                     if int(r.n_signals) == 1 else '数据截止前无成交'),
        }])
        status.to_csv(DAY / name, index=False, encoding='utf-8-sig')
    idx_rows.append({
        '起点信号日': d.strftime('%Y-%m-%d'),
        '当日信号数': int(r.n_signals),
        '笔数': int(r.n),
        '首笔入场日': (pd.Timestamp(r.first_entry).strftime('%Y-%m-%d')
                       if pd.notna(r.first_entry) else ''),
        '末笔退出日': (pd.Timestamp(r.last_exit).strftime('%Y-%m-%d')
                       if pd.notna(r.last_exit) else ''),
        '计划入场日': (pd.Timestamp(r.scheduled_entry).strftime('%Y-%m-%d')
                       if pd.notna(r.scheduled_entry) else ''),
        '期末倍数': float(r.final),
        '总收益(%)': float(r.total_ret) * 100,
        '全期年化(%)': (float(r.cagr) * 100 if pd.notna(r.cagr) else np.nan),
        '1年收益(%)': (float(r['ret_1年']) * 100
                       if pd.notna(r['ret_1年']) else np.nan),
        '3年收益(%)': (float(r['ret_3年']) * 100
                       if pd.notna(r['ret_3年']) else np.nan),
        '5年收益(%)': (float(r['ret_5年']) * 100
                       if pd.notna(r['ret_5年']) else np.nan),
        '样本': r['样本'],
        '文件名': name,
    })
idx = pd.DataFrame(idx_rows)
idx.to_csv(DAY / '00_索引.csv', index=False, encoding='utf-8-sig')

readme = f"""# 每信号日交易记录

> 251 个信号日，每个信号日一份交易记录。  
> 全部交易来自 `03_结果/全部起点逐笔.csv`，按 `起点信号日` 拆分；不是重新回测。

## 文件命名

- `YYYY-MM-DD.csv`：以该信号日为起点，之后所有交易的逐笔记录。
- 例：`2018-04-23.csv` 是 2018-04-23 起点从 10 万开始、一直跑到 2026-10-23 的全部 93 笔。
- `00_索引.csv`：251 个信号日的笔数、期末倍数、年化、1/3/5 年、文件名。
- 两个无成交日（2026-10-22、2026-10-23）也有文件，内容是“尚未入场/无交易”状态，保证 251 天齐全。

## 列

| 列 | 含义 |
|---|---|
| 起点信号日 | 你开始的那天 |
| 信号日 | 这笔交易对应的信号日（可能晚于起点） |
| 代码 | 股票代码 |
| 入场日 / 退出日 | 信号后第3个交易日开盘入场；+20%止盈、15日到期或期末 |
| 退出原因 | +20%止盈 / 15日到期 / 期末未平 |
| 持有交易日 | 1–15；期末未平可能小于15 |
| 入场价/退出价(后复权) | 后复权价；不是不复权实际价 |
| 分配额(初始本金%) | 该笔投入 / 10万；复利后可能超过100% |
| 仓位占当时权益(%) | 该笔投入 / 当时账户权益，≤50% |
| 单笔收益(%) | 扣0.175%/边 |
| 盈亏贡献(初始本金%) | 分配额 × 单笔收益；不是账户总收益的简单加总 |
| 250日回撤/相对强弱14/收盘位置/市场20日 | 信号日指标 |

## 汇总

- 251 个文件；有交易的 249 个，共 {len(TR):,} 笔。
- 无成交 2 个：2026-10-22、2026-10-23（尚未入场）。
- 2026-10-14/15/16 是持有未满，退出原因“期末未平”，单笔收益是截至 2026-10-23 的浮动，不是最终。
- 起点汇总的 `总利润` 和 `期末倍数` 是账户级；逐笔的 `盈亏贡献` 不能直接相加等于账户总收益（复利、现金、仓位变化）。

## 怎么查

1. 打开 `00_索引.csv`，按 `起点信号日` 找文件名。
2. 打开对应 `YYYY-MM-DD.csv`，就是该日起的全部交易。
3. 或直接打开 `../不同信号日起点回测.xlsx` 的 `全部起点逐笔`，按 `起点信号日` 筛选。

> 历史回测不代表未来；不构成投资建议。
"""
(DAY / 'README_每信号日交易记录.md').write_text(readme, encoding='utf-8')

# zip
zip_path = RES / '每信号日交易记录.zip'
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', DAY)

print('date files', len(list(DAY.glob('20*.csv'))), '| index/readme/zip')
print('index rows', len(idx), 'trades', len(TR), 'sum n', int(C.n.sum()))
print('zip', zip_path)
