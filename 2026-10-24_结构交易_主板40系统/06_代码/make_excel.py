# -*- coding: utf-8 -*-
"""生成全中文 Excel 数据表（实际价复核版）。"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, '05_证据')
XLSX = os.path.join(ROOT, '结构交易_主板四成_数据表.xlsx')


def read(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p, encoding='utf-8-sig') if os.path.exists(p) else pd.DataFrame()


def trades(name):
    rich = name.replace('_逐笔.csv', '_逐笔_含实际价.csv')
    d = read(rich) if os.path.exists(os.path.join(OUT, rich)) else read(name)
    return d.rename(columns={
        'code': '代码', 'entry_date': '买入日', 'exit_date': '卖出日',
        'entry': '后复权买入价', 'exit': '后复权卖出价',
        '实际开盘价': '实际买入价', '实际收盘价': '实际卖出价',
        '前复权开盘价': '前复权买入价', '前复权收盘价': '前复权卖出价',
        'ret': '后复权毛收益', '实际毛收益': '实际毛收益', '前复权毛收益': '前复权毛收益',
        'reason': '退出原因', 'hold': '持有天数', 'shares': '股数',
        'pnl': '净盈亏', 'weight': '仓位', 'net_ret': '净收益率',
        'year': '年份', 'board': '板块',
        'entry_actual': '实际买入价', 'exit_actual': '实际卖出价'})


def annual_from_equity(name, start=20000.0):
    d = read(name)
    if len(d) == 0:
        return d
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date')
    rows = []
    prev = start
    for y in range(2018, 2027):
        x = d.loc[str(y)]
        if len(x) == 0:
            continue
        rows.append({'年份': y, '收益': x.equity.iloc[-1] / prev - 1,
                     '年末权益': x.equity.iloc[-1]})
        prev = x.equity.iloc[-1]
    return pd.DataFrame(rows)


def audit_table():
    d = read('市值缺失审计.csv')
    if len(d) == 0:
        return d
    d = d.rename(columns={
        'n': '笔数', 'miss_n': '无市值笔数', 'miss_share': '无市值笔数占比',
        'miss_pnl': '无市值净盈亏', 'total_pnl': '总净盈亏',
        'miss_pnl_share': '无市值利润占比', '版本': '版本',
        '年化': '年化', '最大回撤': '最大回撤', '交易': '交易'})
    d['版本'] = d['版本'].replace({
        '原mcap_不要求市值_核心': '原市值排序_不要求市值_核心',
        '原mcap_要求市值_核心': '原市值排序_要求市值_核心',
        '原mcap_不要求市值_进攻': '原市值排序_不要求市值_进攻',
        '原mcap_要求市值_进攻': '原市值排序_要求市值_进攻',
        '修正amt_核心': '修正成交额排序_核心',
        '修正amt_进攻': '修正成交额排序_进攻'})
    cols = ['版本', '笔数', '无市值笔数', '无市值笔数占比', '无市值净盈亏',
            '总净盈亏', '无市值利润占比', '年化', '最大回撤', '交易']
    return d[[c for c in cols if c in d.columns]]


def st_table():
    d = read('ST口径_AMT.csv')
    if len(d) == 0:
        return d
    d = d.rename(columns={
        'ST口径': '风险警示口径', '无杠杆年化': '无杠杆年化',
        '无杠杆回撤': '无杠杆回撤', '杠杆年化': '融资年化',
        '杠杆回撤': '融资回撤', '交易': '交易'})
    d['风险警示口径'] = d['风险警示口径'].replace({
        '基准STX': '基准口径', '1次5%触板': '一次触板',
        '2次5%触板': '两次触板', '更严STY': '更严口径'})
    return d


def lev_old():
    d = read('融资杠杆表_0.15成本.csv')
    if len(d) == 0:
        return d
    if '最低抵押率' in d.columns:
        d['最低标准担保'] = d['最低抵押率'] + 1
        d = d.drop(columns=['最低抵押率'])
    return d.rename(columns={'杠杆': '杠杆倍数', '年化': '年化',
                             '最大回撤': '最大回撤', '期末权益': '期末权益'})


def actual_compare():
    a = read('实际价过滤_对照.csv')
    b = read('实际价成交_对照.csv')
    if len(a):
        a = a.rename(columns={'价格口径': '口径'})
    d = pd.concat([a, b], ignore_index=True)
    if len(d) == 0:
        return d
    d = d.rename(columns={'最低抵押率': '最低标准担保'})
    cols = ['版本', '口径', '年化', '最大回撤', '交易', '胜率', '盈亏比',
            '期末权益', '最低标准担保', '2018年', '2020年', '2023年', '2026年']
    return d[[c for c in cols if c in d.columns]]


def main():
    note = pd.DataFrame([
        ['文件用途', '结构交易·主板策略的复核修正版数据表；所有收益均为历史回测，不是收益承诺。'],
        ['股票池', '沪深主板：沪 600/601/603/605；深 000/001/002/003；排除创业板、科创板、北交所、风险警示样态。'],
        ['市场口径', '市场宽度与行业宽度用全市场数据（含创业板、科创板、北交所）；实际下单只买主板。'],
        ['价格口径', '2–15 元过滤、100 股整手、成交/费用用不复权实际价；信号/均线/平均真实波幅/止损用后复权。'],
        ['成本口径', '核心/进攻为单边 0.25%；四成版为单边 0.15% + 1.5–2.0 倍融资。'],
        ['重要复核一', '市值字段缺失 79.46%；原“市值升序”实际是“无市值优先”，初版 40.59% 已作废。'],
        ['重要复核二', '后复权价会随分红逐年抬高，不能做固定 2–15 元过滤；实际价 1.5 倍 35.67%，1.9 倍 42.02%/40.12% 但回撤 -51.6%。'],
        ['当前结论', '实际价口径：核心 8.93%、进攻 23.96%、1.5 倍 35.67%；40% 只在 1.9–2.0 倍、0.15% 成本、基准风险警示口径、精确第 10 日同时成立。'],
        ['仓位口径', '1.9 倍 42.02% 假设“单票 50% 总资产”，即单票 95% 净资产、两只 190% 净资产；若按“单票 50% 净资产”，借来的钱闲置，1.9 倍约 20%，40% 不成立。'],
        ['今日信号', '2026-10-22：全市场宽度 11.71%，闸门关，0 信号。'],
    ], columns=['项目', '说明'])

    one = pd.DataFrame([
        ['结论', '看盘框架可确定；稳定四成年化不成立；40% 只在 1.9–2.0 倍、0.15% 成本、精确第 10 日同时成立。'],
        ['复核一', '市值字段缺失 79.46%；原“市值升序”是“无市值优先”。'],
        ['复核二', '2–15 元过滤必须用实际价；后复权价逐年漂移，会无故排除股票。'],
        ['后复权口径', '核心 20.59%/-18.37%；进攻 27.69%/-24.59%；四成 41.94%/-36.01%。'],
        ['实际价口径', '核心 8.93%/-21.57%；进攻 23.96%/-29.35%；四成 1.5 倍 35.67%/-42.61%；1.9 倍 42.02%/40.12%/-51.59%。'],
        ['四成版压力', '1.9倍：0.25%成本 36.16%；风险警示严格 33.92%；持有偏1天均值 22.15%；1.5倍 35.67%（0.25%成本 31.14%）。'],
        ['仓位口径', '42% 假设单票50%总资产（1.9倍时95%净资产）；若单票50%净资产，1.9倍约20%，40%不成立。'],
        ['今日', '全市场宽度 11.71%，闸门关，0 信号。'],
    ], columns=['项目', '内容'])

    rules = pd.DataFrame([
        ['框架', '全市场宽度20>50%；指数在60日均线上方且60日均线向上；行业宽度≥60%、行业20日>5%'],
        ['结构', '收盘在20周均线上方；20周均线在60周均线上方；60周均线向上；价格在周结构线上方'],
        ['趋势', '收盘在60日均线上方；20日均线在60日均线上方；60日均线高于20日前；年龄10-30；平均真实波幅≤4.4%；20日无涨停；距60日高点≥-15%；20日均成交额3000-7000万；实际价2-15；20日波动≥20%'],
        ['排序', '20日均成交额升序（不用市值）'],
        ['仓位', '核心2×45%×波动；进攻2×50%；风险预算≤10%'],
        ['卖出', '核心20日/进攻10日到期；+25%卖一半；+10%保本；+25%后10%移动止盈；-20%灾难止损'],
        ['融资', '四成版1.5–2.0倍，季度再平衡，年利率6.5%；标准维持担保比例=总资产/负债，>160%'],
        ['价格', '选股用不复权实际价；收益/信号用后复权；看图用前复权'],
    ], columns=['环节', '规则'])

    now = pd.DataFrame([
        ['全市场宽度20', '11.71%', '>50%', '关'],
        ['行业最高宽度', '银行28.6%', '≥60%', '关'],
        ['行业20日最高', '传媒+1.6%', '>5%', '关'],
        ['有效信号', '0', '—', '不买'],
    ], columns=['项目', '今天', '要求', '判定'])

    sheets = {
        '说明': note,
        '一页纸': one,
        '规则': rules,
        '当前信号': now,
        '实际价对照': actual_compare(),
        '核心年度': read('核心_AMT_年度.csv'),
        '进攻年度': read('进攻_AMT_年度.csv'),
        '四成版年度': read('进攻40_1.5倍_年度.csv'),
        '核心年度实际价': annual_from_equity('核心_实际价成交_权益.csv'),
        '进攻年度实际价': annual_from_equity('进攻_实际价成交_权益.csv'),
        '四成年度实际价': annual_from_equity('四成_实际价成交_1.5倍_权益.csv'),
        '市值审计': audit_table(),
        '融资杠杆后复权': lev_old(),
        '融资杠杆实际价': read('融资杠杆表_实际价.csv').rename(columns={
            '杠杆': '杠杆倍数', '年化252': '年化252', '年化自然日': '年化自然日',
            '最大回撤': '最大回撤', '最低标准担保': '最低标准担保', '期末权益': '期末权益'}),
        '成本敏感性后复权': read('成本敏感性_AMT.csv'),
        '成本敏感性实际价': read('成本敏感性_实际价.csv'),
        '持有期邻域后复权': read('持有期邻域_AMT.csv'),
        '持有期邻域实际价': read('持有期邻域_实际价.csv'),
        '持有期抖动后复权': read('持有期抖动_AMT.csv'),
        '持有期抖动实际价': read('持有期抖动_实际价.csv'),
        '风险警示后复权': st_table(),
        '风险警示实际价': read('风险警示口径_实际价.csv'),
        '核心逐笔': trades('核心_AMT_逐笔.csv'),
        '进攻逐笔': trades('进攻_AMT_逐笔.csv'),
        '四成版逐笔': trades('进攻40_AMT_逐笔.csv'),
        '核心逐笔实际价': trades('核心_实际价成交_逐笔.csv'),
        '进攻逐笔实际价': trades('进攻_实际价成交_逐笔.csv'),
        '四成逐笔实际价': trades('四成_实际价成交_逐笔.csv'),
        '核心独立仓位': read('核心_AMT_独立仓位.csv'),
        '进攻独立仓位': read('进攻_AMT_独立仓位.csv'),
        '四成版独立仓位': read('进攻40_AMT_独立仓位.csv'),
    }

    with pd.ExcelWriter(XLSX, engine='openpyxl') as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.freeze_panes = 'A2'
            for col in ws.columns:
                width = max(10, min(42, max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2))
                ws.column_dimensions[col[0].column_letter].width = width
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
                cell.fill = cell.fill.copy(fgColor='DDEBF7', fill_type='solid')
            # 百分比格式
            for j, col in enumerate(df.columns, start=1):
                if any(k in str(col) for k in ['收益', '回撤', '占比', '年化', '收益率', '成本', '仓位', '胜率', '担保']):
                    for i in range(2, len(df) + 2):
                        try:
                            ws.cell(row=i, column=j).number_format = '0.00%'
                        except Exception:
                            pass
    print('written', XLSX)


if __name__ == '__main__':
    main()
