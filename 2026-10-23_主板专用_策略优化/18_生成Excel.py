# -*- coding: utf-8 -*-
"""生成《主板专用_策略表.xlsx》：中文表头、中文 sheet 名。"""
import os
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

DIR = '/Users/admin/WorkSpace/trading/2026-10-23_主板专用_策略优化'
OUT = os.path.join(DIR, '主板专用_策略表.xlsx')
NAMES = '/Users/admin/WorkSpace/trading/2026-10-22_持仓天数_涨跌路径/股票名称_2026-10-22.csv'


def rd(name, **kw):
    return pd.read_csv(os.path.join(DIR, name), encoding='utf-8-sig', **kw)


def style_sheet(ws, pct_cols=(), width_max=28):
    header_fill = PatternFill('solid', fgColor='1F4E79')
    header_font = Font(color='FFFFFF', bold=True, size=10)
    thin = Side(style='thin', color='D9D9D9')
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.freeze_panes = 'A2'
    ws.row_dimensions[1].height = 32
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        maxlen = 8
        for cell in col:
            if cell.value is not None:
                maxlen = max(maxlen, min(width_max, len(str(cell.value)) + 2))
            cell.border = Border(bottom=thin)
        ws.column_dimensions[letter].width = maxlen
    for c in pct_cols:
        if c in [cell.value for cell in ws[1]]:
            idx = [cell.value for cell in ws[1]].index(c) + 1
            for row in ws.iter_rows(min_row=2, min_col=idx, max_col=idx):
                for cell in row:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00%'


def main():
    # ---------- 数据 ----------
    annual = rd('13_F2_年度.csv')
    monthly = rd('13_F2_月度.csv')
    trades = rd('13_F2_逐笔.csv', parse_dates=['entry_date', 'exit_date'])
    hold = rd('13_F2_持仓特征.csv', parse_dates=['entry'])
    stress = rd('13_F2_压力.csv')
    status = rd('15_今日F2状态_2026-10-22.csv')
    indgate = rd('15_今日F2_行业闸门.csv')
    near = rd('15_今日F2近信号_2026-10-22.csv')
    cand = rd('03_候选组合.csv')
    grid1 = rd('02_单因素网格_非ST.csv')
    sens = rd('04_核心敏感性.csv')
    agehold = rd('05_年龄区间_持有_分批.csv')
    execu = rd('08_执行与悬崖.csv')
    jitter = rd('09_路径扰动.csv')
    hold20 = rd('11_持有20_稳健网格.csv')
    final = rd('12_最终候选对比.csv')
    walk = rd('14_F2族_走步选择.csv')
    rand = rd('16_F2_随机排序.csv')
    trail = rd('17_F2_移动止盈.csv')

    names = pd.read_csv(NAMES, encoding='utf-8-sig').set_index('thscode')['name']
    trades['名称'] = trades.code.map(names)
    trades = trades.rename(columns={
        'code': '代码', 'entry_date': '买入日', 'exit_date': '卖出日', 'entry': '买入价',
        'exit': '卖出价', 'ret': '毛收益', 'reason': '退出原因', 'hold': '持有天数',
        'shares': '股数', 'pnl': '净盈亏', 'weight': '账户权重', 'net_ret': '净收益率',
        'year': '年份', 'board': '板块'})
    trades = trades[['代码', '名称', '板块', '买入日', '卖出日', '买入价', '卖出价',
                     '毛收益', '退出原因', '持有天数', '股数', '净盈亏', '账户权重',
                     '净收益率', '年份']]

    hold = hold.rename(columns={
        'code': '代码', 'entry': '买入日', 'c5': '60日5%触板次数', 'maxabs60': '60日最大绝对日收益',
        'price': '后复权收盘', 'amt': '20日均成交额_百万', 'mcap': '市值_亿',
        'pnl': '净盈亏'})

    stress = stress.rename(columns={
        'cfg': '口径', '全期_cagr': '全期年化', '全期_maxdd': '最大回撤',
        '全期_trades': '交易数', '全期_win': '退出记录胜率', '全期_pf': '盈亏比',
        '检验_cagr': '2022-2026年化'})

    # ---------- 候选对比 ----------
    rows = [
        dict(口径='全板块 70/15（原基准）', 年化=0.3030, 最大回撤=-0.1964, 交易数=140,
             胜率=0.6214, 胜率口径='交易记录', 盈亏比=2.9653, 负年份=2, 说明='用了创业板/科创板，账户买不了；2025 主要靠非主板'),
        dict(口径='原主板 70/15', 年化=0.1771, 最大回撤=-0.2981, 交易数=139,
             胜率=0.5755, 胜率口径='交易记录', 盈亏比=1.8577, 负年份=4, 说明='含 ST 样态 30 笔，贡献 33% 净利润'),
        dict(口径='主板非ST基线', 年化=0.1233, 最大回撤=-0.2684, 交易数=137,
             胜率=0.5328, 胜率口径='交易记录', 盈亏比=1.5816, 负年份=3, 说明='剔 ST 后的真基线；前5笔占93.5%'),
        dict(口径='C1 固定10日（激进）', 年化=0.3092, 最大回撤=-0.1557, 交易数=226,
             胜率=0.5973, 胜率口径='退出记录', 盈亏比=2.7864, 负年份=1, 说明='路径脆弱；持有9/11/12日只有10.6/22.0/10.4；抖动均值21.7'),
        dict(口径='F2 最终策略', 年化=0.1496, 最大回撤=-0.2114, 交易数=146,
             胜率=0.5276, 胜率口径='独立仓位', 盈亏比=1.9499, 负年份=1, 说明='修复口径（全覆盖市值）；20日持有；分批25+保本10+移动25/10；退出记录胜率54.1%'),
        dict(口径='F2 走步（2021-2026）', 年化=0.2227, 最大回撤=-0.1534, 交易数=np.nan,
             胜率=np.nan, 胜率口径='—', 盈亏比=np.nan, 负年份=1, 说明='每年只用过去3年选参数；更诚实的预期'),
        dict(口径='F2 成本0.4%', 年化=0.1202, 最大回撤=-0.2211, 交易数=146,
             胜率=0.5274, 胜率口径='退出记录', 盈亏比=1.7225, 负年份=1, 说明='单边0.4%压力'),
        dict(口径='F2 ST严（1次5%触板就排）', 年化=0.1838, 最大回撤=-0.2134, 交易数=145,
             胜率=0.5793, 胜率口径='退出记录', 盈亏比=2.1351, 负年份=1, 说明='ST代理更保守'),
        dict(口径='F2 随机排序（30次均值）', 年化=0.1122, 最大回撤=np.nan, 交易数=146,
             胜率=np.nan, 胜率口径='—', 盈亏比=np.nan, 负年份=np.nan, 说明='过滤条件本身约11%；F2 15.0 在83分位；mcap排序贡献有限'),
        dict(口径='20日简单版（mcap，不保本）', 年化=0.0940, 最大回撤=-0.2259, 交易数=117,
             胜率=0.4957, 胜率口径='交易记录', 盈亏比=1.5496, 负年份=1, 说明='只做市值升序、不用分批/保本；修复后只有9.4%'),
        dict(口径='20日简单版（amt，不保本）', 年化=0.1626, 最大回撤=-0.2480, 交易数=117,
             胜率=0.5726, 胜率口径='交易记录', 盈亏比=2.3059, 负年份=1, 说明='成交额升序版本；比纯市值升序更强'),
    ]
    comp = pd.DataFrame(rows)

    # ---------- 参数悬崖 ----------
    pr = []

    def add(dim, variant, cagr, dd, n, win, test, neg, note=''):
        pr.append(dict(维度=dim, 变体=variant, 年化=cagr, 最大回撤=dd, 交易数=n,
                       胜率=win, 检验年化=test, 负年份=neg, 说明=note))

    # 持有：同一底座（C1，age9-25/amt/分批25）固定 8–12 日 + 20 日
    for h in [8, 9, 10, 11, 12]:
        x = execu[execu.cfg == f'持有{h}日']
        if len(x):
            r = x.iloc[0]
            add('持有天数', f'固定{h}日', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'],
                '固定10日是路径峰值；9/11/12日会崩')
    x = jitter[(jitter.hold == 20) & (jitter.jitter == 0)]
    if len(x):
        r = x.iloc[0]
        add('持有天数', '固定20日', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
            r['全期_win'], r['检验_cagr'], r['负年份数'], '同一底座；20日更稳')
    j = final[(final.variant == 'F2 10-30 lock10') & (final['mode'] == 'jitter')]
    add('持有抖动', 'F2 持有20日 ±1天（10次均值）', j['全期_cagr'].mean(), j['全期_maxdd'].mean(),
        j['全期_trades'].mean(), j['全期_win'].mean(), j['检验_cagr'].mean(),
        j['负年份数'].mean(), f"最差回撤 {j['全期_maxdd'].min():.2%}；最差年化 {j['全期_cagr'].min():.2%}")
    # 年龄区间（F2族：mcap/pt25/lock10/no ext）
    for amin, amax in [(9, 25), (10, 25), (10, 30), (9, 30)]:
        name = f'{amin}-{amax} mcap pt0.25 lock0.1 extFalse'
        x = hold20[hold20.name == name]
        if len(x):
            r = x.iloc[0]
            add('年龄区间', f'{amin}-{amax}日', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], 'F2 固定其余条件')
    # 波动（核心：age10-30, DD60-15, 无lock/partial）
    for v in [0.18, 0.20, 0.22, 0.25]:
        x = sens[sens.cfg == f'波动>={v:.0%}']
        if len(x):
            r = x.iloc[0]
            add('20日波动', f'>={v:.0%}', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], '核心口径（未加lock/partial）')
    # 分批
    for pt in [0.20, 0.25, 0.30]:
        name = f'10-30 mcap pt{pt} lock0.1 extFalse'
        x = hold20[hold20.name == name]
        if len(x):
            r = x.iloc[0]
            add('分批止盈', f'涨{pt:.0%}卖一半', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], 'F2 固定其余条件')
    # 保本
    for lock in [0.10, 0.15]:
        name = f'10-30 mcap pt0.25 lock{lock} extFalse'
        x = hold20[hold20.name == name]
        if len(x):
            r = x.iloc[0]
            add('保本触发', f'浮盈{lock:.0%}保本', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], 'F2 固定其余条件')
    # 排序
    for rank in ['amt', 'mcap']:
        name = f'10-30 {rank} pt0.25 lock0.1 extFalse'
        x = hold20[hold20.name == name]
        if len(x):
            r = x.iloc[0]
            add('排序', rank, r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], 'mcap=市值升序；amt=成交额升序')
    # DD60
    for dd in [-0.08, -0.12, -0.15, -0.18]:
        x = sens[sens.cfg == f'DD60>={dd:.0%}']
        if len(x):
            r = x.iloc[0]
            add('DD60下限', f'>={dd:.0%}', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], '核心口径')
    # 闸门
    for g in [0.45, 0.50, 0.55, 0.60]:
        x = sens[sens.cfg == f'宽度>{g:.0%}']
        if len(x):
            r = x.iloc[0]
            add('市场闸门', f'宽度>{g:.0%}', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], '核心口径')
    # 止损
    for s in [0.10, 0.15, 0.20, 0.25]:
        x = sens[sens.cfg == f'止损{s:.0%}']
        if len(x):
            r = x.iloc[0]
            add('止损', f'{s:.0%}', r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], '核心口径')
    # 移动止盈
    for tt, tp, label in [(0.25, 0.05, '25%/5%'), (0.25, 0.10, '25%/10%'),
                          (0.30, 0.10, '30%/10%'), (9.0, 0.10, '不做移动止盈')]:
        x = trail[(trail.tt == tt) & (trail.tp == tp)]
        if len(x):
            r = x.iloc[0]
            add('移动止盈', label, r['全期_cagr'], r['全期_maxdd'], r['全期_trades'],
                r['全期_win'], r['检验_cagr'], r['负年份数'], 'F2 固定其余条件')
    # 成本
    for name in ['F2', '成本0.20%', '成本0.30%', '成本0.40%', '成本0.50%']:
        x = stress[stress['口径'] == name]
        if len(x):
            r = x.iloc[0]
            add('成本', name, r['全期年化'], r['最大回撤'], r['交易数'], r['退出记录胜率'],
                r['2022-2026年化'], 1, '单边成本')
    # ST
    for name in ['ST=one', 'ST=two', 'ST=y']:
        x = stress[stress['口径'] == name]
        if len(x):
            r = x.iloc[0]
            add('ST口径', name, r['全期年化'], r['最大回撤'], r['交易数'], r['退出记录胜率'],
                r['2022-2026年化'], 1, 'one=1次5%触板即排；y=更严')
    cliff = pd.DataFrame(pr)

    # ---------- 过拟合 ----------
    over = pd.DataFrame([
        dict(项目='F2 全期年化', 数值=0.1496, 说明='2018-01 至 2026-10-13；修复口径'),
        dict(项目='F2 最大回撤', 数值=-0.2114, 说明='收盘口径；盘中 -22.06%'),
        dict(项目='F2 年化夏普', 数值=0.8234, 说明='日收益年化'),
        dict(项目='F2 偏度', 数值=0.550, 说明='右偏'),
        dict(项目='F2 峰度', 数值=8.972, 说明='厚尾'),
        dict(项目='走步 2021-2026 年化', 数值=0.2227, 说明='每年只用过去3年选参数'),
        dict(项目='走步 2021-2026 回撤', 数值=-0.1534, 说明='拼接样本外'),
        dict(项目='固定 F2 2021-2026 年化', 数值=0.2022, 说明='同期对照（-19.24%）'),
        dict(项目='PBO 全族192', 数值=0.3050, 说明='修复后下降；仍提示可调参数多'),
        dict(项目='DSR', 数值=0.7165, 说明='192次试验；全程试验数更多，视作上界'),
        dict(项目='随机排序均值', 数值=0.1122, 说明='30次；F2 15.0%在83分位'),
        dict(项目='F2 2022-2026夏普排名', 数值=0.573, 说明='在192个候选中的分位'),
        dict(项目='1年自举回撤P5', 数值=-0.2533, 说明='10日块，3000条'),
        dict(项目='3年自举回撤P5', 数值=-0.3524, 说明='10日块，3000条'),
    ])

    # ---------- 写 Excel ----------
    with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
        # 一页纸
        sheet = pd.DataFrame([
            ('结论', '原主板17.7%含ST；非ST真基线12.3%/-26.8%。F2 修复口径 15.0%/-21.1%，只有2018负。'),
            ('诚实预期', '走步22.3%、抖动15.2%、成本0.4%后12.0%；规划12%-16%，3年尾部-35%。'),
            ('今天', '2026-10-22 宽度11.71%，闸门关；行业宽度最高28.6%，F2可买0。'),
            ('市场', '全A宽度20>50%；不到0买入。'),
            ('股票池', '主板600/601/603/605 + 000/001/002/003；排除ST。'),
            ('个股', '年龄10-30；ATR<=4.4%；20日无涨停；DD60>=-15%；成交额3000-7000万；价2-15；20日波动>=20%。'),
            ('行业', '申万一级宽度>=60%；行业20日>5%。'),
            ('买入', '市值升序（全覆盖市值）；次日开盘；高开>5%/一字/停牌不买；2槽45%×波动系数；热预算10%。'),
            ('卖出', '20日收盘；涨25%卖一半；浮盈10%保本；浮盈25%后10%移动止盈；-20%灾难止损。'),
            ('胜率', '52.76% 独立仓位；54.11% 退出记录；平均盈利+14.20%，平均亏损-5.73%。'),
            ('最大风险', '小盘风格反转（市值中位21.7亿）；右尾集中；3年自举P5 -35%；最大单笔-22.6%。'),
            ('边界', 'ST代理不完美；面板退市不全；无实盘记录；不是投资建议。'),
            ('修正', '2026-10-24 补齐全覆盖市值并修复排序哨兵；旧 25.3% 为数据残缺假象。'),
        ], columns=['项目', '内容'])
        sheet.to_excel(writer, sheet_name='一页纸', index=False)
        style_sheet(writer.sheets['一页纸'], width_max=110)

        # 最终规则
        rules = pd.DataFrame([
            ('股票池', '主板', '沪600/601/603/605；深000/001/002/003'),
            ('股票池', '排除', '创业板300/301、科创板688/689、北交所8/4/920、ST'),
            ('市场', '宽度20', '>50%'),
            ('个股', '上升结构', '收盘>MA60，MA20>MA60，MA60高于20日前'),
            ('个股', '趋势年龄', '10–30个交易日'),
            ('个股', 'ATR14/收盘', '<=4.4%'),
            ('个股', '20日涨停', '=0次'),
            ('个股', 'DD60', '>=-15%'),
            ('个股', '20日均成交额', '3000万–7000万'),
            ('个股', '后复权收盘', '2–15元'),
            ('个股', '20日年化波动', '>=20%'),
            ('行业', '申万一级宽度', '>=60%'),
            ('行业', '行业20日收益', '>5%'),
            ('买入', '排序', '市值升序'),
            ('买入', '时点', '信号次日开盘'),
            ('买入', '不买', '高开>5%、一字涨停、停牌'),
            ('买入', '仓位', '2槽；单票45%×min(1,20%/市场20日波动)；热预算<=10%'),
            ('卖出', '到期', '第20个交易日收盘'),
            ('卖出', '分批', '盘中+25%卖一半'),
            ('卖出', '保本', '盘中最高浮盈>=10%，止损上移到成本'),
            ('卖出', '移动止盈', '盘中最高浮盈>=25%，止损=最高价×0.90'),
            ('卖出', '灾难止损', '-20%'),
            ('成本', '单边', '0.15%，最低5元；100股整手'),
        ], columns=['模块', '规则', '参数'])
        rules.to_excel(writer, sheet_name='最终规则', index=False)
        style_sheet(writer.sheets['最终规则'], width_max=70)

        # 年度
        base = cand[cand.cfg == '非ST基线'].iloc[0]
        orig = pd.read_csv(os.path.join('/Users/admin/WorkSpace/trading/2026-10-22_持仓天数_涨跌路径', '14_主板_年度.csv'), encoding='utf-8-sig')
        orig = orig[orig['口径'] == '主板 70/15'].set_index('年份')['收益']
        ann = annual.copy()
        ann['非ST基线'] = [base[f'{int(y)}年'] for y in ann['年份']]
        ann['原主板70/15'] = [orig.get(int(y), np.nan) for y in ann['年份']]
        ann = ann.rename(columns={'年份': '年份', '笔数': '笔数', '净盈亏': '净盈亏',
                                  '胜率': '退出记录胜率', '仓位胜率': '仓位胜率', '净均值': '净均值',
                                  '最大单笔盈利': '最大单笔盈利', '最大单笔亏损': '最大单笔亏损',
                                  '非ST基线': '非ST基线年化', '原主板70/15': '原主板年化'})
        # 加 F2 年化（从权益曲线现算，避免手填）
        E = pd.read_csv(os.path.join(DIR, '13_F2_权益.csv'), parse_dates=['date']).set_index('date')
        f2y, prev = {}, E.equity.iloc[0]
        for y in range(2018, 2027):
            x = E.equity.loc[str(y)]
            if len(x):
                f2y[y] = x.iloc[-1] / prev - 1
                prev = x.iloc[-1]
        ann.insert(1, 'F2年化', [f2y.get(int(y), np.nan) for y in ann['年份']])
        ann.to_excel(writer, sheet_name='年度表现', index=False)
        style_sheet(writer.sheets['年度表现'],
                    pct_cols=['F2年化', '退出记录胜率', '仓位胜率', '净均值', '非ST基线年化', '原主板年化'])

        # 月度
        monthly.to_excel(writer, sheet_name='月度表现', index=False)
        style_sheet(writer.sheets['月度表现'], pct_cols=['净均值'])

        # 逐笔
        trades.to_excel(writer, sheet_name='逐笔交易', index=False)
        style_sheet(writer.sheets['逐笔交易'], pct_cols=['毛收益', '账户权重', '净收益率'])

        # 持仓特征
        hold.to_excel(writer, sheet_name='持仓特征', index=False)
        style_sheet(writer.sheets['持仓特征'], pct_cols=['60日最大绝对日收益'])

        # 压力
        stress.to_excel(writer, sheet_name='压力测试', index=False)
        style_sheet(writer.sheets['压力测试'], pct_cols=['全期年化', '最大回撤', '退出记录胜率', '2022-2026年化'])

        # 参数悬崖
        cliff.to_excel(writer, sheet_name='参数悬崖', index=False)
        style_sheet(writer.sheets['参数悬崖'], pct_cols=['年化', '最大回撤', '胜率', '检验年化'])

        # 单因素
        g1 = grid1[['类别', '变体', '全期_cagr', '全期_maxdd', '全期_trades', '全期_win',
                    '全期_pf', '训练_cagr', '检验_cagr', '负年份数', '最差年',
                    '全期_最大单笔亏损', '全期_前5笔占比']].copy()
        g1 = g1.rename(columns={'类别': '类别', '变体': '变体', '全期_cagr': '全期年化',
                                '全期_maxdd': '最大回撤', '全期_trades': '交易数',
                                '全期_win': '胜率', '全期_pf': '盈亏比',
                                '训练_cagr': '训练年化', '检验_cagr': '检验年化',
                                '负年份数': '负年份数', '最差年': '最差年',
                                '全期_最大单笔亏损': '最大单笔亏损',
                                '全期_前5笔占比': '前5笔占比'})
        g1.to_excel(writer, sheet_name='单因素', index=False)
        style_sheet(writer.sheets['单因素'],
                    pct_cols=['全期年化', '最大回撤', '胜率', '训练年化', '检验年化',
                              '最差年', '最大单笔亏损', '前5笔占比'])

        # 过拟合
        over.to_excel(writer, sheet_name='过拟合审计', index=False)
        style_sheet(writer.sheets['过拟合审计'], pct_cols=[])
        # 只给真正的百分比行加格式，夏普/偏度/峰度保持普通数字
        ws = writer.sheets['过拟合审计']
        pct_words = ('年化', '回撤', 'PBO', 'DSR', '随机排序均值', '夏普排名')
        for i, name in enumerate(over['项目'], start=2):
            if any(w in name for w in pct_words):
                ws.cell(row=i, column=2).number_format = '0.00%'
        # 走步明细单独一页
        walk.to_excel(writer, sheet_name='走步明细', index=False)
        style_sheet(writer.sheets['走步明细'], pct_cols=['训练年化', '训练回撤'])

        # 候选对比
        comp.to_excel(writer, sheet_name='候选对比', index=False)
        style_sheet(writer.sheets['候选对比'], pct_cols=['年化', '最大回撤', '胜率'])

        # 当前状态、行业、近信号分开
        status.to_excel(writer, sheet_name='当前状态', index=False)
        style_sheet(writer.sheets['当前状态'], pct_cols=['市场宽度'])
        indgate.to_excel(writer, sheet_name='行业闸门', index=False)
        style_sheet(writer.sheets['行业闸门'], pct_cols=['行业宽度', '行业20日收益'])
        near.to_excel(writer, sheet_name='今日近信号', index=False)
        style_sheet(writer.sheets['今日近信号'],
                    pct_cols=['ATR占收盘', '20日波动', '距60日高点', '20日收益',
                              '行业宽度', '行业20日收益'])

        # 说明
        notes = pd.DataFrame([
            ('数据', '全A面板 2016-10-13 至 2026-10-13，5,675只，含102只退市/缺失；今日状态到2026-10-22。'),
            ('价格', '回测用后复权；实际报价用前复权。不要把后复权价当报价。'),
            ('ST', '历史无名称表，用5%触板行为代理；会漏长期不触板的ST，也会误伤极少数低波正常股。'),
            ('幸存者', '退市补漏不完整；小市值主板仍有退市风险。'),
            ('右尾', '前5笔占39.8%；去掉前5笔粗略年化10.7%。'),
            ('风格', '随机排序均值11.2%，F2 15.0%在83分位；mcap排序贡献有限，小盘风格反转会跑输。'),
            ('回撤', '实际-21.1%；1年自举P5 -25.3%；3年P5 -35.2%；最大单笔-22.6%。'),
            ('PBO', '192族30.5%；DSR 0.72（192次试验，视作上界）；F2是固定规则，不要继续调参。'),
            ('修正', '2026-10-24：市值覆盖21%->99.9%，修复排序哨兵；旧25.3%是数据残缺假象，真实15.0%。'),
            ('实盘', '先纸面50笔，再小资金1年；不用生活资金或杠杆。'),
            ('声明', '历史回测，不是未来收益承诺，不构成投资建议。'),
        ], columns=['项目', '说明'])
        notes.to_excel(writer, sheet_name='说明', index=False)
        style_sheet(writer.sheets['说明'], width_max=110)

    print('written', OUT)
    print('sheets', pd.ExcelFile(OUT).sheet_names)


if __name__ == '__main__':
    main()
