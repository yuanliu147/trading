"""只使用合成小数组；直接运行本文件执行账户会计与时序断言。"""
import copy
import hashlib
import unittest

import numpy as np
import pandas as pd

from 账户引擎 import Assumptions, Engine, Spec, _cents, _fees


ACTION_COLUMNS = ['证券', '除权日', '每股分红', '送转比例', '配股比例', '配股价']


def synthetic(symbols=('sh.600001',), end='2011-02-10', dates=None):
    dates = list(dates) if dates is not None else pd.bdate_range('2010-11-01', end).strftime('%Y-%m-%d').tolist()
    shape = (len(symbols), len(dates))
    data = {'meta': {'证券': list(symbols), '日期': dates},
            'index': np.linspace(1000, 2000, len(dates)),
            'actions': pd.DataFrame(columns=ACTION_COLUMNS)}
    values = {'O': 10, 'C': 10, 'PC': 10, 'LUP': 11, 'LDP': 9,
              'H': 10.5, 'L': 9.5, 'V': 1000000, 'AMT': 10000000,
              'PE': 10, 'PB': 1, 'PCT': 0, 'AC': 100, 'AO': 100,
              'ATR20': 2, 'ADV20': 50000000, 'VAVG20': 1000000}
    for key, value in values.items():
        data[key] = np.full(shape, value, dtype=np.float64 if key in ('O', 'C', 'PC', 'LUP', 'LDP') else np.float32)
    for key in ('ST', 'SUSP', 'RESUME', 'U', 'CONFLICT', 'PRESENT'):
        data[key] = np.full(shape, key in ('U', 'PRESENT'), dtype=bool)
    return data


def day(data, date):
    return data['meta']['日期'].index(date)


def actions(data, rows):
    data['actions'] = pd.DataFrame(rows, columns=ACTION_COLUMNS)


def prices(data, i, start, price):
    t = day(data, start)
    for key in ('O', 'C', 'PC'):
        data[key][i, t:] = price
    data['LUP'][i, t:] = price * 1.1
    data['LDP'][i, t:] = price * 0.9


class AccountEngineTests(unittest.TestCase):
    def run_case(self, data, spec=None, assumptions=None, detail=True):
        result = Engine(data).run(spec or Spec('test', band=1, slots=1), 101, assumptions, detail)
        for key, number in result['自检'].items():
            self.assertEqual(number, 0, key)
        nav = result['净值']
        self.assertTrue((nav['现金'] >= 0).all())
        np.testing.assert_allclose(nav['净值'], nav['现金'] + nav['持仓市值'] + nav['应收红利'] - nav['应付税'], atol=1e-9)
        return result

    def events(self, result, name):
        return result['日志'].loc[result['日志']['事件'] == name].reset_index(drop=True)

    def row(self, result, date):
        return result['净值'].set_index('日期').loc[date]

    def test_normal_round_and_fee_inclusive_budget(self):
        data = synthetic(end='2011-04-05')
        data['U'][0, day(data, '2011-03-31'):] = False
        prices(data, 0, '2011-04-01', 12)
        result = self.run_case(data)
        buy, sell = self.events(result, '买入'), self.events(result, '卖出')
        self.assertEqual(len(buy), 1)
        self.assertEqual(buy.iloc[0]['股数'], 900)
        self.assertEqual(buy.iloc[0]['价格'], 10.02)
        self.assertEqual(buy.iloc[0]['费用'], 6)
        self.assertEqual(buy.iloc[0]['现金'], 976)
        self.assertLessEqual(buy.iloc[0]['金额'] + buy.iloc[0]['费用'], buy.iloc[0]['预算'])
        self.assertEqual(sell.iloc[0]['日期'], '2011-04-01')
        self.assertEqual(sell.iloc[0]['价格'], 11.98)
        self.assertEqual(sell.iloc[0]['印花税'], 10.78)
        self.assertEqual(result['汇总']['完整回合数'], 1)
        self.assertEqual(result['汇总']['期末净值'], 11741.22)
        self.assertEqual(result['回合'].iloc[0]['盈亏'], 1741.22)
        self.assertEqual(result['汇总']['胜率'], 1)
        self.assertEqual(result['汇总']['费用合计'], 22.78)

    def test_half_up_fee_dates_exchanges_and_pressure(self):
        self.assertEqual(_cents('1.005'), 101)
        self.assertEqual(_cents('2.675'), 268)
        self.assertEqual(_fees('sh.600001', '2015-07-31', 100, 10, False, 1), (500, 0, 100))
        self.assertEqual(_fees('sz.000002', '2015-07-31', 100, 10, True, 1), (500, 100, 0))
        self.assertEqual(_fees('sh.600001', '2015-07-31', 2000, 10, True, 1), (500, 2000, 200))
        self.assertEqual(_fees('sz.000002', '2015-08-03', 100, 10, True, 1), (500, 100, 2))
        self.assertEqual(_fees('sh.600001', '2015-08-03', 100, 10, True, 2), (1000, 200, 4))
        data = synthetic()
        normal = self.run_case(data)
        doubled = self.run_case(data, assumptions=Assumptions(fee_mult=2))
        b1, b2 = self.events(normal, '买入').iloc[0], self.events(doubled, '买入').iloc[0]
        self.assertEqual(b2['费用'], b1['费用'] * 2)
        self.assertGreater(b2['价格'], b1['价格'])
        self.assertLessEqual(b2['金额'] + b2['费用'], b2['预算'])

    def test_initial_baseline_next_open_and_t_plus_one(self):
        data = synthetic(end='2011-01-07')
        data['ST'][0, day(data, '2011-01-03')] = True
        actions(data, [('sh.600001', '2011-01-03', 1, 1, 1, 5)])
        result = self.run_case(data)
        initial = self.row(result, '2010-12-31')
        self.assertEqual(initial['净值'], 10000)
        self.assertEqual(initial['现金'], 10000)
        self.assertEqual(initial['持仓数'], 0)
        self.assertEqual(self.events(result, '买入').iloc[0]['日期'], '2011-01-03')
        self.assertEqual(self.events(result, '卖出信号').iloc[0]['日期'], '2011-01-03')
        self.assertEqual(self.events(result, '卖出').iloc[0]['日期'], '2011-01-04')
        self.assertEqual(result['汇总']['分红净额'], 0)
        self.assertEqual(result['汇总']['送转税计提'], 0)
        self.assertEqual(result['汇总']['配股放弃股数'], 0)
        one_day = synthetic(end='2011-01-03')
        hold = self.run_case(one_day)['期末持仓'].iloc[0]
        self.assertEqual(hold['可售股数'], 0)
        self.assertEqual(hold['不可立即变现股数'], hold['股数'])

    def test_dividend_bonus_rights_use_same_old_quantity(self):
        data = synthetic(end='2011-01-10')
        actions(data, [('sh.600001', '2011-01-04', 1, 0.5, 0.2, 5)])
        result = self.run_case(data)
        self.assertEqual(result['汇总']['分红净额'], 720)
        self.assertEqual(result['汇总']['送转税计提'], 90)
        self.assertEqual(result['汇总']['配股放弃股数'], 180)
        p = result['期末持仓'].iloc[0]
        self.assertEqual(p['股数'], 1350)
        self.assertEqual(p['可售股数'], 900)
        self.assertEqual(p['锁定股数'], 450)
        self.assertEqual(self.row(result, '2011-01-04')['现金'], 886)
        self.assertEqual(self.row(result, '2011-01-04')['应收红利'], 720)
        for name in ('红利计提', '送转计提', '放弃配股'):
            self.assertEqual(self.events(result, name).iloc[0]['旧股数'], 900)

    def test_dividend_arrives_after_position_fully_sold(self):
        data = synthetic(end='2011-02-07')
        data['ST'][0, day(data, '2011-01-04')] = True
        actions(data, [('sh.600001', '2011-01-04', 1, 0, 0, 0)])
        result = self.run_case(data)
        self.assertEqual(result['回合'].iloc[0]['卖出日'], '2011-01-05')
        self.assertEqual(result['回合'].iloc[0]['净红利'], 720)
        before, after = self.row(result, '2011-02-02'), self.row(result, '2011-02-03')
        self.assertEqual(before['持仓数'], 0)
        self.assertEqual(before['应收红利'], 720)
        self.assertEqual(after['应收红利'], 0)
        self.assertEqual(after['现金'] - before['现金'], 720)
        self.assertEqual(after['净值'], before['净值'])
        self.assertEqual(self.events(result, '红利到账').iloc[0]['日期'], '2011-02-03')

    def test_locked_bonus_partial_sales_form_one_round(self):
        data = synthetic(end='2011-02-04')
        data['ST'][0, day(data, '2011-01-03')] = True
        actions(data, [('sh.600001', '2011-01-04', 1, 0.5, 0, 0)])
        result = self.run_case(data)
        sales = self.events(result, '卖出')
        unlock_t = day(data, '2011-01-04') + 20
        self.assertEqual(sales['日期'].tolist(), ['2011-01-04', data['meta']['日期'][unlock_t]])
        self.assertEqual(sales['股数'].tolist(), [900, 450])
        self.assertEqual(result['汇总']['卖出笔数'], 2)
        self.assertEqual(result['汇总']['完整回合数'], 1)
        self.assertGreater(result['汇总']['卖出延期天次'], 0)
        p = result['回合'].iloc[0]
        expected = sales['金额'].sum() + 720 - p['买入总支出'] - 90
        self.assertAlmostEqual(p['盈亏'], expected, places=8)
        self.assertEqual(p['卖出股数'], p['买入股数'] + p['送转股数'])

    def test_locked_shares_receive_later_actions_and_floor_bonus(self):
        data = synthetic(end='2011-01-10')
        data['ST'][0, day(data, '2011-01-03')] = True
        actions(data, [('sh.600001', '2011-01-04', 0, 0.501, 0, 0),
                       ('sh.600001', '2011-01-06', 2, 0.501, 0.1, 5)])
        result = self.run_case(data)
        bonuses = self.events(result, '送转计提')
        self.assertEqual(bonuses['旧股数'].tolist(), [900, 450])
        self.assertEqual(bonuses['股数'].tolist(), [450, 225])
        self.assertEqual(result['汇总']['分红净额'], 720)
        self.assertEqual(result['汇总']['配股放弃股数'], 45)
        self.assertEqual(result['期末持仓'].iloc[0]['股数'], 675)
        self.assertEqual(result['汇总']['完整回合数'], 0)

    def test_zero_delay_and_zero_lock_available_on_action_day(self):
        data = synthetic(end='2011-01-07')
        data['ST'][0, day(data, '2011-01-03')] = True
        actions(data, [('sh.600001', '2011-01-04', 1, 0.5, 0, 0)])
        result = self.run_case(data, assumptions=Assumptions(dividend_delay=0, bonus_lock=0))
        self.assertEqual(self.events(result, '红利到账').iloc[0]['日期'], '2011-01-04')
        self.assertEqual(self.events(result, '卖出')['股数'].tolist(), [1350])
        self.assertEqual(result['汇总']['完整回合数'], 1)
        self.assertTrue((result['净值']['应收红利'] == 0).all())
        self.assertEqual(result['汇总']['送转税计提'], 90)

    def test_tax_debt_never_overdraws_and_later_dividend_settles(self):
        data = synthetic(end='2011-01-12')
        actions(data, [('sh.600001', '2011-01-04', 10, 10, 0, 0)])
        prices(data, 0, '2011-01-04', 0.9)
        result = self.run_case(data, assumptions=Assumptions(dividend_delay=5))
        event_day = self.row(result, '2011-01-04')
        self.assertEqual(event_day['现金'], 0)
        self.assertEqual(event_day['应付税'], 824)
        self.assertEqual(event_day['应收红利'], 7200)
        self.assertEqual(event_day['净值'], 8910 + 7200 - 824)
        # 1月9日为自然日到期，日线账在其后的首个交易日1月10日到账。
        self.assertEqual(self.events(result, '红利到账').iloc[0]['日期'], '2011-01-10')
        self.assertEqual(self.row(result, '2011-01-10')['应付税'], 0)
        self.assertEqual(self.row(result, '2011-01-10')['现金'], 6376)
        self.assertEqual(self.events(result, '支付送转税')['金额'].sum(), 1800)

    def test_tax_debt_is_paid_from_partial_sale(self):
        data = synthetic(end='2011-01-07')
        actions(data, [('sh.600001', '2011-01-04', 0, 10, 0, 0)])
        data['ST'][0, day(data, '2011-01-04')] = True
        result = self.run_case(data)
        self.assertEqual(self.row(result, '2011-01-04')['应付税'], 824)
        self.assertEqual(self.row(result, '2011-01-05')['应付税'], 0)
        self.assertEqual(result['汇总']['完整回合数'], 0)
        self.assertEqual(result['期末持仓'].iloc[0]['股数'], 9000)

    def test_rights_are_abandoned_without_shares_or_cash_payment(self):
        data = synthetic(end='2011-01-07')
        actions(data, [('sh.600001', '2011-01-04', 0, 0, 0.3, 3)])
        prices(data, 0, '2011-01-04', 8)
        result = self.run_case(data)
        self.assertEqual(result['汇总']['配股放弃股数'], 270)
        self.assertEqual(result['期末持仓'].iloc[0]['股数'], 900)
        self.assertEqual(self.row(result, '2011-01-04')['现金'], 976)
        self.assertEqual(self.row(result, '2011-01-04')['持仓市值'], 7200)
        self.assertEqual(result['汇总']['送转税计提'], 0)
        self.assertEqual(result['汇总']['公司行动参考价不一致天次'], 1)
        self.assertEqual(self.events(result, '公司行动参考价核对').iloc[0]['原因'], '不一致（不删除事件）')

    def test_suspension_limit_down_missing_open_postpone_irrevocably(self):
        data = synthetic(end='2011-01-10')
        data['ST'][0, day(data, '2011-01-03')] = True
        data['SUSP'][0, day(data, '2011-01-04')] = True
        data['O'][0, day(data, '2011-01-05')] = 9
        data['O'][0, day(data, '2011-01-06')] = np.nan
        result = self.run_case(data)
        self.assertEqual(self.events(result, '卖出')['日期'].tolist(), ['2011-01-07'])
        self.assertEqual(self.events(result, '卖出').iloc[0]['原因'], 'ST')
        self.assertEqual(result['汇总']['卖出延期天次'], 3)
        self.assertEqual(result['汇总']['买入笔数'], 1)

    def test_limit_up_blocks_buy_but_allows_sale(self):
        data = synthetic(end='2011-01-07')
        data['O'][0, day(data, '2011-01-03')] = 11
        no_buy = self.run_case(data)
        self.assertEqual(no_buy['汇总']['买入笔数'], 0)
        self.assertEqual(self.events(no_buy, '买入失败').iloc[0]['原因'], '开盘涨停')
        data['O'][0, day(data, '2011-01-03')] = 10
        data['ST'][0, day(data, '2011-01-03')] = True
        data['O'][0, day(data, '2011-01-04')] = 11
        sale = self.run_case(data)
        self.assertEqual(sale['汇总']['卖出笔数'], 1)
        self.assertLessEqual(self.events(sale, '卖出').iloc[0]['价格'], 11)

    def test_insufficient_hundred_share_budget_has_no_replacement(self):
        data = synthetic(('sh.600001', 'sz.000002'), end='2011-01-07')
        ordered = sorted(range(2), key=lambda i: hashlib.sha256(f"101|{data['meta']['证券'][i]}".encode()).digest())
        expensive, cheap = ordered
        prices(data, expensive, '2010-12-31', 200)
        prices(data, cheap, '2010-12-31', 5)
        result = self.run_case(data, Spec('control', score='control', band=0.01, slots=1))
        self.assertEqual(result['汇总']['买入笔数'], 0)
        self.assertEqual(result['汇总']['买入失败数'], 1)
        self.assertEqual(self.events(result, '买入失败').iloc[0]['证券'], data['meta']['证券'][expensive])
        self.assertEqual(result['汇总']['期末净值'], 10000)

    def test_current_volume_rejects_whole_order_not_downsizes(self):
        data = synthetic(end='2011-01-07')
        data['V'][0, day(data, '2011-01-03')] = 850
        result = self.run_case(data)
        failure = self.events(result, '买入失败').iloc[0]
        self.assertEqual(failure['股数'], 900)
        self.assertEqual(failure['原因'], '当日量不足订单')
        self.assertEqual(result['汇总']['买入笔数'], 0)

    def test_only_previous_vavg_sizes_buy_and_partial_sells(self):
        data = synthetic(end='2011-01-18')
        data['VAVG20'][0, day(data, '2011-01-03'):] = 10000
        data['ST'][0, day(data, '2011-01-03')] = True
        result = self.run_case(data)
        self.assertEqual(self.events(result, '买入').iloc[0]['股数'], 900)
        self.assertEqual(self.events(result, '卖出')['股数'].tolist(), [100] * 9)
        self.assertEqual(result['汇总']['完整回合数'], 1)
        self.assertEqual(result['汇总']['卖出延期天次'], 8)
        self.assertEqual(Engine._sell_qty(150, 100), 100)
        self.assertEqual(Engine._sell_qty(50, 100), 50)
        self.assertEqual(Engine._sell_qty(150, 50), 0)
        limited = synthetic(end='2011-01-07')
        limited['VAVG20'][0, day(limited, '2010-12-31')] = 10000
        self.assertEqual(self.events(self.run_case(limited), '买入').iloc[0]['股数'], 100)

    def test_failed_sell_cannot_overfill_slots_or_retry_buy_later(self):
        data = synthetic(('sh.600001', 'sz.000002'), end='2011-04-06')
        data['U'][1, :] = False
        review = day(data, '2011-03-31')
        data['U'][0, review:] = False
        data['U'][1, review:] = True
        data['SUSP'][0, day(data, '2011-04-01')] = True
        result = self.run_case(data)
        self.assertEqual(result['汇总']['买入笔数'], 1)
        self.assertEqual(self.events(result, '卖出')['日期'].tolist(), ['2011-04-04'])
        failure = self.events(result, '买入失败').iloc[0]
        self.assertEqual(failure['证券'], 'sz.000002')
        self.assertEqual(failure['原因'], '待卖未清导致无空槽')
        self.assertEqual(result['汇总']['期末持仓数'], 0)
        self.assertLessEqual(result['净值']['持仓数'].max(), 1)

    def test_keep_pool_holding_without_equal_weight_sales(self):
        data = synthetic(('sh.600001', 'sz.000002'), end='2011-04-06')
        prices(data, 0, '2011-02-01', 30)
        prices(data, 1, '2011-02-01', 5)
        result = self.run_case(data, Spec('keep', band=1, slots=2))
        self.assertEqual(result['汇总']['买入笔数'], 2)
        self.assertEqual(result['汇总']['卖出笔数'], 0)
        self.assertEqual(result['汇总']['期末持仓数'], 2)
        self.assertGreater(result['汇总']['最大单票权重'], 0.5)
        self.assertEqual(result['汇总']['完整回合数'], 0)
        self.assertTrue(np.isnan(result['汇总']['胜率']))

    def test_pool_rank_ceiling_ties_control_and_lagged_admission(self):
        data = synthetic(('sh.600001', 'sh.600002', 'sz.000003'))
        t = day(data, '2010-12-31')
        data['PE'][:, t] = [5, 10, 20]
        data['PB'][:, t] = [1, 2, 4]
        engine = Engine(data)
        self.assertEqual(engine._pool(Spec('p', band=0.34), t), frozenset((0, 1)))
        self.assertEqual(engine._pool(Spec('p', band=0.1, score='ep'), t), frozenset((0,)))
        self.assertEqual(engine._pool(Spec('p', band=0.1, score='bp'), t), frozenset((0,)))
        self.assertEqual(engine._pool(Spec('c', band=0.01, score='control'), t), frozenset((0, 1, 2)))
        tied = synthetic(('sh.600001', 'sh.600002', 'sz.000003'))
        self.assertEqual(Engine(tied)._pool(Spec('tie', band=0.1), t), frozenset((0,)))
        data['PE'][:, t - 2] = [30, 20, 10]
        data['PB'][:, t - 2] = [3, 2, 1]
        self.assertEqual(Engine(data)._pool(Spec('lag', band=0.1, lag=2), t), frozenset((2,)))
        data['U'][2, t] = False
        data['PB'][0, t - 2] = np.nan
        self.assertEqual(Engine(data)._pool(Spec('lag', band=1, lag=2, score='ep'), t), frozenset((1,)))

    def test_market_full_window_missing_or_equal_does_not_buy(self):
        data = synthetic(end='2011-01-07')
        result = self.run_case(data, Spec('ma', band=1, slots=1, exit_kind='market', window=250))
        self.assertEqual(result['汇总']['买入笔数'], 0)
        self.assertEqual(result['汇总']['买入失败数'], 0)
        data['index'][:] = 1000
        equal = self.run_case(data, Spec('ma', band=1, slots=1, exit_kind='market', window=2))
        self.assertEqual(equal['汇总']['买入笔数'], 0)
        t = day(data, '2010-12-31')
        data['index'][t] = 2000
        data['index'][t - 1] = np.nan
        missing = self.run_case(data, Spec('ma', band=1, slots=1, exit_kind='market', window=3))
        self.assertEqual(missing['汇总']['买入笔数'], 0)

    def test_market_daily_exit_next_open_no_daily_refill(self):
        data = synthetic(end='2011-01-10')
        data['index'][day(data, '2011-01-04')] = 10
        result = self.run_case(data, Spec('ma', band=1, slots=1, exit_kind='market', window=2))
        self.assertEqual(result['汇总']['买入笔数'], 1)
        self.assertEqual(self.events(result, '卖出信号').iloc[0]['原因'], 'market')
        self.assertEqual(self.events(result, '卖出信号').iloc[0]['日期'], '2011-01-04')
        self.assertEqual(self.events(result, '卖出').iloc[0]['日期'], '2011-01-05')
        self.assertEqual(result['汇总']['期末持仓数'], 0)

    def test_atr_entry_continuous_space_ratchets_and_next_open(self):
        data = synthetic(end='2011-01-10')
        for date, close, atr in [('2011-01-03', 100, 2), ('2011-01-04', 110, 2),
                                 ('2011-01-05', 106, 10), ('2011-01-06', 103, 10)]:
            t = day(data, date)
            data['AC'][0, t], data['ATR20'][0, t] = close, atr
        result = self.run_case(data, Spec('atr', band=1, slots=1, exit_kind='trail', multiple=3))
        buy = self.events(result, '买入').iloc[0]
        self.assertAlmostEqual(buy['连续成交价'], 100 * buy['价格'] / 10)
        self.assertAlmostEqual(buy['止损连续价'], buy['连续成交价'] - 6)
        signal = self.events(result, '卖出信号').iloc[0]
        self.assertEqual(signal['日期'], '2011-01-06')
        self.assertEqual(signal['原因'], 'trail')
        self.assertEqual(signal['止损连续价'], 104)
        self.assertEqual(self.events(result, '卖出').iloc[0]['日期'], '2011-01-07')

    def test_atr_missing_signal_window_blocks_entry(self):
        data = synthetic(end='2011-01-07')
        data['ATR20'][0, day(data, '2010-12-31')] = np.nan
        result = self.run_case(data, Spec('atr', band=1, slots=1, exit_kind='trail', multiple=3))
        self.assertEqual(result['汇总']['买入笔数'], 0)
        self.assertEqual(self.events(result, '买入失败').iloc[0]['原因'], 'ATR缺窗或连续开盘无效')

    def test_st_priority_over_trail_and_valuation(self):
        data = synthetic(end='2011-04-05')
        t = day(data, '2011-03-31')
        data['ST'][0, t] = True
        data['U'][0, t] = False
        data['AC'][0, t] = 1
        result = self.run_case(data, Spec('priority', band=1, slots=1, exit_kind='trail', multiple=3))
        self.assertEqual(self.events(result, '卖出信号').iloc[0]['原因'], 'ST')
        data['index'][t] = 1
        market = self.run_case(data, Spec('priority', band=1, slots=1, exit_kind='market', window=2))
        self.assertEqual(self.events(market, '卖出信号').iloc[0]['原因'], 'ST')

    def test_mark_last_raw_close_not_fill_buy_price_and_stale_threshold(self):
        data = synthetic(end='2011-02-10')
        buy_t, absent = day(data, '2011-01-03'), day(data, '2011-01-04')
        data['C'][0, buy_t] = np.nan
        data['PRESENT'][0, absent:] = False
        data['U'][0, absent:] = False
        for key in ('O', 'C', 'PC', 'H', 'L', 'V'):
            data[key][0, absent:] = np.nan
        base = self.run_case(data)
        stress = self.run_case(data, assumptions=Assumptions(stale_haircut=0.5))
        self.assertEqual(self.row(base, '2011-01-03')['持仓市值'], 9000)
        self.assertNotEqual(self.events(base, '买入').iloc[0]['金额'], 9000)
        nineteenth, twentieth = data['meta']['日期'][absent + 18], data['meta']['日期'][absent + 19]
        self.assertEqual(self.row(stress, nineteenth)['持仓市值'], 9000)
        self.assertEqual(self.row(stress, twentieth)['持仓市值'], 4500)
        self.assertEqual(base['汇总']['卖出笔数'], 0)
        self.assertEqual(stress['汇总']['卖出笔数'], 0)
        p = stress['期末持仓'].iloc[0]
        self.assertEqual(p['最后报价日'], '2010-12-31')
        self.assertEqual(p['报价缺日数'], len(data['meta']['日期']) - 1 - day(data, '2010-12-31'))
        self.assertEqual(p['连续缺行数'], len(data['meta']['日期']) - absent)
        self.assertEqual(stress['汇总']['期末不可立即变现市值'], 4500)

    def test_total_stale_write_down_has_minus_one_cagr(self):
        data = synthetic(end='2011-02-10')
        absent = day(data, '2011-01-04')
        data['PRESENT'][0, absent:] = False
        data['C'][0, absent:] = np.nan
        result = self.run_case(data, assumptions=Assumptions(fee_mult=0, stale_haircut=1))
        self.assertEqual(result['汇总']['期末净值'], 0)
        self.assertEqual(result['汇总']['CAGR'], -1)
        self.assertEqual(result['汇总']['最大回撤'], -1)
        self.assertEqual(result['汇总']['期末持仓数'], 1)
        self.assertEqual(result['汇总']['卖出笔数'], 0)

    def test_calendar_cagr_and_2016_first_day_is_not_dropped(self):
        dates = ['2010-12-30', '2010-12-31', '2011-01-04', '2015-12-31', '2016-01-04', '2020-12-31']
        data = synthetic(dates=dates)
        prices(data, 0, '2016-01-04', 20)
        result = self.run_case(data, assumptions=Assumptions(fee_mult=0))
        self.assertEqual(result['汇总']['期末净值'], 20000)
        self.assertEqual(result['汇总']['2011-2015_CAGR'], 0)
        days = (pd.Timestamp('2020-12-31') - pd.Timestamp('2015-12-31')).days
        self.assertAlmostEqual(result['汇总']['2016-2020_CAGR'], 2 ** (365.25 / days) - 1)
        all_days = (pd.Timestamp('2020-12-31') - pd.Timestamp('2010-12-31')).days
        self.assertAlmostEqual(result['汇总']['CAGR'], 2 ** (365.25 / all_days) - 1)
        self.assertEqual(result['汇总']['期末持仓数'], 1)
        self.assertEqual(result['汇总']['完整回合数'], 0)

    def test_future_prices_flags_values_and_actions_cannot_change_past(self):
        data = synthetic(('sh.600001', 'sz.000002'), end='2011-04-08')
        actions(data, [('sh.600001', '2011-01-04', 1, 0.5, 0.2, 5)])
        changed = copy.deepcopy(data)
        cutoff = '2011-02-01'
        future = day(data, '2011-02-02')
        for key in ('O', 'C', 'PC', 'H', 'L', 'AC', 'AO', 'PE', 'PB', 'ATR20', 'VAVG20'):
            changed[key][:, future:] *= 3
        changed['U'][:, future:] = False
        changed['ST'][:, future:] = True
        changed['index'][future:] = 10
        changed['actions'] = pd.concat([changed['actions'], pd.DataFrame(
            [('sz.000002', '2011-02-03', 100, 2, 0, 0)], columns=ACTION_COLUMNS)], ignore_index=True)
        for spec in (Spec('base', band=1, slots=2),
                     Spec('market', band=1, slots=2, exit_kind='market', window=2),
                     Spec('trail', band=1, slots=2, exit_kind='trail', multiple=3)):
            original, altered = self.run_case(data, spec), self.run_case(changed, spec)
            for key in ('日志', '净值'):
                left = original[key].loc[original[key]['日期'] <= cutoff].reset_index(drop=True)
                right = altered[key].loc[altered[key]['日期'] <= cutoff].reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right)

    def test_detail_false_retains_true_completed_round_statistics(self):
        data = synthetic(end='2011-02-04')
        data['ST'][0, day(data, '2011-01-03')] = True
        actions(data, [('sh.600001', '2011-01-04', 1, 0.5, 0, 0)])
        engine = Engine(data)
        spec = Spec('same', band=1, slots=1)
        full, brief = engine.run(spec, 101, detail=True), engine.run(spec, 101, detail=False)
        self.assertEqual(set(brief), {'净值', '汇总', '自检'})
        pd.testing.assert_frame_equal(full['净值'], brief['净值'])
        pd.testing.assert_series_equal(pd.Series(full['汇总']), pd.Series(brief['汇总']))
        self.assertEqual(full['自检'], brief['自检'])
        self.assertEqual(brief['汇总']['完整回合数'], 1)
        self.assertEqual(brief['汇总']['卖出笔数'], 2)

    def test_profit_loss_ratio_counts_rounds_not_partial_fills(self):
        data = synthetic(('sh.600001', 'sz.000002'), end='2011-02-04')
        data['ST'][:, day(data, '2011-01-03')] = True
        actions(data, [('sh.600001', '2011-01-04', 0, 1, 0, 0)])
        prices(data, 0, '2011-01-04', 8)
        prices(data, 1, '2011-01-04', 8)
        result = self.run_case(data, Spec('two', band=1, slots=2))
        self.assertEqual(result['汇总']['完整回合数'], 2)
        self.assertEqual(result['汇总']['卖出笔数'], 3)
        pnls = result['回合']['盈亏']
        self.assertEqual(result['汇总']['胜率'], 0.5)
        self.assertAlmostEqual(result['汇总']['盈亏比'], pnls[pnls > 0].mean() / -pnls[pnls < 0].mean())

    def test_quarter_and_half_year_review_schedule(self):
        data = synthetic(end='2011-07-05')
        data['U'][0, day(data, '2011-03-31'):] = False
        quarterly = self.run_case(data, Spec('q', band=1, slots=1, months=3))
        half = self.run_case(data, Spec('h', band=1, slots=1, months=6))
        self.assertEqual(self.events(quarterly, '卖出')['日期'].tolist(), ['2011-04-01'])
        self.assertEqual(self.events(half, '卖出')['日期'].tolist(), ['2011-07-01'])

    def test_prior_cash_sharing_and_initial_target_caps(self):
        data = synthetic(('sh.600001', 'sz.000002'), end='2011-01-07')
        result = self.run_case(data, Spec('budget', band=1, slots=2))
        buys = self.events(result, '买入')
        self.assertEqual(len(buys), 2)
        self.assertTrue((buys['预算'] <= 5000).all())
        self.assertTrue((buys['金额'] + buys['费用'] <= buys['预算']).all())
        # 第二单不能用第一单整手余款突破前日净值/槽位。
        self.assertEqual(buys['股数'].tolist(), [400, 400])

    def test_price_rounding_is_adverse_and_clamped(self):
        data = synthetic(end='2011-01-07')
        t = day(data, '2011-01-03')
        data['O'][0, t] = 10.999
        engine = Engine(data)
        self.assertEqual(engine._price(0, t, 100, False, Assumptions()), 11)
        data['O'][0, t] = 9.001
        self.assertEqual(engine._price(0, t, 100, True, Assumptions()), 9)
        data['O'][0, t] = 10
        self.assertGreaterEqual(engine._price(0, t, 100, False, Assumptions()), 10)
        self.assertLessEqual(engine._price(0, t, 100, True, Assumptions()), 10)

    def test_signal_day_restrictions_and_window_boundary(self):
        for flag in ('ST', 'SUSP', 'RESUME', 'CONFLICT'):
            with self.subTest(flag=flag):
                data = synthetic(end='2011-01-07')
                data[flag][0, day(data, '2010-12-31')] = True
                result = self.run_case(data)
                self.assertEqual(result['汇总']['买入笔数'], 0)
        data = synthetic(dates=['2010-12-31'])
        result = self.run_case(data)
        self.assertEqual(result['汇总']['期末净值'], 10000)
        self.assertEqual(result['汇总']['买入笔数'], 0)
        self.assertEqual(list(result['净值'].columns), Engine.NAV_COLUMNS)
        self.assertEqual(list(result['回合'].columns), Engine.ROUND_COLUMNS)
        self.assertEqual(list(result['期末持仓'].columns), Engine.HOLD_COLUMNS)
        self.assertTrue(np.isnan(result['汇总']['CAGR']))

    def test_missing_row_st_sentinel_is_not_announced_risk(self):
        data = synthetic(end='2011-01-10')
        t = day(data, '2011-01-04')
        data['PRESENT'][0, t] = False
        data['ST'][0, t] = True  # 数据准备对缺行的防买入哨兵，不是已观察到ST。
        data['SUSP'][0, t] = True
        data['U'][0, t] = False
        for key in ('O', 'C', 'PC', 'V'):
            data[key][0, t] = np.nan
        result = self.run_case(data)
        self.assertTrue(self.events(result, '卖出信号').empty)
        self.assertEqual(result['汇总']['卖出笔数'], 0)

    def test_trailing_entry_does_not_use_same_day_close(self):
        data = synthetic(end='2011-01-10')
        changed = copy.deepcopy(data)
        buy_t = day(data, '2011-01-03')
        for key in ('C', 'H', 'L', 'AC', 'AO'):
            changed[key][0, buy_t] = np.nan
        spec = Spec('intraday', band=1, slots=1, exit_kind='trail', multiple=3)
        base = self.run_case(data, spec)
        altered = self.run_case(changed, spec)
        cols = ['日期', '股数', '价格', '连续成交价', '止损连续价']
        pd.testing.assert_frame_equal(self.events(base, '买入')[cols], self.events(altered, '买入')[cols])
        buy = self.events(altered, '买入').iloc[0]
        self.assertAlmostEqual(buy['连续成交价'], buy['价格'] * data['AC'][0, buy_t - 1] / data['PC'][0, buy_t])


if __name__ == '__main__':
    unittest.main(verbosity=2)
