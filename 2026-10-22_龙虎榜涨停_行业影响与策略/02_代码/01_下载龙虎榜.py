# -*- coding: utf-8 -*-
"""下载 2016-2026 龙虎榜明细与机构买卖统计（AkShare/东方财富）。
输出：01_数据/龙虎榜明细_YYYY.parquet, 机构买卖_YYYY.parquet
可重复运行；已存在且非空的文件跳过。
"""
import os, time, sys
import pandas as pd
import akshare as ak

BASE = '/Users/admin/WorkSpace/trading/2026-10-22_龙虎榜涨停_行业影响与策略'
OUT = os.path.join(BASE, '01_数据')
os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2016, 2027))


def fetch(func, start, end, tries=4):
    last = None
    for k in range(tries):
        try:
            df = func(start_date=start, end_date=end)
            if df is None or len(df) == 0:
                raise RuntimeError('empty dataframe')
            return df
        except Exception as e:
            last = e
            print(f'  retry {k+1}/{tries}: {type(e).__name__}: {e}', flush=True)
            time.sleep(2 + 2 * k)
    raise last


def main():
    for y in YEARS:
        start = f'{y}0101'
        end = f'{y}1231' if y < 2026 else '20261022'
        p1 = os.path.join(OUT, f'龙虎榜明细_{y}.parquet')
        p2 = os.path.join(OUT, f'机构买卖_{y}.parquet')
        if os.path.exists(p1) and os.path.getsize(p1) > 1000:
            print(f'[skip] {y} 明细已存在', flush=True)
        else:
            t = time.time()
            df = fetch(ak.stock_lhb_detail_em, start, end)
            df.to_parquet(p1, index=False)
            print(f'[ok] {y} 明细 {len(df):,} 行 {time.time()-t:.1f}s -> {p1}', flush=True)
        if os.path.exists(p2) and os.path.getsize(p2) > 1000:
            print(f'[skip] {y} 机构已存在', flush=True)
        else:
            t = time.time()
            try:
                df = fetch(ak.stock_lhb_jgmmtj_em, start, end)
                df.to_parquet(p2, index=False)
                print(f'[ok] {y} 机构 {len(df):,} 行 {time.time()-t:.1f}s -> {p2}', flush=True)
            except Exception as e:
                print(f'[warn] {y} 机构下载失败: {e}', flush=True)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
