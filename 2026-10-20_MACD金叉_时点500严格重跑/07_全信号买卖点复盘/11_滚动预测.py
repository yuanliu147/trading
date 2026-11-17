#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐年滚动训练“信号质量”模型，只输出数值预测，不使用股票名称。

- 训练：每年初用此前所有已完成的信号；死亡标签按实际卖出日做 purge，
  fwd20/fwd40 标签按信号日提前 60/90 个自然日做 purge；
- 预测：当年所有信号；特征只含信号日收盘可知的数值；
- 输出：信号_滚动预测.csv（code/signal_date 仅用于和行情对齐，不进入模型）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE / "01_数据"
OUT = HERE / "04_单槽精选"
OUT.mkdir(exist_ok=True)

def feature_list(df: pd.DataFrame) -> list[str]:
    feats = ['ret1','ret3','ret5','ret10','ret20','ret60','ret120',
             'dist_ma5','dist_ma10','dist_ma20','dist_ma60','ma20_slope10','ma60_slope20',
             'ma_align','rsi14','atr20','vol20','vol60','vol_ratio','max_dd60','dd250','pos250',
             'amt5_60','amt20_60','amt60_yi','vol5_60','hist_norm','dif_norm','dea_norm',
             'hist_slope5','dif_slope5','hist_accel','zero_axis','dif_pos',
             'n_golden_60','n_death_60','days_since_golden','signal_day_ret',
             'body','upper_shadow','lower_shadow','close_pos','gap','big_move_20',
             'log_price','beta60','corr60','rs20','rs60','idx_ret20','idx_ret60',
             'idx_above_ma20','idx_above_ma60','idx_vol20','idx_dd250',
             'vol120','range250','s_vol','s_heat','breadth_ma20']
    for t in ['低波低量型','低波活跃型','高波低量型','高波活跃型']:
        df['type_' + t] = (df['type'] == t).astype(int)
        feats.append('type_' + t)
    return feats

def main() -> None:
    sig = pd.read_csv(DATA / "信号特征_时点500_内部.csv", parse_dates=["signal_date"])
    sig = sig[sig["in_universe"] == 1].copy()
    tr = pd.read_csv(DATA / "交易_死叉.csv", parse_dates=["信号日", "买入日", "卖出日"])
    tr = tr.rename(columns={"代码": "code", "信号日": "signal_date"})
    tr = tr[["code", "signal_date", "卖出日", "净收益率", "期间最高", "期间最低",
             "持有交易日", "fwd20", "fwd40"]]
    df = sig.merge(tr, on=["code", "signal_date"], how="inner", suffixes=("", "_tr"))
    feats = feature_list(df)
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["signal_date", "code"]).reset_index(drop=True)
    labels = {"死亡": "净收益率", "fwd20": "fwd20", "fwd40": "fwd40"}
    years = sorted(df.signal_date.dt.year.unique())
    rows = []
    for yr in years:
        test = df[df.signal_date.dt.year == yr].copy()
        if test.empty:
            continue
        if yr < 2013:
            test["pred_死亡"] = np.nan; test["pred_fwd20"] = np.nan; test["pred_fwd40"] = np.nan
            rows.append(test[["code", "signal_date", "pred_死亡", "pred_fwd20", "pred_fwd40"]])
            continue
        test_start = pd.Timestamp(f"{yr}-01-01")
        preds = {}
        for key, lab in labels.items():
            if key == "死亡":
                train = df[(df["signal_date"] < test_start) & (df["卖出日"] < test_start)
                           & df[lab].notna()].copy()
            else:
                days = 60 if key == "fwd20" else 90
                train = df[(df["signal_date"] < test_start - pd.Timedelta(days=days))
                           & df[lab].notna()].copy()
            if len(train) < 1500:
                preds[key] = np.full(len(test), np.nan)
                continue
            med = train[feats].median()
            Xtr = train[feats].fillna(med).to_numpy(float)
            Xte = test[feats].fillna(med).to_numpy(float)
            y = train[lab].to_numpy(float)
            lo, hi = np.nanquantile(y, [0.01, 0.99])
            y = np.clip(y, lo, hi)
            m = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.05, max_iter=200,
                min_samples_leaf=500, l2_regularization=1.0, random_state=42)
            m.fit(Xtr, y)
            preds[key] = m.predict(Xte)
            print(f"  {yr} {key}: train {len(train):,} test {len(test):,} "
                  f"pred [{np.nanmin(preds[key]):.3f},{np.nanmax(preds[key]):.3f}]", flush=True)
        out = test[["code", "signal_date"]].copy()
        out["pred_死亡"] = preds["死亡"]
        out["pred_fwd20"] = preds["fwd20"]
        out["pred_fwd40"] = preds["fwd40"]
        rows.append(out)
    pred = pd.concat(rows, ignore_index=True)
    pred = pred.sort_values(["signal_date", "code"]).reset_index(drop=True)
    for k in ["pred_死亡", "pred_fwd20", "pred_fwd40"]:
        pred[k + "_日内分位"] = pred.groupby("signal_date")[k].rank(pct=True)
    pred.to_csv(OUT / "信号_滚动预测.csv", index=False, encoding="utf-8-sig")
    print("\n保存", OUT / "信号_滚动预测.csv", pred.shape)
    print(pred.groupby(pred.signal_date.dt.year)[["pred_死亡", "pred_fwd20", "pred_fwd40"]].count().to_string())

if __name__ == "__main__":
    main()
