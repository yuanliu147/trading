"""账户统计与回测选择偏差诊断。诊断不替代前瞻样本外。"""
from itertools import combinations
import math
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, rankdata, skew


def account_metrics(equity, initial_cash=20000.0, start_date=None):
    """equity为含空仓日的收盘权益；第一天费用也进入收益和回撤。"""
    e = pd.Series(equity, dtype=float)
    if e.empty or (e <= 0).any() or not np.isfinite(e).all():
        raise ValueError("权益须非空、有限且为正")
    idx = pd.DatetimeIndex(e.index)
    start = pd.Timestamp(start_date) if start_date is not None else idx[0]
    years = max((idx[-1] - start).days + 1, 1) / 365.25
    r = e.pct_change()
    r.iloc[0] = e.iloc[0] / initial_cash - 1
    peak = np.maximum.accumulate(np.r_[initial_cash, e.to_numpy()])[1:]
    dd = e.to_numpy() / peak - 1
    underwater = dd < -1e-12
    run = longest = 0
    for v in underwater:
        run = run + 1 if v else 0
        longest = max(longest, run)
    sd = r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0) ** 2))
    total = e.iloc[-1] / initial_cash - 1
    return {
        "总收益": float(total), "年化收益": float((1 + total) ** (1 / years) - 1),
        "最大回撤": float(dd.min()), "夏普": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "索提诺": float(r.mean() / downside * np.sqrt(252)) if downside > 0 else np.nan,
        "期末权益": float(e.iloc[-1]), "最长水下交易日": int(longest),
        "最差交易日": float(r.min()), "日收益波动": float(sd),
        "实际年数": float(years), "交易日数": len(e),
    }


def annual_monthly(equity, initial_cash=20000.0):
    e = pd.Series(equity, dtype=float)
    e.index = pd.DatetimeIndex(e.index)
    out = []
    for freq, name in [("YE", "年度"), ("ME", "月份")]:
        x = e.resample(freq).last().dropna()
        r = x.pct_change()
        if len(r):
            r.iloc[0] = x.iloc[0] / initial_cash - 1
        out.append(pd.DataFrame({name: x.index.strftime("%Y" if freq == "YE" else "%Y-%m"),
                                 "收益率": r.to_numpy(), "期末权益": x.to_numpy()}))
    return tuple(out)


def block_indices(n, block, rng):
    """循环移动区块；保留区块内聚集，首尾拼接属于bootstrap而非历史路径。"""
    if n <= 0:
        raise ValueError("样本不能为空")
    starts = rng.integers(0, n, size=math.ceil(n / block))
    return ((starts[:, None] + np.arange(block)) % n).reshape(-1)[:n]


def block_bootstrap(returns, years, samples=2000, block=20, seed=20261023):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or np.any(r <= -1) or years <= 0:
        raise ValueError("收益须为有限的一维完整序列且每个值大于负一")
    rng = np.random.default_rng(seed)
    out = np.empty((samples, 3))
    for b in range(samples):
        x = r[block_indices(len(r), block, rng)]
        e = np.exp(np.cumsum(np.log1p(x)))
        dd = e / np.maximum.accumulate(np.r_[1.0, e])[1:] - 1
        out[b] = [np.expm1(np.log1p(x).sum() / years),
                  x.mean() / (x.std(ddof=1) + 1e-15) * np.sqrt(252), dd.min()]
    result = {"区块交易日": block, "抽样次数": samples, "原样本年数": years,
              "重采样年化为正比例": float((out[:, 0] > 0).mean()),
              "重采样年化超百分之五十比例": float((out[:, 0] >= .5).mean()),
              "解释": "历史收益的区块重采样区间，不是未来盈利概率；不修复样本选择和数据缺漏"}
    for i, col in enumerate(["年化", "夏普", "回撤"]):
        result[col + "区间下限"] = float(np.quantile(out[:, i], .025))
        result[col + "区间中位"] = float(np.quantile(out[:, i], .5))
        result[col + "区间上限"] = float(np.quantile(out[:, i], .975))
    return result


def deflated_sharpe(returns, candidate_returns, trial_count=None):
    """Bailey/Lopez de Prado DSR近似。使用日频SR、非超额收益现金基准零。

    SR阈值由全部试验日频SR横截面方差估计，试验数取已登记总数。
    偏度/峰度修正不是序列相关修正；另看区块bootstrap。
    """
    r = np.asarray(returns, dtype=float)
    allr = np.asarray(candidate_returns, dtype=float)
    if allr.ndim != 2 or len(r) != len(allr) or not np.isfinite(allr).all() or not np.isfinite(r).all():
        raise ValueError("候选必须是同一日历对齐的完整日收益矩阵")
    n = len(r)
    m = max(int(trial_count or allr.shape[1]), allr.shape[1])
    sd = r.std(ddof=1)
    if n < 30 or sd <= 0:
        return {"调整夏普概率": np.nan, "说明": "样本不足或零波动", "试验数": m}
    sr = r.mean() / sd
    sds = allr.std(axis=0, ddof=1)
    srs = np.divide(allr.mean(axis=0), sds, out=np.zeros_like(sds), where=sds > 0)
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    gamma = 0.5772156649015329
    threshold = 0.0 if m <= 1 else np.sqrt(var_sr) * (
        (1 - gamma) * norm.ppf(1 - 1 / m) + gamma * norm.ppf(1 - 1 / (m * math.e)))
    sk, ku = float(skew(r, bias=False)), float(kurtosis(r, fisher=False, bias=False))
    denom = np.sqrt(max(1 - sk * sr + (ku - 1) * sr ** 2 / 4, 1e-15))
    p = norm.cdf((sr - threshold) * np.sqrt(n - 1) / denom)
    return {"调整夏普概率": float(p), "日频夏普": float(sr),
            "日频选择阈值": float(threshold), "年化选择阈值": float(threshold * np.sqrt(252)),
            "候选夏普方差": var_sr, "试验数": m, "偏度": sk, "非超额峰度": ku,
            "说明": "DSR近似，不是策略未来盈利概率；未显式校正序列相关及项目历史全部搜索"}


def probability_backtest_overfit(candidate_returns, blocks=8):
    """CSCV最高夏普选择器诊断，并非预注册选择流程的PBO。

    反复重组的回看诊断，不能视作可部署walk-forward；没有打乱区块内收益。
    训练最高夏普并列仍以首列确定诊断入选列（明示列顺序依赖）；训练全部无区别、
    测试全部无区别或测试中位logit=0的组合记为并列，排除于失效比例分母。
    """
    a = np.asarray(candidate_returns, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or len(a) < blocks * 10 or not np.isfinite(a).all():
        raise ValueError("候选矩阵需至少两列，足够长且完整")
    if blocks % 2:
        raise ValueError("区块数必须为偶数")
    chunks = np.array_split(np.arange(len(a)), blocks)
    rows = []
    for comb in combinations(range(blocks), blocks // 2):
        train = np.concatenate([chunks[k] for k in comb])
        test = np.concatenate([chunks[k] for k in range(blocks) if k not in comb])
        z, y = a[train], a[test]
        sr0 = z.mean(axis=0) / (z.std(axis=0, ddof=1) + 1e-15)
        sr1 = y.mean(axis=0) / (y.std(axis=0, ddof=1) + 1e-15)
        pick = int(np.argmax(sr0))
        percentile = rankdata(sr1, method="average")[pick] / (a.shape[1] + 1)
        logit = float(np.log(percentile / (1 - percentile)))
        train_ties = int(np.isclose(sr0, sr0[pick], rtol=0, atol=1e-12).sum())
        test_ties = int(np.isclose(sr1, sr1[pick], rtol=0, atol=1e-12).sum())
        reasons = []
        if train_ties == a.shape[1]:
            reasons.append("训练候选无区别")
        if test_ties == a.shape[1]:
            reasons.append("测试候选无区别")
        if np.isclose(logit, 0, rtol=0, atol=1e-12):
            logit = 0.0
            reasons.append("测试中位边界logit=0")
        rows.append({"训练区块": "/".join(str(k + 1) for k in comb), "入选列号": pick,
                     "测试相对排名": float(percentile), "对数胜算": logit,
                     "训练夏普": float(sr0[pick] * np.sqrt(252)),
                     "测试夏普": float(sr1[pick] * np.sqrt(252)),
                     "训练最高夏普并列候选数": train_ties, "测试同分候选数": test_ties,
                     "并列": bool(reasons), "并列原因": "；".join(reasons)})
    frame = pd.DataFrame(rows)
    non_tied = frame.loc[~frame["并列"], "对数胜算"]
    return {"过拟合概率诊断": float((non_tied < 0).mean()) if len(non_tied) else np.nan,
            "诊断标签": "CSCV最高夏普选择器诊断（非预注册选择流程PBO）",
            "判定状态": "可计算（仅非并列组合）" if len(non_tied) else "不可判定（全部并列）",
            "组合数量": len(frame), "非并列分母": len(non_tied), "并列数": int(frame["并列"].sum()),
            "候选数": a.shape[1], "区块数": blocks,
            "并列处理": "训练/测试全部无区别或测试中位logit=0记并列并排除分母（绝对容差1e-12）；"
                        "其他训练最高夏普并列按首列选择，仍依赖列顺序；测试同分使用平均秩",
            "说明": "仅非并列组合的测试相对排名低于中位比例；不是预注册选参流程的PBO、未来亏损概率或干净样本外"}, frame


def white_reality_check(candidate_returns, samples=1000, block=20, seed=20261023):
    """零现金收益基准的最大平均收益单侧区块bootstrap（White RC未学生化版）。"""
    a = np.asarray(candidate_returns, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("需要完整对齐的候选日收益")
    n = len(a)
    observed = np.sqrt(n) * max(float(a.mean(axis=0).max()), 0.0)
    centered = a - a.mean(axis=0)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(samples):
        ix = block_indices(n, block, rng)
        stat = np.sqrt(n) * max(float(centered[ix].mean(axis=0).max()), 0.0)
        exceed += stat >= observed
    return {"多重候选现实检验概率值": (exceed + 1) / (samples + 1),
            "区块交易日": block, "抽样次数": samples, "候选数": a.shape[1],
            "说明": "零现金基准、平均日收益的单侧区块最大值检验；不等于50%年化达标检验"}


if __name__ == "__main__":
    idx = pd.bdate_range("2020-01-01", periods=40)
    e = pd.Series(20000.0, index=idx)
    e.iloc[0] = 19990.0
    assert account_metrics(e)["最大回撤"] == -0.0004999999999999449
    assert account_metrics(pd.Series(20000.0, index=idx))["总收益"] == 0
    ann, mon = annual_monthly(e)
    assert np.isclose(np.prod(1 + mon["收益率"]), e.iloc[-1] / 20000)
    r = np.random.default_rng(1).normal(0.0002, 0.01, size=(320, 5))
    p, detail = probability_backtest_overfit(r)
    assert len(detail) == 70 and 0 <= p["过拟合概率诊断"] <= 1
    assert 0 <= deflated_sharpe(r[:, 0], r)["调整夏普概率"] <= 1
    print("统计函数基础自检通过：初日费用、回撤初值、月度连乘、CSCV、DSR")
