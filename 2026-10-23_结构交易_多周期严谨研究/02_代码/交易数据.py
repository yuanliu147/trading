"""新研究只读适配层。未知历史ST明确标记，不把当前名称或未知状态当已验证。"""
from functools import lru_cache
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from 构建研究数据 import calendar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_数据"
OLD = ROOT.parent / "2026-10-18_成交量融合_2010-2025" / "01_数据"
MAIN_PREFIX = ("000", "001", "002", "003", "600", "601", "603", "605")


def mainboard(code):
    return code.startswith(MAIN_PREFIX)


def round_cent(x):
    return np.floor(np.asarray(x) * 100 + .50000001) / 100


def industries(codes, dates):
    frame = pd.read_csv(OLD / "申万行业分类_时点.csv")
    frame["code"] = frame.code.str.split(".").str[1] + "." + frame.code.str.split(".").str[0].str.upper()
    frame["start_date"] = pd.to_datetime(frame.start_date)
    names = ["未知"] + sorted(frame.industry.dropna().unique())
    mapping = {name: i for i, name in enumerate(names)}
    result = {}
    for code, g in frame.groupby("code"):
        if code not in codes:
            continue
        g = g.sort_values("start_date")
        starts = pd.DatetimeIndex(g.start_date).as_unit("ns").to_numpy()
        ids = g.industry.map(mapping).to_numpy()
        loc = np.searchsorted(starts, dates.as_unit("ns").to_numpy(), side="right") - 1
        result[code] = np.where(loc >= 0, ids[np.maximum(loc, 0)], 0).astype("int16")
    return names, result


class TradingData:
    def __init__(self, unknown_limit="normal", feature_cache=64):
        if unknown_limit not in ("normal", "conservative"):
            raise ValueError("未知涨跌停口径仅支持normal或conservative")
        self.unknown_limit = unknown_limit
        self.full_calendar = calendar()
        self.dates = self.full_calendar[self.full_calendar <= "2026-10-22"]
        self.day_index = {d: i for i, d in enumerate(self.dates)}
        audit = pd.read_csv(DATA / "特征构建审计.csv")
        self.codes = sorted(audit.loc[audit["状态"].eq("成功"), "代码"])
        self.raw = lru_cache(maxsize=256)(self._raw)
        self.features = lru_cache(maxsize=feature_cache)(self._features)
        self.industry_names, self.industry_ids = industries(set(self.codes), self.dates)
        self.market = None
        context_file = DATA / "市场行业上下文.npz"
        if context_file.exists():
            self.market = np.load(context_file)
        self.old_states = {}
        # 只复用已存在的逐日isST与tradestatus；旧缓存的复权preclose不可充当原价参考价。
        old_dir = OLD / "baostock_2016_2026"
        if old_dir.exists():
            for path in old_dir.glob("*.pkl"):
                code = path.stem.split(".")
                if len(code) == 2:
                    code = code[1] + "." + code[0].upper()
                    self.old_states[code] = path

    def _raw(self, code):
        r = pd.read_parquet(DATA / "原始分股" / f"{code}.parquet").reindex(self.dates)
        for col in ("isST", "tradestatus"):
            if col not in r:
                r[col] = np.nan
        exact_reference = pd.Series(np.nan, index=self.dates)
        state_path = DATA / "历史状态" / f"{code}.parquet"
        if state_path.exists():
            s = pd.read_parquet(state_path).set_index("date").reindex(self.dates)
            for col in ("isST", "tradestatus"):
                r[col] = r[col].combine_first(s[col])
            exact_reference = s["preclose"]
        elif code in self.old_states:
            with self.old_states[code].open("rb") as handle:
                s = pickle.load(handle)
            if "date" in s:
                s = s.copy()
                s["date"] = pd.to_datetime(s.date)
                s = s.drop_duplicates("date").set_index("date").reindex(self.dates)
                for col in ("isST", "tradestatus"):
                    if col in s:
                        r[col] = r[col].combine_first(pd.to_numeric(s[col], errors="coerce"))
        # 已知停牌保持缺价但因子可沿用至新观察；不要向后填入未来因子。
        factor = r.adj_factor.ffill()
        previous_adjusted = (r.close * factor).ffill().shift(1)
        r["reference"] = exact_reference.combine_first(previous_adjusted / factor)
        r["adj_factor"] = factor
        r["prev_amount20"] = r.amount.fillna(0).rolling(20, min_periods=20).mean().shift(1)
        r["tradable"] = ((r.volume > 0) & (r.amount > 0) & (r.open > 0)
                         & ~r.tradestatus.eq(0))
        r["st_known"] = r.isST.notna()
        rate = np.full(len(r), .1)
        if code.startswith(("300", "301")):
            rate[self.dates >= "2020-08-24"] = .2
        elif code.startswith(("688", "689")):
            rate[:] = .2
        if mainboard(code):
            rate[r.isST.eq(1).to_numpy() & (self.dates < "2026-07-06")] = .05
            if self.unknown_limit == "conservative":
                rate[~r.st_known.to_numpy() & (self.dates < "2026-07-06")] = .05
        r["limit_up"] = round_cent(r.reference * (1 + rate))
        r["limit_down"] = round_cent(r.reference * (1 - rate))
        r["minimum_buy_size"] = 200 if code.startswith(("688", "689")) else 100
        return r

    def _features(self, code):
        return pd.read_parquet(DATA / "结构特征" / f"{code}.parquet")

    def get_day(self, date, codes):
        out = {}
        if date not in self.day_index:
            raise ValueError("请求日期不在已核验日历")
        i = self.day_index[date]
        cols = ["open", "high", "low", "close", "adj_factor", "prev_amount20", "tradable",
                "limit_up", "limit_down", "minimum_buy_size", "st_known", "isST"]
        for code in codes:
            if code in self.codes:
                row = self.raw(code).iloc[i]
                out[code] = {col: row[col] for col in cols}
        return out

    def context(self, code):
        if self.market is None:
            raise RuntimeError("须先构建市场行业上下文")
        ids = self.industry_ids.get(code, np.zeros(len(self.dates), dtype="int16"))
        index = np.arange(len(self.dates))
        return {
            "mainboard_only": True,
            "market_breadth60": pd.Series(self.market["breadth60"], index=self.dates),
            "industry_breadth60": pd.Series(self.market["industry_breadth60"][index, ids], index=self.dates),
            "industry_ret20": pd.Series(self.market["industry_ret20"][index, ids], index=self.dates),
        }


def build_context():
    data = TradingData(feature_cache=1)
    n, k = len(data.dates), len(data.industry_names)
    counts = np.zeros(n)
    above = np.zeros(n)
    ret_sum, ret_count = np.zeros(n), np.zeros(n)
    amounts, lu, ld = np.zeros(n), np.zeros(n), np.zeros(n)
    inds_n, inds_up, inds_r, inds_rn = [np.zeros((n, k)) for _ in range(4)]
    evidence = []
    index = np.arange(n)
    for pos, code in enumerate(data.codes, 1):
        r, f = data.raw(code), data.features(code)
        close = f.d_close
        mean = close.rolling(60, min_periods=60).mean()
        valid = r.tradable & (f.d_age >= 120) & mean.notna() & ~r.isST.eq(1)
        up = (close > mean) & valid
        counts += valid.to_numpy()
        above += up.to_numpy()
        dr = close.pct_change(fill_method=None)
        rv = valid & dr.notna()
        ret_sum += dr.where(rv, 0).to_numpy()
        ret_count += rv.to_numpy()
        amounts += r.amount.where(valid, 0).fillna(0).to_numpy()
        lu += (valid & r.close.ge(r.limit_up)).to_numpy()
        ld += (valid & r.close.le(r.limit_down)).to_numpy()
        ids = data.industry_ids.get(code, np.zeros(n, dtype="int16"))
        iv = valid.to_numpy() & (ids > 0)
        np.add.at(inds_n, (index, ids), iv)
        np.add.at(inds_up, (index, ids), up.to_numpy() & iv)
        irv = iv & f.d_ret20.notna().to_numpy()
        np.add.at(inds_r, (index, ids), f.d_ret20.where(irv, 0).to_numpy())
        np.add.at(inds_rn, (index, ids), irv)
        evidence.append({"代码": code, "有效观测日": int(valid.sum()),
                         "已知状态日": int((r.tradable & r.st_known).sum()), "有成交日": int(r.tradable.sum())})
        if pos % 500 == 0:
            print(f"市场上下文{pos}/{len(data.codes)}", flush=True)
    divide = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    breadth = divide(above, counts)
    daily = divide(ret_sum, ret_count)
    np.savez_compressed(DATA / "市场行业上下文.npz", breadth60=breadth, daily_return=daily,
                        industry_breadth60=divide(inds_up, inds_n), industry_ret20=divide(inds_r, inds_rn))
    pd.DataFrame({"日期": data.dates, "可得沪深有效股票数": counts, "宽度六十": breadth,
                  "等权标记日收益": daily, "成交额": amounts, "估计涨停占比": divide(lu, counts),
                  "估计跌停占比": divide(ld, counts)}).to_csv(DATA / "市场上下文.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evidence).to_csv(ROOT / "04_证据" / "历史交易状态覆盖.csv", index=False, encoding="utf-8-sig")
    meta = {"股票数": len(data.codes), "日期数": n, "行业名称": data.industry_names,
            "口径": "可得沪深股票、已观察行情；当前日收盘可知；历史ST未知保留并注明，未称完整无偏全A",
            "行业口径": "历史表start_date生效映射；未拥有逐次公告可得时间和数据库版本快照",
            "未知涨跌幅默认": "普通规则10/20%；另跑未知主板5%保守压力；2026-07-06后主板ST10%"}
    (ROOT / "04_证据" / "市场上下文审计.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("市场行业上下文生成完成", flush=True)


if __name__ == "__main__":
    build_context()
