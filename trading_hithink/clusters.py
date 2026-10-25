from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "data" / "work"


def literal(path: Path) -> str:
    return str(path).replace("'", "''")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = WORK / "core.parquet"
    membership_path = WORK / "cluster_membership.parquet"
    daily_path = WORK / "cluster_daily.parquet"
    if membership_path.exists() or daily_path.exists():
        raise FileExistsError("Cluster outputs already exist")
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit TO '12GB'")
    frame = con.execute(
        f"""
        SELECT date, thscode, ret1, amount20, close, listing_session
        FROM read_parquet('{literal(source)}')
        """
    ).fetchdf()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = np.array(sorted(frame["date"].unique()))
    returns = frame.pivot(index="date", columns="thscode", values="ret1").sort_index()
    amounts = frame.pivot(index="date", columns="thscode", values="amount20").sort_index()
    prices = frame.pivot(index="date", columns="thscode", values="close").sort_index()
    sessions = frame.pivot(index="date", columns="thscode", values="listing_session").sort_index()
    months = pd.period_range(pd.Timestamp(dates[0]).to_period("M"), pd.Timestamp(dates[-1]).to_period("M"), freq="M")
    records = []
    diagnostics = []
    for month in months[4:]:
        effective_date = month.start_time
        train_dates = dates[dates < np.datetime64(effective_date)]
        if len(train_dates) < 160:
            continue
        train_dates = train_dates[-120:]
        last_date = train_dates[-1]
        history = returns.loc[train_dates]
        eligible = history.columns[
            (history.count() >= 100)
            & (sessions.loc[last_date] >= 250)
            & (prices.loc[last_date] >= 2)
            & (prices.loc[last_date] <= 100)
            & amounts.loc[last_date].notna()
        ]
        if len(eligible) < 100:
            continue
        selected = amounts.loc[last_date, eligible].nlargest(min(800, len(eligible))).index
        x = history[selected].to_numpy(dtype=np.float64)
        means = np.nanmean(x, axis=0)
        stds = np.nanstd(x, axis=0)
        valid = stds > 1e-8
        x = x[:, valid]
        selected = selected[valid]
        means = means[valid]
        stds = stds[valid]
        x = (x - means) / stds
        x = np.nan_to_num(x, nan=0.0)
        u, singular, vt = np.linalg.svd(x, full_matrices=False)
        component_count = min(7, u.shape[1] - 1)
        correlations = (x.T @ u[:, 1 : component_count + 1]) / np.sqrt(x.shape[0])
        dominant = np.argmax(np.abs(correlations), axis=1)
        strength = np.abs(correlations).max(axis=1)
        signs = np.where(correlations[np.arange(len(selected)), dominant] >= 0, 1, 0)
        cluster_ids = dominant * 2 + signs
        cluster_ids = np.where(strength >= 0.25, cluster_ids, -1)
        month_frame = pd.DataFrame(
            {
                "effective_month": effective_date,
                "thscode": selected,
                "cluster_id": cluster_ids.astype("int16"),
                "cluster_strength": strength,
                "cluster_train_end": last_date,
                "cluster_amount20": amounts.loc[last_date, selected].to_numpy(),
            }
        )
        sizes = month_frame.loc[month_frame["cluster_id"] >= 0, "cluster_id"].value_counts()
        month_frame["cluster_size"] = month_frame["cluster_id"].map(sizes).fillna(0).astype("int16")
        records.append(month_frame)
        diagnostics.append(
            {
                "effective_month": effective_date,
                "train_start": train_dates[0],
                "train_end": last_date,
                "eligible_symbols": len(eligible),
                "clustered_symbols": int((cluster_ids >= 0).sum()),
                "clusters": int(sizes.shape[0]),
                "median_cluster_size": float(sizes.median()) if len(sizes) else 0,
                "median_strength": float(np.median(strength)),
            }
        )
    if not records:
        raise RuntimeError("No cluster memberships generated")
    membership = pd.concat(records, ignore_index=True)
    membership.to_parquet(membership_path, index=False)
    pd.DataFrame(diagnostics).to_csv(WORK / "cluster_diagnostics.csv", index=False)
    sql = f"""
        WITH joined AS (
            SELECT f.date, f.thscode, f.return_5d, f.return_20d, f.return_60d,
                   f.close_ma20, f.close_ma60, f.ret1, f.limit_up_10,
                   m.cluster_id, m.cluster_strength, m.cluster_size
            FROM read_parquet('{literal(source)}') f
            JOIN read_parquet('{literal(membership_path)}') m
              ON m.thscode = f.thscode AND m.effective_month = date_trunc('month', f.date)
        ), aggregates AS (
            SELECT *,
                   count(*) FILTER (WHERE cluster_id >= 0) OVER (PARTITION BY date, cluster_id) AS peer_count,
                   sum(return_5d) OVER (PARTITION BY date, cluster_id) AS sum_return5,
                   sum(return_20d) OVER (PARTITION BY date, cluster_id) AS sum_return20,
                   sum(return_60d) OVER (PARTITION BY date, cluster_id) AS sum_return60,
                   sum(close_ma20) OVER (PARTITION BY date, cluster_id) AS sum_close_ma20,
                   sum(close_ma60) OVER (PARTITION BY date, cluster_id) AS sum_close_ma60,
                   sum((close_ma20 > 0)::INTEGER) OVER (PARTITION BY date, cluster_id) AS sum_breadth20,
                   sum((close_ma60 > 0)::INTEGER) OVER (PARTITION BY date, cluster_id) AS sum_breadth60,
                   sum(ret1) OVER (PARTITION BY date, cluster_id) AS sum_ret1,
                   sum(limit_up_10::INTEGER) OVER (PARTITION BY date, cluster_id) AS sum_limit_up
            FROM joined
        )
        SELECT date, thscode, cluster_id, cluster_strength, cluster_size,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_return5 - return_5d) / (peer_count - 1) END AS peer_return5,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_return20 - return_20d) / (peer_count - 1) END AS peer_return20,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_return60 - return_60d) / (peer_count - 1) END AS peer_return60,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_close_ma20 - close_ma20) / (peer_count - 1) END AS peer_close_ma20,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_close_ma60 - close_ma60) / (peer_count - 1) END AS peer_close_ma60,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_breadth20 - (close_ma20 > 0)::INTEGER) / (peer_count - 1) END AS peer_breadth20,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_breadth60 - (close_ma60 > 0)::INTEGER) / (peer_count - 1) END AS peer_breadth60,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_ret1 - ret1) / (peer_count - 1) END AS peer_return1,
               CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_limit_up - limit_up_10::INTEGER) / (peer_count - 1) END AS peer_limit_up_share,
               return_20d - CASE WHEN cluster_id >= 0 AND peer_count > 1 THEN (sum_return20 - return_20d) / (peer_count - 1) END AS cluster_relative20
        FROM aggregates
    """
    (WORK / "build_cluster_daily.sql").write_text(sql)
    con.execute(f"COPY ({sql}) TO '{literal(daily_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{literal(daily_path)}')").fetchone()[0]
    con.close()
    write_json(
        WORK / "cluster_metadata.json",
        {
            "method": "Monthly trailing-120-session standardized-return PCA; exclude first market component; dominant signed components 2-8",
            "maximum_liquid_symbols_per_month": 800,
            "minimum_history_observations": 100,
            "minimum_absolute_component_correlation": 0.25,
            "membership_uses_current_month_returns": False,
            "official_industry_classification": False,
            "daily_rows": rows,
            "membership_sha256": sha256(membership_path),
            "daily_sha256": sha256(daily_path),
        },
    )
    print(json.dumps({"ok": True, "memberships": len(membership), "daily_rows": rows}))


if __name__ == "__main__":
    main()
