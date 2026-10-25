
        WITH joined AS (
            SELECT f.date, f.thscode, f.return_5d, f.return_20d, f.return_60d,
                   f.close_ma20, f.close_ma60, f.ret1, f.limit_up_10,
                   m.cluster_id, m.cluster_strength, m.cluster_size
            FROM read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/core.parquet') f
            JOIN read_parquet('/Users/admin/WorkSpace/trading/trading_hithink/data/work/cluster_membership.parquet') m
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
    