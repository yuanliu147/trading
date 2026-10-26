from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "data" / "snapshot"
DATABASE = Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb")
EXPORTS = {
    "mainboard_daily": ROOT / "sql" / "export_mainboard.sql",
    "adjustment_events": ROOT / "sql" / "export_events.sql",
    "symbols": ROOT / "sql" / "export_symbols.sql",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def cli_export(sql: str, output: Path) -> dict:
    result = subprocess.run(
        [
            "hithink-finance",
            "db",
            "export",
            "--db",
            str(DATABASE),
            "--sql",
            sql,
            "--output",
            str(output),
            "--file-format",
            "parquet",
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CLI export failed ({result.returncode}): {result.stderr.strip()}")
    envelope = json.loads(result.stdout)
    if not envelope.get("ok"):
        raise RuntimeError(f"CLI export envelope failed: {envelope}")
    return envelope["data"]


def export_manifest() -> dict:
    if not DATABASE.is_file():
        raise FileNotFoundError(DATABASE)
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    exports = {}
    for name, sql_path in EXPORTS.items():
        output = SNAPSHOT / f"{name}.parquet"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen snapshot: {output}")
        sql = sql_path.read_text()
        response = cli_export(sql, output)
        exports[name] = {
            "path": str(output.relative_to(ROOT)),
            "sql_path": str(sql_path.relative_to(ROOT)),
            "sql": sql,
            "sql_sha256": sha256(sql_path),
            "file_sha256": sha256(output),
            "bytes": output.stat().st_size,
            "cli_response": response,
        }
    db_stat = DATABASE.stat()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"database": str(DATABASE), "bytes": db_stat.st_size, "mtime_ns": db_stat.st_mtime_ns},
        "research_config_sha256": sha256(ROOT / "config" / "research.json"),
        "remote_data_used": False,
        "legacy_panel_used": False,
        "exports": exports,
    }


def audit_snapshot() -> tuple[dict, list[dict]]:
    daily_path = SNAPSHOT / "mainboard_daily.parquet"
    events_path = SNAPSHOT / "adjustment_events.parquet"
    symbols_path = SNAPSHOT / "symbols.parquet"
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    for view, path in [("daily", daily_path), ("events", events_path), ("symbols", symbols_path)]:
        literal = str(path).replace("'", "''")
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{literal}')")
    summary = con.execute(
        """
        SELECT count(*) AS rows,
               count(DISTINCT thscode) AS symbols,
               min(date) AS start_date,
               max(date) AS end_date,
               count(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL) AS missing_ohlc,
               count(prev_close) AS nonnull_prev_close,
               count(*) FILTER (WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0) AS nonpositive_ohlc,
               count(*) FILTER (WHERE high < low OR high < greatest(open, close) OR low > least(open, close)) AS invalid_ohlc,
               count(*) FILTER (WHERE volume < 0 OR amount < 0) AS negative_activity,
               count(*) FILTER (WHERE volume = 0) AS zero_volume,
               count(*) FILTER (WHERE amount = 0) AS zero_amount,
               count(*) FILTER (WHERE hfq_open IS NULL OR hfq_high IS NULL OR hfq_low IS NULL OR hfq_close IS NULL) AS missing_hfq,
               count(*) FILTER (WHERE backward_factor IS NULL OR forward_factor IS NULL) AS missing_factor
        FROM daily
        """
    ).fetchdf().iloc[0].to_dict()
    duplicate_rows = con.execute(
        "SELECT count(*) FROM (SELECT thscode, date FROM daily GROUP BY thscode, date HAVING count(*) > 1)"
    ).fetchone()[0]
    prev_close = con.execute(
        """
        WITH x AS (
            SELECT thscode, date, close, prev_close,
                   lag(close) OVER (PARTITION BY thscode ORDER BY date) AS previous_actual_close
            FROM daily
        )
        SELECT count(*) FILTER (WHERE previous_actual_close IS NOT NULL) AS rows_with_previous,
               count(*) FILTER (WHERE previous_actual_close IS NOT NULL AND abs(prev_close - previous_actual_close) > 0.011) AS prev_close_mismatch,
               count(*) FILTER (WHERE previous_actual_close IS NOT NULL AND abs(prev_close - previous_actual_close) > 1.0) AS large_prev_close_mismatch
        FROM x
        """
    ).fetchdf().iloc[0].to_dict()
    factor = con.execute(
        """
        SELECT count(*) FILTER (WHERE abs(hfq_close / close - backward_factor) < 0.0000001) AS backward_formula_matches,
               count(*) FILTER (WHERE abs(hfq_close / close - forward_factor) < 0.0000001) AS forward_formula_matches,
               min(hfq_close / close) AS minimum_hfq_ratio,
               max(hfq_close / close) AS maximum_hfq_ratio
        FROM daily
        """
    ).fetchdf().iloc[0].to_dict()
    symbol_counts = con.execute(
        """
        SELECT count(*) AS current_mainboard_symbols,
               count(name) AS nonnull_names,
               count(*) FILTER (WHERE upper(name) LIKE '%ST%' OR name LIKE '%退%') AS excluded_current_names
        FROM symbols
        """
    ).fetchdf().iloc[0].to_dict()
    event_counts = con.execute(
        """
        SELECT count(*) AS rows,
               count(DISTINCT thscode) AS symbols,
               count(*) FILTER (WHERE coalesce(dividend_per_share, 0) <> 0) AS dividends,
               count(*) FILTER (WHERE coalesce(per_share_bonus, 0) <> 0) AS bonuses,
               count(*) FILTER (WHERE coalesce(rights_ratio, 0) <> 0) AS rights,
               count(*) FILTER (WHERE ex_date > (SELECT max(date) FROM daily)) AS after_last_market_date
        FROM events
        """
    ).fetchdf().iloc[0].to_dict()
    coverage = con.execute(
        """
        SELECT year(date) AS year, count(DISTINCT thscode) AS symbols, count(*) AS rows,
               count(DISTINCT date) AS sessions,
               count(*) FILTER (WHERE volume = 0) AS zero_volume_rows
        FROM daily
        GROUP BY year(date)
        ORDER BY year(date)
        """
    ).fetchdf().to_dict("records")
    con.close()
    audit = {
        "summary": summary,
        "duplicate_key_groups": duplicate_rows,
        "previous_close_audit": prev_close,
        "adjustment_factor_audit": factor,
        "symbol_audit": symbol_counts,
        "event_audit": event_counts,
        "scope_warning": "Current-listed main-board history only; delisted stocks are excluded and survivorship bias remains.",
        "historical_st_available": False,
        "historical_industry_membership_available": False,
        "intraday_path_available": False,
    }
    return audit, coverage


def verify_reuse(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    expected_config = manifest["research_config_sha256"]
    actual_config = sha256(ROOT / "config" / "research.json")
    if expected_config != actual_config:
        raise RuntimeError("Research configuration changed; refusing to reuse snapshot")
    for item in manifest["exports"].values():
        path = ROOT / item["path"]
        if sha256(path) != item["file_sha256"]:
            raise RuntimeError(f"Snapshot hash mismatch: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    manifest_path = SNAPSHOT / "manifest.json"
    if args.reuse:
        manifest = verify_reuse(manifest_path)
    else:
        manifest = export_manifest()
        write_json(manifest_path, manifest)
    audit, coverage = audit_snapshot()
    write_json(SNAPSHOT / "audit.json", audit)
    import pandas as pd

    pd.DataFrame(coverage).to_csv(SNAPSHOT / "coverage_by_year.csv", index=False)
    if audit["duplicate_key_groups"] != 0:
        raise AssertionError("Duplicate thscode/date keys")
    for field in ["missing_ohlc", "nonpositive_ohlc", "invalid_ohlc", "negative_activity", "missing_hfq"]:
        if audit["summary"][field] != 0:
            raise AssertionError(f"Unexpected data quality failure: {field}")
    print(json.dumps({"ok": True, "manifest": str(manifest_path), "audit": audit}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
