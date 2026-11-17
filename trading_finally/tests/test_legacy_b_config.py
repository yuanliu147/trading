import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from trend_system.legacy_b_config import load_legacy_b_config
from trend_system.legacy_b_robustness import run_legacy_b_robustness
from trend_system.legacy_b_runner import run_legacy_b


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "legacy_b_recovery.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_modified(tmp: str, mutate) -> Path:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = Path(tmp) / "modified.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class LegacyBConfigLoaderTest(unittest.TestCase):
    def test_official_config_loads_frozen_values(self):
        config = load_legacy_b_config(CONFIG_PATH)

        self.assertEqual(config.initial_cash, 20000)
        self.assertEqual(config.max_main_drawdown, 0.20)
        self.assertEqual(config.random_paths, 20)
        self.assertEqual(config.random_positive_min, 16)
        self.assertEqual(config.status, "PENDING_DATA_GATE")
        self.assertEqual(len(config.candidates), 1)
        candidate = config.candidates[0]
        self.assertEqual(candidate.name, "early_trend_breadth")
        self.assertEqual(candidate.evaluation_status, "PENDING_DATA_GATE")
        self.assertEqual(candidate.not_evaluable_fields, ())
        self.assertEqual(candidate.max_slots, 6)
        self.assertEqual(candidate.risk_per_trade, 0.005)
        self.assertEqual(candidate.max_hold, 20)
        self.assertEqual(candidate.catastrophe_stop_pct, 0.20)
        self.assertTrue(candidate.gap_filter)
        self.assertEqual(
            {source.path: source.sha256 for source in candidate.sources},
            {
                "/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/02_代码/30b_组合回测_含退市.py":
                    "8406120aedd2ad21cbbca35e87fb784262434730b3399697b7afbe822b4686e8",
                "/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/02_代码/10b_构建全A面板_含退市.py":
                    "d45892c97a2995d1c3fc7349c2dd9162be6b1d11825a330673136c88300f3d37",
            },
        )

    def test_modified_copy_is_rejected_by_official_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_modified(
                tmp, lambda raw: raw["candidates"][0].__setitem__("risk_per_trade", 0.05)
            )
            with self.assertRaisesRegex(ValueError, "official config"):
                load_legacy_b_config(path)

    def test_nonempty_not_evaluable_fields_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_modified(
                tmp,
                lambda raw: raw["candidates"][0].__setitem__(
                    "not_evaluable_fields", ["risk_per_trade"]
                ),
            )
            with patch("trend_system.legacy_b_config.OFFICIAL_CONFIG_SHA256", _sha256(path)):
                with self.assertRaisesRegex(ValueError, "no not_evaluable_fields"):
                    load_legacy_b_config(path)

    def test_wrong_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_modified(
                tmp,
                lambda raw: raw["candidates"][0]["sources"][0].__setitem__("sha256", "0" * 64),
            )
            with patch("trend_system.legacy_b_config.OFFICIAL_CONFIG_SHA256", _sha256(path)):
                with self.assertRaisesRegex(ValueError, "source set/hashes"):
                    load_legacy_b_config(path)

    def test_invalid_candidate_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_modified(
                tmp,
                lambda raw: raw["candidates"][0].__setitem__(
                    "evaluation_status", "NOT_EVALUABLE"
                ),
            )
            with patch("trend_system.legacy_b_config.OFFICIAL_CONFIG_SHA256", _sha256(path)):
                with self.assertRaisesRegex(ValueError, "evaluation_status invalid"):
                    load_legacy_b_config(path)


class LegacyBRunnerGateTest(unittest.TestCase):
    def _root(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "config").mkdir()
        (root / "results").mkdir()
        shutil.copy(CONFIG_PATH, root / "config" / "legacy_b_recovery.json")
        return root

    def _audit(self, root: Path, config_sha: str | None = None) -> None:
        payload = {
            "restored": True,
            "status": "EVALUABLE",
            "config": {"sha256": config_sha or _sha256(root / "config" / "legacy_b_recovery.json")},
        }
        (root / "results" / "legacy_b_recovery_audit.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_main_run_requires_explicit_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            self._audit(root)
            with self.assertRaisesRegex(ValueError, "actual --config"):
                run_legacy_b(None, config_path=None, root=root)

    def test_main_run_rejects_alternative_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            self._audit(root)
            other = _write_modified(
                tmp, lambda raw: raw["candidates"][0].__setitem__("risk_per_trade", 0.05)
            )
            with self.assertRaisesRegex(ValueError, "actual --config"):
                run_legacy_b(None, config_path=other, root=root)

    def test_main_run_rejects_missing_recovery_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            with self.assertRaisesRegex(ValueError, "audit is missing"):
                run_legacy_b(None, config_path=root / "config" / "legacy_b_recovery.json", root=root)

    def test_main_run_rejects_audit_with_other_config_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            self._audit(root, config_sha="0" * 64)
            with self.assertRaisesRegex(ValueError, "does not match the official config"):
                run_legacy_b(None, config_path=root / "config" / "legacy_b_recovery.json", root=root)

    def _robustness_root(self, tmp: str) -> Path:
        root = self._root(tmp)
        self._audit(root)
        config_sha = _sha256(root / "config" / "legacy_b_recovery.json")
        audit_path = root / "results" / "legacy_b_recovery_audit.json"
        (root / "results" / "legacy_b_summary.json").write_text(
            json.dumps({"status": "EVALUABLE"}), encoding="utf-8"
        )
        main_dir = root / "results" / "legacy_candidates" / "early_trend_breadth"
        main_dir.mkdir(parents=True)
        (main_dir / "performance_summary.json").write_text("{}", encoding="utf-8")
        (main_dir / "run_manifest.json").write_text(
            json.dumps({
                "config": {"sha256": config_sha},
                "recovery_audit": {"sha256": _sha256(audit_path)},
                "code": {},
            }),
            encoding="utf-8",
        )
        (root / ".sdd-legacy").mkdir()
        (root / ".sdd-legacy" / "b_main_freeze.json").write_text(
            json.dumps({"files": {}}), encoding="utf-8"
        )
        return root

    def test_robustness_requires_explicit_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._robustness_root(tmp)
            with self.assertRaisesRegex(ValueError, "robustness --config"):
                run_legacy_b_robustness(None, config_path=None, root=root)

    def test_robustness_rejects_alternative_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._robustness_root(tmp)
            other = _write_modified(
                tmp, lambda raw: raw["candidates"][0].__setitem__("risk_per_trade", 0.05)
            )
            with self.assertRaisesRegex(ValueError, "robustness --config"):
                run_legacy_b_robustness(None, config_path=other, root=root)


if __name__ == "__main__":
    unittest.main()
