"""Unit tests for slave/naming.py — pure-python; no Modal calls."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "slave"))


class TestResolveExpId(unittest.TestCase):
    def test_empty_returns_empty(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(os.environ, {"EXP_ID": ""}, clear=False):
            self.assertEqual(resolve_exp_id(), "")

    def test_whitespace_treated_as_empty(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(os.environ, {"EXP_ID": "   "}, clear=False):
            self.assertEqual(resolve_exp_id(), "")

    def test_valid_id_passes_through(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(os.environ, {"EXP_ID": "exp-01"}, clear=False):
            self.assertEqual(resolve_exp_id(), "exp-01")

    def test_valid_id_with_dot_underscore(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(
            os.environ, {"EXP_ID": "exp_01.v2"}, clear=False
        ):
            self.assertEqual(resolve_exp_id(), "exp_01.v2")

    def test_invalid_id_with_space_rejected(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(os.environ, {"EXP_ID": "bad id"}, clear=False):
            with self.assertRaises(ValueError):
                resolve_exp_id()

    def test_invalid_id_with_slash_rejected(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(os.environ, {"EXP_ID": "exp/01"}, clear=False):
            with self.assertRaises(ValueError):
                resolve_exp_id()

    def test_too_long_id_rejected(self) -> None:
        from naming import resolve_exp_id

        with mock.patch.dict(os.environ, {"EXP_ID": "a" * 41}, clear=False):
            with self.assertRaises(ValueError):
                resolve_exp_id()


class TestAppName(unittest.TestCase):
    def test_legacy_unprefixed(self) -> None:
        from naming import app_name

        self.assertEqual(
            app_name("qwen3-8b", exp_id=""), "sglang-slave-qwen3-8b"
        )

    def test_legacy_with_replica(self) -> None:
        from naming import app_name

        self.assertEqual(
            app_name("qwen3-8b", "r0", exp_id=""),
            "sglang-slave-qwen3-8b-r0",
        )

    def test_namespaced(self) -> None:
        from naming import app_name

        self.assertEqual(
            app_name("qwen3-8b", exp_id="exp-01"),
            "exp-01-sglang-slave-qwen3-8b",
        )

    def test_namespaced_with_replica(self) -> None:
        from naming import app_name

        self.assertEqual(
            app_name("qwen3-8b", "r3", exp_id="exp-01"),
            "exp-01-sglang-slave-qwen3-8b-r3",
        )

    def test_implicit_exp_id_from_env(self) -> None:
        from naming import app_name

        with mock.patch.dict(os.environ, {"EXP_ID": "exp-42"}, clear=False):
            self.assertEqual(
                app_name("qwen3-1p7b"),
                "exp-42-sglang-slave-qwen3-1p7b",
            )


class TestVolumeNames(unittest.TestCase):
    def test_data_volume_legacy(self) -> None:
        from naming import data_volume_name

        self.assertEqual(data_volume_name(""), "inference-farm-data")

    def test_data_volume_namespaced(self) -> None:
        from naming import data_volume_name

        self.assertEqual(data_volume_name("exp-01"), "exp-01-data")

    def test_checkpoint_volume_legacy(self) -> None:
        from naming import checkpoint_volume_name

        self.assertEqual(
            checkpoint_volume_name(""), "sglang-slave-checkpoints"
        )

    def test_checkpoint_volume_namespaced(self) -> None:
        from naming import checkpoint_volume_name

        self.assertEqual(
            checkpoint_volume_name("exp-01"), "exp-01-checkpoints"
        )

    def test_hf_cache_always_shared(self) -> None:
        from naming import hf_cache_volume_name

        # Whether or not EXP_ID is set, HF cache stays shared so model
        # weight pulls amortize across experiments.
        self.assertEqual(hf_cache_volume_name(""), "sglang-hf-cache")
        self.assertEqual(
            hf_cache_volume_name("exp-01"), "sglang-hf-cache"
        )


class TestDictNames(unittest.TestCase):
    def test_progress_dict_legacy(self) -> None:
        from naming import progress_dict_name

        self.assertEqual(
            progress_dict_name("qwen3-8b", exp_id=""),
            "sglang-slave-qwen3-8b-progress",
        )

    def test_progress_dict_namespaced_with_replica(self) -> None:
        from naming import progress_dict_name

        self.assertEqual(
            progress_dict_name("qwen3-8b", "r1", exp_id="exp-01"),
            "exp-01-sglang-slave-qwen3-8b-r1-progress",
        )

    def test_fc_dict_legacy(self) -> None:
        from naming import fc_dict_name

        self.assertEqual(
            fc_dict_name("qwen3-8b", exp_id=""),
            "sglang-slave-qwen3-8b-fc",
        )

    def test_fc_dict_namespaced_with_replica(self) -> None:
        from naming import fc_dict_name

        self.assertEqual(
            fc_dict_name("qwen3-8b", "r2", exp_id="exp-01"),
            "exp-01-sglang-slave-qwen3-8b-r2-fc",
        )


class TestPrefetchAppName(unittest.TestCase):
    def test_legacy(self) -> None:
        from naming import prefetch_app_name

        self.assertEqual(
            prefetch_app_name("qwen3-8b", exp_id=""),
            "sglang-prefetch-qwen3-8b",
        )

    def test_namespaced(self) -> None:
        from naming import prefetch_app_name

        self.assertEqual(
            prefetch_app_name("qwen3-8b", exp_id="exp-01"),
            "exp-01-sglang-prefetch-qwen3-8b",
        )


class TestNoCollisionAcrossExperiments(unittest.TestCase):
    """Two experiments running the same model must never share an app /
    volume / dict name. This is the structural guarantee that lets you
    cp -r the repo for parallel runs."""

    def test_app_names_differ(self) -> None:
        from naming import app_name

        a = app_name("qwen3-8b", exp_id="exp-01")
        b = app_name("qwen3-8b", exp_id="exp-02")
        self.assertNotEqual(a, b)

    def test_data_volumes_differ(self) -> None:
        from naming import data_volume_name

        self.assertNotEqual(
            data_volume_name("exp-01"), data_volume_name("exp-02")
        )

    def test_checkpoint_volumes_differ(self) -> None:
        from naming import checkpoint_volume_name

        self.assertNotEqual(
            checkpoint_volume_name("exp-01"),
            checkpoint_volume_name("exp-02"),
        )

    def test_progress_dicts_differ(self) -> None:
        from naming import progress_dict_name

        self.assertNotEqual(
            progress_dict_name("qwen3-8b", "r0", exp_id="exp-01"),
            progress_dict_name("qwen3-8b", "r0", exp_id="exp-02"),
        )


if __name__ == "__main__":
    unittest.main()
