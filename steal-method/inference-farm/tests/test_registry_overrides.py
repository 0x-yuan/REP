"""Unit tests for slave/registry.py override layer.

Verifies that experiment.toml overrides are applied to ModelConfig at
get_config() time and that bad input is rejected with a useful error.
No Modal calls.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "slave"))


class TestOverridesFromToml(unittest.TestCase):
    def setUp(self) -> None:
        # Make sure no env-var override leaks between tests.
        self._env_patch = mock.patch.dict(
            os.environ, {"EXPERIMENT_OVERRIDES_JSON": ""}, clear=False
        )
        self._env_patch.start()

    def tearDown(self) -> None:
        self._env_patch.stop()

    def test_no_file_returns_baseline(self) -> None:
        import registry

        with mock.patch.object(
            registry, "_load_overrides_from_path", return_value={}
        ):
            cfg = registry.get_config("qwen3-8b")
        self.assertEqual(cfg.key, "qwen3-8b")
        # Baseline value from registry.
        self.assertEqual(cfg.context_length, 131072)

    def test_field_overrides_applied(self) -> None:
        import registry

        overrides = {
            "qwen3-8b": {
                "default_max_tokens": 4096,
                "context_length": 65536,
                "max_running_requests": 32,
            }
        }
        with mock.patch.object(
            registry, "_load_overrides_from_path", return_value=overrides
        ):
            cfg = registry.get_config("qwen3-8b")
        self.assertEqual(cfg.default_max_tokens, 4096)
        self.assertEqual(cfg.context_length, 65536)
        self.assertEqual(cfg.max_running_requests, 32)
        # Untouched fields keep baseline values.
        self.assertEqual(cfg.gpu, "H200")

    def test_other_models_untouched(self) -> None:
        import registry

        overrides = {"qwen3-8b": {"context_length": 16384}}
        with mock.patch.object(
            registry, "_load_overrides_from_path", return_value=overrides
        ):
            cfg = registry.get_config("qwen3-14b")
        self.assertEqual(cfg.context_length, 131072)

    def test_unknown_field_rejected(self) -> None:
        import registry

        overrides = {
            "qwen3-8b": {
                "context_length": 65536,
                "made_up_field": True,
            }
        }
        with mock.patch.object(
            registry, "_load_overrides_from_path", return_value=overrides
        ):
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                registry.get_config("qwen3-8b")

    def test_real_toml_file_round_trip(self) -> None:
        """End-to-end: write an actual experiment.toml, point the loader
        at it, parse, apply."""
        import registry

        toml_text = textwrap.dedent(
            """\
            [models.qwen3-8b]
            default_max_tokens = 4096
            context_length = 65536
            cuda_graph_bs = [1, 2, 4, 8]
            """
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "experiment.toml"
            toml_path.write_text(toml_text)
            parsed = registry._load_overrides_from_path(toml_path)
        self.assertIn("qwen3-8b", parsed)
        self.assertEqual(parsed["qwen3-8b"]["default_max_tokens"], 4096)
        self.assertEqual(parsed["qwen3-8b"]["cuda_graph_bs"], [1, 2, 4, 8])

    def test_missing_toml_returns_empty(self) -> None:
        import registry

        result = registry._load_overrides_from_path(
            Path("/no/such/path/experiment.toml")
        )
        self.assertEqual(result, {})


class TestOverridesFromEnv(unittest.TestCase):
    """The container reads overrides from EXPERIMENT_OVERRIDES_JSON env
    var rather than the TOML file (avoids container-filesystem fiddling).
    """

    def test_env_var_takes_precedence(self) -> None:
        import registry

        env_data = {"qwen3-8b": {"context_length": 32768}}
        with mock.patch.dict(
            os.environ,
            {"EXPERIMENT_OVERRIDES_JSON": json.dumps(env_data)},
            clear=False,
        ):
            with mock.patch.object(
                registry,
                "_load_overrides_from_path",
                return_value={"qwen3-8b": {"context_length": 99999}},
            ) as mock_path_loader:
                cfg = registry.get_config("qwen3-8b")
                # Env var won; the TOML loader should never have been called.
                self.assertEqual(cfg.context_length, 32768)
                mock_path_loader.assert_not_called()

    def test_env_var_empty_falls_back_to_file(self) -> None:
        import registry

        with mock.patch.dict(
            os.environ, {"EXPERIMENT_OVERRIDES_JSON": ""}, clear=False
        ):
            with mock.patch.object(
                registry,
                "_load_overrides_from_path",
                return_value={"qwen3-8b": {"context_length": 50000}},
            ) as mock_path_loader:
                cfg = registry.get_config("qwen3-8b")
                self.assertEqual(cfg.context_length, 50000)
                mock_path_loader.assert_called()

    def test_env_var_invalid_json_raises(self) -> None:
        import registry

        with mock.patch.dict(
            os.environ,
            {"EXPERIMENT_OVERRIDES_JSON": "{not json"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "valid JSON"):
                registry.get_config("qwen3-8b")

    def test_env_var_non_object_rejected(self) -> None:
        import registry

        with mock.patch.dict(
            os.environ,
            {"EXPERIMENT_OVERRIDES_JSON": "[1, 2, 3]"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "JSON object"):
                registry.get_config("qwen3-8b")


class TestUnknownModel(unittest.TestCase):
    def test_unknown_model_raises(self) -> None:
        import registry

        with self.assertRaises(KeyError):
            registry.get_config("not-a-model")


if __name__ == "__main__":
    unittest.main()
