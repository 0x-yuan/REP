"""Unit tests for master/launches.py — registry I/O, prefix matching,
zombie detection, and gc behaviour. No subprocess / Modal calls."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_MASTER = _HERE.parent / "master"
if str(_MASTER) not in sys.path:
    sys.path.insert(0, str(_MASTER))

import launches  # noqa: E402


class TestRegistryHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(self.id().replace(".", "_") + "_tmp")
        self._tmp.mkdir(exist_ok=True)
        self._patch = mock.patch.object(launches, "LAUNCHES_DIR", self._tmp)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        for p in self._tmp.glob("*"):
            p.unlink()
        self._tmp.rmdir()

    def _write(self, name: str, **fields) -> Path:
        rec = {
            "launch_id": name,
            "model": fields.get("model", "qwen3-1p7b"),
            "profile": fields.get("profile", "test"),
            "replicas": fields.get("replicas", 1),
            "watch": fields.get("watch", False),
            "log_path": fields.get("log_path", f"/tmp/sglang_pipeline_{name}.log"),
            "started_at": fields.get("started_at", "2026-05-09T18:00:00Z"),
            "pid": fields.get("pid", -1),
            "status": fields.get("status", "running"),
            "exit_code": fields.get("exit_code"),
            "ended_at": fields.get("ended_at"),
        }
        p = self._tmp / f"{name}.json"
        p.write_text(json.dumps(rec))
        return p

    def test_list_registry_returns_sorted_entries(self) -> None:
        self._write("zzzz1234")
        self._write("aaaa5678")
        entries = launches._list_registry()
        ids = [e["launch_id"] for e in entries]
        self.assertEqual(ids, ["aaaa5678", "zzzz1234"])

    def test_list_registry_skips_malformed_json(self) -> None:
        (self._tmp / "broken.json").write_text("not-json{")
        self._write("good1234")
        entries = launches._list_registry()
        self.assertEqual([e["launch_id"] for e in entries], ["good1234"])

    def test_resolve_launch_full_id(self) -> None:
        self._write("aaaaaaaa")
        rec = launches._resolve_launch("aaaaaaaa")
        self.assertEqual(rec["launch_id"], "aaaaaaaa")

    def test_resolve_launch_unique_prefix(self) -> None:
        self._write("aaaa1111")
        self._write("bbbb2222")
        rec = launches._resolve_launch("aa")
        self.assertEqual(rec["launch_id"], "aaaa1111")

    def test_resolve_launch_ambiguous_prefix_raises(self) -> None:
        self._write("aaaa1111")
        self._write("aaaa2222")
        with self.assertRaises(SystemExit) as cm:
            launches._resolve_launch("aaaa")
        self.assertIn("ambiguous", str(cm.exception))

    def test_resolve_launch_no_match_raises(self) -> None:
        self._write("aaaa1111")
        with self.assertRaises(SystemExit) as cm:
            launches._resolve_launch("zzzz")
        self.assertIn("no launch matches", str(cm.exception))

    def test_pid_alive_for_self(self) -> None:
        self.assertTrue(launches._pid_alive(os.getpid()))

    def test_pid_alive_for_dead_pid(self) -> None:
        # PID 1 typically owned by init; -1 / 0 are sentinel "dead" markers.
        self.assertFalse(launches._pid_alive(0))
        self.assertFalse(launches._pid_alive(None))
        self.assertFalse(launches._pid_alive(-1))

    def test_pid_alive_for_unlikely_pid(self) -> None:
        # 9_999_999 is well beyond any reasonable PID on macOS / Linux.
        self.assertFalse(launches._pid_alive(9_999_999))

    def test_runtime_state_running_with_alive_pid(self) -> None:
        rec = {"status": "running", "pid": os.getpid()}
        self.assertEqual(launches._runtime_state(rec), "RUNNING")

    def test_runtime_state_running_with_dead_pid_is_zombie(self) -> None:
        rec = {"status": "running", "pid": 9_999_999}
        self.assertEqual(launches._runtime_state(rec), "ZOMBIE")

    def test_runtime_state_exited_with_code(self) -> None:
        rec = {"status": "exited", "pid": 9_999_999, "exit_code": 7}
        self.assertEqual(launches._runtime_state(rec), "EXITED(7)")

    def test_runtime_state_exited_without_code(self) -> None:
        rec = {"status": "exited", "pid": 9_999_999, "exit_code": None}
        self.assertEqual(launches._runtime_state(rec), "EXITED")


class TestStatusDictName(unittest.TestCase):
    """`launches status` must compute Dict names via slave/naming.py so an
    EXP_ID-prefixed deployment is reachable. Hard-coding `sglang-slave-...`
    works only for legacy (no-EXP_ID) deployments."""

    def setUp(self) -> None:
        self._tmp = Path(self.id().replace(".", "_") + "_tmp")
        self._tmp.mkdir(exist_ok=True)
        self._patch = mock.patch.object(launches, "LAUNCHES_DIR", self._tmp)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        for p in self._tmp.glob("*"):
            p.unlink()
        self._tmp.rmdir()

    def _write(self, lid: str, **fields) -> None:
        rec = {
            "launch_id": lid,
            "exp_id": fields.get("exp_id"),
            "model": fields.get("model", "qwen3-1p7b"),
            "profile": "t",
            "replicas": fields.get("replicas", 1),
            "watch": False,
            "log_path": f"/tmp/{lid}.log",
            "started_at": "2026-05-09T18:00:00Z",
            "pid": fields.get("pid", os.getpid()),
            "status": "running",
            "exit_code": None,
            "ended_at": None,
        }
        (self._tmp / f"{lid}.json").write_text(json.dumps(rec))

    def _capture_dict_names(self, launch_id: str) -> list[str]:
        """Run cmd_status and capture every Dict name it tried to read."""
        captured: list[str] = []

        class _FakeDict(dict):
            def keys(self):
                return []

        def _fake_from_name(name):
            captured.append(name)
            return _FakeDict()

        fake_modal = mock.MagicMock()
        fake_modal.Dict.from_name = _fake_from_name
        with mock.patch.dict(sys.modules, {"modal": fake_modal}):
            ns = mock.Mock(launch_id=launch_id)
            launches.cmd_status(ns)
        return captured

    def test_legacy_single_replica_dict_name(self) -> None:
        self._write("legacysingle", exp_id=None, replicas=1, model="qwen3-1p7b")
        self.assertEqual(
            self._capture_dict_names("legacysingle"),
            ["sglang-slave-qwen3-1p7b-progress"],
        )

    def test_legacy_multi_replica_dict_names(self) -> None:
        self._write("legacymulti", exp_id=None, replicas=3, model="qwen3-8b")
        self.assertEqual(
            self._capture_dict_names("legacymulti"),
            [
                "sglang-slave-qwen3-8b-r0-progress",
                "sglang-slave-qwen3-8b-r1-progress",
                "sglang-slave-qwen3-8b-r2-progress",
            ],
        )

    def test_exp_id_single_replica_dict_name(self) -> None:
        self._write("expsingle1", exp_id="exp-01", replicas=1, model="qwen3-1p7b")
        self.assertEqual(
            self._capture_dict_names("expsingle1"),
            ["exp-01-sglang-slave-qwen3-1p7b-progress"],
        )

    def test_exp_id_multi_replica_dict_names(self) -> None:
        self._write("expmulti01", exp_id="exp-02", replicas=2, model="qwen3-14b")
        self.assertEqual(
            self._capture_dict_names("expmulti01"),
            [
                "exp-02-sglang-slave-qwen3-14b-r0-progress",
                "exp-02-sglang-slave-qwen3-14b-r1-progress",
            ],
        )


class TestGcCommand(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(self.id().replace(".", "_") + "_tmp")
        self._tmp.mkdir(exist_ok=True)
        self._patch = mock.patch.object(launches, "LAUNCHES_DIR", self._tmp)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        for p in self._tmp.glob("*"):
            p.unlink()
        self._tmp.rmdir()

    def _write(self, name: str, **fields) -> Path:
        rec = {
            "launch_id": name,
            "model": "qwen3-1p7b",
            "profile": "t",
            "replicas": 1,
            "watch": False,
            "log_path": f"/tmp/{name}.log",
            "started_at": "2026-05-09T18:00:00Z",
            "pid": fields.get("pid", -1),
            "status": fields.get("status", "running"),
            "exit_code": fields.get("exit_code"),
            "ended_at": fields.get("ended_at"),
        }
        p = self._tmp / f"{name}.json"
        p.write_text(json.dumps(rec))
        return p

    def test_gc_dry_run_does_not_delete(self) -> None:
        self._write("exited01", status="exited", exit_code=0, pid=9_999_999)
        ns = mock.Mock(yes=False)
        launches.cmd_gc(ns)
        self.assertTrue((self._tmp / "exited01.json").exists())

    def test_gc_yes_deletes_exited(self) -> None:
        self._write("exited01", status="exited", exit_code=0, pid=9_999_999)
        ns = mock.Mock(yes=True)
        launches.cmd_gc(ns)
        self.assertFalse((self._tmp / "exited01.json").exists())

    def test_gc_yes_deletes_zombie(self) -> None:
        self._write("zomb0001", status="running", pid=9_999_999)
        ns = mock.Mock(yes=True)
        launches.cmd_gc(ns)
        self.assertFalse((self._tmp / "zomb0001.json").exists())

    def test_gc_keeps_active_running(self) -> None:
        # status=running with our own PID = alive = should NOT be gc'd.
        self._write("alive001", status="running", pid=os.getpid())
        ns = mock.Mock(yes=True)
        launches.cmd_gc(ns)
        self.assertTrue((self._tmp / "alive001.json").exists())


if __name__ == "__main__":
    unittest.main()
