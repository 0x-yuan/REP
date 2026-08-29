"""Unit tests for master/multi_replica_runner.py — pure-python paths only.

Verifies (without any Modal calls):
  * _shard_jsonl correctly splits a JSONL file into N parts and archives
    the original.
  * _maybe_shard_inputs only shards files past the threshold.
  * _worker_loop pulls files from the queue and dispatches them to its
    own replica via a mocked process_one + Modal Cls.
  * File distribution across replicas is roughly balanced (work-stealing).
  * The inbox watcher picks up files added AFTER startup.
  * Drain mode exits when the inbox is empty + workers idle.
  * queue_runner progress formatters render correctly (bar, ETA, duration).

Run:
    cd inference-farm && uv run python tests/test_multi_replica_runner.py
"""
from __future__ import annotations

import collections
import queue as queue_mod
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "master"))
sys.path.insert(0, str(_ROOT / "slave"))


def _write_jsonl(path: Path, n: int, prefix: str = "row") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(f'{{"id": "{prefix}_{i}"}}\n')


class TestShardJsonl(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang_test_"))
        self.inbox = self.tmp / "inbox"
        self.processed = self.tmp / "processed"
        self.inbox.mkdir()
        # Patch the constants the helper imports.
        import multi_replica_runner as mrr

        self._patches = [
            mock.patch.object(mrr, "INBOX", self.inbox),
            mock.patch.object(mrr, "PROCESSED", self.processed),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_jsonl_even_split(self) -> None:
        """Even split (n_total divisible by n_parts)."""
        from multi_replica_runner import _shard_jsonl

        src = self.inbox / "qwen3-8b__exp_x.jsonl"
        _write_jsonl(src, n=12)

        shards = _shard_jsonl(src, n_parts=4, dst_dir=self.inbox)
        self.assertEqual(len(shards), 4)
        for i, sh in enumerate(shards):
            self.assertTrue(sh.exists(), sh)
            self.assertEqual(sh.name, f"qwen3-8b__exp_x__shard{i}of4.jsonl")
            with sh.open() as f:
                rows = f.readlines()
            self.assertEqual(len(rows), 3, sh.name)

        # Original should be archived (not in inbox anymore).
        self.assertFalse(src.exists())

    def test_shard_jsonl_uneven_split(self) -> None:
        """Uneven split — first `extra` shards get one more row."""
        from multi_replica_runner import _shard_jsonl

        src = self.inbox / "qwen3-8b__exp_y.jsonl"
        _write_jsonl(src, n=10)

        shards = _shard_jsonl(src, n_parts=3, dst_dir=self.inbox)
        self.assertEqual(len(shards), 3)
        sizes = []
        for sh in shards:
            with sh.open() as f:
                sizes.append(len(f.readlines()))
        self.assertEqual(sizes, [4, 3, 3])
        self.assertEqual(sum(sizes), 10)
        self.assertFalse(src.exists())

    def test_shard_jsonl_n_parts_one_returns_self(self) -> None:
        from multi_replica_runner import _shard_jsonl

        src = self.inbox / "qwen3-8b__exp_z.jsonl"
        _write_jsonl(src, n=5)

        shards = _shard_jsonl(src, n_parts=1, dst_dir=self.inbox)
        self.assertEqual(shards, [src])
        self.assertTrue(src.exists())

    def test_shard_jsonl_empty_file_returns_empty(self) -> None:
        from multi_replica_runner import _shard_jsonl

        src = self.inbox / "qwen3-8b__exp_empty.jsonl"
        _write_jsonl(src, n=0)

        shards = _shard_jsonl(src, n_parts=4, dst_dir=self.inbox)
        self.assertEqual(shards, [])

    def test_shard_jsonl_rejects_no_model_prefix(self) -> None:
        from multi_replica_runner import _shard_jsonl

        src = self.inbox / "no_double_underscore.jsonl"
        _write_jsonl(src, n=3)
        with self.assertRaises(ValueError):
            _shard_jsonl(src, n_parts=2, dst_dir=self.inbox)


class TestMaybeShardInputs(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang_test_"))
        self.inbox = self.tmp / "inbox"
        self.processed = self.tmp / "processed"
        self.inbox.mkdir()
        import multi_replica_runner as mrr

        self._patches = [
            mock.patch.object(mrr, "INBOX", self.inbox),
            mock.patch.object(mrr, "PROCESSED", self.processed),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_large_files_get_sharded(self) -> None:
        from multi_replica_runner import _maybe_shard_inputs

        big = self.inbox / "qwen3-8b__big.jsonl"
        small = self.inbox / "qwen3-8b__small.jsonl"
        _write_jsonl(big, n=200)
        _write_jsonl(small, n=20)

        out = _maybe_shard_inputs(
            [big, small], n_replicas=4, min_rows=100, tag="[test]"
        )
        # 4 shards from `big` + 1 unchanged `small` = 5.
        self.assertEqual(len(out), 5)
        names = sorted(p.name for p in out)
        self.assertIn("qwen3-8b__small.jsonl", names)
        self.assertEqual(
            sum(1 for n in names if "shard" in n and "big" in n),
            4,
        )

    def test_no_shard_when_replicas_le_one(self) -> None:
        from multi_replica_runner import _maybe_shard_inputs

        big = self.inbox / "qwen3-8b__big.jsonl"
        _write_jsonl(big, n=500)

        out = _maybe_shard_inputs([big], n_replicas=1, min_rows=10, tag="[test]")
        self.assertEqual(out, [big])
        self.assertTrue(big.exists())


class TestWorkerLoop(unittest.TestCase):
    """Verify the queue-driven worker actually distributes work across
    threads (replicas) without losing files or double-processing."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_distribution_balanced(self) -> None:
        """8 files across 4 replicas — every file processed exactly once,
        roughly 2 per replica when each takes the same time."""
        from multi_replica_runner import _worker_loop

        files = [self.tmp / f"qwen3-8b__f{i}.jsonl" for i in range(8)]
        for f in files:
            _write_jsonl(f, n=1)

        file_q: queue_mod.Queue = queue_mod.Queue()
        for f in files:
            file_q.put(f)

        seen: dict[str, list[str]] = collections.defaultdict(list)
        seen_lock = threading.Lock()

        def fake_process_one(slave, jsonl_path, tag, *_a, **_kw):
            # Simulate ~constant work per file so the distribution is roughly even.
            time.sleep(0.05)
            replica = tag.strip("[]").split("/", 1)[0]
            with seen_lock:
                seen[replica].append(jsonl_path.name)

        mock_slave = mock.MagicMock()
        mock_slave.ping.remote.return_value = {"_state": {"phase": "ready"}}

        with mock.patch("multi_replica_runner.process_one", side_effect=fake_process_one), \
             mock.patch(
                 "multi_replica_runner.modal.Cls.from_name",
                 return_value=mock.MagicMock(return_value=mock_slave),
             ):
            failures: list = []
            threads = []
            for rid in ("r0", "r1", "r2", "r3"):
                t = threading.Thread(
                    target=_worker_loop,
                    kwargs=dict(
                        replica_id=rid,
                        model="qwen3-8b",
                        file_q=file_q,
                        poll_interval=0.0,
                        max_retries=0,
                        stall_timeout_s=10,
                        failures=failures,
                    ),
                    daemon=True,
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join(timeout=10)

            self.assertEqual(failures, [])
            # All 8 files processed exactly once.
            all_processed = sorted(name for v in seen.values() for name in v)
            self.assertEqual(
                all_processed, sorted(f.name for f in files)
            )
            # No file double-processed (set == list ⇒ no dups).
            self.assertEqual(
                len(all_processed), len(set(all_processed))
            )
            # All 4 replicas got at least one file.
            self.assertEqual(set(seen.keys()), {"r0", "r1", "r2", "r3"})

    def test_failure_recorded(self) -> None:
        """A SystemExit from process_one (max_retries exhausted) is captured
        in `failures`; the worker continues with the next file."""
        from multi_replica_runner import _worker_loop

        files = [self.tmp / "qwen3-8b__a.jsonl", self.tmp / "qwen3-8b__b.jsonl"]
        for f in files:
            _write_jsonl(f, n=1)

        file_q: queue_mod.Queue = queue_mod.Queue()
        for f in files:
            file_q.put(f)

        def fake_process_one(slave, jsonl_path, tag, *_a, **_kw):
            if "a.jsonl" in jsonl_path.name:
                raise SystemExit(f"{tag} retries exhausted on {jsonl_path.name}")

        mock_slave = mock.MagicMock()
        mock_slave.ping.remote.return_value = {"_state": {"phase": "ready"}}

        with mock.patch("multi_replica_runner.process_one", side_effect=fake_process_one), \
             mock.patch(
                 "multi_replica_runner.modal.Cls.from_name",
                 return_value=mock.MagicMock(return_value=mock_slave),
             ):
            failures: list = []
            t = threading.Thread(
                target=_worker_loop,
                kwargs=dict(
                    replica_id="r0",
                    model="qwen3-8b",
                    file_q=file_q,
                    poll_interval=0.0,
                    max_retries=0,
                    stall_timeout_s=10,
                    failures=failures,
                ),
                daemon=True,
            )
            t.start()
            t.join(timeout=5)
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0][0], "r0")
            self.assertIn("a.jsonl", failures[0][1])


class TestInboxWatcher(unittest.TestCase):
    """End-to-end verification of the live inbox watcher: files added AFTER
    startup are picked up; files in flight are not double-queued."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang_watch_"))
        self.inbox = self.tmp / "inbox"
        self.processed = self.tmp / "processed"
        self.inbox.mkdir()
        import multi_replica_runner as mrr

        self._patches = [
            mock.patch.object(mrr, "INBOX", self.inbox),
            mock.patch.object(mrr, "PROCESSED", self.processed),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_watcher_picks_up_files_added_after_start(self) -> None:
        """Drop a file in inbox AFTER the watcher starts; it must enqueue."""
        from multi_replica_runner import _watcher_loop

        # Patch _list_inbox_files to read from our test inbox via the local
        # name in mrr. The function is imported into mrr's namespace from
        # queue_runner, so patch THERE.
        import multi_replica_runner as mrr

        def _list_in_test_inbox(model: str) -> list[Path]:
            return sorted(
                p
                for p in self.inbox.glob(f"{model}__*.jsonl")
                if not p.name.endswith(".meta.jsonl")
            )

        with mock.patch.object(mrr, "_list_inbox_files", _list_in_test_inbox):
            file_q: queue_mod.Queue = queue_mod.Queue()
            in_flight: set[str] = set()
            in_flight_lock = threading.Lock()
            stop_event = threading.Event()

            t = threading.Thread(
                target=_watcher_loop,
                kwargs=dict(
                    model="qwen3-8b",
                    file_q=file_q,
                    in_flight=in_flight,
                    in_flight_lock=in_flight_lock,
                    stop_event=stop_event,
                    scan_interval=0.05,
                    tag="[test]",
                    pre_shard_min_rows=0,
                    n_replicas=1,
                ),
                daemon=True,
            )
            t.start()
            time.sleep(0.15)  # wait one+ scan cycle on empty inbox

            # Now drop a file in.
            f1 = self.inbox / "qwen3-8b__delayed1.jsonl"
            _write_jsonl(f1, n=3)

            # Wait for it to be enqueued.
            deadline = time.time() + 2.0
            while time.time() < deadline and file_q.empty():
                time.sleep(0.05)
            self.assertFalse(file_q.empty(), "watcher missed delayed file")
            got = file_q.get_nowait()
            self.assertEqual(got.name, "qwen3-8b__delayed1.jsonl")

            # Drop a SECOND file — confirm it also gets enqueued.
            f2 = self.inbox / "qwen3-8b__delayed2.jsonl"
            _write_jsonl(f2, n=3)
            deadline = time.time() + 2.0
            while time.time() < deadline and file_q.empty():
                time.sleep(0.05)
            self.assertFalse(file_q.empty(), "watcher missed second delayed file")

            # Stop the watcher.
            stop_event.set()
            t.join(timeout=2.0)
            self.assertFalse(t.is_alive())

    def test_watcher_skips_files_already_in_flight(self) -> None:
        """Files in `in_flight` must not be re-queued on rescan."""
        from multi_replica_runner import _watcher_loop
        import multi_replica_runner as mrr

        f1 = self.inbox / "qwen3-8b__a.jsonl"
        _write_jsonl(f1, n=3)

        def _list_in_test_inbox(model: str) -> list[Path]:
            return sorted(
                p
                for p in self.inbox.glob(f"{model}__*.jsonl")
                if not p.name.endswith(".meta.jsonl")
            )

        with mock.patch.object(mrr, "_list_inbox_files", _list_in_test_inbox):
            file_q: queue_mod.Queue = queue_mod.Queue()
            # Pre-mark f1 as in-flight, simulating "already enqueued and
            # being processed by a worker".
            in_flight: set[str] = {"qwen3-8b__a.jsonl"}
            in_flight_lock = threading.Lock()
            stop_event = threading.Event()

            t = threading.Thread(
                target=_watcher_loop,
                kwargs=dict(
                    model="qwen3-8b",
                    file_q=file_q,
                    in_flight=in_flight,
                    in_flight_lock=in_flight_lock,
                    stop_event=stop_event,
                    scan_interval=0.05,
                    tag="[test]",
                    pre_shard_min_rows=0,
                    n_replicas=1,
                ),
                daemon=True,
            )
            t.start()
            time.sleep(0.25)
            stop_event.set()
            t.join(timeout=2.0)
            self.assertTrue(file_q.empty(), "watcher re-queued an in-flight file")


class TestProgressFormatters(unittest.TestCase):
    """queue_runner formatters — small, but a regression here breaks every
    log line in the system."""

    def test_fmt_dur(self) -> None:
        from queue_runner import _fmt_dur

        self.assertEqual(_fmt_dur(0), "0s")
        self.assertEqual(_fmt_dur(45), "45s")
        self.assertEqual(_fmt_dur(60), "1m00s")
        self.assertEqual(_fmt_dur(125), "2m05s")
        self.assertEqual(_fmt_dur(3600), "1h00m")
        self.assertEqual(_fmt_dur(3725), "1h02m")
        self.assertEqual(_fmt_dur(None), "?")
        self.assertEqual(_fmt_dur(-1), "?")

    def test_eta_seconds(self) -> None:
        from queue_runner import _eta_seconds

        # half-done in 60s → 60s remaining
        self.assertAlmostEqual(_eta_seconds(50, 100, 60.0), 60.0, places=2)
        # nothing started yet → unknown
        self.assertIsNone(_eta_seconds(0, 100, 30.0))
        # already done → no ETA
        self.assertIsNone(_eta_seconds(100, 100, 30.0))
        # zero elapsed → unknown
        self.assertIsNone(_eta_seconds(50, 100, 0.0))

    def test_progress_bar(self) -> None:
        from queue_runner import _progress_bar

        self.assertEqual(_progress_bar(0, 10, width=10), "[" + "-" * 10 + "]")
        self.assertEqual(_progress_bar(10, 10, width=10), "[" + "#" * 10 + "]")
        self.assertEqual(_progress_bar(5, 10, width=10), "[" + "#" * 5 + "-" * 5 + "]")
        # Past total (e.g. n_resumed > original count): clamps.
        self.assertEqual(_progress_bar(20, 10, width=10), "[" + "#" * 10 + "]")
        # Unknown total.
        self.assertEqual(_progress_bar(5, 0, width=4), "[" + "?" * 4 + "]")

    def test_format_status_line_includes_chunk_when_active(self) -> None:
        from queue_runner import _format_status_line

        # Simulate a record mid-chunk.
        rec = {
            "n_done": 32,
            "n_total": 100,
            "tps_decode": 42.0,
            "current_row_id": "row32",
            "status": "generating",
            "_state": {"phase": "generating"},
            "chunk_index": 1,
            "chunk_n_chunks": 4,
            "chunk_active_s": 18.0,
            "heartbeat_at": time.time(),
        }
        line = _format_status_line(
            tag="[r0/qwen3-8b]",
            rec=rec,
            spawned_at=time.time() - 60.0,
            file_name="qwen3-8b__exp.jsonl",
        )
        self.assertIn("[r0/qwen3-8b]", line)
        self.assertIn("file=qwen3-8b__exp.jsonl", line)
        self.assertIn("32/100", line)
        self.assertIn("(32.0%)", line)
        self.assertIn("chunk=2/4", line)
        self.assertIn("@18s", line)
        self.assertIn("tps=42.0", line)


class TestQueueCli(unittest.TestCase):
    """CLI surface for inbox queue management."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sglang_qcli_"))
        self.inbox = self.tmp / "inbox"
        self.inbox.mkdir()
        import queue_cli

        self._patches = [
            mock.patch.object(queue_cli, "INBOX", self.inbox),
            mock.patch.object(queue_cli, "PROCESSED", self.tmp / "processed"),
            mock.patch.object(queue_cli, "RESULT", self.tmp / "result"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_then_list(self) -> None:
        from queue_cli import cmd_add, cmd_list

        src = self.tmp / "src.jsonl"
        _write_jsonl(src, n=4)

        ns = argparse_ns(
            model="qwen3-8b",
            file=str(src),
            rename_as="myrun",
            move=False,
            force=False,
        )
        rc = cmd_add(ns)
        self.assertEqual(rc, 0)
        self.assertTrue((self.inbox / "qwen3-8b__myrun.jsonl").exists())
        # Source still exists (copy mode).
        self.assertTrue(src.exists())

        # list returns 0 and finds the file.
        rc = cmd_list(argparse_ns(model="qwen3-8b"))
        self.assertEqual(rc, 0)

    def test_add_rejects_bad_filename(self) -> None:
        from queue_cli import cmd_add

        src = self.tmp / "weird_name.jsonl"
        _write_jsonl(src, n=1)
        ns = argparse_ns(
            model="qwen3-8b",
            file=str(src),
            rename_as=None,
            move=False,
            force=False,
        )
        with self.assertRaises(SystemExit):
            cmd_add(ns)

    def test_remove_with_dry_run_then_yes(self) -> None:
        from queue_cli import cmd_remove

        f = self.inbox / "qwen3-8b__run_01.jsonl"
        _write_jsonl(f, n=1)

        # Dry run: file remains.
        rc = cmd_remove(argparse_ns(
            model="qwen3-8b",
            pattern="qwen3-8b__run_*.jsonl",
            yes=False,
        ))
        self.assertEqual(rc, 0)
        self.assertTrue(f.exists(), "dry-run must not delete")

        # With --yes: gone.
        rc = cmd_remove(argparse_ns(
            model="qwen3-8b",
            pattern="qwen3-8b__run_*.jsonl",
            yes=True,
        ))
        self.assertEqual(rc, 0)
        self.assertFalse(f.exists())

    def test_clear_removes_all(self) -> None:
        from queue_cli import cmd_clear

        for name in ("qwen3-8b__a.jsonl", "qwen3-8b__b.jsonl"):
            _write_jsonl(self.inbox / name, n=1)

        rc = cmd_clear(argparse_ns(model="qwen3-8b", yes=True))
        self.assertEqual(rc, 0)
        remaining = list(self.inbox.glob("qwen3-8b__*.jsonl"))
        self.assertEqual(remaining, [])


def argparse_ns(**kwargs):
    """Mini-argparse.Namespace shim for tests without re-invoking the parser."""
    import argparse as _ap

    return _ap.Namespace(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
