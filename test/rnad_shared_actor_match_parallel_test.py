"""Isolated Actor shard scheduling contracts; no AI strength assertion."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.rnad.shared_league_collect import _batch_chunks, _run_actor_jobs


class ActorMatchParallelContracts(unittest.TestCase):
    def setUp(self):
        self.jobs = [
            {"seed": 50000 + i // 2, "seat": i % 2,
             "out": f"accepted/seed-{50000 + i // 2}-seat-{i % 2}.json.gz"}
            for i in range(64)
        ]

    def test_chunks_are_disjoint_deterministic_and_bounded(self):
        for workers in (1, 2, 4):
            chunks = _batch_chunks(self.jobs, workers)
            self.assertEqual(len(chunks), workers)
            self.assertTrue(all(1 <= len(x) <= 64 for x in chunks))
            self.assertEqual(sum(map(len, chunks)), 64)
            for i, chunk in enumerate(chunks):
                self.assertEqual(chunk, self.jobs[i::workers])
            keys = [(j["seed"], j["seat"]) for batch in chunks for j in batch]
            self.assertEqual(len(keys), len(set(keys)))
        for workers in (0, -1, 3, 5):
            with self.assertRaisesRegex(ValueError, "must be 1, 2 or 4"):
                _batch_chunks(self.jobs, workers)

    def test_each_worker_receives_exact_jobs_with_separate_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invoked = []
            def fake_run(cmd, *, check, timeout):
                self.assertTrue(check)
                self.assertEqual(timeout, 1800)
                self.assertEqual(cmd[:2], ["node", "actor.mjs"])
                invoked.append(cmd)
                return subprocess.CompletedProcess(cmd, 0)
            with patch("scripts.rnad.shared_league_collect.subprocess.run",
                       side_effect=fake_run):
                _run_actor_jobs(["node", "actor.mjs"], root, self.jobs, 4)
            self.assertEqual(len(invoked), 4)
            collected = []
            for i in range(4):
                doc = json.loads((root / f"actor-batch-jobs-{i}.json").read_text())
                self.assertEqual(doc["schema"], "rnad-shared-actor-batch-jobs-v1")
                self.assertEqual(doc["jobs"], self.jobs[i::4])
                collected.extend(doc["jobs"])
            self.assertEqual(len(collected), len(self.jobs))
            self.assertEqual(len({(j["seed"],j["seat"]) for j in collected}), 64)

    def test_failed_subprocess_is_never_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def one_failed(cmd, *, check, timeout):
                calls.append(cmd)
                if cmd[-1].endswith("-0.json"):
                    raise subprocess.CalledProcessError(9, cmd)
                return subprocess.CompletedProcess(cmd, 0)
            with patch("scripts.rnad.shared_league_collect.subprocess.run",
                       side_effect=one_failed):
                with self.assertRaises(subprocess.CalledProcessError):
                    _run_actor_jobs(["node", "actor.mjs"], Path(tmp),
                                    self.jobs, 2)
            self.assertEqual(len(calls), 2)

    def test_default_single_worker_retains_original_batch_file(self):
        with tempfile.TemporaryDirectory() as tmp, (
                patch("scripts.rnad.shared_league_collect.subprocess.run")
                ) as mocked:
            _run_actor_jobs(["node", "actor.mjs"], Path(tmp), self.jobs, 1)
            mocked.assert_called_once()
            self.assertTrue((Path(tmp) / "actor-batch-jobs.json").is_file())


if __name__ == "__main__":
    unittest.main()
