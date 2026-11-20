"""Artifact transport concurrency checks; synthetic files are not training."""
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from scripts.rnad.shared_league_artifact_recovery import artifact_name
from scripts.rnad.shared_league_parallel_download import download_actors


class ParallelArtifactDownloadTest(unittest.TestCase):
    def setUp(self):
        self.generation = 2
        self.names = [artifact_name(self.generation, i) for i in range(20)]

    def test_exact_20_disjoint_bounded_downloads(self):
        called = []
        lock = threading.Lock()

        def fake_run(command, *, check, timeout, stdout):
            self.assertTrue(check)
            self.assertEqual(timeout, 300)
            self.assertEqual(command[:3], ["gh", "run", "download"])
            self.assertEqual(command[4:6], ["--repo", "cueikusuta/p-RL1024"])
            name = command[7]
            actor_id = self.names.index(name)
            out = Path(command[9]) / f"actor-{actor_id}.json.gz"
            out.write_bytes(b"synthetic-not-authority")
            with lock:
                called.append(actor_id)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            report = download_actors(
                generation=self.generation, run_id=12345,
                resolved=self.names, output=directory,
                workers=4, run=fake_run, sleep=lambda _: None)
            self.assertEqual(report["actors_downloaded"], 20)
            self.assertEqual(report["download_workers"], 4)
            self.assertTrue(report["strict_trajectory_verification_pending"])
            self.assertEqual(set(called), set(range(20)))
            for i in range(20):
                self.assertTrue(
                    (Path(directory)/f"actor-{i}"/f"actor-{i}.json.gz").is_file())

    def test_g49_g61_transport_rejects_mixed_source(self):
        # New training uses the same strict 20/20 naming contract past G48.
        for generation in (49, 61):
            with self.subTest(generation=generation):
                names = [artifact_name(generation, i) for i in range(20)]
                seen = []

                def fake_run(command, *, check, timeout, stdout):
                    name = command[7]
                    i = names.index(name)
                    seen.append(i)
                    (Path(command[9]) / f"actor-{i}.json.gz").write_bytes(
                        b"synthetic-not-authority")
                    return subprocess.CompletedProcess(command, 0)

                with tempfile.TemporaryDirectory() as directory:
                    report = download_actors(
                        generation=generation, run_id=12345,
                        resolved=names, output=directory, workers=1,
                        run=fake_run, sleep=lambda _: None)
                    self.assertEqual(report["generation"], generation)
                    self.assertEqual(set(seen), set(range(20)))
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        download_actors(
                            generation=generation, run_id=12345,
                            resolved=names[:-1] +
                            [artifact_name(generation-1, 19)],
                            output=directory,
                            run=lambda *_args, **_kw: self.fail(
                                "Wrong-generation actor was downloaded"))

    def test_only_failed_actor_is_retried(self):
        counts = {}
        lock = threading.Lock()
        self.names[5] = artifact_name(self.generation, 5, recovered=True)

        def fake_run(command, *, check, timeout, stdout):
            name = command[7]
            i = next(i for i, n in enumerate(self.names) if n == name)
            with lock:
                counts[i] = counts.get(i, 0) + 1
                n = counts[i]
            if i == 5 and n == 1:
                raise subprocess.CalledProcessError(1, command)
            (Path(command[9])/f"actor-{i}.json.gz").write_bytes(b"ok")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as directory:
            report = download_actors(
                generation=self.generation, run_id=10,
                resolved=self.names, output=directory,
                workers=2, run=fake_run, sleep=lambda _: None)
            self.assertEqual(report["actors_downloaded"], 20)
            self.assertEqual(counts[5], 2)
            self.assertTrue(all(v == 1 for k, v in counts.items() if k != 5))

    def test_wrong_count_or_mixed_generation_fails_before_download(self):
        cases = [self.names[:-1], self.names[:-1]+[self.names[0]],
                 self.names[:-1]+["../../wrong"],
                 [artifact_name(self.generation+1, i)
                  for i in range(20)]]
        with tempfile.TemporaryDirectory() as directory:
            for names in cases:
                with self.subTest(names=names[-1]):
                    with self.assertRaises(ValueError):
                        download_actors(
                            generation=self.generation, run_id=10,
                            resolved=names, output=directory,
                            run=lambda *_args, **_kw: self.fail(
                                "Invalid artifact source was downloaded"))
            for workers in (0, 3, 5, -1):
                with self.assertRaisesRegex(ValueError, "workers"):
                    download_actors(
                        generation=self.generation, run_id=10,
                        resolved=self.names, output=directory,
                        workers=workers)


if __name__ == "__main__":
    unittest.main()
