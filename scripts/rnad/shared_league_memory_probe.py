"""Linux /proc sampled concurrent Learner staging memory, research only.

Unlike max per-worker ru_maxrss, RSS values are sampled at the SAME moment
across the staging parent and all descendant processes. Summed RSS can
double-count shared pages, so cgroup memory.current is recorded separately.
The probe is opt-in and never changes strict Match acceptance or gradients.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path


def _rss_kib(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as source:
            for row in source:
                if row.startswith("VmRSS:"):
                    return int(row.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    return 0


def _child_pids(pid):
    try:
        data = Path(f"/proc/{pid}/task/{pid}/children").read_text()
        return [int(x) for x in data.split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return []


def _descendants(pid):
    seen = {pid}
    pending = [pid]
    while pending and len(seen) < 128:
        parent = pending.pop()
        for child in _child_pids(parent):
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return seen


def _read_kib(path):
    try:
        return int(Path(path).read_text().strip()) // 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _cgroup_memory():
    # GitHub Actions hosted Linux normally uses unified cgroup v2.
    used = _read_kib("/sys/fs/cgroup/memory.current")
    if used is None:
        used = _read_kib("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    limit = _read_kib("/sys/fs/cgroup/memory.max")
    if limit is None:
        limit = _read_kib("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None and limit > (1 << 40):  # v1 unlimited sentinel
        limit = None
    return used, limit


class StageMemorySampler:
    def __init__(self, *, interval_seconds=.05):
        self.interval = interval_seconds
        self.parent_pid = os.getpid()
        self.stop_signal = threading.Event()
        self.thread = None
        self.samples = 0
        self.peak_concurrent_rss_kib = 0
        self.peak_children_rss_kib = 0
        self.peak_cgroup_usage_kib = 0
        self.cgroup_limit_kib = None
        self.minimum_cgroup_headroom_kib = None

    def _sample(self):
        pids = _descendants(self.parent_pid)
        children = sum(_rss_kib(pid) for pid in pids if pid != self.parent_pid)
        total = children + _rss_kib(self.parent_pid)
        self.peak_children_rss_kib = max(self.peak_children_rss_kib, children)
        self.peak_concurrent_rss_kib = max(self.peak_concurrent_rss_kib, total)
        used, limit = _cgroup_memory()
        if used is not None:
            self.peak_cgroup_usage_kib = max(self.peak_cgroup_usage_kib, used)
        if limit is not None:
            self.cgroup_limit_kib = limit
            if used is not None:
                free = max(0, limit - used)
                if (self.minimum_cgroup_headroom_kib is None or
                        free < self.minimum_cgroup_headroom_kib):
                    self.minimum_cgroup_headroom_kib = free
        self.samples += 1

    def _loop(self):
        self._sample()
        while not self.stop_signal.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.thread = threading.Thread(
            target=self._loop, name="rnad-staging-rss-sampler", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, ty, value, traceback):
        self.stop_signal.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Staging memory sampler did not stop cleanly")
        self._sample()

    def result(self):
        return {
            "sample_count": self.samples,
            "sample_interval_seconds": self.interval,
            "peak_parent_and_children_rss_kib": self.peak_concurrent_rss_kib,
            "peak_children_concurrent_rss_kib": self.peak_children_rss_kib,
            "peak_cgroup_memory_kib": self.peak_cgroup_usage_kib or None,
            "cgroup_limit_kib": self.cgroup_limit_kib,
            "minimum_cgroup_headroom_kib": self.minimum_cgroup_headroom_kib,
            "rss_shared_pages_may_be_double_counted": True,
        }
