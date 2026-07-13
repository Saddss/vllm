# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk (L3) tier IO backend for SimpleCPUOffloadConnector.

All disk IO is staged through the worker's pinned CPU tensors: a disk store
reads a CPU block row and ``pwrite``s it to a per-block file; a disk load
``preadv``s the file back into a CPU block row. Disk never touches the GPU (no
GDS). IO runs on a pool of threads draining one priority queue where loads
(staging, on the request critical path) preempt background write-back stores;
a block is one contiguous file (page-first layout), one ``pwrite``/``preadv``.

An event completes only when all its blocks succeed; if any block errors the
event is reported as FAILED (via ``poll_failed``) so the scheduler drops it
instead of caching a partially-read block.

This is a cache, not a durable store: reads only need to see bytes written
earlier in the same process, so we use plain buffered IO and skip ``fsync``.
We deliberately do NOT ``fadvise(DONTNEED)`` on the hot path -- benchmarking
showed that evicting dirty pages right after a write forces synchronous
write-back (and even on reads costs ~25%), while buffered write-back lets the
kernel flush lazily and lets a soon-re-staged block read straight from cache.
The page cache holding cold KV is clean-reclaimable, so it can never OOM.
"""

import hashlib
import itertools
import os
import queue
import threading
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def disk_config_fingerprint(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> str:
    """Digest of the fields that fix a KV block's on-disk bytes: model path,
    revision, dtype, quantization, KV cache dtype/block size, and every group's
    layer set + spec. sha256 keeps it stable across processes
    (PYTHONHASHSEED-independent). Derived from config alone, so the scheduler
    (to find persisted blocks at startup) and the worker (to root the disk
    tier) compute the same value: incompatible configs land in disjoint
    subtrees and can never read each other's block-hash files -- a disk block
    has no content checksum, so a stale file of matching size would otherwise
    be served as valid KV (silent corruption).
    """
    mc = vllm_config.model_config
    cc = vllm_config.cache_config
    parts = [
        str(mc.model),
        str(mc.revision),
        str(mc.dtype),
        str(mc.quantization),
        str(cc.cache_dtype),
        str(cc.block_size),
    ]
    for group in kv_cache_config.kv_cache_groups:
        parts.append(",".join(sorted(group.layer_names)))
        parts.append(repr(group.kv_cache_spec))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def disk_tier_root(base_path: str, fingerprint: str, rank: int) -> str:
    """Directory holding one rank's block files; see disk_config_fingerprint."""
    return f"{base_path}/{fingerprint}/rank{rank}"


class DiskTier:
    def __init__(
        self,
        root: str,
        cpu_kv_caches: dict[str, torch.Tensor],
        n_read_threads: int = 8,
        n_write_threads: int = 8,
    ):
        self.root = root
        os.makedirs(root, exist_ok=True)
        # Fixed segment order defines the on-disk byte layout of a block.
        self.names = list(cpu_kv_caches.keys())
        # Precompute a zero-copy uint8 memoryview for every (segment, block) row
        # ONCE. Building these torch views holds the GIL; doing it per IO would
        # serialize the worker pool. Workers then issue pure pwrite/preadv on the
        # cached views (syscall releases the GIL), so IO scales across threads.
        self._seg_mv: dict[str, list[memoryview]] = {}
        for n in self.names:
            t = cpu_kv_caches[n]
            self._seg_mv[n] = [
                memoryview(t[b].reshape(-1).view(torch.uint8).numpy())
                for b in range(t.shape[0])
            ]
        self.block_nbytes = sum(len(self._seg_mv[n][0]) for n in self.names)

        # One queue drained by ALL threads, so a burst of either kind uses
        # every thread instead of leaving half idle (priority set in _launch).
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._remaining: dict[tuple[bool, int], int] = {}
        self._failed_events: set[tuple[bool, int]] = set()
        self._completed_store: list[int] = []
        self._completed_load: list[int] = []
        self._failed_store: list[int] = []
        self._failed_load: list[int] = []
        self._made_dirs: set[str] = set()

        n_threads = max(1, n_read_threads) + max(1, n_write_threads)
        self._threads: list[threading.Thread] = []
        for _ in range(n_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info(
            "SimpleCPUOffload DiskTier: root=%s block=%.2f MB io_threads=%d",
            root,
            self.block_nbytes / 1e6,
            n_threads,
        )

    def _path(self, key: str) -> str:
        d = os.path.join(self.root, key[:3])
        if d not in self._made_dirs:
            os.makedirs(d, exist_ok=True)
            self._made_dirs.add(d)
        return os.path.join(d, f"{key}.bin")

    @staticmethod
    def _unlink_quiet(path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("DiskTier unlink failed path=%s", path)

    def _store_one(self, block_id: int, key: str) -> None:
        dest = self._path(key)
        if os.path.exists(dest):  # dedup: identical content -> identical file
            return
        tmp = dest + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            off = 0
            for name in self.names:
                mv = self._seg_mv[name][block_id]
                os.pwrite(fd, mv, off)
                off += len(mv)
            os.close(fd)
            fd = -1
            os.replace(tmp, dest)  # atomic publish; readers never see partials
        except BaseException:
            if fd != -1:
                os.close(fd)
            self._unlink_quiet(tmp)  # a partial/failed write must not linger
            raise

    def _load_one(self, block_id: int, key: str) -> None:
        fd = os.open(self._path(key), os.O_RDONLY)
        try:
            off = 0
            for name in self.names:
                mv = self._seg_mv[name][block_id]
                n = os.preadv(fd, [mv], off)
                assert n == len(mv), f"short read {n} != {len(mv)} for {key}"
                off += len(mv)
        finally:
            os.close(fd)

    def _worker(self) -> None:
        while True:
            _, _, kind, event_idx, block_id, key = self._q.get()
            if kind == "delete":
                # Fire-and-forget: no event accounting for unlinks.
                self._unlink_quiet(self._path(key))
                continue
            is_store = kind == "store"
            ok = True
            try:
                if is_store:
                    self._store_one(block_id, key)
                else:
                    self._load_one(block_id, key)
            except Exception as e:
                ok = False
                # One line/block; the scheduler logs the aggregated per-event
                # failure (with the traceback source implied by this warning).
                logger.warning(
                    "DiskTier IO failed key=%s store=%s: %r", key, is_store, e
                )
            with self._lock:
                ek = (is_store, event_idx)
                if not ok:
                    self._failed_events.add(ek)
                self._remaining[ek] -= 1
                if self._remaining[ek] == 0:
                    del self._remaining[ek]
                    failed = ek in self._failed_events
                    self._failed_events.discard(ek)
                    if failed:
                        tgt = self._failed_store if is_store else self._failed_load
                    else:
                        tgt = (
                            self._completed_store if is_store else self._completed_load
                        )
                    tgt.append(event_idx)

    def _launch(
        self, block_ids: list[int], keys: list[str], event_idx: int, is_store: bool
    ) -> None:
        assert len(block_ids) == len(keys)
        with self._lock:
            self._remaining[(is_store, event_idx)] = len(block_ids)
        prio = 1 if is_store else 0  # loads (staging) preempt background stores
        kind = "store" if is_store else "load"
        for bid, key in zip(block_ids, keys):
            self._q.put((prio, next(self._seq), kind, event_idx, bid, key))

    def launch_store(
        self, block_ids: list[int], keys: list[str], event_idx: int
    ) -> None:
        self._launch(block_ids, keys, event_idx, is_store=True)

    def launch_load(
        self, block_ids: list[int], keys: list[str], event_idx: int
    ) -> None:
        self._launch(block_ids, keys, event_idx, is_store=False)

    def poll_completed(self, is_store: bool) -> list[int]:
        with self._lock:
            done = self._completed_store if is_store else self._completed_load
            if not done:
                return []
            out = list(done)
            done.clear()
        return out

    def poll_failed(self, is_store: bool) -> list[int]:
        with self._lock:
            failed = self._failed_store if is_store else self._failed_load
            if not failed:
                return []
            out = list(failed)
            failed.clear()
        return out

    def delete(self, keys: list[str]) -> None:
        """Queue evicted block files for unlink (best effort, asynchronous).

        Unlinks run on the IO threads at store priority, FIFO with stores.
        Doing them inline would stall the engine step: under LRU churn every
        store evicts a key, and thousands of synchronous unlinks per second
        on a write-saturated filesystem back up get_finished (observed as a
        multi-second TTFT collapse on H100).

        A queued delete can race a concurrent re-store of the same key; the
        worst case is a missing file at staging time, which the load-failure
        path already converts to a recompute. The eviction policy in
        disk_coordinator guarantees no in-flight load reads these keys.
        """
        prio = 1  # FIFO with stores: a delete never overtakes an older store
        for key in keys:
            self._q.put((prio, next(self._seq), "delete", 0, 0, key))

    def has_key(self, key: str) -> bool:
        return os.path.exists(self._path(key))
