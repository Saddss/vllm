# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side disk (L3) tier for SimpleCPUOffloadConnector.

All disk IO is staged through the worker's pinned CPU tensors: a disk store
reads a CPU block row and ``pwrite``s it to a per-block file; a disk load
``pread``s the file back into a CPU block row. Disk never touches the GPU (no
GDS). IO runs on a background thread pool so it stays off the critical path;
completion is reported per event_idx and polled non-blocking by the worker.

File layout is page-first: all KV segments of one block are concatenated into a
single contiguous file, so a block is one ``pwrite``/``pread``.
"""

import os
import queue
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _row_bytes(row: torch.Tensor) -> memoryview:
    # Zero-copy uint8 view over a contiguous CPU block row. ``.numpy()`` does
    # not support bf16/fp8, so reinterpret as uint8 first.
    return memoryview(row.reshape(-1).view(torch.uint8).numpy())


class DiskTier:
    def __init__(
        self,
        root: str,
        cpu_kv_caches: dict[str, torch.Tensor],
        num_io_threads: int = 4,
    ):
        self.root = root
        os.makedirs(root, exist_ok=True)
        # Fixed segment order defines the on-disk byte layout of a block.
        self.names = list(cpu_kv_caches.keys())
        self.rows = cpu_kv_caches
        self.seg_nbytes = [
            cpu_kv_caches[n][0].reshape(-1).view(torch.uint8).numel()
            for n in self.names
        ]
        self.block_nbytes = sum(self.seg_nbytes)

        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._remaining: dict[tuple[bool, int], int] = {}
        self._completed_store: list[int] = []
        self._completed_load: list[int] = []
        self._threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(num_io_threads)
        ]
        for t in self._threads:
            t.start()
        logger.info(
            "SimpleCPUOffload DiskTier: root=%s block=%.2f MB io_threads=%d",
            root,
            self.block_nbytes / 1e6,
            num_io_threads,
        )

    def _path(self, key: str) -> str:
        d = os.path.join(self.root, key[:3])
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{key}.bin")

    def _store_one(self, block_id: int, key: str) -> None:
        tmp = self._path(key) + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            off = 0
            for name in self.names:
                mv = _row_bytes(self.rows[name][block_id])
                os.pwrite(fd, mv, off)
                off += len(mv)
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        os.replace(tmp, self._path(key))  # atomic publish

    def _load_one(self, block_id: int, key: str) -> None:
        fd = os.open(self._path(key), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            off = 0
            for name in self.names:
                mv = _row_bytes(self.rows[name][block_id])
                n = os.preadv(fd, [mv], off)
                assert n == len(mv), f"short read {n} != {len(mv)} for {key}"
                off += len(mv)
        finally:
            os.close(fd)

    def _worker(self) -> None:
        while True:
            is_store, event_idx, block_id, key = self._q.get()
            try:
                if is_store:
                    self._store_one(block_id, key)
                else:
                    self._load_one(block_id, key)
            except Exception:
                logger.exception("DiskTier IO failed key=%s store=%s", key, is_store)
            with self._lock:
                ek = (is_store, event_idx)
                self._remaining[ek] -= 1
                if self._remaining[ek] == 0:
                    del self._remaining[ek]
                    (
                        self._completed_store if is_store else self._completed_load
                    ).append(event_idx)

    def _launch(
        self, block_ids: list[int], keys: list[str], event_idx: int, is_store: bool
    ) -> None:
        assert len(block_ids) == len(keys)
        with self._lock:
            self._remaining[(is_store, event_idx)] = len(block_ids)
        for bid, key in zip(block_ids, keys):
            self._q.put((is_store, event_idx, bid, key))

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

    def has_key(self, key: str) -> bool:
        return os.path.exists(self._path(key))
