# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side coordinator for the SimpleCPUOffload disk (L3) tier.

Owns every piece of scheduler-side disk state so the CPU-offload manager
stays disk-agnostic: the on-disk key index (LRU order, optional capacity
with eviction), disk->CPU staging, bounded write-back of newly cached CPU
blocks, in-flight event ledgers, and the startup scan that rebuilds the
index from files a prior process persisted.

Disk bytes live in the worker's CPU tensors; this class only tracks which
block hashes are persisted and decides what the worker should read, write,
or delete each step. Scope (enforced by ``maybe_create``): a single
full-attention group with uniform block size, where one block hash maps
1:1 to one CPU block.

Loads are two-phase: ``try_stage_and_defer`` kicks off disk->CPU staging
and the request is deferred (no GPU blocks held) until the staged blocks
become normal CPU hits served by the existing async CPU->GPU load path.
Disk runs shorter than the staging crossover length recompute instead
(see ``_stage_min_blocks``).
Stores are write-back: newly cached CPU blocks queue UNPINNED and a step
drains up to a pin budget, so write-back can never exhaust the CPU pool
and starve staging. The backlog is bounded with drop-oldest admission, so
when KV production outruns disk write bandwidth persistence degrades to
best-effort instead of queueing unboundedly.
"""

import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.simple_kv_offload.disk_backend import (
    disk_config_fingerprint,
    disk_tier_root,
)
from vllm.v1.simple_kv_offload.metadata import (
    INVALID_JOB_ID,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# (cpu_block_id, block-hash key) pairs describing one block transfer.
DiskSpec = tuple[int, bytes]

# Best-effort deadlines for a deferred request. STALLED counts only steps
# where the request's own staging made no progress (a stalled pipeline means
# recompute will beat waiting); TOTAL is the absolute cap so a request that
# keeps inching forward behind a deep queue still cannot livelock. Progress
# must reset the stall counter: under a bulk reuse storm the queue is many
# steps deep and a plain step counter expires mid-queue, shedding requests to
# GPU recompute that is far slower than the disk pipeline (8xH100: 16k-prefix
# storm TTFT p50 3.6s vs 0.5s for the old direct-read connector).
MAX_STALLED_DISK_DEFERS = 32
MAX_TOTAL_DISK_DEFERS = 128

# A staging event completes as a whole, so every request in it stays deferred
# until the LAST block is read: an H100 preemption storm batched ~67 requests
# into one 8636-block (20 GB) event whose stragglers gated them all. Cap the
# per-step batch so completion is progressive; the remainder stays queued
# (already pinned) and emits on later steps.
STAGE_BLOCKS_PER_STEP = 1024


@dataclass
class DiskStepSpecs:
    """Disk work emitted for one scheduler step (fields mirror metadata)."""

    load_event: int = INVALID_JOB_ID
    load_cpu_blocks: list[int] = field(default_factory=list)
    load_keys: list[str] = field(default_factory=list)
    store_event: int = INVALID_JOB_ID
    store_cpu_blocks: list[int] = field(default_factory=list)
    store_keys: list[str] = field(default_factory=list)
    delete_keys: list[str] = field(default_factory=list)


class _EventLedger:
    """In-flight events of one kind (load or store): a monotonic event
    counter, event -> specs, and per-event worker completion counts so an
    event is acted on only after all ``expected`` workers reported it."""

    def __init__(self, expected: int):
        self.expected = expected
        self._counter = 0
        self.specs: dict[int, list[DiskSpec]] = {}
        self._pending_counts: dict[int, int] = {}

    def open(self, specs: list[DiskSpec]) -> int:
        event = self._counter
        self._counter += 1
        self.specs[event] = specs
        return event

    def all_reported(self, event_idx: int, count: int) -> bool:
        total = self._pending_counts.get(event_idx, 0) + count
        if total >= self.expected:
            self._pending_counts.pop(event_idx, None)
            return True
        self._pending_counts[event_idx] = total
        return False

    def close(self, event_idx: int) -> list[DiskSpec] | None:
        """Pop an event's specs; None if unknown (stale after reset)."""
        # A failure clears any partial completion count for the event.
        self._pending_counts.pop(event_idx, None)
        return self.specs.pop(event_idx, None)

    def reset(self) -> None:
        # The counter is NOT reset: it must stay ahead of worker-side
        # bookkeeping to avoid event index collisions across a cache reset.
        self.specs.clear()
        self._pending_counts.clear()


class DiskTierCoordinator:
    @classmethod
    def maybe_create(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        cpu_block_pool: "BlockPool",
        *,
        num_kv_cache_groups: int,
        fa_gidx: int,
        scheduler_block_size: int,
        fa_block_size: int,
        hash_block_size: int,
        num_cpu_blocks: int,
        cpu_block_bytes: int,
        disk_offload_path: str,
        disk_capacity_bytes: int,
        disk_stage_min_tokens: int,
    ) -> "DiskTierCoordinator | None":
        """Create a coordinator, or None (with a warning) when the model is
        outside the supported scope; see the module docstring."""
        if not disk_offload_path:
            return None
        if num_kv_cache_groups != 1 or not (
            scheduler_block_size == fa_block_size == hash_block_size
        ):
            logger.warning(
                "SimpleCPUOffload disk tier disabled: only a single "
                "full-attention group with uniform block size is supported."
            )
            return None
        logger.info("SimpleCPUOffload disk tier enabled at %s", disk_offload_path)
        return cls(
            vllm_config,
            kv_cache_config,
            cpu_block_pool,
            fa_gidx=fa_gidx,
            hash_block_size=hash_block_size,
            num_cpu_blocks=num_cpu_blocks,
            cpu_block_bytes=cpu_block_bytes,
            disk_offload_path=disk_offload_path,
            disk_capacity_bytes=disk_capacity_bytes,
            disk_stage_min_tokens=disk_stage_min_tokens,
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        cpu_block_pool: "BlockPool",
        *,
        fa_gidx: int,
        hash_block_size: int,
        num_cpu_blocks: int,
        cpu_block_bytes: int,
        disk_offload_path: str,
        disk_capacity_bytes: int,
        disk_stage_min_tokens: int = 0,
    ):
        self._vllm_config = vllm_config
        self._kv_cache_config = kv_cache_config
        self._pool = cpu_block_pool
        self._fa_gidx = fa_gidx
        self._hash_block_size = hash_block_size
        self._path = disk_offload_path

        # Persisted keys in LRU order (oldest first). Capacity 0 = unbounded.
        self._on_disk: OrderedDict[bytes, None] = OrderedDict()
        self._max_blocks = (
            disk_capacity_bytes // max(1, cpu_block_bytes)
            if disk_capacity_bytes > 0
            else 0
        )
        if self._max_blocks:
            logger.info(
                "SimpleCPUOffload disk tier capacity: %d blocks (%.2f GB)",
                self._max_blocks,
                disk_capacity_bytes / (1024**3),
            )
        # In-flight disk->CPU staging: key -> cpu_block_id (dedup + keeps ref).
        self._staging: dict[bytes, int] = {}
        # Write-back backlog (unpinned candidates); policy in module docstring.
        self._backlog: deque[DiskSpec] = deque()
        self._queued: set[bytes] = set()
        self._pinned = 0
        self._pin_budget = max(1, num_cpu_blocks // 8)
        # Admission bound: KV can be produced faster than the disk absorbs it
        # (an H100 fills 2048-token prompts at >1 GB/s of KV vs <1 GB/s NVMe
        # write), so an unbounded backlog only accumulates stale entries.
        # When full, drop the OLDEST candidate: persistence is best-effort and
        # newest blocks are the ones most likely to still be valid at drain.
        self._backlog_limit = 4 * self._pin_budget
        self._stage_blocks_per_step = STAGE_BLOCKS_PER_STEP
        # Staging backpressure: bounds how much of the CPU pool staging can
        # pin (1/3, leaving room for the write pin budget and normal traffic).
        # Sized to keep the disk pipeline full under a bulk reuse storm:
        # requests shed past this bound fall back to GPU recompute, which is
        # far slower than a warm disk read, so a tight bound (R4 used 4
        # chunks) turned a 16k-prefix storm into mass recompute.
        self._max_staging_blocks = max(1, num_cpu_blocks // 3)
        # Staging pays a fixed defer cost (one-plus scheduler round trips)
        # regardless of length, so below a crossover length recompute wins:
        # H100 + NVMe measured disk-hit vs recompute TTFT at +160%/+123%/~0%/
        # -41% for 2k/4k/8k/16k-token prefixes. Disk runs shorter than this
        # many blocks are not staged (the request just recomputes; the files
        # stay on disk). The crossover shifts with GPU speed vs disk
        # bandwidth, hence configurable (disk_stage_min_tokens; 0 = always
        # stage).
        self._stage_min_blocks = max(1, disk_stage_min_tokens // hash_block_size)
        # Specs emitted to the worker at the next emit_step().
        self._pending_load: list[DiskSpec] = []
        self._pending_store: list[DiskSpec] = []
        self._pending_delete: list[bytes] = []
        expected = vllm_config.parallel_config.world_size
        self._loads = _EventLedger(expected)
        self._stores = _EventLedger(expected)
        self._max_stalled_defers = MAX_STALLED_DISK_DEFERS
        self._max_total_defers = MAX_TOTAL_DISK_DEFERS
        # Per-request defer bookkeeping: consecutive no-progress steps, total
        # deferred steps, and last observed in-flight block count (progress
        # signal); see MAX_STALLED_DISK_DEFERS above.
        self._defer_stalls: dict[str, int] = {}
        self._defer_totals: dict[str, int] = {}
        self._defer_last_in_flight: dict[str, int] = {}
        # Stats (logged, not exported).
        self._evicts_total = 0
        self._loads_total = 0
        self._stores_total = 0
        self._failures_total = 0

        self._seed_index()

    # ------------------------------------------------------------------
    # Load path: staging + deferral
    # ------------------------------------------------------------------

    def try_stage_and_defer(
        self, request: "Request", num_computed_tokens: int, num_hash_hit: int
    ) -> bool:
        """Kick off disk->CPU staging for on-disk blocks after the CPU prefix
        and decide whether the request should be deferred this step.

        Returns True to defer (staging launched or still in flight, and the
        request has not exhausted its defer budgets). Once staged, the blocks
        are normal CPU hits and this returns False.
        """
        req_id = request.request_id
        in_flight = self._stage_extension(request, num_computed_tokens, num_hash_hit)
        if in_flight == 0:
            self._drop_defer_state(req_id)
            return False
        prev = self._defer_last_in_flight.get(req_id)
        self._defer_last_in_flight[req_id] = in_flight
        # Progress = strictly fewer in flight; see MAX_STALLED_DISK_DEFERS.
        stalls = (
            0
            if prev is None or in_flight < prev
            else self._defer_stalls.get(req_id, 0) + 1
        )
        total = self._defer_totals.get(req_id, 0) + 1
        if stalls >= self._max_stalled_defers or total >= self._max_total_defers:
            # Deadline hit: stop deferring. In-flight staging still completes
            # into the CPU cache; this request just recomputes the suffix.
            self._drop_defer_state(req_id)
            return False
        self._defer_stalls[req_id] = stalls
        self._defer_totals[req_id] = total
        return True

    def _drop_defer_state(self, req_id: str) -> None:
        self._defer_stalls.pop(req_id, None)
        self._defer_totals.pop(req_id, None)
        self._defer_last_in_flight.pop(req_id, None)

    def _stage_extension(
        self, request: "Request", num_computed_tokens: int, num_hash_hit: int
    ) -> int:
        """Returns how many blocks of this request's prefix are in flight
        (launched now or earlier); 0 = nothing to wait for."""
        num_skipped = num_computed_tokens // self._hash_block_size
        remaining = request.block_hashes[num_skipped:]
        max_hashes = (
            request.num_tokens - 1 - num_computed_tokens
        ) // self._hash_block_size

        # Walk contiguous hashes after the CPU prefix; collect on-disk ones
        # that are not yet being staged. Stop at the first hash that is
        # neither in CPU nor on disk (prefix cache is a contiguous prefix).
        to_stage: list[bytes] = []
        waiting = 0
        i = num_hash_hit
        while i < len(remaining) and i < max_hashes:
            key = bytes(make_block_hash_with_group_id(remaining[i], self._fa_gidx))
            if self._pool.cached_block_hash_to_block.get_one_block(key):
                break
            if key in self._staging:
                waiting += 1
            elif key in self._on_disk:
                to_stage.append(key)
            else:
                break
            i += 1

        if not to_stage:
            return waiting

        # Short runs recompute; see _stage_min_blocks above.
        if len(to_stage) < self._stage_min_blocks:
            return waiting

        # Backpressure: see _max_staging_blocks above.
        if len(self._staging) + len(to_stage) > self._max_staging_blocks:
            return waiting

        # Allocate CPU blocks to receive the disk reads. If the CPU pool is
        # full, fall back to normal recompute (do not defer forever).
        if self._pool.get_num_free_blocks() < len(to_stage):
            return waiting
        cpu_blocks = self._pool.get_new_blocks(len(to_stage))
        for blk, key in zip(cpu_blocks, to_stage):
            blk._block_hash = key  # type: ignore[assignment]
            self._staging[key] = blk.block_id
            self._pending_load.append((blk.block_id, key))
            # LRU touch only for runs actually staged, so declined short runs
            # cannot keep never-read keys MRU and pollute capacity eviction.
            self._on_disk.move_to_end(key)
        return waiting + len(to_stage)

    # ------------------------------------------------------------------
    # Store path: bounded write-back
    # ------------------------------------------------------------------

    def note_cached_blocks(self, cpu_blocks: "list[KVCacheBlock]") -> None:
        """Queue newly cached CPU blocks as UNPINNED write-back candidates."""
        for cpu_block in cpu_blocks:
            key = bytes(cpu_block.block_hash)  # type: ignore[arg-type]
            if key not in self._on_disk and key not in self._queued:
                if len(self._backlog) >= self._backlog_limit:
                    _, dropped = self._backlog.popleft()
                    self._queued.discard(dropped)
                self._queued.add(key)
                self._backlog.append((cpu_block.block_id, key))

    def _drain_backlog(self) -> None:
        """Pin + move backlogged write-backs to pending, up to the budget.

        Each candidate was queued unpinned, so it may have been evicted and
        its CPU block reused since. We re-validate ``block_hash == key``
        before pinning; a mismatch means the block was recycled, so we drop
        it (the content is gone, nothing to persist).
        """
        room = self._pin_budget - self._pinned
        while room > 0 and self._backlog:
            block_id, key = self._backlog.popleft()
            if key in self._on_disk:
                self._queued.discard(key)
                continue
            block = self._pool.blocks[block_id]
            if block.block_hash is None or bytes(block.block_hash) != key:
                self._queued.discard(key)  # evicted + reused
                continue
            self._pool.touch([block])  # pin until the pwrite confirms
            self._pending_store.append((block_id, key))
            self._pinned += 1
            room -= 1

    # ------------------------------------------------------------------
    # Per-step emission and worker feedback
    # ------------------------------------------------------------------

    def emit_step(self) -> DiskStepSpecs:
        """Drain pending work into the disk fields of this step's metadata."""
        out = DiskStepSpecs()
        self._drain_backlog()
        if self._pending_load:
            # Chunked FIFO; see STAGE_BLOCKS_PER_STEP above.
            specs = self._pending_load[: self._stage_blocks_per_step]
            del self._pending_load[: self._stage_blocks_per_step]
            out.load_event = self._loads.open(specs)
            out.load_cpu_blocks = [b for b, _ in specs]
            out.load_keys = [k.hex() for _, k in specs]
        if self._pending_store:
            specs = self._pending_store
            self._pending_store = []
            out.store_event = self._stores.open(specs)
            out.store_cpu_blocks = [b for b, _ in specs]
            out.store_keys = [k.hex() for _, k in specs]
        if self._pending_delete:
            out.delete_keys = [k.hex() for k in self._pending_delete]
            self._pending_delete.clear()
            logger.info(
                "SimpleCPUOffload disk: LRU-evicted %d blocks (total=%d, on_disk=%d)",
                len(out.delete_keys),
                self._evicts_total,
                len(self._on_disk),
            )
        return out

    def on_worker_meta(self, meta: SimpleCPUOffloadWorkerMetadata) -> None:
        """Apply per-worker disk completions/failures (world_size-aggregated).

        A single worker failing an event fails it everywhere, so failures are
        acted on at first report.
        """
        for event_idx, count in meta.completed_disk_load_events.items():
            if self._loads.all_reported(event_idx, count):
                self._on_load_done(event_idx)
        for event_idx, count in meta.completed_disk_store_events.items():
            if self._stores.all_reported(event_idx, count):
                self._on_store_done(event_idx)
        for event_idx in meta.failed_disk_load_events:
            self._on_load_failed(event_idx)
        for event_idx in meta.failed_disk_store_events:
            self._on_store_failed(event_idx)

    def _on_load_done(self, event_idx: int) -> None:
        """Staging done: insert staged CPU blocks into the cache map + unpin."""
        specs = self._loads.close(event_idx)
        if specs is None:
            return
        for cpu_bid, key in specs:
            self._staging.pop(key, None)
            self._mark_on_disk(key)
            self._pool.cached_block_hash_to_block.insert(
                key, self._pool.blocks[cpu_bid]
            )
        self._pool.free_blocks(self._pool.blocks[b] for b, _ in specs)
        self._loads_total += len(specs)
        logger.info(
            "SimpleCPUOffload disk: staged %d blocks disk->CPU (total=%d)",
            len(specs),
            self._loads_total,
        )

    def _on_store_done(self, event_idx: int) -> None:
        """Write-back done: mark hashes persisted + release the write pin."""
        specs = self._stores.close(event_idx)
        if specs is None:
            return
        for _, key in specs:
            self._mark_on_disk(key)
            self._queued.discard(key)
        self._pinned -= len(specs)
        self._pool.free_blocks(self._pool.blocks[b] for b, _ in specs)
        self._stores_total += len(specs)
        logger.info(
            "SimpleCPUOffload disk: wrote %d blocks CPU->disk (total=%d)",
            len(specs),
            self._stores_total,
        )

    def _on_load_failed(self, event_idx: int) -> None:
        """Staging read failed: drop the key (so it re-misses / recomputes)
        and release the reserved CPU blocks without caching a partial read."""
        specs = self._loads.close(event_idx)
        if specs is None:
            return
        for _, key in specs:
            self._staging.pop(key, None)
            self._on_disk.pop(key, None)
            self._pending_delete.append(key)
        self._pool.free_blocks(self._pool.blocks[b] for b, _ in specs)
        self._failures_total += len(specs)
        logger.warning(
            "SimpleCPUOffload disk: %d block(s) failed to stage disk->CPU; "
            "dropping keys for recompute (failures total=%d)",
            len(specs),
            self._failures_total,
        )

    def _on_store_failed(self, event_idx: int) -> None:
        """Write-back failed: release the pin and do NOT mark the key on disk
        (delete any partial file so a later store can retry cleanly)."""
        specs = self._stores.close(event_idx)
        if specs is None:
            return
        for _, key in specs:
            self._queued.discard(key)
            self._pending_delete.append(key)
        self._pinned -= len(specs)
        self._pool.free_blocks(self._pool.blocks[b] for b, _ in specs)
        self._failures_total += len(specs)
        logger.warning(
            "SimpleCPUOffload disk: %d block(s) failed to write CPU->disk; "
            "not persisted (failures total=%d)",
            len(specs),
            self._failures_total,
        )

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def _mark_on_disk(self, key: bytes) -> None:
        """Record a key as persisted (MRU) and evict LRU keys over capacity."""
        self._on_disk[key] = None
        self._on_disk.move_to_end(key)
        if not self._max_blocks:
            return
        # Evict oldest first, but never a key with an in-flight staging read
        # (it is still in _on_disk while being re-loaded into a CPU block).
        while len(self._on_disk) > self._max_blocks:
            victim = next((k for k in self._on_disk if k not in self._staging), None)
            if victim is None:
                break  # everything left is pinned by staging; try again later
            del self._on_disk[victim]
            self._pending_delete.append(victim)
            self._evicts_total += 1

    def _seed_index(self) -> None:
        """Scan the disk tier left by a prior run and rebuild ``_on_disk``.

        The worker roots the tier at ``<path>/<fingerprint>/rank<rank>``; we
        recompute the same fingerprint from config and read every rank's dir.
        A key counts as persisted only if every rank has its file (TP shards
        must all be present to stage), and is seeded oldest-first so LRU
        order and any over-capacity eviction (which unlinks the surplus
        files) match the recency the files were written with.
        """
        fingerprint = disk_config_fingerprint(self._vllm_config, self._kv_cache_config)
        world_size = self._vllm_config.parallel_config.world_size
        per_rank: list[dict[bytes, float]] = []
        for rank in range(world_size):
            root = disk_tier_root(self._path, fingerprint, rank)
            found: dict[bytes, float] = {}
            for dirpath, _, files in os.walk(root):
                for name in files:
                    if not name.endswith(".bin"):
                        continue
                    try:
                        key = bytes.fromhex(name[:-4])
                    except ValueError:
                        continue
                    try:
                        found[key] = os.stat(os.path.join(dirpath, name)).st_mtime
                    except OSError:
                        continue
            per_rank.append(found)

        if not all(per_rank):  # a rank with no files -> nothing stageable
            return
        common = set(per_rank[0]).intersection(*(set(f) for f in per_rank[1:]))
        if not common:
            return
        for key in sorted(common, key=lambda k: per_rank[0][k]):
            self._mark_on_disk(key)
        logger.info(
            "SimpleCPUOffload disk tier: recovered %d persisted blocks", len(common)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_request_finished(self, req_id: str) -> None:
        self._drop_defer_state(req_id)

    def reset(self) -> None:
        """Drop the index and release pins so the CPU cache can reset.

        Orphaned disk files are harmless; stale worker completions are
        ignored by the guarded ``_EventLedger.close`` pops.
        """
        pinned_ids = list(self._staging.values())
        for specs in self._stores.specs.values():
            pinned_ids.extend(b for b, _ in specs)
        pinned_ids.extend(b for b, _ in self._pending_store)
        if pinned_ids:
            self._pool.free_blocks(self._pool.blocks[b] for b in pinned_ids)
        self._staging.clear()
        self._pending_load.clear()
        self._pending_store.clear()
        self._pending_delete.clear()
        self._backlog.clear()
        self._queued.clear()
        self._pinned = 0
        self._loads.reset()
        self._stores.reset()
        self._defer_stalls.clear()
        self._defer_totals.clear()
        self._defer_last_in_flight.clear()
        self._on_disk.clear()
