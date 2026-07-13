# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scheduler-side DiskTierCoordinator.

The coordinator only touches the CPU BlockPool and plain config fields, so
these tests run against a real BlockPool with lightweight config fakes --
no GPU, no engine.
"""

import os
from types import SimpleNamespace

from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.simple_kv_offload.disk_backend import (
    disk_config_fingerprint,
    disk_tier_root,
)
from vllm.v1.simple_kv_offload.disk_coordinator import (
    MAX_DISK_DEFERS,
    DiskTierCoordinator,
)
from vllm.v1.simple_kv_offload.metadata import (
    INVALID_JOB_ID,
    SimpleCPUOffloadWorkerMetadata,
)

BLOCK_SIZE = 16
NUM_CPU_BLOCKS = 16
BLOCK_BYTES = 1024

init_none_hash(sha256)


def _configs(world_size: int = 1):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="m", revision=None, dtype="bfloat16", quantization=None
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=BLOCK_SIZE),
        parallel_config=SimpleNamespace(world_size=world_size),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["l0"], kv_cache_spec="s")]
    )
    return vllm_config, kv_cache_config


def _make(tmp_path, capacity_bytes: int = 0, world_size: int = 1):
    pool = BlockPool(
        num_gpu_blocks=NUM_CPU_BLOCKS,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
    )
    vllm_config, kv_cache_config = _configs(world_size)
    disk = DiskTierCoordinator(
        vllm_config,
        kv_cache_config,
        pool,
        fa_gidx=0,
        hash_block_size=BLOCK_SIZE,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        cpu_block_bytes=BLOCK_BYTES,
        disk_offload_path=str(tmp_path),
        disk_capacity_bytes=capacity_bytes,
    )
    return pool, disk


def _key(i: int) -> bytes:
    # 32-byte hash + 4-byte group id 0, mirroring make_block_hash_with_group_id.
    return i.to_bytes(32, "big") + (0).to_bytes(4, "big")


def _cache_block(pool: BlockPool, key: bytes):
    """Allocate a CPU block, stamp it, and cache it (as a store would)."""
    (blk,) = pool.get_new_blocks(1)
    blk._block_hash = key
    pool.cached_block_hash_to_block.insert(key, blk)
    pool.free_blocks([blk])
    return blk


def _worker_meta(**kw) -> SimpleCPUOffloadWorkerMetadata:
    return SimpleCPUOffloadWorkerMetadata(completed_store_events={}, **kw)


def test_maybe_create_scope_gate(tmp_path):
    pool = BlockPool(4, True, BLOCK_SIZE)
    vllm_config, kv_cache_config = _configs()
    common = dict(
        fa_gidx=0,
        scheduler_block_size=BLOCK_SIZE,
        fa_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        num_cpu_blocks=4,
        cpu_block_bytes=BLOCK_BYTES,
        disk_capacity_bytes=0,
    )
    assert (
        DiskTierCoordinator.maybe_create(
            vllm_config,
            kv_cache_config,
            pool,
            num_kv_cache_groups=1,
            disk_offload_path="",
            **common,
        )
        is None
    )
    assert (
        DiskTierCoordinator.maybe_create(
            vllm_config,
            kv_cache_config,
            pool,
            num_kv_cache_groups=2,  # hybrid model -> out of scope
            disk_offload_path=str(tmp_path),
            **common,
        )
        is None
    )
    assert (
        DiskTierCoordinator.maybe_create(
            vllm_config,
            kv_cache_config,
            pool,
            num_kv_cache_groups=1,
            disk_offload_path=str(tmp_path),
            **common,
        )
        is not None
    )


def test_write_back_drain_respects_pin_budget(tmp_path):
    # Pin budget is num_cpu_blocks // 8 = 2, so 3 queued blocks drain 2 + 1.
    pool, disk = _make(tmp_path)
    keys = [_key(i) for i in range(3)]
    blocks = [_cache_block(pool, k) for k in keys]

    disk.note_cached_blocks(blocks)
    specs = disk.emit_step()
    assert specs.store_event != INVALID_JOB_ID
    assert specs.store_keys == [k.hex() for k in keys[:2]]
    # Emitted blocks are pinned until the write confirms; the third block
    # stays UNPINNED in the backlog (budget exhausted).
    assert all(pool.blocks[b].ref_cnt == 1 for b in specs.store_cpu_blocks)
    assert pool.blocks[blocks[2].block_id].ref_cnt == 0
    assert disk.emit_step().store_event == INVALID_JOB_ID  # still no room

    disk.on_worker_meta(
        _worker_meta(completed_disk_store_events={specs.store_event: 1})
    )
    # Persisted: pins released, keys tracked on disk (visible to staging),
    # and the freed budget lets the backlog remainder drain.
    assert all(pool.blocks[b].ref_cnt == 0 for b in specs.store_cpu_blocks)
    specs2 = disk.emit_step()
    assert specs2.store_keys == [keys[2].hex()]
    disk.on_worker_meta(
        _worker_meta(completed_disk_store_events={specs2.store_event: 1})
    )
    # Re-noting an already-persisted key is a no-op.
    disk.note_cached_blocks([blocks[0]])
    assert disk.emit_step().store_event == INVALID_JOB_ID


def test_write_back_backlog_bounded_drop_oldest(tmp_path):
    """When KV production outruns the disk, the backlog must stay bounded
    and admit newest candidates by dropping the oldest (best-effort)."""
    pool, disk = _make(tmp_path)
    limit = disk._backlog_limit  # 4 * pin_budget = 8 for a 16-block pool
    keys = [_key(i) for i in range(limit + 3)]
    blocks = [_cache_block(pool, k) for k in keys]
    disk.note_cached_blocks(blocks)

    assert len(disk._backlog) == limit
    # The 3 oldest were dropped and are re-admittable (not stuck in _queued).
    for k in keys[:3]:
        assert k not in disk._queued
    # The newest survived.
    assert keys[-1] in disk._queued
    # Dropped keys can be re-queued later (e.g. re-cached after CPU eviction).
    disk.note_cached_blocks([blocks[0]])
    assert keys[0] in disk._queued


def test_write_back_revalidates_recycled_blocks(tmp_path):
    pool, disk = _make(tmp_path)
    key = _key(1)
    blk = _cache_block(pool, key)
    disk.note_cached_blocks([blk])
    # Recycle the block before the drain: hash no longer matches the key.
    blk._block_hash = _key(2)
    specs = disk.emit_step()
    assert specs.store_event == INVALID_JOB_ID  # dropped, nothing pinned
    assert pool.blocks[blk.block_id].ref_cnt == 0


def test_store_failure_releases_pin_and_queues_delete(tmp_path):
    pool, disk = _make(tmp_path)
    blk = _cache_block(pool, _key(1))
    disk.note_cached_blocks([blk])
    specs = disk.emit_step()
    disk.on_worker_meta(_worker_meta(failed_disk_store_events={specs.store_event: 1}))
    assert pool.blocks[blk.block_id].ref_cnt == 0
    # The (possibly partial) file is queued for deletion, key not on disk.
    out = disk.emit_step()
    assert out.delete_keys == [_key(1).hex()]
    assert out.store_event == INVALID_JOB_ID


def _persist_keys(pool, disk, keys):
    """Persist keys to disk and evict them from CPU so staging is required."""
    for k in keys:
        blk = _cache_block(pool, k)
        disk.note_cached_blocks([blk])
        ev = disk.emit_step().store_event
        disk.on_worker_meta(_worker_meta(completed_disk_store_events={ev: 1}))
    evicted = pool.get_new_blocks(pool.get_num_free_blocks())
    pool.free_blocks(evicted)


def _stage_one(pool, disk, key: bytes):
    """Persist `key`, then build a request whose 1st hash is key."""
    _persist_keys(pool, disk, [key])
    return SimpleNamespace(
        request_id="req0",
        block_hashes=[key[:-4]],  # coordinator re-appends the group id
        num_tokens=2 * BLOCK_SIZE,  # room for one full hash block + 1 token
    )


def test_staging_defers_then_serves_as_cpu_hit(tmp_path):
    pool, disk = _make(tmp_path)
    key = _key(7)
    request = _stage_one(pool, disk, key)

    assert disk.try_stage_and_defer(request, 0, 0) is True
    specs = disk.emit_step()
    assert specs.load_event != INVALID_JOB_ID
    assert specs.load_keys == [key.hex()]

    # Still in flight -> keep deferring, but do not double-launch.
    assert disk.try_stage_and_defer(request, 0, 0) is True
    assert disk.emit_step().load_event == INVALID_JOB_ID

    disk.on_worker_meta(_worker_meta(completed_disk_load_events={specs.load_event: 1}))
    # Staged: the key is now a normal CPU hit, so no more deferral.
    assert pool.cached_block_hash_to_block.get_one_block(key) is not None
    assert disk.try_stage_and_defer(request, 0, 0) is False


def test_staging_failure_drops_key_for_recompute(tmp_path):
    pool, disk = _make(tmp_path)
    key = _key(9)
    request = _stage_one(pool, disk, key)

    assert disk.try_stage_and_defer(request, 0, 0) is True
    specs = disk.emit_step()
    disk.on_worker_meta(_worker_meta(failed_disk_load_events={specs.load_event: 1}))

    # Key dropped (re-miss -> recompute), reserved blocks released, file
    # queued for deletion, and the request no longer defers.
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)
    assert disk.emit_step().delete_keys == [key.hex()]
    assert disk.try_stage_and_defer(request, 0, 0) is False


def test_staging_emission_is_chunked_and_progressive(tmp_path):
    """A staging batch larger than the per-step cap must split into multiple
    events so early chunks unblock without waiting for the whole batch."""
    pool, disk = _make(tmp_path)
    disk._stage_blocks_per_step = 2
    keys = [_key(i) for i in range(3)]
    _persist_keys(pool, disk, keys)
    request = SimpleNamespace(
        request_id="req0",
        block_hashes=[k[:-4] for k in keys],
        num_tokens=4 * BLOCK_SIZE,  # room for all three hash blocks
    )

    assert disk.try_stage_and_defer(request, 0, 0) is True
    first = disk.emit_step()
    assert first.load_keys == [k.hex() for k in keys[:2]]
    second = disk.emit_step()
    assert second.load_keys == [keys[2].hex()]

    # Completing only the first chunk caches its keys; the request still
    # defers on the third key, then unblocks when its chunk lands.
    disk.on_worker_meta(_worker_meta(completed_disk_load_events={first.load_event: 1}))
    assert pool.cached_block_hash_to_block.get_one_block(keys[0]) is not None
    assert disk.try_stage_and_defer(request, 0, 2) is True
    disk.on_worker_meta(_worker_meta(completed_disk_load_events={second.load_event: 1}))
    assert disk.try_stage_and_defer(request, 0, 2) is False


def test_staging_backpressure_falls_back_to_recompute(tmp_path):
    """Past the in-flight staging bound, a new request must recompute (no
    defer, no allocation) instead of joining a queue steps deep."""
    pool, disk = _make(tmp_path)
    disk._max_staging_blocks = 1
    keys = [_key(1), _key(2), _key(3)]
    _persist_keys(pool, disk, keys)

    r1 = SimpleNamespace(
        request_id="r1", block_hashes=[keys[0][:-4]], num_tokens=2 * BLOCK_SIZE
    )
    r2 = SimpleNamespace(
        request_id="r2",
        block_hashes=[keys[1][:-4], keys[2][:-4]],
        num_tokens=3 * BLOCK_SIZE,
    )
    assert disk.try_stage_and_defer(r1, 0, 0) is True  # fills the bound
    free_before = pool.get_num_free_blocks()
    assert disk.try_stage_and_defer(r2, 0, 0) is False  # recompute, no defer
    assert pool.get_num_free_blocks() == free_before  # nothing allocated


def test_defer_deadline_gives_up(tmp_path):
    pool, disk = _make(tmp_path)
    key = _key(3)
    request = _stage_one(pool, disk, key)
    for _ in range(MAX_DISK_DEFERS - 1):
        assert disk.try_stage_and_defer(request, 0, 0) is True
    # Deadline reached: stop deferring even though staging is in flight.
    assert disk.try_stage_and_defer(request, 0, 0) is False


def test_capacity_lru_eviction_skips_staging_keys(tmp_path):
    # Capacity of 2 blocks.
    pool, disk = _make(tmp_path, capacity_bytes=2 * BLOCK_BYTES)
    for i in range(3):
        blk = _cache_block(pool, _key(i))
        disk.note_cached_blocks([blk])
        ev = disk.emit_step().store_event
        disk.on_worker_meta(_worker_meta(completed_disk_store_events={ev: 1}))
    # Oldest key (0) was evicted to stay within capacity.
    out = disk.emit_step()
    assert out.delete_keys == [_key(0).hex()]


def test_seed_index_recovers_intersection_and_skips_junk(tmp_path):
    vllm_config, kv_cache_config = _configs(world_size=2)
    fp = disk_config_fingerprint(vllm_config, kv_cache_config)
    keys = [_key(1), _key(2)]
    for rank, present in ((0, keys), (1, keys[:1])):
        root = disk_tier_root(str(tmp_path), fp, rank)
        for k in present:
            h = k.hex()
            os.makedirs(f"{root}/{h[:3]}", exist_ok=True)
            with open(f"{root}/{h[:3]}/{h}.bin", "wb") as f:
                f.write(b"x")
    # Junk that the scan must skip.
    junk_dir = f"{disk_tier_root(str(tmp_path), fp, 0)}/zzz"
    os.makedirs(junk_dir, exist_ok=True)
    with open(f"{junk_dir}/nothex.bin", "wb") as f:
        f.write(b"x")
    with open(f"{junk_dir}/readme.txt", "wb") as f:
        f.write(b"x")

    pool, disk = _make(tmp_path, world_size=2)
    # Only key(1) exists on BOTH ranks; key(2) and junk are not seeded.
    assert list(disk._on_disk) == [keys[0]]


def test_reset_releases_all_pins(tmp_path):
    pool, disk = _make(tmp_path)
    # One pinned write-back in flight + one staging in flight.
    blk = _cache_block(pool, _key(1))
    disk.note_cached_blocks([blk])
    disk.emit_step()
    request = _stage_one(pool, disk, _key(2))
    assert disk.try_stage_and_defer(request, 0, 0) is True

    disk.reset()
    assert all(b.ref_cnt == 0 for b in pool.blocks if not b.is_null)
    # Late (stale) completions after reset are ignored.
    disk.on_worker_meta(_worker_meta(completed_disk_store_events={0: 1}))
    disk.on_worker_meta(_worker_meta(completed_disk_load_events={0: 1}))
    assert disk.emit_step().store_event == INVALID_JOB_ID
