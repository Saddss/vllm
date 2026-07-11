# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SimpleCPUOffload disk (L3) tier round-trip."""

import time
from types import SimpleNamespace

import torch

from vllm.v1.simple_kv_offload.disk_backend import DiskTier, disk_config_fingerprint


def _poll(tier: DiskTier, is_store: bool, event_idx: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if event_idx in tier.poll_completed(is_store):
            return True
        time.sleep(0.01)
    return False


def _poll_failed(tier: DiskTier, is_store: bool, event_idx: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if event_idx in tier.poll_failed(is_store):
            return True
        time.sleep(0.01)
    return False


def _fake_configs(dtype="bfloat16", layers=("a", "b"), spec="spec0"):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="m", revision=None, dtype=dtype, quantization=None
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=list(layers), kv_cache_spec=spec)]
    )
    return vllm_config, kv_cache_config


def test_disk_config_fingerprint_isolates_incompatible_configs():
    base = disk_config_fingerprint(*_fake_configs())
    # Deterministic across calls (and PYTHONHASHSEED-independent by sha256).
    assert base == disk_config_fingerprint(*_fake_configs())
    # Any field that changes on-disk bytes must change the fingerprint.
    assert base != disk_config_fingerprint(*_fake_configs(dtype="float16"))
    assert base != disk_config_fingerprint(*_fake_configs(layers=("a",)))
    assert base != disk_config_fingerprint(*_fake_configs(spec="spec1"))
    # Layer order must not matter (sets of layers are unordered).
    assert base == disk_config_fingerprint(*_fake_configs(layers=("b", "a")))


def test_disk_tier_roundtrip_bit_identical(tmp_path):
    torch.manual_seed(0)
    n_blocks = 8
    # Two segments with different shapes/dtypes, mirroring the worker's layout.
    cpu = {
        "k": torch.randn(n_blocks, 4, 16, dtype=torch.bfloat16),
        "v": torch.randn(n_blocks, 4, 16, dtype=torch.bfloat16),
    }
    ref = {name: t.clone() for name, t in cpu.items()}

    tier = DiskTier(str(tmp_path), cpu, n_read_threads=2, n_write_threads=2)
    block_ids = [1, 3, 5]
    keys = [f"blk{b}" for b in block_ids]

    tier.launch_store(block_ids, keys, event_idx=0)
    assert _poll(tier, True, 0)
    assert all(tier.has_key(k) for k in keys)

    # Corrupt the source rows, then read them back from disk.
    for name in cpu:
        for b in block_ids:
            cpu[name][b].zero_()

    tier.launch_load(block_ids, keys, event_idx=0)
    assert _poll(tier, False, 0)

    for name in cpu:
        for b in block_ids:
            assert torch.equal(cpu[name][b], ref[name][b]), f"{name}[{b}] mismatch"
        # Untouched blocks must be unchanged.
        for b in range(n_blocks):
            if b not in block_ids:
                assert torch.equal(cpu[name][b], ref[name][b])


def test_disk_tier_missing_key():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tier = DiskTier(d, {"k": torch.zeros(2, 8, dtype=torch.float16)})
        assert not tier.has_key("nope")


def test_disk_tier_load_failure_reported(tmp_path):
    """A load of a non-existent key must surface as FAILED, not completed, so
    the scheduler drops the block instead of caching an unwritten row."""
    cpu = {"k": torch.zeros(4, 8, dtype=torch.float16)}
    tier = DiskTier(str(tmp_path), cpu, n_read_threads=2, n_write_threads=2)

    tier.launch_load([0, 1], ["missing0", "missing1"], event_idx=7)
    assert _poll_failed(tier, False, 7)
    # Must NOT also appear as completed.
    assert 7 not in tier.poll_completed(False)


def test_disk_tier_partial_failure_fails_whole_event(tmp_path):
    """If one block of a multi-block event fails, the whole event fails (we
    can't serve a half-loaded prefix)."""
    torch.manual_seed(1)
    cpu = {"k": torch.randn(4, 8, dtype=torch.float16)}
    tier = DiskTier(str(tmp_path), cpu, n_read_threads=2, n_write_threads=2)

    tier.launch_store([0], ["good"], event_idx=0)
    assert _poll(tier, True, 0)

    # One valid key + one missing key in the same event.
    tier.launch_load([0, 1], ["good", "missing"], event_idx=1)
    assert _poll_failed(tier, False, 1)
    assert 1 not in tier.poll_completed(False)


def test_disk_tier_failure_does_not_leak_into_next_event(tmp_path):
    """A failed event must not poison a later clean event's bookkeeping."""
    torch.manual_seed(2)
    cpu = {"k": torch.randn(4, 8, dtype=torch.float16)}
    tier = DiskTier(str(tmp_path), cpu, n_read_threads=2, n_write_threads=2)

    tier.launch_store([0], ["good"], event_idx=0)
    assert _poll(tier, True, 0)

    tier.launch_load([1], ["missing"], event_idx=1)
    assert _poll_failed(tier, False, 1)

    # A subsequent good load on the same tier completes cleanly and is not
    # reported as failed.
    tier.launch_load([0], ["good"], event_idx=2)
    assert _poll(tier, False, 2)
    assert 2 not in tier.poll_failed(False)
