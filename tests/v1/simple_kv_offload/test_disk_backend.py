# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SimpleCPUOffload disk (L3) tier round-trip."""

import time

import torch

from vllm.v1.simple_kv_offload.disk_backend import DiskTier


def _poll(tier: DiskTier, is_store: bool, event_idx: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if event_idx in tier.poll_completed(is_store):
            return True
        time.sleep(0.01)
    return False


def test_disk_tier_roundtrip_bit_identical(tmp_path):
    torch.manual_seed(0)
    n_blocks = 8
    # Two segments with different shapes/dtypes, mirroring the worker's layout.
    cpu = {
        "k": torch.randn(n_blocks, 4, 16, dtype=torch.bfloat16),
        "v": torch.randn(n_blocks, 4, 16, dtype=torch.bfloat16),
    }
    ref = {name: t.clone() for name, t in cpu.items()}

    tier = DiskTier(str(tmp_path), cpu, num_io_threads=2)
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
