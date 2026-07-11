# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False

    # Disk (L3) transfers; keys are block-hash filenames (see disk_backend).
    disk_load_event: int = INVALID_JOB_ID
    disk_load_cpu_blocks: list[int] = field(default_factory=list)
    disk_load_keys: list[str] = field(default_factory=list)
    disk_store_event: int = INVALID_JOB_ID
    disk_store_cpu_blocks: list[int] = field(default_factory=list)
    disk_store_keys: list[str] = field(default_factory=list)
    # LRU-evicted keys whose disk files the worker should unlink.
    disk_delete_keys: list[str] = field(default_factory=list)


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]
    # Disk (L3) completions, same per-world_size aggregation as stores.
    completed_disk_store_events: dict[int, int] = field(default_factory=dict)
    completed_disk_load_events: dict[int, int] = field(default_factory=dict)
    # Disk IO failures; failure semantics in disk_backend.
    failed_disk_store_events: dict[int, int] = field(default_factory=dict)
    failed_disk_load_events: dict[int, int] = field(default_factory=dict)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)

        def _sum(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
            merged = dict(a)
            for k, v in b.items():
                merged[k] = merged.get(k, 0) + v
            return merged

        return SimpleCPUOffloadWorkerMetadata(
            completed_store_events=_sum(
                self.completed_store_events, other.completed_store_events
            ),
            completed_disk_store_events=_sum(
                self.completed_disk_store_events, other.completed_disk_store_events
            ),
            completed_disk_load_events=_sum(
                self.completed_disk_load_events, other.completed_disk_load_events
            ),
            failed_disk_store_events=_sum(
                self.failed_disk_store_events, other.failed_disk_store_events
            ),
            failed_disk_load_events=_sum(
                self.failed_disk_load_events, other.failed_disk_load_events
            ),
        )
