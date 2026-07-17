# SimpleCPUOffload Disk Tier

SimpleCPUOffload uses a pinned-memory block pool as an external prefix cache.
GPU KV cache blocks can be copied into this pool and loaded back asynchronously
on a later hit. CPU memory still limits the working set that can be retained.
The disk tier adds another capacity tier behind the CPU pool:

```text
GPU KV cache  <---- existing ---->  pinned CPU block pool  <---- new ---->  disk
                                                pwrite / preadv
```

The disk tier is not a separate connector and does not access the GPU directly.
It extends
[SimpleCPUOffloadConnector][vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector.SimpleCPUOffloadConnector]:

- Stores read only CPU blocks whose GPU-to-CPU copy has completed.
- Loads write only into CPU blocks allocated specifically for staging.
- After disk-to-CPU staging completes, a block becomes an ordinary CPU cache
  hit and the existing CPU-to-GPU path handles the second transfer.
- A request waiting for disk staging holds no GPU block allocation.

The implementation has two components:

| Component | Process | Responsibility |
| --- | --- | --- |
| [DiskTierCoordinator][vllm.v1.simple_kv_offload.disk_coordinator.DiskTierCoordinator] | Scheduler | Disk index, load staging, write-back, capacity LRU, restart recovery, and event state |
| [DiskTier][vllm.v1.simple_kv_offload.disk_backend.DiskTier] | Worker | I/O thread pool, file reads/writes, asynchronous deletion, and event completion/failure accounting |

## Implementation map

The following table maps the rest of this document to the implementation.
Paths are relative to this document and can be opened directly from GitHub.

| Area | Source locations |
| --- | --- |
| Connector configuration and hook delegation | [`simple_cpu_offload_connector.py`](../../vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py): `SimpleCPUOffloadConnector.__init__`, `get_num_new_matched_tokens`, `build_connector_meta`, `update_connector_output`, `reset_cache` |
| Integration with the CPU offload manager | [`manager.py`](../../vllm/v1/simple_kv_offload/manager.py): `SimpleCPUOffloadScheduler.get_num_new_matched_tokens`, `_process_store_completion`, `request_finished`, `reset` |
| Disk policy and scheduler-side state | [`disk_coordinator.py`](../../vllm/v1/simple_kv_offload/disk_coordinator.py): `DiskTierCoordinator` |
| File format, I/O queue, event completion | [`disk_backend.py`](../../vllm/v1/simple_kv_offload/disk_backend.py): `DiskTier`, `disk_config_fingerprint`, `disk_tier_root` |
| Scheduler-to-worker and worker-to-scheduler messages | [`metadata.py`](../../vllm/v1/simple_kv_offload/metadata.py): `SimpleCPUOffloadMetadata`, `SimpleCPUOffloadWorkerMetadata`; [`kv_connector/utils.py`](../../vllm/distributed/kv_transfer/kv_connector/utils.py): `KVOutputAggregator` |
| Worker CPU storage and transfer submission | [`worker.py`](../../vllm/v1/simple_kv_offload/worker.py): `SimpleCPUOffloadWorker.register_kv_caches`, `get_finished`, `build_connector_worker_meta` |
| Request scheduling and external-KV promotion | [`scheduler.py`](../../vllm/v1/core/sched/scheduler.py): `Scheduler.schedule`, `_update_from_kv_xfer_finished`, `_try_promote_blocked_waiting_request`, `_update_waiting_for_remote_kv` |

## Request lifecycle

The following example follows one prefix through its first computation,
eviction, and reuse. Assume request `R1` computes a 16k-token document and a
later request `R2` uses the same prefix.

### First access

```text
R1 enters the scheduler
  |
  +-- no GPU, CPU, or disk prefix hit
  +-- allocate GPU blocks and run prefill
  |
  +-- SimpleCPUOffload copies complete, confirmed GPU blocks to CPU
  |     +-- eager: discover blocks from per-request progress
  |     +-- lazy: discover cached blocks near the GPU free-queue head
  |
  +-- GPU-to-CPU store event completes
  |     +-- install the block in the CPU prefix cache
  |     +-- make the same CPU block a disk write-back candidate
  |
  +-- R1 may continue decoding or finish; write-back no longer depends on R1
  |
  +-- worker writes and atomically publishes its rank's file
        +-- after every worker reports success, _on_store_done records _on_disk
```

The request does not wait for disk write-back. A file is an asynchronous copy
of an already valid CPU cache block. A write failure means only that the block
was not persisted.

A block may now have three independent copies:

```text
GPU block: still used by the request or retained by the GPU prefix cache
CPU block: a refcount-zero CPU prefix-cache entry
disk file: a published persistent copy
```

Each tier evicts independently. GPU eviction does not affect CPU or disk, and
CPU eviction does not affect disk.

### CPU hit after GPU eviction

If `R2` arrives after GPU eviction but the CPU block is still present:

1. `get_num_new_matched_tokens()` reports the CPU hit.
2. The scheduler allocates GPU destination blocks and moves the request to
   `WAITING_FOR_REMOTE_KVS`.
3. The worker starts CPU-to-GPU DMA.
4. After all ranks complete, the request returns to the ordinary waiting queue
   and can execute.

The disk tier does not read anything in this case.

### Disk-only hit after GPU and CPU eviction

If the consecutive prefix remains only in `_on_disk`:

1. The CPU coordinator misses.
2. `DiskTierCoordinator` allocates CPU staging blocks for the on-disk run.
3. `get_num_new_matched_tokens()` returns `None`. `R2` remains waiting and no
   GPU block is allocated.
4. The worker reads block files into the staging blocks.
5. Completed blocks enter the CPU prefix cache.
6. The next match attempt observes a CPU hit.
7. Only then does the scheduler allocate GPU blocks and move the request to
   `WAITING_FOR_REMOTE_KVS`.
8. The existing CPU-to-GPU path completes, and the request can execute.

The scheduler states are therefore:

```text
WAITING (disk-to-CPU, no GPU blocks)
    -> WAITING_FOR_REMOTE_KVS (CPU-to-GPU, GPU blocks allocated)
    -> WAITING / RUNNING
```

There is no dedicated disk-wait request status. While
`request.num_computed_tokens == 0`, each scheduling attempt calls the connector.
`get_num_new_matched_tokens()` returns `(None, False)`, and
`Scheduler.schedule()` skips the request for that step while leaving it in
`WAITING`. Disk completion is reported through
`completed_disk_load_events` in worker metadata, not through
`finished_recving`.

After staging turns the run into a CPU hit, `(hit_length, True)` starts the
ordinary external-KV path. CPU-to-GPU completion is the phase that uses
`finished_recving`; promotion normally returns the request to `WAITING`, or to
`PREEMPTED` if it already has preemptions.

If the disk run is too short or the CPU pool cannot admit its staging blocks,
no new staging is created. If the request has no shared blocks already in
flight, it is never deferred and immediately follows the recompute path. If it
already has in-flight blocks, it continues waiting for those blocks. A request
also gives up and recomputes when its I/O stops making progress long enough to
reach a defer limit.

### Restart

GPU and CPU caches disappear with the process, while block files remain. A new
process scans file names to rebuild `_on_disk` without prefetching contents.
The first matching request then follows the disk-only lifecycle above.

## Design choices

### Two-stage loading instead of direct disk-to-CPU-to-GPU loading

Disk-to-CPU staging does not allocate GPU blocks. Slow disk I/O or a deep read
queue therefore cannot consume GPU KV capacity that the scheduler could give
to immediately runnable work. Requests that do not hit disk perform only
in-memory index lookups.

The cost is an additional scheduler round trip. A direct-load design allocates
GPU blocks first and chains disk-to-CPU-to-GPU transfers. It is faster for
low-concurrency warm hits and bulk warm reuse, but holds GPU blocks for the
duration of disk I/O and puts filesystem lookup on the request path.

GDS was not selected for the same reason: reading directly into GPU memory
occupies the destination allocation while waiting for the slowest tier, and it
does not remove device I/O latency.

Streaming/direct loading is not implemented. Supporting it would require a
separate worker state machine and load-failure recovery.

### Write after CPU-cache insertion instead of at CPU eviction

The current policy creates disk copies before CPU eviction. CPU LRU eviction
therefore remains a memory-only operation, and restart recovery includes hot
blocks that have never been evicted from CPU.

The cost is write amplification: even zero-reuse requests generate write-back
candidates. The backlog is unpinned and bounded; when KV production exceeds
disk bandwidth, the oldest persistence opportunities are dropped rather than
delaying serving.

Writing only immediately before CPU eviction would reduce write volume, but
would either put disk latency on the allocator path or discard the block. It
would also make restart coverage incomplete.

### In-memory residency index instead of filesystem lookup

`_on_disk` makes a miss an in-memory dictionary lookup. A filesystem-backed
lookup must test file existence for each candidate block and adds syscall tail
latency to requests that never use disk.

The in-memory index requires a startup scan and is not synchronized between
live scheduler processes. Restart recovery and cross-instance sharing are
separate problems; only restart recovery is implemented.

### Buffered I/O instead of `O_DIRECT` or durable writes

Block files are recomputable cache data rather than authoritative storage.
Stores use buffered `pwrite` without `fsync`; loads use `preadv`. The kernel can
coalesce write-back and serve recently written blocks from the page cache.

An event can complete before dirty pages reach stable media. Machine failure
may therefore lose cache files, which is acceptable because a missing file
falls back to recomputation.

### Event-level all-or-nothing installation

If any block in an event fails, none of that event's CPU rows are installed as
valid cache entries. This keeps a prefix consecutive.

The tradeoff is a larger failure blast radius: successfully read blocks in the
same event are discarded. Load events are limited to 1024 blocks to bound both
completion and failure granularity.

The current multi-rank failure path acts on the first worker failure rather
than waiting for every worker to report a terminal state. It releases the
event's CPU references immediately, so a slower worker could still be
accessing those rows. This is a correctness limitation; references must remain
held until every worker has reported success or failure.

### Release staged blocks after event completion

After a disk load event completes, its CPU blocks become ordinary refcount-zero
cache entries instead of remaining pinned until a request consumes them. This
prevents cancelled or isolated requests from occupying the CPU pool.

When a bulk reuse working set exceeds CPU capacity, however, the CPU LRU can
evict staged blocks before waiting requests are rescheduled, causing repeated
reads. Pinning until consumption would improve this workload but requires
request ownership, shared-prefix accounting, cancellation handling, and a CPU
capacity reservation policy.

### Minimum read length, chunking, and backpressure

Three policies bound the cost of deferral:

- `disk_stage_min_tokens` avoids fixed scheduler overhead for short prefixes.
- At most 1024 blocks are emitted per step, preventing many requests from being
  gated by the final block of one very large event.
- Staging may occupy at most one third of the CPU pool.

These values are workload- and hardware-dependent. The implementation keeps
the threshold configurable. The chunk and backpressure limits remain internal
policies: larger chunks reduce event overhead but increase the time staged,
unpinned blocks are exposed to CPU LRU eviction before consumption.

## Disk data model

### Block files

Each KV block is stored in one file:

```text
<disk_offload_path>/<config_fingerprint>/rank<i>/<key[0:3]>/<key>.bin
```

The scheduler stores `BlockHashWithGroupId` keys as bytes in `_on_disk` and
`_staging`. `emit_step()` converts them to hexadecimal strings only at the
scheduler-to-worker metadata boundary; the worker uses that string as the file
name.

The file concatenates every storage segment belonging to the same CPU block.
During KV cache registration, the worker creates one
`[num_cpu_blocks, segment_bytes]` view per unique storage segment. Segment order
is the insertion order of `cpu_kv_caches` (including generated names such as
`<layer>.<segment>`), not a separately sorted canonical order. The disk tier
does not otherwise interpret K/V, layers, or attention backends.

`config_fingerprint` is the first 16 hexadecimal characters of a SHA-256 digest
over model, revision, model dtype, quantization, KV cache dtype, block size, and
every KV group's layer names and cache spec.

Block files do not contain a per-file checksum. The fingerprint directory is a
correctness boundary: different KV layouts can have the same bytes per block,
so a short-read check alone cannot detect a file with the correct size but
contents from another configuration. The current fingerprint does not
explicitly encode the worker-derived raw segment order or parallel topology.
The isolation guarantee is therefore limited to the configuration fields above;
changing low-level layout behavior without changing those fields requires a
fingerprint-version update.

### Scheduler-side state

`DiskTierCoordinator` uses the following in-memory state:

| State | Meaning |
| --- | --- |
| `_on_disk: OrderedDict[key, None]` | Keys known to have complete files, in LRU order |
| `_staging[key] = cpu_block_id` | Blocks queued for or currently undergoing disk-to-CPU staging |
| `_pending_load` | Reads with allocated CPU blocks not yet emitted in step metadata |
| `_backlog` / `_queued` | Write-back candidates not yet write-pinned; backlog length is four times the write pin budget and uses drop-oldest admission |
| `_pending_store` | Write-back blocks that have been revalidated and pinned |
| `_loads` / `_stores` | Event specs and multi-worker completion counts |
| `_pending_delete` | Keys awaiting asynchronous deletion |

## Block ownership and pins

Disk operations use the CPU BlockPool reference count to prevent physical rows
from being reused while an IO operation accesses them:

- Disk staging calls `get_new_blocks()`. Each allocated CPU block starts with
  one reference. `_staging[key]` records its identity while that reference is
  held.
- A successful disk load inserts the block into the CPU cache map and then
  calls `free_blocks()`. The reference becomes zero, but the hash and cache-map
  entry remain until CPU LRU reuse.
- Write-back backlog entries are deliberately unpinned. `_drain_backlog()`
  revalidates the current block hash, then `touch()`es the row before emitting a
  store event. Success and failure both release that write pin.
- A CPU cache hit is temporarily pinned between hit detection and GPU block
  allocation. After allocation, both the CPU source row and GPU destination row
  remain pinned until CPU-to-GPU DMA completes.
- If a request is cancelled after DMA starts, cleanup is deferred until the
  DMA completion releases both pins. Disk staging itself is block-owned rather
  than request-owned and continues after request cancellation.

At the instant `_process_store_completion()` calls `note_cached_blocks()`, a
new CPU block still holds its allocation reference. The same function releases
that reference immediately afterward. “Unpinned backlog” means no additional
write `touch()` is held while queued; it does not mean the reference has already
reached zero at the exact call site.

## Write-back lifecycle

Implementation:
[`SimpleCPUOffloadScheduler._process_store_completion`](../../vllm/v1/simple_kv_offload/manager.py),
[`DiskTierCoordinator.note_cached_blocks`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
[`DiskTierCoordinator._drain_backlog`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
and [`DiskTier._store_one`](../../vllm/v1/simple_kv_offload/disk_backend.py).

The disk tier begins after a GPU-to-CPU store event inserts blocks into the CPU
prefix cache:

```text
CPU cache insertion
    -> note_cached_blocks()
    -> unpinned bounded backlog
    -> revalidate and pin
    -> disk store event
    -> atomic file publish
    -> _on_disk + release pin
```

### Admission and revalidation

`note_cached_blocks()` skips keys already on disk or already queued. It appends
`(cpu_block_id, key)` without pinning. If the backlog is full, it removes the
oldest candidate.

Each step, `_drain_backlog()` can fill only
`write_pin_budget - currently_pinned` slots. The budget is one eighth of the
CPU pool, and incomplete store events from earlier steps continue to consume
it:

- A key already in `_on_disk` is skipped.
- A block whose hash is absent or no longer matches the candidate key was
  evicted/reused and is skipped.
- A valid block is `touch()`ed and moved to `_pending_store`.

### Worker write and completion

`emit_step()` first drains the backlog and then creates one disk store event
containing all `_pending_store` specs admitted under the remaining pin budget.
The CPU contents are stable because the preceding GPU-to-CPU event has already
completed.

`DiskTier._store_one()` writes `<key>.bin.tmp`, `pwrite`s every segment, and
publishes with `os.replace()`. If the destination already exists, the operation
is a no-op success and no temporary file is created. Any exception closes the
fd and removes the temporary file.

The current implementation does not check the byte count returned by
`os.pwrite()`. A short write can therefore be published without raising; later
segments can extend the file and make a simple total-size or short-read check
insufficient. The writer must loop until each segment is complete (or treat a
short write as event failure) before this path can claim complete short-write
protection.

After all workers succeed, `_on_store_done()` records the key as MRU (which can
queue capacity evictions) and then releases the CPU write pin. Physical deletes
are emitted on a later scheduler step. Failure leaves the key absent from
`_on_disk`, releases the pin, and schedules deletion of the published path; the
worker has already removed any `.tmp` file from the failed write.

## Lazy offload support

Implementation:
[`SimpleCPUOffloadScheduler.prepare_store_specs`](../../vllm/v1/simple_kv_offload/manager.py),
[`_prepare_eager_store_specs`](../../vllm/v1/simple_kv_offload/manager.py),
[`_prepare_lazy_store_specs`](../../vllm/v1/simple_kv_offload/manager.py), and
[`_process_store_completion`](../../vllm/v1/simple_kv_offload/manager.py).

`lazy_offload` controls **GPU-to-CPU candidate discovery**, not disk writing.

In eager mode, SimpleCPUOffload tracks complete, confirmed GPU blocks per
request and copies them into CPU early. In lazy mode, it scans the GPU
BlockPool free queue and copies cached blocks close to reuse.

The lazy target sums a per-group estimate: two blocks for Mamba, sliding-window
blocks plus one for sliding-window attention, and
`ceil(max_num_batched_tokens / effective_block_size)` for other groups. It then
doubles that sum as a watermark. Scanning resumes after the last free-queue
block visited; if that saved block is referenced again, the position is
invalidated and scanning restarts at the queue head.

Both modes converge in `_process_store_completion()`, which installs the CPU
cache entry and calls `disk.note_cached_blocks()`:

```text
eager GPU-to-CPU -> CPU cache -> disk backlog
lazy  GPU-to-CPU -> CPU cache -> disk backlog
```

The eager path maintains per-request store state, confirmed-token progress, and
an in-flight GPU-block deduplication set. The lazy path has no request-level
store state and returns no request IDs with its store specs. Both pin selected
GPU blocks during DMA and converge only after the GPU-to-CPU event completes.

The disk tier therefore supports both modes without a separate disk code path.
It does **not** implement lazy disk writes: a CPU block becomes a write-back
candidate immediately after CPU-cache insertion rather than at CPU eviction.

## Disk read lifecycle

Implementation:
[`SimpleCPUOffloadScheduler.get_num_new_matched_tokens`](../../vllm/v1/simple_kv_offload/manager.py),
[`DiskTierCoordinator.try_stage_and_defer`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
[`DiskTierCoordinator._stage_extension`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
[`DiskTierCoordinator.emit_step`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
and [`DiskTier._load_one`](../../vllm/v1/simple_kv_offload/disk_backend.py).

### Consecutive-run discovery

After the CPU coordinator reports its prefix hit,
`try_stage_and_defer(request, num_computed_tokens, num_hash_hit)` continues
through `request.block_hashes`, where
`num_hash_hit = hit_length // hash_block_size`:

- A key in `_staging` counts as already in flight and is not allocated again.
- A key in `_on_disk` is appended to `to_stage`.
- A key that has returned to CPU stops this disk extension; the next attempt
  will recompute the CPU hit.
- A key absent from CPU, staging, and disk stops the consecutive prefix.
- The final incomplete block is not cacheable.

The disk extension check runs before the manager returns the CPU hit. If any
part of the extension is staging, the method returns `(None, False)` for that
step even when the CPU coordinator already found an earlier prefix. The CPU hit
is exposed only after disk deferral finishes or gives up.

### Admission and allocation

`to_stage` must reach
`max(1, disk_stage_min_tokens // hash_block_size)` blocks, remain under the
global one-third CPU staging bound, and fit in currently free CPU blocks. The
conversion currently uses floor division, so a non-block-aligned token
threshold can admit a run shorter than the configured token count.

After admission, the coordinator allocates one CPU block per key, stores the
key in block metadata, records `_staging[key] = cpu_block_id`, and appends the
specs to `_pending_load`. The manager returns `None`, so the request remains
waiting with no GPU block allocation.

### Emission and completion

`emit_step()` emits up to 1024 blocks in FIFO order as one load event. Disk
loads have higher priority than stores and deletes in the worker's shared I/O
queue. Each worker `preadv`s every file segment into its target CPU row.

On all-worker success, `_on_load_done()` removes `_staging`, updates disk
recency, inserts each CPU block into the CPU prefix-cache map, and releases the
staging pin. On the next scheduler attempt, the request sees an ordinary CPU
hit and uses the existing CPU-to-GPU path.

### Progress-aware deferral

The coordinator recounts how many of the request's prefix blocks remain in
`_staging` on every match attempt. A decrease resets the stall counter. After
32 consecutive no-progress attempts the request recomputes. The total counter
permits at most 127 successful defer returns: the attempt that increments it to
128 returns `False`. Submitted I/O continues because staging is block-keyed and
can serve other requests.

## Request finish, preemption, and reset

Implementation:
[`SimpleCPUOffloadScheduler.request_finished`](../../vllm/v1/simple_kv_offload/manager.py),
[`SimpleCPUOffloadWorker.handle_preemptions`](../../vllm/v1/simple_kv_offload/worker.py),
[`SimpleCPUOffloadScheduler.reset`](../../vllm/v1/simple_kv_offload/manager.py),
and [`DiskTierCoordinator.reset`](../../vllm/v1/simple_kv_offload/disk_coordinator.py).

Finishing a request clears only its defer bookkeeping. Disk staging and
write-back are block-keyed, not request-owned, and continue to completion.

GPU preemption synchronizes existing GPU-to-CPU and CPU-to-GPU DMA before GPU
blocks can be reused. Disk I/O operates on separately allocated CPU blocks and
is neither cancelled nor quiesced by GPU preemption.

`DiskTierCoordinator.reset()` currently releases CPU blocks held by staging and
disk stores, clears queues, event ledgers, defer state, and the in-memory disk
index, but does not first quiesce worker disk IO. Ignoring late completion
metadata does not stop an IO thread from reading or writing a released CPU row.
Reset is therefore unsafe while disk load/store events are in flight. A safe
implementation must quiesce the disk worker or retain the references until
every event reaches a terminal state. Block files are left on disk.

The worker is not notified by `reset_cache()`, and
`SimpleCPUOffloadScheduler.reset()` waits only for abandoned GPU DMA state.
It can therefore return `True` and reset the CPU prefix cache while disk worker
threads still access the released rows.

## Multi-rank events and restart recovery

Implementation:
[`SimpleCPUOffloadWorkerMetadata.aggregate`](../../vllm/v1/simple_kv_offload/metadata.py),
[`_EventLedger.all_reported`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
[`DiskTierCoordinator.on_worker_meta`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
and [`DiskTierCoordinator._seed_index`](../../vllm/v1/simple_kv_offload/disk_coordinator.py).

Disk load and store successes take effect only after `world_size` workers have
reported completion. One worker failure currently fails the event immediately
because a missing rank makes the full KV block unusable; as noted above, the
associated references are released before the remaining workers necessarily
stop accessing them.

At startup, `_seed_index()` walks the complete sharded directory tree, skips
non-`.bin` files and invalid hexadecimal names, and collects one key-to-mtime
map per rank. If any rank has no valid files, recovery seeds no keys. Otherwise
it keeps the intersection of rank key sets and restores approximate LRU order
from rank 0 mtimes. Each recovered key passes through `_mark_on_disk()`, so a
recovered set above the current capacity immediately queues its oldest files
for deletion. Recovery rebuilds residency only and does not prefetch contents.

## Capacity and deletion

Implementation:
[`DiskTierCoordinator._mark_on_disk`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
[`DiskTierCoordinator.emit_step`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
and [`DiskTier.delete`](../../vllm/v1/simple_kv_offload/disk_backend.py).

`disk_capacity_bytes` is converted to a key-count limit using
`cpu_capacity_bytes // num_cpu_blocks` as the scheduler's estimate of block
bytes. The configuration is not divided by world size; the scheduler enforces
one global key count while every rank stores one file for each key. The
estimate can differ slightly from the worker's measured sum of segment bytes.

Staging admission moves a key to MRU before its read is emitted; completed
loads and stores also update recency. Capacity overflow removes the oldest key
not currently staging and queues its physical deletion for a later
`emit_step()`.

Deletion is fire-and-forget. The worker queues unlink operations at store
priority rather than running them on the engine path. A deletion failure can
leave an orphan file but cannot corrupt serving.

## Failure semantics

Implementation:
[`DiskTier._worker`](../../vllm/v1/simple_kv_offload/disk_backend.py),
[`DiskTierCoordinator._on_load_failed`](../../vllm/v1/simple_kv_offload/disk_coordinator.py),
and [`DiskTierCoordinator._on_store_failed`](../../vllm/v1/simple_kv_offload/disk_coordinator.py).

| Failure | Behavior |
| --- | --- |
| Missing file, short read, or `preadv` error | Remove staging and residency keys, release CPU blocks, recompute later |
| `pwrite` or publish failure | Do not record residency, release write pin, remove residual file |
| Short `pwrite` without exception | Currently not detected; a partial file can be published |
| ENOSPC | Write event fails; serving continues with lower persistence coverage |
| Delete failure | Log and possibly leave an orphan file |

Events are all-or-nothing: a partially read event never installs any CPU row as
valid KV.

The load-failure path also schedules every failed key for asynchronous file
deletion, so a missing or truncated file is not repeatedly offered as resident.

## Supported features

| Feature | Status | Notes |
| --- | --- | --- |
| Eager GPU-to-CPU offload + disk | Supported | CPU store completion feeds write-back |
| Lazy GPU-to-CPU offload + disk | Supported | Free-queue candidates feed the same write-back |
| Lazy disk write | Not supported | Disk writing does not wait for CPU eviction |
| Two-stage asynchronous load | Supported | Disk stage holds no GPU block |
| Short-prefix recomputation | Supported | `disk_stage_min_tokens`, default 8192 |
| Disk capacity and LRU | Supported | Per-rank capacity; zero is unlimited |
| Cross-process restart recovery | Supported | Startup scan without prefetch |
| Tensor parallelism | Supported | Per-rank files, all-rank event completion, and all-rank restart intersection |
| Context parallelism | Supported for a single group | Effective block size accounts for decode and prefill context parallelism |
| Pipeline parallelism | Partial | Full world size participates in events and recovery; no pipeline-specific coordination is added |
| KV dtype and block layout | Byte-transparent | Backend reads and writes the registered CPU block representation |
| I/O failures and ENOSPC | Partial | Exceptions invalidate and recompute; short writes are not yet detected |
| Preemption | Supported | GPU DMA flush; disk work is block-owned |
| Prefix-cache reset | Unsafe with in-flight disk I/O | Disk worker is not quiesced before CPU references are released |
| Configuration isolation | Partial | Covers high-level model/KV config, not explicit raw segment order or topology |
| GDS | Intentionally unsupported | All disk I/O stages through CPU |

## Unsupported or incomplete features

1. **Hybrid or multi-group KV cache.** The disk tier requires one full-attention
   group and equal scheduler, full-attention, and hash block sizes. SWA/Mamba
   hybrids fall back to CPU-only offload.
2. **Lazy disk writes.**
3. **Streaming/direct loading.** Bulk warm hits remain slower than direct load.
4. **Pin staged blocks until request consumption.** CPU LRU can evict staged
   blocks before allocation when the working set exceeds CPU capacity.
5. **Cross-instance or single-node-DP index sharing.**
6. **Per-file checksums.**
7. **Complete pending-transfer reporting.** `has_pending_transfers()` does not
   include GPU loads, pending/in-flight disk load or store state, or the worker
   I/O queue; it observes only scheduler-side GPU-to-CPU store events.
8. **Eager finish flush.** The final complete block can miss the next eager
   scan if the request finishes in the same step.
9. **Eager scan rollback after CPU eviction.** Per-request scan progress does
   not roll back when an older CPU block is evicted.
10. **Safe reset with in-flight disk IO.** Reset releases CPU rows before disk
    worker quiescence.
11. **All-rank terminal accounting on failure.** The first rank failure
    releases event references before remaining workers report.
12. **Short-write detection.** `pwrite` return values are not checked.

## Configuration

| Setting | Default | Description |
| --- | ---: | --- |
| `disk_offload_path` | `""` | Empty disables the disk tier |
| `disk_io_threads` | 8 | Split into `max(1, floor(n/2))` read-side and `max(1, n-floor(n/2))` write-side workers; all consume one priority queue |
| `disk_capacity_bytes` | 0 | Per-rank capacity; zero is unlimited |
| `disk_stage_min_tokens` | 8192 | Recompute shorter disk runs; zero still requires at least one complete hash block |
