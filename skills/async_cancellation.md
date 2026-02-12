---
name: Async Cancellation Safety
description: For asyncio tasks where interrupts or cancellation behavior matters.
---

## Concurrency/cancellation
- If interrupts or cancellation matter, define behavior before coding.
- Avoid propagating cancellation into running work unless explicitly required.
- Prefer lazy scheduling/backpressure over unbounded concurrency.
- For asyncio: on cancel or KeyboardInterrupt, cancel pending tasks and await all tasks with `return_exceptions=True` so `finally` blocks run.
- For asyncio: shield cleanup (`asyncio.shield(...)` or `try/finally`) so cleanup completes under cancellation.
- For asyncio: avoid cancelling tasks that have already started; stop scheduling new tasks and let in-flight tasks finish.
- For asyncio: if you cancel tasks, always `await` them (prefer `asyncio.shield(asyncio.gather(..., return_exceptions=True))`) before re-raising.
- For asyncio: if you eagerly create tasks gated by a semaphore, cancel tasks still waiting on the semaphore on shutdown; otherwise `await` may hang.
- For asyncio: consider a worker-pool/queue pattern to avoid creating all tasks up front.
- For asyncio: on KeyboardInterrupt with more tasks than `max_concurrent`, only started tasks should emit cleanup; queued tasks may never start.
- For asyncio: include a self-check that cancels with `n_tasks > max_concurrent` and asserts cleanup count equals started count, not total tasks.
- Add a small cancellation self-check if required.
