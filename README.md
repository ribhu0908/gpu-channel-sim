# gpu-channel-sim

A C11 + pthreads model of how a GPU consumes work from many clients at once: **GPFIFO channels, pushbuffer methods, semaphore fences, round-robin channel scheduling, and watchdog-driven channel recovery**. Built to reason about, and test, the firmware/driver logic that sits between "the host submitted work" and "the engine finished it", the part where hangs, bad commands and cross-channel deadlocks have to be survived.

> This is a host-side simulator for learning and testing scheduling and recovery logic. It does not talk to real GPU hardware. The concepts follow the publicly documented NVIDIA / nouveau channel model (GPFIFO, GP_GET/GP_PUT, semaphore acquire/release).

## Demo

```
$ make demo
[   0.2 ms] compute submitted (waits on sem0)
[   5.4 ms] buggy   submitted (will hang)
[  10.5 ms] upload  submitted (engine is stuck on buggy)
[ 126.4 ms] compute finished -> "frame #42 pixels"
[ 126.4 ms] buggy   channel status: FAULT_TIMEOUT
[ 126.6 ms] buggy   recovered and ran new work: OK

stats: entries=3 methods=6 ctx_switches=1 faults=1 watchdog_resets=1
```

What happened:
1. `compute` reached a `SEM_ACQUIRE` whose producer hadn't run yet, so the engine **parked it mid-segment and context-switched** instead of blocking.
2. `buggy` hung the engine. The **watchdog (100 ms) faulted that channel only**, dropped its pending work, and the engine moved on.
3. `upload` ran and released the semaphore; `compute` **resumed exactly where it stopped** and finished.
4. `buggy` was recovered and accepted new work.

## How it flows

```mermaid
sequenceDiagram
    autonumber
    participant C as compute channel
    participant B as buggy channel
    participant U as upload channel
    participant E as GPU engine
    participant W as watchdog
    C->>E: submit [ACQUIRE sem0, COPY, RELEASE sem1]
    E->>E: sem0 not ready: park compute, save cursor
    B->>E: submit [HANG]
    E->>E: stuck executing buggy
    U->>E: submit [COPY, RELEASE sem0] (queued)
    W->>W: entry older than 100 ms
    W->>E: fault buggy (FAULT_TIMEOUT)
    E->>E: drop buggy's pending work
    E->>U: run upload, RELEASE sem0
    E->>C: sem0 reached: resume compute at saved cursor
    C-->>C: RELEASE sem1, host fence wakes
```

## Model

```
 host threads                          engine thread                     watchdog thread
 ------------                          -------------                     ---------------
 pb_fill/pb_copy/...  builds methods
 gpu_submit()  --> GPFIFO[GP_PUT++] --> pick runnable channel (round robin)
               --> doorbell             run ONE entry (timeslice):         active entry older
                                          FILL / COPY  -> VRAM              than timeout?
 gpu_sem_wait() <-- SEM_RELEASE  <--      SEM_RELEASE  -> wake host fences  -> fault channel,
                                          SEM_ACQUIRE  unmet -> park,          abort entry
                                                       save cursor, switch
                                          bad method / bad address -> fault channel
```

Engine loop:

```mermaid
flowchart TD
    S(["engine thread"]) --> P{"runnable channel?<br/>(round robin)"}
    P -- none --> W["wait for doorbell"] --> P
    P -- found --> X["execute one GPFIFO entry<br/>from saved cursor"]
    X --> M{"next method"}
    M -- "FILL / COPY / RELEASE" --> M
    M -- "ACQUIRE not met" --> K["park channel<br/>ctx_switches++"] --> P
    M -- "bad opcode / bad address" --> F["fault this channel only<br/>drop its pending work"] --> P
    M -- "watchdog reset" --> F
    M -- "end of entry" --> D["GP_GET++<br/>wake idle waiters"] --> P
```

| Piece | Behaviour |
|---|---|
| **Pushbuffer** | 32-bit method stream: header `[31:24] opcode, [23:0] arg count` + args. Builders refuse to overflow a segment. |
| **GPFIFO** | Per-channel ring of 16 segments. `gpu_submit` returns `-EAGAIN` when full (back-pressure), `-EIO` once faulted. |
| **Semaphores** | 64 x 32-bit, compared wrap-safe (`(int32_t)(cur - want) >= 0`). Used for host fences and channel-to-channel dependencies. |
| **Scheduling** | One entry per timeslice, round-robin. An unmet `ACQUIRE` parks the channel with its cursor saved; it becomes runnable only when the semaphore is reached. |
| **Fault isolation** | Unknown opcode, wrong arg count, truncated method, out-of-range VRAM or semaphore id each fault *only* the offending channel. |
| **Watchdog** | Checks every `timeout/4`; if one entry has held the engine longer than `timeout`, it faults that channel and kicks the engine. Detection latency is therefore `timeout` to `1.25 x timeout`. |
| **Recovery** | `gpu_channel_recover()` clears the fault and empties the ring so the channel can be reused, refusing with `-EBUSY` while it is still on the engine. |

## Build & test

```sh
make test         # gcc, -Wall -Wextra -Werror -Wconversion
make test CC=clang
make asan         # AddressSanitizer + UBSan
make tsan         # ThreadSanitizer: engine, watchdog and host threads race-free
make portable     # macOS timing path (realtime-clock condvars), runnable on Linux
make demo
```

Runs on Linux and macOS with no GPU required. CI builds on both `ubuntu-latest` and `macos-latest`.

61 checks across 8 tests, all passing under gcc, clang, ASan/UBSan and TSan, including 50 repeated TSan runs and 8 concurrent instances under CPU contention:

| Test | Proves |
|---|---|
| `fill_copy_and_fence` | Methods execute correctly; fence wakes the host |
| `cross_channel_dependency` | Consumer submitted *before* producer is parked, not run early, then resumed |
| `ping_pong` | 80 semaphore hand-offs between two channels inside single segments, no deadlock |
| `invalid_method_is_isolated` | Bad opcode, truncated args and out-of-range VRAM fault only their own channel; recovery works |
| `watchdog_recovers_hang` | Hang detected within the window, queued work dropped, healthy channel unaffected |
| `gpfifo_backpressure` | Ring-full returns `-EAGAIN`; clean shutdown while channels are stalled |
| `pushbuffer_builder_bounds` | Builders never write past a segment |
| `multithreaded_stress` | 4 host threads x 400 submissions with retries: every entry retires, zero faults |

## Design notes

- **One device mutex.** All channel state and semaphores are guarded by it; VRAM is not. Ordering between host and engine comes from `gpu_submit` and `SEM_RELEASE`, both of which pass through the mutex, the same happens-before role a memory barrier plays before a real semaphore release.
- **A fence fires mid-entry.** `SEM_RELEASE` is visible before the GPFIFO entry that contains it retires. Code that needs "entry fully retired" (e.g. reading stats) uses `gpu_channel_idle_wait()`. The stress test originally failed 2 runs in 20 by mixing these up, which is why it is documented here.
- **Hang vs. slow.** The watchdog cannot tell a stuck shader from a very long legitimate copy; both are faulted once they exceed the timeout. Real drivers face the same trade-off, which is why the timeout is configurable.

## Limitations / next steps

- Single engine (no separate copy engine); no preemption inside a method.
- No timeout on a channel parked on `ACQUIRE`, so if the producer channel faults the consumer waits forever. Real hardware has acquire timeouts; this is the next feature.
- No virtual memory / page faults.

## License

MIT
