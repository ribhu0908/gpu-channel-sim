/*
 * gpu_sim.c - see include/gpu_sim.h
 *
 * Threads:
 *   engine    picks a runnable channel round-robin, executes ONE GPFIFO entry
 *             (a timeslice), then moves on. If an entry hits an unmet
 *             SEM_ACQUIRE it saves its position and yields the engine
 *             (a context switch) instead of blocking every other channel.
 *   watchdog  if the engine has been inside one entry for longer than the
 *             timeout, it faults that channel; the engine aborts the entry,
 *             drops the channel's pending work and carries on with others.
 *
 * GPFIFO ring (lock-free): each channel's ring is single-producer /
 * single-consumer, like the real GP_PUT / GP_GET pair.
 *
 *   host    writes the segment into fifo[GP_PUT], then publishes it with a
 *           RELEASE store of GP_PUT. The release guarantees the segment
 *           bytes are visible before the new GP_PUT is (on hardware this is
 *           the write barrier before the USERD GP_PUT / doorbell write).
 *   engine  ACQUIRE-loads GP_PUT, so everything written before the publish
 *           is visible, executes fifo[GP_GET], then RELEASE-stores GP_GET+1
 *           to hand the slot back.
 *   host    ACQUIRE-loads GP_GET before reusing a slot, so it never
 *           overwrites a segment the engine is still reading.
 *
 * Several host threads may submit to one channel; a per-channel submit_lock
 * serialises them into a single producer. The engine never takes it.
 *
 * Locking: the device mutex guards scheduler state (cursor, parking,
 * active channel), semaphores and stats. It is NOT needed to move work
 * through the ring. The doorbell still takes it briefly to wake the engine
 * without a lost wakeup (the engine sleeps on a condvar, not on a register).
 * VRAM is touched without any lock; ordering comes from GP_PUT publishing
 * and from SEM_RELEASE, which goes through the mutex.
 */
#define _POSIX_C_SOURCE 200809L
#include "gpu_sim.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FIFO_MASK (GPU_GPFIFO_ENTRIES - 1u)

/* macOS has no pthread_condattr_setclock(): its timed waits always use the
 * realtime clock, so build deadlines on that clock there. Elapsed-time
 * measurement (watchdog) stays on CLOCK_MONOTONIC everywhere. */
#if defined(__APPLE__) || defined(GPU_SIM_REALTIME_CONDS)
#define COND_CLOCK CLOCK_REALTIME
#define HAVE_CONDATTR_SETCLOCK 0
#else
#define COND_CLOCK CLOCK_MONOTONIC
#define HAVE_CONDATTR_SETCLOCK 1
#endif

typedef enum { EXEC_DONE, EXEC_BLOCKED, EXEC_FAULT } exec_result_t;

typedef struct {
    /* Read lock-free by gpu_submit; written under the device lock. */
    _Atomic bool     open;
    _Atomic int      status;        /* gpu_ch_status_t                      */

    /* Ring indices: free-running, slot = index & FIFO_MASK. */
    _Atomic uint32_t gp_put;        /* host writes (release), engine reads  */
    _Atomic uint32_t gp_get;        /* engine writes (release), host reads  */
    pthread_mutex_t  submit_lock;   /* serialises host producers only       */
    gpu_pb_t         fifo[GPU_GPFIFO_ENTRIES];

    /* Scheduler state below: device lock, or engine-only while active. */
    uint32_t        cursor;         /* resume point inside current entry    */
    bool            waiting;        /* parked on an unmet ACQUIRE           */
    uint32_t        wait_sem;
    uint32_t        wait_val;
    uint64_t        wait_since_ms;  /* when it parked (acquire timeout)     */
    bool            reset_req;      /* watchdog asked engine to abort       */
} channel_t;

struct gpu_dev {
    pthread_mutex_t lock;
    pthread_cond_t  work_cv;        /* doorbell: new work / sem / shutdown  */
    pthread_cond_t  sem_cv;         /* host waiters on semaphores           */
    pthread_cond_t  idle_cv;        /* host waiters on channel idle         */
    pthread_cond_t  wd_cv;          /* watchdog tick + hang abort           */
    pthread_t       engine_thr;
    pthread_t       wd_thr;
    bool            running;

    channel_t       ch[GPU_MAX_CHANNELS];
    uint32_t        sem[GPU_NUM_SEMAPHORES];
    unsigned        rr_next;

    channel_t      *active;         /* entry currently on the engine        */
    uint64_t        active_since_ms;
    uint32_t        watchdog_ms;
    uint32_t        acquire_timeout_ms;     /* 0 = disabled                 */

    uint8_t        *vram;
    size_t          vram_size;
    gpu_stats_t     stats;
};

/* ------------------------------------------------------------ helpers */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

static struct timespec deadline_after(uint32_t ms)
{
    struct timespec ts;
    clock_gettime(COND_CLOCK, &ts);
    ts.tv_sec  += ms / 1000u;
    ts.tv_nsec += (long)(ms % 1000u) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }
    return ts;
}

/* Wrap-safe "a >= b" for free-running 32-bit counters. */
static inline bool sem_reached(uint32_t cur, uint32_t want)
{
    return (int32_t)(cur - want) >= 0;
}

static inline bool range_ok(const gpu_dev_t *d, uint32_t off, uint32_t len)
{
    return (uint64_t)off + len <= d->vram_size;
}

static int init_cond(pthread_cond_t *cv)
{
    pthread_condattr_t a;
    pthread_condattr_init(&a);
#if HAVE_CONDATTR_SETCLOCK
    pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
#endif
    int rc = pthread_cond_init(cv, &a);
    pthread_condattr_destroy(&a);
    return rc;
}

/* ---------------------------------------------------------- pushbuffer */
void pb_reset(gpu_pb_t *pb) { pb->n = 0; }

static int pb_emit(gpu_pb_t *pb, uint32_t op, uint32_t nargs, const uint32_t *args)
{
    if (pb->n + 1u + nargs > GPU_PB_MAX_WORDS)
        return -1;
    pb->w[pb->n++] = (op << 24) | nargs;
    for (uint32_t i = 0; i < nargs; i++)
        pb->w[pb->n++] = args[i];
    return 0;
}

int pb_nop(gpu_pb_t *pb) { return pb_emit(pb, GPU_OP_NOP, 0, NULL); }
int pb_hang(gpu_pb_t *pb) { return pb_emit(pb, GPU_OP_HANG, 0, NULL); }

int pb_fill(gpu_pb_t *pb, uint32_t dst, uint8_t value, uint32_t len)
{
    uint32_t a[3] = { dst, value, len };
    return pb_emit(pb, GPU_OP_FILL, 3, a);
}

int pb_copy(gpu_pb_t *pb, uint32_t dst, uint32_t src, uint32_t len)
{
    uint32_t a[3] = { dst, src, len };
    return pb_emit(pb, GPU_OP_COPY, 3, a);
}

int pb_sem_release(gpu_pb_t *pb, uint32_t sem, uint32_t value)
{
    uint32_t a[2] = { sem, value };
    return pb_emit(pb, GPU_OP_SEM_RELEASE, 2, a);
}

int pb_sem_acquire(gpu_pb_t *pb, uint32_t sem, uint32_t value)
{
    uint32_t a[2] = { sem, value };
    return pb_emit(pb, GPU_OP_SEM_ACQUIRE, 2, a);
}

int pb_raw(gpu_pb_t *pb, uint32_t word)
{
    if (pb->n >= GPU_PB_MAX_WORDS)
        return -1;
    pb->w[pb->n++] = word;
    return 0;
}

const char *gpu_ch_status_str(gpu_ch_status_t s)
{
    switch (s) {
    case GPU_CH_OK:                   return "OK";
    case GPU_CH_FAULT_INVALID_METHOD: return "FAULT_INVALID_METHOD";
    case GPU_CH_FAULT_BAD_ADDRESS:    return "FAULT_BAD_ADDRESS";
    case GPU_CH_FAULT_TIMEOUT:        return "FAULT_TIMEOUT";
    case GPU_CH_FAULT_ACQUIRE_TIMEOUT: return "FAULT_ACQUIRE_TIMEOUT";
    }
    return "UNKNOWN";
}

/* Wake the engine after publishing GP_PUT. The engine checks GP_PUT and goes
 * to sleep under the device lock, so taking it here means the signal cannot
 * fall between its check and its sleep (no lost wakeup). This plays the role
 * of the doorbell register write; no ring data is touched under the lock. */
static void ring_doorbell(gpu_dev_t *d)
{
    pthread_mutex_lock(&d->lock);
    pthread_cond_signal(&d->work_cv);
    pthread_mutex_unlock(&d->lock);
}

/* ------------------------------------------------------------- engine */

/* Execute GPFIFO entry `get` of channel c from c->cursor.
 * Called WITHOUT the device lock held. The caller ACQUIRE-loaded GP_PUT
 * after the host published this slot, so its contents are visible here.
 * On EXEC_FAULT, *fault says why. */
static exec_result_t exec_entry(gpu_dev_t *d, channel_t *c, uint32_t get,
                                gpu_ch_status_t *fault)
{
    const gpu_pb_t *pb = &c->fifo[get & FIFO_MASK];

    while (c->cursor < pb->n) {
        uint32_t hdr   = pb->w[c->cursor];
        uint32_t op    = hdr >> 24;
        uint32_t nargs = hdr & 0x00FFFFFFu;
        const uint32_t *a = &pb->w[c->cursor + 1];

        if (nargs > pb->n - c->cursor - 1u) {
            *fault = GPU_CH_FAULT_INVALID_METHOD;   /* args run off the end */
            return EXEC_FAULT;
        }

        switch (op) {
        case GPU_OP_NOP:
            if (nargs != 0) goto bad_method;
            break;

        case GPU_OP_FILL:
            if (nargs != 3) goto bad_method;
            if (!range_ok(d, a[0], a[2])) goto bad_addr;
            memset(d->vram + a[0], (int)(a[1] & 0xFFu), a[2]);
            break;

        case GPU_OP_COPY:
            if (nargs != 3) goto bad_method;
            if (!range_ok(d, a[0], a[2]) || !range_ok(d, a[1], a[2]))
                goto bad_addr;
            memmove(d->vram + a[0], d->vram + a[1], a[2]);
            break;

        case GPU_OP_SEM_RELEASE:
            if (nargs != 2) goto bad_method;
            if (a[0] >= GPU_NUM_SEMAPHORES) goto bad_addr;
            pthread_mutex_lock(&d->lock);
            d->sem[a[0]] = a[1];
            pthread_cond_broadcast(&d->sem_cv);     /* wake host fences   */
            pthread_mutex_unlock(&d->lock);
            break;

        case GPU_OP_SEM_ACQUIRE: {
            if (nargs != 2) goto bad_method;
            if (a[0] >= GPU_NUM_SEMAPHORES) goto bad_addr;
            pthread_mutex_lock(&d->lock);
            bool ready = sem_reached(d->sem[a[0]], a[1]);
            if (!ready) {
                c->waiting  = true;                 /* park; cursor kept  */
                c->wait_sem = a[0];
                c->wait_val = a[1];
                c->wait_since_ms = now_ms();
            }
            pthread_mutex_unlock(&d->lock);
            if (!ready)
                return EXEC_BLOCKED;
            break;
        }

        case GPU_OP_HANG: {
            if (nargs != 0) goto bad_method;
            /* Spin like a stuck shader until the watchdog pulls the plug. */
            pthread_mutex_lock(&d->lock);
            while (d->running && !c->reset_req) {
                struct timespec ts = deadline_after(1);
                pthread_cond_timedwait(&d->wd_cv, &d->lock, &ts);
            }
            pthread_mutex_unlock(&d->lock);
            *fault = GPU_CH_FAULT_TIMEOUT;
            return EXEC_FAULT;
        }

        default:
            goto bad_method;
        }

        c->cursor += 1u + nargs;
        pthread_mutex_lock(&d->lock);
        d->stats.methods_done++;
        pthread_mutex_unlock(&d->lock);
    }
    return EXEC_DONE;

bad_method:
    *fault = GPU_CH_FAULT_INVALID_METHOD;
    return EXEC_FAULT;
bad_addr:
    *fault = GPU_CH_FAULT_BAD_ADDRESS;
    return EXEC_FAULT;
}

/* Lock held. Round-robin over channels that have work and can make progress.
 * Returns the channel and, in *get_out, the GPFIFO entry to execute. */
static channel_t *pick_runnable(gpu_dev_t *d, unsigned *idx_out,
                                uint32_t *get_out)
{
    for (unsigned k = 0; k < GPU_MAX_CHANNELS; k++) {
        unsigned i = (d->rr_next + k) % GPU_MAX_CHANNELS;
        channel_t *c = &d->ch[i];

        if (!atomic_load(&c->open) || atomic_load(&c->status) != GPU_CH_OK)
            continue;
        /* ACQUIRE pairs with the host's RELEASE publish in gpu_submit. */
        uint32_t put = atomic_load_explicit(&c->gp_put, memory_order_acquire);
        uint32_t get = atomic_load_explicit(&c->gp_get, memory_order_relaxed);
        if (get == put)
            continue;
        if (c->waiting && !sem_reached(d->sem[c->wait_sem], c->wait_val))
            continue;
        c->waiting = false;
        *idx_out = i;
        *get_out = get;
        return c;
    }
    return NULL;
}

/* Lock held, channel not executing on the engine. Drop all pending work
 * and mark the channel faulted. Entries the host publishes after this are
 * dead too (status is no longer OK); gpu_channel_recover() discards them. */
static void fault_channel(gpu_dev_t *d, channel_t *c, gpu_ch_status_t why)
{
    int ok = GPU_CH_OK;
    atomic_compare_exchange_strong(&c->status, &ok, (int)why);
    atomic_store_explicit(&c->gp_get,
                          atomic_load_explicit(&c->gp_put, memory_order_acquire),
                          memory_order_release);
    c->cursor  = 0;
    c->waiting = false;
    d->stats.faults++;
    pthread_cond_broadcast(&d->idle_cv);
}

static void *engine_main(void *arg)
{
    gpu_dev_t *d = arg;

    pthread_mutex_lock(&d->lock);
    while (d->running) {
        unsigned idx;
        uint32_t get;
        channel_t *c = pick_runnable(d, &idx, &get);
        if (c == NULL) {
            pthread_cond_wait(&d->work_cv, &d->lock);
            continue;
        }

        d->active = c;
        d->active_since_ms = now_ms();
        c->reset_req = false;
        pthread_mutex_unlock(&d->lock);

        gpu_ch_status_t why = GPU_CH_OK;
        exec_result_t r = exec_entry(d, c, get, &why);

        pthread_mutex_lock(&d->lock);
        d->active = NULL;
        d->rr_next = (idx + 1u) % GPU_MAX_CHANNELS;

        if (c->reset_req && r != EXEC_FAULT) {      /* watchdog raced us   */
            r = EXEC_FAULT;
            why = GPU_CH_FAULT_TIMEOUT;
        }
        switch (r) {
        case EXEC_DONE:
            c->cursor = 0;
            /* RELEASE: all our reads of the slot happen before the host
             * can see it free and overwrite it. */
            atomic_store_explicit(&c->gp_get, get + 1u, memory_order_release);
            d->stats.entries_done++;
            pthread_cond_broadcast(&d->idle_cv);
            break;
        case EXEC_BLOCKED:
            d->stats.ctx_switches++;
            break;
        case EXEC_FAULT:
            fault_channel(d, c, why);
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return NULL;
}

/* ----------------------------------------------------------- watchdog */
/* Lock held. Check every quarter of the shortest enabled timeout, so a
 * timeout is detected within 1.25x its configured value. */
static uint32_t watchdog_period(const gpu_dev_t *d)
{
    uint32_t t = d->watchdog_ms;
    if (d->acquire_timeout_ms != 0 && d->acquire_timeout_ms < t)
        t = d->acquire_timeout_ms;
    return t / 4u ? t / 4u : 1u;
}

/* Lock held. Fault channels parked on an ACQUIRE that is still unmet after
 * the acquire timeout. A parked channel is off the engine, so it can be torn
 * down right here; the active one is skipped (the hang check covers it). */
static void check_acquire_timeouts(gpu_dev_t *d)
{
    if (d->acquire_timeout_ms == 0)
        return;
    uint64_t now = now_ms();
    for (unsigned i = 0; i < GPU_MAX_CHANNELS; i++) {
        channel_t *c = &d->ch[i];
        if (!atomic_load(&c->open) || atomic_load(&c->status) != GPU_CH_OK ||
            !c->waiting || d->active == c)
            continue;
        if (sem_reached(d->sem[c->wait_sem], c->wait_val))
            continue;                   /* satisfied, just not rescheduled */
        if (now - c->wait_since_ms >= d->acquire_timeout_ms) {
            d->stats.acquire_timeouts++;
            fault_channel(d, c, GPU_CH_FAULT_ACQUIRE_TIMEOUT);
        }
    }
}

static void *watchdog_main(void *arg)
{
    gpu_dev_t *d = arg;

    pthread_mutex_lock(&d->lock);
    while (d->running) {
        struct timespec ts = deadline_after(watchdog_period(d));
        pthread_cond_timedwait(&d->wd_cv, &d->lock, &ts);

        check_acquire_timeouts(d);

        channel_t *c = d->active;
        if (c != NULL && !c->reset_req &&
            now_ms() - d->active_since_ms >= d->watchdog_ms) {
            c->reset_req = true;
            atomic_store(&c->status, GPU_CH_FAULT_TIMEOUT);  /* submit now -EIO */
            d->stats.watchdog_resets++;
            pthread_cond_broadcast(&d->wd_cv);      /* kick the engine     */
        }
    }
    pthread_mutex_unlock(&d->lock);
    return NULL;
}

/* -------------------------------------------------------- public API */
gpu_dev_t *gpu_create(size_t vram_bytes, uint32_t watchdog_ms)
{
    if (vram_bytes == 0 || vram_bytes > UINT32_MAX || watchdog_ms == 0)
        return NULL;

    gpu_dev_t *d = calloc(1, sizeof(*d));
    if (d == NULL)
        return NULL;
    d->vram = calloc(1, vram_bytes);
    if (d->vram == NULL) {
        free(d);
        return NULL;
    }
    d->vram_size   = vram_bytes;
    d->watchdog_ms = watchdog_ms;
    d->running     = true;

    pthread_mutex_init(&d->lock, NULL);
    for (unsigned i = 0; i < GPU_MAX_CHANNELS; i++) {
        channel_t *c = &d->ch[i];
        atomic_init(&c->open, false);
        atomic_init(&c->status, GPU_CH_OK);
        atomic_init(&c->gp_put, 0u);
        atomic_init(&c->gp_get, 0u);
        pthread_mutex_init(&c->submit_lock, NULL);
    }
    init_cond(&d->work_cv);
    init_cond(&d->sem_cv);
    init_cond(&d->idle_cv);
    init_cond(&d->wd_cv);

    if (pthread_create(&d->engine_thr, NULL, engine_main, d) != 0) {
        free(d->vram);
        free(d);
        return NULL;
    }
    if (pthread_create(&d->wd_thr, NULL, watchdog_main, d) != 0) {
        pthread_mutex_lock(&d->lock);
        d->running = false;
        pthread_cond_broadcast(&d->work_cv);
        pthread_mutex_unlock(&d->lock);
        pthread_join(d->engine_thr, NULL);
        free(d->vram);
        free(d);
        return NULL;
    }
    return d;
}

void gpu_destroy(gpu_dev_t *d)
{
    if (d == NULL)
        return;
    pthread_mutex_lock(&d->lock);
    d->running = false;
    pthread_cond_broadcast(&d->work_cv);
    pthread_cond_broadcast(&d->wd_cv);
    pthread_cond_broadcast(&d->sem_cv);
    pthread_cond_broadcast(&d->idle_cv);
    pthread_mutex_unlock(&d->lock);

    pthread_join(d->engine_thr, NULL);
    pthread_join(d->wd_thr, NULL);

    pthread_cond_destroy(&d->work_cv);
    pthread_cond_destroy(&d->sem_cv);
    pthread_cond_destroy(&d->idle_cv);
    pthread_cond_destroy(&d->wd_cv);
    for (unsigned i = 0; i < GPU_MAX_CHANNELS; i++)
        pthread_mutex_destroy(&d->ch[i].submit_lock);
    pthread_mutex_destroy(&d->lock);
    free(d->vram);
    free(d);
}

int gpu_channel_open(gpu_dev_t *d, unsigned *out_ch)
{
    pthread_mutex_lock(&d->lock);
    for (unsigned i = 0; i < GPU_MAX_CHANNELS; i++) {
        if (!atomic_load(&d->ch[i].open)) {
            /* Never opened before (there is no close), so it is still in
             * its zeroed initial state: just make it visible. */
            atomic_store(&d->ch[i].open, true);
            *out_ch = i;
            pthread_mutex_unlock(&d->lock);
            return 0;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return -ENOSPC;
}

int gpu_submit(gpu_dev_t *d, unsigned ch, const gpu_pb_t *pb)
{
    if (ch >= GPU_MAX_CHANNELS || pb == NULL || pb->n > GPU_PB_MAX_WORDS)
        return -EINVAL;

    channel_t *c = &d->ch[ch];
    if (!atomic_load(&c->open))
        return -EINVAL;

    /* Producer side of the SPSC ring. The device lock is not taken to
     * move the work; only the doorbell below touches it. */
    pthread_mutex_lock(&c->submit_lock);
    int rc = 0;
    /* We are the only writer of GP_PUT (submit_lock), so relaxed is enough. */
    uint32_t put = atomic_load_explicit(&c->gp_put, memory_order_relaxed);
    /* ACQUIRE pairs with the engine's RELEASE of GP_GET: once a slot shows
     * as free, the engine has finished reading it. */
    uint32_t get = atomic_load_explicit(&c->gp_get, memory_order_acquire);

    if (atomic_load(&c->status) != GPU_CH_OK)
        rc = -EIO;
    else if (put - get >= GPU_GPFIFO_ENTRIES)       /* wrap-safe distance  */
        rc = -EAGAIN;
    else {
        gpu_pb_t *slot = &c->fifo[put & FIFO_MASK];
        memcpy(slot->w, pb->w, pb->n * sizeof(uint32_t));
        slot->n = pb->n;
        /* RELEASE publish: the segment above is visible to any thread
         * that ACQUIRE-loads this GP_PUT value. */
        atomic_store_explicit(&c->gp_put, put + 1u, memory_order_release);
        ring_doorbell(d);
    }
    pthread_mutex_unlock(&c->submit_lock);
    return rc;
}

int gpu_sem_wait(gpu_dev_t *d, uint32_t sem, uint32_t value, uint32_t timeout_ms)
{
    if (sem >= GPU_NUM_SEMAPHORES)
        return -EINVAL;

    struct timespec dl = deadline_after(timeout_ms);
    int rc = 0;

    pthread_mutex_lock(&d->lock);
    while (!sem_reached(d->sem[sem], value)) {
        if (!d->running ||
            pthread_cond_timedwait(&d->sem_cv, &d->lock, &dl) == ETIMEDOUT) {
            rc = sem_reached(d->sem[sem], value) ? 0 : -ETIMEDOUT;
            break;
        }
    }
    pthread_mutex_unlock(&d->lock);
    return rc;
}

uint32_t gpu_sem_read(gpu_dev_t *d, uint32_t sem)
{
    if (sem >= GPU_NUM_SEMAPHORES)
        return 0;
    pthread_mutex_lock(&d->lock);
    uint32_t v = d->sem[sem];
    pthread_mutex_unlock(&d->lock);
    return v;
}

int gpu_channel_idle_wait(gpu_dev_t *d, unsigned ch, uint32_t timeout_ms)
{
    if (ch >= GPU_MAX_CHANNELS)
        return -EINVAL;

    struct timespec dl = deadline_after(timeout_ms);
    int rc = 0;

    pthread_mutex_lock(&d->lock);
    channel_t *c = &d->ch[ch];
    /* A faulted channel is idle once it is off the engine: whatever is left
     * in its ring is dead work that will never run. */
    while (d->active == c ||
           (atomic_load(&c->status) == GPU_CH_OK &&
            atomic_load_explicit(&c->gp_get, memory_order_acquire) !=
            atomic_load_explicit(&c->gp_put, memory_order_acquire))) {
        if (!d->running ||
            pthread_cond_timedwait(&d->idle_cv, &d->lock, &dl) == ETIMEDOUT) {
            rc = -ETIMEDOUT;
            break;
        }
    }
    if (rc == 0 && atomic_load(&c->status) != GPU_CH_OK)
        rc = -EIO;
    pthread_mutex_unlock(&d->lock);
    return rc;
}

gpu_ch_status_t gpu_channel_status(gpu_dev_t *d, unsigned ch)
{
    if (ch >= GPU_MAX_CHANNELS)
        return GPU_CH_FAULT_BAD_ADDRESS;
    return (gpu_ch_status_t)atomic_load(&d->ch[ch].status);
}

int gpu_channel_recover(gpu_dev_t *d, unsigned ch)
{
    if (ch >= GPU_MAX_CHANNELS)
        return -EINVAL;
    channel_t *c = &d->ch[ch];
    /* Lock order: submit_lock, then device lock (same as gpu_submit). Holding
     * submit_lock freezes GP_PUT while the ring is emptied. */
    pthread_mutex_lock(&c->submit_lock);
    pthread_mutex_lock(&d->lock);
    int rc = 0;
    if (!atomic_load(&c->open))
        rc = -EINVAL;
    else if (d->active == c)
        rc = -EBUSY;                                /* still being torn down */
    else {
        /* Off the engine and faulted, so the engine will not touch the ring:
         * discard everything up to GP_PUT, including dead entries published
         * after the fault, then reopen the channel. */
        atomic_store_explicit(&c->gp_get,
                              atomic_load_explicit(&c->gp_put, memory_order_relaxed),
                              memory_order_release);
        c->cursor    = 0;
        c->waiting   = false;
        c->reset_req = false;
        atomic_store(&c->status, GPU_CH_OK);
    }
    pthread_mutex_unlock(&d->lock);
    pthread_mutex_unlock(&c->submit_lock);
    return rc;
}

int gpu_vram_write(gpu_dev_t *d, uint32_t off, const void *src, uint32_t len)
{
    if (!range_ok(d, off, len) || (len && src == NULL))
        return -EINVAL;
    memcpy(d->vram + off, src, len);
    return 0;
}

int gpu_vram_read(gpu_dev_t *d, uint32_t off, void *dst, uint32_t len)
{
    if (!range_ok(d, off, len) || (len && dst == NULL))
        return -EINVAL;
    memcpy(dst, d->vram + off, len);
    return 0;
}

void gpu_set_acquire_timeout(gpu_dev_t *d, uint32_t ms)
{
    pthread_mutex_lock(&d->lock);
    d->acquire_timeout_ms = ms;
    pthread_cond_broadcast(&d->wd_cv);  /* re-arm the watchdog period now */
    pthread_mutex_unlock(&d->lock);
}

void gpu_get_stats(gpu_dev_t *d, gpu_stats_t *out)
{
    pthread_mutex_lock(&d->lock);
    *out = d->stats;
    pthread_mutex_unlock(&d->lock);
}
