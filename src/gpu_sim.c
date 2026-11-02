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
 * Locking: one device mutex guards all channel state and semaphores. VRAM is
 * touched without the lock; ordering comes from submit / semaphore release,
 * which both go through the mutex (a happens-before edge, as a real GPU uses
 * memory barriers before a semaphore release).
 */
#define _POSIX_C_SOURCE 200809L
#include "gpu_sim.h"

#include <errno.h>
#include <pthread.h>
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
    bool            open;
    gpu_ch_status_t status;
    uint32_t        gp_put;         /* host-owned: next slot to fill        */
    uint32_t        gp_get;         /* engine-owned: next slot to execute   */
    gpu_pb_t        fifo[GPU_GPFIFO_ENTRIES];

    uint32_t        cursor;         /* resume point inside current entry    */
    bool            waiting;        /* parked on an unmet ACQUIRE           */
    uint32_t        wait_sem;
    uint32_t        wait_val;
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
    }
    return "UNKNOWN";
}

/* ------------------------------------------------------------- engine */

/* Execute the channel's current GPFIFO entry from ch->cursor.
 * Called WITHOUT the lock held. On EXEC_FAULT, *fault says why. */
static exec_result_t exec_entry(gpu_dev_t *d, channel_t *c,
                                gpu_ch_status_t *fault)
{
    const gpu_pb_t *pb = &c->fifo[c->gp_get & FIFO_MASK];

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

/* Lock held. Round-robin over channels that have work and can make progress. */
static channel_t *pick_runnable(gpu_dev_t *d, unsigned *idx_out)
{
    for (unsigned k = 0; k < GPU_MAX_CHANNELS; k++) {
        unsigned i = (d->rr_next + k) % GPU_MAX_CHANNELS;
        channel_t *c = &d->ch[i];

        if (!c->open || c->status != GPU_CH_OK || c->gp_get == c->gp_put)
            continue;
        if (c->waiting && !sem_reached(d->sem[c->wait_sem], c->wait_val))
            continue;
        c->waiting = false;
        *idx_out = i;
        return c;
    }
    return NULL;
}

/* Lock held. Drop all pending work and mark the channel faulted. */
static void fault_channel(gpu_dev_t *d, channel_t *c, gpu_ch_status_t why)
{
    if (c->status == GPU_CH_OK)
        c->status = why;
    c->gp_get  = c->gp_put;
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
        channel_t *c = pick_runnable(d, &idx);
        if (c == NULL) {
            pthread_cond_wait(&d->work_cv, &d->lock);
            continue;
        }

        d->active = c;
        d->active_since_ms = now_ms();
        c->reset_req = false;
        pthread_mutex_unlock(&d->lock);

        gpu_ch_status_t why = GPU_CH_OK;
        exec_result_t r = exec_entry(d, c, &why);

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
            c->gp_get++;
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
static void *watchdog_main(void *arg)
{
    gpu_dev_t *d = arg;
    uint32_t period = d->watchdog_ms / 4u ? d->watchdog_ms / 4u : 1u;

    pthread_mutex_lock(&d->lock);
    while (d->running) {
        struct timespec ts = deadline_after(period);
        pthread_cond_timedwait(&d->wd_cv, &d->lock, &ts);

        channel_t *c = d->active;
        if (c != NULL && !c->reset_req &&
            now_ms() - d->active_since_ms >= d->watchdog_ms) {
            c->reset_req = true;
            c->status = GPU_CH_FAULT_TIMEOUT;
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
    pthread_mutex_destroy(&d->lock);
    free(d->vram);
    free(d);
}

int gpu_channel_open(gpu_dev_t *d, unsigned *out_ch)
{
    pthread_mutex_lock(&d->lock);
    for (unsigned i = 0; i < GPU_MAX_CHANNELS; i++) {
        if (!d->ch[i].open) {
            memset(&d->ch[i], 0, sizeof(d->ch[i]));
            d->ch[i].open = true;
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

    pthread_mutex_lock(&d->lock);
    channel_t *c = &d->ch[ch];
    int rc = 0;

    if (!c->open)
        rc = -EINVAL;
    else if (c->status != GPU_CH_OK)
        rc = -EIO;
    else if (c->gp_put - c->gp_get >= GPU_GPFIFO_ENTRIES)
        rc = -EAGAIN;
    else {
        gpu_pb_t *slot = &c->fifo[c->gp_put & FIFO_MASK];
        memcpy(slot->w, pb->w, pb->n * sizeof(uint32_t));
        slot->n = pb->n;
        c->gp_put++;                                /* GP_PUT write        */
        pthread_cond_signal(&d->work_cv);           /* doorbell            */
    }
    pthread_mutex_unlock(&d->lock);
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
    while (c->gp_get != c->gp_put || d->active == c) {
        if (!d->running ||
            pthread_cond_timedwait(&d->idle_cv, &d->lock, &dl) == ETIMEDOUT) {
            rc = -ETIMEDOUT;
            break;
        }
    }
    if (rc == 0 && c->status != GPU_CH_OK)
        rc = -EIO;
    pthread_mutex_unlock(&d->lock);
    return rc;
}

gpu_ch_status_t gpu_channel_status(gpu_dev_t *d, unsigned ch)
{
    if (ch >= GPU_MAX_CHANNELS)
        return GPU_CH_FAULT_BAD_ADDRESS;
    pthread_mutex_lock(&d->lock);
    gpu_ch_status_t s = d->ch[ch].status;
    pthread_mutex_unlock(&d->lock);
    return s;
}

int gpu_channel_recover(gpu_dev_t *d, unsigned ch)
{
    if (ch >= GPU_MAX_CHANNELS)
        return -EINVAL;
    pthread_mutex_lock(&d->lock);
    channel_t *c = &d->ch[ch];
    int rc = 0;
    if (!c->open)
        rc = -EINVAL;
    else if (d->active == c)
        rc = -EBUSY;                                /* still being torn down */
    else {
        c->status    = GPU_CH_OK;
        c->gp_get    = c->gp_put;
        c->cursor    = 0;
        c->waiting   = false;
        c->reset_req = false;
    }
    pthread_mutex_unlock(&d->lock);
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

void gpu_get_stats(gpu_dev_t *d, gpu_stats_t *out)
{
    pthread_mutex_lock(&d->lock);
    *out = d->stats;
    pthread_mutex_unlock(&d->lock);
}
