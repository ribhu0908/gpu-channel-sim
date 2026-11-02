/* Host tests for gpu_sim. Run under ThreadSanitizer with `make tsan`. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "gpu_sim.h"

static int g_checks, g_failures;

#define CHECK(cond)                                                        \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(cond)) {                                                     \
            g_failures++;                                                  \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
        }                                                                  \
    } while (0)

#define RUN(fn) do { printf("[ RUN  ] %s\n", #fn); fn(); } while (0)

#define WD_MS 50u

static uint64_t ms_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

/* ------------------------------------------------------------------ */
static void test_fill_copy_and_fence(void)
{
    gpu_dev_t *g = gpu_create(4096, WD_MS);
    unsigned ch = 0;
    CHECK(g != NULL);
    if (g == NULL)
        return;
    CHECK(gpu_channel_open(g, &ch) == 0);

    const char msg[] = "hello from the host";
    CHECK(gpu_vram_write(g, 0, msg, sizeof msg) == 0);

    gpu_pb_t pb;
    pb_reset(&pb);
    pb_fill(&pb, 1024, 0x5A, 64);
    pb_copy(&pb, 2048, 0, sizeof msg);
    pb_sem_release(&pb, 0, 1);                  /* fence */
    CHECK(gpu_submit(g, ch, &pb) == 0);

    CHECK(gpu_sem_wait(g, 0, 1, 1000) == 0);
    char out[sizeof msg];
    uint8_t filled[64];
    gpu_vram_read(g, 2048, out, sizeof out);
    gpu_vram_read(g, 1024, filled, sizeof filled);
    CHECK(memcmp(out, msg, sizeof msg) == 0);
    CHECK(filled[0] == 0x5A && filled[63] == 0x5A);

    CHECK(gpu_channel_idle_wait(g, ch, 1000) == 0);  /* entry retired */
    gpu_stats_t s;
    gpu_get_stats(g, &s);
    CHECK(s.entries_done == 1 && s.methods_done == 3);
    gpu_destroy(g);
}

/* Channel B depends on channel A, but B is submitted FIRST. The engine must
 * park B, run A, then resume B - not deadlock and not run B early. */
static void test_cross_channel_dependency(void)
{
    gpu_dev_t *g = gpu_create(4096, WD_MS);
    unsigned a, b;
    gpu_channel_open(g, &a);
    gpu_channel_open(g, &b);

    gpu_pb_t pb;
    pb_reset(&pb);
    pb_sem_acquire(&pb, 5, 1);                  /* wait for A's result   */
    pb_copy(&pb, 512, 0, 16);                   /* consume it            */
    pb_sem_release(&pb, 6, 1);
    CHECK(gpu_submit(g, b, &pb) == 0);

    /* Give the engine time to discover B is blocked. */
    CHECK(gpu_sem_wait(g, 6, 1, 30) == -ETIMEDOUT);

    pb_reset(&pb);
    pb_fill(&pb, 0, 0x7E, 16);                  /* produce               */
    pb_sem_release(&pb, 5, 1);
    CHECK(gpu_submit(g, a, &pb) == 0);

    CHECK(gpu_sem_wait(g, 6, 1, 1000) == 0);
    uint8_t v[16];
    gpu_vram_read(g, 512, v, sizeof v);
    CHECK(v[0] == 0x7E && v[15] == 0x7E);       /* B saw A's data        */

    gpu_stats_t s;
    gpu_get_stats(g, &s);
    CHECK(s.ctx_switches >= 1);
    gpu_destroy(g);
}

/* Two channels hand a token back and forth 80 times through semaphores. */
static void test_ping_pong(void)
{
    gpu_dev_t *g = gpu_create(4096, WD_MS);
    unsigned a, b;
    gpu_channel_open(g, &a);
    gpu_channel_open(g, &b);

    gpu_pb_t pa, pbb;
    pb_reset(&pa);
    pb_reset(&pbb);
    for (uint32_t k = 1; k <= 40; k++) {   /* 40 rounds x 6 words fits one segment */
        pb_sem_acquire(&pa, 1, k - 1);          /* B finished round k-1  */
        pb_sem_release(&pa, 0, k);
        pb_sem_acquire(&pbb, 0, k);             /* A finished round k    */
        pb_sem_release(&pbb, 1, k);
    }
    CHECK(gpu_submit(g, b, &pbb) == 0);
    CHECK(gpu_submit(g, a, &pa) == 0);
    CHECK(gpu_sem_wait(g, 1, 40, 2000) == 0);
    CHECK(gpu_sem_read(g, 0) == 40);
    gpu_destroy(g);
}

static void test_invalid_method_is_isolated(void)
{
    gpu_dev_t *g = gpu_create(4096, WD_MS);
    unsigned bad, good;
    gpu_channel_open(g, &bad);
    gpu_channel_open(g, &good);

    gpu_pb_t pb;
    pb_reset(&pb);
    pb_raw(&pb, (0x77u << 24));                 /* unknown opcode        */
    CHECK(gpu_submit(g, bad, &pb) == 0);
    CHECK(gpu_channel_idle_wait(g, bad, 1000) == -EIO);
    CHECK(gpu_channel_status(g, bad) == GPU_CH_FAULT_INVALID_METHOD);
    CHECK(gpu_submit(g, bad, &pb) == -EIO);     /* faulted: refuses work */

    pb_reset(&pb);
    pb_raw(&pb, (GPU_OP_FILL << 24) | 3);       /* header says 3 args... */
    pb_raw(&pb, 0);                             /* ...only 1 present     */
    unsigned trunc;
    gpu_channel_open(g, &trunc);
    gpu_submit(g, trunc, &pb);
    CHECK(gpu_channel_idle_wait(g, trunc, 1000) == -EIO);
    CHECK(gpu_channel_status(g, trunc) == GPU_CH_FAULT_INVALID_METHOD);

    pb_reset(&pb);
    pb_fill(&pb, 4000, 1, 200);                 /* runs past VRAM end    */
    unsigned oob;
    gpu_channel_open(g, &oob);
    gpu_submit(g, oob, &pb);
    CHECK(gpu_channel_idle_wait(g, oob, 1000) == -EIO);
    CHECK(gpu_channel_status(g, oob) == GPU_CH_FAULT_BAD_ADDRESS);

    pb_reset(&pb);
    pb_fill(&pb, 0, 9, 8);
    pb_sem_release(&pb, 2, 1);
    CHECK(gpu_submit(g, good, &pb) == 0);       /* others unaffected     */
    CHECK(gpu_sem_wait(g, 2, 1, 1000) == 0);
    CHECK(gpu_channel_status(g, good) == GPU_CH_OK);

    CHECK(gpu_channel_recover(g, bad) == 0);    /* reuse after recovery  */
    pb_reset(&pb);
    pb_sem_release(&pb, 3, 7);
    CHECK(gpu_submit(g, bad, &pb) == 0);
    CHECK(gpu_sem_wait(g, 3, 7, 1000) == 0);
    gpu_destroy(g);
}

/* A hung channel must be detected within ~watchdog time, torn down, and
 * must not block a healthy channel queued behind it. */
static void test_watchdog_recovers_hang(void)
{
    gpu_dev_t *g = gpu_create(4096, WD_MS);
    unsigned hung, ok;
    gpu_channel_open(g, &hung);
    gpu_channel_open(g, &ok);

    gpu_pb_t pb;
    pb_reset(&pb);
    pb_hang(&pb);
    pb_sem_release(&pb, 10, 1);                 /* must never run        */
    gpu_submit(g, hung, &pb);
    gpu_submit(g, hung, &pb);                   /* queued work dropped   */

    pb_reset(&pb);
    pb_sem_release(&pb, 11, 1);
    gpu_submit(g, ok, &pb);

    uint64_t t0 = ms_now();
    CHECK(gpu_sem_wait(g, 11, 1, 2000) == 0);
    uint64_t waited = ms_now() - t0;
    CHECK(waited >= WD_MS / 2 && waited < 10 * WD_MS);

    CHECK(gpu_channel_idle_wait(g, hung, 1000) == -EIO);
    CHECK(gpu_channel_status(g, hung) == GPU_CH_FAULT_TIMEOUT);
    CHECK(gpu_sem_read(g, 10) == 0);

    gpu_stats_t s;
    gpu_get_stats(g, &s);
    CHECK(s.watchdog_resets == 1 && s.faults == 1);
    gpu_destroy(g);
}

static void test_gpfifo_backpressure(void)
{
    gpu_dev_t *g = gpu_create(4096, WD_MS);
    unsigned ch;
    gpu_channel_open(g, &ch);

    gpu_pb_t pb;
    pb_reset(&pb);
    pb_sem_acquire(&pb, 20, 1);                 /* stall the channel     */
    CHECK(gpu_submit(g, ch, &pb) == 0);

    pb_reset(&pb);
    pb_nop(&pb);
    int accepted = 1;
    while (gpu_submit(g, ch, &pb) == 0)
        accepted++;
    CHECK(accepted == GPU_GPFIFO_ENTRIES);
    CHECK(gpu_submit(g, ch, &pb) == -EAGAIN);

    CHECK(gpu_submit(g, 99, &pb) == -EINVAL);
    unsigned extra, n = 1;
    while (gpu_channel_open(g, &extra) == 0)
        n++;
    CHECK(n == GPU_MAX_CHANNELS);

    gpu_destroy(g);                             /* clean shutdown while stalled */
}

/* Builders must refuse to overflow a segment instead of corrupting it. */
static void test_pushbuffer_builder_bounds(void)
{
    gpu_pb_t pb;
    pb_reset(&pb);
    int ok = 0;
    while (pb_fill(&pb, 0, 0, 1) == 0)
        ok++;
    CHECK(ok == GPU_PB_MAX_WORDS / 4);
    CHECK(pb.n <= GPU_PB_MAX_WORDS);
}

/* 4 host threads x 1 channel each, 400 submissions apiece, with retries on
 * back-pressure. Every region must end up with the last value written. */
#define STRESS_THREADS 4
#define STRESS_ITERS   400

typedef struct { gpu_dev_t *g; unsigned ch; uint32_t id; } stress_arg_t;

static void *stress_worker(void *p)
{
    stress_arg_t *a = p;
    gpu_pb_t pb;
    for (uint32_t i = 1; i <= STRESS_ITERS; i++) {
        pb_reset(&pb);
        pb_fill(&pb, a->id * 256u, (uint8_t)i, 256);
        pb_sem_release(&pb, a->id, i);
        int rc;
        while ((rc = gpu_submit(a->g, a->ch, &pb)) == -EAGAIN)
            gpu_sem_wait(a->g, a->id, i - GPU_GPFIFO_ENTRIES / 2u, 10);
        if (rc != 0)
            return (void *)1;
    }
    return NULL;
}

static void test_multithreaded_stress(void)
{
    gpu_dev_t *g = gpu_create(STRESS_THREADS * 256, 500);
    pthread_t t[STRESS_THREADS];
    stress_arg_t a[STRESS_THREADS];

    for (uint32_t i = 0; i < STRESS_THREADS; i++) {
        a[i].g = g;
        a[i].id = i;
        gpu_channel_open(g, &a[i].ch);
        pthread_create(&t[i], NULL, stress_worker, &a[i]);
    }
    int errors = 0;
    for (uint32_t i = 0; i < STRESS_THREADS; i++) {
        void *r;
        pthread_join(t[i], &r);
        errors += r != NULL;
    }
    CHECK(errors == 0);

    for (uint32_t i = 0; i < STRESS_THREADS; i++) {
        CHECK(gpu_sem_wait(g, i, STRESS_ITERS, 5000) == 0);
        uint8_t buf[256];
        gpu_vram_read(g, i * 256u, buf, sizeof buf);
        CHECK(buf[0] == (uint8_t)STRESS_ITERS && buf[255] == (uint8_t)STRESS_ITERS);
        /* A fence fires mid-entry; wait for the entry to retire before
           reading completion stats. */
        CHECK(gpu_channel_idle_wait(g, a[i].ch, 5000) == 0);
    }
    gpu_stats_t s;
    gpu_get_stats(g, &s);
    CHECK(s.entries_done == STRESS_THREADS * STRESS_ITERS);
    CHECK(s.faults == 0);
    gpu_destroy(g);
}

/* 4 host threads share ONE channel. submit_lock must serialise them into a
 * single producer: no lost entry, no torn segment, per-thread order kept. */
#define SHARED_THREADS 4
#define SHARED_ITERS   300

typedef struct { gpu_dev_t *g; unsigned ch; uint32_t id; uint32_t retries; } shared_arg_t;

static void *shared_worker(void *p)
{
    shared_arg_t *a = p;
    gpu_pb_t pb;
    for (uint32_t i = 1; i <= SHARED_ITERS; i++) {
        pb_reset(&pb);
        pb_fill(&pb, a->id * 64u, (uint8_t)i, 64);
        pb_sem_release(&pb, a->id, i);          /* must rise monotonically */
        int rc;
        while ((rc = gpu_submit(a->g, a->ch, &pb)) == -EAGAIN) {
            struct timespec ts = { 0, 100000L };    /* 100 us */
            a->retries++;
            nanosleep(&ts, NULL);
        }
        if (rc != 0)
            return (void *)1;
    }
    return NULL;
}

static void test_shared_channel_many_producers(void)
{
    gpu_dev_t *g = gpu_create(SHARED_THREADS * 64, 500);
    unsigned ch;
    gpu_channel_open(g, &ch);
    pthread_t t[SHARED_THREADS];
    shared_arg_t a[SHARED_THREADS];

    for (uint32_t i = 0; i < SHARED_THREADS; i++) {
        a[i] = (shared_arg_t){ g, ch, i, 0 };
        pthread_create(&t[i], NULL, shared_worker, &a[i]);
    }
    int errors = 0;
    for (uint32_t i = 0; i < SHARED_THREADS; i++) {
        void *r;
        pthread_join(t[i], &r);
        errors += r != NULL;
    }
    CHECK(errors == 0);
    CHECK(gpu_channel_idle_wait(g, ch, 5000) == 0);

    for (uint32_t i = 0; i < SHARED_THREADS; i++) {
        CHECK(gpu_sem_read(g, i) == SHARED_ITERS);
        uint8_t buf[64];
        gpu_vram_read(g, i * 64u, buf, sizeof buf);
        CHECK(buf[0] == (uint8_t)SHARED_ITERS && buf[63] == (uint8_t)SHARED_ITERS);
    }
    gpu_stats_t s;
    gpu_get_stats(g, &s);
    CHECK(s.entries_done == SHARED_THREADS * SHARED_ITERS);
    CHECK(s.methods_done == 2u * SHARED_THREADS * SHARED_ITERS);
    CHECK(s.faults == 0);
    gpu_destroy(g);
}

int main(void)
{
    RUN(test_fill_copy_and_fence);
    RUN(test_cross_channel_dependency);
    RUN(test_ping_pong);
    RUN(test_invalid_method_is_isolated);
    RUN(test_watchdog_recovers_hang);
    RUN(test_gpfifo_backpressure);
    RUN(test_pushbuffer_builder_bounds);
    RUN(test_multithreaded_stress);
    RUN(test_shared_channel_many_producers);

    printf("\n%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
