/* Benchmark for gpu_sim: host-side submit cost and end-to-end throughput.
 * Build against any revision of the simulator, for example:
 *   gcc -std=c11 -O2 -Iinclude src/gpu_sim.c bench/bench.c -o bench -lpthread && ./bench 7 100000
 * Prints one JSON line. Compare revisions by checking each out and re-running. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpu_sim.h"

static uint64_t ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
typedef struct { gpu_dev_t *g; unsigned ch; int iters; uint64_t sub_ns, ok, eagain; int err; } arg_t;

static void *worker(void *p)
{
    arg_t *a = p; gpu_pb_t pb; pb_reset(&pb); pb_nop(&pb);
    for (int i = 0; i < a->iters;) {
        uint64_t t0 = ns(); int rc = gpu_submit(a->g, a->ch, &pb); uint64_t t1 = ns();
        if (rc == 0) { a->sub_ns += t1 - t0; a->ok++; i++; }
        else if (rc == -EAGAIN) { a->eagain++; sched_yield(); }
        else { a->err = rc; return NULL; }
    }
    return NULL;
}
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* returns entries/s and mean ns per successful gpu_submit */
static int run(int threads, int shared, int iters, double *tput, double *sub_ns, double clk)
{
    gpu_dev_t *g = gpu_create(4096, 60000);
    int nch = shared ? 1 : threads; unsigned ch[8];
    for (int i = 0; i < nch; i++) gpu_channel_open(g, &ch[i]);
    pthread_t t[8]; arg_t a[8]; memset(a, 0, sizeof a);
    uint64_t t0 = ns();
    for (int i = 0; i < threads; i++) { a[i] = (arg_t){ g, ch[shared ? 0 : i], iters, 0, 0, 0, 0 }; pthread_create(&t[i], NULL, worker, &a[i]); }
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    for (int i = 0; i < nch; i++) if (gpu_channel_idle_wait(g, ch[i], 60000) != 0) { gpu_destroy(g); return -1; }
    uint64_t t1 = ns();
    uint64_t sum = 0, ok = 0;
    for (int i = 0; i < threads; i++) { if (a[i].err) { gpu_destroy(g); return -2; } sum += a[i].sub_ns; ok += a[i].ok; }
    gpu_stats_t s; gpu_get_stats(g, &s);
    if (s.entries_done != (uint64_t)threads * (uint64_t)iters || s.faults != 0) { gpu_destroy(g); return -3; }
    gpu_destroy(g);
    *tput = (double)ok / ((double)(t1 - t0) / 1e9);
    *sub_ns = (double)sum / (double)ok - clk;
    return 0;
}

int main(int argc, char **argv)
{
    int runs = argc > 1 ? atoi(argv[1]) : 7, iters = argc > 2 ? atoi(argv[2]) : 100000;
    uint64_t c0 = ns(); for (int i = 0; i < 1000000; i++) (void)ns(); double clk = (double)(ns() - c0) / 1e6;
    struct { const char *name; int threads, shared; } cfg[] = {
        { "1 host thread, 1 channel", 1, 0 }, { "4 host threads, 4 channels (one each)", 4, 0 }, { "4 host threads, 1 shared channel", 4, 1 } };
    printf("{\"clock_overhead_ns\":%.1f,\"runs\":%d,\"iters_per_thread\":%d,\"configs\":[", clk, runs, iters);
    for (int c = 0; c < 3; c++) {
        double tp[32], sn[32];
        for (int r = 0; r < runs; r++) { int rc = run(cfg[c].threads, cfg[c].shared, iters, &tp[r], &sn[r], clk); if (rc) { fprintf(stderr, "run failed %d\n", rc); return 1; } }
        qsort(tp, (size_t)runs, sizeof *tp, cmp); qsort(sn, (size_t)runs, sizeof *sn, cmp);
        printf("%s{\"name\":\"%s\",\"tput_median\":%.0f,\"tput_min\":%.0f,\"tput_max\":%.0f,\"submit_ns_median\":%.0f,\"submit_ns_min\":%.0f,\"submit_ns_max\":%.0f}",
               c ? "," : "", cfg[c].name, tp[runs / 2], tp[0], tp[runs - 1], sn[runs / 2], sn[0], sn[runs - 1]);
    }
    printf("]}\n");
    return 0;
}
