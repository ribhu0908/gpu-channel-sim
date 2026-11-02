/*
 * demo.c - three clients share one GPU engine:
 *   upload   copies host data into VRAM, then releases semaphore 0
 *   compute  submitted FIRST, but must wait (ACQUIRE sem 0) for upload
 *   buggy    hangs the engine; the watchdog must kill it without
 *            stalling the other two
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "gpu_sim.h"

static struct timespec t0;

static void pause_ms(long m)    /* let the engine make progress between submits */
{
    struct timespec ts = { 0, m * 1000000L };
    nanosleep(&ts, NULL);
}

static double ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)(t.tv_sec - t0.tv_sec) * 1e3 + (double)(t.tv_nsec - t0.tv_nsec) / 1e6;
}

int main(void)
{
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gpu_dev_t *g = gpu_create(1 << 16, 100);       /* 64 KB VRAM, 100 ms watchdog */
    unsigned upload, compute, buggy;
    gpu_channel_open(g, &upload);
    gpu_channel_open(g, &compute);
    gpu_channel_open(g, &buggy);

    const char frame[] = "frame #42 pixels";
    gpu_vram_write(g, 0, frame, sizeof frame);
    gpu_pb_t pb;

    pb_reset(&pb);                                  /* compute: depends on upload */
    pb_sem_acquire(&pb, 0, 1);
    pb_copy(&pb, 0x8000, 0x1000, sizeof frame);
    pb_sem_release(&pb, 1, 1);
    gpu_submit(g, compute, &pb);
    printf("[%6.1f ms] compute submitted (waits on sem0)\n", ms());
    pause_ms(5);

    pb_reset(&pb);                                  /* buggy: hangs the engine */
    pb_hang(&pb);
    gpu_submit(g, buggy, &pb);
    printf("[%6.1f ms] buggy   submitted (will hang)\n", ms());
    pause_ms(5);

    pb_reset(&pb);                                  /* upload */
    pb_copy(&pb, 0x1000, 0, sizeof frame);
    pb_sem_release(&pb, 0, 1);
    gpu_submit(g, upload, &pb);
    printf("[%6.1f ms] upload  submitted (engine is stuck on buggy)\n", ms());

    gpu_sem_wait(g, 1, 1, 2000);
    char out[sizeof frame];
    gpu_vram_read(g, 0x8000, out, sizeof out);
    printf("[%6.1f ms] compute finished -> \"%s\"\n", ms(), out);

    gpu_channel_idle_wait(g, buggy, 2000);
    printf("[%6.1f ms] buggy   channel status: %s\n", ms(),
           gpu_ch_status_str(gpu_channel_status(g, buggy)));

    gpu_channel_recover(g, buggy);
    pb_reset(&pb);
    pb_sem_release(&pb, 2, 1);
    gpu_submit(g, buggy, &pb);
    gpu_sem_wait(g, 2, 1, 1000);
    printf("[%6.1f ms] buggy   recovered and ran new work: %s\n", ms(),
           gpu_ch_status_str(gpu_channel_status(g, buggy)));

    gpu_stats_t s;
    gpu_get_stats(g, &s);
    printf("\nstats: entries=%llu methods=%llu ctx_switches=%llu faults=%llu watchdog_resets=%llu\n",
           (unsigned long long)s.entries_done, (unsigned long long)s.methods_done,
           (unsigned long long)s.ctx_switches, (unsigned long long)s.faults,
           (unsigned long long)s.watchdog_resets);
    gpu_destroy(g);
    return 0;
}
