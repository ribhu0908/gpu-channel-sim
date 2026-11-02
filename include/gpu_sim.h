/*
 * gpu_sim.h - A host-side model of how a GPU consumes work from channels.
 *
 * Concepts mirror the publicly documented NVIDIA / nouveau channel model:
 *
 *   pushbuffer   a stream of 32-bit "methods" (header word + arguments)
 *   GPFIFO       per-channel ring of pushbuffer segments; the host advances
 *                GP_PUT to submit, the engine advances GP_GET as it finishes
 *   semaphores   memory the engine RELEASEs (writes) when work completes and
 *                ACQUIREs (waits on) before dependent work may start - this is
 *                how fences and cross-channel dependencies work
 *   RC recovery  a channel that hangs or sends a bad method is faulted and
 *                torn down by a watchdog, without stopping other channels
 *
 * This is a simulator for learning and testing scheduling / recovery logic
 * in plain C with pthreads. It does not talk to real hardware.
 */
#ifndef GPU_SIM_H
#define GPU_SIM_H

#include <stddef.h>
#include <stdint.h>

#define GPU_MAX_CHANNELS    8
#define GPU_GPFIFO_ENTRIES  16      /* power of two */
#define GPU_PB_MAX_WORDS    256     /* max words per pushbuffer segment */
#define GPU_NUM_SEMAPHORES  64

/* Method header: [31:24] opcode, [23:0] argument count. */
enum gpu_op {
    GPU_OP_NOP         = 0x00,      /* args: none                        */
    GPU_OP_FILL        = 0x01,      /* args: dst, byte value, len        */
    GPU_OP_COPY        = 0x02,      /* args: dst, src, len               */
    GPU_OP_SEM_RELEASE = 0x03,      /* args: sem id, value               */
    GPU_OP_SEM_ACQUIRE = 0x04,      /* args: sem id, value (wait >=)     */
    GPU_OP_HANG        = 0xEE       /* args: none - test hook for a hang */
};

typedef enum {
    GPU_CH_OK = 0,
    GPU_CH_FAULT_INVALID_METHOD,    /* unknown opcode or wrong arg count  */
    GPU_CH_FAULT_BAD_ADDRESS,       /* access outside VRAM / bad sem id   */
    GPU_CH_FAULT_TIMEOUT            /* watchdog fired on this channel     */
} gpu_ch_status_t;

/* ---------------------------------------------------------- pushbuffer */
typedef struct {
    uint32_t w[GPU_PB_MAX_WORDS];
    uint32_t n;
} gpu_pb_t;

/* Builders return 0, or -1 if the segment is full (nothing appended). */
void pb_reset(gpu_pb_t *pb);
int  pb_nop(gpu_pb_t *pb);
int  pb_fill(gpu_pb_t *pb, uint32_t dst, uint8_t value, uint32_t len);
int  pb_copy(gpu_pb_t *pb, uint32_t dst, uint32_t src, uint32_t len);
int  pb_sem_release(gpu_pb_t *pb, uint32_t sem, uint32_t value);
int  pb_sem_acquire(gpu_pb_t *pb, uint32_t sem, uint32_t value);
int  pb_hang(gpu_pb_t *pb);
int  pb_raw(gpu_pb_t *pb, uint32_t word);     /* emit any word (tests) */

/* ------------------------------------------------------------- device */
typedef struct gpu_dev gpu_dev_t;

typedef struct {
    uint64_t entries_done;      /* GPFIFO entries completed               */
    uint64_t methods_done;      /* individual methods executed            */
    uint64_t ctx_switches;      /* channel yielded on an unmet ACQUIRE    */
    uint64_t faults;            /* channels torn down (any reason)        */
    uint64_t watchdog_resets;   /* of which: hang detected by watchdog    */
} gpu_stats_t;

/* Starts the engine and watchdog threads. NULL on failure. */
gpu_dev_t *gpu_create(size_t vram_bytes, uint32_t watchdog_ms);
void       gpu_destroy(gpu_dev_t *dev);

/* Returns 0 and the channel id, or -ENOSPC if all channels are in use. */
int gpu_channel_open(gpu_dev_t *dev, unsigned *out_ch);

/* Copy a pushbuffer segment into the channel's GPFIFO and ring the doorbell.
 * 0 on success, -EAGAIN if the GPFIFO is full (back-pressure),
 * -EIO if the channel is faulted, -EINVAL on bad arguments. */
int gpu_submit(gpu_dev_t *dev, unsigned ch, const gpu_pb_t *pb);

/* Wait until semaphore >= value (wrap-safe). 0 or -ETIMEDOUT. */
int      gpu_sem_wait(gpu_dev_t *dev, uint32_t sem, uint32_t value,
                      uint32_t timeout_ms);
uint32_t gpu_sem_read(gpu_dev_t *dev, uint32_t sem);

/* Wait until every submitted entry on ch has finished or been dropped.
 * 0 if idle and healthy, -EIO if faulted, -ETIMEDOUT. */
int gpu_channel_idle_wait(gpu_dev_t *dev, unsigned ch, uint32_t timeout_ms);

gpu_ch_status_t gpu_channel_status(gpu_dev_t *dev, unsigned ch);

/* Clear a fault so the channel can be reused (like re-creating it after RC). */
int gpu_channel_recover(gpu_dev_t *dev, unsigned ch);

/* Host access to VRAM. Order with the engine through submit / sem_wait:
 * write before submitting work that reads it, read after its fence. */
int gpu_vram_write(gpu_dev_t *dev, uint32_t off, const void *src, uint32_t len);
int gpu_vram_read(gpu_dev_t *dev, uint32_t off, void *dst, uint32_t len);

void gpu_get_stats(gpu_dev_t *dev, gpu_stats_t *out);

const char *gpu_ch_status_str(gpu_ch_status_t s);

#endif /* GPU_SIM_H */
