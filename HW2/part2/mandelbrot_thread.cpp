#include <stdio.h>
#include <thread>
#include <cmath>
#include "cycle_timer.h"

typedef int v4si __attribute__ ((vector_size (16)));
typedef union {
    v4si v;
    int e[4];
    __int128 bits;
} ve4si;

typedef float v4sf __attribute__ ((vector_size (16)));
typedef union {
    v4sf v;
    float e[4];
} ve4sf;

typedef struct {
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int numThreads;
    int threadId;
    double runtime;
} WorkerArgs;

static inline void vmandel(ve4sf *vc_re, ve4sf *vc_im, int count, ve4si *vout)
{
    ve4sf vz_re, vz_im;
    vz_re.v = vc_re->v;
    vz_im.v = vc_im->v;

    ve4si v0 = {0, 0, 0, 0};
    ve4si v1 = {1, 1, 1, 1};
    ve4sf v2f = {2, 2, 2, 2};
    ve4sf v4f = {4, 4, 4, 4};
    ve4si vmask = {0, 0, 0, 0};
    vout->v = v0.v;

    for (int i = 0; i < count; ++i) {
        ve4si vcmpmask;
        ve4sf vnew_re;
        ve4sf vnew_im;

        vcmpmask.v = (vz_re.v * vz_re.v + vz_im.v * vz_im.v) > v4f.v;
        vmask.v = vmask.v | vcmpmask.v;

        if (!(~vmask.bits)) break; // 四個都逃逸時提前結束

        vnew_re.v = vz_re.v * vz_re.v - vz_im.v * vz_im.v;
        vnew_im.v = v2f.v * vz_re.v * vz_im.v;

        vz_re.v = vc_re->v + vnew_re.v;
        vz_im.v = vc_im->v + vnew_im.v;

        vout->v = vout->v + (v1.v & (!vmask.v));
    }
}

// Padding 版本: 全 SIMD、無 scalar fallback
static void mandelbrotSerialSIMD(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int numThreads, int threadId,
    int maxIterations,
    int output[])
{
    const int simdWidth = 4;
    const int paddedWidth = ((width + simdWidth - 1) / simdWidth) * simdWidth; // 向上補齊到4的倍數

    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    ve4sf vy0 = {y0, y0, y0, y0};
    ve4sf vx0 = {x0, x0, x0, x0};
    ve4sf vdy = {dy, dy, dy, dy};
    ve4sf vdx = {dx, dx, dx, dx};
    ve4sf v03 = {0, 1, 2, 3};
    ve4sf vwidth = {(float)paddedWidth, (float)paddedWidth, (float)paddedWidth, (float)paddedWidth};

    for (int j = threadId; j < height; j += numThreads) {
        ve4sf vj = {(float)j, (float)j, (float)j, (float)j};
        ve4sf v_y0 = vy0;
        v_y0.v += vj.v * vdy.v;

        for (int i = 0; i < paddedWidth; i += simdWidth) {
            ve4sf vi = {(float)i, (float)i, (float)i, (float)i};
            ve4sf v_x = vx0;
            v_x.v += (vi.v + v03.v) * vdx.v;

            ve4si vret;
            vmandel(&v_x, &v_y0, maxIterations, &vret);

            for (int k = 0; k < simdWidth; ++k) {
                int idx = j * paddedWidth + i + k;
                if (i + k < width)  // 只輸出有效像素
                    output[j * width + i + k] = vret.e[k];
            }
        }
    }
}

void workerThreadStart(WorkerArgs *const args)
{
    double startTime = CycleTimer::current_seconds();

    mandelbrotSerialSIMD(
        args->x0,
        args->y0,
        args->x1,
        args->y1,
        args->width,
        args->height,
        args->numThreads,
        args->threadId,
        args->maxIterations,
        args->output);

    double endTime = CycleTimer::current_seconds();
    args->runtime = endTime - startTime;
}

void mandelbrot_thread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS) {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++) {
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].output = output;
        args[i].numThreads = numThreads;
        args[i].threadId = i;
    }

    for (int i = 1; i < numThreads; i++)
        workers[i] = std::thread(workerThreadStart, &args[i]);

    workerThreadStart(&args[0]);

    for (int i = 1; i < numThreads; i++)
        workers[i].join();

    printf("Finish Time Report:\n");
    for (int i = 0; i < numThreads; i++)
        printf("Thread %d: %f\n", i, args[i].runtime);
}

