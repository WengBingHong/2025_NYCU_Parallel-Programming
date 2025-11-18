#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "kernel.h"

namespace {

// Extract the even bits of a 32-bit integer and compact them (inverse of part1by1)
__device__ __forceinline__
unsigned int compact1by1(unsigned int x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

// Decode a 2D Morton(Z-order) code into (x, y)
__device__ __forceinline__
void morton_decode_2d(unsigned int code, unsigned int& x, unsigned int& y) {
    x = compact1by1(code);
    y = compact1by1(code >> 1);
}

// Determine whether the point is inside the main cardioid or the period-2 bulb
// If so, it is guaranteed to be non-escaping (belongs to the Mandelbrot set)
__device__ __forceinline__
bool in_main_cardioid_or_period2(float x, float y) {
    // Period-2 bulb: (x + 1)^2 + y^2 < 1/16
    float xp1 = x + 1.0f;
    if (xp1 * xp1 + y * y < 0.0625f) {
        return true;
    }

    // Main cardioid test
    // q(q + x') < 1/4 * y^2 where x' = x - 1/4
    float x_shift = x - 0.25f;
    float q = x_shift * x_shift + y * y;
    if (q * (q + x_shift) < 0.25f * y * y) {
        return true;
    }

    return false;
}

// Compute Mandelbrot iteration count for a given complex coordinate c = (c_re, c_im)
// Returns the number of iterations until divergence, or max_iterations if bounded
__device__ __forceinline__
int mandel_iterations(float c_re, float c_im, int max_iterations) {
    // Early exit: skip iteration loop if inside non-escaping regions
    if (in_main_cardioid_or_period2(c_re, c_im)) {
        return max_iterations;
    }

    float z_re = c_re;
    float z_im = c_im;

    int i = 0;

    #pragma unroll 32
    for (i = 0; i < max_iterations; ++i) {
        float z_re2 = z_re * z_re;
        float z_im2 = z_im * z_im;

        // Escape condition: magnitude^2 > 4
        if (z_re2 + z_im2 > 4.0f) {
            break;
        }

        // z_{n+1} = z_n^2 + c
        float new_re = z_re2 - z_im2 + c_re;
        float new_im = 2.0f * z_re * z_im + c_im;
        z_re = new_re;
        z_im = new_im;
    }

    return i;
}

// Kernel using Morton (Z-curve) order + grid-stride loop to cover image pixels
__global__ void mandel_kernel_morton(float upper_x,
                                     float upper_y,
                                     float lower_x,
                                     float lower_y,
                                     int res_x,
                                     int res_y,
                                     float step_x,
                                     float step_y,
                                     int max_iterations,
                                     unsigned int morton_limit,
                                     int* __restrict__ output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Iterate over Morton index space using grid-stride
    for (unsigned int code = static_cast<unsigned int>(tid);
         code < morton_limit;
         code += static_cast<unsigned int>(total_threads)) {

        unsigned int px_u, py_u;
        morton_decode_2d(code, px_u, py_u);

        // Morton range represents a power-of-two square region.
        // Pixel coordinates outside the real resolution must be skipped.
        if (px_u >= static_cast<unsigned int>(res_x) ||
            py_u >= static_cast<unsigned int>(res_y)) {
            continue;
        }

        int px = static_cast<int>(px_u);
        int py = static_cast<int>(py_u);

        float x = lower_x + static_cast<float>(px) * step_x;
        float y = lower_y + static_cast<float>(py) * step_y;

        int idx = py * res_x + px;
        output[idx] = mandel_iterations(x, y, max_iterations);
    }
}

} // namespace

// CPU side front-end caller
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int* img,
             int res_x,
             int res_y,
             int max_iterations) {
    float step_x = (upper_x - lower_x) / static_cast<float>(res_x);
    float step_y = (upper_y - lower_y) / static_cast<float>(res_y);

    int num_pixels = res_x * res_y;

    int* device_buffer = nullptr;
    cudaMalloc(&device_buffer, num_pixels * sizeof(int));

    // Determine the Morton traversal's maximum encoded range:
    // Find the larger dimension and extend it to the next power of two
    unsigned int max_dim = static_cast<unsigned int>(
        (res_x > res_y) ? res_x : res_y);

    unsigned int bits = 0;
    while ((1u << bits) < max_dim) {
        ++bits;
    }

    // Morton range = (2^bits)^2 = 2^(2*bits)
    unsigned int morton_limit = 1u << (2 * bits);

    // Kernel launch configuration
    int block_size = 256;
    int grid_size = (num_pixels + block_size - 1) / block_size;
    int max_grid = 1024;
    if (grid_size > max_grid) {
        grid_size = max_grid;
    }

    mandel_kernel_morton<<<grid_size, block_size>>>(
        upper_x, upper_y,
        lower_x, lower_y,
        res_x, res_y,
        step_x, step_y,
        max_iterations,
        morton_limit,
        device_buffer);

    cudaMemcpy(img,
               device_buffer,
               num_pixels * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_buffer);
}
