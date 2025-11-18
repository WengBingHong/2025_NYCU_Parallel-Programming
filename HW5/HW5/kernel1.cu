#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "kernel.h"

namespace {

__device__ int mandel_device(float c_re, float c_im, int max_iterations) {
    float z_re = c_re;
    float z_im = c_im;
    int i;
    for (i = 0; i < max_iterations; ++i) {
        float z_re2 = z_re * z_re;
        float z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.0f) {
            break;
        }

        float new_re = z_re2 - z_im2;
        float new_im = 2.0f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandel_kernel(float upper_x,
                              float upper_y,
                              float lower_x,
                              float lower_y,
                              int res_x,
                              int res_y,
                              float step_x,
                              float step_y,
                              int max_iterations,
                              int* output) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= res_x || py >= res_y) {
        return;
    }

    float x = lower_x + static_cast<float>(px) * step_x;
    float y = lower_y + static_cast<float>(py) * step_y;

    int index = py * res_x + px;
    output[index] = mandel_device(x, y, max_iterations);
}

} // namespace

// Method 1: new + cudaMalloc, one pixel per thread
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

    // host buffer: new (pageable)
    int* host_buffer = new int[num_pixels];

    // device buffer: linear
    int* device_buffer = nullptr;
    cudaMalloc(&device_buffer, num_pixels * sizeof(int));

    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (res_x + block_dim.x - 1) / block_dim.x,
        (res_y + block_dim.y - 1) / block_dim.y);

    mandel_kernel<<<grid_dim, block_dim>>>(
        upper_x, upper_y,
        lower_x, lower_y,
        res_x, res_y,
        step_x, step_y,
        max_iterations,
        device_buffer);

    cudaDeviceSynchronize();

    cudaMemcpy(host_buffer, device_buffer,
               num_pixels * sizeof(int),
               cudaMemcpyDeviceToHost);

    // copy result from our own host buffer back to img
    for (int i = 0; i < num_pixels; ++i) {
        img[i] = host_buffer[i];
    }

    cudaFree(device_buffer);
    delete[] host_buffer;
}
