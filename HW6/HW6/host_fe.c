#include "host_fe.h"
#include "helper.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>

#define TILE_W 16
#define TILE_H 16

void host_fe(
    int filter_width,
    float *filter,
    int image_height,
    int image_width,
    float *input_image,
    float *output_image,
    cl_device_id *device,
    cl_context *context,
    cl_program *program)
{
    cl_int err;

    /* --- 靜態資源，在多次調用中可重複使用 --- */
    static cl_command_queue queue = NULL;
    static cl_kernel kernel = NULL;
    static cl_mem buf_img = NULL, buf_out = NULL, buf_flt = NULL;

    static int oldW = 0, oldH = 0, oldFw = 0;
    static float *oldImg = NULL, *oldFlt = NULL;

    size_t img_bytes = sizeof(float) * image_width * image_height;
    size_t flt_bytes = sizeof(float) * filter_width * filter_width;

    /* 建立 queue（只建立一次） */
    if (queue == NULL) {
        queue = clCreateCommandQueue(*context, *device, 0, &err);
        CHECK(err, "clCreateCommandQueue");
    }

    /* 建立 kernel（只建立一次） */
    if (kernel == NULL) {
        kernel = clCreateKernel(*program, "convolution", &err);
        CHECK(err, "clCreateKernel");
    }

    /* --- 建立/更新 buffer：影像大小或濾波器變動時才重建 --- */

    if (buf_img == NULL || image_width != oldW || image_height != oldH) {
        if (buf_img) { clReleaseMemObject(buf_img); clReleaseMemObject(buf_out); }
        buf_img = clCreateBuffer(*context, CL_MEM_READ_ONLY, img_bytes, NULL, &err);
        CHECK(err, "clCreateBuffer input_image");

        buf_out = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, img_bytes, NULL, &err);
        CHECK(err, "clCreateBuffer output_image");

        oldW = image_width; oldH = image_height;
        oldImg = NULL;  /* 強制重新複製 */
    }

    if (buf_flt == NULL || oldFw != filter_width) {
        if (buf_flt) clReleaseMemObject(buf_flt);

        buf_flt = clCreateBuffer(*context, CL_MEM_READ_ONLY, flt_bytes, NULL, &err);
        CHECK(err, "clCreateBuffer filter");

        oldFw = filter_width;
        oldFlt = NULL;
    }

    /* --- Host to Device 複製（如果資料指標不同才複製） --- */

    if (oldImg != input_image) {
        CHECK(clEnqueueWriteBuffer(queue, buf_img, CL_TRUE, 0, img_bytes, input_image, 0, NULL, NULL),
              "write input_image");
        oldImg = input_image;
    }

    if (oldFlt != filter) {
        CHECK(clEnqueueWriteBuffer(queue, buf_flt, CL_TRUE, 0, flt_bytes, filter, 0, NULL, NULL),
              "write filter");
        oldFlt = filter;
    }

    /* --- 設定 kernel 參數 --- */

    CHECK(clSetKernelArg(kernel, 0, sizeof(int), &filter_width), "arg0");
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_flt), "arg1");
    CHECK(clSetKernelArg(kernel, 2, sizeof(int), &image_height), "arg2");
    CHECK(clSetKernelArg(kernel, 3, sizeof(int), &image_width), "arg3");
    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_img), "arg4");
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &buf_out), "arg5");

    /* --- 工作區域 --- */

    size_t local[2]  = {TILE_W, TILE_H};
    size_t global[2] = {
        (size_t)((image_width + TILE_W - 1) / TILE_W) * TILE_W,
        (size_t)((image_height + TILE_H - 1) / TILE_H) * TILE_H
    };

    /* --- 執行 Kernel --- */

    CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL),
          "NDRange");

    CHECK(clFinish(queue), "clFinish");

    /* --- Device → Host --- */

    CHECK(clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, img_bytes, output_image, 0, NULL, NULL),
          "read output_image");
}
