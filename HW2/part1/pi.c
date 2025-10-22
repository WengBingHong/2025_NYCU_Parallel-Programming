#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <immintrin.h> 
#include <math.h>

typedef struct {
    long long num_tosses_per_thread;
    unsigned int seed;
} ThreadData;

void* monte_carlo_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    long long local_hits = 0;

    __m256i seed = _mm256_set_epi32(
        data->seed, data->seed + 1, data->seed + 2, data->seed + 3,
        data->seed + 4, data->seed + 5, data->seed + 6, data->seed + 7
    );
    __m256i a = _mm256_set1_epi32(1664525);
    __m256i c = _mm256_set1_epi32(1013904223);

    __m256 const_1_0f = _mm256_set1_ps(1.0f);
    __m256 max_int_inv = _mm256_set1_ps(1.0f / 0x7FFFFFFF);

    long long i = 0;
for (; i + 16 <= data->num_tosses_per_thread; i += 16) {
    // 第 1 組 8 個樣本
    seed = _mm256_add_epi32(_mm256_mullo_epi32(seed, a), c);
    __m256i rand_int_x1 = _mm256_and_si256(seed, _mm256_set1_epi32(0x7FFFFFFF));
    __m256 x1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(rand_int_x1), max_int_inv), const_1_0f);
    seed = _mm256_add_epi32(_mm256_mullo_epi32(seed, a), c);
    __m256i rand_int_y1 = _mm256_and_si256(seed, _mm256_set1_epi32(0x7FFFFFFF));
    __m256 y1 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(rand_int_y1), max_int_inv), const_1_0f);

    __m256 dist_sq1 = _mm256_add_ps(_mm256_mul_ps(x1, x1), _mm256_mul_ps(y1, y1));
    __m256 cmp_mask1 = _mm256_cmp_ps(dist_sq1, const_1_0f, _CMP_LE_OQ);
    local_hits += __builtin_popcount(_mm256_movemask_ps(cmp_mask1));

    // 第 2 組 8 個樣本
    seed = _mm256_add_epi32(_mm256_mullo_epi32(seed, a), c);
    __m256i rand_int_x2 = _mm256_and_si256(seed, _mm256_set1_epi32(0x7FFFFFFF));
    __m256 x2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(rand_int_x2), max_int_inv), const_1_0f);
    seed = _mm256_add_epi32(_mm256_mullo_epi32(seed, a), c);
    __m256i rand_int_y2 = _mm256_and_si256(seed, _mm256_set1_epi32(0x7FFFFFFF));
    __m256 y2 = _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(rand_int_y2), max_int_inv), const_1_0f);

    __m256 dist_sq2 = _mm256_add_ps(_mm256_mul_ps(x2, x2), _mm256_mul_ps(y2, y2));
    __m256 cmp_mask2 = _mm256_cmp_ps(dist_sq2, const_1_0f, _CMP_LE_OQ);
    local_hits += __builtin_popcount(_mm256_movemask_ps(cmp_mask2));
}

    unsigned int scalar_seed = data->seed;
    for (; i < data->num_tosses_per_thread; ++i) {
        double x_s = (double)rand_r(&scalar_seed) / RAND_MAX * 2.0 - 1.0;
        double y_s = (double)rand_r(&scalar_seed) / RAND_MAX * 2.0 - 1.0;
        if (x_s * x_s + y_s * y_s <= 1.0)
            local_hits++;
    }

    return (void*)local_hits;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads> <num_tosses>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long long total_tosses = atoll(argv[2]);

    if (num_threads <= 0 || total_tosses <= 0) {
        fprintf(stderr, "Error: arguments must be positive.\n");
        return 1;
    }

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));

    long long tosses_per_thread = total_tosses / num_threads;
    long long remainder = total_tosses % num_threads;

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].num_tosses_per_thread = tosses_per_thread + (i == 0 ? remainder : 0);
        thread_data[i].seed = i * 1000 + 1234;
        pthread_create(&threads[i], NULL, monte_carlo_worker, &thread_data[i]);
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        void* status;
        pthread_join(threads[i], &status);
        total_hits += (long long)status;
    }

    double pi_estimate = 4.0 * total_hits / total_tosses;
    printf("%.10f\n", pi_estimate);

    free(threads);
    free(thread_data);
    return 0;
}

