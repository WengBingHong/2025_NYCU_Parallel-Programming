#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int local_tosses = tosses / world_size;
    if (world_rank == 0) {
        local_tosses += (tosses % world_size);
    }

    unsigned int seed = (unsigned int)(time(NULL) + world_rank * getpid()); 
    
    long long int local_number_in_circle = 0;
    for (long long int i = 0; i < local_tosses; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_number_in_circle++;
        }
    }

    // TODO: use MPI_Gather
    long long int *all_hits = NULL;
    if (world_rank == 0) {
        // 只有 rank 0 需要分配接收陣列 (大小為 world_size)
        all_hits = (long long int *)malloc(sizeof(long long int) * world_size);
    }

    // 所有 process 都呼叫 MPI_Gather
    // 每個 process 都發送 1 個 local_number_in_circle
    // rank 0 會將收到的 world_size 個數字存放到 all_hits 陣列中
    MPI_Gather(&local_number_in_circle, // send_data
               1,                       // send_count
               MPI_LONG_LONG,           // send_type
               all_hits,                // recv_data (只有 rank 0 有用)
               1,                       // recv_count (每個 process 接收 1 個)
               MPI_LONG_LONG,           // recv_type
               0,                       // root
               MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        // Rank 0 手動加總 all_hits 陣列
        long long int total_number_in_circle = 0;
        for (int i = 0; i < world_size; i++) {
            total_number_in_circle += all_hits[i];
        }
        
        pi_result = 4.0 * (double)total_number_in_circle / (double)tosses;

        free(all_hits);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}