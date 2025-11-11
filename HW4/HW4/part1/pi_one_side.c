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

    MPI_Win win;

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

    long long int total_hits = 0;

    // 建立一個 Window
    // 只有 rank 0 實際 "開放" 自己的 total_hits 記憶體
    // 其他 process 則建立一個大小為 0 的 window
    MPI_Win_create(&total_hits,                        // base address
                   (world_rank == 0) ? sizeof(long long int) : 0, // size
                   sizeof(long long int),              // displacement unit
                   MPI_INFO_NULL,                      // info
                   MPI_COMM_WORLD,                     // communicator
                   &win);                              // output window

    
    // MPI_Win_fence 是集體操作，用於同步
    // 參數 0 表示 "no preceding synchronization"
    MPI_Win_fence(0, win);

    // 所有 process (包括 rank 0) 都將自己的 local_number_in_circle
    // "累加" (MPI_SUM) 到 rank 0 (target_rank 0) 的 window (位移 0) 上
    MPI_Accumulate(&local_number_in_circle, // origin_addr
                   1,                       // origin_count
                   MPI_LONG_LONG,           // origin_datatype
                   0,                       // target_rank (root)
                   0,                       // target_disp (offset)
                   1,                       // target_count
                   MPI_LONG_LONG,           // target_datatype
                   MPI_SUM,                 // operation
                   win);

    // 第二個 fence，確保所有的 Accumulate 操作都已完成
    MPI_Win_fence(0, win);

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        // 此時 rank 0 上的 total_hits 已經是總和
        pi_result = 4.0 * (double)total_hits / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}