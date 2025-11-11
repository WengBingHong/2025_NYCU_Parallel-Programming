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

    if (world_rank > 0)
    {
        // TODO: MPI workers
        // Worker process：發送 local_number_in_circle 給 rank 0
        MPI_Send(&local_number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

        // 需要 world_size - 1 個空間來存放來自 worker (1...N-1) 的結果
        long long int *received_hits = (long long int *)malloc(sizeof(long long int) * (world_size - 1));
        MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * (world_size - 1));

        // 對所有 worker 呼叫 Irecv (non-blocking receive)
        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(&received_hits[i - 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        // 等待所有 Irecv "請求" 都完成
        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

        // 累加結果
        long long int total_number_in_circle = local_number_in_circle; // 加上 rank 0 自己的
        for (int i = 0; i < world_size - 1; i++) {
            total_number_in_circle += received_hits[i];
        }

        // TODO: PI result
        pi_result = 4.0 * (double)total_number_in_circle / (double)tosses;

        free(received_hits);
        free(requests);
    }

    if (world_rank == 0)
    {
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}