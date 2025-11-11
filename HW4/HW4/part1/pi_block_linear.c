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

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long int local_tosses = tosses / world_size;
    long long int total_number_in_circle = 0;
    
    // 讓 rank 0 處理餘數，確保總和為 tosses
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
        // TODO: handle workers
        // Worker process：發送 local_number_in_circle 給 rank 0
        MPI_Send(&local_number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: main
        // Rank 0：接收來自所有 worker process 的結果
        total_number_in_circle = local_number_in_circle;
        long long int received_number;
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&received_number, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_number_in_circle += received_number;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = 4.0 * (double)total_number_in_circle / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}