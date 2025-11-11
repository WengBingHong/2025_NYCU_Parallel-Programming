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

    // TODO: binary tree redunction
    long long int total_hits = local_number_in_circle;
    
    // d 是每一輪的通訊 "距離"
    for (int d = 1; d < world_size; d *= 2) {
        // rank 是 2*d 的倍數，代表他是接收者
        if (world_rank % (2 * d) == 0) {
            // 檢查發送者 (world_rank + d) 是否存在
            if (world_rank + d < world_size) {
                long long int received_hits;
                MPI_Recv(&received_hits, 1, MPI_LONG_LONG, world_rank + d, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_hits += received_hits;
            }
        } 
        // rank 是 d 的倍數，但不是 2*d 的倍數，代表他是發送者
        else if (world_rank % d == 0) {
            MPI_Send(&total_hits, 1, MPI_LONG_LONG, world_rank - d, 0, MPI_COMM_WORLD);
            break; // 發送完畢，此 process 的歸約任務結束
        }
    }


    if (world_rank == 0)
    {
        // TODO: PI result
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