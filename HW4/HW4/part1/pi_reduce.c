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

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Work split
    long long int base = (world_size > 0) ? (tosses / world_size) : 0;
    long long int rem  = (world_size > 0) ? (tosses % world_size) : 0;
    long long int local_tosses = base + (world_rank < rem ? 1 : 0);

    // RNG per-rank
    const unsigned SEED = 86413u;
    srand(SEED * (unsigned)(world_rank + 1));

    // Local Monte Carlo
    long long int local_count = 0;
    for (long long int i = 0; i < local_tosses; ++i) {
        double x = (double)rand() / (double)RAND_MAX;
        double y = (double)rand() / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) ++local_count;
    }

    // Use MPI_Reduce with MPI_SUM to rank 0. Root must not reuse the same buffer as recv.
    long long int reduced_sum = 0;
    MPI_Reduce(&local_count, &reduced_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        pi_result = (tosses > 0) ? (4.0 * (double)reduced_sum / (double)tosses) : 0.0;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}