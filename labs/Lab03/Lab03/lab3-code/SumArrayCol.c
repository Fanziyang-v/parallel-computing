// Program that computes the sum of an array of elements in parallel using
// MPI_Reduce.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv)
{


    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int pid, np, n = sizeof(a) / sizeof(int),
                 elements_per_process,
                 local_sum = 0;
    // np -> no. of processes
    // pid -> process id

    // Creation of parallel processes
    MPI_Init(&argc, &argv);

    // find out process ID,
    // and how many processes were started
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    elements_per_process = ceil(n * 1.0 / np);
    int i, init = elements_per_process * pid;
    for (i = init; i < init + elements_per_process && i < n; i++)
        local_sum += a[i];

    // Print the random numbers on each process
    printf("Local sum for process %d - %d\n",
            pid, local_sum);

    // Reduce all of the local sums into the global sum
    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0,
                MPI_COMM_WORLD);

    // Print the result
    if (pid == 0)
    {
        printf("Total sum = %d\n", global_sum);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}