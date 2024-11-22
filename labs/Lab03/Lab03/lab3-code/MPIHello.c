#include<stdio.h>
#include<mpi.h>
int main (int argc, char *argv[])
{
    int rank;
    int number_of_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    printf("hello from process %d of %d\n", rank, number_of_processes);
    MPI_Finalize();
    return 0;
}
