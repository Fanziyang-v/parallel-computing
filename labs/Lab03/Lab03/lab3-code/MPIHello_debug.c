#include<stdio.h>
#include<mpi.h>
#include<unistd.h>
#include<netdb.h>
int main (int argc, char *argv[])
{
    int rank;
    int number_of_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD , &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD , &rank);
    
    volatile int debug = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == debug)
        sleep(5);
    
    printf("hello from process %d of %d\n", rank, number_of_processes);
    MPI_Finalize();
    return 0;
}
