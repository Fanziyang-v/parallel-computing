#include <iostream>
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[])
{
    int n;
    double *v, *w;
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (rank == 0)
    {
        /* Master code: read n from standard input stream. */
        cout << "Enter number of elmenets per process: " << endl;
        cin >> n;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    v = new double[n];
    w = new double[n * numprocs];
    for (int i = 0; i < n; i++)
        v[i] = rank;
    MPI_Allgather(v, n, MPI_DOUBLE, w, n, MPI_DOUBLE, MPI_COMM_WORLD);
    if (rank == numprocs - 1)
    {
        cout << "Gathering result: " << endl;
        for (int i = 0; i < n * numprocs; i++)
            cout << w[i] << " ";
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}