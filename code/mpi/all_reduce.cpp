#include <iostream>
#include <cmath>
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
        cout << "Enter n: " << endl;
        cin >> n;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    v = new double[n];
    w = new double[n];
    for (int i = 0; i < n; i++)
        v[i] = i * rank;
    MPI_Allreduce(v, w, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == numprocs - 1)
    {
        cout << "Reduction result: " << endl;
        for (int i = 0; i < n; i++)
            cout << w[i] << " ";
        cout << endl;
    }
    MPI_Finalize();
    return 0;
}