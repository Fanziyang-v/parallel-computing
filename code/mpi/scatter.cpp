#include <iostream>
#include <fstream>
#include "mpi.h"

using namespace std;

void initialize(double *x, int n)
{
    srand(10213u);
    for (int i = 0; i < n; i++)
        x[i] = (double)rand() / RAND_MAX;
}

void print_vec(double *x, int n)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ofstream file("P" + to_string(rank) + ".txt");
    file << "Process " << rank << " has received data from master process." << endl;
    for (int i = 0; i < n; i++)
        file << x[i] << " ";
    file << endl;
}

int main(int argc, char *argv[])
{
    int n, nums;
    double *x, *y;
    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        cout << "Enter n: " << endl;
        cin >> n;
        x = new double[n];
        initialize(x, n);
        for (int i = 0; i < n; i++)
            cout << x[i] << " ";
        cout << endl;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    nums = n / numprocs;
    y = new double[nums];
    MPI_Scatter(x, nums, MPI_DOUBLE, y, nums, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    print_vec(y, nums);
    MPI_Finalize();
    return 0;
}