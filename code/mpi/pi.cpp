#include <iostream>
#include <iomanip>
#include "mpi.h"
#include <cmath>

const double PI = 3.141592653589793;
double func(double x)
{
    return 4.0 / (x * x + 1);
}

int main(int argc, char *argv[])
{
    int i, n, pid, numprocs;
    double pi, h, sum, x;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    if (pid == 0)
    {
        /* Master code */
        std::cout << "Enter n" << std::endl;
        std::cin >> n;
    }

    /* broadcast n */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    h = 1.0 / n;
    sum = 0.0;
    for (i = pid; i < n; i += numprocs)
    {
        x = i * h;
        sum += func(x);
    }
    sum *= h;
    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (pid == 0)
        std::cout << "PI is approximately " << std::setprecision(16) << pi << ", Error is " << fabs(pi - PI) << std::endl;

    MPI_Finalize();
    return 0;
}