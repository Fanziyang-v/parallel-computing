#include <iostream>
#include <cmath>
#include "mpi.h"

using namespace std;

/* Perform matrix-vector multiplication. */
void mv_mul(double *M, double *x, double *y, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += M[i * n + j] * x[j];
        y[i] = sum;
    }
}

/* Initialize matrix and vector with random floating-point values. */
void initialize(double *M, double *x, int m, int n)
{
    int i, j;
    /* Setup random seed. */
    srand(10213u);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            M[i * n + j] = (double)rand() / RAND_MAX;
    for (j = 0; j < n; j++)
        x[j] = (double)rand() / RAND_MAX;
}

/* Print vector. */
void print_vector(double *v, int n)
{
    int i;
    for (i = 0; i < n; i++)
        cout << v[i] << " ";
    cout << endl;
}

/* Print matrix. */
void print_matrix(double *M, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            cout << M[i * n + j] << " ";
        cout << endl;
    }
}

/* Sanity check. */
bool check(double *u, double *v, int n)
{
    const double EPSILON = 1.e-12;
    for (int i = 0; i < n; i++)
        if (fabs(u[i] - v[i]) > EPSILON)
            return false;
    return true;
}

int main(int argc, char *argv[])
{
    int i, m, n;
    double *M, *x, *y, *z;
    double start, end, tp, ts;
    int rank, numprocs;
    int num_rows;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        /* Master code: read m, n from standard input stream */
        cout << "Enter m, n:" << endl;
        cin >> m >> n;
        /* Initialize matrix and vector with random floating-point values. */
        M = new double[m * n];
        x = new double[n];
        y = new double[m];
        z = new double[m];
        initialize(M, x, m, n);
        start = MPI_Wtime();
    }
    /* Broadcast m, n. */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    num_rows = m / numprocs;
    if (rank == numprocs - 1)
        num_rows = m - (numprocs - 1) * num_rows;

    if (rank == 0)
    {
        /* Master code. */
        MPI_Request *requests = new MPI_Request[numprocs - 1];
        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (i = 1; i < numprocs - 1; i++)
            MPI_Isend(M + i * num_rows * n, num_rows * n, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &requests[i - 1]);
        MPI_Isend(M + (numprocs - 1) * num_rows * n, (m - (numprocs - 1) * num_rows) * n, MPI_DOUBLE, numprocs - 1, numprocs - 1, MPI_COMM_WORLD, &requests[numprocs - 2]);
        /* Perform its own matrix-vector multiplication. */
        mv_mul(M, x, y, num_rows, n);
        MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);
        /* Collect results from other processes. */
        for (i = 1; i < numprocs - 1; i++)
            MPI_Irecv(y + i * num_rows, num_rows, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &requests[i - 1]);
        MPI_Irecv(y + (numprocs - 1) * num_rows, m - (numprocs - 1) * num_rows, MPI_DOUBLE, numprocs - 1, numprocs - 1, MPI_COMM_WORLD, &requests[numprocs - 2]);
        MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);
        end = MPI_Wtime();
        tp = (end - start) * 1000.0;
        /* Print parallel matrix-vector multiplication result. */
        cout << "Parallel matrix-vector multiplication result: " << endl;
        print_vector(y, n);
        cout << "Parallel matrix-vector multiplication time is " << tp << " ms" << endl;

        /* Perform serial matrix-vector multiplication. */
        start = MPI_Wtime();
        mv_mul(M, x, z, m, n);
        end = MPI_Wtime();
        ts = (end - start) * 1000.0;
        cout << "Serial result: " << endl;
        print_vector(z, n);
        if (check(y, z, n))
            cout << "Sanity check passes!" << endl;
        else
            cout << "Sanity check fails!" << endl;
        cout << "Serial matrix-vector multiplication time is " << ts << " ms" << endl;
        cout << "Speedup is " << ts / tp << endl;
        delete[] z;
        delete[] requests;
    }
    else
    {
        /* Slave code. */
        M = new double[num_rows * n];
        x = new double[n];
        y = new double[num_rows];
        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Recv(M, num_rows * n, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mv_mul(M, x, y, num_rows, n);
        MPI_Send(y, num_rows, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }
    delete[] M;
    delete[] x;
    delete[] y;
    MPI_Finalize();
    return 0;
}