#include <iostream>
#include <cmath>
#include "mpi.h"

using namespace std;

/* Perform matrix multiplication. */
void mat_mul(double *M, double *N, double *P, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += M[i * n + k] * N[k * n + j];
            P[i * p + j] = sum;
        }
    }
}

/* Initialize matrix with random floating-point vlaues. */
void initialize(double *M, double *N, int m, int n, int p)
{
    /* Setup random seed. */
    srand(10213u);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            M[i * n + j] = (double)rand() / RAND_MAX;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            N[i * p + j] = (double)rand() / RAND_MAX;
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
bool check(double *P, double *Q, int m, int p)
{
    const double EPSILON = 1.e-12;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            if (fabs(P[i * p + j] - Q[i * p + j]) > EPSILON)
                return false;
    return true;
}

int main(int argc, char *argv[])
{
    int m, n, p;
    double *M, *N, *P, *Q;
    double start, end, tp, ts;
    int rank, numprocs;
    int num_rows;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        cout << "Enter m, n, p: " << endl;
        cin >> m >> n >> p;
        /* Initialize matrix M, N with random floating-point values. */
        M = new double[m * n];
        N = new double[n * p];
        P = new double[m * p];
        Q = new double[m * p];
        initialize(M, N, m, n, p);
        start = MPI_Wtime();
    }
    /* Broadcast m, n, p */
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);

    num_rows = m / numprocs;
    if (rank == numprocs - 1)
        num_rows = m - (numprocs - 1) * num_rows;

    if (rank == 0)
    {
        /* Master code. */
        MPI_Request *requests = new MPI_Request[numprocs - 1];
        /* Broadcast matrix N. */
        MPI_Bcast(N, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        /* Send sub matrix of M to other processes. */
        for (int i = 1; i < numprocs - 1; i++)
            MPI_Isend(M + i * num_rows * n, num_rows * n, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &requests[i - 1]);
        MPI_Isend(M + (numprocs - 1) * num_rows * n, (m - (numprocs - 1) * num_rows) * n, MPI_DOUBLE, numprocs - 1, numprocs - 1, MPI_COMM_WORLD, &requests[numprocs - 2]);
        /* Perform its own matrix multiplication. */
        mat_mul(M, N, P, num_rows, n, p);
        MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);
        /* Collect results from other processes. */
        for (int i = 1; i < numprocs - 1; i++)
            MPI_Irecv(P + i * num_rows * p, num_rows * p, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &requests[i - 1]);
        MPI_Irecv(P + (numprocs - 1) * num_rows * p, (m - (numprocs - 1) * num_rows) * p, MPI_DOUBLE, numprocs - 1, numprocs - 1, MPI_COMM_WORLD, &requests[numprocs - 2]);
        MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);
        end = MPI_Wtime();
        tp = (end - start) * 1000.0;
        cout << "Parallel matrix multiplication time is " << tp << " ms" << endl;

        /* Perform serial matrix multiplication. */
        start = MPI_Wtime();
        mat_mul(M, N, Q, m, n, p);
        end = MPI_Wtime();
        ts = (end - start) * 1000.0;
        cout << "Serial matrix multiplication time is " << ts << " ms" << endl;
        if (check(P, Q, m, p))
            cout << "Sanity check passes!" << endl;
        else
            cout << "Sanity check fails!" << endl;
        cout << "Speedup is " << ts / tp << endl;
        delete[] Q;
        delete[] requests;
    }
    else
    {
        /* Slave code. */
        M = new double[num_rows * n];
        N = new double[n * p];
        P = new double[num_rows * p];
        MPI_Bcast(N, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Recv(M, num_rows * n, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Perform its own matrix multiplication. */
        mat_mul(M, N, P, num_rows, n, p);
        /* Send result to master process. */
        MPI_Send(P, num_rows * p, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }
    delete[] M;
    delete[] N;
    delete[] P;

    MPI_Finalize();
    return 0;
}