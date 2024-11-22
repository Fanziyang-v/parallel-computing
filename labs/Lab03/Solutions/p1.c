#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void mat_mul(double *M, double *N, double *P, int m, int n, int p)
{
    double sum;
    int i, j, k;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum = 0;
            for (k = 0; k < p; k++)
                sum += M[i * n + k] * N[k * p + j];
            P[i * p + j] = sum;
        }
    }
}

void input(const char *filename, int *m, int *n, int *p)
{
    FILE *fp;
    if ((fp = fopen(filename, "r")) == NULL)
    {
        printf("Unable to open %s for reading.\n", filename);
        exit(0);
    }
    fscanf(fp, "%d,%d,%d", m, n, p);
}

void output(const char *filename, double ts, double tp)
{
    FILE *fp;
    if ((fp = fopen(filename, "w")) == NULL)
    {
        printf("Unable to open %s for writing.\n", filename);
        exit(0);
    }
    fprintf(fp, "%.3lf,%.3lf\n", ts, tp);
}

void initialize(double *M, double *N, int m, int n, int p)
{
    int i, j;
    srand(10213u);
    // Initialize Matrix M
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            M[i * n + j] = (double)rand() / RAND_MAX;
    // Initialize Matrix N
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            N[i * p + j] = (double)rand() / RAND_MAX;
}

// sanity check
int check(double *A, double *B, int m, int n)
{
    int i, j;
    double epsilon = 1.e-3;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            if (fabs(A[i * n + j] - B[i * n + j]) > epsilon)
                return 0;
    }
    return 1;
}

void print_matrix(double *M, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            printf("%lf ", M[i * n + j]);
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int i, j, srow, srow_last;
    int m, n, p;
    double *M, *N, *P, *Q;
    int numprocs, pid;
    MPI_Request *requests;
    double start, end, elapsed_time;
    double tp, ts;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    printf("I am proc %d\n", pid);

    if (pid == 0)
    {
        // Read m, n, p from input.txt file.
        input("input1.txt", &m, &n, &p);
        // allocate memory and initialize with random floating-point values.
        M = (double *)malloc(m * n * sizeof(double));
        N = (double *)malloc(n * p * sizeof(double));
        P = (double *)malloc(m * p * sizeof(double));
        Q = (double *)malloc(m * p * sizeof(double));
        initialize(M, N, m, n, p);
        requests = (MPI_Request *)malloc((numprocs - 1) * sizeof(MPI_Request));
    }
    if (numprocs == 1)
    {
        P = (double *)malloc(m * p * sizeof(double));
        mat_mul(M, N, P, m, n, p);
        // print_matrix(P, m, p);
    }
    else
    {
        start = MPI_Wtime();
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("process %d gets m = %d, n = %d, p=%d\n", pid, m, n, p);

        srow = m / numprocs;
        srow_last = m - srow * (numprocs - 1);
        if (pid == numprocs - 1)
            srow = srow_last;
        if (pid == 0)
        {
            // master code.
            // Broadcast matrix n
            MPI_Bcast(N, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // Send sub-matrices to other processes.
            for (i = 1; i < numprocs - 1; i++)
                MPI_Isend(M + i * srow * n, srow * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
            MPI_Isend(M + i * srow * n, srow_last * n, MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD, &requests[i - 1]);

            // perform its own calculation
            mat_mul(M, N, P, srow, n, p);
            MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);

            // Collect results from other processes
            for (i = 1; i < numprocs - 1; i++)
                MPI_Irecv(P + i * srow * p, srow * p, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
            MPI_Irecv(P + i * srow * p, srow_last * p, MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD, &requests[i - 1]);
            MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);

            end = MPI_Wtime();
            tp = (end - start) * 1000.0;

            start = MPI_Wtime();
            // perform matrix multiplication in one process.
            mat_mul(M, N, Q, m, n, p);
            end = MPI_Wtime();
            ts = (end - start) * 1000.0;
            printf("Parallel matrix multiplication time: %.3fms\n", tp);
            printf("Serial matrix multiplication time: %.3fms\n", ts);
            printf("Speedup: %.3f\n", ts / tp);
            // sanity check.
            if (check(P, Q, m, p))
                printf("Sanity Check passes!\n");
            else
                printf("Sanity Check fails\n");
            output("output1.txt", ts, tp);
        }
        else
        {
            // slave code.
            // allocate memory for sub-matrix M, and matrix N.
            M = (double *)malloc(srow * n * sizeof(double));
            N = (double *)malloc(n * p * sizeof(double));
            P = (double *)malloc(srow * p * sizeof(double));

            // Broadcast matrix n
            MPI_Bcast(N, n * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // receive sub-matrix M and matrix N elements from master process
            MPI_Recv(M, srow * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // perform its own calculation
            mat_mul(M, N, P, srow, n, p);
            // send the results to master process
            MPI_Send(P, srow * p, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    free(M);
    free(N);
    free(P);
    MPI_Finalize();
    return 0;
}
