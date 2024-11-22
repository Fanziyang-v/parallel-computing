#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
/*  This function calculates y = Ax, where
     A is a (row x col) matrix;
     x is a (col x 1) vector;
     y is a (row x 1) vector. 
*/
void mat_vec(double *A, double *x, double *y, int row, int col)
{
    double sum;
    int i, j;
    for (i = 0; i < row; i++)
    {
        sum = 0.0;
        for (j = 0; j < col; j++)
            sum += A[i * col + j] * x[j];
        y[i] = sum;
    }
}

double* read_mat_from_file(const char *s, int *n_row, int *n_col)
{
    FILE *fp; 
    if ((fp = fopen(s, "r")) == NULL)
    {
        printf("Unable to open %s for reading.\n", s);
        exit(0);
    }
    fscanf(fp, "%d%d", n_row, n_col);
    int m = *n_row, n = *n_col;
    double *in_matrix = (double *)malloc(m * n * sizeof(double));
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            fscanf(fp, "%lf", &in_matrix[i * n + j]);
    fclose(fp);
    // printf("%lf\n", in_matrix[0]);
    return in_matrix;
} 

int main(int argc, char **argv)
{
    int m = 0, n = 0, myid, numprocs, srow = 0, i, p, srow_last = 0;
    double *A, *x, *y;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    printf("I am proc %d\n", myid);
    // Read Matrix and Vector
    if (myid == 0)
    {
        A = read_mat_from_file("matrix.txt", &m, &n);
        x = read_mat_from_file("vector.txt", &n, &p);
    }
    if (numprocs == 1) {
        y = (double *)malloc(m * sizeof(double));
        mat_vec(A, x, y, m, n);
        for (i = 0; i < n; i++)
            printf("%lf ", y[i]);
        printf("\n");
    }
    else{
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("process %d gets n = %d, m = %d\n", myid, n, m);
        if (myid != 0)
            x = (double *)malloc(n * sizeof(double));

        /* broadcast vector x */
        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        srow = m / numprocs;
        srow_last = m - srow * (numprocs - 1);
        if (myid == numprocs - 1)
            srow = srow_last;
        printf("process %d calculates %d rows\n", myid, srow);
        if (myid == 0)
        {
            /* master code */
            /* allocate memory for matrix A, vectors y, and initialize them */
            // A = (double *)malloc(m * n * sizeof(double));
            y = (double *)malloc(m * sizeof(double));

            /* send sub-matrices to other processes */
            for (i = 1; i < numprocs - 1; i++)
                MPI_Send(A + i * srow * n, srow * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(A + i * srow * n, srow_last * n, MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD);

            /* perform its own calculation for the 1st sub-matrix */
            mat_vec(A, x, y, srow, n);

            /* collect results from other processes */
            for (i = 1; i < numprocs - 1; i++)
                MPI_Recv(y + i * srow, srow, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(y + i * srow, srow_last, MPI_DOUBLE, numprocs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (i = 0; i < n; i++)
                printf("%lf ", y[i]);
            printf("\n");
        }
        else
        {
            /* slave code */
            /* allocate memory for sub-matrix A, and sub-sector y */
            A = (double *)malloc(srow * n * sizeof(double));
            y = (double *)malloc(srow * sizeof(double));

            /* receive sub-matrix from process 0 */
            MPI_Recv(A, srow * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* perform the calculation on the sub-matrix */
            mat_vec(A, x, y, srow, n);

            /* send the results to process 0 */
            MPI_Send(y, srow, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    free(A);
    free(x);
    free(y);
    MPI_Finalize();
    return 0;
}
