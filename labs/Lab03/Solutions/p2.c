#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#define MAX(a, b) (a > b ? a : b)
#define REDUCE(a, b, op) (op == MPI_SUM ? a + b : MAX(a, b))

void Ring_AllreduceV1(const double *sendbuf, double *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int i, t, base, tiles;
    int prev_pid, next_pid;
    int send_pos, recv_pos;
    int numprocs, pid;
    double *buf, val;
    MPI_Request requests[2];
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &pid);
    printf("I am proc %d\n", pid);

    tiles = (int)ceil(count * 1.0 / numprocs);
    buf = (double *)malloc(numprocs * tiles * sizeof(double));
    memcpy(buf, sendbuf, count * sizeof(double));
    prev_pid = (pid - 1 + numprocs) % numprocs;
    next_pid = (pid + 1) % numprocs;
    for (t = 0; t < tiles; t++)
    {
        base = t * numprocs;
        for (i = 0; i < 2 * (numprocs - 1); i++)
        {
            // Send sendbuf[send_pos] to process next_pid and receive buf[recv_pos] from process prev_pid
            send_pos = (base + pid - (i % numprocs) + numprocs) % numprocs;
            recv_pos = (base + pid - (i % numprocs) - 1 + numprocs) % numprocs;
            MPI_Isend(&buf[base + send_pos], 1, datatype, next_pid, base + i, comm, &requests[0]);
            MPI_Irecv(&val, 1, datatype, prev_pid, base + i, comm, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            if (i < numprocs - 1)
                buf[base + recv_pos] = REDUCE(buf[base + recv_pos], val, op); // perform reduce operation.
            else
                buf[base + recv_pos] = val;
        }
    }
    memcpy(recvbuf, buf, count * sizeof(double));
    free(buf);
}

void Ring_AllreduceV2(const double *sendbuf, double *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int i, j;
    int prev_pid, next_pid;
    int numprocs, pid;
    double *sbuf, *rbuf;
    MPI_Request requests[2];
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &pid);
    printf("I am proc %d\n", pid);

    sbuf = (double *)malloc(count * sizeof(double));
    rbuf = (double *)malloc(count * sizeof(double));
    memcpy(sbuf, sendbuf, count * sizeof(double));
    memcpy(recvbuf, sendbuf, count * sizeof(double));
    prev_pid = (pid - 1 + numprocs) % numprocs;
    next_pid = (pid + 1) % numprocs;

    for (i = 0; i < numprocs - 1; i++)
    {
        MPI_Isend(sbuf, count, datatype, next_pid, 0, comm, &requests[0]);
        MPI_Irecv(rbuf, count, datatype, prev_pid, 0, comm, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
#pragma unroll
        for (j = 0; j < count; j++)
        {
            // perform reduce operation.
            recvbuf[j] = REDUCE(rbuf[j], recvbuf[j], op);
        }

        // memcpy(sbuf, rbuf, count * sizeof(double));
        double *temp = sbuf;
        sbuf = rbuf;
        rbuf = temp;
    }
    free(sbuf);
    free(rbuf);
}
// Initialize array with random floating-point values.
void initialize(double *a, int n, unsigned seed)
{
    int i;
    srand(seed);
    for (i = 0; i < n; i++)
        a[i] = (double)rand() / RAND_MAX;
}

// sanity check
int check(double *a, double *b, int n)
{
    int i;
    double eps = 1.e-5;
    for (i = 0; i < n; i++)
        if (fabs(a[i] - b[i]) > eps)
            return 0;
    return 1;
}

int main(int argc, char *argv[])
{
    FILE *infile, *outfile;
    int i, n = 100000;
    double *a, *b, *c;
    char strop[10];
    int numprocs, pid;
    MPI_Op op;
    double start, end, elapsed_time, t1, t2;
    if ((infile = fopen("input2.txt", "r")) == NULL)
    {
        perror("Error open input2.txt");
        exit(-1);
    }
    if ((outfile = fopen("output2.txt", "w")) == NULL)
    {
        perror("Error open output2.txt");
        exit(-1);
    }

    fscanf(infile, "%d,%s", &n, strop);
    printf("n=%d,op=%s\n", n, strop);
    op = strcmp(strop, "sum") == 0 ? MPI_SUM : MPI_MAX;
    // exit(0);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // allocate memory and initialize array a with random floating-point values.
    a = (double *)malloc(n * sizeof(double));
    b = (double *)malloc(n * sizeof(double));
    c = (double *)malloc(n * sizeof(double));
    initialize(a, n, 10213u + pid);

    // Perform Ring-based all reduce.
    start = MPI_Wtime();
    // Ring_AllreduceV1(a, b, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Ring_AllreduceV2(a, b, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    end = MPI_Wtime();
    t1 = elapsed_time = (end - start) * 1000.0;
    printf("Ring_AllreduceV2 operation took %lf ms on process %d\n", elapsed_time, pid);

    // Perform MPI all reduce.
    start = MPI_Wtime();
    MPI_Allreduce(a, c, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    end = MPI_Wtime();
    t2 = elapsed_time = (end - start) * 1000.0;
    printf("MPI_Allreduce operation took %lf ms on process %d\n", elapsed_time, pid);

    if (check(b, c, n))
        printf("Sanity check in Process %d passes!\n", pid);
    else
        printf("Sanity check in Process %d fails!\n", pid);
    
    
    fprintf(outfile, "%.3f,%.3f\n", t2, t1);
    free(a);
    free(b);
    free(c);
    MPI_Finalize();
    return 0;
}
