#include <iostream>
#include <cstring>
#include <cmath>
#include "mpi.h"

using namespace std;

double reduce(double a, double b, string op)
{
    if (op == "sum")
        return a + b;
    else if (op == "max")
        return max(a, b);
    else
    {
        cout << "Unkown operator " << op << endl;
        exit(-1);
    }
}

void Reduce_Array(double *dst, double *src, int n, string op)
{
    for (int i = 0; i < n; i++)
        dst[i] = reduce(dst[i], src[i], op);
}

void Ring_Based_Allreduce(const double *sendbuf, double *recvbuf, int count, string op, MPI_Comm comm)
{
    int block_size, last_block_size;
    int rank, numprocs;
    double *buf;
    MPI_Request sendreq, recvreq;
    MPI_Comm_size(comm, &numprocs);
    MPI_Comm_rank(comm, &rank);
    /* Locate the previous and next process id. */
    int prev_id = (rank - 1 + numprocs) % numprocs;
    int next_id = (rank + 1) % numprocs;

    /* Compute block size. */
    block_size = count / numprocs;
    last_block_size = block_size + count % numprocs;
    buf = new double[last_block_size];

    memcpy(recvbuf, sendbuf, count * sizeof(double));
    for (int i = 0; i < numprocs - 1; i++)
    {
        int sblk_idx = (rank - i + numprocs) % numprocs;
        int rblk_idx = (sblk_idx - 1 + numprocs) % numprocs;
        MPI_Isend(recvbuf + sblk_idx * block_size, sblk_idx == numprocs - 1 ? last_block_size : block_size, MPI_DOUBLE, next_id, i, comm, &sendreq);
        MPI_Irecv(buf, rblk_idx == numprocs - 1 ? last_block_size : block_size, MPI_DOUBLE, prev_id, i, comm, &recvreq);
        /* Once process rank received the elements from previous process, then do reduction. */
        MPI_Wait(&recvreq, MPI_STATUS_IGNORE);
        Reduce_Array(recvbuf + rblk_idx * block_size, buf, rblk_idx == numprocs - 1 ? last_block_size : block_size, op);
        MPI_Wait(&sendreq, MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < numprocs - 1; i++)
    {
        int sblk_idx = (rank + 1 - i + numprocs) % numprocs;
        int rblk_idx = (sblk_idx - 1 + numprocs) % numprocs;
        MPI_Isend(recvbuf + sblk_idx * block_size, sblk_idx == numprocs - 1 ? last_block_size : block_size, MPI_DOUBLE, next_id, i, comm, &sendreq);
        MPI_Irecv(recvbuf + rblk_idx * block_size, rblk_idx == numprocs - 1 ? last_block_size : block_size, MPI_DOUBLE, prev_id, i, comm, &recvreq);
        MPI_Wait(&sendreq, MPI_STATUS_IGNORE);
        MPI_Wait(&recvreq, MPI_STATUS_IGNORE);
    }
}

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
    double start, end, t1, t2, sum_t1, sum_t2;
    double *x, *y, *z;
    string op;
    int n, rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        cout << "Enter n, op(sum | max): " << endl;
        cin >> n >> op;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    x = new double[n];
    y = new double[n];
    z = new double[n];
    for (int i = 0; i < n; i++)
        x[i] = rank;

    /* Perform Ring-based Allreduce. */
    start = MPI_Wtime();
    Ring_Based_Allreduce(x, y, n, op, MPI_COMM_WORLD);
    end = MPI_Wtime();
    t1 = (end - start) * 1000.0;

    /* Perform MPI_Allreduce. */
    start = MPI_Wtime();
    MPI_Allreduce(x, z, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    end = MPI_Wtime();
    t2 = (end - start) * 1000.0;

    MPI_Reduce(&t1, &sum_t1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t2, &sum_t2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        double avg_t1 = sum_t1 / numprocs;
        double avg_t2 = sum_t2 / numprocs;
        cout << "Ring-Based Allreduce time is " << avg_t1 << " ms" << endl;
        cout << "MPI_Allreduce time is " << avg_t2 << " ms" << endl;
        cout << "Speedup is " << avg_t2 / avg_t1 << endl;
        if (check(y, z, n))
            cout << "Sanity check passes!" << endl;
        else
            cout << "Sanity check fails!" << endl;
    }
    MPI_Finalize();
    return 0;
}
