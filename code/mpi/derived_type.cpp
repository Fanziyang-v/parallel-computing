#include <iostream>
#include <cmath>
#include "mpi.h"
#include <cstddef>

using namespace std;

struct mystruct
{
    char x;
    double y[6];
    int z[4];
};

int rand_int(int a, int b)
{
    int range = (b - a);
    return a + (int)floor((double)rand() / RAND_MAX * range);
}

void initialize(struct mystruct *data, int n)
{
    srand(10213u);
    for (int i = 0; i < n; i++)
    {
        data[i].x = 'a' + rand_int(0, 26);
        for (int j = 0; j < 6; j++)
            data[i].y[j] = j;
        for (int k = 0; k < 4; k++)
            data[i].z[k] = k;
    }
}

void print(struct mystruct *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        cout << data[i].x << " ";
        for (int j = 0; j < 6; j++)
            cout << data[i].y[j] << " ";
        for (int k = 0; k < 4; k++)
            cout << data[i].z[k] << " ";
        cout << endl;
    }
}

int main(int argc, char *argv[])
{
    struct mystruct mydata[1000];
    int rank, numprocs;
    MPI_Datatype dtype;
    MPI_Datatype types[3] = {MPI_CHAR, MPI_DOUBLE, MPI_INT};
    int block_lengths[3] = {1, 6, 4};
    MPI_Aint disp[3];
    disp[0] = offsetof(mystruct, x);
    disp[1] = offsetof(mystruct, y);
    disp[2] = offsetof(mystruct, z);
    MPI_Init(&argc, &argv);
    MPI_Type_create_struct(3, block_lengths, disp, types, &dtype);
    MPI_Type_commit(&dtype);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        initialize(mydata, 1000);
        MPI_Send(mydata, 1000, dtype, 1, 0, MPI_COMM_WORLD);
        cout << "Master process has sent mydata array to Process 1" << endl;
    }
    else
    {
        MPI_Recv(mydata, 1000, dtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "process 1 has received mydata array from master process" << endl;
        cout << "The first 50 elements are as follows:" << endl;
        print(mydata, 50);
    }
    MPI_Type_free(&dtype);
    MPI_Finalize();
    return 0;
}