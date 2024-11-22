#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <sys/time.h>

using namespace std;

const int TILE_WIDTH = 32;
__global__ void MatrixTransposeKernel(float *d_M, float *d_N, int m, int n)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Identify the row and column of the matrix N element to work on.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    // (col, row) is the corresponding position in Matrix M.
    if (row < n && col < m)
    {
        ds_M[ty][tx] = d_M[col * n + row];
        d_N[row * m + col] = ds_M[ty][tx];
    }
}

// shared memory will not be used in here.
__global__ void MatrixTransposeKernel2(float *d_M, float *d_N, int m, int n)
{   
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Identify the row and column of the matrix N element to work on.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    // (col, row) is the corresponding position in Matrix M.
    if (row < n && col < m)
        d_N[row * m + col] = d_M[col * n + row];
}

// perform matrix transpose using CPU.
void MatrixTranspose(float *h_M, float *h_N, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            h_N[j * m + i] = h_M[i * n + j];
}

// Initialize Matrix and Vector element with random floating-point values.
void initialize(float *h_M, int m, int n)
{
    srand(10213u);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // generate random number in range [0, 1).
            float number = (float)rand() / RAND_MAX * 1.0;
            h_M[i * n + j] = number;
        }
    }
}

// Compute cpu time(ms).
double cpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (1000 * (double)tp.tv_sec + (double)tp.tv_usec * 1.e-3);
}

bool check(float *h_N, float *h_NN, int m, int n)
{
    double epsilon = 1.0e-3;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (fabs(h_N[i * m + j] - h_NN[i * m + j]) > epsilon)
                return false;
    return true;
}

void printMatrix(float *mat, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%.2f ", mat[i * n + j]);
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int m = 100, n = 100;
    FILE *in_file, *out_file;
    if ((in_file = fopen("input2.txt", "r")) == NULL)
    {
        perror("Can not open file input2.txt!\n");
        exit(-1);
    }
    if ((out_file = fopen("output2.txt", "w")) == NULL)
    {
        perror("Can not open file output2.txt!\n");
        exit(-1);
    }
    fscanf(in_file, "%d,%d", &m, &n);
    printf("m=%d, n=%d\n", m, n);
    
    float *h_M, *h_N, *h_NN, *d_M, *d_N, *d_NN;
    // allocate host memory.
    h_M = (float *)malloc(m * n * sizeof(float));
    h_N = (float *)malloc(m * n * sizeof(float));
    h_NN = (float *)malloc(m * n * sizeof(float));

    // allocate device memory.
    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, m * n * sizeof(float));
    cudaMalloc((void **)&d_NN, m * n * sizeof(float));

    // initialize Matrix and Vector.
    initialize(h_M, m, n);

    // Copy Matrix elements from host to device memory.
    cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH, TILE_WIDTH);
    dim3 dim_grid((int)ceil(m * 1.0 / TILE_WIDTH), (int)ceil(n * 1.0 / TILE_WIDTH));
    printf("number of thread blocks in grid: %d\n", dim_grid.x * dim_grid.y);
    printf("number of threads in thread block: %d\n", dim_block.x * dim_block.y);

    printf("dim_grid=(%d, %d)\n", dim_grid.x, dim_grid.y);
    printf("dim_block=(%d, %d)\n", dim_block.x, dim_block.y);

    double begin, end, t1;
    begin = cpuTime();
    // perform matrix transpose using CPU.
    MatrixTranspose(h_M, h_NN, m, n);
    end = cpuTime();
    t1 = end - begin;
    printf("CPU matrix transpose time: %.3f ms\n", t1);

    // printf("CPU Matrix Tranpose Result:\n");
    // printMatrix(h_NN, n, m);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    // invoke Matrix Transpose kernel(with shared memory).
    MatrixTransposeKernel<<<dim_grid, dim_block>>>(d_M, d_N, m, n);
    
    cudaEventRecord(stop1, 0);
    // cudaDeviceSynchronize();
    cudaEventSynchronize(stop1);
    float t2;
    cudaEventElapsedTime(&t2, start1, stop1);
    printf("Kernel Elpased Time(With shared memory): %.3f ms\n", t2);
    
    cudaMemcpy(h_N, d_N, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // printf("GPU Matrix Tranpose Result(With shared memoery):\n");
    // printMatrix(h_N, n, m);

    // check correctness.
    if (check(h_N, h_NN, m, n))
        printf("Correctness checking passes.\n");
    else
        printf("Correctness checking fails\n");

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);
    // invoke Matrix Transpose kernel(without shared memory).
    MatrixTransposeKernel2<<<dim_grid, dim_block>>>(d_M, d_NN, m, n);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float t3;
    cudaEventElapsedTime(&t3, start2, stop2);
    printf("Kernel Elpased Time(Without shared memory): %.3f ms\n", t3);

    cudaMemcpy(h_N, d_NN, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("GPU Matrix Tranpose Result(Without shared memoery):\n");
    // printMatrix(h_N, n, m);
    // check correctness.
    if (check(h_N, h_NN, m, n))
        printf("Correctness checking passes.\n");
    else
        printf("Correctness checking fails\n");

    printf("Speedup(With Shared meomry): %.3f\n", t1 / t2);
    printf("Speedup(Without Shared meomry): %.3f\n", t1 / t3);

    fprintf(out_file, "%.2f,%.2f,%.2f", t1, t3, t2);

    // free memory.
    free(h_M);
    free(h_N);
    free(h_NN);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_NN);
    return 0;
}

