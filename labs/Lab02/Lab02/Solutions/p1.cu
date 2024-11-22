#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <sys/time.h>

using namespace std;

const int TILE_WIDTH = 1024;
__global__ void MatrixVectorMulKernel(float *d_M, float *d_v, float *d_w, int m, int n)
{
    __shared__ float ds_v[TILE_WIDTH];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Identify the row of the vector w element to work on.
    int row = bx * TILE_WIDTH + tx;
    float value = 0;
    for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; t++)
    {
        if (t * TILE_WIDTH + tx < n)
            ds_v[tx] = d_v[t * TILE_WIDTH + tx];
        else
            ds_v[tx] = 0;
        __syncthreads();
        if (row < m)
        {
            for (int i = 0; i < TILE_WIDTH; i++)
                value += d_M[row * n + t * TILE_WIDTH + i] * ds_v[i];
        }
        __syncthreads();
    }
    if (row < m)
        d_w[row] = value;
}

// Perform Matrix-Vector multiplication using GPU.
void MatrixVectorMul(float *h_M, float *h_v, float *h_w, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        float value = 0;
        for (j = 0; j < n; j++)
            value += h_M[i * n + j] * h_v[j];
        h_w[i] = value;
    }
}

// Initialize Matrix and Vector element with random floating-point values.
void initialize(float *h_M, float *h_v, int m, int n)
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
    for (int k = 0; k < n; k++)
    {
        // generate random number in range [0, 1).
        float number = (float)rand() / RAND_MAX * 1.0;
        h_v[k] = number;
    }
}

// Correctness checking.
bool check(float *h_w, float *h_ww, int m)
{
    int i;
    double epsilon = 1.0e-3;
    for (i = 0; i < m; i++)
        if (fabs(h_w[i] - h_ww[i]) > epsilon)
            return false;
    return true;
}

// Compute cpu time(ms).
double cpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (1000 * (double)tp.tv_sec + (double)tp.tv_usec * 1.e-3);
}

int main(int argc, char *argv[])
{
    int m = 32, n = 32;
    float *h_M, *h_v, *d_M, *d_v;
    float *h_w, *h_ww, *d_w;
    FILE *in_file, *out_file;
    if ((in_file = fopen("input1.txt", "r")) == NULL)
    {
        perror("Can not open file input1.txt!\n");
        exit(-1);
    }
    if ((out_file = fopen("output1.txt", "w")) == NULL)
    {
        perror("Can not open file output1.txt!\n");
        exit(-1);
    }
    fscanf(in_file, "%d,%d", &m, &n);
    printf("m=%d, n=%d\n", m, n);
    // allocate host memory.
    h_M = (float *)malloc(m * n * sizeof(float));
    h_v = (float *)malloc(n * sizeof(float));
    h_w = (float *)malloc(m * sizeof(float));
    h_ww = (float *)malloc(m * sizeof(float));

    // allocate device memory.
    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_v, n * sizeof(float));
    cudaMalloc((void **)&d_w, m * sizeof(float));

    // initialize Matrix and Vector.
    initialize(h_M, h_v, m, n);

    // Copy Matrix and Vector elements from host to device memory.
    cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH);
    dim3 dim_grid((int)ceil(m * 1.0 / TILE_WIDTH));

    printf("number of thread blocks in grid: %d\n", dim_grid.x);
    printf("number of threads in thread block: %d\n", dim_block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Invoke Matrix-Vector multiplication kernel.
    MatrixVectorMulKernel<<<dim_grid, dim_block>>>(d_M, d_v, d_w, m, n);

    cudaEventRecord(stop, 0);
    // cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float ElapsedTime;
    cudaEventElapsedTime(&ElapsedTime, start, stop);
    printf("Kernel Elpased Time: %.3f ms\n", ElapsedTime);

    cudaMemcpy(h_w, d_w, m * sizeof(float), cudaMemcpyDeviceToHost);

    double begin, end, cpu_time;
    begin = cpuTime();
    // perform matrix-vector multiplication using CPU.
    MatrixVectorMul(h_M, h_v, h_ww, m, n);
    end = cpuTime();
    cpu_time = end - begin;
    printf("CPU matrix-vector multiplication time: %.3f ms\n", cpu_time);

    // check correctness.
    if (check(h_w, h_ww, m))
        printf("Correctness checking passes.\n");
    else
        printf("Correctness checking fails\n");

    printf("Speedup: %.3f\n", cpu_time / ElapsedTime);
    fprintf(out_file, "%.2f,%.2f", cpu_time, ElapsedTime);
    // free memory.
    free(h_M);
    free(h_v);
    free(h_w);
    free(h_ww);
    cudaFree(d_M);
    cudaFree(d_v);
    cudaFree(d_w);

    return 0;
}
