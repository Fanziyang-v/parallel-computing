#include <iostream>
#include <cmath>
#include <cuda.h>

using namespace std;

const int TILE_WIDTH = 32;
__global__ void MatrixTransposeKernel(float *d_M, float *d_N, int m, int n)
{
    /* one column padding in order to prevent bank conflict. */
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH + 1];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* compute the left-upper corner coordinate in matrix M. */
    int startx = bx * TILE_WIDTH;
    int starty = by * TILE_WIDTH;
    /* identify the row and column of the matrix N element to work on */
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    if (startx + ty < m && starty + tx < n)
        ds_N[tx][ty] = d_M[(startx + ty) * n + (starty + tx)];
    __syncthreads();
    if (row < n && col < m)
        d_N[row * m + col] = ds_N[ty][tx];
}

void matrix_transpose(float *M, float *N, int m, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            N[i * m + j] = M[j * n + i];
}

void initialize(float *M, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            M[i * n + j] = (float)rand() / RAND_MAX;
}

bool check(float *M, float *N, int m, int n)
{
    const float EPS = 1.e-6;
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            if (fabs(M[i * n + j] - N[i * n + j]) > EPS)
                return false;
    return true;
}

int main()
{
    int m, n;
    float *h_M, *h_N, *h_P;
    float *d_M, *d_N;
    dim3 dim_blcok(TILE_WIDTH, TILE_WIDTH);
    dim3 dim_grid;

    cout << "Enter m, n" << endl;
    cin >> m >> n;
    dim_grid.x = (int)ceil(m * 1.0 / TILE_WIDTH);
    dim_grid.y = (int)ceil(n * 1.0 / TILE_WIDTH);
    cout << "Number of thread blocks in grid: " << dim_grid.x * dim_grid.y << endl;
    cout << "Number of threads in thread block: " << dim_blcok.x * dim_blcok.y << endl;

    /* Allocate memory */
    h_M = new float[m * n];
    h_N = new float[m * n];
    h_P = new float[m * n];
    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, m * n * sizeof(float));

    /* Initialize matrix M with random floating-point values. */
    initialize(h_M, m, n);

    /* Copy data from host memory to device memory */
    cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float tp;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* Invoke kernel */
    MatrixTransposeKernel<<<dim_grid, dim_blcok>>>(d_M, d_N, m, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tp, start, stop);

    cout << "Parallel computation time is " << tp << " ms" << endl;
    cudaMemcpy(h_N, d_N, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float ts;
    clock_t t;
    t = clock();
    matrix_transpose(h_M, h_P, m, n);
    ts = (float)(clock() - t) / CLOCKS_PER_SEC * 1000.0;

    cout << "Serial computation time is " << ts << " ms" << endl;
    cout << "Speedup is " << ts / tp << endl;

    if (check(h_N, h_P, n, m))
        cout << "Sanity check passes!" << endl;
    else
        cout << "Sanity check fails!" << endl;

    delete[] h_M;
    delete[] h_N;
    delete[] h_P;
    cudaFree(d_M);
    cudaFree(d_N);
    return 0;
}