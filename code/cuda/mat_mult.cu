#include <iostream>
#include <cmath>
#include <cuda.h>

using namespace std;

const int TILE_WIDTH = 32;
__global__ void MatrixMultKernel(float *d_M, float *d_N, float *d_P, int m, int n, int k)
{
    /* one column padding in order to prevent bank conflict. */
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH + 1];
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    /* identity row and column of matrix P elment to work on */
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0;
    for (int t = 0; t < (int)ceil(n * 1.0 / TILE_WIDTH); t++)
    {
        /* Load data from global memory. */
        ds_M[ty][tx] = row < m && t * TILE_WIDTH + tx < n ? d_M[row * n + t * TILE_WIDTH + tx] : 0;
        ds_N[ty][tx] = col < k && t * TILE_WIDTH + ty < n ? d_N[(t * TILE_WIDTH + ty) * k + col] : 0;
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++)
            sum += ds_M[ty][i] * ds_N[i][tx];
        __syncthreads();
    }
    if (row < m && col < k)
        d_P[row * k + col] = sum;
}

void matrix_mult(float *M, float *N, float *P, int m, int n, int p)
{
    int i, j, k;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < p; j++)
        {
            float sum = 0.0;
            for (k = 0; k < n; k++)
                sum += M[i * n + k] * N[k * p + j];
            P[i * p + j] = sum;
        }
    }
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
    int m, n, k;
    float *h_M, *h_N, *h_P, *h_Q;
    float *d_M, *d_N, *d_P;
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH);
    dim3 dim_grid;

    cout << "Enter m, n, k" << endl;
    cin >> m >> n >> k;
    dim_grid.x = (int)ceil(k * 1.0 / TILE_WIDTH);
    dim_grid.y = (int)ceil(m * 1.0 / TILE_WIDTH);

    /* allocate memory. */
    h_M = new float[m * n];
    h_N = new float[n * k];
    h_P = new float[m * k];
    h_Q = new float[m * k];
    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_N, n * k * sizeof(float));
    cudaMalloc((void **)&d_P, m * k * sizeof(float));

    /* Initialize matrix with random floating-point values */
    initialize(h_M, m, n);
    initialize(h_N, n, k);
    cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, n * k * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float tp;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* Invoke kernel */
    MatrixMultKernel<<<dim_grid, dim_block>>>(d_M, d_N, d_P, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tp, start, stop);

    cudaMemcpy(h_P, d_P, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t t;
    float ts;
    t = clock();
    matrix_mult(h_M, h_N, h_Q, m, n, k);
    ts = (float)(clock() - t) / CLOCKS_PER_SEC * 1000.0;


    if (check(h_P, h_Q, m, k))
        cout << "Sanity check passes!" << endl;
    else
        cout << "Sanity check fails!" << endl;

    cout << "Parallel computation time is " << tp << " ms" << endl;
    cout << "Serial computation time is " << ts << " ms" << endl;
    cout << "Speedup is " << ts / tp << endl;

    /* free memory */
    delete[] h_M;
    delete[] h_N;
    delete[] h_P;
    delete[] h_Q;
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}