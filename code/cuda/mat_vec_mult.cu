#include <iostream>
#include <cuda.h>
#include <cmath>

using namespace std;

const int TILE_WIDTH = 256;
__global__ void MatrixVectorMultKernel(float *d_M, float *d_v, float *d_out, int m, int n)
{
    /* Share memory. */
    __shared__ float ds_v[TILE_WIDTH];
    int num_iterations = (int)ceil(n * 1.0 / TILE_WIDTH);

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int row = bx * TILE_WIDTH + tx;
    float value = 0;
    for (int t = 0; t < num_iterations; t++)
    {
        /* Load a part of vector data. */
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
        d_out[row] = value;
}

void initialize(float *h_M, float *h_v, int m, int n)
{
    int i, j;
    srand(10213u);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            h_M[i * n + j] = (float)rand() / RAND_MAX;
    for (j = 0; j < n; j++)
        h_v[j] = (float)rand() / RAND_MAX;
}

void mat_vec_mult(float *M, float *v, float *w, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        float sum = 0;
        for (j = 0; j < n; j++)
            sum += M[i * n + j] * v[j];
        w[i] = sum;
    }
}

void print_vector(float *v, int n)
{
    for (int i = 0; i < n; i++)
        cout << v[i] << " ";
    cout << endl;
}

bool check(float *u, float *v, int n)
{
    const double EPS = 1.e-3;
    int i;
    for (i = 0; i < n; i++)
        if (fabs(u[i] - v[i]) > EPS)
            return false;
    return true;
}

int main(int argc, char *argv[])
{
    int m = 10, n = 10;
    float *h_M, *h_v, *h_out, *w;
    float *d_M, *d_v, *d_out;
    int num_threads = TILE_WIDTH;
    int num_blocks;

    cout << "Enter m, n:" << endl;
    cin >> m >> n;
    num_blocks = (int)ceil(m * 1.0 / TILE_WIDTH);
    /* allocate memory */
    h_M = new float[m * n];
    h_v = new float[n];
    h_out = new float[m];
    w = new float[m];
    initialize(h_M, h_v, m, n);
    cudaMalloc((void **)&d_M, m * n * sizeof(float));
    cudaMalloc((void **)&d_v, n * sizeof(float));
    cudaMalloc((void **)&d_out, m * sizeof(float));

    /* Copy data from host memory to device memory. */
    cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    /* Invoke kernel. */
    MatrixVectorMultKernel<<<num_blocks, num_threads>>>(d_M, d_v, d_out, m, n);

    cudaEventRecord(stop, 0);
    // cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float tp;
    cudaEventElapsedTime(&tp, start, stop);
    cout << "Kernel Elpased Time is " << tp << " ms" << endl;

    /* copy results from device to host memory. */
    cudaMemcpy(h_out, d_out, m * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Parallel computation result: " << endl;
    print_vector(h_out, m);
    float ts;
    clock_t t = clock();
    /* Perform matrix-vector multiplicaiton in CPU. */
    mat_vec_mult(h_M, h_v, w, m, n);
    ts = (double)(clock() - t) / CLOCKS_PER_SEC * 1000.0;
    cout << "Serial computation result: " << endl;
    print_vector(w, m);
    if (check(h_out, w, m))
        cout << "Sanity check passes!" << endl;
    else
        cout << "Sanity check fails!" << endl;
    
    cout << "Serial computation time is " << ts << " ms" << endl;
    cout << "Speedup is " << ts / tp << endl;
    
    delete[] h_M;
    delete[] h_v;
    delete[] h_out;
    delete[] w;
    cudaFree(d_M);
    cudaFree(d_v);
    cudaFree(d_out);
    return 0;
}