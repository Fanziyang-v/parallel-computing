#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <sys/time.h>

using namespace std;

// TILE_WIDTH must be greater than kernel size.
const int TILE_WIDTH = 32;
const int MAX_KERNEL_SIZE = 32;
__global__ void convolutionKernel(int *d_image, int *d_feature_map, int *d_kernel, int height, int width, int kernel_size)
{
    __shared__ int ds_kernel[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];

    int h_out = (height - kernel_size) + 1;
    int w_out = (width - kernel_size) + 1;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // identify the row and col of output feature map element to work on.
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    if (ty < kernel_size && tx < kernel_size)
        ds_kernel[ty][tx] = d_kernel[ty * kernel_size + tx];
    __syncthreads();

    // perform convolution.
    int value = 0;
    if (row < h_out && col < w_out)
    {
        #pragma unroll
        for (int i = 0; i < kernel_size; i++)
            for (int j = 0; j < kernel_size; j++)
                value += d_image[(i + row) * width + (j + col)] * ds_kernel[i][j];
        
        d_feature_map[row * w_out + col] = value;
    }
}

// Perform convolution using CPU.
void convolution(int *image, int *feature_map, int *kernel, int height, int width, int kernel_size)
{
    // Compute output feature map size.
    int h_out = (height - kernel_size) + 1;
    int w_out = (width - kernel_size) + 1;

    for (int i = 0; i < h_out; i++)
    {
        for (int j = 0; j < w_out; j++)
        {
            // Compute feature_map[i][j].
            int value = 0;
            #pragma unroll
            for (int k = 0; k < kernel_size; k++)
                for (int l = 0; l < kernel_size; l++)
                    value += image[(i + k) * width + (j + l)] * kernel[k * kernel_size + l];
            feature_map[i * w_out + j] = value;
        }
    }
}

void initialize(int *image, int *kernel, int height, int width, int kernel_size)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // Initialize image with random integer values.
            image[i * width + j] = (int)((double)rand() / RAND_MAX * 10);
        }
    }

    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            // Initialize kernel with random integer values.
            kernel[i * kernel_size + j] = (int)((double)rand() / RAND_MAX * 10); 
        }
    }
}


// Sanity Check.
bool check(int *feat_map1, int *feat_map2, int height, int width)
{
    double epsilon = 1.0e-3;
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            if (fabs(feat_map1[i * width + j] - feat_map2[i * width + j]) > epsilon)
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
    int height = 32, width = 32, kernel_size = 3;
    int h_out, w_out;
    FILE *in_file, *out_file;
    if ((in_file = fopen("input3.txt", "r")) == NULL)
    {
        perror("Can not open file input3.txt!\n");
        exit(-1);
    }
    if ((out_file = fopen("output3.txt", "w")) == NULL)
    {
        perror("Can not open file output3.txt!\n");
        exit(-1);
    }
    fscanf(in_file, "%d,%d,%d", &height, &width, &kernel_size);
    printf("Height=%d, Width=%d\n", height, width);
    printf("Kernel Size=%d\n", kernel_size);

    // Compute output feature map size.
    h_out = (height - kernel_size) + 1;
    w_out = (width - kernel_size) + 1;

    int *h_image, *h_feat_map, *h_feat_map2;
    int *h_kernel, *d_kernel;
    int *d_image, *d_feat_map;

    // allocate host memory.
    h_image = (int *)malloc(height * width * sizeof(int));
    h_feat_map = (int *)malloc(h_out * w_out * sizeof(int));
    h_feat_map2 = (int *)malloc(h_out * w_out * sizeof(int));
    h_kernel = (int *)malloc(kernel_size * kernel_size * sizeof(int));

    // allocate device memory.
    cudaMalloc((void **)&d_kernel, kernel_size * kernel_size * sizeof(int));
    cudaMalloc((void **)&d_image, height * width * sizeof(int));
    cudaMalloc((void **)&d_feat_map, h_out * w_out * sizeof(int));

    initialize(h_image, h_kernel, height, width, kernel_size);

    // Copy image and kernel from host memory to device memory
    cudaMemcpy(d_image, h_image, height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH, TILE_WIDTH);
    dim3 dim_grid((int)ceil(w_out * 1.0 / TILE_WIDTH), (int)ceil(h_out * 1.0 / TILE_WIDTH));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // invoke convolution kernel.
    convolutionKernel<<<dim_grid, dim_block>>>(d_image, d_feat_map, d_kernel, height, width, kernel_size);

    cudaEventRecord(stop, 0);
    // cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    float ElapsedTime;
    cudaEventElapsedTime(&ElapsedTime, start, stop);
    printf("Kernel Elpased Time: %.3f ms\n", ElapsedTime);

    cudaMemcpy(h_feat_map, d_feat_map, h_out * w_out * sizeof(int), cudaMemcpyDeviceToHost);
    
    double begin, end, cpu_time;
    begin = cpuTime();
    // perform convolution using CPU.
    convolution(h_image, h_feat_map2, h_kernel, height, width, kernel_size);
    end = cpuTime();
    cpu_time = end - begin;
    printf("CPU convolution time: %.3f ms\n", cpu_time);

    // check correctness.
    if (check(h_feat_map, h_feat_map2, h_out, w_out))
        printf("Correctness checking passes.\n");
    else
        printf("Correctness checking fails\n");
    
    printf("Speedup: %.3f\n", cpu_time / ElapsedTime);
    fprintf(out_file, "%.2f,%.2f", cpu_time, ElapsedTime);
    // free memory.
    free(h_image);
    free(h_feat_map);
    free(h_feat_map2);
    free(h_kernel);
    cudaFree(d_kernel);
    cudaFree(d_image);
    cudaFree(d_feat_map);

    return 0;
}
