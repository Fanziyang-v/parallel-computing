#include <cuda.h>
#include <stdio.h>

// Device code
__global__ void mykernel(int* a) {
    int idx= blockIdx.x* blockDim.x+ threadIdx.x;  // locate the data item handled by this thread
    a[idx] = threadIdx.x;
}

// Host code
int main() {
    int dimx= 16, num_bytes= dimx*sizeof(int);
    int*d_a= 0, *h_a= 0; // device and host pointers

    h_a= (int*)malloc(num_bytes);
    cudaMalloc((void**)&d_a, num_bytes);
    
    cudaMemset(d_a, 0, num_bytes);
    dim3  grid, block;
    block.x= 4;         // each block has 4 threads
    grid.x= dimx / block.x;     // # of blocks is calculated
    
    mykernel<<<grid, block>>>(d_a);
    cudaMemcpy(h_a, d_a, num_bytes, cudaMemcpyDeviceToHost);
    
    for(int i= 0; i< dimx; i++)
        printf("%d\n", h_a[i]);
    
    free(h_a);
    cudaFree(d_a);
    return 0;
}



