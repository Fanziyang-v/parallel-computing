#include <cuda.h>
#include <stdio.h>

int main() {
    int dimx= 16;
    int num_bytes= dimx* sizeof(int);
    int*d_a= 0, *h_a= 0;   // device and host pointers

    h_a= (int*)malloc(num_bytes);
    cudaMalloc((void**)&d_a, num_bytes);

    if (0 == h_a|| 0 == d_a) {
        printf("couldn't allocate memory\n");
        return 1;
    }

    cudaMemset(d_a, 0, num_bytes);
    cudaMemcpy(h_a, d_a, num_bytes, cudaMemcpyDeviceToHost);

    for (int i= 0; i< dimx; i++)
        printf("%d\n", h_a[i]);
    
    free(h_a);
    cudaFree(d_a);
    return 0;
}
