#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <iostream>

__global__ void VectorAddition(int *a_device_this_process, int *b_device_this_process, int *c_device_this_process, int elements_per_process)
{  	
	int block_num = gridDim.x;
  	int block_id = blockIdx.x;
  	int thread_num = blockDim.x;
  	int thread_id = threadIdx.x;
    
	int total_thread_num = block_num*thread_num;
	int id = block_id * thread_num + thread_id;
    for (int i = id; i <elements_per_process; i+=total_thread_num)
	{
		c_device_this_process[i]= a_device_this_process[i] + b_device_this_process[i];
	}
}

int main(int argc, char* argv[]) 
{

	int provided, pid, np;			 

	int *a, *b, *c;
	
	srand(time(NULL));

	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	int total_num, elements_per_process, n_elements_recieved;
	int block_num, thread_num;
    if (pid == 0) {
		block_num = 3; 
		thread_num = 2;
		elements_per_process = block_num*thread_num * (rand() % 5 + 1);
		total_num = np * elements_per_process;
				
		a = (int *)malloc(total_num * sizeof(int));
        b = (int *)malloc(total_num * sizeof(int));
        c = (int *)malloc(total_num * sizeof(int));
        for (int i = 0; i < total_num; i++) {
			a[i] = rand() % 10;
            b[i] = rand() % 10;
        }
	}

	int *a_device, *b_device, *c_device;
	int *a_device_this_process, *b_device_this_process, *c_device_this_process;
	if (pid==0) {
		cudaMalloc( (void **)&a_device, total_num * sizeof(int));
		cudaMalloc( (void **)&b_device, total_num*sizeof(int));
		cudaMalloc( (void **)&c_device, total_num * sizeof(int));
		cudaMemcpy( (void *)a_device, (void *)a, total_num * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy( (void *)b_device, (void *)b, total_num * sizeof(int), cudaMemcpyHostToDevice);
	}

	MPI_Bcast(&elements_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&thread_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

	cudaMalloc( (void **)&a_device_this_process, elements_per_process * sizeof(int));
	cudaMalloc( (void **)&b_device_this_process, elements_per_process*sizeof(int));
	cudaMalloc( (void **)&c_device_this_process, elements_per_process * sizeof(int));
	
	MPI_Scatter(a_device, elements_per_process, MPI_INT, 
				a_device_this_process, elements_per_process, MPI_INT,
				0, MPI_COMM_WORLD); 
    MPI_Scatter(b_device, elements_per_process, MPI_INT, 
				b_device_this_process, elements_per_process, MPI_INT,
				0, MPI_COMM_WORLD); 

	VectorAddition<<<block_num, thread_num>>>(a_device_this_process, b_device_this_process, c_device_this_process, elements_per_process);	

	MPI_Gather(c_device_this_process, elements_per_process, MPI_INT,
               c_device, elements_per_process, MPI_INT, 0,
               MPI_COMM_WORLD); 

	if (pid == 0) {
		cudaMemcpy( (void *)c, (void *)c_device, total_num * sizeof(int), cudaMemcpyDeviceToHost);	
	}

	if (pid==0) {
		cudaFree(a_device);
		cudaFree(b_device);
		cudaFree(c_device);
	}
	cudaFree(a_device_this_process);
	cudaFree(b_device_this_process);
	cudaFree(c_device_this_process);

	if (pid == 0) {
		printf("A:");
		for (int i = 0; i < total_num ; i++) {
			printf("%2d ",a[i]);
		}
		printf("\n");

		printf("B:");
		for (int i = 0; i < total_num ; i++) {
			printf("%2d ",b[i]);
		}
		printf("\n");

		printf("C:");
		for (int i = 0; i < total_num ; i++) {
			printf("%2d ",c[i]);
		}
		printf("\n");

		free(a);
		free(b);
		free(c);
	}


	MPI_Finalize(); 

	return 0; 
} 
