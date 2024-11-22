#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <iostream>

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

   // add your code here

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
