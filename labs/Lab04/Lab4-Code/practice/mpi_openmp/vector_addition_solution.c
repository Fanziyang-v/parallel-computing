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
    if (pid == 0) {
		elements_per_process = omp_get_num_threads() * (rand() % 5 + 1);
		total_num = np * elements_per_process;
				
		a = (int *)malloc(total_num * sizeof(int));
        b = (int *)malloc(total_num * sizeof(int));
        c = (int *)malloc(total_num * sizeof(int));
        for (int i = 0; i < total_num; i++) {
			a[i] = rand() % 10;
            b[i] = rand() % 10;
        }
	}

	int *a_this_process, *b_this_process, *c_this_process;
	#pragma omp parallel
	{
		
		#pragma omp master 
		{
			MPI_Bcast(&elements_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
			a_this_process = (int *)malloc(elements_per_process * sizeof(int));
        	b_this_process = (int *)malloc(elements_per_process * sizeof(int));
        	c_this_process = (int *)malloc(elements_per_process * sizeof(int));			
			MPI_Scatter(a, elements_per_process, MPI_INT, 
				        a_this_process, elements_per_process, MPI_INT,
						0, MPI_COMM_WORLD); 
            MPI_Scatter(b, elements_per_process, MPI_INT, 
				        b_this_process, elements_per_process, MPI_INT,
						0, MPI_COMM_WORLD); 
		}
		#pragma omp barrier

		#pragma omp for 
		for (int i = 0; i < elements_per_process; i++)
			c_this_process[i] = a_this_process[i] + b_this_process[i];

		#pragma omp master 
		{
			MPI_Gather(c_this_process, elements_per_process, MPI_INT,
             c, elements_per_process, MPI_INT, 0,
            MPI_COMM_WORLD); 
		}
	}
	free(a_this_process);
	free(b_this_process);
	free(c_this_process);

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
