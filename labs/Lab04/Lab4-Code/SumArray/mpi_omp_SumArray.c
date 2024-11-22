#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <iostream>

int main(int argc, char* argv[]) 
{

	int pid, np, provided, local_sum = 0, global_sum;			 
	// np -> no. of processes 
	// pid -> process id
	int *a, *a2;
	srand(time(NULL));

	MPI_Status status;

	// Creation of parallel processes
	// MPI_Init(&argc, &argv);
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

	// find out process ID,
	// and how many processes were started
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	int total_num, elements_per_process, n_elements_recieved;

	#pragma omp parallel
	{
		if (pid == 0) {
			#pragma omp master
			{
				elements_per_process = omp_get_num_threads() * (rand() % 5 + 1);
				total_num = np * elements_per_process;
				
				a = (int *)malloc(total_num * sizeof(int));
				printf("# processes: %d, # threads per process: %d, # elements per process: %d\n", np, omp_get_num_threads(), elements_per_process);
				printf("Total Num of the array: %d\n", total_num);
			}
			
			#pragma omp barrier
			
			#pragma omp for
			for (int i = 0; i < total_num; i++)
				a[i] = i + 1;//rand() % 20;

			#pragma omp master 
			{
				printf("Array: ");
				for (int i = 0; i < total_num; i++) printf("%d ", a[i]);
				printf("\n");
				for (int i = 1; i < np; i++) { 

					MPI_Send(&elements_per_process, 
							1, MPI_INT, i, 0, 
							MPI_COMM_WORLD); 
					MPI_Send(&a[i * elements_per_process], 
							elements_per_process, 
							MPI_INT, i, 0, 
							MPI_COMM_WORLD); 
				}
			}
			#pragma omp barrier

			#pragma omp for reduction(+:local_sum)
			for (int i = 0; i < elements_per_process; i++)
				local_sum += a[i];
		}
		else {

			#pragma omp master 
			{
				MPI_Recv(&n_elements_recieved, 
				1, MPI_INT, 0, 0, 
				MPI_COMM_WORLD, 
				&status); 

				a2 = (int *)malloc(n_elements_recieved * sizeof(int));

				// stores the received array segment 
				// in local array a2 
				MPI_Recv(a2, n_elements_recieved, 
						MPI_INT, 0, 0, 
						MPI_COMM_WORLD, 
						&status);
			}
			#pragma omp barrier
			#pragma omp for reduction(+:local_sum)
			for (int i = 0; i < n_elements_recieved; i++)
				local_sum += a2[i];
		}
	}

	MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (!pid) {
		printf("Global sum: %d\n", global_sum);
		free(a);
	}
	else free(a2);

	// cleans up all MPI state before exit of process 
	MPI_Finalize(); 

	return 0; 
} 
