#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
#include <math.h>
#include <mpi-ext.h>

#define BLOCKSIZE 16
int IntializingMatrixVectors(float **, float **, float **, int , int , int , int);

//Pragma routine to report the detail of cuda error
#define CUDA_SAFE_CALL(call)                                                         \
            do{                                                                      \
                 cudaError_t err = call;                                             \
                 if(err != cudaSuccess)                                              \
                 {                                                                   \
                        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                         __FILE__, __LINE__, cudaGetErrorString( err) );             \
                         exit(1);                                                    \
                 }                                                                   \
               } while (0)                                                           \


void cuda_aware_check()
{
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
 
    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
}


//Kernel that performs Matrix Vector Multiplication
__global__ void MatrixVectorMultiplication(float *Matrix,float *Vector,float *Solution, int RowsNo, int ColsNo, int RowsNo2, int ColsNo2, int VectorLength, int ScatterSize, int ThreadDim, int MyRank, int NumberofProcessors)
{  	
    int tidx = threadIdx.x;
   
    int count,ThreadColumnIndex,pass = 0 ;
    float TempResult = 0.0f;
   
    for (int i = 0; i < RowsNo / NumberofProcessors; i++) {
        for (tidx = 0; tidx < ColsNo2; tidx++) {
            float sum = 0.0;
            for (int k = 0; k < RowsNo2; k++)
                sum = sum + Matrix[i * ColsNo + k] * Vector[k * ColsNo2 + tidx];

            Solution[i * ColsNo2 + tidx] = sum;
        }
	}

    __syncthreads();
}

int main(int argc, char **argv)
{

	int MyRank, NumberOfProcessors;
	int Status = 1;
	float *MatrixA, *VectorB, *ResultVector, *MatrixB, *ResultMatrix;
	float *MyMatrixA, *MyResultMatrix;
	float *DeviceRootMatrixA, *DeviceRootResultVector;
	float *DeviceMyMatrixA, *DeviceMyResultVector, *DeviceVectorB, *DeviceMatrixB, *CPUResultVector;
	int RowsNo, ColsNo, RowsNo2, ColsNo2, VectorSize, ScatterSize, IndexCol, IndexValue, DeviceStatus;
	int matrixBsize, pinned;
	int print = 0;
	int verify = 0;

	//MPI Intialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);

	cuda_aware_check();

	double start_time = MPI_Wtime();

	//Checking if valid number of arguements have been passed
	if(argc < 5)
	{
		if(MyRank == 0)
			printf("Usage:\n"
			"mpirun -np <# processors> <./executable> "
			"<# rows of matrix A> <# columns of matrix A> "
			"<# rows of matrix B> <# columns of matrix B> "
			" <-v if verification is required> "
			" <-p if print is required>\n");
			
		MPI_Finalize();
		exit(-1);
	}
	if ((argc >= 6 && strcmp(argv[5],"-v") == 0) || (argc >= 7 && strcmp(argv[6],"-v") == 0))
		verify = 1;

	if ((argc >= 6 && strcmp(argv[5],"-p") == 0) || (argc == 7 && strcmp(argv[6],"-p") == 0)) 
		print = 1;
	
	//Assigning values to RowsNo, ColsNo, VectorSize from the arguements passed
	RowsNo = atoi( argv[1] );
	ColsNo = atoi( argv[2] );
	RowsNo2= atoi( argv[3] );
	ColsNo2= atoi( argv[4] );
	
	matrixBsize = RowsNo2 * ColsNo2;
	if (MyRank==0)
		printf("\nResultant Matrix Number of Elements is %d\n\n", matrixBsize);

	int elements;

	//Checking if columns is equal to vector size
	if( ColsNo != RowsNo2)
	{
		if(MyRank == 0)
			printf("Entered wrong input, Number of columns of matrix should be equal to number of rows \n");
		MPI_Finalize();
		exit(-1);
	}

	if(RowsNo < NumberOfProcessors)
	{
		if(MyRank == 0)
			printf("Given number of Rows of the matrix should be more than number of processors \n");
		MPI_Finalize();
		exit(-1);
	}

	//Checking if Matrix can be distributed evenly to all the nodes
	if(RowsNo % NumberOfProcessors != 0)
	{
		if(MyRank == 0)
			printf("The Rows of the matrix can not be distributed evenly among processors \n");
		MPI_Finalize();
		exit(-1);
	}

	//Calculating the Scatter size of the Matrix
	ScatterSize = RowsNo / NumberOfProcessors;

	elements = (RowsNo * ColsNo2) / NumberOfProcessors;

	//Root node intializes the Matrix, Vector and Result Vector
	if(MyRank == 0) {
		Status = IntializingMatrixVectors(&MatrixA, &MatrixB, &ResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2);
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceRootMatrixA, RowsNo * ColsNo * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceRootResultVector, RowsNo * ColsNo2 * sizeof(float) ) );
		cudaMemcpy( (void *)DeviceRootMatrixA, (void *)MatrixA, RowsNo * ColsNo * sizeof(float), cudaMemcpyHostToDevice );
	}
	printf("[%s:%i]Entering MPI Barrier...\n", __FILE__, __LINE__);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("[%s:%i]broadcast status start...\n", __FILE__, __LINE__);
	MPI_Bcast(&Status, 1, MPI_INT, 0, MPI_COMM_WORLD);

	printf("[%s:%i]broadcast status finished...\n", __FILE__, __LINE__);

	//Allocating the Memory on the device memory
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyMatrixA, ScatterSize * ColsNo * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMatrixB, matrixBsize*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyResultVector, elements * sizeof(float) ) );

	if(MyRank == 0) {
		printf("Copy Matrix B on root node...");
		cudaMemcpy( (void *)DeviceMatrixB, (void *)MatrixB, matrixBsize * sizeof(float), cudaMemcpyHostToDevice );
	}
	printf("[%s:%i]broadcast Matrix B start...\n", __FILE__, __LINE__);
	//Broad casting the Vector to all the nodes from root node
	MPI_Bcast(DeviceMatrixB, matrixBsize, MPI_FLOAT, 0, MPI_COMM_WORLD);
	printf("[%s:%i]broadcast Matrix B finished...\n", __FILE__, __LINE__);

	//Distributing the Matrix among to all the nodes
	MPI_Scatter(DeviceRootMatrixA, ScatterSize * ColsNo, MPI_FLOAT, DeviceMyMatrixA, ScatterSize * ColsNo, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Calling the kernel which performs Matrix Vector Product
	MatrixVectorMultiplication<<<1, 256>>>(DeviceMyMatrixA, DeviceMatrixB, DeviceMyResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2, ColsNo, ScatterSize, BLOCKSIZE, MyRank, NumberOfProcessors);	

	MPI_Barrier(MPI_COMM_WORLD);
	
	//Root processor gathering from all nodes to get the final result vector
	MPI_Gather(DeviceMyResultVector, elements, MPI_FLOAT, DeviceRootResultVector, elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (MyRank == 0)
		cudaMemcpy( (void *)ResultVector, (void *)DeviceRootResultVector, RowsNo * ColsNo2 * sizeof(float), cudaMemcpyDeviceToHost );
	
	MPI_Barrier(MPI_COMM_WORLD);

	//To verify:
	//Compute on CPU
	if (MyRank == 0 && verify == 1){
		CPUResultVector = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
		for (int i = 0; i < RowsNo; i++) {
			for (int j = 0; j < ColsNo2; j++) {
				float sum = 0.0;
				for (int k = 0; k < RowsNo2; k++)
					sum = sum + MatrixA[i * ColsNo + k] * MatrixB[k * ColsNo2 + j];
		
				CPUResultVector[i * ColsNo2 + j] = sum;
			}
		}
		int flag = 1;
		for(int i = 0; i < ColsNo2 * RowsNo; i++) {
			float a = ResultVector[i];
			float b = CPUResultVector[i];
			if (fabs(a-b) > 0.01) {
				printf("Error in computation and values are %f and %f",ResultVector[i], CPUResultVector[i]);
				flag = 0;
			}
		}
		if (flag)
			printf("\nVerification Passed\n\n");
		free(CPUResultVector);
	}

	//Root processor printing the resultant vector if print specified
	if(MyRank == 0 && print == 1)
	{
		printf("The resultant vector with size %d is \n",RowsNo*ColsNo2);
		for(int i = 0; i < ColsNo2 * RowsNo; i++)
			printf(" %f \n", ResultVector[i]);

	}

	if (MyRank == 0) {
		printf("\n\n Computation Done .....\n Exiting \n\n");
		//freeing the Vectors allocated by the root node
		free(MatrixA);
		free(MatrixB);
		free(ResultVector);
		CUDA_SAFE_CALL( cudaFree( DeviceRootMatrixA ) );
		CUDA_SAFE_CALL( cudaFree( DeviceRootResultVector ) );
	}	
	
	
	//Freeing the device memory
	CUDA_SAFE_CALL( cudaFree( DeviceMyMatrixA ) );
	CUDA_SAFE_CALL( cudaFree( DeviceMatrixB ) );
	CUDA_SAFE_CALL( cudaFree( DeviceMyResultVector ) );

	if (MyRank==0) {
		double end_time = MPI_Wtime();
		std::cout << "Running time:" << end_time-start_time << "s" << std::endl;
	}

	MPI_Finalize();

	return(0);
}

int IntializingMatrixVectors(float **MatrixA, float **MatrixB, float **ResultVector, int RowsNo, int ColsNo, int RowsNo2, int ColsNo2)
{
	float *TempMatrixA, *TempVectorB, *TempResultVector, *TempMatrixB;
	int Status;

	//Allocating memory on the host
	TempMatrixA = (float *)malloc(RowsNo * ColsNo * sizeof(float));
	if(TempMatrixA == NULL)
		Status = 0;
	TempMatrixB = (float *)malloc(RowsNo2 * ColsNo2 * sizeof(float));
	if(TempMatrixB == NULL)
		Status = 0;
	TempResultVector = (float *)malloc(RowsNo * ColsNo2 * sizeof(float));
	if(TempResultVector == NULL)
		Status = 0;

	//Intializing the Matrix and the Vectors
	srand(time(NULL));
	int a = 10;
	for(int i = 0; i < RowsNo * ColsNo; i++)
		TempMatrixA[i] = (float)rand()/(float)(RAND_MAX / a);
		
	printf("Matrix A initialized\n\n");		

	for(int i = 0; i < RowsNo2 * ColsNo2; i++)
		TempMatrixB[i] = (float)rand()/(float)(RAND_MAX / a);

	printf("Matrix B initilized\n\n");

	for(int i = 0; i < ColsNo2 * RowsNo; i++)
		TempResultVector[i] = 0.0f;

	*MatrixA = TempMatrixA;
	*MatrixB = TempMatrixB;
	*ResultVector = TempResultVector;
	
	return Status;
}
