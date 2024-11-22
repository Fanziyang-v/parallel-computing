#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
#include <math.h>

#define BLOCKSIZE 16
int IntializingMatrixVectors(float **, float **, float **, int , int , int , int);
int CheckDevice(int );

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
	float *DeviceMyMatrixA, *DeviceMyResultVector, *DeviceVectorB, *DeviceMatrixB, *CPUResultVector;
	int RowsNo, ColsNo, RowsNo2, ColsNo2, VectorSize, ScatterSize, IndexCol, IndexValue, DeviceStatus;
	int matrixBsize, pinned;
	int print = 0;
	int verify = 0;

	//MPI Intialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
	MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);
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

	//Root node intializes the Matrix, Vector and Result Vector
	if(MyRank == 0)
			Status = IntializingMatrixVectors(&MatrixA, &MatrixB, &ResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(&Status, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Allocating memory for the Vector by all nodes expect root node
	if(MyRank != 0)
		MatrixB = (float *)malloc(matrixBsize * sizeof(float));

	//Broad casting the Vector to all the nodes from root node
	MPI_Bcast(MatrixB, matrixBsize, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//Calculating the Scatter size of the Matrix
	ScatterSize = RowsNo / NumberOfProcessors;

	elements = (RowsNo * ColsNo2) / NumberOfProcessors;

	//Allocating the memory on the host for the MyMatrixA and MyResultVector by all nodes
	MyMatrixA = (float *)malloc(ScatterSize * ColsNo * sizeof(float) );
	if(MyMatrixA == NULL)
		Status = 0;

	MyResultMatrix = (float *)malloc(elements* sizeof(float));
	if(MyResultMatrix == NULL)
		Status = 0;
	
	//Distributing the Matrix among to all the nodes
	MPI_Scatter(MatrixA, ScatterSize * ColsNo, MPI_FLOAT, MyMatrixA, ScatterSize * ColsNo, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	DeviceStatus = CheckDevice(MyRank);

    if(DeviceStatus == 0) {
        printf("Processor with rank %d doing partial product of two vectors on CPU \n",MyRank);
		for(int i = 0 ; i < ScatterSize ; i++) {
			MyResultMatrix[i] =0;
			IndexValue = i * ColsNo;
			for(IndexCol = 0; IndexCol < ColsNo; IndexCol++) 
			MyResultMatrix[i] += (MyMatrixA[IndexValue++] * VectorB[IndexCol]);
        }
	}
	else { // do calculation on CPU
		
		//Allocating the Memory on the device memory
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyMatrixA, ScatterSize * ColsNo * sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMatrixB, matrixBsize*sizeof(float) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **)&DeviceMyResultVector, elements * sizeof(float) ) );

		//Copying the data from host to device
		cudaMemcpy( (void *)DeviceMyMatrixA, (void *)MyMatrixA, ScatterSize * ColsNo * sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( (void *)DeviceMatrixB, (void *)MatrixB,  matrixBsize*sizeof(float), cudaMemcpyHostToDevice );

		//Calling the kernel which performs Matrix Vector Product
		MatrixVectorMultiplication<<<1, 256>>>(DeviceMyMatrixA, DeviceMatrixB, DeviceMyResultVector, RowsNo, ColsNo, RowsNo2, ColsNo2, ColsNo, ScatterSize, BLOCKSIZE, MyRank, NumberOfProcessors);	
			
		//Copying the value of patial result vector from device to host
		cudaMemcpy( (void *)MyResultMatrix, (void *)DeviceMyResultVector, elements * sizeof(float), cudaMemcpyDeviceToHost );
	}        
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	//Root processor gathering from all nodes to get the final result vector
	MPI_Gather(MyResultMatrix,elements, MPI_FLOAT, ResultVector, elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

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
		free(ResultVector);
	}

	//Freeing the host memory	
	free(MyMatrixA);
	free(MatrixB);
	free(MyResultMatrix);
	
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

int CheckDevice(int MyRank)
{
	int DeviceCount, Device;
	struct cudaDeviceProp Properties;

	cudaGetDeviceCount(&DeviceCount);
	if(DeviceCount >= 1)
	{
		cudaGetDevice(&Device);
		cudaGetDeviceProperties(&Properties, Device);
		printf("Processor with rank %d has the Device by name %s and computation is done on this device \n",MyRank, Properties.name);
	}

	return(DeviceCount);
}
