#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

void SequentialCalculation(const int &n,
                           const int &m,
                           const std::vector<std::vector<int>> &A,
                           const std::vector<std::vector<int>> &B,
                           std::vector<std::vector<int>> *C)
{

  std::vector<std::vector<int>> B_power, next_B_power;
  std::vector<std::vector<int>> D;
  (*C) = A;
  B_power = B;
  int tmp;
  for (int t = 1; t <= m; t++)
  {
    D = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < n; k++)
        {
          D[i][j] = (D[i][j] + A[i][k] * B_power[k][j]) % 2;
        }
      }
    }
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        (*C)[i][j] = ((*C)[i][j] + D[i][j]) % 2;
      }
    }
    if (t == m)
      break;
    next_B_power = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        for (int k = 0; k < n; k++)
          next_B_power[i][j] = (next_B_power[i][j] + B_power[i][k] * B[k][j]) % 2;
      }
    }
    B_power = next_B_power;
  }
}

bool LoadFile(const std::string &input_file_path, int *n, int *m, std::vector<std::vector<int>> *A,
              std::vector<std::vector<int>> *B)
{
  std::ifstream fin(input_file_path.c_str());
  if (!fin.is_open())
  {
    return false;
  }
  fin >> (*n) >> (*m);
  *A = std::vector<std::vector<int>>(*n, std::vector<int>(*n, 0));
  *B = std::vector<std::vector<int>>(*n, std::vector<int>(*n, 0));
  for (int i = 0; i < (*n); i++)
    for (int j = 0; j < (*n); j++)
      fin >> (*A)[i][j];
  for (int i = 0; i < (*n); i++)
    for (int j = 0; j < (*n); j++)
      fin >> (*B)[i][j];
  fin.close();
  return true;
}

void TestAnswerCorrectness(const std::vector<std::vector<int>> &sequential_answer,
                           const std::vector<std::vector<int>> &parallel_answer)
{
  if (sequential_answer.size() != parallel_answer.size())
  {
    std::cout << "Error! The number of sequential_answer and parallel_answer "
                 "is not the same"
              << std::endl;
    return;
  }
  long long sum_sequential_answer = 0;
  long long sum_parallel_answer = 0;
  int sum_error = 0;
  for (uint i = 0; i < sequential_answer.size(); i++)
  {
    if (sequential_answer[i].size() != parallel_answer[i].size())
    {
      std::cout << "Error! The number of sequential_answer and parallel_answer "
                   "is not the same"
                << std::endl;
      return;
    }
    for (uint j = 0; j < sequential_answer[i].size(); j++)
    {
      sum_error += abs(sequential_answer[i][j] - parallel_answer[i][j]);
      sum_sequential_answer += sequential_answer[i][j];
      sum_parallel_answer += parallel_answer[i][j];
    }
  }
  std::cout << "sum_sequential_answer = " << sum_sequential_answer << std::endl;
  std::cout << "sum_parallel_answer = " << sum_parallel_answer << std::endl;

  if (sum_error > 0)
  {
    std::cout << "Wrong Answer" << std::endl;
  }
  else
  {
    std::cout << "Correct!!!" << std::endl;
  }
}

// ==============================================================
// ====    Write your functions below this line    ====
// ==============================================================
// ==============================================================

/* Matrix Multiplication and Mod 2 Kernel Function */
__global__ void MatrixMultMod2Kernel(int *d_M, int *d_N, int *d_P, int n)
{
  int num_blocks = gridDim.x;   /* number of thread blocks in grid. */
  int num_threads = blockDim.x; /* number of threads per thread block. */
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int total_num_threads = num_blocks * num_threads;

  int num_iterations = (int)ceil(n * n * 1.0 / total_num_threads);
  for (int t = 0; t < num_iterations; t++)
  {
    /* compute row and col of matrix P element to work on */
    int id = t * total_num_threads + block_id * num_threads + thread_id;
    int row = id / n;
    int col = id % n;
    if (row >= n)
      break;
    int sum = 0;
    for (int k = 0; k < n; k++)
      sum += d_M[row * n + k] * d_N[k * n + col];
    d_P[row * n + col] = sum % 2; /* mod 2 */
  }
}

__global__ void MatrixAddMod2Kernel(int *d_M, int *d_N, int *d_P, int n)
{
  int num_blocks = gridDim.x;   /* number of thread blocks in grid. */
  int num_threads = blockDim.x; /* number of threads per thread block. */
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int total_num_threads = num_blocks * num_threads;

  int num_iterations = (int)ceil(n * n * 1.0 / total_num_threads);
  for (int t = 0; t < num_iterations; t++)
  {
    /* compute row and col of matrix P element to work on */
    int id = t * total_num_threads + block_id * num_threads + thread_id;
    int row = id / n;
    int col = id % n;
    if (row >= n)
      break;
    // d_P[row * n + col] = (d_M[row * n + col] + d_N[row * n + col]) % 2;
    d_P[row * n + col] = d_M[row * n + col] ^ d_N[row * n + col];
  }
}

void ParallelCalculation(const int &n,
                         const int &m,
                         std::vector<std::vector<int>> &A,
                         std::vector<std::vector<int>> &B,
                         std::vector<std::vector<int>> &C,
                         const int &num_blocks,
                         const int &num_threads)
{
  int i, j, k;
  int numprocs, rank;
  MPI_Request *requests;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int *h_A, *h_B, *h_C, *h_B_Pwr, *h_buf;
  int *d_A, *d_B, *d_C, *d_B_Pwr, *d_buf;
  /* allocate host memory and device memory */
  h_A = new int[n * n];
  h_B = new int[n * n];
  h_C = new int[n * n];
  h_buf = new int[n * n];
  h_B_Pwr = new int[n * n];

  cudaMalloc((void **)&d_A, n * n * sizeof(int));
  cudaMalloc((void **)&d_B, n * n * sizeof(int));
  cudaMalloc((void **)&d_C, n * n * sizeof(int));
  cudaMalloc((void **)&d_B_Pwr, n * n * sizeof(int));
  cudaMalloc((void **)&d_buf, n * n * sizeof(int));
  cudaMemset(d_C, 0, n * n * sizeof(int));

  if (rank == 0)
  {
    /* Master code */
    requests = new MPI_Request[numprocs - 1];
    for (i = 0; i < n; i++)
    {
      memcpy(h_A + i * n, A[i].data(), n * sizeof(int));
      memcpy(h_B + i * n, B[i].data(), n * sizeof(int));
    }
    /* Broadcast Matrix A. */
    MPI_Bcast(h_A, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    /* Copy matrix elements from host to device */
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_Pwr, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    /* Compute B to the power of numprocs */
    for (i = 1; i < numprocs; i++)
    {
      /* Invoke kernel */
      MatrixMultMod2Kernel<<<num_blocks, num_threads>>>(d_B_Pwr, d_B, d_buf, n);
      /* Swap pointer between d_B_Pwr and d_buf */
      int *temp = d_buf;
      d_buf = d_B_Pwr;
      d_B_Pwr = temp;
      /* Send B^(i+1) to process i. */
      cudaMemcpy(h_B_Pwr, d_B_Pwr, n * n * sizeof(int), cudaMemcpyDeviceToHost);
      MPI_Isend(h_B_Pwr, n * n, MPI_INT, i, i, MPI_COMM_WORLD, &requests[i - 1]);
    }
    MPI_Waitall(numprocs - 1, requests, MPI_STATUSES_IGNORE);
    /* Broadcast h_B_Pwr */
    MPI_Bcast(h_B_Pwr, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    delete[] requests;
  }
  else
  {
    /* Broadcast Matrix A. */
    MPI_Bcast(h_A, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    /* Slave code: receive data from master process. */
    MPI_Recv(h_B, n * n, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* Broadcast h_B_Pwr */
    MPI_Bcast(h_B_Pwr, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    /* Copy matrix A, B and B_Pwr elements from host to device */
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_Pwr, h_B_Pwr, n * n * sizeof(int), cudaMemcpyHostToDevice);
  }

  for (int t = rank; t < m; t += numprocs)
  {
    /* Perform matrix multiplication */
    MatrixMultMod2Kernel<<<num_blocks, num_threads>>>(d_A, d_B, d_buf, n);
    /* Perform reduction */
    MatrixAddMod2Kernel<<<num_blocks, num_threads>>>(d_C, d_buf, d_C, n);
    /* Update matrix B by (B x B_Pwr) mod 2 */
    MatrixMultMod2Kernel<<<num_blocks, num_threads>>>(d_B_Pwr, d_B, d_buf, n);
    /* Swap pointer between d_B and d_buf */
    int *temp = d_buf;
    d_buf = d_B;
    d_B = temp;
    /* Now d_B is up-to-date. */
  }
  /* Now computation is over. Copy computation result to host memory. */
  cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  // perform reduction.
  MPI_Reduce(h_C, h_buf, n * n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0)
  {
    C.assign(n, std::vector<int>(n, 0));
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
        C[i][j] = h_buf[i * n + j] % 2;
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_buf;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_B_Pwr);
  cudaFree(d_buf);
}

// ==============================================================
// ====    Write your functions above this line    ====
// ==============================================================
// ==============================================================

int main(int argc, char **argv)
{
  int number_of_processes, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double parallel_start_time;

  int number_of_block_in_a_grid;
  int number_of_thread_in_a_block;
  int n, m;
  std::vector<std::vector<int>> A;
  std::vector<std::vector<int>> B;
  if (rank == 0)
  {
    if (argc < 4)
    {
      std::cout << "Error! Please use \"mpiexec -n [process number] "
                   "[--hostfile hostfile] multiple [number_of_block_in_a_grid] [number_of_thread_in_a_block] [data_file_name]\"\n";
      return 1;
    }
    else
    {
      number_of_block_in_a_grid = std::atoi(argv[1]);
      number_of_thread_in_a_block = std::atoi(argv[2]);
      std::string input_file_path = std::string(argv[3]);
      std::cout << "number_of_block_in_a_grid:" << number_of_block_in_a_grid << std::endl;
      std::cout << "number_of_thread_in_a_block:" << number_of_thread_in_a_block << std::endl;
      if (!LoadFile(input_file_path, &n, &m, &A, &B))
      {
        std::cout << "Error! Please check the format of input file\n";
        return 1;
      }
    }
  }
  std::vector<std::vector<int>> parallel_answer;

  if (rank == 0)
  {
    parallel_start_time = MPI_Wtime();
  }

  // ==============================================================
  // ====    Write your implementation below this line    ====
  // ==============================================================
  // ==============================================================

  /* Broadcast some params. */
  MPI_Bcast(&number_of_block_in_a_grid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&number_of_thread_in_a_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

  ParallelCalculation(n, m, A, B, parallel_answer, number_of_block_in_a_grid, number_of_thread_in_a_block);
  // ==============================================================
  // ====    Write your implementation above this line    ====
  // ==============================================================
  // ==============================================================
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
  {
    double parallel_end_time = MPI_Wtime();
    double parallel_running_time = parallel_end_time - parallel_start_time;
    std::cout << "parallel running time:" << parallel_running_time << std::endl;
    std::vector<std::vector<int>> sequential_answer;
    double sequential_start_time = MPI_Wtime();

    SequentialCalculation(n, m, A, B, &sequential_answer);
    double sequential_end_time = MPI_Wtime();
    double sequential_running_time =
        sequential_end_time - sequential_start_time;
    std::cout << "sequential running time:" << sequential_running_time
              << std::endl;
    std::cout << "speed up:" << sequential_running_time / parallel_running_time
              << std::endl;
    TestAnswerCorrectness(sequential_answer, parallel_answer);
  }
  MPI_Finalize();
  return 0;
}