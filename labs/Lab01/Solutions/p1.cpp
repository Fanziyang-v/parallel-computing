#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

// Global variables.
int N;
double **A, **B, **C, **D;

// Initialization.
void initialize()
{
    std::ifstream file("input1.txt");
    file >> N;
    file.close();
    std::cout << "N=" << N << std::endl;
    // Initialize matrix A, B and C
    A = new double *[N];
    B = new double *[N];
    C = new double *[N];
    D = new double *[N];
    for (int i = 0; i < N; i++)
    {
        A[i] = new double[N];
        B[i] = new double[N];
        C[i] = new double[N];
        D[i] = new double[N];
        for (int j = 0; j < N; j++)
        {
            A[i][j] = j * 1;
            B[i][j] = i * j + 2;
            D[i][j] = C[i][j] = j - i * 2;
        }
    }
}

// Sanity Check.
bool check()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (C[i][j] != D[i][j])
                return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    int i, j, k, id;
    double sum;
    double start, end, t1, t2;
    initialize();

    // Parallel computing.
    start = omp_get_wtime(); // start time measurement
#pragma omp parallel for num_threads(12) private(i, j, k, sum)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    end = omp_get_wtime(); // end time measurement

    std::cout << "Time of computation(Parallel): " << (end - start) << " seconds." << std::endl;
    t1 = end - start;

    // Serial Computing.
    start = omp_get_wtime(); // start time measurement
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            D[i][j] = sum;
        }
    }
    end = omp_get_wtime(); // end time measurement
    std::cout << "Time of computation(Serial): " << (end - start) << " seconds." << std::endl;
    t2 = end - start;

    // Ouput the result.
    std::fstream out_file("output1.txt");
    out_file.precision(2);
    out_file << std::fixed << t1 * 1000.0 << "," << t2 * 1000.0;
    std::cout << "Speedup=" << t2 / t1 << std::endl;
    if (check())
        std::cout << "The result of parallel computing and serial computing for matrix multiplication are the same." << std::endl;
    else
        std::cout << "Sanity Check Fail!!!" << std::endl;
    return 0;
}
