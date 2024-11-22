#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <fstream>

int NUM_THREADS = 12;

void serial_histogram(float *array, int n, int *bins, int num_bins)
{
    int i;
    /* Counting */
    int idx;
    for (i = 0; i < n; i++)
    {
        int val = (int)array[i];
        if (val == num_bins)
        { /* Ensure 10 numbers go to the last bin */
            idx = num_bins - 1;
        }
        else
        {
            idx = val % num_bins;
        }
        bins[idx]++;
    }
}

void parallel_histogram(float *array, int n, int *bins, int num_bins)
{
    int i, j, id, sum;
    /* Counting */
    int idx;
#pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_bins[num_bins] = {0}; // Thread-private copy of bins
#pragma omp for private(i, idx)
        for (i = 0; i < n; i++)
        {
            int val = (int)array[i];

            if (val == num_bins)
            { /* Ensure numbers equal to num_bins go to the last bin */
                idx = num_bins - 1;
            }
            else
            {
                idx = val % num_bins;
            }
            thread_bins[idx]++;
        }
/* Merge the thread-private bins into the global bins */
#pragma omp critical
        for (i = 0; i < num_bins; i++)
        {
            bins[i] += thread_bins[i];
        }
    }
}

void generate_random_numbers(float *array, int n)
{
    int i;
    float a = 10.0;
    for (i = 0; i < n; ++i)
        array[i] = ((float)rand() / (float)(RAND_MAX)) * a;
}

bool check(int *bins1, int *bins2, int num_bins)
{
    for (int i = 0; i < num_bins; i++)
        if (bins1[i] != bins2[i])
            return false;

    return true;
}

int main(int argc, char *argv[])
{
    
    int n;
    int num_bins = 10;
    float *array;
    int *bins1, *bins2;
    double t1, t2;
    bins1 = (int *)calloc(num_bins, sizeof(int));
    bins2 = (int *)calloc(num_bins, sizeof(int));
    std::ifstream file("input2.txt");
    file >> n;
    std::cout << n << std::endl;
    array = (float *)malloc(sizeof(float) * n);
    // Serial Computing.
    double start = omp_get_wtime();
    generate_random_numbers(array, n);
    serial_histogram(array, n, bins1, num_bins);
    double end = omp_get_wtime();
    printf("Results of Serial Histogram\n");
    int i;
    for (i = 0; i < num_bins; i++)
    {
        printf("bins[%d]: %d\n", i, bins1[i]);
    }
    printf("Running time: %f seconds\n\n", (t2 = end - start));

    // Parallel computing.
    start = omp_get_wtime();
    parallel_histogram(array, n, bins2, num_bins);
    end = omp_get_wtime();
    printf("Results of Parallel Histogram\n");
    for (i = 0; i < num_bins; i++)
    {
        printf("bins[%d]: %d\n", i, bins2[i]);
    }
    printf("Running time: %f seconds\n", (t1 = end - start));
    std::cout << "Speedup: " << t2 / t1 << std::endl;

    // Sanity Check.
    if (check(bins1, bins2, num_bins))
        std::cout << "Sanity Check Pass!" << std::endl;
    else
        std::cout << "Sanity Check Fail!" << std::endl;


    std::ofstream out_file("output2.txt");
    for (i = 0; i < num_bins; i++)
    {
        out_file << bins2[i] << ",";
    }
    out_file.precision(2);
    out_file << std::fixed << t1 * 1000.0 << "," << t2 * 1000.0;
    free(array);
    free(bins1);
    free(bins2);
    return 0;
}
