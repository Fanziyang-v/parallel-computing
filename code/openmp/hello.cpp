#include <iostream>
#include <omp.h>

using namespace std;

void hello(void);

int main(int argc, char *argv[])
{
    /* Get number of threads from command line argument. */
    int thread_count = strtol(argv[1], NULL, 10);
#pragma omp parallel num_threads(thread_count)
    hello();
    return 0;
}

void hello(void)
{
    int rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    cout << "Hello from thread " << rank << " of " << thread_count << endl;
}