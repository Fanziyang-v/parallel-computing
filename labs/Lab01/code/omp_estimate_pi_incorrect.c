#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    double factor = 1.0;
    double sum = 0.0;
    int n = 100000;

    double pi_approx;
    int k;

    int thread_count; 
    thread_count = strtol(argv[1], NULL, 10);
    int my_rank;

#   pragma omp parallel for num_threads(thread_count) reduction(+:sum)
    for (k=0; k < n; k++){
        sum += factor/(2*k+1);
        factor = -factor;
        my_rank = omp_get_thread_num();

        printf("Factor is modified to %f by thread %d \n", factor, my_rank);
    }
    pi_approx = 4.0*sum; 
    printf("Approximation value: %f\n", pi_approx);

    return 0;
}
