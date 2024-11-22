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

#   pragma omp parallel for num_threads(thread_count) reduction(+:sum) private(factor)
    for (k=0; k < n; k++){
        if (k % 2 == 0)
            factor = 1.0;
        else
            factor = -1.0;
        sum += factor/(2*k+1);        
    }
    pi_approx = 4.0*sum; 
    printf("Approximation value: %f\n", pi_approx);

    return 0;
}
