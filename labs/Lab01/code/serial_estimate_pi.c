#include <stdio.h>


int main(int argc, char* argv[]) {
    double factor = 1.0;
    double sum = 0.0;
    int n = strtol(argv[1], NULL, 10);
    double pi_approx;
    int k;

    for (k=0; k < n; k++){
        sum += factor/(2*k+1);
        factor = -factor;
    }
    pi_approx = 4.0*sum; 
    printf("Approximation value: %f\n", pi_approx);

    return 0;
}
