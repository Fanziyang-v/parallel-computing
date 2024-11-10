#include <iostream>
#include <omp.h>

int main(int argc, char *argv[])
{
    double sum = 0.0, factor;
    int n, thread_count;

    thread_count = strtol(argv[1], NULL, 10);

    std::cout << "Enter n: " << std::endl;
    std::cin >> n;

#pragma omp parallel for num_threads(thread_count) private(factor) reduction(+ : sum)
    for (int k = 0; k < n; k++)
    {
        factor = k % 2 == 0 ? 1.0 : -1.0;
        sum += factor / (2 * k + 1);
    }

    sum *= 4.0;
    std::cout << "With n = " << n << ", the estimate value of PI is " << sum << std::endl;
    return 0;
}