#include <iostream>
#include <omp.h>

double func(double x)
{
    return 3 * x * x + 2 * x + 1;
}

double integral(double a, double b, int n, double (*f)(double))
{
    double sum = 0.0, start;
    double h = (b - a) / n;
    int rank, thread_count, m;

    rank = omp_get_thread_num();
    thread_count = omp_get_num_threads();
    m = n / thread_count;
    if (rank == thread_count - 1)
        m = n - (thread_count - 1) * m;

    start = a + rank * ((b - a) / thread_count);
    for (int i = 0; i < m; i++)
    {
        double x = start + i * h;
        sum += f(x);
    }
    sum *= h;
    return sum;
}

int main(int argc, char *argv[])
{
    double a, b, h;
    double result = 0.0;
    int n, thread_count;

    thread_count = strtol(argv[1], NULL, 10);
    std::cout << "Enter a, b and n" << std::endl;
    std::cin >> a >> b >> n;

    // V1: using critical directive.
    /* #pragma omp parallel num_threads(thread_count)
        {
            double local_result = integral(a, b, n, func);
    #pragma omp critical
            result += local_result;
        } */

    // V2: using reduction clause.
    /* #pragma omp parallel num_threads(thread_count) reduction(+ : result)
        result += integral(a, b, n, func); */

    // V3: using parallel for.
    h = (b - a) / n;
#pragma omp parallel for num_threads(thread_count) reduction(+ : result)
    for (int i = 0; i < n; i++)
        result += func(a + i * h);

    result *= h;

    std::cout << "With n = " << n << " trapezoids, our estimate of the integral from "
              << a << " to " << b << " = " << result << std::endl;
    return 0;
}
