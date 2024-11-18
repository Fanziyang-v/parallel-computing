#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    int private_num = 10213;
    int n = 5;
#pragma omp parallel num_threads(n) private(private_num)
    {
#pragma omp for
        for (int i = 0; i < n; i++)
            cout << "Thread number is " << omp_get_thread_num() << endl;
        private_num = omp_get_thread_num();
    }
    cout << "private_num = " << private_num << endl;
    return 0;
}