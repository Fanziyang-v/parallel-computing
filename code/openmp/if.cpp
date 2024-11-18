#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    const int n = 4;
#pragma omp parallel for if (n % 2 == 0)
    for (int i = 0; i < n; i++)
        cout << "Thread number is " << omp_get_thread_num() << endl;
    return 0;
}