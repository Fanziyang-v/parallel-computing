#include <iostream>
#include <sstream>
#include <omp.h>

using namespace std;

int main()
{
    const int thread_count = 4;
    int a = 5;
#pragma omp parallel num_threads(thread_count) firstprivate(a)
    {
        stringstream ss;
        a = omp_get_thread_num();
        ss << "Thread id = " << omp_get_thread_num() << ", a = " << a << endl;
        cout << ss.str();
    }
    cout << "a = " << a << endl;
    return 0;
}