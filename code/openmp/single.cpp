#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    const int thread_count = 5;
#pragma omp parallel num_threads(thread_count)
    {
#pragma omp single
        {
            int thread_id = omp_get_thread_num();
            cout << "Thread id = " << thread_id << endl;
            cout << "A " << "race " << "car" << endl;
        }
    }
    return 0;
}