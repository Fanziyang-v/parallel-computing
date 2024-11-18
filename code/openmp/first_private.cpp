#include <iostream>
#include <sstream>
#include <omp.h>

using namespace std;

int main()
{
    int private_num = 10213;
    int n = 5;
#pragma omp parallel num_threads(n) firstprivate(private_num)
    {
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            stringstream ss;
            ss << "Thread number = " << omp_get_thread_num() << endl;
            ss << "Init private num = " << private_num << endl;
            private_num = omp_get_thread_num();
            ss << "After modification, private num = " << private_num << endl;
            cout << ss.str();
        }
    }
    cout << "private_num = " << private_num << endl;
    return 0;
}