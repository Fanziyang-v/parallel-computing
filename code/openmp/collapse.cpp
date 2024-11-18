#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                cout << "Thread number is " << omp_get_thread_num() << endl;
}
