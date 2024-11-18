#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
#pragma omp parallel
    {
#pragma omp single
        {
            cout << "A ";
#pragma omp task
            {
                cout << "race ";
            }
#pragma omp task
            {
                cout << "car ";
            }
#pragma omp taskwait
            cout << "is fun to watch ";
        }
        /*
         * Result 1: A race car is fun to watch.
         * Result 2: A car race is fun to watch.
         */
    }
    cout << endl;
}