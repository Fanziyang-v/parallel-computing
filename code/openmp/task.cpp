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
            cout << "is fun to watch ";
        }
        /*
         * Result 1: A is fun to watch race car.
         * Result 2: A is fun to watch car race.
         */
    }
    cout << endl;
}