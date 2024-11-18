#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
    const int N = 8;
    int x = 0;
#pragma omp parallel for lastprivate(x)
    for (int i = 0; i < N; i++)
        x = i;
    cout << "x = " << x << endl; // x = 7
    return 0;
}