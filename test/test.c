#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "..\\code\\input.h"
int main()
{
    FILE *test = fopen(INPUTFILE, "w");
    int cnt = 0;
    double a,b;
    while (cnt < DATASIZE)
    {
        
        for(int i = 0; i < DATASIZE; i++)
        fprintf(test,"0 0 0\n0 1 1\n1 0 1\n1 1 0\n");
        cnt++;
    }
    fclose(test);
    return 0;
}