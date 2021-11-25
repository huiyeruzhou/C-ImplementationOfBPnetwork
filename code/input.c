#include "input.h"
#include <stdio.h>
void GetInput(FILE* inputFile, double** inputData)
{
    int inputSize = 0;
    while(inputSize < DATASIZE)
    {
        for(int i = 0; i < layerNodes[0] + layerNodes[LAYER + 1]; i ++)
            fscanf(inputFile,"%lf", inputData[inputSize] + i);
       inputSize++;
    }
}
void Output(FILE* outputFile, double** outputData)
{
    ; 
}
