#ifndef __INPUT_H__
#define __INPUT_H__
#include <stdio.h>
#define LAYER 1
#define EPOCH 10000
#define DATASIZE 4
#define BATCHSIZE 1
#define INITBIAS 0.01
#define INPUTFILE "..\\input.txt"
#define RESULTFILE "..\\output.txt"
#define ERRORFILE "..\\error.csv"
#define LEARNINGRATE 0.1
static int layerNodes[LAYER + 2] = {2,4,1};


void GetInput(FILE* inputFile, double** inputData);


#endif//__INPUT_H__
