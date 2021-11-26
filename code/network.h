#ifndef __NETWORK_H__
#define __NETWORK_H__
#include <stdio.h>
#include "calculate.h"
typedef struct 
{
    //初始数据
    int epoch;
    int batchSize;
    int layer;
    int dataSize;
    double** input;
    REMINMAX reMinMax;
    double learningRate;
    FILE* inputFile;
    FILE* resultFile;
    FILE* errorFile;
    FILE* testFile;
    FILE* checkFile;
    //前向传播
    double** nodes;
    double** weights;
    double** biases;
    double* errors;
    double e;
    //反向传播
    double** deltas;
    double** weightDeriv;
    double** biasesDeriv;
}BPnetwork;

void InitNetwork(BPnetwork* network);
void InitTrain(BPnetwork* network, int numOfData);
void InitTest(BPnetwork *network);
void TrainNetwork(BPnetwork* network);
void TestNetwork(BPnetwork *network);
void Backward(BPnetwork* network);
void Forward(BPnetwork* network);
void InitDeriv(BPnetwork* network);
void NetworkLearn(BPnetwork* network);
void PrintResult(BPnetwork* network, FILE* resultFile);
void PrintError(BPnetwork* network, FILE* errorFile);
void test(BPnetwork *network);
void PrintTest(BPnetwork *network, int numOfTest);

#endif//__NETWORK_H__