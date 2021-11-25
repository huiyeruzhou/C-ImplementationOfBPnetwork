#ifndef __NETWORK_H__
#define __NETWORK_H__
#include <stdio.h>
typedef struct 
{
    //初始数据
    int epoch;
    int batchSize;
    int layer;
    int dataSize;
    double** Input;
    double scaler;
    double learningRate;
    FILE* inputFile;
    FILE* resultFile;
    FILE* errorFile;
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
void TrainNetwork(BPnetwork* network);
void Backward(BPnetwork* network);
void Forward(BPnetwork* network);
void InitDeriv(BPnetwork* network);
void NetworkLearn(BPnetwork* network);
void PrintResult(BPnetwork* network);
void PrintError(BPnetwork* network);
void test(BPnetwork *network);

#endif//__NETWORK_H__