#include <stdlib.h>
#include <stdio.h>
#include "input.h"
#include "network.h"
#include "calculate.h"

void TrainNetwork(BPnetwork *network)
{
    network->scaler = MinMaxScaler(network->Input, network->dataSize);
    InitDeriv(network);
    for(int j = 0 ; j < network->epoch; j ++)
    for (int i = 0; i < network->dataSize; i++)
    {
        InitTrain(network, i);
        Forward(network);
        Backward(network);
        fprintf(network->resultFile,"NO.%d in %d rounds",i + 1, j + 1);
        PrintResult(network);
        if (i + 1 && !(i + 1 % BATCHSIZE))
        {
            fprintf(network->errorFile,"%d,",j * network->dataSize + i + 1);
            PrintError(network);
            NetworkLearn(network);
            InitDeriv(network);
        }
        
    }
    fclose(network->resultFile);
    fclose(network->errorFile);

}
void InitNetwork(BPnetwork *network)
{
    
    //���ļ�
    errno = 0;
    network->inputFile = fopen(INPUTFILE, "r");
    if(!network->inputFile)
    {
        perror("Error in open inputFile");
        exit(-1);
    }
    network->resultFile = fopen(RESULTFILE, "w");
    if(!network->resultFile)
    {
        perror("Error in open resultFile");
        exit(-1);
    }
    network->errorFile = fopen(ERRORFILE, "w");
    if(!network->errorFile)
    {
        perror("Error in open errorFile");
        exit(-1);
    }
    fprintf(network->errorFile, "error,rounds\n");

    //Ϊ��������ʵ�帳ֵ
    network->e = 0;
    network->epoch = EPOCH;
    network->layer = LAYER;
    network->dataSize = DATASIZE;
    network->batchSize = BATCHSIZE;
    network->learningRate = LEARNINGRATE;

    //Ϊ����ָ�����洢�ռ�
    //ѵ������
    network->Input = (double **)malloc(DATASIZE * sizeof(double *));
    for (int i = 0; i < DATASIZE; i++)
    {
        network->Input[i] = (double *)malloc((layerNodes[0] + layerNodes[LAYER + 1]) * sizeof(double));
    }
    GetInput(network->inputFile, network->Input);
    fclose(network->inputFile);


    //ƫ���������Ȩ�ؾ���
    //һ�������������+1��
    network->biases = (double **)malloc((LAYER + 1) * sizeof(double *));
    for (int i = 0; i < LAYER + 1; i++)
    {
        //ÿ���ƫ������������һ��Ľڵ���,����Ϊ1
        network->biases[i] = (double *)malloc(layerNodes[i + 1] * sizeof(double));
        for (int j = 0; j < layerNodes[i + 1]; j++)
        {
            network->biases[i][j] = INITBIAS;
        }
    }
    //Ȩ�ؾ���һ�������������+1��
    network->weights = (double **)malloc((LAYER + 1) * sizeof(double *));
    for (int i = 0; i < LAYER + 1; i++)
    {
        //ÿ���Ȩ�������Ǹò����ԵĽڵ����,��������һ�����ԵĽڵ����
        network->weights[i] = (double *)malloc(layerNodes[i] * layerNodes[i + 1] * sizeof(double));
        for (int j = 0; j < layerNodes[i] * layerNodes[i + 1]; j++)
        {
            network->weights[i][j] = 2 * rand() / (RAND_MAX + 1.0) - 1;
        }
    }

    //����źž���,��״��ƫ�þ�����ͬ
    network->deltas = (double **)malloc((LAYER + 1) * sizeof(double));
    for (int i = 0; i < LAYER + 1; i++)
    {
        network->deltas[i] = (double *)malloc(layerNodes[i + 1] * sizeof(double));
        for (int j = 0; j < layerNodes[i + 1]; j++)
        {
            network->deltas[i][j] = 0;
        }
    }

    //ƫ��΢�־���
    network->biasesDeriv = (double **)malloc((LAYER + 1) * sizeof(double *));
    for (int i = 0; i < LAYER + 1; i++)
    {
        network->biasesDeriv[i] = (double *)malloc(layerNodes[i + 1] * sizeof(double));
        for (int j = 0; j < layerNodes[i + 1]; j++)
        {
            network->biases[i][j] = 0;
        }
    }
    //Ȩ��΢�־���
    network->weightDeriv = (double **)malloc((LAYER + 1) * sizeof(double *));
    for (int i = 0; i < LAYER + 1; i++)
    {
        network->weightDeriv[i] = (double *)malloc(layerNodes[i] * layerNodes[i + 1] * sizeof(double));
        for (int j = 0; j < layerNodes[i] * layerNodes[i + 1]; j++)
        {
            network->weights[i][j] = rand() / (RAND_MAX + 1.0);
        }
    }

    //�����������ŵ������ֵ��Ԥ�ڵĲ�ֵ
    network->errors = (double *)malloc(layerNodes[LAYER + 1] * sizeof(double));

    network->nodes = (double **)malloc((LAYER + 2) * sizeof(double *));
    for (int i = 0; i < LAYER + 2; i++)
    {
        network->nodes[i] = (double *)malloc(
            2 * layerNodes[i] * sizeof(double));
        for (int j = 0; j < 2 * layerNodes[i]; j++)
        {
            network->nodes[i][j] = 0;
        }
    }
}

void InitTrain(BPnetwork *network, int numOfData)
{
    int i; 
    for (i = 0; i < layerNodes[0]; i++)
    {
        network->nodes[0][2 * i + 1] = network->nodes[0][2 * i] = network->Input[numOfData][i]; 
    }
    for(int j = 0; j < layerNodes[LAYER + 1];i++, j ++)
    {
        network->errors[j] = -network->Input[numOfData][i];
    }
}

void Forward(BPnetwork *network)
{
    for (int l = 0; l < LAYER + 1; l++)
    {
        //����
        LinearCombine(network->weights[l], layerNodes[l + 1], layerNodes[l], network->nodes[l], network->biases[l], network->nodes[l + 1]);
        //����
        GetSigmoid(network->nodes[l + 1], layerNodes[l + 1]);
    }
    for (int i = 0; i < layerNodes[LAYER + 1]; i++)
    {
        //����������
        network->errors[i] += network->nodes[LAYER + 1][2 * i + 1];
        //���������
        network->e += 0.5 * network->errors[i] * network->errors[i];
    }
}
void Delta(BPnetwork *network)
{
    for (int i = 0; i < layerNodes[LAYER + 1]; i++)
    {
        network->deltas[LAYER][i] = 0;
        network->deltas[LAYER][i] = (network->errors[i]) * (SigmoidDeriv(network->nodes[LAYER + 1][2 * i + 1]));
    }
    for (int l = LAYER - 1; l >= 0; l--)
    {
        for (int i = 0; i < layerNodes[l + 1]; i++)
        {
            network->deltas[l][i] = 0;
            for (int k = 0; k < layerNodes[l + 2]; k++)
                network->deltas[l][i] += network->deltas[l + 1][k] *
                                         network->weights[l + 1][k * layerNodes[l + 1] + i];
            network->deltas[l][i] *= SigmoidDeriv(network->nodes[l + 1][2 * i + 1]);
        }
    }
}
void Backward(BPnetwork *network)
{
    Delta(network);
    for (int l = 0; l < LAYER + 1; l++)
    {
        for (int i = 0; i < layerNodes[l + 1]; i++)
        {
            for (int j = 0; j < layerNodes[l]; j++)
            {
                network->weightDeriv[l][i * layerNodes[l] + j] +=
                    network->deltas[l][i] * network->nodes[l][2 * j + 1];
            }
            network->biasesDeriv[l][i] += network->deltas[l][i];
        }
    }
}
void InitDeriv(BPnetwork *network)
{
    network->e = 0;
    for (int i = 0; i < LAYER + 1; i++)
    {
        for (int j = 0; j < layerNodes[i + 1]; j++)
        {
            network->biasesDeriv[i][j] = 0;
        }
    }
    for (int i = 0; i < LAYER + 1; i++)
    {
        for (int j = 0; j < layerNodes[i] * layerNodes[i + 1]; j++)
        {
            network->weightDeriv[i][j] = 0;
        }
    }
}
void NetworkLearn(BPnetwork *network)
{
    for (int l = 0; l < LAYER + 1; l++)
    {
        for (int i = 0; i < layerNodes[l + 1]; i++)
        {
            for (int j = 0; j < layerNodes[l]; j++)
            {
                network->weights[l][i * layerNodes[l] + j] -=
                    1./network->batchSize * network->learningRate * network->weightDeriv[l][i * layerNodes[l] + j];
            }
            network->biases[l][i] -= 1./network->batchSize * network->learningRate * network->biasesDeriv[l][i];
        }
    }
}
void test(BPnetwork *network)
{
   static double w1[4] = {0.15, 0.20, 0.25, 0.30};
    *network->weights = w1;
   static  double w2[4] = {0.40, 0.45, 0.50, 0.55};
    *(network->weights + 1) = w2;
   static  double b1[2] = {0.35, 0.35};
    static double b2[2] = {0.60, 0.60};
    *network->biases = b1;
    *(network->biases + 1) = b2;
   static  double n0[4] = {0.05, 0, 0.10, 0};
    network->nodes[0] = n0;
   static  double e[2] = {-0.01, -0.99};
    network->errors = e;
}
void PrintError(BPnetwork* network)
{
    fprintf(network->errorFile, "%lf\n",network->e);
}
void PrintResult(BPnetwork* network)
{
    fprintf(network->resultFile,"\nthe input data is : ");
    for(int i = 0; i < layerNodes[0]; i ++)
    {
        fprintf(network->resultFile,"%.2lf ",network->scaler * network->nodes[0][2 * i]);
    }
    fprintf(network->resultFile,"\nthe output data is : ");
    for(int i = 0; i < layerNodes[LAYER + 1]; i ++)
    {
        fprintf(network->resultFile,"%.2lf ", network->nodes[LAYER + 1][2* i + 1]);
    }
    fprintf(network->resultFile,"\nerror = %lf\n", network->e);
}