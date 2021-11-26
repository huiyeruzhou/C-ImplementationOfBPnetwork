#ifndef __INPUT_H__
#define __INPUT_H__
#include <stdio.h>
//基本参数:隐含层数/每层节点数
#define LAYER 1
static int layerNodes[LAYER + 2] = {2,2,1};

//基本参数:训练数据量,测试数据量
#define DATASIZE 3
#define TESTSIZE 1

//训练参数:总迭代次数,批次大小,学习率,初始偏置值
#define EPOCH 50000
#define BATCHSIZE 3
#define LEARNINGRATE 0.04
#define INITBIAS 0.1

//输入输出参数:打印结果的频率,打印累计均方差的频率
#define PRT_RESULT 1000
#define PRT_ERROR 100
#define INPUTFILE "..\\input.txt"
#define RESULTFILE "..\\output.txt"
#define ERRORFILE "..\\error.csv"
#define TESTFILE "..\\test.txt"
#define CHECKFILE "..\\check.csv"




void GetInput(FILE* inputFile, double** inputData);


#endif//__INPUT_H__
