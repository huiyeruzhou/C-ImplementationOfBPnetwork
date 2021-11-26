#ifndef __INPUT_H__
#define __INPUT_H__
#include <stdio.h>
//��������:��������/ÿ��ڵ���
#define LAYER 1
static int layerNodes[LAYER + 2] = {2,2,1};

//��������:ѵ��������,����������
#define DATASIZE 3
#define TESTSIZE 1

//ѵ������:�ܵ�������,���δ�С,ѧϰ��,��ʼƫ��ֵ
#define EPOCH 50000
#define BATCHSIZE 3
#define LEARNINGRATE 0.04
#define INITBIAS 0.1

//�����������:��ӡ�����Ƶ��,��ӡ�ۼƾ������Ƶ��
#define PRT_RESULT 1000
#define PRT_ERROR 100
#define INPUTFILE "..\\input.txt"
#define RESULTFILE "..\\output.txt"
#define ERRORFILE "..\\error.csv"
#define TESTFILE "..\\test.txt"
#define CHECKFILE "..\\check.csv"




void GetInput(FILE* inputFile, double** inputData);


#endif//__INPUT_H__
