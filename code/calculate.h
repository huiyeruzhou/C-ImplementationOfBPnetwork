#ifndef __CALCULATE_H__
#define __CALCULATE_H__
typedef struct 
{
    double* min;
    double* scaler;
}REMINMAX;


REMINMAX MinMaxScaler(double** input, int dataSize);//ÌØ»¯
int ij(int i, int col_x, int j);
void LinearCombine(double*weight, int row_weight, int column_weight,
double*node, double* biases, double *result);
void GetSigmoid(double* x,int data_num);
double SigmoidDeriv(double x);
void WeightDeriv(double learingRate, double delta, double OutNet);
void BiasDeriv(double learingRate, double* delta);
#endif//__CALCULATE_H__
