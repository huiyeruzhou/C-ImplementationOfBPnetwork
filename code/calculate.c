#include <stdlib.h>
#include <math.h>
#include "input.h"
#include "calculate.h"
#define E 2.718281828

REMINMAX MinMaxScaler(double **input, int dataSize)
{
    double *min = (double *)malloc((layerNodes[0] + layerNodes[LAYER + 1]) * sizeof(double));
    double *max = (double *)malloc((layerNodes[0] + layerNodes[LAYER + 1]) * sizeof(double));
    double *scaler = (double *)malloc((layerNodes[0] + layerNodes[LAYER + 1]) * sizeof(double));

    for (int j = 0; j < layerNodes[0] + layerNodes[LAYER + 1]; j++)
    {
        max[j] = min[j] = input[0][j];
        for (int i = 0; i < dataSize; i++)
        {
            if (input[i][j] < min[j])
            {
                min[j] = input[i][j];
            }
            else if (input[i][j] > max[j])
            {
                max[j] = input[i][j];
            }
        }
        scaler[j] = max[j] - min[j];
    }
    for (int i = 0; i < dataSize; i++)
    {
        for (int j = 0; j < layerNodes[0] + layerNodes[LAYER + 1]; j++)
            input[i][j] = (input[i][j] - min[j]) /scaler[j] ;
    }
    REMINMAX reMinMax;
    reMinMax.min = min;
    reMinMax.scaler = scaler;
    return reMinMax;
}
double Mij(double *x, int col_x, int i, int j)
{
    return *(x + i * col_x + j);
}
void LinearCombine(double *weight, int row_weight, int column_weight,
                   double *node, double *biases, double *result)
{
    for (int i = 0; i < row_weight; i++)
    {
        result[2 * i] = 0;
        for (int j = 0; j < column_weight; j++)
        {
            result[2 * i] += weight[i * column_weight + j] * node[2 * j + 1];
        }
        result[2 * i] += biases[i];
    }
}
void GetSigmoid(double *x, int data_num)
{
    for (int i = 0; i < data_num; i++)
    {
        x[2 * i + 1] = 1. / (1. + pow(E, -x[2 * i]));
    }
}
double SigmoidDeriv(double x)
{
    return x * (1 - x);
}
