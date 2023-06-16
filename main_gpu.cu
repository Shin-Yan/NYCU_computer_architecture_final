#include <stdio.h>
#include <stdlib.h>
#include "matrix_gpu.h"

int main(){
    Matrix *m1 = InitMatrix(10,1000000), *m2 = InitMatrix(1000000, 4), *m;
    double* bias = (double*)malloc(sizeof(double) * 10);
    generateRandomMatrix(m1);
    generateRandomMatrix(m2);
    // dumpMatrix(m1);
    // dumpMatrix(m2);
    m = matrixMultiplyAddBiasActivation(m1,m2,bias);
    if(m){
        // printf("dumping multiplied matrix\n");
        dumpMatrix(m);
        MatrixFree(m);
    }
    MatrixFree(m1);
    MatrixFree(m2);
    return 0;
}