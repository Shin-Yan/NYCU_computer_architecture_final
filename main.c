#include <stdio.h>
#include <stdlib.h>
#include "matrix_cpu.h"

int main(){
    Matrix *m1 = InitMatrix(256,128), *m2 = InitMatrix(128, 64), *m;
    double* bias = (double*)malloc(sizeof(double) * 256);
    generateRandomVector(bias,2);
    // dumpVector(bias, 2);
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