#include <stdio.h>
#include <stdlib.h>
#include "matrix_cpu.h"

int main(){
    Matrix *m1 = InitMatrix(10,20), *m2 = InitMatrix(20, 10), *m;
    float* bias = (float*)malloc(sizeof(float) * 10);
    generateRandomVector(bias,10);
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