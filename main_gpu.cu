#include <stdio.h>
#include <stdlib.h>
#include "matrix_gpu.h"

int main(){
    Matrix *m1 = InitMatrix(2,4), *m2 = InitMatrix(4, 2), *m;
    float* bias = (float*)malloc(sizeof(float) * 2);
    generateRandomVector(bias, 2);
    dumpVector(bias, 2);
    generateRandomMatrix(m1);
    generateRandomMatrix(m2);
    dumpMatrix(m1);
    dumpMatrix(m2);
    m = matrixMultiplyAddBiasActivation(m1,m2,bias);
    if(m){
        // printf("dumping multiplied matrix\n");
        // dumpMatrix(m);
        MatrixFree(m);
    }
    MatrixFree(m1);
    MatrixFree(m2);
    free(bias);
    return 0;
}
