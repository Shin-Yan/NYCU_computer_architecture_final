#include <stdio.h>
#include <stdlib.h>
#include "matrix_cpu.h"

int main(){
    Matrix *m1 = InitMatrix(2,3), *m2 = InitMatrix(3, 4), *m;
    generateRandomMatrix(m1);
    generateRandomMatrix(m2);
    // dumpMatrix(m1);
    // dumpMatrix(m2);
    m = matrixMultiply(m1,m2);
    if(m){
        // printf("dumping multiplied matrix\n");
        dumpMatrix(m);
        MatrixFree(m);
    }
    MatrixFree(m1);
    MatrixFree(m2);
    return 0;
}