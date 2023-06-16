#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(){
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    InitMatrix(m, 10, 10);
    generateRandomMatrix(m);
    dumpMatrix(m);
    MatrixFree(m);
    return 0;
}