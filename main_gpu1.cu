#include <stdio.h>
#include <stdlib.h>
#include "matrix_gpu2.h"

int main(){
    Matrix *m1 = InitMatrix(MAT1ROWS, MAT12), *m2 = InitMatrix(MAT12, MAT2COLS), *m3, *m4;
    type_m* bias = (type_m*)malloc(sizeof(type_m) * MAT1ROWS);
    generateRandomVector(bias, MAT1ROWS);
    // dumpVector(bias, MAT1ROWS);
    generateRandomMatrix(m1);
    generateRandomMatrix(m2);
    // dumpMatrix(m1);
    // dumpMatrix(m2);
    m3 = matrixMultiplyAddBias_cpu(m1,m2,bias);

    IndexSave* Ind = (IndexSave*)malloc(sizeof(IndexSave) * MAT1ROWS * MAT2COLS);
    m4 = matrixMultiplyAddBias_gpu(m1,m2,bias,Ind);
    if(m3 && m4){
        // dump the index of gpu counted matrix and cpu counted matrix
        // dumpIndex(Ind, m3, m4);

        // check the gpu counted matrix and cpu counted matrix is same or not
        matrixEqual(m3,m4);
            
        MatrixFree(m3);
        MatrixFree(m4);
    }
    MatrixFree(m1);
    MatrixFree(m2);
    free(bias);
    return 0;
}
