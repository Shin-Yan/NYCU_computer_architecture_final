/*
 * matrix.c
 */

#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

void InitMatrix(Matrix* matrix, int rows, int cols)
{
    matrix->rows_ = rows;
    matrix->cols_ = cols;
    allocSpace(matrix);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix->mat[i][j] = 0;
        }
    }

}

void MatrixFree(Matrix* matrix){
    for(int i = 0; i <matrix->rows_ ; ++i){
        free(matrix->mat[i]);
    }
    free(matrix->mat);
    free(matrix);
}

void allocSpace(Matrix* matrix){
    matrix->mat = (double**)malloc(matrix->rows_*sizeof(double*));
    for(int i = 0 ; i < matrix->rows_ ; ++i){
        matrix->mat[i] =  (double*)malloc(matrix->cols_*sizeof(double));
    }
}

void dumpMatrix(Matrix* matrix){
    for(int i = 0 ; i < matrix->rows_ ; ++i){
        for(int j = 0 ; j < matrix->cols_; ++j){
            printf("%lf ", matrix->mat[i][j]);
        }
        printf("\n");
    }
}