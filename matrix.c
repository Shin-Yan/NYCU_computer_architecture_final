/*
 * matrix.c
 */

#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

Matrix* InitMatrix(int rows, int cols)
{
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows_ = rows;
    matrix->cols_ = cols;
    allocSpace(matrix);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix->mat[i][j] = 0;
        }
    }
    return matrix;
}

void generateRandomMatrix(Matrix* matrix){
    for(int i = 0 ; i < matrix->rows_ ; ++i){
        for(int j = 0 ; j < matrix->cols_; ++j){
            matrix->mat[i][j] = (double)rand() / (double)RAND_MAX;
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

double innerProudct(double* vec1, double* vec2, int size){

    double ret;
    for(int i = 0 ; i < size ; ++i){
        ret += vec1[i] * vec2[i];
    }
    return ret;
}

double* vectorAddition(double* vec1, double* vec2, int size){
    double *ret = (double*)malloc(size * sizeof(double));
    for(int i = 0 ; i < size ; ++i){
        ret[i] = vec1[i] + vec2[i];
    }
    return ret;
}

Matrix* transpose(Matrix* matrix){
    Matrix* new_mat = (Matrix*)malloc(sizeof(Matrix));
    new_mat->rows_ = matrix->cols_;
    new_mat->cols_ = matrix->rows_;
    allocSpace(new_mat); 
    for(int i = 0 ; i < matrix->rows_ ; ++i){
        for(int j = 0 ; j < matrix->cols_; ++j){
            new_mat->mat[j][i] = matrix->mat[i][j];
        }
    }
    return new_mat;
}

Matrix* matrixMultiply(Matrix* matrix1, Matrix* matrix2){

    if(matrix1->cols_ != matrix2->rows_){
        printf("matrix1 cols doesn't match matrix2 rows\n");
        return NULL;
    }

    Matrix* new_mat = (Matrix*)malloc(sizeof(Matrix));
    new_mat->rows_ = matrix1->rows_;
    new_mat->cols_ = matrix2->cols_;
    allocSpace(new_mat);

    Matrix* matrix2_trans = transpose(matrix2);
    for(int i = 0 ; i < new_mat->rows_ ; ++i){
        for(int j = 0 ; j < new_mat->cols_ ; ++j){
            new_mat->mat[i][j] = innerProudct(matrix1->mat[i], matrix2_trans->mat[j], matrix1->cols_);
        }
    }
    return new_mat;
}