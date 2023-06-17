/*
 * matrix.c
 */

#include "matrix_cpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

Matrix* InitMatrix(int rows, int cols)
{
    Matrix *matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows_ = rows;
    matrix->cols_ = cols;
    allocSpace(matrix);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix->mat[i*cols+j] = 0;
        }
    }
    return matrix;
}

void generateRandomMatrix(Matrix* matrix){
    int rows = matrix->rows_;
    int cols = matrix->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            // int value = (double)rand()/(double)RAND_MAX *10;
            matrix->mat[i*cols+j] = (double)rand()/(double)RAND_MAX;
        }
    }
}

void generateRandomVector(double* vec, int size){
    for(int i = 0 ; i < size ; i++){
        // int value = (double)rand()/(double)RAND_MAX *10;
        vec[i] = (double)rand()/(double)RAND_MAX;
    }
}

void MatrixFree(Matrix* matrix){
    free(matrix->mat);
    free(matrix);
}

void allocSpace(Matrix* matrix){
    matrix->mat = (double*)malloc(matrix->rows_ * matrix->cols_ *sizeof(double*));
}

void dumpMatrix(Matrix* matrix){
    int rows = matrix->rows_;
    int cols = matrix->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            printf("%lf ", matrix->mat[i*cols+j]);
        }
        printf("\n");
    }
}

void dumpVector(double* vec, int size){
    for(int i = 0 ; i < size ; i ++){
        printf("%lf ", vec[i]);
    }
    printf("\n");
}

double innerProduct(double *vec1, double *vec2, int n){
    double ret = 0 ;
    for(int i = 0 ; i < n ; ++n){
        ret += vec1[i] * vec2[i];
    }
    return ret;
}

double* addVector(double *vec1, double *vec2, int n){
    double* new_vec = (double*)malloc(n * sizeof(double));
    for(int i = 0 ; i < n ; ++n){
        new_vec[i] = vec1[i] + vec2[i];
    }
}

double* substractVector(double *vec1, double *vec2, int n){
    double* new_vec = (double*)malloc(n * sizeof(double));
    for(int i = 0 ; i < n ; ++n){
        new_vec[i] = vec1[i] - vec2[i];
    }
}

Matrix* transpose(Matrix* matrix){
    Matrix* new_mat = (Matrix*)malloc(sizeof(Matrix));
    int cols = new_mat->rows_ = matrix->cols_;
    int rows = new_mat->cols_ = matrix->rows_;
    allocSpace(new_mat); 
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols; ++j){
            new_mat->mat[j*rows+i] = matrix->mat[i*cols+j];
        }
    }
    return new_mat;
}

Matrix* matrixMultiplyAddBiasActivation(Matrix* matrix1, Matrix* matrix2, double* bias){

    if(matrix1->cols_ != matrix2->rows_){
        printf("matrix1 cols doesn't match matrix2 rows\n");
        return NULL;
    }

    Matrix* new_mat = (Matrix*)malloc(sizeof(Matrix));
    new_mat->rows_ = matrix1->rows_;
    new_mat->cols_ = matrix2->cols_;
    allocSpace(new_mat);

    // Matrix* matrix2_trans = transpose(matrix2);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int rows = new_mat->rows_;
    int cols = new_mat->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            for(int k = 0 ; k < matrix1->cols_ ; ++k){
                new_mat->mat[i*cols+j] += matrix1->mat[i*matrix1->cols_+k]*matrix2->mat[k*matrix2->cols_ + j];
            }
            new_mat->mat[i*cols+j] += bias[i];
            new_mat->mat[i*cols+j] = 1.0 / (1.0 + exp(-new_mat->mat[i*cols+j]));
        }
    }
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Time elapsed is %5.2f ms\n", (float)micros / 1000);
    return new_mat;
}