/*
 * matrix.c
 */

#include "matrix_gpu.h"
#include "parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cudaMatrixMuladdBias(double* new_mat, double* m1, double* m2, int n, 
                                    int m, int p, size_t p_1, size_t p_2, size_t p_new){
        int x = threadIdx.x + blockIdx.x* blockDim.x;
        int y = threadIdx.y + blockIdx.y* blockDim.y;

        if(x < p && y < n){
            double sum = 0;
            for(int k = 0 ; k < m ;++k){
                double* row1 = (double*)((char*)m1 + y * p_1);
                double* row2 = (double*)((char*)m2 + k * p_2);
                sum += row1[k] * row2[x];
            }
            double *row_new = (double*)((char*)new_mat + y * p_new);
            row_new[x] = sum;
        }
}

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

void generateRandomVector(double* vec, int size){
    for(int i = 0 ; i < size ; i++){
        vec[i] = (double)rand() / (double)RAND_MAX;
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
    return new_vec;
}

double* substractVector(double *vec1, double *vec2, int n){
    double* new_vec = (double*)malloc(n * sizeof(double));
    for(int i = 0 ; i < n ; ++n){
        new_vec[i] = vec1[i] - vec2[i];
    }
    return new_vec;
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

Matrix* matrixMultiplyAddBiasActivation(Matrix* matrix1, Matrix* matrix2, double* bias){

    if(matrix1->cols_ != matrix2->rows_){
        printf("matrix1 cols doesn't match matrix2 rows\n");
        return NULL;
    }

    Matrix* new_mat = (Matrix*)malloc(sizeof(Matrix));
    new_mat->rows_ = matrix1->rows_;
    new_mat->cols_ = matrix2->cols_;
    allocSpace(new_mat);

    // create time event
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    // allocate memory space on device
    double *device_new, *device_matrix1, *device_matrix2, *device_bias;
    size_t pitch_new, pitch_matrix1, pitch_matrix2;
    size_t size_new = new_mat->cols_ * sizeof(double);
    size_t size_matrix1 = matrix1->cols_ * sizeof(double);
    size_t size_matrix2 = matrix2->cols_ * sizeof(double);
    cudaMallocPitch((void **)&device_new, &pitch_new, size_new, new_mat->rows_);
    cudaMallocPitch((void **)&device_matrix1, &pitch_matrix1, size_matrix1, matrix1->rows_);
    cudaMallocPitch((void **)&device_matrix2, &pitch_matrix2, size_matrix2, matrix2->rows_);
    cudaMalloc((void **)&device_bias, sizeof(double) * new_mat->rows_);

    // call kernel function
    dim3 dimBlock(16,16);
    dim3 dimGrid((new_mat->cols_ + dimBlock.x - 1) / dimBlock.x, (new_mat->rows_ + dimBlock.y - 1) / dimBlock.y);
    cudaMatrixMuladdBias<<<dimGrid, dimBlock>>>(device_new, device_matrix1, device_matrix2, matrix1->rows_, matrix1->cols_, 
                                                matrix2->cols_, pitch_matrix1, pitch_matrix2, pitch_new);
    // copy result nack to host
    cudaMemcpy2D(new_mat->mat, size_new, device_new, pitch_new, size_new, new_mat->rows_, cudaMemcpyDeviceToHost);

    // release memory space on device
    cudaFree(device_matrix1);
    cudaFree(device_matrix2);
    cudaFree(device_new);
    cudaFree(device_bias);

    // Calculate Elapsed Time
  	float elapsedTime; 
  	cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time elapsed is %5.2f ms\n", elapsedTime);

    return new_mat;
}