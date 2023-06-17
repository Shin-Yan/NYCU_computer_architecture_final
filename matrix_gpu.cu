/*
 * matrix.c
 */

#include "matrix_gpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cudaMatrixMuladdBias(double* new_mat, double* m1, double* m2, double* bias, int m, 
                                    int n, int p){
        int stripe = blockDim.x*gridDim.x;
        int head = (blockIdx.x*blockDim.x + threadIdx.x);
        for(int i = head ; i < m ; i+=stripe){
            for(int k = 0 ; k < p ; ++k){
                for(int j = 0 ; j < n ; ++j)
                {
                    new_mat[i*p+k] += m1[i*n+j]*m2[j*p+k];
                } 
            }
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
            int value = (double)rand()/(double)RAND_MAX *10;
            matrix->mat[i*cols+j] = value;
        }
    }
}

void generateRandomVector(double* vec, int size){
    for(int i = 0 ; i < size ; i++){
        int value = (double)rand()/(double)RAND_MAX *10;
        vec[i] = value;
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

    // create time event
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
    cudaEventRecord(start, 0);

    // allocate memory space on device
    double *device_new, *device_matrix1, *device_matrix2, *device_bias;
    
    cudaMalloc((void **)&device_new, sizeof(double) * new_mat->rows_ * new_mat->cols_);
    cudaMalloc((void **)&device_matrix1, sizeof(double) * matrix1->rows_ * matrix1->cols_);
    cudaMalloc((void **)&device_matrix2, sizeof(double) * matrix2->rows_ * matrix2->cols_);
    cudaMalloc((void **)&device_bias, sizeof(double) * new_mat->rows_);

    // copy data from host to device
    cudaMemcpy(device_new, new_mat->mat, sizeof(double)* new_mat->cols_ * new_mat->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix1, matrix1->mat, sizeof(double)* matrix1->cols_ * matrix1->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix2, matrix2->mat, sizeof(double)* matrix2->cols_ * matrix2->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias, bias, sizeof(double) * matrix1->rows_, cudaMemcpyHostToDevice);

    // call kernel function
    dim3 dimGrid(1);
    dim3 dimBlock(2);
    
    cudaMatrixMuladdBias<<<dimGrid, dimBlock>>>(device_new, device_matrix1, device_matrix2, bias, matrix1->rows_, matrix1->cols_, 
                                                matrix2->cols_);
    // copy result back to host
    cudaMemcpy(new_mat->mat, device_new, sizeof(double)* new_mat->rows_ * new_mat->cols_, cudaMemcpyDeviceToHost);
    
    // release memory space on device
    cudaFree(device_matrix1);
    cudaFree(device_matrix2);
    cudaFree(device_new);
    cudaFree(device_bias);

    // Calculate Elapsed Time
    float elapsedTime; 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time elapsed is %5.2f ms\n", elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return new_mat;
}
