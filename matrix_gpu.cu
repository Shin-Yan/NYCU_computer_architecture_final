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

__global__ void cudaMatrixMuladdBias(float* new_mat, float* m1, float* m2, float* bias, IndexSave* ind, int m, 
                                    int n, int p){
        int stripe = blockDim.x*gridDim.x;
        int head = (blockIdx.x*blockDim.x + threadIdx.x);
        for(int i = head ; i < m ; i+=stripe){
            ind[i].blockInd_x = blockIdx.x;
            ind[i].threadInd_x = threadIdx.x;
            ind[i].head = head;
            ind[i].stripe = stripe;
            for(int k = 0 ; k < p ; ++k){
                for(int j = 0 ; j < n ; ++j)
                {
                    new_mat[i*p+k] += m1[i*n+j]*m2[j*p+k];
                }
                new_mat[i*p+k] += bias[i]; 
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
            int value = (float)rand()/(float)RAND_MAX *10;
            matrix->mat[i*cols+j] = value;
        }
    }
}

void generateRandomVector(float* vec, int size){
    for(int i = 0 ; i < size ; i++){
        int value = (float)rand()/(float)RAND_MAX *10;
        vec[i] = value;
    }
}

void MatrixFree(Matrix* matrix){
    free(matrix->mat);
    free(matrix);
}

void allocSpace(Matrix* matrix){
    matrix->mat = (float*)malloc(matrix->rows_ * matrix->cols_ *sizeof(float*));
}

void dumpMatrix(Matrix* matrix){
    int rows = matrix->rows_;
    int cols = matrix->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            printf("%f ", matrix->mat[i*cols+j]);
        }
        printf("\n");
    }
}

void dumpVector(float* vec, int size){
    for(int i = 0 ; i < size ; i ++){
        printf("%f ", vec[i]);
    }
    printf("\n");
}

void dumpIndex(IndexSave* Ind, Matrix* new_mat){
    for(int i = 0 ; i < new_mat->rows_ ; ++i ){
	printf("%d : blockInd_x=%d, threadInd_x=%d, head=%d, stripe=%d\n", i, Ind[i].blockInd_x, Ind[i].threadInd_x, Ind[i].head, Ind[i].stripe);
	printf("    GPU result :");
	    for(int j = 0 ; j < new_mat->cols_ ; ++j){
	        printf("%f ", new_mat->mat[i*new_mat->cols_ + j]);
	    }
	printf("\n");
    }
}

float innerProduct(float *vec1, float *vec2, int n){
    float ret = 0 ;
    for(int i = 0 ; i < n ; ++n){
        ret += vec1[i] * vec2[i];
    }
    return ret;
}

float* addVector(float *vec1, float *vec2, int n){
    float* new_vec = (float*)malloc(n * sizeof(float));
    for(int i = 0 ; i < n ; ++n){
        new_vec[i] = vec1[i] + vec2[i];
    }
    return new_vec;
}

float* substractVector(float *vec1, float *vec2, int n){
    float* new_vec = (float*)malloc(n * sizeof(float));
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

Matrix* matrixMultiplyAddBiasActivation(Matrix* matrix1, Matrix* matrix2, float* bias){

    if(matrix1->cols_ != matrix2->rows_){
        printf("matrix1 cols doesn't match matrix2 rows\n");
        return NULL;
    }

    Matrix* new_mat = (Matrix*)malloc(sizeof(Matrix));
    new_mat->rows_ = matrix1->rows_;
    new_mat->cols_ = matrix2->cols_;
    allocSpace(new_mat);
    // IndSave 
    IndexSave* Ind = (IndexSave*)malloc(sizeof(IndexSave) * new_mat->rows_);
    IndexSave* dInd;
    // create time event
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    // allocate memory space on device
    float *device_new, *device_matrix1, *device_matrix2, *device_bias;

    cudaMalloc((void **)&device_new, sizeof(float) * new_mat->rows_ * new_mat->cols_);
    cudaMalloc((void **)&device_matrix1, sizeof(float) * matrix1->rows_ * matrix1->cols_);
    cudaMalloc((void **)&device_matrix2, sizeof(float) * matrix2->rows_ * matrix2->cols_);
    cudaMalloc((void **)&device_bias, sizeof(float) * new_mat->rows_);
    cudaMalloc((void **)&dInd, sizeof(IndexSave) * new_mat->rows_);

    // copy data from host to device
    // cudaMemcpy(device_new, new_mat->mat, sizeof(float)* new_mat->cols_ * new_mat->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix1, matrix1->mat, sizeof(float)* matrix1->cols_ * matrix1->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix2, matrix2->mat, sizeof(float)* matrix2->cols_ * matrix2->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias, bias, sizeof(float) * matrix1->rows_, cudaMemcpyHostToDevice);
    // start timer
    cudaEventRecord(start, 0);

    // call kernel function
    dim3 dimGrid(1);
    dim3 dimBlock(2);
    
    cudaMatrixMuladdBias<<<dimGrid, dimBlock>>>(device_new, device_matrix1, device_matrix2, device_bias, dInd, matrix1->rows_, matrix1->cols_, 
    		    matrix2->cols_);
    // stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // copy result back to host
    cudaMemcpy(new_mat->mat, device_new, sizeof(float)* new_mat->rows_ * new_mat->cols_, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ind, dInd, sizeof(IndexSave) * new_mat->rows_, cudaMemcpyDeviceToHost);

    dumpIndex(Ind, new_mat);

    // release memory space on device
    cudaFree(device_matrix1);
    cudaFree(device_matrix2);
    cudaFree(device_new);
    cudaFree(device_bias);

    // Calculate Elapsed Time
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time elapsed is %5.2f ms\n", elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return new_mat;
}
