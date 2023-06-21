/*
 * matrix.c
 */

#include "matrix_gpu2.h"
#include "parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cudaMatrixMuladdBias(type_m* new_mat, type_m* m1, type_m* m2, type_m* bias, IndexSave* ind, int m, 
                                    int n, int p){
        int TotalThread = blockDim.x*gridDim.x;
        int stripe = (m*p)/TotalThread;
        int head = (blockIdx.x*blockDim.x + threadIdx.x)*stripe;
        int LoopLim = head+stripe;
        for(int i = head ; i < LoopLim ; ++i){
            ind[i].blockInd_x = blockIdx.x;
            ind[i].threadInd_x = threadIdx.x;
            ind[i].head = head;
            ind[i].stripe = stripe;
            // m = 4, p = 32, i = 31
            int idx1 = i / p;
            int idx2 = i % p;
            for(int j = 0 ; j < n ; ++j)
            {
                new_mat[i] += m1[idx1*n+j]*m2[idx2*n+j];
            }
            new_mat[i] += bias[idx1]; 

            // sigmoid: activation function
            new_mat[i] = 1.0 / (1.0 + exp(-new_mat[i]));
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

bool matrixEqual(Matrix* m1, Matrix* m2){
    bool flag_equal = true;
    if(m1->rows_ != m2->rows_){
        printf("row size is not equal!\n");
        return false;
    }
    if(m1->cols_ != m2->cols_){
        printf("column size is not equal!\n");
        return false;
    }
    for(int i = 0 ; i < m1->rows_ ;++i){
        for(int j = 0 ; j < m1->cols_ ; ++j){
            if(m1->mat[i*m1->cols_+j] != m2->mat[i*m2->cols_+j] && abs(m2->mat[i*m2->cols_+j] -m1->mat[i*m1->cols_+j])>0.000001){
                printf("not equal at row: %d, col: %d!\n", i , j);
                printf("matrix1: %f, matrix2: %f\n", m1->mat[i*m1->cols_+j], m2->mat[i*m2->cols_+j]);
                flag_equal = false;
            }
        }
    }
    if(!flag_equal)
        return false;
    printf("Two matrix are equal.\n");
    return true;
}

void generateRandomMatrix(Matrix* matrix){
    int rows = matrix->rows_;
    int cols = matrix->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            // int value = (float)rand()/(float)RAND_MAX *10;
            matrix->mat[i*cols+j] = (type_m)rand()/(type_m)RAND_MAX;
        }
    }
}

void generateRandomVector(type_m* vec, int size){
    for(int i = 0 ; i < size ; i++){
        // int value = (type_m)rand()/(type_m)RAND_MAX *10;
        vec[i] = (type_m)rand()/(type_m)RAND_MAX;
    }
}

void MatrixFree(Matrix* matrix){
    free(matrix->mat);
    free(matrix);
}

void allocSpace(Matrix* matrix){
    matrix->mat = (type_m*)malloc(matrix->rows_ * matrix->cols_ *sizeof(type_m));
}

void dumpMatrix(Matrix* matrix){
    int rows = matrix->rows_;
    int cols = matrix->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            printf(expr_m, matrix->mat[i*cols+j]);
        }
        printf("\n");
    }
}

void dumpVector(type_m* vec, int size){
    for(int i = 0 ; i < size ; i ++){
        printf(expr_m, vec[i]);
    }
    printf("\n");
}

void dumpIndex(IndexSave* Ind, Matrix* cpu_mat, Matrix* gpu_mat){
    for(int i = 0 ; i < gpu_mat->rows_ * gpu_mat->cols_ ; ++i ){
        printf("%d : blockInd_x=%d, threadInd_x=%d, head=%d, stripe=%d\n", i, Ind[i].blockInd_x, Ind[i].threadInd_x, Ind[i].head, Ind[i].stripe);
        printf("    GPU result :");
        printf(expr_m, gpu_mat->mat[i]);
        printf("\n");
        printf("    CPU result :");
        printf(expr_m, cpu_mat->mat[i]);
        printf("\n");
    }
}

type_m innerProduct(type_m *vec1, type_m *vec2, int n){
    type_m ret = 0 ;
    for(int i = 0 ; i < n ; ++n){
        ret += vec1[i] * vec2[i];
    }
    return ret;
}

type_m* addVector(type_m *vec1, type_m *vec2, int n){
    type_m* new_vec = (type_m*)malloc(n * sizeof(type_m));
    for(int i = 0 ; i < n ; ++n){
        new_vec[i] = vec1[i] + vec2[i];
    }
    return new_vec;
}

type_m *substractVector(type_m *vec1, type_m *vec2, int n){
    type_m* new_vec = (type_m*)malloc(n * sizeof(type_m));
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

Matrix* matrixMultiplyAddBias_gpu(Matrix* matrix1, Matrix* matrix2, type_m* bias, IndexSave* Ind){

    if(matrix1->cols_ != matrix2->rows_){
        printf("matrix1 cols doesn't match matrix2 rows\n");
        return NULL;
    }

    Matrix* new_mat = InitMatrix(matrix1->rows_, matrix2->cols_);
    // IndSave 
    Matrix* matrix2_t = transpose(matrix2);

    IndexSave* dInd;
    // create time event
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    // allocate memory space on device
    type_m *device_new, *device_matrix1, *device_matrix2, *device_bias;

    cudaMalloc((void **)&device_new, sizeof(type_m) * new_mat->rows_ * new_mat->cols_);
    cudaMalloc((void **)&device_matrix1, sizeof(type_m) * matrix1->rows_ * matrix1->cols_);
    cudaMalloc((void **)&device_matrix2, sizeof(type_m) * matrix2->rows_ * matrix2->cols_);
    cudaMalloc((void **)&device_bias, sizeof(type_m) * new_mat->rows_);
    cudaMalloc((void **)&dInd, sizeof(IndexSave) * new_mat->rows_ * new_mat->cols_);

    // m*n , n*p
    // copy data from host to device
    cudaMemcpy(device_new, new_mat->mat, sizeof(type_m)* new_mat->cols_ * new_mat->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix1, matrix1->mat, sizeof(type_m)* matrix1->cols_ * matrix1->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix2, matrix2_t->mat, sizeof(type_m)* matrix2_t->cols_ * matrix2_t->rows_, cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias, bias, sizeof(type_m) * matrix1->rows_, cudaMemcpyHostToDevice);
    // start timer
    cudaEventRecord(start, 0);

    // call kernel function
    dim3 dimGrid(GRID);
    dim3 dimBlock(BLOCK);
    
    cudaMatrixMuladdBias<<<dimGrid, dimBlock>>>(device_new, device_matrix1, device_matrix2, device_bias, dInd, matrix1->rows_, matrix1->cols_, 
    		    matrix2->cols_);
    // stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // copy result back to host
    cudaMemcpy(new_mat->mat, device_new, sizeof(type_m)* new_mat->rows_ * new_mat->cols_, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ind, dInd, sizeof(IndexSave) * new_mat->rows_ * new_mat->cols_, cudaMemcpyDeviceToHost);

    // dumpIndex(Ind, new_mat);

    // release memory space on device
    cudaFree(device_matrix1);
    cudaFree(device_matrix2);
    cudaFree(device_new);
    cudaFree(device_bias);

    // Calculate Elapsed Time
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU Time elapsed is %5.2f ms\n", elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    MatrixFree(matrix2_t);
    return new_mat;
}

Matrix* matrixMultiplyAddBias_cpu(Matrix* matrix1, Matrix* matrix2, type_m* bias){

    if(matrix1->cols_ != matrix2->rows_){
        printf("matrix1 cols doesn't match matrix2 rows\n");
        return NULL;
    }

    Matrix* new_mat = InitMatrix(matrix1->rows_, matrix2->cols_);

    Matrix* matrix2_t = transpose(matrix2);
    // struct timeval start, end;
    // gettimeofday(&start, NULL);
    int rows = new_mat->rows_;
    int cols = new_mat->cols_;
    for(int i = 0 ; i < rows ; ++i){
        for(int j = 0 ; j < cols ; ++j){
            for(int k = 0 ; k < matrix1->cols_ ; ++k){
                new_mat->mat[i*cols+j] += matrix1->mat[i*matrix1->cols_+k]*matrix2_t->mat[j*matrix2_t->cols_+k];
            }
            new_mat->mat[i*cols+j] += bias[i];

            // activation function 有很多選項，這裡選擇sigmoid
            // sigmoid function
            new_mat->mat[i*cols+j] = 1.0 / (1.0 + exp(-new_mat->mat[i*cols+j]));
        }
    }
    // gettimeofday(&end, NULL);
    // long seconds = (end.tv_sec - start.tv_sec);
    // long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    // printf("CPU Time elapsed is %5.2f ms\n", (int)micros / 1000);
    MatrixFree(matrix2_t);
    return new_mat;
}