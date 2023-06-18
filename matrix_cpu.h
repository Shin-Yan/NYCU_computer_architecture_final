/*
 * matrix.h
 */

#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

#include <stdio.h>

typedef struct Matrix{
    float *mat;
    int rows_, cols_;
} Matrix;

Matrix* InitMatrix(int rows, int cols);
void generateRandomMatrix(Matrix* matrix);
void generateRandomVector(float* vec, int size);
void MatrixFree(Matrix* matrix);
void allocSpace(Matrix* matrix);
void dumpMatrix(Matrix* matrix);
void dumpVector(float* vec, int size);
float innerProduct(float *vec1, float *vec2, int n);
float* addVector(float *vec1, float *vec2, int n);
float* substractVector(float *vec1, float *vec2, int n);

Matrix* transpose(Matrix* matrix);
Matrix* matrixMultiplyAddBiasActivation(Matrix* matrix1, Matrix* matrix2, float* bias);

#endif
