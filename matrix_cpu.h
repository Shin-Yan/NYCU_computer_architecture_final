/*
 * matrix.h
 */

#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

#include <stdio.h>

typedef struct Matrix{
    double *mat;
    int rows_, cols_;
} Matrix;

Matrix* InitMatrix(int rows, int cols);
void generateRandomMatrix(Matrix* matrix);
void generateRandomVector(double* vec, int size);
void MatrixFree(Matrix* matrix);
void allocSpace(Matrix* matrix);
void dumpMatrix(Matrix* matrix);
void dumpVector(double* vec, int size);
double innerProduct(double *vec1, double *vec2, int n);
double* addVector(double *vec1, double *vec2, int n);
double* substractVector(double *vec1, double *vec2, int n);

Matrix* transpose(Matrix* matrix);
Matrix* matrixMultiplyAddBiasActivation(Matrix* matrix1, Matrix* matrix2, double* bias);

#endif
