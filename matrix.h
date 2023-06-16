/*
 * matrix.h
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdio.h>

typedef struct Matrix{
    double **mat;
    int rows_, cols_;
} Matrix;

Matrix* InitMatrix(int rows, int cols);
void generateRandomMatrix(Matrix* matrix);
void MatrixFree(Matrix* matrix);
void allocSpace(Matrix* matrix);
void dumpMatrix(Matrix* matrix);
double innerProudct(double* vec1, double* vec2, int size);
double* vectorAddition(double* vec1, double* vec2, int size);
Matrix* transpose(Matrix* matrix);
Matrix* matrixMultiply(Matrix* matrix1, Matrix* matrix2);

#endif
