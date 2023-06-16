/*
 * matrix.h
 */

#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

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

Matrix* transpose(Matrix* matrix);
Matrix* matrixMultiply(Matrix* matrix1, Matrix* matrix2);

#endif
