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

void InitMatrix(Matrix* matrix,int rows, int cols);
void generateRandomMatrix(Matrix* matrix);
void MatrixFree(Matrix* matrix);
void allocSpace(Matrix* matrix);
void dumpMatrix(Matrix* matrix);

#endif
