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

void InitMatrix(Matrix* ,int rows, int cols);
void MatrixFree(Matrix*);
void allocSpace(Matrix* mat);
void dumpMatrix(Matrix* mat);

#endif
