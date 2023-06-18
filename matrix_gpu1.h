/*
 * matrix.h
 */

#ifndef __MATRIX_CPU_H__
#define __MATRIX_CPU_H__

#include <stdio.h>
#include "parameters.h"

#define MAT1ROWS 32
#define MAT12 4
#define MAT2COLS 16
#define type_m float 
#define expr_m "%f " 

typedef struct Matrix{
    type_m *mat;
    int rows_, cols_;
} Matrix;

Matrix* InitMatrix(int rows, int cols);
bool matrixEqual(Matrix* matrix1, Matrix* matrix2);
void generateRandomMatrix(Matrix* matrix);
void generateRandomVector(type_m* vec, int size);
void MatrixFree(Matrix* matrix);
void allocSpace(Matrix* matrix);
void dumpIndex(IndexSave* Ind, Matrix* cpu_mat, Matrix* gpu_mat);
void dumpMatrix(Matrix* matrix);
void dumpVector(type_m* vec, int size);
type_m innerProduct(type_m *vec1, type_m *vec2, int n);
type_m* addVector(type_m *vec1, type_m *vec2, int n);
type_m* substractVector(type_m *vec1, type_m *vec2, int n);

Matrix* transpose(Matrix* matrix);
Matrix* matrixMultiplyAddBias_cpu(Matrix* matrix1, Matrix* matrix2, type_m* bias);
Matrix* matrixMultiplyAddBias_gpu(Matrix* matrix1, Matrix* matrix2, type_m* bias, IndexSave* Ind);

#endif
