#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

enum MATRIX_MAJOR {COL_MAJOR, ROW_MAJOR};

double* access(double *M, int width, int height, int x, int y, enum MATRIX_MAJOR type) {
  switch(type) {
    case COL_MAJOR: 
      return &(M[x + y*width]);
    case ROW_MAJOR:
      return &(M[y + x*height]);
    default:
      return NULL;
  }
}

void printMat(double *M, int width, int height, enum MATRIX_MAJOR type) {
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      printf("%10lf ", *access(M, width, height, i, j, type));
    }
    printf("\n");
  }
  printf("\n");
}

void identity(double *Id, int size) {
  memset((void*)Id, 0, size * size * sizeof(double));

  for(int i = 0; i < size; i++) {
    *access(Id, size, size, i, i, COL_MAJOR) = 1;
  }
}

void house(double *v, double *H, int size) {
  identity(H, size);
  cblas_dger(CblasRowMajor, size, size, -2., v, 1, v, 1, H, size);
}

void bidiag(double* A, int width, int height, double* B) {
  memcpy(B, A, width * height * sizeof(double));

  double *Be = (double*)malloc(width*width*sizeof(double));
  double *v = NULL;

  for(int j = 0; j < height; j++) {
    v = &(B[j + j*height]);
    house(v, Be, width - j);

    // copy to extract submatrix
  }
}

int main(int argc, char** argv) {
  int size = 5;
  double *A = (double*)malloc(size*size*sizeof(double));
  double v[] = {2,4,3,1,0};
  house(v, A, 5);
  printMat(A, 5, 5, COL_MAJOR);
  free(A);

  exit(0);
}
