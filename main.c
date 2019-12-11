#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

enum MATRIX_MAJOR {COL_MAJOR, ROW_MAJOR};
typedef enum MATRIX_MAJOR MATRIX_MAJOR;


typedef struct {
  double *M;
  MATRIX_MAJOR type;
  int width, height;
  int Xstart, Xend;
  int Ystart, Yend;
} Matrix;

void initMatrix(Matrix *M, int width, int height, MATRIX_MAJOR type) {
  M->M = (double*)malloc(width*height*sizeof(double));
  M->type = type;
  M->width = width;
  M->height = height;
  M->Xstart = M->Ystart = 0;
  M->Xend = width - 1;
  M->Yend = height - 1;
}

void initSubMatrix(Matrix M, Matrix *sM, int Xstart, int Xend, int Ystart, int Yend) {
  sM->M = M.M;
  sM->type = M.type;
  sM->width = M.width;
  sM->height = M.height;
  sM->Xstart = Xstart;
  sM->Xend = Xend;
  sM->Ystart = Ystart;
  sM->Yend = Yend;
}

double* access(Matrix M, int x, int y) {
  switch(M.type) {
    case COL_MAJOR: 
      return &(M.M[(y + M.Ystart) + (x + M.Xstart)*M.height]);
    case ROW_MAJOR:
      return &(M.M[(x + M.Xstart) + (y + M.Ystart)*M.width]);
    default:
      return NULL;
  }
}


void printMat(Matrix M) {
  for(int i = 0; i <= M.Yend - M.Ystart; i++) {
    for(int j = 0; j <= M.Xend - M.Xstart; j++) {
      printf("%10lf ", *access(M, j, i));
    }
    printf("\n");
  }
  printf("\n");
}


/*
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
*/
int main(int argc, char** argv) {
  Matrix M;
  initMatrix(&M, 5, 5, COL_MAJOR);
  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++) {
      *access(M, i, j) = i*5+j;
    }
  }
  printMat(M);

  Matrix sM;
  initSubMatrix(M, &sM, 1, 3, 0, 4);
  printMat(sM);

  Matrix c;
  initMatrix(&c, 3, 5, COL_MAJOR);

  double alpha = 1.0, beta = 0.;
  int m = c.height, n = c.width, k = 5;
  int lda = M.height, ldb = sM.height, ldc = c.height;

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, access(M, 0, 0), lda, access(sM, 0, 0), ldb, beta, access(c, 0, 0), ldc);

  printMat(c);

  exit(0);
}
