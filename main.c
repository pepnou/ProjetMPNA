#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
  M->M = (double*)calloc(width*height, sizeof(double));
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

void setParams(Matrix *M, int width, int height) {
  M->width = width;
  M->height = height;
  M->Xstart = M->Ystart = 0;
  M->Xend = width - 1;
  M->Yend = height - 1;
}

void clearMatrix(Matrix m) {
  memset((void*)m.M, 0, m.width * m.height * sizeof(double));
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
      printf("%7.3lf ", *access(M, j, i));
    }
    printf("\n");
  }
  printf("\n");
}



void identity(Matrix m) {
  memset((void*)m.M, 0, m.width * m.height * sizeof(double));

  for(int i = 0; i < m.width; i++) {
    *access(m, i, i) = 1;
  }
}

void house(double *x, int size, double* v, double *beta) {
  double theta = cblas_ddot(size - 1, &x[1], 1, &x[1], 1);

  v[0] = 1.;
  memcpy(&v[1], &x[1], (size-1)*sizeof(double));
    
  if(theta == 0) {
    *beta = 0;
  } else {
    double nu = sqrt(x[0] * x[0] + theta);
    if(x[0] <= 0) {
      v[0] = x[0] - nu;
    } else {
      v[0] = -theta / (x[0] + nu);
    }
    double tmp = v[0]*v[0];
    *beta = 2*tmp / (theta + tmp);
    cblas_dscal(size, 1./v[0], v, 1);
  }
}

void bidiag(Matrix A, Matrix B) {
  memcpy(B.M, A.M, B.width * B.height * sizeof(double));

  Matrix Bvvt, BvvtA, sB;
  initMatrix(&Bvvt , B.height, B.height, COL_MAJOR);
  initMatrix(&BvvtA, B.height, B.height, COL_MAJOR);


  double *x = NULL, *v = (double*)malloc(B.height * sizeof(double));
  double beta;
  
  int size;

  for(int j = 0; j < B.width; j++) {
    x = access(B, j, j);
    
    size = B.height - j;

    setParams(&Bvvt , size, size);
    setParams(&BvvtA, B.width - j, size);

    initSubMatrix(B, &sB, j, B.width-1, j, B.height-1);

    house(x, size, v, &beta);

    cblas_dger(CblasColMajor, size, size, -beta, v, 1, v, 1, Bvvt.M, size);
  
    for(int i = 0; i < size; i++) {
      *access(Bvvt, i, i) += 1.;
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, sB.Xend - sB.Xstart + 1, size, 1., Bvvt.M, size, access(sB, 0, 0), sB.height, 0., BvvtA.M, size);


    for(int i = 0; i < BvvtA.width; i++) {
      memcpy(access(sB, i, 0), access(BvvtA, i, 0), size*sizeof(double));
    }

    memcpy(access(B, j, j+1), &x[1], (size - j)*sizeof(double));

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);
  }

  free(v);
  free(Bvvt.M);
  free(BvvtA.M);
}



int main(int argc, char** argv) {
  /*Matrix M;
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

  printMat(c);*/

  int m = 3, n = 2;

  Matrix A, B;
  initMatrix(&A, n, m, COL_MAJOR);
  initMatrix(&B, n, m, COL_MAJOR);

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      *access(A, i, j) = i + j*n;
    }
  }

  printMat(A);

  bidiag(A, B);

  printMat(B);

  free(A.M);
  free(B.M);

  exit(0);
}
