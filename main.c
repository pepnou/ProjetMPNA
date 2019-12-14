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


  double *x  = NULL, 
         *v  = (double*)malloc(B.height * sizeof(double)),
         *xt = (double*)malloc(B.width  * sizeof(double));
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
    
    //memcpy(access(B, j, j+1), &v[1], (size - 1)*sizeof(double));

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);


    if(j < B.width - 1) {
      size = B.width - j - 1;

      setParams(&Bvvt , size, size);
      setParams(&BvvtA, size, B.height - j);

      initSubMatrix(B, &sB, j + 1, B.width-1, j, B.height-1);

      cblas_dcopy(size, access(B, j+1, j), B.height, xt, 1);
      house(xt, size, v, &beta);

      cblas_dger(CblasColMajor, size, size, -beta, v, 1, v, 1, Bvvt.M, size);

      for(int i = 0; i < size; i++) {
        *access(Bvvt, i, i) += 1.;
      }

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sB.Yend - sB.Ystart + 1, size, size, 1., access(sB, 0, 0), sB.height, Bvvt.M, size, 0., BvvtA.M, sB.Yend - sB.Ystart + 1);

      for(int i = 0; i < BvvtA.width; i++) {
        memcpy(access(sB, i, 0), access(BvvtA, i, 0), BvvtA.height*sizeof(double));
      }
      
      //cblas_dcopy(size - 1, &v[1], 1, access(B, j+2, j), B.height);
    }

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);
  }

  free(v);
  free(xt);
  free(Bvvt.M);
  free(BvvtA.M);
}

void givens(double a, double b, double *c, double *s) {
  if( b == 0 ) {
    *c = 1;
    *s = 0;
  } else {
    if( abs(b) > abs(a) ) {
      double tau = -a / b;
      *s = 1 / sqrt(1 + tau * tau);
      *c = *s * tau;
    } else {
      double tau = -b / a;
      *c = 1 / sqrt(1 + tau * tau);
      *s = *c * tau;
    }
  }
}

double sign(double a) {
  if( a < 0 ) {
    return -1;
  } else {
    return 1;
  }
}

void applyGivensLeft(Matrix T, int i, int k, double c, double s) {
  for(int j = 0; j < T.width; j++) {
    double tau1 = *access(T, j, i);
    double tau2 = *access(T, j, k);

    *access(T, j, 1) = c * tau1 - s * tau2;
    *access(T, j, 2) = s * tau1 + c * tau2;
  }
}

void applyGivensRight(Matrix T, int i, int k, double c, double s) {
  for(int j = 0; j < T.height; j++) {
    double tau1 = *access(T, i, j);
    double tau2 = *access(T, k, j);

    *access(T, i, j) = c * tau1 - s * tau2;
    *access(T, k, j) = s * tau1 + c * tau2;
  }
}


void QR(Matrix T) {
  double TNN     = *access(T, T.width - 1, T.height - 1),
         TNNM1   = *access(T, T.width - 2, T.height - 1),
         //TNM1N   = *access(T, T.width - 1, T.height - 2),
         TNM1NM1 = *access(T, T.width - 2, T.height - 2);


  double d = ( TNM1NM1 - TNN ) / 2;
  double mu = TNN - TNNM1 * TNNM1 / ( d + sign(d) * sqrt(d*d + TNNM1 * TNNM1) );

  double x = *access(T, 0, 0) - mu;
  double z = *access(T, 0, 1);

  double c, s;

  for(int k = 0; k < T.width - 1; k++) {
    givens(x, z, &c, &s);

    applyGivensLeft(T , k, k+1, c, s);
    applyGivensRight(T, k, k+1, c, s);
    
    if(k < T.width - 2) {
      x = *access(T, k, k+1);
      z = *access(T, k, k+2);
    }
  }
}

void SVD(Matrix A, Matrix U, Matrix V) {
  Matrix B, T;
  initMatrix(&B, A.width, A.height, A.type);
  initMatrix(&T, B.width, B.width, A.type);

  bidiag(A, B);

  printMat(A);
  printMat(B);


  //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, B.width, B.width, B.height, 1., access(B, 0, 0), B.height, access(B, 0, 0), B.height, 0., access(T, 0, 0), T.height);
  //printMat(T);







  free(B.M);
  free(T.M);
}


int main(int argc, char** argv) {
  int m = 5, n = 5;

  Matrix A, U, V;
  initMatrix(&A, n, m, COL_MAJOR);

  srand(0);

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      //*access(A, i, j) = i+1 + j*n;
      *access(A, i, j) = rand() % 100;
    }
  }

  SVD(A, U, V);

  free(A.M);

  exit(0);
}
