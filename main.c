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

void bidiag(Matrix A, Matrix B, Matrix U, Matrix V) {
  memcpy(B.M, A.M, B.width * B.height * sizeof(double));

  Matrix Bvvt, BvvtA, sB, sU, sV, BvvtU, BvvtV;
  initMatrix(&Bvvt , B.height, B.height, COL_MAJOR);
  initMatrix(&BvvtA, B.height, B.height, COL_MAJOR);
  initMatrix(&BvvtU, U.height, U.height, COL_MAJOR);
  initMatrix(&BvvtV, V.width , V.width, COL_MAJOR);


  identity(U);
  identity(V);


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
    setParams(&BvvtU, size, size);

    initSubMatrix(B, &sB, j, B.width-1, j, B.height-1);
    initSubMatrix(U, &sU, j, U.height-1, j, U.height-1);

    house(x, size, v, &beta);

    cblas_dger(CblasColMajor, size, size, -beta, v, 1, v, 1, Bvvt.M, size);
  
    for(int i = 0; i < size; i++) {
      *access(Bvvt, i, i) += 1.;
    }

    

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1., Bvvt.M, size, access(sU, 0, 0), sU.height, 0., BvvtU.M, size);
  
    for(int i = 0; i < BvvtU.width; i++) {
      memcpy(access(sU, i, 0), access(BvvtU, i, 0), size*sizeof(double));
    }


    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, sB.Xend - sB.Xstart + 1, size, 1., Bvvt.M, size, access(sB, 0, 0), sB.height, 0., BvvtA.M, size);


    for(int i = 0; i < BvvtA.width; i++) {
      memcpy(access(sB, i, 0), access(BvvtA, i, 0), size*sizeof(double));
    }
    
    //memcpy(access(B, j, j+1), &v[1], (size - 1)*sizeof(double));

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);
    clearMatrix(BvvtU);


    if(j < B.width - 1) {
      size = B.width - j - 1;

      setParams(&Bvvt , size, size);
      setParams(&BvvtA, size, B.height - j);
      setParams(&BvvtV, size, size);

      initSubMatrix(B, &sB, j + 1, B.width-1, j, B.height-1);
      initSubMatrix(V, &sV, j + 1, B.width-1, j, B.width-1);

      cblas_dcopy(size, access(B, j+1, j), B.height, xt, 1);
      house(xt, size, v, &beta);

      cblas_dger(CblasColMajor, size, size, -beta, v, 1, v, 1, Bvvt.M, size);

      for(int i = 0; i < size; i++) {
        *access(Bvvt, i, i) += 1.;
      }



      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, size, size, 1., access(sV, 0, 0), sV.height, Bvvt.M, size, 0., BvvtV.M, BvvtV.height);

      for(int i = 0; i < size; i++) {
        memcpy(access(sV, i, 0), access(BvvtV, i, 0), size*sizeof(double));
      }




      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sB.Yend - sB.Ystart + 1, size, size, 1., access(sB, 0, 0), sB.height, Bvvt.M, size, 0., BvvtA.M, sB.Yend - sB.Ystart + 1);

      for(int i = 0; i < BvvtA.width; i++) {
        memcpy(access(sB, i, 0), access(BvvtA, i, 0), BvvtA.height*sizeof(double));
      }
      
      //cblas_dcopy(size - 1, &v[1], 1, access(B, j+2, j), B.height);
    }

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);
    clearMatrix(BvvtV);
  }



  free(v);
  free(xt);
  free(Bvvt.M);
  free(BvvtA.M);
  free(BvvtU.M);
  free(BvvtV.M);
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


double wilkinsonshift(double a1, double b, double a2) {
  double d = (a1 - a2) / 2;

  if(d == 0) {
    if(a2 > 0) {
      return a2 + abs(b);
    } else {
      return a2 - abs(b);
    }
  } else {
    return a2 - b*b / (d * sign(d) * sqrt(d*d + b*b));
  }
}

void GKSVDstep(Matrix B, Matrix U, Matrix V) {
  double mu;

  int N = B.Xend - B.Xstart;
  //int M = B.Yend - B.Ystart;

  if(N > 2) {
    mu = wilkinsonshift(
        *access(B, N-1, N-1) * *access(B, N-1, N-1), 
        *access(B, N-1, N-1) * *access(B, N, N-1),
        *access(B, N, N) * *access(B, N, N) + *access(B, N, N-1) * *access(B, N, N-1));
  } else if(N == 2) {
    mu = wilkinsonshift(
        *access(B, N-1, N-1) * *access(B, N-1, N-1),
        *access(B, N-1, N-1) * *access(B, N, N-1),
        *access(B, N, N) * *access(B, N, N) + *access(B, N, N-1) * *access(B, N, N-1));
  } else {
    exit(1);
  }

  double x = *access(B, 0, 0) * *access(B, 0, 0) - mu;
  double y = *access(B, 0, 0) * *access(B, 1, 0);
  double bulge = 0., c, s;

  for(int k = 0; k < N-1; k++) {
    givens(x, y, &c, &s);
    applyGivensRight(V, k, k+1, c, s);

    if(k > 0) {
      *access(B, k, k-1) = c * *access(B, k, k-1) - s * bulge;
    }

    double Bk = *access(B, k , k);
    bulge = -s * *access(B, k+1, k+1);
    *access(B, k, k) = c * Bk - s * *access(B, k+1, k);
    *access(B, k+1, k) = s * Bk + c * *access(B, k+1, k);
    *access(B, k+1, k+1) = c * *access(B, k+1, k+1);

    x = *access(B, k, k);
    y = bulge;
    givens(x, y, &c, &s);
    applyGivensRight(U, k, k+1, c, s);

    *access(B, k, k) = c * *access(B, k, k) - s * bulge;
    double Bk2 = *access(B, k+1, k);
    *access(B, k+1, k) = c * Bk2 - s * *access(B, k+1, k+1);
    bulge = -s * *access(B, k+2, k+1);
    *access(B, k+1, k+1) = s * Bk2 + c * *access(B, k+1, k+1);
    *access(B, k+2, k+1) = c * *access(B, k+2, k+1);

    x = *access(B, k+1, k);
    y = bulge;
  }

  *access(B, N+1, N) = 0;
}

void GKSVD(Matrix B, Matrix D, Matrix U, Matrix V, double tol) {
  int q = 0, p;

  Matrix sB, sU, sV;

  //int M = U.height;
  int N = V.height;

  while(q < N) {
    q = 0;
    //p = N - 1;
    p = N - 1;

    for(int k = N - 2; k >= 0; k--) {
      if( abs(*access(B, k + 1, k)) <= tol * (abs(*access(B, k, k)) * abs(*access(B, k+1, k+1))) ) {
        *access(B, k+1, k) = 0;
        if(q == N - k - 2) {
          //q = N - k - 1;
          q++;
          //p = k - 1;
          p--;
        }
      } else {
        if(p == k + 1) {
          //p = k - 1;
          p--;
        }
      }

      printf("k: %d, p: %d, q: %d\n", k, p, q);
    }

    printf("%d %d\n", q, p);

    printMat(B);
    for(unsigned long l = 0; l < 1000000000; l++);

    if(q == B.width) {
      q = B.width;
    } else {
      int k = p;

      while(k < B.width - q - 1 && abs(*access(B, k, k)) > tol) {
        k++;
      }

      printf("%d\n", k);

      if(abs(*access(B, k, k)) < tol) {
        *access(B, k, k) = 0;

        if(k < B.width - q) {
          double bulge = *access(B, k+1, k);
          *access(B, k+1, k) = 0;

          for(int j = k + 1; j < B.width - q - 1; j++) {
            double c, s;
            givens(*access(B, j, j), bulge, &c, &s);
            *access(B, j, j) = -s * bulge + s * *access(B, j, j);
            bulge = s * *access(B, j+1, j);
            *access(B, j+1, j) = c * *access(B, j+1, j);
            applyGivensRight(U, k, j, c, s);
          }
        } else {
          double bulge = *access(B, B.width - q, B.width - q - 1);
          *access(B, B.width - q, B.width - q - 1) = 0;

          for(int j = B.width - q; j > p; j--) {
            double c, s;
            givens(*access(B, j, j), bulge, &c, &s);
            *access(B, j, j) = c * *access(B, j, j) - s * bulge;
            if(j > p+1) {
              bulge = s * *access(B, j, j-1);
              *access(B, j, j-1) = c * *access(B, j, j-1);
            }
            applyGivensRight(V, j, k, c, s);
          }
        }
      } else {
        initSubMatrix(B, &sB, 0, B.width-1, p, B.width - 1 - q);
        initSubMatrix(U, &sU, p, U.width - 1 - q, 0, U.height - 1);
        initSubMatrix(V, &sV, p, V.width - 1 - q, 0, V.height - 1);

        GKSVDstep(sB, sU, sV);
      }
    }
  }

  for(int i = 0; i < B.width; i++) {
    if(*access(B, i, i) < 0) {
      *access(D, i, i) = - *access(B, i, i);
      for(int j = 0; j < V.height; j++) {
        *access(V, i, j) = - *access(V, i, j);
      }
    } else {
      *access(D, i, i) = - *access(B, i, i);
    }
  }
}

void SVD(Matrix A, Matrix D, Matrix U, Matrix V) {
  Matrix B;
  initMatrix(&B, A.width, A.height, A.type);

  double tol = 0.0001 ;

  bidiag(A, B, U, V);

  GKSVD(B, D, U, V, tol);
  


  printMat(A);
  printMat(B);
  printMat(U);
  printMat(V);


  free(B.M);
}


int main(int argc, char** argv) {
  int m = 5, n = 5;

  Matrix A, D, U, V;
  initMatrix(&A, n, m, COL_MAJOR);
  initMatrix(&D, n, m, COL_MAJOR);
  initMatrix(&U, m, m, COL_MAJOR);
  initMatrix(&V, n, n, COL_MAJOR);

  srand(0);

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      //*access(A, i, j) = i+1 + j*n;
      *access(A, i, j) = rand() % 100;
    }
  }

  SVD(A, D, U, V);

  free(A.M);
  free(U.M);
  free(V.M);

  exit(0);
}
