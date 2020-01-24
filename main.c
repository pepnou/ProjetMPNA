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

  Matrix Bvvt, sBvvt, BvvtA, sB, sU, sV, tmpU, tmpV;
  initMatrix(&Bvvt , B.height, B.height, COL_MAJOR);
  initMatrix(&BvvtA, B.height, B.height, COL_MAJOR);
  initMatrix(&tmpU, U.height, U.height, COL_MAJOR);
  initMatrix(&tmpV, V.width , V.width, COL_MAJOR);


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

    setParams(&Bvvt , B.height, B.height);
    setParams(&BvvtA, B.width - j, size);

    initSubMatrix(B, &sB, j, B.width-1, j, B.height-1);
    initSubMatrix(U, &sU, j, U.height-1, j, U.height-1);
    initSubMatrix(Bvvt, &sBvvt, Bvvt.width-size, Bvvt.width-1, Bvvt.height-size, Bvvt.height-1);

    house(x, size, v, &beta);


    cblas_dger(CblasColMajor, size, size, -beta, v, 1, v, 1, access(sBvvt, 0, 0), Bvvt.height);
  
    for(int i = 0; i < size; i++) {
      *access(sBvvt, i, i) += 1.;
    }


    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, size, sB.Xend - sB.Xstart + 1, size, 1., access(sBvvt, 0, 0), Bvvt.height, access(sB, 0, 0), sB.height, 0., BvvtA.M, size);


    for(int i = 0; i < BvvtA.width; i++) {
      memcpy(access(sB, i, 0), access(BvvtA, i, 0), size*sizeof(double));
    }


    for(int i = 0; i < sBvvt.Xstart; i++) {
      *access(Bvvt, i, i) = 1;
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, U.height, Bvvt.width, U.width, 1., access(U, 0, 0), U.height, access(Bvvt, 0, 0), Bvvt.height, 0., access(tmpU, 0, 0), tmpU.height);
    memcpy(U.M, tmpU.M, U.width*U.height*sizeof(double));
    

    
    //memcpy(access(B, j, j+1), &v[1], (size - 1)*sizeof(double));

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);
    clearMatrix(Bvvt);


    if(j < B.width - 1) {
      size = B.width - j - 1;

      setParams(&Bvvt , B.width, B.width);
      setParams(&BvvtA, size, B.height - j);

      initSubMatrix(B, &sB, j + 1, B.width-1, j, B.height-1);
      initSubMatrix(V, &sV, j + 1, B.width-1, j, B.width-1);
      initSubMatrix(Bvvt, &sBvvt, Bvvt.width-size, Bvvt.width-1, Bvvt.height-size, Bvvt.height-1);


      cblas_dcopy(size, access(B, j+1, j), B.height, xt, 1);
      house(xt, size, v, &beta);

      cblas_dger(CblasColMajor, size, size, -beta, v, 1, v, 1, access(sBvvt, 0, 0), sBvvt.height);

      for(int i = 0; i < size; i++) {
        *access(sBvvt, i, i) += 1.;
      }

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, sB.Yend - sB.Ystart + 1, size, size, 1., access(sB, 0, 0), sB.height, access(sBvvt, 0, 0), sBvvt.height, 0., BvvtA.M, sB.Yend - sB.Ystart + 1);

      for(int i = 0; i < BvvtA.width; i++) {
        memcpy(access(sB, i, 0), access(BvvtA, i, 0), BvvtA.height*sizeof(double));
      }


      for(int i = 0; i < sBvvt.Xstart; i++) {
        *access(Bvvt, i, i) = 1;
      }

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, V.height, Bvvt.width, V.width, 1., access(V, 0, 0), V.height, access(Bvvt, 0, 0), Bvvt.height, 0., access(tmpV, 0, 0), tmpV.height);
      memcpy(V.M, tmpV.M, V.width*V.height*sizeof(double));


      //cblas_dcopy(size - 1, &v[1], 1, access(B, j+2, j), B.height);
    }

    clearMatrix(Bvvt);
    clearMatrix(BvvtA);
  }



  free(v);
  free(xt);
  free(Bvvt.M);
  free(BvvtA.M);
  free(tmpU.M);
  free(tmpV.M);
}

void bidiag_lanczos(Matrix A, Matrix U, Matrix V, double tol) {
  double *p = (double*) calloc(A.width, sizeof(double));
  double *tau = (double*) calloc(A.height, sizeof(double));
  *access(V, 0, 0) = 1.;
  p[0] = 1.;
  double beta = 1., alpha = 0.;
  int k = 0;

  while(beta > tol && k < A.width - 1) {
    cblas_daxpy(A.width, 1./beta, p, 1, access(V, k+1, 0), 1);
    k++;

    memcpy(tau, access(U, k-1, 0), A.height * sizeof(double));
    cblas_dgemv(CblasColMajor, CblasNoTrans, A.height, A.width, 1., access(A, 0, 0), A.height, access(V, k, 0), 1, beta, tau, 1);

    alpha = cblas_dnrm2(A.height, tau, 1);

    if(k == 0) {
      cblas_daxpy(A.height, 1./alpha, tau, 1, access(U, k, 0), 1);
    } else {
      cblas_daxpy(A.height, 1./alpha, tau, 1, access(U, k-1, 0), 1);
    }

    memcpy(p, access(V, k, 0), A.width * sizeof(double));
    cblas_dgemv(CblasColMajor, CblasTrans, A.height, A.width, 1., access(A, 0, 0), A.height, access(U, k, 0), 1, alpha, p, 1);

    beta = cblas_dnrm2(A.width, p, 1);
  }
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

void GKSVDstep(Matrix B, Matrix U, Matrix V) {
  double mu;

  int size = B.Xend - B.Xstart + 1;
  int N = size - 1;

  double  dm = *access(B, N - 1, N - 1),
          dn = *access(B, N, N),
          fm = *access(B, N, N - 1),
          fm1 = *access(B, N - 1, N - 2);

  double tnn   = dn*dn+fm*fm,
         tn1n1 = dm*dm+fm1*fm1,
         tnn1  = dm*fm;

  double d = (tn1n1 - tnn) / 2;
  mu = tnn - pow(tnn1, 2) / (d + sign(d) * sqrt(pow(d, 2) + pow(tnn1, 2)));


  printf("[%lf %lf\n %lf %lf]\n",
      dm*dm+fm1*fm1,
      dm*fm,
      dm*fm,
      dn*dn+fm*fm);

  printf("mu : %lf\n", mu);

  double x = *access(B, 0, 0) * *access(B, 0, 0) - mu;
  double y = *access(B, 0, 0) * *access(B, 1, 0);
  double c, s;
  double bulge;

  for(int k = 0; k <= N-1; k++) {
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

  /*for(int k = 0; k <= N-1; k++) {
    givens(x, y, &c, &s);
    applyGivensRight(B, k, k+1, c, s);
    applyGivensRight(V, k, k+1, c, s);

    x = *access(B, k, k);
    y = *access(B, k, k+1);

    givens(x, y, &c, &s);
    applyGivensLeft(B, k, k+1, c, s);
    applyGivensLeft(U, k, k+1, c, s);

    if(k < N-1) {
      x = *access(B, k+1, k);
      y = *access(B, k+1, k);
    }
  }*/
}

void GKSVD(Matrix B, Matrix D, Matrix U, Matrix V, double tol) {
  int q = 0, p;

  Matrix sB, sU, sV;

  int N = V.height;

  while(q < N) {
    q = 0;
    p = N - 1;

    for(int k = N - 2; k >= 0; k--) {
      if( abs(*access(B, k + 1, k)) <= tol * (abs(*access(B, k, k)) * abs(*access(B, k+1, k+1))) ) {
        *access(B, k+1, k) = 0;
        if(q == N - k - 2) {
          q++;
          p--;
        }
      } else {
        if(p == k + 1) {
          p--;
        }
      }
    }

    printf("q : %d, p : %d\n", q, p);
    printf("B :\n");
    printMat(B);

    if(q == B.width - 1) {
      q = B.width;
      break;
    } else {
      int k = p;

      while(k < B.width - q && abs(*access(B, k, k)) > tol) {
        k++;
      }

      if(k < B.width - q) {
        *access(B, k, k) = 0.;

        if(k < B.width - q - 1) {
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
        printf("B' : \n");
        printMat(B);


        //*access(B, k+1, k) = 0.;
      } else {
        initSubMatrix(B, &sB, p, N - 1 - q, p, N - 1 - q);
        initSubMatrix(U, &sU, p, N - 1 - q, p, N - 1 - q);
        initSubMatrix(V, &sV, p, N - 1 - q, p, N - 1 - q);

        GKSVDstep(sB, sU, sV);

        printf("B'' : \n");
        printMat(B);
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
      *access(D, i, i) = *access(B, i, i);
    }
  }
}

void SVD(Matrix A, Matrix D, Matrix U, Matrix V) {
  Matrix B;
  initMatrix(&B, A.width, A.height, A.type);

  double tol = 0.00001 ;

  bidiag(A, B, U, V);

  printf("Bidiag B :\n");
  printMat(B);
  printf("Bidiag U :\n");
  printMat(U);
  printf("Bidiag V :\n");
  printMat(V);

  GKSVD(B, D, U, V, tol);
  

  printf("GKSVD D :\n");
  printMat(D);
  printf("GKSVD U :\n");
  printMat(U);
  printf("GKSVD V :\n");
  printMat(V);

  free(B.M);
}


int main(int argc, char** argv) {
  int m = atoi(argv[1]), n = atoi(argv[2]);

  Matrix A, D, U, V;
  initMatrix(&A, n, m, COL_MAJOR);
  initMatrix(&D, n, m, COL_MAJOR);
  initMatrix(&U, m, m, COL_MAJOR);
  initMatrix(&V, n, n, COL_MAJOR);

  srand(0);

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      //*access(A, i, j) = rand() % 100;
      *access(A, i, j) = i + j*n + 1;
    }
  }

  printf("Input Matrix :\n");
  printMat(A);

  SVD(A, D, U, V);

  free(A.M);
  free(U.M);
  free(V.M);

  exit(0);
}
