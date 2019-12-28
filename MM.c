#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

double min(double a, double b) {
  return (a <= b)?a:b;
}
double max(double a, double b) {
  return (a >= b)?a:b;
}

void getPosition(int pos[2], int rank, int size, int m, int n, int k) {
//x = max(j, m-1) - (m-1) + j
//y = min(j, m-1) - j

  int x, y, count = 0;
  for(int i = 0; i < m + k - 1; i++) {
    x = max(i, m-1) - (m-1);
    y = min(i, m-1);
    for(; x < k && y >= 0; x++, y--) {
      if(count == rank) {
        pos[0] = x;
        pos[1] = y;
        return;
      }
      count++;
    }
  }
}

void matMul(double *A, double *B, double *C, int m, int n, int k) {
  int pos[2];
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if(rank == 0) {
    int tmp[3] = {m, n, k};
    MPI_Bcast(tmp, 3, MPI_INT, 0, MPI_COMM_WORLD);
  }

  getPosition(pos, rank, size, m, n, k);

  //printf("rank %d : %d %d\n", rank, pos[0], pos[1]);

  // create line and column communicator
  MPI_Comm lineComm, columnComm;
  MPI_Comm_split(MPI_COMM_WORLD, pos[0], rank, &columnComm);
  MPI_Comm_split(MPI_COMM_WORLD, pos[1], rank, &lineComm);
  
  
  //create datatype for scatter
  MPI_Datatype type1, type2;
  MPI_Type_vector (n, 1, m, MPI_DOUBLE, &type1);
  MPI_Type_commit(&type1);
  MPI_Type_create_resized(type1, 0, sizeof(double), &type2);
  MPI_Type_commit(&type2);


  double *Aline   = malloc(n*sizeof(double)),
         *Bcolumn = malloc(n*sizeof(double)),
         *Ccolumn = malloc(m*sizeof(double));

  // Scatter
  if(pos[0] == 0) {
    MPI_Scatter(A, 1, type2, Aline, n, MPI_DOUBLE, 0, columnComm);
  }

  if(pos[1] == 0) {
    MPI_Scatter(B, n, MPI_DOUBLE, Bcolumn, n, MPI_DOUBLE, 0, lineComm);
  }

  MPI_Bcast(Aline, n, MPI_DOUBLE, 0, lineComm);

  MPI_Bcast(Bcolumn, n, MPI_DOUBLE, 0, columnComm);



  double res = 0.;
#pragma omp parallel for reduction(+:res)
  for(int i = 0; i < n; i++)
    res = res + (Aline[i] * Bcolumn[i]);

  /*for(int i = 0; i < size; i++) {
    if(i == rank) {
      printf("\nrank : %d\n", rank);
      for(int j = 0; j < n; j++) {
        printf("%3.0lf ", Aline[j]);
      }
      printf("\n");
      for(int j = 0; j < n; j++) {
        printf("%3.0lf ", Bcolumn[j]);
      }
      printf("\n%3.0lf\n", res);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/

  
  MPI_Gather(&res, 1, MPI_DOUBLE, Ccolumn, 1, MPI_DOUBLE, 0, columnComm);

  if(pos[1] == 0) {
    MPI_Gather(Ccolumn, m, MPI_DOUBLE, C, m, MPI_DOUBLE, 0, lineComm);
  }
}

void matMul2(double *A, double *B, double *C, int m, int n, int k) {
  
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank == 0) {
    int tmp[3] = {m, n, k};
    MPI_Bcast(tmp, 3, MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Comm matMulComm;
  MPI_Comm_split(MPI_COMM_WORLD, rank >= m, rank, &matMulComm);
  if(rank >= m) {
    MPI_Comm_free(&matMulComm);
    return;
  }

  MPI_Comm_size(matMulComm, &size);

  int sizes[size], displs[size];
  int tmp1 = m/size;
  int tmp2 = m%size;
  for(int i = 0; i < tmp2; i++) {
    sizes[i] = tmp1 + 1;
  }
  for(int i = tmp2; i < size; i++) {
    sizes[i] = tmp1;
  }
  displs[0] = 0;
  for(int i = 1; i < size; i++) {
    displs[i] = displs[i-1] + sizes[i-1];
  }
  
  //create datatype for scatter
  MPI_Datatype type1, type2;
  MPI_Type_vector (n, 1, m, MPI_DOUBLE, &type1);
  MPI_Type_commit(&type1);
  MPI_Type_create_resized(type1, 0, sizeof(double), &type2);
  MPI_Type_commit(&type2);


  double *Alines  = (double*)malloc(n*sizes[rank]*sizeof(double)),
         *Ccolumn = (double*)calloc(k*sizes[rank],sizeof(double));

  if(rank > 0) {
    B = malloc(n*k*sizeof(double));
  }

  MPI_Scatterv(A, sizes, displs, type2, Alines, n*sizes[rank], MPI_DOUBLE, 0, matMulComm);

  MPI_Bcast(B, n*k, MPI_DOUBLE, 0, matMulComm);


  for(int i = 0; i < size; i++) {
    if(i == rank) {
      printf("\nrank : %d\n", rank);
      for(int j = 0; j < n*sizes[rank]; j++) {
        printf("%3.0lf ", Alines[j]);
      }
      printf("\n");
    }
    MPI_Barrier(matMulComm);
  }


//#pragma omp parallel for
  for(int i = 0; i < sizes[rank]; i++) {
//#pragma omp parallel for
    for(int j = 0; j < k; j++) {
//#pragma omp parallel for reduction(+:Ccolumn[j+i*n])
      for(int l = 0; l < n; l++) { 
        Ccolumn[j+i*k] = Ccolumn[j+i*k] + Alines[l + i*n] * B[l + j*n];
      }
    }
  }
  printf("\n");

  for(int i = 0; i < size; i++) {
    if(i == rank) {
      printf("\nrank : %d\n", rank);
      for(int j = 0; j < k*sizes[rank]; j++) {
        printf("%3.0lf ", Ccolumn[j]);
      }
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(matMulComm);
  }



  /*for(int i = 0; i < size; i++) {
    if(i == rank) {
      printf("\nrank : %d\n", rank);
      for(int j = 0; j < n; j++) {
        printf("%3.0lf ", Aline[j]);
      }
      printf("\n");
      for(int j = 0; j < n; j++) {
        printf("%3.0lf ", Bcolumn[j]);
      }
      printf("\n%3.0lf\n", res);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }*/

  
  /*MPI_Gather(&res, 1, MPI_DOUBLE, Ccolumn, 1, MPI_DOUBLE, 0, columnComm);

  if(pos[1] == 0) {
    MPI_Gather(Ccolumn, m, MPI_DOUBLE, C, m, MPI_DOUBLE, 0, lineComm);
  }*/

  free(Alines);
  free(Ccolumn);
  if(rank > 0) {
    free(B);
  }

  MPI_Comm_free(&matMulComm);
}




int gcd(int a, int b) {
  while(a != b) {
    if(a > b) {
      a = a - b;
    } else {
      b = b - a;
    }
  }
  return a;
}

void p0() {
  int m = 3, n = 2, k = 2;

  double *A = malloc(m*n*sizeof(double)), 
         *B = malloc(n*k*sizeof(double)), 
         *C = malloc(m*k*sizeof(double));
  for(int i = 0; i < m*n; i++) {
    A[i] = i+1;
  }
  for(int i = 0; i < n*k; i++) {
    B[i] = i + m*n + 1;
  }

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      printf("%3.0lf ", A[i + m*j]);
    }
    printf("\n");
  }

  matMul2(A, B, C, m, n, k);

  /*printf("\n");
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      printf("%3.0lf ", C[i + m*j]);
    }
    printf("\n");
  }*/

  free(A);
  free(B);
  free(C);


  // notify every one to stop
  int tmp[3] = {0,0,0};
  MPI_Bcast(tmp, 3, MPI_INT, 0, MPI_COMM_WORLD);
}

void pn() {
  int tmp[3];
  
  while(1) {
    MPI_Bcast(tmp, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if(tmp[0] > 0) {
      matMul2(NULL, NULL, NULL, tmp[0], tmp[1], tmp[2]);
    } else {
      return;
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    p0();
  } else {
    pn();
  }

  MPI_Finalize();
  exit(0);
}
