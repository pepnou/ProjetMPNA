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

  getPosition(pos, rank, size, m, n, k);

  printf("rank %d : %d %d\n", rank, pos[0], pos[1]);

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

  if(rank == 0) {
    for(int i = 0; i < m; i++) {
      for(int j = 0; j < n; j++) {
        printf("%3.0lf ", C[i + m*j]);
      }
      printf("\n");
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

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

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == 0) {
    for(int i = 0; i < m; i++) {
      for(int j = 0; j < n; j++) {
        printf("%3.0lf ", A[i + m*j]);
      }
      printf("\n");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  matMul(A, B, C, m, n, k);


  MPI_Finalize();
  exit(0);
}
