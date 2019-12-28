#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


void matMul(double *A, double *B, double *C, int m, int n, int k) {
  
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
  if(rank == 0) {
    MPI_Type_vector (n, 1, m, MPI_DOUBLE, &type1);
    MPI_Type_commit(&type1);
    MPI_Type_create_resized(type1, 0, sizeof(double), &type2);
    MPI_Type_commit(&type2);
  }

  double *Alines  = (double*)malloc(n*sizes[rank]*sizeof(double)),
         *Ccolumn = (double*)calloc(k*sizes[rank],sizeof(double));

  if(rank > 0) {
    B = malloc(n*k*sizeof(double));
  }

  MPI_Scatterv(A, sizes, displs, type2, Alines, n*sizes[rank], MPI_DOUBLE, 0, matMulComm);

  MPI_Bcast(B, n*k, MPI_DOUBLE, 0, matMulComm);


#pragma omp parallel for
  for(int i = 0; i < sizes[rank]; i++) {
#pragma omp parallel for
    for(int j = 0; j < k; j++) {
#pragma omp parallel for reduction(+:Ccolumn[j+i*n])
      for(int l = 0; l < n; l++) { 
        Ccolumn[j+i*k] = Ccolumn[j+i*k] + Alines[l + i*n] * B[l + j*n];
      }
    }
  }

  //create datatype for gather
  MPI_Datatype type3, type4;
  if(rank == 0) {
    MPI_Type_vector (k, 1, m, MPI_DOUBLE, &type3);
    MPI_Type_commit(&type3);
    MPI_Type_create_resized(type3, 0, sizeof(double), &type4);
    MPI_Type_commit(&type4);
  }

  MPI_Gatherv(Ccolumn, k*sizes[rank], MPI_DOUBLE, C, sizes, displs, type4, 0, matMulComm);


  free(Alines);
  free(Ccolumn);
  if(rank == 0) {
    MPI_Type_free(&type1);
    MPI_Type_free(&type2);
    MPI_Type_free(&type3);
    MPI_Type_free(&type4);
  } else {
    free(B);
  }

  MPI_Comm_free(&matMulComm);
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

  matMul(A, B, C, m, n, k);

  printf("\n");
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      printf("%3.0lf ", C[i + m*j]);
    }
    printf("\n");
  }

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
      matMul(NULL, NULL, NULL, tmp[0], tmp[1], tmp[2]);
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
