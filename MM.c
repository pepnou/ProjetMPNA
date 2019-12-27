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
  MPI_Comm_split(MPI_COMM_WORLD, pos[0], rank, &lineComm);
  MPI_Comm_split(MPI_COMM_WORLD, pos[1], rank, &columnComm);
  
  
  //create datatype for scatter
  MPI_Datatype type1, type2;
  MPI_Type_vector (n, 1, m, MPI_DOUBLE, &type1);
  MPI_Type_commit(&type1);
  MPI_Type_create_resized(type1, 0, sizeof(double), &type2);
  MPI_Type_commit(&type2);

  //int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

  
  double *tmp = malloc(n*m*sizeof(double));
  memset(tmp, 0, n*m*sizeof(double));

  // Scatter
  if(pos[0] == 0) {
    MPI_Scatter(A, 2, type2, tmp, n*2, MPI_DOUBLE, 0, columnComm);
  }

  for(int i = 0; i < size; i++) {
    if(i == rank && pos[0] == 0) {
      printf("\nRank %d\n", rank);
      for(int j = 0; j < n*m; j++) {
        printf("%3.0lf ", tmp[j]);
      }
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /*if(rank == 0) {
    MPI_Send(A, m, type2, 1, 0, MPI_COMM_WORLD);
  }

  if(rank == 1) {
    MPI_Recv(tmp, m*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    for(int i = 0; i < m*n; i++) {
      printf("%3.0lf ", tmp[i]);
    }
    printf("\n");
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0) {
    for(int i = 0; i < m*n; i++) {
      printf("%3.0lf ", A[i]);
    }
    printf("\n");
  }*/

  if(pos[1] == 0) {

  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int m = 6, n = 2, k = 2;
  
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
