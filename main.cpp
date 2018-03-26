#include "kmeans_sgd.h"

#include <iostream>
#include <fstream>

using std::cin;
using std::cout;
using std::clog;
using std::endl;

int main ( int argc, char * argv[] ) {
   MPI_Init ( &argc, &argv );
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   std::ifstream inputFile ( "./benchmarks/s1.txt" );
   kMeansSGD solver ( inputFile );

   solver.setK(15);
   solver.solve();

   if ( rank == 0 )
      cout << solver;

   MPI_Finalize ();
   return 0;
}
