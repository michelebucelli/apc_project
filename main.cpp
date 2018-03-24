#include "kmeans_sdg.h"

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

   // Since data is read from a file, we can have all the processes read it
   // concurrently, instead of having rank 0 receive it and then send it to the
   // other processes

   std::ifstream inputFile ( "./benchmarks/s1.txt" );
   kMeansSDG solver ( inputFile );

   if ( rank == 0 ) {
      clog << "Input file:    ./benchmarks/s1.txt" << endl;
      clog << "N. of samples: " << solver.size() << endl;
      clog << "Dimension:     " << solver.getN() << endl;
   }

   solver.setK ( 15 );
   solver.solve();

   if ( rank == 0 ) {
      cout << solver;
   }

   MPI_Finalize();
   return 0;
}
