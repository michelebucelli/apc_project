#include "kmeans_sgd.h"
#include "kmeans.h"

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

   std::ifstream datasetIn ( "./benchmarks/g2-2-30.txt" );
   std::ifstream trueLabelsIn ( "./benchmarks/g2-2-30-truelabels.txt" );
   kMeansSGD solver ( datasetIn, trueLabelsIn );

   solver.setStop ( 100, -1, 1 );
   solver.setBatchSize ( 100 );

   solver.setK ( 2 );
   solver.solve ();
   real purity = solver.purity();

   if ( rank == 0 ) {
      clog << "Dataset size: " << solver.size() << endl;
      clog << "Converged in " << solver.getIter() << " iterations" << endl;
      clog << "Clustering purity: " << purity << endl;
      cout << solver;
   }

   MPI_Finalize ();
   return 0;
}
