#include "kmeans_sgd.h"
#include "kmeans.h"

#include "timer.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

using std::cout;
using std::clog;
using std::endl;

int main ( int argc, char * argv[] ) {
   MPI_Init ( &argc, &argv );
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   timer tm;

   // Read command line parameters
   std::string test, method;
   int k = 0;

   if ( argc < 3 ) {
      clog << "Too few arguments. Usage: main.out <test file> <cluster count> <method>" << endl;
      return 1;
   }

   else {
      test = argv[1];
      k = std::atoi(argv[2]);
      method = argv[3];
   }

   std::ifstream datasetIn ( "./benchmarks/" + test + ".txt" );
   std::ifstream trueLabelsIn ( "./benchmarks/" + test + "-truelabels.txt" );

   kMeansBase *solver = nullptr;

   if ( method == "kmeans" ) {
      solver = new kMeans ( datasetIn, trueLabelsIn );
   }

   else if ( method == "kmeansSGD" ) {
      kMeansSGD * slv = new kMeansSGD ( datasetIn, trueLabelsIn );
      slv->setBatchSize ( 100 );
      solver = slv;
   }

   else {
      clog << "Unknown method " << method << "; available methods: kmeans kmeansSGD" << endl;
      return 1;
   }

   solver->setStop ( 1000, -1, 1 );
   solver->setK ( k );

   if ( rank == 0 ) {
      clog << "Test name: " << test << endl;
      clog << "Dataset source: ./benchmarks/" << test << ".txt" << endl;
      clog << "True labels source: ./benchmarks/" << test << "-truelabels.txt" << endl;
      clog << "Method: " << method << endl;
   }

   tm.start();
   solver->solve ();
   tm.stop();

   if ( rank == 0 )
      clog << "Elapsed time: " << tm.getTime() << " microseconds" << endl;

   real purity = solver->purity();

   if ( rank == 0 ) {
      clog << "Dataset size: " << solver->size() << endl;
      clog << "Converged in " << solver->getIter() << " iterations" << endl;
      clog << "Clustering purity: " << purity << endl;
      cout << (*solver);
   }

   delete solver;

   MPI_Finalize ();
   return 0;
}
