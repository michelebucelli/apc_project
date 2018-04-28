 #include "kmeans_seq.h"
#include "kmeans.h"
#include "kmeans_sgd.h"

#include "timer.h"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include "GetPot"

using std::cout;
using std::clog;
using std::endl;

int main ( int argc, char * argv[] ) {
   MPI_Init ( &argc, &argv );
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   GetPot cmdLine ( argc, argv );
   timer tm;

   std::string test = cmdLine.follow("s1", 2, "-t", "--test" );
   int k = cmdLine.follow(15, 1, "-k" );
   std::string method = cmdLine.follow("kmeans", 2, "-m", "--method" );
   bool purityTest = cmdLine.search("-p") || cmdLine.search("--purity" );

   std::ifstream datasetIn ( "./benchmarks/" + test + ".txt" );
   std::ifstream trueLabelsIn ( "./benchmarks/" + test + "-truelabels.txt" );

   kMeansBase *solver = nullptr;

   if ( method == "kmeans" ) {
      solver = purityTest ? new kMeans ( datasetIn, trueLabelsIn ) : new kMeans ( datasetIn );
      solver->setStop ( 1000, -1, 1 );
   }

   else if ( method == "sequential" ) {
      solver = purityTest ? new kMeansSeq ( datasetIn, trueLabelsIn ) : new kMeansSeq ( datasetIn );
      solver->setStop ( 1000, -1, 1 );
   }

   else if ( method == "kmeansSGD" ) {
      kMeansSGD * slv = purityTest ? new kMeansSGD ( datasetIn, trueLabelsIn ) : new kMeansSGD ( datasetIn );
      slv->setBatchSize ( 20*size );
      solver = slv;
      solver->setStop ( solver->size(), -1, 8 );
   }

   else {
      clog << "Unknown method " << method << "; available methods: sequential, kmeans, kmeansSGD" << endl;
      return 1;
   }

   solver->setK ( k );

   if ( rank == 0 ) {
      clog << "-----------------------------------------" << endl;
      clog << "Test name: " << test << endl;
      clog << "Dataset source: ./benchmarks/" << test << ".txt" << endl;
      clog << "Dataset size: " << solver->size() << endl;
      clog << "Clusters: " << k << endl;
      if ( purityTest ) clog << "True labels source: ./benchmarks/" << test << "-truelabels.txt" << endl;
      clog << "Method: " << method << endl;
      clog << "-----------------------------------------" << endl;
   }

   if ( method != "sequential" || rank == 0 ) {
      tm.start();
      solver->solve ();
      tm.stop();
   }

   if ( rank == 0 ) {
      clog << "Elapsed time: " << tm.getTime() << " msec" << endl;
      clog << "Converged in " << solver->getIter() << " iterations" << endl;
      if ( purityTest ) clog << "Clustering purity: " << solver->purity() << endl;
      clog << "-----------------------------------------" << endl;

      clog << "Cluster counts:" << endl;
      for ( int kk = 0; kk < k; ++kk ) clog << solver->getClusterCount(kk) << " ";
      clog << endl;

      clog << "-----------------------------------------" << endl;

      // cout << (*solver);
   }

   delete solver;

   MPI_Finalize ();
   return 0;
}
