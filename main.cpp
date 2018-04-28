#include "kmeans_seq.h"
#include "kmeans.h"
#include "kmeans_sgd.h"

#include "timer.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

#include "GetPot"

using std::cout;
using std::clog;
using std::endl;

void printHelp ( void ) {
   clog << "Stochastic Gradient Descent applied to K-Means" << endl;
   clog << "Michele Bucelli, Jose' Villafan" << endl;
   clog << "Algorithms and Parallel Computing course" << endl;
   clog << "Politecnico di Milano - A.Y. 2017/2018" << endl << endl;
   clog << "Purpose: the program takes in input a dataset consisting of points\n"
           "in R^n and performs k-means clustering on it, using one of three\n"
           "possible algorithms, making use of parallel computing where needed." << endl << endl;
   clog << "Usage: mpirun -np <processes> main.out -t|--test <testname>\n"
        << "              -k <clusters> -m|--method <method> [--purity]\n"
        << "              [--no-output] [--no-log]" << endl << endl;
   clog << "Output: result of the clustering is printed on the standard output\n"
        << "in an Octave/MatLab-compatible format." << endl << endl;
   clog << "Parameters:\n"
        << " -t|--test <testname> : specifies the name of the test; there must\n"
        << "      be a corresponding <testname>.txt file in the benchmarks\n"
        << "      subfolder; if purity testing is enabled, there must also be\n"
        << "      a <testname>-truelabels.txt file in the benchmarks subfolder\n"
        << " -k <clusters> : number of clusters the algorithm should produce\n"
        << " -m|--method <method> : specifies the method to be used; available\n"
        << "      methods are:\n"
        << "         sequential - performs k-means without parallelization\n"
        << "         kmeans - performs k-means in parallel\n"
        << "         kmeansSGD - performs k-means with stochastic gradient descent\n"
        << "         compare - tests all methods reporting timing results; no\n"
        << "           output is produced in this case\n"
        << " --purity : enables purity evaluation for the produced clusters\n"
        << " --no-output : disables output result\n"
        << " --no-log : disables logging\n" << endl;
}

int main ( int argc, char * argv[] ) {
   MPI_Init ( &argc, &argv );
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Read parameters from command line
   GetPot cmdLine ( argc, argv );

   if ( cmdLine.search("-h") || cmdLine.search("--help") ) {
      printHelp();
      return 0;
   }

   std::string test = cmdLine.follow("g1M-20-5", 2, "-t", "--test" ); // Test name
   int k = cmdLine.follow(5, 1, "-k" ); // Number of clusters
   std::string method = cmdLine.follow("sequential", 2, "-m", "--method" ); // Method : sequential, kmeans, kmeansSGD, compare
   bool purityTest = cmdLine.search("-p") || cmdLine.search("--purity" ); // Purity flag test
   bool suppressOutput = cmdLine.search("--no-output"); // Disable output
   bool suppressLog = cmdLine.search("--no-log"); // Disable log

   // Read the dataset
   std::ifstream datasetIn ( "./benchmarks/" + test + ".txt" );
   kMeansDataset dataset;
   datasetIn >> dataset;
   unsigned int n = dataset[0].getN();

   // Read the true labels
   std::ifstream trueLabelsIn ( "./benchmarks/" + test + "-truelabels.txt" );
   std::vector<int> trueLabels;
   trueLabelsIn >> trueLabels;

   // Dataset info on log
   if ( rank == 0 && !suppressLog ) {
      clog << "-----------------------------------------" << endl;
      clog << "Test name: " << test << endl;
      clog << "Dataset source: ./benchmarks/" << test << ".txt" << endl;
      clog << "Dataset size: " << dataset.size() << endl;
      clog << "Dataset dimension: " << n << endl;
      clog << "Clusters: " << k << endl;
      if ( purityTest ) clog << "True labels source: ./benchmarks/" << test << "-truelabels.txt" << endl;
      clog << "-----------------------------------------" << endl;
   }

   std::vector<std::string> methods = { "sequential", "kmeans", "kmeansSGD" };

   for ( auto i : methods ) {
      MPI_Barrier(MPI_COMM_WORLD);

      if ( method != i && method != "compare" ) continue;
      if ( i == "sequential" && rank != 0 ) continue;

      kMeansBase * solver = nullptr;

      if ( i == "sequential" ) {
         solver = new kMeansSeq ( n, dataset );
         solver->setStop ( 100, -1, 1 );
      }

      else if ( i == "kmeans" ) {
         solver = new kMeans ( n, dataset );
         solver->setStop ( -1, -1, 1 );
      }

      else if ( i == "kmeansSGD" ) {
         auto tmp = new kMeansSGD ( n, dataset );
         tmp->setBatchSize ( 360 );
         tmp->setStop ( -1, -1, 20 );

         solver = tmp;
      }

      if ( purityTest ) solver->setTrueLabels ( trueLabels );
      solver->setK ( k );

      timer tm;

      tm.start();
      solver->solve();
      tm.stop();

      if ( rank == 0 && !suppressLog ) {
         clog << "Method: " << i << endl;
         clog << "Elapsed time: " << tm.getTime() << " msec" << endl;
         clog << "Converged in " << solver->getIter() << " iterations" << endl;
         if ( purityTest ) clog << "Clustering purity: " << solver->purity() << endl;
         clog << "Cluster counts: ";
         for ( int kk = 0; kk < k; ++kk ) clog << solver->getClusterCount(kk) << " ";
         clog << endl;

         clog << "-----------------------------------------" << endl;

         if ( !suppressOutput && method != "compare" ) cout << (*solver);
      }

      delete solver;
   }

   MPI_Finalize ();
   return 0;
}
