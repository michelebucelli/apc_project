#include "kmeans_seq.h"
#include "kmeans.h"
#include "kmeans_sgd.h"

#include "timer.h"

#include <iostream>
#include <iomanip>
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
        << "       - sequential - performs k-means without parallelization\n"
        << "       - kmeans - performs k-means in parallel\n"
        << "       - kmeansSGD - performs k-means with stochastic gradient descent\n"
        << "       - compare - tests both kmeans and kmeansSGD methods reporting\n"
        << "         timing results; no output is produced in this case\n"
        << " --purity : enables purity evaluation for the produced clusters\n"
        << " --no-output : disables output result\n"
        << " -q|--quiet : disables logging\n" << endl;
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
   std::string method = cmdLine.follow("sequential", 2, "-m", "--method" ); // Method : sequential, kmeans, kmeansSGD, compare
   int k = cmdLine.follow(5, 1, "-k" ); // Number of clusters
   bool purityTest = cmdLine.search("-p") || cmdLine.search("--purity"); // Purity flag test
   bool suppressOutput = cmdLine.search("--no-output"); // Disable output
   bool suppressLog = cmdLine.search("-q") || cmdLine.search("--quiet"); // Disable log
   bool verbose = cmdLine.search("-v") || cmdLine.search("--verbose"); // Verbose log

   if ( rank == 0 && !suppressLog && verbose ) {
      clog << "-----------------------------------------" << endl;
      clog << "Dataset source: ./benchmarks/" << test << ".txt" << endl;
      if ( purityTest ) clog << "True labels source: ./benchmarks/" << test << "-truelabels.txt" << endl;
   }

   // Read the dataset
   std::ifstream datasetIn ( "./benchmarks/" + test + ".txt" );

   if ( datasetIn.fail() ) {
      if ( rank == 0 ) clog << "Error: couldn't read dataset file" << endl;
      return 1;
   }

   kMeansDataset dataset;
   datasetIn >> dataset;
   unsigned int n = dataset[0].getN();
   datasetIn.close();

   // Read the true labels
   std::ifstream trueLabelsIn ( "./benchmarks/" + test + "-truelabels.txt" );

   if ( purityTest && trueLabelsIn.fail() ) {
      if ( rank == 0 ) clog << "Error: couldn't read true labels file" << endl;
      return 1;
   }

   std::vector<int> trueLabels;
   trueLabelsIn >> trueLabels;
   trueLabelsIn.close();

   // Dataset info on log
   if ( rank == 0 && !suppressLog && verbose ) {
      clog << "-----------------------------------------" << endl;
      clog << "Test name: " << test << endl;
      clog << "Dataset size: " << dataset.size() << endl;
      clog << "Dataset dimension: " << n << endl;
      clog << "Clusters: " << k << endl;
      clog << "-----------------------------------------" << endl;
   }

   std::vector<std::string> methods = { "sequential", "kmeans", "kmeansSGD" };

   for ( auto i : methods ) {
      MPI_Barrier(MPI_COMM_WORLD);

      if ( method != i && method != "compare" ) continue;
      if ( i == "sequential" && (rank != 0 || method == "compare") ) continue;

      // Allocate and configurate the solver
      kMeansBase * solver = nullptr;

      // Sequential kMeans
      if ( i == "sequential" ) {
         solver = new kMeansSeq ( n, dataset.begin(), dataset.end() );
         solver->setStop ( 100, -1, 1 );

         if ( purityTest )
            solver->setTrueLabels ( trueLabels.begin(), trueLabels.end() );
      }

      // Parallel kMeans
      else if ( i == "kmeans" ) {
         solver = new kMeans ( n, dataset.begin(), dataset.end() );
         solver->setStop ( -1, -1, 1 );
      }

      // Stochastic gradient descent kMeans
      else if ( i == "kmeansSGD" ) {
         auto tmp = new kMeansSGD ( n, dataset.begin(), dataset.end() );

         tmp->setBatchSize ( 1000 );
         tmp->setStop ( -1, -1, 50 );

         solver = tmp;
      }

      solver->setK ( k );

      if ( purityTest )
         solver->setTrueLabels ( trueLabels.begin(), trueLabels.end() );

      // We delete the dataset, if it is no longer necessary
      if ( method != "compare" ) {
         dataset.resize(0, point(n));
         trueLabels.resize(0);
      }

      timer tm;

      tm.start();
      solver->solve();
      tm.stop();

      double purity = purityTest ? solver->purity() : 0;

      if ( rank == 0 && !suppressLog ) {
         if ( verbose ) {
            clog << "Method: " << i << endl;
            clog << "Elapsed time: " << tm.getTime() << " msec" << endl;
            clog << "Converged in " << solver->getIter() << " iterations" << endl;
            if ( purityTest ) clog << "Clustering purity: " << purity << endl;
            clog << "-----------------------------------------" << endl;
         }

         else {
            clog << std::setw(10) << i << " | " << std::setw(10) << tm.getTime() << " msec | " << std::setw(10) << solver->getIter() << " iter";
            if ( purityTest ) clog << " | " << std::setw(10) << purity << " purity";
            clog << endl;
         }
      }

      if ( !suppressOutput && method != "compare" ) solver->printOutput( cout );

      delete solver;
   }

   MPI_Finalize ();
   return 0;
}
