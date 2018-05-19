#include "kmeans_parallel.h"
using std::clog; using std::endl;

kMeansParallelBase::kMeansParallelBase ( unsigned int n, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) : kMeansBase(n) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   datasetSize = b - a;
   int r = datasetSize % size;

   datasetShare = datasetSize / size + ( rank < r );
   datasetBegin = ( rank < r ? datasetShare * rank : (datasetShare + 1)*r + datasetShare*(rank - r) );

   dataset.assign ( a + datasetBegin, a + datasetBegin + datasetShare );
}

void kMeansParallelBase::randomize ( void ) {
   std::default_random_engine eng;
   std::uniform_int_distribution<unsigned int> dist ( 0, k - 1 );

   counts = std::vector<int>(k,0);

   for ( unsigned int i = 0; i < dataset.size(); i += 1 ) {
      eng.seed ( (i + datasetBegin) * 1000 );
      unsigned int lab = dist(eng);
      dataset[i].setLabel ( lab );
      counts[lab]++;
   }
}

void kMeansParallelBase::computeCentroids ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Centroids are computed in parallel
   // Each process accumulates, for each cluster, the sum of the points in that
   // cluster and their amount, over a portion of the whole dataset
   // Partial results are then gathered by process 0, who computes the average
   // and assign the result to the centroids member

   centroids = std::vector<point> ( k, point(n) );

   // Each process computes the local sums
   for ( unsigned int i = 0; i < dataset.size(); i++ ) {
      unsigned int l = dataset[i].getLabel();
      for ( unsigned int nn = 0; nn < n; ++nn )
         centroids[l][nn] += dataset[i][nn];
   }

   // Cluster counts are collected across processes
   std::vector<int> allcounts ( k, 0 );
   MPI_Allreduce ( counts.data(), allcounts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

   // Partial sums are collected and the average is calculated
   for ( unsigned int kk = 0; kk < k; ++kk ) {
      mpi_point_allreduce ( &centroids[kk] );
      centroids[kk] = centroids[kk] / allcounts[kk];
   }
}

double kMeansParallelBase::purity ( void ) const {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // True labels of the clusters
   std::vector<int> trueLabels ( k, -1 );

   // Counts of the labels assigned to each cluster
   // counts[i*k + j] = n means that the cluster i has n elements with label j
   std::vector<int> counts ( k*k, 0 );

   // Iterate through the whole dataset and compute the counts
   for ( const auto & i : dataset ) {
      counts[ i.getLabel()*k + i.getTrueLabel() ] += 1;
   }

   MPI_Allreduce ( MPI_IN_PLACE, counts.data(), k*k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

   // Compute the true labels
   for ( unsigned int kk = 0; kk < k; ++kk ) {
      int maxIdx = 0, maxCount = counts[kk * k];

      for ( unsigned int j = 1; j < k; ++j ) {
         if ( counts[kk * k + j] > maxCount ) {
            maxCount = counts[kk * k + j];
            maxIdx = j;
         }
      }

      trueLabels[kk] = maxIdx;
   }

   // Compute purity
   int result = 0;

   for ( unsigned int i = 0; i < dataset.size(); ++i )
      if ( dataset[i].getTrueLabel() == trueLabels[dataset[i].getLabel()] ) result += 1;

   MPI_Allreduce ( MPI_IN_PLACE, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

   return result / double(datasetSize);
}

void kMeansParallelBase::setTrueLabels ( std::vector<int>::const_iterator a, std::vector<int>::const_iterator b, int offset ) {
   kMeansBase::setTrueLabels ( a + datasetBegin, a + datasetBegin + datasetShare, offset );
}

void kMeansParallelBase::printOutput ( std::ostream &out ) const {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Process 0 receives the data from the others and prints it
   if ( rank == 0 ) {
      // General info about the dataset
      out << "dim = " << n << ";\nclusters = " << k << ";\n";
      out << "dataset = [ " << dataset[0];

      // Print process 0's own portion of dataset
      unsigned int i = 1;
      for ( ; i < this->size(); ++i )
         out << ";\n" << dataset[i];

      // Receive and print the others' portions
      for ( int proc = 1; proc < size; ++proc ) {
         // First receive the number of points of that process ...
         int share = 0;
         MPI_Recv ( &share, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

         // ... then receive and print the actual points
         for ( int i = 0; i < share; ++i )
            out << ";\n" << mpi_point_recv ( proc, n );
      }

      out << "];";
   }

   // Other processes just send the result to rank 0
   // First they send the local share of points, then they send the actual points
   else {
      int share = datasetShare;
      MPI_Send ( &share, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );

      for ( int i = 0; i < share; ++i )
         mpi_point_send ( 0, dataset[i] );
   }
}
