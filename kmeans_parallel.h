#ifndef _KMEANS_PARALLEL_H
#define _KMEANS_PARALLEL_H

#include "kmeans_base.h"

// Parallel k-means base class
// Computations of base functions ( computeCentroids, randomize ) are done in
// parallel. Each process is meant to store only a portion of the dataset.
// Thus, the field counts contains only local counts of points in each cluster
template<typename dist_type = dist_euclidean>
class kMeansParallelBase : public kMeansBase<dist_type> {
protected:
   // Info about the portion of dataset assigned to the process
   int datasetSize = 0; // Size of the complete dataset
   int datasetShare = 0; // Size of the local share of the dataset
   int datasetBegin = 0; // Index of the complete dataset where the local portion begins
public:
   kMeansParallelBase ( unsigned int, kMeansDataset::const_iterator, kMeansDataset::const_iterator );

   void randomize ( void ) override;
   void computeCentroids ( void ) override;
   double purity ( void ) const override;

   void setTrueLabels ( std::vector<int>::const_iterator, std::vector<int>::const_iterator, int = -1 ) override;

   // We have to override here because the dataset is split across different processes.
   // Output is done by process 0, which collects the results from other processes too
   void printOutput ( std::ostream& ) const override;
};

template<typename dist_type>
kMeansParallelBase<dist_type>::kMeansParallelBase ( unsigned int n, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b )
   : kMeansBase<dist_type> (n) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   datasetSize = b - a;
   int r = datasetSize % size;

   datasetShare = datasetSize / size + ( rank < r );
   datasetBegin = ( rank < r ? datasetShare * rank : (datasetShare + 1)*r + datasetShare*(rank - r) );

   this->dataset.assign ( a + datasetBegin, a + datasetBegin + datasetShare );
}

template<typename dist_type>
void kMeansParallelBase<dist_type>::randomize ( void ) {
   std::default_random_engine eng;
   std::uniform_int_distribution<unsigned int> dist ( 0, this->k - 1 );

   this->counts = std::vector<int>(this->k,0);

   for ( unsigned int i = 0; i < this->dataset.size(); i += 1 ) {
      eng.seed ( (i + datasetBegin) * 1000 );
      unsigned int lab = dist(eng);
      this->dataset[i].setLabel ( lab );
      this->counts[lab]++;
   }
}

template<typename dist_type>
void kMeansParallelBase<dist_type>::computeCentroids ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Centroids are computed in parallel
   // Each process accumulates, for each cluster, the sum of the points in that
   // cluster and their amount, over a portion of the whole dataset
   // Partial results are then gathered by process 0, who computes the average
   // and assign the result to the centroids member

   this->centroids = std::vector<point> ( this->k, point(this->n) );

   // Each process computes the local sums
   for ( unsigned int i = 0; i < this->dataset.size(); i++ ) {
      unsigned int l = this->dataset[i].getLabel();
      for ( unsigned int nn = 0; nn < this->n; ++nn )
         this->centroids[l][nn] += this->dataset[i][nn];
   }

   // Cluster counts are collected across processes
   std::vector<int> allcounts ( this->k, 0 );
   MPI_Allreduce ( this->counts.data(), allcounts.data(), this->k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

   // Partial sums are collected and the average is calculated
   for ( unsigned int kk = 0; kk < this->k; ++kk ) {
      mpi_point_allreduce ( &this->centroids[kk] );
      this->centroids[kk] = this->centroids[kk] / allcounts[kk];
   }
}

template<typename dist_type>
double kMeansParallelBase<dist_type>::purity ( void ) const {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // True labels of the clusters
   std::vector<int> trueLabels ( this->k, -1 );

   // Counts of the labels assigned to each cluster
   // counts[i*k + j] = n means that the cluster i has n elements with label j
   std::vector<int> counts ( this->k * this->k, 0 );

   // Iterate through the whole dataset and compute the counts
   for ( const auto & i : this->dataset ) {
      counts[ i.getLabel()*this->k + i.getTrueLabel() ] += 1;
   }

   MPI_Allreduce ( MPI_IN_PLACE, counts.data(), this->k * this->k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

   // Compute the true labels
   for ( unsigned int kk = 0; kk < this->k; ++kk ) {
      int maxIdx = 0, maxCount = counts[kk * this->k];

      for ( unsigned int j = 1; j < this->k; ++j ) {
         if ( counts[kk * this->k + j] > maxCount ) {
            maxCount = counts[kk * this->k + j];
            maxIdx = j;
         }
      }

      trueLabels[kk] = maxIdx;
   }

   // Compute purity
   int result = 0;

   for ( unsigned int i = 0; i < this->dataset.size(); ++i )
      if ( this->dataset[i].getTrueLabel() == trueLabels[this->dataset[i].getLabel()] ) result += 1;

   MPI_Allreduce ( MPI_IN_PLACE, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

   return result / double(datasetSize);
}

template<typename dist_type>
void kMeansParallelBase<dist_type>::setTrueLabels ( std::vector<int>::const_iterator a, std::vector<int>::const_iterator b, int offset ) {
   kMeansBase<dist_type>::setTrueLabels ( a + datasetBegin, a + datasetBegin + datasetShare, offset );
}

template<typename dist_type>
void kMeansParallelBase<dist_type>::printOutput ( std::ostream &out ) const {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Process 0 receives the data from the others and prints it
   if ( rank == 0 ) {
      // General info about the dataset
      out << "dim = " << this->n << ";\nclusters = " << this->k << ";\n";
      out << "dataset = [ " << this->dataset[0];

      // Print process 0's own portion of dataset
      unsigned int i = 1;
      for ( ; i < this->size(); ++i )
         out << ";\n" << this->dataset[i];

      // Receive and print the others' portions
      for ( int proc = 1; proc < size; ++proc ) {
         // First receive the number of points of that process ...
         int share = 0;
         MPI_Recv ( &share, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

         // ... then receive and print the actual points
         for ( int i = 0; i < share; ++i )
            out << ";\n" << mpi_point_recv ( proc, this->n );
      }

      out << "];";
   }

   // Other processes just send the result to rank 0
   // First they send the local share of points, then they send the actual points
   else {
      int share = datasetShare;
      MPI_Send ( &share, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );

      for ( int i = 0; i < share; ++i )
         mpi_point_send ( 0, this->dataset[i] );
   }
}

#endif
