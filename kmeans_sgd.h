#ifndef _KMEANS_SGD_H
#define _KMEANS_SGD_H

#include "kmeans_parallel.h"
#include "timer.h"

// K-Means algorithm using stochastic gradient descent

// At each iteration, we pick a random point from the dataset, we compute its
// distance from all the centroids and we assign to it the label of the closest
// centroid. After doing that, we recompute the centroids

// Iterations stop when there are no more changes in the centroids

template<typename dist_type = dist_euclidean>
class kMeansSGD : public kMeansParallelBase<dist_type> {
private:
   // Batch size
   // Each process will sample batchSize/nproc elements from its portion of the
   // dataset at each iteration
   int batchSize = 20;
public:
   kMeansSGD ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) :
      kMeansParallelBase<dist_type> ( nn, a, b ) { }

   void solve ( void ) override;

   // Batch size get-set
   int getBatchSize ( void ) const { return batchSize; }
   void setBatchSize ( int bs ) { batchSize = bs; }
};

template<typename dist_type>
void kMeansSGD<dist_type>::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   // Parallelization: we draw entries in batches. Each process draws a portion
   // of the batch and performs the algorithm, Size of each batch is in the
   // member batchSize

   this->iter = 0;

   // Randomize initial assignments
   this->randomize();
   this->computeCentroids();

   int changesCount = this->stoppingCriterion.minLabelChanges + 1;
   double centroidDispl = this->stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   // Counts for how many subsequent iterations the stopping criteria are met; the
   // algorithm stops when it reaches a fixed number (see below)
   // This helps checking that actual convergence takes place
   int stopIters = 0;

   std::default_random_engine eng ( 10000 * rank );
   std::uniform_int_distribution<unsigned int> distro ( 0, this->dataset.size() - 1 );

   std::vector<int> oldGlobalCounts ( this->k, 0 );
   std::vector<int> newGlobalCounts ( this->k, 0 );

   while ( stopIters < 15 ) {
      // Checks if stopping criterion is satisfied at this iteration, and possibly
      // increment the counter
      if ( (this->stoppingCriterion.maxIter <= 0 || this->iter < this->stoppingCriterion.maxIter)
        && (this->stoppingCriterion.minLabelChanges <= 0 || changesCount >= this->stoppingCriterion.minLabelChanges)
        && (this->stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= this->stoppingCriterion.minCentroidDisplacement) ) stopIters = 0;
      else stopIters++;

      if ( this->stoppingCriterion.minCentroidDisplacement > 0 )
         oldCentroids = this->centroids;

      changesCount = 0;

      std::vector<point> centroidDiff  ( this->k, point(this->n) );

      MPI_Allreduce ( this->counts.data(), oldGlobalCounts.data(), this->k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      // Determines the changes to be made on the assigned portion of the batch.
      // Those changes are initially stored in the vector changes, and only later
      // sent to the other processes and applied
      for ( int i = rank; i < batchSize; i += size ) {
         // Randomly selects a point
         unsigned int idx = distro(eng);

         // Find the nearest centroid to the selected point
         int nearestLabel = 0;
         double nearestDist = this->dist ( this->dataset[idx], this->centroids[0] );

         for ( unsigned int kk = 1; kk < this->k; ++kk ) {
            double d = this->dist ( this->dataset[idx], this->centroids[kk] );
            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         // Assigns the chosen label
         int oldLabel = this->dataset[idx].getLabel();

         if ( oldLabel != nearestLabel ) {
            this->counts[oldLabel] -= 1;
            this->counts[nearestLabel] += 1;
            this->dataset[idx].setLabel(nearestLabel);
            changesCount++;

            for ( unsigned int nn = 0; nn < this->n; ++nn ) {
               centroidDiff[oldLabel][nn] -= this->dataset[idx][nn];
               centroidDiff[nearestLabel][nn] += this->dataset[idx][nn];
            }
         }
      }

      MPI_Allreduce ( this->counts.data(), newGlobalCounts.data(), this->k, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      for ( unsigned int kk = 0; kk < this->k; ++kk ) {
         mpi_point_allreduce ( &centroidDiff[kk] );
         for ( unsigned int nn = 0; nn < this->n; ++nn )
            this->centroids[kk][nn] = ( this->centroids[kk][nn] * oldGlobalCounts[kk] + centroidDiff[kk][nn] ) / newGlobalCounts[kk];
      }

      MPI_Allreduce ( MPI_IN_PLACE, &changesCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      // Compute the max displacement of the centroids for the stopping criterion
      if ( this->stoppingCriterion.minCentroidDisplacement > 0 ) {
         centroidDispl = 0;
         for ( unsigned kk = 0; kk < this->k; ++kk ) {
            double displ = this->dist ( oldCentroids[kk], this->centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++this->iter;
   }
}

#endif
