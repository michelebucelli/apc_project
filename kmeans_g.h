#ifndef _KMEANS_H
#define _KMEANS_H

#include "kmeans_parallel.h"
#include "timer.h"

template<typename dist_type = dist_euclidean>
class kMeansG : public kMeansParallelBase<dist_type> {
public:
   kMeansG ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) :
      kMeansParallelBase<dist_type> ( nn, a, b ) { }

   // Solve method
   void solve ( void ) override;
};

template<typename dist_type>
void kMeansG<dist_type>::solve ( void ) {
   int size; MPI_Comm_size ( MPI_COMM_WORLD, &size );
   int rank; MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

   this->iter = 0;

   int changesCount = this->stoppingCriterion.minLabelChanges + 1;
   double centroidDispl = this->stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   this->randomize();
   this->computeCentroids();

   while ( (this->stoppingCriterion.maxIter <= 0 || this->iter < this->stoppingCriterion.maxIter)
        && (this->stoppingCriterion.minLabelChanges <= 0 || changesCount >= this->stoppingCriterion.minLabelChanges)
        && (this->stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= this->stoppingCriterion.minCentroidDisplacement) ) {

      if ( this->stoppingCriterion.minCentroidDisplacement > 0 )
        oldCentroids = this->centroids;

      changesCount = 0;

      // Assigns each point to the group of the closest centroid. The changes to
      // be made are initially stored in the vector changes, and are applied only
      // later while broadcasting them to the other processes

      double nearestDist = 0, d = 0;
      int nearestLabel = 0;

      for ( unsigned int i = 0; i < this->dataset.size(); i += 1 ) {
         nearestDist = this->dist ( this->dataset[i], this->centroids[0] );
         nearestLabel = 0;

         // Finding the nearest of the centroids
         for ( unsigned int kk = 1; kk < this->k; ++kk ) {
            d = this->dist ( this->dataset[i], this->centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         int oldLabel = this->dataset[i].getLabel();
         if ( oldLabel != nearestLabel ) {
            this->counts[oldLabel] -= 1;
            this->counts[nearestLabel] += 1;
            this->dataset[i].setLabel(nearestLabel);
            changesCount++;
         }
      }

      // Recomputes the centroids in the current configuration
      this->computeCentroids();
      MPI_Allreduce ( MPI_IN_PLACE, &changesCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

      // Compute the max displacement of the centroids
      if ( this->stoppingCriterion.minCentroidDisplacement > 0 ) {
         centroidDispl = 0;
         for ( unsigned kk = 0; kk < this->k; kk += 1 ) {
            double displ = this->dist ( oldCentroids[kk], this->centroids[kk] );
            if ( displ > centroidDispl ) centroidDispl = displ;
         }
         centroidDispl = sqrt(centroidDispl);
      }

      ++this->iter;
   }

}

#endif
