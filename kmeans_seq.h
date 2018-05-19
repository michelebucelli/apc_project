#ifndef _KMEANS_SEQ_H
#define _KMEANS_SEQ_H

#include "kmeans_base.h"

// The class performs classic kmeans algorithm without parallelization
// Used for timing reference
template<typename dist_type = dist_euclidean>
class kMeansSeq : public kMeansBase<dist_type> {
public:
   kMeansSeq ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) :
      kMeansBase<dist_type> ( nn, a, b ) { }

   // Randomize and compute centroids are overridden to be without parallelization
   // Function to recompute the centroids
   void computeCentroids ( void ) override;

   // Solve method
   void solve ( void ) override;
};

template<typename dist_type>
void kMeansSeq<dist_type>::computeCentroids ( void ) {
   this->centroids = std::vector<point> ( this->k, point(this->n) );

   for ( auto & i : this->dataset ) {
      int lab = i.getLabel();
      for ( unsigned int nn = 0; nn < this->n; ++nn )
         this->centroids[lab][nn] += i[nn] / this->counts[lab];
   }
}

template<typename dist_type>
void kMeansSeq<dist_type>::solve ( void ) {
   this->randomize();

   this->iter = 0;

   // This variable counts how many points have changed label during each iteration
   int changes = this->stoppingCriterion.minLabelChanges + 1;
   double centroidDispl = this->stoppingCriterion.minCentroidDisplacement + 1;
   std::vector<point> oldCentroids;

   this->computeCentroids();

   while ( (this->stoppingCriterion.maxIter <= 0 || this->iter < this->stoppingCriterion.maxIter)
        && (this->stoppingCriterion.minLabelChanges <= 0 || changes >= this->stoppingCriterion.minLabelChanges)
        && (this->stoppingCriterion.minCentroidDisplacement <= 0 || centroidDispl >= this->stoppingCriterion.minCentroidDisplacement) ) {

      if ( this->stoppingCriterion.minCentroidDisplacement > 0 )
         oldCentroids = this->centroids;

      changes = 0;

      // Assigns each point to the group of the closest centroid
      for ( unsigned int i = 0; i < this->dataset.size(); i++ ) {
         double nearestDist = this->dist ( this->dataset[i], this->centroids[0] );
         int nearestLabel = 0;

         for ( unsigned int kk = 1; kk < this->k; ++kk ) {
            double d = this->dist ( this->dataset[i], this->centroids[kk] );

            if ( d < nearestDist ) {
               nearestDist = d;
               nearestLabel = kk;
            }
         }

         int oldLabel = this->dataset[i].getLabel();
         if ( oldLabel != nearestLabel ) {
            changes++;
            this->counts[oldLabel]--;
            this->counts[nearestLabel]++;
            this->dataset[i].setLabel(nearestLabel);
         }

      }

      // Computes the centroids in the current configuration
      this->computeCentroids();

      // Compute the max displacement of the centroids
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
