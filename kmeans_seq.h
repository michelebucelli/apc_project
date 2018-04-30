#ifndef _KMEANS_SEQ_H
#define _KMEANS_SEQ_H

#include "kmeans_base.h"

// The class performs classic kmeans algorithm without parallelization
// Used for timing reference
class kMeansSeq : public kMeansBase {
public:
   kMeansSeq ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) :
      kMeansBase ( nn, a, b ) { }

   // Randomize and compute centroids are overridden to be without parallelization
   // Function to recompute the centroids
   void computeCentroids ( void ) override;

   // Solve method
   void solve ( void ) override;
};

#endif
