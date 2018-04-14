#ifndef _KMEANS_SEQ_H
#define _KMEANS_SEQ_H

#include "kmeans_base.h"

// The class performs classic kmeans algorithm without parallelization
// Used for timing reference
class kMeansSeq : public kMeansBase {
public:
   kMeansSeq ( unsigned int nn, const std::vector<point> & pts ) : kMeansBase ( nn, pts ) { }
   kMeansSeq ( std::istream& in ) : kMeansBase(in) { };
   kMeansSeq ( std::istream& datasetIn, std::istream& trueLabelsIn ) : kMeansBase ( datasetIn, trueLabelsIn ) { };

   // Randomize and compute centroids are overridden to be without parallelization
   // Function to recompute the centroids
   void computeCentroids ( void ) override;

   // Solve method
   void solve ( void ) override;
};

#endif
