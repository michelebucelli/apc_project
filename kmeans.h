#ifndef _KMEANS_H
#define _KMEANS_H

#include "kmeans_base.h"

class kMeans : public kMeansBase {
public:
   kMeans ( unsigned int nn, const std::vector<point> & pts ) : kMeansBase ( nn, pts ) { }
   kMeans ( std::istream& in ) : kMeansBase(in) { };

   // Function to recompute the centroids
   // Computation is executed in parallel
   // Move to clasic kMeans
   void computeCentroids ( void );

   // Assigns random labels to the points of the dataset
   // Must be called after k has been set
   // Process 0 generates the values and then sends them to the other processes
   // Move to classic kMeans
   void randomize ( void );

   // Solve method
   void solve ( void ) override;
};

#endif
