#ifndef _KMEANS_H
#define _KMEANS_H

#include "kmeans_parallel.h"

class kMeansG : public kMeansParallelBase {
public:
   kMeansG ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) :
      kMeansParallelBase ( nn, a, b ) { }

   // Solve method
   void solve ( void ) override;
};

#endif
