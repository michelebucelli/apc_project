#ifndef _KMEANS_PARALLEL_H
#define _KMEANS_PARALLEL_H

#include "kmeans_base.h"

// Parallel k-means base class
// Computations of base functions ( computeCentroids, randomize ) are done in
// parallel. Each process is meant to store only a portion of the dataset.
// Thus, the field counts contains only local counts of points in each cluster
class kMeansParallelBase : public kMeansBase {
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

#endif
