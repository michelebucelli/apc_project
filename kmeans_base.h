#ifndef _KMEANS_BASE_H
#define _KMEANS_BASE_H

#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>

#include "point.h"

struct kMeansStop {
   // Maximum iterations
   // Negative value means ignore
   int maxIter = 1;

   // Minimum centroid displacement
   // Negative value means ignore
   double minCentroidDisplacement = -1;

   // Minimum number of labels that change at each iteration
   // Negative value means ignore
   int minLabelChanges = 1;
};

typedef std::vector<point> kMeansDataset;
std::istream& operator>> ( std::istream &, kMeansDataset & );

// K-means solver base class
class kMeansBase {
protected:
   // Number of clusters we are looking for
   unsigned int k = 1;

   // Dimensions of the points
   unsigned int n = 1;

   // Points of the data set
   kMeansDataset dataset;

   // Centroids
   // Centroid for cluster of label 0 is centroids[0], etc...
   std::vector<point> centroids;

   // Counts of the points assigned to each cluster
   std::vector<int> counts;

   // Iterations counter
   int iter = 0;

   // Stopping criterion
   kMeansStop stoppingCriterion;

   // Protected constructor that allows derived classes to construct  without a
   // dataset
   kMeansBase ( unsigned int nn ) : n(nn) { }

public:
   // Constructor: requires the dimension of the points and the dataset, as a range
   kMeansBase ( unsigned int, kMeansDataset::const_iterator, kMeansDataset::const_iterator );

   // Destructor
   virtual ~kMeansBase ( void ) = default;

   // Getter and setter for the stopping criterion
   void setStop ( int maxIter, double minDispl, int minLabCh ) {
      stoppingCriterion.maxIter = maxIter;
      stoppingCriterion.minCentroidDisplacement = minDispl;
      stoppingCriterion.minLabelChanges = minLabCh;
   }
   kMeansStop getStop ( void ) const { return stoppingCriterion; }

   // Miscellaneous getters and setters
   const std::vector<point> & getDataset ( void ) const { return dataset; }
   unsigned int getN ( void ) const { return n; }
   void setK ( unsigned int );
   unsigned int getK ( void ) const { return k; }
   unsigned int size ( void ) const { return dataset.size(); }
   unsigned int getIter ( void ) const { return iter; }

   // Solve function
   virtual void solve ( void ) = 0;

   // Assigns random labels to the points of the dataset
   // Must be called after k has been set
   virtual void randomize ( void );

   // Function to compute the centroids
   virtual void computeCentroids ( void ) = 0;

   // Set the true labels from a vector
   virtual void setTrueLabels ( std::vector<int>::const_iterator, std::vector<int>::const_iterator, int = -1 );

   // Compute and return the purity of the clustering
   // Each cluster is assigned to the true label that is most frequent in it, then
   // we sum up the assignments to that label. Purity is the fraction of points in
   // the dataset that were assigned to the corresponding "true" cluster
   // True labels need to be set for the function to work (use setTrueLabels for that...)
   virtual double purity ( void ) const;

   // Output of the dataset on a stream
   // Output is made in an Octave/MatLab-like syntax to facilitate interaction
   // with other scripts
   virtual void printOutput ( std::ostream& ) const;
};

// Read a vector of integers from a stream
// Used to read true labels from file
std::istream& operator>> ( std::istream&, std::vector<int> & );

#endif
