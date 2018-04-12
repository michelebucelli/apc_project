#ifndef _KMEANS_BASE_H
#define _KMEANS_BASE_H

#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <limits>

#include <mpi.h>

// Type used for real values
typedef double real;

// Class used for labeled points
class point {
private:
   // Dimension of the point
   unsigned int n;

   // Vector of the coordinates of the point
   std::vector<real> coords;

   // Label of the points (indicates the cluster to which the point belongs)
   int label = -1;

   // True label of the point
   int trueLabel = -1;

public:
   point ( unsigned int nn ) : n(nn), coords(nn,0) { }
   point ( unsigned int nn, std::vector<real> cc ) : n(nn), coords(cc) { coords.resize(nn,0); }

   // Coordinate access : operator[]
   real& operator[] ( unsigned int idx ) {
      assert ( idx < n );
      return coords[idx];
   }

   const real& operator[] ( unsigned int idx ) const {
      assert ( idx < n );
      return coords[idx];
   }

   // Get dimension
   const unsigned int getN ( void ) const { return n; }

   // Get vector raw data (for communication)
   real * data ( void ) { return coords.data(); }
   const real * data ( void ) const { return coords.data(); }

   // Label get and set
   const int getLabel ( void ) const { return label; }
   void setLabel ( int l ) { label = l; }

   // True label get and set
   const int getTrueLabel ( void ) const { return trueLabel; }
   void setTrueLabel ( int l ) { trueLabel = l; }
};

// Output the point on a stream
// Format: single line,
// <label> <coord. 0> <coord. 1> ... <coord. N>
std::ostream& operator<< ( std::ostream&, const point& );

// Squared distance between two points
real dist2 ( const point &, const point & );

// Element-wise operations between points
// Useful for calculating centroids
point operator+ ( const point &, const point & );
point operator- ( const point &, const point & );
point& operator+= ( point &, const point & );
point operator/ ( const point &, real );

// Routines for communicating point via MPI
// When sending point with these routines, only the coordinates are sent
// Therefore, the dimension of the point has to be known in advance by the
// receiving process when calling mpi_point_recv
void  mpi_point_send ( unsigned int, const point& ); // Send point
point mpi_point_recv ( unsigned int, unsigned int ); // Receive point

struct kMeansStop {
   // Maximum iterations
   // Negative value means ignore
   int maxIter = 10;

   // Maximum centroid displacement
   // Negative value means ignore
   real minCentroidDisplacement = std::numeric_limits<double>::epsilon() * 20;

   // Minimum number of labels that change at each iteration
   // Negative value means ignore
   int minLabelChanges = 1;
};

// K-means solver base class
class kMeansBase {
protected:
   // Number of clusters we are looking for
   unsigned int k = 1;

   // Dimensions of the points
   unsigned int n = 1;

   // Points of the data set
   std::vector<point> dataset;

   // Centroids
   // Centroid for cluster of label 0 is centroids[0], etc...
   std::vector<point> centroids;

   friend std::istream& operator>> ( std::istream&, kMeansBase& );
   friend std::ostream& operator<< ( std::ostream&, const kMeansBase& );

   // Iterations counter
   int iter = 0;

   // Stopping criterion
   kMeansStop stoppingCriterion;
public:
   // Constructor
   kMeansBase ( unsigned int nn, const std::vector<point> & pts ) :
      n(nn), dataset(pts) { }

   // Constructor: reads the dataset from the given input stream
   kMeansBase ( std::istream& in ) { in >> (*this); }

   // Constructor: reads the dataset from the first input stream, and gets the
   // true labels from the second input stream
   kMeansBase ( std::istream& datasetIn, std::istream& trueLabelsIn ) {
      datasetIn >> (*this);
      getTrueLabels(trueLabelsIn);
   }

   // Destructor
   virtual ~kMeansBase ( void ) = default;

   // Getter and setter for the number of clusters
   void setK ( unsigned int );
   unsigned int getK ( void ) const { return k; }

   // Getter and setter for the stopping criterion
   void setStop ( int maxIter, real minDispl, int minLabCh ) {
      stoppingCriterion.maxIter = maxIter;
      stoppingCriterion.minCentroidDisplacement = minDispl;
      stoppingCriterion.minLabelChanges = minLabCh;
   }
   kMeansStop getStop ( void ) const { return stoppingCriterion; }

   // Get the dimension of the points
   unsigned int getN ( void ) const { return n; }

   // Get the dimension of the dataset
   unsigned int size ( void ) const { return dataset.size(); }

   // Virtual solve function
   // Each derived class shall implement their own solving algorithm
   virtual void solve ( void ) = 0;

   // Get number of iterations performed
   unsigned int getIter ( void ) const { return iter; }

   // Function to recompute the centroids
   // Computation is executed in parallel
   void computeCentroids ( void );

   // Assigns random labels to the points of the dataset
   // Must be called after k has been set
   // Process 0 generates the values and then sends them to the other processes
   void randomize ( void );

   // Reads the true labels from an input stream
   // The labels are assumed to be in the same order as the points in the dataset
   void getTrueLabels ( std::istream&, int = -1 );

   // Compute and return the purity of the clustering
   // Each cluster is assigned to the true label that is most frequent in it, then
   // we sum up the assignments to that label. Purity is the fraction of points in
   // the dataset that were assigned to the corresponding "true" cluster
   // True labels need to be set for the function to work (use getTrueLabels for that...)
   real purity ( void ) const;
};

// Read a dataset from an input stream and stores it into a kmeans object
// File format:
// <dimension of the points>
// <coordinates 0>
// <coordinates 1>
// ...
std::istream& operator>> ( std::istream&, kMeansBase & );

// Output the results on a stream
// Output file format:
// <dimension of the points>
// <number of clusters>
// <label 0> <coordinates 0>
// <label 1> <coordinates 1>
// ...
std::ostream& operator<< ( std::ostream&, const kMeansBase & );

#endif
