#ifndef _KMEANS_BASE_H
#define _KMEANS_BASE_H

#include <vector>
#include <numeric>
#include <istream>
#include <cassert>
#include <mpi.h>

// Type used for real values
typedef double real;

// Class used for labelled points
class point {
private:
   // Dimension of the point
   unsigned int n;

   // Vector of the coordinates of the point
   std::vector<real> coords;

   // Label of the points (indicates the cluster to which the point belongs)
   unsigned int label;

public:
   point ( unsigned int nn ) : n(nn), coords(nn,0) { }
   point ( unsigned int nn, std::vector<real> cc ) : n(nn), coords(cc) { coords.resize(nn); }

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
   unsigned int getN ( void ) const { return n; }

   // Get vector raw data (for communication)
   real * data ( void ) { return coords.data(); }
   const real * data ( void ) const { return coords.data(); }

   // Label get and set
   unsigned int getLabel ( void ) const { return label; }
   void setLabel ( unsigned int l ) { label = l; }
};

// Squared distance between two points
real dist2 ( const point &, const point & );

// Element-wise operations between points
// Useful for calculating centroids
point operator+ ( const point &, const point & );
point& operator+= ( const point &, const point & );
point operator/ ( const point &, real );

// Routines for communicating point via MPI
// When sending point with these routines, only the coordinates are sent
// Therefore, the dimension of the point has to be known in advance by the
// receiving process when calling mpi_point_recv
void  mpi_point_send ( unsigned int, const point& ); // Send point
point mpi_point_recv ( unsigned int, unsigned int ); // Receive point

// K-means solver base class
class kMeansBase {
private:
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
public:
   // Constructor
   kMeansBase ( unsigned int nn, const std::vector<point> & pts ) :
      n(nn), dataset(pts) { }

   kMeansBase ( std::istream& in ) { in >> (*this); }

   // Getter and setter for the number of clusters
   void setK ( unsigned int kk ) { k = kk; }
   unsigned int getK ( void ) const { return k; }

   // Get the dimension of the points
   unsigned int getN ( void ) const { return n; }

   // Get the dimension of the dataset
   unsigned int size ( void ) const { return dataset.size(); }

   // Virtual solve function
   // Each derived class shall implement their own solving algorithm
   // void solve ( void ) = 0;

   // Function to recompute the centroids
   void computeCentroids ( void );
};

// Read a dataset from an input stream and stores it into a kmeans object
// File format : <dimension of the points> <coordinates 1> <coordinates 2> ...
std::istream& operator>> ( std::istream&, kMeansBase & );

#endif
