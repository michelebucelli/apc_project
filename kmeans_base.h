#ifndef _KMEANS_BASE_H
#define _KMEANS_BASE_H

#include <array>
#include <vector>
#include <numeric>
#include <istream>

// Type used for real values
typedef double real;

// Type used for points
// Template parameter N is the dimension (number of coordinates) of the point
template < unsigned int N >
using point = std::array<real,N>;

// Class used for labelled points
// Template parameter N is the dimension
template < unsigned int N >
class labPoint : point<N> {
private:
   unsigned int label;

public:
   labPoint ( const point<N> &p, unsigned int l = 0 ) : point<N>(p), label(l) { }
};

// Squared distance between two points
template < unsigned int N >
real dist2 ( const point<N> &a, const point<N> &b ) {
   real sum = 0;
   for ( size_t i = 0; i < N; ++i ) {
      real x = a[i] - b[i];
      sum += x*x;
   }
   return sum;
}

// K-means solver base class
template < unsigned int N >
class kMeansBase {
private:
   // Number of clusters we are looking for
   unsigned int k;

   // Points of the data set
   std::vector<labPoint<N>> dataset;

   // Centroids
   // Centroid for cluster of label 0 is centroids[0], etc...
   std::vector<point<N>> centroids;

public:
   // Constructor
   kMeansBase ( const std::vector<point<N>> & pts ) {
      for ( const auto & pt : pts )
         dataset.emplace_back ( pt );
   };

   kMeansBase ( std::istream& in ) { in >> (*this); }

   // Function to add a point
   void addPoint ( const point<N>& p ) {
      dataset.push_back( labPoint<N>(p) );
   }

   // Getter and setter for the number of clusters
   void setK ( unsigned int kk ) { k = kk; }
   unsigned int getK ( void ) const { return k; }

   // Get the dimension of the dataset
   size_t size ( void ) const { return dataset.size(); }

   // Virtual solve function
   // Each derived class shall implement their own solving algorithm
   // void solve ( void ) = 0;

   // Function to recompute the centroids
   void computeCentroids ( void ) {
   }
};

template < unsigned int N >
std::istream& operator>> ( std::istream& in, kMeansBase<N> &km ) {
   unsigned int n = 0;
   real tmp = 0;
   point<N> p;

   while ( in >> tmp ) {
      p[n] = tmp;
      if ( n == N-1 ) {
         km.addPoint ( p );
         n = 0;
      }
      else n++;
   }

   return in;
}

#endif
