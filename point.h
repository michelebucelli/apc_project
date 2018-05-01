#ifndef _POINT_H
#define _POINT_H

#include <vector>
#include <ostream>
#include <cassert>
#include <mpi.h>

// Class used for labeled points
class point {
private:
   // Dimension of the point
   unsigned int n;

   // Vector of the coordinates of the point
   std::vector<double> coords;

   // Label of the points (indicates the cluster to which the point belongs)
   int label = -1;

   // True label of the point
   int trueLabel = -1;

public:
   point ( unsigned int nn ) : n(nn), coords(nn,0) { }
   point ( unsigned int nn, std::vector<double> cc ) : n(nn), coords(cc) { coords.resize(nn,0); }

   // Coordinate access : operator[]
   double& operator[] ( unsigned int idx ) {
      assert ( idx < n );
      return coords[idx];
   }

   const double& operator[] ( unsigned int idx ) const {
      assert ( idx < n );
      return coords[idx];
   }

   // Get dimension
   const unsigned int getN ( void ) const { return n; }

   // Get vector raw data (for communication)
   double * data ( void ) { return coords.data(); }
   const double * data ( void ) const { return coords.data(); }

   // Label get and set
   const int getLabel ( void ) const { return label; }
   void setLabel ( int l ) { label = l; }

   // True label get and set
   int getTrueLabel ( void ) const { return trueLabel; }
   void setTrueLabel ( int l ) { trueLabel = l; }
};

// Output the point on a stream
// Format: single line,
// <label> <coord. 0> <coord. 1> ... <coord. N>
std::ostream& operator<< ( std::ostream&, const point& );

// Squared distance between two points
double dist2 ( const point &, const point & );

// Element-wise operations between points
// Useful for calculating centroids
point operator+ ( const point &, const point & );
point operator- ( const point &, const point & );
point& operator+= ( point &, const point & );
point operator/ ( const point &, double );

// Sum points across processes
// Used for parallel computation of the centroids
void mpi_point_reduce ( point* );

// Point send and receive
// The dimension of the point is required before when receiving a point, in order
// to properly allocate memory. These functions communicate labels as well
void  mpi_point_send ( unsigned int, const point& ); // Send point
point mpi_point_recv ( unsigned int, unsigned int ); // Receive point

#endif
