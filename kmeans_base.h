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
// The template parameter is a type that has a member function dist that takes
// two const point& parameters and computes the distance between the points,
// according to the desired metric
template<typename dist_type = dist_euclidean>
class kMeansBase : public dist_type {
protected:
   using dist_type::dist;
   
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

std::istream& operator>> ( std::istream &in, kMeansDataset &km ) {
   unsigned int i = 0;
   unsigned int n = 0;
   double tmp = 0;

   in >> n;
   point p ( n );

   while ( in >> tmp ) {
      p[i] = tmp;
      if ( i == n - 1 ) {
         km.push_back(p);
         i = 0;
      }
      else i++;
   }

   return in;
}

template <typename dist_type>
kMeansBase<dist_type>::kMeansBase ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) : n(nn) {
   dataset.assign ( a, b );
}

template <typename dist_type>
void kMeansBase<dist_type>::setK ( unsigned int kk ) {
   k = kk;
   centroids = std::vector<point> ( kk, point(n) );
   counts = std::vector<int> ( kk, 0 );
}

template <typename dist_type>
void kMeansBase<dist_type>::randomize ( void ) {
   std::default_random_engine eng;
   std::uniform_int_distribution<unsigned int> dist ( 0, k - 1 );

   counts = std::vector<int>(k,0);

   for ( unsigned int i = 0; i < dataset.size(); i += 1 ) {
      eng.seed ( i * 1000 );
      unsigned int lab = dist(eng);
      dataset[i].setLabel ( lab );
      counts[lab]++;
   }
}

template <typename dist_type>
void kMeansBase<dist_type>::setTrueLabels ( std::vector<int>::const_iterator a, std::vector<int>::const_iterator b, int offset ) {
   auto cur = a;
   for ( unsigned int i = 0; i < (b - a); ++i ) {
      dataset[i].setTrueLabel ( (*cur) + offset );
      cur++;
   }
}

template<typename dist_type>
double kMeansBase<dist_type>::purity ( void ) const {
   // True labels of the clusters
   std::vector<int> trueLabels ( k, -1 );

   // Counts of the labels assigned to each cluster
   // On the rows ( i.e. counts[i] ) we have the vector of the amounts of points
   // for each label assigned to the cluster i ( that is : counts[i][j] is the
   // number of points with true label j assigned to cluster i)
   std::vector<std::vector<int>> counts ( k, std::vector<int>(k,0) );

   // Iterate through the whole dataset and compute the counts
   for ( unsigned int i = 0; i < dataset.size(); ++i )
      counts[dataset[i].getLabel()][dataset[i].getTrueLabel()] += 1;

   // Compute the true labels
   for ( unsigned int kk = 0; kk < k; ++kk ) {
      int maxIdx = 0, maxCount = counts[kk][0];
      for ( unsigned int j = 1; j < k; ++j ) {
         if ( counts[kk][j] > maxCount ) {
            maxCount = counts[kk][j];
            maxIdx = j;
         }
      }

      trueLabels[kk] = maxIdx;
   }

   // Compute purity
   double result = 0;

   for ( unsigned int i = 0; i < dataset.size(); ++i )
      if ( dataset[i].getTrueLabel() == trueLabels[dataset[i].getLabel()] ) result += 1;

   return result / dataset.size();
}

template<typename dist_type>
void kMeansBase<dist_type>::printOutput ( std::ostream &out ) const {
   out << "dim = " << n << ";\nclusters = " << k << ";\n";
   out << "dataset = [ ";

   unsigned int i = 0;
   for ( ; i < size()-1; ++i )
      out << dataset[i] << ";\n";

   out << dataset[i] << "];";
}

std::istream& operator>> ( std::istream &in, std::vector<int> & out ) {
   int tmp;
   while ( in >> tmp ) out.push_back(tmp);
   return in;
}

#endif
