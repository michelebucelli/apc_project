#include "kmeans_base.h"
using std::clog; using std::endl;

kMeansBase::kMeansBase ( unsigned int nn, kMeansDataset::const_iterator a, kMeansDataset::const_iterator b ) : n(nn) {
   dataset.assign ( a, b );
}

void kMeansBase::setK ( unsigned int kk ) {
   k = kk;
   centroids = std::vector<point> ( kk, point(n) );
   counts = std::vector<int> ( kk, 0 );
}

void kMeansBase::randomize ( void ) {
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

void kMeansBase::setTrueLabels ( std::vector<int>::const_iterator a, std::vector<int>::const_iterator b, int offset ) {
   auto cur = a;
   for ( unsigned int i = 0; i < (b - a); ++i ) {
      dataset[i].setTrueLabel ( (*cur) + offset );
      cur++;
   }
}

double kMeansBase::purity ( void ) const {
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

std::istream& operator>> ( std::istream &in, std::vector<int> & out ) {
   int tmp;
   while ( in >> tmp ) out.push_back(tmp);
   return in;
}
