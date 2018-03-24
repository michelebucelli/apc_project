#include "kmeans_sdg.h"

#include <iostream>
#include <fstream>

using std::cin;
using std::cout;
using std::clog;
using std::endl;

int main ( int argc, char * argv[] ) {
   kMeansSDG solver ( cin );

   solver.setK(2);
   solver.solve();

   cout << solver;

   return 0;
}
