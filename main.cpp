#include "kmeans_base.h"

#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main () {
   kMeansBase<2> b ( cin );
   cout << b.size() << endl;
   return 0;
}
