#ifndef _TIMER_H
#define _TIMER_H

#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

class timer {
public:
   using clock = std::chrono::high_resolution_clock;
   using timePoint = std::chrono::time_point<clock>;

private:
   timePoint startTime;
   timePoint stopTime;
   double cumulate;

public:
   timer () : startTime ( clock::now() ), stopTime ( clock::now() ), cumulate( 0 ) { };
   void start ( void ) { startTime = stopTime = clock::now(); };

   void stop ( void ) {
      stopTime = clock::now();
      cumulate += getTime();
   };

   double getTime ( void ) const {
      auto tspan = std::chrono::duration_cast<std::chrono::nanoseconds> ( stopTime - startTime );
      return tspan.count() / 1000000.0;
   }

   double getCumulate ( void ) const { return cumulate; }
};

#endif
