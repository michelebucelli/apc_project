#ifndef _TIMER_H
#define _TIMER_H

#include <chrono>

class timer {
public:
   using clock = std::chrono::high_resolution_clock;
   using timePoint = std::chrono::time_point<clock>;

private:
   timePoint startTime;
   timePoint stopTime;

public:
   timer () : startTime ( clock::now() ), stopTime ( clock::now() ) { };
   void start ( void ) { startTime = stopTime = clock::now(); };
   void stop ( void ) { stopTime = clock::now(); };

   double getTime ( void ) const {
      auto tspan = std::chrono::duration_cast<std::chrono::nanoseconds> ( stopTime - startTime );
      return tspan.count() / 1000.0;
   }
};

#endif
