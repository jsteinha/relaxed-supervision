#include <cmath>
#include <cstdlib>
#include "util.h"

double lse(double a, double b){
  if(a < b) return b + log(1 + exp(a-b));
  else return a + log(1 + exp(b-a));
}

bool flip(double p){
  return rand() < p * (RAND_MAX + 1.0);
}
