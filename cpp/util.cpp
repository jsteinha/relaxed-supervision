#include <cmath>
#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include "util.h"

double lse(double a, double b){
  if(a < b) return b + log(1 + exp(a-b));
  else return a + log(1 + exp(b-a));
}

bool flip(double p){
  return rand() < p * (RAND_MAX + 1.0);
}

// returns an integer in the range [0, max_val)
// p(k) is proportional to (k+1)^(-r)
int power_law(int max_val, double r){
  assert(r >= 0.0); // if r < 0, results are undefined
  assert(fabs(r-1) >= .05); // avoid numerical instabilities near r = 1
  double s = r - 1;
  double u = rand() / (RAND_MAX + 1.0);
  double v = -1 + pow(u + (1-u) * pow(max_val+1, -s), -1.0/s);
  return (int)v;
}

