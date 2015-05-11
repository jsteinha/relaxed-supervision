#ifndef GLOBAL_H
#define GLOBAL_H

#include <algorithm>
#include <vector>
#include <set>
using namespace std;

// global constants
static int algorithm;
static double beta_val;
static bool fixed_beta, tied_beta;
static int theta_dim, beta_dim, dim;

// global typedefs
typedef vector<int> X;
typedef multiset<int> Y;
typedef vector<int> Z;
typedef vector<pair<int,double>> LIN;
typedef struct {
  X x;
  Y y;
  vector<int> u;
} example;

#endif
