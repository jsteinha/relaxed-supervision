#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <map>
#include <vector>
#include <set>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>
#include "prettyprint.hpp"

using namespace std;

const int W = 21; // vocabulary size
const int L = 12; // sentence length
const int N = 1; // number of examples
const int TR = 1; // number of training iterations
const int S = 10;

typedef vector<int> X;
typedef multiset<int> Y;
typedef vector<int> Z;
typedef struct {
  X x;
  Y y;
} example;

example make_example(){
  example e;
  for(int i = 0; i < L; i++){
    int c = rand() % W;
    e.x.push_back(c);
    e.y.insert(c);
  }
  return e;
}

typedef pair<int,int> T;
typedef int B;
int to_int(T t){
  return W*t.first + t.second;
}
int to_int(B b){
  return W*W + b;
}

map<int,double> theta;

double lse(double a, double b){
  if(a < b) return b + log(1 + exp(a-b));
  else return a + log(1 + exp(b-a));
}

int sample_once(int xi, const Y &y){
  double logZ = -INFINITY;
  for(Y::iterator yj = y.begin(); yj != y.end(); yj = y.upper_bound(*yj)){
    logZ = lse(logZ, theta[to_int(T(xi, *yj))]);
  }
  double u = rand() / (double) RAND_MAX;
  double cur = -INFINITY;
  for(Y::iterator yj = y.begin(); yj != y.end(); yj = y.upper_bound(*yj)){
    cur = lse(cur, theta[to_int(T(xi, *yj))]);
    if(u < exp(cur - logZ)){
      return (*yj);
    }
  }
  cout << "UH OH" << endl;
  assert(false);
}

double p_accept(const Z &z, const Y &y){
  double cost = 0.0;
  Y yhat;
  for(Z::const_iterator zi = z.begin(); zi != z.end(); ++zi){
    yhat.insert(*zi);
  }
  Y::const_iterator y1 = y.begin(), end1 = y.end();
  Y::const_iterator y2 = yhat.cbegin(), end2 = yhat.cend();
  while(y1 != end1 || y2 != end2){
    //cout << "y1 " << (*y1) << " y2 " << (*y2) << endl;
    if(y1 == end1){
      cost += theta[to_int(*y2)];
      y2 = yhat.upper_bound(*y2);
    } else if(y2 == end2) {
      cost += theta[to_int(*y1)];
      y1 = y.upper_bound(*y1);
    } else {
      if(*y1 > *y2){ 
        cost += theta[to_int(*y2)];
        y2 = yhat.upper_bound(*y2);
      } else if(*y2 > *y1) {
        cost += theta[to_int(*y1)];
        y1 = y.upper_bound(*y1);
      } else { ++y1; ++y2; }
    }
  }
  cout << "cost: " << cost << endl;
  return exp(-cost);
}

Z sample(const X &x, const Y &y){
  int num_samples = 0;
  while(true){
    ++num_samples;
    Z z;
    for(int i = 0; i < x.size(); i++){
      z.push_back(sample_once(x[i], y));
    }
    if(rand() < p_accept(z, y) * RAND_MAX){
      cout << num_samples << " samples" << endl;
      return z;
    }
  }
}

int main(){
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
  cout << "Generating examples..." << endl;
  vector<example> examples;
  for(int i = 0; i < N; i++){
    examples.push_back(make_example());
  }
  cout << examples[0].x << endl;
  cout << examples[0].y << endl;

  cout << "Initializing beta..." << endl;
  for(int i = 0; i < W; i++){
    theta[to_int(i)] = 1.0/L;
  }
  for(int t = 0; t < TR; t++){
    // for each example
    for(int i = 0; i < N; i++){
      // generate S samples
      cout << "Generating samples..." << endl;
      for(int s = 0; s < S; s++){
        Z z = sample(examples[i].x, examples[i].y);
        cout << "z: " << z << endl;
      }
    }
  }

  return 0;
}
