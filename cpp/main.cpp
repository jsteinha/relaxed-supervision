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

const int W = 15; // vocabulary size
const int L = 8; // sentence length
const int N = 50; // number of examples
const int TR = 10; // number of training iterations
const int S = 10;
const double eta = 0.06; // step size for learning

typedef vector<int> X;
typedef multiset<int> Y;
typedef vector<int> Z;
typedef struct {
  X x;
  Y y;
} example;

example make_example_iid(){
  example e;
  for(int i = 0; i < L; i++){
    int c = rand() % W;
    e.x.push_back(c);
    e.y.insert(c);
  }
  return e;
}

example make_example_con(){
  assert(W%3==0);
  assert(L%2==0);
  example e;
  for(int i = 0; i < L; i++){
    int c;
    if(i%2==0){ c = 3 * (rand() % (W/3)); }
    else { c = e.x[i-1] + 1 + rand() % 2; }
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

set<int> diff(const Y &y1, const Y &y2){
  set<int> ret;
  Y::const_iterator it1 = y1.begin(), end1 = y1.end();
  Y::const_iterator it2 = y2.begin(), end2 = y2.end();
  while(it1 != end1 || it2 != end2){
    //cout << "y1 " << (*y1) << " y2 " << (*y2) << endl;
    if(it1 == end1){
      ret.insert(*it2);
      it2 = y2.upper_bound(*it2);
    } else if(it2 == end2) {
      ret.insert(*it1);
      it1 = y1.upper_bound(*it1);
    } else {
      if(*it1 > *it2){ 
        ret.insert(*it2);
        it2 = y2.upper_bound(*it2);
      } else if(*it2 > *it1) {
        ret.insert(*it1);
        it1 = y1.upper_bound(*it1);
      } else { ++it1; ++it2; }
    }
  }
  return ret;
}

Y z2y(const Z &z){
  Y yhat;
  for(Z::const_iterator zi = z.begin(); zi != z.end(); ++zi){
    yhat.insert(*zi);
  }
  return yhat;
}

double p_accept(const Z &z, const Y &y){
  double cost = 0.0;
  Y yhat = z2y(z);
  set<int> ydiff = diff(y, yhat);
  for(set<int>::iterator yi = ydiff.begin(); yi != ydiff.end(); ++yi){
    cost += theta[to_int(*yi)];
  }
  //cout << "cost: " << cost << endl;
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
    examples.push_back(make_example_con());
  }
  cout << examples[0].x << endl;
  cout << examples[0].y << endl;

  cout << "Initializing beta..." << endl;
  for(int i = 0; i < W; i++){
    theta[to_int(i)] = 1.0/L;
  }
  for(int t = 0; t < TR; t++){
    // for each example
    for(example ex : examples){
      cout << "Printing example..." << endl;
      cout << ex.x << endl;
      cout << ex.y << endl;
      // generate S samples
      cout << "Generating samples..." << endl;
      vector<Z> zs;
      for(int s = 0; s < S; s++){
        zs.push_back(sample(ex.x, ex.y));
        cout << "z: " << zs[s] << endl;
      }
      cout << "Updating gradient..." << endl;
      // for now, just do gradient on log-likelihood
      // theta: +sum of theta values in samples, -sum of average theta
      // beta:  -sum of diffs in samples, + (L-1)exp(-beta)/(1+(L-1)exp(-beta))
      for(Y::iterator yj = ex.y.begin(); yj != ex.y.end(); yj = ex.y.upper_bound(*yj)){
        double& beta = theta[to_int(*yj)];
        beta += eta * (L-1)*exp(-beta)/(1+(L-1)*exp(-beta));
      }
      for(int x : ex.x){
        double logZ = -INFINITY;
        for(int y = 0; y < W; y++){
          logZ = lse(logZ, theta[to_int(T(x,y))]);
        }
        for(int y = 0; y < W; y++){
          double &th = theta[to_int(T(x,y))];
          th -= eta * exp(th - logZ);
        }
      }
      for(int s = 0; s < S; s++){
        for(int j = 0; j < L; j++){
          theta[to_int(T(ex.x[j], zs[s][j]))] += eta/S;
        }
        set<int> ydiff = diff(ex.y, z2y(zs[s]));
        for(int y : ydiff){
          theta[to_int(y)] -= eta/S;
        }
      }
    }
    cout << "Printing params..." << endl;
    cout << "THETA:" << endl;
    for(int x = 0; x < W; x++){
      double logZ = -INFINITY;
      for(int y = 0; y < W; y++) logZ = lse(logZ, theta[to_int(T(x,y))]);
      for(int y = 0; y < W; y++) printf("%.2f ", exp(theta[to_int(T(x,y))]-logZ));
      printf("\n");
    }
    cout << "BETA:" << endl;
    for(int y = 0; y < W; y++) printf("%.2f ", theta[to_int(y)]);
    printf("\n");
  }

  return 0;
}
