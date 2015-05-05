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
#include "snoptProblem.hpp"

using namespace std;

const int W = 42; // vocabulary size
const int L = 24; // sentence length
const int N = 400; // number of examples
const int TR = 30; // number of training iterations
const int S = 10;
//const double eta = 0.06; // step size for learning
const double TAU = 200.0; // number of samples

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

//map<int,double> theta;
vector<double> theta(W*W+W);

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

double compute_cost(const Z &z, const Y &y){
  double cost = 0.0;
  Y yhat = z2y(z);
  set<int> ydiff = diff(y, yhat);
  for(set<int>::iterator yi = ydiff.begin(); yi != ydiff.end(); ++yi){
    cost += theta[to_int(*yi)];
  }
  //cout << "cost: " << cost << endl;
  return cost;
}

int sample_num = 0, sample_denom = 0;
Z sample(const X &x, const Y &y, double *logZ){
  int num_samples = 0;
  *logZ = -INFINITY;
  while(true){
    ++num_samples;
    Z z;
    for(int i = 0; i < x.size(); i++){
      z.push_back(sample_once(x[i], y));
    }
    double cost = compute_cost(z, y);
    *logZ = lse(*logZ, -cost);
    if(rand() < exp(-cost) * RAND_MAX){
      //cout << num_samples << " samples" << endl;
      *logZ = (*logZ) - log(num_samples);
      sample_num += num_samples;
      sample_denom += 1;
      return z;
    }
  }
}

// declare some structures that will be useful for the optimization
typedef vector<pair<int,double>> LIN;
vector<double>   c_vec;
vector<LIN>      A_vec;
vector<set<int>> u_vec;
vector<int>      xtot;

vector<example> examples;

void usrfun ( int *mode,  int *nnObj, int *nnCon,
     int *nnJac, int *nnL,   int *negCon, double w[],
     double *fObj,  double gObj[],
     double fCon[], double gCon[], int *Status,
     char    *cu, int *lencu,
     int    iu[], int *leniu,
     double ru[], int *lenru )
{
  double Objective = 0.0;
  
  // create regularizer
  double lambda = 0.0; //L/(W * sqrt(N));
  for(int i = 0; i < W*W; i++){
    Objective += lambda * w[i] * w[i];
    gObj[i] = 2 * lambda * w[i];
  }
  
  // initialize other values
  vector<double> logC(W*W+W); // log of constraint
  for(int i = W*W; i < W*W + W; i++) gObj[i] = 0;

  double Constraint = 0.0;
  // loop through examples
  for(int n = 0; n < N; n++){
    Objective += c_vec[n] / N;
    double logConstraint = c_vec[n];
    for(auto p : A_vec[n]){
      Objective -= w[p.first] * p.second / N;
      gObj[p.first] -= p.second / N;
      logConstraint -= w[p.first] * p.second;
    }
    for(int x : examples[n].x){
      double logZ = -INFINITY;
      for(int u : u_vec[n]){
        logZ = lse(logZ, w[to_int(T(x,u))]);
      }
      logConstraint += logZ;
    }
    logC[n] = logConstraint;
    Constraint += exp(logConstraint) / N;
  }
  // add additional terms that we accumulated over examples
  for(int i = 0; i < W; i++){
    double beta = w[to_int(i)];
    Objective += log(1 + (L-1) * exp(-beta));
    gObj[to_int(i)] -= (L-1) / (exp(beta) + (L-1));
  }
  for(int x = 0; x < W; x++){
    double logZ = -INFINITY;
    for(int y = 0; y < W; y++){
      logZ = lse(logZ, w[to_int(T(x,y))]);
    }
    Objective += xtot[x] * logZ / N;
    for(int y = 0; y < W; y++){
      double theta = w[to_int(T(x,y))];
      gObj[to_int(T(x,y))] += xtot[x] * exp(theta - logZ) / N;
    }
  }

  //cout << "Objective: " << Objective << endl;
  //cout << "Constraint: " << Constraint << endl;

  //Constraint = 0.0;
  //double MULT = 4.0;
  *fObj   =  Objective; // + MULT * (log(max(TAU, Constraint)) - log(TAU));
  fCon[0] = log(Constraint);
  //cout << "Constraint: " << fCon[0] << " vs " << log(TAU) << endl;

  if ( *mode == 0 || *mode == 2 ) {
    // we already updated fObj
  }

  if ( *mode == 1 || *mode == 2 ) {
    for(int i = 0; i < W*W+W; i++) gCon[i] = 0.0;
    // compute gradient of constraint; only do this if necessary
    for(int n = 0; n < N; n++){
      double wt = exp(logC[n]) / (N * Constraint);
      for(auto p : A_vec[n]){
        gCon[p.first] -= p.second * wt;
      }
      for(int x : examples[n].x){
        double logZ = -INFINITY;
        for(int u : u_vec[n]){
          logZ = lse(logZ, w[to_int(T(x,u))]);
        }
        for(int u : u_vec[n]){
          gCon[to_int(T(x,u))] += exp(w[to_int(T(x,u))]-logZ) * wt;
        }
      }
    }
    
  }

}

void constraint ( int *mode,  int *nnCon, int *nnJac, int *negCon,
         double x[], double fCon[], double gCon[], int *nState,
         char    *cu, int *lencu,
         int    iu[], int *leniu,
         double ru[], int *lenru )
{

  if ( *mode == 0 || *mode == 2 ) {
    // no actual constraints
    fCon[0] =  0;
  }

  if ( *mode == 1 || *mode == 2 ) {
    // no actual constraints
  }

}



int main(){
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

  /* Begin SNOPT initialization */
  cout << "Initializing SNOPT structures..." << endl;
  int n = W*W + W;
  int m = 1;
  int ne = n;
  int nnCon = 1;
  int nnObj = n;
  int nnJac = n;

  int    *indJ = new int[ne];
  int    *locJ = new int[n+1];
  double *valJ = new double[ne];
  
  double *w  = new double[n+m];
  double *bl = new double[n+m];
  double *bu = new double[n+m];
  double *pi = new double[m];
  double *rc = new double[n+m];
  int    *hs = new    int[n+m];
  
  int    iObj    = -1;
  double ObjAdd  = 0;

  for(int i = 0; i <= n; i++){
    locJ[i] = i;
    if(i < n){
      indJ[i] = 0;
      valJ[i] = 0.0;
    }
  }

  int Cold = 0, Basis = 1, Warm = 2;

  for(int i = 0; i < W*W; i++){
    bl[i] = -5; bu[i] = 5;
  }
  for(int i = W*W; i < W*W + W; i++){
    bl[i] = 1.0/L; bu[i] = 5;
  }
  bl[n] = -1.1e20; bu[n] = log(TAU);
  cout << "bu[" << n << "] = " << bu[n] << endl;

  for ( int i = 0; i < n+m; i++ ) {
    hs[i] = 0;
     w[i] = 0;
    rc[i] = 0;
  }

  for ( int i = 0; i < m; i++ ) {
    pi[i] = 0;
  }
  /* End SNOPT initialization */


  cout << "Generating examples..." << endl;
  //vector<example> examples;
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
    sample_num = sample_denom = 0;
    cout << "Beginning iteration " << (t+1) << endl;
    // initialize optimization structures
    c_vec.clear();
    A_vec.clear();
    u_vec.clear();
    xtot.clear();
    xtot = vector<int>(W);

    // for each example
    int ex_num = 0;
    for(example ex : examples){
      if((++ex_num) % 25 == 0) cout << ex_num << " examples" << endl;
      // create c, A, u
      double c_cur = 0.0;
      LIN A_cur;
      set<int> u_cur;
      // update xtot
      for(int x : ex.x) xtot[x]++;
      // set u_cur
      for(Y::iterator yj = ex.y.begin(); yj != ex.y.end(); yj = ex.y.upper_bound(*yj)){
        u_cur.insert(*yj);
      }

      //cout << "Printing example..." << endl;
      //cout << ex.x << endl;
      //cout << ex.y << endl;
      // generate S samples
      //cout << "Generating samples..." << endl;
      vector<Z> zs;
      double logZ = 0.0, logZcur;
      //double logZ = -INFINITY, logZcur;
      for(int s = 0; s < S; s++){
        zs.push_back(sample(ex.x, ex.y, &logZcur));
        logZ += logZcur / S;
        //logZ = lse(logZ, logZcur);
        //cout << "z: " << zs[s] << endl;
      }
      //logZ -= log(S);
      c_cur -= logZ;
      //cout << "Updating gradient..." << endl;
      // for now, just do gradient on log-likelihood
      // theta: +sum of theta values in samples, -sum of average theta
      // beta:  -sum of diffs in samples, + (L-1)exp(-beta)/(1+(L-1)exp(-beta))
      /*
      for(int y : u_cur){
        double& beta = theta[to_int(y)];
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
      */
      for(int s = 0; s < S; s++){
        for(int j = 0; j < L; j++){
          int index = to_int(T(ex.x[j], zs[s][j]));
          double val = 1.0/S;
          A_cur.push_back(pair<int,double>(index, val));
          c_cur += theta[index] * val;
          //theta[to_int(T(ex.x[j], zs[s][j]))] += eta/S;
        }
        set<int> ydiff = diff(ex.y, z2y(zs[s]));
        for(int y : ydiff){
          int index = to_int(y);
          double val = -1.0/S;
          A_cur.push_back(pair<int,double>(index, val));
          c_cur += theta[index] * val;
          //theta[to_int(y)] -= eta/S;
        }
      }
      for(int x : ex.x){
        double logZ = -INFINITY;
        for(int y : u_cur){
          logZ = lse(logZ, theta[to_int(T(x,y))]);
        }
        c_cur -= logZ;
      }
      c_vec.push_back(c_cur);
      A_vec.push_back(A_cur);
      u_vec.push_back(u_cur);
    }

    printf("Average number of samples: %.2f\n", sample_num / (double) sample_denom);

    cout << "Building SNOPT problem..." << endl;
    snoptProblemC prob("the_optimization");

    for(int i=0;i<n;i++) w[i]=theta[i];

    prob.setProblemSize ( m, n, nnCon, nnJac, nnObj );
    prob.setObjective   ( iObj, ObjAdd );
    prob.setJ           ( ne, valJ, indJ, locJ );
    prob.setX           ( bl, bu, w, pi, rc, hs );
    
    //prob.setFunobj      ( objective );
    //prob.setFuncon      ( constraint );
    prob.setUserFun     ( usrfun );
    
    prob.setSpecsFile   ( "prob.spc" );
    prob.setIntParameter( "Verify level", 0 );
    prob.setIntParameter( "Derivative option", 3 );
    
    //prob.solve          ( Cold );
    prob.solve          ( Warm );
    //prob.solve          ( Warm );

    for(int i = 0; i < n; i++) theta[i] = w[i];

    cout << "Printing params..." << endl;
    cout << "THETA:" << endl;
    double trace = 0.0;
    for(int x = 0; x < W; x++){
      double logZ = -INFINITY;
      for(int y = 0; y < W; y++) logZ = lse(logZ, theta[to_int(T(x,y))]);
      for(int y = 0; y < W; y++){
        double prob = exp(theta[to_int(T(x,y))]-logZ);
        printf("%.2f ", prob);
        if(x == y) trace += prob;
      }
      printf("\n");
    }
    cout << "BETA:" << endl;
    for(int y = 0; y < W; y++) printf("%.2f ", theta[to_int(y)]);
    printf("\n");
    printf("Trace: %.2f\n\n", trace);

    //cout << "Cleaning up..." << endl;
    //delete []indJ;  delete []locJ; delete []valJ;

    //delete []w;     delete []bl;   delete []bu;
    //delete []pi;    delete []rc;   delete []hs;

  }

  return 0;
}
