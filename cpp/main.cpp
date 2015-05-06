#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <thread>
#include <atomic>
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

const int W = 102; // vocabulary size
const int L = 36; // sentence length
const int N = 1200; // number of examples
const int TR = 50; // number of training iterations
const int S = 20; // number of samples
const double TAU = 200.0; // number of rejections per samples

const int DEFAULT = 0,
          NO_CONSTRAINT = 1,
          IMPORTANCE = 2;
int algorithm = DEFAULT;
bool fixed_beta = false;
bool tied_beta = false;
double beta_val;

const int numThreads = 6;
const int MAX_DIM = W*W+W;
int theta_dim, beta_dim, dim; // = W*W+W;

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
  if(tied_beta) return theta_dim;
  else return theta_dim + b;
}

double theta[MAX_DIM];
//vector<double> theta(MAX_DIM);

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
  cout << "UH OH " << u << " " << cur << " " << logZ << endl;
  return *(y.begin()); // just do something
}

set<int> diff(const Y &y1, const Y &y2){
  set<int> ret;
  Y::const_iterator it1 = y1.begin(), end1 = y1.end();
  Y::const_iterator it2 = y2.begin(), end2 = y2.end();
  while(it1 != end1 || it2 != end2){
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
  return cost;
}

atomic<int> sample_num(0), sample_denom(0);
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
      *logZ = (*logZ) - log(num_samples);
      sample_num += num_samples;
      sample_denom += 1;
      return z;
    }
  }
}

// declare some structures that will be useful for the optimization
typedef vector<pair<int,double>> LIN;
//vector<double>      c_vec;
//vector<LIN>         A_vec;
//vector<vector<int>> u_vec;
//vector<int>         xtot;
double c_vec[N];
LIN A_vec[N];
vector<int> u_vec[N];
int xtot[W];

vector<example> examples;

double fObjParts[numThreads];
double gObjParts[numThreads][MAX_DIM];
double fConParts[numThreads];
double gConParts[numThreads][MAX_DIM];

void process_part(int index, int start, int end, double w[]){
  // initialize other values
  vector<double> logC(end-start); // log of constraint
  double* gObj = gObjParts[index];
  double* gCon = gConParts[index];

  for(int i = 0; i < dim; i++) gObj[i] = 0.0;
  for(int i = 0; i < dim; i++) gCon[i] = 0.0;

  double Objective = 0.0, Constraint = 0.0;

  // loop through examples
  for(int n = start; n < end; n++){
    Objective += c_vec[n] / N;
    double logConstraint = c_vec[n];
    for(auto p : A_vec[n]){
      Objective -= w[p.first] * p.second / N;
      gObj[p.first] -= p.second / N;
      logConstraint -= w[p.first] * p.second;
    }
    if(algorithm == DEFAULT){ // only compute constraint for DEFAULT alg
      for(int x : examples[n].x){
        double logZ = -INFINITY;
        for(int u : u_vec[n]){
          logZ = lse(logZ, w[to_int(T(x,u))]);
        }
        logConstraint += logZ;
      }
      logC[n-start] = logConstraint;
      Constraint += exp(logConstraint) / N;
    }
  }

  //*fObj   =  Objective;
  //fCon[0] = log(Constraint);
  fObjParts[index] = Objective;
  fConParts[index] = Constraint;

  // compute gradient of constraint
  if(algorithm == DEFAULT){ // only compute constraint for DEFAULT alg
    for(int n = start; n < end; n++){
      double wt = exp(logC[n-start]) / (N * Constraint);
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

void process_words(int index, int start, int end, double gObj[], double w[]){
  double Objective = 0.0;
  for(int x = start; x < end; x++){
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
  fObjParts[index] = Objective;
}

void usrfun ( int *mode,  int *nnObj, int *nnCon,
     int *nnJac, int *nnL,   int *negCon, double w[],
     double *fObj,  double gObj[],
     double fCon[], double gCon[], int *Status,
     char    *cu, int *lencu,
     int    iu[], int *leniu,
     double ru[], int *lenru )
{
  double Objective = 0.0, Constraint = 0.0;
  
  /*
  // create regularizer
  double lambda = 0.0; //L/(W * sqrt(N));
  for(int i = 0; i < W*W; i++){
    Objective += lambda * w[i] * w[i];
    gObj[i] = 2 * lambda * w[i];
  }
  */
  
  // create blocks for each thread
  // thread i has [blocks[i], blocks[i+1])
  vector<thread> threads;
  for(int i = 0; i < numThreads; i++){
    int start = (N * i) / numThreads,
        end = (N * (i+1)) / numThreads;
    threads.push_back(thread(process_part, i, start, end, w));
  }
  for(auto &thread : threads){
    thread.join();
  }

  for(int i = 0; i < dim; i++){
    gObj[i] = gCon[i] = 0.0;
  }
  for(int t = 0; t < numThreads; t++){
    Objective += fObjParts[t];
    Constraint += fConParts[t];
  }
  for(int t = 0; t < numThreads; t++){
    double ratio = fConParts[t] / Constraint;
    for(int i = 0; i < dim; i++){
      gObj[i] += gObjParts[t][i];
      if(algorithm == DEFAULT) gCon[i] += ratio * gConParts[t][i];
    }
  }

  // add additional terms that we accumulated over examples
  // this should be multi-threaded; fortunately it is threadsafe
  threads.clear();
  for(int i = 0; i < numThreads; i++){
    int start = (W * i) / numThreads,
        end = (W * (i+1)) / numThreads;
    threads.push_back(thread(process_words, i, start, end, gObj, w));
  }
  for(auto &thread : threads) thread.join();
  for(int t = 0; t < numThreads; t++){
    Objective += fObjParts[t];
  }
  // this can be single-threaded
  for(int i = 0; i < W; i++){
    double beta = w[to_int(i)];
    Objective += log(1 + (L-1) * exp(-beta));
    gObj[to_int(i)] -= (L-1) / (exp(beta) + (L-1));
  }


  *fObj = Objective;
  if(algorithm == DEFAULT){
    fCon[0] = log(Constraint);
  } else {
    fCon[0] = 0.0;
  }

}

void process_examples(int start, int end){

    for(int ex_num = start; ex_num < end; ex_num++){
      if(ex_num % 25 == 0) cout << "example " << ex_num << endl;
      example ex = examples[ex_num];
      // create c, A, u
      double c_cur = 0.0;
      LIN A_cur;
      set<int> u_cur;
      // update xtot -- NO LONGER NEEDED
      //for(int x : ex.x) xtot[x]++;
      // set u_cur
      for(Y::iterator yj = ex.y.begin(); yj != ex.y.end(); yj = ex.y.upper_bound(*yj)){
        u_cur.insert(*yj);
      }

      // generate S samples
      vector<Z> zs;
      double logZ = 0.0, logZcur;
      for(int s = 0; s < S; s++){
        zs.push_back(sample(ex.x, ex.y, &logZcur));
        logZ += logZcur / S;
      }
      c_cur -= logZ;
      for(int s = 0; s < S; s++){
        for(int j = 0; j < L; j++){
          int index = to_int(T(ex.x[j], zs[s][j]));
          double val = 1.0/S;
          A_cur.push_back(pair<int,double>(index, val));
          c_cur += theta[index] * val;
        }
        set<int> ydiff = diff(ex.y, z2y(zs[s]));
        for(int y : ydiff){
          int index = to_int(y);
          double val = -1.0/S;
          A_cur.push_back(pair<int,double>(index, val));
          c_cur += theta[index] * val;
        }
      }
      for(int x : ex.x){
        double logZ = -INFINITY;
        for(int y : u_cur){
          logZ = lse(logZ, theta[to_int(T(x,y))]);
        }
        c_cur -= logZ;
      }

      c_vec[ex_num] = c_cur;
      // sorting (maybe) improves cache efficiency
      sort(A_cur.begin(), A_cur.end());
      A_vec[ex_num] = A_cur;
      // convert from set to vector for efficiency
      u_vec[ex_num] = vector<int>(u_cur.begin(), u_cur.end());
    }


}

int main(int argc, char *argv[]){
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

  int opt;
  while((opt = getopt(argc, argv, "a:b:t")) != -1){
    switch(opt){
      case 'a':
        sscanf(optarg, "%d", &algorithm);
        cout << "Using algorithm " << algorithm << endl;
        break;
      case 'b':
        sscanf(optarg, "%lf", &beta_val);
        fixed_beta = true;
        tied_beta = true;
        cout << "Using fixed beta value " << beta_val << endl;
        break;
      case 't':
        tied_beta = true;
        cout << "Tying beta vales" << endl;
        break;
      default:
        cout << "Exiting" << endl;
        exit(0);
    }
  }
  theta_dim = W*W;
  if(fixed_beta){
    beta_dim = 0;
  } else if(tied_beta){
    beta_dim = 1;
  } else {
    beta_dim = W;
  }
  dim = theta_dim + beta_dim;

  /* Begin SNOPT initialization */
  cout << "Initializing SNOPT structures..." << endl;
  //int n = W*W + W; -- replace with dim
  int m = 1;
  int ne = dim;
  int nnCon = 1;
  int nnObj = dim;
  int nnJac = dim;

  int    *indJ = new int[ne];
  int    *locJ = new int[dim+1];
  double *valJ = new double[ne];
  
  double *w  = new double[dim+m];
  double *bl = new double[dim+m];
  double *bu = new double[dim+m];
  double *pi = new double[m];
  double *rc = new double[dim+m];
  int    *hs = new    int[dim+m];
  
  int    iObj    = -1;
  double ObjAdd  = 0;

  for(int i = 0; i <= dim; i++){
    locJ[i] = i;
    if(i < dim){
      indJ[i] = 0;
      valJ[i] = 0.0;
    }
  }

  int Cold = 0, Basis = 1, Warm = 2;

  for(int i = 0; i < theta_dim; i++){
    bl[i] = -5; bu[i] = 5;
  }
  for(int i = theta_dim; i < dim; i++){
    bl[i] = 1.0/L; bu[i] = 5;
  }
  bl[dim] = -1.1e20; bu[dim] = algorithm == DEFAULT ? log(TAU) : 1.1e20;
  cout << "bu[" << dim << "] = " << bu[dim] << endl;

  for ( int i = 0; i < dim+m; i++ ) {
    hs[i] = 0;
     w[i] = 0;
    rc[i] = 0;
  }

  for ( int i = 0; i < m; i++ ) {
    pi[i] = 0;
  }
  /* End SNOPT initialization */


  cout << "Generating examples..." << endl;
  for(int i = 0; i < N; i++){
    examples.push_back(make_example_con());
  }
  cout << examples[0].x << endl;
  cout << examples[0].y << endl;

  cout << "Initializing beta..." << endl;
  if(fixed_beta) theta[theta_dim] = beta_val;
  else if(tied_beta) theta[theta_dim] = 1.0/L;
  else {
    for(int i = 0; i < W; i++){
      theta[to_int(i)] = 1.0/L;
    }
  }

  cout << "Populating xtot..." << endl;
  for(int x = 0; x < W; x++) xtot[x] = 0;
  for(example ex: examples){
    for(int x : ex.x) xtot[x]++;
  }

  for(int t = 0; t < TR; t++){
    sample_num = sample_denom = 0;
    cout << "Beginning iteration " << (t+1) << endl;
    // initialize optimization structures

    // launch threads for each segment of examples
    vector<thread> threads;
    for(int i = 0; i < numThreads; i++){
      int start = (N * i) / numThreads,
          end = (N * (i+1)) / numThreads;
      threads.push_back(thread(process_examples, start, end));
    }
    for(auto &th : threads) th.join();

    printf("Average number of samples: %.2f\n", sample_num / (double) sample_denom);

    cout << "Building SNOPT problem..." << endl;
    snoptProblemC prob("the_optimization");

    for(int i=0;i<dim;i++) w[i]=theta[i];

    prob.setProblemSize ( m, dim, nnCon, nnJac, nnObj );
    prob.setObjective   ( iObj, ObjAdd );
    prob.setJ           ( ne, valJ, indJ, locJ );
    prob.setX           ( bl, bu, w, pi, rc, hs );
    
    prob.setUserFun     ( usrfun );
    
    prob.setSpecsFile   ( "prob.spc" );
    prob.setIntParameter( "Verify level", 0 );
    prob.setIntParameter( "Derivative option", 3 );
    
    if(t == 0) prob.solve( Cold );
    else       prob.solve( Warm );

    for(int i = 0; i < dim; i++) theta[i] = w[i];

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

  }

  return 0;
}
