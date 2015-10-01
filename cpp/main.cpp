#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <thread>
#include <vector>
#include <set>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>
#include "prettyprint.hpp"
#include "snoptProblem.hpp"
#include "global.h"
#include "util.h"

#include "Task.h"
#include "ByDenotation.cpp"
#include "ByDerivation.cpp"

using namespace std;

int N = 300; // number of examples
int W = 102; // vocabulary size
int L = 36; // sentence length
const int TR = 50; // number of training iterations
int S = 50; // number of samples
double TAU = 200.0; // number of rejections per samples

const int DEFAULT = 0,
          NO_CONSTRAINT = 1,
          IMPORTANCE = 2;
const int DECIPHERMENT = 0, PREDICATE = 1;
int stage = 1;

const int numThreads = 24;
const int MAX_DIM = 1999999; //W*W+W+5;

double theta[MAX_DIM];

// declare some structures that will be useful for the optimization
double c_vec[MAX_DIM];
LIN A_vec[MAX_DIM];

vector<example> examples;
Task* task;

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
    task->logZ(examples[n].x, Objective, gObj, 1.0/N, w);
    if(algorithm == DEFAULT){ // only compute constraint for DEFAULT alg
      assert(stage == 1 || stage == 2);
      if(stage == 1){
        logConstraint = task->sumBeta(examples[n], w);
      } else {
        logConstraint += task->logZu(examples[n], w);
      }
      logC[n-start] = logConstraint;
      Constraint += exp(logConstraint) / N;
    }
  }

  fObjParts[index] = Objective;
  fConParts[index] = Constraint;

  // compute gradient of constraint
  if(algorithm == DEFAULT){ // only compute constraint for DEFAULT alg
    for(int n = start; n < end; n++){
      double wt = exp(logC[n-start]) / (N * Constraint);
      assert(stage == 1 || stage == 2);
      if(stage == 1){
        task->nablaSumBeta(examples[n], gCon, wt, w);
      } else {
        for(auto p : A_vec[n]){
          gCon[p.first] -= p.second * wt;
        }
        task->nablaLogZu(examples[n], gCon, wt, w);
      }
    }
  }
}

double rho = 0.000;
void usrfun ( int *mode,  int *nnObj, int *nnCon,
     int *nnJac, int *nnL,   int *negCon, double w[],
     double *fObj,  double gObj[],
     double fCon[], double gCon[], int *Status,
     char    *cu, int *lencu,
     int    iu[], int *leniu,
     double ru[], int *lenru )
{
  double Objective = 0.0, Constraint = 0.0;
  
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

  // this can be single-threaded
  task->logZbeta(Objective, gObj, w);

  // L2 reg
  for(int i = 0; i < theta_dim; i++){
    Objective += 0.5 * rho * w[i] * w[i];
    gObj[i] += rho * w[i];
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

      // generate S samples
      vector<Z> zs;
      double logZcur, logZ = -INFINITY;
      int num_samples = 0;
      for(int s = 0; s < S; s++){
        int num_samples_cur = 0;
        zs.push_back(task->sample(ex, logZcur, num_samples_cur, false));
        num_samples += num_samples_cur;
        logZ = lse(logZ, logZcur);
        // Z z0 = task->sample(ex, logZcur, num_samples_cur, true);
        // for(auto &a : task->extract_features(ex, z0)){
        //   if(a.first >= theta_dim){
        //     c_cur -= theta[a.first] * a.second/S;
        //   }
        // }
      }
      c_cur -= logZ - log(S);
      printf("SAMPLES %d %.2f\n", ex_num, num_samples / (double) S);
      for(int s = 0; s < S; s++){
        for(auto &a : task->extract_features(ex, zs[s])){
          A_cur.push_back(pair<int,double>(a.first, a.second/S));
          //if(a.first < theta_dim){
          c_cur += theta[a.first] * a.second/S;
          //}
        }
      }
      c_cur -= task->logZu(ex, theta);

      c_vec[ex_num] = c_cur;
      // sorting (maybe) improves cache efficiency
      sort(A_cur.begin(), A_cur.end());
      A_vec[ex_num] = A_cur;
    }


}

double gObj[MAX_DIM];
double gCon[MAX_DIM];

int main(int argc, char *argv[]){
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");

  // default values
  algorithm = DEFAULT;
  fixed_beta = false;
  tied_beta = false;

  // see if defaults are overridden by args
  int opt;
  int seed = 0;
  int model = 0;
  bool relax = false;
  int U = 300, P = 90, num_bad = 0;
  double delta = 0.0, delta2 = 0.0, r = 0.0;
  double alpha = 0.95;
  while((opt = getopt(argc, argv, "a:b:d:e:f:r:s:m:W:U:P:N:S:T:L:B:tx")) != -1){
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
      case 's':
        sscanf(optarg, "%d", &seed);
        break;
      case 'U':
        sscanf(optarg, "%d", &U);
        break;
      case 'P':
        sscanf(optarg, "%d", &P);
        break;
      case 'N':
        sscanf(optarg, "%d", &N);
        break;
      case 'S':
        sscanf(optarg, "%d", &S);
        break;
      case 'T':
        sscanf(optarg, "%lf", &TAU);
        break;
      case 'L':
        sscanf(optarg, "%d", &L);
        break;
      case 'W':
        sscanf(optarg, "%d", &W);
        break;
      case 'B':
        sscanf(optarg, "%d", &num_bad);
        break;
      case 'd':
        sscanf(optarg, "%lf", &delta);
        break;
      case 'e':
        sscanf(optarg, "%lf", &delta2);
        break;
      case 'f':
        sscanf(optarg, "%lf", &alpha);
        break;
      case 'r':
        sscanf(optarg, "%lf", &r);
        break;
      case 'm':
        sscanf(optarg, "%d", &model);
        break;
      case 'x':
        relax = true;
        cout << "using relaxed constraint" << endl;
        break;
      default:
        cout << "Exiting" << endl;
        exit(0);
    }
  }
  srand(seed);
  printf("OPTION algorithm %d\n", algorithm);
  printf("OPTION fixed_beta %d\n", fixed_beta);
  printf("OPTION tied_beta %d\n", tied_beta);
  printf("OPTION beta_val %lf\n", beta_val);
  printf("OPTION seed %d\n", seed);
  printf("OPTION N %d\n", N);
  printf("OPTION S %d\n", S);
  printf("OPTION TAU %lf\n", TAU);
  printf("OPTION L %d\n", L);
  printf("OPTION W %d\n", W);
  printf("OPTION U %d\n", U);
  printf("OPTION P %d\n", P);
  printf("OPTION num_bad %d\n", num_bad);
  printf("OPTION delta %lf\n", delta);
  printf("OPTION delta2 %lf\n", delta2);
  printf("OPTION alpha %lf\n", alpha);
  printf("OPTION r %lf\n", r);
  printf("OPTION model %d\n", model);
  printf("OPTION rho %lf\n", rho);

  //task = new ByDerivation(theta, W, L);
  //task = new ByDenotationBinary(theta, b, W, L);
  //task = new ByDenotation(theta, 100, 30, 20, 0.9, 10);
  if(model == DECIPHERMENT){
    task = new ByDerivation(theta, W, L, delta, delta2, r);
    printf("OPTION task ByDerivation(%d, %d, %lf, %lf)\n", W, L, delta, delta2);
  } else if(model == PREDICATE){
    task = new ByDenotation(theta, U, W, P, alpha, L, delta, r);
    printf("OPTION task ByDenotation(%d, %d, %lf, %lf)\n", W, L, delta, delta2);
  } else {
    cout << "Invalid model: " << model << endl;
    cout << "Exiting" << endl;
    exit(0);
  }
  double init_beta = task->init_beta();

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
  
  double *bl = new double[dim+m];
  double *bu = new double[dim+m];
  double *pi = new double[m];
  double *rc = new double[dim+m];
  double *w  = new double[dim+m];
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

  int Cold = 0, /*Basis = 1, */Warm = 2;

  for(int i = 0; i < theta_dim; i++){
    bl[i] = -5; bu[i] = 5;
  }
  for(int i = theta_dim; i < dim; i++){
    bl[i] = init_beta; bu[i] = 5;
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
    if(i < num_bad) examples.push_back(task->make_example_bad());
    else examples.push_back(task->make_example());
  }

  cout << "Initializing beta..." << endl;
  if(fixed_beta) theta[theta_dim] = beta_val;
  else if(tied_beta) theta[theta_dim] = init_beta;
  else {
    for(int i = theta_dim; i < dim; i++){
      theta[i] = init_beta;
    }
  }

  for(int t = 0; t < TR; t++){
    if(t >= TR/2) stage = 2;
    task->sample_num = task->sample_denom = 0;
    cout << "Beginning iteration " << (t+1) << " (stage=" << stage << ")" << endl;

    // launch threads for each segment of examples
    vector<thread> threads;
    for(int i = 0; i < numThreads; i++){
      int start = (N * i) / numThreads,
          end = (N * (i+1)) / numThreads;
      threads.push_back(thread(process_examples, start, end));
    }
    for(auto &th : threads) th.join();

    printf("Average number of samples: %.2f\n", task->sample_num / (double) task->sample_denom);

    cout << "Building SNOPT problem..." << endl;
    snoptProblemC prob("the_optimization");

    for(int i=0;i<dim;i++) w[i]=theta[i];

    if(relax && algorithm == DEFAULT){
      cout << "Measuring TAU" << endl;
      int foo = 0;
      int *bar = &foo;
      double baz = 0.0;
      double *bat = &baz;
      double *tauA = new double[1];
      tauA[0] = -55555;
      usrfun(bar,  &nnObj, &nnCon, &nnJac, bar, bar, theta, bat, gObj, tauA, gCon, 
             bar, (char*)bar, bar, bar, bar, bat, bar);
      double log_tau_measured = tauA[0];
      cout << ">>> TAU (measured): " << exp(log_tau_measured) << endl;
      cout << ">>> TAU (actual)  : " << TAU << endl;
      bu[dim] = max(log(TAU), log_tau_measured);
    }

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

    task->print();

  }

  return 0;
}
