#ifndef TASK_H
#define TASK_H

#include <algorithm>
#include <vector>
#include <set>
#include <atomic>
#include <cmath>
#include <iostream>
#include <cassert>
#include "global.h"
#include "util.h"

using namespace std;
/* 
  Abstraction for tasks
  For each task we should be able to:
    sample
    extract features
    compute logZ(theta) and its gradient
    compute logZ(beta) and its gradient
    optional: if logZ(theta) for conditional inference 
              is different, can override it with logZu
      
 */
class Task {
  protected:
    double* theta;
  public:
    atomic<int> sample_num, sample_denom;
    Task(double theta[]) : theta(theta), sample_num(0), sample_denom(0) {}

    virtual example make_example() = 0;
    virtual double init_beta() = 0;
    virtual void print() = 0;

    virtual Z sample(const example &e, double &logZ) = 0;
    virtual vector<pair<int,double>> extract_features(const X &x, const Z &z, const Y &y) = 0;
    virtual void logZ(const X &x, double& Objective, double gObj[], double wt, double w[]) = 0;
    virtual double logZu(example e, double params[]) = 0;
    virtual void nablaLogZu(example e, double gCon[], double wt, double w[]) = 0;
    virtual void logZbeta(double &Objective, double gObj[], double w[]) = 0;
};

#endif
