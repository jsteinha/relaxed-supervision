#include "Task.h"
class ByDenotationBinary : public Task {
  private:
    int b, W0, W;
    int L;
    typedef pair<int,int> T;
    typedef int B;
    int to_int(T t){
      return W0*t.first + t.second;
    }
    int to_int(B b){
      if(tied_beta) return theta_dim;
      else return theta_dim + b;
    }
    inline int f(int x){
      return min(x, W0-1);
    }
    int sample_once(int x, int y){
      double logZ = -INFINITY;
      for(int z = 0; z < W0; z++){
        if((z & y) == y){
          logZ = lse(logZ, theta[to_int(T(x,z))]);
        }
      }
      double u = rand() / (1.0 + RAND_MAX);
      double cur = -INFINITY;
      for(int z = 0; z < W0; z++){
        if((z & y) == y){
          cur = lse(cur, theta[to_int(T(x,z))]);
          if(u < exp(cur - logZ)){
            return z;
          }
        }
      }
      cout << "UH OH " << u << " " << cur << " " << logZ << endl;
      assert(false);
    }
    double compute_cost(const Z &z, int y1){
      double ret = 0.0;
      int y2 = -1;
      for(int j = 0; j < L; j++) y2 = (y2 & z[j]);
      for(int j = 0; j < b; j++){
        if( (1<<j) & (y1 ^ y2) ){
          ret += theta[to_int(j)];
        }
      }
      return ret;
    }
  public:
    ByDenotationBinary(double theta[], int b, int W, int L) : Task(theta), b(b), W(W), L(L) {
      assert(b <= 30 && W >= (1<<b));
      W0 = (1<<b);
      theta_dim = W * W0;
      if(fixed_beta){
        beta_dim = 0;
      } else if(tied_beta){
        beta_dim = 1;
      } else {
        beta_dim = b;
      }
      dim = theta_dim + beta_dim;
    }

    virtual example make_example(){
      example ex;
      int i = rand() % b;
      int y0 = (1<<i), y = -1;
      for(int j = 0; j < L; j++){
        int x;
        do x = rand() % W; while((f(x)&y)==0);
        y = y & f(x);
        ex.x.push_back(x);
      }
      // special case: have 0 occur sometimes
      if(rand() % (b+1) == 0){
        ex.x[rand()%L]=0;
        y = 0;
      }
      ex.y.insert(y);
      return ex;
    }
    virtual double init_beta(){
      return 1.0/L;
    }
    virtual void print(){
      cout << "Printing params..." << endl;
      cout << "THETA:" << endl;
      double trace = 0.0;
      for(int x = 0; x < W; x++){
        double logZ = -INFINITY;
        for(int z = 0; z < W0; z++) logZ = lse(logZ, theta[to_int(T(x,z))]);
        for(int z = 0; z < W0; z++){
          double prob = exp(theta[to_int(T(x,z))]-logZ);
          printf("%.2f ", prob);
          if(f(x) == z) trace += prob;
        }
        printf("\n");
      }
      cout << "BETA:" << endl;
      for(int j = 0; j < b; j++) printf("%.2f ", theta[to_int(j)]);
      printf("\n");
      printf("Trace: %.2f\n\n", trace);
    }


    virtual Z sample(const X &x, const Y &y, double &logZ){
      int num_samples = 0;
      logZ = -INFINITY;
      int y1 = *(y.begin());
      while(true){
        ++num_samples;
        Z z;
        for(int i = 0; i < L; i++){
          z.push_back(sample_once(x[i], y1));
        }
        double cost = compute_cost(z, y1);
        logZ = lse(logZ, -cost);
        if(rand() < exp(-cost) * RAND_MAX){
          logZ -= log(num_samples);
          sample_num += num_samples;
          sample_denom += 1;
          return z;
        }
      }
    }
    virtual vector<pair<int,double>> extract_features(const X &x, const Z &z, const Y &y){
      vector<pair<int,double>> ret;
      int y1 = *(y.begin()), y2 = -1;
      // x to z
      for(int j = 0; j < L; j++){
        int index = to_int(T(x[j],z[j]));
        ret.push_back(pair<int,double>(index, 1.0));
        y2 = (y2 & z[j]);
      }
      // z to y
      for(int j = 0; j < b; j++){
        if( (1<<j) & (y1 ^ y2) ){
          int index = to_int(j);
          ret.push_back(pair<int,double>(index, -1.0));
        }
      }
      return ret;
    }
    virtual void logZ(const X &xs, double &Objective, double gObj[], double wt, double w[]){
      for(int x : xs){
        double logZ = -INFINITY;
        for(int z = 0; z < W0; z++){
          logZ = lse(logZ, w[to_int(T(x,z))]);
        }
        Objective += logZ * wt;
        for(int z = 0; z < W0; z++){
          double th = w[to_int(T(x,z))];
          gObj[to_int(T(x,z))] += exp(th - logZ) * wt;
        }
      }
    }
    virtual double logZu(example e, double params[]){
      double ret = 0.0;
      int y = *(e.y.begin());
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int z = 0; z < W0; z++){
          if((z & y) == y){
            logZ = lse(logZ, params[to_int(T(x,z))]);
          }
        }
        ret += logZ;
      }
      return ret;
    }
    virtual void nablaLogZu(example e, double gCon[], double wt, double w[]){
      int y = *(e.y.begin());
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int z = 0; z < W0; z++){
          if((z & y) == y){
            logZ = lse(logZ, w[to_int(T(x,z))]);
          }
        }
        for(int z = 0; z < W0; z++){
          if((z & y) == y){
            gCon[to_int(T(x,z))] += exp(w[to_int(T(x,z))]-logZ) * wt;
          }
        }
      }
    }
    virtual void logZbeta(double &Objective, double gObj[], double w[]){
      for(int j = 0; j < b; j++){
        double beta = w[to_int(j)];
        Objective += log(1 + exp(-beta));
        gObj[to_int(j)] -= 1 / (exp(beta) + 1);
      }
    }
};
