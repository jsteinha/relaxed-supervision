#include "Task.h"

class ByDerivation : public Task {
  private:
    int W, L;
    double delta, delta2;
    double r;
    vector<int> freqs;
    typedef pair<int,int> T;
    typedef int B;
    int to_int(T t){
      return W*t.first + t.second;
    }
    int to_int(B b){
      if(tied_beta) return theta_dim;
      else return theta_dim + b;
    }

    example make_example_iid(){
      example e;
      bool prev_bad = false;
      for(int i = 0; i < L; i++){
        int c = power_law(W/3, r);
        e.x.push_back(c);
        bool bad = (prev_bad && flip(delta2)) || flip(delta);
        if(!bad){
          e.y.insert(c);
        } else {
          int c2 = rand() % W;
          e.y.insert(c2);
        }
        prev_bad = bad;
      }
      return e;
    }
    
    example make_example_con(){
      assert(W%3==0);
      assert(L%2==0);
      example e;
      bool prev_bad = false;
      for(int i = 0; i < L; i++){
        int c;
        if(i%2==0){ c = 3 * power_law(W/3, r); }
        else { c = e.x[i-1] + 1 + rand() % 2; }
        e.x.push_back(c);
        bool bad = (prev_bad && flip(delta2)) || flip(delta);
        if(!bad){
          e.y.insert(c);
        } else {
          int c2 = rand() % W;
          e.y.insert(c2);
        }
        prev_bad = bad;
      }
      return e;
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
  public:
    ByDerivation(double theta[], int W, int L, double delta=0.0, double delta2=0.0, double r=0.0)
        : Task(theta), W(W), L(L), delta(delta), delta2(delta2), r(r), freqs(W) {
      theta_dim = W*W;
      if(fixed_beta){
        beta_dim = 0;
      } else if(tied_beta){
        beta_dim = 1;
      } else {
        beta_dim = W;
      }
      dim = theta_dim + beta_dim;
    }

    virtual example make_example(){
      example ex = make_example_con();
      set<int> u_cur;
      for(Y::iterator yj = ex.y.begin(); 
                      yj != ex.y.end(); 
                      yj = ex.y.upper_bound(*yj)){
        u_cur.insert(*yj);
      }
      ex.u = vector<int>(u_cur.begin(), u_cur.end());
      for(int x : ex.x) freqs[x]++;
      return ex;
    }
    virtual double init_beta(){
      return 1.0/L;
    }
    virtual void print(){
      cout << "Printing params..." << endl;
      cout << "THETA:" << endl;
      double trace = 0.0, trace2 = 0.0;
      vector<double> diag;
      for(int x = 0; x < W; x++){
        double logZ = -INFINITY;
        for(int y = 0; y < W; y++) logZ = lse(logZ, theta[to_int(T(x,y))]);
        for(int y = 0; y < W; y++){
          double prob = exp(theta[to_int(T(x,y))]-logZ);
          printf("%.2f ", prob);
          if(x == y){
            trace += prob;
            if(prob > 0.75) trace2 += 1.0;
            diag.push_back(prob);
          }
        }
        printf("\n");
      }
      cout << "THETA_diag:" << endl;
      for(int x = 0; x < W; x++) printf("%.2f ", diag[x]);
      printf("\n");
      cout << "BETA:" << endl;
      for(int y = 0; y < W; y++) printf("%.2f ", theta[to_int(y)]);
      printf("\n");
      cout << "FREQS:" << endl;
      for(int x = 0; x < W; x++) printf("%d ", freqs[x]);
      printf("\n");
      printf("Trace: %.2f\n", trace);
      printf("Trace2: %.2f\n", trace2);
      printf("\n");
    }


    virtual Z sample(const example &e, double &logZ, int &num_samples, bool auto_accept){
      num_samples = 0;
      logZ = -INFINITY;
      while(true){
        Z z;
        for(unsigned int i = 0; i < e.x.size(); i++){
          z.push_back(sample_once(e.x[i], e.y));
        }
        if(auto_accept) return z;
        ++num_samples;
        double cost = compute_cost(z, e.y);
        logZ = lse(logZ, -cost);
        if(rand() < exp(-cost) * RAND_MAX){
          logZ -= log(num_samples);
          sample_num += num_samples;
          sample_denom += 1;
          return z;
        }
      }
    }
    virtual vector<pair<int,double>> extract_features(const example &e, const Z &z){
      vector<pair<int,double>> ret;
      for(int j = 0; j < L; j++){
        int index = to_int(T(e.x[j],z[j]));
        ret.push_back(pair<int,double>(index, 1.0));
      }
      for(int yj : diff(e.y, z2y(z))){
        int index = to_int(yj);
        ret.push_back(pair<int,double>(index, -1.0));
      }
      return ret;
    }
    virtual void logZ(const X &xs, double& Objective, double gObj[], double wt, double w[]){
      for(int x : xs){
        double logZ = -INFINITY;
        for(int y = 0; y < W; y++){
          logZ = lse(logZ, w[to_int(T(x,y))]);
        }
        Objective += logZ * wt;
        for(int y = 0; y < W; y++){
          double th = w[to_int(T(x,y))];
          gObj[to_int(T(x,y))] += exp(th - logZ) * wt;
        }
      }
    }
    virtual double logZu(example e, double params[]){
      double ret = 0.0;
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int u : e.u){
          logZ = lse(logZ, params[to_int(T(x,u))]);
        }
        ret += logZ;
      }
      return ret;
    }
    virtual double sumBeta(example e, double w[]){
      double sum = 0.0;
      for(int u : e.u){
        sum += w[to_int(u)];
      }
      return sum;
    }
    virtual void nablaLogZu(example e, double gCon[], double wt, double w[]){
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int u : e.u){
          logZ = lse(logZ, w[to_int(T(x,u))]);
        }
        for(int u : e.u){
          gCon[to_int(T(x,u))] += exp(w[to_int(T(x,u))]-logZ) * wt;
        }
      }
    }
    virtual void nablaSumBeta(example e, double gCon[], double wt, double w[]){
      for(int u : e.u){
        gCon[to_int(u)] += wt;
      }
    }
    virtual void logZbeta(double &Objective, double gObj[], double w[]){
      for(int i = 0; i < W; i++){
        double beta = w[to_int(i)];
        Objective += log(1 + (L-1) * exp(-beta));
        gObj[to_int(i)] -= (L-1) / (exp(beta) + (L-1));
      }
    }
};
