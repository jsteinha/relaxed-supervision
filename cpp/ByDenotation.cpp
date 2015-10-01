#include "Task.h"
#include <map>
class ByDenotation : public Task {
  private:
    vector<vector<bool>> predicates;
    int U; // size of universe
    int V; // size of vocabulary
    int P; // number of predicates (P <= V)
    double alpha; // fraction of universe for each predicate
    int L; // length of sentence
    double delta, r;
    vector<int> freqs;
    typedef pair<int,int> T; // pair (x,z) where x in [0,V) and y in [0,P]
    typedef int B; // in [0,U)
    int to_int(T t){
      return (P+1) * t.first + t.second;
    }
    int to_int(B b){
      if(tied_beta) return theta_dim;
      else return theta_dim + b;
    }
    inline int f(int x){
      if(x < V-P) return P;
      else return x-(V-P);
      //return min(x, P);
    }
    bool contains(const vector<bool> &pred, const Y &y){
      for(auto &u : y){
        if(!pred[u]) return false;
      }
      return true;
    }
    int sample_once(int x, const vector<int> &u){ //const Y &y){
      double logZ = -INFINITY;
      for(int z = 0; z <= P; z++){
        if(u[z]){
          logZ = lse(logZ, theta[to_int(T(x,z))]);
        }
      }
      double v = rand() / (1.0 + RAND_MAX);
      double cur = -INFINITY;
      for(int z = 0; z <= P; z++){
        if(u[z]){
          cur = lse(cur, theta[to_int(T(x,z))]);
          if(v < exp(cur - logZ)){
            return z;
          }
        }
      }
      cout << "UH OH " << v << " " << cur << " " << logZ << endl;
      assert(false);
    }
    vector<int> diff(const Z &z, const Y &y, const vector<int> &contains_y){
      // z is already constructed so that denote(z) \supset y
      // now, for each predicate p such that y \subset p, 
      //   we also need denote(z) \subset p
      // so, first compute denote(z)
      // then loop through predicates and check containment
      Y denotation;
      for(int u = 0; u < U; u++){
        bool in = true;
        for(int p : z){
          if(!predicates[p][u]){
            in = false;
            break;
          }
        }
        if(in) denotation.insert(u);
      }
      vector<int> ret;
      for(int p = 0; p <= P; p++){
        if(contains_y[p] && !contains(predicates[p], denotation)){
          ret.push_back(p);
        }
      }
      return ret;
    }
    double compute_cost(const Z &z, const example &e){
      double cost = 0.0;
      for(int p : diff(z, e.y, e.u)){
        cost += theta[to_int(p)];
      }
      return cost;
    }
  public:
    ByDenotation(double theta[], int U, int V, int P, double alpha, int L, double delta, double r) :
                 Task(theta), U(U), V(V), P(P), alpha(alpha), L(L), delta(delta), r(r), freqs(V) {
      for(int p = 0; p < P; p++){
        vector<bool> pred;
        for(int u = 0; u < U; u++){
          pred.push_back(rand() < alpha * RAND_MAX);
        }
        predicates.push_back(pred);
      }
      vector<bool> pred;
      for(int u = 0; u < U; u++){
        pred.push_back(true);
      }
      predicates.push_back(pred);
      
      theta_dim = V * (P+1);
      if(fixed_beta){
        beta_dim = 0;
      } else if(tied_beta){
        beta_dim = 1;
      } else {
        beta_dim = P+1;
      }
      dim = theta_dim + beta_dim;
    }
    example make_example(bool Bad){
      example ex;
      vector<int> zs;
      for(int j = 0; j < L; j++){
        int x = power_law(V, r);
        ex.x.push_back(x);
        int z = f(ex.x[j]);
        if(Bad || flip(delta)) z = rand() % (P+1);
        zs.push_back(z);
      }
      bool any = false;
      for(int u = 0; u < U; u++){
        bool in = true;
        for(int j = 0; j < L; j++){
          int p = zs[j];
          if(!predicates[p][u]){
            in = false;
            break;
          }
        }
        if(in){
          ex.y.insert(u);
          any = true;
        }
      }
      if(!any){
        cout << "re-generating... (y = [])" << endl;
        return make_example(Bad); // then try again
      }
      for(int p = 0; p <= P; p++){
        ex.u.push_back(contains(predicates[p], ex.y));
      }
      for(int x : ex.x) freqs[x]++;
      return ex;
    }
    virtual example make_example(){
      return make_example(false);
    }
    virtual example make_example_bad(){
      return make_example(true);
    }
    virtual double init_beta(){
      return 1.0/L;
    }
    virtual void print(){
      cout << "Printing params..." << endl;
      cout << "THETA:" << endl;
      double trace = 0.0, trace2 = 0.0;
      vector<double> diag;
      for(int x = 0; x < V; x++){
        double logZ = -INFINITY;
        for(int z = 0; z <= P; z++) logZ = lse(logZ, theta[to_int(T(x,z))]);
        for(int z = 0; z <= P; z++){
          double prob = exp(theta[to_int(T(x,z))]-logZ);
          printf("%.2f ", prob);
          if(f(x) == z){
            trace += prob;
            if(prob > 0.50) trace2 += 1.0;
            diag.push_back(prob);
          }
        }
        printf("\n");
      }
      cout << "THETA_diag:" << endl;
      for(double diag_x : diag) printf("%.2f ", diag_x);
      printf("\n");
      cout << "BETA:" << endl;
      for(int j = 0; j < beta_dim; j++) printf("%.2f ", theta[to_int(j)]);
      printf("\n");
      cout << "FREQS:" << endl;
      for(int x = 0; x < V; x++) printf("%d ", freqs[x]);
      printf("\n");
      printf("Trace: %.2f\n", trace);
      printf("Trace2: %.2f\n", trace2);
      printf("\n");
    }
    

    virtual Z sample(const example &e, double &logZ, int &num_samples, bool auto_accept){
      num_samples = 0;
      logZ = -INFINITY;
      while(true){
        ++num_samples;
        Z z;
        for(int i = 0; i < L; i++){
          z.push_back(sample_once(e.x[i], e.u));
        }
        if(auto_accept) return z;
        double cost = compute_cost(z, e);
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
      // x to z
      for(int j = 0; j < L; j++){
        int index = to_int(T(e.x[j],z[j]));
        ret.push_back(pair<int,double>(index, 1.0));
      }
      // z to y
      for(int u : diff(z,e.y,e.u)){
        int index = to_int(u);
        ret.push_back(pair<int,double>(index, -1.0));
      }
      return ret;
    }
    virtual void logZ(const X &xs, double &Objective, double gObj[], double wt, double w[]){
      for(int x : xs){
        double logZ = -INFINITY;
        for(int z = 0; z <= P; z++){
          logZ = lse(logZ, w[to_int(T(x,z))]);
        }
        Objective += logZ * wt;
        for(int z = 0; z <= P; z++){
          double th = w[to_int(T(x,z))];
          gObj[to_int(T(x,z))] += exp(th - logZ) * wt;
        }
      }
    }
    virtual double logZu(example e, double params[]){
      double ret = 0.0;
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int z = 0; z <= P; z++){
          if(e.u[z]/*contains(predicates[z], e.y)*/){
            logZ = lse(logZ, params[to_int(T(x,z))]);
          }
        }
        ret += logZ;
      }
      return ret;
    }
    virtual double sumBeta(example e, double w[]){
      double sum = 0.0;
      for(int p = 0; p <= P; p++){
        if(e.u[p]){
          sum += w[to_int(p)];
        }
      }
      return sum;
    }
    virtual void nablaLogZu(example e, double gCon[], double wt, double w[]){
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int z = 0; z <= P; z++){
          if(e.u[z]/*contains(predicates[z], e.y)*/){
            logZ = lse(logZ, w[to_int(T(x,z))]);
          }
        }
        for(int z = 0; z <= P; z++){
          if(e.u[z]/*contains(predicates[z], e.y)*/){
            gCon[to_int(T(x,z))] += exp(w[to_int(T(x,z))]-logZ) * wt;
          }
        }
      }
    }
    virtual void nablaSumBeta(example e, double gCon[], double wt, double w[]){
      for(int p = 0; p <= P; p++){
        if(e.u[p]){
          gCon[to_int(p)] += wt;
        }
      }
    }
    virtual void logZbeta(double &Objective, double gObj[], double w[]){
      for(int j = 0; j < U; j++){
        double beta = w[to_int(j)];
        Objective += log(1 + exp(-beta));
        gObj[to_int(j)] -= 1 / (exp(beta) + 1);
      }
    }
};
