class Task {
  public:
    virtual double logZu(example e, vector<int> u_vec) = 0;
    virtual void nablaLogZu(example e, vector<int> u_vec, double[] gCon, double wt) = 0;
    virtual void logZ(int x, double& Objective, double[] gObj, double wt) = 0;
}

class ByDerivation : public Task {
  public:
    double logZu(example e, vector<int> u_vec){
      double ret = 0.0;
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int u : u_vec){
          logZ = lse(logZ, w[to_int(T(x,u))]);
        }
        ret += logZ;
      }
      return ret;
    }
    void nablaLogZu(example e, vector<int> u_vec, double[] gCon, double wt){
      for(int x : e.x){
        double logZ = -INFINITY;
        for(int u : u_vec){
          logZ = lse(logZ, w[to_int(T(x,u))]);
        }
        for(int u : u_vec){
          gCon[to_int(T(x,u))] += exp(w[to_int(T(x,u))]-logZ) * wt;
        }
      }
    }
    void logZ(int x, double& Objective, double[] gObj, double wt){
      double logZ = -INFINITY;
      for(int y = 0; y < W; y++){
        logZ = lse(logZ, w[to_int(T(x,y))]);
      }
      Objective += logZ * wt;
      for(int y = 0; y < W; y++){
        double theta = w[to_int(T(x,y))];
        gObj[to_int(T(x,y))] += exp(theta - logZ) * wt;
      }
    }
}
