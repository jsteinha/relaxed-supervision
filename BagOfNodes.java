import java.io.*;
import java.util.*;

public class BagOfNodes {
  final static int N = 30;
  final static int K = 10;
  final static int S = 1000000;
  static Pair[][] theta = new Pair[N][N];
  static Pair beta = new ClippedPair();
  static {
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
        theta[i][j] = new Pair();
      }
    }
  }
  static double prob(Pair[] ps, int i){
    double logZ = Double.NEGATIVE_INFINITY;
    for(Pair p : ps) logZ = lse(logZ, p.v);
    return Math.exp(ps[i].v - logZ);
  }
  static int[] sample(int[] x){
    int[] z = new int[x.length];
    for(int i = 0; i < x.length; i++){
      z[i] = sample(theta[x[i]]);
    }
    return z;
  }
  static int sample(Pair[] ps){
    double logZ = Double.NEGATIVE_INFINITY;
    for(Pair p : ps) logZ = lse(logZ, p.v);
    double rnd = Math.random(), sum = 0.0;
    for(int i = 0; i < ps.length; i++){
      sum += Math.exp(ps[i].v - logZ);
      if(sum > rnd) return i;
    }
    throw new RuntimeException("invalid sample");
  }
  static double lse(double a, double b){
    if(a<b) return b + Math.log(1 + Math.exp(a-b));
    else if(b<a) return a + Math.log(1 + Math.exp(b-a));
    else return a + Math.log(2);
  }
  static void print(String name, int[] arr){
    System.out.printf("%s:", name);
    for(int i=0;i<arr.length;i++) System.out.printf(" %d", arr[i]);
    System.out.println();
  }

  public static void main(String[] args) throws Exception {
    Example[] train = new Example[1000];
    for(int i = 0; i < train.length; i++) train[i] = new Example(K, N);
    for(int tr = 0; tr < 10; tr++){
      for(Example ex : train){
        double[][] gradientTheta = new double[N][N];
        double gradientBeta = 0.0;
        int s = 0;
        print("   x", ex.x);
        while(s < S){
          int[] z = sample(ex.x);
          if(s == 0){ // negative gradient
            // theta
            print("zSrc", z);
            for(int i = 0; i < K; i++){
              gradientTheta[ex.x[i]][z[i]]-=1.0;
            }
            // beta
            gradientBeta += K * (N-1) * Math.exp(-beta.v) / (1 + (N-1) * Math.exp(-beta.v));
            s++; continue;
          }
          int diff = ex.diff(z);
          double pAccept = Math.exp(-beta.v * diff);
          if(Math.random() < pAccept){ // positive gradient
            // theta
            print("zTar", z);
            for(int i = 0; i < K; i++){
              gradientTheta[ex.x[i]][z[i]]+=1.0;
            }
            // beta
            gradientBeta -= diff;
            System.out.println("diff: " + diff);
            break;
          }
          s++;
        }
        System.out.println("number of samples: " + s);
        if(s == S){
          System.out.println("maximum number of samples exceeded, skipping example...");
          continue;
        }
        for(int i = 0; i < N; i++){
          for(int j = 0; j < N ; j++){
            theta[i][j].update(gradientTheta[i][j]);
          }
        }
        beta.update(gradientBeta);
        System.out.println("beta: " + beta.v);
        System.out.print("probs:");
        for(int i = 0; i < N; i++) System.out.printf(" %.2f", prob(theta[i], i));
        System.out.println();
      }
    }
  }
}

class Example {
  int[] x, y;
  public Example(int K, int N){
    x = new int[K];
    y = new int[K];
    for(int i=0;i<K;i++){
      x[i]=y[i]=(int)(Math.random() * N);
    }
  }
  int diff(int[] z){
    int[] counts = new int[BagOfNodes.N];
    for(int i=0;i<z.length;i++) counts[z[i]]++;
    for(int i=0;i<y.length;i++) counts[y[i]]--;
    int ret = 0;
    for(int i=0;i<counts.length;i++) ret += Math.abs(counts[i]);
    return ret;
  }
}

class Pair {
  final static double eta = 0.3;
  double v, s;
  public Pair(){
    v = 0.0;
    s = 1e-4;
  }
  void update(double x){
    s += x*x;
    v += eta * x / Math.sqrt(s);
  }
}

class ClippedPair extends Pair {
  final static double MIN_VAL = 0.5;
  public ClippedPair(){
    super();
    v = MIN_VAL;
  }
  @Override
  void update(double x){
    s += x*x;
    v += eta * x / Math.sqrt(s);
    v = Math.max(v, MIN_VAL);
  }
}
