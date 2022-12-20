#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "mnistreader.hh"
#include <map>
#include <fstream>
#include <thread>
#include "misc.hh"

using namespace std;
using namespace Eigen;

static float sigmoid(const float z)
{
  return 1.0 / (1.0 + exp(-z));
} 

static float gelu(const float x)
{
  constexpr float invsqrt2 = .70710678118654752440; // 1/sqrt(2)
  return 0.5*x*(1+erff(x*invsqrt2));  
}

template<int INPUTS, int OUTPUTS>
struct TLayer
{
  Matrix<float, OUTPUTS, INPUTS> weights, weightsGrad;
  Matrix<float, 1, OUTPUTS>  bias, biasGrad;
  std::function<float(float)> action{[](float x){return x;}};

  void randomize()
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
 
    std::normal_distribution<> d{0, 1};

    auto nd=[&gen, &d](int) {
      return (float)d(gen);
    };
    weights = Matrix<float, OUTPUTS,INPUTS>::Zero().unaryExpr(nd);
    bias = Matrix<float, 1, OUTPUTS>::Zero().unaryExpr(nd);
  }
  
  Matrix<float, 1, OUTPUTS> process(const Matrix<float, INPUTS, 1>& input) const
  {
    Matrix<float, 1, OUTPUTS> ret = weights * input;
    ret += bias;
    ret = ret.unaryExpr(action);
    return ret;
  }

  void doGrad(std::function<float()> lossfunc)
  {
    double h=0.001;
    double origloss = lossfunc();
    for(int r=0; r < weights.rows(); ++r) {
      for(int c=0; c < weights.cols(); ++c) {
        float oldval = weights(r,c);
        weights(r,c) += h; // tweak a bit
        double newloss = lossfunc();
        weights(r,c) = oldval;
        weightsGrad(r,c) = (newloss-origloss)/h;
      }
    }
    biasGrad.setZero();
    for(int r=0; r < bias.rows(); ++r) {
      for(int c=0; c < bias.cols(); ++c) {
        float oldval = bias(r,c);
        bias(r,c) += h; // tweak a bit
        double newloss = lossfunc();
        bias(r,c) = oldval;
        biasGrad(r,c) = (newloss-origloss)/h;
      }
    }
  }

  void applyGrad(float lr)
  {
    weights -= lr * weightsGrad;
    bias -= lr * biasGrad;
  }
};


typedef Matrix<float, 28*28,1> img_t;

void printImg(const img_t& img)
{
  cout<<"Image: "<<img.mean()<<"\n";
  for(int y=0; y < 28; ++y) {
    for(int x=0; x < 28; ++x) {
      float val = img(x*28+y);
      if(val > 0.5)
        cout<<'X';
      else if(val > 0.25)
        cout<<'*';
      else if(val > 0.125)
        cout<<'.';
      else
        cout<<' ';
    }
    cout<<'\n';
  }
  cout << img(28*28-1);
  cout<<"\n";
}

template<typename T>
void scoreModel(const T& l, const MNISTReader& mntest)
{
  unsigned int corrects=0, wrongs=0;
  int threes=0, sevens=0, threepreds=0, sevenpreds=0;
  for(unsigned int i = 0 ; i < mntest.num() - 1; ++i){
    int label = mntest.getLabel(i);
    if(label==3)
      threes++;
    else if(label == 7)
      sevens++;
    else continue;

    img_t pic = mntest.getImageEigen(i);
    MatrixXf result = l.process(pic); 

    int verdict = result(0) < 0.5 ? 3 : 7;
    //    cout<<"label "<<(int)label<<" result(0) "<<result(0)<<" verdict " <<verdict<<" pic "<<pic.mean()<<endl;
    //cout<<"l1bias "<<l.l1.bias<<" l2bias "<<l.l2.bias<<endl;
    //    cout<<l.l1.weights.cwiseAbs().mean()<<endl;
    if(verdict == label) {
      corrects++;
    }
    else {
      wrongs++;
    }

    if(verdict==3)
      threepreds++;
    else if(verdict == 7)
      sevenpreds++;
  }
  double perc = corrects*100.0/(corrects+wrongs);
  cout<<perc<<"% correct, threes "<<threes<<" sevens "<<sevens<<" threepreds "<<threepreds<<" sevenpreds "<<sevenpreds<<"\n";
  static ofstream ofs("./results.csv");
  static int lcount;
  if(!lcount) {
    ofs<<"count,perc,threes,sevens"<<endl;
  }
  ofs<<lcount<<","<< perc <<","<<threepreds<<","<<sevenpreds<<endl;
  lcount++;
  static int perclim;
  if(perc > perclim) {
    ofstream ofs2("model-"+to_string(perclim)+".ppm");
    ofs2<<"P6\n# example from the man page\n280 280\n255\n";
    float maxval = l.weights.maxCoeff(), minval=l.weights.minCoeff();
    string line;
    for(int n=0; n < l.weights.cols(); ++n) {
      if(!(n%28)) {
        for(int rep = 0; rep < 10; ++rep)
          ofs2<<line;
        line.clear();
      }
      float val = l.weights(n);
      unsigned char r{0},g{0},b{0};
      if(val > 0)
        r = 255 * log(1+val)/log(maxval);
      else if(val < 0)
        b = 255 * log(-val +1)/log(-minval);
      for(int rep = 0; rep < 10; ++rep) {
        line.append(1, (char)r);
        line.append(1, (char)g);
        line.append(1, (char)b);
      }

    }
    for(int rep = 0; rep < 10; ++rep)
      ofs2<<line;

    perclim = (int)perc+1;
  }
}

int main()
{
  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  cout<<"Have "<<mn.num()<<" images"<<endl;

  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  cout<<"Start training"<<endl;
  
  srandom(time(0));

  TLayer<28*28, 1> l1;
  l1.randomize();
  l1.action = sigmoid;

  struct TwoLayers
  {
    TLayer<28*28, 30> l1;
    TLayer<30, 1> l2;
    Matrix<float, 1, 1> process(const img_t& in) const
    {
      //return l1.process(in);
      return l2.process(l1.process(in));
    }
    TwoLayers()
    {
      l1.randomize();
      l1.action = gelu;
      l2.randomize();
      l2.action = sigmoid;
    }
    
    void doGrad(std::function<double()> lossfunc) 
    {
      l1.doGrad(lossfunc);
      l2.doGrad(lossfunc);
    }
    
    void applyGrad(float lr)
    {
      l1.applyGrad(lr);
      l2.applyGrad(lr);
    }
  };
  TwoLayers tl;
  
  vector<int> threeseven;
  for(unsigned int i = 0 ; i < mn.num(); ++i) {
    int label = mn.getLabel(i);
    if(label==3 || label == 7)
      threeseven.push_back(i);
  }
  cout<<"Have "<<threeseven.size()<<" threes and sevens"<<endl;

  auto& themodel = l1; // or tl

  for(;;) {
    cout<<"Starting with a freshly shuffled batcher"<<endl;
    Batcher batcher(threeseven);
    
    for(;;) {
      scoreModel(themodel, mntest);
      auto batch = batcher.getBatch(64);
      if(batch.empty())
        break;

      auto lossfunc = [&batch, &mn, &themodel]() {
        double totloss = 0;
        for(auto& idx : batch) {
          int label = mn.getLabel(idx);
          
          const img_t& leImage = mn.getImageEigen(idx);
          
          Matrix<float, 1, 1> result = themodel.process(leImage);
          
          MatrixXf expected(1,1);
          expected << (label == 3 ? 0.0 : 1.0);
          double thisloss = abs((result-expected).mean());
          totloss += thisloss;
        }

        totloss /= batch.size();
        return totloss;
      };
      
      themodel.doGrad(lossfunc);
      constexpr float lr = 1; // was 0.04
      themodel.applyGrad(lr);
    }
  }
}

  
