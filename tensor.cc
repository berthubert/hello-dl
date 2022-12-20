#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <Eigen/Dense>
#include <optional>
#include "array.hh"
#include "tracked.hh"
#include "mnistreader.hh"
#include "misc.hh"

using namespace Eigen;
using namespace std;

ofstream g_tree; //("tree.part");

typedef Matrix<float, 28*28,1> img_t;
void pushImage(const img_t& src, NNArray<float, 28, 28>& dest)
{
  for(int row=0 ; row < 28; ++row)
    for(int col=0 ; col < 28; ++col)
      dest(row, col) = src(row+28*col, 0);
}

int main()
{
  //  g_tree<<"digraph G { \n";
  cout<<"Start!"<<endl;

  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  cout<<"Have "<<mn.num()<<" images"<<endl;

  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  vector<int> threeseven;
  for(unsigned int i = 0 ; i < mn.num(); ++i) {
    int label = mn.getLabel(i);
    if(label==3 || label == 7)
      threeseven.push_back(i);
  }
  cout<<"Have "<<threeseven.size()<<" threes and sevens"<<endl;
  
  NNArray<float, 1, 28*28> weights;
  NNArray<float, 1, 1> bias;
  weights.randomize();
  bias.randomize();

  struct State {
    NNArray<float, 28, 28> img;
    NNArray<float, 28*28, 1> flat;
    NNArray<float, 1, 1> output;
    int label;
    TrackedFloat score;
    TrackedFloat expected;
    TrackedFloat loss;

  };

  Batcher batcher(threeseven); // badgerbadger
  

  vector<int> prevbatch;
  for(int tries=0; tries < 200;++tries) {
    auto batch = batcher.getBatch(128);
    if(batch.empty())
      break;
    vector<State> states;
    cout<<"Start batch"<<endl;
    for(auto idx: batch) {
      State s;
      
      const img_t& leImage = mn.getImageEigen(idx);
      s.label = mn.getLabel(idx);
      pushImage(leImage, s.img);
      s.flat = s.img.flatViewRow();
      s.output = weights * s.flat + bias;
      s.score = doFunc(s.output(0,0), SigmoidFunc());
      if(s.label == 7)
        s.expected = 1;
      if(s.label == 3)
        s.expected = 0;
      // squared error
      s.loss = (s.expected - s.score) * (s.expected - s.score);
      states.push_back(s);
    }
    cout<<"Tying"<<endl;
    TrackedFloat totalLoss=0;
    for(auto& s : states)
      totalLoss = totalLoss + s.loss;
    
    cout<<"Average loss: ";
    cout.flush();
    cout<<totalLoss.getVal() / states.size() <<endl;
    int corrects=0, wrongs=0;
    int threepreds=0, sevenpreds=0;
    for(auto& s : states) {
      int predicted = s.score.impl->d_val > 0.5 ? 7 : 3; // get the precalculated number
      if(predicted == s.label)
        corrects++;
      else wrongs++;
      if(predicted == 7)
        sevenpreds++;
      else if(predicted ==3)
        threepreds++;
    }
    cout<<"Percent correct: "<<100.0*corrects/(corrects+wrongs)<<" threepreds "<<threepreds<<" sevenpreds " <<sevenpreds<<endl;
    totalLoss.backward();
    auto grad = weights.getGrad();
    double lr=0.1;
    for(int n=0; n < 100; ++n)
      cout<<weights(0,n).getVal()<<" -= "<< (lr*grad(0,n)) <<"\t";
    cout<<endl;
    
    grad *= lr;

    weights -= grad;

    weights.zeroGrad();
    
    auto biasgrad = bias.getGrad();
    biasgrad *= lr;
    bias -= biasgrad;
    cout<<bias(0,0).getVal() << " -= " << lr*biasgrad(0,0) <<endl;

    bias.zeroGrad();
  }

  

  //  g_tree<<"}"<<endl;
}

