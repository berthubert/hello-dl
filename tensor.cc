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

struct State {
  NNArray<float, 28, 28> img;
  NNArray<float, 28*28, 1> flat;
  NNArray<float, 1, 1> output;
  int label;
  TrackedFloat score;
  TrackedFloat expected;
  TrackedFloat loss;
  void init(NNArray<float, 1, 28*28>& weights,  NNArray<float, 1, 1>& bias)
  {
    img.zero();
    flat = img.flatViewRow();
    output = weights * flat + bias;
    score = doFunc(output(0,0), SigmoidFunc());
    expected = 0;
    loss = (expected - score) * (expected - score);
  }
};


template<typename M, typename W, typename B>
void scoreModel(W& weights, B& bias, const MNISTReader& mntest)
{
  unsigned int corrects=0, wrongs=0;
  int threes=0, sevens=0, threepreds=0, sevenpreds=0;
  M s;
  s.init(weights, bias);
  
  for(unsigned int i = 0 ; i < mntest.num() - 1; ++i){
    int label = mntest.getLabel(i);
    if(label==3)
      threes++;
    else if(label == 7)
      sevens++;
    else continue;

    mntest.pushImage(i, s.img);
    
    int verdict = s.score.getVal() < 0.5 ? 3 : 7;
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
}


int main()
{
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
  cout<<"Have "<<threeseven.size()<<" threes and sevens to train with"<<endl;
  
  NNArray<float, 1, 28*28> weights;
  NNArray<float, 1, 1> bias;
  weights.randomize();
  bias.randomize();


  cout<<"Configuring network.. ";
  cout.flush();

  vector<State> states;  
  for(int n=0; n <128; ++n) {
    State s;
    s.init(weights, bias);
    states.push_back(s);
  }
  TrackedFloat totalLoss=0;
  for(auto& s : states)
    totalLoss = totalLoss + s.loss;

  cout<<"done"<<endl;
  Batcher batcher(threeseven); // badgerbadger
  
  for(int tries=0; tries < 100;++tries) {
    if(!(tries %32))
      scoreModel<State>(weights, bias, mntest);
    
    auto batch = batcher.getBatch(states.size());
    if(batch.empty())
      break;

    for(size_t i = 0; i < batch.size(); ++i) {
      State& s = states[i];
      auto idx = batch[i];
      
      mn.pushImage(idx, s.img);
      s.label = mn.getLabel(idx);

      if(s.label == 7)
        s.expected = 1;
      if(s.label == 3)
        s.expected = 0;
    }
    
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
    cout<<"Percent batch correct: "<<100.0*corrects/(corrects+wrongs)<<" threepreds "<<threepreds<<" sevenpreds " <<sevenpreds<<endl;
    totalLoss.backward();
    auto grad = weights.getGrad();
    double lr=0.07;
    /*
    for(int n=0; n < 100; ++n)
      cout<<weights(0,n).getVal()<<" -= "<< (lr*grad(0,n)) <<"\t";
    cout<<endl;
    */
    
    grad *= lr;

    weights -= grad;

    weights.zeroGrad();
    
    auto biasgrad = bias.getGrad();
    biasgrad *= lr;
    bias -= biasgrad;
    //    cout<<bias(0,0).getVal() << " -= " << lr*biasgrad(0,0) <<endl;

    bias.zeroGrad();
  }
  scoreModel<State>(weights, bias, mntest);
}

