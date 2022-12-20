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

struct ReluModel {
  NNArray<float, 28, 28> img;

  int label;
  TrackedFloat score;
  TrackedFloat expected;
  TrackedFloat loss;

  struct State
  {
    NNArray<float, 30, 28*28> w1;
    NNArray<float, 30, 1> b1;
    
    NNArray<float, 1, 30> w2;
    NNArray<float, 1, 1> b2;
  };
  
  void init(State& s)
  {
    img.zero();
    auto flat = img.flatViewRow();
    auto output = s.w1 * flat + s.b1;
    auto output2 = output.applyFunc(ReluFunc());
    auto output3 = s.w2 * output2 + s.b2;
    score = doFunc(output3(0,0), SigmoidFunc());
    expected = 0;
    loss = (expected - score) * (expected - score);
  }
};


template<typename M, typename S>
void scoreModel(S& s, const MNISTReader& mntest)
{
  unsigned int corrects=0, wrongs=0;
  int threes=0, sevens=0, threepreds=0, sevenpreds=0;
  M model;
  model.init(s);
  
  for(unsigned int i = 0 ; i < mntest.num() - 1; ++i){
    int label = mntest.getLabel(i);
    if(label==3)
      threes++;
    else if(label == 7)
      sevens++;
    else continue;

    mntest.pushImage(i, model.img);
    
    int verdict = model.score.getVal() < 0.5 ? 3 : 7;
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

  ReluModel::State s;
  s.w1.randomize();
  s.b1.randomize();
  
  s.w2.randomize();
  s.b2.randomize();


  cout<<"Configuring network.. ";
  cout.flush();

  vector<ReluModel> models;  
  for(int n=0; n <128; ++n) {
    ReluModel rm;
    rm.init(s);
    models.push_back(rm);
  }
  TrackedFloat totalLoss=0;
  for(auto& m : models)
    totalLoss = totalLoss + m.loss;

  cout<<"done"<<endl;
  Batcher batcher(threeseven); // badgerbadger
  
  for(int tries=0; tries < 100;++tries) {
    if(!(tries %32))
      scoreModel<ReluModel, ReluModel::State>(s, mntest);
    
    auto batch = batcher.getBatch(models.size());
    if(batch.empty())
      break;

    for(size_t i = 0; i < batch.size(); ++i) {
      ReluModel& m = models[i];
      auto idx = batch[i];
      
      mn.pushImage(idx, m.img);
      m.label = mn.getLabel(idx);
      m.expected = (m.label == 3) ? 0 : 1;
    }
    
    cout<<"Average loss: ";
    cout.flush();
    cout<<totalLoss.getVal() / models.size() <<endl;
    int corrects=0, wrongs=0;
    int threepreds=0, sevenpreds=0;
    for(auto& m : models) {
      int predicted = m.score.impl->d_val > 0.5 ? 7 : 3; // get the precalculated number 
      if(predicted == m.label)
        corrects++;
      else wrongs++;
      if(predicted == 7)
        sevenpreds++;
      else if(predicted ==3)
        threepreds++;
    }
    cout<<"Percent batch correct: "<<100.0*corrects/(corrects+wrongs)<<" threepreds "<<threepreds<<" sevenpreds " <<sevenpreds<<endl;
    totalLoss.backward();


    double lr=0.07;    
    {
      auto grad = s.w1.getGrad();
      grad *= lr;
      s.w1 -= grad;
      s.w1.zeroGrad();
      
      auto biasgrad = s.b1.getGrad();
      biasgrad *= lr;
      s.b1 -= biasgrad;
      s.b1.zeroGrad();
    }
    {
      auto grad = s.w2.getGrad();
      grad *= lr;
      s.w2 -= grad;
      s.w2.zeroGrad();
      
      auto biasgrad = s.b2.getGrad();
      biasgrad *= lr;
      s.b2 -= biasgrad;
      s.b2.zeroGrad();
    }

  }
  scoreModel<ReluModel>(s, mntest);
}

