#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <optional>
#include "array.hh"
#include "tracked.hh"
#include "mnistreader.hh"
#include "misc.hh"
#include <fenv.h>

using namespace std;

ofstream g_tree;//("tree.part");

struct ReluModel {
  NNArray<float, 28, 28> img;

  int label;
  NNArray<float, 10, 1> scores;
  NNArray<float, 1, 10> expected;

  TrackedFloat loss;

  struct State
  {
    //             out  in
    NNArray<float, 128, 28*28> w1;
    NNArray<float, 128, 1> b1;
    
    NNArray<float, 64, 128> w2;
    NNArray<float, 64, 1> b2;

    NNArray<float, 10, 64> w3;
    NNArray<float, 10, 1> b3;

  };
  
  void init(State& s)
  {
    img.zero();
    auto flat =    img.flatViewRow();

    auto output =  s.w1 * flat + s.b1;

    auto output2 = output.applyFunc(ReluFunc());

    auto output3 = s.w2 * output2 + s.b2;
    auto output4 = output3.applyFunc(ReluFunc());

    auto output5 = s.w3 * output4 + s.b3;

    scores = output5.logSoftMax();
    expected.zero();
    loss = TrackedFloat(0)-(expected*scores)(0,0);
  }
};

template<typename M, typename S>
void scoreModel(S& s, const MNISTReader& mntest)
{
  unsigned int corrects=0, wrongs=0;

  M model;
  model.init(s);

  unsigned int limit = mntest.num() - 1;
  limit=100;
  for(unsigned int i = 0 ; i < limit; ++i){
    cout<<".";
    cout.flush();
    int label = mntest.getLabel(i);

    mntest.pushImage(i, model.img);
    
    int verdict = model.scores.maxValueIndexOfColumn(0);
    //    cout<<"label "<<(int)label<<" result(0) "<<result(0)<<" verdict " <<verdict<<" pic "<<pic.mean()<<endl;
    //cout<<"l1bias "<<l.l1.bias<<" l2bias "<<l.l2.bias<<endl;
    //    cout<<l.l1.weights.cwiseAbs().mean()<<endl;
    if(verdict == label) {
      corrects++;
    }
    else {
      wrongs++;
    }
  }
  cout<<"\n";
  double perc = corrects*100.0/(corrects+wrongs);
  cout<<perc<<"% correct\n";
}


int main()
{
  cout<<"Start!"<<endl;
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" training images and "<<mntest.num()<<" test images"<<endl;
  
  ReluModel::State s;
  s.w1.randomize(sqrt(1/(28.0*28.0)));
  s.b1.randomize(sqrt(1/(28.0*28.0)));
  
  s.w2.randomize(sqrt(1/(128.0)));
  s.b2.randomize(sqrt(1/(128.0)));
                 
  s.w3.randomize(sqrt(1/(64.0)));
  s.b3.randomize(sqrt(1/(64.0)));


  cout<<"Configuring network";
  cout.flush();

  vector<ReluModel> models;  
  for(int n=0; n <256; ++n) {
    ReluModel rm;
    rm.init(s);
    models.push_back(rm);
    cout<<".";
    cout.flush();
  }

  cout<<endl<<"Tying... ";
  TrackedFloat totalLoss=0;
  for(auto& m : models)
    totalLoss = totalLoss + m.loss;
  totalLoss = totalLoss/TrackedFloat(models.size());
  cout<<"done"<<endl;
  cout<<"Getting topology.. ";
  cout.flush();
  auto topo = totalLoss.getTopo();
  cout<<"done"<<endl;
  
  cout<<TrackedNumberImp<float>::getCount()<<" instances"<<endl;
  Batcher batcher(mn.num()); // badgerbadger
 
  for(int tries=0;;++tries) {
    
    if(!(tries %32))
      scoreModel<ReluModel, ReluModel::State>(s, mntest);

    
    auto batch = batcher.getBatch(models.size());
    if(batch.empty())
      break;
    for(size_t i = 0; i < batch.size(); ++i) {
      ReluModel& m = models.at(i);

      auto idx = batch.at(i);

      mn.pushImage(idx, m.img);

      m.label = mn.getLabel(idx);

      m.expected.zero();
      m.expected(0,m.label) = 1;
    }
    
    cout<<"Average loss: ";
    cout.flush();
    cout<<totalLoss.getVal() <<". ";
    int corrects=0, wrongs=0;

    for(auto& m : models) {
      if(corrects + wrongs == 0) {
        cout<<m.expected<<endl;
        cout<<m.scores<<endl;
        cout<<"Loss: "<<m.loss.getVal()<<endl;
      }

      int predicted = m.scores.maxValueIndexOfColumn(0);
      if(predicted == m.label)
        corrects++;
      else wrongs++;
    }
    cout<<"Percent batch correct: "<<100.0*corrects/(corrects+wrongs)<<"%"<<endl;
    totalLoss.backward(topo);
    cout<<"Done backwarding"<<endl;

    double lr=0.2;//.2;

    auto doLearn=[&lr](auto& v){
      auto grad = v.getGrad();
      grad *= lr;
      v -= grad;
    };

    doLearn(s.w1);
    doLearn(s.w2);
    doLearn(s.w3);
    doLearn(s.b1);
    doLearn(s.b2);
    doLearn(s.b3);
    
    cout<<"Resetting grads"<<endl;
    totalLoss.zeroGrad(topo);
  }
  scoreModel<ReluModel>(s, mntest);
}

