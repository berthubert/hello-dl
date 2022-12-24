#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <optional>
#include "array.hh"
#include "tracked.hh"
#include "mnistreader.hh"
#include "misc.hh"

using namespace std;

ofstream g_tree; //("tree.part");

template<typename T>
void printImg(const T& img)
{
  for(unsigned int y=0; y < img.getRows(); ++y) {
    for(unsigned int x=0; x < img.getCols(); ++x) {
      float val = img(y,x).getVal();
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
  cout<<"\n";
}

/*
Gratefully copied from 'mnist.cpp' in the PyTorch example repository

This model takes MNIST 28*28 input and:

  * normalizes to "0.1307, 03081", torch::data::transforms::Normalize<>(0.1307, 0.3081)

  * applies a 5*5 kernel convolution `conv1`, configured to emit 10 layers, 23*23
  * does max_pool2d on these, which takes non-overlapping 2*2 rectangles 
    and emits max value per rectangle. Delivers 12*12 values for each layer (1 pixel is invented)
  * ReLu
  * does another 5x5 convolution `conv2` on the 10 layers, turning them into 20
  * randomly *zeroes* half of the 20 layers `conv2_drop` - no state, Bernoulli 
    STILL MISSING!
  * another max_pool2d
  * ReLu
  * flatten to 320 values
  * linear combination 320x50 (fc1)
  * ReLU
  * zero out half of values randomly during training (STILL MISSING)
  * another linear combination, 50x10 (fc2)
  * log_softmax on the 10 values
  * the 10 outputs are probabilities per digit
  * highest probability is chosen
 */

struct CNNModel {
  NNArray<float, 28, 28> img;

  int label;
  NNArray<float, 10, 1> scores;
  NNArray<float, 1, 10> expected;

  TrackedFloat loss;

  struct State
  {
    std::array<NNArray<float, 5, 5>,10> c1w; // convo
    std::array<NNArray<float, 1, 1>,10> c1b; // convo
    // max2d
    std::array<NNArray<float, 5, 5>,20> c2w; // convo
    std::array<NNArray<float, 1, 1>,20> c2b; // convo
    // randomly zero half of the 20 XXX MISSING
    // max2d
    //            out  in
    NNArray<float, 50, 320> fc1w; // will become 320
    NNArray<float, 50, 1> fc1b;
    // relu
    // flatten
    // zero half of parameters  XXX MISSING
    NNArray<float, 10, 50> fc2w;
    NNArray<float, 10, 1> fc2b;
    // log_softmax

    void randomize()
    {
      for(auto &c : c1w)
        c.randomize(sqrt(1/25.0)); // formula on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
      for(auto &c : c2w)//  v- input chanels
        c.randomize(sqrt(1/(10*25.0)));  // pytorch appears to do uniform distribution

      for(auto &c : c1b)
        c.randomize(sqrt(1/25.0));
      for(auto &c : c2b)
        c.randomize(sqrt(1/(10*25.0)));
      
      fc1w.randomize(sqrt(1/320.0));
      fc1b.randomize(sqrt(1/320.0));
      fc2w.randomize(sqrt(1/50.0));
      fc2b.randomize(sqrt(1/50.0));
    }
  };
  
  void init(State& s)
  {
    img.zero();
    std::array<NNArray<float, 24,24>, 10> step1;

    int ctr=0;

    // this one works - we make 10 output layers based on 1 input layer, using
    // 10 sets of filters (weights)
    
    for(auto& p : step1) {
      p=img.Convo2d<5>(s.c1w[ctr], s.c1b[ctr]);
      ctr++;
    }
    
    std::array<NNArray<float, 12,12>, 10> step2;
    ctr=0;
    for(auto& p : step2)
      p = step1[ctr++].Max2d<2>().applyFunc(ReluFunc());

    // The 20 output layers of the next convo2d have 20 filters
    // these filters need to be applied to all 10 input layers
    // and the output is the addition of the outputs of those filters
    
    ctr = 0;
    std::array<NNArray<float, 8,8>, 20> step3;
    for(auto& p : step3) {
      p.zero();
      for(auto& p2 : step2)
        p = p +  p2.Convo2d<5>(s.c2w[ctr], s.c2b[ctr]);
      ctr++;
    }
    ctr = 0;
    std::array<NNArray<float, 4,4>, 20> step4;
    for(auto& p : step4) {
      p = step3[ctr++].Max2d<2>().applyFunc(ReluFunc());
    }

    NNArray<float, 4*4*20, 1> flat;
    ctr=0;
    for(const auto& p : step4) {
      auto flatpart = p.flatViewRow();
      for(unsigned int i = 0; i < 4*4; ++i) // XX make dynamic somehow
        flat(ctr++, 0) = flatpart(i, 0);
    }
    auto output = s.fc1w * flat + s.fc1b;
    auto output2 = output.applyFunc(ReluFunc());
    auto output3 = s.fc2w * output2 + s.fc2b;

    scores = output3.logSoftMax();
    expected.zero();
    loss = TrackedFloat(0) - (expected*scores)(0,0);
  }
};

template<typename M, typename S>
void scoreModel(S& s, const MNISTReader& mntest, int batchno)
{
  unsigned int corrects=0, wrongs=0;
  static ofstream vcsv("validation.csv");
  static bool notfirst;
  if(!notfirst) {
    vcsv<<"batchno,corperc,avgloss\n";
    notfirst=true;
  }
  
  M model;
  model.init(s);

  auto topo = model.loss.getTopo();
  Batcher batcher(mntest.num());
  auto batch = batcher.getBatch(100);
  double totalLoss=0;
  for(auto i : batch) {
    cout<<".";    cout.flush();
    int label = mntest.getLabel(i);

    mntest.pushImage(i, model.img);
    model.expected.zero();
    model.expected(0,label) = 1; // "one hot vector"
    
    int verdict = model.scores.maxValueIndexOfColumn(0);

    if(verdict == label) {
      corrects++;
    }
    else {
      wrongs++;
    }
    totalLoss += model.loss.getVal();
    model.loss.zeroGrad(topo);
  }
  cout<<"\n";
  double perc = corrects*100.0/(corrects+wrongs);
  double avgLoss = totalLoss/batch.size();
  cout<<perc<<"% correct, average loss "<<avgLoss<<"\n";
  vcsv << batchno <<"," << perc << ", " << avgLoss << endl; 
}

int main()
{
  cout<<"Start!"<<endl;

  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  cout<<"Have "<<mn.num()<<" images"<<endl;

  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  CNNModel::State s;
  s.randomize();

  cout<<"Configuring network";
  cout.flush();

  vector<CNNModel> models;  
  for(int n=0; n < 64; ++n) {
    CNNModel rm;
    rm.init(s);
    models.push_back(rm);
    cout<<".";
    cout.flush();
  }

  cout<<endl<<"Tying... ";
  TrackedFloat totalLoss=0;
  for(auto& m : models)
    totalLoss = totalLoss + m.loss;
  totalLoss= totalLoss/TrackedFloat(models.size());
  cout<<"done"<<endl;
  cout<<"Getting topology.. ";
  cout.flush();
  auto topo = totalLoss.getTopo();
  cout<<"done"<<endl;
  
  cout<<TrackedNumberImp<float>::getCount()<<" instances"<<endl;
  unsigned int batchno=0;
  ofstream tcsv("training.csv");
  tcsv<<"batchno,corperc,avgloss"<<endl;
  for(;;) {
    Batcher batcher(mn.num()); // badgerbadger
    
    for(int tries=0;;++tries) {
      if(!(tries %32))
        scoreModel<CNNModel, CNNModel::State>(s, mntest, batchno);
      
      auto batch = batcher.getBatch(models.size());
      if(batch.size() != models.size())
        break;
      for(size_t i = 0; i < batch.size(); ++i) {
        CNNModel& m = models.at(i);

        auto idx = batch.at(i);
        mn.pushImage(idx, m.img);
        m.label = mn.getLabel(idx);
        
        m.expected.zero();
        m.expected(0,m.label) = 1; // "one hot vector"
      }
    
      cout<<"Average loss: ";
      cout.flush();
      cout<<totalLoss.getVal() <<". ";
      int corrects=0, wrongs=0;
      
      for(auto& m : models) {
        int predicted = m.scores.maxValueIndexOfColumn(0);        
        if(corrects + wrongs == 0) {
          cout<<"Predicted: "<<predicted<<", actual: "<<m.label<<": ";
          if(predicted == m.label)
            cout<<"We got it right!"<<endl;
          else
            cout<<"More learning to do.."<<endl;
          cout<<"Loss: "<<m.loss.getVal()<<", "<<m.scores.flatViewCol()<<endl;
          printImg(m.img);
        }

        if(predicted == m.label)
          corrects++;
        else wrongs++;
      }
      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Percent batch correct: "<<perc<<"%"<<endl;

      tcsv << batchno <<"," << perc << ", " << totalLoss.getVal() << endl; 
      
      totalLoss.backward(topo);
      //      cout<<"Done backwarding"<<endl;
      double lr=0.005; // mnist.cpp has 0.01, plus "momentum" of 0.5, which we don't have
      
      auto doLearn=[&lr](auto& v){
        auto grad = v.getGrad();
        grad *= lr;
        v -= grad;
      };
      
      doLearn(s.fc1w);
      doLearn(s.fc2w);
      doLearn(s.fc1b);
      doLearn(s.fc2b);
      
      for(auto& c : s.c1w)
        doLearn(c);
      for(auto& c : s.c2w)
        doLearn(c);

      for(auto& c : s.c1b)
        doLearn(c);
      for(auto& c : s.c2b)
        doLearn(c);

      totalLoss.zeroGrad(topo);
      batchno++;
    }

  }
  scoreModel<CNNModel>(s, mntest, batchno);
}
