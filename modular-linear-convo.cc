#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <optional>
#include "array.hh"
#include "tracked.hh"
#include "mnistreader.hh"
#include "misc.hh"
#include <string.h>

#include <fenv.h>
#include "cnn1.hh"
#include "cnn-alphabet.hh"
#include "fvector.hh"

using TheModel = CNNAlphabetModel<fvector<8>>;

using namespace std;

ofstream g_tree; //("tree.part");

template<typename M, typename S>
void scoreModel(S& s, const MNISTReader& mntest, int batchno)
{
  unsigned int corrects=0, wrongs=0;
  static ofstream vcsv("validation2.csv");
  static bool notfirst;
  if(!notfirst) {
    vcsv<<"batchno,corperc,avgloss\n";
    notfirst=true;
  }
  
  M mod;
  mod.init(s);

  auto topo = mod.loss.getTopo();
  Batcher batcher(mntest.num());
  auto batch = batcher.getBatch(80);
  
  fvector<8> totLoss=0.0; 
  
  while(!batch.empty()) {
    mod.expected.zero();
    fvector<8> label = 0;
    for(int n = 0; n < 8 ; ++n) {
      auto idx = batch.front();
      batch.pop_front();
      
      mntest.pushImage(idx, mod.img, n);
      label.v[n] = mntest.getLabel(idx) - 1;  // a == 1..
      
      mod.expected(0, label.v[n]).impl->d_val.v[n] = 1; // "one hot vector"
    }
    
    totLoss += mod.loss.getVal();
    
    for(int n = 0; n < 8; ++n) {
      int verdict = mod.scores.getUnparallel(n).maxValueIndexOfColumn(0);
      if(verdict == label.v[n])
        corrects++;
      else
        wrongs++;
    }
    mod.loss.zeroGrad(topo);
  }
  double avgLoss = totLoss.sum()/80.0;
  double perc = 100.0*corrects/(corrects+wrongs);
  cout<<perc<<"% correct, average loss "<<avgLoss<<"\n";
  vcsv << batchno <<"," << perc << ", " << avgLoss << endl; 
}

int main(int argc, char** argv)
{
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  string kind = "letters"; // or digits
  MNISTReader mn("gzip/emnist-"+kind+"-train-images-idx3-ubyte.gz", "gzip/emnist-"+kind+"-train-labels-idx1-ubyte.gz");

  MNISTReader mntest("gzip/emnist-"+kind+"-test-images-idx3-ubyte.gz", "gzip/emnist-"+kind+"-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" images for training, "<<mntest.num()<<" for validation"<<endl;
  
  TheModel::State s;

  if(argc > 1) {
    cout<<"Loading model state from "<<argv[1]<<endl;
    loadModelState(s, argv[1]);
  }

  TheModel mod;
  mod.init(s);

  cout<<"Getting topo"<<endl;
  auto topo = mod.loss.getTopo();
  
  cout<<TrackedNumberImp<float>::getCount()<<" instances, model has "<<s.size()<<" parameters, "<<topo.size()<<" nodes to be visited\n";

  unsigned int batchno=0;
  ofstream tcsv("training2.csv");
  tcsv<<"batchno,corperc,avgloss"<<endl;

  constexpr int batchsize=64;

  for(;;) {
    Batcher batcher(mn.num()); // badgerbadger

    for(int tries=0;;++tries) {
      if(!((tries+0) %32))
        scoreModel<TheModel, TheModel::State>(s, mntest, batchno);
      DTime dt;
      dt.start();
      auto batch = batcher.getBatch(batchsize);
      if(batch.size() != batchsize)
        break;

      fvector<8> totLoss=0.0; 
      int corrects=0, wrongs=0;

      TheModel::State gather;
      gather.zeroGrad();
      
      while(!batch.empty()) {
        mod.expected.zero();
        fvector<8> label=0;
        for(int n = 0; n < 8 ; ++n) {
          auto idx = batch.front();
          batch.pop_front();
          
          mn.pushImage(idx, mod.img, n);
          label.v[n] = mn.getLabel(idx) - 1;  // a == 1..
          
          mod.expected(0, label.v[n]).impl->d_val.v[n] = 1; // "one hot vector"
        }

        totLoss += mod.loss.getVal();
        mod.loss.backward(topo);

        gather.addGrad(s);

        for(int n = 0; n < 8; ++n) {
          int verdict = mod.scores.getUnparallel(n).maxValueIndexOfColumn(0);

          if(corrects + wrongs == 0) {
            cout<<"Predicted: '"<< (char)('a'+verdict)<<"', actual: '"<< (char)('a'+label.v[n]) <<"': ";
            if(verdict == label.v[n])
              cout<<"We got it right!"<<endl;
            else
              cout<<"More learning to do.."<<endl;
            cout<<"Loss: "<<mod.loss.getVal().v[n]<<", "<<mod.scores.getUnparallel(n).flatViewCol()<<endl;
            printImg(mod.img.getUnparallel(n));
          }
          
          if(verdict == label.v[n])
            corrects++;
          else
            wrongs++;
        }

        mod.loss.zeroGrad(topo);
      }

      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Percent batch correct: "<<perc<<"%, average loss: "<<totLoss.sum()/batchsize <<endl;

      tcsv << batchno <<"," << perc << ", " << totLoss.sum() / batchsize << endl; 
      
      double lr=0.005; // mnist.cpp has 0.01, plus "momentum" of 0.5, which we don't have
      s.setGrad(gather, batchsize);
      s.learn(lr);
      batchno++;
      cout<<dt.lapUsec()/1000000.0<<" seconds/batch of " << batchsize << endl;;
      saveModelState(s, "cnn-lin.state");
    }
  }
}
