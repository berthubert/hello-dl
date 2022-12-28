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
#include <sstream>
#include <fenv.h>
#include "cnn-alphabet.hh"
#include "fvector.hh"
#include <mutex>
#include <atomic>
#include <thread>

using TheModel = CNNAlphabetModel<fvector<8>>;

using namespace std;

ofstream g_tree; //("tree.part");

template<typename M, typename S>
void scoreModel(S& s, const MNISTReader& mntest, int batchno)
{
  unsigned int corrects=0, wrongs=0;
  static ofstream vcsv("validation3.csv");
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
    for(int n = 0; n < 8 ; ++n) {
      auto idx = batch.front();
      batch.pop_front();
      
      mntest.pushImage(idx, mod.img, n);
      mod.label.a[n] = mntest.getLabel(idx) - 1;  // a == 1..
      
      mod.expected(0, mod.label.a[n]).impl->d_val.a[n] = 1; // "one hot vector"
    }
    
    totLoss += mod.loss.getVal();
    
    for(int n = 0; n < 8; ++n) {
      int verdict = mod.scores.getUnparallel(n).maxValueIndexOfColumn(0);
      if(verdict == mod.label.a[n])
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

  struct ModelEnv
  {
    TheModel mod;
    TheModel::State s;
    vector<TrackedNumberImp<fvector<8>>*> topo; // UGH
  };
  vector<ModelEnv> threadmods(4);
  for(auto& tm : threadmods) {
    cout<<"Building thread..\n";
    tm.mod.init(tm.s);
    tm.topo = tm.mod.loss.getTopo();
  }
  
  cout<<TrackedNumberImp<float>::getCount()<<" instances, model has "<<s.size()<<" parameters, "<<topo.size()<<" nodes to be visited\n";

  unsigned int batchno=0;
  ofstream tcsv("training3.csv");
  tcsv<<"batchno,corperc,avgloss"<<endl;

  constexpr int batchsize=64;
  std::mutex lock;  
  
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
      std::atomic<unsigned int> corrects=0, wrongs=0;

      TheModel::State gather;
      gather.zeroGrad();
      
      while(!batch.empty()) {
        string modstate;
        s.save(modstate);
        for(auto& tm : threadmods) 
          tm.s.load(modstate);
        
        auto work = [&batch, &mn, &totLoss, &gather, &corrects, &wrongs, &lock](ModelEnv* me) {
          for(int loops = 0 ; loops < 2; ++loops) {
            me->mod.expected.zero();
            
            for(int n = 0; n < 8 ; ++n) {
              unsigned int idx;
              {
                std::lock_guard<std::mutex> m(lock);
                
                idx = batch.front(); // XXX MUTEX
                batch.pop_front();  // XXX MUTEX
              }
              mn.pushImage(idx, me->mod.img, n);
              me->mod.label.a[n] = mn.getLabel(idx) - 1;  // a == 1..
              
              me->mod.expected(0, me->mod.label.a[n]).impl->d_val.a[n] = 1; // "one hot vector"
            }
            
            auto loss = me->mod.loss.getVal();
            
            me->mod.loss.backward(me->topo);
            {
              std::lock_guard<std::mutex> m(lock);
              totLoss += loss; // XXX MUTEX
              gather.addGrad(me->s); // XXX MUTEX
            }
            
            for(int n = 0; n < 8; ++n) {
              int verdict = me->mod.scores.getUnparallel(n).maxValueIndexOfColumn(0);
              
              if(corrects + wrongs == 0) {
                cout<<"Predicted: '"<< (char)('a'+verdict)<<"', actual: '"<< (char)('a'+me->mod.label.a[n]) <<"': ";
                if(verdict == me->mod.label.a[n])
                  cout<<"We got it right!"<<endl;
                else
                  cout<<"More learning to do.."<<endl;
                cout<<"Loss: "<<me->mod.loss.getVal().a[n]<<", "<<me->mod.scores.getUnparallel(n).flatViewCol()<<endl;
                printImg(me->mod.img.getUnparallel(n));
              }
              
              if(verdict == me->mod.label.a[n])
                corrects++;
              else
                wrongs++;
            }
            
            me->mod.loss.zeroGrad(me->topo);
          }
        };
        vector<std::thread> workers;

        for(auto &tm : threadmods) {
          workers.emplace_back(work, &tm);
        }
        for(auto &w : workers)
          w.join();
          
      }

      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Percent batch correct: "<<perc<<"%, average loss: "<<totLoss.sum()/batchsize <<endl;

      tcsv << batchno <<"," << perc << ", " << totLoss.sum() / batchsize << endl; 
      
      double lr=0.005; // mnist.cpp has 0.01, plus "momentum" of 0.5, which we don't have
      s.setGrad(gather, batchsize);
      s.learn(lr);
      batchno++;
      cout<<dt.lapUsec()/1000000.0<<" seconds/batch of " << batchsize << endl;;
      saveModelState(s, "cnn-lin-threaded.state");
    }
  }
}