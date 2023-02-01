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
#include <mutex>
#include <fenv.h>
#include <atomic>
#include <thread>
#include "cnn-alphabet.hh"
#include "fvector.hh"


double g_lastBatchTook=0.0;
std::atomic<unsigned int> g_trained=0;
float g_lr=0.005;
std::vector<float> g_losses;
constexpr int batchsize=128;


using TheModel = CNNAlphabetModel<float>;

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
  
  fvector<4> totLoss=0.0; 
  
  while(!batch.empty()) {
    mod.expected.zero();
    for(int n = 0; n < 8 ; ++n) {
      auto idx = batch.front();
      batch.pop_front();
      
      mntest.pushImage(idx, mod.img, n);
      mod.label = mntest.getLabel(idx) - 1;  // a == 1..
      
      mod.expected(0, mod.label).impl->d_val = 1; // "one hot vector"
    }
    
    totLoss += mod.loss.getVal();
    
    for(int n = 0; n < 8; ++n) {
      int verdict = mod.scores.maxValueIndexOfColumn(0);
      if(verdict == mod.label.v[n])
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

  //  std::thread graphics(graphicsThread);
  
  string kind = "letters"; // or digits
  MNISTReader mn("gzip/emnist-"+kind+"-train-images-idx3-ubyte.gz", "gzip/emnist-"+kind+"-train-labels-idx1-ubyte.gz");

  MNISTReader mntest("gzip/emnist-"+kind+"-test-images-idx3-ubyte.gz", "gzip/emnist-"+kind+"-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" images for training, "<<mntest.num()<<" for validation"<<endl;
  
  TheModel::State s;

  if(argc > 1) {
    cout<<"Loading model state from "<<argv[1]<<endl;
    loadModelState(s, argv[1]);
  }

  auto mod = make_unique<TheModel>();
  mod->init(s);

  cout<<"Getting topo"<<endl;
  auto topo = mod->loss.getTopo();

  auto expected = mod->expected;
  auto scores = mod->scores;
  auto img = mod->img;
  expected.setVariable();
  mod->scores.setVariable();
  mod->img.setVariable();
  cout<<"scores variable, needsgrad: "<<scores(0,0).impl->d_variable<<" " <<scores(0,0).impl->d_needsgrad << " mode "<< (unsigned int)scores(0,0).impl->d_mode<<endl;
  cout<<"Compiling work"<<endl;
  auto w = mod->loss.getWork<float>(topo);  
  
  //  mod->scores.zero(); 
  cout<<TrackedNumberImp<float>::getCount()<<" instances before clean"<<endl;
  mod.reset(); // save memory, although actual cleanup only follows later
  
  cout<<TrackedNumberImp<float>::getCount()<<" instances, model has "<<s.size()<<" parameters, "<<topo.size()<<" nodes to be visited, worker has "<<w.work.size()<<" entries of size "<<sizeof(w.work[0])<<", total " <<w.work.size()*sizeof(w.work[0])/1000000.0<<"MB"<< endl;
  cout<<w.dyns.size()<<" variable"<<endl;
  
  unsigned int batchno=0;
  ofstream tcsv("training2.csv");
  tcsv<<"batchno,corperc,avgloss"<<endl;

  vector<decltype(w)> works(3);
  for(auto& nw : works) {
    cout<<"Copying\n";
    nw = w;
  }
  
  for(;;) {
    Batcher batcher(mn.num()); // badgerbadger

    for(int tries=0;;++tries) {
      //      if(!((tries+0) %32))
      //scoreModel<TheModel, TheModel::State>(s, mntest, batchno);
      DTime dt;
      dt.start();
      auto batch = batcher.getBatch(batchsize);
      if(batch.size() != batchsize)
        break;

      float totLoss = 0.0;
      std::atomic<unsigned int> corrects = 0, wrongs=0;

      s.zeroGrad(); // we accumulate grads here
      std::mutex lock;

      auto threadWorker = [&mn, &corrects, &wrongs, &batch, &lock, &totLoss, &img, &expected, &scores](decltype(w)* wptr) {
        auto& w = *wptr;
        for(;;) {
          unsigned int idx;
          int label;
          {
            DTime dt;
            dt.start();
            std::lock_guard<std::mutex> m(lock);
            //            cout<<"Spent "<<dt.lapUsec()<<" waiting for lock\n";
            dt.start();
            if(batch.empty())
              break;
            g_trained++;
            idx = batch.front();
            batch.pop_front();
            mn.pushImage(idx, img); // XX MUTEX
            label = mn.getLabel(idx) - 1;  // a == 1..
            expected.zero(); // XXX MUTEX
            expected(0, label).impl->d_val = 1; // "one hot vector"
            w.syncVariable(); // XXX MUTEX
            //            cout<<"Spent "<<dt.lapUsec()<<" usec locked"<<endl;
          }
        
          float loss = w.getResult(); // runs the whole calculation
          
          w.backward();
          
          decltype(scores.flatViewCol().getS()) ourscores;
          int verdict;
          {
            std::lock_guard<std::mutex> m(lock);
            totLoss += loss;
            w.syncBack(); // so the scores match again
            verdict = scores.maxValueIndexOfColumn(0);
            ourscores = scores.flatViewCol().getS();
          }
          
          if(corrects + wrongs < 1) {
            cout<<"Predicted: '"<< (char)('a'+verdict)<<"', actual: '"<< (char)('a' + label) <<"': ";
            if(verdict == label)
              cout<<"We got it right!"<<endl;
            else
              cout<<"More learning to do.."<<endl;
            cout<<"Loss: "<< loss <<", ";
            //// cols, see 'flatviewcol.getS above ////
            for(unsigned int x = 0 ; x < ourscores.getCols(); ++x) {
              cout << (char)('a'+x)  <<": "<<ourscores(0, x)<<" ";
            }
            cout<<endl;
            printImg(img);
          }
          if(verdict == label)
            corrects++;
          else
            wrongs++;
          
          {
            std::lock_guard<std::mutex> m(lock);
            w.syncAddGrad();  // gather grads
          }
          w.zeroGrad(); // 16 milliseconds
        }
      };
      threadWorker(&w);
      //      std::thread t1(threadWorker, &w);
      //      std::thread t2(threadWorker, &works[0]);
      //      std::thread t3(threadWorker, &works[1]);
      //std::thread t4(threadWorker, &works[2]);
      //      t1.join(); t2.join(); //t3.join(); t4.join();

      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Percent batch correct: "<<perc<<"%, average loss: "<<totLoss/batchsize <<endl;
      g_losses.reserve(1024);
      g_losses.push_back(totLoss/batchsize);

      tcsv << batchno <<"," << perc << ", " << totLoss / batchsize << endl; 
      
      double lr=g_lr/batchsize; // mnist.cpp has 0.01, plus "momentum" of 0.5, which we don't have
      s.learn(lr);
      s.zeroGrad();
      batchno++;
      g_lastBatchTook = dt.lapUsec()/1000000.0;
      cout<<g_lastBatchTook<<" seconds/batch of " << batchsize << endl;;
      saveModelState(s, "worker-convo.state");
    }
  }
}
