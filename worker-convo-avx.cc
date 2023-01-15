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

#include <stdio.h>

using TheModel = CNNAlphabetModel<float>;

using namespace std;

ofstream g_tree; //("tree.part");

std::shared_ptr<HyperParameters> g_hyper;
struct TrainingProgress g_progress;

template<typename W, typename IMGPROJ, typename SCOPROJ, typename EXPPROJ>
void scoreModel(W& w8, const IMGPROJ& imgproj, const SCOPROJ& scoresproj, const EXPPROJ& expproj, const MNISTReader& mntest, int batchno)
{
  unsigned int corrects=0, wrongs=0;
  static ofstream vcsv("validation2.csv");
  std::call_once([&vcsv]() {
    vcsv<<"batchno,corperc,avgloss\n";
  });
  Batcher batcher(mntest.num());

  
  unsigned int batchsize = 256;
  DTime dt;

  NNArray<fvector<8>, 28, 28> img8;
  NNArray<fvector<8>, 26, 1> scores8; // this should be easier to transpose
  NNArray<fvector<8>, 1, 26> expected8;
  cout<<"Start scoring against validation set\n";
  dt.start();
  float totLoss = 0.0;
  for(unsigned int macrobatch = 0 ; macrobatch < batchsize/8; ++macrobatch) {
    auto minibatch = batcher.getBatch(8);
    if(minibatch.size() != 8)
      return;

    fvector<8> label = 0;
    expected8.zero();
    img8.zero();
    for(int n =0 ; n < 8 ; ++n) {
      int idx = minibatch.at(n);
      g_progress.trained++;
      mntest.pushImage(idx, img8, n); 
      label.v[n] = mntest.getLabel(idx) - 1;  // a == 1..
      expected8(0, label.v[n]).impl->d_val.v[n] = 1; // "one hot vector"
    }
    projForward(imgproj, img8, w8);
    projForward(expproj, expected8, w8);
    float loss = w8.getResult().sum(); // runs the whole calculation
    totLoss += loss;
    projBack(scoresproj, w8, scores8);
        
    for(unsigned int n=0 ; n < 8; ++n) {
      int verdict = scores8.getUnparallel(n).maxValueIndexOfColumn(0);
      if(verdict == label.v[n])
        corrects++;
      else
        wrongs++;
    }
  }
  
  double avgLoss = totLoss/batchsize;
  double perc = 100.0*corrects/(corrects+wrongs);
  cout<<perc<<"% correct, average loss "<<avgLoss<<"\n";
  vcsv << batchno <<"," << perc << ", " << avgLoss << endl; 
}

int main(int argc, char** argv)
{
  //  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  auto hp = make_shared<HyperParameters>();
  hp->lr = 0.005;
  hp->batchMult = 8;
  hp->momentum = 0.5;
  g_hyper = hp;

  g_progress.losses.resize(1000);
  g_progress.corrects.resize(1000);
  for(auto& i : g_progress.losses)
    i=NAN;
  for(auto& i : g_progress.corrects)
    i=NAN;
  
  std::thread graphThread(graphicsThread);
  
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

  cout<<"Compiling work"<<endl;
  auto w = mod->loss.getWork<float>(topo);  
  
  //  mod->scores.zero(); 
  cout<<TrackedNumberImp<float>::getCount()<<" instances before clean"<<endl;

  auto imgproj = makeProj(img, w);
  auto scoresproj = makeProj(scores, w);
  auto expproj = makeProj(expected, w);

  s.makeProj(w);
  cout<<s.fc1.d_weights(0,0).getVal()<<endl;
  TheModel::State gather = s;
  cout<<s.fc1.d_weights(0,0).getVal()<<endl;
  cout<<gather.fc1.d_weights(0,0).getVal()<<endl;
  gather.reset(); // uncouple
  cout<<s.fc1.d_weights(0,0).getVal()<<endl;

  
  auto w8 = mod->loss.getWork<fvector<8>>(topo);  
  mod.reset(); // save memory, although actual cleanup only follows later
  expected.reset();
  NNArray<fvector<8>, 28, 28> img8;
  NNArray<fvector<8>, 26, 1> scores8; // this should be easier to transpose
  NNArray<fvector<8>, 1, 26> expected8;
    
  
  cout<<TrackedNumberImp<float>::getCount()<<" instances, model has "<<s.size()<<" parameters, "<<topo.size()<<" nodes to be visited, worker has "<<w8.work.size()<<" entries of size "<<sizeof(w8.work[0])<<", total " <<w8.work.size()*sizeof(w8.work[0])/1000000.0<<"MB"<< endl;
  cout<<w8.dyns.size()<<" variable"<<endl;
  
  unsigned int batchno=0;
  ofstream tcsv("training2.csv");
  tcsv<<"batchno,corperc,avgloss,batchsize,lr,momentum"<<endl;

  s.zeroGrad(); 
  
  for(;;) { // start a new epoch
    Batcher batcher(mn.num()); // badgerbadger

    for(;;) { // start a new batch
      hp = g_hyper; // possibly updated parameters
      
      std::atomic<unsigned int> corrects = 0, wrongs=0;

      gather.zeroGrad(); // we accumulate grads here
      s.projForward(w8);

      if(!(batchno%64))
        scoreModel(w8, imgproj, scoresproj, expproj, mntest, batchno);
      
      DTime dt;
      dt.start();
      float totLoss = 0.0;
      for(unsigned int macrobatch = 0 ; macrobatch < hp->getBatchSize()/8; ++macrobatch) {
        DTime dt;
        dt.start();
        auto minibatch = batcher.getBatch(8);
        if(minibatch.size() != 8)
          goto batcherEmpty;
        
        
        expected8.zero();
        fvector<8> label = 0;
        for(int n =0 ; n < 8 ; ++n) {
          int idx = minibatch.at(n);
          g_progress.trained++;
          mn.pushImage(idx, img8, n); 
          label.v[n] = mn.getLabel(idx) - 1;  // a == 1..
          expected8(0, label.v[n]).impl->d_val.v[n] = 1; // "one hot vector"
        }
        projForward(imgproj, img8, w8);
        projForward(expproj, expected8, w8);
        float loss = w8.getResult().sum(); // runs the whole calculation
        totLoss += loss;
        projBack(scoresproj, w8, scores8);
        
        w8.backward();
        
        for(unsigned int n=0 ; n < 8; ++n) {
          int verdict = scores8.getUnparallel(n).maxValueIndexOfColumn(0);
          
          if(corrects + wrongs < 1) {
            cout<<"Predicted: '"<< (char)('a'+verdict)<<"', actual: '"<< (char)('a' + label.v[n]) <<"': ";
            if(verdict == label.v[n])
              cout<<"We got it right!"<<endl;
            else
              cout<<"More learning to do.."<<endl;
            
            for(unsigned int x = 0 ; x < scores8.getUnparallel(n).getRows(); ++x) {
              cout << (char)('a'+x)  <<": "<< scores8.getUnparallel(n)(x, 0).getVal()<<" ";
            }
            cout<<endl;
            printImg(img8.getUnparallel(n));
          }
          if(verdict == label.v[n])
            corrects++;
          else
            wrongs++;
        }
        gather.projBackGrad(w8);
        w8.zeroGrad();
      }
      // done with a whole batch
      
      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Percent batch correct: "<<perc<<"%, average loss: "<<totLoss/hp->getBatchSize() <<endl;
      g_progress.losses[batchno % g_progress.losses.size()]=(totLoss/hp->getBatchSize());
      g_progress.corrects[batchno % g_progress.losses.size()]= perc;

      for(int wo = 1 ; wo < 5; ++wo) {
        g_progress.losses[(batchno+wo) % g_progress.losses.size()]=NAN;
        g_progress.corrects[(batchno+wo) % g_progress.losses.size()]= NAN;
      }  
      
      tcsv << batchno << "," << perc << ", " << totLoss / hp->getBatchSize() << ","<<hp->getBatchSize()<<","<<hp->lr<<","<<hp->momentum<<endl; 
      
      double lr=hp->lr/hp->getBatchSize(); // mnist.cpp has 0.01, plus "momentum" of 0.5, which we don't have

      s.momentum(gather, hp->momentum);
      s.learn(lr);
      //      s.zeroGrad();
      batchno++;
      g_progress.lastTook = dt.lapUsec()/1000000.0;
      cout<<g_progress.lastTook<<" seconds/batch of " << hp->getBatchSize() <<", lr = "<<hp->lr<<", momentum = "<<hp->momentum<< endl;;
      saveModelState(s, "worker-convo-avx.state");
    }
  batcherEmpty:
    ;
  }

}
