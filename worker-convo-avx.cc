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
double g_lastBatchTook=0.0;
std::atomic<unsigned int> g_trained=0;
float g_lr=0.005;
std::vector<float> g_losses;
constexpr int batchsize=64;

using TheModel = CNNAlphabetModel<float>;

using namespace std;

ofstream g_tree; //("tree.part");

template<typename W, typename IMGPROJ, typename SCOPROJ, typename EXPPROJ>
void scoreModel(W& w8, const IMGPROJ& imgproj, const SCOPROJ& scoresproj, const EXPPROJ& expproj, const MNISTReader& mntest, int batchno)
{
  unsigned int corrects=0, wrongs=0;
  static ofstream vcsv("validation2.csv");
  static bool notfirst;
  if(!notfirst) {
    vcsv<<"batchno,corperc,avgloss\n";
    notfirst=true;
  }
  Batcher batcher(mntest.num());

  unsigned int batchsize = 64;
  DTime dt;

  NNArray<fvector<8>, 28, 28> img8;
  NNArray<fvector<8>, 26, 1> scores8; // this should be easier to transpose
  NNArray<fvector<8>, 1, 26> expected8;
  
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
      g_trained++;
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
  tcsv<<"batchno,corperc,avgloss"<<endl;
  
  for(;;) { // start a new epoch
    Batcher batcher(mn.num()); // badgerbadger

    for(;;) { // start a new batch
      std::atomic<unsigned int> corrects = 0, wrongs=0;
      s.zeroGrad(); // we accumulate grads here
      s.projForward(w8);

      if(!(batchno%32))
        scoreModel(w8, imgproj, scoresproj, expproj, mntest, batchno);
    

      
      DTime dt;
      dt.start();
      float totLoss = 0.0;
      for(unsigned int macrobatch = 0 ; macrobatch < batchsize/8; ++macrobatch) {
        //      if(!((tries+0) %32))
        //scoreModel<TheModel, TheModel::State>(s, mntest, batchno);
        DTime dt;
        dt.start();
        auto minibatch = batcher.getBatch(8);
        if(minibatch.size() != 8)
          goto batcherEmpty;
        
        
        expected8.zero();
        fvector<8> label = 0;
        for(int n =0 ; n < 8 ; ++n) {
          int idx = minibatch.at(n);
          g_trained++;
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
        s.projBackGrad(w8);
        w8.zeroGrad();
      }
      // done with a whole batch
      
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
      saveModelState(s, "worker-convo-avx.state");
    }
  batcherEmpty:
    ;
  }

}
