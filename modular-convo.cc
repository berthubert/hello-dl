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

using TheModel = CNNAlphabetModel<float>;

using namespace std;

ofstream g_tree; //("tree.part");

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
    int label = mntest.getLabel(i) - 1;

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

int main(int argc, char** argv)
{
  cout<<"Start!"<<endl;
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  string kind = "letters"; // or digits
  MNISTReader mn("gzip/emnist-"+kind+"-train-images-idx3-ubyte.gz", "gzip/emnist-"+kind+"-train-labels-idx1-ubyte.gz");
  cout<<"Have "<<mn.num()<<" images"<<endl;

  MNISTReader mntest("gzip/emnist-"+kind+"-test-images-idx3-ubyte.gz", "gzip/emnist-"+kind+"-test-labels-idx1-ubyte.gz");

  TheModel::State s;

  if(argc > 1) {
    cout<<"Loading model state from "<<argv[1]<<endl;
    loadModelState(s, argv[1]);
  }
  
  cout<<"Configuring network";
  cout.flush();

  vector<TheModel> models;  
  for(int n=0; n < 22; ++n) {
    TheModel rm;
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
        scoreModel<TheModel, TheModel::State>(s, mntest, batchno);
      
      auto batch = batcher.getBatch(models.size());
      if(batch.size() != models.size())
        break;
      for(size_t i = 0; i < batch.size(); ++i) {
        TheModel& m = models.at(i);

        auto idx = batch.at(i);
        mn.pushImage(idx, m.img);
        m.label = mn.getLabel(idx) -1;
        
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
      
      s.learn(lr);
      totalLoss.zeroGrad(topo);
      batchno++;

      saveModelState(s, "cnn.state");
    }

  }
  scoreModel<TheModel>(s, mntest, batchno);
}
