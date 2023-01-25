#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <optional>
#include "tensor2.hh"
#include "mnistreader.hh"
#include "misc.hh"
#include <fenv.h>
#include "tensor-layers.hh"
#include "convo-alphabet.hh"
#include <time.h>

using namespace std;

template<typename M, typename S>
void testModel(SQLiteWriter& sqw, S& s, const MNISTReader& mn, unsigned int startID, int batchno, std::mt19937& rangen)
{
  M m;
  m.init(s, true); // production
  
  Batcher b(mn.num(), rangen);
  auto batch = b.getBatch(128);
  float totalLoss=0;
  int corrects=0, wrongs=0;
  
  auto topo = m.loss.getTopo();
  DTime dt;
  dt.start();
  for(const auto& idx : batch) {
    m.loss.zerograd(topo);
    mn.pushImage(idx, m.img);
    int label = mn.getLabel(idx) - 1;
    m.expected.zero();
    m.expected(0, label) = 1;

    totalLoss += m.modelloss(0,0); // turns it into a float
      
    int predicted = m.scores.maxValueIndexOfColumn(0);

    if(corrects + wrongs == 0) {
      printImgTensor(m.img);
      cout<<"predicted: "<<(char)(predicted+'a')<<", actual: "<<(char)('a'+label)<<", loss: "<<m.modelloss<<"\n";
    }
    
    if(predicted == label)
      corrects++;
    else wrongs++;

  }
  double perc=100.0*corrects/(corrects+wrongs);
  cout<<"Validation batch average loss: "<<totalLoss/batch.size()<<", percentage correct: "<<perc<<", took "<<dt.lapUsec()/1000<<" ms for "<<batch.size()<<" images\n";
  
  sqw.addValue({
      {"startID", startID},
      {"batchno", batchno},
      {"cputime", (double) clock()/CLOCKS_PER_SEC},
      {"corperc", perc},
      {"avgloss", totalLoss/batch.size()}}, "validation");
}

int main(int argc, char **argv)
{
  cout<<"Start!"<<endl;
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mn("gzip/emnist-letters-train-images-idx3-ubyte.gz", "gzip/emnist-letters-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" training images and "<<mntest.num()<<" test images"<<endl;

  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;
  srandom(0);

  if(argc==2) {
    cout<<"Loading model state from file '"<<argv[1]<<"'\n";
    loadModelState(s, argv[1]);
  }
  else
    s.randomize();

  //  std::random_device rd;
  //  std::mt19937 rangen(rd());
  std::mt19937 rangen(0);
  
  m.init(s);

  auto topo = m.loss.getTopo();
  cout<<"Topo.size(): "<<topo.size()<<endl;

  SQLiteWriter sqw("convo-vals.sqlite3");
  int64_t startID=time(0);
  
  int batchno = 0;

  for(;;) {
    Batcher batcher(mn.num(), rangen);

    DTime dt;
    for(unsigned int tries = 0 ;; ++tries) {
      auto batch = batcher.getBatch(64);
      if(batch.empty())
        break;

      if(!(tries % 32)) {
        testModel<ConvoAlphabetModel>(sqw, s, mntest, startID, batchno, rangen);
        saveModelState(s, "tensor-convo.state");
      }
      if(batchno < 32 || !(tries%32)) {
        s.emit(sqw, startID, batchno, batch.size());
      }
      dt.start();
      
      batchno++;

      float totalLoss = 0, totalWeightsLoss=0;
      unsigned int corrects=0, wrongs=0;
      
      m.loss.zeroAccumGrads(topo);

      for(const auto& idx : batch) {     
        mn.pushImage(idx, m.img);
        int label = mn.getLabel(idx) -1;
        m.expected.oneHotColumn(label);
        
        totalLoss += m.modelloss(0,0); // turns it into a float
        totalWeightsLoss += m.weightsloss(0,0);
        int predicted = m.scores.maxValueIndexOfColumn(0);
        
        if(corrects + wrongs == 0) {
          cout<<"predicted: "<<(char)(predicted+'a')<<", actual: "<<(char)(label+'a')<<", loss: "<<m.modelloss<<"\n";
          printImgTensor(m.img);
        }
        
        if(predicted == label)
          corrects++;
        else wrongs++;
        
        // backward the thing
        m.loss.backward(topo);
        m.loss.accumGrads(topo); 
        // clear grads & havevalue
        m.loss.zerograd(topo);
      }
      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Batch "<<batchno<<" average loss " << totalLoss/batch.size()<<", weightsloss " <<totalWeightsLoss/batch.size()<<", percent batch correct "<<perc<<"%, "<<dt.lapUsec()/1000<<"ms/batch"<<endl;

      double lr=0.005 / batch.size();
      double momentum = 0.9;
      s.learn(lr, 0.9);

      // tcsv<<"batchno,cputime,corperc,avgloss,batchsize,lr,momentum"<<endl;
      sqw.addValue({
          {"startID", startID},
          {"batchno", batchno},
          {"cputime", (double)clock()/CLOCKS_PER_SEC},
          {"corperc", perc},
          {"avgloss", totalLoss/batch.size()},
          {"batchsize", (int)batch.size()},
          {"lr", lr*batch.size()},
          {"momentum", momentum}}, "training");
    }
  }
}

