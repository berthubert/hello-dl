#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <optional>
#include "tensor2.hh"
#include "mnistreader.hh"
#include "misc.hh"
#include "vizi.hh"
#include <fenv.h>
#include "tensor-layers.hh"
#include "convo-alphabet.hh"
#include <time.h>
#include <fcntl.h>              /* Obtain O_* constant definitions */
#include <unistd.h>
#include <thread>

using namespace std;

template<typename M, typename S>
void testModel(SQLiteWriter& sqw, S& s, const MNISTReader& mn, unsigned int startID, int batchno)
{
  M m;
  m.init(s, true); // production
  
  Batcher b(mn.num());
  auto batch = b.getBatch(128);
  float totalLoss=0;
  int corrects=0, wrongs=0;
  
  auto topo = m.loss.getTopo();
  DTime dt;
  dt.start();
  for(const auto& idx : batch) {
    m.loss.zerograd(topo);
    mn.pushImage(idx, m.img);
    // normalize
    m.img.normalize(0.172575, 0.25);
    
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


template<typename M>
struct ParMod
{
  ParMod(int readpipe, int writepipe, MNISTReader& mn, std::atomic<unsigned int>& corrects, std::atomic<unsigned int>& wrongs) : d_readpipe(readpipe), d_writepipe(writepipe), d_mn(mn), d_corrects(corrects), d_wrongs(wrongs)
  {
    d_model.init(d_state);
    d_thread = std::thread(&ParMod<M>::worker, this);
    d_topo = d_model.loss.getTopo();
  }

  ~ParMod()
  {
    d_thread.join();
  }
  void worker()
  {
    size_t idx;
    for(;;) {
      int rc =read(d_readpipe, (void*)&idx, sizeof(idx));
      if(rc == 0)
        break;
      if(rc != sizeof(idx))
        throw std::runtime_error("Partial read or error: rc = "+to_string(rc));

      // ....

      d_mn.pushImage(idx, d_model.img);
        // normalize
      d_model.img.normalize(0.172575, 0.25);
      
      int label = d_mn.getLabel(idx) - 1; // they count from 1 over at NIST!
      d_model.expected.oneHotColumn(label);
      
      d_model.modelloss(0,0); // turns it into a float
      int predicted = d_model.scores.maxValueIndexOfColumn(0);
      if(predicted == label)
        d_corrects++;
      else
        d_wrongs++;
        
      // backward the thing
      d_model.loss.backward(d_topo);
      d_model.loss.accumGrads(d_topo); 
      // clear grads & havevalue
      d_model.loss.zerograd(d_topo);

      rc = write(d_writepipe, (void*)&idx, sizeof(idx));
      if(rc == 0)
        break;
      if(rc != sizeof(idx))
        throw std::runtime_error("Partial write: rc = " + to_string(rc));
    }
  }
  
  M d_model;
  typename M::State d_state;
  int d_readpipe, d_writepipe;
  std::vector<TensorImp<float>*> d_topo;
  std::thread d_thread;
  MNISTReader& d_mn;
  std::atomic<unsigned int>& d_corrects;
  std::atomic<unsigned int>&  d_wrongs;
};

int main(int argc, char **argv)
{
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mn("gzip/emnist-letters-train-images-idx3-ubyte.gz", "gzip/emnist-letters-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" training images and "<<mntest.num()<<" test images"<<endl;

  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;

  if(argc==2) {
    cout<<"Loading model state from file '"<<argv[1]<<"'\n";
    loadModelState(s, argv[1]);
  }
  else
    s.randomize();
  
  m.init(s);

  auto topo = m.loss.getTopo();
  cout<<"Topo.size(): "<<topo.size()<<endl;

  std::atomic<unsigned int> corrects=0, wrongs=0;
  
  int toworker[2], fromworker[2];
  if(pipe(toworker) < 0 || pipe(fromworker) < 0)
    throw std::runtime_error("Creating pipe");
  
  vector<std::unique_ptr<ParMod<ConvoAlphabetModel>>> pms;
  for(unsigned int n=0; n < 4; ++n)  // [0] = read, [1]=write
    pms.push_back(std::make_unique<ParMod<ConvoAlphabetModel>>(toworker[0], fromworker[1], mn, corrects, wrongs)); 

  SQLiteWriter sqw("convo-vals-par.sqlite3");
  int64_t startID=time(0);
  
  int batchno = 0;

  for(;;) {
    Batcher batcher(mn.num());

    DTime dt;
    for(unsigned int tries = 0 ;; ++tries) {
      auto batch = batcher.getBatch(64);
      if(batch.empty())
        break;

      if(!(tries % 32)) {
        testModel<ConvoAlphabetModel>(sqw, s, mntest, startID, batchno);
        saveModelState(s, "tensor-convo-par.state");
      }
      if(batchno < 32 || !(tries%32)) {
        s.emit(sqw, startID, batchno, batch.size());
      }
      dt.start();
      
      batchno++;

      float totalLoss = 0, totalWeightsLoss=0;

      
      m.loss.zeroAccumGrads(topo);

      for(auto& pm : pms) {
        pm->d_model.loss.zeroAccumGrads(pm->d_topo);
        m.loss.copyParams(topo, pm->d_topo);
      }
      corrects = wrongs = 0;
      for(const auto& idx : batch) {
        size_t w = idx;
        write(toworker[1], &w, sizeof(w));
      }
      for(size_t pos = 0 ; pos < batch.size(); ++pos) {
        size_t ret;
        read(fromworker[0], &ret, sizeof(ret));
      }

      for(auto& pm : pms) {
        m.loss.addAccumGrads(pm->d_topo, topo);
      }
      
      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Batch "<<batchno<<" average loss " << totalLoss/batch.size()<<", weightsloss " <<totalWeightsLoss/batch.size()<<", percent batch correct "<<perc<<"%, "<<dt.lapUsec()/1000<<"ms/batch"<<endl;

      double lr=0.010 / batch.size(); // 0.010 works well at the beginning
      double momentum = 0.9;
      s.learn(lr, momentum);

      // tcsv<<"batchno,cputime,corperc,avgloss,batchsize,lr,momentum"<<endl;
      sqw.addValue({
          {"startID", startID}, {"batchno", batchno},   {"cputime", (double)clock()/CLOCKS_PER_SEC},
          {"corperc", perc},    {"avgloss", totalLoss/batch.size()},
          {"batchsize", (int)batch.size()}, {"lr", lr*batch.size()}, {"momentum", momentum}}, "training");
    }
  }
}

