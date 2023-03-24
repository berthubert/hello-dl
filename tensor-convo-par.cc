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
#include "ext/argparse.hpp"

using namespace std;
time_t g_starttime;

bool g_mutateOnLearn;
bool g_mutateOnValidate;
void mutateImage(Tensor<float>& img)
{
  Tensor<float>::EigenMatrix orig = img.raw();
  
  int cshift = -2 + (random() % 5);
  int rshift = -2 + (random() % 5);
  for(int c = 0 ; c < 28; ++c) {
    for(int r = 0 ; r < 28 ; ++r) {
      int o_r = r + rshift;
      int o_c = c + cshift;
      if(o_r >= 0 && o_c >=0 && o_r < 28 && o_c < 28)
        img(r, c) = orig(o_r, o_c);
      else
        img(r, c) = 0;
    }
  }

  for(int n = 0 ; n < 5; ++n) {
    int r = random()% 28, c = random()% 28;
    img(r,c)= 1.0 - img(r, c);
  }

  
}

template<typename M, typename S>
void testModel(SQLiteWriter& sqw, const S& s_in, const MNISTReader& mn, unsigned int startID, int batchno, double epoch)
{
  M m;
  ostringstream os;
  s_in.save(os);

  S s;
  istringstream is(os.str());
  s.load(is);
  
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
    m.img.zero();
    mn.pushImage(idx, m.img);
    // normalize
    m.img.normalize(0.172575, 0.25);

    if(g_mutateOnValidate) { // corrupt somewhat
      mutateImage(m.img);
    }

    int label = mn.getLabel(idx) - 1;
    m.expected.oneHotColumn(label);

    totalLoss += m.modelloss(0,0); // turns it into a float
      
    int predicted = m.scores.maxValueIndexOfColumn(0);

    if(corrects + wrongs == 0) {
      printImgTensor(m.img);
      cout<<"predicted: "<<(char)(predicted+'a')<<", actual: "<<(char)('a'+label)<<", loss: "<<m.modelloss<<"\n";
      cout<<m.scores<<endl;
    }
    
    if(predicted == label)
      corrects++;
    else wrongs++;
  }
  double perc=100.0*corrects/(corrects+wrongs);
  cout<<"Validation batch average loss: "<<totalLoss/batch.size()<<", percentage correct: "<<perc<<"%, took "<<dt.lapUsec()/1000<<" ms for "<<batch.size()<<" images\n";
  
  sqw.addValue({
      {"startID", startID},  {"batchno", batchno}, {"epoch", epoch},  {"time", time(0)},
      {"cputime", (double) clock()/CLOCKS_PER_SEC}, {"elapsed", time(0) - g_starttime},
      {"corperc", perc}, {"avgloss", totalLoss/batch.size()}}, "validation");
}


template<typename M>
struct ParMod
{
  ParMod(int readpipe, int writepipe, MNISTReader& mn, std::atomic<unsigned int>& corrects, std::atomic<unsigned int>& wrongs, bool production) : d_readpipe(readpipe), d_writepipe(writepipe), d_mn(mn), d_corrects(corrects), d_wrongs(wrongs)
  {
    d_model.init(d_state, production);
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

      d_mn.pushImage(idx, d_model.img);
        // normalize
      d_model.img.normalize(0.172575, 0.25);
      if(g_mutateOnLearn) { // corrupt somewhat
        mutateImage(d_model.img);
      }

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
  std::atomic<unsigned int>& d_wrongs;
};

int main(int argc, char **argv)
{
  argparse::ArgumentParser program("tensor-convo-par");

  program.add_argument("state-file").help("state file to read from").default_value(std::string());
  program.add_argument("--lr", "--learning-rate").default_value(0.01).scan<'g', double>();
  program.add_argument("--alpha").default_value(0.001).scan<'g', double>();
  program.add_argument("--momentum").default_value(0.9).scan<'g', double>();
  program.add_argument("--batch-size").default_value(64).scan<'i', int>();
  program.add_argument("--dropout").default_value(false).implicit_value(true);
  program.add_argument("--adam").default_value(false).implicit_value(true);
  program.add_argument("--threads").default_value(4).scan<'i', int>();
  program.add_argument("--mut-on-learn").default_value(false).implicit_value(true);
  program.add_argument("--mut-on-validate").default_value(false).implicit_value(true);
  
  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  
  //  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;
  cout<<"state-file: "<<program.get<string>("state-file") << endl;
  cout<<"momentum: "<<program.get<double>("--momentum") << endl;
  cout<<"lr: "<<program.get<double>("--lr") << endl;
  cout<<"batch-size: "<<program.get<int>("--batch-size") << endl;
  cout<<"threads: "<<program.get<int>("--threads") << endl;
  cout<<"dropout: "<<program.get<bool>("--dropout") << endl;
  g_mutateOnLearn=program.get<bool>("mut-on-learn");
  g_mutateOnValidate=program.get<bool>("mut-on-validate");
  cout<<"Mutate while learning "<<g_mutateOnLearn<<", while validating: "<<g_mutateOnValidate<<endl;
  if(!program.get<string>("state-file").empty()) {
    cout<<"Loading model state from file '"<< program.get<string>("state-file") <<"'\n";
    loadModelState(s, program.get<string>("state-file"));
  }
  else {
    cout<<"Starting from random state"<<endl;
    srandom(time(0)); // weak 
    s.randomize();
  }

  m.init(s, !program.get<bool>("--dropout")); // passes 'production'

  MNISTReader     mn("gzip/emnist-letters-train-images-idx3-ubyte.gz", "gzip/emnist-letters-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz",   "gzip/emnist-letters-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" training images and "<<mntest.num()<<" test images"<<endl;

  auto topo = m.loss.getTopo();
  cout<<"Topo.size(): "<<topo.size()<<endl;

  std::atomic<unsigned int> corrects=0, wrongs=0;
  
  int toworker[2], fromworker[2];
  if(pipe(toworker) < 0 || pipe(fromworker) < 0)
    throw std::runtime_error("Creating pipe");
  
  vector<std::unique_ptr<ParMod<ConvoAlphabetModel>>> pms;
  for(int n=0; n < program.get<int>("threads"); ++n)  // [0] = read, [1]=write
    pms.push_back(std::make_unique<ParMod<ConvoAlphabetModel>>(toworker[0], fromworker[1], mn, corrects, wrongs, !program.get<bool>("--dropout"))); 

  SQLiteWriter sqw("convo-vals-par.sqlite3");
  int64_t startID=time(0);
  g_starttime = startID;
  
  int batchno = 0;

  for(;;) {
    Batcher batcher(mn.num());

    DTime dt;
    for(unsigned int tries = 0 ;; ++tries) {
      auto batch = batcher.getBatch(program.get<int>("--batch-size"));
      if(batch.empty())
        break;

      if(!(tries % 32)) {
        testModel<ConvoAlphabetModel>(sqw, s, mntest, startID, batchno, 1.0*batchno*batch.size()/mn.num()); // epoch
        saveModelState(s, "tensor-convo-par.state");
      }
      if(batchno < 32 || !(tries%32)) {
        s.emit(sqw, startID, batchno, batch.size());
      }
      dt.start();
      
      batchno++;

      float totalLoss = 0, totalWeightsLoss=0;
      
      m.loss.zeroAccumGrads(topo);
      saveModelState(s, "orig.state");
      int scounter=0;
      for(auto& pm : pms) {
        pm->d_model.loss.zeroAccumGrads(pm->d_topo);
        m.loss.copyParams(topo, pm->d_topo);
        saveModelState(pm->d_state, "copy-" + to_string(scounter++)+".state");
      }
      corrects = wrongs = 0;
      for(const auto& idx : batch) {
        size_t w = idx;
        if(write(toworker[1], &w, sizeof(w)) != sizeof(w))
          throw runtime_error("Partial write");
      }
      for(size_t pos = 0 ; pos < batch.size(); ++pos) {
        size_t ret;
        if(read(fromworker[0], &ret, sizeof(ret)) != sizeof(ret))
          throw runtime_error("Partial read");
      }

      for(auto& pm : pms) {
        m.loss.addAccumGrads(pm->d_topo, topo);
      }
      
      double perc = 100.0*corrects/(corrects+wrongs);
      cout<<"Batch "<<batchno<<" average loss " << totalLoss/batch.size()<<", weightsloss " <<totalWeightsLoss/batch.size()<<", percent batch correct "<<perc<<"%, "<<dt.lapUsec()/1000<<"ms/batch"<<endl;

      double lr= program.get<double>("--lr") / batch.size(); // 0.010 works well at the beginning
      double momentum = program.get<double>("--momentum");

      if(program.get<bool>("--adam"))
        s.learnAdam(1.0/batch.size(), batchno, program.get<double>("--alpha"));
      else
        s.learn(lr, momentum);

      // tcsv<<"batchno,cputime,corperc,avgloss,batchsize,lr,momentum"<<endl;
      sqw.addValue({
          {"startID", startID}, {"batchno", batchno}, {"epoch", 1.0*batchno*batch.size()/mn.num()}, {"time", time(0)}, {"elapsed", time(0) - g_starttime},
          {"cputime", (double)clock()/CLOCKS_PER_SEC},
          {"corperc", perc},    {"avgloss", totalLoss/batch.size()},
          {"batchsize", (int)batch.size()}, {"lr", lr*batch.size()}, {"momentum", momentum}}, "training");
    }
  }
}

