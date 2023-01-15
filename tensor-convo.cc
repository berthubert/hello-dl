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
using namespace std;

  
struct ConvoAlphabetModel {
  Tensor<float> img{28,28};
  Tensor<float> scores{26, 1};
  Tensor<float> expected{1,26};
  Tensor<float> loss{1,1};
  Tensor<float> rawscores;
  struct State
  {
    //           r_in c   k c_i  c_out
    Conv2d<float, 28, 28, 3, 1,  32> c1; // -> 26*26 -> max2d -> 13*13
    Conv2d<float, 13, 13, 3, 32, 64> c2; // -> -> 11*11 -> max2d -> 6*6 //padding
    Conv2d<float, 6,   6, 3, 64, 128> c3; // -> 4*4 -> max2d -> 2*2
    // flattened to 512
           //      IN OUT
    Linear<float, 512, 64> fc1;  
    Linear<float, 64, 128> fc2;
    Linear<float, 128, 26> fc3; 
    
    void randomize()
    {
      c1.randomize();
      c2.randomize();
      c3.randomize();
      fc1.randomize();
      fc2.randomize();
      fc3.randomize();
    }

    void learn(float lr)
    {
      c1.learn(lr);
      c2.learn(lr);
      c3.learn(lr);
      fc1.learn(lr);
      fc2.learn(lr);
      fc3.learn(lr);
    }
  };
  
  void init(State& s)
  {
    img.zero();
    img.d_imp->d_nograd=true;
    auto step1 = s.c1.forward(img);
    
    std::array<Tensor<float>, 32> step2; // 13x13
    unsigned ctr=0;
    for(auto& p : step2)
      p = makeFunction<ReluFunc>(step1[ctr++].makeMax2d(2));

    std::array<Tensor<float>, 64> step3 = s.c2.forward(step2);  // 11x11
    std::array<Tensor<float>, 64> step4;                   // 6x6

    ctr=0;
    for(auto& p : step4) {
      p = makeFunction<ReluFunc>(step3[ctr++].makeMax2d(2));
    }

    std::array<Tensor<float>, 128> step5 = s.c3.forward(step4); // 4x4
    std::array<Tensor<float>, 128> step6;                  // 2x2

    ctr=0;
    for(auto& p : step6) {
      p = makeFunction<ReluFunc>(step5[ctr++].makeMax2d(2));
    }
    
    Tensor<float> flat = makeFlatten(step6); // 2*2*128 * 1
    auto output = s.fc1.forward(flat);
    auto output2 = makeFunction<ReluFunc>(output);
    auto output3 = makeFunction<ReluFunc>(s.fc2.forward(output2));
    auto output4 = makeFunction<ReluFunc>(s.fc3.forward(output3));
    rawscores = output4;
    scores = makeLogSoftMax(output4);
    loss = -(expected*scores).sum();
  }
};

template<typename M>
void testModel(M& m, const MNISTReader& mn, int batchno)
{
  static ofstream vcsv("validation3.csv");
  static bool first=true;
  if(first) {
    vcsv<<"batchno,corperc,avgloss\n";
    first=false;
  }

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
    int label = mn.getLabel(idx) - 1;
    m.expected.zero();
    m.expected(0, label) = 1;

    totalLoss += m.loss(0,0); // turns it into a float
      
    int predicted = m.scores.maxValueIndexOfColumn(0);

    if(corrects + wrongs == 0) {
      printImgTensor(m.img);
      cout<<"predicted: "<<(char)(predicted+'a')<<", actual: "<<(char)('a'+label)<<", loss: "<<m.loss<<"\n";
    }
    
    if(predicted == label)
      corrects++;
    else wrongs++;

  }
  double perc=100.0*corrects/(corrects+wrongs);
  cout<<"Validation batch average loss: "<<totalLoss/batch.size()<<", percentage correct: "<<perc<<", took "<<dt.lapUsec()/1000<<" ms for "<<batch.size()<<" images\n";
  vcsv << batchno <<"," << perc << ", " << totalLoss/batch.size() << endl; 
}

int main()
{
  cout<<"Start!"<<endl;
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mn("gzip/emnist-letters-train-images-idx3-ubyte.gz", "gzip/emnist-letters-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" training images and "<<mntest.num()<<" test images"<<endl;

  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;
  srandom(time(0));
  s.randomize();
  //  cout<<s.fc1.d_weights<<endl;
  m.init(s);

  auto topo = m.loss.getTopo();
  cout<<"Topo.size(): "<<topo.size()<<endl;

  std::unordered_map<int, int> tcounts;
  for(const auto& t : topo) {
    tcounts[(int)t->d_mode]++;
  }
  for(const auto& tc: tcounts) {
    cout<<tc.first<<": "<<tc.second<<endl;
  }
  return 0;
  ofstream tcsv("training3.csv");
  tcsv<<"batchno,corperc,avgloss,batchsize,lr,momentum"<<endl;

  int batchno = 0;
  for(;;) {
    Batcher batcher(mn.num());
    
    DTime dt;
    for(unsigned int tries = 0 ;; ++tries) {
      if(!(tries % 32))
        testModel(m, mntest, batchno);
      dt.start();
      auto batch = batcher.getBatch(64);
      if(batch.empty())
        break;
      float totalLoss = 0;
      unsigned int corrects=0, wrongs=0;
      
      m.loss.zeroAccumGrads(topo);
      
      for(const auto& idx : batch) {     
        mn.pushImage(idx, m.img);
        int label = mn.getLabel(idx) -1;
        m.expected.zero();
        m.expected(0, label) = 1;
        
        totalLoss += m.loss(0,0); // turns it into a float
        
        int predicted = m.scores.maxValueIndexOfColumn(0);
        
        if(corrects + wrongs == 0) {
          cout<<"predicted: "<<(char)(predicted+'a')<<", actual: "<<(char)(label+'a')<<", loss: "<<m.loss<<"\n";
          printImgTensor(m.img);
          //        cout<<m.rawscores<<endl;
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
      cout<<"Batch "<<batchno<<" average loss " << totalLoss/batch.size()<<", percent batch correct "<<perc<<"%, "<<dt.lapUsec()/1000<<"ms/batch"<<endl;
      
      double lr=0.01 / batch.size();
      s.learn(lr);
      tcsv << batchno << "," << perc << ", " << totalLoss / batch.size() << ","<< batch.size() <<","<<lr*batch.size()<<","<<0<<endl; 
    }
  }
}

