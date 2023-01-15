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

template<typename T>
struct ModelState
{
  ModelState() {}
  ModelState(const ModelState&) = delete;
  ModelState& operator=(const ModelState& rhs) = delete;
  void randomize()
  {
    for(auto& m : d_members)
      m->randomize();
  }
  void learn(float lr)
  {
    for(auto& m : d_members)
      m->learn(lr);
  }

  void save(std::ostream& out) const
  {
    for(const auto& m : d_members)
      m->save(out);
  }
  void load(std::istream& in)
  {
    for(auto& m : d_members)
      m->load(in);
  }

  
  std::vector<TensorLayer<T>*> d_members;
};
  
struct ReluDigitModel {
  Tensor<float> img{28,28};
  Tensor<float> scores{10, 1};
  Tensor<float> expected{1,10};
  Tensor<float> loss{1,1};
  struct State : public ModelState<float>
  {
    Linear<float, 28*28, 128> lc1;
    Linear<float, 128,    64> lc2;
    Linear<float, 64,     10> lc3;

    State()
    {
      this->d_members = {&lc1, &lc2, &lc3};
    }
    
    
  };
  
  void init(State& s)
  {
    auto output = s.lc1.forward(makeFlatten({img}));
    auto output2 = makeFunction<ReluFunc>(output);
    auto output3 = s.lc2.forward(output2);
    auto output4 = makeFunction<ReluFunc>(output3);
    auto output5 = s.lc3.forward(output4);
    scores = makeLogSoftMax(output5);
    loss = -(expected*scores).sum();
  }
};

template<typename M>
void testModel(M& m, const MNISTReader& mn)
{
  Batcher b(mn.num());
  auto batch = b.getBatch(128);
  float totalLoss=0;
  int corrects=0, wrongs=0;
  auto topo = m.loss.getTopo();
  for(const auto& idx : batch) {
    m.loss.zerograd(topo);
    mn.pushImage(idx, m.img);
    int label = mn.getLabel(idx);
    m.expected.zero();
    m.expected(0, label) = 1;
    
    totalLoss += m.loss(0,0); // turns it into a float
      
    int predicted = m.scores.maxValueIndexOfColumn(0);

    if(corrects + wrongs == 0) {
      printImgTensor(m.img);
      cout<<"predicted: "<<(int)predicted<<", actual: "<<label<<", loss: "<<m.loss<<"\n";
    }
    
    if(predicted == label)
      corrects++;
    else wrongs++;

  }
  cout<<"Validation batch average loss: "<<totalLoss/batch.size()<<", percentage correct: "<<(100.0*corrects/(corrects+wrongs))<<"%\n";
}

int main(int argc, char** argv)
{
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" training images and "<<mntest.num()<<" test images"<<endl;

  ReluDigitModel m;
  ReluDigitModel::State s;
  if(argc==2) {
    cout<<"Loading model state from '"<<argv[1]<<"'\n";
    loadModelState(s, argv[1]);
  }
  else
    s.randomize();

  m.init(s);

  auto topo = m.loss.getTopo();
  
  Batcher batcher(mn.num()); 
  for(unsigned int tries = 0 ;; ++tries) {
    if(!(tries % 32)) {
      testModel(m, mntest);
      saveModelState(s, "tensor-relu.state");
    }
    
    auto batch = batcher.getBatch(64);
    if(batch.empty())
      break;
    float totalLoss = 0;
    unsigned int corrects=0, wrongs=0;

    m.loss.zeroAccumGrads(topo);
    
    for(const auto& idx : batch) {

      mn.pushImage(idx, m.img);
      int label = mn.getLabel(idx);
      m.expected.zero();
      m.expected(0, label) = 1;

      totalLoss += m.loss(0,0); // turns it into a float
      
      int predicted = m.scores.maxValueIndexOfColumn(0);
#if 0
      if(corrects + wrongs == 0) {
        printImgTensor(m.img);
        cout<<"predicted: "<<(int)predicted<<", actual: "<<label<<", loss: "<<m.loss<<"\n";
      }
#endif 
      if(predicted == label)
        corrects++;
      else wrongs++;

      // backward the thing
      m.loss.backward(topo);
      m.loss.accumGrads(topo); 
      // clear grads & havevalue
      m.loss.zerograd(topo);
    }
    cout<<tries<<": Average loss " << totalLoss/batch.size()<<", percent batch correct "<<100.0*corrects/(corrects+wrongs)<<"%\n";
    
    double lr=0.1 / batch.size();
    s.learn(lr);
  }
}

