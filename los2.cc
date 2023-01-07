#include <typeinfo>
#include <typeindex>
#include <malloc.h>
#include <fenv.h>
#include <random>
#include <chrono>
#include <fstream>
#include "fvector.hh"
#include <iostream>
#include <array>
#include "gru.hh"
#include "textsupport.hh"
#include <sstream>
#include <dlfcn.h>
#include <unordered_map>
#include "cnn-alphabet.hh"

std::ofstream g_tree;//("./tree.part");
#include "tracked.hh"
#include "misc.hh"
#include <fstream>

#include <initializer_list>
using namespace std;

template<typename T>
struct GRUModel
{
  struct State
  {
    GRULayer<T, 98, 250> gm1;
    GRULayer<T, 250, 250> gm2;
    Linear<T, 250, 98> fc;
    void zeroGrad()
    {
      gm1.zeroGrad();
      gm2.zeroGrad();
      fc.zeroGrad();
    }
    void addGrad(const State& rhs)
    {
      gm1.addGrad(rhs.gm1);
      gm2.addGrad(rhs.gm2);
      fc.addGrad(rhs.fc);
    }

    void setGrad(const State& rhs, float divisor)
    {
      gm1.setGrad(rhs.gm1, divisor);
      gm2.setGrad(rhs.gm2, divisor);
      fc.setGrad(rhs.fc, divisor);
    }
    
    void save(std::ostream& out) const
    {
      gm1.save(out);
      gm2.save(out);
      fc.save(out);
    }

    void load(std::istream& in) 
    {
      gm1.load(in);
      gm2.load(in);
      fc.load(in);
    }
  };
  vector<NNArray<T, 98, 1>> invec;
  vector<NNArray<T, 1, 98>> expvec;
  vector<NNArray<T, 98, 1>> scorevec;

  TrackedNumber<T> totloss;
  
  void unroll(State& s, unsigned int choplen)
  {
    cout<<"Unrolling the GRU";
    totloss = TrackedNumber<T>(0.0);
    for(size_t i = 0 ; i < choplen; ++i) {
      cout<<"."; cout.flush();
      NNArray<T, 98, 1> in;
      NNArray<T, 1, 98> expected;
      in.zero();  
      expected.zero(); 
      
      invec.push_back(in);
      expvec.push_back(expected);
      auto res1 = s.fc.forward(s.gm2.forward(s.gm1.forward(in)));
      auto score = res1.logSoftMax();
      scorevec.push_back(score);
      auto loss = TrackedNumber<T>(0.0) - (expected*score)(0,0);
      totloss = totloss + loss;
    }
    totloss = totloss/TrackedNumber<T>(choplen);
    cout<<"\n";
  }

};

void benchexp()
{
  vector<float> in;
  in.resize(1000000);
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, 1};

  for(auto& item : in) 
    item = (float)d(gen);

}

int main()
{

  cout << getFunc<float>(g_fss[0])(2.5) << endl;
  fvector<8> x=4.2;
  cout << getFunc<fvector<8>>(g_fss[0])(x) << endl;
  cout << getFunc<fvector<8>>(g_fss[1])(x) << endl;
  cout << getDeriv<fvector<8>>(g_fss[1])(x) << endl;

  
  benchexp();
  return 0;
  
  typedef float T;
  //typedef fvector<4> T;

  auto f = std::make_unique<GRUModel<T>>();
  GRUModel<T>::State s;

  unsigned int choplen = 10;
  f->unroll(s, choplen - 1);
  
  //  f->scorevec.setVariable();
  //  f->expected.setVariable();
  cout<<"Getting topo"<<endl;
  auto topo = f->totloss.getTopo();

  DTime dt;
  dt.start();
  auto val = f->totloss.getVal();
  cout<<val<<", "<< dt.lapUsec()<<" usec"<<endl;

  cout<<"Conventional backward: "<<endl;
  dt.start();
  f->totloss.backward(topo);
  cout<<dt.lapUsec()<<endl;
  
  cout<<"Getting work"<<endl;
  auto w = f->totloss.getWork<float>(topo);

  cout<<"Dropping full model"<<endl;
  f.reset();
  
  unsigned int funcs=0;
  unordered_map<string, unsigned int> funccounts;
  for(const auto& item : w.work) {
    if(item.mode == TrackedNumberImp<float>::Modes::Func) {
      //funccounts[w.fundevs[item.findex].name]++;
      ++funcs;
    }
  }
  cout<<"funcs "<<funcs<<endl;
  for(const auto& fu : funccounts)
    cout<<fu.first<<": "<<fu.second<<endl;
  w.syncVariable();

  dt.start();
  for(int n = 0 ; n < 10; ++n)
    cout << w.getResult() << "\n";
  cout<<dt.lapUsec()/10.0<<endl;
  cout<<"Work items: "<<w.work.size()<<", sizeof: "<<sizeof(w.work[0])<<", memory: "<<
    w.work.size()*sizeof(w.work[0])/1000000.0<<"MB"<<endl;

  
  cout<<"Worker backward: ";

  dt.start();
  unsigned int zerousec=0;
  for(int n=0; n<10;++n) {
    w.backward();
    DTime dt2;
    dt2.start();
    w.zeroGrad();
    zerousec += dt2.lapUsec();
  }
  cout<<(dt.lapUsec()-zerousec)/10.0<<endl;


}


  #if 0
  {
    ofstream ofs("tmp.cc");
    f.loss.getCode(ofs, topo, g_dyns);
  }
  cout<<"dyns size: "<<g_dyns.size()<<endl;
  sort(g_dyns.begin(), g_dyns.end());
  //  system("clang++ -std=c++17 -O0 tmp.cc -o tmp.so -fPIC -shared");
  void* handle=dlopen("./tmp.so", RTLD_NOW|RTLD_GLOBAL);

  void * fsym = dlsym(handle, "_Z7getLossv");
  typedef void(*setval_t)(unsigned int, float);
  setval_t setval= (setval_t)dlsym(handle, "_Z6setvaljf");
  
  cout<<"dyns size: "<<g_dyns.size()<<", (void*)fsym "<<(void*)fsym<<", setval "<<(void*)setval<<endl;

  for(const auto& g : g_dyns)
    setval(g.first, g.second->d_val);

  typedef float(*func_t)();
  func_t theFunc = (func_t)fsym;
  
  dt.start();
  for(int n=0; n < 100; ++n)
    val = theFunc();
  
  cout<<dt.lapUsec() /100.0<<", "<<calls/100<<endl;
  cout<<"val: "<<val<<endl;

  g_dyns.clear();
  for(const auto& g : g_dyns)
    setval(g.first, g.second->d_val);
  #endif


#if 0
  NNArray<float, 50, 10> ax;
  NNArray<float, 10, 50> bx;

  ax.randomize();
  bx.randomize();
  auto ares = ax*bx;

  auto sum = ares.sum();
  DTime dt;
  dt.start();
  float res = sum.getVal();
  cout<<dt.lapUsec();
  cout<< "// "<< res << endl;

  auto topo = sum.getTopo();
  if(0)
  {
    ofstream ofs("tmp.cc");
    sum.getCode(ofs, topo, g_dyns);
  }
  sort(g_dyns.begin(), g_dyns.end());
  system("clang++ -std=c++17 tmp.cc -o tmp.so -fPIC -shared");
  void* handle=dlopen("./tmp.so", RTLD_NOW|RTLD_GLOBAL);

  void * f = dlsym(handle, "_Z7getLossv");
  
  cout<<"dyns size: "<<g_dyns.size()<<", (void*)f "<<(void*)f<<endl;
  typedef float(*func_t)();
  func_t theFunc = (func_t)f;
  dt.start();
  res = theFunc();
  cout<<dt.lapUsec()<<"\n";
  cout<<"res: "<<res<<endl;
#endif
#if 0 
float getval(unsigned int v)
{
  calls++;
  return g_vect[v];
  pair<unsigned int, TrackedNumberImp<float>*> fnd(v, 0);
  auto iter = lower_bound(g_dyns.begin(), g_dyns.end(), fnd);
  return (iter)->second->d_val;
}
#endif 
float getLoss() __attribute__((weak));

float getLoss()
{
  cout<<"Weak version called"<<endl;
  return -1;
}
