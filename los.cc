
#include <malloc.h>
#include <fenv.h>
#include <random>
#include <chrono>
#include <fstream>
#include "fvector.hh"
#include <iostream>
#include <array>


std::ofstream g_tree("./tree.part");
#include "tracked.hh"

#include <initializer_list>
using namespace std;

struct Test
{
  Test(const float& val) : d_val(val)
  {
    cout<<"Float constructor\n";
  }

  Test(std::initializer_list<float> vals)
  {
    cout<<"Initializer list contructor"<<endl;
  }
  float d_val;
};

int main()
{
  struct Test2
  {
    Test d_t = 0.0; // calls initializer list constructor..
  };
  Test2 t2;
}


#if 0
  typedef fvector<8> fvect;
  TrackedNumber<fvect> x(fvect({1,2,3,4,5,6,7,8}));
  TrackedNumber<fvect> y(fvect({5,4,3,2,1,0,0,0}));
  TrackedNumber<fvect> mres = x*x*x + x*y;

  cout<< (fvect({1,2,3,4,5,6,7,8}) < fvect({8,7,6,5,4,3,2,1})) <<endl;
  
  fvect n2 = mres.getVal();
  cout<<n2<<endl;
  mres.backward();
  cout<<x.getGrad()<<endl;
  // 3*x^2
  // 3, 12, 27, 48
  //               +y
  // 8  16  30  50

  cout<<y.getGrad()<<endl;
  // 1  2   3  4
  
  fvect vsum;
  vsum=(float)0.0;

  cout<<"vsum: "<<vsum<<endl;

  //  avxtest(); 
}
#endif
#if 0
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );  

  vector<float> a(40000000), b(40000000), res(40000000);
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, 1};
  
  for(auto& item : a) {
    item = d(gen);
  }
  for(auto& item : b) {
    item = d(gen);
  }

  auto start = chrono::steady_clock::now();

  for(size_t i = 0; i < a.size(); ++i) {
    res[i] = a[i] < b[i] ? b[i] : a[i];
  }
  float sum=0;
  
  g_tree<<"digraph D {"<<endl;

  TrackedFloat x(2.0);
  TrackedFloat y(3.0);
  TrackedFloat res = x + y * y; 
  cout << res.getVal() << " ==? 11"<< endl;
  res.backward();
  cout << x.getGrad() << " ==? 1 "<< endl; // 1
  cout << y.getGrad() << " ==? 6 "<<endl; // 2*y = 6
  
  g_tree<<"}"<<endl;
}


  for(size_t i = 0; i < a.size(); i += 4) {
    vsum.v += ((fvector*)&a[i])->v;
  }
  cout<<"vsum: "<<vsum<<endl;
  cout << "Sum: "<<vsum.sum()<<endl;
  auto usec = std::chrono::duration_cast<std::chrono::microseconds>(chrono::steady_clock::now()- start).count();
  cout << usec/1000.0 <<" msec\n";

  
  cout<<sizeof(TrackedNumberImp<float>)<<endl;
  cout<<sizeof(std::function<float(float)>)<<endl;
  cout<<sizeof(float(*)(float))<<endl;
  //  malloc_info(0, stdout);
  TrackedFloat a, b, c, d;
  //  malloc_info(0, stdout);
  malloc_stats();
  a=1;
  malloc_stats();
  b=1;
  malloc_stats();
  c=1;
  malloc_stats();
  d=1;
  malloc_stats();
  //  malloc_info(0, stdout);
  return 0;


#endif
