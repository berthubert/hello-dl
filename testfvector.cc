#include "ext/doctest.h"
#include "array.hh"
#include <sstream>
#include "fvector.hh"
#include "tracked.hh"
using namespace std;

TEST_CASE("basic fvect4 test") {
  fvector<4> val=1234;
  CHECK(val.v[0] == 1234);
  CHECK(val.v[1] == 1234);
  CHECK(val.v[2] == 1234);
  CHECK(val.v[3] == 1234);

  CHECK(val.sum() == 4*1234);

  val.v[0]++;
  val *= 2;

  CHECK(val.v[0] == 2*1235);
  
}


TEST_CASE("basic fvect8 test") {
  fvector<8> val=1234;
  CHECK(val.v[0] == 1234);
  CHECK(val.v[1] == 1234);
  CHECK(val.v[2] == 1234);
  CHECK(val.v[3] == 1234);

  CHECK(val.sum() == 8*1234);

  val.v[0]++;
  val *= 2;

  CHECK(val.v[0] == 2*1235);
  
}

TEST_CASE("dev fvect8 test") {
  fvector<8> val({1.0, 2.0, 3.0, 4., 5., 6., 7., 8.});
  auto inv = 1.0/val;

  CHECK(inv.v[0]==1);
  CHECK(inv.v[4]==doctest::Approx(0.2));
}

TEST_CASE("fvect8 cmp test") {
  fvector<8> x({1,2,3,4,5,6,7,8});
  fvector<8> y({5,4,3,2,1,0,0,0});
  fvector<8> r = x<y;
  CHECK(r.v[0] == 1);
  CHECK(r.v[1] == 1);
  CHECK(r.v[2] == 0);
  CHECK(r.v[3] == 0);
  CHECK(r.v[4] == 0);
  CHECK(r.v[5] == 0);
  CHECK(r.v[6] == 0);
  CHECK(r.v[7] == 0);
  
  
}

TEST_CASE("fvect8 max test") {
  fvector<8> x({1,2,3,4,5,6,7,8});
  fvector<8> y({5,4,3,2,1,0,0,0});
  fvector<8> sum1=0, sum2=0;
  fvector<8> res = x<y;


  
  sum1 += res * x;
  sum2 += (!res) * x;

  //  cout<<"sum1: "<<sum1<<endl;
  //  cout<<"sum2: "<<sum2<<endl;
  //  cout<<"comp: "<<(sum1+sum2)<<endl;
  CHECK((sum1+sum2) == x);
  auto res2= maxFunc(x,y);
  CHECK(res2.v[0] == 5);
  CHECK(res2.v[1] == 4);
  
}

typedef TrackedNumber<fvector<8>> TrackedVec;
TEST_CASE("tracked fvect8 test") {
  /*
  TrackedVec a = fvector<8>({-3,-2,-1,0,1,2,3,4});
  TrackedVec b = makeFunc(a, ReluFunc()) + a;
  
  fvector<8> res = b.getVal();
  cout<<"res: "<<res<<endl;
  b.backward();
  cout<<a.getGrad()<<endl;
  */
}

TEST_CASE("tracked max test") {
  TrackedVec a = fvector<8>({-3,-2,-1, 0,1,2,3,4});
  cout<<a.getVal()<<endl;
  fvector<8> tmp({ 2, -9,-4, 3,2,1,4,7});
  TrackedVec b = fvector<8>(tmp);

  //    -3,-2,-1,0,1,2,3,4
  //b:   2 -9 -4 3 2 1 4 7 
  //res: 2 -2 -1 3 2 2 4 7 


  //  cout<< "a: "<<a.getVal()<<endl;
  //  cout<< "b: "<<b.getVal()<<endl;

  
  auto res = makeMax(a,b);
  //  cout<<"m: "<<res.getVal()<<endl;
  res.backward();
  cout<<a.getGrad()<<endl;
  cout<<b.getGrad()<<endl;
  //  0 1 1 0 0 1 0 0 
  //  1 0 0 1 1 0 1 1

  CHECK(a.getGrad().v[0] == 0);
  CHECK(a.getGrad().v[1] == 1);
  CHECK(a.getGrad().v[2] == 1);
  CHECK(a.getGrad().v[3] == 0);
  CHECK(a.getGrad().v[4] == 0);
  CHECK(a.getGrad().v[5] == 1);
  CHECK(a.getGrad().v[6] == 0);
  CHECK(a.getGrad().v[7] == 0);

  CHECK(b.getGrad().v[0] == 1);
  CHECK(b.getGrad().v[1] == 0);
  CHECK(b.getGrad().v[2] == 0);
  CHECK(b.getGrad().v[3] == 1);
  CHECK(b.getGrad().v[4] == 1);
  CHECK(b.getGrad().v[5] == 0);
  CHECK(b.getGrad().v[6] == 1);
  CHECK(b.getGrad().v[7] == 1);

  
}

TEST_CASE("tracked log test") {
  TrackedVec a = fvector<8>({3,2,1, 1,1,2,3,4});
  auto res = makeFunc(a, LogFunc());
  cout<<"m: "<<res.getVal()<<endl;
  res.backward();
  cout<<a.getGrad()<<endl;
  
}

TEST_CASE("assignment test") {
  fvector<8> a({1,2,3});
  CHECK(a.v[0]==1);
  CHECK(a.v[2]==3);
  CHECK(a.v[3]==0);
  a=1.0;
  a=fvector<8>({0});
  CHECK(a.v[5]==0.0);
}


TEST_CASE("tanh test") {
  TrackedVec x(1.0);
  TrackedVec y(1.0);
  TrackedVec res = makeFunc(x-y, TanhFunc());
  
  CHECK(res.getVal().v[5] == doctest::Approx(0.0));
  res.backward();
  CHECK(x.getGrad().v[4] == doctest::Approx(1.0));
  CHECK(y.getGrad().v[3] == doctest::Approx(-1.0));

  res.zeroGrad();
  x = 2;
  CHECK(res.getVal().v[7] ==   doctest::Approx(tanhf(1.0)));
  res.backward();
  CHECK(x.getGrad().v[0] == doctest::Approx(1 - tanhf(1.0)*tanhf(1.0)));
  CHECK(y.getGrad().v[1] == -x.getGrad().v[6]);
  
}

  
  
