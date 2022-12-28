#include "ext/doctest.h"
#include "array.hh"
#include <sstream>
#include "fvector.hh"
#include "tracked.hh"
using namespace std;

TEST_CASE("basic fvect4 test") {
  fvector<4> val=1234;
  CHECK(val.a[0] == 1234);
  CHECK(val.a[1] == 1234);
  CHECK(val.a[2] == 1234);
  CHECK(val.a[3] == 1234);

  CHECK(val.sum() == 4*1234);

  val.a[0]++;
  val *= 2;

  CHECK(val.a[0] == 2*1235);
  
}


TEST_CASE("basic fvect8 test") {
  fvector<8> val=1234;
  CHECK(val.a[0] == 1234);
  CHECK(val.a[1] == 1234);
  CHECK(val.a[2] == 1234);
  CHECK(val.a[3] == 1234);

  CHECK(val.sum() == 8*1234);

  val.a[0]++;
  val *= 2;

  CHECK(val.a[0] == 2*1235);
  
}

TEST_CASE("dev fvect8 test") {
  fvector<8> val({1.0, 2.0, 3.0, 4., 5., 6., 7., 8.});
  auto inv = 1.0/val;

  CHECK(inv.a[0]==1);
  CHECK(inv.a[4]==doctest::Approx(0.2));
}


TEST_CASE("fvect8 max test") {
  fvector<8> x({1,2,3,4,5,6,7,8});
  fvector<8> y({5,4,3,2,1,0,0,0});
  fvector<8> sum1=0, sum2=0;
  fvector<8> res = x<y;
  cout<<res<<endl;

  sum1 += res * x;
  sum2 += (!res) * x;

  cout<<"sum1: "<<sum1<<endl;
  cout<<"sum2: "<<sum2<<endl;
  cout<<"comp: "<<(sum1+sum2)<<endl;
  CHECK((sum1+sum2) == x);

  
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


  cout<< "a: "<<a.getVal()<<endl;
  cout<< "b: "<<b.getVal()<<endl;

  
  auto res = makeMax(a,b);
  cout<<"m: "<<res.getVal()<<endl;
  res.backward();
  cout<<a.getGrad()<<endl;
  cout<<b.getGrad()<<endl;
  //  0 1 1 0 0 1 0 0 
  //  1 0 0 1 1 0 1 1

  CHECK(a.getGrad().a[0] == 0);
  CHECK(a.getGrad().a[1] == 1);
  CHECK(a.getGrad().a[2] == 1);
  CHECK(a.getGrad().a[3] == 0);
  CHECK(a.getGrad().a[4] == 0);
  CHECK(a.getGrad().a[5] == 1);
  CHECK(a.getGrad().a[6] == 0);
  CHECK(a.getGrad().a[7] == 0);

  CHECK(b.getGrad().a[0] == 1);
  CHECK(b.getGrad().a[1] == 0);
  CHECK(b.getGrad().a[2] == 0);
  CHECK(b.getGrad().a[3] == 1);
  CHECK(b.getGrad().a[4] == 1);
  CHECK(b.getGrad().a[5] == 0);
  CHECK(b.getGrad().a[6] == 1);
  CHECK(b.getGrad().a[7] == 1);

  
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
  CHECK(a.a[0]==1);
  CHECK(a.a[2]==3);
  CHECK(a.a[3]==0);
  a=1.0;
  a=fvector<8>({0});
  CHECK(a.a[5]==0.0);
}
  
  
