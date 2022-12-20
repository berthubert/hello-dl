#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "ext/doctest.h"
#include "tracked.hh"
using namespace std;

ofstream g_tree; 

TEST_CASE("basic test") {
  TrackedFloat x(2.0);
  TrackedFloat y(3.0);
  TrackedFloat res = x * y;
  CHECK(res.getVal() == 6.0);
}


TEST_CASE("quadratic test") {
  TrackedFloat x(2.0);
  TrackedFloat y(3.0);
  TrackedFloat res = x + y * y;
  CHECK(res.getVal() == 11.0);
  res.backward();
  CHECK(x.getGrad() == 1.0);
  CHECK(y.getGrad() == 6.0);
}

TEST_CASE("quadratic test negative") {
  TrackedFloat x(-1.0);
  TrackedFloat y(-3.0);
  TrackedFloat res = x + y * y;
  CHECK(res.getVal() == 8.0);
  res.backward();
  CHECK(x.getGrad() == 1.0);
  CHECK(y.getGrad() == -6.0);
}

TEST_CASE("cube test negative") {
  TrackedFloat x(-1.0);
  TrackedFloat y(-4.0);
  TrackedFloat res = x*x + y * y * y;
  CHECK(res.getVal() == -63.0);
  res.backward();
  CHECK(x.getGrad() == -2.0);
  CHECK(y.getGrad() == 48.0); // -> 3*y^2 = 48
}


TEST_CASE("addition test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(-4.0);
  TrackedFloat res = x+x+x+x +y +y +y +x +x; 
  CHECK(res.getVal() == -18.0);
  res.backward();
  CHECK(x.getGrad() == 6);
  CHECK(y.getGrad() == 3);
}


TEST_CASE("subtraction test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(-4.0);
  TrackedFloat res = x-x - y;
  CHECK(res.getVal() == 4.0);
  res.backward();
  CHECK(x.getGrad() == 0);
  CHECK(y.getGrad() == -1);
}

TEST_CASE("reuse test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat res = x+y;
  CHECK(res.getVal() == 1.0);
  y=3.0;
  CHECK(res.getVal() == 2.0);
}

TEST_CASE("reuse test grad") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat res = x+y*y;
  CHECK(res.getVal() == 3.0);
  res.backward();
  CHECK(x.getGrad() == 1.0);
  CHECK(y.getGrad() == 4.0);
  res.zeroGrad();
  CHECK(res.getVal() == 3.0);
  res.backward();
  CHECK(x.getGrad() == 1.0);
  CHECK(y.getGrad() == 4.0);

  res.zeroGrad();
  y=3.0;

  CHECK(res.getVal() == 8.0);
  res.backward();
  CHECK(x.getGrad() == 1.0);
  CHECK(y.getGrad() == 6.0);

}


TEST_CASE("relu test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat res = doFunc(x+y, ReluFunc());
  
  CHECK(res.getVal() == 1.0);
  res.backward();
  CHECK(x.getGrad() == 1.0);
  CHECK(y.getGrad() == 1.0);
  res.zeroGrad();

  y=0.0;
  CHECK(res.getVal() == 0.0);
  res.backward();
  CHECK(x.getGrad() == 0.0);
  CHECK(y.getGrad() == 0.0);
}

TEST_CASE("sigmoid test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat res = doFunc(x+y, SigmoidFunc());
  
  CHECK(res.getVal() ==   SigmoidFunc::func(1.0));
  res.backward();
  CHECK(x.getGrad() == SigmoidFunc::deriv(1.0));
  CHECK(y.getGrad() == SigmoidFunc::deriv(1.0));

  res.zeroGrad();
  x=-20;
  CHECK(res.getVal() ==   SigmoidFunc::func(-18.0));
  res.backward();
  CHECK(x.getGrad() == SigmoidFunc::deriv(-18.0));
  CHECK(y.getGrad() == SigmoidFunc::deriv(-18.0));
  
}


TEST_CASE("sigmoid advanced test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat res = doFunc(x*x+y*y*y, SigmoidFunc());
  
  CHECK(res.getVal() ==   SigmoidFunc::func(9.0));
  res.backward();
  CHECK(x.getGrad() == -2*SigmoidFunc::deriv(9.0));
  CHECK(y.getGrad() == 3*4*SigmoidFunc::deriv(9.0));
}


TEST_CASE("vector test") {
  vector<TrackedFloat> vec;
  for(int n=0; n < 100; ++n)
    vec.emplace_back(n);

  TrackedFloat res(0.0);
  for(const auto& v : vec)
    res = res + v;
  
  CHECK(res.getVal() == 4950);
  res.backward();
  CHECK(vec[0].getGrad() == 1);
  CHECK(vec[10].getGrad() == 1);
  CHECK(vec[20].getGrad() == 1);

  res.zeroGrad();
  vec[0] = 1.0;
  
  CHECK(res.getVal() == 4951);  
}


