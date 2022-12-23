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
  CHECK(y.getGrad() == 6.0); // 2 * y 
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


TEST_CASE("division test") {
  TrackedFloat x(12.0);
  TrackedFloat y(2.0);
  TrackedFloat res = x/y;
  CHECK(res.getVal() == 6.0);
  res.backward();     // x * y^-1 
  CHECK(x.getGrad() == 0.5); // y^-1 -> 0.5
  CHECK(y.getGrad() == -3); // -y^-2 * x = -0.25 * 12 = -3
  res.zeroGrad();

  res = (x+y)/y;            // x/y + 1
  CHECK(res.getVal() == 7.0);
  res.backward();
  CHECK(x.getGrad() == 0.5);
  CHECK(y.getGrad() == -3);

  res.zeroGrad();
  res = (x +y*y)/x; // 1 + (y*y)*x^-1
  CHECK(res.getVal() == doctest::Approx(16.0/12.0));;
  res.backward();
  // -x^-2 * (y*y) = y*y/(12*12) 
  CHECK(x.getGrad() == doctest::Approx(-4.0/(12*12)));
  // 2*y/x
  CHECK(y.getGrad() == doctest::Approx(2*2.0/12.0));
}


TEST_CASE("reuse test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat res = x+y;
  CHECK(res.getVal() == 1.0);
  y=3.0;
  res.zeroGrad(); // to redo the calculation
  CHECK(res.getVal() == 2.0);
}

TEST_CASE("temporaries") {
  TrackedFloat x(3), y(1);
  TrackedFloat res = makeFunc((x - y), SquareFunc());
  CHECK(res.getVal() == 4);
  res = makeFunc((x + y), SquareFunc());
  CHECK(res.getVal() == 16);
}

TEST_CASE("self") {
  TrackedFloat x(3), y(2);
  x = x/y;
  CHECK(x.getVal() == doctest::Approx(3.0/2.0));
  x.backward();
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
  TrackedFloat res = makeFunc(x+y, ReluFunc());
  
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
  TrackedFloat res = makeFunc(x+y, SigmoidFunc());
  
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
  TrackedFloat res = makeFunc(x*x+y*y*y, SigmoidFunc());
  
  CHECK(res.getVal() ==   SigmoidFunc::func(9.0));
  res.backward();
  CHECK(x.getGrad() == -2*SigmoidFunc::deriv(9.0));
  CHECK(y.getGrad() == 3*4*SigmoidFunc::deriv(9.0));
}

TEST_CASE("sigmoid negative test") {
  TrackedFloat x(-1.0);
  TrackedFloat y(2.0);
  TrackedFloat one(1.0);
  TrackedFloat res = one-makeFunc(x*x+y*y*y, SigmoidFunc());
  
  CHECK(res.getVal() ==   1-SigmoidFunc::func(9.0));
  res.backward();
  CHECK(x.getGrad() == 2*SigmoidFunc::deriv(9.0));
  CHECK(y.getGrad() == -3*4*SigmoidFunc::deriv(9.0));
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


TEST_CASE("typical loss func test") {
  TrackedFloat result(0.75);
  TrackedFloat expected(1.0);
  TrackedFloat loss = (expected - result) * (expected - result);

  // expected^2 -2*expected*result +result^2
  // deriv result:  -2*expected + 2 * result -> -0.5
  
  CHECK(loss.getVal() == (1-0.75)*(1-0.75));
  loss.backward();
  double grad = result.getGrad();
  //  cout<<"grad: "<< grad <<endl;
  loss.zeroGrad();

  result = result.getVal() - 0.1*grad;
  CHECK(loss.getVal() == doctest::Approx((1.0-0.80)*(1.0-0.80)));
}


TEST_CASE("max test") {

  TrackedFloat x(2), y(3);
  auto res = makeMax(x,y);
  CHECK(res.getVal() == 3);
  res.backward();
  CHECK(x.getGrad()==0);
  CHECK(y.getGrad()==1);
  res.zeroGrad();
  auto res2 = TrackedFloat(4)*res;
  CHECK(res2.getVal() == 12);
  res2.backward();
  CHECK(x.getGrad()==0);
  CHECK(y.getGrad()==4);
  
}
