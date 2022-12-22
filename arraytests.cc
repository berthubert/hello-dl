#include "ext/doctest.h"
#include "array.hh"
using namespace std;


TEST_CASE("basic array test") {
  NNArray<float, 1,1> test1, test2;
  test1(0,0)=12;
  test2(0,0)=2;
  NNArray<float, 1,1> res;
  res = test1*test2;
  
  CHECK(test1(0,0).getVal() == 12);
  CHECK(res(0,0).getVal() == 24);
}


TEST_CASE("basic rect array test") {
  NNArray<float, 2,3> test1;
  NNArray<float, 3,2> test2;
  
  test1(0,0) = 1;  test1(0,1) = 2; test1(0,2) = 3;
  test1(1,0) = 4;  test1(1,1) = 5; test1(1,2) = 6;

  test2(0,0) = 7;  test2(0,1) = 8;
  test2(1,0) = 9;  test2(1,1) = 10;
  test2(2,0) = 11;  test2(2,1) = 12;

  NNArray<float, 2, 2> res = test1*test2;
  
  CHECK(res(0,0).getVal() == 58.0);
  CHECK(res(1,0).getVal() == 139.0);
  CHECK(res(0,1).getVal() == 64.0);
  CHECK(res(1,1).getVal() == 154.0);

  NNArray<float, 3, 3> res2 = test2*test1;
  CHECK(res2(0,0).getVal() == 39.0);
  CHECK(res2(1,0).getVal() == 49.0);
  CHECK(res2(2,0).getVal() == 59.0);

  CHECK(res2(0,1).getVal() == 54.0);
  CHECK(res2(1,1).getVal() == 68.0);
  CHECK(res2(2,1).getVal() == 82.0);

  CHECK(res2(0,2).getVal() == 69.0);
  CHECK(res2(1,2).getVal() == 87.0);
  CHECK(res2(2,2).getVal() == 105.0);
}


TEST_CASE("array applyfunc test") {
  NNArray<float, 1,1> test1, test2;
  test1(0,0)=12;
  test2(0,0)=2;
  NNArray<float, 1,1> res;

  res = test1 - test2;
  CHECK(test1(0,0).getVal() == 12);
  CHECK(test2(0,0).getVal() == 2);
  CHECK(res(0,0).getVal() == 10);

  auto res2 = (test2-test1).applyFunc(SquareFunc());
  CHECK(res2(0,0).getVal() == 100);


  auto res3 = (test2+test1).applyFunc(SquareFunc());
  CHECK(res3(0,0).getVal() == 196);
}

TEST_CASE("array sum test") {
  NNArray<float, 4, 1> in;
  in(0,0)=1;
  in(1,0)=2;
  in(2,0)=3;
  in(3,0)=4;

  
  TrackedNumber<float> res = in.sum();
  CHECK(res.getVal() == 10);
  res.backward();
  CHECK(in(0,0).getGrad() == 1);
  res.zeroGrad();
  
  NNArray<float, 1,4> w;
  w(0,0) = 1; w(0,1)=2 ; w(0,2)=3; w(0,3)=4;

  auto mres = in*w;
  auto s=mres.sum();
  cout<<s.getVal()<<endl;
  s.backward();

  cout<<in(0,0).getGrad()<<endl;
  
}

TEST_CASE("array mean test") {

  NNArray<float, 4, 1> in;
  in(0,0)=1;
  in(1,0)=2;
  in(2,0)=4;
  in(3,0)=3;

  TrackedNumber m = in.mean();
  CHECK(m.getVal() == doctest::Approx(2.5));
  m.backward();
  CHECK(in(0,0).getGrad() == doctest::Approx(0.25));

  CHECK(in.maxValueIndexOfColumn(0) == 2);
}


TEST_CASE("array logsoftmax") {

  NNArray<float, 4, 1> in;
  in(0,0)=1; // e
  in(1,0)=0; // 1
  in(2,0)=0; // 1 
  in(3,0)=1; // e

  auto m = in.logSoftMax();
  TrackedNumber msum = m.sum();

  CHECK(m(0,0).getVal() == doctest::Approx(-1.00641));
  CHECK(m(1,0).getVal() == doctest::Approx(-2.00641));
  CHECK(m(2,0).getVal() == doctest::Approx(-2.00641));
  CHECK(m(3,0).getVal() == doctest::Approx(-1.00641));
}
