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

