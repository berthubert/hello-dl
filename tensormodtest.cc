#include "ext/doctest.h"
#include "tensor2.hh"
#include <iostream>
#include <Eigen/Dense>
#include "misc.hh"
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include "tensor-layers.hh"
using namespace std;


TEST_CASE("tensor model load save test") {
  struct TestState : ModelState<float>
  {
    Conv2d<float, 28, 28, 3, 1,  32> d_lc1; // -> 26*26 -> max2d -> 13*13
    Linear<float, 512, 64> d_fc1;  
    TestState()
    {
      d_members = {{&d_lc1, "lc1"}, {&d_fc1, "fc1"}};
    }
  };

  TestState ts;
  ts.randomize();
  
  saveModelState(ts, "tensormodetest-test.state");

  TestState ts2;
  ts2.randomize(); // just to confuse things
  loadModelState(ts2, "tensormodetest-test.state");

  CHECK(ts.d_lc1.d_filters[16](2,2) == ts2.d_lc1.d_filters[16](2,2));
  CHECK(ts.d_fc1.d_weights(30,32) == ts2.d_fc1.d_weights(30,32));
}


TEST_CASE("max2d array") {
  std::array<Tensor<float>, 3> in;
  for(unsigned int n=0; n < in.size(); ++n) {
    in[n]=Tensor<float>(4,4);
    in[n].iota(n);
  }
  auto res = Max2dfw(in, 2);
  // 0   1  2  3
  // 4   5  6  7
  // 8   9 10 11
  // 12 13 14 15

  CHECK(res[0](0,0) == 5);
  CHECK(res[0](0,1) == 7);

  CHECK(res[1](0,0) == 6);
  CHECK(res[1](0,1) == 8);

  CHECK(res[2](1,0) == 15);
  CHECK(res[2](1,1) == 17);
}

