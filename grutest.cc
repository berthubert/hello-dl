#include "ext/doctest.h"
#include "tracked.hh"
#include "gru.hh"
#include <iostream>

using namespace std;

// a sequence needs to be dragged through
#if 0
TEST_CASE("single GRU") {
  GRULayer<float, 26, 250> gm;
  //  cout<<"gm.size(): "<<gm.size() << endl;
  Linear<float, 250, 26> fc;
  //  cout<<"fc.size(): "<<fc.size() << endl;
  
  NNArray<float, 26, 1> in;
  NNArray<float, 1, 26> expected;

  std::string input = "hellothisisbert";
  std::string output;
  TrackedFloat totloss=0.0;
  for(size_t pos = 0 ; pos < input.size() - 1; ++pos) {
    in.zero();  in(input.at(pos)-'a', 0) = 1.0;
    
    expected.zero(); expected(0, input.at(pos+1)-'a') = 1.0;
    auto res1 = fc.forward(gm.forward(in));
    auto score = res1.logSoftMax();
    auto loss = TrackedNumber<float>(0.0) - (expected*score)(0,0);
    totloss = totloss + loss;
    output.append(1, 'a' + score.maxValueIndexOfColumn(0));
    //    cout<<"score: "<<loss.getVal() << endl;
  }
  //  cout<<output<<endl;
  totloss.backward();
  //  cout<<gm.d_w_ir.getGrad()<<endl;
}


TEST_CASE("multi GRU") {
  GRULayer<float, 10, 250> gm1, gm2, gm3;
  //  cout<<"gm.size(): "<<gm1.size() << endl;
  Linear<float, 250, 10> fc1, fc2, fc3;
  
  NNArray<float, 10, 1> in1, in2, in3;
  in1.zero();  in1(3,0) = 1.0;
  in2.zero();  in2(1,0) = 1.0;
  in3.zero();  in3(4,0) = 1.0;
  auto res1 = gm1.forward(in1);
  gm2.d_prevh = res1;

  auto res2 = gm2.forward(in2);
  gm3.d_prevh = res2;

  auto res3 = gm3.forward(in3);

  cout<<"GRU: "<<endl;
  cout << fc1.forward(res1).logSoftMax().maxValueIndexOfColumn(0) << endl;
  cout << fc2.forward(res2).logSoftMax().maxValueIndexOfColumn(0) << endl;
  cout << fc3.forward(res3).logSoftMax().maxValueIndexOfColumn(0) << endl;
  
  
}
#endif
