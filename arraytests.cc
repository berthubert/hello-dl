#include "ext/doctest.h"
#include "array.hh"
#include <sstream>
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
  //  cout<<s.getVal()<<endl;
  s.backward();

  //  cout<<in(0,0).getGrad()<<endl;
}

TEST_CASE("array mean, min, max test") {

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
  CHECK(in.minValueIndexOfColumn(0) == 0);
}

TEST_CASE("array meanstd test") {

  NNArray<float, 4, 1> in;
  in(0,0)=1;
  in(1,0)=2;
  in(2,0)=4;
  in(3,0)=3;

  auto [mean, std] = in.getMeanStd();
  CHECK(mean == doctest::Approx(2.5));
  CHECK(std == doctest::Approx(sqrt(((1-2.5)*(1-2.5) +
                                     (2-2.5)*(2-2.5) +
                                     (4-2.5)*(4-2.5) +
                                     (3-2.5)*(3-2.5))/3.0)));

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


TEST_CASE("max2d") {

  NNArray<float, 4, 4> in;
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;        in(0,3)=1;      
  in(1,0)=0;        in(1,1)=0;        in(1,2)=0;        in(1,3)=0;      
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;        in(2,3)=-5;   
  in(3,0)=1;        in(3,1)=7;        in(3,2)=-4;        in(3,3)=-3;      

  auto m = in.Max2d<2>();

  CHECK(m(0,0).getVal() == 2);
  CHECK(m(0,1).getVal() == 3);
  CHECK(m(1,0).getVal() == 7);
  CHECK(m(1,1).getVal() == -3);

  auto s = m.sum();
  s.backward();
  CHECK(in(0,0).getGrad() == 0);
  CHECK(in(0,1).getGrad() == 1);
}

TEST_CASE("max2d padding") {

  NNArray<float, 3, 3> in;
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;       
  in(1,0)=0;        in(1,1)=0;        in(1,2)=0;       
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;      


  NNArray<float, 2, 2> m = in.Max2d<2>();

  CHECK(m(0,0).getVal() == 2);
  CHECK(m(0,1).getVal() == 3);
  CHECK(m(1,0).getVal() == 0);
  CHECK(m(1,1).getVal() == -9);

  auto s = m.sum();
  s.backward();
  CHECK(in(0,0).getGrad() == 0);
  CHECK(in(0,1).getGrad() == 1);
}



TEST_CASE("convo2d simple") {
  NNArray<float, 4, 4> in;
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;        in(0,3)=1;      
  in(1,0)=0;        in(1,1)=0;        in(1,2)=0;        in(1,3)=0;      
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;        in(2,3)=-5;   
  in(3,0)=1;        in(3,1)=7;        in(3,2)=-4;        in(3,3)=-3;      

  NNArray<float, 1, 1> w;
  w(0,0)=2;
  NNArray<float, 1, 1> b;
  b(0,0)=0;
  auto m = in.Convo2d<1>(w,b);

  CHECK(m(0,0).getVal() == 2);
  CHECK(m(0,1).getVal() == 4);
  CHECK(m(1,0).getVal() == 0);
  CHECK(m(1,1).getVal() == 0);
  CHECK(m(3,3).getVal() == -6);

  
}

TEST_CASE("convo2d more") {
  NNArray<float, 4, 4> in;
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;        in(0,3)=1;      
  in(1,0)=1;        in(1,1)=0;        in(1,2)=0;        in(1,3)=0;      
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;        in(2,3)=-5;   
  in(3,0)=1;        in(3,1)=7;        in(3,2)=-4;        in(3,3)=-3;      

  NNArray<float, 4, 4> w;
  NNArray<float, 1, 1> b;
  w.constant(2);
  b.constant(1);
  auto m = in.Convo2d<4>(w,b);
  CHECK(in.sum().getVal() == -5);
  CHECK(m(0,0).getVal() == -9);  // 2*-5 + 1

  auto s = m.sum();
  s.backward();
  CHECK(in(0,0).getGrad() == 2);
  CHECK(w(0,0).getGrad() == 1);
  CHECK(w(0,1).getGrad() == 2);
  CHECK(b(0,0).getGrad() == 1);
}

TEST_CASE("combined") {
  NNArray<float, 28, 28> img;
  int ctr=0;
  for(unsigned int r=0; r < img.getRows(); ++r)
    for(unsigned int c=0; c < img.getCols(); ++c)
      img(r,c)=(ctr++);

  // 0 28 ...
  // 1 29
  // 2 30
  // 3 31
  // . ..


  NNArray<float, 5,5> kernel;
  NNArray<float, 1,1> bias;
  bias.zero();
  kernel.constant(2);
  kernel(4,4) = 0; // a hole
  
  auto res = img.Convo2d<5>(kernel, bias);
  CHECK(res.getRows() == 24);
  CHECK(res.getCols() == 24);

  float val=0;

  for(int r=0; r< 5; ++r) {
    for(int c=0; c< 5; ++c) {
      val+=c * 28 + r;
    }
  }
  val -= 4*28 + 4;
  val *= 2;
  
  CHECK(res(0,0).getVal() == val);

  res(0,0).backward();
  CHECK(img(0,0).getGrad() == 2);
  CHECK(img(1,2).getGrad() == 2);
  CHECK(img(3,2).getGrad() == 2);
  CHECK(img(4,4).getGrad() == 0);
}

TEST_CASE("cross entropy") {
  NNArray<float, 4, 1> in;
  in(0,0)=2; 
  in(1,0)=0; 
  in(2,0)=0; 
  in(3,0)=1;

  auto logscores = in.logSoftMax();
  //  cout<<"Logscores:\n"<<logscores<<endl;

  NNArray<float, 1, 4> expected;
  expected.zero();
  expected(0,0)=1; 
  //  cout<<"Expected: "<<expected<<endl;
  auto loss = TrackedFloat(0) - (expected*logscores)(0,0);
  auto oldloss = loss.getVal();
  //  cout<<"Loss: "<<loss.getVal()<<endl;
  loss.backward();

  //  cout << in(0,0).getGrad()<<endl;
  //  cout << in(1,0).getGrad()<<endl;

  // "learn a bit"
  in(0,0) = in(0,0).getVal() - 0.2 * in(0,0).getGrad();
  loss.zeroGrad();
  //  cout<<"New loss: "<<loss.getVal() << endl;
  CHECK(loss.getVal() < oldloss);
}

TEST_CASE("array dot hadamard test")
{
  NNArray<float, 2, 2> a;
  a(0,0)=1;   a(0,1)=2; 
  a(1,0)=3;   a(1,1)=4; 

  NNArray<float, 2, 2> b;
  b(0,0)=4;   b(0,1)=3; 
  b(1,0)=1;   b(1,1)=2;

  NNArray<float, 2, 2> res = a.dot(b);
  CHECK(res(0,0).getVal() == 4);
  CHECK(res(0,1).getVal() == 6);
  CHECK(res(1,0).getVal() == 3);
  CHECK(res(1,1).getVal() == 8);

  NNArray<float, 2, 2> res2 = b.dot(a);
  CHECK(res2(0,0).getVal() == 4);
  CHECK(res2(0,1).getVal() == 6);
  CHECK(res2(1,0).getVal() == 3);
  CHECK(res2(1,1).getVal() == 8);

  auto sum = res2.sum();
  sum.backward();
  CHECK(a(0,0).getGrad() == 4.0);
  CHECK(b(0,1).getGrad() == 2.0);
}

TEST_CASE("Saving and restoring arrays")
{
  ostringstream str;
  NNArray<float, 20, 25> f;
  f.randomize();

  f.save(str);

  ofstream ofs("test.arr");
  f.save(ofs);
  
  string saved = str.str();

  NNArray<float, 20, 25> restored;
  restored.zero();
  istringstream istr(saved);

  restored.load(istr);

  auto diff = restored - f;
  CHECK(diff.sum().getVal() ==  0);
  
}
