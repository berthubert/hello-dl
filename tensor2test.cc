#include "ext/doctest.h"
#include "tensor2.hh"
#include <iostream>
#include <Eigen/Dense>
#include "misc.hh"
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
using namespace std;


TEST_CASE("basic tensor")
{
  Tensor a(4, 2);
  Tensor b(4, 2);
  Tensor c(4, 2);

  a(0,0) = 3;
  b(0,0) = 4;
  c(2,1) = 5;
  
  auto res = a + b + c;
  CHECK(res(0,0) == 7);
  CHECK(res(1,1) == 0);
  CHECK(res(2,1) == 5);
  cout<<res<<endl;
  
  Tensor x(1,4);
  Tensor y(4,1);
  x(0,0)=2.0;
  y(0,0)=2.5;
  auto z = x * y;

  cout<<"z: '"<< (z) <<"'\n";
  Tensor q(1,1);
  q(0,0)=3;
  cout<<"Los: '"<< (z+q) <<"'\n";

}


TEST_CASE("basic tensor backwards")
{
  Tensor a(4, 2);
  Tensor b(2, 4);
  Tensor c(2, 2);

  a(0,0) = 1; // grad 4
  a(1,0) = 2; // grad 5
  a(2,0) = 3; // grad 6
  a(3,0) = 4; // grad 7

  //  grad 1             2             3             4
  b(0,0) = 4;   b(0,1) = 5;   b(0,2) = 6;   b(0,3) = 7;

  c(0,0) = 3;
  c(1,0) = 2;
  auto res = (b * a);
  cout<<"res:\n"<<res<<endl; 

  auto tot = res.sum();
  cout<<"tot: "<<tot<<endl;

  cout<<"a:\n"<<a<<endl;
  cout<<"b:\n"<<b<<endl;
  
  auto topo = tot.getTopo();
  tot.backward(topo);
  
  cout << "a grad: \n"<<a.getGrad() << endl;
  cout << "b grad: \n"<<b.getGrad() << endl;
  cout << "c grad: \n"<<c.getGrad() << endl;

  CHECK(a.getGrad()(0,0) == 4);
  CHECK(a.getGrad()(0,1) == 4);
  CHECK(a.getGrad()(3,0) == 7);
  CHECK(a.getGrad()(3,1) == 7);

  CHECK(b.getGrad()(0,0) == 1);
  CHECK(b.getGrad()(0,1) == 2);
  CHECK(b.getGrad()(1,2) == 3);
  CHECK(b.getGrad()(1,3) == 4);
}


TEST_CASE("complexer tensor backwards")
{

  //                             a a a
  //                             a a a
  //                             a a a
  Tensor a(4, 3);   //           a a a
  Tensor b(2, 4);   //  b b b b  
  Tensor c(2, 3);   //  b b b b

  a(0,0) = 1;  a(0,1) = 1.5;
  a(1,0) = 2;  a(1,1) = 1.6;
  a(2,0) = 3;  a(2,1) = 1.7;
  a(3,0) = 4;  a(3,1) = 1.8;

  //  grad 1             2             3             4
  b(0,0) = 4;     b(0,1) = 5;     b(0,2) = 6;     b(0,3) = 7;
  b(1,0) = 1.1;   b(1,1) = 2.2;   b(1,2) = 3.3;   b(1,3) = 4.4;

  c(0,0) = 3;
  c(1,0) = 2;

  cout<< "b*a:\n"<< (b*a) <<endl;
  
  auto res = ((b * a));
  cout<<"Sum:\n"<<res<<endl; 

  auto tot = res.sum();
  cout<<"tot: "<<tot<<endl;

  cout<<"a:\n"<<a<<endl;
  cout<<"b:\n"<<b<<endl;
  
  auto topo = tot.getTopo();
  tot.backward(topo);
  
  cout << "a grad: \n"<<a.getGrad() << endl;
  cout << "b grad: \n"<<b.getGrad() << endl;
  cout << "c grad: \n"<<c.getGrad() << endl;

}


TEST_CASE("func tensor backwards")
{
  Tensor x(2,2);
  x(0,0) = 4;   x(1,1) = 7;
  x(1,0) = -3;
  
  auto res = makeFunction<ReluFunc>(x);
  CHECK(res(0,0) == 4);
  CHECK(res(1,1) == 7);
  CHECK(res(1,0) == 0);

  auto s = res.sum();

  cout<<"Sum: "<<s<<endl;
  auto topo = s.getTopo();
  s.backward(topo);

  CHECK(x.getGrad()(0,0) == 1);
  CHECK(x.getGrad()(1,0) == 0);
  CHECK(x.getGrad()(1,1) == 1);
}


TEST_CASE("tensor logsoftmax")
{
  Tensor x(1, 5);
  x(0,0) = 1; x(0,1) = 2; x(0,2) = 3; x(0,3) = 4; x(0,4) = 5;

  auto res=makeLogSoftMax(x);
  cout<<res<<endl;
  // pytorch
  // -4.4519, -3.4519, -2.4519, -1.4519, -0.4519

  CHECK(res(0,0) == doctest::Approx(-4.4519));
  CHECK(res(0,4) == doctest::Approx(-0.4519));

  auto sum = res.sum();
  auto topo = sum.getTopo();
  sum.backward(topo);
  // pytoch
  // tensor([[ 0.9417,  0.8416,  0.5694, -0.1706, -2.1820]])
  auto grads = x.getGrad();
  CHECK(grads(0,0) == doctest::Approx(0.9417));
  CHECK(grads(0,1) == doctest::Approx(0.841575));
  CHECK(grads(0,2) == doctest::Approx(0.569357));
  CHECK(grads(0,3) == doctest::Approx(-0.170608));
  CHECK(grads(0,4) == doctest::Approx(-2.18204));
  cout<<x.getGrad()<<endl;
}


TEST_CASE("negative tensor backwards")
{
  Tensor x(4,4);
  x(0,0) = 3;
  x(2,0) = 4;
  x(2,1) = 5;
  auto res = -x;

  CHECK(res(0,0) == -3);
  CHECK(res(2,0) == -4);
  CHECK(res(2,1) == -5);
  CHECK(res(3,3) == 0);
  
  auto topo = res.getTopo();
  res.backward(topo);

  CHECK(x.getGrad()(0,0) == -1);
  CHECK(x.getGrad()(1,1) == -1);
  CHECK(x.getGrad()(2,1) == -1);
  CHECK(x.getGrad()(2,3) == -1);
}

TEST_CASE("tensor cross entropy")
{
  Tensor in(4,1);
  in(0,0)=2; 
  in(1,0)=0; 
  in(2,0)=0; 
  in(3,0)=1;

  auto logscores = makeLogSoftMax(in);
  cout<<"tensor Logscores:\n"<<logscores<<endl;

  Tensor expected(1,4);
  //  expected.zero();
  expected(0,0)=1; 
  cout<<"product: "<<expected*logscores<<endl;
  auto loss = -(expected*logscores);
  cout<<"Loss: "<<loss<<endl;
  //  float oldloss = loss(0,0);
  
  auto topo = loss.getTopo();
  loss.backward(topo);

  cout<<"in.getGrad():\n"<<in.getGrad() << endl;
  //  cout << in(0,0).getGrad()<<endl;
  //  cout << in(1,0).getGrad()<<endl;


  /*
  -0.389704
0.0825946
0.0825946
 0.224515
  */
  CHECK(in.getGrad()(0,0) == doctest::Approx(-0.389704));
  CHECK(in.getGrad()(1,0) == doctest::Approx(0.0825946));
  CHECK(in.getGrad()(2,0) == doctest::Approx(0.0825946));
  CHECK(in.getGrad()(3,0) == doctest::Approx( 0.224515));

  
  // "learn a bit"
  in -= 0.2 * in.getGrad();
  loss.zerograd(topo);
  cout<<"New loss: "<<loss << endl;

}

TEST_CASE("tensor dot test")
{
  Tensor x(5, 5), y(5,5);
  int count = 1;
  for(unsigned int r = 0 ; r < 5; ++r) {
    for(unsigned int c = 0 ; c < 5; ++c) {
      x(r,c) = count++;
      y(r,c) = count % 2;
    }
  }
  auto ret = x.dot(y);
  cout << ret << endl;
  CHECK(ret(0,0) == 0);
  CHECK(ret(3,0) == 16);
  CHECK(ret(3,4) == 20);

  auto sum = ret.sum();
  cout<<"Sum: "<<sum;
  CHECK(sum(0,0) == 156);

  auto topo = sum.getTopo();
  sum.backward(topo);
  cout << x.getGrad() << endl;
  CHECK(x.getGrad() == y.d_imp->d_val);
}

TEST_CASE("tensor sum test")
{
  Tensor x(5,1);
  x.iota(1);

  Tensor m(1,1);
  m.identity(4);
  
  cout << "x:\n" << x << endl;
  cout << "x*m:\n" << (x*m) << endl;
  auto sum = (x*m).sum();
  cout<<sum<<endl;
  auto ssum = sum+sum;
  auto topo = ssum.getTopo();
  ssum.backward(topo);

  CHECK(x.getGrad()(0,0)==8);
  CHECK(x.getGrad()(1,0)==8);
  CHECK(x.getGrad()(2,0)==8);
  CHECK(x.getGrad()(3,0)==8);
  CHECK(x.getGrad()(4,0)==8);
  
}

TEST_CASE("tensor dot grad test")
{
  Tensor x(5,5), y(5,5), z(5,5);
  x.iota(1);  // 1  2  3  4  5
              // 6  7  8  9  10
  y.iota(5);  // 5  6  7  8  9
              // 10 11
  z.identity(2);

  Tensor res = (x.dot(y)*z).sum();
  cout << res << endl;

  auto topo = res.getTopo();
  res.backward(topo);

  CHECK(x.getGrad()(0,0)==10);
  CHECK(x.getGrad()(1,1)== 11*2);
  CHECK(y.getGrad()(1,1)== 7*2); 
  
}

TEST_CASE("tensor slice and dot test")
{
  Tensor x(5, 5);
  int count = 1;
  for(unsigned int r = 0 ; r < 5; ++r)
    for(unsigned int c = 0 ; c < 5; ++c)
      x(r,c) = count++;
  cout<<"x:\n"<<x<<endl;

  cout<<"makeSlice: "<<endl;
  auto s = x.makeSlice(0, 0, 3);
  cout << s << endl;

  auto sum = s.sum();
  cout<<"sum: "<<sum<<endl;

  auto topo= sum.getTopo();
  sum.backward(topo);

  cout<<"x.getGrad():\n"<<x.getGrad()<<endl;
  sum.zerograd(topo);
  
  Tensor w(3,3);
  w.randomize(0.1);

  
  sum = w.dot(s).sum();
  topo = sum.getTopo();
  sum.backward(topo);
  cout<<"New sum: "<<sum<<endl;
  cout<<"w:\n"<< w << endl;

  cout<<"x.getGrad():\n"<<x.getGrad()<<endl;
  
}


TEST_CASE("tensor flatten test")
{
  Tensor x(5,5);
  Tensor y(2,3);
  y.iota(75);
  
  int count=0;

  for(unsigned int c = 0 ; c < 5; ++c)
    for(unsigned int r = 0 ; r < 5; ++r)
      x(r,c) = count++;
  cout<<"x:\n"<<x<<endl;
  auto f = makeFlatten({x,y});
  cout << "f:\n"<< f << endl;


  Tensor m(1, 25+6);
  count=0;
  for(unsigned int c = 0; c < 25+6; ++c)
    m(0, c) = count++;

  cout<<"m:\n"<<m<<endl;
  cout<<"m*f:\n"<<(m*f)<<endl;
  auto s = (m*f).sum();

  cout << "sum: "<< s <<endl;
  auto topo = s.getTopo();
  s.backward(topo);

  count=0;
  for(unsigned int c = 0 ; c < 5; ++c) {
    for(unsigned int r = 0 ; r < 5; ++r) {
      CHECK(x.getGrad()(r,c) == count);
      ++count;
    }
  }
  cout<<"y.getGrad():\n"<<y.getGrad()<<endl;
}


TEST_CASE("tensor division test")
{
  Tensor x(5,5);
  int count=1;
  for(unsigned int c = 0 ; c < 5; ++c)
    for(unsigned int r = 0 ; r < 5; ++r)
      x(r,c) = count++;

  Tensor d(1,1);
  d(0,0) = 3.0;
  auto res = x/d;
  
  count = 1;
  for(unsigned int c = 0 ; c < 5; ++c) {
    for(unsigned int r = 0 ; r < 5; ++r) { 
      CHECK(res(r,c) == doctest::Approx(count/3.0));
      count++;
    }
  }
  auto s = res.sum();
  auto topo = s.getTopo();
  s.backward(topo);
  for(unsigned int c = 0 ; c < 5; ++c) {
    for(unsigned int r = 0 ; r < 5; ++r) { 
      CHECK(x.getGrad()(r,c) == doctest::Approx(1./3.0));
    }
  }
  
  
  
}


TEST_CASE("tensor subtract test")
{
  Tensor x(2,3), y(2,3);
  x(0,0)=1;   x(0,1)=2;   x(0,2)=1;

  y(0,0)=4;   y(0,1)=3;   y(0,2)=2;

  auto res = x - y;
  CHECK(res(0,0) == 1 - 4);
  CHECK(res(0,1) == 2 - 3);
  CHECK(res(0,2) == 1 - 2);
  CHECK(res(1,1) == 0);

  res = y - x;
  CHECK(res(0,0) == 3);
  CHECK(res(0,1) == 1);
  CHECK(res(0,2) == 1);
  CHECK(res(1,1) == 0);
}

TEST_CASE("tensor convo2d more") {
  Tensor in(4,4);
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;        in(0,3)=1;      
  in(1,0)=1;        in(1,1)=0;        in(1,2)=0;        in(1,3)=0;      
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;        in(2,3)=-5;   
  in(3,0)=1;        in(3,1)=7;        in(3,2)=-4;        in(3,3)=-3;      

  Tensor w(2,2);
  Tensor b(1,1);
  w(0,0) = 1;   w(0,1) = 2;
  w(1,0) = 3;   w(1,1) = 4;
  b.constant(1);
  auto m = in.makeConvo(2, w, b);
  CHECK(in.sum()(0,0) == -5);
  CHECK(m(0,0) == 1*1 + 2*2 + 3*1 +4*0 + 1);
  CHECK(m(0,1) == 1*2 + 2*3 + 3*0 +4*0 + 1);
  CHECK(m(2,1) == 1*0 + 2*-9 + 3*7 +4*-4 + 1);
  CHECK(m(2,2) == 1*-9 + 2*-5 + 3*-4 +4*-3 + 1);

  /*
  auto s = m.sum();
  s.backward();
  CHECK(in(0,0).getGrad() == 2);
  CHECK(w(0,0).getGrad() == 1);
  CHECK(w(0,1).getGrad() == 2);
  CHECK(b(0,0).getGrad() == 1);
  */
}

TEST_CASE("tensor convo2d backward") {
  Tensor input(6,6);
  input.iota(1);
  input(0,0)=11.0;
  cout<<"input:\n"<<input<<endl;
#if 0
  Tensor filter(3,3);
  filter(0,0) = 0.1107; filter(0,1)=  0.2178; filter(0,2)= -0.1075;
  filter(1,0)= 0.0788; filter(1,1)= 0.1591; filter(1,2)=  0.1667;
  filter(2,0)=-0.2994; filter(2,1)= 0.1177; filter(2,2)=  0.2621;
#endif
  Tensor filter(2,2);
  filter(0,0) = -0.0352; filter(0,1)= 0.0890;// filter(0,2)= -0.1075;
  filter(1,0)= 0.4843; filter(1,1)= 0.3177; //filter(1,2)=  0.1667;
  //  filter(2,0)=-0.2994; filter(2,1)= 0.1177; filter(2,2)=  0.2621;


  
  Tensor factor(6,6);
  factor.identity(2.0);
  
  /*
Parameter containing:
tensor([[[[ 0.1107,  0.2178, -0.1075],
          [ 0.0788,  0.1591,  0.1667],
          [-0.2994,  0.1177,  0.2621]]]], requires_grad=True)
Parameter containing:
tensor([0.1104], requires_grad=True)
   */
  
  cout<<"filter:\n"<<filter<<endl;

  Tensor bias(1,1);
  auto c = (input*factor).makeConvo(2, filter, bias);
  cout<<"c:\n"<<c<<endl;
  auto s=c.sum();
  auto topo=s.getTopo();
  s.backward(topo);

  cout << "input.getGrad():\n"<<input.getGrad()<<endl;
  cout << "filter.getGrad():\n"<<filter.getGrad()<<endl;
  cout << "bias.getGrad():\n"<<c.d_imp->d_convop.bias->d_grads<<endl;
  /*
  CHECK(input.getGrad()(0,0) == doctest::Approx(0.1107));
  CHECK(input.getGrad()(4,2) == doctest::Approx(0.485));
  CHECK(input.getGrad()(5,5) == doctest::Approx(0.2621));
  CHECK(filter.getGrad()(0,0)==184);
  //  CHECK(filter.getGrad()(0,2)==216); */
  CHECK(bias.getGrad()(0,0) == 25);
  /* These numbers match PyTorch
c:
 5.2356  5.9416  6.6476  7.3536
 9.4716 10.1776 10.8836 11.5896
13.7076 14.4136 15.1196 15.8256
17.9436 18.6496 19.3556 20.0616
input.getGrad():
 0.1107  0.3285   0.221   0.221  0.1103 -0.1075
 0.1895  0.5664  0.6256  0.6256  0.4361  0.0592
-0.1099  0.3847   0.706   0.706  0.8159  0.3213
-0.1099  0.3847   0.706   0.706  0.8159  0.3213
-0.2206  0.0562   0.485   0.485  0.7056  0.4288
-0.2994 -0.1817  0.0804  0.0804  0.3798  0.2621
filter.getGrad():
184 200 216
280 296 312
376 392 408
bias.getGrad():
16
  */
  
}


TEST_CASE("tensor max2d") {

  Tensor in(6,4);
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;        in(0,3)=1;      
  in(1,0)=0;        in(1,1)=0;        in(1,2)=0;        in(1,3)=0;      
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;        in(2,3)=-5;   
  in(3,0)=1;        in(3,1)=7;        in(3,2)=-4;        in(3,3)=-3;      
  in(4,0)=0;        in(4,1)=0;        in(4,2)=-9;        in(4,3)=-5;   
  in(5,0)=1;        in(5,1)=7;        in(5,2)=-4;        in(5,3)=-3;      

  Tensor f(4,4);
  f(0,0) = 3.5;
  f(1,1) = 3.5;
  f(2,2) = 3.5;
  f(3,3) = 3.5;

  
  auto m = (in*f).makeMax2d(2);

  CHECK(m(0,0) == 2*3.5);
  CHECK(m(0,1) == 3*3.5);
  CHECK(m(1,0) == 7*3.5);
  CHECK(m(1,1) == -3*3.5);

  auto s = m.sum();

  auto topo = s.getTopo();
  s.backward(topo);
  CHECK(in.getGrad()(0,0) == 0);
  CHECK(in.getGrad()(0,1) == 3.5);

  CHECK(in.getGrad()(2,0) == 0);
  CHECK(in.getGrad()(3,1) == 3.5);
  
  CHECK(in.getGrad()(5,3) == 3.5);
  CHECK(in.getGrad()(4,3) == 0);
}


TEST_CASE("max2d padding") {

  Tensor in(3,3);
  in(0,0)=1;        in(0,1)=2;        in(0,2)=3;       
  in(1,0)=0;        in(1,1)=0;        in(1,2)=0;       
  in(2,0)=0;        in(2,1)=0;        in(2,2)=-9;      

  auto m = in.makeMax2d(2); // 2*2

  CHECK(m(0,0) == 2);
  CHECK(m(0,1) == 3);
  CHECK(m(1,0) == 0);
  CHECK(m(1,1) == -9);

  auto s = m.sum();
  auto topo = s.getTopo();
  s.backward(topo);
  CHECK(in.getGrad()(0,0) == 0);
  CHECK(in.getGrad()(0,1) == 1);
}


TEST_CASE("tensor save and load")
{
  ostringstream str;
  Tensor f(20, 25);
  f.randomize(1.0);

  f.save(str);

  {
    ofstream ofs("tensor.arr");
    f.save(ofs);
  }
  unlink("tensor.arr");
  
  string saved = str.str();

  Tensor restored(20,25);
  restored.zero();
  istringstream istr(saved);

  restored.load(istr);

  auto diff = restored - f;
  CHECK(diff.sum()(0,0) ==  0);
  
}

TEST_CASE("relu")
{
  Tensor x(1,1);
  x(0,0) = 0;
  auto relu=makeFunction<ReluFunc>(x);
  CHECK(relu(0,0) == 0);
  auto topo = relu.getTopo();
  relu.zerograd(topo);
  x(0,0) = 12;
  CHECK(relu(0,0) == 12);
  relu.zerograd(topo);
  x(0,0) = -12;
  CHECK(relu(0,0) == 0);

}

TEST_CASE("gelu")
{
  Tensor x(1,1);
  x(0,0) = 0;
  auto gelu=makeFunction<GeluFunc>(x);
  CHECK(gelu(0,0) == 0);
  
  auto topo = gelu.getTopo();

  gelu.zerograd(topo);
  x(0,0) = 12;
  CHECK(gelu(0,0) == 12);
  
  gelu.zerograd(topo);
  x(0,0) = -50;
  CHECK(gelu(0,0) == doctest::Approx(0.0));

  gelu.zerograd(topo);
  x(0,0) = 50;
  CHECK(gelu(0,0) == doctest::Approx(50.0));


  gelu.zerograd(topo);
  x(0,0) = 0;
  gelu.backward(topo);
  CHECK(x.getGrad()(0,0) == doctest::Approx(0.5));

  gelu.zerograd(topo);
  x(0,0) = 50;
  gelu.backward(topo);
  CHECK(x.getGrad()(0,0) == doctest::Approx(1.0));

  gelu.zerograd(topo);
  x(0,0) = -50;
  gelu.backward(topo);
  CHECK(x.getGrad()(0,0) == doctest::Approx(0.0));

  gelu.zerograd(topo);
  x(0,0) = 1;
  gelu.backward(topo);
  CHECK(x.getGrad()(0,0) == doctest::Approx(1.0833));

  gelu.zerograd(topo);
  x(0,0) = -1;
  gelu.backward(topo);
  CHECK(x.getGrad()(0,0) == doctest::Approx(-0.0833155));

  
  
}
