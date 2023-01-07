#include "tracked.hh"
#include "array.hh"
#include "ext/doctest.h"

using namespace std;

TEST_CASE("basic worker test") {
  TrackedFloat a(2.0), b(3.0), c(1.0);
  auto d = a * b + c;
  a.setVariable();
  float res = d.getVal();
  cout<<"res is: "<<res<<endl;

  auto topo = d.getTopo();

  auto w = d.getWork<float>(topo);
  for(const auto& g : w.dyns) {
    cout<<"Setting variable ["<<g.first<<"] to "<<g.second->d_val<<"\n";
    w.work[g.first].ourval = g.second->d_val;
  }


  int num=0;
  for(const auto& item : w.work) {
    cout<< "["<<num<<"] "<<item <<endl;
    num++;
  }

  cout<<"worker res: "<<w.getResult()<<endl;
  a.impl->d_val = 1.0;
  d.backward();
  d.zeroGrad();
  
  cout<<"New value: "<<d.getVal() << endl;
  w.syncVariable();
  cout<<"New worker val after resync: "<<w.getResult()<<endl;
  CHECK(w.getResult() == d.getVal());
}

TEST_CASE("grad worker test") {
  cout<<"START!"<<endl;
  TrackedFloat a(2.0), b(3.0), c(1.0);
  auto d = a * a * a + b * b;
  a.setVariable();
  b.needsGrad();
  c.needsGrad();
  
  float res = d.getVal();
  d.backward();

  cout<<"res is: "<<res<<", a grad is: " << a.getGrad()<<endl;
  cout<<"b grad is: " << b.getGrad()<<endl;
  cout<<"c grad is: " << c.getGrad()<<endl;

  auto topo = d.getTopo();
  auto w = d.getWork<float>(topo);
  float wresult = w.getResult();
  cout<<"Worker result: "<<wresult<<endl;

  CHECK(res == wresult);  
  w.backward();
  // syncback needgrads

  int num=0;
  for(const auto& item : w.work) {
    cout<< "["<<num<<"] "<<item <<endl;
    num++;
  }
}

TEST_CASE("array worker test") {
  cout<<"Array worker test"<<endl;
  NNArray<float, 2, 2> img, weights;
  img.randomize();
  weights.randomize();
  weights.needsGrad();

  auto c = weights * img;
  auto sumval = c.sum();
  float res = sumval.getVal();
  sumval.backward();
  img.zeroGrad(); // backward populates grads, when it shouldn't
  cout<<"res is: "<<res<<", grad of weights(0,0): "<<weights(0,0).getGrad()<<", val of weights(0,0) "<<weights(0,0).getVal()<<endl;

  auto topo = sumval.getTopo();
  auto wgrad = weights.getGrad();
  
  auto w = sumval.getWork<float>(topo);
  CHECK(w.dyns.size() == 8);
  cout<<"worker res: "<<w.getResult()<<endl;

  CHECK(w.getResult() == sumval.getVal());
  
  cout<<"a(0,0).getGrad() "<< weights(0,0).getGrad()<<endl;
  w.backward();
  int num=0;
  for(const auto& item : w.work) {
    cout<< "["<<num<<"] "<<item << " grad " <<w.grads[num]<<endl;
    num++;
  }
  w.syncGrad();

  auto wgrad2 = weights.getGrad();
  CHECK(wgrad2 == wgrad);

  SArray<float, 2, 2> zero;
  zero.setZero();
  auto imggrad = img.getGrad();
  CHECK(imggrad == zero);
  
  cout<<"sizeof: "<< sizeof(w.work[0]) <<endl;
}


TEST_CASE("array worker projection test") {

  NNArray<float, 2, 5> a;
  NNArray<float, 5, 2> b;
  a.randomize();
  b.randomize();
  TrackedFloat res = (a * b).sum();
  float nresult = res.getVal();

  auto topo = res.getTopo();
  auto w=res.getWork<float>(topo);

  auto proj = makeProj(a, w);
  
  float wresult = w.getResult();
  CHECK(nresult == wresult);

  a(0,0) = a(0,0).getVal() + 1;
  res.zeroGrad(); // flushes cache

  projForward(proj, a, w);
  cout<<"nresult: "<<nresult<<endl;
  CHECK(res.getVal() == w.getResult());
}

TEST_CASE("array worker AVX2 test") {

  NNArray<float, 2, 5> a;
  NNArray<float, 5, 2> b;
  a.randomize();
  b.randomize();
  TrackedFloat res = (a * b).sum();
  float nresult = res.getVal();

  auto topo = res.getTopo();
  auto w=res.getWork<fvector<8>>(topo);

  auto proj = makeProj(a, w);
  
  auto wresult = w.getResult();
  CHECK(nresult == wresult.v[0]);
  CHECK(nresult == wresult.v[1]);
  CHECK(nresult == wresult.v[2]);
  CHECK(nresult == wresult.v[7]);

  a(0,0) = a(0,0).getVal() + 1;
  res.zeroGrad(); // flushes cache

  projForward(proj, a, w);
  cout<<"nresult: "<<nresult<<endl;
  CHECK(res.getVal() == w.getResult().v[3]);

  
}

TEST_CASE("worker AVX2 test") {

  NNArray<float, 1, 5> a;
  NNArray<float, 1, 5> b;
  a.zero();
  b.zero();
  a.setVariable();
  b.setVariable();
  
  auto c = a+b;
  auto res = c(0,0);
  float nresult = res.getVal();

  CHECK(nresult == 0.0);
  
  auto topo = res.getTopo();

  auto w = res.getWork<float>(topo);
  auto aproj = makeProj(a, w);
  auto bproj = makeProj(b, w);
  
  auto w8=res.getWork<fvector<8>>(topo);
  
  auto wresult = w8.getResult();
  CHECK(wresult.v[0] == 0 );
  CHECK(wresult.v[1] == 0 );
  CHECK(wresult.v[2] == 0 );
  CHECK(wresult.v[3] == 0 );
    

  NNArray<fvector<8>, 1, 5> a8;
  a8(0,0) = 0;
  a8(0,0).impl->d_val.v[0]=1;
  a8(0,0).impl->d_val.v[1]=2;
  a8(0,0).impl->d_val.v[2]=4;


  NNArray<fvector<8>, 1, 5> b8;
  b8(0,0) = 0;
  b8(0,0).impl->d_val.v[0]=4;
  b8(0,0).impl->d_val.v[1]=5;
  b8(0,0).impl->d_val.v[2]=6;
  
  
  res.zeroGrad(); // flushes cache

  projForward(aproj, a8, w8);
  projForward(bproj, b8, w8);
  
  wresult = w8.getResult();
  CHECK(wresult.v[0] == 5);
  CHECK(wresult.v[1] == 7);
  CHECK(wresult.v[2] == 10);

  
}
