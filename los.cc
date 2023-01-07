
#include <malloc.h>
#include <fenv.h>
#include <random>
#include <chrono>
#include <fstream>
#include "fvector.hh"
#include <iostream>
#include <array>
#include "gru.hh"
#include "textsupport.hh"

std::ofstream g_tree;//("./tree.part");
#include "tracked.hh"
#include "misc.hh"

#include <initializer_list>
using namespace std;

template<typename T>
struct GRUModel
{
  struct State
  {
    GRULayer<T, 98, 250> gm1;
    GRULayer<T, 250, 250> gm2;
    Linear<T, 250, 98> fc;
    void zeroGrad()
    {
      gm1.zeroGrad();
      gm2.zeroGrad();
      fc.zeroGrad();
    }
    
    void save(std::ostream& out) const
    {
      gm1.save(out);
      gm2.save(out);
      fc.save(out);
    }

    void load(std::istream& in) 
    {
      gm1.load(in);
      gm2.load(in);
      fc.load(in);
    }

    template<typename W>
    void makeProj(const W& w)
    {
      gm1.makeProj(w);
      gm2.makeProj(w);
      fc.makeProj(w);
    }

    template<typename W>
    void projForward(W& w) const
    {
      gm1.projForward(w);
      gm2.projForward(w);
      fc.projForward(w);
    }
    template<typename W>
    void projBackGrad(const W& w) 
    {
      gm1.projBackGrad(w);
      gm2.projBackGrad(w);
      fc.projBackGrad(w);
    }
  };
  vector<NNArray<T, 98, 1>> invec;
  vector<NNArray<T, 1, 98>> expvec;
  vector<NNArray<T, 98, 1>> scorevec;

  TrackedNumber<T> totloss;
  
  void unroll(State& s, unsigned int choplen)
  {
    cout<<"Unrolling the GRU";
    totloss = TrackedNumber<T>(0.0);
    for(size_t i = 0 ; i < choplen; ++i) {
      cout<<"."; cout.flush();
      NNArray<T, 98, 1> in;
      NNArray<T, 1, 98> expected;
      in.zero();  
      expected.zero(); 
      
      invec.push_back(in);
      expvec.push_back(expected);
      auto res1 = s.fc.forward(s.gm2.forward(s.gm1.forward(in)));
      auto score = res1.logSoftMax();
      scorevec.push_back(score);
      auto loss = TrackedNumber<T>(0.0) - (expected*score)(0,0);
      totloss = totloss + loss;
    }
    totloss = totloss/TrackedNumber<T>(choplen);
    cout<<"\n";
  }

};


int main(int argc, char **argv)
{
  BiMapper bm("corpus.txt", 98);
  constexpr int choplen= 75;
  vector<string> sentences=textChopper("corpus.txt", choplen, 10);
  cout<<"Got "<<sentences.size()<<" sentences"<<endl;
  Batcher batcher(sentences.size());

  auto grum = make_unique<GRUModel<float>>();
  GRUModel<float>::State s;

  if(argc > 1) {
    cout<<"Loading model state from "<<argv[1]<<endl;
    loadModelState(s, argv[1]);
  }
  grum->unroll(s, choplen - 1);

  constexpr int batchsize = 64;

  cout<<"\nDoing topo.."; cout.flush();
  auto topo = grum->totloss.getTopo();
  cout<<" "<< topo.size() <<" entries"<<endl;

  vector<std::array<unsigned int, 98>> invecprojarray, expvecprojarray, scorevecprojarray;
  cout<<"Making float worker"<<endl;
  for(auto& x : grum->scorevec) {
    x.setVariable();
  }
  for(auto& x : grum->invec) {
    x.setVariable();
  }
  for(auto& x : grum->expvec) {
    x.setVariable();
  }

  auto w = grum->totloss.getWork<float>(topo);
  
  for(const auto& x : grum->scorevec) {
    scorevecprojarray.push_back(makeProj(x, w));
  }
  
  for(const auto& x : grum->invec)
    invecprojarray.push_back(makeProj(x, w));
  
  for(const auto& x : grum->expvec)
    expvecprojarray.push_back(makeProj(x, w));
  
  s.makeProj(w);
  
  cout<<"Resetting GRU"<<endl;
  cout<<TrackedNumberImp<float>::getCount()<<" instances before clean"<<endl;
    
  grum.reset();
  cout<<TrackedNumberImp<float>::getCount()<<" instances after clean"<<endl;
    
  vector<NNArray<fvector<8>, 98, 1>> invec(choplen);
  vector<NNArray<fvector<8>, 1, 98>> expvec(choplen);
  vector<NNArray<fvector<8>, 98, 1>> scorevec(choplen);

  cout<<"Making AVX2 worker"<<endl;
  auto w8 = w.convert<fvector<8>>(); 

  cout<<"Starting the work"<<endl;

  for(;;) { // the batch loop
    cout<<TrackedNumberImp<float>::getCount()<<" instances"<<endl;
    double batchloss = 0;
    w8.zeroGrad();
    s.zeroGrad();
    s.projForward(w8);
    for(unsigned int minibatchno = 0 ; minibatchno < batchsize/8; ++minibatchno) {
      s.gm1.d_prevh.reset();
      s.gm2.d_prevh.reset();

      auto minibatch = batcher.getBatch(8);
      if(minibatch.size() != 8)
        goto batcherEmpty;

      for(size_t pos = 0 ; pos < choplen - 1; ++pos) {
        invec[pos].zero();
        expvec[pos].zero();
      }

      int avxindex =0;
      for(const auto& idx : minibatch) {
        string input = sentences[idx];
        std::string output;

        for(size_t pos = 0 ; pos < input.size() - 1; ++pos) {
          cout<<input.at(pos);
          invec[pos](bm.c2i(input.at(pos)), 0).impl->d_val.v[avxindex] = 1.0;
          expvec[pos](0, bm.c2i(input.at(pos+1))).impl->d_val.v[avxindex] = 1.0;
        }
        cout<<endl;
        avxindex++;
      }

      for(unsigned int projpos = 0; projpos < scorevecprojarray.size(); ++projpos) {
        projForward(invecprojarray[projpos], invec[projpos], w8);
        projForward(expvecprojarray[projpos], expvec[projpos], w8);
      }

      
      auto numloss = w8.getResult().sum()/8; // this triggers the whole calculation

      for(unsigned int projpos = 0; projpos < scorevecprojarray.size(); ++projpos)
        projBack(scorevecprojarray[projpos], w8, scorevec[projpos]);

      
      for(int t =0 ; t< 8; ++t) {
        for(size_t pos = 0 ; pos < choplen - 1; ++pos) {
          cout<< bm.i2c(scorevec[pos].getUnparallel(t).maxValueIndexOfColumn(0));
        }
        cout<<"\n";
      }
      batchloss += numloss;
      cout<<"\naverage loss: "<<numloss<<endl;
      s.projBackGrad(w8);
    }
    // done with all the minibatches, have a batch
    


    batchloss /= batchsize;
    
    float lr=0.01/batchsize;
    cout<<"Average batch loss: "<<batchloss<<endl;
    s.gm1.learn(lr);
    s.gm2.learn(lr);
    s.fc.learn(lr);
    //    cout<<"\n\n";
    saveModelState(s, "los-worker.state");
  }
 batcherEmpty:;
}

