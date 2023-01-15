#include "tensor-layers.hh"
#include "textsupport.hh"
#include "misc.hh"
using namespace std;

// hidden state=> dense linear => output x

// x is input
// h_t, h_{t-1} = hidden state

// gate_{reset} = \sigma(W_{input_{reset}} \cdot x_t + W_{hidden_{reset}} \cdot h_{t-1})

// W_input_reset  - ^^ normal matrix products
// W_input_hidden

// pytorch:

// r_t​ = σ(W_{ir} ​x_t + b_{ir}​+W_{hr}​h_{t−1}​ +b_{hr}​)            // reset gate
// z_t​ = σ(W_{iz} ​x_t ​+ b_{iz} ​+W_{hz}​h_{t−1}​ + b_{hz}​)          // update
// n_t​ = tanh(W_{in}​x_t​+b_{in}​+ r_t​*(W_{hn} ​h_{t−1}​ + b_{hn}​))   // "new" - * is dotproduct
// h_t​=(1−z_t​)*n_t​+z_t​*h_{t−1)​}                                  // new h

// the hidden state is also the output, which needs linear combination to turn into input size again
// https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

template<typename T, unsigned int IN, unsigned int HIDDEN>
struct GRULayer 
{
  Tensor<T> d_w_ir{HIDDEN, IN}; // reset
  Tensor<T> d_w_iz{HIDDEN, IN}; // update
  Tensor<T> d_w_in{HIDDEN, IN}; // new

  Tensor<T> d_w_hr{HIDDEN, HIDDEN};  // hidden reset
  Tensor<T> d_w_hz{HIDDEN, HIDDEN}; // hidden update
  Tensor<T> d_w_hn{HIDDEN, HIDDEN}; // hidden "new"

  Tensor<T> d_prevh{HIDDEN, 1};


  GRULayer()
  {
    randomize();
  }

  // https://blog.floydhub.com/gru-with-pytorch/
  //  https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18
  // these appear to be slightly different
  auto forward(const Tensor<T>& xt)
  {
    auto r_t = makeFunction<SigmoidFunc>(d_w_ir * xt + d_w_hr * d_prevh); // reset gate
    auto z_t = makeFunction<SigmoidFunc>(d_w_iz * xt + d_w_hz * d_prevh);
    auto n_t = makeFunction<TanhFunc>(d_w_in * xt + r_t.dot(d_w_hn *d_prevh));
    Tensor<T> one(256, 1);  // XXX this is a SUPER wart
    // the problem is we have no support for 1 - Tensor() kind of operations
    // and we also don't know the size of z_t here yet
    for(unsigned int r=0 ; r < one.getRows(); ++r)
      for(unsigned int c=0 ; c < one.getCols(); ++c)
        one(r,c)=1;
    auto h_t = (one - z_t).dot(n_t) + z_t.dot(d_prevh);
    d_prevh = h_t; // "this is where the magic happens"
    return h_t;
  }

  void learn(float lr) 
  {
    { auto grad1 = d_w_ir.getAccumGrad(); grad1 *= lr; d_w_ir -= grad1;  }
    { auto grad1 = d_w_iz.getAccumGrad(); grad1 *= lr; d_w_iz -= grad1;  }
    { auto grad1 = d_w_in.getAccumGrad(); grad1 *= lr; d_w_in -= grad1;  }
    
    { auto grad1 = d_w_hr.getAccumGrad(); grad1 *= lr; d_w_hr -= grad1;  }
    { auto grad1 = d_w_hz.getAccumGrad(); grad1 *= lr; d_w_hz -= grad1;  }
    { auto grad1 = d_w_hn.getAccumGrad(); grad1 *= lr; d_w_hn -= grad1;  }
  }

  void randomize() // "Xavier initialization"  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  {
    d_w_ir.randomize(1.0/sqrt(HIDDEN));
    d_w_iz.randomize(1.0/sqrt(HIDDEN));
    d_w_in.randomize(1.0/sqrt(HIDDEN));

    d_w_hr.randomize(1.0/sqrt(HIDDEN));
    d_w_hz.randomize(1.0/sqrt(HIDDEN));
    d_w_hn.randomize(1.0/sqrt(HIDDEN));
    d_prevh.zero();
  }
};



template<typename T>
struct GRUModel
{
  struct State
  {
    GRULayer<T, 98, 256> gm1;
    GRULayer<T, 256, 256> gm2;
    Linear<T, 256, 98> fc;
    void zeroGrad()
    {
      gm1.zeroGrad();
      gm2.zeroGrad();
      fc.zeroGrad();
    }
    
  };
  vector<Tensor<T>> invec;   // 98  1
  vector<Tensor<T>> expvec;  // 1  98
  vector<Tensor<T>> scorevec;// 98  1

  Tensor<T> totloss;
  
  void unroll(State& s, unsigned int choplen)
  {
    cout<<"Unrolling the GRU";
    totloss= Tensor<T>(1, 1);
    totloss(0,0) = 0.0;
    Tensor<T> choplent(1,1);
    choplent(0,0) = choplen;

    for(size_t i = 0 ; i < choplen; ++i) {
      cout<<"."; cout.flush();
      Tensor<T> in(98, 1);
      Tensor<T> expected(1,98);
      in.zero();  
      expected.zero(); 
      
      invec.push_back(in);
      expvec.push_back(expected);
      auto res1 = s.fc.forward(s.gm2.forward(s.gm1.forward(in)));
      auto score = makeLogSoftMax(res1);
      scorevec.push_back(score);
      auto loss =  -(expected*score);
      totloss = totloss + loss;
    }
    totloss = totloss/choplent;
    cout<<"\n";
  }

};


int main(int argc, char **argv)
{
  BiMapper bm("corpus.txt", 98);
  constexpr int choplen= 75;
  vector<string> sentences=textChopper("corpus.txt", choplen, 10);
  cout<<"Got "<<sentences.size()<<" sentences"<<endl;

  GRUModel<float> grum;
  GRUModel<float>::State s;

  grum.unroll(s, choplen - 1);

  constexpr int batchsize = 100;

  cout<<"\nDoing topo.."; cout.flush();
  auto topo = grum.totloss.getTopo();
  cout<<" "<< topo.size() <<" entries"<<endl;

  cout<<"Starting the work"<<endl;
  for(;;) {
    Batcher batcher(sentences.size());
    for(;;) { // the batch loop
      auto batch = batcher.getBatch(batchsize);
      if(batch.empty())
        break;
      float batchloss = 0;
      
      grum.totloss.zeroAccumGrads(topo);
      for(const auto& idx : batch) {
        // wipe out history
        s.gm1.d_prevh = Tensor<float> (s.gm1.d_prevh.getRows(), s.gm1.d_prevh.getCols()); 
        s.gm2.d_prevh = Tensor<float> (s.gm2.d_prevh.getRows(), s.gm2.d_prevh.getCols()); 
        string input = sentences[idx];
        std::string output;
        
        for(size_t pos = 0 ; pos < input.size() - 1; ++pos) {
          cout<<input.at(pos);
          grum.invec[pos].zero();
          grum.invec[pos](bm.c2i(input.at(pos)), 0) = 1.0;
          grum.expvec[pos].zero();
          grum.expvec[pos](0, bm.c2i(input.at(pos+1)))= 1.0;
        }
        cout<<"\n";
        batchloss += grum.totloss(0,0); // triggers the calculation
        for(size_t pos = 0 ; pos < choplen - 1; ++pos) {
          cout<< bm.i2c(grum.scorevec[pos].maxValueIndexOfColumn(0));
        }
        cout<<"\n";
        grum.totloss.backward(topo);
        grum.totloss.accumGrads(topo);
        grum.totloss.zerograd(topo);
      }
      
      batchloss /= batchsize;
      
      float lr=0.01/batchsize;
      cout<<"Average batch loss: "<<batchloss<<endl;
      s.gm1.learn(lr);
      s.gm2.learn(lr);
      s.fc.learn(lr);
      cout<<"\n\n";
      //    saveModelState(s, "los-worker.state");
    }
  }
}

