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
struct GRULayer : TensorLayer<T>
{
  Tensor<T> d_w_ir{HIDDEN, IN}; // reset
  Tensor<T> d_w_iz{HIDDEN, IN}; // update
  Tensor<T> d_w_in{HIDDEN, IN}; // new

  Tensor<T> d_w_hr{HIDDEN, HIDDEN};  // hidden reset
  Tensor<T> d_w_hz{HIDDEN, HIDDEN}; // hidden update
  Tensor<T> d_w_hn{HIDDEN, HIDDEN}; // hidden "new"

  Tensor<T> d_origprevh{HIDDEN, 1};
  Tensor<T> d_prevh{HIDDEN, 1};

  GRULayer()
  {
    this->d_params={
        {&d_w_ir, "w_ir"},         {&d_w_iz, "w_iz"},         {&d_w_in, "w_in"},
        {&d_w_hr, "w_hr"},         {&d_w_hz, "w_hz"},         {&d_w_hn, "w_hn"}};
    randomize();
    Tensor one(HIDDEN, HIDDEN);
    one.identity(1.0);
    d_prevh = one*d_origprevh;
  }

  // https://blog.floydhub.com/gru-with-pytorch/
  //  https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18
  // these appear to be slightly different
  auto forward(const Tensor<T>& xt)
  {
    auto r_t = makeFunction<SigmoidFunc>(d_w_ir * xt + d_w_hr * d_prevh); // reset gate
    auto z_t = makeFunction<SigmoidFunc>(d_w_iz * xt + d_w_hz * d_prevh);
    // z_t dimensions: rows from d_w_iz, columns from xt -> HIDDEN,IN
    auto n_t = makeFunction<TanhFunc>(d_w_in * xt + r_t.dot(d_w_hn *d_prevh));

    Tensor<T> one(HIDDEN, 1);  // XXX this is a SUPER wart
    // the problem is we have no support for 1 - Tensor() kind of operations, so we need to make an appropriately sized 'one'
    for(unsigned int r=0 ; r < one.getRows(); ++r)
      for(unsigned int c=0 ; c < one.getCols(); ++c)
        one(r,c)=1;
    
    auto h_t = (one - z_t).dot(n_t) + z_t.dot(d_prevh);
    d_prevh = h_t; // "this is where the magic happens"
    return h_t;
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

template<typename T, unsigned int SYMBOLS>
struct GRUModel 
{
  struct State : ModelState<T>
  {
    //          IN   HIDDEN
    GRULayer<T, SYMBOLS, 256> gm1;
    GRULayer<T, 256, 256> gm2;
    Linear<T, 256, SYMBOLS> fc;

    State()
    {
      this->d_members = {{&gm1, "gm1"}, {&gm2, "gm2"}, {&fc, "fc"}};
    }
    
  };
  vector<Tensor<T>> invec;   // SYMBOLS  1
  vector<Tensor<T>> expvec;  // 1  SYMBOLS
  vector<Tensor<T>> scorevec;// SYMBOLS  1

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
      Tensor<T> in(SYMBOLS, 1);
      Tensor<T> expected(1,SYMBOLS);
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
    totloss = totloss/choplent; // otherwise the gradient is too high
    cout<<"\n";
  }
};

int sampleWithTemperature(const Tensor<float>::EigenMatrix& in, float t [[maybe_unused]] = 0)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  vector<double> ref;
  for(int r = 0; r< in.rows();++r)
    for(int c = 0; c< in.cols();++c)
      ref.push_back(exp(3.5*in(r,c)));

  std::discrete_distribution<> d(ref.begin(), ref.end());
  int pick = d(gen);

  /*
  double sum=0;
  for(int c = 0 ; c< ref.size() ; ++c) {
    cout<<ref[c];
    sum += ref[c];
    if(c==pick)
      cout<<"**";
    cout<<" ";
  }
  cout<<": sum = "<<sum<<endl;
  */
  return pick;
}

int main(int argc, char **argv)
{
  BiMapper bm("corpus.txt", 75);
  constexpr int choplen= 75;
  vector<string> sentences=textChopper("corpus.txt", choplen, 10);
  cout<<"Got "<<sentences.size()<<" sentences"<<endl;

  GRUModel<float, 75> grum;
  GRUModel<float, 75>::State s;

  vector<char> charset;
  for(int n=0; n < 75;++n) {
    cout<<bm.i2c(n);
    charset.push_back(bm.i2c(n));
  }
  cout<<endl;
  sort(charset.begin(), charset.end());
  for(const auto& c: charset) {
    cout<<c;
  }
  cout<<endl;
  if(argc > 2) {
    grum.unroll(s, 300);
    cout<<"Loading state from "<<argv[1]<<endl;
    loadModelState(s, argv[1]);
    auto topo = grum.scorevec[0].getTopo();
    int res;
    int n;
    for(n=0; n < strlen(argv[2]); ++n) {
      grum.invec[n].oneHotRow(bm.c2i(argv[2][n])); // "prompt"
      grum.scorevec[n](0,0);

      res = grum.scorevec[n].maxValueIndexOfColumn(0);
      cout << bm.i2c(res);
    }
    cout<<endl;
    for(; n < 300 ; ++n) {
      grum.invec[n].oneHotRow(res);      
      grum.scorevec[n](0,0);
      res = sampleWithTemperature(grum.scorevec[n].d_imp->d_val);
      //      res = grum.scorevec[n].maxValueIndexOfColumn(0);
      cout << bm.i2c(res);
    }
    cout<<endl;
    return 0;
  }
  
  grum.unroll(s, choplen - 1);

  // is now randomized
  if(argc>1) {
    cout<<"Loading state from "<<argv[1]<<endl;
    loadModelState(s, argv[1]);
  }
  
  constexpr int batchsize = 100;

  auto topo = grum.totloss.getTopo();
  cout<<"Topo has "<< topo.size() <<" entries"<<endl;
  unsigned int batchno = 0;
  cout<<"Starting the work"<<endl;
  for(;;) {
    Batcher batcher(sentences.size());
    for(;;) { // the batch loop
      auto batch = batcher.getBatch(batchsize);
      if(batch.empty())
        break;
      float batchloss = 0;
      ++batchno;
      grum.totloss.zeroAccumGrads(topo);
      for(const auto& idx : batch) {
        s.gm1.d_origprevh.zero();
        s.gm2.d_origprevh.zero();

        //        cout << "s.gm1.d_prevh:\n"<< s.gm1.d_prevh<<endl;
        //cout << "s.gm1.d_origprevh:\n"<< s.gm1.d_origprevh<<endl;
        
        string input = sentences[idx];
        std::string output;

        cout<<"IN:  ";
        for(size_t pos = 0 ; pos < input.size() - 1; ++pos) {
          cout<<input.at(pos);
          grum.invec[pos].zero();
          grum.invec[pos](bm.c2i(input.at(pos)), 0) = 1.0;
          grum.expvec[pos].zero();
          grum.expvec[pos](0, bm.c2i(input.at(pos+1)))= 1.0;
        }
        cout<<"\nOUT:  ";
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
      
      float lr=0.001/batchsize;
      cout<<"Average batch loss: "<<batchloss<<endl;
      s.learnAdam(1.0/batchsize, batchno, 0.001);
      batchno++;
      cout<<"\n\n";
      saveModelState(s, "tensor-gru.state");
    }
  }
}

