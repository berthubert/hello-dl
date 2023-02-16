#include "gru-layer.hh"
#include "textsupport.hh"
#include "misc.hh"

using namespace std;

template<typename T, unsigned int SYMBOLS>
struct GRUModel 
{
  struct State : ModelState<T>
  {
    //          IN   HIDDEN
    GRULayer<T, SYMBOLS, 256> gm1;
    GRULayer<T, 256, 256> gm2;
    GRULayer<T, 256, 256> gm3;
    Linear<T, 256, SYMBOLS> fc;

    State()
    {
      this->d_members = {{&gm1, "gm1"}, {&gm2, "gm2"}, {&gm3, "gm3"}, {&fc, "fc"}};
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
      auto res1 = s.fc.forward(s.gm3.forward(s.gm2.forward(s.gm1.forward(in))));
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
      ref.push_back(exp(3.0*in(r,c)));

  std::discrete_distribution<> d(ref.begin(), ref.end());
  int pick = d(gen);

  return pick;
}

int main(int argc, char **argv)
{
  constexpr int tokens = 95;
  BiMapper bm("corpus.txt",tokens);
  constexpr int choplen= 75;
  vector<string> sentences=textChopper("corpus.txt", choplen, 10);
  cout<<"Got "<<sentences.size()<<" sentences"<<endl;

  GRUModel<float, tokens> grum;
  GRUModel<float, tokens>::State s;

  vector<char> charset;
  for(int n=0; n < tokens;++n) {
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
    int len = 1000;
    grum.unroll(s, len);
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
    for(; n < len ; ++n) {
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
        s.gm3.d_origprevh.zero();

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
      s.learnAdam(1.0/batchsize, batchno, 0.001); // 0.001 is start
      batchno++;
      cout<<"\n\n";
      saveModelState(s, "tensor-gru.state");
    }
  }
}

