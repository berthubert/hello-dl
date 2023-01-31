#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include "tensor-layers.hh"
#include "convo-alphabet.hh"
#include "vizi.hh"
#include "mnistreader.hh"
using namespace std;


int main(int argc, char** argv)
{
  if(argc < 3) {
    cout<<"Syntax: imagine fromletter toletter modelname"<<endl;
    return EXIT_FAILURE;
  }

  int fromlabel = *argv[1]-'a' + 1;
  int tolabel = *argv[2]-'a' + 1;
  
  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;
  cout<<"Loading model state from file '"<<argv[3]<<"'\n";
  loadModelState(s, argv[3]);
  m.init(s, true);

  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");
  m.img.zero();
  for(int n = 0 ; n < mntest.num(); ++n) {
    if(mntest.getLabel(n) == fromlabel) {
      mntest.pushImage(n, m.img);
      break;
    }
  }
  /*
  m.img.randomize(1.0);
  m.img.d_imp->d_val = m.img.d_imp -> d_val.unaryExpr([](float v) { return fabs(v); });
  */
  
  m.img.normalize(0.172575, 0.25);

  auto specscore = m.scores.makeSlice(tolabel, 0, 1, 1);
  auto topo = specscore.getTopo();
  for(unsigned int tries = 0 ; tries < 10000; ++tries) {
    cout<<specscore<<endl;
    specscore.backward(topo);
    auto grad = m.img.getGrad();
    grad *= 0.2;
    m.img.d_imp->d_val += grad;

    if(!(tries %4))
      saveTensor(m.img, "imagine-"+to_string(tries)+".png", 252, true);
    specscore.zerograd(topo);
  }
  
}
