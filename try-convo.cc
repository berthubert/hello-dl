#include <iostream>
#include <memory>
#include "mnistreader.hh"
#include "misc.hh"
#include <string.h>
#include <fenv.h>
#include "tensor-layers.hh"
#include "vizi.hh"

using namespace std;

int main(int argc, char** argv)
{
  if(argc <  2) {
    cerr<<"Syntax: try-convo index"<<endl;
    return 0;
  }

  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");
  cout<<"Have "<<mntest.num()<<" validation images"<<endl;

  Tensor img(28, 28);
  
  int idx = atoi(argv[1]);
  mntest.pushImage(idx, img);

  Conv2d<float, 28, 28, 3, 1, 1> convo;
  auto& f1 = convo.d_filters[0];
  f1(0,0) = -1;  f1(0,1) = -1; f1(0,2)=1;
  f1(1,0) = -1;  f1(0,1) = -1; f1(1,2)=1;
  f1(2,0) =  1;  f1(2,1) =  1; f1(2,2)=1;

  convo.d_bias[0](0,0) = 0;

  Tensor out = convo.forward(img)[0].makeMax2d(2);
  out(0,0);
  cout<<"out:\n"<<out<<endl;
  saveTensor(img, "input.png", 252, true);
  saveTensor(out, "convolved.png", 9*out.getRows(), false);
}
