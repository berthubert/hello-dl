
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb/stb_image_write.h"

#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include "mnistreader.hh"
#include <fenv.h>
#include "misc.hh"

using namespace std;

int main(int argc, char **argv)
{
  int filt=-1;
  if(argc == 2)
    filt= 1 + argv[1][0] - 'a';
  
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  MNISTReader mn("gzip/emnist-letters-train-images-idx3-ubyte.gz", "gzip/emnist-letters-train-labels-idx1-ubyte.gz");
    //MNISTReader mn("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");

  cout<<"Have "<<mn.num()<<" images"<<endl;

  constexpr int imgrows=1200, imgcols=1900;
  vector<uint8_t> out;
  out.resize(imgcols*imgrows);
  auto pix = [&out, &imgrows, &imgcols](int col, int row) -> uint8_t&
  {
    return out[col + row*imgcols];
  };
  
  int count=0;
  Batcher batcher(mn.num());
  for(;;) {
    auto b = batcher.getBatch(1);
    if(b.empty())
      break;
    int n=b[0];
    if(filt >=0 && mn.getLabel(n) != filt)
        continue;
    
    Tensor img(28,28);
    mn.pushImage(n, img);
    
    int x = 30 * (count % (imgcols/30 - 1)); // this many per row
    int y = 30 * (count / (imgcols/30 - 1));
    count++;

    if(x+30 >= imgcols || y+30 >= imgrows)
      break;
    
    for(unsigned int r=0; r < img.getRows(); ++r)
      for(unsigned int c=0; c < img.getCols(); ++c)
        pix(x+c, y+r) = 255 - img(r,c)*255;
  }
  stbi_write_png("poster.png", imgcols, imgrows, 1, &out[0], imgcols);
}
