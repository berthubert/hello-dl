#pragma once
#include <iostream>
#include <functional>
#include "tensor2.hh"
#include "ext/stb/stb_truetype.h"

template<typename T>
void printImgTensor(const T& img)
{
  for(unsigned int y=0; y < img.getRows(); ++y) {
    for(unsigned int x=0; x < img.getCols(); ++x) {
      float val = img(y,x);
      if(val > 0.5)
        std::cout<<'X';
      else if(val > 0.25)
        std::cout<<'*';
      else if(val > 0.125)
        std::cout<<'.';
      else
        std::cout<<' ';
    }
    std::cout<<'\n';
  }
  std::cout<<"\n";
}

struct FontWriter
{
  FontWriter();
  void writeChar(char ch, int s, int c, int r, std::function<void(int, int, int, int, int)> f);
  stbtt_fontinfo d_font;
  std::vector<char> d_ttf_buffer; 
};

void saveTensor(const Tensor<float>& t, const std::string& fname, int size, bool monochrome=false);
