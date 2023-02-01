#pragma once
#include <string>
#include <vector>
#include <unordered_map>
//#include "array.hh"
//#include "fvector.hh"

#include "tensor2.hh"
class MNISTReader
{
public:
  MNISTReader(const std::string& images, const std::string& labels);
  unsigned int num() const
  {
    return d_num;
  }
  std::vector<uint8_t> getImage(int n) const;
  const std::vector<float>& getImageFloat(int n) const
  {
    if(auto iter = d_converted.find(n); iter != d_converted.end())
      return iter->second;
    else
      throw std::runtime_error("Could not find image "+std::to_string(n));
  }

  template<typename T>
  void pushImage(int n, Tensor<T>& dest) const
  {
    assert(dest.d_imp && dest.d_imp->d_mode == TMode::Parameter);
    const auto& src = getImageFloat(n);
    for(int row=0 ; row < 28; ++row)
      for(int col=0 ; col < 28; ++col)
        dest(row, col) = src.at(row+28*col);
  }
  /*
  template<typename UT>
  void pushImage(int n, NNArray<UT, 28, 28>& dest, int idx) const
  {
    const auto& src = getImageFloat(n);
    for(int row=0 ; row < 28; ++row)
      for(int col=0 ; col < 28; ++col) {
        if(!dest(row,col).impl) // XXX FUGLY
          dest(row, col) = 0;
        dest(row, col).impl->d_val.v[idx] = src.at(row+28*col);
      }
  }
  */
  
  char getLabel(int n) const;
private:
  std::vector<uint8_t> d_images;
  std::vector<char> d_labels;
  unsigned int d_rows, d_cols, d_stride, d_num;
  std::unordered_map<int, std::vector<float>> d_converted;
};
