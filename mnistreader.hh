#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
class MNISTReader
{
public:
  MNISTReader(const std::string& images, const std::string& labels);
  unsigned int num() const
  {
    return d_num;
  }
  std::vector<uint8_t> getImage(int n) const;
  const Eigen::Matrix<float, 28*28, 1>& getImageEigen(int n) const
  {
    if(auto iter = d_converted.find(n); iter != d_converted.end())
      return iter->second;
    else
      throw std::runtime_error("Could not find image "+std::to_string(n));
  }

  char getLabel(int n) const;
private:
  std::vector<uint8_t> d_images;
  std::vector<char> d_labels;
  unsigned int d_rows, d_cols, d_stride, d_num;
  std::unordered_map<int, Eigen::Matrix<float, 28*28, 1>> d_converted;
};
