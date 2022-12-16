#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
class MNISTReader
{
public:
  MNISTReader(const std::string& images, const std::string& labels);
  unsigned int num() const
  {
    return d_num;
  }
  std::vector<uint8_t> getImage(int n) const;
  Eigen::MatrixXf getImageEigen(int n) const;
  char getLabel(int n) const;
private:
  std::vector<uint8_t> d_images;
  std::vector<char> d_labels;
  unsigned int d_rows, d_cols, d_stride, d_num;
};
