#include "mnistreader.hh"
#include <string.h>
#include <utility>
#include <memory>
#include <stdexcept>
#include "zlib.h"
#include <arpa/inet.h>

using namespace std;

static auto safeOpen(const std::string& fname)
{
  gzFile fp = gzopen(fname.c_str(), "rb");
  if(!fp)
    throw runtime_error("Error opening file "+fname+": "+strerror(errno));
  return fp;
}

MNISTReader::MNISTReader(const std::string& images, const std::string& labels)
{
  struct idx1header
  {
    uint32_t magic;
    uint32_t num;
  } __attribute__((packed));

  struct idx3header
  {
    uint32_t magic;
    uint32_t num;
    uint32_t rows;
    uint32_t cols;
  } __attribute__((packed));

  auto imgfp = safeOpen(images);
  auto labelsfp = safeOpen(labels);

  idx1header i1h;
  idx3header i3h;
  if(gzfread(&i1h, sizeof(idx1header), 1, labelsfp) != 1)
    throw std::runtime_error("Label file too short");
  if(gzfread(&i3h, sizeof(idx3header), 1, imgfp) != 1)
    throw std::runtime_error("Images file too short");

  i1h.magic = htonl(i1h.magic);
  i1h.num = htonl(i1h.num);

  i3h.magic = htonl(i3h.magic);
  i3h.num = htonl(i3h.num);
  i3h.rows = htonl(i3h.rows);
  i3h.cols = htonl(i3h.cols);

  d_rows = i3h.rows;
  d_cols = i3h.cols;
  d_stride = d_rows * d_cols;
  d_num = i3h.num;
  if(i1h.magic != 2049)
    throw runtime_error("Magic value of labels file wrong "+to_string(i1h.magic));
  if(i3h.magic != 2051)
    throw runtime_error("Magic value of images file wrong "+to_string(i1h.magic));

  if(i3h.num != i1h.num)
    throw runtime_error("Mismatch between number of labels and number of images");

  
  d_images.resize(i3h.num*i3h.cols*i3h.rows);
  if(gzfread((char*)&d_images[0], i3h.cols*i3h.rows, i3h.num, imgfp) != i3h.num)
    throw runtime_error("Could not read all "+to_string(i3h.num)+" images");

  d_labels.resize(i3h.num);
  if(gzfread((char*)&d_labels[0], 1, i3h.num, labelsfp) != i3h.num)
    throw runtime_error("Could not read all "+to_string(i3h.num)+" labels");

  gzclose(imgfp);
  gzclose(labelsfp);

  Eigen::Matrix<float, 28*28,1> tmp;
  for(unsigned int n=0 ; n < d_num; ++n) {
    unsigned int pos = n * d_stride;
    for(unsigned int i=0; i < d_stride; ++i) {
      tmp(i) = d_images.at(pos+i);
    }
    d_converted[n]=tmp/256.0;
  }
}

vector<uint8_t> MNISTReader::getImage(int n) const
{
  unsigned int pos = n*d_rows*d_cols;
  vector<uint8_t> ret(&d_images.at(pos), &d_images.at(pos + d_rows*d_cols));
  return ret;
}

  

char MNISTReader::getLabel(int n) const
{
  return d_labels.at(n);
}

