#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ext/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb/stb_image_write.h"

#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include "tensor-layers.hh"
#include "convo-alphabet.hh"

using namespace std;

int main(int argc, char** argv)
{
  int cols, rows, n;
  unsigned char *data = stbi_load(argv[1], &cols, &rows, &n, 3);
  if(!data) {
    cerr << "Could not load: "<<stbi_failure_reason() << endl;
    exit(1);
  }
  //cout<<"cols "<< cols <<" rows "<< rows << " n "<< n << endl;

  // in scanline order in data, 8 bits per signal
  const uint8_t *ptr = &data[0];
  std::vector<unsigned int> ints(256);
  std::vector<unsigned int> rowints(rows), colints(cols);
  
  for(int r = 0 ; r < rows; ++r) {
    for(int c = 0 ; c < cols; ++c) {
      const uint8_t cR = *ptr++;
      const uint8_t cG = *ptr++;
      const uint8_t cB = *ptr++;
      uint32_t intensity = (cR + cG + cB)/3;
      ints[intensity]++;
      if(intensity < 20) {
        rowints[r] ++;
        colints[c] ++;
      }
    }
  }

  ofstream colcsv("col.csv");
  colcsv << "col,int"<<endl;
  int startc=-1, stopc=-1;
  for(int c = 0 ; c< cols ;++c) {
    if(colints[c] && startc==-1)
      startc=c;
    if(colints[c])
      stopc=c;
    colcsv << c <<',' << colints[c] <<'\n';
  }
  cout<<"Action from column "<<startc<<" to "<<stopc<<endl;
  
  ofstream rowcsv("row.csv");
  int startrow=-1, stoprow=-1;
  rowcsv <<  "row,int"<<endl;
  for(int r = 0 ; r < rows ;++r) {
    if(rowints[r] && startrow==-1)
      startrow=r;
    if(rowints[r])
      stoprow = r;
    rowcsv << r <<',' << rowints[r] <<'\n';
  }
  cout<<"Action from row "<<startrow<<" to "<<stoprow<<endl;

  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;
  cout<<"Loading model state from file '"<<argv[2]<<"'\n";
  loadModelState(s, argv[2]);
  m.init(s);
  auto topo = m.loss.getTopo();

  struct Rectangle
  {
    char c;
    int lstartcol, lstopcol;
    int lstartrow, lstoprow;
  };

  vector<Rectangle> rects =
    {
      {'a', 327, 430, 468, 559},
      {'b', 448, 563, 418, 557},
      {'c', 591, 670, 451, 545},
      {'d', 696, 789, 418, 517},
      {'e', 814, 897, 439, 514},
      {'f', 901, 954, 418, 514},
      {'g', 978, 1042, 442, 543},
      {'h', 1078, 1140, 426, 505},
      {'i', 1143, 1188, 451, 498},
      {'j', 1189, 1239, 453, 516},
      {'k', 1236, 1294, 441, 502}
    };
  
  for(const auto& l: rects) {
    cout<<"Size: cols "<< l.lstopcol-l.lstartcol<<" rows "<<l.lstoprow-l.lstartrow<<endl;
    
    vector<uint8_t> newpic;
    newpic.reserve((l.lstopcol-l.lstartcol) * (l.lstoprow-l.lstartrow));
    for(int r= l.lstartrow; r < l.lstoprow; ++r) {
      for(int c= l.lstartcol ; c < l.lstopcol; ++c) {
      const uint8_t* ptr = &data[3*(c + cols*r)];
      const uint8_t cR = *ptr++;
      const uint8_t cG = *ptr++;
      const uint8_t cB = *ptr++;
      uint32_t intensity = (cR + cG + cB)/3;
      if(intensity < 10)
        newpic.push_back(255);
      else if(intensity > 100)
        newpic.push_back(0);
      else
        newpic.push_back(255*(1-(intensity-10)/150.0));
      }
    }
    /*
      STBIRDEF int stbir_resize_uint8(     const unsigned char *input_pixels , int input_w , int input_h , int input_stride_in_bytes,
      unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                     int num_channels);
    */
    vector<uint8_t> scaledpic(28*28);
    stbir_resize_uint8(&newpic[0], l.lstopcol - l.lstartcol, l.lstoprow - l.lstartrow, 0,
                       &scaledpic[0], 28, 28, 0, 1);
    
    
    //int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
    
    //    cout<<"RC: "<<stbi_write_png("letter.png", 28, 28, 1, &scaledpic[0], 28)<<endl;
    //cout<<"RC: "<<stbi_write_png("unscaled.png", lstopcol - lstartcol, lstoprow - lstartrow, 1, &newpic[0], lstopcol - lstartcol)<<endl;
    
    m.loss.zerograd(topo);
    for(unsigned int r=0; r < 28; ++r)
      for(unsigned int c=0; c < 28; ++c)
        m.img(r,c) = scaledpic[c+r*28]/175.0;
    int label = l.c-'a';
    m.expected.oneHotColumn(label);
    m.modelloss(0,0); // makes the calculation happen
    
    int predicted = m.scores.maxValueIndexOfColumn(0);
    printImgTensor(m.img);
    cout<<"predicted: "<<(char)(predicted+'a')<<", actual: "<<l.c<<", loss: "<<m.modelloss<<"\n";
    std::vector<pair<float, char>> scores;
    for(unsigned int c = 0 ; c < m.scores.getRows(); ++c)
      scores.push_back({-m.scores(c, 0), 'a'+c});
    sort(scores.begin(), scores.end());
    for(int pos = 0; pos < 5; ++pos)
      cout<<scores[pos].second<<": "<< -scores[pos].first<<'\n';
    
  }
}
