#include "vizi.hh"
#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "ext/stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb/stb_image_write.h"
#define STB_TRUETYPE_IMPLEMENTATION  // force following include to generate implementation
#include "ext/stb/stb_truetype.h"
#include <vector>

using namespace std;

FontWriter::FontWriter()
{
  FILE* fp = fopen("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", "rb");
  fread(d_ttf_buffer, 1, 1<<18, fp);
  fclose(fp);
  stbtt_InitFont(&d_font, d_ttf_buffer, stbtt_GetFontOffsetForIndex(d_ttf_buffer,0));
}

void FontWriter::writeChar(char ch, int s, int c, int r, std::function<void(int, int, int, int, int)> f)
{
  int w,h,i,j;
  unsigned char *bitmap = stbtt_GetCodepointBitmap(&d_font, 0,stbtt_ScaleForPixelHeight(&d_font, s), ch, &w, &h, 0,0);
  c -= w/2; // center
  for (j=0; j < h; ++j) {
    for (i=0; i < w; ++i) {
      f(c + i, r + j, 255-bitmap[j*w+i], 255-bitmap[j*w+i], 255-bitmap[j*w+i]);
    }
  }
}


void saveTensor(const Tensor<float>& t, const std::string& fname, int size, bool monochrome)
{
  vector<uint8_t> out;
  out.resize(size*size*3);
  struct Pixel {
    uint8_t r, g, b;
  };
  static_assert(sizeof(Pixel)==3);
  
  auto pix = [&out, &size](int col, int row) -> Pixel&
  {
    return *(Pixel*)&out[3 *(col + row*size)];
  };

  float lemin, lemax;
  lemin = lemax = t(0,0);
  
  for(unsigned int row = 0 ; row < t.getRows(); ++row) {
    for(unsigned int col = 0 ; col < t.getCols(); ++col) {
      float v = t(row, col);
      if(v > lemax)
        lemax = v;
      if(v < lemin)
        lemin = v;
    }
  }

  unsigned int hboxsize = size/t.getCols();
  unsigned int vboxsize = size/t.getRows();

  auto box = [&pix, &out](int col, int row, int w, int h, uint8_t cr, uint8_t cg, uint8_t cb) {
    for(int c = col ; c < col + w; ++c)
      for(int r = row ; r < row + h; ++r)
        pix(c, r) = {cr,cg,cb};
  };
  
  for(unsigned int row = 0 ; row < t.getRows(); ++row) {
    for(unsigned int col = 0 ; col < t.getCols(); ++col) {
      float v = t(row, col);

      if(monochrome) {
        uint8_t color = 255.0*(v - lemin)/(lemax-lemin);
        box(col * hboxsize, row * vboxsize, hboxsize, vboxsize, color, color, color);
      }
      else {
        if(v > 0)  // red
          box(col * hboxsize, row * vboxsize, hboxsize, vboxsize, 255 * v/lemax, 0, 0);
        else if(v < 0) // blue
          box(col * hboxsize, row * vboxsize, hboxsize, vboxsize, 0, 0, 255 * v/lemin);
      }
    }
  }
      
  
  stbi_write_png(fname.c_str(), size, size, 3, &out[0], 3*size);
}
