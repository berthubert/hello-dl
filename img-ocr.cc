#include "ext/stb/stb_image.h"
#include "ext/stb/stb_image_resize.h"
#include "ext/stb/stb_image_write.h"


#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include "tensor-layers.hh"
#include "convo-alphabet.hh"
#include "vizi.hh"

using namespace std;


int main(int argc, char** argv)
{
  if(argc < 2) {
    cout<<"Syntax: img-ocr imagename modelname"<<endl;
    return EXIT_FAILURE;
  }
  int cols, rows, n;
  unsigned const char *data = stbi_load(argv[1], &cols, &rows, &n, 3);
  if(!data) {
    cerr << "Could not load: "<<stbi_failure_reason() << endl;
    exit(1);
  }

  vector<uint8_t> paintdata(3*cols*rows); //r, g and b
  memcpy(&paintdata[0], data, 3*cols*rows);
  
  //cout<<"cols "<< cols <<" rows "<< rows << " n "<< n << endl;

  // in scanline order in data, 8 bits per signal
  std::vector<unsigned int> ints(256);
  std::vector<unsigned int> rowints(rows);

  auto getintens=[&data, &rows, &cols](int c, int r) {
    const uint8_t* ptr = &data[3*(c + r*cols)];
    return (ptr[0] + ptr[1] + ptr[2])/3;
  };

  auto paint = [&paintdata,&rows,&cols](int col, int row, int red=0, int green=255, int blue=0) {
    if(col < 0 || row < 0 || col>=cols || row >= rows)
      return;
    uint8_t* ptr = &paintdata[3*(col + cols*row)];
    ptr[0]=red;
    ptr[1]=green;
    ptr[2]=blue;
  };

  for(int r = 0 ; r < rows; ++r) {
    for(int c = 0 ; c < cols; ++c) {
      uint32_t intensity = getintens(c, r);
      ints[intensity]++;
    }
  }

  ofstream histocsv("histo.csv");
  histocsv<<"int,count"<<endl;
  float cumul=0;
  int whitebalhigh=0, whiteballow=0;
  for(int i = 0; i < 256; ++i) {
    histocsv << i << ',' << ints[i] <<'\n';
    cumul += ints[i];
    if(cumul < 0.000002 * rows * cols)
      whiteballow=i;
        
    if(cumul < 0.01 * rows * cols)
      whitebalhigh=i;
  }
  //  whiteballow=0; whitebalhigh = 154;
  cout<<"Everything below "<<whiteballow<<" is black"<<endl;
  cout<<"Everything beyond "<<whitebalhigh<<" is white"<<endl;

  for(int r = 0 ; r < rows; ++r) {
    for(int c = 0 ; c < cols; ++c) {
      int intensity = getintens(c, r);
      if(intensity < whitebalhigh) {
        rowints[r] ++;
      }
    }
  }

  
  /*
  ofstream colcsv("col.csv");
  colcsv << "col,int"<<endl;
  int startc=-1, stopc=-1;
  for(int c = 0 ; c < cols ;++c) {
    if(colints[c] && startc==-1)
      startc=c;
    if(colints[c])
      stopc=c;
    colcsv << c <<',' << colints[c] <<'\n';
  }
  cout<<"Action from column "<<startc<<" to "<<stopc<<endl;
  */
  
  ofstream rowcsv("row.csv");
  int startrow=-1, stoprow=-1;
  rowcsv <<  "row,int"<<endl;
  vector<pair<int,int>> vblocks;
  for(int r = 0 ; r < rows ;++r) {
    if(startrow==-1) {
      if(rowints[r]) {
        startrow=r;
        stoprow=r;
      }
    }
    else { // we're in a block
      if(rowints[r])
        stoprow = r; // extend it
      else {
        cout<<"Possible end at row "<<r<<" after "<<r-startrow<<" rows\n";
        if(r - startrow > 10) {
          cout<<"   Yes, end, startrow was "<<startrow<<endl;
          vblocks.emplace_back(startrow, r);
          startrow=-1;
        }
        else stoprow = r; // extend it anyhow
      }
    }
    rowcsv << r <<',' << rowints[r] <<'\n';
  }
  if(startrow >= 0) {
    vblocks.emplace_back(startrow, stoprow);
  }
  cout<<"Identified "<<vblocks.size()<<" vertical blocks\n";


  struct Rectangle
  {
    int lstartcol, lstopcol;
    int lstartrow, lstoprow;
  };

  vector<Rectangle> rects;
  
  for(const auto& vb: vblocks) {
    vector<unsigned int> colints(cols);
    for(int r = vb.first ; r < vb.second; ++r) {
      for(int c = 0 ; c < cols; ++c) {
        int intensity = getintens(c, r);
        ints[intensity]++;
        if(intensity < whitebalhigh) {
          colints[c] ++;
        }
      }
    }

    int startc=-1, stopc=-1;
    vector<pair<int, int>> hblocks;
    for(int c = 0 ; c < cols ;++c) {
      if(startc==-1) {
        if(colints[c])
          startc=c;
      }
      else {
        if(!colints[c] && c - startc > 2) {
          hblocks.emplace_back(startc, c);
          startc=-1;
        }
        else if(colints[c] < 2 && c - startc > 40) { // if letter is wide enough, split it more easily
          hblocks.emplace_back(startc, c);
          startc=-1;
        }
        else
          stopc = c;
      }
    }
    if(startc>=0)
      hblocks.emplace_back(startc, stopc);

    
    for(const auto& hb : hblocks) {
      if(hb.second - hb.first < 5) {
        cout<<"Skipping tiny block"<<endl;
        continue;
      }
      
      // take some stuff from the top
      
      int startrow = vb.first;
      cout<<"Original vertical start row (column "<<hb.first<<"): "<< startrow <<endl;
      for(; startrow < vb.second; ++startrow) {
        // scan row from left to right
        int c;
        int pixcount=0;
        for(c = hb.first; c < hb.second; ++c) {
          if(getintens(c, startrow) < whitebalhigh) {
            pixcount++;
          }
        }
        if(pixcount > 2) //
          break;
      }
      cout<<"Adjusted vertical start row: "<< startrow <<endl;

      int stoprow = vb.second;
      cout<<"Original vertical stop row (column "<<hb.first<<"): "<< stoprow <<endl;
      for(; stoprow > vb.first; --stoprow) {
        // scan row from left to right
        int c;
        int pixcount=0;
        for(c = hb.first; c < hb.second; ++c) {
          if(getintens(c, stoprow) < whitebalhigh) {
            pixcount++;
          }
        }
        if(pixcount > 2)
          break;
      }
      cout<<"Adjusted vertical stopt row: "<< stoprow <<endl;
      int startcol = hb.first, stopcol = hb.second;

      int aspectdelta = (stoprow - startrow) - (hb.second - hb.first);
      if( aspectdelta < 0 )  { // wider than tall
        cout<<"Was too wide, moving top up by "<<-aspectdelta/2<<", and bottom same but down"<<endl;
        startrow += aspectdelta/2; // make taller
        stoprow -= aspectdelta/2;
      }
      else if(aspectdelta > 0) {// taller than wide
        cout<<"Was too tall, moving left limit by "<<aspectdelta/2<<" pixels and right limit similar but right"<<endl;
        startcol -= aspectdelta/2;
        stopcol += aspectdelta/2; // make wider
      }

      startcol -= 2;
      stopcol += 2;
      startrow -=2;
      stoprow +=2;
      if(startcol > 0 && startrow > 0 && stopcol < cols && stoprow < rows && stopcol - startcol > 12 && stoprow - startrow > 12) {
        rects.push_back({startcol, stopcol, startrow, stoprow});
        
        for(int c = startcol ; c < stopcol; ++c) {
          paint(c, startrow);
          paint(c, stoprow);
        }
        for(int r = startrow; r < stoprow; ++r) {
          paint(startcol, r);
          paint(stopcol, r);
        }

        for(int c = startcol + 1 ; c < stopcol - 1; ++c) {
          for(int r = startrow + 1; r < stoprow -1 ; ++r) {
            int intens=getintens(c,r);
            if(intens <= whiteballow)
              intens=0;
            else if(intens > whitebalhigh)
              intens=255;
            else
              intens = 255* pow((intens - whiteballow) / (whitebalhigh - whiteballow), 1);
            paint(c, r, intens, intens, intens);
          }
        }
      }
    }
  }
  // vertical rulers
  for(const auto& vb : vblocks) {
    cout<<"Attempting to draw a line from c = 0 to c= "<<cols<<" at rows "<< vb.first <<" " << vb.second<<endl;
    
    for(int lc = 0 ; lc < cols; ++lc) {
      paint(lc, vb.first, 0,0,0);
      paint(lc, vb.second, 0,0,0);
    }
  }

  FontWriter fw;

  
  cout<<"Got "<<rects.size()<<" rectangles to look at"<<endl;

  
  ConvoAlphabetModel m;
  ConvoAlphabetModel::State s;
  cout<<"Loading model state from file '"<<argv[2]<<"'\n";
  loadModelState(s, argv[2]);
  m.init(s, true);
  auto topo = m.loss.getTopo();

  
  for(const auto& l: rects) {
    cout<<"Size: cols "<< l.lstopcol-l.lstartcol<<" rows "<<l.lstoprow-l.lstartrow<<endl;
    
    vector<uint8_t> newpic;
    newpic.reserve((l.lstopcol-l.lstartcol) * (l.lstoprow-l.lstartrow));
    for(int r= l.lstartrow; r < l.lstoprow; ++r) {
      for(int c= l.lstartcol ; c < l.lstopcol; ++c) {
      int intensity = getintens(c, r);
      if(intensity < whiteballow)
        newpic.push_back(255);
      else if(intensity > whitebalhigh)
        newpic.push_back(0);
      else
        newpic.push_back(255*(1- pow((intensity - whiteballow)/(whitebalhigh - whiteballow), 1) ));
      }
    }
    /*
      STBIRDEF int stbir_resize_uint8(     const unsigned char *input_pixels , int input_w , int input_h , int input_stride_in_bytes,
      unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                     int num_channels);
    */
    vector<uint8_t> scaledpic(24*24);
    stbir_resize_uint8(&newpic[0], l.lstopcol - l.lstartcol, l.lstoprow - l.lstartrow, 0,
                       &scaledpic[0], 24, 24, 0, 1);
    
    
    //int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
    
    //    cout<<"RC: "<<stbi_write_png("letter.png", 28, 28, 1, &scaledpic[0], 28)<<endl;
    //cout<<"RC: "<<stbi_write_png("unscaled.png", lstopcol - lstartcol, lstoprow - lstartrow, 1, &newpic[0], lstopcol - lstartcol)<<endl;

    m.img.zero();
    for(unsigned int r=0; r < 24; ++r)
      for(unsigned int c=0; c < 24; ++c)
        m.img(2+r,2+c) = scaledpic[c+r*24]/255.0;

    float prevmean = m.img.d_imp->d_val.mean();
    float prevdev = sqrt((m.img.d_imp->d_val.array() - prevmean).unaryExpr([](float v) { return v*v; }).sum()/(28*28.0));
    m.img.normalize(0.172575, 0.25);
    
    float mean = m.img.d_imp->d_val.mean();
    cout << "mean: "<<prevmean<<" -> "<< mean  <<  endl;
    cout << "dev: "<<prevdev<< " -> "<< sqrt((m.img.d_imp->d_val.array() - mean).unaryExpr([](float v) { return v*v; }).sum()/(28*28.0)) << endl;
    
    int label = 0;
    m.expected.oneHotColumn(label);
    m.modelloss(0,0); // makes the calculation happen
    
    int predicted = m.scores.maxValueIndexOfColumn(0);
    printImgTensor(m.img);
    cout<<"predicted: "<<(char)(predicted+'a')<<endl;


    fw.writeChar(predicted+'a', 80,  (l.lstartcol + l.lstopcol) / 2, l.lstoprow +10 , paint);
    
    std::vector<pair<float, char>> scores;
    for(unsigned int c = 0 ; c < m.scores.getRows(); ++c)
      scores.push_back({-m.scores(c, 0), 'a'+c});
    sort(scores.begin(), scores.end());
    for(int pos = 0; pos < 5; ++pos)
      cout<<scores[pos].second<<": "<< -scores[pos].first<<'\n';
    m.loss.zerograd(topo);
  }
  stbi_write_png("boxed.png", cols, rows, 3, &paintdata[0], 3*cols);
}
