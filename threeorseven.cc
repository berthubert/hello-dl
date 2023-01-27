#include "mnistreader.hh"
#include "vizi.hh"
#include <iostream>
#include "ext/sqlitewriter/sqlwriter.hh"
#include <unistd.h>

using namespace std;

int main()
{
  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  cout << "Have "<<mn.num() << " training images and " << mntest.num() << " validation images." <<endl;

  Tensor threes(28, 28), sevens(28, 28);
  
  float threecount = 0, sevencount=0;
  for(unsigned int n = 0 ; n < mn.num(); ++n) {
    int label = mn.getLabel(n);
    if(label != 3 && label != 7)
      continue;
    
    Tensor img(28,28);
    mn.pushImage(n, img);

    if(label == 3) {
      threecount++;
      threes.raw() += img.raw();
    }
    else {
      sevencount++;
      sevens.raw() += img.raw();
    }
  }
  saveTensor(threes, "threes.png", 252, true);
  saveTensor(sevens, "sevens.png", 252, true);

  Tensor totcount(threecount + sevencount);
  auto delta = (sevens - threes) / totcount;

  saveTensor(delta, "diff.png", 252);

  float threesmean = 0, sevensmean = 0;

  unlink("threeorseven.sqlite3");
  SQLiteWriter sqw("threeorseven.sqlite3");
  
  for(unsigned int n = 0 ; n < mn.num(); ++n) {
    int label = mn.getLabel(n);
    if(label != 3 && label != 7)
      continue;
    Tensor img(28,28);
    mn.pushImage(n, img);

    float res = (img.dot(delta).sum()(0,0)); // the calculation

    sqw.addValue({{"label", label}, {"res", res}});
    if(label == 3) 
      threesmean += res;
    else 
      sevensmean += res;
  }

  cout<<"Three average result: "<<threesmean/threecount<<", seven average result: "<<sevensmean/sevencount<<endl;
  float middle = (sevensmean/sevencount + threesmean/threecount) / 2;
  //  middle = 0;
  cout<<"Middle: "<< middle <<endl;

  float bias = -middle;
  
  unsigned int corrects=0, wrongs=0;
  int haveseven=0, havethree=0;
  for(unsigned int n = 0 ; n < mntest.num(); ++n) {
    int label = mntest.getLabel(n);
    if(label != 3 && label != 7)
      continue;
    Tensor img(28,28);
    mntest.pushImage(n, img);

    float score = (img.dot(delta).sum()(0,0)) + bias; // the calculation
    int predict = score > 0 ? 7 : 3;                  // the verdict
    
    if(predict == label) {
      if(haveseven < 5 && label==7) {
        saveTensor(img, "seven-"+to_string(haveseven)+".png", 252, true);
        Tensor prod  = img.dot(delta);
        saveTensor(prod, "prod7-"+to_string(haveseven)+".png", 252);
        haveseven++;
      }
      if(havethree < 5 && label==3) {
        saveTensor(img, "three-"+to_string(havethree)+".png", 252, true);
        Tensor prod  = img.dot(delta);
        saveTensor(prod, "prod3-"+to_string(havethree)+".png", 252);
        havethree++;
      }

      corrects++;
    }
    else {
      saveTensor(img, "wrong-"+to_string(label)+"-"+to_string(wrongs)+".png", 252);
      wrongs++;
    }
  }
  cout << 100.0*corrects/(corrects+wrongs) << "% correct" << endl;
}
