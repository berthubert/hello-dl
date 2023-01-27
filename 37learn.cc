#include "mnistreader.hh"
#include "vizi.hh"
#include <iostream>
#include "ext/sqlitewriter/sqlwriter.hh"
#include <unistd.h>

using namespace std;

float doTest(const MNISTReader& mntest, const Tensor<float>& weights, float bias, SQLiteWriter* sqw=0)
{
  unsigned int corrects=0, wrongs=0;
  
  for(unsigned int n = 0 ; n < mntest.num(); ++n) {
    int label = mntest.getLabel(n);
    if(label != 3 && label != 7)
      continue;
    Tensor img(28,28);
    mntest.pushImage(n, img);

    float score = (img.dot(weights).sum()(0,0)) + bias; // the calculation

    int predict = score > 0 ? 7 : 3;                  // the verdict

    if(sqw)
      sqw->addValue({{"label", label}, {"res", score}, {"verdict", predict}});

    
    if(predict == label) {
      corrects++;
    }
    else {
      wrongs++;
    }
  }
  float perc = 100.0*corrects/(corrects+wrongs); 
  cout << perc << "% correct" << endl;
  return perc;
}

int main()
{
  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

  cout << "Have "<<mn.num() << " training images and " << mntest.num() << " validation images." <<endl;

  Tensor weights(28,28);
  weights.randomize(1.0/sqrt(28*28));

  saveTensor(weights, "random-weights.png", 252);

  float bias=0;
  
  unlink("37learn.sqlite3");
  SQLiteWriter sqw("37learn.sqlite3");
  int count=0;

  Tensor lr(28,28);
  lr.identity(0.01);

  for(unsigned int n = 0 ; n < mn.num(); ++n) {
    
    int label = mn.getLabel(n);
    if(label != 3 && label != 7)
      continue;

    if(!(count % 4)) {
      if(doTest(mntest, weights, bias) > 98.0)
        break;
      saveTensor(weights, "weights-"+to_string(count)+".png", 252);
    }

    Tensor img(28,28);
    mn.pushImage(n, img);
    float res = (img.dot(weights).sum()(0,0)); // the calculation
    if(count == 25001) {
      auto prod = img.dot(weights);
      saveTensor(img, "random-image.png", 252, true);
      saveTensor(prod, "random-prod.png", 252);
      cout<<"res for first image: " << res << '\n';
    }
    int verdict = res > 0 ? 7 : 3;

    if(label == 7) {
      if(res < 2.0) {
        weights.raw() = weights.raw() + img.raw() * lr.raw();
        bias += 0.1;
      }
    } else {
      if(res > -2.0) {
        weights.raw() = weights.raw() - img.raw() * lr.raw();
        bias -= 0.1;
      }
    }
    

    ++count;
  }
  saveTensor(weights, "weights-final.png", 252);
  doTest(mntest, weights, bias, &sqw);
}
