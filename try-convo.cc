#include <iostream>
#include <memory>
#include "mnistreader.hh"
#include "misc.hh"
#include <string.h>

#include <fenv.h>
#include "cnn1.hh"
using namespace std;

ofstream g_tree; //("tree.part");

int main(int argc, char** argv)
{
  if(argc <  3) {
    cerr<<"Syntax: try-convo model-file index"<<endl;
    return 0;
  }

  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

  // MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");
  MNISTReader mntest("gzip/emnist-letters-test-images-idx3-ubyte.gz", "gzip/emnist-letters-test-labels-idx1-ubyte.gz");


  cout<<"Have "<<mntest.num()<<" validation images"<<endl;

  CNNModel m;
  CNNModel::State s;

  cout<<"Loading model state from "<<argv[1]<<endl;
  loadModelState(s, argv[1]);

  m.init(s);

  int idx = atoi(argv[2]);
  mntest.pushImage(idx, m.img);

  int prediction = m.scores.maxValueIndexOfColumn(0);
  int label=mntest.getLabel(idx);
  cout<<"Our prediction: "<<prediction<<", label: "<<label<<endl;
  printImg(m.img);

  std::vector<pair<double, int>> scores;
  for(int n=0; n < 10; ++n) {
    scores.push_back({-m.scores(n,0).getVal(), n});
  }
  sort(scores.begin(), scores.end());
  string stars;
  for(const auto& s : scores) {
    cout<<s.second<<": "<<-s.first;
    stars ="\t";
    if(s.second == label)
      stars += "!\t";
    else
      stars += "\t";
    stars.append((int)(30*exp(-s.first)), '*');
    cout<<stars<<endl;
  }

}
