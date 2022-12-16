#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "mnistreader.hh"
#include <map>
#include "misc.hh"

using namespace std;
using namespace Eigen;

static float sigmoid(const float z)
{
  return 1.0 / (1.0 + exp(-z));
} 

static float gelu(const float x)
{
  //  constexpr double s2pi = .79788456081426795426; // ~ sqrt(2/pi)
  //  return x*0.5*(1.0 + tanh(s2pi*(x+0.044715*pow(x,3))));
  constexpr float invsqrt2 = .70710678118654752440; // 1/sqrt(2)
  return 0.5*x*(1+erff(x*invsqrt2));  
}

struct Layer
{
  MatrixXf weights, weightsGrad;
  MatrixXf bias, biasGrad;
  std::function<float(float)> action{[](float x){return x;}};
};
typedef Matrix<float, 28*28,1> img_t;

MatrixXf doStuff(const img_t& img, const Layer &l)
{
  MatrixXf ret = l.weights*img;
  ret += l.bias;
  ret = ret.unaryExpr(l.action);
  return ret;
}

void printImg(const img_t& img)
{
  cout<<"Image: "<<img.mean()<<"\n";
  for(int y=0; y < 28; ++y) {
    for(int x=0; x < 28; ++x) {
      float val = img(x*28+y);
      if(val > 0.5)
        cout<<'X';
      else if(val > 0.25)
        cout<<'*';
      else if(val > 0.125)
        cout<<'.';
      else
        cout<<' ';
    }
    cout<<'\n';
  }
  cout << img(28*28-1);
  cout<<"\n";
}


void scoreModel(const Layer& l, const MNISTReader& mntest)
{
  /*
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist2(0, mntest.num()-1);
  */
  //  cout<<"Have "<<mntest.num()<<" test digits"<<endl;
  unsigned int corrects=0, wrongs=0;
  int threes=0, sevens=0, threepreds=0, sevenpreds=0;
  for(unsigned int i = 0 ; i < mntest.num() - 1; ++i){
    int label = mntest.getLabel(i);
    if(label==3)
      threes++;
    else if(label == 7)
      sevens++;
    else continue;

    MatrixXf pic = mntest.getImageEigen(i)/256.0;
    MatrixXf result = doStuff(pic, l);

    int verdict = result(0) < 0.5 ? 3 : 7;
    //    cout<<(int)label<<": verdict "<<result(0)<<" " <<verdict<<" pic "<<pic.mean()<<" weights "<<l.weights.mean()<<" bias "<<l.bias.mean()<<endl;
    if(verdict == label) {
      corrects++;
    }
    else {
      wrongs++;
    }

    if(verdict==3)
      threepreds++;
    else if(verdict == 7)
      sevenpreds++;

    
  }
  cout<<corrects*100.0/(corrects+wrongs)<<"% correct, threes "<<threes<<" sevens "<<sevens<<" threepreds "<<threepreds<<" sevenpreds "<<sevenpreds<<"\n";
}

int main()
{
  MNISTReader mn("gzip/emnist-digits-train-images-idx3-ubyte.gz", "gzip/emnist-digits-train-labels-idx1-ubyte.gz");
  cout<<"Have "<<mn.num()<<" images"<<endl;

  map<char, Matrix<float, 28*28, 1>> ideals;
  map<char, unsigned int> counts;

  for(unsigned int i = 0 ; i < mn.num()  ; ++i) {
    int label = mn.getLabel(i);
    if(label != 3 && label != 7)
      continue;
    if(!ideals.count(mn.getLabel(i)))
      ideals[mn.getLabel(i)].setZero();
    
    auto& id = ideals[mn.getLabel(i)];
    auto pic = mn.getImageEigen(i)/256.0;
    id += pic;
    counts[mn.getLabel(i)]++;
  }
  for(auto& id : ideals) {
    id.second /= counts[id.first];

    cout<<"Ideal for label "<<(int)id.first<<": \n";
    printImg(id.second);
  }
  MNISTReader mntest("gzip/emnist-digits-test-images-idx3-ubyte.gz", "gzip/emnist-digits-test-labels-idx1-ubyte.gz");

#if 0  
  std::random_device r;


  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist2(0, mntest.num()-1);

  cout<<"Have "<<mntest.num()<<" test digits"<<endl;
  unsigned int corrects=0, wrongs=0;
  for(unsigned int i = 0 ; i < mntest.num() - 1; ++i){
    int label = mntest.getLabel(i);
    if(label != 3 && label != 7)
      continue;
    
    /*    int idx = uniform_dist2(e1);
    cout<<"Image with label '"<<(int)mntest.getLabel(idx)<<"': \n";
    */
    int idx = i;
    auto pic = mntest.getImageEigen(idx)/256;
    printImg(pic);


    map<double, int> scores;
    for(auto& id : ideals) {
      scores[(id.second - pic).cwiseAbs().mean()]=id.first;
    }
    for(const auto& s : scores) {
      cout<< s.first <<": "<<(int)s.second<<'\n';
    }
    if(scores.begin()->second == mntest.getLabel(idx)) {
      cout<<"CORRECT!"<<endl;
      corrects++;
    }
    else {
      cout<<"WRONG!"<<endl;
      wrongs++;
    }
  }
  cout<<corrects*100.0/(corrects+wrongs)<<"% correct\n";
#endif
  
  srandom(time(0));
  Layer l;
  l.weights = MatrixXf::Random(1, 28*28);
  l.bias = MatrixXf::Random(1, 1);
  l.action = sigmoid;

  vector<int> threeseven;
  for(unsigned int i = 0 ; i < mn.num(); ++i) {
    int label = mn.getLabel(i);
    if(label==3 || label == 7)
      threeseven.push_back(i);
  }
  cout<<"Have "<<threeseven.size()<<" threes and sevens"<<endl;
  for(;;) {
    Batcher batcher(threeseven);

    for(;;) {
      scoreModel(l, mntest);
      auto batch = batcher.getBatch(256);
      if(batch.empty())
        break;

      double origloss = 0;
      for(auto& idx : batch) {
        int label = mn.getLabel(idx);
        if(label != 3 && label != 7) {
          cerr<<"Impossible label"<<endl;
          exit(1);
        }

        MatrixXf leImage = mn.getImageEigen(idx)/256.0;
        //        printImg(leImage);
        MatrixXf result = doStuff(leImage, l);
        MatrixXf expected(1,1);
        expected << (label == 3 ? 0.0 : 1.0);
        double thisloss = abs((result-expected).mean());
        //        cout<<"label " <<(int)label<<" result "<<result<<" expected "<<expected<<" loss  "<<thisloss<<endl;
        origloss += thisloss;
      }
      origloss = origloss/batch.size();
      
      l.weightsGrad = MatrixXf::Zero(1, 28*28);    
      for(int r=0; r < l.weights.rows(); ++r) {
        for(int c=0; c < l.weights.cols(); ++c) {
          Layer lprime = l;
          lprime.weights(r,c) += 0.001; // tweak a bit

          double newloss=0;
          for(auto& idx : batch) {
            MatrixXf leImage = mn.getImageEigen(idx)/256.0;
            MatrixXf result = doStuff(leImage, lprime);
          
            MatrixXf expected(1,1);
            int label = mn.getLabel(idx);
            expected << (label == 3 ? 0.0 : 1.0);
          
            double thisloss = abs((result-expected).mean());
            newloss += thisloss;
          }
          newloss = newloss/batch.size();
          //          cout<<"Change ("<<r<<','<<c<<"): "<<origloss<<" -> "<<newloss<<", grad "<<1000*(newloss-origloss) <<"\n";
          l.weightsGrad(r,c) = 1000.0*(newloss-origloss);
        }
        l.biasGrad = MatrixXf::Zero(1, 1);
        for(int r=0; r < l.bias.rows(); ++r) {
          for(int c=0; c < l.bias.cols(); ++c) {
            Layer lprime = l;

            lprime.bias(r,c) += 0.001; // tweak a bit
          
            double newloss=0;
            for(auto& idx : batch) {
              MatrixXf leImage = mn.getImageEigen(idx)/256.0;
              MatrixXf result = doStuff(leImage, lprime);
            
              MatrixXf expected(1,1);
              expected << (mn.getLabel(idx) == 3 ? 0.0 : 1.0);
            
              double thisloss = abs((result-expected).mean());
              newloss += thisloss;
            }
            newloss = newloss/batch.size();
            //          cout<<"Change ("<<r<<','<<c<<"): "<<loss<<" -> "<<newloss<<", delta "<<(newloss-loss)<<endl;
            l.biasGrad(r,c) = 1000.0*(newloss-origloss);
          }
        }

        l.weights -= 0.04*l.weightsGrad;
        l.bias -= 0.04*l.biasGrad;
        /*
        cout<<"Weights grad: "<<l.weightsGrad<<endl;
        cout<<"Bias grad: "<<l.biasGrad<<endl;
        cout<<"Magnitude weights "<<l.weights.cwiseAbs().mean()<<" " << 0.1*l.weightsGrad.cwiseAbs().mean()<<endl;
        cout<<"Magnitude bias "<<l.bias.cwiseAbs().mean()<<" " << 0.1*l.biasGrad.cwiseAbs().mean()<<endl;
        */

        double finalloss=0;
        for(auto& idx : batch) {
          MatrixXf leImage = mn.getImageEigen(idx)/256.0;
          MatrixXf result = doStuff(leImage, l);
        
          MatrixXf expected(1,1);
          expected << (mn.getLabel(idx) == 3 ? 0.0 : 1.0);
        
          double thisloss = abs((result-expected).mean());
          finalloss += thisloss;
        }
        finalloss = finalloss/batch.size();
        //        cout<<"origloss "<<origloss<<" finalloss "<<finalloss<<endl;
      
      }
    }
  }
}
  
