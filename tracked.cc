#if 0
  typedef TrackedNumber<MatrixXf> TrackedMatrix;
  if(0)
  {
    TrackedMatrix input(MatrixXf::Random(8,1), "input"), weights(MatrixXf::Random(2,8), "weights");
    Matrix<float, 1, 2> tmp;
    tmp<<1,2;
    TrackedMatrix final(tmp, "final");
    cout<<"Gradients should be all zeros, horizontal, two rows: "<<endl;
    cout<<weights.getGrad()<<endl;

    cout<<"Input, should be 1 column: "<<endl;
    cout<<input.getVal()<<endl;

    cout<<"Final, should be 1 2: "<<endl;
    cout<<final.getVal()<<endl;
    
    TrackedMatrix res = weights*input; 
    cout<<"Result should be two numbers"<<endl;
    cout<<res.getVal()<<endl;

    TrackedMatrix output = final * res;
    cout<<"Output, should be one number: "<<output.getVal()<<endl;
    cout<<"--- starting backward ---"<<endl;
    output.backward();
    cout<<"And the grad: "<<endl;
    cout<<weights.getGrad()<<endl;
  }

  if(0)
  {
    TrackedMatrix x(MatrixXf::Constant(1,1, 2), "x");
    TrackedMatrix y(MatrixXf::Constant(1,1, 3), "y");
    TrackedMatrix z(MatrixXf::Constant(1,1, 4), "z");
    TrackedMatrix res = y * (x + x + x + x ) + z; //  y*(4*x)
    // 
    cout<<"Result: "<<res.getVal()<<endl;
    res.backward();
    cout<<x.getGrad()<<endl;  // 4*y = 12
    cout<<y.getGrad()<<endl;  // 4*x = 8
    cout<<z.getGrad()<<endl;  // 1
  }

  if(0)
  {
    TrackedMatrix x(MatrixXf::Constant(1,1, 2), "x");
    TrackedMatrix y(MatrixXf::Constant(1,1, 3), "y");
    TrackedMatrix z(MatrixXf::Constant(1,1, 4.5), "z");
    TrackedMatrix w1(MatrixXf::Constant(1,1, 1.0), "w");
    TrackedMatrix w2(MatrixXf::Constant(1,1, 2.0), "w");
    TrackedMatrix res = (w1+w2) * (x * x * x * y + z * z); 
    // 
    cout<<"Result: "<<res.getVal()<<endl;
    res.backward();
    cout<<x.getGrad()<<endl;  // 3*x^2*y = 36
    cout<<y.getGrad()<<endl;  // x^3 = 8
    cout<<z.getGrad()<<endl;  // 2*z = 9
  }

  if(0)
  {
    TrackedMatrix x(MatrixXf::Constant(1,1, -1), "x");
    TrackedMatrix y(MatrixXf::Constant(1,1, M_PI/2), "y");
    TrackedMatrix z(MatrixXf::Constant(1,1, 3), "z");

    TrackedMatrix res = doFunc(x, ReluFunc()) + doFunc(y, SinFunc()) + doFunc(z, ReluFunc());
    // 
    cout<<"Result: "<<res.getVal()<<endl; // 1+3 = 4
    res.backward();
    cout<<x.getGrad()<<endl;  // 0
    cout<<y.getGrad()<<endl;  // 0
    cout<<z.getGrad()<<endl;  // 1
  }

  if(0)
  {
    TrackedMatrix x(MatrixXf::Constant(1,1, 3), "x");

    TrackedMatrix res = doFunc(x*x, IDFunc());
    cout<<"Result: "<<res.getVal()<<endl; 
    res.backward();
    cout<<x.getGrad()<<endl;    // x^2 -> 2*x -> 6
  }
#endif
  if(0) {
    vector<TrackedFloat> weights, input;
    for(int n=0; n < 10; ++n) {
      weights.emplace_back(n, "w"+to_string(n));
      float v = random()/65535000.0;
      input.emplace_back(v, "i"+to_string(n));
    }
    TrackedFloat res(0.0, "res");
    for(int n=0; n < 10; ++n)
      res=res+weights[n]*input[n];
    
    cout<<res.getVal()<<endl;
    
    res.backward();
    for(int n=0; n < 10 ; ++n)
      cout<<weights[n].getGrad()<<endl;
  }

  if(0)
  {
    TrackedFloat x(2, "x");
    TrackedFloat y(3, "y");
    TrackedFloat res = y*x*x; // 12
    cout << res.getVal() <<endl;
    res.backward();
    cout<<x.getGrad()<<endl;    // 2*x*y = 12
    cout<<y.getGrad()<<endl;    // x^2 = 4
    
    
  }
