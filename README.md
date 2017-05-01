Implement softmax classifier using c++.
I implement the algorithm of back propagation and update parameters using SGD.
The algorithm is run on CPU by default, if want to run on GPU, just call useGPU(1) like this:
maxsoft classifier;
classifier.useGPU(1);
...

It seems GPU version didn't cost less time than CPU version. Next step is to optimize the cuda code.
