#pragma once
#include<iostream>
#include <random>
#include <time.h>
//#include<vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class softmax
{
public:
	void cleanup_Resize(int categoryNum = 0,int featureNum = 0);
	void setLearningRate(float rate);
	float updatePara(float* x,int* label, int batch, float* backPropagationForInput);
	float test(float* x, int* label, int batch);
	void useGPU(bool);
	~softmax();
private:
	float* W = 0;
	float* b = 0;
	bool gpu = 0;
	int categories;
	int features;
	float learningRate = 0.0005;
};