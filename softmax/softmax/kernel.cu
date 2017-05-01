
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>

#define SIZE 512

#define ONERROR(expression)  cudaStatus = expression;\
if(cudaStatus != cudaSuccess) \
{\
    fprintf(stderr, "%s failed on line %d!\n",__FUNCTION__,__LINE__);\
    goto Error;\
}

__global__ void multiply(int categories, int features,int batches, float* W, float* b, float* input, float* score)
{
	int feature = threadIdx.x;
	int category = blockIdx.x;
	int batch = blockIdx.y;
	float Wx;
	float Wx_plus_b;
	float result;

	__shared__ float temp[SIZE];//SIZE * categories
	temp[feature % SIZE] = 0;

	__syncthreads();



	{
		Wx = W[feature + category * features] * input[feature + batch * features];
		temp[feature % SIZE] += Wx;
		__syncthreads();
		for (int thread = SIZE / 2; thread > 0; thread /= 2)
		{
			if (threadIdx.x < thread)
			{
				temp[threadIdx.x] += temp[threadIdx.x + thread];
			}
			__syncthreads();
		}

		score[category + categories * batch] = temp[0];

	}
}
__global__ void update(int categories, int features, int batches, float* W, float* b, float* input, int* labels,float* score,float learningRate)
{
	int category = threadIdx.x;
	int batch = blockIdx.x;
	extern __shared__ float diff[];
	__shared__ float temp[SIZE];
	diff[category] = exp(score[category + batch * categories]);
	temp[category % SIZE] = 0;
	__syncthreads();
	temp[category % SIZE] += diff[category];
	__syncthreads();
	for (int thread = SIZE / 2; thread > 0; thread /= 2)
	{
		if (threadIdx.x < thread)
		{
			temp[threadIdx.x] += temp[threadIdx.x + thread];
		}
		__syncthreads();
	}
	diff[category] /= temp[0];
	if (category == labels[batch])
		diff[category]--;
	__syncthreads();
	for (int feature = 0; feature < features; feature++)
	{
		W[feature + category*features] -= learningRate * diff[category] * input[feature + batch * features];
	}
	b[category] -= learningRate * diff[category];
}




extern "C"
cudaError_t mallocInitPara(float** dev_W, float** dev_b, int features, int categories,float* host_W, float* host_b)
{
	cudaError_t cudaStatus;
	ONERROR(cudaMalloc((void**)dev_W, features * categories * sizeof(float)));
	ONERROR(cudaMemcpy(*dev_W, host_W, sizeof(float)*features*categories, cudaMemcpyHostToDevice));

	ONERROR(cudaMalloc((void**)dev_b, categories * sizeof(float)));
	ONERROR(cudaMemcpy(*dev_b, host_b, sizeof(float)*categories, cudaMemcpyHostToDevice));
	return cudaSuccess;
Error:
	cudaFree(*dev_W);
	cudaFree(*dev_b);
	return cudaStatus;
}
using namespace std;

extern "C"
cudaError_t updatePara_SGD(float* x, int* label, int batches, float* backPropagationForInput,float* dev_W,float* dev_b, int features, int categories,float learningRate, float* correctRate)
{
	cudaError_t cudaStatus = cudaSuccess;

	
	float* dev_x;
	float* dev_score;
	float* host_score = new float[categories * batches];
	int* dev_labels;
	ONERROR(cudaMalloc(&dev_x,sizeof(float)*features*batches));
	ONERROR(cudaMalloc(&dev_labels, sizeof(int)*batches));
	ONERROR(cudaMemcpy(dev_x, x, sizeof(float)*features*batches, cudaMemcpyHostToDevice));
	ONERROR(cudaMemcpy(dev_labels, label, sizeof(int)*batches, cudaMemcpyHostToDevice));
	ONERROR(cudaMalloc(&dev_score, sizeof(float)*categories*batches));
/*
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	int a = properties.multiProcessorCount;
	int b =  properties.maxThreadsPerMultiProcessor;*/

	dim3 gridShape = dim3(categories, batches);
	dim3 blockShape = dim3(features);
	multiply << <gridShape, blockShape >> > (categories, features, batches, dev_W, dev_b, dev_x, dev_score);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	ONERROR(cudaDeviceSynchronize());
	update << <batches, categories, categories*sizeof(float) >> > (categories, features, batches, dev_W, dev_b, dev_x, dev_labels, dev_score, learningRate);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	ONERROR(cudaDeviceSynchronize());
	ONERROR(cudaMemcpy(host_score,dev_score, sizeof(float)*categories*batches,cudaMemcpyDeviceToHost));

	int correctCount = 0;
	for (int i = 0; i < batches; i++)
	{
		float max = -999999.0;
		int maxlabel = 0;
		for (int j = 0; j < categories; j++)
		{
			if (host_score[categories * i + j] > max)
			{
				max = host_score[categories * i + j];
				maxlabel = j;
			}
		}
		if (maxlabel == label[i])
			correctCount++;
	}
	delete[] host_score;
	correctRate[0] = (float)correctCount / batches;
Error:
	cudaFree(dev_x);
	cudaFree(dev_labels);
	cudaFree(dev_score);
	return cudaStatus;
}
