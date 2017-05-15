
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>

#define SIZE 1024

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

	int init = threadIdx.x;
	__shared__ float temp[SIZE];
	while (init < SIZE)
	{
		temp[init] = 0;
		init += blockDim.x;
	}

	__syncthreads();

	{
		Wx = W[feature + category * features] * input[feature + batch * features];
		temp[feature] += Wx;
		__syncthreads();
		for (int thread = 1024 / 2; thread > 32; thread /= 2)
		{
			if (threadIdx.x < thread)
			{
				temp[threadIdx.x] += temp[threadIdx.x + thread];
			}
			__syncthreads();
		}

		if (threadIdx.x < 16)
		{
			volatile float* volatiletemp = temp;
			volatiletemp[threadIdx.x] += volatiletemp[threadIdx.x + 32];
			volatiletemp[threadIdx.x] += volatiletemp[threadIdx.x + 16];
			volatiletemp[threadIdx.x] += volatiletemp[threadIdx.x + 8];
			volatiletemp[threadIdx.x] += volatiletemp[threadIdx.x + 4];
			volatiletemp[threadIdx.x] += volatiletemp[threadIdx.x + 2];
			volatiletemp[threadIdx.x] += volatiletemp[threadIdx.x + 1];
			if (threadIdx.x == 0)
				score[category + categories * batch] = volatiletemp[0];
		}

	}
}
__global__ void update1(int categories, int features, int batches, int* labels,float* score,float* diff)
{
	int category = threadIdx.x;
	int batch = blockIdx.x;
//	extern __shared__ float diff[];
	__shared__ float temp[SIZE];

	diff[category + batch * categories] = exp(score[category + batch * categories]);

	int init = threadIdx.x;
	while (init < SIZE)
	{
		temp[init] = 0;
		init += blockDim.x;
	}
	__syncthreads();


	temp[category] += diff[category + batch * categories];
	__syncthreads();
	for (int thread = SIZE / 2; thread > 0; thread /= 2)
	{
		if (threadIdx.x < thread)
		{
			temp[threadIdx.x] += temp[threadIdx.x + thread];
		}
		__syncthreads();
	}

	diff[category + batch * categories] /= temp[0];
	if (category == labels[batch])
		diff[category + batch * categories]--;
}

__global__ void update2(int categories, int features, int batches,float* input, float* diff,float* diffW)
{
	int category = threadIdx.x;
	int batch = threadIdx.y;
	int feature = blockIdx.x;
	//	extern __shared__ float diff[];
	diffW[category + categories*batch + feature*categories*batches] = diff[category + batch * categories] * input[feature + batch * features];
}

__global__ void update3(int categories, int features, int batches, float* W, float* b, float* diff, float* diffW,float learningRate)
{
	int feature = threadIdx.x;
	int category = blockIdx.x;
	float batchDiffW = 0;
	float batchDiffb = 0;
	for (int batch = 0; batch < batches; batch++)
	{
		batchDiffW += learningRate * diffW[category + categories*batch + feature*categories*batches];
		if (feature == 0)
			batchDiffb += learningRate * diff[category + categories* batch];
	}
	W[feature + features * category] -= batchDiffW;
	if (feature == 0)
		b[category] -= batchDiffb;
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
	float* dev_diff;
	float* dev_diffW;
	int* dev_labels;
	cudaStream_t stream0;
	ONERROR(cudaStreamCreate(&stream0));
	ONERROR(cudaMalloc(&dev_x,sizeof(float)*features*batches));
	ONERROR(cudaMalloc(&dev_labels, sizeof(int)*batches));
	ONERROR(cudaMemcpyAsync(dev_x, x, sizeof(float)*features*batches, cudaMemcpyHostToDevice, stream0));
	ONERROR(cudaMemcpyAsync(dev_labels, label, sizeof(int)*batches, cudaMemcpyHostToDevice, stream0));
	ONERROR(cudaMalloc(&dev_score, sizeof(float)*categories*batches));
	ONERROR(cudaMalloc(&dev_diff, sizeof(float)*categories*batches));
	ONERROR(cudaMalloc(&dev_diffW, sizeof(float)*categories*batches*features));
/*
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	int a = properties.multiProcessorCount;
	int b =  properties.maxThreadsPerMultiProcessor;*/

	dim3 gridShape = dim3(categories, batches);
	dim3 blockShape = dim3(features);
	int shareMemSize = 1;


	multiply << <gridShape, blockShape,0, stream0 >> > (categories, features, batches, dev_W, dev_b, dev_x, dev_score);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	ONERROR(cudaMemcpyAsync(host_score, dev_score, sizeof(float)*categories*batches, cudaMemcpyDeviceToHost,stream0));
	ONERROR(cudaStreamSynchronize(stream0));




	update1 << <batches, categories,0, stream0 >> > (categories, features, batches, dev_labels, dev_score, dev_diff);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}




	update2 << <features, dim3(categories,batches),0, stream0 >> > (categories, features, batches,dev_x, dev_diff, dev_diffW);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	update3 << <categories, features,0, stream0 >> >(categories, features, batches, dev_W, dev_b, dev_diff, dev_diffW, learningRate);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	
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
	ONERROR(cudaStreamSynchronize(stream0));
	delete[] host_score;
	correctRate[0] = (float)correctCount / batches;

Error:
	cudaFree(dev_x);
	cudaFree(dev_labels);
	cudaFree(dev_score);
	cudaFree(dev_diff);
	cudaFree(dev_diffW);
	cudaStreamDestroy(stream0);
	return cudaStatus;
}
