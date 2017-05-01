#include"softmax.h"

extern "C"
cudaError_t mallocInitPara(float** dev_W, float** dev_b, int features, int categories, float* host_W, float* host_b);
extern "C"
cudaError_t updatePara_SGD(float* x, int* label, int batches, float* backPropagationForInput, float* dev_W, float* dev_b, int features, int categories, float learningRate, float* correctRate);
void softmax::cleanup_Resize(int categoryNum, int featureNum)
{
	categories = categoryNum;
	features = featureNum;
	if (!gpu)
	{
		if (W != 0)
			delete[] W;
		if (b != 0)
			delete[] b;
		if (categoryNum != 0 && featureNum != 0)
		{
			W = new float[categoryNum * featureNum];
			b = new float[categoryNum];
			const float initial_mean = 0.0;
			const float initial_variance = 0.05;
			static std::default_random_engine e;
			e.seed(time(0));
			static std::normal_distribution<float> normal(initial_mean, initial_variance);

			for (int category = 0; category < categoryNum; category++)
			{
				for (int feature = 0; feature < featureNum; feature++)
				{
					W[feature + category * featureNum] = normal(e);
				}
				b[category] = normal(e);
			}
		}
	}
	else
	{
		if (W != 0)
			cudaFree(W);
		if (b != 0)
			cudaFree(b);

		if (categoryNum != 0 && featureNum != 0)
		{
			float* W_host = new float[categoryNum * featureNum];
			float* b_host = new float[categoryNum];
			const float initial_mean = 0.0;
			const float initial_variance = 0.05;
			static std::default_random_engine e;
			e.seed(time(0));
			static std::normal_distribution<float> normal(initial_mean, initial_variance);

			for (int category = 0; category < categoryNum; category++)
			{
				for (int feature = 0; feature < featureNum; feature++)
				{
					W_host[feature + category * featureNum] = normal(e);
				}
				b_host[category] = normal(e);
			}
			mallocInitPara(&W, &b, features, categories, W_host, b_host);
			delete[] W_host;
			delete[] b_host;
		}
	}
}
void softmax::setLearningRate(float rate)
{
	learningRate = rate;
}
softmax::~softmax()
{
	cleanup_Resize();
}

void softmax::useGPU(bool GPU)
{
	gpu = GPU;
	if(gpu)
		cudaSetDevice(0);
}

float softmax::test(float* x, int* label, int batches)
{
	int correct = 0;
	for (int batch = 0; batch < batches; batch++)
	{
		float max = -99999999.0;
		int max_label = 0;
		for (int category = 0; category < categories; category++)
		{
			float result = 0;
			for (int feature = 0; feature < features; feature++)
			{
				result += W[category * features + feature] * x[features * batch + feature];
			}
			result += b[category];
			max_label = result > max ? category : max_label;
			max = result > max ? result : max;
		}
		if (max_label == label[batch])
			correct++;
	}
	return (float)correct / batches;
}






float softmax::updatePara(float* x,int* label, int batches, float* backPropagationForInput)//SGD
{
	if (gpu)
	{
		float correctRate;

//		updatePara_SGD(x, label, batches, 0, W, b, features, categories, learningRate,&correctRate);
		updatePara_SGD(x, label, batches, 0, W, b, features, categories, learningRate, &correctRate);
		return correctRate;
	}
	float* Wx_plus_b = new float[batches * categories];
	float* diff_W = new float[categories * features];
	float* diff_b = new float[categories];
	float* diff = new float[batches * categories];    // assume z = Wx+b ,then P = ∂(outputLoss)/∂z
	memset(diff_W, 0, sizeof(float) * categories * features);
	memset(diff_b, 0, sizeof(float) * categories);
	memset(diff, 0, sizeof(float) * batches * categories);

	float finalLoss = 0;//new float[batches];
	int correct = 0;
	for (int batch = 0; batch < batches; batch++)
	{
		float sum = 0;
		float max = -99999999.0;
		int max_label = 0;
		for (int category = 0; category < categories; category++)
		{
			float result = 0;
			for (int feature = 0; feature < features; feature++)
			{
				result += W[category * features + feature] * x[features * batch + feature];
			}
			result += b[category];
			Wx_plus_b[category + categories * batch] = result;
			max_label = result > max ? category : max_label;
			max = result > max ? result : max;
			sum += exp(result);
		}

		finalLoss += - log (exp(Wx_plus_b[label[batch] + categories * batch]) / sum);

		if (max_label == label[batch])
			correct++;
		for (int category = 0; category < categories; category++)
		{
			float z = exp(Wx_plus_b[category + categories * batch]);
			diff[category + categories * batch] = label[batch] == category ? (z / sum - 1) : z / sum;
			for (int feature = 0; feature < features; feature++)
			{
				diff_W[feature + category * features] += diff[category + categories * batch] * x[features * batch + feature];
			}
			diff_b[category] += 1 * diff[category + categories * batch];
		}

		if (backPropagationForInput != NULL)
		{
			for (int feature = 0; feature < features; feature++)
			{
				float sum_bp = 0;
				for (int category = 0; category < categories; category++)
				{
					sum_bp += W[feature + category * features] * diff[category + categories * batch];
				}
				backPropagationForInput[feature + features * batch] = sum_bp;
			}
		}

	}


	for (int category = 0; category < categories; category++)
	{
		for (int feature = 0; feature < features; feature++)
		{
			float update = diff_W[feature + category * features] * learningRate;
			W[category * features + feature] -= update;
		}
		b[category] -= (diff_b[category] * learningRate);
	};

	delete[] diff;
	delete[] diff_W;
	delete[] diff_b;
	delete[] Wx_plus_b;
	return (float)correct / batches;
}