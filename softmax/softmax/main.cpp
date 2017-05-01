#include"MNIST.h"
#include"softmax.h"
#include<iostream>

int main()
{
	MNIST mnist;
	mnist.load();
	softmax classifier;
	classifier.useGPU(1);
	int batch_count = 50;

	classifier.cleanup_Resize(10, mnist.getWidth()*mnist.getHeight());
	classifier.setLearningRate(0.00001);
	for (int count = 0; count < 50000; count++)
	{
		unsigned char* input;
		unsigned char* type = mnist.getBatchFromTrainSet(batch_count, &input);

		// Transfer the type
		float* x = new float[batch_count * mnist.getWidth()*mnist.getHeight()];
		int* label = new int[batch_count];
		for (int i = 0; i < batch_count*mnist.getWidth()*mnist.getHeight(); i++)
		{
			x[i] = (float)input[i] / 256;
		}
		delete[] input;
		for (int i = 0; i < batch_count; i++)
		{
			label[i] = type[i];
		}
		delete[] type;
		float correct_rate = classifier.updatePara(x, label, batch_count,0);
		if(count % 1000 == 0)
			std::cout << "training " << count << " times, correct rate is " << correct_rate << std::endl;
		delete[]x;
		delete[] label;
	}
	/*
	{
		unsigned char* input;
		unsigned char* type = mnist.getTestSet(&input);

		// Transfer the type
		float* x = new float[batch_count * mnist.getWidth()*mnist.getHeight()];
		int* label = new int[batch_count];
		for (int i = 0; i < batch_count*mnist.getWidth()*mnist.getHeight(); i++)
		{
			x[i] = (float)input[i] / 256;
		}
		delete[] input;
		for (int i = 0; i < batch_count; i++)
		{
			label[i] = type[i];
		}
		delete[] type;
		float correct_rate = classifier.test(x, label, batch_count);
		std::cout << "On test set, correct rate is " << correct_rate << std::endl;
		delete[] x;
		delete[] label;
	}
	*/

	system("pause");
}