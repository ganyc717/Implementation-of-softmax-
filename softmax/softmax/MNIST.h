#pragma once
#include <string>
#include <iostream>  
#include <fstream>  
#include <cstdlib>
#include <time.h>
#include <assert.h>
#include <vector>

//#include <sstream>

#define TRAIN_IMAGES "train-images.idx3-ubyte"
#define TRAIN_LABELS "train-labels.idx1-ubyte"
#define TEST_IMAGES "t10k-images.idx3-ubyte"
#define TEST_LABELS "t10k-labels.idx1-ubyte"


class MNIST
{
public:
	bool load(std::string directory = "");
	unsigned char getImageFromTrainSet(int, unsigned char**);
	unsigned char getImageFromTestSet(int, unsigned char**);
	int getWidth();
	int getHeight();
	unsigned char* getBatchFromTrainSet(int batch, unsigned char** pixel);
	unsigned char* getTestSet(unsigned char** pixel);
	~MNIST();
private:
	int number_of_test_image = 0;
	int number_of_test_label = 0;
	int number_of_train_image = 0;
	int number_of_train_label = 0;

	int test_image_col = 0;
	int test_image_row = 0;
	int train_image_col = 0;
	int train_image_row = 0;
	std::vector<unsigned char* > train_image;
	std::vector<unsigned char> train_label;
	std::vector<unsigned char* > test_image;
	std::vector<unsigned char> test_label;
};