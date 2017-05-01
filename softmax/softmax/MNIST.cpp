#include"MNIST.h"

int32_t Reverse(int32_t i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 0xFF;
	ch2 = (i >> 8) & 0xFF;
	ch3 = (i >> 16) & 0xFF;
	ch4 = (i >> 24) & 0xFF;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

MNIST::~MNIST()
{
	for (int i = 0; i < number_of_train_image; i++)
	{
		delete[] train_image[i];
	}
	for (int i = 0; i < number_of_test_image; i++)
	{
		delete[] test_image[i];
	}
}

bool MNIST::load(std::string directory)
{
	std::string train_image_path = directory + TRAIN_IMAGES;
	std::string train_label_path = directory + TRAIN_LABELS;
	std::string test_image_path = directory + TEST_IMAGES;
	std::string test_label_path = directory + TEST_LABELS;

	std::ifstream fp;
	fp.open(train_image_path, std::ios::binary| std::ios::in);
	if (!fp.is_open())
	{
		std::cout << "Error opening file " << train_image_path << std::endl;
		return false;
	}
	int32_t magic_num;
	fp.read((char*)&magic_num, sizeof(int32_t));
	magic_num = Reverse(magic_num);
	assert(magic_num == 2051);

	fp.read((char*)&number_of_train_image, sizeof(int32_t));
	number_of_train_image = Reverse(number_of_train_image);

	fp.read((char*)&train_image_row, sizeof(int32_t));
	train_image_row = Reverse(train_image_row);

	fp.read((char*)&train_image_col, sizeof(int32_t));
	train_image_col = Reverse(train_image_col);

	train_image.clear();
	train_image.resize(number_of_train_image);
	for (int i = 0; i < number_of_train_image; i++)
	{
		train_image[i] = new unsigned char[train_image_col * train_image_row];
		fp.read((char*)train_image[i], sizeof(unsigned char) * train_image_col * train_image_row);
	}
	fp.close();

	fp.open(train_label_path, std::ios::binary | std::ios::in);
	if (!fp.is_open())
	{
		std::cout << "Error opening file " << train_label_path << std::endl;
		return false;
	}
	fp.read((char*)&magic_num, sizeof(int32_t));
	magic_num = Reverse(magic_num);
	assert(magic_num == 2049);

	fp.read((char*)&number_of_train_label, sizeof(int32_t));
	number_of_train_label = Reverse(number_of_train_label);

	train_label.clear();
	train_label.resize(number_of_train_label);
	for (int i = 0; i < number_of_train_label; i++)
	{
		fp.read((char*)&train_label[i], sizeof(unsigned char));
	}
	fp.close();

	//////////////////////////

	fp.open(test_image_path, std::ios::binary | std::ios::in);
	if (!fp.is_open())
	{
		std::cout << "Error opening file " << test_image_path << std::endl;
		return false;
	}

	fp.read((char*)&magic_num, sizeof(int32_t));
	magic_num = Reverse(magic_num);
	assert(magic_num == 2051);

	fp.read((char*)&number_of_test_image, sizeof(int32_t));
	number_of_test_image = Reverse(number_of_test_image);

	fp.read((char*)&test_image_row, sizeof(int32_t));
	test_image_row = Reverse(test_image_row);

	fp.read((char*)&test_image_col, sizeof(int32_t));
	test_image_col = Reverse(test_image_col);

	test_image.clear();
	test_image.resize(number_of_test_image);
	for (int i = 0; i < number_of_test_image; i++)
	{
		test_image[i] = new unsigned char[test_image_col * test_image_row];
		fp.read((char*)test_image[i], sizeof(unsigned char) * test_image_col * test_image_row);
	}
	fp.close();



	fp.open(test_label_path, std::ios::binary | std::ios::in);
	if (!fp.is_open())
	{
		std::cout << "Error opening file " << train_label_path << std::endl;
		return false;
	}
	fp.read((char*)&magic_num, sizeof(int32_t));
	magic_num = Reverse(magic_num);
	assert(magic_num == 2049);

	fp.read((char*)&number_of_test_label, sizeof(int32_t));
	number_of_test_label = Reverse(number_of_test_label);

	test_label.clear();
	test_label.resize(number_of_test_label);
	for (int i = 0; i < number_of_test_label; i++)
	{
		fp.read((char*)&test_label[i], sizeof(unsigned char));
	}
	fp.close();
	return true;
}

int MNIST::getWidth()
{
	return test_image_row;
}
int MNIST::getHeight()
{
	return test_image_col;
}

unsigned char MNIST::getImageFromTrainSet(int index, unsigned char** image)
{
	if (index >= train_image.size())
		return 10;	//Invalid vaule
	*image = train_image[index];
	return train_label[index];
}

unsigned char MNIST::getImageFromTestSet(int index, unsigned char** image)
{
	if (index >= test_image.size())
		return 10;	//Invalid vaule
	*image = test_image[index];
	return test_label[index];
}

unsigned char* MNIST::getTestSet(unsigned char** pixel)
{
	int batch = test_image.size();
	*pixel = new unsigned char[batch * test_image_row * test_image_col];
	unsigned char* labels = new unsigned char[batch];
	srand((unsigned)time(NULL));
	for (int i = 0; i < batch; i++)
	{
		unsigned char* image;
		unsigned char label = getImageFromTestSet(rand() % test_image.size(), &image);
		memcpy((*pixel) + i * test_image_row * test_image_col, image, sizeof(unsigned char) * test_image_row * test_image_col);
		labels[i] = label;
	}
	return labels;
}


unsigned char* MNIST::getBatchFromTrainSet(int batch, unsigned char** pixel)
{
	*pixel = new unsigned char[batch * train_image_row * train_image_col];
	unsigned char* labels = new unsigned char[batch];
	srand((unsigned)time(NULL));
	for (int i = 0; i < batch; i++)
	{
		unsigned char* image;
		unsigned char label = getImageFromTrainSet(rand() % train_image.size(), &image);
		memcpy((*pixel) + i * train_image_row * train_image_col, image, sizeof(unsigned char) * train_image_row * train_image_col);
		labels[i] = label;
	}
	return labels;
}