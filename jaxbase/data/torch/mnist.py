import torchvision
import numpy as np

def OneHot(x):
	return np.eye(10)[x]

def Flatten(x):
	return x.reshape(x.shape[0],-1)

def numpy(data_dir, download=False, flatten=True, one_hot=True, normalize=True):
	traindata = torchvision.datasets.MNIST(data_dir, train=True, download=download)
	testdata = torchvision.datasets.MNIST(data_dir, train=False, download=download)
	train_x, train_y, test_x, test_y = traindata.data, traindata.targets, testdata.data, testdata.targets
	if flatten:
		train_x, test_x = Flatten(train_x), Flatten(test_x)
	if one_hot:
		train_y, test_y = OneHot(train_y), OneHot(test_y)
	if normalize:
		train_x, test_x = np.array(train_x,dtype=np.float32)/255, np.array(test_x,dtype=np.float32)/255
	return train_x,train_y,test_x,test_y
