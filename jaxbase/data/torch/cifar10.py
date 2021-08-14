import torchvision
import torchvision.transforms as T
import numpy as np

CIFAR10_MEAN = np.array([125.30691805, 122.95039414, 113.86538318])
CIFAR10_STD = np.array([62.99321928, 62.08870764, 66.70489964])

def OneHot(x):
	return np.eye(10)[x]
	
def CIFAR10_Normalize(x):
	x = np.asarray(x, dtype=np.float32)
	x = (x - CIFAR10_MEAN) / CIFAR10_STD
	return x

def numpy(data_dir, download=False, normalize=True, one_hot=True):
	traindata = torchvision.datasets.CIFAR10(data_dir, train=True, download=download)
	testdata = torchvision.datasets.CIFAR10(data_dir, train=False, download=download)
	train_x, train_y, test_x, test_y = traindata.data, traindata.targets, testdata.data, testdata.targets
	if normalize:
		train_x, test_x = CIFAR10_Normalize(train_x), CIFAR10_Normalize(test_x)
	if one_hot:
		train_y, test_y = OneHot(train_y), OneHot(test_y)
	return train_x,train_y,test_x,test_y

def torch(data_dir, download=False, normalize=True, one_hot=True, data_aug=False):
	normalizations = [CIFAR10_Normalize] if normalize else []
	augmentations = [T.RandomCrop(32, padding=4),T.RandomHorizontalFlip()] if data_aug else []
	transform_train = T.Compose(augmentations + normalizations)
	transform_test = T.Compose(normalizations)
	target_transform = OneHot if one_hot else None
	traindata = torchvision.datasets.CIFAR10(data_dir,
						 transform=transform_train,
						 target_transform=target_transform,
						 train=True, 
						 download=download)
	testdata = torchvision.datasets.CIFAR10(data_dir,
						transform=transform_test,
						target_transform=target_transform,
						train=False,
						download=download)
	return traindata, testdata
