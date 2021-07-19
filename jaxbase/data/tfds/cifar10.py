import tensorflow_datasets as tfds
import numpy as np

def np_data(data_dir, download=False, normalize=True, one_hot=True):
	ds = tfds.load('cifar10', split=['train','test'], batch_size=-1, as_supervised=True, data_dir=data_dir, download=download)
	(train_x,train_y),(test_x,test_y) = tfds.as_numpy(ds)
	if normalize:
		data_mean = train_x.mean((0,1,2))
		data_std = train_x.std((0,1,2))
		train_x, test_x = (train_x-data_mean)/data_std, (test_x-data_mean)/data_std
	if one_hot:
		train_y, test_y = np.eye(10)[train_y], np.eye(10)[test_y]
	return train_x,train_y,test_x,test_y