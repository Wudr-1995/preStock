import numpy as np
# import torchsnooper
from torch.utils.data import Dataset
import torch

class myDataSet(Dataset):
	def __init__(self, path, ID=0, transform=torch.as_tensor, loader=np.loadtxt):
		super(myDataSet, self).__init__()
		self.loader = loader
		dataIn = self.loader(path, dtype=np.float32)
		length = np.size(dataIn, 0) - 19
		n = 0
		datas = []
		while n < length:
			datas.append((dataIn[n + 10 : n + 19], dataIn[n, ID]))
			n += 1
		self.datas = datas
		self.transform = transform
		
	def __getitem__(self, index):
		dataX, Label = self.datas[index]
		dataX = np.expend_dims(dataX, 0)
		dataX = self.transform(dataX)
		return dataX, Label
	
	def __len__(self):
		return len(self.datas)
