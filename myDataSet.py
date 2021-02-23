import numpy as np
# import torchsnooper
from torch.utils.data import Dataset
import torch

class myDataSet(Dataset):
	def __init__(self, Input, Label, transform=torch.as_tensor, loader=np.loadtxt):
		super(myDataSet, self).__init__()
		self.loader = loader
		dataIn = self.loader(Input, dtype=np.float32)
		labelIn = self.loader(Label, dtype=np.float32)
		length = np.size(labelIn, 0)
		n = 0
		datas = []
		while n < length:
			datas.append((dataIn[n * 10 : n * 10 + 10], labelIn[n]))
			n += 1
		self.datas = datas
		self.transform = transform
		
	def __getitem__(self, index):
		dataX, Label = self.datas[index]
		dataX = np.expand_dims(dataX, 0)
		dataX = self.transform(dataX)
		return dataX, Label
	
	def __len__(self):
		return len(self.datas)
