# import torchsnooper
import torch.nn as nn

class preModel(nn.Module):
	def __init__(self):
		super(preModel, self).__init__()
		self.regression = nn.Sequential(
			nn.Linear(41000, 4100),
			nn.ReLU(),
			nn.Linear(4100, 2000),
			nn.ReLU(),
			nn.Linear(2000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 200),
			nn.ReLU(),
			nn.Linear(200, 50),
			nn.ReLU(),
			nn.Linear(50, 1)
		)
	
	def forward(self, x):
		x = x.float()
		pre = self.regression(x)
		return pre

	def test(self):
		print("It's right!")