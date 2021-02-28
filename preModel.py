# import torchsnooper
import torch.nn as nn

class preModel(nn.Module):
	def __init__(self):
		super(preModel, self).__init__()
		self.regression = nn.Sequential(
			nn.Linear(17360, 8000),
			nn.ReLU(),
			nn.Linear(8000, 4000),
			nn.ReLU(),
			nn.Linear(4000, 2000),
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
		x = x.view(x.size(0), -1)
		pre = self.regression(x)
		return pre

	def test(self):
		print("It's right!")