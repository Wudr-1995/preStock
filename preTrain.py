from myDataSet import myDataSet
# import torchsnooper
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
import torch
import time
import copy
from torch.optim import Adam
from torch.optim import Adadelta
import argparse
from preModel import preModel
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_parser():
	parser = argparse.ArgumentParser(description='ML')
	parser.add_argument("--load", type=int, default=0)
	parser.add_argument("--inputData", default='./data/samTrainInput.txt')
	parser.add_argument("--labelData", default='./data/samTrainLabel.txt')
	parser.add_argument("--modelInput", default='./model.pkl')
	parser.add_argument("--outPath", default='./model.pkl')
	parser.add_argument("--epoch", type=int)
	parser.add_argument("--learnRate", type=int, default=0.0003)
	return parser

def trainModel(model, dataLoader, trainRate, criterion, optimizer, nEpoch, outPath):
	
	bestLoss = 1000
	trainLossAll = []
	since = time.time()

	for epoch in range(nEpoch):
		print('Epoch {}/{}'.format(epoch, nEpoch - 1))
		print('-' * 20)
		trainLoss = 0.0
		trainNum = 0

		for step, (X, Y) in enumerate(dataLoader):
			model.train()
			output = model(X)
			output = output[:, 0]
			loss = criterion(output, Y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			trainLoss += loss.item() * X.size(0)
			trainNum += X.size(0)

		trainLossAll.append(trainLoss / trainNum)
		print('{} Train Loss: {:.4f} '.format(epoch, trainLossAll[-1]))

		if trainLossAll[-1] < bestLoss:
			bestLoss = trainLossAll[-1]
		timeUse = time.time() - since
		print('Train and val complete in {:.0f}m {:.0f}'.format(timeUse // 60, timeUse % 60))
		torch.save(model, outPath)

if __name__ == "__main__":
	torch.set_default_tensor_type(torch.FloatTensor)
	parser = get_parser()
	args = parser.parse_args()
	if args.load == 0:
		initNet = preModel()
	else:
		initNet = torch.load(args.modelInput)

	trainData = args.inputData
	trainLabel = args.labelData
	trainDataSet = myDataSet(trainData, trainLabel)
	trainLoader = Data.DataLoader(
		dataset=trainDataSet,
		shuffle=True,
		batch_size=64,
		num_workers=1
	)

	optimizer = torch.optim.Adam(initNet.parameters(), lr=0.0003)
	lossFunc = nn.MSELoss()
	model, process = trainModel(initNet, trainLoader, 0.0003, lossFunc, optimizer, args.epoch, args.outPath)