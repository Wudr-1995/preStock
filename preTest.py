from torch.utils.data.dataloader import DataLoader
from preModel import preModel
from myDataSet import myDataSet
import time
import numpy as np
import torch
import argparse
import torch.utils.data as Data

def get_parser():
	parser = argparse.ArgumentParser(description='ML')
	parser.add_argument("--inputData", default='./trainData')
	parser.add_argument("--modelInput", default='./model.pkl')
	parser.add_argument("--outPath", default='./model.pkl')
	parser.add_argument("--epoch", type=int)
	parser.add_argument("learnRate", type=int, default=0.0003)
	return parser

def testModel(model, dataLoader, outPath):
	since = time.time()
	for step, (X, Y) in enumerate(dataLoader):
		model.eval()
		pre = model(X)
		out = np.column_stack((pre, Y))
		fileOut = np.loadtxt(outPath)
		data = np.row_stack((fileOut, out))
		np.savetxt(outPath, data)

if __name__ == "__main__":
	torch.set_default_tensor_type(torch.FloatTensor)
	parser = get_parser()
	args = parser.parse_args()

	testData = args.inputData
	testDataSet = myDataSet(testData)
	testLoader = Data.dataloader(
		dataset=testDataSet,
		batch_size=64,
		shuffle=True,
		num_workers=1
	)
	model = torch.load(args.modelInput)
	testModel(model, testLoader, args.outPath)
