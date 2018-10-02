'''
Name : Xingjia Wang
Date : 10/02/2018
Project Name: PyTorch Based MNIST Demo with Linear Models
Project Description: 
	A simple machine learning project with a self-built neural network linear model based on the 
MNIST handwritten number training (60000 samples) and testing datasets (10000 samples.) Training takes 
10 epochs, and each epoch is set with a batch size of 64 samples. The accuracy after 10 epochs reached
96% on my computer. 
'''

import torch

from torch.autograd import Variable # Container for tensors which a neural network model will take in

import torch.nn as nn # For building neural network
import torch.nn.functional as F # Activation functions

import torch.optim as optim # Optimizer

import torchvision # MNIST datasets
import torchvision.transforms as transforms # Data to tensor transformation

# ---------------- PREP -----------------

# Training settings
#
# Note:
#	1 epoch = 1 time training through the entire dataset
#	1 iteration = 1 time training through the set of BATCH_SIZE number of samples

BATCH_SIZE = 64

# Function to transform mnist pics to tensors
transform = transforms.ToTensor()

# Load Datasets from MNIST
train_set = torchvision.datasets.MNIST('./mnist_data/', train = True, download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True, num_workers = 2) 

test_set = torchvision.datasets.MNIST('./mnist_data/', train = False, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle = False, num_workers = 2) 
# Note-to-self question: What is the significance of num_workers?

# ------------ MODEL BUILDING ------------

# Building the neural network model

class nnMNIST(nn.Module):
	def __init__(self):
		super(nnMNIST, self).__init__()		# inheriting class structure
		self.lin1 = nn.Linear(28*28, 512)	# MNIST images have sizes of 28*28 pixels
		self.lin2 = nn.Linear(512, 256)
		self.lin3 = nn.Linear(256, 128)
		self.lin4 = nn.Linear(128, 10)		# four hidden layers in total, outputting 0-9 (10 in total)
		
	def forward(self, x):
		x = x.view(-1, 784) 			# flattening the original 2d dataset
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))		# calling activation function through each inner layer
		y_hat = self.lin4(x)
		return y_hat
		

model = nnMNIST()

# Parameters for Training

criterion = nn.CrossEntropyLoss()					# Loss function with softmax
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)	# Optimizing with gradient descent, learning rate = 0.01
									# also consider optim.Adam()

# Function to train the neural network
def train(epoch):
	model.train()
	for batch_pos, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)		# Tensors -> Variables
		optimizer.zero_grad()					# Standardized step: need to look up why this step... something to clear out?
		output = model(data)					# Put data through the network model and get the output
		loss = criterion(output, target)			# Calculate loss
		loss.backward()						# Standardized step: something related to gradient descent... also need to look up
		optimizer.step()					# Standardized step: same as above... I don't quite remember

		# Screen output
		# Sample: Train Epoch: 1 [0/60000] (0%)    Loss: 2.000000
		if batch_pos % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_pos * len(data), len(train_loader.dataset), 100. * batch_pos / len(train_loader), loss.data[0])) 

# Function to test the neural network
def test():
	model.eval()		# I don't remember why... need to look up
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data, target = Variable(data, volatile = True), Variable(target)
		output = model(data)
		test_loss += criterion(output, target).data[0]		# sum up batch loss
		pred = output.data.max(1, keepdim = True)[1]		# get index of max
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)

	# Screen output
	# Sample: Test set: Average loss: 0.1000, Accuracy: 9700/10000 (97%)
	print ('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset))) 

# ---------------- MAIN -----------------

		
for epoch in range(1, 11):
	train(epoch)
	test()

