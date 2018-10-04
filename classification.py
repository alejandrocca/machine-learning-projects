import torch
from torch.autograd import Variable
import torch.nn.functional as F #all activational functions
import matplotlib.pyplot as plt

# Data
n_data = torch.ones(100,2) 
x0 = torch.normal(2*n_data,1) # type0 x data(tensor), shape = (100,2)
y0 = torch.zeros(100) # type0 y data(tensor), shape = (100,1)
x1 = torch.normal(-2*n_data, 1) # type1 x data(tensor), shape =(100,2)
y1 = torch.ones(100) #type1 ydata (tensor), shape = (100,1)

# Combining Data
x = torch.cat((x0,x1),0).type(torch.FloatTensor) # 32-bit float
y = torch.cat((y0,y1),).type(torch.LongTensor) # 64-bit int

x,y = Variable(x), Variable(y)


'''
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

'''


# Creating a neural network method 1
class Net(torch.nn.Module): #inheriting the module from torch
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net,self).__init__() #inheriting __init__ functions
		# assigning tools for each process layer
		self.hidden = torch.nn.Linear(n_feature, n_hidden) # regression output into the hidden layer
		self.predict = torch.nn.Linear(n_hidden, n_output) # regression output into the output layer
		
	def forward(self,x): # the forward function of Module
		# forwarding the input and neural analyze the output
		x = F.relu(self.hidden(x)) # activational function relu (the regression output of the hidden layer)
		x = self.predict(x) # output
		return x


# data x has two features(the x-axis and y-axis)		
net1 = Net(n_feature=2, n_hidden=10, n_output=2)
'''
print(net)
'''

# method 2
net2 = torch.nn.Sequential(
	torch.nn.Linear(2, 10),
	torch.nn.ReLU(),
	torch.nn.Linear(10, 2)
)

plt.ion()
plt.show()

# Training with optimizer
optimizer = torch.optim.SGD(net2.parameters(), lr = 0.02)
# inputting all parameters of net and a learning rate

#torch.nn.MSELoss() used for regression
loss_func = torch.nn.CrossEntropyLoss() #for classification


for t in range(100): #number of training steps
	out = net2(x) #F.softmax(out)
	
	loss = loss_func(out, y)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if t % 2 == 0:
		plt.cla()
		prediction = torch.max(F.softmax(out),1)[1]
		pred_y = prediction.data.numpy().squeeze()
		target_y = y.data.numpy()
		plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1], c=pred_y, s= 100, lw=0, cmap = 'RdYlGn')
		accuracy = sum(pred_y == target_y)/200.
		plt.text(1.5,-4,'Accuracy = %.2f' % accuracy, fontdict = {'size':20,'color':'red'})
		plt.pause(0.1)
		
plt.ioff()
plt.show()
	
	

