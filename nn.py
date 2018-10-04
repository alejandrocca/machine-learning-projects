import torch
from torch.autograd import Variable
import torch.nn.functional as F #all activational functions
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1,1,100), dim = 1)
#transforming linspace dots into 2d data

y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x), Variable(y)


'''
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

plt.ion()
plt.show()
'''
'''
# Data
n_data = torch.ones(100,2) 
x0 = torch.normal(2*n_data,1) # type0 x data(tensor), shape = (100,2)
y0 = torch.zeros(100) # type0 y data(tensor), shape = (100,1)
x1 = torch.normal(-2*n_data, 1) # type1 x data(tensor), shape =(100,2)
y1 = torch.ones(100) #type1 ydata (tensor), shape = (100,1)

# Combining Data
x = torch.cat((x0,x1),0).type(torch.FloatTensor) # 32-bit float
y = torch.cat((y0,y1),).type(torch.LongTensor) # 64-bit int
'''

# Creating a neural network
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


		
net = Net(n_feature=1, n_hidden=10, n_output=1)
'''
print(net)
'''

plt.ion()
plt.show()

# Training with optimizer
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
# inputting all parameters of net and a learning rate

loss_func = torch.nn.MSELoss()
#loss_func = torch.nn.CrossEntropyLoss()

for t in range(100): #number of training steps
	prediction = net(x)
	
	loss = loss_func(prediction, y)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if t % 5 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(),y.data.numpy())
		plt.plot(x.data.numpy(),prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5,0,'Loss=%.4f' % loss.data[0], fontdict = {'size': 20, 'color':'red'})
		plt.pause(0.1)
		
plt.ioff()
plt.show()
	
	

'''
for t in range(100):
	out = net(x) # training data x to net, output the analyzed value
	
	loss = loss_func(out, y) #calculate error
	optimizer.zero_grad() #clear the previous step
	loss.backward() #bp the loss and calculate the updated val
	optimizer.step() #update the value to the parameters of net
	
	# Visualizing the training process
	
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
'''

