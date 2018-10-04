import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#fake data
x = torch.linspace(-5,5,200) # x data (tensor), shape = (100,1)
x = Variable(x)
x_np = x.data.numpy() #plotting

y_relu = F.relu(x).data.numpy()
y_sigmoid = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
#y_softmax = F.softmax(x)

#Plotting
plt.figure(1,figsize = (8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='red',label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c = 'red', label = 'sigmoid')
plt.ylim((-0.2,1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c = 'red', label = 'tanh')
plt.ylim((-1.2,1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c = 'red', label = 'sigmoid')
plt.ylim((-0.2,6))
plt.legend(loc='best')

plt.show()
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad = True)

#print(tensor)
#print(variable)

t_out = torch.mean(tensor*tensor) # x^2
v_out = torch.mean(variable*variable)

#print(t_out)
#print(v_out)
#print(variable.grad)
v_out.backward()
#print(variable.grad)

#print(variable)
#print(variable.data)
#print(variable.data.numpy())


