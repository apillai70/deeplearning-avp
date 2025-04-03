"""1(Python Basic with Numpy) of deeplearning.ai"""
#Attention: this is my practice of deeplearning.ai, so please do not copy anything from it, thanks!
#Sigmoid f
### 1 START CODE HERE ###
import math 
def basic_sigmoid(x):
	s = 1/(1 + math.exp(-x))
	return s
### END CODE HERE ###

#numpy
### 2 START CODE HERE ###
import numpy as np
def Sigmoid(x):
	s = 1/(1 + np.exp(-x))
	return s
x = np.array([1,2,3])
print("x = " + str(x))
print("Sigmoid(x) = " + str(Sigmoid(x)))
### END CODE HERE ###

#Sigmoid gradient
### 3 START CODE HERE ###
def sigmoid_derivative(x):
	ds = Sigmoid(x) * (1 - Sigmoid(x))
	return ds
print("Sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
### END CODE HERE ###

#Image
### 4 START CODE HERE ###
def image2vector(image):
	v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
	return v
image = np.random.rand(3,3,2)
print("image2vector(image) = " + str(image2vector(image)))
### END CODE HERE ###

#Noralizing rows
### 5 START CODE HERE ###
def normalizeRows(x):
	x_norm = np.linalg.norm(x,ord = 2,axis = 1,keepdims = True)
	x_normalized = x/x_norm
	return x_normalized
x =  np.array([[1,1],[2,2]])
print("normalizeRows(x) = " + str(normalizeRows(x)))
### END CODE HERE ###

#Broadcasting and the softmax function
### 6 START CODE HERE ###
def softmax(x):
	x_exp = np.exp(x)
	x_sum = np.sum(np.exp(x),axis = 1,keepdims = True)
	s = x_exp/x_sum
	return s
x = np.array([[9,2,5,0,0],[7,5,0,0,0]])
print("softmax(x) = " + str(softmax(x)))
### END CODE HERE ###

#Vectorization
import time

x1 = [9,2,5,0,0,7,5,0,0,0,9,2,5,0,0]
x2 = [9,2,2,9,0,9,2,5,0,0,9,2,5,0,0]

### CLASSIC DOT PRODUCT OF VECTOR IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
	dot += x1[i] * x2[i]
toc = time.process_time()
print(toc,tic)
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")
### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic =time.process_time()
outer = np.zeros((len(x1),len(x2)))
for i in range(len(x1)):
	for j in range(len(x2)):
		outer[i,j] = x1[i] * x2[j]
toc = time.process_time()
print("Outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

# L1 loss function
### 7 START CODE HERE ###
def L1(y_hat, y):
	"""
	Arguments:
	y_hat -- vector of size m(predicted labels)
	y -- vector of size m(true labels)
	returns:
	loss -- the value of the L1 loss function defined above
	"""
	s = np.sum(np.abs(y - y_hat))
	return s
y_hat = np.array([.9,0.2,0.1,.4,.9])
y = np.array([1,0,0,1,1])
print("L1 = " + str(L1(y_hat,y)))
### END CODE HERE ###

# L2 loss function
###  START CODE HERE ###
def L2(y_hat,y):
	s = np.dot(y - y_hat, y - y_hat)
	#s = np.sum(np.square(y - y_hat))
	return s
print("L2 = " + str(L2(y_hat,y)))
### END CODE HERE ###