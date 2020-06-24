# Pytorch Projects
This is a directory for exploring pytorch. This repo contains projects for learning pytorch and exploring Computer Vision field. 

## PyTorch
Pytorch is a Neural Network Library, made for Computer Vision and NLP by Facebook Researchers. At it's core, PyTorch is a library for processing tensors. A tensor is a number, or a n-dimentional array (matrix, vector, array). 
```
import torch
t1 = torch.tensor(2.) 	#this creates a floating number 2.0 
t1.dtype 				#this will give out int


t2 = torch.tensor([1., 2, 3, 4])	#here, t2 is a array. 
									#Each element should have same datatype. Pytorch converts to same data-type
 

 #here, t3 is a 3-dimentional array
t3 = torch.tensor([
    [[11, 12, 13], 
     [13, 14, 15]], 
    [[15, 16, 17], 
     [17, 18, 19.]]])

 #Tensors must have uniformly shaped size, or an error is thrown

t3.shape 		#this gives out the shape of tensor => torch.Size([2, 2, 3])
```

### Tensor Operations
- Tensors can be combined via  basic arithmetic operations like +,-,*,/ etc
- Grads: Gradients can be taken into consideration by a simple parameter
	- Gradients are used heavily in gradient descent algorithm
```
# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b

# Arithmetic operations
y = w * x + b
y

# Compute derivatives
y.backward()

# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

```

### Interoperability with Numpy
Among other great features, pytorch's integration with numpy is pretty fantastic. Numpy can be used seamlessly ans so gives us access to matplotlib, opencv, pandas and other libraries with little setup. This makes pyTorch very powerful and fun to work with. Also, most datasets are likely shared to be preprocessed in numpy arrays anyeay. 

```
import numpy as np

x = np.array([[1, 2], [3, 4.]])
x

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)
y

# Convert a torch tensor to a numpy array
z = y.numpy()
z


```



