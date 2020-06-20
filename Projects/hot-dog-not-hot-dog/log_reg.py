# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy
import imageio
from PIL import Image
from lr_utils import load_dataset
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()

# config - change these to run the code natively

test_hot_dogs_dir = "/Users/shivanshsuhane/Desktop/code/hot-dog-not-hot-dog/datasets/test/hot_dog"
test_not_hot_dogs_dir = "/Users/shivanshsuhane/Desktop/code/hot-dog-not-hot-dog/datasets/test/not_hot_dog"

train_hot_dogs_dir = "/Users/shivanshsuhane/Desktop/code/hot-dog-not-hot-dog/datasets/train/hot_dog" 
train_not_hot_dogs_dir = "/Users/shivanshsuhane/Desktop/code/hot-dog-not-hot-dog/datasets/train/not_hot_dog"

dimention = 64         #convert any img to 64x64 img in preprocessing


#returns train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
def load_dataset_native():
    
    #1. convert Image dirs to numpy arrays 4x
        #1. call preProcess on each image then add image to array
    
    print("Loading the datasets to numpy arrays")
    test_hd_arr = imageDirToNumpyArr(test_hot_dogs_dir, dimention)          #1 res
    test_nhd_arr = imageDirToNumpyArr(test_not_hot_dogs_dir,dimention)      #0 res
    
    train_hd_arr = imageDirToNumpyArr(train_hot_dogs_dir, dimention)        #1 res
    train_nhd_arr = imageDirToNumpyArr(train_not_hot_dogs_dir, dimention)   #0 res
    

    #2. flatten the array, add the result
    print("Flattening the images to [64*64*3 x 1] dimentions")
    flat_test_hd = flattenAndAddResult(test_hd_arr,1)
    flat_test_nhd = flattenAndAddResult(test_nhd_arr,0)
    flat_train_hd = flattenAndAddResult(train_hd_arr,1)
    flat_train_nhd = flattenAndAddResult(train_nhd_arr,0)
    
    
    #2. Merge the two types of testing, and two types of training data to 2 classes, then mix
    
    flat_test = mergeAndMixArrays(flat_test_hd, flat_test_nhd)
    flat_train = mergeAndMixArrays(flat_train_hd, flat_train_nhd)
        
    #3. Separate the y (last column from train and test)
    yTest = flat_test[-1]
    yTrain = flat_train[-1]
    
    #delete the last row as its saved in the y
    #np.delete(flat_test, -1, 0)
    #np.delete(flat_train, -1, 0)
    
    return flat_train[:-1], yTrain, flat_test[:-1], yTest

#returns a numpy array from the directory address for an image directory
def imageDirToNumpyArr(im_dir_path, zero_or_one):
    arr = os.listdir(im_dir_path)
    result_arr = np.empty([len(arr), dimention, dimention, 3], dtype='int16');
    
    # preprocess each image
    for i in range(0,len(arr)):
        f_path = im_dir_path + '/' +arr[i]
        if f_path == im_dir_path + '/.DS_Store':
            continue
        im_array=preprocess(f_path, dimention)
        result_arr[i]= im_array
    
    return result_arr
    

def preprocess(image_path, num_px):
    print()
    print("Preprocessing Image: ", image_path)
    image = Image.open(image_path)
    print("Image Size: ",image.size)
    print()
    
    resized_image = image.resize((num_px,num_px))
    print("Image Size: ",resized_image.size)
    image_arr = np.array(resized_image)
    return image_arr

    
def flattenAndAddResult(arr, res_zero_or_one):
    # Flatten
    flat = arr.reshape(arr.shape[0], -1).T
    # Add the result ->?
    
    print ("arr shape: " + str(flat.shape))
    
    # standardize
    flat=flat/255       # divide every row of the dataset by 255 (the maximum value of a pixel channel)
    
    # Add a row of res_zero_or_one
    if(res_zero_or_one == 1):
        newRow = np.ones((1, flat.shape[1]),dtype=int)
        res = np.vstack((flat,newRow))
        
    else:
        newRow = np.zeros((1, flat.shape[1]),dtype=int)
        res = np.vstack((flat,newRow))
            
    return res

    

# merge teain test subarrays, mix them afterwordss
def mergeAndMixArrays(flat_arr1, flat_arr2):
    print("Merging, mixing data")
    
    # merge
    res = np.column_stack((flat_arr1, flat_arr2))
    
    # mix and return
    np.random.shuffle(np.transpose(res))
    return res
    

    
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1 + np.exp(-z))
    
    return s



# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b



def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X)+b)                                                               # compute activation
    cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))* 1 / (- m)                     # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X,(A-Y).T) * 1 / ( m)
    db = np.sum(A-Y) * 1 / ( m)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


epsilon = 1e-5    



# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w,b,X,Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, False)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

if __name__ == "__main__":
    # Loading the data hot-dog/not-hot-dog
    train_set_x, train_set_y, test_set_x, test_set_y = load_dataset_native()
    
    m_train = len(train_set_x)
    m_test = len(test_set_x)
    num_px = train_set_x.shape[1]
    
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))
    
   # test_set_y = [test_set_y]
    #test_set_y = np.array(test_set_y).T

   # train_set_y = [train_set_y]
  #  train_set_y = np.array(train_set_y).T
    
   # train_set_x = train_set_x.T
   # test_set_x = test_set_x.T
    
   # logisticRegr.fit(train_set_x, train_set_y)
   # predictions = logisticRegr.predict(test_set_x)
   # score = logisticRegr.score(test_set_x, test_set_y)
   # print(score)
    
    print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
    dim = 2
    
    
    
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))
    
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    grads, cost = propagate(w, b, X, Y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))
    
    params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

    print ("w = " + str(params["w"]))
    print ("b = " + str(params["b"]))
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    
    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
    print ("predictions = " + str(predict(w, b, X)))
    
    print("Training the model")
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.002, print_cost = True)

    # Example of a picture that was wrongly classified.
    index = 1
    #plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    #print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
    
    
    # Plot learning curve (with costs)
    #costs = np.squeeze(d['costs'])
    #plt.plot(costs)
    #plt.ylabel('cost')
    #plt.xlabel('iterations (per hundreds)')
    #plt.title("Learning rate =" + str(d["learning_rate"]))
    #plt.show()
    
    learning_rates = [0.02, 0.01, 0.005]
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')
    
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    
    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    
    ## PUT YOUR IMAGE NAME) 
    my_image = "my_image2.jpg"   # change this to the name of your image file 
    
    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(imageio.imread(fname))
    my_image = Image.fromarray(image)
    my_image = my_image.resize(size=(dimention,dimention))
    my_image = np.reshape(my_image, (1, dimention*dimention*3)).T
    
    main_model = models[str(0.02)]
    my_predicted_image = predict(d["w"], d["b"], my_image)
    print("The given image is Hotdog: " + str(my_predicted_image))
    
    plt.imshow(image)
    #print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    
    
        
    
    
    

    
    
    
    
    
    
    