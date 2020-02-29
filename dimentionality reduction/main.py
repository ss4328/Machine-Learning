import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.linalg import eig
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def standardize(data):
    for obs in data:
        mean = np.mean(obs,0)
        std = np.std(obs,0)
        for elem in obs:
            elem -= mean
            elem /= std

# def standardize(data):
#     imgSize = len(data[0])
#     #over 1600 elems (cols)
#     for i in range(0, imgSize):
#         meanI = mean(data, axis=i)
#         stdI = std(data, axis=i)
#         # data[i] = (data[i] - mean(data, axis=i)) / std(data, axis=i)  not working??
#         #over 154 elems(rows)
#         for j in range(0,len(data)):
#             data[i][j] = (data[i][j] - meanI) / meanI

# def standardize(data):
#     data = (data - mean(data)) / std(data)


def getMean(data)->[]:
    res=[]
    sum=0
    for i in range(0, len(data[0])):
        sum+=data[i][count]

    return res


print("welcome to hw1")
path = os.getcwd() + '/yalefaces'
print("Current path: " + os.getcwd())

files = []
twoD = []
data = []
intData = []
pcaData = []
# Step 1: Read in the list of files
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        # if '.txt' in file:
        files.append(os.path.join(r, file))


# step 2a: read in image as 2d array (243x320px)
count = 0
for f in files:
    im = Image.open(f)
    # step 2b: resize image to become 40x40 px
    twoD.append(im.resize((40, 40)))

for im in twoD:
    # step 2c: flatten to 1D(1x1600)
    # step 2d: Concatenate this as a row of your data matrix.
    data.append(im.resize((1, 1600)))

for im in data:
    intData.append(np.matrix(im))



intData = np.array(intData, dtype=np.float32)

intData = np.squeeze(intData, axis=2)

# step 3a: Standardizes the data
# standardize(intData)
print(np.shape(intData))
intData = preprocessing.scale(np.squeeze(intData))
row = 1
# for column in range(0,1600):
#     m = mean(data_matrix(1:154, column));
#     s = std(data_matrix(1:154, column));
#     data_matrix(:, row) = (data_matrix(:, row) - m)./ s;
#     row = row + 1;


# step 3b: Reduces the data to 2D using PCA         154x1600->154x2
print(np.shape(intData))
covMat = np.cov(np.squeeze(intData))
print(np.shape(covMat))

# get eigen values, eigen vectors from cov. matrix
values, vectors = eig(covMat)

P = vectors.T.dot(np.squeeze(intData))
print(P.T)



#fancy little code to figure out data retention 1st = 77%, +2nd = 85%
var_exp = [(i / sum(values))*100 for i in sorted(values, reverse=True)]
np.cumsum(var_exp)

# step 3c: Plot the data as points in 2D space for visualization
# plt.plot(np.real(P),"wo")

plt.scatter(np.real(P)[:,0], np.real(P)[:,1])
plt.show()
