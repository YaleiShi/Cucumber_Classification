import numpy as np


train_x = np.load("./train_x.npy")
train_y = np.load("./train_y.npy")
train_y_encoded = np.load("./train_y_encoded.npy")

test_x = np.load("./test_x.npy")
test_y = np.load("./test_y.npy")
test_y_encoded = np.load("./test_y_encoded.npy")


print("shape of train_x" ,np.shape(train_x))
print("shape of train_y" ,np.shape(train_y))
print("shape of train_y_encoded" ,np.shape(train_y_encoded))


print("shape of test_x" ,np.shape(test_x))
print("shape of test_y" ,np.shape(test_y))
print("shape of test_y_encoded" ,np.shape(test_y_encoded))
