import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd


df = pd.read_csv("spam.csv")
data = df.to_numpy()

x = []
for i in range(len(data)):
    x.append(np.array([float(i) for i in data[i][0].split()]))
x = np.array(x)

Spam = sum(x[:,-1] == True)
NSpam = sum(x[:,-1] == False)

print("Spam email Percentage = {0:.3f}".format(Spam/len(data)))
print("Non Spam email Percentage = {0:.3f}".format(NSpam/len(data)))

# We make a random permutation of data

x = np.random.permutation(x)

x_train = x[0:int(len(x)*0.7)]  # train data 70%
x_val = x[int(len(x)*0.7)-1:-1] # train data 30%






