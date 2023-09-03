# Name: Lachlan Fisher
# Student number: c3379928
# 
# Your code goes here

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models import LinearRegressionModel
from sklearn.model_selection import train_test_split

# load the dataframe in
df = pd.read_csv('magnetic_field_experiment1.csv')

# find the elements of only the corrupt and missing inputs

x_flags = df['flag'].values
inds_corrupt = np.where(x_flags == 'corrupt')
inds_missing = np.where(x_flags == 'missing')

x1 = df['x1'].values
x2 = df['x2'].values

x1_corrupted = x1[inds_corrupt]
x2_corrupted = x2[inds_corrupt]

x1_missing = x1[inds_missing]
x2_missing = x2[inds_missing]

# replace flags 'corrupt' and 'missing' with NaN values
df = df.replace(to_replace='corrupt', value=np.nan)
df = df.replace(to_replace='missing', value=np.nan)


# drop all NaN values
df = df.dropna()

# retrieve values from the dataframe
x1 = df['x1'].values
x2 = df['x2'].values
y1 = df['y1'].values
y2 = df['y2'].values
y3 = df['y3'].values

# create 3 x N matrix of outputs
Y = np.vstack((y1, y2, y3))

# create 2 x N matrix of inputs
X = np.vstack((x1, x2))

# plot the position data
# plt.plot(x1, x2, label='Data')
# plt.plot(x1_missing, x2_missing, 'r.', label='Missing Data')
# plt.plot(x1_corrupted, x2_corrupted, 'rx', label='Corrupt Data')
# plt.title("Position Data")
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.legend()
# plt.show()


mu_1 = np.linspace(0,1,20)
mu_2 = np.linspace(0,1,20)
mu = np.vstack((mu_1,mu_2))

test_model = LinearRegressionModel(mu,lengthscale=0.1)
theta1 = np.ones(np.shape(mu_1))
theta2 = np.zeros(np.shape(mu_1))
theta3 = np.linspace(-1,1,20)
X_line = np.vstack((np.linspace(0,1,400),np.linspace(0,1,400)))

test_model.set_parameters(theta1, theta2, theta3)

y1hat, y2hat, y3hat = test_model.predict(X_line)
# plt.plot(X_line[0, :], y1hat)
# plt.plot(X_line[0, :], y2hat)
# plt.plot(X_line[0, :], y3hat)
# plt.show()


#--------------------------------------
# 5.1
#-------------------------------------

N_mesh_x = np.linspace(0, 1.6, 41)
N_mesh_y = np.linspace(0, 2.1, 41)

xlocation, ylocation = np.meshgrid(N_mesh_x, N_mesh_y)
xlocation = np.reshape(xlocation, (1, 41*41))
xlocation -= 0.1
ylocation = -ylocation - 1.7
ylocation = np.reshape(ylocation, (1, 41*41))
locations = np.vstack((xlocation, ylocation))
plt.plot(xlocation-0.1, -ylocation-1.7, '.')
# plt.plot(x1, x2, label='Data')
# plt.plot(x1_missing, x2_missing, 'r.', label='Missing Data')
# plt.plot(x1_corrupted, x2_corrupted, 'rx', label='Corrupt Data')
plt.show()

fit_model = LinearRegressionModel(locations=locations, lengthscale=0.2, gamma=1.0)



N = len(X[0, :])
N_train = np.floor(N * 0.2).astype(int)
X_train = X[:, :N_train]
Y_train = Y[:, :N_train]
fit_model.fit(X_train, Y_train)
y1hat, y2hat, y3hat = fit_model.predict(X_train)


plt.plot(np.linspace(0, 100, len(y1hat)), y1hat, label='pred')
plt.plot(np.linspace(0, 100, len(Y_train[0, :])), Y_train[0, :], label='true')
plt.legend()
plt.show()