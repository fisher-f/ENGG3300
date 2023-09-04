# Name: Lachlan Fisher
# Student number: c3379928
# 
# Your code goes here

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models import LinearRegressionModel
from sklearn.model_selection import KFold

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

# N = len(X[0, :])
# N_train = np.floor(N * 0.2).astype(int)
# X_train = X[:, :N_train]
# Y_train = Y[:, :N_train]
# fit_model.fit(X_train, Y_train)
# y1hat, y2hat, y3hat = fit_model.predict(X_train)

#
# plt.plot(np.linspace(0, 100, len(y1hat)), y2hat, label='pred')
# plt.plot(np.linspace(0, 100, len(Y_train[0, :])), Y_train[1, :], label='true')
# plt.legend()
# plt.show()
#
# print('MSE', np.mean((y1hat - Y_train[0,:]) ** 2))

#--------------------------------------
# 5.1
#-------------------------------------

lengthscales = np.sqrt(np.logspace(-2.5,-0.8,10))
gammas = np.logspace(-2,1,10)
MSE_store = np.zeros((len(lengthscales), len(gammas)))

# for i in range(len(lengthscales)):
#     for j in range(len(gammas)):
#         gamma = gammas[j]
#         ls = lengthscales[i]
#         kf = KFold(5)
#         MSE1 = 0
#         MSE2 = 0
#         MSE3 = 0
#         for train, test in kf.split(X.T, Y.T):
#             fit_model = LinearRegressionModel(locations=locations, lengthscale=ls, gamma=gamma)
#             X_train = X[:, train]
#             Y_train = Y[:, train]
#             Y_test = Y[:, test]
#             X_test = X[:, test]
#             fit_model.fit(X_train, Y_train)
#             y1hat, y2hat, y3hat = fit_model.predict(X_test)
#             MSE1 += np.mean((y1hat - Y_test[0, :]) ** 2)
#             MSE2 += np.mean((y2hat - Y_test[1, :]) ** 2)
#             MSE3 += np.mean((y3hat - Y_test[2, :]) ** 2)
#         MSE1 /= 5
#         MSE2 /= 5
#         MSE3 /= 5
#         MSE_store[i, j] = np.mean((MSE1, MSE2, MSE3))
#         print('\nMSE',MSE_store[i, j], 'gam', gamma, 'ls', ls)


# minMSE = np.min(MSE_store)
# ls_ideal_inds, gam_ideal_inds = np.where(MSE_store == minMSE)
#
# ls_ideal = lengthscales[ls_ideal_inds]
# gam_ideal = gammas[gam_ideal_inds]

ls_ideal = 0.13420781
gam_ideal = 0.04641589
print('\nls', ls_ideal, gam_ideal, 'gamma')



(X1grid, X2grid) = np.meshgrid(np.linspace(-0.09,1.48,100),np.linspace(-3.77,-1.73,100))
locations = np.vstack((X1grid, X2grid))
final_model = LinearRegressionModel(locations=locations, lengthscale=ls_ideal, gamma=gam_ideal)
final_model.save_params("model parameters.npz")
final_model.fit(X, Y)
y1hat_final, y2hat_final, y3hat_final = final_model.predict(X)

X1, X2 = np.meshgrid(x1, x2)
y1hat_final = y1hat_final.reshape(X1.shape)
y2hat_final = y2hat_final.reshape(X1.shape)
y3hat_final = y3hat_final.reshape(X1.shape)

# plt.contour(np.log10(lengthscales),np.log10(gammas),np.log10(MSE_store),25)
# plt.xlabel("Lengthscales []")
# plt.ylabel('Gamma []')
# plt.title('MSE Contour')
# plt.show()
#
fig, axs = plt.subplots(2, 2)

# axs[0].plot(x1, x2)
# axs[0].set_title('Original Position Data')
# axs[0].set_ylabel('x2')
# axs[0].set_xlabel('x1')

axs[0, 1].pcolor(x1, x2, y1hat_final)
axs[0, 1].set_title('Predictions of Y1')
axs[0, 1].set_ylabel('x2')
axs[0, 1].set_xlabel('x1')

axs[1, 0].pcolor(x1, x2, y2hat_final)
axs[1, 0].set_title('Predictions of Y2')
axs[1, 0].set_ylabel('x2')
axs[1, 0].set_xlabel('x1')

axs[1, 1].pcolor(x1, x2, y3hat_final)
axs[1, 1].set_title('Predictions of Y3')
axs[1, 1].set_ylabel('x2')
axs[1, 1].set_xlabel('x1')
#
# set colour bars
colourbarY1 = fig.colorbar(y1hat_final, ax=axs[1])
colourbarY1.set_label('Y1 Prediction')

colourbarY2 = fig.colorbar(y2hat_final, ax=axs[2])
colourbarY2.set_label('Y2 Prediction')

colourbarY3 = fig.colorbar(y3hat_final, ax=axs[3])
colourbarY3.set_label('Y3 Prediction')
plt.show()



print(x1.shape, x2.shape, y1hat_final.shape)