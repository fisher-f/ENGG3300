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
plt.plot(x1, x2, label='Data')
plt.plot(x1_missing, x2_missing, 'r.', label='Missing Data')
plt.plot(x1_corrupted, x2_corrupted, 'rx', label='Corrupt Data')
plt.title("Position Data")
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


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

fig, axs = plt.subplots(3, 1)

fig.suptitle("Model testing with lengthscale = 0.1")
axs[0].plot(X_line[0, :], y1hat)
axs[0].set_ylabel("y1")
axs[0].set_xlabel("x1")

axs[1].plot(X_line[0, :], y2hat)
axs[1].set_ylabel("y1")
axs[1].set_xlabel("x1")

axs[2].plot(X_line[0, :], y3hat)
axs[2].set_ylabel("y1")
axs[2].set_xlabel("x1")
plt.show()


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

# plot of the RBF locations over the position data to check that locations go over the initial plot
# plt.plot(xlocation, ylocation, '.')
# plt.plot(x1, x2, label='Data')
# plt.plot(x1_missing, x2_missing, 'r.', label='Missing Data')
# plt.plot(x1_corrupted, x2_corrupted, 'rx', label='Corrupt Data')
# plt.show()

fit_model = LinearRegressionModel(locations=locations, lengthscale=0.2, gamma=1.0)
#
N = len(X[0, :])
N_train = np.floor(N * 0.2).astype(int)
X_train = X[:, :N_train]
Y_train = Y[:, :N_train]
fit_model.fit(X_train, Y_train)
y1hat, y2hat, y3hat = fit_model.predict(X_train)

x_length = np.linspace(0, len(y1hat), len(y1hat))
#
fig, axs = plt.subplots(3, 1)

axs[0].plot(x_length, y1hat, label="Predicted")
axs[0].plot(x_length, Y_train[0, :], label="True")
axs[0].set_title("Y1 Prediction")
axs[0].set_ylabel("Y Data")
axs[0].set_xlabel("Sample No.")
axs[0].legend()

axs[1].plot(x_length, y2hat, label="Predicted")
axs[1].plot(x_length, Y_train[1, :], label="True")
axs[1].set_title("Y2 Prediction")
axs[1].set_ylabel("Y Data")
axs[1].set_xlabel("Sample No.")
axs[1].legend()

axs[2].plot(x_length, y3hat, label="Predicted")
axs[2].plot(x_length, Y_train[2, :], label="True")
axs[2].set_title("Y2 Prediction")
axs[2].set_ylabel("Y Data")
axs[2].set_xlabel("Sample No.")
axs[2].legend()
plt.tight_layout()
plt.show()



# plt.plot(np.linspace(0, 100, len(y1hat)), y2hat, label='pred')
# plt.plot(np.linspace(0, 100, len(Y_train[0, :])), Y_train[1, :], label='true')
# plt.legend()
# plt.show()
#
print('The MSE of the first 20% of the measurements is', np.mean((y1hat - Y_train[0,:]) ** 2))

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


minMSE = np.min(MSE_store)
ls_ideal_inds, gam_ideal_inds = np.where(MSE_store == minMSE)

ls_ideal = lengthscales[ls_ideal_inds]
gam_ideal = gammas[gam_ideal_inds]

ls_ideal = 0.13420781
gam_ideal = 0.04641589
print('\nls', ls_ideal, gam_ideal, 'gamma')



(X1grid, X2grid) = np.meshgrid(np.linspace(-0.09,1.48,100),np.linspace(-3.77,-1.73,100))
X1grid = np.reshape(X1grid, (1, 100*100))
X2grid = np.reshape(X2grid, (1, 100*100))
Xnew = np.vstack((X1grid, X2grid))
final_model = LinearRegressionModel(locations=locations, lengthscale=ls_ideal, gamma=gam_ideal) # change to initial locations
final_model.fit(X, Y)
y1hat_save, y2hat_save, y3hat_save = final_model.predict(X)
final_model.save_params("model_parameters.npz")
y1hat_final, y2hat_final, y3hat_final = final_model.predict(Xnew)  #change to locations final


plt.plot(np.linspace(0, 100, len(y1hat_final)), y1hat_final, label='pred')
plt.plot(np.linspace(0, 100, len(Y[0, :])), Y[0, :], label='true')
plt.legend()
plt.show()

(X1, X2) = np.meshgrid(x1,x2)

print(X1.shape, X2.shape, y1hat_final.shape)

# fig, axs = plt.subplots()
# MSEPlot = axs.contour(np.log10(lengthscales),np.log10(gammas),np.log10(MSE_store),25)
# axs.set_title("MSE Contour")
# axs.set_xlabel("lengthscale")
# axs.set_ylabel("gamma")
# colourbarMSE = fig.colorbar(MSEPlot, ax=axs)
# colourbarMSE.set_label('Mean Sqaured Error')
# plt.show()

# Clip the data so that we can plot it
# reshape the output y as a function of x1 and x2
y1hat_plot = np.reshape(y1hat_final, (100, 100))      #change this to 100 by 100 as its what we put into final model
y2hat_plot = np.reshape(y2hat_final, (100, 100))
y3hat_plot = np.reshape(y3hat_final, (100, 100))

yMagnitude = np.sqrt((y1hat_plot**2) + (y2hat_plot**2) + (y3hat_plot**2))
# note this gives shape 80, 80 for both X1plot, X2plot and the y[i]_plot
(X1plot, X2plot) = np.meshgrid(np.linspace(-0.09,1.48,100), np.linspace(-3.77,-1.73,100))         #change to size 100

fig, axs = plt.subplots(2, 2)

yM_plot = axs[0, 0].pcolor(X1plot, X2plot, yMagnitude,  shading='auto')
axs[0, 0].set_title('Predictions of Y Magnitude')
axs[0, 0].set_ylabel('x2')
axs[0, 0].set_xlabel('x1')

y1_plot = axs[0, 1].pcolor(X1plot, X2plot, y1hat_plot,  shading='auto')
axs[0, 1].set_title('Predictions of Y1')
axs[0, 1].set_ylabel('x2')
axs[0, 1].set_xlabel('x1')

y2_plot = axs[1, 0].pcolor(X1plot, X2plot, y2hat_plot,  shading='auto')
axs[1, 0].set_title('Predictions of Y2')
axs[1, 0].set_ylabel('x2')
axs[1, 0].set_xlabel('x1')

y3_plot = axs[1, 1].pcolor(X1plot, X2plot, y3hat_plot,  shading='auto')
axs[1, 1].set_title('Predictions of Y3')
axs[1, 1].set_ylabel('x2')
axs[1, 1].set_xlabel('x1')
#
# set colour bars
colourbarYM = fig.colorbar(yM_plot, ax=axs[0, 0])
colourbarYM.set_label('Y Magnitude Prediction')

colourbarY1 = fig.colorbar(y1_plot, ax=axs[0, 1])
colourbarY1.set_label('Y1 Prediction')

colourbarY2 = fig.colorbar(y2_plot, ax=axs[1, 0])
colourbarY2.set_label('Y2 Prediction')

colourbarY3 = fig.colorbar(y3_plot, ax=axs[1, 1])
colourbarY3.set_label('Y3 Prediction')


plt.show()


## find the variance of the predictions

# use model to calculate the variance
Sigma_y_star = final_model.calculate_variances(X, Xnew)

# calculate the sqrt of diagonal elements in the matrix
# this gives the standard deviation at each point
stand_dev = np.sqrt(np.diag(Sigma_y_star))
stand_dev = np.reshape(stand_dev, (100, 100))
# plot the SD as function of X1plot and X2plot

fig, axs = plt.subplots()
SDPlot = axs.pcolor(X1plot, X2plot, stand_dev, shading='auto')
axs.set_title("Standard Deviation at each point")
axs.set_xlabel("X1")
axs.set_ylabel("X2")
colourbarSD = fig.colorbar(SDPlot, ax=axs)
colourbarSD.set_label('Standard Deviation')
plt.show()